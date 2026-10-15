"""One-call HAL walker for the five supported ensembles."""

from __future__ import annotations

import json
import os
import sys
import time
from dataclasses import dataclass, field

import numpy as np
from ase import units
from ase.md.langevin import Langevin
from ase.md.velocitydistribution import MaxwellBoltzmannDistribution, Stationary, ZeroRotation

from tensorpotential.uq.hal.bias import PairTable
from tensorpotential.uq.hal.calculator import HALCalculator, HALStop
from tensorpotential.uq.hal.collector import DEFAULT_ALPHAS, HALCollector, check_collector_options
from tensorpotential.uq.hal.features import FEATURES_FILE
from tensorpotential.uq.hal.model_dir import resolve_model_dir
from tensorpotential.uq.hal.dynamics import (
    FRAGMENT_MULT,
    RampedLangevin,
    VolumeGuard,
    cluster_wall,
    default_timestep_fs,
    estimate_bulk_modulus,
)

PERIODIC_ENSEMBLES = ("nvt", "nvt_ramp", "npt", "npt_stress")
ENSEMBLES = PERIODIC_ENSEMBLES + ("cluster",)
OUTPUT_FILES = ("hal_collected.extxyz", "hal_live_artifacts.npz", "hal_history.npz",
                "hal_summary.json", "hal_diagnostics.png", "hal_settings.json", FEATURES_FILE)


@dataclass
class HALResult:
    frames: list = field(default_factory=list)
    history: dict = field(default_factory=dict)
    summary: dict = field(default_factory=dict)
    live_model: object = None


class ProgressPrinter:
    """Observer (attach after the collector): one status line every ``every`` steps.

    Reads only the collector's last history row and counters: no model or GMM work.
    """

    def __init__(self, collector, n_steps, every, stream=None):
        self.col, self.n_steps, self.every = collector, int(n_steps), int(every)
        self.stream = stream or sys.stdout
        self._t0 = None
        self._last_printed = None

    def _line(self, row, suffix=""):
        step = int(row["step"])
        dt = time.perf_counter() - (self._t0 or time.perf_counter())
        rate = step / dt if dt > 0 else float("nan")
        print(f"step {step}/{self.n_steps} | max γ {row['gamma_max']:.2f} | "
              f"collected {self.col.n_collected} | rejected {sum(self.col.rejected.values())} | "
              f"T {row['temperature']:.0f} K | {rate:.0f} steps/s{suffix}",
              file=self.stream, flush=True)
        self._last_printed = step

    def __call__(self):
        if self._t0 is None:
            self._t0 = time.perf_counter()
        row = self.col.history.last()
        step = int(row.get("step", 0))
        if step == 0 or step % self.every:
            return
        self._line(row)

    def final(self, stop_reason=None):
        """Print the last recorded step if it has not been printed (early stop)."""
        row = self.col.history.last()
        if not row or int(row["step"]) == self._last_printed:
            return
        early = stop_reason and stop_reason != "max_steps"
        self._line(row, f" | stop: {stop_reason}" if early else "")


def _write_settings(output_dir: str, settings: dict):
    """Persist the run settings so a killed walker can still be replotted."""
    with open(os.path.join(output_dir, "hal_settings.json"), "w") as fh:
        json.dump(settings, fh, indent=2, default=str)


def _is_int(x) -> bool:
    return isinstance(x, (int, np.integer)) and not isinstance(x, bool)


def _validate_options(atoms, *, ensemble, n_steps, tau_ema, zbl, zbl_switch,
                      reject_bond_factor, pair_overrides, trigger_min_atoms, update_work_floor,
                      update, alphas, update_inflate, fragment_mult=FRAGMENT_MULT,
                      feature_gamma_min=1.0):
    """Model-free option checks, run before the output dir is touched or the model loads.

    A bad option must not cost the previous walker's files (``overwrite=True``) or a model
    load. The components (collector, calculator, dynamics) keep their own checks as a
    second line of defence.
    """
    if ensemble not in ENSEMBLES:
        raise ValueError(f"ensemble must be one of {ENSEMBLES}, got {ensemble!r}")
    pbc = np.asarray(atoms.pbc, dtype=bool)
    if pbc.all() and atoms.cell.rank == 3:
        periodic = True
    elif not pbc.any():
        periodic = False
    else:
        raise ValueError(
            "structure must be fully periodic or fully non-periodic (mixed pbc is not "
            f"supported); ensembles: periodic {PERIODIC_ENSEMBLES}, non-periodic ('cluster',)"
        )
    if periodic and ensemble == "cluster":
        raise ValueError(f"ensemble 'cluster' is for non-periodic structures; use one of {PERIODIC_ENSEMBLES}")
    if not periodic and ensemble != "cluster":
        raise ValueError(f"non-periodic structure: use ensemble='cluster' (got {ensemble!r})")
    if not _is_int(n_steps) or n_steps < 1:
        raise ValueError(f"n_steps must be an integer >= 1, got {n_steps!r}")
    if not _is_int(tau_ema) or tau_ema < 1:
        raise ValueError(f"tau_ema must be an integer >= 1, got {tau_ema!r}")
    if zbl not in ("auto", "off"):
        raise ValueError(f"zbl must be 'auto' or 'off', got {zbl!r}")
    PairTable(zbl_switch, reject_bond_factor, pair_overrides)  # zbl_switch + override keys
    check_collector_options(trigger_min_atoms=trigger_min_atoms,
                            update_work_floor=update_work_floor, update=update, alphas=alphas,
                            update_inflate=update_inflate, fragment_mult=fragment_mult,
                            feature_gamma_min=feature_gamma_min)


def _prepare_output_dir(output_dir: str, overwrite: bool):
    existing = [f for f in OUTPUT_FILES if os.path.exists(os.path.join(output_dir, f))]
    if existing and not overwrite:
        raise FileExistsError(
            f"{output_dir} already holds HAL output ({', '.join(existing)}); "
            "choose another directory (--output-dir; output_dir= in Python) or replace "
            "them with --overwrite (overwrite=True in Python)"
        )
    for f in existing:
        os.remove(os.path.join(output_dir, f))
    os.makedirs(output_dir, exist_ok=True)


def run_hal(atoms, *, temperature_K, n_steps, model_yaml=None, checkpoint=None,
            uq_artifact=None, model_dir=None, ensemble="nvt", live_start_artifact=None, n_collect=None, timestep_fs=None,
            tau=0.15, tau_ema=10, equil_steps=0, trigger_gamma=1.5, trigger_gap=10,
            trigger_min_atoms=1, update="alpha_search", update_target=1.25, alphas=None,
            update_inflate=True, inflate_max_rounds=3, inflate_max_factor=4.0,
            reject_gamma=10.0, reject_bond_factor=0.65, zbl="auto", zbl_switch=(0.55, 0.75),
            pair_overrides=None, nvt_ramp=(1.0, 1.4), stress_tau=0.02, bulk_modulus="auto",
            volume_guard=(0.75, 1.30), cluster_wall_margin=2.0, cluster_wall_k=1.0,
            friction=0.01, tdamp_fs=100.0, pdamp_fs=1000.0, remove_update_work=True,
            update_work_floor=0.5, seed=42, walker_id=0, output_dir="hal_w0", overwrite=False,
            keep_frames=True, save_every=10, history_every=1000, calc_kwargs=None,
            plot=True, progress_every=0, fragment_mult=FRAGMENT_MULT,
            feature_gamma_min=1.0) -> HALResult:
    """Run one HAL walker. See docs/gracemaker/uq_workflows.md for the protocol.

    The model is given either as ``model_yaml`` + ``checkpoint`` + ``uq_artifact`` or as
    ``model_dir`` (see :func:`resolve_model_dir`; explicit paths override the search).
    ``progress_every > 0`` prints a status line every that many steps.
    ``feature_gamma_min`` (None: off) keeps the features of the atoms with
    ``gamma_pristine >= feature_gamma_min`` of every collected frame in
    ``hal_features.npz``, the input of ``grace_uq select --hal-dirs``.
    """
    settings = {k: v for k, v in locals().items() if k != "atoms"}
    if model_dir is not None:
        model_yaml, checkpoint, uq_artifact = resolve_model_dir(
            model_dir, model_yaml=model_yaml, checkpoint=checkpoint, artifact_path=uq_artifact)
        if progress_every:  # quiet by default in Python, like the progress lines
            print(f"HAL model from {model_dir}: {model_yaml} | {checkpoint} | {uq_artifact}",
                  flush=True)
        settings.update(model_yaml=model_yaml, checkpoint=checkpoint, uq_artifact=uq_artifact)
    missing = [n for n, v in (("model_yaml", model_yaml), ("checkpoint", checkpoint),
                              ("uq_artifact", uq_artifact)) if v is None]
    if missing:
        raise ValueError(f"pass model_dir or all of model_yaml, checkpoint, uq_artifact (missing: {missing})")
    _validate_options(
        atoms, ensemble=ensemble, n_steps=n_steps, tau_ema=tau_ema, zbl=zbl,
        zbl_switch=zbl_switch, reject_bond_factor=reject_bond_factor,
        pair_overrides=pair_overrides, trigger_min_atoms=trigger_min_atoms,
        update_work_floor=update_work_floor, update=update, alphas=alphas,
        update_inflate=update_inflate, fragment_mult=fragment_mult,
        feature_gamma_min=feature_gamma_min,
    )
    # explicit alphas win; otherwise the default (additive-update) grid
    alphas = DEFAULT_ALPHAS if alphas is None else tuple(float(a) for a in alphas)
    settings["alphas"] = list(alphas)
    _prepare_output_dir(output_dir, overwrite)

    from tensorpotential.uq.factories import get_gmm_uq_calculator
    from tensorpotential.uq.gmmuq import GMMUQModel

    base = get_gmm_uq_calculator(
        model_yaml=model_yaml, checkpoint=checkpoint,
        gmm_artifact_path=live_start_artifact or uq_artifact,
        compute_dsigma_dr=True, **(calc_kwargs or {}),
    )
    missing = sorted(set(atoms.get_chemical_symbols()) - set(base.element_map))
    if missing:
        raise ValueError(f"structure has elements not in the model: {missing}")
    element_indices = np.array([base.element_map[s] for s in atoms.get_chemical_symbols()])
    pristine = GMMUQModel(uq_artifact)
    # Both towers re-inverted from their (float64) scatter before the first base call:
    # the stored float32 inverse differs from what an update re-inverts by ~1e-4 in
    # gamma, which would let gamma_live exceed gamma_pristine at a cluster's first
    # update. Now both start from the same inverses and gamma_live <= gamma_pristine
    # holds on the statistics (live_start_artifact: its saved scatter, likewise).
    # Only the structure's elements: a walker never evaluates the others.
    t_reinvert = time.perf_counter()
    present = np.unique(element_indices)
    for tower in (getattr(base, "gmm_uq_model", None), pristine):  # None: a non-UQ base
        if tower is not None:
            tower.reinvert_all(present)
    settings["reinvert_s"] = time.perf_counter() - t_reinvert

    b_mod = None
    if ensemble == "npt_stress":
        b_mod = estimate_bulk_modulus(atoms, base) if bulk_modulus == "auto" else float(bulk_modulus)
        settings["bulk_modulus_used"] = b_mod
    wall = None
    if ensemble == "cluster":
        wall = cluster_wall(atoms, cluster_wall_margin, cluster_wall_k)
        settings["cluster_wall_radius"] = wall[1]

    hal = HALCalculator(
        base, tau=tau, tau_ema=tau_ema, zbl=zbl, zbl_switch=zbl_switch,
        reject_bond_factor=reject_bond_factor, pair_overrides=pair_overrides,
        stress_tau=stress_tau if ensemble == "npt_stress" else None, bulk_modulus=b_mod,
        wall=wall,
    )
    atoms = atoms.copy()
    atoms.calc = hal
    rng = np.random.default_rng(seed)
    MaxwellBoltzmannDistribution(atoms, temperature_K=temperature_K, rng=rng)
    # no temperature preservation: rescaling would move the removed share into the
    # internal modes (same reason as the per-step reset below)
    Stationary(atoms, preserve_temperature=False)
    if ensemble == "cluster":
        ZeroRotation(atoms, preserve_temperature=False)
    dt = (timestep_fs or default_timestep_fs(atoms.numbers)) * units.fs
    settings["timestep_fs_used"] = dt / units.fs

    # Langevin: fixcm=False plus a COM reset every step (observer attached before the
    # collector, so recorded T and frames are post-reset). ASE's fixcm=True projection of
    # the random force assumes equal masses and runs hot with mixed masses (ethanol at
    # 300 K: ~380 K). The reset must not preserve the temperature: rescaling after removing
    # the COM momentum would move its share into the internal modes (T*N/(N-1)).
    if ensemble in ("nvt", "cluster"):
        dyn = Langevin(atoms, dt, temperature_K=temperature_K,
                       friction=friction / units.fs, fixcm=False, rng=rng)
    elif ensemble == "nvt_ramp":
        dyn = RampedLangevin(atoms, dt, temperature_K=temperature_K,
                             friction=friction / units.fs, n_steps=n_steps,
                             v_start=nvt_ramp[0], v_end=nvt_ramp[1], fixcm=False, rng=rng)
    else:  # npt, npt_stress
        from ase.md.nose_hoover_chain import MTKNPT

        dyn = MTKNPT(atoms, dt, temperature_K=temperature_K, pressure_au=units.bar,
                     tdamp=tdamp_fs * units.fs, pdamp=pdamp_fs * units.fs)
        dyn.attach(VolumeGuard(atoms, *volume_guard), interval=1)
    langevin = ensemble in ("nvt", "nvt_ramp", "cluster")
    if langevin:
        dyn.attach(Stationary, 1, atoms, preserve_temperature=False)
    # MTKNPT integrates its own momentum copy (ASE 3.26) and discards momenta set on the
    # atoms, so the update-work removal can only act on the Langevin ensembles.
    settings["update_work_removal_active"] = bool(remove_update_work and langevin)

    collector = HALCollector(
        atoms, hal, pristine=pristine, element_indices=element_indices,
        output_dir=output_dir, equil_steps=equil_steps, trigger_gamma=trigger_gamma,
        trigger_gap=trigger_gap, trigger_min_atoms=trigger_min_atoms,
        reject_gamma=reject_gamma, update=update,
        update_target=update_target, alphas=alphas,
        update_inflate=update_inflate,
        inflate_max_rounds=inflate_max_rounds, inflate_max_factor=inflate_max_factor,
        n_collect=n_collect,
        save_every=save_every, history_every=history_every, walker_id=walker_id,
        keep_frames=keep_frames, capacity=n_steps + 1, temperature_K=temperature_K,
        remove_update_work=settings["update_work_removal_active"],
        update_work_floor=update_work_floor, fragment_mult=fragment_mult,
        feature_gamma_min=feature_gamma_min,
        feature_meta={"model_yaml": os.path.abspath(model_yaml)},
    )
    # the collector hashed the pristine artifact for the sidecar (select checks it)
    settings["uq_artifact_sha256"] = (collector.candidates.meta["artifact_sha256"] or None
                                      if collector.candidates is not None else None)
    dyn.attach(collector, interval=1)
    settings["n_fragments0"] = collector.n_fragments0
    progress = ProgressPrinter(collector, n_steps, progress_every) if progress_every else None
    if progress is not None:
        dyn.attach(progress, interval=1)

    _write_settings(output_dir, settings)

    try:
        dyn.run(n_steps)
        reason = "max_steps"
    except HALStop as stop:
        reason = stop.reason
    except BaseException as exc:
        collector.finalize(f"error: {type(exc).__name__}", settings)
        raise
    if progress is not None:
        progress.final(reason)
    summary = collector.finalize(reason, settings)
    if plot:
        from tensorpotential.uq.hal.plot import plot_diagnostics

        plot_diagnostics(collector.history.as_dict(), summary,
                         os.path.join(output_dir, "hal_diagnostics.png"))
    return HALResult(collector.frames, collector.history.as_dict(), summary, collector.live)
