"""HAL collection observer: trigger, reject, write, update the live CALM tower."""

from __future__ import annotations

import json
import os
import time
import warnings

import numpy as np
from ase import units
from ase.io import write

from tensorpotential.uq.artifact_builder import atomic_savez
from tensorpotential.uq.hal.calculator import HALStop
from tensorpotential.uq.hal.dynamics import (
    FRAGMENT_MULT,
    count_fragments,
    internal_temperature,
    radius_of_gyration,
)
from tensorpotential.uq.hal.features import (
    FEATURES_FILE,
    CandidateFeatures,
    check_feature_gamma_min,
    file_sha256,
)

# HAL updates the live tower additively (C += alpha * C_frame, GMMUQModel mode="additive");
# the alpha that brings a frame to the update target is often 0.01-0.05
DEFAULT_ALPHAS = (0.0005, 0.001, 0.002, 0.005, 0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 1.0)
HISTORY_FIELDS = (
    "step", "gamma_max", "gamma_sum", "frac_gamma_gt1", "tau_r", "f_norm", "g_norm",
    "temperature", "v_ratio", "rg_ratio", "wall_energy", "pressure_gpa", "d_min",
    "bond_ratio_min", "status", "alpha", "gamma_pristine_max", "n_gamma_above", "update_work",
)
STATUS_NONE, STATUS_COLLECTED, STATUS_REJECTED = 0, 1, 2


def check_collector_options(*, trigger_min_atoms=1, update_work_floor=0.5,
                            update="alpha_search", alphas=None, update_inflate=True,
                            fragment_mult=FRAGMENT_MULT, feature_gamma_min=1.0):
    """Model-free option checks of :class:`HALCollector`.

    ``run_hal`` calls this before it touches the output directory or loads the model;
    ``HALCollector.__init__`` repeats it for hand-composed walkers.
    """
    if (isinstance(trigger_min_atoms, bool) or not isinstance(trigger_min_atoms, (int, np.integer))
            or trigger_min_atoms < 1):
        raise ValueError(f"trigger_min_atoms must be an integer >= 1, got {trigger_min_atoms!r}")
    if not fragment_mult > 0:
        raise ValueError(f"fragment_mult must be > 0, got {fragment_mult!r}")
    check_feature_gamma_min(feature_gamma_min)
    if not update_work_floor >= 0:
        raise ValueError(f"update_work_floor must be >= 0, got {update_work_floor!r}")
    if update not in ("alpha_search", "none"):
        raise ValueError(f"update must be 'alpha_search' or 'none', got {update!r}")
    if alphas is None:
        alphas = DEFAULT_ALPHAS
    if update == "alpha_search" and update_inflate and float(alphas[-1]) != 1.0:
        raise ValueError(
            "inflation is only defined after a full alpha = 1 update: end --alphas with "
            f"1.0 (got alphas={tuple(alphas)}) or pass --no-update-inflate"
        )


class History:
    """Preallocated per-step arrays; grows by doubling."""

    def __init__(self, capacity: int = 1024):
        self._a = {k: np.full(max(1, capacity), np.nan) for k in HISTORY_FIELDS}
        self.n = 0

    def append(self, **row):
        if self.n == len(self._a["step"]):
            for k, v in self._a.items():
                self._a[k] = np.concatenate([v, np.full(len(v), np.nan)])
        for k in HISTORY_FIELDS:
            self._a[k][self.n] = row.get(k, np.nan)
        self.n += 1

    def last(self) -> dict:
        """The most recent row (empty dict before the first append)."""
        return {k: v[self.n - 1] for k, v in self._a.items()} if self.n else {}

    def as_dict(self) -> dict:
        return {k: v[: self.n].copy() for k, v in self._a.items()}

    def save(self, path: str):
        """The file is for diagnostics only: float32 and compressed (about 4.5x smaller),
        except ``step``, which stays an exact int64. The in-memory arrays (``as_dict``,
        ``run_hal(...).history``) stay float64."""
        atomic_savez(path, compressed=True,
                     **{k: v.astype(np.int64 if k == "step" else np.float32)
                        for k, v in self.as_dict().items()})


class HALCollector:
    """ASE observer (attach with interval=1). Reads only ``calc.results``.

    A frame triggers when at least ``trigger_min_atoms`` atoms have gamma >
    ``trigger_gamma`` (1 = the max-gamma rule); rejection and the alpha search stay on max
    gamma. A rejected trigger also starts the ``trigger_gap`` window (intended, so a bad
    region is not re-checked every step).

    Update work: between updates the bias force tau_r * grad(sum gamma) derives from
    U_b = -tau_r * sum gamma. A live update lowers sum gamma at fixed positions, so U_b
    jumps up by dU_b = tau_r * (sum gamma_before - sum gamma_after) and the walker turns
    that into heat. With ``remove_update_work`` the collector takes dU_b back out of the
    kinetic energy by a uniform momentum rescale, never below
    ``update_work_floor * temperature_K`` (the rest is recorded as not removed).

    Every recorded temperature (history, frame ``T``, the floor) counts 3N-3 dof
    (``internal_temperature``): the COM momentum is zero in all ensembles. Frame ``T`` is
    after the removal, consistent with the stored momenta; ``T_before_removal`` is the
    value before it.

    Fragments: ``n_fragments`` of each collected frame (all ensembles) counts connected
    components with bonds below ``fragment_mult`` (default 1.25) x the covalent-radius sum
    (``dynamics.count_fragments``;
    see its bulk caveat); ``n_fragmented`` counts frames with more fragments than the
    starting structure (``n_fragments0``).

    Warning: the removal rescales ``atoms`` momenta, which only integrators that read the
    momenta back from the atoms honour (ASE Langevin). ASE ``MTKNPT`` keeps its own copy
    and discards them, so a hand-composed collector on MTKNPT records removals that never
    happen: pass ``remove_update_work=False`` there (``run_hal`` does this for npt and
    npt_stress).

    Candidate features: with ``feature_gamma_min`` (not None) every collection keeps the
    collected features of the atoms with ``gamma_pristine >= feature_gamma_min`` in
    ``hal_features.npz`` (see :mod:`tensorpotential.uq.hal.features`), written at history
    saves and at finalize; ``feature_meta`` (``artifact_sha256``, ``model_yaml``) goes into
    its metadata. Without an ``artifact_sha256`` there, the collector hashes the file the
    pristine model was loaded from (``pristine.source_path``), once; a pristine model
    built in memory leaves the sidecar without a hash (with a warning). Frames carry ``collection_index`` (0-based), the sidecar's link.
    """

    def __init__(self, atoms, calc, *, pristine, element_indices, output_dir,
                 equil_steps=0, trigger_gamma=1.5, trigger_gap=10, reject_gamma=10.0,
                 update="alpha_search", update_target=1.25, alphas=None,
                 update_inflate=True, inflate_max_rounds=3, inflate_max_factor=4.0,
                 n_collect=None, save_every=10, history_every=1000, walker_id=0,
                 keep_frames=True, capacity=1024, trigger_min_atoms=1, temperature_K=None,
                 remove_update_work=True, update_work_floor=0.5, fragment_mult=FRAGMENT_MULT,
                 feature_gamma_min=1.0, feature_meta=None):
        check_collector_options(trigger_min_atoms=trigger_min_atoms,
                                update_work_floor=update_work_floor, update=update,
                                alphas=alphas, update_inflate=update_inflate,
                                fragment_mult=fragment_mult,
                                feature_gamma_min=feature_gamma_min)
        if remove_update_work and temperature_K is None:
            raise ValueError(
                "remove_update_work needs temperature_K (the floor is "
                "update_work_floor * temperature_K); pass it or remove_update_work=False"
            )
        self.live = getattr(calc.base, "gmm_uq_model", None) if update == "alpha_search" else None
        if update == "alpha_search" and self.live is None:
            raise ValueError(
                "update='alpha_search' needs an updatable live CALM model: build the base "
                "calculator from model.yaml + checkpoint (get_gmm_uq_calculator / "
                "load_uq_artifacts). A SavedModel can only run with update='none'."
            )
        self.atoms, self.calc, self.pristine = atoms, calc, pristine
        self.element_indices = np.asarray(element_indices, dtype=np.int32)
        self.equil_steps, self.trigger_gamma, self.trigger_gap = equil_steps, trigger_gamma, trigger_gap
        self.trigger_min_atoms = int(trigger_min_atoms)
        self.reject_gamma, self.update_target = reject_gamma, update_target
        self.alphas = tuple(float(a) for a in (DEFAULT_ALPHAS if alphas is None else alphas))
        self.update_inflate = update_inflate
        self.inflate_max_rounds, self.inflate_max_factor = inflate_max_rounds, inflate_max_factor
        self.n_collect, self.save_every, self.history_every = n_collect, save_every, history_every
        self.walker_id, self.keep_frames = walker_id, keep_frames
        self.temperature_K = temperature_K
        self.remove_update_work, self.update_work_floor = bool(remove_update_work), float(update_work_floor)
        os.makedirs(output_dir, exist_ok=True)
        self.output_dir = output_dir
        self.frames_path = os.path.join(output_dir, "hal_collected.extxyz")
        self.live_path = os.path.join(output_dir, "hal_live_artifacts.npz")
        self.history_path = os.path.join(output_dir, "hal_history.npz")
        self.summary_path = os.path.join(output_dir, "hal_summary.json")
        self.history = History(capacity)
        self.frames, self.collections = [], []
        self.rejected: dict[str, int] = {}
        self.n_collected, self.step = 0, 0
        self._last_trigger = -np.inf
        self.periodic = atoms.cell.rank == 3 and all(atoms.pbc)
        self.v0 = atoms.get_volume() if self.periodic else np.nan
        self.rg0 = np.nan if self.periodic else radius_of_gyration(atoms)
        self.fragment_mult = float(fragment_mult)
        self.n_fragments0 = count_fragments(atoms, self.fragment_mult)  # start (normally 1)
        self.candidates = None
        if feature_gamma_min is not None:
            meta = dict(feature_meta or {})
            if not meta.get("artifact_sha256"):
                src = getattr(pristine, "source_path", None)
                if src:
                    meta["artifact_sha256"] = file_sha256(src)
                else:
                    warnings.warn(
                        "HALCollector: the pristine CALM model was not loaded from a file "
                        "(no source_path), so hal_features.npz carries no artifact hash: "
                        "grace_uq select --hal-dirs cannot check it against --artifact or "
                        "mix it with other walkers. Load the pristine tower with "
                        "GMMUQModel(path) or pass feature_meta={'artifact_sha256': ...}.",
                        stacklevel=2,
                    )
            self.candidates = CandidateFeatures(
                os.path.join(output_dir, FEATURES_FILE), feature_gamma_min, atoms.numbers,
                walker_id=walker_id, **meta)
        self._t0 = time.perf_counter()

    def _shape_ratios(self):
        """(V/V0, Rg/Rg0): the first for periodic cells, the second for clusters."""
        if self.periodic:
            return self.atoms.get_volume() / self.v0, np.nan
        return np.nan, radius_of_gyration(self.atoms) / self.rg0

    # -- per step ----------------------------------------------------------
    def __call__(self):
        r = self.calc.results
        gamma = np.asarray(r["gamma"], dtype=np.float64)
        gmax = float(gamma.max())
        n_above = int(np.count_nonzero(gamma > self.trigger_gamma))
        status, alpha, gmax_pristine, update_work = STATUS_NONE, np.nan, np.nan, np.nan
        if (self.step >= self.equil_steps and n_above >= self.trigger_min_atoms
                and self.step - self._last_trigger >= self.trigger_gap):
            self._last_trigger = self.step
            g_pristine, _ = self.pristine.gamma_from_features(
                np.asarray(r["features"]), self.element_indices
            )
            g_pristine = np.asarray(g_pristine, dtype=np.float64)
            gmax_pristine = float(g_pristine.max())
            reason = self._reject_reason(r, gmax, gmax_pristine)
            if reason:
                self.rejected[reason] = self.rejected.get(reason, 0) + 1
                status = STATUS_REJECTED
            else:
                alpha, update_work = self._collect(r, gamma, g_pristine, n_above)
                status = STATUS_COLLECTED
        stress = np.asarray(r["stress"]).reshape(-1)
        v_ratio, rg_ratio = self._shape_ratios()
        self.history.append(
            step=self.step, gamma_max=gmax, gamma_sum=float(gamma.sum()),
            frac_gamma_gt1=float(np.mean(gamma > 1.0)), tau_r=r["tau_r"],
            f_norm=r["f_norm_ema"], g_norm=r["g_norm_ema"],
            temperature=internal_temperature(self.atoms),
            v_ratio=v_ratio, rg_ratio=rg_ratio, wall_energy=r.get("wall_energy", np.nan),
            pressure_gpa=-float(stress[:3].mean()) / units.GPa if self.periodic else np.nan,
            d_min=r["d_min"], bond_ratio_min=r["bond_ratio_min"], status=status, alpha=alpha,
            gamma_pristine_max=gmax_pristine,  # NaN except at trigger steps
            n_gamma_above=n_above, update_work=update_work,  # NaN except at collection steps
        )
        self.step += 1
        if self.history_every and self.step % self.history_every == 0:
            self.history.save(self.history_path)
            self.save_candidates()
        if self.n_collect and self.n_collected >= self.n_collect:
            raise HALStop("n_collect")

    def _reject_reason(self, r, gmax, gmax_pristine):
        if gmax > self.reject_gamma:
            return "gamma"
        if gmax_pristine > self.reject_gamma:
            return "gamma_pristine"
        if r["bond_ratio_min"] < 1.0:
            return "short_bond"
        return None

    def _remove_kinetic(self, work: float) -> float:
        """Rescale all momenta to lower the kinetic energy by ``work`` (eV), not below the floor.

        Returns the energy actually removed.
        """
        ke, t_now = self.atoms.get_kinetic_energy(), internal_temperature(self.atoms)
        if ke <= 0 or t_now <= 0:
            return 0.0
        t_floor = self.update_work_floor * (self.temperature_K or 0.0)
        removed = min(work, max(0.0, ke * (1.0 - t_floor / t_now)))
        if removed > 0:
            self.atoms.set_momenta(self.atoms.get_momenta() * np.sqrt((ke - removed) / ke))
        return float(removed)

    def _update_work(self, r, feats, gamma):
        """(dU_b, removed) of the live update just made; one gamma evaluation."""
        tau_r = float(r["tau_r"])
        if self.live is None or not tau_r > 0:
            return 0.0, 0.0
        g_after, _ = self.live.gamma_from_features(feats, self.element_indices)
        work = tau_r * float(gamma.sum() - np.sum(np.asarray(g_after, dtype=np.float64)))
        removed = self._remove_kinetic(work) if self.remove_update_work and work > 0 else 0.0
        return work, removed

    def _collect(self, r, gamma, g_pristine, n_above):
        feats = np.asarray(r["features"])
        alpha = np.nan
        n_inflate, inflate_factor = 0, 1.0
        t_update = time.perf_counter()
        if self.live is not None:
            alpha, reached = self.live.alpha_search_update(
                feats, self.element_indices, self.alphas, self.update_target)
            if not reached and self.update_inflate:
                # Paper protocol: bounded inflation only when the alpha search ended at
                # alpha = 1 without reaching the target (never after alpha < 1: the
                # inflation would compound across collections and blind the live model).
                res = self.live.inflate(
                    feats, self.element_indices, self.update_target,
                    self.inflate_max_rounds, self.inflate_max_factor,
                )
                n_inflate = int(res.n_rounds)
                inflate_factor = float(np.prod(res.factors)) if n_inflate else 1.0
        t_before = internal_temperature(self.atoms)
        update_work, work_removed = self._update_work(r, feats, gamma)  # frame is post-removal
        update_ms = 1e3 * (time.perf_counter() - t_update)  # alpha search + update + inflation
        frame = self.atoms.copy()
        frame.calc = None
        # connected components with the start-relative pair table; recorded, never filtered
        n_fragments = count_fragments(frame, self.fragment_mult)
        frame.arrays["gamma_live"] = gamma.copy()
        frame.arrays["gamma_pristine"] = g_pristine
        frame.arrays["grace_forces"] = np.asarray(r["grace_forces"]).copy()
        v_ratio, rg_ratio = self._shape_ratios()
        shape = {"V_V0": v_ratio} if self.periodic else {"Rg_Rg0": rg_ratio}
        # one record for the frame's info and for the summary's collections list
        rec = {
            "step": self.step, "gamma_max_live": float(gamma.max()),
            "gamma_max_pristine": float(g_pristine.max()), "alpha": float(alpha),
            "n_inflate": n_inflate, "inflate_factor": inflate_factor, "n_gamma_above": n_above,
            "update_work": update_work, "update_work_removed": work_removed,
            "n_fragments": n_fragments,
        }
        frame.info.update(
            rec, walker_id=self.walker_id, T=internal_temperature(self.atoms),
            T_before_removal=t_before, **shape, tau_r=r["tau_r"],
            grace_energy=r["grace_energy"], grace_stress=np.asarray(r["grace_stress"]).copy(),
            collection_index=self.n_collected,
        )
        write(self.frames_path, frame, format="extxyz", append=True)
        # after the frame is on disk: a failed write leaves no sidecar row behind
        if self.candidates is not None:  # pristine features of the candidate atoms
            self.candidates.add(self.n_collected, self.step, frame.numbers,
                                self.element_indices, feats, g_pristine)
        self.n_collected += 1
        self.collections.append({**rec, "update_ms": update_ms})
        if self.keep_frames:
            self.frames.append(frame)
        if self.live is not None and self.save_every and self.n_collected % self.save_every == 0:
            self.save_live()
        return alpha, update_work

    # -- outputs -----------------------------------------------------------
    def save_live(self):
        if self.live is not None:
            self.live.save(self.live_path)  # GMMUQModel.save is atomic

    def save_candidates(self):
        if self.candidates is not None:
            self.candidates.save()  # atomic

    def finalize(self, stop_reason: str, settings: dict | None = None) -> dict:
        self.history.save(self.history_path)
        self.save_candidates()
        self.save_live()
        wall = time.perf_counter() - self._t0
        try:
            from importlib.metadata import version

            tp_version = version("tensorpotential")
        except Exception:
            tp_version = "unknown"
        hist = self.history.as_dict()
        temps, gmax = hist["temperature"], hist["gamma_max"]
        summary = {
            "stop_reason": stop_reason,
            "n_steps": self.step,
            "n_collected": self.n_collected,
            "rejected": dict(self.rejected),
            "collections": self.collections,
            "temperature_mean_K": float(np.nanmean(temps)) if np.isfinite(temps).any() else float("nan"),
            # empty history: the first model call stopped the walker (e.g. non_finite)
            "gamma_max_live": float(np.nanmax(gmax)) if np.isfinite(gmax).any() else float("nan"),
            "update_work_total_eV": float(sum(c["update_work"] for c in self.collections)),
            "update_work_removed_eV": float(sum(c["update_work_removed"] for c in self.collections)),
            "wall_time_s": wall,
            "steps_per_s": self.step / wall if wall > 0 else float("nan"),
            "n_model_calls": getattr(self.calc, "n_base_calls", None),
            "tensorpotential_version": tp_version,
            "settings": settings or {},
        }
        # frames with more fragments than the starting structure (normally 1)
        summary["n_fragments0"] = self.n_fragments0
        summary["n_fragmented"] = int(sum(c["n_fragments"] > self.n_fragments0
                                          for c in self.collections))
        with open(self.summary_path, "w") as fh:
            json.dump(summary, fh, indent=2, default=str)
        return summary
