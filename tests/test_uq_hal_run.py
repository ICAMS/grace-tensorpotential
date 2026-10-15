"""run_hal end to end on the test model.

Every ensemble is run once (module-scoped, on demand) and several tests assert on the
same walker, because a walker costs ~10-20 s (per-walker XLA compile), not its steps.
"""

import json
import os

import numpy as np
import pytest
from ase import Atoms
from ase.io import read

from tensorpotential.uq.hal import DEFAULT_ALPHAS, run_hal

FILES = ("hal_collected.extxyz", "hal_live_artifacts.npz", "hal_history.npz",
         "hal_summary.json", "hal_settings.json", "hal_features.npz")
LANGEVIN = ("nvt", "nvt_ramp", "cluster")

# npt carries the non-default options (removal is off on MTK anyway): they must reach
# hal_settings.json and the records. update_target 1e-3 forces the alpha = 1 + inflation path.
OPTIONS = dict(trigger_min_atoms=2, update_target=1e-3, inflate_max_rounds=2,
               inflate_max_factor=3.0, remove_update_work=False, update_work_floor=0.25)


def _kw(setup, out, **over):
    kw = dict(model_yaml=setup["model_yaml"], checkpoint=setup["checkpoint"],
              uq_artifact=setup["artifact_path"], temperature_K=300, n_steps=30,
              equil_steps=2, trigger_gamma=0.0, trigger_gap=5, history_every=10,
              output_dir=str(out), calc_kwargs={"param_dtype": "float64"}, plot=False)
    kw.update(over)
    return kw


def _structure(setup):
    return max(setup["atoms"], key=len).copy()


def _cluster(setup):
    bulk_atoms = _structure(setup)
    return Atoms(bulk_atoms.get_chemical_symbols(), positions=bulk_atoms.positions)  # no cell, no pbc


@pytest.fixture(scope="module")
def walker(hal_uq_setup, tmp_path_factory):
    """walker(ensemble) -> (result, output dir, Langevin spies, caller's input Atoms); cached."""
    from tensorpotential.uq.hal import run as run_mod

    cache = {}

    def run(ensemble):
        if ensemble in cache:
            return cache[ensemble]
        out = tmp_path_factory.mktemp(f"walker_{ensemble}")
        built = []
        extra = {"bulk_modulus": 0.5}
        if ensemble in ("npt", "npt_stress"):
            # the random-init model has |P| ~ 1e3 GPa: 0.2 fs keeps MTK inside the guard
            extra["timestep_fs"] = 0.2
        if ensemble == "npt":
            extra.update(OPTIONS)
        if ensemble == "nvt":
            extra["plot"] = True  # the one walker that writes the PNG
        if ensemble == "cluster":
            extra["feature_gamma_min"] = 0.0  # every atom a candidate
        atoms = _cluster(hal_uq_setup) if ensemble == "cluster" else _structure(hal_uq_setup)
        given = atoms.copy()
        with pytest.MonkeyPatch.context() as mp:
            for name in ("Langevin", "RampedLangevin"):
                cls = getattr(run_mod, name)
                mp.setattr(run_mod, name, type(f"Spy{name}", (cls,), {
                    "__init__": lambda self, *a, _c=cls, **k: (_c.__init__(self, *a, **k), built.append(self))[0]
                }))
            res = run_hal(atoms, **_kw(hal_uq_setup, out, ensemble=ensemble, **extra))
        cache[ensemble] = (res, out, built, atoms, given)
        return cache[ensemble]

    return run


@pytest.mark.parametrize("ensemble", ["nvt", "nvt_ramp", "npt", "npt_stress", "cluster"])
def test_run_hal_smoke(walker, ensemble):
    res, out, built, _, _ = walker(ensemble)
    s = res.summary
    # the random-init model can push the npt_stress cell out of the guard band
    assert s["stop_reason"] == "max_steps" or s["stop_reason"].startswith("volume_guard")
    assert s["n_collected"] >= 1 and len(res.frames) == s["n_collected"]
    for f in FILES + (("hal_diagnostics.png",) if ensemble == "nvt" else ()):
        assert os.path.exists(out / f), f
    assert len(read(str(out / "hal_collected.extxyz"), ":")) == s["n_collected"]
    # exactly one base-model call per collector call (step 0 included); a volume-guard stop
    # happens after the step's model call and before its collector call
    extra = 1 if s["stop_reason"].startswith("volume_guard") else 0
    assert s["n_model_calls"] == s["n_steps"] + extra
    assert s["n_fragments0"] == 1 and s["settings"]["fragment_mult"] == 1.25
    assert s["settings"]["reinvert_s"] >= 0.0  # both towers re-inverted at set-up
    assert "n_fragmented" in s and all("n_fragments" in f.info for f in res.frames)
    assert all(k in f.info for f in res.frames for k in ("update_work", "update_work_removed"))
    assert np.isfinite(s["temperature_mean_K"])


@pytest.mark.parametrize("ensemble", LANGEVIN)
def test_langevin_fixcm_false_com_reset_and_removal(walker, ensemble):
    res, _, built, _, _ = walker(ensemble)
    assert len(built) == 1 and built[0].fix_com is False
    for f in res.frames:  # frames are written after the per-step COM reset
        assert np.abs(f.get_momenta().sum(axis=0)).max() < 1e-10
    s = res.summary["settings"]
    assert s["remove_update_work"] is True and s["update_work_removal_active"] is True
    assert s["alphas"] == list(DEFAULT_ALPHAS)


@pytest.mark.parametrize("ensemble", ["nvt", "cluster"])
def test_feature_sidecar_matches_frames(walker, hal_uq_setup, ensemble):
    from tensorpotential.uq.hal.features import file_sha256, read_features

    res, out, _, _, _ = walker(ensemble)
    sc = read_features(str(out))
    fgm = 0.0 if ensemble == "cluster" else 1.0
    assert sc["feature_gamma_min"] == fgm == res.summary["settings"]["feature_gamma_min"]
    assert sc["artifact_sha256"] == file_sha256(hal_uq_setup["artifact_path"])
    assert sc["model_yaml"] == os.path.abspath(hal_uq_setup["model_yaml"])
    frames = read(str(out / "hal_collected.extxyz"), ":")
    assert len(sc["n_atoms"]) == len(frames)
    for k, f in enumerate(frames):
        assert f.info["collection_index"] == k and sc["collection_step"][k] == f.info["step"]
        rows = sc["collection"] == k
        gp = f.arrays["gamma_pristine"]
        np.testing.assert_array_equal(sc["atom"][rows], np.flatnonzero(gp >= fgm))
        np.testing.assert_allclose(sc["gamma_pristine"][rows], gp[sc["atom"][rows]], rtol=1e-6)
        np.testing.assert_array_equal(sc["Z"][rows], f.numbers[sc["atom"][rows]])
    if ensemble == "cluster":
        assert len(sc["atom"]) == sum(len(f) for f in frames) > 0
        assert sc["features"].shape[1] == sc["feature_dim"] > 0


def test_npt_options_reach_settings_and_records(walker):
    """MTKNPT keeps its own momentum copy, so the removal is off; options reach the records."""
    res, out, built, _, _ = walker("npt")
    s = res.summary
    assert built == []  # no Langevin
    st = s["settings"]
    assert st["update_work_removal_active"] is False and s["update_work_removed_eV"] == 0.0
    with open(out / "hal_settings.json") as fh:
        on_disk = json.load(fh)
    for key, val in OPTIONS.items():
        assert st[key] == val and on_disk[key] == val, key
    assert on_disk["alphas"] == list(DEFAULT_ALPHAS)
    for c in s["collections"]:
        assert c["alpha"] == 1.0 and c["n_inflate"] == 2 and 1.0 < c["inflate_factor"] <= 9.0
        assert c["n_gamma_above"] >= 2
    assert read(str(out / "hal_collected.extxyz"), 0).info["n_inflate"] == 2
    assert all(f.info["n_gamma_above"] >= 2 for f in res.frames)


def test_mtk_turns_update_work_removal_off(walker):
    """npt_stress keeps the default remove_update_work=True; MTKNPT discards momenta set on
    the atoms, so run_hal turns the removal off (the rule, not a user option)."""
    res, _, _, _, _ = walker("npt_stress")
    st = res.summary["settings"]
    assert st["remove_update_work"] is True and st["update_work_removal_active"] is False
    assert res.summary["update_work_removed_eV"] == 0.0


def test_cluster_wall_and_callers_atoms_untouched(walker):
    res, _, _, atoms, given = walker("cluster")
    s = res.summary
    assert s["settings"]["cluster_wall_radius"] > 2.0
    h = res.history
    assert np.isfinite(h["rg_ratio"]).all() and np.isnan(h["v_ratio"]).all()
    # the base calculator pads non-periodic structures in place; run_hal must not let that
    # reach the caller's Atoms
    assert not atoms.pbc.any() and atoms.cell.rank == 0
    np.testing.assert_array_equal(atoms.positions, given.positions)


def test_live_start_loads_the_live_tower(walker, hal_uq_setup, tmp_path):
    """The chained walker's step-0 gamma is the gamma under the first walker's live tower."""
    from tensorpotential.uq.gmmuq import GMMUQModel

    first, out, _, _, _ = walker("npt")  # update_target 1e-3 + inflation: a tower far from training
    live = str(out / "hal_live_artifacts.npz")
    second = run_hal(_structure(hal_uq_setup),
                     **_kw(hal_uq_setup, tmp_path / "w1", n_steps=5, live_start_artifact=live,
                           seed=1))
    assert second.summary["settings"]["live_start_artifact"] == live
    atoms = hal_uq_setup["atoms"]
    k = max(range(len(atoms)), key=lambda i: len(atoms[i]))
    start = sum(len(a) for a in atoms[:k])
    sl = slice(start, start + len(atoms[k]))
    feats = np.asarray(hal_uq_setup["features"])[sl]
    eidx = np.asarray(hal_uq_setup["element_indices"])[sl]
    stored = GMMUQModel(live)  # the saved (float32) inverses
    reinverted = GMMUQModel(live)
    reinverted.reinvert_all()  # what run_hal does at set-up, before the first model call
    g_stored = float(np.max(stored.gamma_from_features(feats, eidx)[0]))
    g_live = float(np.max(reinverted.gamma_from_features(feats, eidx)[0]))
    g_train = float(np.max(GMMUQModel(hal_uq_setup["artifact_path"]).gamma_from_features(feats, eidx)[0]))
    assert g_live < g_train * (1 - 1e-2)  # the first walker's updates changed the tower
    # step 0 (the first model call) already sees the re-inverted live tower
    assert second.history["gamma_max"][0] == pytest.approx(g_live, rel=1e-6)
    assert second.history["gamma_max"][0] != pytest.approx(g_stored, rel=1e-6)


def test_run_hal_finalizes_on_non_finite(hal_uq_setup, tmp_path, monkeypatch):
    from tensorpotential.uq.hal import calculator as calc_mod

    original = calc_mod.HALCalculator.calculate
    calls = {"n": 0}

    def flaky(self, *a, **k):
        calls["n"] += 1
        if calls["n"] == 6:
            raise calc_mod.HALStop("non_finite")
        return original(self, *a, **k)

    monkeypatch.setattr(calc_mod.HALCalculator, "calculate", flaky)
    res = run_hal(_structure(hal_uq_setup), **_kw(hal_uq_setup, tmp_path / "w0", n_steps=20))
    assert res.summary["stop_reason"] == "non_finite"
    assert os.path.exists(tmp_path / "w0" / "hal_summary.json")


def test_run_hal_failed_setup_allows_rerun(hal_uq_setup, tmp_path):
    bad = _structure(hal_uq_setup)
    bad[0].symbol = "Al"
    kw = _kw(hal_uq_setup, tmp_path / "w0", n_steps=5)
    with pytest.raises(ValueError, match="Al"):
        run_hal(bad, **kw)
    run_hal(_structure(hal_uq_setup), **kw)  # no overwrite needed


def test_hand_composed_collector_writes_run_hal_hash(walker, hal_uq_setup, tmp_path):
    """A HALCollector built by hand, with a file-loaded pristine tower and no
    feature_meta, writes the same artifact hash as run_hal."""
    from ase.build import bulk

    from tensorpotential.uq.gmmuq import GMMUQModel
    from tensorpotential.uq.hal import HALCollector
    from tensorpotential.uq.hal.features import read_features
    from tests.hal_fakes import FakeHALCalc

    _, out, _, _, _ = walker("nvt")
    pristine = GMMUQModel(hal_uq_setup["artifact_path"])
    assert pristine.source_path == os.path.abspath(hal_uq_setup["artifact_path"])
    atoms = bulk("Cu", "fcc", a=3.6, cubic=True)
    col = HALCollector(atoms, FakeHALCalc(len(atoms)), pristine=pristine,
                       element_indices=np.zeros(len(atoms), int), output_dir=str(tmp_path / "h"),
                       update="none", remove_update_work=False)
    col.finalize("max_steps")
    sha = read_features(str(out))["artifact_sha256"]
    assert sha and read_features(str(tmp_path / "h"))["artifact_sha256"] == sha
    with open(out / "hal_settings.json") as fh:
        assert json.load(fh)["uq_artifact_sha256"] == sha
