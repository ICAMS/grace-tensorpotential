import json
import os

import itertools

import numpy as np
import pytest
from ase import Atoms
from ase.build import bulk
from ase.io import read

from tensorpotential.uq.hal.calculator import HALStop
from tensorpotential.uq.hal.collector import DEFAULT_ALPHAS, HALCollector, History
from tests.hal_fakes import FAKE_META, FakeHALCalc, FakeLive, FakePristine

GRID = (0.05, 0.1, 0.2, 0.4, 0.7, 1.0)  # alpha grid of _setup, one FakeLive trial value each


_DIRS = itertools.count()  # every collector its own output dir (frames append)


def _setup(tmp_path, trial=(2.0, 1.3, 1.1, 0.9, 0.5, 0.4), pristine=0.5, **kw):
    atoms = bulk("Cu", "fcc", a=3.6, cubic=True)
    live = FakeLive(trial)
    calc = FakeHALCalc(len(atoms), live)
    col = HALCollector(atoms, calc, pristine=FakePristine(pristine),
                       element_indices=np.zeros(len(atoms), int),
                       output_dir=str(tmp_path / f"w{next(_DIRS)}"),
                       **{"temperature_K": 300.0, "alphas": GRID,
                          "feature_meta": {"artifact_sha256": "fake"}, **kw})
    return atoms, calc, live, col


def _steps(col, calc, gammas):
    for g in gammas:
        calc.gamma = np.full(calc.n, g)
        col()


def _step_vec(col, calc, gamma):
    calc.gamma = np.asarray(gamma, dtype=float)
    col()


def test_trigger_equilibration_threshold_and_gap(tmp_path):
    _, calc, _, col = _setup(tmp_path, equil_steps=3, trigger_gap=1)
    _steps(col, calc, [5.0, 5.0, 5.0])  # equilibration: no collection
    assert col.n_collected == 0
    _steps(col, calc, [5.0])
    assert col.n_collected == 1
    _, calc, _, col = _setup(tmp_path, equil_steps=0, trigger_gamma=1.5, trigger_gap=3)
    _steps(col, calc, [1.4, 1.6, 1.6, 1.6, 1.6])  # collect at step 1, next allowed at 4
    assert [c["step"] for c in col.collections] == [1, 4]


def test_equil_steps_default_zero_collects_from_the_first_step(tmp_path):
    """Default equil_steps=0 (user decision 2026-10-06): collection may start at step 0; the
    bias still waits for the EMAs (tau_ema), which is the calculator's business."""
    import inspect

    from tensorpotential.uq.hal import run_hal
    from tensorpotential.uq.cli.hal import _build_parser

    assert inspect.signature(HALCollector).parameters["equil_steps"].default == 0
    assert inspect.signature(run_hal).parameters["equil_steps"].default == 0
    assert _build_parser().parse_args([]).equil_steps == 0
    kw = {"trigger_gap": 1}
    _, calc, _, col = _setup(tmp_path, **kw)
    _steps(col, calc, [5.0])
    assert [c["step"] for c in col.collections] == [0]


def test_trigger_min_atoms(tmp_path):
    _, calc, live, col = _setup(tmp_path, equil_steps=0, trigger_gap=1, trigger_gamma=1.5,
                                trigger_min_atoms=3, reject_gamma=10)
    _step_vec(col, calc, [2.0, 2.0, 1.0, 1.0])  # 2 above: no trigger
    _step_vec(col, calc, [1.5, 1.5, 1.5, 3.0])  # gamma == trigger_gamma does not count
    _step_vec(col, calc, [2.0, 2.0, 2.0, 1.0])  # 3 above: trigger
    assert [c["step"] for c in col.collections] == [2]
    h = col.history.as_dict()
    np.testing.assert_array_equal(h["n_gamma_above"], [2, 1, 3])
    np.testing.assert_array_equal(h["status"], [0, 0, 1])
    # rejection stays on max gamma, not on the count
    _step_vec(col, calc, [20.0, 2.0, 2.0, 1.0])
    assert col.rejected == {"gamma": 1} and len(live.updates) == 1


def test_rejection_by_gamma_and_bond(tmp_path):
    _, calc, live, col = _setup(tmp_path, equil_steps=0, trigger_gap=1, reject_gamma=10,
                                pristine=12.0)
    _steps(col, calc, [20.0])  # live gamma wins over gamma_pristine
    assert col.rejected == {"gamma": 1}
    _, calc, live, col = _setup(tmp_path, equil_steps=0, trigger_gap=1, reject_gamma=10)
    calc.ratio = 0.9
    _steps(col, calc, [2.0])
    assert col.n_collected == 0 and col.rejected == {"short_bond": 1}
    assert live.updates == []
    assert not os.path.exists(col.frames_path)


def test_rejection_by_pristine_gamma(tmp_path):
    _, calc, live, col = _setup(tmp_path, equil_steps=0, trigger_gap=3, reject_gamma=10,
                                pristine=12.0)
    _steps(col, calc, [2.0] * 4)  # live gamma is fine; rejected at 0, gap window until 3
    assert col.n_collected == 0
    assert col.rejected == {"gamma_pristine": 2}
    assert live.updates == [] and live.inflations == []
    assert not os.path.exists(col.frames_path)
    np.testing.assert_array_equal(col.history.as_dict()["status"], [2, 0, 0, 2])


def test_history_records_pristine_gamma_at_triggers_only(tmp_path):
    _, calc, _, col = _setup(tmp_path, equil_steps=0, trigger_gap=1, reject_gamma=10,
                             pristine=0.8)
    _steps(col, calc, [1.0, 2.0, 1.0])  # collected at step 1
    calc.ratio = 0.9
    _steps(col, calc, [2.0])  # rejected (short bond) at step 3
    h = col.history.as_dict()
    np.testing.assert_array_equal(h["status"], [0, 1, 0, 2])
    np.testing.assert_allclose(h["gamma_pristine_max"], [np.nan, 0.8, np.nan, 0.8])


def test_alpha_search_picks_smallest_alpha_below_target(tmp_path):
    _, calc, live, col = _setup(tmp_path, equil_steps=0, update_target=1.25)
    _steps(col, calc, [2.0])
    assert live.updates == [GRID[2]]  # trial[2] = 1.1 < 1.25
    assert live.targets == [1.25]  # the collector asks trial_gamma to stop early
    assert live.modes == [("trial", "additive"), ("update", "additive")]
    assert live.inflations == []
    f = read(col.frames_path, 0)
    assert f.info["n_inflate"] == 0 and f.info["inflate_factor"] == 1.0
    assert col.collections[0]["n_inflate"] == 0 and col.collections[0]["inflate_factor"] == 1.0


def test_inflate_after_alpha_one_fallback(tmp_path):
    _, calc, live, col = _setup(tmp_path, trial=(5, 5, 5, 5, 5, 5), equil_steps=0,
                                update_target=1.25, inflate_max_rounds=2,
                                inflate_max_factor=3.0)
    _steps(col, calc, [2.0])
    assert live.updates == [GRID[-1]]
    assert live.inflations == [(1.25, 2, 3.0)]
    f = read(col.frames_path, 0)
    assert f.info["n_inflate"] == 2 and f.info["inflate_factor"] == 8.0
    c = col.collections[0]
    assert c["n_inflate"] == 2 and c["inflate_factor"] == 8.0 and c["alpha"] == 1.0
    assert col.history.as_dict()["alpha"][0] == 1.0


def test_inflation_gate(tmp_path):
    """Inflate only when no alpha met the target, inflation is on and there is an update."""
    _, calc, live, col = _setup(tmp_path, trial=(5, 5, 5, 5, 5, 1.0), equil_steps=0)
    _steps(col, calc, [2.0])  # alpha = 1 itself met the target
    assert live.updates == [1.0] and live.inflations == []
    _, calc, live, col = _setup(tmp_path, trial=(5, 5, 5, 5, 5, 5), equil_steps=0,
                                update_inflate=False)
    _steps(col, calc, [2.0])
    assert live.updates == [1.0] and live.inflations == []
    assert col.collections[0]["n_inflate"] == 0 and col.collections[0]["inflate_factor"] == 1.0
    # alphas not ending in 1.0 are allowed when inflation is off
    _, calc, live, col = _setup(tmp_path, trial=(5, 5), alphas=(0.1, 0.5),
                                update_inflate=False, equil_steps=0)
    _steps(col, calc, [2.0])
    assert live.updates == [0.5] and live.inflations == []
    atoms = bulk("Cu", "fcc", a=3.6, cubic=True)
    calc = FakeHALCalc(len(atoms), None)
    col = HALCollector(atoms, calc, pristine=FakePristine(), feature_meta=FAKE_META,
                       element_indices=np.zeros(len(atoms), int),
                       output_dir=str(tmp_path / "n"), equil_steps=0, update="none",
                       temperature_K=300.0)
    _steps(col, calc, [2.0])
    assert col.collections[0]["n_inflate"] == 0


def test_frames_carry_grace_keys_not_calculator(tmp_path):
    _, calc, _, col = _setup(tmp_path, equil_steps=0, trigger_gap=1, trigger_gamma=1.5)
    _step_vec(col, calc, [1.0, 1.0, 1.0, 1.0])
    _step_vec(col, calc, [2.0, 2.0, 1.0, 1.0])
    _step_vec(col, calc, [2.0, 2.0, 2.0, 2.0])
    frames = read(col.frames_path, ":")
    assert len(frames) == 2
    f = frames[0]
    assert f.calc is None  # predictions never look like reference labels
    np.testing.assert_allclose(f.arrays["gamma_live"], [2.0, 2.0, 1.0, 1.0])
    np.testing.assert_allclose(f.arrays["gamma_pristine"], 0.5)
    assert "grace_forces" in f.arrays
    assert f.info["grace_energy"] == -1.0 and f.info["step"] == 1
    np.testing.assert_array_equal(col.history.as_dict()["n_gamma_above"], [0, 2, 4])
    assert [fr.info["n_gamma_above"] for fr in frames] == [2, 4]
    summary = col.finalize("max_steps")
    assert [c["n_gamma_above"] for c in summary["collections"]] == [2, 4]
    assert all(c["update_ms"] >= 0.0 for c in summary["collections"])


def test_n_collect_stops(tmp_path):
    _, calc, _, col = _setup(tmp_path, equil_steps=0, trigger_gap=1, n_collect=2)
    calc.gamma = np.full(calc.n, 2.0)
    col()
    with pytest.raises(HALStop) as exc:
        col()
    assert exc.value.reason == "n_collect"


def test_collector_constructor_errors(tmp_path):
    atoms = bulk("Cu", "fcc", a=3.6, cubic=True)
    kw = dict(pristine=FakePristine(), feature_meta=FAKE_META, element_indices=np.zeros(len(atoms), int),
              output_dir=str(tmp_path / "w"))
    with pytest.raises(ValueError, match="SavedModel"):
        HALCollector(atoms, FakeHALCalc(len(atoms), None), update="alpha_search",
                     temperature_K=300.0, **kw)
    calc = FakeHALCalc(len(atoms), FakeLive((0.5,) * 11))
    with pytest.raises(ValueError, match="temperature_K"):
        HALCollector(atoms, calc, **kw)
    col = HALCollector(atoms, calc, remove_update_work=False, **kw)  # fine without a target T
    assert col.alphas == DEFAULT_ALPHAS


def test_collector_frames_survive_exception(tmp_path):
    _, calc, _, col = _setup(tmp_path, equil_steps=0, trigger_gap=1, save_every=1)
    _steps(col, calc, [2.0, 2.0])
    # walker "killed" here: no finalize()
    assert len(read(col.frames_path, ":")) == 2
    assert os.path.exists(col.live_path)
    assert not [p for p in os.listdir(os.path.dirname(col.live_path)) if ".tmp" in p]


def test_finalize_writes_history_summary_live(tmp_path):
    _, calc, _, col = _setup(tmp_path, equil_steps=0, trigger_gap=1)
    _steps(col, calc, [1.0, 2.0, 1.0])
    summary = col.finalize("max_steps", settings={"trigger_gamma": 1.5})
    with open(col.summary_path) as fh:
        on_disk = json.load(fh)
    assert on_disk["stop_reason"] == "max_steps" == summary["stop_reason"]
    assert on_disk["n_collected"] == 1 and on_disk["n_steps"] == 3
    hist = np.load(col.history_path)
    np.testing.assert_array_equal(hist["status"], [0, 1, 0])
    assert os.path.exists(col.live_path)


def test_cluster_history_uses_radius_of_gyration(tmp_path):
    atoms = Atoms("Cu3", positions=[[0, 0, 0], [2.5, 0, 0], [0, 2.5, 0]])
    calc = FakeHALCalc(len(atoms), FakeLive((0.5,) * 11))
    col = HALCollector(atoms, calc, pristine=FakePristine(), feature_meta=FAKE_META,
                       element_indices=np.zeros(3, int), output_dir=str(tmp_path / "c"),
                       equil_steps=0, trigger_gap=1, temperature_K=300.0)
    calc.gamma = np.full(3, 2.0)
    col()
    atoms.positions *= 1.1
    col()
    h = col.history.as_dict()
    np.testing.assert_allclose(h["rg_ratio"], [1.0, 1.1])
    assert np.isnan(h["v_ratio"]).all() and np.isnan(h["pressure_gpa"]).all()
    assert "Rg_Rg0" in read(col.frames_path, 0).info


def test_history_grows_beyond_capacity():
    h = History(capacity=2)
    for k in range(5):
        h.append(step=k, gamma_max=float(k))
    d = h.as_dict()
    np.testing.assert_array_equal(d["step"], [0, 1, 2, 3, 4])
    assert np.isnan(d["tau_r"]).all()


# -- candidate-feature sidecar (hal_features.npz) -------------------------------------


def test_feature_sidecar_rows_link_to_frames(tmp_path):
    from tensorpotential.uq.hal.features import FEATURES_FILE, read_features

    g_pristine = [0.5, 1.0, 2.0, 0.9]  # feature_gamma_min 1.0 is inclusive: atoms 1, 2
    _, calc, _, col = _setup(tmp_path, equil_steps=0, trigger_gap=1, pristine=g_pristine,
                             history_every=2, feature_gamma_min=1.0,
                             feature_meta={"artifact_sha256": "abc", "model_yaml": "m.yaml"})
    feats = [np.arange(16.0).reshape(4, 4), -np.arange(16.0).reshape(4, 4)]
    path = os.path.join(col.output_dir, FEATURES_FILE)
    calc.features = feats[0]
    _steps(col, calc, [2.0])
    assert not os.path.exists(path)  # written at history saves and finalize only
    calc.features = feats[1]
    _steps(col, calc, [2.0])  # step 2: history save
    sc = read_features(path)
    np.testing.assert_array_equal(sc["collection"], [0, 0, 1, 1])
    col.finalize("max_steps")
    sc = read_features(col.output_dir)
    np.testing.assert_array_equal(sc["atom"], [1, 2, 1, 2])
    np.testing.assert_array_equal(sc["Z"], [29] * 4)
    np.testing.assert_array_equal(sc["element_index"], [0] * 4)
    np.testing.assert_allclose(sc["gamma_pristine"], [1.0, 2.0, 1.0, 2.0])
    assert sc["features"].dtype == np.float32 and sc["features"].shape == (4, 4)
    np.testing.assert_array_equal(sc["features"], np.concatenate([f[[1, 2]] for f in feats]))
    np.testing.assert_array_equal(sc["n_atoms"], [4, 4])
    np.testing.assert_array_equal(sc["collection_step"], [0, 1])
    np.testing.assert_array_equal(sc["species_Z"], [29])
    np.testing.assert_array_equal(sc["elem_counts"], [[4], [4]])
    assert sc["feature_gamma_min"] == 1.0 and sc["feature_dim"] == 4
    assert sc["artifact_sha256"] == "abc" and sc["model_yaml"] == "m.yaml"
    frames = read(col.frames_path, ":")
    assert [f.info["collection_index"] for f in frames] == [0, 1]
    for row in range(4):  # sidecar row -> frame -> same atom
        f = frames[sc["collection"][row]]
        assert f.info["step"] == sc["collection_step"][sc["collection"][row]]
        assert f.arrays["gamma_pristine"][sc["atom"][row]] == sc["gamma_pristine"][row]


def test_feature_sidecar_off_and_validation(tmp_path):
    from tensorpotential.uq.hal.collector import check_collector_options

    _, calc, _, col = _setup(tmp_path, equil_steps=0, trigger_gap=1, feature_gamma_min=None)
    _steps(col, calc, [2.0])
    col.finalize("max_steps")
    assert not os.path.exists(os.path.join(col.output_dir, "hal_features.npz"))
    assert read(col.frames_path, 0).info["collection_index"] == 0
    for bad in (-0.1, float("nan"), "1.0", True):
        with pytest.raises(ValueError, match="feature_gamma_min"):
            check_collector_options(feature_gamma_min=bad)


def test_failed_frame_write_leaves_no_sidecar_row(tmp_path, monkeypatch):
    from tensorpotential.uq.hal import collector as col_mod
    from tensorpotential.uq.hal.features import read_features

    _, calc, _, col = _setup(tmp_path, equil_steps=0, trigger_gap=1, pristine=2.0)
    real = col_mod.write

    def full_disk(*a, **k):
        raise OSError("disk full")

    monkeypatch.setattr(col_mod, "write", full_disk)
    with pytest.raises(OSError):
        _steps(col, calc, [2.0])
    monkeypatch.setattr(col_mod, "write", real)
    _steps(col, calc, [2.0, 2.0])  # trigger gap; then collection index 0 again
    col.finalize("max_steps")
    sc = read_features(col.output_dir)
    frames = read(col.frames_path, ":")
    assert len(frames) == len(sc["n_atoms"]) == 1
    assert frames[0].info["collection_index"] == 0


def test_sidecar_hash_from_pristine_source_path(tmp_path):
    from tensorpotential.uq.hal.features import file_sha256, read_features

    art = tmp_path / "art.npz"
    art.write_bytes(b"not really an artifact")
    pristine = FakePristine()
    pristine.source_path = str(art)
    atoms = bulk("Cu", "fcc", a=3.6, cubic=True)
    kw = dict(element_indices=np.zeros(4, int), update="none", remove_update_work=False)
    col = HALCollector(atoms, FakeHALCalc(4), pristine=pristine,
                       output_dir=str(tmp_path / "a"), **kw)
    col.finalize("max_steps")
    assert read_features(str(tmp_path / "a"))["artifact_sha256"] == file_sha256(str(art))
    # an explicit hash wins; no sidecar: no hashing, no warning
    col = HALCollector(atoms, FakeHALCalc(4), pristine=pristine, output_dir=str(tmp_path / "b"),
                       feature_meta={"artifact_sha256": "given"}, **kw)
    col.finalize("max_steps")
    assert read_features(str(tmp_path / "b"))["artifact_sha256"] == "given"
    # built in memory: warned once, empty hash
    with pytest.warns(UserWarning, match="no artifact hash") as rec:
        col = HALCollector(atoms, FakeHALCalc(4), pristine=FakePristine(),
                           output_dir=str(tmp_path / "c"), **kw)
    assert len([w for w in rec if "no artifact hash" in str(w.message)]) == 1
    col.finalize("max_steps")
    assert read_features(str(tmp_path / "c"))["artifact_sha256"] == ""


def test_history_file_is_compressed_float32(tmp_path):
    """hal_history.npz is for diagnostics: float32 (step: exact int64), zip-compressed;
    memory stays float64."""
    import zipfile

    from tensorpotential.uq.hal.collector import HISTORY_FIELDS

    h = History(capacity=4)
    rng = np.random.default_rng(0)
    for i in range(2000):
        h.append(step=i, gamma_max=float(rng.random()), temperature=300.0 + rng.random())
    path = str(tmp_path / "hal_history.npz")
    h.save(path)
    with zipfile.ZipFile(path) as z:
        assert all(info.compress_type == zipfile.ZIP_DEFLATED for info in z.infolist())
    with np.load(path) as f:
        assert set(f.files) == set(HISTORY_FIELDS)
        assert f["step"].dtype == np.int64
        assert all(f[k].dtype == np.float32 for k in f.files if k != "step")
        assert np.array_equal(f["step"], np.arange(2000, dtype=np.float32))
        assert np.allclose(f["gamma_max"], h.as_dict()["gamma_max"], rtol=1e-6)
    assert h.as_dict()["gamma_max"].dtype == np.float64


def test_summary_has_the_run_max_gamma_live(tmp_path):
    atoms = bulk("Cu", "fcc", a=3.6, cubic=True)
    calc = FakeHALCalc(len(atoms), FakeLive((0.5,) * 11))
    col = HALCollector(atoms, calc, pristine=FakePristine(), feature_meta=FAKE_META,
                       element_indices=np.zeros(len(atoms), int), output_dir=str(tmp_path / "w"),
                       equil_steps=0, trigger_gap=1, temperature_K=300.0)
    for g in (0.4, 2.0, 0.7):
        calc.gamma = np.full(len(atoms), g)
        col()
    s = col.finalize("max_steps")
    assert s["gamma_max_live"] == pytest.approx(2.0)
