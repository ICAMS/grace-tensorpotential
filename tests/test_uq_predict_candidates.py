"""`grace_uq predict --features-gamma-min`: candidate-only feature columns, read by select
like a HAL pool. Model-free: a fake UQ calculator, a synthetic CALM artifact."""

import numpy as np
import pandas as pd
import pytest
from ase.build import bulk
from ase.calculators.calculator import Calculator, all_changes

from tensorpotential.uq import constants as uqc
from tensorpotential.uq.cli import predict as predict_mod
from tensorpotential.uq.cli import select as select_mod
from tensorpotential.uq.hal.features import file_sha256
from tests.test_uq_select_candidates import D, _write_artifact


class FakeUQ(Calculator):
    """gamma from atoms.arrays["g"]; features 3 * N(0, 1) seeded by the positions."""

    implemented_properties = ["energy", "forces", "stress"]

    def __init__(self):
        super().__init__()
        self.feature_shapes = []  # all-atom feature arrays handed out, per structure

    def calculate(self, atoms=None, properties=None, system_changes=all_changes):
        super().calculate(atoms, properties, system_changes)
        n = len(atoms)
        seed = int(abs(atoms.positions).sum() * 1000) % 2**32
        feats = 3.0 * np.random.default_rng(seed).standard_normal((n, D))
        self.feature_shapes.append(feats.shape)
        self.results = {
            "energy": 0.0, "forces": np.zeros((n, 3)), "stress": np.zeros(6),
            uqc.ATOMIC_GAMMA: np.asarray(
                atoms.arrays["g"] if "g" in atoms.arrays else atoms.arrays["gamma_pristine"],
                float),
            uqc.ATOMIC_SIGMA: np.ones(n), uqc.FEATURES: feats,
        }


def _dataset(n=12, seed=0):
    rng = np.random.default_rng(seed)
    rows = []
    for i in range(n):
        at = bulk("Cu", "fcc", a=3.6, cubic=True)
        at.positions += rng.normal(scale=0.05, size=at.positions.shape)
        g = np.full(4, 0.5)
        g[rng.choice(4, 2, replace=False)] = rng.uniform(1.0, 3.0, 2)
        at.arrays["g"] = g
        rows.append({"ase_atoms": at, "energy": 0.0})
    return pd.DataFrame(rows)


def test_predict_columns_keep_only_candidate_features():
    df = _dataset(4)
    calc = FakeUQ()
    cols = predict_mod._predict_columns(df["ase_atoms"].tolist(), calc, False, True,
                                        features_gamma_min=1.0)
    assert uqc.FEATURES not in cols  # no all-atom features column
    for i, at in enumerate(df["ase_atoms"]):
        idx = cols["cand_atom_index"][i]
        np.testing.assert_array_equal(idx, np.flatnonzero(at.arrays["g"] >= 1.0))
        assert cols["cand_features"][i].dtype == np.float32
        assert cols["cand_features"][i].shape == (len(idx), D)
        np.testing.assert_allclose(cols["cand_gamma"][i], at.arrays["g"][idx])
    both = predict_mod._predict_columns(df["ase_atoms"].tolist(), FakeUQ(), True, True,
                                        features_gamma_min=1.0)
    assert both[uqc.FEATURES][0].shape == (4, D) and "cand_features" in both


def _predict(tmp_path, art, *extra):
    raw = tmp_path / "raw.pkl.gz"
    _dataset().to_pickle(raw, compression="gzip")
    out = tmp_path / "pred.pkl.gz"
    assert predict_mod.predict_main([
        "--model", str(tmp_path), "--dataset", str(raw), "--output", str(out),
        "--n-workers", "1", "--artifact", str(art), *extra]) == 0
    return out


@pytest.fixture
def fake_model(monkeypatch):
    monkeypatch.setattr(predict_mod, "_build_calculator", lambda args: FakeUQ())


def test_predict_writes_candidate_columns_not_features(tmp_path, fake_model):
    art = _write_artifact(tmp_path / "a.npz")
    df = pd.read_pickle(_predict(tmp_path, art, "--features-gamma-min", "1.0"),
                        compression="gzip")
    assert uqc.FEATURES not in df.columns
    assert {"gamma", "cand_atom_index", "cand_features", "cand_gamma"} <= set(df.columns)
    assert (df["cand_feature_gamma_min"] == 1.0).all()
    assert (df["cand_artifact_sha256"] == file_sha256(art)).all()
    for at, idx in zip(df["ase_atoms"], df["cand_atom_index"]):  # rows keep input order
        np.testing.assert_array_equal(idx, np.flatnonzero(at.arrays["g"] >= 1.0))


def test_predict_select_round_trip(tmp_path, fake_model, capsys):
    art = _write_artifact(tmp_path / "a.npz")
    pred = _predict(tmp_path, art, "--features-gamma-min", "1.0")
    assert select_mod._is_candidate_df(pd.read_pickle(pred, compression="gzip"))

    def no_predict(args):
        raise AssertionError("select must not predict again")

    select_mod._maybe_run_predict, real = no_predict, select_mod._maybe_run_predict
    try:
        out = tmp_path / "sel.pkl.gz"
        for flag in ("--predicted", "--dataset"):
            assert select_mod.select_main([flag, str(pred), "--artifact", art, "-n", "3",
                                           "--no-element-stratified", "--output", str(out)]) == 0
            sel = pd.read_pickle(out, compression="gzip")
            assert 0 < len(sel) <= 3 and "gamma_at_pick" in sel.columns
            assert not [c for c in sel.columns if c.startswith("cand_")]
        assert "strategy='calm-greedy'" in capsys.readouterr().out
        with pytest.raises(SystemExit, match="feature_gamma_min"):
            select_mod.select_main(["--predicted", str(pred), "--artifact", art, "-n", "3",
                                    "--gamma-min", "0.5", "--output", str(out)])
        other = _write_artifact(tmp_path / "b.npz", seed=4)
        with pytest.raises(SystemExit, match="not the CALM artifact"):
            select_mod.select_main(["--predicted", str(pred), "--artifact", other, "-n", "3",
                                    "--output", str(out)])
    finally:
        select_mod._maybe_run_predict = real


def test_with_save_features_the_predicted_path_is_used(tmp_path, fake_model):
    art = _write_artifact(tmp_path / "a.npz")
    df = pd.read_pickle(_predict(tmp_path, art, "--features-gamma-min", "1.0",
                                 "--save-features"), compression="gzip")
    assert uqc.FEATURES in df.columns and "cand_features" in df.columns
    assert not select_mod._is_candidate_df(df)


def test_features_gamma_min_refuses_extxyz_output(tmp_path, capsys):
    with pytest.raises(SystemExit):
        predict_mod.predict_main(["--model", str(tmp_path), "--dataset", "x.pkl.gz",
                                  "--output", str(tmp_path / "p.extxyz"),
                                  "--features-gamma-min", "1.0"])
    assert "pkl.gz" in capsys.readouterr().err


def test_n_workers_auto_rules(monkeypatch):
    from tensorpotential.uq.cli import _common

    assert _common.auto_n_workers(27) == 1
    assert _common.auto_n_workers(200) == 1
    assert _common.auto_n_workers(201) == 2  # ceil(n / 200)
    assert _common.auto_n_workers(450) == 3
    assert _common.auto_n_workers(10_000) == 4  # at most 4
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "-1")
    assert _common.visible_gpu_ids() == []
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "0,2")
    assert _common.visible_gpu_ids() == ["0", "2"]
    assert _common.visible_gpu_ids("3") == ["3"]
    with pytest.raises(ValueError):
        _common.parse_n_workers("0")
    assert _common.parse_n_workers("AUTO") == "auto" and _common.parse_n_workers("3") == 3


def test_predict_auto_runs_one_worker_in_process(tmp_path, fake_model, monkeypatch):
    def no_spawn(args):
        raise AssertionError("one worker must run in this process")

    monkeypatch.setattr(predict_mod, "_spawn_workers", no_spawn)
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "-1")
    raw = tmp_path / "raw.pkl.gz"
    _dataset().to_pickle(raw, compression="gzip")
    out = tmp_path / "p.pkl.gz"
    assert predict_mod.predict_main(["--model", str(tmp_path), "--dataset", str(raw),
                                     "--output", str(out)]) == 0
    assert len(pd.read_pickle(out, compression="gzip")) == 12


@pytest.mark.parametrize("cvd,gpus", [("0,1,2,3", "0,1,2,3"), ("1", "1"), ("-1", "-1")])
def test_predict_auto_spreads_workers(tmp_path, monkeypatch, cvd, gpus):
    """480 structures: ceil(480 / 200) = 3 workers, round robin over the visible GPUs,
    on CPU ("-1") without one."""
    seen = {}

    def capture(args):
        seen.update(n=args.n_workers, gpus=args.gpus)
        raise SystemExit("stop")

    monkeypatch.setattr(predict_mod, "_spawn_workers", capture)
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", cvd)
    raw = tmp_path / "raw.pkl.gz"
    pd.concat([_dataset()] * 40, ignore_index=True).to_pickle(raw, compression="gzip")  # 480
    with pytest.raises(SystemExit, match="stop"):
        predict_mod.predict_main(["--model", str(tmp_path), "--dataset", str(raw),
                                  "--output", str(tmp_path / "p.pkl.gz")])
    assert seen == {"n": 3, "gpus": gpus}


def test_predict_auto_in_process_reads_dataset_once(tmp_path, fake_model, monkeypatch):
    calls = []
    real = predict_mod.load_dataset_any
    monkeypatch.setattr(predict_mod, "load_dataset_any",
                        lambda p, *a: calls.append(p) or real(p, *a))
    raw = tmp_path / "raw.pkl.gz"
    _dataset().to_pickle(raw, compression="gzip")
    assert predict_mod.predict_main(["--model", str(tmp_path), "--dataset", str(raw),
                                     "--output", str(tmp_path / "p.pkl.gz")]) == 0
    assert calls == [str(raw)]


def test_features_gamma_min_refuses_no_ase_atoms(tmp_path, capsys):
    with pytest.raises(SystemExit):
        predict_mod.predict_main(["--model", str(tmp_path), "--dataset", "x.pkl.gz",
                                  "--output", str(tmp_path / "p.pkl.gz"),
                                  "--features-gamma-min", "1.0", "--no-ase-atoms"])
    assert "--no-ase-atoms" in capsys.readouterr().err


@pytest.mark.parametrize("flag", ["--artifact", "--uq"])
def test_select_inline_predict_keeps_candidates_only(tmp_path, fake_model, flag):
    """select --dataset RAW with the artifact known: the inline prediction stores only
    candidate features and the artifact hash, which select's check then accepts (also
    for a SavedModel with --uq, where the artifact is recorded, not loaded)."""
    art = _write_artifact(tmp_path / "a.npz")
    raw = tmp_path / "raw.pkl.gz"
    _dataset().to_pickle(raw, compression="gzip")
    out = tmp_path / "sel.pkl.gz"
    assert select_mod.select_main(["--dataset", str(raw), "--model", str(tmp_path), flag, art,
                                   "-n", "3", "--no-element-stratified",
                                   "--output", str(out)]) == 0
    pred = pd.read_pickle(tmp_path / ".select_predicted.pkl.gz", compression="gzip")
    assert uqc.FEATURES not in pred.columns and "cand_features" in pred.columns
    assert (pred["cand_artifact_sha256"] == file_sha256(art)).all()
    assert (pred["cand_feature_gamma_min"] == 1.0).all()
    assert 0 < len(pd.read_pickle(out, compression="gzip")) <= 3


@pytest.mark.parametrize("strategy", ["fps-all"])
def test_select_inline_predict_all_atom_strategies_get_features(tmp_path, fake_model, strategy):
    art = _write_artifact(tmp_path / "a.npz")
    raw = tmp_path / "raw.pkl.gz"
    _dataset().to_pickle(raw, compression="gzip")
    out = tmp_path / "sel.pkl.gz"
    assert select_mod.select_main(["--dataset", str(raw), "--model", str(tmp_path),
                                   "--artifact", art, "--strategy", strategy, "-n", "3",
                                   "--no-element-stratified", "--output", str(out)]) == 0
    pred = pd.read_pickle(tmp_path / ".select_predicted.pkl.gz", compression="gzip")
    assert uqc.FEATURES in pred.columns and "cand_features" not in pred.columns
    assert len(pd.read_pickle(out, compression="gzip")) == 3


def test_pool_then_predict_features_gamma_min_then_select(tmp_path, fake_model, caplog):
    """A HAL pool already has cand_* columns: predict --features-gamma-min replaces them
    (one log line), no duplicate columns, and select works on the result."""
    from tensorpotential.uq.hal import pool_frames
    from tests.test_uq_select_candidates import _walkers

    art = _write_artifact(tmp_path / "a.npz")
    pool = tmp_path / "pool.pkl.gz"
    pool_frames(_walkers(tmp_path, art)).to_pickle(pool, compression="gzip")
    out = tmp_path / "pred.pkl.gz"
    with caplog.at_level("INFO", logger="grace_uq.predict"):
        assert predict_mod.predict_main([
            "--model", str(tmp_path), "--dataset", str(pool), "--output", str(out),
            "--artifact", str(art), "--features-gamma-min", "1.0"]) == 0
    assert "replaced by this prediction" in caplog.text and "cand_features" in caplog.text
    df = pd.read_pickle(out, compression="gzip")
    assert not df.columns.duplicated().any()
    sel = tmp_path / "sel.pkl.gz"
    assert select_mod.select_main(["--predicted", str(out), "--artifact", art, "-n", "2",
                                   "--output", str(sel)]) == 0
    assert len(pd.read_pickle(sel, compression="gzip")) == 2


def test_single_worker_device_pinning():
    pin = predict_mod.single_worker_device
    assert pin("1", ["1"], {}) == "1"  # --gpus names a GPU
    assert pin("2,3", ["2", "3"], {"CUDA_VISIBLE_DEVICES": "0"}) == "2"
    assert pin("-1", [], {}) == "-1"  # --gpus given, no GPU: CPU
    assert pin(None, ["0", "1"], {"CUDA_VISIBLE_DEVICES": "0,1"}) is None  # left alone
    assert pin(None, [], {"CUDA_VISIBLE_DEVICES": "-1"}) is None
    assert pin(None, ["0", "1"], {}) == "0"  # nothing set, nvidia-smi found GPUs
    assert pin(None, [], {}) is None  # CPU-only host


def test_auto_one_worker_sets_cuda_visible_devices(tmp_path, monkeypatch):
    seen = {}

    def capture(args):
        seen["cvd"] = predict_mod.os.environ.get("CUDA_VISIBLE_DEVICES")
        seen["n"] = args.n_workers
        raise SystemExit("stop")

    monkeypatch.setattr(predict_mod, "run_master", capture)
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "0")
    raw = tmp_path / "raw.pkl.gz"
    _dataset().to_pickle(raw, compression="gzip")
    with pytest.raises(SystemExit, match="stop"):
        predict_mod.predict_main(["--model", str(tmp_path), "--dataset", str(raw),
                                  "--output", str(tmp_path / "p.pkl.gz"), "--gpus", "-1"])
    assert seen == {"cvd": "-1", "n": 1}


def test_predict_without_artifact_refusal_suggests_predict(tmp_path, fake_model):
    art = _write_artifact(tmp_path / "a.npz")
    raw = tmp_path / "raw.pkl.gz"
    _dataset().to_pickle(raw, compression="gzip")
    pred = tmp_path / "pred.pkl.gz"
    assert predict_mod.predict_main(["--model", str(tmp_path), "--dataset", str(raw),
                                     "--output", str(pred), "--features-gamma-min", "1.0"]) == 0
    with pytest.raises(SystemExit, match=r"re-run `grace_uq predict .*--artifact"):
        select_mod.select_main(["--predicted", str(pred), "--artifact", art, "-n", "2",
                                "--output", str(tmp_path / "s.pkl.gz")])


def test_failed_prediction_rows_warning(tmp_path, fake_model, caplog):
    art = _write_artifact(tmp_path / "a.npz")
    df = pd.read_pickle(_predict(tmp_path, art, "--features-gamma-min", "1.0"),
                        compression="gzip")
    df.at[3, "cand_features"] = None  # as written for a structure whose prediction failed
    df.at[3, "cand_atom_index"] = None
    df.at[3, "cand_gamma"] = None
    select_mod._candidate_pool_from_df(df, ["Cu", "Al"], 1.0, None, art)
    assert "1 row(s) have no candidate data (the prediction failed for them)" in caplog.text


def test_subprocess_workers_forward_features_gamma_min(tmp_path, monkeypatch):
    seen = []

    def capture(cmds, **kw):
        seen.extend(cmds)
        return []

    monkeypatch.setattr(predict_mod, "spawn_and_monitor", capture)
    monkeypatch.setattr(predict_mod, "_merge_shards", lambda args: pd.DataFrame({"x": [1]}))
    monkeypatch.setattr(predict_mod, "save_dataset_any", lambda *a, **k: None)
    art = _write_artifact(tmp_path / "a.npz")
    assert predict_mod.predict_main(["--model", str(tmp_path), "--dataset", "d.pkl.gz",
                                     "--output", str(tmp_path / "p.pkl.gz"), "--n-workers", "2",
                                     "--artifact", art, "--features-gamma-min", "1.5"]) == 0
    assert [i for i, _ in seen] == [0, 1]
    for _, cmd in seen:
        assert cmd[cmd.index("--features-gamma-min") + 1] == "1.5"
        assert cmd[cmd.index("--artifact") + 1] == art and "--save-features" not in cmd


def test_two_workers_merge_keeps_order_with_candidate_columns(tmp_path, fake_model):
    """run_worker for ids 0 and 1, then _merge_shards: rows in input order, each row's
    candidate columns still belong to its own structure."""
    import argparse

    art = _write_artifact(tmp_path / "a.npz")
    raw = tmp_path / "raw.pkl.gz"
    df = _dataset(n=9)
    df.index = np.arange(9)[::-1] * 3  # a non-trivial input index
    df.to_pickle(raw, compression="gzip")
    args = predict_mod._build_parser().parse_args([
        "--model", str(tmp_path), "--dataset", str(raw), "--output", str(tmp_path / "p.pkl.gz"),
        "--n-workers", "2", "--artifact", art, "--features-gamma-min", "1.0"])
    predict_mod._resolve_paths(args)
    for w in (0, 1):
        assert predict_mod.run_worker(argparse.Namespace(**{**vars(args), "worker_id": w})) == 0
    merged = predict_mod._merge_shards(args)
    assert len(merged) == 9 and not merged.columns.duplicated().any()
    for at, ref, idx in zip(merged["ase_atoms"], df["ase_atoms"], merged["cand_atom_index"]):
        np.testing.assert_array_equal(at.positions, ref.positions)
        np.testing.assert_array_equal(idx, np.flatnonzero(ref.arrays["g"] >= 1.0))


def test_visible_gpu_ids_nvidia_smi_fallback(monkeypatch):
    import subprocess
    import types

    from tensorpotential.uq.cli import _common

    monkeypatch.delenv("CUDA_VISIBLE_DEVICES", raising=False)
    out = "GPU 0: NVIDIA T1000 (UUID: x)\nGPU 1: NVIDIA RTX 3060 (UUID: y)\n"
    monkeypatch.setattr(_common.subprocess, "run",
                        lambda *a, **k: types.SimpleNamespace(stdout=out))
    assert _common.visible_gpu_ids() == ["0", "1"]

    def missing(*a, **k):
        raise FileNotFoundError("nvidia-smi")

    monkeypatch.setattr(_common.subprocess, "run", missing)
    assert _common.visible_gpu_ids() == []

    def hang(*a, **k):
        raise subprocess.TimeoutExpired("nvidia-smi", 30)

    monkeypatch.setattr(_common.subprocess, "run", hang)
    assert _common.visible_gpu_ids() == []
