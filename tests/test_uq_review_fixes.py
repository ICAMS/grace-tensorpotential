"""Regression tests for the code-review findings on the UQ/HAL branch."""

import copy
import os
import sys

import numpy as np
import pytest
from ase.build import bulk

import tensorpotential.uq.hal.walkers as walkers
from tensorpotential.uq import constants as uqc
from tensorpotential.uq.artifact_builder import GMMUQArtifactBuilder
from tensorpotential.uq.cli import predict as predict_mod
from tensorpotential.uq.gmmuq import GMMUQModel
from tensorpotential.uq.hal.collector import History
from tests.test_uq_hal_walkers import FAKE_WALKER, MODEL_KW

D, K = 4, 2


def _element_arts(seed):
    rng = np.random.default_rng(seed)
    a = rng.standard_normal((K, D, D))
    cov = np.einsum("kij,klj->kil", a, a) / D + 0.5 * np.eye(D)
    counts = np.full(K, 50, dtype=np.int64)
    return {
        uqc.CENTROIDS: 3.0 * rng.standard_normal((K, D)),
        uqc.INV_COV: np.linalg.inv(cov + 1e-6 * np.eye(D)),
        uqc.COUNTS: counts,
        uqc.EFFECTIVE_COUNT: counts.astype(np.float64),
        uqc.SCATTER: cov * 50.0,
    }


def _frame(n=6, elements=(0, 1)):
    rng = np.random.default_rng(7)
    eidx = np.array([elements[i % len(elements)] for i in range(n)], dtype=np.int32)
    return 4.0 * rng.standard_normal((n, D)), eidx


def test_trial_gamma_skips_element_without_training_data(tmp_path):
    """Element 1 is in the element map but has no UQ training data: trial_gamma must
    skip it like update() and inflate() do, not raise KeyError."""
    path = str(tmp_path / "a.npz")
    GMMUQArtifactBuilder.save_artifacts(
        path, {0: _element_arts(0)}, interp_thresholds=np.ones((2, K)),
        element_map=np.array(["Cu", "Al"]))
    with pytest.warns(UserWarning):
        gmm = GMMUQModel(path)
    feats, eidx = _frame()
    got = gmm.trial_gamma(feats, eidx, [0.5, 1.0])
    only0 = gmm.trial_gamma(feats[eidx == 0], eidx[eidx == 0], [0.5, 1.0])
    np.testing.assert_allclose(got, only0)


def test_from_artifacts_does_not_modify_the_callers_arrays():
    arts = {e: _element_arts(e) for e in (0, 1)}
    before = copy.deepcopy(arts)
    gmm = GMMUQModel.from_artifacts(arts, extra_data={"interp_thresholds": np.ones((2, K))})
    feats, eidx = _frame()
    gmm.update_one(feats, eidx, alpha=0.5, mode="additive")
    for e in arts:
        for key, v in before[e].items():
            np.testing.assert_array_equal(arts[e][key], v, err_msg=f"{e} {key}")


def test_predict_workers_respect_cuda_visible_devices(tmp_path, monkeypatch):
    seen = {}

    def capture(cmds, **kw):
        seen.update(kw)
        return []

    monkeypatch.setattr(predict_mod, "spawn_and_monitor", capture)
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "2,3")
    args = predict_mod._build_parser().parse_args(
        ["--model", str(tmp_path), "--dataset", "pool.pkl.gz", "--n-workers", "2"])
    args.threads_per_worker_int = 1
    predict_mod._spawn_workers(args)
    assert seen["gpus"] == ["2", "3"]


def test_history_keeps_integer_steps_exact(tmp_path):
    h = History(4)
    big = 2**24 + 1  # not representable in float32
    h.append(step=big, gamma_max=1.0)
    h.append(step=big + 2, gamma_max=2.0)
    path = str(tmp_path / "hal_history.npz")
    h.save(path)
    with np.load(path) as d:
        assert d["step"].dtype.kind == "i"
        assert d["step"].tolist() == [big, big + 2]
        assert d["gamma_max"].dtype == np.float32


def test_walker_dying_before_reading_its_job_does_not_stop_the_others(tmp_path, monkeypatch):
    """The first child exits before reading stdin; the job pickle is larger than the pipe
    buffer, so the parent's write fails (BrokenPipeError). That walker is reported as
    failed and the next one still runs."""
    marker = tmp_path / "died_once"
    script = tmp_path / "fake_walker.py"
    script.write_text(
        "import os, sys\n"
        f"if not os.path.exists({str(marker)!r}):\n"
        f"    open({str(marker)!r}, 'w').close()\n"
        "    sys.exit(5)\n" + FAKE_WALKER)
    monkeypatch.setattr(walkers, "CHILD_CMD", (sys.executable, str(script)))
    big = bulk("Al", "fcc", a=4.05, cubic=True).repeat((12, 12, 12))  # pickle >> 64 KiB
    out = str(tmp_path / "w")
    with pytest.warns(UserWarning, match=r"1 of 2 walkers failed: \[0\]"):
        df = walkers.run_walkers([big], 2, walkers_per_gpu=1, output_dir=out, gpus="",
                                 temperature_K=300, n_steps=5, **MODEL_KW)
    assert df["exit_code"].tolist() == [5, 0]
    assert os.path.exists(os.path.join(out, "hal_w1", "fake.json"))
