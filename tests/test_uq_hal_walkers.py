"""grace_uq hal --repeats / run_walkers: K walkers per setting as subprocesses, M per GPU at a time.

The scheduler is tested with a fake walker subprocess (no model); one small real run of
two walkers on the CPU checks the actual child.
"""

import json
import os
import sys

import pandas as pd
import pytest
from ase.build import bulk
from ase.io import write

import tensorpotential.uq.hal.walkers as walkers
from tensorpotential.uq.cli.hal import hal_main
from tensorpotential.uq.hal.walkers import run_walkers

MODEL = os.path.join(os.path.dirname(__file__), "..", "examples", "data", "AlLi-1L-small")
MODEL_KW = dict(model_yaml=os.path.join(MODEL, "model.yaml"),
                checkpoint=os.path.join(MODEL, "checkpoint"),
                uq_artifact=os.path.join(MODEL, "gmm_artifacts.npz"))

FAKE_WALKER = r'''
import json, os, pickle, sys, time
job = pickle.load(sys.stdin.buffer)
kw = job["kwargs"]
out = kw["output_dir"]
t0 = time.time()
time.sleep(0.4)
rec = {"seed": kw["seed"], "walker_id": kw["walker_id"], "a": float(job["atoms"].cell[0, 0]),
       "tau": kw.get("tau"), "temperature_K": kw["temperature_K"], "ensemble": kw.get("ensemble"),
       "keep_frames": kw["keep_frames"], "overwrite": kw["overwrite"],
       "env": {k: os.environ.get(k) for k in ("CUDA_VISIBLE_DEVICES", "TF_FORCE_GPU_ALLOW_GROWTH",
                                              "OMP_NUM_THREADS")},
       "t0": t0, "t1": time.time()}
with open(os.path.join(out, "fake.json"), "w") as fh:
    json.dump(rec, fh)
print("fake walker", kw["walker_id"])
if kw["seed"] == int(os.environ.get("FAKE_FAIL_SEED", "-1")):
    sys.exit(3)
summary = {"n_collected": kw["walker_id"], "rejected": {"gamma": 1} if kw["walker_id"] % 2 else {},
           "stop_reason": "max_steps", "temperature_mean_K": 300.0, "n_steps": kw["n_steps"],
           "steps_per_s": 10.0}
with open(os.path.join(out, "hal_summary.json"), "w") as fh:
    json.dump(summary, fh)
'''


@pytest.fixture
def fake_child(tmp_path, monkeypatch):
    script = tmp_path / "fake_walker.py"
    script.write_text(FAKE_WALKER)
    monkeypatch.setattr(walkers, "CHILD_CMD", (sys.executable, str(script)))
    return script


def _frames():
    return [bulk("Al", "fcc", a=a, cubic=True) for a in (4.0, 4.05, 4.1)]


def _records(out, n):
    recs = []
    for i in range(n):
        with open(os.path.join(out, f"hal_w{i}", "fake.json")) as fh:
            recs.append(json.load(fh))
    return recs


def _max_overlap(recs):
    events = sorted([(r["t0"], 1) for r in recs] + [(r["t1"], -1) for r in recs])
    cur = best = 0
    for _, e in events:
        cur += e
        best = max(best, cur)
    return best


def test_layout_seeds_start_frames_cpu_slots_and_csv(fake_child, tmp_path):
    out = str(tmp_path / "walkers")
    df = run_walkers(_frames(), 5, walkers_per_gpu=2, output_dir=out, seed=7,
                     structure_offset=1, gpus="", temperature_K=500, n_steps=11, tau=0.3,
                     **MODEL_KW)
    recs = _records(out, 5)
    assert [r["walker_id"] for r in recs] == list(range(5))
    assert [r["seed"] for r in recs] == [7, 8, 9, 10, 11]
    a = [f.cell[0, 0] for f in _frames()]
    assert [r["a"] for r in recs] == pytest.approx([a[(1 + i) % 3] for i in range(5)])
    assert all(r["tau"] == 0.3 and r["temperature_K"] == 500 for r in recs)
    assert all(r["keep_frames"] is False and r["overwrite"] is False for r in recs)
    assert all(r["env"]["CUDA_VISIBLE_DEVICES"] == "-1" for r in recs)  # CPU: no GPU at all
    assert all(r["env"]["TF_FORCE_GPU_ALLOW_GROWTH"] == "true" for r in recs)
    assert {r["env"]["OMP_NUM_THREADS"] for r in recs} == {str(max(1, (os.cpu_count() or 1) // 2))}
    assert _max_overlap(recs) <= 2  # walkers_per_gpu on the CPU
    assert all(os.path.exists(os.path.join(out, f"hal_w{i}", "walker.log")) for i in range(5))
    assert "fake walker 3" in open(os.path.join(out, "hal_w3", "walker.log")).read()
    csv = pd.read_csv(os.path.join(out, "walkers.csv"))
    assert csv["walker"].tolist() == list(range(5)) == df["walker"].tolist()
    assert csv["start_frame"].tolist() == [1, 2, 0, 1, 2]
    assert csv["collected"].tolist() == [0, 1, 2, 3, 4]
    assert csv["rejected_gamma"].tolist() == [0, 1, 0, 1, 0]
    assert (csv["exit_code"] == 0).all() and (csv["device"] == "cpu").all()


def test_gpus_oversubscribed_round_robin(fake_child, tmp_path):
    out = str(tmp_path / "w")
    df = run_walkers(_frames()[0], 6, walkers_per_gpu=2, output_dir=out, gpus="0,1",
                     temperature_K=300, n_steps=5, **MODEL_KW)
    recs = _records(out, 6)
    by_gpu = {g: [r for r in recs if r["env"]["CUDA_VISIBLE_DEVICES"] == g] for g in ("0", "1")}
    assert sorted(len(v) for v in by_gpu.values()) == [3, 3]
    assert all(_max_overlap(v) <= 2 for v in by_gpu.values())
    assert _max_overlap(recs) >= 3  # both GPUs were busy at once
    assert sorted(df["device"]) == ["0", "0", "0", "1", "1", "1"]


def test_failing_walker_is_reported_and_others_finish(fake_child, tmp_path, monkeypatch):
    monkeypatch.setenv("FAKE_FAIL_SEED", "44")  # walker 2
    out = str(tmp_path / "w")
    with pytest.warns(UserWarning, match=r"1 of 4 walkers failed: \[2\]"):
        df = run_walkers(_frames(), 4, walkers_per_gpu=4, output_dir=out, gpus="",
                         temperature_K=300, n_steps=5, **MODEL_KW)
    assert df["exit_code"].tolist() == [0, 0, 3, 0]
    assert df.loc[2, "stop_reason"] == "failed before the run"
    assert df.loc[[0, 1, 3], "stop_reason"].eq("max_steps").all()



def test_grid_of_ensemble_temperature_and_tau(fake_child, tmp_path):
    """Several values of tau / ensemble / temperature_K: `repeats` walkers per combination,
    all sharing the slots; seed and start frame depend on the repeat only."""
    out = str(tmp_path / "grid")
    df = run_walkers(_frames(), 2, walkers_per_gpu=3, output_dir=out, seed=7, gpus="",
                     ensemble=["nvt", "npt"], temperature_K=300, tau=[0.2, 0.5], n_steps=5,
                     **MODEL_KW)
    labels = ["nvt_tau_0.2", "nvt_tau_0.5", "npt_tau_0.2", "npt_tau_0.5"]
    recs = []
    for lab in labels:
        for r in range(2):
            with open(os.path.join(out, lab, f"hal_w{r}", "fake.json")) as fh:
                recs.append(dict(json.load(fh), label=lab, repeat=r))
    assert sorted(r["walker_id"] for r in recs) == list(range(8))
    assert all(r["seed"] == 7 + r["repeat"] for r in recs)
    a = [f.cell[0, 0] for f in _frames()]
    assert all(r["a"] == pytest.approx(a[r["repeat"]]) for r in recs)
    assert all(r["label"] == f"{r['ensemble']}_tau_{r['tau']:g}" for r in recs)
    assert all(r["temperature_K"] == 300 for r in recs)
    assert _max_overlap(recs) <= 3  # one scheduler for the whole grid
    assert len(df) == 8 and df["walker"].tolist() == list(range(8))
    assert df[["ensemble", "temperature_K", "tau"]].drop_duplicates().shape[0] == 4
    assert df["repeat"].tolist() == [0, 1] * 4
    csv = pd.read_csv(os.path.join(out, "walkers.csv"))
    assert csv["output_dir"].str.contains("npt_tau_0.5").sum() == 2


def test_single_combination_keeps_flat_layout(fake_child, tmp_path):
    out = str(tmp_path / "w")
    df = run_walkers(_frames(), 2, output_dir=out, gpus="", temperature_K=[300],
                     tau=(0.3,), n_steps=5, **MODEL_KW)
    assert os.path.exists(os.path.join(out, "hal_w1", "fake.json"))
    assert df["tau"].tolist() == [0.3, 0.3] and df["temperature_K"].tolist() == [300, 300]


@pytest.mark.parametrize("bad, match", [
    (dict(tau=[0.2, 0.2]), "duplicate"),
    (dict(ensemble=["nvt", "cluster"]), "cluster"),
    (dict(walker_id=3), "per walker"),
    (dict(ensemble="cluster"), "cluster"),
    (dict(tau_ema=0), "tau_ema"),
    (dict(bogus=1), "unknown"),
    (dict(uq_artifact="nope.npz"), "not found"),
    (dict(checkpoint="nope"), "not found"),
])
def test_bad_options_refused_before_any_walker_starts(fake_child, tmp_path, bad, match):
    out = tmp_path / "w"
    kw = {**MODEL_KW, "temperature_K": 300, "n_steps": 5, **bad}
    with pytest.raises((ValueError, TypeError, FileNotFoundError), match=match):
        run_walkers(_frames(), 2, output_dir=str(out), gpus="", **kw)
    assert not out.exists()


def test_existing_walker_output_refused_without_overwrite(fake_child, tmp_path):
    out = tmp_path / "w"
    (out / "hal_w1").mkdir(parents=True)
    (out / "hal_w1" / "hal_summary.json").write_text("{}")
    with pytest.raises(FileExistsError, match=r"hal_w1.*--overwrite.*overwrite=True"):
        run_walkers(_frames(), 2, output_dir=str(out), gpus="", temperature_K=300, n_steps=5,
                    **MODEL_KW)
    assert not (out / "hal_w0").exists()
    df = run_walkers(_frames(), 2, output_dir=str(out), gpus="", temperature_K=300, n_steps=5,
                     overwrite=True, **MODEL_KW)
    assert (df["exit_code"] == 0).all()


def test_cli_walkers_passes_options_through(tmp_path, monkeypatch, capsys):
    seen = {}

    def fake_run_walkers(frames, n, **kw):
        seen.update(kw, frames=frames, n=n)
        return pd.DataFrame([{"walker": 0, "start_frame": 2, "device": "cpu", "exit_code": 0,
                              "collected": 3, "rejected": "0", "n_rejected": 0,
                              "stop_reason": "max_steps", "mean_T_K": 300.0, "steps": 5,
                              "steps_per_s": 1.0, "wall_s": 1.0, "output_dir": "x"},
                             {"walker": 1, "start_frame": 0, "device": "cpu", "exit_code": 1,
                              "collected": float("nan"), "rejected": "", "n_rejected": 0,
                              "stop_reason": "failed before the run", "mean_T_K": 0,
                              "steps": 0, "steps_per_s": 0, "wall_s": 1.0,
                              "output_dir": "y"}])

    monkeypatch.setattr(walkers, "run_walkers", fake_run_walkers)
    structure = str(tmp_path / "s.extxyz")
    write(structure, _frames())
    code = hal_main(["--model-dir", MODEL, "--structure", structure, "--structure-index", "2",
                     "-T", "400", "--n-steps", "9", "--repeats", "2", "--seed", "5",
                     "--tau", "0.2", "--trigger-gap", "4"])
    assert code == 1  # walker 1 failed
    assert seen["n"] == 2 and len(seen["frames"]) == 3
    assert seen["structure_offset"] == 2 and seen["seed"] == 5
    assert seen["output_dir"] == "hal_walkers" and seen["walkers_per_gpu"] == 4
    assert seen["tau"] == [0.2] and seen["trigger_gap"] == 4 and seen["temperature_K"] == [400.0]
    assert seen["model_dir"] == MODEL and seen["equil_steps"] == 0
    assert not {"walker_id", "keep_frames"} & set(seen)
    out = capsys.readouterr().out
    assert "1 of 2 walkers finished" in out and "FAILED: [1]" in out



def test_cli_walkers_grid_passes_lists(tmp_path, monkeypatch, capsys):
    seen = {}

    def fake_run_walkers(frames, n, **kw):
        seen.update(kw, n=n)
        return pd.DataFrame([{"walker": 0, "repeat": 0, "ensemble": "nvt", "temperature_K": 300.0,
                              "tau": 0.2, "start_frame": 0, "device": "cpu", "exit_code": 0,
                              "collected": 1, "rejected": "0", "n_rejected": 0,
                              "stop_reason": "max_steps", "mean_T_K": 300.0, "steps": 5,
                              "steps_per_s": 1.0, "wall_s": 1.0, "output_dir": "x"}])

    monkeypatch.setattr(walkers, "run_walkers", fake_run_walkers)
    structure = str(tmp_path / "s.extxyz")
    write(structure, _frames())
    assert hal_main(["--model-dir", MODEL, "--structure", structure, "--n-steps", "9",
                     "--repeats", "2", "--tau", "0.2", "0.5", "--ensemble", "nvt", "npt",
                     "-T", "300", "600"]) == 0
    assert seen["tau"] == [0.2, 0.5] and seen["ensemble"] == ["nvt", "npt"]
    assert seen["temperature_K"] == [300.0, 600.0] and seen["n"] == 2



def test_cli_walkers_table_shows_only_varying_settings(tmp_path, monkeypatch, capsys):
    """Settings equal for all walkers go into one line above the table; walkers.csv keeps all."""
    def row(w, tau):
        return {"walker": w, "repeat": 0, "ensemble": "npt", "temperature_K": 300.0, "tau": tau,
                "start_frame": 0, "device": "0", "exit_code": 0, "collected": w, "rejected": "0",
                "n_rejected": 0, "stop_reason": "max_steps", "mean_T_K": 300.0, "steps": 5,
                "steps_per_s": 1.0, "wall_s": 1.0, "output_dir": f"o/tau_{tau:g}/hal_w0"}

    monkeypatch.setattr(walkers, "run_walkers",
                        lambda frames, n, **kw: pd.DataFrame([row(0, 0.25), row(1, 0.5)]))
    structure = str(tmp_path / "s.extxyz")
    write(structure, _frames())
    assert hal_main(["--model-dir", MODEL, "--structure", structure, "--n-steps", "5",
                     "-T", "300", "--ensemble", "npt", "--repeats", "1",
                     "--tau", "0.25", "0.5"]) == 0
    out = capsys.readouterr().out
    same = next(line for line in out.splitlines() if line.startswith("same for all walkers:"))
    for text in ("repeat 0", "ensemble npt", "temperature_K 300", "start_frame 0", "device 0"):
        assert text in same
    header = next(line for line in out.splitlines() if line.lstrip().startswith("walker"))
    assert "tau" in header.split() and "collected" in header.split()
    assert not {"ensemble", "temperature_K", "repeat", "device", "start_frame"} & set(header.split())


@pytest.mark.parametrize("extra, match", [
    (["--walkers-per-gpu", "2"], "applies with --repeats"),
    (["--ensemble", "nvt", "npt"], "need --repeats"),
    (["-T", "300", "600"], "need --repeats"),
    (["--repeats", "2", "--walker-id", "3"], "single walker"),
    (["--repeats", "0"], ">= 1"),
    (["--repeats", "2", "--pool", "a"], "does not combine"),
])
def test_cli_walkers_refusals(extra, match, capsys):
    with pytest.raises(SystemExit):
        hal_main(["--model-dir", MODEL, "--structure", "s.xyz", "-T", "300", "--n-steps", "5",
                  *extra])
    assert match in capsys.readouterr().err


def test_two_real_walkers_on_cpu(hal_uq_setup, tmp_path, monkeypatch):
    monkeypatch.setenv("PYTHONPATH", os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
    atoms = max(hal_uq_setup["atoms"], key=len).copy()
    out = str(tmp_path / "w")
    df = run_walkers([atoms], 2, walkers_per_gpu=2, output_dir=out, gpus="",
                     model_yaml=hal_uq_setup["model_yaml"],
                     checkpoint=hal_uq_setup["checkpoint"],
                     uq_artifact=hal_uq_setup["artifact_path"], temperature_K=300,
                     n_steps=15, trigger_gamma=0.0, trigger_gap=5, plot=False,
                     calc_kwargs={"param_dtype": "float64"})
    assert (df["exit_code"] == 0).all(), open(os.path.join(out, "hal_w0", "walker.log")).read()
    assert (df["steps"] >= 15).all() and (df["collected"] >= 1).all()
    for i in range(2):
        with open(os.path.join(out, f"hal_w{i}", "hal_settings.json")) as fh:
            s = json.load(fh)
        assert s["seed"] == 42 + i and s["walker_id"] == i
        assert os.path.exists(os.path.join(out, f"hal_w{i}", "hal_features.npz"))


def test_hal_help_mentions_repeats():
    from tensorpotential.uq.cli.hal import _build_parser

    assert "--repeats" in _build_parser().description


def test_scan_tau_is_one_walker_per_tau(fake_child, tmp_path):
    """scan_tau = run_walkers with repeats=1 over the tau values: one layout
    (OUTPUT_DIR/tau_<tau>/hal_w0/) and one table (walkers.csv) for every multi-walker run."""
    from tensorpotential.uq.hal import tools

    out = str(tmp_path / "scan")
    df = tools.scan_tau([0.15, 0.3], _frames()[0], output_dir=out, walkers_per_gpu=2,
                        temperature_K=300, n_steps=5, **MODEL_KW)
    assert list(df["tau"]) == [0.15, 0.3]
    for tau in ("0.15", "0.3"):
        assert os.path.exists(os.path.join(out, f"tau_{tau}", "hal_w0", "fake.json"))
    assert list(pd.read_csv(os.path.join(out, walkers.WALKERS_CSV))["tau"]) == [0.15, 0.3]
    with pytest.raises(ValueError, match="duplicate"):
        tools.scan_tau([0.2, 0.3, 0.2], _frames()[0], output_dir=str(tmp_path / "d"),
                       temperature_K=300, n_steps=5, **MODEL_KW)


def test_cli_several_taus_without_repeats_run_one_walker_each(tmp_path, monkeypatch, capsys):
    seen = {}

    def fake_run_walkers(frames, n, **kw):
        seen.update(kw, n=n)
        return pd.DataFrame([{"walker": i, "repeat": 0, "tau": t, "start_frame": 0,
                              "device": "cpu", "exit_code": 0, "collected": 1, "rejected": "0",
                              "n_rejected": 0, "stop_reason": "max_steps", "mean_T_K": 300.0,
                              "steps": 5, "steps_per_s": 1.0, "wall_s": 1.0, "output_dir": "x"}
                             for i, t in enumerate(kw["tau"])])

    monkeypatch.setattr(walkers, "run_walkers", fake_run_walkers)
    structure = str(tmp_path / "s.extxyz")
    write(structure, _frames())
    assert hal_main(["--model-dir", MODEL, "--structure", structure, "-T", "300",
                     "--n-steps", "9", "--tau", "0.2", "0.4", "--walkers-per-gpu", "1"]) == 0
    assert seen["n"] == 1 and seen["tau"] == [0.2, 0.4] and seen["walkers_per_gpu"] == 1
    assert seen["output_dir"] == "hal_walkers"
    assert "walkers.csv" in capsys.readouterr().out


def test_walker_row_has_the_tau_scan_columns(tmp_path):
    out = tmp_path / "hal_w0"
    out.mkdir()
    summary = {"n_collected": 2, "rejected": {"gamma": 1}, "stop_reason": "max_steps",
               "temperature_mean_K": 301.0, "n_steps": 50, "steps_per_s": 9.0,
               "gamma_max_live": 2.5, "n_fragmented": 1,
               "collections": [{"gamma_max_pristine": 1.6}, {"gamma_max_pristine": 2.1}]}
    (out / "hal_summary.json").write_text(json.dumps(summary))
    job = {"walker": 0, "repeat": 0, "combo": {"tau": 0.3}, "frame": 0, "out": str(out)}
    row = walkers._row(job, None, 0, 1.0)
    assert row["max_gamma_live"] == 2.5 and row["max_gamma_pristine_collected"] == 2.1
    assert row["n_fragmented"] == 1
