import json
import os

import pytest
from ase.io import write

from tensorpotential.scripts.grace_uq import _dispatch
from tensorpotential.uq.cli.hal import _build_parser, hal_main


# CLI dest -> run_hal parameter where the names differ
_RENAMED = {"artifact_path": "uq_artifact", "live_start_artifact_path": "live_start_artifact",
            "tdamp": "tdamp_fs", "pdamp": "pdamp_fs", "pair_radii": "pair_overrides",
            "timestep": "timestep_fs", "temperature": "temperature_K"}
# documented differences: the CLI prints progress, takes several tau / ensembles (with
# --repeats), says 'auto' for the step
_CLI_ONLY_DEFAULTS = {"progress_every": (100, 0), "tau": ([0.15], 0.15), "timestep": ("auto", None),
                      "ensemble": (["nvt"], "nvt"),
                      "output_dir": (None, "hal_w0")}  # --repeats: hal_walkers


def test_cli_defaults_match_run_hal():
    """Parser defaults == run_hal signature defaults. Not compared, by construction: values
    hal_main hard-codes instead of taking from the parser (keep_frames=False on the CLI,
    True in run_hal: the CLI walker writes its frames to disk only)."""
    import inspect

    from tensorpotential.uq.hal import run_hal

    params = inspect.signature(run_hal).parameters
    args = vars(_build_parser().parse_args([]))
    compared = 0
    for dest, cli_default in args.items():
        name = _RENAMED.get(dest, dest)
        if name not in params or params[name].default is inspect.Parameter.empty:
            continue
        py_default = params[name].default
        if dest in _CLI_ONLY_DEFAULTS:
            assert (cli_default, py_default) == _CLI_ONLY_DEFAULTS[dest], dest
        else:
            cli_v = tuple(cli_default) if isinstance(cli_default, list) else cli_default
            assert cli_v == py_default, dest
        compared += 1
    assert compared >= 35


def test_cli_missing_required_args(capsys):
    with pytest.raises(SystemExit) as exc:
        hal_main(["--model-yaml", "m.yaml"])
    assert exc.value.code == 2
    err = capsys.readouterr().err
    assert "--artifact-path" in err and "--structure" in err and "--model-dir" in err


def test_flags_reach_run_hal(monkeypatch, tmp_path):
    import types

    import ase.io
    from ase import Atoms

    import tensorpotential.uq.hal as hal_pkg

    seen = {}

    def fake_run_hal(atoms, **kw):
        seen["atoms"] = atoms
        seen.update(kw)
        return types.SimpleNamespace(
            summary=dict(n_steps=7, n_collected=1, rejected=0, stop_reason="done", steps_per_s=1.0)
        )

    monkeypatch.setattr(hal_pkg, "run_hal", fake_run_hal)
    atoms = Atoms("H2", positions=[[0, 0, 0], [0, 0, 0.8]])
    monkeypatch.setattr(ase.io, "read", lambda f, index=0: atoms if (f, index) == ("s.extxyz", 3) else None)
    monkeypatch.delenv("CUDA_VISIBLE_DEVICES", raising=False)
    pr = tmp_path / "pairs.json"
    pr.write_text('{"H-H": {"zbl": [0.3, 0.5], "reject": 0.4}}')
    argv = [
        "--model-yaml", "m.yaml", "--checkpoint", "ckpt", "--artifact-path", "a.npz",
        "--live-start-artifact-path", "live.npz", "--structure", "s.extxyz", "--structure-index", "3",
        "--param-dtype", "float32", "--ensemble", "npt_stress", "-T", "777", "--n-steps", "7",
        "--n-collect", "4", "--timestep", "1.5", "--friction", "0.02", "--tdamp", "50",
        "--pdamp", "500", "--nvt-ramp", "0.9", "1.2", "--stress-tau", "0.05",
        "--bulk-modulus", "0.6", "--volume-guard", "0.8", "1.2", "--cluster-wall-margin", "3.0",
        "--cluster-wall-k", "2.0", "--seed", "11", "--tau", "0.3", "--tau-ema", "5",
        "--zbl", "off", "--zbl-switch", "0.4", "0.6", "--pair-radii", str(pr),
        "--equil-steps", "9", "--trigger-gamma", "2.5", "--trigger-gap", "3",
        "--trigger-min-atoms", "4",
        "--reject-gamma", "8.0", "--reject-bond-factor", "0.5", "--update", "none",
        "--update-target", "1.5", "--alphas", "0.1", "0.5",
        "--no-update-inflate",
        "--inflate-max-rounds", "2", "--inflate-max-factor", "2.5", "--output-dir", "outdir",
        "--walker-id", "2", "--save-every", "4", "--history-every", "50", "--overwrite",
        "--gpus", "3", "--no-remove-update-work", "--update-work-floor", "0.2",
        "--fragment-mult", "1.5", "--feature-gamma-min", "none",
    ]
    assert hal_main(argv) == 0
    assert os.environ["CUDA_VISIBLE_DEVICES"] == "3"
    monkeypatch.delenv("CUDA_VISIBLE_DEVICES")
    assert seen.pop("atoms") is atoms
    assert seen == dict(
        model_yaml="m.yaml", checkpoint="ckpt", uq_artifact="a.npz",
        live_start_artifact="live.npz", ensemble="npt_stress", temperature_K=777.0,
        n_steps=7, n_collect=4, timestep_fs=1.5, tau=0.3, tau_ema=5, equil_steps=9,
        trigger_gamma=2.5, trigger_gap=3, trigger_min_atoms=4, update="none", update_target=1.5,
        alphas=(0.1, 0.5), update_inflate=False, inflate_max_rounds=2,
        inflate_max_factor=2.5, reject_gamma=8.0, reject_bond_factor=0.5, zbl="off",
        zbl_switch=(0.4, 0.6), pair_overrides={"H-H": {"zbl": [0.3, 0.5], "reject": 0.4}},
        nvt_ramp=(0.9, 1.2), stress_tau=0.05, bulk_modulus=0.6, volume_guard=(0.8, 1.2),
        cluster_wall_margin=3.0, cluster_wall_k=2.0, friction=0.02, tdamp_fs=50.0,
        pdamp_fs=500.0, remove_update_work=False, update_work_floor=0.2, seed=11, walker_id=2,
        output_dir="outdir", overwrite=True, keep_frames=False, save_every=4, history_every=50,
        calc_kwargs={"param_dtype": "float32"}, model_dir=None, progress_every=100,
        fragment_mult=1.5, feature_gamma_min=None,
    )
    seen.clear()
    assert hal_main(["--model-dir", "MD", "--checkpoint", "C", "--structure", "s.extxyz",
                     "--structure-index", "3", "-T", "300", "--n-steps", "1"]) == 0
    assert seen["model_dir"] == "MD" and seen["checkpoint"] == "C"
    assert seen["model_yaml"] is None and seen["uq_artifact"] is None
    assert seen["feature_gamma_min"] == 1.0


def test_cli_end_to_end_and_plot_only(hal_uq_setup, tmp_path):
    structure = str(tmp_path / "seed.extxyz")
    write(structure, max(hal_uq_setup["atoms"], key=len).copy())
    out = str(tmp_path / "w0")
    argv = [
        "--model-yaml", hal_uq_setup["model_yaml"], "--checkpoint", hal_uq_setup["checkpoint"],
        "--artifact-path", hal_uq_setup["artifact_path"], "--structure", structure,
        "-T", "300", "--n-steps", "20", "--equil-steps", "2", "--trigger-gamma", "0",
        "--trigger-gap", "5", "--output-dir", out, "--param-dtype", "float64",
    ]
    assert _dispatch(["hal", *argv]) == 0
    with open(os.path.join(out, "hal_summary.json")) as fh:
        assert json.load(fh)["n_steps"] >= 20
    # a killed walker: no summary, no PNG; --plot-only rebuilds from settings + history + frames
    os.remove(os.path.join(out, "hal_summary.json"))
    os.remove(os.path.join(out, "hal_diagnostics.png"))
    assert hal_main(["--plot-only", out]) == 0
    assert os.path.exists(os.path.join(out, "hal_diagnostics.png"))
