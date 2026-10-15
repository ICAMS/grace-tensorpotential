"""HAL usability: progress lines, --model-dir, pool helper, tau scan, n_fragments."""

import io
import os
import types

import numpy as np
import pandas as pd
import pytest
from ase import Atoms
from ase.build import bulk, molecule
from ase.io import read, write

from tensorpotential.uq.cli.hal import _build_parser, hal_main
from tensorpotential.uq.hal.collector import HALCollector
from tests.hal_fakes import FAKE_META, FakeHALCalc, FakeLive, FakePristine


# -- 5. fragments -----------------------------------------------------------------

def test_fragments_ethanol():
    from tensorpotential.uq.hal.dynamics import count_fragments

    m = molecule("CH3CH2OH")
    assert count_fragments(m) == 1
    h = np.nonzero(m.numbers == 1)[0]
    lost_h = m.copy()
    lost_h.positions[h[0]] = m.positions.max(axis=0) + [2.5, 0, 0]  # > 2 A from every atom
    assert count_fragments(lost_h) == 2
    h2 = m.copy()  # two H leave and form H2 (0.74 A) far away: one more fragment, not two
    h2.positions[h[:2]] = [[20, 0, 0], [20.74, 0, 0]]
    assert count_fragments(h2) == 2
    # H2 is one fragment with margin (H-H cutoff 1.25 x 0.62 = 0.775 A; 1.2 gave 0.744 and flickered)
    assert count_fragments(Atoms("H2", positions=[[0, 0, 0], [0.74, 0, 0]])) == 1


def _alli_cell():
    cell = bulk("Al", "fcc", a=4.05, cubic=True) * (3, 3, 3)  # 108 atoms
    cell.symbols[::4] = "Li"
    return cell


def test_fragments_periodic_void():
    from tensorpotential.uq.hal.dynamics import count_fragments

    assert count_fragments(bulk("Al", "fcc", a=4.05)) == 1  # one atom, bonded to its images
    cell = _alli_cell()
    assert count_fragments(cell) == 1
    slab = cell.copy()
    slab.cell[2, 2] += 10.0  # vacuum gap along z: still connected through the x, y images
    assert count_fragments(slab) == 1
    slab.positions[0] = [1.0, 1.0, cell.cell[2, 2] + 5.0]  # one atom into the gap
    assert count_fragments(slab) == 2


def test_collector_n_fragments_two_molecule_start(tmp_path):
    atoms = molecule("H2O") + Atoms("H2", positions=[[15, 0, 0], [15.74, 0, 0]])  # 2 at start
    calc = FakeHALCalc(len(atoms), FakeLive((0.5,) * 11))
    col = HALCollector(atoms, calc, pristine=FakePristine(), feature_meta=FAKE_META, element_indices=np.zeros(len(atoms), int),
                       output_dir=str(tmp_path / "w0"), equil_steps=0, trigger_gap=1,
                       temperature_K=300.0)
    assert col.n_fragments0 == 2
    calc.gamma = np.full(len(atoms), 2.0)
    col()  # still 2: not fragmented relative to the start
    atoms.positions[-1] = [25, 0, 0]  # H2 breaks up
    col()
    assert [c["n_fragments"] for c in col.collections] == [2, 3]
    assert read(col.frames_path, 1).info["n_fragments"] == 3
    s = col.finalize("max_steps")
    assert s["n_fragments0"] == 2 and s["n_fragmented"] == 1


# -- 1. progress --------------------------------------------------------------------

def test_progress_printer(tmp_path):
    from tensorpotential.uq.hal.run import ProgressPrinter

    atoms = bulk("Cu", "fcc", a=3.6, cubic=True)
    calc = FakeHALCalc(len(atoms), FakeLive((0.5,) * 6))
    col = HALCollector(atoms, calc, pristine=FakePristine(), feature_meta=FAKE_META, element_indices=np.zeros(len(atoms), int),
                       output_dir=str(tmp_path / "w0"), equil_steps=0, trigger_gap=1,
                       temperature_K=300.0)
    out = io.StringIO()
    pp = ProgressPrinter(col, n_steps=4, every=2, stream=out)
    for g in (1.0, 1.0, 2.0, 1.0, 1.0):  # collection at step 2
        calc.gamma = np.full(len(atoms), g)
        col()
        pp()
    lines = out.getvalue().splitlines()
    assert [ln.split(" |")[0] for ln in lines] == ["step 2/4", "step 4/4"]  # cadence
    assert "collected 1" in lines[0]
    # early stop between two lines: one final line with the reason, printed once
    pp = ProgressPrinter(col, n_steps=100, every=50, stream=out)
    col()
    pp.final("n_collect")
    pp.final("n_collect")
    last = out.getvalue().splitlines()[2:]
    assert len(last) == 1 and last[0].startswith("step 5/100 |") and last[0].endswith("| stop: n_collect")


# -- 2. model directory -------------------------------------------------------------

def _touch(p):
    os.makedirs(os.path.dirname(p), exist_ok=True)
    open(p, "w").close()


def _examples_layout(d):
    for f in ("model.yaml", "checkpoint.index", "checkpoint.data-00000-of-00001", "gmm_artifacts.npz"):
        _touch(os.path.join(d, f))
    return d


def _fit_layout(d, uq_dir=True):
    for f in ("model.yaml", "checkpoints/checkpoint.best_test_loss.index",
              "checkpoints/checkpoint.best_test_loss.data-00000-of-00001",
              "checkpoints/checkpoint.index", "checkpoints/checkpoint.data-00000-of-00001"):
        _touch(os.path.join(d, f))
    _touch(os.path.join(d, "UQ/gmm_artifacts.npz" if uq_dir else "gmm_artifacts.npz"))
    return d


def test_resolve_model_dir_layouts(tmp_path):
    from tensorpotential.uq.hal import resolve_model_dir

    d = _examples_layout(str(tmp_path / "m"))
    assert resolve_model_dir(d) == (os.path.join(d, "model.yaml"), os.path.join(d, "checkpoint"),
                                    os.path.join(d, "gmm_artifacts.npz"))
    for uq_dir in (True, False):
        d = _fit_layout(str(tmp_path / f"fit{uq_dir}"), uq_dir=uq_dir)
        y, c, a = resolve_model_dir(d)
        assert y == os.path.join(d, "model.yaml")
        assert c == os.path.join(d, "checkpoints", "checkpoint.best_test_loss")  # best wins
        assert a == os.path.join(d, *(("UQ",) if uq_dir else ()), "gmm_artifacts.npz")
    y, c, a = resolve_model_dir(d, model_yaml="x.yaml", artifact_path="other.npz")  # overrides
    assert (y, a) == ("x.yaml", "other.npz")
    assert c == os.path.join(d, "checkpoints", "checkpoint.best_test_loss")


def test_resolve_model_dir_errors(tmp_path):
    from tensorpotential.uq.hal import resolve_model_dir

    d = str(tmp_path / "half")
    _touch(os.path.join(d, "model.yaml"))
    with pytest.raises(FileNotFoundError) as exc:
        resolve_model_dir(d)
    msg = str(exc.value)
    assert "checkpoints/checkpoint.best_test_loss.index" in msg
    assert "UQ/gmm_artifacts.npz" in msg and "--checkpoint" in msg
    d = _fit_layout(str(tmp_path / "both"))
    _touch(os.path.join(d, "gmm_artifacts.npz"))  # top level and UQ/
    with pytest.raises(ValueError, match="ambiguous"):
        resolve_model_dir(d)
    resolve_model_dir(d, artifact_path=os.path.join(d, "UQ", "gmm_artifacts.npz"))  # override resolves it


def test_resolve_model_dir_optional_checkpoint_ambiguity(tmp_path):
    from tensorpotential.uq.hal import resolve_model_dir

    d = _examples_layout(str(tmp_path / "m"))
    _touch(os.path.join(d, "checkpoints", "checkpoint.index"))
    y, c, a = resolve_model_dir(d, require=("artifact",))
    assert y == os.path.join(d, "model.yaml")
    assert c is None
    assert a == os.path.join(d, "gmm_artifacts.npz")
    with pytest.raises(ValueError, match="ambiguous checkpoint"):
        resolve_model_dir(d)
    _, c, _ = resolve_model_dir(d, checkpoint="explicit", require=("artifact",))
    assert c == "explicit"


def test_run_hal_model_dir_reaches_factory(tmp_path, monkeypatch, capsys):
    from tensorpotential.uq import factories
    from tensorpotential.uq.hal import run_hal

    d = _examples_layout(str(tmp_path / "m"))
    seen = {}

    def stop(**kw):
        seen.update(kw)
        raise RuntimeError("stop here")

    monkeypatch.setattr(factories, "get_gmm_uq_calculator", stop)
    with pytest.raises(RuntimeError, match="stop here"):
        run_hal(bulk("Al", cubic=True), model_dir=d, temperature_K=300, n_steps=5,
                output_dir=str(tmp_path / "w0"))
    assert seen["model_yaml"] == os.path.join(d, "model.yaml")
    assert seen["checkpoint"] == os.path.join(d, "checkpoint")
    assert seen["gmm_artifact_path"] == os.path.join(d, "gmm_artifacts.npz")
    assert capsys.readouterr().out == ""  # quiet by default in Python
    with pytest.raises(RuntimeError, match="stop here"):
        run_hal(bulk("Al", cubic=True), model_dir=d, temperature_K=300, n_steps=5,
                output_dir=str(tmp_path / "w1"), progress_every=100)
    assert "gmm_artifacts.npz" in capsys.readouterr().out  # printed with progress lines on


# -- 3. pool ------------------------------------------------------------------------

def _walker_dir(path, gammas, symbol="Cu"):
    os.makedirs(path, exist_ok=True)
    for i, g in enumerate(gammas):
        a = bulk(symbol, "fcc", a=3.6, cubic=True)
        a.info.update(step=10 * i, gamma_max_pristine=g, grace_energy=-1.0, n_fragments=1)
        a.arrays["gamma_live"] = np.full(len(a), 1.6)
        write(os.path.join(path, "hal_collected.extxyz"), a, format="extxyz", append=True)
    return path


def test_pool_frames_columns_filter_and_skip(tmp_path):
    from tensorpotential.uq.hal import pool_frames

    d1 = _walker_dir(str(tmp_path / "w1"), [1.2, 2.0])
    d2 = _walker_dir(str(tmp_path / "w2"), [0.8])
    empty = str(tmp_path / "w3")
    os.makedirs(empty)
    with pytest.warns(UserWarning, match="w3"):
        df = pool_frames([d1, d2, empty])
    assert len(df) == 3 and "ase_atoms" in df.columns
    assert {"hal_dir", "step", "gamma_max_pristine", "grace_energy"} <= set(df.columns)
    assert list(df["hal_dir"]) == [d1, d1, d2]
    assert all(a.calc is None for a in df["ase_atoms"])  # never reference-looking labels
    with pytest.warns(UserWarning):
        df2 = pool_frames([d1, d2, empty], gamma_pristine_min=1.0)
    assert list(df2["gamma_max_pristine"]) == [1.2, 2.0]


def test_pool_frames_carries_candidate_columns(tmp_path):
    from tests.hal_fakes import make_walker_dir

    from tensorpotential.uq.hal import pool_frames

    atoms = bulk("Cu", "fcc", a=3.6, cubic=True)
    feats = [np.arange(16.0).reshape(4, 4) + 100 * k for k in range(3)]
    gp = [[0.5, 1.5, 2.0, 0.1], [0.2, 0.2, 0.2, 0.2], [3.0, 0.0, 0.0, 1.0]]
    d1 = str(tmp_path / "w1")
    make_walker_dir(d1, atoms, np.zeros(4, int), feats, gp,
                    feature_meta={"artifact_sha256": "h1"})
    d2 = _walker_dir(str(tmp_path / "w2"), [1.2])  # no sidecar
    df = pool_frames([d1, d2])
    assert len(df) == 4
    np.testing.assert_array_equal(df["cand_atom_index"][0], [1, 2])
    np.testing.assert_array_equal(df["cand_features"][0], feats[0][[1, 2]])
    assert df["cand_features"][0].dtype == np.float32
    np.testing.assert_allclose(df["cand_gamma"][0], [1.5, 2.0])
    assert len(df["cand_atom_index"][1]) == 0 and df["cand_features"][1].shape == (0, 4)
    np.testing.assert_array_equal(df["cand_atom_index"][2], [0, 3])
    assert df["cand_feature_gamma_min"][0] == 1.0 and df["cand_artifact_sha256"][0] == "h1"
    assert df["cand_atom_index"].iloc[3] is None or pd.isna(df["cand_atom_index"].iloc[3])
    # the gamma filter keeps each row's own candidates
    df2 = pool_frames([d1], gamma_pristine_min=2.5)
    assert list(df2["collection_index"]) == [2]
    np.testing.assert_array_equal(df2["cand_atom_index"][0], [0, 3])


def test_cli_pool_extxyz_drops_candidates(tmp_path):
    from tests.hal_fakes import make_walker_dir

    atoms = bulk("Cu", "fcc", a=3.6, cubic=True)
    d1 = str(tmp_path / "w1")
    make_walker_dir(d1, atoms, np.zeros(4, int), [np.ones((4, 4))], [[2.0] * 4])
    out = str(tmp_path / "pool.extxyz")
    with pytest.warns(UserWarning) as rec:
        assert hal_main(["--pool", d1, "--pool-output", out]) == 0
    msgs = [str(w.message) for w in rec]
    assert any("candidate-atom features" in m for m in msgs)
    assert any("reference energies" in m for m in msgs)
    assert len(read(out, ":")) == 1


def test_cli_pool(tmp_path, capsys):
    from tensorpotential.uq.cli._common import load_dataset_any

    d1 = _walker_dir(str(tmp_path / "w1"), [1.2, 2.0])
    out = str(tmp_path / "pool.pkl.gz")
    assert hal_main(["--pool", d1, "--pool-output", out, "--gamma-pristine-min", "1.5"]) == 0
    df = load_dataset_any(out)
    assert len(df) == 1 and df["gamma_max_pristine"].iloc[0] == 2.0
    assert "pooled 1 frames from 1 dirs" in capsys.readouterr().out
    out = str(tmp_path / "pool.extxyz")
    with pytest.warns(UserWarning, match="pkl.gz"):
        assert hal_main(["--pool", d1, "--pool-output", out]) == 0
    frames = read(out, ":")
    assert len(frames) == 2
    assert all("energy" not in (f.calc.results if f.calc else {}) for f in frames)


def test_pool_select_round_trip(hal_uq_setup, tmp_path):
    """A pooled pkl.gz goes straight into `grace_uq select --dataset` (yaml mode, CPU)."""
    from tensorpotential.uq.cli.select import select_main
    from tensorpotential.uq.hal import pool_frames

    w = str(tmp_path / "w1")
    os.makedirs(w)
    for i, a in enumerate(hal_uq_setup["atoms"][:4]):
        a = a.copy()
        a.info.update(step=i, gamma_max_pristine=2.0, grace_energy=-1.0, n_fragments=1)
        write(os.path.join(w, "hal_collected.extxyz"), a, format="extxyz", append=True)
    pool = str(tmp_path / "pool.pkl.gz")
    assert hal_main(["--pool", w, "--pool-output", pool]) == 0
    assert len(pool_frames([w])) == 4
    out = str(tmp_path / "sel.pkl.gz")
    rc = select_main(["--dataset", pool, "--model", hal_uq_setup["model_yaml"],
                      "--checkpoint", hal_uq_setup["checkpoint"],
                      "--artifact", hal_uq_setup["artifact_path"], "-n", "2",
                      "--gamma-min", "0", "--no-element-stratified", "--n-workers", "1",
                      "--gpus", "-1", "--output", out])
    assert rc == 0
    assert len(pd.read_pickle(out, compression="gzip")) == 2


# -- 4. tau scan --------------------------------------------------------------------

def test_cli_model_dir_help_matches_search_order():
    from tensorpotential.uq.cli.hal import _MODEL_DIR_HELP
    from tensorpotential.uq.hal.model_dir import SEARCH_ORDER_HELP

    assert _MODEL_DIR_HELP == SEARCH_ORDER_HELP


def test_cli_pool_flags_and_output_prefix(capsys):
    for extra in (["--pool-output", "p.pkl.gz"], ["--gamma-pristine-min", "1.0"]):
        with pytest.raises(SystemExit):
            hal_main(["--model-dir", "d", "--structure", "s", "-T", "300", "--n-steps", "1", *extra])
        assert "--pool" in capsys.readouterr().err
    # --output still abbreviates --output-dir (no --output flag of its own)
    assert _build_parser().parse_args(["--model-dir", "d", "--output", "w1"]).output_dir == "w1"


def test_expanded_bulk_reads_fragmented_documented_caveat(tmp_path):
    """Covalent cutoffs (Al-Al 3.03 A, Al-Li 3.11 A) sit just above the fcc Al-Li
    nearest-neighbour distance (2.86 A): at V/V0 1.4 (3.20 A) the cell reads as fragments."""
    atoms = _alli_cell()
    calc = FakeHALCalc(len(atoms), FakeLive((0.5,) * 11))
    col = HALCollector(atoms, calc, pristine=FakePristine(), feature_meta=FAKE_META, element_indices=np.zeros(len(atoms), int),
                       output_dir=str(tmp_path / "w0"), equil_steps=0, trigger_gap=1,
                       temperature_K=300.0)
    atoms.set_cell(atoms.cell * 1.4 ** (1 / 3), scale_atoms=True)
    calc.gamma = np.full(len(atoms), 2.0)
    col()
    assert col.collections[0]["n_fragments"] > 1
    s = col.finalize("max_steps")
    assert s["n_fragments0"] == 1 and s["n_fragmented"] == 1


def test_cli_single_tau_runs_one_walker_in_process(tmp_path, monkeypatch):
    import ase.io

    import tensorpotential.uq.hal as hal_pkg

    monkeypatch.setattr(ase.io, "read", lambda f, index=0: bulk("Cu", cubic=True))
    single = {}
    monkeypatch.setattr(hal_pkg, "run_hal", lambda atoms, **kw: (single.update(kw), types.SimpleNamespace(
        summary=dict(n_steps=1, n_collected=0, rejected={}, stop_reason="max_steps",
                     steps_per_s=1.0)))[1])
    assert hal_main(["--model-dir", "MD", "--structure", "s", "-T", "300", "--n-steps", "10",
                     "--output-dir", str(tmp_path / "out"), "--tau", "0.3"]) == 0
    assert single["tau"] == 0.3 and single["output_dir"] == str(tmp_path / "out")
