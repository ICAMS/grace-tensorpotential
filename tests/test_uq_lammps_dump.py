"""grace_uq predict / select read LAMMPS text dumps as --dataset."""

import logging

import numpy as np
import pytest

from tensorpotential.uq.cli import predict as predict_mod
from tensorpotential.uq.cli import select as select_mod
from tensorpotential.uq.cli._common import load_dataset_any


def _write_dump(path, n_frames=2, with_element=True):
    """Text dump of a 4-atom cubic cell: types 1 1 1 2 (Al Al Al Li), box 4 Å, pbc."""
    cols = "id type element x y z" if with_element else "id type x y z"
    pos = np.array([[0, 0, 0], [0, 2, 2], [2, 0, 2], [2, 2, 0]], dtype=float)
    types, names = [1, 1, 1, 2], ["Al", "Al", "Al", "Li"]
    lines = []
    for step in range(n_frames):
        lines += ["ITEM: TIMESTEP", str(10 * step), "ITEM: NUMBER OF ATOMS", "4",
                  "ITEM: BOX BOUNDS pp pp pp", "0 4", "0 4", "0 4", f"ITEM: ATOMS {cols}"]
        for i, (t, el, (x, y, z)) in enumerate(zip(types, names, pos + 0.01 * step), 1):
            mid = f"{t} {el}" if with_element else f"{t}"
            lines.append(f"{i} {mid} {x} {y} {z}")
    path.write_text("\n".join(lines) + "\n")
    return str(path)


def test_dump_with_element_column(tmp_path):
    df = load_dataset_any(_write_dump(tmp_path / "extrapolative.dump"))
    assert list(df.columns) == ["ase_atoms"]
    assert len(df) == 2
    at = df["ase_atoms"].iloc[1]
    assert at.get_chemical_symbols() == ["Al", "Al", "Al", "Li"]
    assert at.pbc.all()
    np.testing.assert_allclose(at.cell.lengths(), [4, 4, 4])
    np.testing.assert_allclose(at.positions[1], [0.01, 2.01, 2.01])


def test_dump_without_element_needs_mapping(tmp_path):
    path = _write_dump(tmp_path / "dump.lammpstrj", with_element=False)
    with pytest.raises(SystemExit, match="--dump-elements"):
        load_dataset_any(path)


def test_dump_without_element_uses_mapping(tmp_path):
    path = _write_dump(tmp_path / "dump.md", with_element=False)
    df = load_dataset_any(path, dump_elements=["Al", "Li"])
    assert df["ase_atoms"].iloc[0].get_chemical_symbols() == ["Al", "Al", "Al", "Li"]


def test_dump_mapping_too_short(tmp_path):
    path = _write_dump(tmp_path / "dump.md", with_element=False)
    with pytest.raises(SystemExit, match="type 2"):
        load_dataset_any(path, dump_elements=["Al"])


def test_dump_element_column_wins_over_mapping(tmp_path, caplog):
    path = _write_dump(tmp_path / "extrapolative.dump")
    with caplog.at_level(logging.WARNING, logger="grace_uq"):
        df = load_dataset_any(path, dump_elements=["Cu", "Ni"])
    assert df["ase_atoms"].iloc[0].get_chemical_symbols() == ["Al", "Al", "Al", "Li"]
    assert "--dump-elements" in caplog.text


def test_select_forwards_dump_elements_to_predict(tmp_path, monkeypatch):
    path = _write_dump(tmp_path / "pool.dump", with_element=False)
    seen = {}

    def capture(argv):
        seen["argv"] = argv
        return 1

    monkeypatch.setattr(predict_mod, "predict_main", capture)
    saved = tmp_path / "saved_model"
    saved.mkdir()
    with pytest.raises(RuntimeError, match="prediction step failed"):
        select_mod.select_main(["--dataset", path, "--model", str(saved),
                                "--dump-elements", "Al", "Li", "--strategy", "random-extrap", "-n", "1",
                                "--output", str(tmp_path / "sel.pkl.gz")])
    argv = seen["argv"]
    i = argv.index("--dump-elements")
    assert argv[i + 1:i + 3] == ["Al", "Li"]


def test_predict_workers_get_dump_elements(tmp_path, monkeypatch):
    seen = {}

    def capture(cmds, **kw):
        seen["cmds"] = cmds
        return []

    monkeypatch.setattr(predict_mod, "spawn_and_monitor", capture)
    args = predict_mod._build_parser().parse_args(
        ["--model", str(tmp_path), "--dataset", "pool.dump", "--n-workers", "2",
         "--dump-elements", "Al", "Li"])
    args.threads_per_worker_int = 1
    predict_mod._spawn_workers(args)
    for _, cmd in seen["cmds"]:
        i = cmd.index("--dump-elements")
        assert cmd[i + 1:i + 3] == ["Al", "Li"]
