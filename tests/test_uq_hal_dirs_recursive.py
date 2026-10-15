"""grace_uq select --hal-dirs and grace_uq hal --pool find walker folders at any depth."""

import os

import pandas as pd
import pytest

from tensorpotential.uq.cli.hal import hal_main
from tensorpotential.uq.hal.tools import find_walker_dirs
from tests.test_uq_select_candidates import _select, _walkers, _write_artifact


def _tree(tmp_path, art):
    """grid/a/hal_w0, grid/a/hal_w1 (depth 2) and grid/b/deep/hal_w0 (depth 3)."""
    a = _walkers(tmp_path / "grid" / "a", art, n_frames=(3, 2), seed=1)
    b = _walkers(tmp_path / "grid" / "b" / "deep", art, n_frames=(4,), seed=2)
    return sorted(a + b)


def test_finds_walkers_at_any_depth(tmp_path):
    art = _write_artifact(tmp_path / "a.npz")
    walkers = _tree(tmp_path, art)
    assert find_walker_dirs([str(tmp_path / "grid")]) == walkers


def test_walker_dir_given_directly_and_no_duplicates(tmp_path):
    art = _write_artifact(tmp_path / "a.npz")
    walkers = _tree(tmp_path, art)
    got = find_walker_dirs([walkers[0], str(tmp_path / "grid"), walkers[0]])
    assert got == [walkers[0]] + [w for w in walkers if w != walkers[0]]


def test_walker_that_collected_nothing_is_found(tmp_path):
    empty = tmp_path / "grid" / "hal_w9"
    empty.mkdir(parents=True)
    (empty / "hal_settings.json").write_text("{}")
    assert find_walker_dirs([str(tmp_path / "grid")]) == [str(empty)]


def test_folder_without_walkers_is_an_error(tmp_path):
    (tmp_path / "nothing" / "here").mkdir(parents=True)
    with pytest.raises(ValueError, match="nothing"):
        find_walker_dirs([str(tmp_path / "nothing")])


def test_select_hal_dirs_parent_equals_explicit_list(tmp_path):
    art = _write_artifact(tmp_path / "a.npz")
    walkers = _tree(tmp_path, art)
    out_parent, out_list = tmp_path / "parent.pkl.gz", tmp_path / "list.pkl.gz"
    assert _select("--hal-dirs", tmp_path / "grid", "--artifact", art, "--output", out_parent) == 0
    assert _select("--hal-dirs", *walkers, "--artifact", art, "--output", out_list) == 0
    a, b = pd.read_pickle(out_parent), pd.read_pickle(out_list)
    assert len(a) > 0
    assert list(a["gamma_at_pick"]) == list(b["gamma_at_pick"])
    for x, y in zip(a["ase_atoms"], b["ase_atoms"]):
        assert (x.positions == y.positions).all()


def test_select_hal_dirs_without_walkers_stops(tmp_path):
    art = _write_artifact(tmp_path / "a.npz")
    (tmp_path / "empty").mkdir()
    with pytest.raises(SystemExit, match="empty"):
        _select("--hal-dirs", tmp_path / "empty", "--artifact", art,
                "--output", tmp_path / "sel.pkl.gz")


def test_hal_pool_parent_folder(tmp_path):
    art = _write_artifact(tmp_path / "a.npz")
    walkers = _tree(tmp_path, art)
    out = str(tmp_path / "pool.pkl.gz")
    assert hal_main(["--pool", str(tmp_path / "grid"), "--pool-output", out]) == 0
    df = pd.read_pickle(out)
    assert len(df) == 3 + 2 + 4
    assert sorted(df["hal_dir"].unique()) == walkers
    assert all(os.path.isabs(d) for d in df["hal_dir"])
