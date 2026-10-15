"""run_hal validates every option before touching the output dir or loading the model."""

import os

import pytest
from ase import Atoms
from ase.build import bulk

from tensorpotential.uq import factories
from tensorpotential.uq.hal import run_hal

OLD = ("hal_collected.extxyz", "hal_history.npz", "hal_summary.json", "hal_settings.json")


def _old_walker(tmp_path):
    out = tmp_path / "w0"
    out.mkdir(parents=True)
    for f in OLD:
        (out / f).write_text("old")
    return out


def _cluster():
    a = bulk("Al", cubic=True)
    return Atoms(a.get_chemical_symbols(), positions=a.positions)  # no cell, no pbc


def _mixed():
    a = bulk("Al", cubic=True)
    a.pbc = [True, True, False]
    return a


BAD = [
    ({"trigger_min_atoms": 0}, "trigger_min_atoms"),
    ({"trigger_min_atoms": True}, "trigger_min_atoms"),
    ({"update_work_floor": -0.1}, "update_work_floor"),
    ({"fragment_mult": 0.0}, "fragment_mult"),
    ({"feature_gamma_min": -1.0}, "feature_gamma_min"),
    ({"update": "always"}, "update"),
    ({"alphas": (0.1, 0.5)}, "alpha = 1"),
    ({"zbl": "on"}, "zbl"),
    ({"zbl_switch": (0.8, 0.5)}, "zbl_switch"),
    ({"pair_overrides": {"Al-Qq": {"reject": 1.0}}}, "Qq"),
    ({"n_steps": 0}, "n_steps"),
    ({"tau_ema": 0}, "tau_ema"),
    ({"ensemble": "nph"}, "ensemble"),
    ({"ensemble": "cluster"}, "ensemble"),  # periodic structure
    ({"atoms": _cluster, "ensemble": "nvt"}, "ensemble"),
    ({"atoms": _cluster, "ensemble": "npt_stress"}, "ensemble"),
    ({"atoms": _mixed, "ensemble": "nvt"}, "ensemble"),
]


def _no_factory(monkeypatch):
    def boom(**kw):
        raise AssertionError("model factory must not be called")

    monkeypatch.setattr(factories, "get_gmm_uq_calculator", boom)


def test_bad_option_keeps_old_outputs_and_never_loads_model(tmp_path, monkeypatch):
    _no_factory(monkeypatch)
    for k, (bad, match) in enumerate(BAD):
        out = _old_walker(tmp_path / str(k))
        kw = dict(model_yaml="m.yaml", checkpoint="c", uq_artifact="a.npz", temperature_K=300,
                  n_steps=10, output_dir=str(out), overwrite=True)
        kw.update(bad)
        atoms = kw.pop("atoms", lambda: bulk("Al", cubic=True))()
        with pytest.raises(ValueError, match=match):
            run_hal(atoms, **kw)
        for f in OLD:
            assert (out / f).read_text() == "old", (bad, f)
        assert sorted(os.listdir(out)) == sorted(OLD), bad


def test_existing_output_refused_without_overwrite(tmp_path, monkeypatch):
    _no_factory(monkeypatch)
    out = _old_walker(tmp_path)
    with pytest.raises(FileExistsError, match=r"--output-dir.*--overwrite.*overwrite=True"):
        run_hal(bulk("Al", cubic=True), model_yaml="m.yaml", checkpoint="c", uq_artifact="a.npz",
                temperature_K=300, n_steps=10, output_dir=str(out))
    assert all((out / f).read_text() == "old" for f in OLD)


def test_good_options_reach_the_model_factory(tmp_path, monkeypatch):
    import numpy as np

    called = []

    def stop(**kw):
        called.append(kw)
        raise RuntimeError("stop here")

    monkeypatch.setattr(factories, "get_gmm_uq_calculator", stop)
    with pytest.raises(RuntimeError, match="stop here"):
        run_hal(bulk("Al", cubic=True), model_yaml="m.yaml", checkpoint="c",
                uq_artifact="a.npz", temperature_K=300, n_steps=10,
                output_dir=str(tmp_path / "w0"), pair_overrides={"Al-Al": {"reject": 1.0}},
                update="none", alphas=(0.1, 0.5), trigger_min_atoms=np.int64(2))
    assert len(called) == 1
