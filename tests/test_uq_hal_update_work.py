"""Removal of the bias energy injected by a live CALM update."""

import itertools

import numpy as np
import pytest
from ase.build import bulk
from ase.io import read
from ase.md.velocitydistribution import MaxwellBoltzmannDistribution, Stationary

from tensorpotential.uq.hal.collector import HISTORY_FIELDS, HALCollector
from tensorpotential.uq.hal.dynamics import internal_temperature
from tests.hal_fakes import FAKE_META, FakeHALCalc, FakeLive, FakePristine

T_TARGET = 300.0


_DIRS = itertools.count()  # every collector its own output dir (frames append)


def _setup(tmp_path, *, t_start=600.0, gamma_after=2.9, tau_r=0.1, update="alpha_search", **kw):
    atoms = bulk("Cu", "fcc", a=3.6, cubic=True) * (2, 2, 2)  # 32 atoms
    MaxwellBoltzmannDistribution(atoms, temperature_K=t_start, rng=np.random.default_rng(0))
    Stationary(atoms)
    live = FakeLive((2.0, 1.3, 1.1, 0.9, 0.5, 0.4), gamma_after=gamma_after)
    calc = FakeHALCalc(len(atoms), live if update == "alpha_search" else None)
    calc.tau_r = tau_r
    col = HALCollector(atoms, calc, pristine=FakePristine(), feature_meta=FAKE_META, element_indices=np.zeros(len(atoms), int),
                       output_dir=str(tmp_path / f"w{next(_DIRS)}"), equil_steps=0, trigger_gap=1, update=update,
                       temperature_K=T_TARGET, **kw)
    return atoms, calc, live, col


def _collect_once(col, calc, gamma=3.0):
    calc.gamma = np.full(calc.n, gamma)
    col()


def test_kinetic_energy_drops_by_update_work(tmp_path):
    atoms, calc, live, col = _setup(tmp_path)
    calc.gamma = np.full(calc.n, 1.0)
    col()  # below trigger: no collection
    ke0, p0 = atoms.get_kinetic_energy(), atoms.get_momenta().copy()
    t_before = internal_temperature(atoms)
    _collect_once(col, calc)
    work = 0.1 * 32 * (3.0 - 2.9)  # tau_r * (sum gamma before - sum gamma after)
    assert live.n_gamma_calls == 1  # one extra gamma evaluation per collection
    assert atoms.get_kinetic_energy() == pytest.approx(ke0 - work, rel=1e-12)
    # uniform rescaling: direction of every momentum kept
    np.testing.assert_allclose(atoms.get_momenta(), p0 * np.sqrt((ke0 - work) / ke0), rtol=1e-12)
    c = col.collections[0]
    assert c["update_work"] == pytest.approx(work) and c["update_work_removed"] == pytest.approx(work)
    f = read(col.frames_path, 0)
    assert f.info["update_work"] == pytest.approx(work)
    assert f.info["update_work_removed"] == pytest.approx(work)
    assert f.info["T"] == pytest.approx(internal_temperature(atoms))  # frame is post-removal
    assert f.info["T_before_removal"] == pytest.approx(t_before, rel=1e-12)
    assert f.info["T_before_removal"] > f.info["T"]
    h = col.history.as_dict()
    assert "update_work" in HISTORY_FIELDS
    assert np.isnan(h["update_work"][0]) and h["update_work"][1] == pytest.approx(work)
    summary = col.finalize("max_steps")
    assert summary["temperature_mean_K"] == pytest.approx(np.mean(h["temperature"]))
    assert summary["update_work_total_eV"] == pytest.approx(work)
    assert summary["update_work_removed_eV"] == pytest.approx(work)


def test_floor_respected_and_remainder_recorded(tmp_path):
    atoms, calc, _, col = _setup(tmp_path, t_start=400.0, gamma_after=0.0, update_work_floor=0.5)
    ke0 = atoms.get_kinetic_energy()
    t0 = internal_temperature(atoms)
    _collect_once(col, calc)  # work = 0.1 * 32 * 3 = 9.6 eV >> ke0 (~1.6 eV)
    assert internal_temperature(atoms) == pytest.approx(0.5 * T_TARGET, rel=1e-10)
    removed = ke0 * (1.0 - 0.5 * T_TARGET / t0)
    c = col.collections[0]
    assert c["update_work"] == pytest.approx(9.6)
    assert c["update_work_removed"] == pytest.approx(removed, rel=1e-10)
    summary = col.finalize("max_steps")
    assert summary["update_work_total_eV"] == pytest.approx(9.6)
    assert summary["update_work_removed_eV"] == pytest.approx(removed, rel=1e-10)
    # already below the floor: never heated up to it
    atoms, calc, _, col = _setup(tmp_path, t_start=100.0, update_work_floor=0.5)
    p0 = atoms.get_momenta().copy()
    _collect_once(col, calc)
    np.testing.assert_array_equal(atoms.get_momenta(), p0)
    assert col.collections[0]["update_work"] > 0 and col.collections[0]["update_work_removed"] == 0.0


def test_no_removal_cases(tmp_path):
    """(setup kwargs, expected update_work, expected gamma evaluations); momenta untouched."""
    cases = [
        (dict(remove_update_work=False), 0.32, 1),  # still measured
        (dict(tau_r=0.0), 0.0, 0),  # bias off (tau_ema warm-up)
        (dict(update="none"), 0.0, 0),  # no live update
        (dict(gamma_after=3.5), 0.1 * 32 * -0.5, 1),  # gamma went up: recorded, not removed
    ]
    for kw, work, n_gamma in cases:
        atoms, calc, live, col = _setup(tmp_path, **kw)
        p0 = atoms.get_momenta().copy()
        _collect_once(col, calc)
        np.testing.assert_array_equal(atoms.get_momenta(), p0)
        c = col.collections[0]
        assert c["update_work"] == pytest.approx(work) and c["update_work_removed"] == 0.0, kw
        if live is not None and kw.get("update") != "none":
            assert live.n_gamma_calls == n_gamma, kw
        f = read(col.frames_path, -1)
        assert f.info["T_before_removal"] == pytest.approx(f.info["T"], rel=1e-12)
