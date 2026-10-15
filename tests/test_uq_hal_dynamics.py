import numpy as np
import pytest
from ase import units
from ase.build import bulk
from ase.calculators.calculator import Calculator, all_changes
from ase.calculators.emt import EMT
from ase.md.nose_hoover_chain import MTKNPT
from ase.md.velocitydistribution import MaxwellBoltzmannDistribution

from tensorpotential.uq.hal.calculator import HALStop
from tensorpotential.uq.hal.dynamics import (
    RampedLangevin,
    VolumeGuard,
    cluster_wall,
    default_timestep_fs,
    estimate_bulk_modulus,
    radius_of_gyration,
)


class CountingEMT(EMT):
    def __init__(self):
        super().__init__()
        self.n_calls = 0

    def calculate(self, atoms=None, properties=("energy",), system_changes=all_changes):
        self.n_calls += 1
        super().calculate(atoms, properties, system_changes)


class QuadraticEV(Calculator):
    implemented_properties = ["energy"]

    def __init__(self, v0, b0):
        super().__init__()
        self.v0, self.b0 = v0, b0

    def calculate(self, atoms=None, properties=("energy",), system_changes=all_changes):
        super().calculate(atoms, properties, system_changes)
        v = atoms.get_volume()
        self.results = {"energy": 0.5 * self.b0 * (v - self.v0) ** 2 / self.v0}


def _cu():
    atoms = bulk("Cu", "fcc", a=3.6, cubic=True) * (2, 2, 2)
    MaxwellBoltzmannDistribution(atoms, temperature_K=300, rng=np.random.default_rng(0))
    return atoms


def test_ramp_follows_schedule_with_one_call_per_step():
    atoms = _cu()
    v0 = atoms.get_volume()
    calc = CountingEMT()
    atoms.calc = calc
    dyn = RampedLangevin(atoms, 2 * units.fs, temperature_K=300, friction=0.01 / units.fs,
                         n_steps=20, v_start=1.0, v_end=1.2, rng=np.random.default_rng(1))
    dyn.run(10)
    assert atoms.get_volume() / v0 == pytest.approx(1.0 + 0.2 * 10 / 20, rel=1e-10)
    assert calc.n_calls == 10 + 1
    dyn.run(15)  # past n_steps: holds the end value
    assert atoms.get_volume() / v0 == pytest.approx(1.2, rel=1e-10)
    assert calc.n_calls == 25 + 1


def test_volume_guard_raises_outside_band():
    atoms = _cu()
    guard = VolumeGuard(atoms, lo=0.9, hi=1.1)
    guard()
    atoms.set_cell(atoms.cell * 1.05, scale_atoms=True)  # V/V0 = 1.157
    with pytest.raises(HALStop) as exc:
        guard()
    assert exc.value.reason.startswith("volume_guard")


def test_bulk_modulus():
    atoms = _cu()
    assert estimate_bulk_modulus(atoms, QuadraticEV(atoms.get_volume(), 0.8)) == pytest.approx(0.8, rel=1e-6)
    with pytest.raises(ValueError, match="bulk-modulus"):
        estimate_bulk_modulus(atoms, QuadraticEV(atoms.get_volume(), -0.8))


def test_default_timestep():
    assert default_timestep_fs([13, 13]) == 2.0
    assert default_timestep_fs([13, 3]) == 0.5


def test_cluster_wall_and_rg():
    from ase import Atoms

    atoms = Atoms("Cu2", positions=[[0, 0, 0], [2.0, 0, 0]])
    center, r_wall, k = cluster_wall(atoms, margin=2.0, k=1.5)
    np.testing.assert_allclose(center, [1.0, 0, 0])
    assert r_wall == pytest.approx(3.0) and k == 1.5
    assert radius_of_gyration(atoms) == pytest.approx(1.0)


def test_mtknpt_overwrites_externally_set_momenta():
    """Canary for HAL: ASE MTKNPT integrates its own momentum copy, so momenta set on the
    atoms between steps are discarded and the update-work removal cannot act on npt. If
    this starts failing, ASE honours external momenta and the removal could be enabled."""
    atoms = _cu()
    atoms.calc = EMT()
    dyn = MTKNPT(atoms, 2 * units.fs, temperature_K=300, pressure_au=units.bar,
                 tdamp=100 * units.fs, pdamp=1000 * units.fs)
    dyn.run(2)
    ke = atoms.get_kinetic_energy()
    atoms.set_momenta(np.zeros_like(atoms.get_momenta()))
    dyn.run(1)
    assert atoms.get_kinetic_energy() > 0.5 * ke
