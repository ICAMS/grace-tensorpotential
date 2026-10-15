"""HAL thermostat: internal (3N-3 dof) temperature at the target with mixed masses."""

import numpy as np
import pytest
from ase import Atoms, units
from ase.calculators.lj import LennardJones
from ase.cluster import Icosahedron

from tensorpotential.uq import constants as uqc
from tensorpotential.uq.hal import run_hal
from tensorpotential.uq.hal.dynamics import internal_temperature
from tests.hal_fakes import FakePristine

ETHANOL_MASSES = [12.011, 12.011, 15.999] + [1.008] * 6


class _LJBase:
    """Base-calculator stand-in: Lennard-Jones forces, zero gamma (no bias, no collections)."""

    element_map = {"Ar": 0}
    gmm_uq_model = None
    cutoff = 10.0  # covers the reject cutoff, so HALCalculator skips its own neighbour list
    current_min_dist = np.inf

    def __init__(self):
        self.lj = LennardJones(sigma=3.4, epsilon=0.3, rc=10.0, smooth=True)
        self.results = {}

    def calculate(self, atoms, properties=None, system_changes=None):
        n = len(atoms)
        self.results = {
            "energy": self.lj.get_potential_energy(atoms), "forces": self.lj.get_forces(atoms),
            "stress": np.zeros(6), uqc.ATOMIC_GAMMA: np.zeros(n), uqc.DSIGMA_DR: np.zeros((n, 3)),
            uqc.FEATURES: np.zeros((n, 4)),
        }


def test_internal_temperature_uses_3n_minus_3_dof():
    atoms = Atoms("Ar4", positions=np.eye(4, 3) * 4.0)
    atoms.set_momenta(np.arange(12.0).reshape(4, 3))
    assert internal_temperature(atoms) == pytest.approx(atoms.get_temperature() * 12 / 9)
    assert internal_temperature(Atoms("Ar", positions=[[0, 0, 0]])) == 0.0


def _run_cluster(tmp_path, monkeypatch, n_steps, seed=1, friction=0.1):
    """run_hal(ensemble='cluster') on a 9-atom LJ cluster with ethanol masses.

    Returns (history, summary, KE per collector call); the KE is read by a collector spy
    after the per-step COM reset, independent of the recorded temperature.
    """
    import tensorpotential.uq.factories as factories
    import tensorpotential.uq.gmmuq as gmmuq
    from tensorpotential.uq.hal import run as run_mod

    ke = []

    class Spy(run_mod.HALCollector):
        def __call__(self):
            ke.append(self.atoms.get_kinetic_energy())
            super().__call__()

    monkeypatch.setattr(factories, "get_gmm_uq_calculator", lambda **kw: _LJBase())
    monkeypatch.setattr(gmmuq, "GMMUQModel", lambda path: FakePristine())
    monkeypatch.setattr(run_mod, "HALCollector", Spy)
    atoms = Atoms("Ar9", positions=Icosahedron("Ar", 2).positions[:9])
    atoms.set_masses(ETHANOL_MASSES)
    res = run_hal(atoms, model_yaml="m.yaml", checkpoint="c", uq_artifact="a.npz", temperature_K=300.0,
                  n_steps=n_steps, ensemble="cluster", timestep_fs=1.0, friction=friction, tau=0.0,
                  update="none", zbl="off", equil_steps=0, history_every=0, plot=False,
                  seed=seed, output_dir=str(tmp_path / "w0"), overwrite=True,
                  feature_gamma_min=None)  # no sidecar: the artifact path is fake
    return res.history, res.summary, np.array(ke)


def test_cluster_mixed_masses_internal_temperature_at_target(tmp_path, monkeypatch):
    """Per-step COM reset without temperature preservation, and T recorded over 3N-3 dof.

    With Stationary(preserve_temperature=True) every step the internal modes sit at
    T*N/(N-1) (+12.5 % for N = 9); a 3N-dof record would read T*(N-1)/N (-11 %).
    Statistics: one sample of T over 24 internal dof fluctuates by sqrt(2/24) = 29 %. Over
    eight seeds this run (8 ps at friction 0.1/fs, 0.5 ps burn-in) gave a mean internal T of
    298.4 K with a seed-to-seed spread of 4.7 K (1.6 %). The 5 % tolerance is ~3 sigma and
    far from the +12.5 % of the preserve_temperature=True bug (~337 K).
    """
    n_steps, burn_in = 8000, 500
    hist, summary, ke = _run_cluster(tmp_path, monkeypatch, n_steps)
    t = hist["temperature"]
    assert len(t) == len(ke) >= n_steps
    t_int = 2.0 * ke / ((3 * 9 - 3) * units.kB)  # independent of the code under test
    np.testing.assert_allclose(t, t_int, rtol=1e-12)  # recorded T is the 3N-3 one
    assert np.mean(t_int[burn_in:]) == pytest.approx(300.0, rel=0.05)
    assert summary["temperature_mean_K"] == pytest.approx(np.mean(t))
