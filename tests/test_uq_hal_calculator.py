import numpy as np
import pytest
from ase import Atoms

from tensorpotential.uq import constants as uqc
from tensorpotential.uq.factories import get_gmm_uq_calculator
from tensorpotential.uq.hal.bias import model_virial_to_ase
from tensorpotential.uq.hal.calculator import HALCalculator, HALStop
from tests.hal_fakes import FakeUQCalculator


@pytest.fixture(scope="module")
def base_calc(hal_uq_setup):
    return get_gmm_uq_calculator(
        model_yaml=hal_uq_setup["model_yaml"],
        checkpoint=hal_uq_setup["checkpoint"],
        gmm_artifact_path=hal_uq_setup["artifact_path"],
        param_dtype="float64",
    )


@pytest.fixture
def structure(hal_uq_setup):
    return max(hal_uq_setup["atoms"], key=len).copy()


def _sum_gamma(base, atoms):
    base.calculate(atoms, ["energy", "forces", "stress"], ["positions"])
    return float(np.sum(base.results[uqc.ATOMIC_GAMMA]))


def test_tau_zero_matches_base(base_calc, structure):
    base_calc.calculate(structure, ["energy", "forces", "stress"], ["positions"])
    e0 = base_calc.results["energy"]
    f0 = base_calc.results["forces"].copy()
    s0 = np.array(base_calc.results["stress"]).copy()
    hal = HALCalculator(base_calc, tau=0.0, zbl="off")
    structure.calc = hal
    assert structure.get_potential_energy() == pytest.approx(e0)
    np.testing.assert_allclose(structure.get_forces(), f0)
    np.testing.assert_allclose(structure.get_stress(), s0)
    assert hal.n_base_calls == 1


def test_uncertainty_gradient_matches_finite_differences(base_calc, structure):
    """dsigma_dr = d(sum gamma)/dr and virial_sigma = d(sum gamma)/d(strain), float64 model.

    The structure is rattled by 0.02 A first: one atom of the training structure sits exactly
    on its GMM centroid, where sigma = sqrt(d^2 + eps) has its eps-smoothed cone tip and a
    central difference converges only as O(h) (benign: the bias gradient just goes smoothly
    to zero there). Off the centroid the convergence is O(h^2).
    """
    structure.rattle(0.02, seed=1)
    n = len(structure)
    _sum_gamma(base_calc, structure)
    grad = np.array(base_calc.results[uqc.DSIGMA_DR])[:n]
    w = model_virial_to_ase(base_calc.results[uqc.VIRIAL_SIGMA])
    u = np.random.default_rng(0).normal(size=(n, 3))
    h = 1e-4  # FD rel. error ~1e-5 here (1e-3 at h=1e-3, 1e-7 at h=1e-5)
    plus, minus = structure.copy(), structure.copy()
    plus.positions += h * u
    minus.positions -= h * u
    fd = (_sum_gamma(base_calc, plus) - _sum_gamma(base_calc, minus)) / (2 * h)
    assert np.sum(grad * u) == pytest.approx(fd, rel=5e-5)
    h = 1e-5  # sum gamma is strongly curved in volumetric strain: xx rel. error ~3e-5 here
    for (a, b), voigt in [((0, 0), 0), ((1, 2), 3)]:
        eps = np.zeros((3, 3))
        eps[a, b] = eps[b, a] = h
        p, m = structure.copy(), structure.copy()
        p.set_cell(structure.cell @ (np.eye(3) + eps), scale_atoms=True)
        m.set_cell(structure.cell @ (np.eye(3) - eps), scale_atoms=True)
        d = (_sum_gamma(base_calc, p) - _sum_gamma(base_calc, m)) / (2 * h)
        assert w[voigt] * (1.0 if a == b else 2.0) == pytest.approx(d, rel=1e-4)


def test_bias_added_along_gradient_after_warmup(base_calc, structure):
    hal = HALCalculator(base_calc, tau=0.5, tau_ema=1, zbl="off")
    hal.calculate(structure, ["forces"], ["positions"])
    assert hal.results["tau_r"] == 0.0
    hal.calculate(structure, ["forces"], ["positions"])
    tau_r = hal.results["tau_r"]
    assert tau_r > 0
    grad = np.array(base_calc.results[uqc.DSIGMA_DR])[: len(structure)]
    np.testing.assert_allclose(
        hal.results["forces"] - hal.results["grace_forces"], tau_r * grad, atol=1e-10
    )


def test_stress_bias_along_virial_sigma(base_calc, structure):
    base_calc.calculate(structure, ["energy", "forces", "stress"], ["positions"])
    w = model_virial_to_ase(base_calc.results[uqc.VIRIAL_SIGMA])
    w = w / np.linalg.norm(w)
    hal = HALCalculator(base_calc, tau=0.0, zbl="off", stress_tau=0.1, bulk_modulus=0.5)
    hal.calculate(structure, ["stress"], ["positions"])
    np.testing.assert_allclose(
        hal.results["stress"] - hal.results["grace_stress"], -0.1 * 0.5 * w, atol=1e-10
    )


def test_calculator_guards():
    fake = FakeUQCalculator()
    hal = HALCalculator(fake)
    fake_calc = fake.calculate

    def no_grad(atoms, properties=None, system_changes=None):
        fake_calc(atoms)
        del fake.results[uqc.DSIGMA_DR]

    fake.calculate = no_grad
    with pytest.raises(RuntimeError, match="dsigma_dr"):
        hal.calculate(Atoms("Al2", positions=[[0, 0, 0], [3, 0, 0]]), ["forces"], ["positions"])
    hal = HALCalculator(FakeUQCalculator(gamma=[np.nan, 1.0]))
    with pytest.raises(HALStop) as exc:
        hal.calculate(Atoms("Al2", positions=[[0, 0, 0], [3, 0, 0]]), ["forces"], ["positions"])
    assert exc.value.reason == "non_finite"


def test_zbl_guard_and_bond_ratio():
    hal = HALCalculator(FakeUQCalculator(), tau=0.0)
    atoms = Atoms("Al2", positions=[[0, 0, 0], [1.0, 0, 0]])
    hal.calculate(atoms, ["forces"], ["positions"])
    assert hal.results["bond_ratio_min"] < 1.0
    far = Atoms("Al2", positions=[[0, 0, 0], [2.8, 0, 0]])
    hal.calculate(far, ["forces"], ["positions"])
    assert hal.results["bond_ratio_min"] == np.inf
    assert not hal.results["forces"].any()


def test_neighbour_list_skip(monkeypatch):
    from tensorpotential.uq.hal import calculator as calc_mod

    calls = []
    real = calc_mod.neighbor_list
    monkeypatch.setattr(calc_mod, "neighbor_list", lambda *a, **k: calls.append(1) or real(*a, **k))
    fake = FakeUQCalculator()
    fake.cutoff = 6.0
    hal = HALCalculator(fake, tau=0.0)
    far = Atoms("Al2", positions=[[0, 0, 0], [2.8, 0, 0]])
    fake.current_min_dist = 2.8
    hal.calculate(far, ["forces"], ["positions"])
    assert calls == []  # Al-Al pair cutoff is 1.8 A < 2.8 A
    assert hal.results["bond_ratio_min"] == np.inf and not hal.results["forces"].any()
    close = Atoms("Al2", positions=[[0, 0, 0], [1.0, 0, 0]])
    fake.current_min_dist = 1.0
    hal.calculate(close, ["forces"], ["positions"])
    assert calls == [1]
    assert hal.results["bond_ratio_min"] < 1.0 and hal.results["forces"].any()
    # a base cutoff shorter than the pair cutoff cannot rule pairs out: fall back to the list
    fake.cutoff = 1.0
    fake.current_min_dist = 2.8
    HALCalculator(fake, tau=0.0).calculate(far, ["forces"], ["positions"])
    assert calls == [1, 1]
    # per-pair model cutoffs: the base bond list is used unless a pair cutoff is too short
    from tensorpotential import constants

    calls.clear()
    fake = FakeUQCalculator()
    fake.cutoff, fake.dense_reshape, fake.current_min_dist = 6.0, False, 1.0
    fake.data = {
        constants.BOND_IND_I: np.array([0, 1]),
        constants.BOND_IND_J: np.array([1, 0]),
        constants.BOND_VECTOR: np.array([[1.0, 0, 0], [-1.0, 0, 0]]),
    }
    atoms = Atoms("Al2", positions=[[0, 0, 0], [1.0, 0, 0]])
    fake.cutoff_dict = {("Al", "Al"): 6.0}
    hal = HALCalculator(fake, tau=0.0)
    hal.calculate(atoms, ["forces"], ["positions"])
    assert calls == [] and hal.results["bond_ratio_min"] < 1.0
    fake.cutoff_dict = {("Al", "Al"): 1.0}  # model pair cutoff below the ZBL cutoff (1.8 A)
    HALCalculator(fake, tau=0.0).calculate(atoms, ["forces"], ["positions"])
    assert calls == [1]


def _compress_first_pair(atoms, frac=0.6):
    from ase.data import covalent_radii

    atoms = atoms.copy()
    s = covalent_radii[atoms.numbers[0]] + covalent_radii[atoms.numbers[1]]
    u = atoms.positions[1] - atoms.positions[0]
    atoms.positions[1] = atoms.positions[0] + frac * s * u / np.linalg.norm(u)
    return atoms


@pytest.mark.parametrize("periodic", [True, False])
def test_base_bond_list_matches_neighbour_list(base_calc, structure, monkeypatch, periodic):
    from tensorpotential.uq.hal import calculator as calc_mod

    atoms = _compress_first_pair(structure)
    if not periodic:
        atoms.set_pbc(False)
        atoms.set_cell(None)
    ref = HALCalculator(base_calc, tau=0.0)
    monkeypatch.setattr(ref, "_base_pairs", lambda atoms: None)
    ref.calculate(atoms.copy(), ["forces"], ["positions"])
    assert ref.results["bond_ratio_min"] < 1.0 and np.abs(ref.results["forces"]).max() > 0
    if periodic:  # HAL = GRACE + switched ZBL (tau = 0): energy, forces and stress
        from ase.neighborlist import neighbor_list

        from tensorpotential.uq.hal.bias import PairTable, zbl_switched

        table = PairTable()
        i, j, d, D = neighbor_list("ijdD", atoms, table.max_cutoff(atoms.numbers))
        z = atoms.numbers
        r_in, r_out, _ = table.radii(z[i], z[j])
        e_z, f_z, s_z = zbl_switched(i, j, d, D, z[i], z[j], r_in, r_out, len(atoms),
                                     atoms.get_volume())
        assert e_z > 0
        r = ref.results
        assert r["energy"] - r["grace_energy"] == pytest.approx(e_z, rel=1e-10)
        np.testing.assert_allclose(r["forces"] - r["grace_forces"], f_z, atol=1e-10)
        np.testing.assert_allclose(r["stress"] - r["grace_stress"], s_z, atol=1e-10)

    def no_neighbor_list(*a, **k):
        raise AssertionError("neighbor_list called on the base-bond-list path")

    monkeypatch.setattr(calc_mod, "neighbor_list", no_neighbor_list)
    hal = HALCalculator(base_calc, tau=0.0)
    hal.calculate(atoms.copy(), ["forces"], ["positions"])
    for key in ("energy", "forces", "stress"):
        np.testing.assert_allclose(hal.results[key], ref.results[key], rtol=1e-5, atol=1e-6)
    assert hal.results["bond_ratio_min"] == pytest.approx(ref.results["bond_ratio_min"], rel=1e-5)


def test_wall_adds_restoring_force_not_energy():
    hal = HALCalculator(FakeUQCalculator(), tau=0.0, zbl="off", wall=(np.zeros(3), 3.0, 1.0))
    atoms = Atoms("Al2", positions=[[0, 0, 0], [4.0, 0, 0]])
    hal.calculate(atoms, ["forces"], ["positions"])
    assert hal.results["forces"][1, 0] == pytest.approx(-1.0)
    assert hal.results["wall_energy"] == pytest.approx(0.5)
    assert hal.results["energy"] == 0.0 and hal.results["n_outside_wall"] == 1
