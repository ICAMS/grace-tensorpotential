import numpy as np
import pytest
from ase import Atoms
from ase.neighborlist import neighbor_list

from tensorpotential.uq.hal.bias import (
    ForceBiasScaler,
    PairTable,
    StressBiasDirection,
    containment_wall,
    model_virial_to_ase,
    stress_bias,
    zbl_switched,
)


def test_wall_forces_match_finite_differences():
    rng = np.random.default_rng(1)
    pos = rng.uniform(-6, 6, (6, 3))
    c = np.array([0.2, -0.1, 0.3])
    _, f, _ = containment_wall(pos, c, 4.0, 1.5)
    h = 1e-6
    for a in range(6):
        for d in range(3):
            p, m = pos.copy(), pos.copy()
            p[a, d] += h
            m[a, d] -= h
            fd = -(containment_wall(p, c, 4.0, 1.5)[0] - containment_wall(m, c, 4.0, 1.5)[0]) / (2 * h)
            assert f[a, d] == pytest.approx(fd, rel=1e-5, abs=1e-8)


def test_force_bias_scaler():
    s = ForceBiasScaler(tau=0.5, ema=3)
    f = np.ones((4, 3))
    g = 2 * np.ones((4, 3))
    assert [s(f, g) for _ in range(3)] == [0.0, 0.0, 0.0]  # warm-up
    assert s(f, g) == pytest.approx(0.5 * 0.5)  # tau * |F| / |g| = 0.5 * 1/2
    s = ForceBiasScaler(tau=1.0, ema=2)  # EMA of the norms
    s(np.ones((1, 3)), np.ones((1, 3)))
    s(3 * np.ones((1, 3)), np.ones((1, 3)))
    assert s.f_norm == pytest.approx(np.sqrt(3) * (1 + 0.5 * (3 - 1)))


def test_pair_table_covalent_defaults_and_override():
    t = PairTable(zbl_switch=(0.55, 0.75), reject_factor=0.65)
    r_in, r_out, r_rej = t.radii(np.array([13, 1]), np.array([13, 1]))
    np.testing.assert_allclose(r_out, [0.75 * 2 * 1.21, 0.75 * 2 * 0.31])
    np.testing.assert_allclose(r_rej, [0.65 * 2 * 1.21, 0.65 * 2 * 0.31])
    t2 = PairTable(overrides={"Al-Al": {"zbl": [1.6, 2.4], "reject": 1.5}})
    r_in, r_out, r_rej = t2.radii(np.array([13]), np.array([13]))
    assert (r_in[0], r_out[0], r_rej[0]) == (1.6, 2.4, 1.5)
    assert t2.max_cutoff([13, 13]) == pytest.approx(2.4)


def _zbl(atoms, table):
    cut = table.max_cutoff(atoms.numbers)
    i, j, d, D = neighbor_list("ijdD", atoms, cut)
    z = atoms.numbers
    r_in, r_out, _ = table.radii(z[i], z[j])
    vol = atoms.get_volume() if atoms.cell.rank == 3 else None
    return zbl_switched(i, j, d, D, z[i], z[j], r_in, r_out, len(atoms), vol)


def test_zbl_forces_match_finite_differences():
    # distances 1.1-1.7 A; the H-Al pairs sit inside the switching zone
    atoms = Atoms("AlAlHAl", positions=[[0, 0, 0], [1.1, 0.2, 0], [0.3, 1.2, 0.1], [1.0, 1.0, 0.9]])
    table = PairTable()
    _, forces, _ = _zbl(atoms, table)
    h = 1e-5
    for a in range(4):
        for c in range(3):
            p, m = atoms.copy(), atoms.copy()
            p.positions[a, c] += h
            m.positions[a, c] -= h
            fd = -(_zbl(p, table)[0] - _zbl(m, table)[0]) / (2 * h)
            assert forces[a, c] == pytest.approx(fd, rel=1e-4, abs=1e-6)


def test_zbl_stress_matches_strain_derivative():
    atoms = Atoms("Al2", positions=[[0, 0, 0], [1.2, 0.3, 0.1]], cell=[4, 4, 4], pbc=True)
    table = PairTable()
    _, _, stress = _zbl(atoms, table)
    h = 1e-6
    for (a, b), voigt in [((0, 0), 0), ((1, 1), 1), ((2, 2), 2), ((1, 2), 3), ((0, 2), 4), ((0, 1), 5)]:
        eps = np.zeros((3, 3))
        eps[a, b] = eps[b, a] = h
        p, m = atoms.copy(), atoms.copy()
        p.set_cell(atoms.cell @ (np.eye(3) + eps), scale_atoms=True)
        m.set_cell(atoms.cell @ (np.eye(3) - eps), scale_atoms=True)
        de = (_zbl(p, table)[0] - _zbl(m, table)[0]) / (2 * h)
        factor = 1.0 if a == b else 2.0
        assert stress[voigt] == pytest.approx(de / (factor * atoms.get_volume()), rel=1e-4, abs=1e-8)


def test_zbl_switches_off_smoothly():
    table = PairTable()
    r_out = table.radii(np.array([13]), np.array([13]))[1][0]
    far = Atoms("Al2", positions=[[0, 0, 0], [r_out - 1e-7, 0, 0]])
    e, f, _ = _zbl(far, table)
    assert abs(e) < 1e-8 and np.abs(f).max() < 1e-5
    beyond = Atoms("Al2", positions=[[0, 0, 0], [r_out + 0.1, 0, 0]])
    e, f, _ = _zbl(beyond, table)
    assert e == 0.0 and not f.any()


def test_stress_bias_direction_is_unit_and_smoothed():
    np.testing.assert_array_equal(model_virial_to_ase([0, 1, 2, 3, 4, 5]), [0, 1, 2, 5, 4, 3])
    s = StressBiasDirection(ema=2)
    w1 = s(np.array([2.0, 0, 0, 0, 0, 0]))
    np.testing.assert_allclose(w1, [1, 0, 0, 0, 0, 0])
    w2 = s(np.array([0, 3.0, 0, 0, 0, 0]))
    assert np.linalg.norm(w2) == pytest.approx(1.0)
    assert w2[0] > 0 and w2[1] > 0
    np.testing.assert_allclose(stress_bias(w1, 0.1, 0.5), -0.05 * w1)


def test_pair_table_rejects_malformed_overrides():
    for overrides, match in [({"Al-Xx": {"reject": 1.0}}, "Xx"),
                             ({"Al-Al": {"cutoff": 1.0}}, "cutoff"),
                             ({"Al-Al": {"zbl": [2.0, 1.0]}}, "r_in < r_out")]:
        with pytest.raises(ValueError, match=match):
            PairTable(overrides=overrides)
