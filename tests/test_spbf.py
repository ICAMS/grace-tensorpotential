"""Tests for the SPBF (Single Particle Basis Function) instruction."""

import os

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

import numpy as np
import tensorflow as tf
import pytest
from scipy.spatial.transform import Rotation
from tensorflow import float64

from tensorpotential.instructions import (
    BondLength,
    FunctionReduce,
    FunctionReduceN,
    InstructionManager,
    ProductFunction,
    ScalarChemicalEmbedding,
    ScaledBondVector,
    SingleParticleBasisFunctionScalarInd,
    SingleParticleBasisFunctionEquivariantInd,
    SphericalHarmonic,
    RadialBasis,
    MLPRadialFunction_v2,
    SPBF,
)
from tensorpotential import constants
from tensorpotential.utils import Parity


def _make_input_dict(n_types=2, seed=322):
    """Build a minimal input_data dict from two copies of 3 atoms (original + rotated)."""
    np.random.seed(seed)

    axis = np.array([1, 1, 3], dtype=float)
    theta = np.pi / 2.2
    axis = axis / np.linalg.norm(axis)
    rot = Rotation.from_rotvec(theta * axis)

    coord = np.array([[0.2, 0.3, 2.1], [0.1, 3.2, 0.5], [0.3, 1, 2]])
    coord_r = rot.apply(coord)
    coord_2 = np.vstack([coord, coord_r]).astype(np.float64)

    indj = [1, 2, 0, 2, 0, 1, 4, 5, 3, 5, 3, 4]
    indi = [0, 0, 1, 1, 2, 2, 3, 3, 4, 4, 5, 5]
    atomic_mu_i = [0, 1, 0, 0, 1, 0]
    mui = np.take(atomic_mu_i, indi, axis=0)
    muj = np.take(atomic_mu_i, indj, axis=0)
    rj = np.take(coord_2, indj, axis=0)
    ri = np.take(coord_2, indi, axis=0)
    rij = rj - ri

    return {
        constants.BOND_VECTOR: rij,
        constants.BOND_IND_J: indj,
        constants.BOND_MU_I: mui,
        constants.BOND_MU_J: muj,
        constants.BOND_IND_I: indi,
        constants.N_ATOMS_BATCH_TOTAL: len(coord_2),
        constants.ATOMIC_MU_I: atomic_mu_i,
    }


def _build_base_instructions(inpt_dict, lmax, rcut, embedding_size=32):
    """Build BondLength, ScaledBondVector, SphericalHarmonic, ChemEmb and add to input dict."""
    d_ij = BondLength()
    d_ij.build(float64)
    inpt_dict = d_ij(inpt_dict)

    rhat = ScaledBondVector(bond_length=d_ij)
    rhat.build(float64)
    inpt_dict = rhat(inpt_dict)

    Y = SphericalHarmonic(vhat=rhat, lmax=lmax, name="Y")
    Y.build(float64)
    inpt_dict = Y(inpt_dict)

    z = ScalarChemicalEmbedding(
        element_map={"H": 0, "C": 1}, embedding_size=embedding_size, name="Z"
    )
    z.build(float64)
    inpt_dict = z(inpt_dict)

    return inpt_dict, d_ij, Y, z


# ---------------------------------------------------------------------------
# Test: SPBF scalar mode (no indicator) produces correct shapes
# ---------------------------------------------------------------------------
def test_spbf_scalar_no_indicator_shape():
    lmax = 3
    n_rad_max = 4
    n_rad_basis = 8
    rcut = 6.0

    np.random.seed(322)
    tf.random.set_seed(322)

    inpt_dict = _make_input_dict()
    inpt_dict, d_ij, Y, z = _build_base_instructions(inpt_dict, lmax, rcut)

    spbf = SPBF(
        name="A",
        bonds=d_ij,
        angular=Y,
        chemical_embedding=z,
        n_rad_max=n_rad_max,
        n_rad_basis=n_rad_basis,
        rcut=rcut,
    )
    spbf.build(float64)
    inpt_dict = spbf(inpt_dict)

    out = inpt_dict["A"]
    n_atoms = 6
    n_lm = (lmax + 1) ** 2
    assert out.shape == (n_atoms, n_rad_max, n_lm)


# ---------------------------------------------------------------------------
# Test: SPBF scalar mode (with indicator) produces correct shapes
# ---------------------------------------------------------------------------
def test_spbf_scalar_with_indicator_shape():
    lmax = 3
    n_rad_max = 4
    n_rad_basis = 8
    rcut = 6.0

    np.random.seed(322)
    tf.random.set_seed(322)

    inpt_dict = _make_input_dict()
    inpt_dict, d_ij, Y, z = _build_base_instructions(inpt_dict, lmax, rcut)

    spbf = SPBF(
        name="A",
        bonds=d_ij,
        angular=Y,
        chemical_embedding=z,
        n_rad_max=n_rad_max,
        n_rad_basis=n_rad_basis,
        rcut=rcut,
        indicator=z,
    )
    spbf.build(float64)
    inpt_dict = spbf(inpt_dict)

    out = inpt_dict["A"]
    n_atoms = 6
    n_lm = (lmax + 1) ** 2
    assert out.shape == (n_atoms, n_rad_max, n_lm)


# ---------------------------------------------------------------------------
# Test: SPBF scalar mode is rotationally invariant (L=0 output)
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("use_indicator", [False, True])
def test_spbf_scalar_rotational_invariance(use_indicator):
    lmax = 4
    n_rad_max = 5
    n_rad_basis = 8
    rcut = 6.0

    np.random.seed(322)
    tf.random.set_seed(322)

    inpt_dict = _make_input_dict()
    inpt_dict, d_ij, Y, z = _build_base_instructions(inpt_dict, lmax, rcut)

    indicator = z if use_indicator else None

    A = SPBF(
        name="A",
        bonds=d_ij,
        angular=Y,
        chemical_embedding=z,
        n_rad_max=n_rad_max,
        n_rad_basis=n_rad_basis,
        rcut=rcut,
        indicator=indicator,
    )
    A.build(float64)
    inpt_dict = A(inpt_dict)

    # Product to get invariants
    AA = ProductFunction(left=A, right=A, name="AA", lmax=lmax, Lmax=0)
    AA.build(float64)
    inpt_dict = AA(inpt_dict)

    out = inpt_dict["AA"].numpy()
    # First 3 atoms = original, next 3 = rotated -> should match
    assert np.allclose(out[:3], out[3:], atol=1e-10), (
        f"Rotational invariance failed: max diff = {np.max(np.abs(out[:3] - out[3:]))}"
    )


# ---------------------------------------------------------------------------
# Test: SPBF equivariant mode shape and rotational invariance
# ---------------------------------------------------------------------------
def test_spbf_equivariant_rotational_invariance():
    lmax = 3
    n_rad_max = 4
    n_rad_basis = 8
    rcut = 6.0

    np.random.seed(322)
    tf.random.set_seed(322)

    inpt_dict = _make_input_dict()
    inpt_dict, d_ij, Y, z = _build_base_instructions(inpt_dict, lmax, rcut)

    # First layer: scalar SPBF
    A = SPBF(
        name="A",
        bonds=d_ij,
        angular=Y,
        chemical_embedding=z,
        n_rad_max=n_rad_max,
        n_rad_basis=n_rad_basis,
        rcut=rcut,
    )
    A.build(float64)
    inpt_dict = A(inpt_dict)

    # Product to get equivariant indicator
    AA = ProductFunction(left=A, right=A, name="AA", lmax=lmax, Lmax=lmax)
    AA.build(float64)
    inpt_dict = AA(inpt_dict)

    # Second layer: equivariant SPBF
    B = SPBF(
        name="B",
        bonds=d_ij,
        angular=Y,
        chemical_embedding=z,
        n_rad_max=n_rad_max,
        n_rad_basis=n_rad_basis,
        rcut=rcut,
        indicator=AA,
        lmax=lmax,
        Lmax=lmax,
        keep_parity=Parity.REAL_PARITY,
        normalize_cg=True,
    )
    B.build(float64)
    inpt_dict = B(inpt_dict)

    # Reduce to invariants via another product
    BB = ProductFunction(left=B, right=B, name="BB", lmax=lmax, Lmax=0)
    BB.build(float64)
    inpt_dict = BB(inpt_dict)

    out = inpt_dict["BB"].numpy()
    assert np.allclose(out[:3], out[3:], atol=1e-10), (
        f"Rotational invariance failed: max diff = {np.max(np.abs(out[:3] - out[3:]))}"
    )


# ---------------------------------------------------------------------------
# Test: SPBF equivariant mode with chemical embedding
# ---------------------------------------------------------------------------
def test_spbf_equivariant_with_chemical_embedding():
    lmax = 3
    n_rad_max = 4
    n_rad_basis = 8
    rcut = 6.0

    np.random.seed(322)
    tf.random.set_seed(322)

    inpt_dict = _make_input_dict()
    inpt_dict, d_ij, Y, z = _build_base_instructions(inpt_dict, lmax, rcut)

    # First layer: scalar SPBF with indicator
    A = SPBF(
        name="A",
        bonds=d_ij,
        angular=Y,
        chemical_embedding=z,
        n_rad_max=n_rad_max,
        n_rad_basis=n_rad_basis,
        rcut=rcut,
        indicator=z,
    )
    A.build(float64)
    inpt_dict = A(inpt_dict)

    AA = ProductFunction(left=A, right=A, name="AA", lmax=lmax, Lmax=lmax)
    AA.build(float64)
    inpt_dict = AA(inpt_dict)

    # Use FunctionReduceN as indicator (merges histories -> single L=0 component)
    indicator = FunctionReduceN(
        instructions=[A, AA],
        name="E",
        ls_max=[lmax, lmax],
        n_out=n_rad_max,
        is_central_atom_type_dependent=False,
        allowed_l_p=Parity.REAL_PARITY,
    )
    indicator.build(float64)
    inpt_dict = indicator(inpt_dict)

    # Second layer: equivariant SPBF with chemical embedding
    B = SPBF(
        name="B",
        bonds=d_ij,
        angular=Y,
        chemical_embedding=z,
        n_rad_max=n_rad_max,
        n_rad_basis=n_rad_basis,
        rcut=rcut,
        indicator=indicator,
        lmax=lmax,
        Lmax=lmax,
        keep_parity=Parity.REAL_PARITY,
        normalize_cg=True,
    )
    B.build(float64)
    inpt_dict = B(inpt_dict)

    # Check shape
    out = inpt_dict["B"]
    n_atoms = 6
    n_lm = len(B.coupling_meta_data)
    assert out.shape == (n_atoms, n_rad_max, n_lm)

    # Reduce to invariants and check rotational invariance
    BB = ProductFunction(left=B, right=B, name="BB", lmax=lmax, Lmax=0)
    BB.build(float64)
    inpt_dict = BB(inpt_dict)

    out_inv = inpt_dict["BB"].numpy()
    assert np.allclose(out_inv[:3], out_inv[3:], atol=1e-10), (
        f"Rotational invariance failed: max diff = "
        f"{np.max(np.abs(out_inv[:3] - out_inv[3:]))}"
    )


# ---------------------------------------------------------------------------
# Test: SPBF MLP hidden layers have trainable biases
# ---------------------------------------------------------------------------
def test_spbf_mlp_has_biases():
    lmax = 2
    n_rad_max = 3
    n_rad_basis = 6
    rcut = 5.0

    spbf = SPBF(
        name="A",
        bonds="bonds",
        angular=SphericalHarmonic(vhat="rhat", lmax=lmax, name="Y"),
        chemical_embedding=ScalarChemicalEmbedding(
            element_map={"H": 0, "C": 1}, embedding_size=8, name="Z",
        ),
        n_rad_max=n_rad_max,
        n_rad_basis=n_rad_basis,
        rcut=rcut,
        hidden_layers=[32, 32],
    )
    spbf.build(float64)

    # Hidden layers (all except last) should have bias
    for layer in spbf.mlp_layers[:-1]:
        assert layer.use_bias, f"Hidden layer {layer.name} should have bias"
        assert hasattr(layer, "b"), f"Hidden layer {layer.name} missing bias variable"
        assert layer.b.trainable

    # Output layer should NOT have bias
    assert not spbf.mlp_layers[-1].use_bias


# ---------------------------------------------------------------------------
# Test: SPBF rcut is stored as non-trainable tf.Variable
# ---------------------------------------------------------------------------
def test_spbf_rcut_variable():
    rcut = 5.5
    spbf = SPBF(
        name="A",
        bonds="bonds",
        angular=SphericalHarmonic(vhat="rhat", lmax=2, name="Y"),
        chemical_embedding=ScalarChemicalEmbedding(
            element_map={"H": 0, "C": 1}, embedding_size=8, name="Z",
        ),
        n_rad_max=3,
        n_rad_basis=6,
        rcut=rcut,
    )
    spbf.build(float64)

    assert isinstance(spbf.rc, tf.Variable)
    assert not spbf.rc.trainable
    assert np.isclose(spbf.rc.numpy(), rcut)


# ---------------------------------------------------------------------------
# Test: SPBF with custom hidden layers and activations
# ---------------------------------------------------------------------------
def test_spbf_custom_mlp_config():
    lmax = 2
    n_rad_max = 3
    n_rad_basis = 8
    rcut = 5.0

    np.random.seed(322)
    tf.random.set_seed(322)

    inpt_dict = _make_input_dict()
    inpt_dict, d_ij, Y, z = _build_base_instructions(inpt_dict, lmax, rcut)

    spbf = SPBF(
        name="A",
        bonds=d_ij,
        angular=Y,
        chemical_embedding=z,
        n_rad_max=n_rad_max,
        n_rad_basis=n_rad_basis,
        rcut=rcut,
        hidden_layers=[16, 32, 16],
        activation=["silu", "tanh", "silu"],
    )
    spbf.build(float64)
    inpt_dict = spbf(inpt_dict)

    out = inpt_dict["A"]
    n_atoms = 6
    n_lm = (lmax + 1) ** 2
    assert out.shape == (n_atoms, n_rad_max, n_lm)

    # 3 hidden layers + 1 output = 4 total
    assert len(spbf.mlp_layers) == 4
    # First 3 have bias, last does not
    for layer in spbf.mlp_layers[:3]:
        assert layer.use_bias
    assert not spbf.mlp_layers[3].use_bias


# ---------------------------------------------------------------------------
# Test: SPBF with per-species neighbor averaging
# ---------------------------------------------------------------------------
def test_spbf_per_species_avg_n_neigh():
    lmax = 2
    n_rad_max = 3
    n_rad_basis = 6
    rcut = 6.0

    np.random.seed(322)
    tf.random.set_seed(322)

    inpt_dict = _make_input_dict()
    inpt_dict, d_ij, Y, z = _build_base_instructions(inpt_dict, lmax, rcut)

    spbf = SPBF(
        name="A",
        bonds=d_ij,
        angular=Y,
        chemical_embedding=z,
        n_rad_max=n_rad_max,
        n_rad_basis=n_rad_basis,
        rcut=rcut,
        avg_n_neigh={0: 2.0, 1: 3.0},
    )
    spbf.build(float64)
    inpt_dict = spbf(inpt_dict)

    out = inpt_dict["A"]
    assert out.shape == (6, n_rad_max, (lmax + 1) ** 2)


# ---------------------------------------------------------------------------
# Test: SPBF with lmax smaller than angular lmax (slicing)
# ---------------------------------------------------------------------------
def test_spbf_angular_slicing():
    angular_lmax = 5
    spbf_lmax = 3
    n_rad_max = 4
    n_rad_basis = 8
    rcut = 6.0

    np.random.seed(322)
    tf.random.set_seed(322)

    inpt_dict = _make_input_dict()
    inpt_dict, d_ij, Y, z = _build_base_instructions(inpt_dict, angular_lmax, rcut)

    spbf = SPBF(
        name="A",
        bonds=d_ij,
        angular=Y,
        chemical_embedding=z,
        n_rad_max=n_rad_max,
        n_rad_basis=n_rad_basis,
        rcut=rcut,
        lmax=spbf_lmax,
    )
    spbf.build(float64)
    inpt_dict = spbf(inpt_dict)

    out = inpt_dict["A"]
    n_lm = (spbf_lmax + 1) ** 2
    assert out.shape == (6, n_rad_max, n_lm)
    assert spbf.lmax == spbf_lmax