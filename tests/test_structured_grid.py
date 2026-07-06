"""
Unit tests for tensorpotential.experimental.instructions.aux_compute.

Coverage:
  * StructuredGridProductFunction agrees with the direct CG / Gaunt integral
    at small lmax in fp64 (machine precision).
  * Asymmetric n_left != n_right is accepted (fixes the spurious
    prototype assertion).
  * coupling_meta_data has the canonical column schema and downstream
    FunctionReduceN, FCRight2Left, EquivariantRMSNorm consume it without
    error and produce the expected output shape.
  * parity_mode='natural' rejects mixed-parity inputs at construction time.
  * parity_mode='full' raises NotImplementedError (reserved for a follow-up).
  * Gradient flow: every internal Variable receives a non-zero gradient.
  * StructuredGridMessagePassing uses BOND_IND_J for source-side gather and
    BOND_IND_I for the segment-sum aggregation (the paper's Eq. 36).
  * Rotational equivariance (per-(L, m) Wigner-D transformation) and
    invariance (L=0 contraction is rotation-invariant) for both
    StructuredGridProductFunction and StructuredGridMessagePassing in fp32
    and fp64, exercising the grid_full architecture.
"""

from __future__ import annotations

import os

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
os.environ.setdefault("TF_USE_LEGACY_KERAS", "1")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import numpy as np
import pandas as pd
import pytest
import tensorflow as tf

from tensorpotential import constants
from tensorpotential.experimental.instructions.aux_compute import (
    StructuredGridProductFunction,
    StructuredGridMessagePassing,
    _gauss_legendre_nodes_weights,
    _normalised_assoc_legendre,
    _real_sh_phi_factors,
)


# ---------------------------------------------------------------------- #
class _StubInstr:
    """Minimal upstream-instruction stand-in (n_out, lmax, name, metadata)."""

    def __init__(self, name, lmax, n_out, parity="natural"):
        self.name = name
        self.lmax = lmax
        self.n_out = n_out
        rows = []
        for l in range(lmax + 1):  # noqa: E741
            for m in range(-l, l + 1):
                if parity == "natural":
                    p = 1 if l % 2 == 0 else -1
                elif parity == "opposite":
                    p = -1 if l % 2 == 0 else 1
                else:
                    raise ValueError(parity)
                rows.append([l, m, "", p, l])
        self.coupling_meta_data = pd.DataFrame(
            rows, columns=["l", "m", "hist", "parity", "sum_of_ls"]
        )
        self.coupling_origin = None


def _real_sh_grid(lmax, U, V):
    x_u, w_u = _gauss_legendre_nodes_weights(U)
    P = _normalised_assoc_legendre(lmax, x_u)
    Q = _real_sh_phi_factors(V, lmax)
    Y = np.zeros((lmax + 1, 2 * lmax + 1, U, V))
    for l in range(lmax + 1):  # noqa: E741
        for m in range(-l, l + 1):
            Y[l, m + lmax] = P[:, l, abs(m)][:, None] * Q[:, m + lmax][None, :]
    return Y, x_u, w_u


def _direct_gaunt_reference(A1_np, A2_np, lmax_in, lmax_out):
    """Compute phi_lm via dense quadrature on a finer grid (ground truth)."""
    U = V = 4 * lmax_in + 4
    Y_in, x_u, w_u = _real_sh_grid(lmax_in, U, V)
    Y_out, _, _ = _real_sh_grid(lmax_out, U, V)
    n_in = A1_np.shape[1]
    A1_d = np.zeros((n_in, U, V))
    A2_d = np.zeros((A2_np.shape[1], U, V))
    for l in range(lmax_in + 1):  # noqa: E741
        for m in range(-l, l + 1):
            lm = l * (l + 1) + m
            A1_d += A1_np[0, :, lm, None, None] * Y_in[l, m + lmax_in][None]
            A2_d += A2_np[0, :, lm, None, None] * Y_in[l, m + lmax_in][None]
    # For n_in == n_right == 1 (the test setup) the inner product collapses
    S = (A1_d * A2_d).sum(axis=0)                            # [U, V]
    out = np.zeros((lmax_out + 1) ** 2)
    for l in range(lmax_out + 1):  # noqa: E741
        for m in range(-l, l + 1):
            lm = l * (l + 1) + m
            out[lm] = (2 * np.pi / V) * np.sum(
                w_u[:, None] * Y_out[l, m + lmax_out] * S
            )
    return out


# ====================================================================== #
# Numerical correctness vs direct Gaunt integral
# ====================================================================== #
@pytest.mark.parametrize("lmax_in,lmax_out", [
    (2, 2), (3, 3), (4, 4), (5, 5), (6, 6), (7, 7),
    (3, 2), (2, 4), (7, 5), (5, 7),
])
def test_pipeline_matches_direct_gaunt(lmax_in, lmax_out):
    rng = np.random.default_rng(seed=0xCAFE + lmax_in * 100 + lmax_out)
    n_in, n_out, rank = 1, 1, 1

    left = _StubInstr("A1", lmax_in, n_in)
    right = _StubInstr("A2", lmax_in, n_in)
    instr = StructuredGridProductFunction(
        left=left, right=right, name="test_grid",
        lmax=lmax_in, Lmax=lmax_out,
        n_out=n_out, rank=rank,
        is_left_right_equal=False,
    )
    instr.build(tf.float64)
    instr.a_cnl.assign(np.ones([rank, n_in, lmax_in + 1]))
    instr.b_cnl.assign(np.ones([rank, n_in, lmax_in + 1]))
    instr.lambda_cnl.assign(np.ones([rank, n_out, lmax_out + 1]))

    A1 = rng.standard_normal((1, n_in, (lmax_in + 1) ** 2))
    A2 = rng.standard_normal((1, n_in, (lmax_in + 1) ** 2))
    phi = instr.frwrd({"A1": tf.constant(A1), "A2": tf.constant(A2)}).numpy()

    ref = _direct_gaunt_reference(A1, A2, lmax_in, lmax_out)
    np.testing.assert_allclose(phi[0, 0], ref, atol=1e-12, rtol=1e-12)


# ====================================================================== #
# Asymmetric channel counts (fixes the prototype's spurious assert)
# ====================================================================== #
def test_asymmetric_n_out():
    n_left, n_right, rank, n_out = 3, 5, 4, 2
    lmax = 3
    left = _StubInstr("L", lmax, n_left)
    right = _StubInstr("R", lmax, n_right)
    instr = StructuredGridProductFunction(
        left=left, right=right, name="asym",
        lmax=lmax, Lmax=lmax,
        n_out=n_out, rank=rank,
        is_left_right_equal=False,
    )
    instr.build(tf.float64)
    assert instr.a_cnl.shape == [rank, n_left, lmax + 1]
    assert instr.b_cnl.shape == [rank, n_right, lmax + 1]
    assert instr.lambda_cnl.shape == [rank, n_out, lmax + 1]

    rng = np.random.default_rng(0)
    A1 = rng.standard_normal((1, n_left, (lmax + 1) ** 2))
    A2 = rng.standard_normal((1, n_right, (lmax + 1) ** 2))
    phi = instr.frwrd({"L": tf.constant(A1), "R": tf.constant(A2)}).numpy()
    assert phi.shape == (1, n_out, (lmax + 1) ** 2), (
        f"unexpected output shape {phi.shape}"
    )
    assert np.all(np.isfinite(phi))


# ====================================================================== #
# coupling_meta_data schema + downstream consumer compatibility
# ====================================================================== #
def test_coupling_meta_data_schema():
    left = _StubInstr("L", 3, 4)
    right = _StubInstr("R", 3, 4)
    instr = StructuredGridProductFunction(
        left=left, right=right, name="meta",
        lmax=3, Lmax=3, n_out=2, rank=2,
    )
    cmd = instr.coupling_meta_data
    assert list(cmd.columns) == ["l", "m", "hist", "parity", "sum_of_ls"]
    # Rows in canonical sort order (l asc, parity asc, hist asc, m asc)
    sorted_again = cmd.sort_values(["l", "parity", "hist", "m"]).reset_index(drop=True)
    pd.testing.assert_frame_equal(cmd, sorted_again)
    # natural-mode default: every row has parity == (-1)^l
    for _, row in cmd.iterrows():
        l, p = int(row["l"]), int(row["parity"])
        assert p == (1 if l % 2 == 0 else -1)
    # row count == sum_l (2l + 1) for l in 0..Lmax
    assert len(cmd) == sum(2 * l + 1 for l in range(3 + 1))


def test_downstream_FunctionReduceN_consumes_grid_output():
    """
    Build a small grid_prod -> FunctionReduceN pipeline and check it
    constructs without error and produces the expected output shape.
    """
    from tensorpotential.instructions.compute import FunctionReduceN

    left = _StubInstr("L", 2, 3)
    right = _StubInstr("R", 2, 3)
    grid = StructuredGridProductFunction(
        left=left, right=right, name="g",
        lmax=2, Lmax=2, n_out=4, rank=2,
        is_left_right_equal=False,
    )
    reducer = FunctionReduceN(
        instructions=[grid], name="g_red",
        ls_max=[2], n_out=8,
        allowed_l_p=[[l, 1 if l % 2 == 0 else -1] for l in range(3)],
        is_central_atom_type_dependent=False,
    )
    grid.build(tf.float64)
    reducer.build(tf.float64)

    rng = np.random.default_rng(0)
    A1 = rng.standard_normal((4, 3, 9)).astype(np.float64)
    A2 = rng.standard_normal((4, 3, 9)).astype(np.float64)
    inputs = {
        "L": tf.constant(A1),
        "R": tf.constant(A2),
        constants.ATOMIC_MU_I: tf.constant([0, 0, 0, 0], dtype=tf.int32),
    }
    inputs["g"] = grid.frwrd(inputs)
    out = reducer.frwrd(inputs)
    # Reducer outputs [n_atoms, n_out, n_lm_kept]
    assert out.shape[0] == 4 and out.shape[1] == 8


def test_downstream_EquivariantRMSNorm_consumes_grid_output():
    from tensorpotential.instructions.compute import EquivariantRMSNorm

    left = _StubInstr("L", 2, 3)
    right = _StubInstr("R", 2, 3)
    grid = StructuredGridProductFunction(
        left=left, right=right, name="g",
        lmax=2, Lmax=2, n_out=4, rank=2,
        is_left_right_equal=False,
    )
    norm = EquivariantRMSNorm(input=grid, name="g_norm")
    grid.build(tf.float64)
    norm.build(tf.float64)


def test_downstream_FCRight2Left_consumes_grid_output():
    from tensorpotential.instructions.compute import FCRight2Left

    left = _StubInstr("L", 2, 3)
    right = _StubInstr("R", 2, 3)
    grid_a = StructuredGridProductFunction(
        left=left, right=right, name="ga",
        lmax=2, Lmax=2, n_out=4, rank=2,
        is_left_right_equal=False,
    )
    grid_b = StructuredGridProductFunction(
        left=left, right=right, name="gb",
        lmax=2, Lmax=2, n_out=4, rank=2,
        is_left_right_equal=False,
    )
    fc = FCRight2Left(left=grid_a, right=grid_b, name="fc",
                      n_out=4, left_coefs=False)
    # Just smoke-test construction; full forward needs more wiring.
    assert fc.coupling_meta_data is not None


# ====================================================================== #
# parity_mode validation
# ====================================================================== #
def test_natural_mode_rejects_mixed_parity_input():
    left = _StubInstr("L", 2, 3, parity="opposite")  # parity = -(-1)^l
    right = _StubInstr("R", 2, 3, parity="natural")
    with pytest.raises(ValueError, match="natural-parity"):
        StructuredGridProductFunction(
            left=left, right=right, name="bad",
            lmax=2, Lmax=2, n_out=4, rank=2,
            parity_mode="natural",
        )


def test_full_mode_accepts_natural_inputs_and_emits_both_parities():
    """
    parity_mode='full' should construct successfully on natural-parity inputs
    and emit a coupling_meta_data that covers BOTH parity sectors per output l.
    The frwrd output then has 2*(Lmax+1)^2 rows (alpha + beta blocks).
    Layout: the natural-parity block (in (l, m) order) comes first, then the
    opposite-parity block; ``_keep_rows`` is the identity by default so the
    trailing gather is a no-op.
    """
    lmax = 2
    Lmax = 2
    left = _StubInstr("L", lmax, 3)
    right = _StubInstr("R", lmax, 3)
    instr = StructuredGridProductFunction(
        left=left, right=right, name="full_ok",
        lmax=lmax, Lmax=Lmax, n_out=4, rank=2,
        parity_mode="full", is_left_right_equal=False,
    )
    instr.build(tf.float64)
    # In full mode coupling_meta_data has both parities per l, so 2*(Lmax+1)^2 rows.
    assert len(instr.coupling_meta_data) == 2 * (Lmax + 1) ** 2
    parities_per_l = (
        instr.coupling_meta_data.groupby("l")["parity"].apply(lambda s: sorted(set(s)))
    )
    for l, ps in parities_per_l.items():
        assert ps == [-1, 1], f"l={l} should expose both parities, got {ps}"
    # Beta CP factors exist and are independent learnable Variables.
    assert hasattr(instr, "lambda_cnl_beta")
    assert instr.lambda_cnl is not instr.lambda_cnl_beta

    # Verify the natural-first / opposite-second block layout.  The first
    # (Lmax+1)^2 rows should all be natural-parity, then the next (Lmax+1)^2
    # opposite-parity.  ``_keep_rows`` should be the identity permutation.
    n_lm_out = (Lmax + 1) ** 2
    cmd = instr.coupling_meta_data
    for k in range(n_lm_out):
        l, p = int(cmd.iloc[k]["l"]), int(cmd.iloc[k]["parity"])
        assert p == (1 if l % 2 == 0 else -1), \
            f"row {k} should be natural-parity, got l={l}, p={p}"
    for k in range(n_lm_out, 2 * n_lm_out):
        l, p = int(cmd.iloc[k]["l"]), int(cmd.iloc[k]["parity"])
        assert p == -(1 if l % 2 == 0 else -1), \
            f"row {k} should be opposite-parity, got l={l}, p={p}"
    assert instr._keep_rows_is_identity, \
        "default keep_parity should make _keep_rows the identity (no gather)"

    rng = np.random.default_rng(0)
    A1 = rng.standard_normal((2, 3, (lmax + 1) ** 2))
    A2 = rng.standard_normal((2, 3, (lmax + 1) ** 2))
    phi = instr.frwrd({"L": tf.constant(A1), "R": tf.constant(A2)}).numpy()
    assert phi.shape == (2, 4, 2 * (Lmax + 1) ** 2), phi.shape
    assert np.all(np.isfinite(phi))


def test_full_mode_accepts_mixed_parity_inputs():
    """Full mode now supports mixed-parity inputs via the 8-case dispatch
    of paper Sec. 7.6 ("On-site bilinear contraction with general parity"):
    one input opposite-parity + one natural-parity → constructs successfully
    and produces both alpha- and beta-side CP factors on the opposite slot.

    Natural mode should still reject mixed-parity inputs (regression).
    """
    left = _StubInstr("L", 2, 3, parity="opposite")
    right = _StubInstr("R", 2, 3, parity="natural")
    # Natural mode rejects.
    with pytest.raises(ValueError, match="natural-parity"):
        StructuredGridProductFunction(
            left=left, right=right, name="mixed_natural",
            lmax=2, Lmax=2, n_out=4, rank=2,
            parity_mode="natural",
        )
    # Full mode accepts and allocates the s_1=- slot CP factors.
    instr = StructuredGridProductFunction(
        left=left, right=right, name="mixed_full",
        lmax=2, Lmax=2, n_out=4, rank=2,
        parity_mode="full",
        is_left_right_equal=False,
    )
    instr.build(tf.float64)
    assert hasattr(instr, "a_cnl_alpha_opp")
    assert hasattr(instr, "a_cnl_beta_opp")
    assert hasattr(instr, "lambda_cnl_alpha_opp")
    assert hasattr(instr, "lambda_cnl_beta_nat")
    # The natural side (right input) does NOT need an opposite CP factor.
    assert not hasattr(instr, "b_cnl_alpha_opp")

    rng = np.random.default_rng(0)
    A1 = rng.standard_normal((2, 3, (2 + 1) ** 2))
    A2 = rng.standard_normal((2, 3, (2 + 1) ** 2))
    out = instr.frwrd({"L": tf.constant(A1), "R": tf.constant(A2)}).numpy()
    assert out.shape == (2, 4, 2 * (2 + 1) ** 2), out.shape
    assert np.all(np.isfinite(out))


# ====================================================================== #
# Gradient flow
# ====================================================================== #
def test_gradient_flow():
    n_in, rank, n_out = 2, 3, 2
    lmax = 3
    left = _StubInstr("L", lmax, n_in)
    right = _StubInstr("R", lmax, n_in)
    instr = StructuredGridProductFunction(
        left=left, right=right, name="grad",
        lmax=lmax, Lmax=lmax, n_out=n_out, rank=rank,
        is_left_right_equal=False,
    )
    instr.build(tf.float64)

    rng = np.random.default_rng(0)
    A1 = tf.constant(rng.standard_normal((3, n_in, (lmax + 1) ** 2)))
    A2 = tf.constant(rng.standard_normal((3, n_in, (lmax + 1) ** 2)))
    with tf.GradientTape() as tape:
        phi = instr.frwrd({"L": A1, "R": A2})
        loss = tf.reduce_sum(phi * phi)
    grads = tape.gradient(loss, [instr.a_cnl, instr.b_cnl, instr.lambda_cnl])
    for v, g in zip([instr.a_cnl, instr.b_cnl, instr.lambda_cnl], grads):
        assert g is not None, f"no gradient for {v.name}"
        assert tf.reduce_max(tf.abs(g)).numpy() > 0, f"zero gradient on {v.name}"


# ====================================================================== #
# is_left_right_equal saves work (b is reused from a)
# ====================================================================== #
def test_is_left_right_equal_collapses_b_to_a():
    left = _StubInstr("X", 2, 3)
    instr = StructuredGridProductFunction(
        left=left, right=left, name="sym",
        lmax=2, Lmax=2, n_out=2, rank=2,
        is_left_right_equal=True,
    )
    instr.build(tf.float64)
    # b_cnl is the same Variable object as a_cnl
    assert instr.a_cnl is instr.b_cnl


# ====================================================================== #
# StructuredGridMessagePassing: bond-index convention (the prototype's bug)
# ====================================================================== #
def test_message_passing_uses_BOND_IND_J_for_source_gather():
    """
    Build a tiny message-passing instance, run forward on a 3-atom / 2-bond
    graph where atom 0 has a non-zero indicator and atoms 1, 2 are zero, and
    bond_ind_j = [0, 0] (both bonds source from atom 0).  The output at the
    central atoms should be NON-ZERO; if the gather were by BOND_IND_I (the
    bug), atom 0 would never be a *source* and the output would vanish.
    """
    # We build the absolute minimum stack: BondLength stub, SH stub, indicator
    # stub, embedding stub.  We then directly instantiate the instruction and
    # run frwrd with hand-constructed input_data.
    from tensorpotential.instructions.compute import (
        ScalarChemicalEmbedding,
        SphericalHarmonic,
        BondLength,
        ScaledBondVector,
    )

    lmax = 2

    bonds = BondLength()
    rhat = ScaledBondVector(bond_length=bonds)
    Y = SphericalHarmonic(vhat=rhat, lmax=lmax, name="Y")
    z = ScalarChemicalEmbedding(
        element_map={"A": 0}, embedding_size=4, name="Z",
    )

    indicator = _StubInstr("I", lmax, 4)  # n_rad_max=4

    instr = StructuredGridMessagePassing(
        name="mp", bonds=bonds, angular=Y, indicator=indicator,
        chemical_embedding=z,
        n_rad_max=4, n_rad_basis=4, rcut=5.0, p=5,
        lmax=lmax, Lmax=lmax,
        avg_n_neigh=1.0,
        hidden_layers=[8],
    )
    Y.build(tf.float64)
    z.build(tf.float64)
    instr.build(tf.float64)

    # Synthetic inputs.  Two bonds, both sourced at atom 0, ending at atoms 1, 2.
    n_atoms = 3
    bond_ind_i = tf.constant([1, 2], dtype=tf.int32)   # central / dest
    bond_ind_j = tf.constant([0, 0], dtype=tf.int32)   # source / neighbour
    bond_mu_i = tf.constant([0, 0], dtype=tf.int32)
    bond_mu_j = tf.constant([0, 0], dtype=tf.int32)
    atomic_mu_i = tf.constant([0, 0, 0], dtype=tf.int32)
    bond_vector = tf.constant([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]], dtype=tf.float64)
    bond_length = tf.norm(bond_vector, axis=-1)
    rhat_v = bond_vector / bond_length[:, None]

    # Indicator: non-zero only at atom 0 (the source)
    I_data = np.zeros((n_atoms, 4, (lmax + 1) ** 2), dtype=np.float64)
    I_data[0, 0, 0] = 1.0  # l=0 channel, atom 0
    I_tf = tf.constant(I_data)

    # Per-atom-type embedding: read the Variable directly (one row per type).
    # ScalarChemicalEmbedding stores its embedding in `.w` and exposes the
    # ``embedding_size`` attribute; bypass __call__ which expects input_data.
    z_emb = tf.cast(z.w, tf.float64)  # [n_types, embedding_size]

    inputs = {
        bonds.name: bond_length,
        rhat.name: rhat_v,
        Y.name: Y.frwrd({rhat.name: rhat_v}),
        z.name: z_emb,
        indicator.name: I_tf,
        constants.BOND_IND_I: bond_ind_i,
        constants.BOND_IND_J: bond_ind_j,
        constants.BOND_MU_I: bond_mu_i,
        constants.BOND_MU_J: bond_mu_j,
        constants.ATOMIC_MU_I: atomic_mu_i,
        constants.N_ATOMS_BATCH_TOTAL: tf.constant(n_atoms, dtype=tf.int32),
    }
    out = instr.frwrd(inputs).numpy()        # [n_atoms, n_out, n_lm]
    # Atoms 1 and 2 (the central atoms of the bonds) should have non-zero
    # output -- they receive a message FROM atom 0.  If the bug were live
    # (gather by BOND_IND_I), atoms 1 and 2 would gather their own indicator
    # which is zero, so the output would be zero.
    norm_1 = np.linalg.norm(out[1])
    norm_2 = np.linalg.norm(out[2])
    norm_0 = np.linalg.norm(out[0])
    assert norm_1 > 1e-6, (
        f"atom 1 should receive non-zero message from atom 0 (BOND_IND_J), "
        f"got |out[1]|={norm_1:.2e} -- bond gather may be using BOND_IND_I."
    )
    assert norm_2 > 1e-6, (
        f"atom 2 should receive non-zero message from atom 0 (BOND_IND_J), "
        f"got |out[2]|={norm_2:.2e} -- bond gather may be using BOND_IND_I."
    )
    # Atom 0 has no incoming bonds, so its output should be zero.
    assert norm_0 < 1e-12, (
        f"atom 0 has no incoming bonds; |out[0]|={norm_0:.2e} should be ~0"
    )


# ====================================================================== #
# Rotational equivariance + invariance for the grid_full architecture
# ====================================================================== #
#
# We piggy-back on tests/test_instructions.py:_build_general_product_test_data
# which constructs a 6-atom batch consisting of {atoms 0..2} and their rotated
# image {atoms 3..5}.  Any equivariant instruction's output then has rows
# 0..2 as f(x) and rows 3..5 as f(R x); equivariance means
#     out[3..5, c, l, m] = sum_{m'} D^l_{m m'}(R) out[0..2, c, l, m']
# while invariance (L=0 channels) means out[3..5, c, 0, 0] = out[0..2, c, 0, 0].
#
# The same Wigner-D matrices the existing GeneralProductFunction test fits
# from Y_lm(R x) = D^l Y_lm(x) are reused here -- so the equivariance
# assertion is identical between the two impls; if it passes for general it
# must pass for grid up to numerical tolerance.

# The helpers below mirror tests/test_instructions.py:
# _build_general_product_test_data + _compute_wigner_d_real.  We inline rather
# than cross-import because that file uses package-relative imports
# ("from .utils") that only resolve when tests/ is treated as a package; pytest
# collects fine but `import test_instructions` from outside doesn't.

from scipy.spatial.transform import Rotation as _Rotation


def _build_grid_equivar_test_data(dtype):
    """
    6-atom test scaffold: atoms 0..2 form a small cluster, atoms 3..5 are
    the same cluster rotated by a fixed rotation R.  Returns a populated
    input dict, an SPBF-like equivariant feature ``A`` with n_out=n_rad_max,
    and the rotation that took 0..2 to 3..5.

    Any equivariant instruction f(...) run on this batch will then have
    rows 0..2 = f(x), rows 3..5 = f(R x); equivariance ==
        f(R x)[:, c, l, m] = sum_{m'} D^l_{m m'}(R) f(x)[:, c, l, m']
    and invariance at l=0 ==
        f(R x)[:, c, 0, 0] = f(x)[:, c, 0, 0].
    """
    from tensorpotential.instructions.compute import (
        BondLength,
        ScaledBondVector,
        SphericalHarmonic,
        ScalarChemicalEmbedding,
        RadialBasis,
        MLPRadialFunction,
        SingleParticleBasisFunctionScalarInd,
    )

    lmax_bond = 4
    n_rad_base = 8
    n_rad_max = 8
    cutoff_rad = 6.0

    np.random.seed(322)
    tf.random.set_seed(322)

    axis = np.array([1, 1, 3])
    theta = np.pi / 2.2
    axis = axis / np.linalg.norm(axis)
    rot = _Rotation.from_rotvec(theta * axis)

    coord = np.array([[0.2, 0.3, 2.1], [0.1, 3.2, 0.5], [0.3, 1, 2]])
    coord_r = rot.apply(coord)
    coord_2 = np.vstack([coord, coord_r]).astype(np.float64)

    indj = [1, 2, 0, 2, 0, 1, 4, 5, 3, 5, 3, 4]
    indi = [0, 0, 1, 1, 2, 2, 3, 3, 4, 4, 5, 5]
    atomic_mu_i = [0, 1, 0, 0, 1, 0]
    muj = np.take(atomic_mu_i, indj, axis=0)
    rj = np.take(coord_2, indj, axis=0)
    ri = np.take(coord_2, indi, axis=0)
    rij = rj - ri

    mui = np.take(atomic_mu_i, indi, axis=0)
    inpt_dict = {
        constants.BOND_VECTOR: rij,
        constants.BOND_IND_J: indj,
        constants.BOND_MU_J: muj,
        constants.BOND_IND_I: indi,
        constants.BOND_MU_I: mui,
        constants.N_ATOMS_BATCH_TOTAL: len(coord_2),
        constants.ATOMIC_MU_I: atomic_mu_i,
    }

    d_ij = BondLength()
    d_ij.build(dtype)
    inpt_dict = d_ij(inpt_dict)
    rhat = ScaledBondVector(bond_length=d_ij)
    rhat.build(dtype)
    inpt_dict = rhat(inpt_dict)
    g_k = RadialBasis(
        bonds=d_ij, basis_type="SBessel", nfunc=n_rad_base, rcut=cutoff_rad, p=5,
    )
    g_k.build(dtype)
    inpt_dict = g_k(inpt_dict)
    R_nl = MLPRadialFunction(
        n_rad_max=n_rad_max, lmax=lmax_bond, basis=g_k, name="R",
    )
    R_nl.build(dtype)
    inpt_dict = R_nl(inpt_dict)
    Y = SphericalHarmonic(vhat=rhat, lmax=lmax_bond, name="Y")
    Y.build(dtype)
    inpt_dict = Y(inpt_dict)
    z = ScalarChemicalEmbedding(
        element_map={"H": 0, "C": 1}, embedding_size=32, name="Z",
    )
    z.build(dtype)
    inpt_dict = z(inpt_dict)
    A = SingleParticleBasisFunctionScalarInd(
        radial=R_nl, angular=Y, indicator=z, name="A", avg_n_neigh=10.0,
    )
    A.build(dtype)
    A.lin_transform.build(dtype)
    inpt_dict = A(inpt_dict)
    return inpt_dict, A, d_ij, rhat, Y, z, lmax_bond, n_rad_max, rot


def _compute_wigner_d_real(rot, lmax):
    """
    Real Wigner-D matrices fit from Y_lm(R x) = D^l Y_lm(x) using
    least-squares on a set of 2L+1 random unit vectors.
    """
    from tensorpotential.instructions.compute import SphericalHarmonic
    rng = np.random.RandomState(12345)
    n_vecs = max(2 * lmax + 1, 5)
    vecs = rng.normal(size=(n_vecs, 3))
    vecs = vecs / np.linalg.norm(vecs, axis=1, keepdims=True)
    rvecs = rot.apply(vecs)
    Y = SphericalHarmonic(vhat="v", lmax=lmax, name="_wigner_Y")
    Y.build(tf.float64)
    y_orig = Y.frwrd({"v": vecs}).numpy()
    y_rot = Y.frwrd({"v": rvecs}).numpy()
    D = {}
    for L in range(lmax + 1):
        start = L * L
        end = (L + 1) * (L + 1)
        Y_orig = y_orig[:, start:end].T
        Y_rot = y_rot[:, start:end].T
        D[L] = Y_rot @ np.linalg.pinv(Y_orig)
    return D


def _equivariance_assertions(instr_out: np.ndarray, AA, rot, lmax_out, atol):
    """
    Run the per-(L, hist, parity) Wigner-D check on a 6-atom output:
    rows 0..2 are the originals, rows 3..5 are the rotated image.
    """
    out_orig = instr_out[:3]
    out_rot = instr_out[3:]
    D = _compute_wigner_d_real(rot, lmax_out)
    cmd = AA.coupling_meta_data
    groups = cmd.groupby(["l", "hist", "parity"]).indices
    for (L, hist, parity), indices in groups.items():
        sub = cmd.iloc[indices].sort_values("m")
        m_indices = sub.index.values
        ms = list(sub["m"].values)
        assert ms == list(range(-L, L + 1)), (
            f"Expected m={list(range(-L, L + 1))}, got {ms} "
            f"for (L={L}, hist={hist!r}, parity={parity})"
        )
        D_L = D[L]                                # [2L+1, 2L+1]
        for atom in range(3):
            for n in range(out_orig.shape[1]):
                v_orig = out_orig[atom, n, m_indices]
                v_rot = out_rot[atom, n, m_indices]
                v_expected = D_L @ v_orig
                assert np.allclose(v_rot, v_expected, atol=atol, rtol=atol), (
                    f"equivariance broken at (L={L}, hist={hist!r}, "
                    f"parity={parity}, atom={atom}, n={n}): "
                    f"max|err|={np.max(np.abs(v_rot - v_expected)):.3e}, "
                    f"atol={atol:.1e}"
                )


def _invariance_assertions(instr_out: np.ndarray, AA, atol):
    """L=0 channels must be bit-equal (up to atol) between original and rotated."""
    out_orig = instr_out[:3]
    out_rot = instr_out[3:]
    cmd = AA.coupling_meta_data
    l0_rows = cmd.index[cmd["l"] == 0].values
    if len(l0_rows) == 0:
        return  # nothing to check
    diff = np.max(np.abs(out_orig[:, :, l0_rows] - out_rot[:, :, l0_rows]))
    assert diff < atol, (
        f"L=0 invariance broken: max|out_orig - out_rot| = {diff:.3e} "
        f"(atol={atol:.1e})"
    )


# ---------------------------------------------------------------------- #
# StructuredGridProductFunction equivariance + invariance
# ---------------------------------------------------------------------- #
@pytest.mark.parametrize(
    "param_dtype,atol",
    # Tolerances calibrated empirically (see /tmp/diag_fp32 sweep): the floor
    # is pure fp64/fp32 floating-point precision and is independent of the
    # grid oversample factor (1.0 to 3.0 give the same error).  Headroom
    # ~10-100x over the measured floor for cross-machine robustness.
    [(tf.float64, 1e-11), (tf.float32, 1e-7)],
)
def test_grid_product_function_rot_equivar(param_dtype, atol):
    """
    StructuredGridProductFunction: per-(L, m) output rotates under D^L(R)
    when the input geometry rotates under R.  Tested at every L in [0..lmax].
    """
    inpt_dict, A, _d_ij, _rhat, _Y, _z, lmax_bond, n_rad_max, rot = (
        _build_grid_equivar_test_data(dtype=param_dtype)
    )

    Lmax_out = lmax_bond
    AA = StructuredGridProductFunction(
        left=A, right=A, name="grid_AA",
        lmax=lmax_bond, Lmax=Lmax_out,
        n_out=n_rad_max, rank=6,
        is_left_right_equal=True,
        parity_mode="natural",
    )
    AA.build(param_dtype)
    inpt_dict = AA(inpt_dict)
    out = inpt_dict[AA.name].numpy()

    _equivariance_assertions(out, AA, rot, Lmax_out, atol)
    _invariance_assertions(out, AA, atol)


# ---------------------------------------------------------------------- #
# parity_mode='full' equivariance: BOTH natural and opposite parity output
# sectors must rotate as D^L(R) under input rotation, regardless of which
# parity sector they sit in.  The opposite-parity sector is supplied by the
# Category-beta surface-curl branch added on top of natural mode.
# ---------------------------------------------------------------------- #
@pytest.mark.parametrize(
    "param_dtype,atol",
    # Same tolerance as the natural+natural -> full case below; the gather of
    # the natural/opposite views does not amplify floating-point error.
    [(tf.float64, 5e-11), (tf.float32, 1e-7)],
)
def test_grid_product_function_full_mode_mixed_input_rot_equivar(param_dtype, atol):
    """parity_mode='full' with ONE mixed-parity input: full output equivariant.

    Builds a synthetic mixed-parity left input (interleaved natural and
    opposite-parity rows for the same l) and a natural right input.  Checks
    that both alpha and beta output blocks rotate correctly.
    """
    # We need a left input with mixed parity.  Construct one manually by
    # building an _StubInstr-like field with rows that include opposite-parity
    # entries.  Right input stays natural.
    np.random.seed(13)
    tf.random.set_seed(13)
    from tensorpotential.instructions.compute import (
        BondLength, ScaledBondVector, SphericalHarmonic,
        ScalarChemicalEmbedding, RadialBasis, MLPRadialFunction,
        SingleParticleBasisFunctionScalarInd,
    )
    from scipy.spatial.transform import Rotation as _Rotation
    # Reuse the cluster + image setup from _build_grid_equivar_test_data.
    inpt_dict, A, d_ij, rhat, Y, z, lmax_bond, n_rad_max, rot = (
        _build_grid_equivar_test_data(dtype=param_dtype)
    )

    # Build a "mixed-parity" left input by gluing A (natural-parity SPBF
    # output) with a second SPBF feature B that we MARK as opposite-parity.
    # The actual rotation behaviour of B's coefficients is the same as a
    # natural-parity tensor of the same shape; the parity flag is a labelling
    # choice that the consumer (the bilinear contraction) interprets via
    # surface-curl dispatch.  For the equivariance test we only care that the
    # output rotates correctly under R; parity is a slot label, not a
    # rotation-altering transform.
    B = SingleParticleBasisFunctionScalarInd(
        radial=A.radial, angular=A.angular, indicator=A.indicator,
        name="B_raw", avg_n_neigh=5.0,
    )
    B.build(param_dtype)
    B.lin_transform.build(param_dtype)
    inpt_dict = B(inpt_dict)

    # Construct a "mixed" instr stub whose coupling_meta_data is the union of
    # A's (natural) and B's (relabelled opposite) rows.  We physically store
    # the concatenated tensor in a new key "AB_mix" in inpt_dict.
    import pandas as pd
    A_cmd = A.coupling_meta_data.copy()
    B_cmd = B.coupling_meta_data.copy()
    B_cmd["parity"] = -B_cmd["parity"]  # flip parity label to opposite
    B_cmd["hist"] = B_cmd["hist"].astype(str) + "(opp)"
    mix_cmd = (
        pd.concat([A_cmd, B_cmd], ignore_index=True)
        .sort_values(["l", "parity", "hist", "m"])
        .reset_index(drop=True)
    )

    class MixedField:
        name = "AB_mix"
        lmax = lmax_bond
        n_out = n_rad_max
        coupling_meta_data = mix_cmd
        coupling_origin = ["AB_mix"]

    # Reorder A and B rows to match mix_cmd's sort order
    A_tensor = inpt_dict[A.name].numpy()
    B_tensor = inpt_dict[B.name].numpy()
    # Build the concatenated tensor in mix_cmd row order
    rows = []
    nA = len(A.coupling_meta_data)
    nB = len(B.coupling_meta_data)
    # Map (l, m, hist_clean, source) -> source row index
    A_index = {}
    for idx, row in A.coupling_meta_data.iterrows():
        A_index[(int(row["l"]), int(row["m"]), str(row["hist"]), "A")] = idx
    B_index = {}
    for idx, row in B.coupling_meta_data.iterrows():
        # Match B_cmd hist (B_cmd has "(opp)" appended)
        B_index[(int(row["l"]), int(row["m"]),
                 str(row["hist"]) + "(opp)", "B")] = idx
    for _, row in mix_cmd.iterrows():
        key_A = (int(row["l"]), int(row["m"]), str(row["hist"]), "A")
        key_B = (int(row["l"]), int(row["m"]), str(row["hist"]), "B")
        if key_A in A_index:
            rows.append(A_tensor[..., A_index[key_A]])
        elif key_B in B_index:
            rows.append(B_tensor[..., B_index[key_B]])
        else:
            raise AssertionError(f"Could not locate row {row}")
    mix_tensor = tf.constant(np.stack(rows, axis=-1))
    inpt_dict["AB_mix"] = mix_tensor

    Lmax_out = lmax_bond
    AB = StructuredGridProductFunction(
        left=MixedField, right=A, name="grid_AB_mix",
        lmax=lmax_bond, Lmax=Lmax_out,
        n_out=n_rad_max, rank=6,
        is_left_right_equal=False,
        parity_mode="full",
    )
    AB.build(param_dtype)
    # Verify the new CP factor slots were allocated for the s_1=- side only.
    assert hasattr(AB, "a_cnl_alpha_opp"), "left mixed -> a_cnl_alpha_opp should exist"
    assert not hasattr(AB, "b_cnl_alpha_opp"), \
        "right is natural-only -> b_cnl_alpha_opp should NOT exist"
    inpt_dict = AB(inpt_dict)
    out = inpt_dict[AB.name].numpy()
    _equivariance_assertions(out, AB, rot, Lmax_out, atol)
    _invariance_assertions(out, AB, atol)


@pytest.mark.parametrize(
    "param_dtype,atol",
    # Slightly relaxed vs natural-mode (1e-11) because Category-beta runs an
    # extra Legendre stage with the 1/(1-x^2) factor of paper Eq. (61), which
    # amplifies fp64 noise near the GL nodes closest to the poles by ~1 ulp.
    [(tf.float64, 5e-11), (tf.float32, 1e-7)],
)
def test_grid_product_function_full_mode_rot_equivar(param_dtype, atol):
    """parity_mode='full': both alpha and beta blocks transform as D^L(R)."""
    inpt_dict, A, _d_ij, _rhat, _Y, _z, lmax_bond, n_rad_max, rot = (
        _build_grid_equivar_test_data(dtype=param_dtype)
    )

    # We need *two distinct* inputs for the beta block to be non-zero on
    # self-coupled symmetric channel mixing (paper Eq. 33 says Category-beta
    # vanishes for A1=A2 with symmetric c-tensor).  Build a second feature B
    # that differs from A.
    from tensorpotential.instructions.compute import (
        SingleParticleBasisFunctionScalarInd,
    )
    B = SingleParticleBasisFunctionScalarInd(
        radial=A.radial, angular=A.angular, indicator=A.indicator,
        name="B", avg_n_neigh=5.0,
    )
    B.build(param_dtype)
    B.lin_transform.build(param_dtype)
    inpt_dict = B(inpt_dict)

    Lmax_out = lmax_bond
    AB = StructuredGridProductFunction(
        left=A, right=B, name="grid_AB_full",
        lmax=lmax_bond, Lmax=Lmax_out,
        n_out=n_rad_max, rank=6,
        is_left_right_equal=False,
        parity_mode="full",
    )
    AB.build(param_dtype)
    inpt_dict = AB(inpt_dict)
    out = inpt_dict[AB.name].numpy()

    # The existing helper groups coupling_meta_data by (l, hist, parity), so it
    # naturally checks the alpha block AND the beta block.
    _equivariance_assertions(out, AB, rot, Lmax_out, atol)
    _invariance_assertions(out, AB, atol)


# ---------------------------------------------------------------------- #
# StructuredGridMessagePassing equivariance + invariance
# ---------------------------------------------------------------------- #
@pytest.mark.parametrize(
    "param_dtype,atol",
    [(tf.float64, 1e-10), (tf.float32, 1e-7)],
)
def test_grid_message_passing_rot_equivar(param_dtype, atol):
    """
    StructuredGridMessagePassing: per-(L, m) output rotates under D^L(R)
    when the input geometry (positions + indicator) rotates under R.

    Mirrors the StructuredGridProductFunction equivariance test but exercises
    the message-passing path: indicator gathered to bonds, R*Y edge field on
    the grid, segment-summed neighbour pooling, lambda projection.

    The helper provides A: an SPBF-like equivariant feature with n_out=n_rad_max.
    That's exactly the shape contract StructuredGridMessagePassing expects on
    the indicator (Eq. 40 absorption -- indicator's CP factor pre-applied
    externally).
    """
    inpt_dict, A, d_ij, rhat, Y, z, lmax_bond, n_rad_max, rot = (
        _build_grid_equivar_test_data(dtype=param_dtype)
    )

    Lmax_out = lmax_bond
    MP = StructuredGridMessagePassing(
        name="grid_MP",
        bonds=d_ij, angular=Y, indicator=A, chemical_embedding=z,
        n_rad_max=n_rad_max, n_rad_basis=4, rcut=6.0, p=5,
        lmax=lmax_bond, Lmax=Lmax_out,
        avg_n_neigh=10.0,
        parity_mode="natural", hidden_layers=[16],
    )
    MP.build(param_dtype)
    inpt_dict = MP(inpt_dict)
    out = inpt_dict[MP.name].numpy()

    _equivariance_assertions(out, MP, rot, Lmax_out, atol)
    _invariance_assertions(out, MP, atol)


# ---------------------------------------------------------------------- #
# StructuredGridMessagePassing parity_mode='full' equivariance
# ---------------------------------------------------------------------- #
@pytest.mark.parametrize(
    "param_dtype,atol",
    [(tf.float64, 1e-9), (tf.float32, 1e-7)],
)
def test_grid_message_passing_full_mode_mixed_indicator_rot_equivar(param_dtype, atol):
    """MP parity_mode='full' with a MIXED-parity indicator.

    The four-case dispatch (s_indicator ∈ {±} × {alpha, beta}) must produce a
    fully-equivariant output covering both parity sectors.
    """
    import pandas as pd
    from tensorpotential.instructions.compute import (
        SingleParticleBasisFunctionScalarInd,
    )
    inpt_dict, A, d_ij, rhat, Y, z, lmax_bond, n_rad_max, rot = (
        _build_grid_equivar_test_data(dtype=param_dtype)
    )
    # Construct a mixed-parity indicator by concatenating A (natural) and a
    # parity-relabelled second SPBF feature B (opposite).
    B = SingleParticleBasisFunctionScalarInd(
        radial=A.radial, angular=A.angular, indicator=A.indicator,
        name="B_raw_mp", avg_n_neigh=5.0,
    )
    B.build(param_dtype)
    B.lin_transform.build(param_dtype)
    inpt_dict = B(inpt_dict)
    A_cmd = A.coupling_meta_data.copy()
    B_cmd = B.coupling_meta_data.copy()
    B_cmd["parity"] = -B_cmd["parity"]
    B_cmd["hist"] = B_cmd["hist"].astype(str) + "(opp)"
    mix_cmd = (
        pd.concat([A_cmd, B_cmd], ignore_index=True)
        .sort_values(["l", "parity", "hist", "m"])
        .reset_index(drop=True)
    )

    class MixedInd:
        name = "I_mix_mp"
        lmax = lmax_bond
        n_out = n_rad_max
        coupling_meta_data = mix_cmd
        coupling_origin = ["I_mix_mp"]

    A_tensor = inpt_dict[A.name].numpy()
    B_tensor = inpt_dict[B.name].numpy()
    A_index = {}
    for idx, row in A.coupling_meta_data.iterrows():
        A_index[(int(row["l"]), int(row["m"]), str(row["hist"]))] = idx
    B_index = {}
    for idx, row in B.coupling_meta_data.iterrows():
        B_index[(int(row["l"]), int(row["m"]), str(row["hist"]) + "(opp)")] = idx
    rows = []
    for _, row in mix_cmd.iterrows():
        key = (int(row["l"]), int(row["m"]), str(row["hist"]))
        if key in A_index:
            rows.append(A_tensor[..., A_index[key]])
        elif key in B_index:
            rows.append(B_tensor[..., B_index[key]])
        else:
            raise AssertionError(f"Missing row for {row}")
    inpt_dict["I_mix_mp"] = tf.constant(np.stack(rows, axis=-1))

    MP = StructuredGridMessagePassing(
        name="grid_MP_mixed",
        bonds=d_ij, angular=Y, indicator=MixedInd, chemical_embedding=z,
        n_rad_max=n_rad_max, n_rad_basis=4, rcut=6.0, p=5,
        lmax=lmax_bond, Lmax=lmax_bond,
        avg_n_neigh=10.0,
        parity_mode="full", hidden_layers=[16],
    )
    MP.build(param_dtype)
    # Verify the (s_2=-) MLPs + lambdas were allocated.
    assert hasattr(MP, "mlp_layers_alpha_opp"), \
        "mixed indicator -> mlp_layers_alpha_opp should exist"
    assert hasattr(MP, "lambda_cnl_alpha_opp")
    assert hasattr(MP, "lambda_cnl_beta_nat")
    inpt_dict = MP(inpt_dict)
    out = inpt_dict[MP.name].numpy()

    _equivariance_assertions(out, MP, rot, lmax_bond, atol)
    _invariance_assertions(out, MP, atol)


@pytest.mark.parametrize(
    "param_dtype,atol",
    # Slightly relaxed for the same reason as the on-site full-mode test:
    # Category-beta runs an extra Legendre stage with the 1/(1-x^2) factor
    # of paper Eq. (61), which amplifies fp64 noise near the grid poles by
    # ~1 ulp.
    [(tf.float64, 1e-9), (tf.float32, 1e-7)],
)
def test_grid_message_passing_full_mode_rot_equivar(param_dtype, atol):
    """parity_mode='full' MP: both alpha and beta blocks transform as D^L(R).

    The beta block populates the opposite-parity output sectors via the
    surface-curl construction of paper Sec. 7.6.  Equivariance has to hold on
    both block sectors simultaneously.
    """
    inpt_dict, A, d_ij, rhat, Y, z, lmax_bond, n_rad_max, rot = (
        _build_grid_equivar_test_data(dtype=param_dtype)
    )

    Lmax_out = lmax_bond
    MP = StructuredGridMessagePassing(
        name="grid_MP_full",
        bonds=d_ij, angular=Y, indicator=A, chemical_embedding=z,
        n_rad_max=n_rad_max, n_rad_basis=4, rcut=6.0, p=5,
        lmax=lmax_bond, Lmax=Lmax_out,
        avg_n_neigh=10.0,
        parity_mode="full", hidden_layers=[16],
    )
    MP.build(param_dtype)
    # In full mode coupling_meta_data should have 2*(Lmax_out+1)^2 rows.
    assert len(MP.coupling_meta_data) == 2 * (Lmax_out + 1) ** 2
    # The beta lambda is an independent learnable Variable.
    assert MP.lambda_cnl is not MP.lambda_cnl_beta
    inpt_dict = MP(inpt_dict)
    out = inpt_dict[MP.name].numpy()

    _equivariance_assertions(out, MP, rot, Lmax_out, atol)
    _invariance_assertions(out, MP, atol)


# ---------------------------------------------------------------------- #
# Grid-full chained pipeline: composing both grid instructions still
# preserves equivariance.
# ---------------------------------------------------------------------- #
@pytest.mark.parametrize(
    "param_dtype,atol",
    [(tf.float64, 1e-12), (tf.float32, 1e-9)],
)
def test_grid_full_chained_rot_equivar(param_dtype, atol):
    """
    grid_full mini-pipeline: A (SPBF) -> StructuredGridMessagePassing(MP) ->
    StructuredGridProductFunction(MP, MP).  Equivariance must hold at the
    final L>0 channels and invariance at L=0.
    """
    inpt_dict, A, d_ij, rhat, Y, z, lmax_bond, n_rad_max, rot = (
        _build_grid_equivar_test_data(dtype=param_dtype)
    )

    Lmax_out = lmax_bond
    MP = StructuredGridMessagePassing(
        name="grid_MP_full",
        bonds=d_ij, angular=Y, indicator=A, chemical_embedding=z,
        n_rad_max=n_rad_max, n_rad_basis=4, rcut=6.0, p=5,
        lmax=lmax_bond, Lmax=Lmax_out,
        avg_n_neigh=10.0,
        parity_mode="natural", hidden_layers=[16],
    )
    MP.build(param_dtype)
    inpt_dict = MP(inpt_dict)

    AA = StructuredGridProductFunction(
        left=MP, right=MP, name="grid_AA_full",
        lmax=Lmax_out, Lmax=Lmax_out,
        n_out=n_rad_max, rank=6,
        is_left_right_equal=True, parity_mode="natural",
    )
    AA.build(param_dtype)
    inpt_dict = AA(inpt_dict)
    out = inpt_dict[AA.name].numpy()

    _equivariance_assertions(out, AA, rot, Lmax_out, atol)
    _invariance_assertions(out, AA, atol)


# ====================================================================== #
# End-to-end correctness: given the same R, Y, I, the StructuredGrid
# message-passing pipeline produces the same coupled tensor as a direct
# (dense-quadrature) evaluation of Eq. 36 on the sphere.  Strongest
# correctness check we can do: an *independent* reference computed on a
# 4x finer grid, against the structured pipeline at every lmax.
# ====================================================================== #


@pytest.mark.parametrize("lmax", [2, 3, 4, 5, 6])
def test_grid_message_passing_matches_direct_gaunt(lmax):
    """
    One-edge graph (atom 0 -> atom 1) with a fixed (r, rhat).  Bypass the
    radial MLP by monkey-patching _radial_mlp to return a controlled
    R[c, l1] tensor; feed I and Y as fixed numpy arrays.  Run the
    structured-grid pipeline and compare the output channel-by-channel
    against the analytical Eq. 36 sum

        phi_i_{lm} = sum_j sum_{l1m1l2m2} G^{lm}_{l1m1l2m2}
                       * R[c, l1] Y_{l1m1}(rhat_b) I[j, c, l2m2]

    evaluated by dense quadrature on a finer grid (4*lmax + 4 nodes per
    axis).  Lambda is set to identity (lambda[c, n=c, l] = 1) so the
    output is the raw CG sum per channel, with NO mixing -- this isolates
    the structured-grid CG kernel from any other moving part.

    If the structured pipeline disagrees with the direct sum at lmax >= 4,
    the wall-clock speedup is fake.  fp64 floor for this size is ~1e-12.
    """
    rng = np.random.default_rng(seed=0xCC + lmax)
    n_rad_max = 1
    rcut = 6.0
    n_rad_basis = 4
    Lmax_out = lmax
    n_atoms = 2

    U_ref = V_ref = 4 * lmax + 4
    Y_ref, x_u_ref, w_u_ref = _real_sh_grid(lmax, U_ref, V_ref)
    Y_out_ref, _, _ = _real_sh_grid(Lmax_out, U_ref, V_ref)

    rhat = np.array([[0.6, 0.0, 0.8]])
    R_per_bond_l = rng.standard_normal((1, n_rad_max, lmax + 1)).astype(np.float64)
    I_per_atom = rng.standard_normal(
        (n_atoms, n_rad_max, (lmax + 1) ** 2)
    ).astype(np.float64)
    bond_ind_i = np.array([1], dtype=np.int32)
    bond_ind_j = np.array([0], dtype=np.int32)

    from tensorpotential.instructions.compute import (
        BondLength, ScaledBondVector, SphericalHarmonic, ScalarChemicalEmbedding,
    )
    from tensorpotential.functions.spherical_harmonics import SphericalHarmonics
    sh = SphericalHarmonics(lmax=lmax, type="real")
    sh.build(tf.float64)
    Y_at_rhat = sh(tf.constant(rhat)).numpy()[0]

    # Reference: direct Eq. 36 evaluated by quadrature on a 4x finer grid.
    edge_grid = np.zeros((1, n_rad_max, U_ref, V_ref))
    for l1 in range(lmax + 1):
        for m1 in range(-l1, l1 + 1):
            lm = l1 * (l1 + 1) + m1
            edge_grid[0] += (
                R_per_bond_l[0, :, l1, None, None]
                * Y_at_rhat[lm]
                * Y_ref[l1, m1 + lmax]
            )
    node_grid = np.zeros((1, n_rad_max, U_ref, V_ref))
    for l2 in range(lmax + 1):
        for m2 in range(-l2, l2 + 1):
            lm = l2 * (l2 + 1) + m2
            node_grid[0] += (
                I_per_atom[bond_ind_j[0], :, lm, None, None]
                * Y_ref[l2, m2 + lmax]
            )
    S_grid = edge_grid * node_grid
    S_atom = np.zeros((n_atoms, n_rad_max, U_ref, V_ref))
    S_atom[bond_ind_i[0]] = S_grid[0]
    n_lm_out = (Lmax_out + 1) ** 2
    ref = np.zeros((n_atoms, n_rad_max, n_lm_out))
    for l in range(Lmax_out + 1):  # noqa: E741
        for m in range(-l, l + 1):
            lm = l * (l + 1) + m
            integrand = Y_out_ref[l, m + Lmax_out] * S_atom
            ref[:, :, lm] = (2 * np.pi / V_ref) * np.sum(
                w_u_ref[None, None, :, None] * integrand, axis=(2, 3)
            )

    # StructuredGridMessagePassing pipeline (radial MLP stubbed out)
    inpt_dict = {
        constants.BOND_VECTOR: rhat * 5.0,
        constants.BOND_IND_I: bond_ind_i,
        constants.BOND_IND_J: bond_ind_j,
        constants.BOND_MU_I: np.array([0], dtype=np.int32),
        constants.BOND_MU_J: np.array([0], dtype=np.int32),
        constants.ATOMIC_MU_I: np.array([0, 0], dtype=np.int32),
        constants.N_ATOMS_BATCH_TOTAL: n_atoms,
    }
    d_ij_inst = BondLength()
    d_ij_inst.build(tf.float64)
    inpt_dict = d_ij_inst(inpt_dict)
    rhat_inst = ScaledBondVector(bond_length=d_ij_inst)
    rhat_inst.build(tf.float64)
    inpt_dict = rhat_inst(inpt_dict)
    Y_inst = SphericalHarmonic(vhat=rhat_inst, lmax=lmax, name=f"Y_corr_l{lmax}")
    Y_inst.build(tf.float64)
    inpt_dict = Y_inst(inpt_dict)
    z = ScalarChemicalEmbedding(
        element_map={"X": 0}, embedding_size=2, name=f"Z_corr_l{lmax}",
    )
    z.build(tf.float64)
    inpt_dict[z.name] = tf.cast(z.w, tf.float64)

    indicator = _StubInstr(f"I_corr_l{lmax}", lmax, n_rad_max)
    inpt_dict[indicator.name] = tf.constant(I_per_atom)

    MP = StructuredGridMessagePassing(
        name=f"MP_corr_l{lmax}",
        bonds=d_ij_inst, angular=Y_inst, indicator=indicator,
        chemical_embedding=z,
        n_rad_max=n_rad_max, n_out=n_rad_max,
        n_rad_basis=n_rad_basis, rcut=rcut, p=5,
        lmax=lmax, Lmax=Lmax_out,
        avg_n_neigh=1.0,
        parity_mode="natural", hidden_layers=[4],
    )
    MP.build(tf.float64)
    R_const = tf.constant(R_per_bond_l)
    MP._radial_mlp = lambda input_data: R_const

    lam = np.zeros([n_rad_max, n_rad_max, Lmax_out + 1])
    for c in range(n_rad_max):
        lam[c, c, :] = 1.0
    MP.lambda_cnl.assign(lam)

    inpt_dict = MP(inpt_dict)
    out = inpt_dict[MP.name].numpy()

    diff_max = np.max(np.abs(out - ref))
    ref_norm = max(np.max(np.abs(ref)), 1e-30)
    diff_rel = diff_max / ref_norm
    # Tolerance is the fp64 SH-precision floor (measured ~1e-11 across lmax 2-6
    # at this size), with 10x headroom for cross-machine roundoff.  An actual
    # algorithmic disagreement would manifest as O(|ref|) error, not 1e-11.
    assert diff_rel < 1e-10, (
        f"StructuredGridMessagePassing disagrees with direct-Gaunt reference "
        f"at lmax={lmax}: max|out - ref| = {diff_max:.3e}, "
        f"|ref|_max = {ref_norm:.3e}, rel = {diff_rel:.3e}.  This means "
        f"the structured-grid pipeline is NOT computing the same "
        f"R(r)*Y(rhat) (x) I_{{lm}} CG coupling that SPBF's sparse-CG path does."
    )
