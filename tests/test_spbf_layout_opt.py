"""Parity tests for the dense-GEMM equivariant CG couple
(``compute._USE_GEMM_COUPLE``, default True).

The GEMM couple (``prod_flat @ W``) is a pure reformulation of the original
``transpose + gather_nd + segment_sum`` couple, so forward + gradients (incl.
double-backward, for force training) must match to fp64 precision regardless of
the flag. Run on CPU.
"""

import os

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

import numpy as np
import tensorflow as tf
import pytest
from tensorflow import float64

from tensorpotential.instructions import compute
from tensorpotential.instructions.compute import (
    FunctionReduceParticular,
    CollectInvarBasis,
    CropProductFunction,
)
from tensorpotential.instructions import (
    FunctionReduce,
    GeneralProductFunction,
    ProductFunction,
    ScalarChemicalEmbedding,
    SingleParticleBasisFunctionScalarInd,
    SingleParticleBasisFunctionEquivariantInd,
    RadialBasis,
    MLPRadialFunction_v2,
    SPBF,
    BondLength,
    ScaledBondVector,
    SphericalHarmonic,
)
from tensorpotential import constants
from tensorpotential.utils import Parity

try:
    # When `tests` is imported as a package (e.g. CI `pytest` with tests/__init__.py,
    # repo root on sys.path), the bare sibling import does not resolve.
    from tests.test_spbf import _make_input_dict, _build_base_instructions
except ModuleNotFoundError:
    from test_spbf import _make_input_dict, _build_base_instructions


# ---------------------------------------------------------------------------
# Build an equivariant SPBF (combined class, used by GRACE_CE A2/A3) and an
# equivariant-indicator instruction (used by GRACE_2LAYER) — both route the
# CG couple through compute._equiv_cg_couple.
# ---------------------------------------------------------------------------
def _build_spbf_equiv_chain(inpt_dict, lmax, rcut, dense_nbr=False):
    inpt_dict, d_ij, Y, z = _build_base_instructions(inpt_dict, lmax, rcut)
    n_rad_max, n_rad_basis = 5, 8

    A = SPBF(name="A", bonds=d_ij, angular=Y, chemical_embedding=z,
             n_rad_max=n_rad_max, n_rad_basis=n_rad_basis, rcut=rcut)
    A.build(float64)
    inpt_dict = A(inpt_dict)

    AA = ProductFunction(left=A, right=A, name="AA", lmax=lmax, Lmax=lmax,
                         keep_parity=Parity.REAL_PARITY)
    AA.build(float64)
    inpt_dict = AA(inpt_dict)

    YI = SPBF(name="YI", bonds=d_ij, angular=Y, chemical_embedding=z,
              n_rad_max=n_rad_max, n_rad_basis=n_rad_basis, rcut=rcut,
              indicator=AA, lmax=lmax, Lmax=lmax, dense_nbr=dense_nbr)
    YI.build(float64)
    return inpt_dict, A, YI


def _build_equivind_chain(inpt_dict, lmax, rcut, indicator_lmax=2):
    inpt_dict, d_ij, Y, z = _build_base_instructions(inpt_dict, lmax, rcut)
    n_rad_max = 6
    g_k = RadialBasis(bonds=d_ij, basis_type="Cheb", nfunc=8, rcut=rcut)
    g_k.build(float64)
    inpt_dict = g_k(inpt_dict)
    R = MLPRadialFunction_v2(n_rad_max=n_rad_max, lmax=lmax, basis=g_k, name="R",
                             hidden_layers=[16, 16], activation=["silu", "silu"])
    R.build(float64)
    inpt_dict = R(inpt_dict)

    A = SingleParticleBasisFunctionScalarInd(radial=R, angular=Y, indicator=z,
                                             name="A", avg_n_neigh=1.0)
    A.build(float64)
    inpt_dict = A(inpt_dict)
    AA = ProductFunction(left=A, right=A, name="AA", lmax=lmax, Lmax=indicator_lmax,
                         keep_parity=Parity.REAL_PARITY, is_left_right_equal=True,
                         normalize=True)
    AA.build(float64)
    inpt_dict = AA(inpt_dict)
    ind = FunctionReduce(instructions=[AA], name="IND", ls_max=indicator_lmax,
                         n_out=n_rad_max, allowed_l_p=Parity.REAL_PARITY)
    ind.build(float64)
    inpt_dict = ind(inpt_dict)
    YI = SingleParticleBasisFunctionEquivariantInd(
        radial=R, angular=Y, indicator=ind, name="YI", avg_n_neigh=1.0,
        lmax=lmax, Lmax=lmax)
    YI.build(float64)
    return inpt_dict, YI


def _couple_parity(inpt_dict, YI, watch_keys):
    """Run YI with GEMM couple on/off; return (outputs, grads) for each."""
    refs = {k: tf.constant(inpt_dict[k]) for k in watch_keys}

    def run(gemm):
        compute._USE_GEMM_COUPLE = gemm
        d = dict(inpt_dict)
        vs = {k: tf.Variable(refs[k]) for k in watch_keys}
        d.update(vs)
        with tf.GradientTape() as t:
            out = YI.frwrd(d)
            loss = tf.reduce_sum(out * out)
        grads = t.gradient(loss, [vs[k] for k in watch_keys])
        return out.numpy(), [tf.convert_to_tensor(g).numpy() for g in grads]

    try:
        out_a, g_a = run(False)  # fallback gather couple
        out_b, g_b = run(True)   # GEMM couple (default)
    finally:
        compute._USE_GEMM_COUPLE = True
    return (out_a, g_a), (out_b, g_b)


def _couple_double_grad(inpt_dict, YI, watch_keys):
    """Run YI with GEMM couple on/off; return the second-order gradient for each.

    Force training backpropagates through the force computation (dE/dR), i.e. it
    differentiates the couple twice. The GEMM path's double-backward graph
    (einsum over a scatter_nd-built matrix) is structurally different from the
    fallback's (gather_nd + segment_sum), so we verify second derivatives match,
    not just first. We differentiate ``d = d/dx sum(g1**2)`` where
    ``g1 = d loss/dx`` -- a scalar that depends on the full first-order backward
    graph -- so a discrepancy anywhere in the double-backward shows up.
    """
    refs = {k: tf.constant(inpt_dict[k]) for k in watch_keys}

    def run(gemm):
        compute._USE_GEMM_COUPLE = gemm
        d = dict(inpt_dict)
        vs = {k: tf.Variable(refs[k]) for k in watch_keys}
        var_list = [vs[k] for k in watch_keys]
        d.update(vs)
        with tf.GradientTape() as t2:
            with tf.GradientTape() as t1:
                out = YI.frwrd(d)
                loss = tf.reduce_sum(out * out)
            g1 = t1.gradient(loss, var_list)
            # Densify any IndexedSlices (gathered inputs) before squaring.
            g1 = [tf.convert_to_tensor(g) for g in g1 if g is not None]
            seed = tf.add_n([tf.reduce_sum(g * g) for g in g1])
        h = t2.gradient(seed, var_list)
        return [None if x is None else tf.convert_to_tensor(x).numpy() for x in h]

    try:
        h_a = run(False)  # fallback gather couple
        h_b = run(True)   # GEMM couple (default)
    finally:
        compute._USE_GEMM_COUPLE = True
    return h_a, h_b


def _assert_double_grad_parity(h_a, h_b, keys):
    # second backward accumulates more than the first, so use a slightly looser
    # absolute tolerance than the first-order checks (still ~1e-14 relative on
    # the largest-magnitude entries observed).
    saw_nonzero = False
    for ha, hb, nm in zip(h_a, h_b, keys):
        assert (ha is None) == (hb is None), f"{nm}: one path has no 2nd grad"
        if ha is None:
            continue
        saw_nonzero = saw_nonzero or np.max(np.abs(ha)) > 0
        assert np.allclose(ha, hb, atol=1e-8, rtol=0), (
            f"{nm} 2nd grad differs, max {np.max(np.abs(ha - hb))}"
        )
    assert saw_nonzero, "double-backward is trivially zero; test is vacuous"


def test_gemm_couple_parity_spbf():
    """Equivariant SPBF (GRACE_CE A2/A3 path)."""
    np.random.seed(322)
    tf.random.set_seed(322)
    inpt_dict = _make_input_dict()
    inpt_dict, _A, YI = _build_spbf_equiv_chain(inpt_dict, lmax=3, rcut=6.0)
    keys = [YI.angular.name]
    (out_a, g_a), (out_b, g_b) = _couple_parity(inpt_dict, YI, keys)

    assert out_a.shape == out_b.shape
    assert np.allclose(out_a, out_b, atol=1e-10, rtol=0), (
        f"forward differs, max {np.max(np.abs(out_a - out_b))}"
    )
    for ga, gb in zip(g_a, g_b):
        assert np.allclose(ga, gb, atol=1e-9, rtol=0), (
            f"grad differs, max {np.max(np.abs(ga - gb))}"
        )


def test_gemm_couple_parity_equivind():
    """SingleParticleBasisFunctionEquivariantInd (GRACE_2LAYER path)."""
    np.random.seed(322)
    tf.random.set_seed(322)
    inpt_dict = _make_input_dict()
    inpt_dict, YI = _build_equivind_chain(inpt_dict, lmax=3, rcut=6.0)
    keys = [YI.radial.name, YI.angular.name, YI.indicator.name]
    (out_a, g_a), (out_b, g_b) = _couple_parity(inpt_dict, YI, keys)

    assert np.allclose(out_a, out_b, atol=1e-10, rtol=0), (
        f"forward differs, max {np.max(np.abs(out_a - out_b))}"
    )
    for ga, gb, nm in zip(g_a, g_b, keys):
        assert np.allclose(ga, gb, atol=1e-9, rtol=0), (
            f"{nm} grad differs, max {np.max(np.abs(ga - gb))}"
        )

def test_gemm_couple_double_backward_spbf():
    """Second-derivative parity for the equivariant SPBF path (force training)."""
    np.random.seed(322)
    tf.random.set_seed(322)
    inpt_dict = _make_input_dict()
    inpt_dict, _A, YI = _build_spbf_equiv_chain(inpt_dict, lmax=3, rcut=6.0)
    keys = [YI.angular.name]
    h_a, h_b = _couple_double_grad(inpt_dict, YI, keys)
    _assert_double_grad_parity(h_a, h_b, keys)


def test_gemm_couple_double_backward_equivind():
    """Second-derivative parity for the SingleParticleBasisFunctionEquivariantInd path."""
    np.random.seed(322)
    tf.random.set_seed(322)
    inpt_dict = _make_input_dict()
    inpt_dict, YI = _build_equivind_chain(inpt_dict, lmax=3, rcut=6.0)
    keys = [YI.radial.name, YI.angular.name, YI.indicator.name]
    h_a, h_b = _couple_double_grad(inpt_dict, YI, keys)
    _assert_double_grad_parity(h_a, h_b, keys)



def test_dense_nbr_arg():
    """dense_nbr=True -> the equivariant SPBF runs the dense reshape compute
    (self.dense_nbr) and is dense_capable, while declaring NO extra input (same
    signature as segment_sum -- the reshape consumes a different bond layout, not a
    new tensor). False (default) -> segment_sum. (Model-level E/F parity vs
    segment_sum is covered by tests/test_calculator.py.)"""
    np.random.seed(322)
    tf.random.set_seed(322)
    _, _A, YI_on = _build_spbf_equiv_chain(_make_input_dict(), lmax=3, rcut=6.0,
                                           dense_nbr=True)
    _, _A2, YI_off = _build_spbf_equiv_chain(_make_input_dict(), lmax=3, rcut=6.0)
    assert YI_on.dense_nbr is True
    assert YI_on.dense_capable is True
    assert YI_off.dense_nbr is False
    # reshape adds no input -> identical input signature to segment_sum
    assert set(YI_on.input_tensor_spec) == set(YI_off.input_tensor_spec)


def test_dense_nbr_manager_broadcast_and_scalar_mask():
    """InstructionManager(dense_nbr=True) broadcasts to instructions that opt in:
    the equivariant SPBF resolves dense_nbr=None to True; a scalar SPBF (no
    equivariant neighbor product) stays False."""
    from tensorpotential.instructions.base import InstructionManager

    np.random.seed(322)
    tf.random.set_seed(322)
    inpt_dict = _make_input_dict()
    with InstructionManager(dense_nbr=True):
        # dense_nbr defaults to None -> resolved against the manager.
        _, A_scalar, YI = _build_spbf_equiv_chain(inpt_dict, lmax=3, rcut=6.0,
                                                  dense_nbr=None)
    assert YI.dense_nbr is True, "equivariant SPBF should adopt manager default"
    assert A_scalar.dense_nbr is False, "scalar SPBF should mask dense_nbr to False"


def test_gemm_couple_is_default():
    assert compute._USE_GEMM_COUPLE is True


# ---------------------------------------------------------------------------
# cp_lL compact projection (GeneralProductFunction mode="cp_lL").
# The per-CG-pair U/V weight depends on the pair only through (group, lm-channel),
# and there are only D << n_cg distinct such pairs, so the projection runs over
# the D distinct pairs then gathers D -> n_cg. This is the only cp_lL path (both
# layouts), validated two ways: (1) the standard layout against an independent
# per-n_cg oracle (the pre-compact arithmetic, written out below); (2) the
# lm_first layout against the standard layout (cross-layout parity, via
# _lmfirst_parity). fp64 throughout. Run on CPU.
# ---------------------------------------------------------------------------
def _build_cpll_gpf(inpt_dict, lmax, rcut, rank=4):
    """A small GeneralProductFunction(mode='cp_lL') over a self-coupled
    equivariant SPBF — exercises the per-(l, L) projection the compact path
    reorders. Returns (inpt_dict, AA)."""
    inpt_dict, d_ij, Y, z = _build_base_instructions(inpt_dict, lmax, rcut)
    A = SPBF(name="A", bonds=d_ij, angular=Y, chemical_embedding=z,
             n_rad_max=5, n_rad_basis=8, rcut=rcut)
    A.build(float64)
    inpt_dict = A(inpt_dict)
    AA = GeneralProductFunction(
        left=A, right=A, name="AA", lmax=lmax, Lmax=lmax, rank=rank,
        mode="cp_lL", keep_parity=Parity.REAL_PARITY, is_left_right_equal=True,
    )
    AA.build(float64)
    return inpt_dict, AA


def _naive_cp_lL_standard(AA, left, right):
    """Independent per-n_cg reference for mode='cp_lL', standard layout
    [atoms, n, lm]: gather lm -> n_cg, project each CG pair by its group's U/V,
    CG-couple, optional S. This is the pre-compact arithmetic; the in-production
    compact path must reproduce it bit-for-bit (fp64)."""
    lft = tf.gather(left, AA.left_ind, axis=2)  # [atoms, n, n_cg]
    rght = tf.gather(right, AA.right_ind, axis=2)
    U_per_cg = tf.gather(AA.U, AA.cg_u_group, axis=0)  # [n_cg, R, n]
    V_per_cg = tf.gather(AA.V, AA.cg_v_group, axis=0)
    lft_proj = tf.einsum("wrn,anw->arw", U_per_cg, lft) * AA.norm_u  # [atoms, R, n_cg]
    rght_proj = tf.einsum("wrn,anw->arw", V_per_cg, rght) * AA.norm_v
    prod = lft_proj * rght_proj * tf.cast(AA.cg, lft_proj.dtype)
    prod = tf.transpose(prod, [2, 0, 1])
    prod = tf.math.unsorted_segment_sum(prod, AA.m_sum_ind, num_segments=AA.nfunc)
    prod = tf.transpose(prod, [1, 2, 0])  # [atoms, R, nfunc]
    if AA.use_S:
        prod = tf.einsum("kr,arL->akL", AA.S, prod) * AA.norm_s
    return prod


def _cpll_std_fwd_grad(inpt_dict, AA, use_naive):
    """Forward + first-order grad of AA.frwrd (compact) or the naive oracle,
    watching the shared left/right input (standard layout)."""
    key = AA.left.name
    v = tf.Variable(tf.constant(inpt_dict[key]))
    d = dict(inpt_dict)
    d[key] = v
    with tf.GradientTape() as t:
        out = (
            _naive_cp_lL_standard(AA, d[AA.left.name], d[AA.right.name])
            if use_naive
            else AA.frwrd(d)
        )
        loss = tf.reduce_sum(out * out)
    g = tf.convert_to_tensor(t.gradient(loss, v))
    return out.numpy(), g.numpy()


def test_cpll_compact_matches_naive():
    """Standard-layout compact cp_lL forward + grad vs the independent per-n_cg
    oracle."""
    np.random.seed(322)
    tf.random.set_seed(322)
    inpt_dict, AA = _build_cpll_gpf(_make_input_dict(), lmax=3, rcut=6.0)
    out_n, g_n = _cpll_std_fwd_grad(inpt_dict, AA, use_naive=True)
    out_c, g_c = _cpll_std_fwd_grad(inpt_dict, AA, use_naive=False)
    assert out_n.shape == out_c.shape
    assert np.max(np.abs(out_c)) > 0, "forward is trivially zero; test is vacuous"
    assert np.allclose(out_n, out_c, atol=1e-10, rtol=0), (
        f"forward differs, max {np.max(np.abs(out_n - out_c))}"
    )
    assert np.allclose(g_n, g_c, atol=1e-9, rtol=0), (
        f"grad differs, max {np.max(np.abs(g_n - g_c))}"
    )


def test_lmfirst_general_product_cp_lL():
    """lm_first cp_lL must match the standard layout (the compact projection is
    the only path), forward + first-order grad, transpose-aligned."""
    inpt = _make_input_dict()

    def build(lm_first):
        i, A = _spbf_producer(dict(inpt), lmax=3, rcut=6.0, lm_first=lm_first)
        AA = GeneralProductFunction(
            left=A, right=A, name="AA", lmax=3, Lmax=3, rank=4, mode="cp_lL",
            keep_parity=Parity.REAL_PARITY, is_left_right_equal=True,
            lm_first=lm_first)
        AA.build(float64)
        return i, A, AA
    _assert_fwd_grad(*_lmfirst_parity(build))


def test_lmfirst_general_product_cp_lL_double_backward():
    """Second-derivative cross-layout parity for cp_lL (force training)."""
    inpt = _make_input_dict()

    def build(lm_first):
        i, A = _spbf_producer(dict(inpt), lmax=3, rcut=6.0, lm_first=lm_first)
        AA = GeneralProductFunction(
            left=A, right=A, name="AA", lmax=3, Lmax=3, rank=4, mode="cp_lL",
            keep_parity=Parity.REAL_PARITY, is_left_right_equal=True,
            lm_first=lm_first)
        AA.build(float64)
        return i, A, AA
    (_, h_a), (_, h_b) = _lmfirst_parity(build, double=True)
    assert np.max(np.abs(h_a)) > 0, "double-backward trivially zero; test vacuous"
    assert np.allclose(h_a, h_b, atol=1e-8, rtol=0), (
        f"2nd grad differs, max {np.max(np.abs(h_a - h_b))}"
    )


# ---------------------------------------------------------------------------
# lm_first layout for the previously atom-major-only equivariant instructions
# (ProductFunction, CropProductFunction, FunctionReduce, FunctionReduceParticular,
# CollectInvarBasis). lm_first keeps the lm/coupling axis leading; it is a pure
# layout reformulation, so building the SAME chain (same seed -> identical
# weights) with lm_first=False vs True must give bit-identical fp64 results once
# the equivariant lm_first output [lm, atoms, n] is transposed back to [atoms, n,
# lm]. Gradients (wrt the producer output, fed in the matching layout) must match
# transpose-aligned. Invariant outputs (CollectInvarBasis) are layout-agnostic.
# ---------------------------------------------------------------------------
def _lmfirst_parity(build_fn, equivariant_out=True, double=False):
    """build_fn(lm_first) -> (inpt_dict, producer, instr). Watch the producer's
    output (the direct input to instr), fed in the layout matching lm_first."""
    def run(lm_first):
        np.random.seed(404)
        tf.random.set_seed(404)
        inpt_dict, producer, instr = build_fn(lm_first)
        ref = tf.constant(inpt_dict[producer.name])
        d = dict(inpt_dict)
        v = tf.Variable(ref)
        d[producer.name] = v

        def fwd():
            out = instr.frwrd(d)
            if lm_first and equivariant_out:
                out = tf.transpose(out, [1, 2, 0])  # [lm,a,n] -> [a,n,lm]
            return out

        if not double:
            with tf.GradientTape() as t:
                out = fwd()
                loss = tf.reduce_sum(out * out)
            g = tf.convert_to_tensor(t.gradient(loss, v))
            if lm_first:
                g = tf.transpose(g, [1, 2, 0])  # align grad layout to standard
            return out.numpy(), g.numpy()
        with tf.GradientTape() as t2:
            with tf.GradientTape() as t1:
                out = fwd()
                loss = tf.reduce_sum(out * out)
            g1 = tf.convert_to_tensor(t1.gradient(loss, v))
            seed = tf.reduce_sum(g1 * g1)
        h = tf.convert_to_tensor(t2.gradient(seed, v))
        if lm_first:
            h = tf.transpose(h, [1, 2, 0])
        return None, h.numpy()

    return run(False), run(True)


def _spbf_producer(inpt_dict, lmax, rcut, lm_first):
    inpt_dict, d_ij, Y, z = _build_base_instructions(inpt_dict, lmax, rcut)
    A = SPBF(name="A", bonds=d_ij, angular=Y, chemical_embedding=z,
             n_rad_max=5, n_rad_basis=8, rcut=rcut, lm_first=lm_first)
    A.build(float64)
    inpt_dict = A(inpt_dict)
    return inpt_dict, A


def _product_producer(inpt_dict, lmax, rcut, lm_first, Lmax):
    """SPBF -> ProductFunction, used as the equivariant input to the reduces."""
    inpt_dict, A = _spbf_producer(inpt_dict, lmax, rcut, lm_first)
    AA = ProductFunction(left=A, right=A, name="AA", lmax=lmax, Lmax=Lmax,
                         keep_parity=Parity.REAL_PARITY, is_left_right_equal=True,
                         lm_first=lm_first)
    AA.build(float64)
    inpt_dict = AA(inpt_dict)
    return inpt_dict, AA


def _assert_fwd_grad(a, b, fwd_atol=1e-10, g_atol=1e-9):
    (out_a, g_a), (out_b, g_b) = a, b
    assert out_a.shape == out_b.shape, f"{out_a.shape} vs {out_b.shape}"
    assert np.max(np.abs(out_a)) > 0, "forward trivially zero; test vacuous"
    assert np.allclose(out_a, out_b, atol=fwd_atol, rtol=0), (
        f"forward differs, max {np.max(np.abs(out_a - out_b))}")
    assert np.allclose(g_a, g_b, atol=g_atol, rtol=0), (
        f"grad differs, max {np.max(np.abs(g_a - g_b))}")


def test_lmfirst_product_function():
    inpt = _make_input_dict()

    def build(lm_first):
        i, A = _spbf_producer(dict(inpt), lmax=3, rcut=6.0, lm_first=lm_first)
        AA = ProductFunction(left=A, right=A, name="AA", lmax=3, Lmax=3,
                             keep_parity=Parity.REAL_PARITY, is_left_right_equal=True,
                             lm_first=lm_first)
        AA.build(float64)
        return i, A, AA
    _assert_fwd_grad(*_lmfirst_parity(build))


def test_lmfirst_crop_product_function():
    inpt = _make_input_dict()

    def build(lm_first):
        i, A = _spbf_producer(dict(inpt), lmax=3, rcut=6.0, lm_first=lm_first)
        AA = CropProductFunction(left=A, right=A, name="AA", lmax=3, Lmax=3,
                                 n_crop=3, keep_parity=Parity.REAL_PARITY,
                                 is_left_right_equal=True, lm_first=lm_first)
        AA.build(float64)
        return i, A, AA
    _assert_fwd_grad(*_lmfirst_parity(build))


def test_lmfirst_function_reduce():
    inpt = _make_input_dict()

    def build(lm_first):
        i, AA = _product_producer(dict(inpt), lmax=3, rcut=6.0, lm_first=lm_first, Lmax=2)
        RN = FunctionReduce(instructions=[AA], name="IND", ls_max=2,
                            n_out=4, allowed_l_p=Parity.REAL_PARITY, lm_first=lm_first)
        RN.build(float64)
        return i, AA, RN
    _assert_fwd_grad(*_lmfirst_parity(build))


def test_lmfirst_function_reduce_particular():
    inpt = _make_input_dict()

    def build(lm_first):
        i, AA = _product_producer(dict(inpt), lmax=3, rcut=6.0, lm_first=lm_first, Lmax=2)
        RN = FunctionReduceParticular(instructions=[AA], name="PART", selected_l=2,
                                      selected_p=1, n_out=4, lm_first=lm_first)
        RN.build(float64)
        return i, AA, RN
    _assert_fwd_grad(*_lmfirst_parity(build))


def test_lmfirst_collect_invar_basis():
    inpt = _make_input_dict()

    # CollectInvarBasis is abstract (missing upd_init_args_new_elements) and unused
    # in presets; a concrete stub lets us validate the lm_first frwrd logic.
    class _ConcreteCollectInvar(CollectInvarBasis):
        def upd_init_args_new_elements(self, new_element_map):
            pass

    def build(lm_first):
        i, AA = _product_producer(dict(inpt), lmax=3, rcut=6.0, lm_first=lm_first, Lmax=2)
        CB = _ConcreteCollectInvar(instructions=[AA], name="CB", ls_max=0, lm_first=lm_first)
        CB.build(float64)
        return i, AA, CB
    # invariant output [atoms, features] is layout-agnostic -> no transpose
    _assert_fwd_grad(*_lmfirst_parity(build, equivariant_out=False))


def test_lmfirst_product_double_backward():
    inpt = _make_input_dict()

    def build(lm_first):
        i, A = _spbf_producer(dict(inpt), lmax=3, rcut=6.0, lm_first=lm_first)
        AA = ProductFunction(left=A, right=A, name="AA", lmax=3, Lmax=3,
                             keep_parity=Parity.REAL_PARITY, is_left_right_equal=True,
                             lm_first=lm_first)
        AA.build(float64)
        return i, A, AA
    (_, h_a), (_, h_b) = _lmfirst_parity(build, double=True)
    assert np.max(np.abs(h_a)) > 0, "double-backward trivially zero; test vacuous"
    assert np.allclose(h_a, h_b, atol=1e-8, rtol=0), (
        f"2nd grad differs, max {np.max(np.abs(h_a - h_b))}")


if __name__ == "__main__":
    import sys

    sys.exit(pytest.main([__file__, "-v"]))
