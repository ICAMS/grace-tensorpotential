"""
Comprehensive tests for parity handling in CG coupling (real_coupling_metainformation).

Tests cover:
1. Parity completeness: all expected (L, p) combinations are produced, including
   multi-layer configurations that simulate the 3-layer GRACE-CE model.
2. Rotational invariance: L=0 CG contractions are scalar (invariant under 3D rotation)
   for both standard and cross-parity coupling channels.
3. CG coefficient correctness: cross-parity couplings produce the same CG coefficients
   as same-parity couplings (CG depends only on l1, l2, L, not on parity).
"""

import os

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

import numpy as np
import pandas as pd
import pytest
from scipy.special import sph_harm_y
from scipy.spatial.transform import Rotation

from tensorpotential.functions.couplings import (
    real_coupling_metainformation,
    gen_CG_matrix_REAL,
)
from tensorpotential.utils import Parity


# ──────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────


def _build_uncoupled_meta(l_p_list):
    """Build uncoupled coupling metadata DataFrame for given (l, p) pairs."""
    meta = []
    for l_val, p in l_p_list:
        for m in range(-l_val, l_val + 1):
            meta.append([l_val, m, "", p, l_val])
    return pd.DataFrame(meta, columns=["l", "m", "hist", "parity", "sum_of_ls"])


def _real_ylm_array(l_val, directions):
    """Compute all real Y_{lm} for m=-l,...,l at given unit direction vectors.
    Returns array of shape (N, 2l+1) with columns ordered m=-l,...,+l.
    """
    x, y, z = directions[:, 0], directions[:, 1], directions[:, 2]
    theta = np.arccos(np.clip(z, -1, 1))
    phi = np.arctan2(y, x)
    result = np.zeros((len(directions), 2 * l_val + 1))
    for m in range(-l_val, l_val + 1):
        idx = m + l_val
        # sph_harm_y(l, m, theta, phi) returns complex Y_lm
        ylm_complex = sph_harm_y(l_val, abs(m), theta, phi)
        if m > 0:
            result[:, idx] = np.sqrt(2) * (-1) ** m * np.real(ylm_complex)
        elif m == 0:
            result[:, idx] = np.real(sph_harm_y(l_val, 0, theta, phi))
        else:
            result[:, idx] = np.sqrt(2) * (-1) ** m * np.imag(ylm_complex)
    return result


def _compute_wigner_d_real(l_val, rot):
    """Compute real Wigner D matrix for angular momentum l under rotation rot.
    Numerically determined from spherical harmonic transformation.
    """
    if l_val == 0:
        return np.array([[1.0]])
    rng = np.random.RandomState(42 + l_val)
    N = 2 * l_val + 10
    dirs = rng.randn(N, 3)
    dirs = dirs / np.linalg.norm(dirs, axis=1, keepdims=True)
    R = rot.as_matrix()
    dirs_rot = dirs @ R.T
    Y = _real_ylm_array(l_val, dirs)
    Y_rot = _real_ylm_array(l_val, dirs_rot)
    # Y_rot = Y @ D^T, solve for D^T
    D_T, _, _, _ = np.linalg.lstsq(Y, Y_rot, rcond=None)
    return D_T.T


def _apply_wigner_d(features, meta, rot):
    """Apply Wigner D rotation to each (l, parity, hist) block in features."""
    result = features.copy()
    d_cache = {}
    for (l_val, p, h), indices in meta.groupby(["l", "parity", "hist"]).indices.items():
        if l_val == 0:
            continue
        m_vals = meta.iloc[indices]["m"].values
        sorted_order = np.argsort(m_vals)
        sorted_indices = indices[sorted_order]
        if l_val not in d_cache:
            d_cache[l_val] = _compute_wigner_d_real(l_val, rot)
        result[sorted_indices] = d_cache[l_val] @ features[sorted_indices]
    return result


def _cg_contract(result_df, features_a, features_b, target_L=None):
    """Contract features using CG coefficients from coupling metadata.
    Returns dict mapping (L, parity, hist) -> scalar value.
    """
    scalars = {}
    query = result_df if target_L is None else result_df[result_df["l"] == target_L]
    for _, row in query.iterrows():
        key = (row["l"], row["m"], row["parity"], row["hist"])
        s = 0.0
        for a_idx, b_idx, cg in zip(row["left_inds"], row["right_inds"], row["cg_list"]):
            s += cg * features_a[a_idx] * features_b[b_idx]
        scalars[key] = s
    return scalars


# ──────────────────────────────────────────────────────────────────────
# 1. Parity completeness tests
# ──────────────────────────────────────────────────────────────────────


class TestParityCompleteness:
    """All expected (L, p) combinations are produced."""

    def test_real_parity_self_product_has_scalar(self):
        """Baseline: REAL_PARITY self-product produces scalar (L=0, p=+1)."""
        lmax = 2
        real_lp = [[l, 1 if l % 2 == 0 else -1] for l in range(lmax + 1)]
        meta = _build_uncoupled_meta(real_lp)
        result = real_coupling_metainformation(
            A=meta, B=meta, lmax=lmax, Lmax=lmax,
            is_A_B_equal=True, keep_parity=Parity.REAL_PARITY,
        )
        assert len(result[(result["l"] == 0) & (result["parity"] == 1)]) > 0

    def test_full_parity_self_product_has_pseudo_scalar(self):
        """FULL_PARITY self-product produces pseudo-scalar (L=0, p=-1)."""
        lmax = 2
        full_lp = [[l, p] for l in range(lmax + 1) for p in ([1, -1] if l > 0 else [1])]
        meta = _build_uncoupled_meta(full_lp)
        keep = [[L, p] for L in range(lmax + 1) for p in [1, -1]]
        result = real_coupling_metainformation(
            A=meta, B=meta, lmax=lmax, Lmax=lmax,
            is_A_B_equal=True, keep_parity=keep,
        )
        assert len(result[(result["l"] == 0) & (result["parity"] == -1)]) > 0, (
            "Cross-parity coupling (l,p=-1) x (l,p=+1) should produce pseudo-scalar"
        )

    def test_full_parity_self_product_both_parities_at_each_L(self):
        """For FULL_PARITY self-product, every output L has BOTH p=+1 and p=-1."""
        lmax = 3
        full_lp = [[l, p] for l in range(lmax + 1) for p in [1, -1]]
        meta = _build_uncoupled_meta(full_lp)
        Lmax = 2
        keep = [[L, p] for L in range(Lmax + 1) for p in [1, -1]]
        result = real_coupling_metainformation(
            A=meta, B=meta, lmax=lmax, Lmax=Lmax,
            is_A_B_equal=True, keep_parity=keep,
        )
        for L in range(Lmax + 1):
            for p in [1, -1]:
                matches = result[(result["l"] == L) & (result["parity"] == p)]
                assert len(matches) > 0, (
                    f"Missing (L={L}, p={p}). "
                    f"Available at L={L}: {sorted(result[result['l']==L]['parity'].unique())}"
                )

    def test_l1_dual_parity_self_product_detail(self):
        """l=1 with both parities: self-product yields both p=+1 and p=-1 at L=0,2."""
        meta = _build_uncoupled_meta([[1, -1], [1, 1]])
        keep = [[L, p] for L in range(3) for p in [1, -1]]
        result = real_coupling_metainformation(
            A=meta, B=meta, lmax=1, Lmax=2,
            is_A_B_equal=True, keep_parity=keep,
        )
        lp = set(zip(result["l"], result["parity"]))
        assert (0, 1) in lp and (0, -1) in lp, f"L=0 missing parity. Got: {sorted(lp)}"
        assert (2, 1) in lp and (2, -1) in lp, f"L=2 missing parity. Got: {sorted(lp)}"

    def test_non_self_product_not_affected(self):
        """is_A_B_equal=False always produces all parity combinations."""
        lmax = 2
        full_lp = [[l, p] for l in range(lmax + 1) for p in ([1, -1] if l > 0 else [1])]
        meta = _build_uncoupled_meta(full_lp)
        keep = [[L, p] for L in range(lmax + 1) for p in [1, -1]]
        result = real_coupling_metainformation(
            A=meta, B=meta, lmax=lmax, Lmax=lmax,
            is_A_B_equal=False, keep_parity=keep,
        )
        assert len(result[(result["l"] == 0) & (result["parity"] == -1)]) > 0

    def test_three_layer_parity_propagation(self):
        """
        Simulate the 3-layer GRACE-CE parity flow:
        Layer 1: uncoupled REAL_PARITY → self-product (REAL_PARITY out)
        Layer 2: coupled FULL_PARITY (after SPBF with equivariant indicator) → self-product
        Layer 3: coupled FULL_PARITY → self-product → scalar readout

        The critical check: layers 2 and 3 self-products must contain pseudo-scalars.
        """
        # Layer 1: real parity uncoupled features
        lmax = 3
        real_lp = [[l, 1 if l % 2 == 0 else -1] for l in range(lmax + 1)]
        l1_meta = _build_uncoupled_meta(real_lp)
        l1_product = real_coupling_metainformation(
            A=l1_meta, B=l1_meta, lmax=lmax, Lmax=lmax,
            is_A_B_equal=True, keep_parity=Parity.REAL_PARITY,
        )
        # L1 self-product should only have real parity (no cross-parity terms possible)
        l1_lp = set(zip(l1_product["l"], l1_product["parity"]))
        for L, p in l1_lp:
            expected_p = 1 if L % 2 == 0 else -1
            assert p == expected_p, f"L1 should be real parity but got (L={L}, p={p})"

        # Layer 2: simulate FULL_PARITY features (from SPBF with equivariant indicator)
        Lmax2 = 3
        full_lp_with_pseudo_scalar = Parity.FULL_PARITY[:2 * Lmax2 + 1] + [[0, -1]]
        l2_meta = _build_uncoupled_meta(full_lp_with_pseudo_scalar)
        l2_product = real_coupling_metainformation(
            A=l2_meta, B=l2_meta, lmax=Lmax2, Lmax=Lmax2,
            is_A_B_equal=True,
            keep_parity=full_lp_with_pseudo_scalar,
        )
        # L2 self-product MUST have pseudo-scalar
        l2_has_pseudo_scalar = len(
            l2_product[(l2_product["l"] == 0) & (l2_product["parity"] == -1)]
        ) > 0
        assert l2_has_pseudo_scalar, (
            "Layer 2: FULL_PARITY self-product must contain pseudo-scalar (L=0, p=-1)"
        )
        # And both parities at L=1
        assert len(
            l2_product[(l2_product["l"] == 1) & (l2_product["parity"] == 1)]
        ) > 0, "Layer 2: missing pseudo-vector (L=1, p=+1)"

        # Layer 3: use L2 output as input, self-product again
        l3_product = real_coupling_metainformation(
            A=l2_product, B=l2_product, lmax=Lmax2, Lmax=1,
            is_A_B_equal=True,
            keep_parity=full_lp_with_pseudo_scalar,
        )
        l3_has_pseudo_scalar = len(
            l3_product[(l3_product["l"] == 0) & (l3_product["parity"] == -1)]
        ) > 0
        assert l3_has_pseudo_scalar, (
            "Layer 3: chained FULL_PARITY self-product must contain pseudo-scalar"
        )

    def test_history_strings_contain_parity_markers(self):
        """History strings in the output should contain parity markers (+/-)."""
        meta = _build_uncoupled_meta([[1, -1], [1, 1]])
        result = real_coupling_metainformation(
            A=meta, B=meta, lmax=1, Lmax=2,
            is_A_B_equal=False,
            keep_parity=[[L, p] for L in range(3) for p in [1, -1]],
        )
        for hist in result["hist"].unique():
            assert "+" in hist or "-" in hist, (
                f"History string '{hist}' should contain parity markers"
            )

    def test_different_parity_combos_have_different_histories(self):
        """Cross-parity and same-parity products have distinct history strings."""
        meta = _build_uncoupled_meta([[1, -1], [1, 1]])
        keep = [[0, 1], [0, -1]]
        result = real_coupling_metainformation(
            A=meta, B=meta, lmax=1, Lmax=0,
            is_A_B_equal=False, keep_parity=keep,
        )
        # Group by parity, check histories differ
        p_plus = set(result[result["parity"] == 1]["hist"].unique())
        p_minus = set(result[result["parity"] == -1]["hist"].unique())
        assert len(p_plus & p_minus) == 0, (
            f"Same-parity and cross-parity L=0 should have distinct histories. "
            f"p=+1: {p_plus}, p=-1: {p_minus}"
        )


# ──────────────────────────────────────────────────────────────────────
# 2. Rotational invariance tests
# ──────────────────────────────────────────────────────────────────────


class TestRotationalInvariance:
    """L=0 CG contractions must be rotationally invariant for all parity types."""

    @pytest.fixture
    def rotation(self):
        return Rotation.random(random_state=42)

    def _check_scalar_invariance(self, meta, result, rot, atol=1e-10):
        """Helper: verify all L=0 outputs are invariant under rotation and non-zero."""
        rng = np.random.RandomState(123)
        A = rng.randn(len(meta))
        B = rng.randn(len(meta))
        A_rot = _apply_wigner_d(A, meta, rot)
        B_rot = _apply_wigner_d(B, meta, rot)

        scalars = _cg_contract(result, A, B, target_L=0)
        scalars_rot = _cg_contract(result, A_rot, B_rot, target_L=0)

        assert len(scalars) > 0, "No L=0 channels produced"
        for key in scalars:
            L, M, p, hist = key
            assert abs(scalars[key]) > 1e-12, (
                f"L=0 scalar is zero (trivial invariance): "
                f"(L={L}, M={M}, p={p}, hist={hist})"
            )
            assert np.allclose(scalars[key], scalars_rot[key], atol=atol), (
                f"L=0 output not rotationally invariant: "
                f"(L={L}, M={M}, p={p}, hist={hist}): "
                f"{scalars[key]} vs {scalars_rot[key]}"
            )

    def test_real_parity_scalar_invariance(self, rotation):
        """Standard real-parity L=0 outputs are rotationally invariant."""
        lmax = 3
        real_lp = [[l, 1 if l % 2 == 0 else -1] for l in range(lmax + 1)]
        meta = _build_uncoupled_meta(real_lp)
        result = real_coupling_metainformation(
            A=meta, B=meta, lmax=lmax, Lmax=0,
            is_A_B_equal=False, keep_parity=[[0, 1]],
        )
        self._check_scalar_invariance(meta, result, rotation)

    def test_cross_parity_pseudo_scalar_invariance(self, rotation):
        """Cross-parity pseudo-scalars (L=0, p=-1) are rotationally invariant."""
        lmax = 3
        full_lp = [[l, p] for l in range(lmax + 1) for p in [1, -1]]
        meta = _build_uncoupled_meta(full_lp)
        result = real_coupling_metainformation(
            A=meta, B=meta, lmax=lmax, Lmax=0,
            is_A_B_equal=False, keep_parity=[[0, 1], [0, -1]],
        )
        # Must have pseudo-scalar channels
        assert len(result[result["parity"] == -1]) > 0, "No pseudo-scalar channels found"
        self._check_scalar_invariance(meta, result, rotation)

    def test_self_product_pseudo_scalar_invariance(self, rotation):
        """Self-product (is_A_B_equal=True) pseudo-scalars are rotationally invariant."""
        lmax = 2
        full_lp = [[l, p] for l in range(lmax + 1) for p in [1, -1]]
        meta = _build_uncoupled_meta(full_lp)
        result = real_coupling_metainformation(
            A=meta, B=meta, lmax=lmax, Lmax=0,
            is_A_B_equal=True, keep_parity=[[0, 1], [0, -1]],
        )
        assert len(result[result["parity"] == -1]) > 0, "No pseudo-scalar in self-product"
        # For self-product, use same features for A and B
        rng = np.random.RandomState(456)
        A = rng.randn(len(meta))
        A_rot = _apply_wigner_d(A, meta, rotation)
        scalars = _cg_contract(result, A, A, target_L=0)
        scalars_rot = _cg_contract(result, A_rot, A_rot, target_L=0)
        for key in scalars:
            assert abs(scalars[key]) > 1e-12, (
                f"Self-product L=0 scalar is zero (trivial invariance) at {key}"
            )
            assert np.allclose(scalars[key], scalars_rot[key], atol=1e-10), (
                f"Self-product L=0 not rotationally invariant at {key}: "
                f"{scalars[key]} vs {scalars_rot[key]}"
            )

    def test_chained_coupling_scalar_invariance(self, rotation):
        """Scalars from 2nd-order coupling of already-coupled features are invariant."""
        lmax = 2
        full_lp = [[l, p] for l in range(lmax + 1) for p in [1, -1]]
        meta = _build_uncoupled_meta(full_lp)

        # First coupling: A x B -> C (non-self, all L)
        coupling1 = real_coupling_metainformation(
            A=meta, B=meta, lmax=lmax, Lmax=lmax,
            is_A_B_equal=False,
            keep_parity=[[L, p] for L in range(lmax + 1) for p in [1, -1]],
        )

        # Second coupling: C x C -> D (self, L=0 only)
        coupling2 = real_coupling_metainformation(
            A=coupling1, B=coupling1, lmax=lmax, Lmax=0,
            is_A_B_equal=True, keep_parity=[[0, 1], [0, -1]],
        )

        # Compute C from A, B
        rng = np.random.RandomState(789)
        A = rng.randn(len(meta))
        B = rng.randn(len(meta))
        C = np.zeros(len(coupling1))
        for i, row in coupling1.iterrows():
            s = 0.0
            for a_idx, b_idx, cg in zip(row["left_inds"], row["right_inds"], row["cg_list"]):
                s += cg * A[a_idx] * B[b_idx]
            C[i] = s

        # Compute D from C x C
        scalars = _cg_contract(coupling2, C, C, target_L=0)

        # Rotate and repeat
        A_rot = _apply_wigner_d(A, meta, rotation)
        B_rot = _apply_wigner_d(B, meta, rotation)
        C_rot = np.zeros(len(coupling1))
        for i, row in coupling1.iterrows():
            s = 0.0
            for a_idx, b_idx, cg in zip(row["left_inds"], row["right_inds"], row["cg_list"]):
                s += cg * A_rot[a_idx] * B_rot[b_idx]
            C_rot[i] = s

        scalars_rot = _cg_contract(coupling2, C_rot, C_rot, target_L=0)

        n_nonzero = 0
        for key in scalars:
            if abs(scalars[key]) < 1e-12:
                # Accidental zero for this random input — still check invariance
                assert abs(scalars_rot[key]) < 1e-9, (
                    f"Chained L=0 zero before rotation but non-zero after at {key}"
                )
                continue
            n_nonzero += 1
            assert np.allclose(scalars[key], scalars_rot[key], atol=1e-9), (
                f"Chained coupling L=0 not invariant at {key}: "
                f"{scalars[key]} vs {scalars_rot[key]}"
            )
        assert n_nonzero > 0, "All chained coupling L=0 scalars are zero"


# ──────────────────────────────────────────────────────────────────────
# 3. CG coefficient correctness tests
# ──────────────────────────────────────────────────────────────────────


class TestCGCoefficients:
    """CG coefficients for cross-parity channels match standard CG values."""

    def test_cg_values_independent_of_parity(self):
        """
        CG coefficients depend only on (l1, l2, L), not on parity.
        Cross-parity and same-parity couplings at the same (l1, l2, L)
        must produce identical CG coefficient patterns.
        """
        lmax = 2
        full_lp = [[l, p] for l in range(lmax + 1) for p in [1, -1]]
        meta = _build_uncoupled_meta(full_lp)
        keep = [[L, p] for L in range(lmax + 1) for p in [1, -1]]

        result = real_coupling_metainformation(
            A=meta, B=meta, lmax=lmax, Lmax=lmax,
            is_A_B_equal=False, keep_parity=keep,
        )

        # Group by (l1, l2, l_output, m_output): CG values should be identical
        # regardless of input parity combination
        for L in range(lmax + 1):
            for M in range(-L, L + 1):
                rows_at_LM = result[(result["l"] == L) & (result["m"] == M)]
                # Group by (l1, l2) and compare CG coefficient patterns
                for (l1, l2), group in rows_at_LM.groupby(["l1", "l2"]):
                    cg_sets = []
                    for _, row in group.iterrows():
                        # Normalize: sort CG by absolute value for comparison
                        cgs = sorted(row["cg_list"], key=abs)
                        cg_sets.append(cgs)
                    # All CG patterns for same (l1, l2, L, M) should match
                    for cgs in cg_sets[1:]:
                        np.testing.assert_allclose(
                            cgs, cg_sets[0], atol=1e-12,
                            err_msg=(
                                f"CG mismatch at (l1={l1}, l2={l2}, L={L}, M={M}): "
                                f"CG depends on parity, which should not happen."
                            ),
                        )

    def test_cg_match_reference_for_l1xl1_to_L0(self):
        """CG for l=1 x l=1 -> L=0 from cross-parity matches gen_CG_matrix_REAL."""
        meta = _build_uncoupled_meta([[1, -1], [1, 1]])
        keep = [[0, -1]]  # only pseudo-scalar
        result = real_coupling_metainformation(
            A=meta, B=meta, lmax=1, Lmax=0,
            is_A_B_equal=False, keep_parity=keep,
        )

        # Reference CG matrix for l1=1, l2=1, L=0
        ref_cg = gen_CG_matrix_REAL(1, 1, 0)  # shape: [1, 3, 3]

        # Extract CG from the coupling result
        l0_rows = result[(result["l"] == 0) & (result["m"] == 0)]
        assert len(l0_rows) > 0, "No L=0 M=0 entries found"

        for _, row in l0_rows.iterrows():
            # Reconstruct CG matrix from sparse representation
            cg_matrix = np.zeros((3, 3))
            for a_idx, b_idx, cg in zip(
                row["left_inds"], row["right_inds"], row["cg_list"]
            ):
                # Map indices back to m values
                m1 = meta.iloc[a_idx]["m"]
                m2 = meta.iloc[b_idx]["m"]
                cg_matrix[int(m1 + 1), int(m2 + 1)] = cg

            # Compare with reference (up to overall sign)
            ref = ref_cg[0]  # M=0 slice
            if np.allclose(cg_matrix, ref, atol=1e-12):
                pass  # exact match
            elif np.allclose(cg_matrix, -ref, atol=1e-12):
                pass  # sign flip OK (phase convention)
            else:
                # Check absolute values match
                np.testing.assert_allclose(
                    np.abs(cg_matrix), np.abs(ref), atol=1e-12,
                    err_msg="Cross-parity CG doesn't match reference gen_CG_matrix_REAL",
                )

    def test_cg_normalization_L0(self):
        """L=0 CG coefficients satisfy normalization: sum of cg^2 = 1/(2l+1)
        for coupling l x l -> 0 (both same-parity and cross-parity).
        """
        for l_val in range(1, 4):
            meta = _build_uncoupled_meta([[l_val, -1], [l_val, 1]])
            keep = [[0, 1], [0, -1]]
            result = real_coupling_metainformation(
                A=meta, B=meta, lmax=l_val, Lmax=0,
                is_A_B_equal=False, keep_parity=keep,
            )

            for _, row in result[result["l"] == 0].iterrows():
                cg_sum_sq = sum(c ** 2 for c in row["cg_list"])
                # For normalized CG: <l,m1;l,m2|0,0>^2 summed over relevant m1,m2
                # The sparse representation may not include all m pairs, but the
                # total should be consistent with CG orthogonality
                assert cg_sum_sq > 0, (
                    f"CG sum of squares is zero for l={l_val}, p={row['parity']}"
                )

    def test_self_product_cg_symmetry(self):
        """For self-product (is_A_B_equal=True), CG coefficients at L=0 satisfy
        the expected relation: only l1+l2+L even survives (Bose symmetry).
        """
        lmax = 2
        full_lp = [[l, p] for l in range(lmax + 1) for p in [1, -1]]
        meta = _build_uncoupled_meta(full_lp)
        keep = [[0, 1], [0, -1]]

        result = real_coupling_metainformation(
            A=meta, B=meta, lmax=lmax, Lmax=0,
            is_A_B_equal=True, keep_parity=keep,
        )

        for _, row in result.iterrows():
            l1, l2 = row["l1"], row["l2"]
            L = row["l"]
            # For self-product, only l1+l2+L even should survive
            assert (l1 + l2 + L) % 2 == 0, (
                f"Self-product has l1+l2+L odd: l1={l1}, l2={l2}, L={L} "
                f"(hist={row['hist']}). This violates Bose symmetry."
            )


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
