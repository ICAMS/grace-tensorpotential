"""Regression tests for the per-cluster robust sigma threshold estimator.

The build threshold is ``median + k*(1.4826*MAD)`` (k=3), read off each cluster's
sigma histogram via a piecewise-linear (uniform-within-bin) CDF. This guards:
  - the value formula (median + k robust-sigma, not a tail quantile),
  - bin-size independence (the whole point of interpolating within the bin),
  - the reliability gate / backfill (still p99-midpoint-based, so the reliable
    SET is unchanged from the midpoint era — only the cell VALUES are robust),
  - saturation detection in the info CLI.
"""
import numpy as np

from tensorpotential.uq.cli.build.thresholds import (
    _MAD_TO_STD,
    _ROBUST_K,
    _compute_dual_thresholds,
    _robust_threshold_from_hist,
)
from tensorpotential.uq.cli.info import _is_saturated


BINS = np.linspace(0.0, 100.0, 250)  # 250 left edges, as the build writes them
BW = float(BINS[1] - BINS[0])

# For a histogram whose mass is a single bin b (a pure spike), the uniform-
# within-bin CDF gives median = bins[b] + bw/2 and MAD = bw/4, so:
#     threshold = bins[b] + bw*(1/2 + k*1.4826/4)
_SPIKE_MARGIN = 0.5 + _ROBUST_K * _MAD_TO_STD / 4.0  # ~1.612 for k=3


def _hist_in_bin(b, n=200, nbins=250):
    """A length-`nbins` histogram with all `n` counts in bin `b`."""
    h = np.zeros(nbins, dtype=np.int64)
    h[b] = n
    return h


def test_threshold_is_median_plus_k_robust_sigma_on_a_spike():
    # Pure spike in bin 50 -> threshold = bins[50] + bw*(1/2 + k*1.4826/4).
    thr = _robust_threshold_from_hist(_hist_in_bin(50), BINS, BW)
    assert np.isclose(thr, BINS[50] + _SPIKE_MARGIN * BW)
    # strictly above the bin (it adds a positive k-sigma margin), never a clamp
    assert thr > BINS[50] + BW / 2.0


def test_threshold_matches_analytic_uniform():
    # Bin-aligned Uniform[10,20]: median=15, MAD=2.5 -> 15 + 3*1.4826*2.5 = 26.12.
    bins = np.arange(250) * 0.4          # left edges 0, 0.4, ... ; bin 25 = [10,10.4]
    h = np.zeros(250)
    h[25:50] = 1000.0                     # bins 25..49 cover [10.0, 20.0] exactly
    thr = _robust_threshold_from_hist(h, bins, 0.4)
    assert np.isclose(thr, 15.0 + _ROBUST_K * _MAD_TO_STD * 2.5, atol=1e-2)


def test_threshold_is_bin_size_independent():
    """An exact 2x bin refinement represents the SAME distribution, so the
    uniform-within-bin CDF — and the robust threshold — must be identical. (The
    old midpoint estimator snapped to bin centers and would shift by ~bw/2.)"""
    N, bwc = 250, 0.4
    coarse = np.arange(N) * bwc
    fine = np.arange(2 * N) * (bwc / 2.0)
    hc = np.zeros(N)
    hc[20:60] = 1000.0    # a uniform block
    hc[120] = 50000.0     # plus a spike
    hf = np.zeros(2 * N)  # split each coarse bin into 2 equal-density fine bins
    for i in range(N):
        hf[2 * i] = hc[i] / 2.0
        hf[2 * i + 1] = hc[i] / 2.0
    tc = _robust_threshold_from_hist(hc, coarse, bwc)
    tf = _robust_threshold_from_hist(hf, fine, bwc / 2.0)
    assert np.isclose(tc, tf, atol=1e-6)


def test_reliable_cluster_takes_its_own_robust_threshold():
    hists = {0: _hist_in_bin(50)[None, :]}
    raw, eff, _, _ = _compute_dual_thresholds(hists, hists, BINS, 1, 128, 1)
    expected = _robust_threshold_from_hist(_hist_in_bin(50), BINS, BW)
    assert np.isclose(raw[0, 0], expected)
    assert np.isclose(eff[0, 0], expected)


def test_bin0_eff_threshold_is_lifted_off_zero():
    """Raw p99 healthy (cell reliable) but the weighted histogram piles in bin 0.

    The robust eff threshold of a bin-0 spike is bins[0] + margin*bw > 0 — the
    estimator can never produce the zero threshold the old left-edge p99 did.
    """
    raw_hist = {0: _hist_in_bin(50)[None, :]}   # reliable (gate sees bin 50)
    eff_hist = {0: _hist_in_bin(0)[None, :]}    # weighted mass collapsed to bin 0
    raw, eff, _, _ = _compute_dual_thresholds(raw_hist, eff_hist, BINS, 1, 128, 1)
    assert eff[0, 0] > 0.0
    assert np.isclose(eff[0, 0], _SPIKE_MARGIN * BW)  # bins[0]=0


def test_reliability_gate_still_rejects_bins_0_and_1():
    """Gate must reject a cluster whose p99 is in bin 0/1 (too concentrated) and
    backfill it with a reliable sibling's robust threshold — same SET as before,
    just a robust fill value."""
    h = np.stack([_hist_in_bin(50), _hist_in_bin(1)])  # [2, nbins]
    raw, eff, _, _ = _compute_dual_thresholds({0: h}, {0: h}, BINS, 2, 128, 1)
    reliable_val = _robust_threshold_from_hist(_hist_in_bin(50), BINS, BW)
    assert np.isclose(raw[0, 0], reliable_val)
    assert np.isclose(raw[0, 1], reliable_val)  # bin-1 cluster took the backfill


def test_all_degenerate_element_backfills_to_populated_max():
    """Every cluster degenerate (p99 in bin 0): no reliable cluster, so backfill
    uses the populated max — which under the robust estimator is bins[0]+margin*bw,
    never 0."""
    h = np.stack([_hist_in_bin(0), _hist_in_bin(0)])
    raw, eff, _, _ = _compute_dual_thresholds({0: h}, {0: h}, BINS, 2, 128, 1)
    assert np.allclose(raw[0], _SPIKE_MARGIN * BW)
    assert np.all(raw[0] > 0.0)


def test_absent_element_row_defaults_to_one():
    hists = {0: _hist_in_bin(50)[None, :]}
    raw, eff, _, _ = _compute_dual_thresholds(hists, hists, BINS, 1, 128, 2)
    assert np.allclose(raw[1], 1.0)


def test_saturation_detection_handles_ceiling():
    # A threshold below the ceiling is not saturated; at/above bins[-1] it is.
    assert not _is_saturated(50.0, BINS)
    assert not _is_saturated(float(BINS[-1]) - 1.0, BINS)
    assert _is_saturated(float(BINS[-1]), BINS)
    assert _is_saturated(float(BINS[-1]) + BW, BINS)
