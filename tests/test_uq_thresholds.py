"""Regression tests for the per-cluster robust sigma threshold estimator.

The build threshold is ``median + k*(1.4826*MAD)`` (k=3), read off each cluster's
sigma histogram via a piecewise-linear (uniform-within-bin) CDF. This guards:
  - the value formula (median + k robust-sigma, not a tail quantile),
  - bin-size independence (the whole point of interpolating within the bin),
  - the reliability gate / backfill (p99-based, so the reliable SET does not
    depend on the estimator — only the cell VALUES are robust),
  - saturation detection in the info CLI,
  - the opt-in --threshold-percentile estimator, and that selecting it leaves
    the default path and the reliability gate untouched,
  - the log-spaced bin grid of new builds, and that older uniform-grid artifacts
    still read the same.
"""
import numpy as np

from tensorpotential.uq.cli.build.thresholds import (
    _MAD_TO_STD,
    _ROBUST_K,
    _compute_dual_thresholds,
    _percentile_threshold_from_hist,
    _robust_threshold_from_hist,
    narrow_sigma_clusters,
    print_narrow_sigma_warning,
    sigma_bin_index,
    sigma_hist_edges,
)
from tensorpotential.uq.cli.info import _is_saturated


# The uniform grid of older builds: 250 left edges as they stored them (hist_bins)
# and the edges read_artifact_metadata rebuilds from them. The estimators take any
# edges; new builds store the log-spaced sigma_hist_edges().
BINS = np.linspace(0.0, 100.0, 250)
BW = float(BINS[1] - BINS[0])
EDGES = np.append(BINS, BINS[-1] + BW)

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
    thr = _robust_threshold_from_hist(_hist_in_bin(50), EDGES)
    assert np.isclose(thr, BINS[50] + _SPIKE_MARGIN * BW)
    # strictly above the bin (it adds a positive k-sigma margin), never a clamp
    assert thr > BINS[50] + BW / 2.0


def test_threshold_matches_analytic_uniform():
    # Bin-aligned Uniform[10,20]: median=15, MAD=2.5 -> 15 + 3*1.4826*2.5 = 26.12.
    edges = np.arange(251) * 0.4         # edges 0, 0.4, ... ; bin 25 = [10,10.4]
    h = np.zeros(250)
    h[25:50] = 1000.0                     # bins 25..49 cover [10.0, 20.0] exactly
    thr = _robust_threshold_from_hist(h, edges)
    assert np.isclose(thr, 15.0 + _ROBUST_K * _MAD_TO_STD * 2.5, atol=1e-2)


def test_threshold_is_bin_size_independent():
    """An exact 2x bin refinement represents the SAME distribution, so the
    uniform-within-bin CDF — and the robust threshold — must be identical. (The
    old midpoint estimator snapped to bin centers and would shift by ~bw/2.)"""
    N, bwc = 250, 0.4
    coarse = np.arange(N + 1) * bwc
    fine = np.arange(2 * N + 1) * (bwc / 2.0)
    hc = np.zeros(N)
    hc[20:60] = 1000.0    # a uniform block
    hc[120] = 50000.0     # plus a spike
    hf = np.zeros(2 * N)  # split each coarse bin into 2 equal-density fine bins
    for i in range(N):
        hf[2 * i] = hc[i] / 2.0
        hf[2 * i + 1] = hc[i] / 2.0
    tc = _robust_threshold_from_hist(hc, coarse)
    tf = _robust_threshold_from_hist(hf, fine)
    assert np.isclose(tc, tf, atol=1e-6)


def test_reliable_cluster_takes_its_own_robust_threshold():
    hists = {0: _hist_in_bin(50)[None, :]}
    raw, eff, _, _ = _compute_dual_thresholds(hists, hists, EDGES, 1, 128, 1)
    expected = _robust_threshold_from_hist(_hist_in_bin(50), EDGES)
    assert np.isclose(raw[0, 0], expected)
    assert np.isclose(eff[0, 0], expected)


def test_bin0_eff_threshold_is_lifted_off_zero():
    """Raw p99 healthy (cell reliable) but the weighted histogram piles in bin 0.

    The robust eff threshold of a bin-0 spike is bins[0] + margin*bw > 0 — the
    estimator can never produce the zero threshold the old left-edge p99 did.
    """
    raw_hist = {0: _hist_in_bin(50)[None, :]}   # reliable (gate sees bin 50)
    eff_hist = {0: _hist_in_bin(0)[None, :]}    # weighted mass collapsed to bin 0
    raw, eff, _, _ = _compute_dual_thresholds(raw_hist, eff_hist, EDGES, 1, 128, 1)
    assert eff[0, 0] > 0.0
    assert np.isclose(eff[0, 0], _SPIKE_MARGIN * BW)  # bins[0]=0


def test_reliability_gate_still_rejects_bins_0_and_1():
    """Gate must reject a cluster whose p99 is below 0.8 (bin 0/1 of the old grid:
    too concentrated) and backfill it with a reliable sibling's robust threshold."""
    h = np.stack([_hist_in_bin(50), _hist_in_bin(1)])  # [2, nbins]
    raw, eff, _, _ = _compute_dual_thresholds({0: h}, {0: h}, EDGES, 2, 128, 1)
    reliable_val = _robust_threshold_from_hist(_hist_in_bin(50), EDGES)
    assert np.isclose(raw[0, 0], reliable_val)
    assert np.isclose(raw[0, 1], reliable_val)  # bin-1 cluster took the backfill


def test_all_degenerate_element_backfills_to_populated_max():
    """Every cluster degenerate (p99 in bin 0): no reliable cluster, so backfill
    uses the populated max — which under the robust estimator is bins[0]+margin*bw,
    never 0."""
    h = np.stack([_hist_in_bin(0), _hist_in_bin(0)])
    raw, eff, _, _ = _compute_dual_thresholds({0: h}, {0: h}, EDGES, 2, 128, 1)
    assert np.allclose(raw[0], _SPIKE_MARGIN * BW)
    assert np.all(raw[0] > 0.0)


def test_absent_element_row_defaults_to_one():
    hists = {0: _hist_in_bin(50)[None, :]}
    raw, eff, _, _ = _compute_dual_thresholds(hists, hists, EDGES, 1, 128, 2)
    assert np.allclose(raw[1], 1.0)


def test_saturation_detection_handles_ceiling():
    # A threshold below the ceiling is not saturated; at/above bins[-1] it is.
    assert not _is_saturated(50.0, BINS)
    assert not _is_saturated(float(BINS[-1]) - 1.0, BINS)
    assert _is_saturated(float(BINS[-1]), BINS)
    assert _is_saturated(float(BINS[-1]) + BW, BINS)


# ---------------------------------------------------------------------------
# --threshold-percentile: opt-in tail-quantile estimator
# ---------------------------------------------------------------------------


def _spike_plus_tail(nbins=250):
    """Sharp bulk plus a thin heavy tail — the shape that makes median+k*MAD
    flag a large slice of the TRAINING set (MAD sees only the spike)."""
    h = np.zeros(nbins, dtype=np.float64)
    h[2] = 10_000.0          # the bulk
    h[10:60] = 20.0          # 1000 counts smeared over a long tail
    return h


def test_percentile_threshold_matches_analytic_uniform():
    # Bin-aligned Uniform[10,20]: p99 = 10 + 0.99*10 = 19.9.
    edges = np.arange(251) * 0.4
    h = np.zeros(250)
    h[25:50] = 1000.0
    thr = _percentile_threshold_from_hist(h, edges, 99.0)
    assert np.isclose(thr, 19.9, atol=1e-2)


def test_percentile_threshold_is_bin_size_independent():
    """Same guarantee as the robust estimator: an exact 2x refinement describes
    the same distribution, so the quantile must not move."""
    N, bwc = 250, 0.4
    coarse = np.arange(N + 1) * bwc
    hc = _spike_plus_tail(N)
    fine = np.arange(2 * N + 1) * (bwc / 2)
    hf = np.repeat(hc, 2) / 2.0
    assert np.isclose(
        _percentile_threshold_from_hist(hc, coarse, 99.0),
        _percentile_threshold_from_hist(hf, fine, 99.0),
        atol=1e-9,
    )


def test_percentile_is_above_robust_on_a_heavy_tail():
    """The motivating case: the robust fence sits just above the bulk, so much of
    the training distribution reads gamma > 1. p99 rides the tail instead, which
    is exactly the trade --threshold-percentile makes."""
    h = _spike_plus_tail()
    robust = _robust_threshold_from_hist(h, EDGES)
    p99 = _percentile_threshold_from_hist(h, EDGES, 99.0)
    assert p99 > robust
    # by construction ~1% of the mass sits above the p99 threshold ...
    above_p99 = h[BINS + BW / 2.0 > p99].sum() / h.sum()
    assert above_p99 < 0.02
    # ... while the robust fence leaves far more of the TRAINING mass above 1
    above_robust = h[BINS + BW / 2.0 > robust].sum() / h.sum()
    assert above_robust > 5 * above_p99


def test_default_path_is_unchanged_by_the_new_parameter():
    """Omitting threshold_percentile and passing None explicitly must be the same
    computation — the option must not perturb existing artifacts."""
    h = _spike_plus_tail()
    hists = {0: np.stack([h, h * 2]), 1: np.stack([h * 3, np.zeros_like(h)])}
    default = _compute_dual_thresholds(hists, hists, EDGES, 2, 50, 2)
    explicit = _compute_dual_thresholds(
        hists, hists, EDGES, 2, 50, 2, threshold_percentile=None
    )
    for a, b in zip(default[:2], explicit[:2]):
        assert np.array_equal(a, b, equal_nan=True)
    assert default[2:] == explicit[2:]


def _cluster_hist(spike_bin, n=10_000.0, tail=None, tail_h=20.0, nbins=250):
    """Sharp bulk in `spike_bin`, optionally smeared with a thin heavy tail."""
    h = np.zeros(nbins, dtype=np.float64)
    h[spike_bin] = n
    if tail is not None:
        h[tail[0]:tail[1]] = tail_h
    return h


def _backfilled_mask(mat):
    """Cells that took the row fill (the element-wide max of reliable cells).

    `_backfill` leaves no NaN behind, so comparing NaN patterns compares two
    all-False arrays and pins nothing. Backfilled cells are exactly those equal
    to the row fill, which IS observable in the returned matrix — provided the
    fixture gives the reliable clusters DIFFERENT thresholds, otherwise every
    cell equals the max and the mask is all-True and equally useless.
    """
    return np.isclose(mat, np.nanmax(mat, axis=1, keepdims=True))


def test_percentile_changes_values_but_not_the_reliability_set():
    """Constraint: the gate stays on p99 whichever estimator runs,
    so the reliable-cluster SET and every backfill position are identical.

    Fixture: two well-populated clusters with distinct thresholds plus one
    under-populated cluster that must backfill — so the mask is [F, T, T] and
    actually discriminates.
    """
    hists = {
        0: np.stack([
            _cluster_hist(2, tail=(10, 60)),    # reliable, low threshold
            _cluster_hist(30, tail=(40, 90)),   # reliable, high threshold
            _cluster_hist(2, n=10.0),           # 10 atoms < min_atoms=50 -> backfilled
        ])
    }
    robust = _compute_dual_thresholds(hists, hists, EDGES, 3, 50, 1)
    pctl = _compute_dual_thresholds(
        hists, hists, EDGES, 3, 50, 1, threshold_percentile=99.0
    )

    # the fixture must actually exercise backfill AND discriminate, or this
    # test would pass no matter what the gate did
    assert robust[2] == [(0, 2, 10)], robust[2]
    assert not _backfilled_mask(robust[0]).all()

    assert not np.allclose(robust[0], pctl[0])                 # values move ...
    for a, b in zip(robust[:2], pctl[:2]):                     # ... positions do not
        assert np.array_equal(_backfilled_mask(a), _backfilled_mask(b))
    # underpop warnings carry (elem, cluster, n_calib) — counts, so estimator-free.
    # elementwise_fallback_warnings carry a fill VALUE and legitimately differ.
    assert robust[2] == pctl[2]


def test_backfill_mask_fixture_detects_a_gate_change():
    """Guard on the guard: if the reliability gate moved, the mask above must
    change — otherwise the invariant test is decorative."""
    hists = {
        0: np.stack([
            _cluster_hist(2, tail=(10, 60)),
            _cluster_hist(30, tail=(40, 90)),
            _cluster_hist(2, n=10.0),
        ])
    }
    base = _compute_dual_thresholds(hists, hists, EDGES, 3, 50, 1)
    stricter = _compute_dual_thresholds(hists, hists, EDGES, 3, 50_000, 1)
    assert not np.array_equal(
        _backfilled_mask(base[0]), _backfilled_mask(stricter[0])
    )


def test_percentile_rejects_out_of_range_q():
    """Q outside (0, 100] gives a degenerate threshold (0 -> gamma=inf, >100 ->
    gamma~0), so it must raise rather than silently produce a broken artifact."""
    h = _spike_plus_tail()
    for bad in (0.0, -1.0, 100.5, float("nan"), float("inf")):
        try:
            _percentile_threshold_from_hist(h, EDGES, bad)
        except ValueError:
            continue
        raise AssertionError(f"percentile {bad} was accepted")


# ---------------------------------------------------------------------------
# Bin grid shared by the step-3 workers (binning) and the master (estimator)
# ---------------------------------------------------------------------------


def _exact_threshold(s):
    med = np.median(s)
    return med + _ROBUST_K * _MAD_TO_STD * np.median(np.abs(s - med))


def _binned_threshold(s):
    h = np.bincount(sigma_bin_index(s), minlength=len(sigma_hist_edges()) - 1)
    return _robust_threshold_from_hist(h, sigma_hist_edges())


def test_every_sigma_lies_in_the_bin_the_edges_give():
    """The workers' bin index and the master's edges describe one grid; sigma
    outside the grid is clipped into the first/last bin."""
    edges = sigma_hist_edges()
    s = np.geomspace(edges[0], edges[-1], 200_001)[:-1]
    idx = sigma_bin_index(s)
    assert np.all(edges[idx] <= s) and np.all(s < edges[idx + 1])
    assert sigma_bin_index([0.0, 1e-6])[0] == 0
    assert sigma_bin_index([1e6])[0] == len(edges) - 2


def test_log_grid_has_constant_relative_width():
    edges = sigma_hist_edges()
    rel = edges[1:] / edges[:-1] - 1.0
    assert np.allclose(rel, rel[0]) and rel[0] < 0.012


def test_worker_binned_threshold_matches_exact_on_a_broad_distribution():
    rng = np.random.default_rng(0)
    s = rng.normal(50.0, 6.0, 400_000)
    assert abs(_binned_threshold(s) / _exact_threshold(s) - 1.0) < 1e-3


def test_narrow_cluster_threshold_is_accurate_on_the_log_grid():
    """A narrow cluster at small sigma (MAD ~ 0.2 of the old 0.4 bin width) was
    tens of percent off on the uniform grid; the log grid resolves it."""
    rng = np.random.default_rng(1)
    s = rng.normal(2.0, 0.12, 20_000)
    exact = _exact_threshold(s)
    assert abs(_binned_threshold(s) / exact - 1.0) < 0.01
    old = np.clip((s / 0.4).astype(int), 0, 249)
    h_old = np.bincount(old, minlength=250)
    edges_old = np.arange(251) * 0.4
    assert abs(_robust_threshold_from_hist(h_old, edges_old) / exact - 1.0) > 0.05


def test_histogram_from_another_grid_is_refused():
    """Step-3 worker files of an older build (250 uniform bins) must not be read
    against the log grid."""
    hists = {0: _hist_in_bin(50)[None, :]}
    try:
        _compute_dual_thresholds(hists, hists, sigma_hist_edges(), 1, 128, 1)
    except ValueError as err:
        assert "250 bins" in str(err)
    else:
        raise AssertionError("a 250-bin histogram was accepted on the log grid")


def test_metadata_reads_uniform_and_log_grids(tmp_path):
    """An older artifact stores only the left edges of its uniform grid (hist_bins);
    a new one stores hist_bin_edges. Both give edges and left edges."""
    from tensorpotential.uq.cli._common import read_artifact_metadata
    from tests.test_uq_select_candidates import _write_artifact

    old = _write_artifact(tmp_path / "old.npz", store_fp32=True)
    with np.load(old) as d:
        data = dict(d)
    np.savez(old, **data, hist_bins=BINS)
    meta = read_artifact_metadata(old)
    np.testing.assert_allclose(meta["hist_bin_edges"], EDGES)
    np.testing.assert_allclose(meta["hist_bins"], BINS)

    new = str(tmp_path / "new.npz")
    np.savez(new, **data, hist_bin_edges=sigma_hist_edges())
    meta = read_artifact_metadata(new)
    np.testing.assert_allclose(meta["hist_bin_edges"], sigma_hist_edges())
    np.testing.assert_allclose(meta["hist_bins"], sigma_hist_edges()[:-1])


def _log_hist(s):
    return np.bincount(sigma_bin_index(s), minlength=len(sigma_hist_edges()) - 1)


def test_narrow_sigma_clusters_flags_mad_below_bin_width():
    """Build-time warning: clusters with enough atoms whose sigma MAD is below
    ~1.4 widths of the bin at their median (the uniform-within-bin CDF then
    overestimates the threshold)."""
    edges = sigma_hist_edges()
    i = int(sigma_bin_index([8.4])[0])
    narrow = np.zeros(len(edges) - 1)
    narrow[i:i + 2] = 3000.0                 # MAD = half a bin width
    broad = _log_hist(np.random.default_rng(0).normal(8.0, 2.0, 6000)).astype(float)
    small = np.zeros(len(edges) - 1)
    small[i:i + 2] = 5.0                     # narrow, but only 10 atoms
    hists = {0: np.stack([narrow, broad]), 1: small[None, :]}
    found = narrow_sigma_clusters(hists, hists, edges, min_atoms=50)
    assert [(e, k) for e, k, _med, _mad in found] == [(0, 0)]
    _e, _k, med, mad = found[0]
    assert np.isclose(med, edges[i + 1]) and np.isclose(mad, (edges[i + 1] - edges[i]) / 2, rtol=0.02)


def test_narrow_sigma_clusters_skips_floored_clusters():
    """p99 below 0.8 is backfilled by the reliability gate; not this warning."""
    edges = sigma_hist_edges()
    i = int(sigma_bin_index([0.3])[0])
    h = np.zeros(len(edges) - 1)
    h[i:i + 2] = 3000.0
    hists = {0: h[None, :]}
    assert narrow_sigma_clusters(hists, hists, edges, min_atoms=50) == []


def test_build_warns_about_narrow_clusters_and_names_them(capsys):
    print_narrow_sigma_warning([(0, 1, 1.37, 0.117)], ["Al"], n_clusters=4)
    out = capsys.readouterr().out
    assert "WARNING" in out
    assert "Al" in out and "cluster 1" in out and "MAD=0.117" in out
    assert "--n-clusters" in out          # the advice applies: K > 1
    assert "--threshold-percentile" in out  # a tail quantile is bin-robust


def test_build_narrow_warning_is_silent_without_findings(capsys):
    print_narrow_sigma_warning([], ["Al"], n_clusters=4)
    assert capsys.readouterr().out == ""


def test_narrow_sigma_clusters_flags_spread_up_to_1_4_bin_widths():
    """Tight peaks split across neighbouring bins read a binned MAD above one bin
    width although the true MAD is far smaller (old grid, tutorial K=2, Al cluster
    0: binned MAD 1.23 bw, true 0.20 bw, threshold +123 %). The cut is 1.4 bin
    widths; uniform over 5 bins of the old grid gives MAD = 1.25 bw."""
    h = np.zeros(250)
    h[20:25] = 1000.0
    hists = {0: h[None, :]}
    found = narrow_sigma_clusters(hists, hists, EDGES, min_atoms=50)
    assert [(e, k) for e, k, _m, _d in found] == [(0, 0)]
