"""Tests for the ``grace_uq info`` artifact health diagnostics.

The checks read ONLY a ``meta`` dict (the shape ``read_artifact_metadata``
returns) — no model load, no GPU — so we build synthetic metas here and assert
each check fires (or stays silent) on the condition it targets. The headline is
``sigma_imbalance``: the GRACE-2L-SMAX over-lenient-gamma pathology where the
p99 threshold is set by a tiny outlier tail and the bulk reads gamma << 1.
"""
import numpy as np

from tensorpotential.uq import constants as uq_constants
from tensorpotential.uq.cli.diagnostics import diagnose, print_diagnostics


BINS = np.linspace(0.0, 100.0, 250)  # 250 left edges, as the build writes them
BW = float(BINS[1] - BINS[0])


def _hist_with_median_in_bin(b, n=10000, nbins=250):
    """Histogram with all `n` counts in bin `b` -> median and p99 both at bin b."""
    h = np.zeros(nbins, dtype=np.int64)
    h[b] = n
    return h


def _spike_tail_hist(spike_bin=0, tail_bin=60, spike=9990, tail=10, nbins=250):
    """Bulk piled in `spike_bin`, a thin tail in `tail_bin`.

    median sits in the spike bin while p99 (1% of mass) reaches the tail bin —
    the sigma-imbalance shape, where threshold >> median sigma.
    """
    h = np.zeros(nbins, dtype=np.int64)
    h[spike_bin] = spike
    h[tail_bin] = tail
    return h


def _make_meta(
    *,
    counts,          # list[int] per cluster (real clusters; sentinels appended)
    thr,             # list[float] per cluster threshold
    hist,            # list[np.ndarray] per cluster histogram
    D=128,
    n_sentinel=0,
    cond=None,
    rank=None,
    ntrunc=None,
    eff_count=None,  # list[float] per cluster effective (weighted) count
    element_map=("X",),
    bad_centroid=False,
):
    K = len(counts) + n_sentinel
    centroids = np.zeros((K, D), dtype=np.float64)
    for j in range(len(counts), K):
        centroids[j, :] = uq_constants.SENTINEL_CENTROID_VALUE \
            if hasattr(uq_constants, "SENTINEL_CENTROID_VALUE") else 1e10
    if bad_centroid:
        centroids[0, 0] = np.nan
    counts_arr = np.array(list(counts) + [0] * n_sentinel, dtype=np.int64)
    hist_arr = np.zeros((K, len(BINS)), dtype=np.int64)
    for k, hk in enumerate(hist):
        hist_arr[k] = hk
    thr_row = np.full(K, 1.0, dtype=np.float64)
    thr_row[: len(thr)] = thr
    art = {
        uq_constants.CENTROIDS: centroids,
        uq_constants.COUNTS: counts_arr,
        uq_constants.INV_COV: np.tile(np.eye(D), (K, 1, 1)),
    }
    if eff_count is not None:
        art[uq_constants.EFFECTIVE_COUNT] = np.array(
            list(eff_count) + [0.0] * n_sentinel, dtype=np.float64
        )
    if cond is not None:
        art[uq_constants.COND_NUMBER] = np.array(list(cond) + [1.0] * n_sentinel)
    if rank is not None:
        art[uq_constants.EFFECTIVE_RANK] = np.array(list(rank) + [D] * n_sentinel)
    if ntrunc is not None:
        art[uq_constants.N_TRUNCATED] = np.array(list(ntrunc) + [0] * n_sentinel)
    return {
        "D": D,
        "K_max": K,
        "n_elements": 1,
        "element_map": list(element_map),
        "artifacts": {0: art},
        "interp_thresholds": thr_row[None, :],
        "eff_interp_thresholds": None,
        "hist_bins": BINS,
        "hist_arrays": {0: hist_arr},
        "eff_hist_arrays": None,
    }


def _codes(meta):
    return {g.code: g for g in diagnose(meta)}


def test_sigma_imbalance_fires_on_spike_tail():
    # bulk median in bin 0 (~0.2), threshold set by the tail (p99 ~ bin 60).
    meta = _make_meta(
        counts=[10000],
        thr=[float(BINS[60] + BW / 2.0)],   # ~24.3
        hist=[_spike_tail_hist(spike_bin=0, tail_bin=60)],
    )
    g = _codes(meta)
    assert "sigma_imbalance" in g
    # one element flagged, typical gamma far below 1
    lbl, msg, val = g["sigma_imbalance"].items[0]
    assert val < 0.1
    assert "X(0)" == lbl
    # the message names the dominant cluster and its population
    assert "cluster 0" in msg
    assert "atoms" in msg


def test_sigma_imbalance_reports_effective_count_when_weighted():
    meta = _make_meta(
        counts=[10000],
        thr=[float(BINS[60] + BW / 2.0)],
        hist=[_spike_tail_hist(spike_bin=0, tail_bin=60)],
        eff_count=[2500.0],   # weighted build: eff != raw
    )
    msg = _codes(meta)["sigma_imbalance"].items[0][1]
    assert "eff. atoms" in msg
    assert "2.5k" in msg


def test_sigma_imbalance_omits_bin0_clause_when_bulk_above_bin0():
    # At-type case: bulk in bin ~5 (median ~2.2), nothing in bin 0, far tail.
    # "0% of sigma in bin 0" is noise and must not appear; the ceiling note may.
    meta = _make_meta(
        counts=[10000],
        thr=[float(BINS[150] + BW / 2.0)],   # ~60
        hist=[_spike_tail_hist(spike_bin=5, tail_bin=150, spike=9970, tail=30)],
    )
    g = _codes(meta)
    assert "sigma_imbalance" in g
    msg = g["sigma_imbalance"].items[0][1]
    assert "in bin 0" not in msg


def test_sigma_imbalance_keeps_bin0_clause_when_bulk_piles_there():
    meta = _make_meta(
        counts=[10000],
        thr=[float(BINS[60] + BW / 2.0)],
        hist=[_spike_tail_hist(spike_bin=0, tail_bin=60, spike=9990, tail=10)],
    )
    msg = _codes(meta)["sigma_imbalance"].items[0][1]
    assert "of sigma in bin 0" in msg


def test_sigma_imbalance_silent_on_healthy_cluster():
    # median sigma ~ p99 threshold -> typical gamma ~0.8, not flagged.
    h = np.zeros(len(BINS), dtype=np.int64)
    h[18:24] = 1000  # mass clustered near bins 18..23
    meta = _make_meta(
        counts=[6000],
        thr=[float(BINS[23] + BW / 2.0)],   # ~9.4 ; median bin ~20 -> ~8.2
        hist=[h],
    )
    assert "sigma_imbalance" not in _codes(meta)


def test_zero_threshold_is_fail():
    meta = _make_meta(
        counts=[10000],
        thr=[0.0],
        hist=[_hist_with_median_in_bin(50)],
    )
    g = _codes(meta)
    assert "zero_threshold" in g
    assert g["zero_threshold"].severity == "FAIL"
    assert "cluster 0" in g["zero_threshold"].items[0][1]  # names the cluster


def test_floored_threshold_warns():
    # p99 in bin 1 -> threshold ~bw*1.5 floor; no usable calibration.
    meta = _make_meta(
        counts=[10000],
        thr=[float(BINS[1] + BW / 2.0)],
        hist=[_hist_with_median_in_bin(1)],
    )
    g = _codes(meta)
    assert "floored_threshold" in g
    # names the cluster, its size and the offending threshold value
    msg = g["floored_threshold"].items[0][1]
    assert "cluster 0" in msg and "10.0k" in msg and "thr=" in msg


def test_dominant_cluster_collapse():
    h_big = _hist_with_median_in_bin(20, n=99900)
    h_small = _hist_with_median_in_bin(22, n=100)
    meta = _make_meta(
        counts=[99900, 100],
        thr=[float(BINS[23] + BW / 2.0), float(BINS[24] + BW / 2.0)],
        hist=[h_big, h_small],
    )
    g = _codes(meta)
    assert "dominant_cluster" in g
    assert g["dominant_cluster"].items[0][2] >= 0.95
    assert "cluster 0" in g["dominant_cluster"].items[0][1]


def test_underpopulated_cluster():
    # a real cluster with fewer than max(50, D)=128 atoms.
    meta = _make_meta(
        counts=[10000, 10],
        thr=[float(BINS[20] + BW / 2.0), float(BINS[20] + BW / 2.0)],
        hist=[_hist_with_median_in_bin(20), _hist_with_median_in_bin(20, n=10)],
    )
    g = _codes(meta)
    assert "underpopulated" in g
    # names the under-populated cluster (1) and its 10-atom size, not cluster 0
    msg = g["underpopulated"].items[0][1]
    assert "cluster 1 (10)" in msg


def test_ill_conditioned_and_rank_deficient():
    meta = _make_meta(
        counts=[10000],
        thr=[float(BINS[20] + BW / 2.0)],
        hist=[_hist_with_median_in_bin(20)],
        cond=[5e11],
        rank=[100],
        ntrunc=[28],
    )
    g = _codes(meta)
    assert "ill_conditioned" in g
    assert "rank_deficient" in g
    assert "cluster 0" in g["ill_conditioned"].items[0][1]
    assert "cond=" in g["ill_conditioned"].items[0][1]
    assert "rank=100/128" in g["rank_deficient"].items[0][1]


def test_saturated_threshold_info():
    meta = _make_meta(
        counts=[10000],
        thr=[float(BINS[-1])],   # at the ceiling
        hist=[_hist_with_median_in_bin(20)],
    )
    g = _codes(meta)
    assert "saturated_threshold" in g
    assert g["saturated_threshold"].severity == "INFO"
    assert "cluster 0" in g["saturated_threshold"].items[0][1]


def test_corruption_nan_centroid_is_fail():
    meta = _make_meta(
        counts=[10000],
        thr=[float(BINS[20] + BW / 2.0)],
        hist=[_hist_with_median_in_bin(20)],
        bad_centroid=True,
    )
    g = _codes(meta)
    assert "corruption" in g
    assert "cluster 0" in g["corruption"].items[0][1]
    assert g["corruption"].severity == "FAIL"


def test_diagnose_sorted_fail_first_and_print_smoke(capsys):
    # zero threshold (FAIL) + dominant collapse (WARN) -> FAIL sorts first.
    meta = _make_meta(
        counts=[99900, 100],
        thr=[0.0, float(BINS[20] + BW / 2.0)],
        hist=[_hist_with_median_in_bin(20, n=99900),
              _hist_with_median_in_bin(20, n=100)],
    )
    groups = diagnose(meta)
    assert groups[0].severity == "FAIL"
    print_diagnostics(meta)
    out = capsys.readouterr().out
    assert "DIAGNOSTICS" in out
    assert "fail" in out and "warn" in out


def test_clean_artifact_reports_no_problems(capsys):
    h = np.zeros(len(BINS), dtype=np.int64)
    h[18:24] = 1000
    meta = _make_meta(
        counts=[6000],
        thr=[float(BINS[23] + BW / 2.0)],
        hist=[h],
        cond=[1e3],
        rank=[128],
        ntrunc=[0],
    )
    assert diagnose(meta) == []
    print_diagnostics(meta)
    assert "no problems detected" in capsys.readouterr().out
