"""Per-cluster robust (median + k*MAD) sigma thresholds, elbow selection,
covariance diagnostics."""

from __future__ import annotations

import logging

import numpy as np

from tensorpotential.uq import constants as uq_constants
from tensorpotential.uq.cli._common import format_elem_label, is_sentinel_mask

log = logging.getLogger("grace_uq")

def _normalize_inertia(values):
    """Normalize inertia by the maximum value → range (0, 1]."""
    ys = np.asarray(values, dtype=float)
    return ys / (ys[0] + 1e-12)


# ---------------------------------------------------------------------------
# Robust per-cluster sigma threshold:  median + k * (1.4826 * MAD)
# ---------------------------------------------------------------------------
# gamma = sigma / threshold is normalized per cluster by a threshold read off
# that cluster's training-sigma histogram. We deliberately do NOT use a tail
# quantile (e.g. p99) for that threshold, because the per-cluster sigma
# distribution is, in practice, a spike + heavy tail:
#
#   A clean, correctly-scaled Gaussian feature model gives sigma ~ chi(D)
#   (median ~ sqrt(D) ~ 11.3 for D=128). But a handful of outlier environments
#   (clashing / high-force atoms, and deviations along rank-deficient null
#   directions amplified by covariance regularization) inflate the cluster
#   covariance, so the bulk collapses to sigma ~ 0.2 while a thin tail stretches
#   to 100. A p99 anchor then RIDES that tail: the threshold is set by the
#   outliers, the bulk reads gamma ~ 0.01, and ordinary atoms can never approach
#   gamma = 1 (over-lenient UQ — the GRACE-2L-SMAX blow-up).
#
# A robust location+scale anchor is immune to that tail. The MEDIAN (50%
# breakdown point) ignores the outliers; the MAD (median absolute deviation,
# also 50% breakdown) gives a tail-immune scale. We scale MAD by 1.4826 so it
# estimates the standard deviation for a normal (MAD_normal = sigma / 1.4826),
# then add k of them:
#
#       threshold = median + k * (1.4826 * MAD)        # == median + k robust std
#
# Reading of k (since 1.4826*MAD is one robust standard deviation):
#   - For a normal — and chi(D) is ~normal for large D — this is a k-sigma fence
#     with one-sided coverage Phi(k):  k=2 -> 97.7%,  k=3 -> 99.87% (~p99.9),
#     k=4 -> 99.997%.
#   - On a CLEAN chi(128) cluster, median+3*MAD lands within a few percent of
#     p99 (bulk gamma ~0.84). So it is a safe drop-in that loses nothing on
#     well-behaved clusters; it only DIVERGES from p99 on heavy-tailed clusters,
#     where it refuses to let the tail set the boundary. That divergence is the
#     entire point.
#
# k is HARDCODED to 3 (the conventional 3-sigma outlier fence). It is not a user
# knob: changing it rescales every gamma and breaks comparability across already
# deployed models. Re-validate (gamma vs force-error) before ever touching it.
_ROBUST_K: float = 3.0
_MAD_TO_STD: float = 1.4826  # 1 / Phi^-1(0.75): MAD -> std for a normal distribution


def _hist_cdf_parts(h):
    """Precompute (h, total, exclusive-prefix-sums) for the piecewise-linear CDF."""
    h = np.asarray(h, dtype=np.float64)
    s = float(h.sum())
    c_before = np.cumsum(h) - h
    return h, s, c_before


def _interp_quantile(h, s, c_before, bins, bw, q):
    """Quantile ``q`` of the piecewise-linear CDF implied by the histogram (mass
    assumed uniform within each bin). Interpolating WITHIN the bin makes the
    estimate independent of the bin grid: splitting a bin into equal-density
    sub-bins leaves this CDF — hence the quantile — unchanged. (The legacy
    estimator snapped to a bin edge/midpoint and was bin-size-bound.)"""
    target = q * s
    cum_upper = c_before + h
    i = int(np.searchsorted(cum_upper, target, side="left"))
    i = min(i, len(h) - 1)
    if h[i] <= 0:
        return float(bins[i])
    frac = (target - c_before[i]) / h[i]
    return float(bins[i] + bw * min(max(frac, 0.0), 1.0))


def _cdf_at(x, h, s, c_before, bins, bw):
    """Value of the piecewise-linear (uniform-within-bin) CDF at ``x``."""
    if x <= bins[0]:
        return 0.0
    if x >= bins[-1] + bw:
        return 1.0
    i = int((x - bins[0]) / bw)
    i = min(max(i, 0), len(h) - 1)
    frac = min(max((x - bins[i]) / bw, 0.0), 1.0)
    return (c_before[i] + frac * h[i]) / s


def _mad_from_hist(h, s, c_before, bins, bw, median):
    """median(|x - median|) via the folded CDF ``G(t) = F(median+t) - F(median-t)``,
    solved for ``G(t) = 0.5`` by bisection. Uses the same bin-size-proof
    piecewise-linear ``F`` as the median, so the MAD is also grid-independent.
    (The one irreducible case is a pure single-bin spike, where the uniform
    within-bin assumption yields MAD = bw/4 — but for sigma the resulting GAMMA
    is still bin-size-invariant, since median and MAD both scale with bw and
    cancel in the ratio.)"""
    lo, hi = 0.0, max(median - bins[0], (bins[-1] + bw) - median)
    for _ in range(60):  # 60 bisections -> width far below machine eps * range
        t = 0.5 * (lo + hi)
        g = (_cdf_at(median + t, h, s, c_before, bins, bw)
             - _cdf_at(median - t, h, s, c_before, bins, bw))
        if g < 0.5:
            lo = t
        else:
            hi = t
    return 0.5 * (lo + hi)


def _robust_threshold_from_hist(h, bins, bw):
    """``median + _ROBUST_K * (1.4826 * MAD)`` from a sigma histogram (NaN if empty).

    This is the per-cluster threshold value. See the block comment above for the
    rationale and the meaning of k.
    """
    h, s, c_before = _hist_cdf_parts(h)
    if s <= 0:
        return np.nan
    median = _interp_quantile(h, s, c_before, bins, bw, 0.5)
    sigma_hat = _MAD_TO_STD * _mad_from_hist(h, s, c_before, bins, bw, median)
    return median + _ROBUST_K * sigma_hat


def _p99_midpoint_from_hist(h, bins, bw):
    """p99 as the MIDPOINT of the bin holding it.

    NOT the threshold anymore — used ONLY for the reliability GATE in
    ``_compute_dual_thresholds``, so the reliable-cluster SET (and every backfill
    decision) stays byte-identical to the validated midpoint-era behavior. A
    cluster whose p99 sits in bin 0/1 (<= 1.5*bw) is too concentrated for its own
    histogram to define a trustworthy spread, so it borrows a sibling's threshold
    via backfill instead of using its own median+k*MAD value.
    """
    h = np.asarray(h, dtype=np.float64)
    s = float(h.sum())
    if s <= 0:
        return np.nan
    idx = int(np.searchsorted(np.cumsum(h) / s, 0.99))
    idx = min(idx, len(h) - 1)
    return float(bins[idx] + bw / 2.0)


def _compute_dual_thresholds(
    total_hists: dict,
    total_eff_hists: dict,
    bins: np.ndarray,
    n_clusters: int,
    min_atoms_for_threshold: int,
    n_elements: int,
):
    """Compute raw + effective per-cluster sigma threshold matrices in lockstep.

    The threshold VALUE is the robust ``median + k*(1.4826*MAD)`` estimator (see
    the block comment above ``_ROBUST_K``), NOT a tail quantile — so a heavy
    outlier tail can't set the boundary and leave the bulk over-lenient.

    Both matrices share reliability mask and backfill positions: the raw atom
    count drives the ``min_atoms_for_threshold`` gate (statistical confidence is
    about sample size, not weight), while each matrix's cells use its own
    histogram's robust threshold. Unreliable cells in *each* matrix are
    backfilled with that matrix's own element-wide max of reliable cells.

    Returns matrices of shape ``[n_elements, n_clusters]`` indexed by element
    idx (NOT dense over present elements). Inference (``tf.gather_nd`` in
    gmmuq) and the info CLI both look up rows by element idx, so elements
    missing from the calibration histograms (e.g. filtered out at build time)
    get a default-1.0 row instead of silently displacing the rows of higher-
    indexed elements.

    Returns
    -------
    thresh_matrix : np.ndarray [n_elements, n_clusters]
        Raw robust thresholds (default 1.0 for elements with no histogram).
    eff_thresh_matrix : np.ndarray [n_elements, n_clusters]
        Effective (weighted) robust thresholds (used by inference under weighting).
    underpop_warnings : list of (elem, cluster, n_calib) for clusters below the floor
    elementwise_fallback_warnings : list of (elem, n_pop, fill) for elements with no reliable cluster

    A cell is *reliable* iff its raw count ≥ ``min_atoms_for_threshold`` and its
    p99 clears bin 0/1 (the gate still uses p99 so the reliable SET is identical
    to the validated baseline). Non-reliable cells are backfilled with the
    element-wide max of reliable cells, falling back to the element-wide max of
    any populated cells, and finally to 1.0 when the row is entirely empty (the
    case for elements absent from the calibration histograms).
    """
    bin_width = bins[1] - bins[0]
    underpop_warnings: list = []

    raw_thresh = np.full((n_elements, n_clusters), np.nan, dtype=np.float64)
    eff_thresh = np.full((n_elements, n_clusters), np.nan, dtype=np.float64)
    reliable_mask = np.zeros((n_elements, n_clusters), dtype=bool)

    for e, h_raw_full in total_hists.items():
        h_eff_full = total_eff_hists.get(e)
        for k in range(n_clusters):
            h_raw = h_raw_full[k]
            n_calib = int(h_raw.sum())
            # Threshold VALUE: robust median + k*MAD on each histogram (see the
            # block comment above _ROBUST_K). The effective (weighted) histogram
            # drives the threshold inference uses when weights are non-trivial.
            raw_thresh[e, k] = _robust_threshold_from_hist(h_raw, bins, bin_width)
            eff_thresh[e, k] = (
                _robust_threshold_from_hist(h_eff_full[k], bins, bin_width)
                if h_eff_full is not None
                else np.nan
            )
            # Reliability GATE (deliberately unchanged from the midpoint era):
            # raw-sample confidence AND not-too-concentrated. The gate still uses
            # the p99 midpoint so the reliable SET — and thus every backfill
            # decision — is byte-identical to the validated baseline; only the
            # cell VALUES above switched to the robust estimator.
            gate_p99 = _p99_midpoint_from_hist(h_raw, bins, bin_width)
            if n_calib <= 0:
                rel = False
            elif gate_p99 <= 1.5 * bin_width:
                # p99 in bin 0/1 -> too concentrated to define its own spread;
                # this cluster backfills from a reliable sibling instead.
                rel = False
            else:
                rel = n_calib >= min_atoms_for_threshold
                if not rel:
                    underpop_warnings.append((int(e), k, n_calib))
            reliable_mask[e, k] = rel

    def _backfill(mat):
        out = mat.copy()
        warns = []
        for e in range(out.shape[0]):
            row = out[e]
            row_reliable = reliable_mask[e]
            row_populated = ~np.isnan(row)
            if row_reliable.any():
                fill = float(np.max(row[row_reliable]))
            elif row_populated.any():
                fill = float(np.max(row[row_populated]))
                warns.append((e, int(row_populated.sum()), fill))
            else:
                fill = 1.0
            out[e] = np.where(row_reliable, row, fill)
            out[e] = np.where(np.isnan(out[e]), fill, out[e])
        return out, warns

    thresh_matrix, ew_warns_raw = _backfill(raw_thresh)
    eff_thresh_matrix, _ = _backfill(eff_thresh)
    return thresh_matrix, eff_thresh_matrix, underpop_warnings, ew_warns_raw


def select_optimal_clusters(k_values: list, inertias: list) -> int:
    """Return the elbow k via maximum perpendicular distance to the chord.

    Scale- and shift-invariant: result is unchanged by any affine rescaling of
    the inertia values.

    A virtual flat-plateau point is appended beyond the last candidate so that
    the largest k can also be selected (without this, endpoints always have
    zero distance to the chord and can never win).
    """
    if len(k_values) <= 2:
        return int(k_values[0])
    y = np.asarray(inertias, dtype=float)
    if y[0] - y[-1] < 1e-12 * max(abs(y[0]), 1.0):
        log.warning("Elbow method: inertias are nearly constant; defaulting to first k")
        return int(k_values[0])
    k = np.array(k_values, dtype=float)

    # Append a virtual point that assumes inertia plateaus beyond the last
    # candidate.  This turns the last real candidate into an interior point.
    k_ext = np.append(k, k[-1] + (k[-1] - k[0]))
    y_ext = np.append(y, y[-1])

    k_norm = (k_ext - k_ext[0]) / (k_ext[-1] - k_ext[0])
    # Map y to [0, 1] for the chord geometry (independent of display normalization).
    y_01 = (y_ext - y_ext[-1]) / (y_ext[0] - y_ext[-1] + 1e-12)
    dists = np.abs(k_norm + y_01 - 1) / np.sqrt(2)

    # Only consider the original candidates (exclude the virtual point).
    real_dists = dists[: len(k_values)]
    # Deterministic tiebreaker: when several k candidates are within
    # ``tie_tol`` of the max distance, pick the smallest k (Occam's razor —
    # the cheapest model that hits the elbow). Log when this happens so a
    # close call is visible in the build report instead of being hidden by
    # argmax's first-wins behavior.
    tie_tol = 1e-3
    d_max = float(real_dists.max())
    tied_idx = np.flatnonzero(real_dists >= d_max - tie_tol)
    chosen_idx = int(tied_idx[0])
    if tied_idx.size > 1:
        tied_ks = [int(k_values[i]) for i in tied_idx]
        log.info(
            "Elbow tiebreak: %d candidates within tol=%.0e of d_max=%.4f "
            "(k=%s); picking smallest k=%d.",
            tied_idx.size, tie_tol, d_max, tied_ks, int(k_values[chosen_idx]),
        )
    return int(k_values[chosen_idx])


def _save_elbow_report(
    k_values,
    per_elem_inertias,
    per_elem_optimal_k,
    global_optimal_k,
    output_path,
    element_names=None,
):
    """Save a text report summarising the elbow analysis for every element."""
    report_path = output_path.replace(".npz", "_elbow.txt")
    k_header = "".join(f"{'k=' + str(k):>14s}" for k in k_values)

    lines = []
    lines.append("GRACE-UQ  Elbow Cluster Selection Report")
    lines.append("=" * 55)
    lines.append(f"Candidate k values: {k_values}")
    lines.append("Method: max perpendicular distance to chord (scale-invariant)")
    lines.append("")

    # Per-sample inertia table
    lines.append("Per-sample inertia (lower is better):")
    lines.append(f"{'Element':>14s}{k_header}  {'optimal_k':>10s}")
    lines.append("-" * (14 + 14 * len(k_values) + 12))

    for e in sorted(per_elem_inertias.keys()):
        lbl = format_elem_label(e, element_names)
        row = "".join(f"{v:14.6e}" for v in per_elem_inertias[e])
        lines.append(f"{lbl:>14s}{row}  {per_elem_optimal_k[e]:>10d}")

    lines.append("-" * (14 + 14 * len(k_values) + 12))
    lines.append(f"{'GLOBAL':>14s}{'':>{14 * len(k_values)}s}  {global_optimal_k:>10d}")
    lines.append("")
    optima_str = ", ".join(
        f"{format_elem_label(e, element_names)}={k}"
        for e, k in sorted(per_elem_optimal_k.items())
    )
    lines.append(
        f"Selected k = {global_optimal_k}  (max over per-element optima: {optima_str})"
    )
    lines.append("")

    # Normalised inertia table (for reference)
    lines.append("Normalised inertia (fraction of k=1 inertia, lower is better):")
    lines.append(f"{'Element':>14s}{k_header}")
    lines.append("-" * (14 + 14 * len(k_values)))
    for e in sorted(per_elem_inertias.keys()):
        lbl = format_elem_label(e, element_names)
        row = "".join(f"{v:14.6f}" for v in _normalize_inertia(per_elem_inertias[e]))
        lines.append(f"{lbl:>14s}{row}")

    text = "\n".join(lines) + "\n"
    with open(report_path, "w") as f:
        f.write(text)
    print(f"  [Master] Elbow report saved to {report_path}")


def _save_elbow_plot(
    k_values,
    per_elem_inertias,
    per_elem_optimal_k,
    global_optimal_k,
    output_path,
    element_names=None,
):
    """Save a PNG showing per-element normalized inertia curves with elbow markers."""
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        log.warning("matplotlib not installed — skipping elbow plot")
        return

    fig, ax = plt.subplots(figsize=(8, 5))
    colors = plt.cm.tab10.colors
    for idx, e in enumerate(sorted(per_elem_inertias.keys())):
        ys_norm = _normalize_inertia(per_elem_inertias[e])
        color = colors[idx % len(colors)]
        lbl = format_elem_label(e, element_names)
        ax.plot(k_values, ys_norm, marker="o", color=color, label=lbl)
        ax.axvline(per_elem_optimal_k[e], color=color, linestyle=":", alpha=0.6)
    ax.axvline(
        global_optimal_k,
        color="black",
        linestyle="--",
        linewidth=2,
        label=f"selected k={global_optimal_k}",
    )
    ax.set_xlabel("n_clusters (k)")
    ax.set_ylabel("normalized inertia")
    ax.set_title("Elbow method: per-element inertia")
    ax.set_xticks(k_values)
    ax.legend()
    plot_path = output_path.replace(".npz", "_elbow.png")
    fig.savefig(plot_path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"  [Master] Elbow plot saved to {plot_path}")


def _print_covariance_diagnostics(artifacts, element_names=None):
    """Print a per-element covariance health summary after Step 2 finalize.

    Sentinel-padded cluster slots are excluded from the per-element min/max
    so the diagnostics describe only the data-fitted clusters.
    """
    print("\n  [Master] Covariance diagnostics:")
    _COND_WARN_THRESHOLD = 1e10
    any_bad = False
    for elem in sorted(artifacts.keys()):
        data = artifacts[elem]
        cond = data.get(uq_constants.COND_NUMBER)
        rank = data.get(uq_constants.EFFECTIVE_RANK)
        n_trunc = data.get(uq_constants.N_TRUNCATED)
        centroids = data.get(uq_constants.CENTROIDS)
        counts = data.get(uq_constants.COUNTS)
        eff_counts = data.get(uq_constants.EFFECTIVE_COUNT)
        if cond is None or centroids is None:
            continue
        lbl = format_elem_label(elem, element_names)
        K = len(cond)
        D = data[uq_constants.INV_COV].shape[1]
        sentinel_mask = is_sentinel_mask(centroids)
        real_idx = np.flatnonzero(~sentinel_mask)
        K_eff = int(real_idx.size)
        sent_note = f" ({K - K_eff} sentinel)" if K_eff < K else ""
        if K_eff == 0:
            print(f"    {lbl:>10s}: no real clusters (all sentinels)")
            continue
        max_cond = float(np.max(cond[real_idx]))
        min_rank = int(np.min(rank[real_idx]))
        max_trunc = int(np.max(n_trunc[real_idx]))
        n_bad = int(np.sum(cond[real_idx] > _COND_WARN_THRESHOLD))
        status = (
            "OK"
            if n_bad == 0
            else f"WARNING: {n_bad}/{K_eff} clusters above {_COND_WARN_THRESHOLD:.0e}"
        )
        # Show effective vs raw counts when they diverge (i.e. weighted build).
        count_note = ""
        if counts is not None and eff_counts is not None:
            raw_sum = int(counts[real_idx].sum())
            eff_sum = float(eff_counts[real_idx].sum())
            if abs(eff_sum - raw_sum) > 0.5:
                count_note = f", raw_atoms={raw_sum}, eff_total={eff_sum:.1f}"
        print(
            f"    {lbl:>10s}: K_eff={K_eff}/{K}, max_cond={max_cond:.2e}, "
            f"min_rank={min_rank}/{D}, max_truncated={max_trunc}{count_note}  "
            f"[{status}]{sent_note}"
        )
        if n_bad > 0:
            any_bad = True
            bad_clusters = real_idx[cond[real_idx] > _COND_WARN_THRESHOLD]
            for k in bad_clusters:
                print(
                    f"              cluster {k}: cond={cond[k]:.2e}, "
                    f"rank={rank[k]}/{D}, truncated={n_trunc[k]}"
                )
    if any_bad:
        print(
            "\n  [Master] WARNING: Some clusters have high condition numbers.\n"
            "  This may produce unreliable uncertainty estimates for atoms\n"
            "  assigned to those clusters. Consider increasing regularization\n"
            "  or collecting more training data for the affected elements."
        )
    print()
