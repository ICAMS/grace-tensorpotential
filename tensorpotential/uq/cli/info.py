"""Info subcommand: print a human-readable summary of a UQ .npz artifact."""

from __future__ import annotations

import argparse
import os

import numpy as np

from tensorpotential.uq import constants as uq_constants
from tensorpotential.uq.cli._common import (
    format_elem_label,
    is_sentinel_mask,
    read_artifact_metadata,
)
from tensorpotential.uq.cli.diagnostics import print_diagnostics


_COND_WARN_THRESHOLD = 1e10


def _is_saturated(threshold: float, bins) -> bool:
    """True iff `threshold` is at the histogram ceiling (it was clipped there).

    A threshold that lands in the last bin takes that bin's representative
    value: its left edge ``bins[-1]`` (legacy left-edge estimator) or
    ``bins[-1] + bin_width/2`` (midpoint / robust median+MAD estimators). Both
    sit at/above ``bins[-1]``, while every non-saturated threshold is the
    midpoint of an earlier bin and stays strictly below ``bins[-1]``. Testing
    ``>=`` therefore flags saturation for artifacts written by any estimator.
    """
    if bins is None or threshold is None or not np.isfinite(threshold):
        return False
    return bool(float(threshold) >= float(bins[-1]) - 1e-6)


def _effective_quantile_label(hist_row, bins) -> str:
    """Quantile that `bins[-1]` actually represents in this cluster's histogram.

    Returns ``'p{XX.X}'`` when the cluster has data with mass in the saturating
    last bin (the threshold being capped at the ceiling is data-driven), or
    ``'bkfill'`` when the cluster's own histogram has no last-bin mass — in
    that case the displayed ceiling came from element-wide backfill of an
    unreliable cluster, not from this cluster's tail.
    """
    if hist_row is None or bins is None:
        return ""
    s = float(np.asarray(hist_row).sum())
    if s <= 0:
        return "bkfill"
    last_frac = float(hist_row[-1]) / s
    if last_frac <= 0:
        return "bkfill"
    return f"p{(1.0 - last_frac) * 100.0:.1f}"


def _count_saturated(thr_row, bins) -> int:
    """Count clusters in `thr_row` whose threshold is at the histogram ceiling."""
    if thr_row is None or bins is None:
        return 0
    return int(np.sum(np.asarray(thr_row, dtype=float) >= float(bins[-1]) - 1e-6))


def _format_thr_cell(value, hist_row, bins) -> str:
    """Format a single (threshold, hist) cell for the verbose per-cluster table.

    When the threshold equals the histogram ceiling (e.g. 100), the cell shows
    ``ceil* p{q}`` where ``p{q}`` is the actual quantile that the ceiling
    represents in this cluster's histogram (or ``bkfill`` when the value came
    from element-wide backfill rather than the cluster's own tail). Total width
    is 12 characters to match the existing column.
    """
    if value is None:
        return f"{'-':>12s}"
    if _is_saturated(value, bins):
        q_lbl = _effective_quantile_label(hist_row, bins)
        cell = f"{float(value):.3g}*"
        if q_lbl:
            cell += f" {q_lbl}"
        return f"{cell:>12s}"
    return f"{float(value):>12.3g}"


def _human_size(nbytes: int) -> str:
    val = float(nbytes)
    for unit in ("B", "KiB", "MiB", "GiB", "TiB"):
        if val < 1024.0:
            return f"{val:.1f} {unit}"
        val /= 1024.0
    return f"{val:.1f} PiB"


def _print_header(artifact_path: str, meta: dict):
    print()
    print("=" * 78)
    print(f"  GRACE UQ artifact: {artifact_path}")
    print(f"  size: {_human_size(os.path.getsize(artifact_path))}")
    print("=" * 78)
    centroid_dtype = next(iter(meta["artifacts"].values()))[
        uq_constants.CENTROIDS
    ].dtype
    print(f"  feature_dim D       : {meta['D']}")
    print(f"  max clusters K_max  : {meta['K_max']}")
    print(f"  n_elements (fitted) : {meta['n_elements']}")
    if meta["element_map"] is not None:
        print(f"  element_map         : {meta['element_map']}")
    print(f"  centroid dtype      : {centroid_dtype}")
    print(f"  thresholds present  : {meta['has_thresholds']}", end="")
    if meta.get("has_eff_thresholds"):
        print(" (raw + effective)")
    else:
        print()
    print(f"  histograms present  : {meta['has_histograms']}", end="")
    if meta.get("has_eff_histograms"):
        print(" (raw + effective)")
    else:
        print()
    print()


def _weighted_artifact(meta: dict) -> bool:
    """True iff at least one element has effective_count != counts."""
    for a in meta["artifacts"].values():
        eff = a.get(uq_constants.EFFECTIVE_COUNT)
        if eff is None:
            continue
        if not np.allclose(eff, a[uq_constants.COUNTS].astype(np.float64)):
            return True
    return False


def _print_per_element_table(meta: dict):
    artifacts = meta["artifacts"]
    element_map = meta["element_map"]
    thresholds = meta["interp_thresholds"]
    eff_thresholds = meta.get("eff_interp_thresholds")
    bins = meta.get("hist_bins")
    K_max = meta["K_max"]
    D = meta["D"]
    is_weighted = _weighted_artifact(meta)
    show_eff_thr = (
        eff_thresholds is not None
        and thresholds is not None
        and not np.allclose(eff_thresholds, thresholds)
    )
    show_sat = thresholds is not None and bins is not None

    header_thresh = (
        f"{'thr_min':>10s}{'thr_med':>10s}{'thr_max':>10s}"
        if thresholds is not None
        else ""
    )
    if show_eff_thr:
        header_thresh += f"{'eff_thr_med':>13s}"
    if show_sat:
        header_thresh += f"{'sat':>6s}"
    eff_count_header = f"{'eff_count':>12s}" if is_weighted else ""
    print(
        f"  {'Element':>10s}{'K_eff':>7s}{'n_atoms':>10s}{eff_count_header}"
        f"{'n/clust':>14s}"
        f"{'max_cond':>12s}{'min_rank':>10s}{'max_trunc':>11s}{header_thresh}"
    )
    print(
        "  "
        + "-"
        * (
            10 + 7 + 10 + (12 if is_weighted else 0) + 14 + 12 + 10 + 11
            + (30 if thresholds is not None else 0)
            + (13 if show_eff_thr else 0)
            + (6 if show_sat else 0)
        )
    )

    for elem_idx in sorted(artifacts.keys()):
        a = artifacts[elem_idx]
        centroids = a[uq_constants.CENTROIDS]  # [K, D]
        sentinel_mask = is_sentinel_mask(centroids)
        k_eff = int(np.sum(~sentinel_mask))
        counts = a[uq_constants.COUNTS]
        n_atoms = int(np.sum(counts))
        eff_counts = a.get(uq_constants.EFFECTIVE_COUNT)
        cond = a.get(uq_constants.COND_NUMBER)
        rank = a.get(uq_constants.EFFECTIVE_RANK)
        n_trunc = a.get(uq_constants.N_TRUNCATED)

        real_counts = counts[~sentinel_mask] if k_eff > 0 else counts[:0]
        if len(real_counts) > 0:
            n_per_cluster_str = f"{int(np.min(real_counts))}..{int(np.max(real_counts))}"
        else:
            n_per_cluster_str = "-"

        if cond is not None and len(cond) > 0:
            real_cond = cond[~sentinel_mask] if k_eff > 0 else cond
            real_rank = rank[~sentinel_mask] if rank is not None and k_eff > 0 else rank
            real_trunc = (
                n_trunc[~sentinel_mask] if n_trunc is not None and k_eff > 0 else n_trunc
            )
            max_cond_str = f"{np.max(real_cond):.2e}"
            min_rank_str = f"{int(np.min(real_rank))}/{D}" if real_rank is not None else "-"
            max_trunc_str = f"{int(np.max(real_trunc))}" if real_trunc is not None else "-"
        else:
            max_cond_str, min_rank_str, max_trunc_str = "-", "-", "-"

        thr_str = ""
        sat_count = 0
        if thresholds is not None and elem_idx < thresholds.shape[0]:
            row = thresholds[elem_idx]
            real_row = row[:k_eff] if 0 < k_eff <= len(row) else row
            if len(real_row) > 0:
                max_val = np.max(real_row)
                max_str = (
                    f"{max_val:.3g}*"
                    if show_sat and _is_saturated(max_val, bins)
                    else f"{max_val:.3g}"
                )
                thr_str = (
                    f"{np.min(real_row):>10.3g}"
                    f"{np.median(real_row):>10.3g}"
                    f"{max_str:>10s}"
                )
                sat_count = _count_saturated(real_row, bins)
            else:
                thr_str = f"{'-':>10s}{'-':>10s}{'-':>10s}"

        if show_eff_thr and elem_idx < eff_thresholds.shape[0]:
            erow = eff_thresholds[elem_idx]
            real_erow = erow[:k_eff] if 0 < k_eff <= len(erow) else erow
            if len(real_erow) > 0:
                thr_str += f"{np.median(real_erow):>13.3g}"
            else:
                thr_str += f"{'-':>13s}"

        if show_sat:
            sat_lbl = f"{sat_count}/{k_eff}" if k_eff > 0 else "-"
            thr_str += f"{sat_lbl:>6s}"

        eff_count_str = ""
        if is_weighted:
            if eff_counts is not None:
                eff_count_str = f"{float(np.sum(eff_counts)):>12.3g}"
            else:
                eff_count_str = f"{'-':>12s}"

        lbl = format_elem_label(elem_idx, element_map)
        sentinel_note = f" ({K_max - k_eff} sentinel)" if k_eff < K_max else ""
        print(
            f"  {lbl:>10s}{k_eff:>7d}{n_atoms:>10d}{eff_count_str}"
            f"{n_per_cluster_str:>14s}"
            f"{max_cond_str:>12s}{min_rank_str:>10s}{max_trunc_str:>11s}{thr_str}"
            + sentinel_note
        )

    # Column legend (printed once below the table)
    print()
    print("  Column legend:")
    print(
        "    Element    : element symbol(idx); idx is the model's internal element index."
    )
    print(
        "    K_eff      : number of *real*, data-fitted clusters for this element."
    )
    print(
        "                 Sentinel-padded slots (added to keep tensor shapes uniform"
    )
    print(
        "                 across elements) are excluded. K_eff < K_max happens when"
    )
    print(
        "                 the element has fewer than K_max * D training atoms."
    )
    print(
        "    n_atoms    : total training atoms of this element used to fit the GMM."
    )
    print(
        "    n/clust    : range (min..max) of training-atom counts per non-sentinel"
    )
    print(
        "                 cluster. Use --verbose to see per-cluster counts."
    )
    print(
        "    max_cond   : largest condition number among the K_eff cluster"
    )
    print(
        f"                 inverse-covariance matrices. Healthy < ~1e10; "
        f"> {_COND_WARN_THRESHOLD:.0e} flags numerically"
    )
    print(
        "                 unstable clusters whose gamma may be unreliable."
    )
    print(
        "    min_rank   : smallest data-supported rank (eigenvalues of unregularized"
    )
    print(
        "                 sample covariance) across the K_eff clusters. Capped at min(D, N-1);"
    )
    print(
        "                 a small value means one cluster has very few atoms."
    )
    print(
        "    max_trunc  : largest count of singular values truncated by pinv across"
    )
    print(
        "                 K_eff clusters (rcond=1e-15). >0 means a cluster's covariance"
    )
    print(
        "                 was rank-deficient even after Tikhonov regularization."
    )
    print(
        "    thr_min /  : min / median / max of the per-cluster robust sigma thresholds"
    )
    print(
        "    thr_med /    used in gamma = sigma / threshold. An atom whose sigma equals"
    )
    print(
        "    thr_max      its cluster's threshold gives gamma ~= 1 (boundary of training"
    )
    print(
        "                 distribution). thr_min == thr_med == thr_max means every"
    )
    print(
        "                 cluster shares the same backfilled threshold."
    )
    print(
        "    eff_count  : Σ per-atom weight from --train-data-weighted. Shown only when"
    )
    print(
        "                 a weighted build produced effective_count ≠ raw counts."
    )
    print(
        "    eff_thr_med: median of per-cluster robust thresholds computed from the"
    )
    print(
        "                 weighted (effective) sigma histogram. Inference uses these"
    )
    print(
        "                 when present. Shown only when they differ from raw thresholds."
    )
    print(
        "    (N sentinel): N of K_max cluster slots are padded; they never attract atoms."
    )
    if show_sat:
        print(
            "    sat        : sat_k / K_eff — count of clusters whose threshold equals the"
        )
        print(
            f"                 histogram ceiling ({float(bins[-1]):g}). Saturated thresholds are"
        )
        print(
            "                 right-censored: the true threshold is >= the ceiling. thr_max is"
        )
        print(
            "                 suffixed with '*' when it is at the ceiling. Use --verbose"
        )
        print(
            "                 to see the actual percentile each ceiling represents."
        )

    # Health summary
    bad_clusters = 0
    for a in artifacts.values():
        cond = a.get(uq_constants.COND_NUMBER)
        if cond is not None:
            bad_clusters += int(np.sum(cond > _COND_WARN_THRESHOLD))
    if bad_clusters:
        print()
        print(
            f"  WARNING: {bad_clusters} cluster(s) have cond_number > {_COND_WARN_THRESHOLD:.0e}; "
            "uncertainty estimates for atoms assigned there may be unreliable."
        )
    total_sat = (
        _count_saturated(thresholds, bins) if show_sat else 0
    )
    if total_sat:
        print()
        print(
            f"  NOTE: {total_sat} cluster(s) have threshold at histogram ceiling "
            f"({float(bins[-1]):g}); displayed value is right-censored. "
            "Use --verbose to see the actual percentile each ceiling represents."
        )
    print()


def _cell(vec, k, width: int, fmt: str, missing: str = "-") -> str:
    """Format ``vec[k]`` with ``fmt`` and right-pad to ``width``.

    Returns the ``missing`` placeholder (right-aligned to ``width``) when
    ``vec`` is ``None`` or ``vec[k]`` is non-finite. The build pipeline
    guarantees length-K vectors, so no bounds check is needed.
    """
    if vec is None:
        return f"{missing:>{width}s}"
    v = vec[k]
    if isinstance(v, float) and not np.isfinite(v):
        return f"{missing:>{width}s}"
    return f"{format(v, fmt):>{width}s}"


def _print_per_cluster_breakdown(meta: dict):
    """Per-element, per-cluster verbose table (counts, cond, rank, thresholds)."""
    artifacts = meta["artifacts"]
    element_map = meta["element_map"]
    thresholds = meta["interp_thresholds"]
    eff_thresholds = meta.get("eff_interp_thresholds")
    bins = meta.get("hist_bins")
    hist_arrays = meta.get("hist_arrays")
    eff_hist_arrays = meta.get("eff_hist_arrays")
    D = meta["D"]
    is_weighted = _weighted_artifact(meta)
    show_eff_thr = (
        eff_thresholds is not None
        and thresholds is not None
        and not np.allclose(eff_thresholds, thresholds)
    )

    print("  Per-cluster breakdown (--verbose):")
    print()
    for elem_idx in sorted(artifacts.keys()):
        a = artifacts[elem_idx]
        centroids = a[uq_constants.CENTROIDS]
        sentinel_mask = is_sentinel_mask(centroids)
        counts = a[uq_constants.COUNTS]
        eff_counts = a.get(uq_constants.EFFECTIVE_COUNT)
        cond = a.get(uq_constants.COND_NUMBER)
        rank = a.get(uq_constants.EFFECTIVE_RANK)
        n_trunc = a.get(uq_constants.N_TRUNCATED)
        thr_row = thresholds[elem_idx] if thresholds is not None else None
        eff_thr_row = eff_thresholds[elem_idx] if show_eff_thr else None
        hist_row = hist_arrays.get(elem_idx) if hist_arrays is not None else None
        eff_hist_row = (
            eff_hist_arrays.get(elem_idx) if eff_hist_arrays is not None else None
        )

        lbl = format_elem_label(elem_idx, element_map)
        print(f"  {lbl}")
        header = f"    {'cluster':>7s}{'n_atoms':>10s}"
        if is_weighted:
            header += f"{'eff_count':>12s}"
        header += f"{'cond':>12s}{'rank':>10s}{'n_trunc':>10s}"
        if thr_row is not None:
            header += f"{'threshold':>12s}"
        if eff_thr_row is not None:
            header += f"{'eff_thresh':>12s}"
        header += f"{'kind':>10s}"
        print(header)
        print("    " + "-" * (len(header) - 4))

        K = centroids.shape[0]
        for k in range(K):
            kind = "sentinel" if sentinel_mask[k] else "real"
            row = (
                f"    {k:>7d}"
                f"{int(counts[k]):>10d}"
            )
            if is_weighted:
                row += _cell(eff_counts, k, 12, ".3g")
            row += (
                _cell(cond, k, 12, ".2e")
                + (f"{int(rank[k])}/{D}" if rank is not None else "-").rjust(10)
                + _cell(n_trunc, k, 10, "d")
            )
            if thr_row is not None:
                row += _format_thr_cell(
                    thr_row[k],
                    hist_row[k] if hist_row is not None and k < hist_row.shape[0] else None,
                    bins,
                )
            if eff_thr_row is not None:
                row += _format_thr_cell(
                    eff_thr_row[k],
                    eff_hist_row[k] if eff_hist_row is not None and k < eff_hist_row.shape[0] else None,
                    bins,
                )
            row += f"{kind:>10s}"
            print(row)
        print()


_INFO_EPILOG = """\
Examples
--------
  # Inspect an artifact written by `grace_uq build`:
  grace_uq info UQ/gmm_artifacts.npz

  # The output shows: feature dim D, max clusters K, n_elements,
  # per-element K_eff / atom counts / max condition number / threshold range,
  # and flags any sentinel-padded clusters.
"""


def info_main(argv=None) -> int:
    parser = argparse.ArgumentParser(
        prog="grace_uq info",
        description="Summarize a UQ .npz artifact",
        epilog=_INFO_EPILOG,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "artifact_path",
        nargs="?",
        default="gmm_artifacts.npz",
        help="Path to gmm_artifacts.npz (default: %(default)s)",
    )
    parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="Print per-cluster breakdown for every element (counts, cond, rank, "
        "truncated, threshold, sentinel/real).",
    )
    args = parser.parse_args(argv)

    if not os.path.exists(args.artifact_path):
        parser.error(f"Artifact not found: {args.artifact_path}")

    meta = read_artifact_metadata(args.artifact_path)
    _print_header(args.artifact_path, meta)
    _print_per_element_table(meta)
    if args.verbose:
        _print_per_cluster_breakdown(meta)
    realigned = meta.get("hist_realigned_elems") or []
    unaligned = meta.get("hist_unaligned_elems") or []
    if realigned:
        print(
            f"  WARNING: histograms AND thresholds for {len(realigned)} element(s) "
            "were stored in a different cluster order than counts/centroids. "
            "Build-pipeline bug: step-3's hist_*/interp_thresholds aren't permuted "
            "alongside the post-finalize sort. INFERENCE ON THIS ARTIFACT IS "
            "AFFECTED — each cluster is matched against the wrong threshold. "
            "Display values shown here are corrected via row-sum matching; the "
            "saved file itself still needs rebuilding to fix inference."
        )
    if unaligned:
        print(
            f"  WARNING: histograms for {len(unaligned)} element(s) could not be "
            "aligned to the centroid order; per-cluster diagnostics are "
            "suppressed for those elements."
        )
    print_diagnostics(meta)
    return 0


if __name__ == "__main__":
    raise SystemExit(info_main())
