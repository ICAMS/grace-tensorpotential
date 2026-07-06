"""Health diagnostics for UQ (GMM-LLRP) artifacts.

``grace_uq info`` runs these checks and prints a DIAGNOSTICS section flagging
calibration problems that make per-atom ``gamma = sigma / threshold`` unreliable
or insensitive. Every check reads ONLY the stored artifact (per-element GMM
stats + sigma histograms + thresholds) — no model load, no GPU.

Checks
------
- sigma_imbalance : the dominant cluster's *typical* gamma (median sigma /
  threshold) is far below 1, i.e. the threshold is set by a small outlier tail
  and ordinary atoms can never approach gamma=1 (over-lenient UQ). This is the
  "99% of sigma in bin 0" pathology: a sparse set of extreme-feature atoms
  inflates the covariance, so the bulk sits at sigma~0. Healthy chi-like
  clusters give typical gamma ~0.85.
- zero_threshold      : threshold <= 0 -> gamma blows up to ~1e9 (legacy bin-0 bug).
- floored_threshold   : 0 < threshold <= ~1 bin width -> p99 fell in bin 0/1;
  cluster has no usable calibration, gamma there is a floor artifact.
- dominant_cluster    : one cluster holds ~all atoms (KMeans collapsed onto one
  mode) -> gamma resolution limited; satellites are tiny.
- underpopulated      : real clusters below the min-atoms floor -> threshold
  backfilled / covariance regularization-dominated.
- ill_conditioned     : inv_cov condition number above ~1e10 -> numerically
  unstable sigma.
- rank_deficient      : effective rank < D or truncated singular values ->
  covariance under-determined (collinear features / too few atoms).
- saturated_threshold : threshold at the histogram ceiling -> right-censored p99
  (the true p99 is >= ceiling).
- corruption          : NaN/Inf in centroids/inv_cov/thresholds, or an
  element_map length mismatch.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from tensorpotential.uq import constants as uq_constants
from tensorpotential.uq.cli._common import format_elem_label, is_sentinel_mask

FAIL, WARN, INFO = "FAIL", "WARN", "INFO"
_SEV_ORDER = {FAIL: 0, WARN: 1, INFO: 2}
_SEV_MARK = {FAIL: "✗", WARN: "⚠", INFO: "ℹ"}

# Heuristic thresholds.
_LENIENT_GAMMA = 0.1     # typical gamma below this -> over-lenient calibration
_DOMINANT_FRAC = 0.95    # one cluster holding this share -> collapsed clustering
_CEILING_TAIL = 1e-3     # fraction of sigma at the ceiling worth calling out
_BIN0_NOTE_MIN = 0.05    # only mention "X% of sigma in bin 0" above this share
_COND_WARN = 1e10        # condition number above this -> unstable inv_cov


@dataclass
class Group:
    severity: str
    code: str
    title: str
    asc: bool = False  # sort items ascending by value (True) or descending (False)
    items: list = field(default_factory=list)  # (scope, message, value)


def _fmt_count(n) -> str:
    """Compact atom count: 12.3M / 456k / 87."""
    n = float(n)
    for div, suf in ((1e9, "G"), (1e6, "M"), (1e3, "k")):
        if n >= div:
            return f"{n / div:.1f}{suf}"
    return f"{n:.0f}"


def _csize(row) -> str:
    """Cluster population as a compact string, preferring the effective
    (weighted) count when it diverges from the raw atom count."""
    ec = row.get("eff_count")
    if ec is not None and abs(ec - row["count"]) > 0.5:
        return f"{_fmt_count(ec)} eff."
    return f"{_fmt_count(row['count'])}"


def _cref(row, extra: str = "") -> str:
    """``cluster {k} ({size}[, extra])`` — names the cluster a finding is about
    and how many atoms it holds, per the standing "always say which cluster and
    its size" rule."""
    inside = _csize(row) + (f", {extra}" if extra else "")
    return f"cluster {row['k']} ({inside})"


def _cref_list(rows, extra=None) -> str:
    """Join ``_cref`` for several clusters. ``extra`` (optional) is a callable
    ``row -> str`` adding a per-cluster metric (cond, rank, …)."""
    return ", ".join(_cref(r, extra(r) if extra else "") for r in rows)


def _hist_stats(h, bins, bw):
    s = float(np.sum(h))
    if s <= 0:
        return None
    cum = np.cumsum(h) / s
    med = float(bins[np.searchsorted(cum, 0.5)] + bw / 2.0)
    return {
        "n": s,
        "median": med,
        "frac_bin0": float(h[0]) / s,
        "frac_ceiling": float(h[-1]) / s,
    }


def _iter_real_clusters(meta):
    """Yield per-(element, real-cluster) dicts with the stats every check needs.

    Uses the effective (weighted) threshold/histogram when present — that is what
    inference consumes — else the raw ones.
    """
    arts = meta["artifacts"]
    emap = meta["element_map"]
    bins = meta.get("hist_bins")
    bw = float(bins[1] - bins[0]) if bins is not None else None
    eff_thr = meta.get("eff_interp_thresholds")
    thr = eff_thr if eff_thr is not None else meta.get("interp_thresholds")
    eff_hist = meta.get("eff_hist_arrays") or {}
    raw_hist = meta.get("hist_arrays") or {}
    for e in sorted(arts.keys()):
        a = arts[e]
        cents = a[uq_constants.CENTROIDS]
        smask = is_sentinel_mask(cents)
        counts = a[uq_constants.COUNTS]
        eff_counts = a.get(uq_constants.EFFECTIVE_COUNT)
        cond = a.get(uq_constants.COND_NUMBER)
        rank = a.get(uq_constants.EFFECTIVE_RANK)
        ntr = a.get(uq_constants.N_TRUNCATED)
        he = eff_hist.get(e)
        if he is None:
            he = raw_hist.get(e)
        rows = []
        for k in range(cents.shape[0]):
            if smask[k]:
                continue
            stats = None
            if he is not None and bins is not None and k < he.shape[0]:
                stats = _hist_stats(he[k], bins, bw)
            rows.append({
                "k": k,
                "count": int(counts[k]),
                "eff_count": float(eff_counts[k]) if eff_counts is not None else None,
                "thr": float(thr[e, k]) if thr is not None and e < thr.shape[0] else None,
                "cond": float(cond[k]) if cond is not None else None,
                "rank": int(rank[k]) if rank is not None else None,
                "ntrunc": int(ntr[k]) if ntr is not None else None,
                "stats": stats,
            })
        yield e, format_elem_label(e, emap), rows, bw


def _check_sigma_imbalance(meta):
    g = Group(WARN, "sigma_imbalance",
              "Over-lenient gamma (sigma imbalance): threshold set by the outlier "
              "tail; typical atoms read gamma << 1 (healthy ~0.85)", asc=True)
    for e, lbl, rows, bw in _iter_real_clusters(meta):
        if not rows:
            continue
        dom = max(rows, key=lambda r: r["count"])  # cluster atoms actually land in
        st, thr = dom["stats"], dom["thr"]
        if st is None or not thr or thr <= 0:
            continue
        tg = st["median"] / thr
        if tg < _LENIENT_GAMMA:
            # Identify the cluster the bulk lands in and its population. Prefer
            # the effective (weighted) count when it differs from the raw atom
            # count, since the thresholds are computed from the weighted sigma.
            ec = dom["eff_count"]
            if ec is not None and abs(ec - dom["count"]) > 0.5:
                pop = f"{_fmt_count(ec)} eff. atoms"
            else:
                pop = f"{_fmt_count(dom['count'])} atoms"
            # Describe where the imbalance comes from, omitting non-informative
            # clauses: bin-0 share only when the bulk actually piles there, and
            # the ceiling share only when atoms reach it. When the bulk sits
            # above bin 0 with no ceiling mass, the median<<threshold gap in the
            # leading text already tells the story (a thin mid-range tail).
            notes = []
            if st["frac_bin0"] >= _BIN0_NOTE_MIN:
                notes.append(f"{st['frac_bin0']:.0%} of sigma in bin 0")
            if st["frac_ceiling"] >= _CEILING_TAIL:
                notes.append(f"{st['frac_ceiling']:.2%} at the ceiling")
            note = f"; {', '.join(notes)}" if notes else ""
            g.items.append((lbl,
                f"cluster {dom['k']} ({pop}): bulk gamma~{tg:.2g} "
                f"(median sigma={st['median']:.2g}, threshold={thr:.3g}){note}",
                tg))
    return g


def _check_thresholds(meta):
    """Zero (FAIL) and floored/degenerate (WARN) thresholds, in two groups."""
    zero = Group(FAIL, "zero_threshold",
                 "Zero threshold: gamma = sigma/threshold blows up (~1e9)")
    floored = Group(WARN, "floored_threshold",
                    "Floored threshold: p99 fell in bin 0/1 -> no usable "
                    "calibration; gamma there is a floor artifact")
    def _thr(r):
        return f"thr={r['thr']:.3g}"

    for e, lbl, rows, bw in _iter_real_clusters(meta):
        with_thr = [r for r in rows if r["thr"] is not None]
        if not with_thr:
            continue
        zrows = [r for r in with_thr if r["thr"] <= 0]
        frows = ([r for r in with_thr if 0 < r["thr"] <= 1.5 * bw]
                 if bw is not None else [])
        if zrows:
            zero.items.append((lbl, f"{len(zrows)}/{len(with_thr)} with threshold <= 0: "
                               f"{_cref_list(zrows)}", len(zrows)))
        if frows:
            floored.items.append((lbl, f"{len(frows)}/{len(with_thr)} at the bin-0/1 floor: "
                                  f"{_cref_list(frows, _thr)}", len(frows)))
    return [zero, floored]


def _check_dominant_cluster(meta):
    g = Group(WARN, "dominant_cluster",
              "Collapsed clustering: one cluster holds nearly all atoms; gamma "
              "resolution limited (more clusters won't help if inertia is flat)")
    for e, lbl, rows, bw in _iter_real_clusters(meta):
        if len(rows) < 2:
            continue
        tot = sum(r["count"] for r in rows)
        if tot <= 0:
            continue
        dom = max(rows, key=lambda r: r["count"])
        frac = dom["count"] / tot
        if frac >= _DOMINANT_FRAC:
            g.items.append((lbl, f"{_cref(dom)} = {frac:.1%} of atoms "
                                 f"({len(rows)} real clusters)", frac))
    return g


def _check_underpopulated(meta):
    minat = max(50, meta["D"])
    g = Group(WARN, "underpopulated",
              f"Under-populated clusters (< {minat} atoms): threshold backfilled / "
              "covariance regularization-dominated")
    for e, lbl, rows, bw in _iter_real_clusters(meta):
        under = [r for r in rows if r["count"] < minat]
        if under:
            g.items.append((lbl, f"{len(under)}/{len(rows)} below {minat} atoms: "
                            f"{_cref_list(under)}", len(under)))
    return g


def _check_covariance(meta):
    D = meta["D"]
    cond = Group(WARN, "ill_conditioned",
                 f"Ill-conditioned covariance (cond > {_COND_WARN:.0e}): unstable sigma")
    rankg = Group(INFO, "rank_deficient",
                  f"Rank-deficient covariance (rank < D={D} or truncated SVs)")
    def _cnd(r):
        return f"cond={r['cond']:.1e}"

    def _rnk(r):
        return f"rank={r['rank']}/{D}, trunc={r['ntrunc'] or 0}"

    for e, lbl, rows, bw in _iter_real_clusters(meta):
        bad_cond = sorted((r for r in rows if r["cond"] is not None and r["cond"] > _COND_WARN),
                          key=lambda r: r["cond"], reverse=True)
        if bad_cond:
            cond.items.append((lbl, _cref_list(bad_cond, _cnd), bad_cond[0]["cond"]))
        bad_rank = [r for r in rows
                    if (r["rank"] is not None and r["rank"] < D)
                    or (r["ntrunc"] is not None and r["ntrunc"] > 0)]
        bad_rank.sort(key=lambda r: (r["rank"] if r["rank"] is not None else D))
        if bad_rank:
            worst = D - (bad_rank[0]["rank"] if bad_rank[0]["rank"] is not None else D)
            rankg.items.append((lbl, _cref_list(bad_rank, _rnk), worst))
    return [cond, rankg]


def _check_saturation(meta):
    g = Group(INFO, "saturated_threshold",
              "Saturated thresholds at the histogram ceiling: right-censored p99 "
              "(true p99 >= ceiling)")
    bins = meta.get("hist_bins")
    if bins is None:
        return g
    ceiling = float(bins[-1]) - 1e-6
    for e, lbl, rows, bw in _iter_real_clusters(meta):
        sat = [r for r in rows if r["thr"] is not None and r["thr"] >= ceiling]
        if sat:
            g.items.append((lbl, f"{len(sat)}/{len(rows)} at the ceiling: "
                            f"{_cref_list(sat)}", len(sat)))
    return g


def _check_corruption(meta):
    g = Group(FAIL, "corruption", "Corrupted artifact (NaN/Inf or shape mismatch)")
    for e, lbl, rows, bw in _iter_real_clusters(meta):
        a = meta["artifacts"][e]
        cents = a[uq_constants.CENTROIDS]
        ic = a.get(uq_constants.INV_COV)
        bad = []  # (row, "centroid/inv_cov")
        for r in rows:
            k = r["k"]
            where = []
            if k < cents.shape[0] and not np.all(np.isfinite(cents[k])):
                where.append("centroid")
            if ic is not None and k < ic.shape[0] and not np.all(np.isfinite(ic[k])):
                where.append("inv_cov")
            if where:
                bad.append((r, "/".join(where)))
        if bad:
            g.items.append((lbl, ", ".join(f"{_cref(r)} non-finite {w}" for r, w in bad),
                            len(bad)))
    for key in ("interp_thresholds", "eff_interp_thresholds"):
        t = meta.get(key)
        if t is not None and not np.all(np.isfinite(t)):
            g.items.append(("artifact", f"non-finite values in {key}", 1))
    emap = meta["element_map"]
    if emap is not None and len(emap) != meta["n_elements"]:
        g.items.append(("artifact",
                        f"element_map length {len(emap)} != n_elements {meta['n_elements']}", 1))
    return g


def diagnose(meta) -> list:
    """Run every check; return non-empty Groups sorted most-severe first."""
    groups = []
    groups.append(_check_corruption(meta))
    groups.append(_check_sigma_imbalance(meta))
    groups.extend(_check_thresholds(meta))
    groups.append(_check_dominant_cluster(meta))
    groups.append(_check_underpopulated(meta))
    groups.extend(_check_covariance(meta))
    groups.append(_check_saturation(meta))
    groups = [g for g in groups if g.items]
    groups.sort(key=lambda g: _SEV_ORDER[g.severity])
    return groups


def print_diagnostics(meta) -> None:
    groups = diagnose(meta)
    print("  DIAGNOSTICS")
    print("  " + "-" * 74)
    if not groups:
        print("    ✓ no problems detected")
        print()
        return
    n_fail = sum(1 for g in groups if g.severity == FAIL)
    n_warn = sum(1 for g in groups if g.severity == WARN)
    for g in groups:
        items = sorted(g.items, key=lambda it: it[2], reverse=not g.asc)
        mark = _SEV_MARK[g.severity]
        print(f"    {mark} [{g.severity}] {g.title}  ({len(items)} element(s))")
        for scope, msg, _v in items:
            print(f"        {scope:>10s}: {msg}")
    print()
    print(f"    Summary: {n_fail} fail, {n_warn} warn across "
          f"{len(groups)} check(s) with findings.")
    print()
