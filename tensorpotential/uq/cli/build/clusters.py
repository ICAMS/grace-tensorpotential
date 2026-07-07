"""Cluster merging, compaction and per-element centroid selection helpers."""

from __future__ import annotations

import logging

import numpy as np

from tensorpotential.uq import constants as uq_constants
from tensorpotential.uq.cli._common import (
    SENTINEL_CENTROID_VALUE as _SENTINEL_CENTROID_VALUE,
    is_sentinel_mask,
)

log = logging.getLogger("grace_uq")

def _merge_small_clusters(centroids, counts, min_atoms, lbl=None):
    """Iteratively merge underpopulated clusters into their nearest neighbor.

    For the smallest cluster with count < ``min_atoms``, find the closest
    (Euclidean) other centroid and merge: the survivor's centroid becomes
    the count-weighted average of the two, and the counts are summed. The
    underpopulated cluster is dropped. Repeats until every remaining cluster
    has count ≥ min_atoms, or only one cluster remains.

    Returns
    -------
    (centroids, counts, n_merged) : (np.ndarray [k', D], np.ndarray [k'], int)
    """
    centroids = np.asarray(centroids, dtype=np.float64).copy()
    counts = np.asarray(counts, dtype=np.float64).copy()
    n_merged = 0
    label = f" [{lbl}]" if lbl else ""
    log.debug(
        "merge%s: start K=%d, counts=%s, sum=%.1f, min_atoms=%d",
        label,
        len(counts),
        np.array2string(counts, precision=1, separator=",", suppress_small=True),
        float(counts.sum()),
        int(min_atoms),
    )
    while True:
        if len(centroids) <= 1:
            break
        small = np.where(counts < min_atoms)[0]
        if small.size == 0:
            break
        smallest = small[np.argmin(counts[small])]
        others = np.delete(np.arange(len(centroids)), smallest)
        dists = np.linalg.norm(centroids[others] - centroids[smallest], axis=1)
        target_local = int(np.argmin(dists))
        target = int(others[target_local])

        c_small = float(counts[smallest])
        c_target = float(counts[target])
        total = c_small + c_target

        log.debug(
            "merge%s: step %d — drop cluster %d (count=%.1f) into "
            "cluster %d (count=%.1f, dist=%.3g) → new count=%.1f, K %d → %d",
            label,
            n_merged + 1,
            int(smallest),
            c_small,
            target,
            c_target,
            float(dists[target_local]),
            total,
            len(counts),
            len(counts) - 1,
        )

        centroids[target] = (
            centroids[smallest] * c_small + centroids[target] * c_target
        ) / total
        counts[target] = total

        keep = np.delete(np.arange(len(centroids)), smallest)
        centroids = centroids[keep]
        counts = counts[keep]
        n_merged += 1

    log.debug(
        "merge%s: done. final K=%d, counts=%s, sum=%.1f, "
        "min count=%.1f, n_merged=%d",
        label,
        len(counts),
        np.array2string(counts, precision=1, separator=",", suppress_small=True),
        float(counts.sum()),
        float(counts.min()) if len(counts) else float("nan"),
        n_merged,
    )
    return centroids, counts, n_merged


def _merge_step2_clusters(
    centroids,
    scatter,
    counts,
    min_atoms,
    lbl=None,
    sentinel_value=_SENTINEL_CENTROID_VALUE,
    effective_counts=None,
):
    """Merge real clusters with raw ``count < min_atoms`` into their nearest
    real neighbor, using the **parallel-axis theorem** to combine scatter
    matrices exactly. Operates on the step-2 artifact (post-scatter
    accumulation), so counts are real per-cluster atom counts — eliminating
    the step-1-vs-step-2 redistribution drift that the meta-centroid-based
    step-1 merge cannot see.

    Two counts are tracked: raw (int atom count, used for the ``min_atoms``
    reliability gate) and effective (sum of weights, used as the weight in
    the parallel-axis math because the SCATTER was already built as
    ``Σ w·δδᵀ``). When ``effective_counts`` is None it defaults to
    ``counts.astype(float64)`` — back-compat for unweighted runs.

    For clusters A (effective n_A, centroid c_A, scatter S_A) and B
    (effective n_B, centroid c_B, scatter S_B), merging into B gives:
        c_new   = (n_A c_A + n_B c_B) / (n_A + n_B)
        n_new   = n_A + n_B
        S_new   = S_A + S_B + (n_A n_B / (n_A + n_B)) · (c_A − c_B)(c_A − c_B)ᵀ
    The dropped cluster's slot is demoted to a sentinel (centroid set to
    ``sentinel_value``, scatter zeroed, both counts zeroed) so the artifact
    tensor shape ``[K_max, ...]`` is preserved.

    Iterates until every real cluster has raw count ≥ min_atoms, or only one
    real cluster remains (Tikhonov handles that case downstream).
    """
    centroids = np.asarray(centroids, dtype=np.float64).copy()
    scatter = np.asarray(scatter, dtype=np.float64).copy()
    counts = np.asarray(counts, dtype=np.float64).copy()
    if effective_counts is None:
        effective_counts = counts.copy()
    else:
        effective_counts = np.asarray(effective_counts, dtype=np.float64).copy()
    label = f" [{lbl}]" if lbl else ""
    K0_real = int((~is_sentinel_mask(centroids)).sum())
    n_merged = 0
    while True:
        is_real = ~is_sentinel_mask(centroids)
        real_idx = np.where(is_real)[0]
        if real_idx.size <= 1:
            break
        small_real = real_idx[counts[real_idx] < min_atoms]
        if small_real.size == 0:
            break
        smallest = int(small_real[np.argmin(counts[small_real])])
        other_real = real_idx[real_idx != smallest]
        dists = np.linalg.norm(
            centroids[other_real] - centroids[smallest], axis=1
        )
        target = int(other_real[np.argmin(dists)])

        eff_s = float(effective_counts[smallest])
        eff_t = float(effective_counts[target])
        eff_total = eff_s + eff_t
        raw_s = float(counts[smallest])
        raw_t = float(counts[target])
        delta = centroids[smallest] - centroids[target]

        if eff_total > 0:
            centroids[target] = (
                eff_s * centroids[smallest] + eff_t * centroids[target]
            ) / eff_total
            scatter[target] = (
                scatter[smallest]
                + scatter[target]
                + (eff_s * eff_t / eff_total) * np.outer(delta, delta)
            )
            effective_counts[target] = eff_total
            counts[target] = raw_s + raw_t

        log.debug(
            "step2 merge%s: drop slot %d (raw=%.1f, eff=%.1f) into slot %d "
            "(raw=%.1f, eff=%.1f, dist=%.3g) → new raw=%.1f eff=%.1f",
            label, smallest, raw_s, eff_s, target,
            raw_t, eff_t, float(dists[np.argmin(dists)]),
            counts[target], effective_counts[target],
        )

        centroids[smallest] = sentinel_value
        scatter[smallest] = 0.0
        counts[smallest] = 0
        effective_counts[smallest] = 0.0
        n_merged += 1

    is_real_final = ~is_sentinel_mask(centroids)
    K1_real = int(is_real_final.sum())
    if n_merged > 0:
        log.info(
            "[Master] Step 2 merge%s: %d cluster(s) with < %d atoms folded "
            "into nearest neighbor (real K %d → %d). Final raw counts: %s",
            label, n_merged, int(min_atoms), K0_real, K1_real,
            np.array2string(
                counts[is_real_final],
                precision=1, separator=",", suppress_small=True,
            ),
        )
    return centroids, scatter, counts, effective_counts, n_merged


def _compact_step2_artifacts(base, K_max_new=None, sentinel_value=_SENTINEL_CENTROID_VALUE):
    """Pack non-sentinel slots to the front per element and trim trailing
    sentinel rows so every element's tensor shrinks to ``K_max_new``.

    If ``K_max_new`` is None it is computed as ``max(K_eff_e for e)`` so the
    artifact's K_max equals the largest post-merge real-cluster count across
    all elements — eliminating spurious all-empty sentinel rows that every
    element would otherwise carry.

    Mutates the per-element tensors in ``base`` in place. Returns the new
    K_max actually applied.
    """
    is_real_per_e = {
        e: ~is_sentinel_mask(d[uq_constants.CENTROIDS])
        for e, d in base.items()
    }
    if K_max_new is None:
        K_max_new = max(int(m.sum()) for m in is_real_per_e.values())
    K_max_new = max(K_max_new, 1)

    for e, d in base.items():
        is_real = is_real_per_e[e]
        order = np.concatenate(
            [np.where(is_real)[0], np.where(~is_real)[0]]
        )
        c = d[uq_constants.CENTROIDS][order][:K_max_new]
        if c.shape[0] < K_max_new:
            pad = np.full(
                (K_max_new - c.shape[0], c.shape[1]),
                sentinel_value,
                dtype=c.dtype,
            )
            c = np.vstack([c, pad])
        d[uq_constants.CENTROIDS] = c

        n = np.asarray(d[uq_constants.COUNTS])[order][:K_max_new]
        if n.shape[0] < K_max_new:
            n = np.concatenate(
                [n, np.zeros(K_max_new - n.shape[0], dtype=n.dtype)]
            )
        d[uq_constants.COUNTS] = n

        if d.get(uq_constants.EFFECTIVE_COUNT) is not None:
            ne = np.asarray(d[uq_constants.EFFECTIVE_COUNT])[order][:K_max_new]
            if ne.shape[0] < K_max_new:
                ne = np.concatenate(
                    [ne, np.zeros(K_max_new - ne.shape[0], dtype=ne.dtype)]
                )
            d[uq_constants.EFFECTIVE_COUNT] = ne

        if d.get(uq_constants.SCATTER) is not None:
            S = d[uq_constants.SCATTER][order][:K_max_new]
            if S.shape[0] < K_max_new:
                pad = np.zeros(
                    (K_max_new - S.shape[0],) + S.shape[1:], dtype=S.dtype
                )
                S = np.concatenate([S, pad], axis=0)
            d[uq_constants.SCATTER] = S
    return K_max_new


def _pick_centroids_for_element(
    *,
    e,
    n_atoms_e,
    n_clusters_candidates,
    elem_optimal_k,
    global_kms_by_k,
    all_centroids_by_k,
    min_atoms,
    lbl=None,
    regularization=1e-6,
    weighted=False,
):
    """Pick centroids for one element. Run the KMeans at the largest
    candidate K ≤ ``elem_optimal_k`` (this element's own elbow optimum,
    NOT the global max), then merge any cluster with < ``min_atoms`` atoms
    into its nearest neighbor. Atoms ultimately re-bind to whichever
    surviving centroid is nearest in step 2, so merging is a pure step-1
    redistribution that preserves all atoms.

    Falls back to K=1 only when no KMeans was fit at any candidate K (very
    rare — only happens for elements with too few meta-centroids for the
    smallest k). The resulting K=1 centroid is the weighted mean of the
    meta-centroids, with a warning logged so the user knows Tikhonov
    regularization is doing the heavy lifting on an under-determined
    covariance.

    Returns
    -------
    (real_centroids, best_k) : tuple of (np.ndarray [best_k, D], int)
    """
    k_min = min(n_clusters_candidates)
    label = lbl if lbl is not None else str(e)

    km = None
    chosen_k = 0
    for k in sorted(n_clusters_candidates, reverse=True):
        if k > elem_optimal_k:
            continue
        candidate = global_kms_by_k.get(k, {}).get(e)
        if candidate is not None:
            km = candidate
            chosen_k = k
            break

    if km is None:
        parts = all_centroids_by_k[k_min][e]
        c_all = np.vstack([p[0] for p in parts])
        w_all = np.concatenate([p[1] for p in parts])
        real_centroids = np.average(c_all, axis=0, weights=w_all)[None, :]
        log.warning(
            "[Master] Step 1: element %s (%d atoms) has no fitted KMeans "
            "for any candidate k; synthesizing K=1 from meta-centroids. "
            "Tikhonov regularization (eps=%.0e) will stabilize the "
            "covariance.",
            label, n_atoms_e, regularization,
        )
        return real_centroids, 1

    parts = all_centroids_by_k[chosen_k][e]
    c_all = np.vstack([p[0] for p in parts])
    w_all = np.concatenate([p[1] for p in parts])
    labels = km.predict(c_all)
    counts = np.bincount(labels, weights=w_all, minlength=chosen_k)
    centroids = km.cluster_centers_

    log.info(
        "[Master] Step 1: element %s — KMeans K=%d on %d meta-centroids "
        "(weights sum=%.1f, raw atom count=%d). Per-cluster counts: %s",
        label,
        chosen_k,
        len(c_all),
        float(w_all.sum()),
        n_atoms_e,
        np.array2string(counts, precision=1, separator=",", suppress_small=True),
    )
    # Under unweighted builds, meta-centroid weights are raw atom counts and
    # must match ``n_atoms_e`` exactly. Under ``--train-data-weighted``, the
    # weights are *effective* counts (Σ per-atom weight), so a mismatch is
    # expected by construction — don't cry wolf.
    if not weighted and abs(float(w_all.sum()) - float(n_atoms_e)) > 0.5:
        log.warning(
            "[Master] Step 1: element %s — meta-centroid weights sum (%.1f) "
            "does NOT match raw atom count (%d). Per-cluster floor checks "
            "below will use the (likely wrong) meta-centroid weights.",
            label, float(w_all.sum()), n_atoms_e,
        )

    centroids, counts, n_merged = _merge_small_clusters(
        centroids, counts, min_atoms, lbl=label
    )

    final_k = len(centroids)
    if n_merged > 0:
        log.info(
            "[Master] Step 1: element %s — merged %d underpopulated "
            "cluster(s) into nearest neighbors (K=%d → %d, min count after "
            "merge = %d).",
            label, n_merged, chosen_k, final_k, int(counts.min()),
        )
    if final_k == 1 and counts[0] < min_atoms:
        log.warning(
            "[Master] Step 1: element %s has only %d atoms (< min %d); "
            "K=1 with Tikhonov regularization (eps=%.0e) will stabilize the "
            "under-determined covariance.",
            label, n_atoms_e, min_atoms, regularization,
        )

    return centroids, final_k
