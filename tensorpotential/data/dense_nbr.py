"""Host-side data prep for the dense (reshape) equivariant neighbor aggregation.

The dense aggregation (``compute._dense_reshape_einsum``) replaces the per-bond
``einsum + unsorted_segment_sum`` (an atomic scatter) with a batched matmul over a
per-atom-uniform ``[n_atoms*max_neigh, ...]`` bond layout. This module produces
that layout host-side: a permutation that places each atom's bonds in a contiguous
``max_neigh`` block (padded), so the model's aggregation is a plain ``reshape``
(no in-graph gather -> free backward). ``max_neigh`` is the layout's per-atom width,
so XLA specializes/recompiles per distinct tier and reuses otherwise.

Lives in the data module (not ``instructions/compute``) because it is pure data
preparation: it consumes a bond list and reorders bond arrays; no TensorFlow /
autodiff dependency.
"""
import numpy as np

# Padding-boundary granularity for ``max_neigh``: the true max neighbor count is
# rounded UP to a multiple of this tier. Coarser -> fewer distinct ``max_neigh``
# values -> fewer XLA recompiles, but more padding waste.
DENSE_TIER = 16


def build_dense_reshape_perm(ind_i, n_atoms, sentinel, tier=DENSE_TIER, force_max_neigh=None):
    """Permutation for the per-atom-uniform reshape layout: ``perm`` is
    ``[n_atoms*max_neigh]`` int32 mapping each flat slot ``atom*max_neigh + s`` to a
    source bond index, with ``sentinel`` for padded slots. Apply it host-side (once)
    to the bond arrays (see ``reorder_bonds_for_reshape``) so the model aggregation
    is a ``reshape``.

    Built from REAL bonds only (``ind_i`` = central-atom index per real bond; exclude
    padding bonds so they don't inflate ``max_neigh``). Order-agnostic: does NOT assume
    ``ind_i`` is sorted (the aggregation sums over the slot axis, so any consistent
    per-atom slot assignment is fine). ``max_neigh`` = true max neighbor count rounded
    up to ``tier`` (or ``force_max_neigh`` if given, which must be >= true max).
    ``sentinel`` is the bond index for empty slots — set it to the bond count so the
    appended dummy row is hit. Returns ``(perm[int32], max_neigh)``.
    """
    ind_i = np.asarray(ind_i)
    nb = ind_i.shape[0]
    counts = np.bincount(ind_i, minlength=n_atoms)[:n_atoms]
    true_max = int(counts.max()) if counts.size else 0
    if force_max_neigh is not None:
        assert force_max_neigh >= true_max, f"{force_max_neigh} < true max {true_max}"
        max_neigh = int(force_max_neigh)
    else:
        max_neigh = int(np.ceil(max(true_max, 1) / tier) * tier)
    # slot = position of each bond within its atom's group (no sortedness assumed)
    order = np.argsort(ind_i, kind="stable")
    sorted_ii = ind_i[order]
    first = np.searchsorted(sorted_ii, sorted_ii, side="left")
    slot = np.empty(nb, dtype=np.int64)
    slot[order] = np.arange(nb) - first
    perm = np.full(n_atoms * max_neigh, sentinel, dtype=np.int32)
    perm[ind_i * max_neigh + slot] = np.arange(nb, dtype=np.int32)
    return perm, max_neigh


def reorder_bonds_for_reshape(bond_arrays, perm, dummy_values):
    """Host-side: reorder/pad bond arrays into the per-atom-uniform reshape layout.
    ``bond_arrays`` maps key->array ``[n_bonds, ...]``; for each, a dummy row
    (``dummy_values[key]``) is appended (index == sentinel) and the array is gathered
    by ``perm`` -> ``[n_atoms*max_neigh, ...]``. Padded slots get the dummy row; set
    the bond-vector dummy beyond the cutoff so the envelope zeros its contribution.
    """
    out = {}
    for k, arr in bond_arrays.items():
        arr = np.asarray(arr)
        dummy = np.full((1,) + arr.shape[1:], dummy_values[k], dtype=arr.dtype)
        out[k] = np.concatenate([arr, dummy], axis=0)[perm]
    return out


def auto_slot_budget(nat, max_neigh, batch_size):
    """Auto dense slot budget ``= batch_size * mean(nat) * mean(max_neigh)``: a mean-coordination
    batch then holds ~``batch_size`` structures, denser buckets proportionally fewer. ``mean`` (not
    ``median``) gives fuller batches / fewer launches than median without raising peak per-batch
    footprint; adding a std term was measured to over-pack and INCREASE padding on heavy-tailed
    (foundation) data, so it is intentionally omitted. Pure (numpy only)."""
    nat = np.asarray(nat)
    max_neigh = np.asarray(max_neigh)
    if nat.size == 0:
        return 1
    return int(round(batch_size * float(np.mean(nat)) * float(np.mean(max_neigh))))


def _max_neigh_width_dp(nat, max_neigh, max_buckets_cap):
    """DP for the adaptive width bucketing (shared by ``bucket_by_max_neigh`` and the adaptive
    split search). Partition the distinct ``max_neigh`` values (ascending) into contiguous
    buckets; cost of a bucket = ``(Σ nat in bucket) · bucket_max``. ``dp[k][M]`` = min total
    width-only slots ``Σ nat·width`` using exactly ``k`` buckets. Returns
    ``(vals, inv, dp, bnd, M, cap)``; one O(cap·M²) pass over the histogram (M = #distinct
    values → scales to any N). Pure (numpy only)."""
    vals, inv = np.unique(max_neigh, return_inverse=True)  # distinct max_neigh, ascending
    M = len(vals)
    nat_per = np.bincount(inv, weights=nat, minlength=M).astype(np.float64)
    P = np.concatenate([[0.0], np.cumsum(nat_per)])
    cap = int(min(max_buckets_cap, M)) if M else 0
    INF = float("inf")
    dp = np.full((cap + 1, M + 1), INF)
    dp[0, 0] = 0.0
    bnd = np.zeros((cap + 1, M + 1), dtype=np.int64)
    for k in range(1, cap + 1):
        for i in range(1, M + 1):
            w = float(vals[i - 1])  # bucket [j, i) -> width vals[i-1]
            cost = dp[k - 1, :i] + (P[i] - P[:i]) * w
            j = int(np.argmin(cost))
            dp[k, i] = cost[j]
            bnd[k, i] = j
    return vals, inv, dp, bnd, M, cap


def _widths_for_k(vals, inv, bnd, M, K):
    """Reconstruct ``width_per_structure`` for a ``K``-bucket partition from a DP table."""
    width_of_val = np.empty(M, dtype=np.int64)
    i, k = M, K
    while k > 0:
        j = int(bnd[k, i])
        width_of_val[j:i] = int(vals[i - 1])
        i, k = j, k - 1
    return width_of_val[inv].astype(np.int64)


def bucket_by_max_neigh(
    nat, max_neigh, n_neigh, n_buckets="auto", max_padding=0.15, max_buckets_cap=64
):
    """Adaptive reshape-width bucketing: assign each structure a reshape ``width`` by
    partitioning structures into contiguous ``max_neigh`` buckets, each padded to the bucket's
    OWN max ``max_neigh`` (the minimal width that fits its members). The partition minimizes
    total padded neighbor slots ``Σ nat·width`` for the chosen bucket count, via DP on the
    ``max_neigh`` histogram (O(``n_buckets`` · M²), M = #distinct ``max_neigh`` values) — so it
    scales to any dataset size. More buckets → less over-padding (down toward the
    intra-structure floor), at the cost of more distinct widths (XLA recompiles).

    ``n_buckets``: an int K, or ``"auto"`` → the smallest K in ``[1, max_buckets_cap]`` whose
    width-padding fraction ``1 - Σn_neigh / Σ(nat·width)`` is ``<= max_padding`` (best effort at
    the cap). Returns ``(width_per_structure[int64, len n], n_buckets_used)``. Pure (numpy only)."""
    nat = np.asarray(nat)
    max_neigh = np.asarray(max_neigh)
    n_neigh = np.asarray(n_neigh)
    if nat.shape[0] == 0:
        return np.zeros(0, dtype=np.int64), 0
    vals, inv, dp, bnd, M, cap = _max_neigh_width_dp(nat, max_neigh, max_buckets_cap)
    real = float(np.asarray(n_neigh).sum())
    if n_buckets == "auto":
        K = cap
        for kk in range(1, cap + 1):
            slots = dp[kk, M]
            if slots < float("inf") and (slots <= 0 or (1.0 - real / slots) <= max_padding):
                K = kk
                break
    else:
        K = max(1, min(int(n_buckets), M))
    return _widths_for_k(vals, inv, bnd, M, K), K


def pack_structures_elastic(nat, width_per_struct, batch_size, slot_budget):
    """Group structures into footprint-bounded batches within each reshape-``width`` group
    (widths come from ``bucket_by_max_neigh``). Within a width group, sort by ``nat`` and
    greedily fill a batch while ``(running_nat + nat) * width <= slot_budget`` and
    ``n_structs < batch_size``; a lone structure that alone exceeds the budget still becomes its
    own size-1 batch. Returns ``list[{"structure_ind": list[int], "width": int}]`` with
    positions into the passed arrays. Pure (numpy only).

    Note: structures are reordered relative to input order (grouped by width, sorted by ``nat``);
    per-structure identity is carried by ``DATA_STRUCTURE_ID`` so downstream attribution is
    unaffected."""
    nat = np.asarray(nat)
    width_per_struct = np.asarray(width_per_struct)
    out = []
    for w in np.unique(width_per_struct):
        w = int(w)
        sel = np.where(width_per_struct == w)[0]
        order = sel[np.argsort(nat[sel], kind="stable")]
        cur, cur_nat = [], 0
        for j in order:
            jn = int(nat[j])
            if cur and ((cur_nat + jn) * w > slot_budget or len(cur) >= batch_size):
                out.append({"structure_ind": cur, "width": w})
                cur, cur_nat = [], 0
            cur.append(int(j))
            cur_nat += jn
        if cur:
            out.append({"structure_ind": cur, "width": w})
    return out


def _bucket_by_atoms(recs, max_n_buckets, max_padding_fraction, hard_cap=32):
    """Group per-batch records into buckets sharing ``max_nat``. Mirrors the dataset bucketing
    overhead sweep but on the ATOM metric (the width is already fixed per band). Adds a ``+1``
    fake atom and ``+1`` fake structure to a bucket whenever it pads atoms. ``recs`` is a list
    of ``{"pos": list[int], "n_atoms": int, "n_structures": int}``. ``hard_cap`` bounds the number
    of nat buckets (the per-width-group share of the total compile budget). Returns
    ``list[(bucket_recs, max_nat, max_nstruct)]``. Pure (numpy only)."""
    recs = sorted(recs, key=lambda r: r["n_atoms"], reverse=True)
    n = len(recs)
    if n == 0:
        return []
    cap = max(1, min(n, hard_cap))
    if max_n_buckets == "auto":
        total_real = sum(r["n_atoms"] for r in recs) or 1
        k = cap
        for kk in range(1, cap + 1):
            parts = np.array_split(np.arange(n), kk)
            padded = sum(max(recs[i]["n_atoms"] for i in p) * len(p) for p in parts)
            if padded / total_real - 1.0 <= max_padding_fraction:
                k = kk
                break
    else:
        k = max(1, min(int(max_n_buckets), n, hard_cap))
    out = []
    for p in np.array_split(np.arange(n), k):
        bucket = [recs[i] for i in p]
        max_nat = max(r["n_atoms"] for r in bucket)
        max_nstruct = max(r["n_structures"] for r in bucket)
        if any(r["n_atoms"] != max_nat for r in bucket):
            max_nat += 1
            max_nstruct += 1
        out.append((bucket, max_nat, max_nstruct))
    return out


def _assemble_batches(
    idx, nat, width_per_struct, batch_size, slot_budget, atom_overhead, max_n_buckets, nat_hard_cap=32
):
    """Pack structures (by width group) into batches and apply the within-width nat bucketing
    with the given ``atom_overhead`` budget, capped at ``nat_hard_cap`` nat buckets per width group
    (the per-group share of the compile budget). Returns the batch dict list (indices mapped
    through ``idx`` to original positions). Pure (numpy only)."""
    raw = pack_structures_elastic(nat, width_per_struct, batch_size, slot_budget)
    batches = []
    for w in sorted({rb["width"] for rb in raw}):
        recs = [
            {
                "pos": rb["structure_ind"],
                "n_atoms": int(sum(int(nat[p]) for p in rb["structure_ind"])),
                "n_structures": len(rb["structure_ind"]),
            }
            for rb in raw
            if rb["width"] == w
        ]
        for bucket, max_nat, max_nstruct in _bucket_by_atoms(
            recs, max_n_buckets, atom_overhead, hard_cap=nat_hard_cap
        ):
            for r in bucket:
                batches.append(
                    {
                        "structure_ind": [int(idx[p]) for p in r["pos"]],
                        "max_nat": int(max_nat),
                        "max_nstruct": int(max_nstruct),
                        "max_neigh": int(w),
                    }
                )
    return batches


def _plan_net_shapes(batches, real_total):
    """(net neighbor-padding fraction, #distinct (max_nat, max_neigh) shapes) for a batch list."""
    slots = sum(b["max_nat"] * b["max_neigh"] for b in batches)
    net = (1.0 - real_total / slots) if slots else 0.0
    shapes = len({(b["max_nat"], b["max_neigh"]) for b in batches})
    return net, shapes


def plan_dense_batches(
    nat,
    max_neigh,
    n_neigh,
    *,
    batch_size,
    slot_budget="auto",
    n_neigh_buckets="auto",
    net_padding=0.15,
    max_shapes=64,
    max_neigh_cap=None,
    max_n_buckets="auto",
):
    """Full dense batch plan (pure, numpy only). Returns ``(batches, dropped)``:

    - ``batches``: ``list[{"structure_ind": list[int], "max_nat": int, "max_nstruct": int,
      "max_neigh": int}]`` — indices into the original arrays; ``max_nat``/``max_nstruct``
      include the ``+1`` fake atom/structure when the bucket pads atoms; ``max_neigh`` is the
      adaptive bucket's reshape width. ``max_nat`` is the per-batch TOTAL atom count (the flat
      atom axis, = ``N_ATOMS_BATCH_TOTAL``), so downstream ``n_bonds == max_nat * max_neigh``.
    - ``dropped``: original indices removed by ``max_neigh_cap`` (empty when the cap is off).

    ``net_padding`` is the UNIFIED target on the **net neighbor-padding fraction**
    ``1 - Σn_neigh / Σ(max_nat·max_neigh)`` (the perf-relevant metric — total dense slots; in the
    dense layout a fake atom occupies ``width`` neighbor slots so the atom/neighbor axes are
    bound). ``max_shapes`` is a HARD cap on the number of distinct ``(max_nat, max_neigh)`` shapes
    (== XLA recompiles) — the compile-budget control.

    **Adaptive split:** sweep the number of width buckets K; for each, give the within-width
    ``nat`` bucketing exactly the slack budget that lands net at ``net_padding``, build, and
    measure (net, shapes). Pick the plan that meets the net target with the FEWEST shapes (so a
    larger ``batch_size`` — where fake atoms vanish — naturally spends the whole budget on width
    and uses fewer widths). If no plan meets the target within ``max_shapes``, the compile cap
    wins: take the lowest-padding plan that fits (the caller logs the shortfall). ``n_neigh_buckets``
    (explicit width K) and ``max_n_buckets`` (explicit nat-bucket cap) override the auto search."""
    nat = np.asarray(nat)
    max_neigh = np.asarray(max_neigh)
    n_neigh = np.asarray(n_neigh)
    idx = np.arange(nat.shape[0])
    dropped = []
    if max_neigh_cap is not None:
        keep = max_neigh <= int(max_neigh_cap)
        dropped = idx[~keep].tolist()
        idx, nat, max_neigh, n_neigh = idx[keep], nat[keep], max_neigh[keep], n_neigh[keep]
    if nat.size == 0:
        return [], dropped
    if slot_budget in ("auto", None):
        slot_budget = auto_slot_budget(nat, max_neigh, batch_size)
    slot_budget = max(int(slot_budget), 1)

    real_total = float(n_neigh.sum())
    vals, inv, dp, bnd, M, cap = _max_neigh_width_dp(nat, max_neigh, max_shapes)

    def overhead_for(width_pad):
        return max(0.0, (1.0 - width_pad) / (1.0 - net_padding) - 1.0)

    # Explicit width-K override: skip the search, give nat the slack budget for net_padding.
    if n_neigh_buckets != "auto":
        K = max(1, min(int(n_neigh_buckets), M))
        widths = _widths_for_k(vals, inv, bnd, M, K)
        wpad = 1.0 - real_total / float((nat * widths).sum())
        nat_cap = max(1, max_shapes // K)
        batches = _assemble_batches(
            idx, nat, widths, batch_size, slot_budget, overhead_for(wpad), max_n_buckets, nat_cap
        )
        return batches, dropped

    # Adaptive search over width-bucket count K (1..max_shapes distinct widths). For each K the
    # remaining compile budget goes to the nat bucketing (nat_cap = max_shapes // K per width
    # group), so total shapes <= max_shapes BY CONSTRUCTION. Pick the plan meeting the net target
    # with the fewest shapes; else the lowest-padding plan within the cap.
    best_meet = None  # (shapes, batches): meets net target, min shapes
    best_fit = None   # (net, shapes, batches): min net within the compile cap
    for K in range(1, cap + 1):
        s0 = dp[K, M]
        if not np.isfinite(s0):
            continue
        wpad = 1.0 - real_total / s0 if s0 else 0.0
        nat_cap = max(1, max_shapes // K)
        widths = _widths_for_k(vals, inv, bnd, M, K)
        batches = _assemble_batches(
            idx, nat, widths, batch_size, slot_budget, overhead_for(wpad), max_n_buckets, nat_cap
        )
        net, shapes = _plan_net_shapes(batches, real_total)
        if best_fit is None or net < best_fit[0]:
            best_fit = (net, shapes, batches)
        if net <= net_padding + 1e-9 and (best_meet is None or shapes < best_meet[0]):
            best_meet = (shapes, batches)

    if best_meet is not None:
        return best_meet[1], dropped
    return best_fit[2], dropped


def dense_padding_fractions(padding_stats):
    """Padded-of-total fractions for the dense layout from a ``padding_stats`` dict
    (keys ``pad_nneigh``/``nreal_neigh``/``pad_nat``/``nreal_atoms``/``pad_nstruct``/
    ``nreal_struc``). Returns ``{"neigh","atoms","struct"}`` in ``[0, 1]``."""

    def frac(pad, real):
        tot = pad + real
        return (pad / tot) if tot else 0.0

    return {
        "neigh": frac(padding_stats["pad_nneigh"], padding_stats["nreal_neigh"]),
        "atoms": frac(padding_stats["pad_nat"], padding_stats["nreal_atoms"]),
        "struct": frac(padding_stats["pad_nstruct"], padding_stats["nreal_struc"]),
    }
