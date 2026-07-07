"""Select subcommand: pick N structures from a candidate pool.

Algorithm: a strategy yields atoms in preferred order; the master loop walks
that iterator, accumulating per-structure pick counts. Two hard guarantees:
  1. Per-element stratification floor over the selected structures' atoms
     (default ON, floor = D+1 from the UQ artifact — strict floor for
     non-degenerate per-element covariance). If --uq is omitted, D is
     inferred from the per-atom `features` column when present (e.g. from
     `grace_uq predict --save-features`); otherwise the user must pass
     --min-per-element or --no-element-stratified.
  2. Exactly N structures in the output (top-N by pick count).
If the strategy iterator is exhausted before both guarantees are met,
:func:`_repair_stratification` swaps low-pick-count structures for donor
structures rich in under-represented elements.
"""

from __future__ import annotations

import argparse
import logging
import os
from collections import Counter
from typing import Iterator

import numpy as np
import pandas as pd
from tqdm import tqdm

from tensorpotential.data.databuilder import symbols_to_indices
from tensorpotential.uq import constants as uq_constants
from tensorpotential.uq.cli._common import (
    apply_master_thread_caps,
    format_elem_label,
    is_extxyz_path,
    load_dataset_any,
    load_element_map_from_savedmodel,
    read_artifact_metadata,
    resolve_threads_per_worker,
    save_dataset_any,
)
from tensorpotential.uq.cli.strategies import get_strategy, list_strategies

log = logging.getLogger("grace_uq.select")


# Per-atom prediction artifacts produced by `grace_uq predict` plus the
# select-only `n_atoms_selected` book-keeping column. These are dropped from
# the saved selection because the goal is a structure list for retraining,
# not a copy of the (often large) prediction artifacts.
_PREDICTION_COLUMNS = (
    "energy_predicted",
    "forces_predicted",
    "stress_predicted",
    "gamma",
    "sigma",
    uq_constants.FEATURES,
    "n_atoms_selected",
)

# Selection-summary table layout (chosen so up to 8-digit values with comma
# separators — i.e. ~99,999,999 atoms — fit cleanly).
_SUMMARY_BAR_WIDTH = 40
_SUMMARY_LBL_W = 10
_SUMMARY_NUM_W = 12
_SUMMARY_LABEL_COL_W = 38

# `_collect_top_n` performance knobs. The defaults pair: floor-check cadence
# is at least one cycle per 256 atoms (or per `n_structures`, whichever is
# larger), and after N unique structures are hit the loop processes at most
# `n_structures * 10` more atoms before deferring to `_repair_stratification`
# — otherwise an unreachable-from-top-N floor would force exhausting the
# whole iterator.
_FLOOR_CHECK_MIN_INTERVAL = 256
_POST_N_ATOM_BUDGET_MULT = 10

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")


# ---------------------------------------------------------------------------
# Atom pool construction (vectorize per-structure data into flat arrays)
# ---------------------------------------------------------------------------


def _atomic_numbers_to_indices(
    numbers: np.ndarray, element_map: list[str] | None
) -> np.ndarray:
    """Map ASE atomic numbers to model element indices via the artifact's element_map.

    If element_map is None, return atomic numbers unchanged (rare path; the
    artifact should always include an element_map). Unknown elements map to
    ``-1`` so they're filtered out of stratification floors.
    """
    if element_map is None:
        return numbers.astype(np.int32, copy=False)
    from ase.data import chemical_symbols

    sym_to_idx = {sym: idx for idx, sym in enumerate(element_map)}
    symbols = (chemical_symbols[int(z)] for z in numbers)
    return symbols_to_indices(symbols, sym_to_idx, default=-1)


def _build_atom_pool(
    df: pd.DataFrame,
    element_map: list[str] | None,
    *,
    need_features: bool,
) -> dict:
    """Flatten per-row arrays into per-atom struct-of-arrays.

    Required columns on `df`:
      - ase_atoms                       (for atomic numbers)
      - gamma                           (for *-extrap strategies)
    Optional:
      - features                        (for fps-* strategies)
    """
    if "ase_atoms" not in df.columns:
        raise SystemExit(
            "select requires `ase_atoms` column on the predicted dataset."
        )
    if "gamma" not in df.columns:
        raise SystemExit(
            "select requires `gamma` column. Re-run `grace_uq predict` on this dataset."
        )
    struct_id_chunks, elem_chunks, gamma_chunks = [], [], []
    feat_chunks = [] if need_features else None
    df = df.reset_index(drop=False).rename(columns={"index": "_orig_index"})
    for sid, row in enumerate(df.itertuples(index=False)):
        at = getattr(row, "ase_atoms")
        gamma = np.asarray(getattr(row, "gamma"))
        nat = len(at)
        if len(gamma) != nat:
            raise SystemExit(
                f"row {sid}: len(gamma)={len(gamma)} but n_atoms={nat}"
            )
        struct_id_chunks.append(np.full(nat, sid, dtype=np.int64))
        elem_chunks.append(
            _atomic_numbers_to_indices(at.get_atomic_numbers(), element_map)
        )
        gamma_chunks.append(gamma.astype(np.float32, copy=False))
        if need_features:
            feats = getattr(row, uq_constants.FEATURES, None)
            if feats is None:
                raise SystemExit(
                    f"row {sid}: --strategy fps-* needs per-atom `features`. "
                    "Re-run `grace_uq predict --save-features`."
                )
            feats = np.asarray(feats)
            if feats.shape[0] != nat:
                raise SystemExit(
                    f"row {sid}: features shape {feats.shape} but n_atoms={nat}"
                )
            feat_chunks.append(feats.astype(np.float32, copy=False))
    pool = {
        "struct_id": np.concatenate(struct_id_chunks),
        "element": np.concatenate(elem_chunks),
        "gamma": np.concatenate(gamma_chunks),
        "_orig_index": df["_orig_index"].to_numpy(),
    }
    if need_features:
        pool[uq_constants.FEATURES] = np.vstack(feat_chunks)
    return pool


def _structure_element_counts(df: pd.DataFrame, element_map: list[str] | None):
    """Per-structure per-element atom counts. Returns int matrix [N_structs, n_elem]."""
    n_elem = len(element_map) if element_map is not None else 119
    counts = np.zeros((len(df), n_elem), dtype=np.int64)
    for sid, at in enumerate(df["ase_atoms"]):
        idx = _atomic_numbers_to_indices(at.get_atomic_numbers(), element_map)
        for e in idx:
            if 0 <= e < n_elem:
                counts[sid, e] += 1
    return counts


# ---------------------------------------------------------------------------
# Strategy-iterator consumer
# ---------------------------------------------------------------------------


def _topk_from_counts(
    pick_counts: dict[int, int], gamma_max_per_struct: dict[int, float], k: int
) -> list[int]:
    """Top-k struct_ids by (pick_count, max-gamma-of-picked-atoms)."""
    if k >= len(pick_counts):
        ranked = list(pick_counts.keys())
    else:
        ranked = sorted(
            pick_counts,
            key=lambda s: (pick_counts[s], gamma_max_per_struct.get(s, 0.0), -s),
            reverse=True,
        )[:k]
    return ranked


def _collect_top_n(
    atom_iter: Iterator[int],
    atom_pool: dict,
    struct_elem_counts: np.ndarray,
    *,
    n_structures: int,
    floors: dict[int, int] | None,
) -> tuple[list[int], dict[int, int], dict[int, float]]:
    """Walk the strategy iterator; stop when N structures + floors are met.

    The strategy generator (e.g. ``fps-extrap``) decides the order in which
    atoms are yielded — this function only counts and stops; it does not make
    selection decisions of its own.

    Returns (top_struct_ids, pick_counts, gamma_max_per_struct).
    """
    struct_ids = atom_pool["struct_id"]
    gammas = atom_pool["gamma"]

    pick_counts: dict[int, int] = Counter()
    gamma_max: dict[int, float] = {}
    top: list[int] = []

    bar = tqdm(total=n_structures, desc="select", unit="struct", dynamic_ncols=True)
    atoms_seen = 0
    check_every = max(_FLOOR_CHECK_MIN_INTERVAL, n_structures)
    post_n_budget = n_structures * _POST_N_ATOM_BUDGET_MULT
    last_check = 0
    n_structs_hit_at = 0  # 0 == "not yet"; truthy once N structures are hit
    try:
        for atom_idx in atom_iter:
            atoms_seen += 1
            sid = int(struct_ids[atom_idx])
            new_struct = sid not in pick_counts
            pick_counts[sid] += 1
            g = float(gammas[atom_idx])
            if g > gamma_max.get(sid, -np.inf):
                gamma_max[sid] = g
            if new_struct and len(pick_counts) <= n_structures:
                bar.update(1)
            if atoms_seen % 1024 == 0:
                bar.set_postfix_str(f"atoms={atoms_seen}", refresh=False)

            if len(pick_counts) < n_structures:
                continue
            if not n_structs_hit_at:
                n_structs_hit_at = atoms_seen
            if floors is None:
                top = _topk_from_counts(pick_counts, gamma_max, n_structures)
                break
            if (atoms_seen - last_check) < check_every:
                continue
            last_check = atoms_seen
            top = _topk_from_counts(pick_counts, gamma_max, n_structures)
            out_elem_counts = struct_elem_counts[top].sum(axis=0)
            if all(out_elem_counts[e] >= floor for e, floor in floors.items()):
                break
            if (atoms_seen - n_structs_hit_at) >= post_n_budget:
                log.info(
                    "_collect_top_n: exceeded %d-atom budget after hitting N=%d "
                    "structures with floors still unmet by current top-N; "
                    "stopping iteration and deferring to stratification repair.",
                    post_n_budget,
                    n_structures,
                )
                break
        else:
            # iterator exhausted; recompute top from whatever we have
            top = _topk_from_counts(pick_counts, gamma_max, n_structures)
        bar.set_postfix_str(f"atoms={atoms_seen}", refresh=True)
    finally:
        bar.close()

    return top, dict(pick_counts), gamma_max


def _repair_stratification(
    top: list[int],
    pick_counts: dict[int, int],
    struct_elem_counts: np.ndarray,
    floors: dict[int, int],
    n_total_structs: int,
):
    """Swap low-pick structures in `top` for donors rich in deficit elements."""
    floor_keys = np.array(sorted(floors), dtype=np.int64)
    floor_vals = np.array([floors[int(e)] for e in floor_keys], dtype=np.int64)

    # Work on a mutable copy so we preserve the caller's pick-count ordering
    # across swaps (downstream consumers index `df.iloc[top]` and zip with
    # `n_atoms_selected`, both of which assume `top`'s order is meaningful).
    top = list(top)
    top_set = set(top)
    out_counts = struct_elem_counts[top].sum(axis=0).astype(np.int64)
    n_unmet_initial = int(np.sum(out_counts[floor_keys] < floor_vals))
    if n_unmet_initial == 0:
        return top

    all_struct_ids = np.arange(n_total_structs, dtype=np.int64)
    outside = np.setdiff1d(all_struct_ids, np.fromiter(top_set, dtype=np.int64))
    if len(outside) == 0:
        log.warning("stratification repair: no donor structures available")
        return top

    bar = tqdm(
        total=n_unmet_initial,
        desc="stratify-repair",
        unit="elem",
        dynamic_ncols=True,
    )
    bar.set_postfix_str(f"unmet={n_unmet_initial}", refresh=False)

    # Each successful swap strictly reduces the total deficit (we only accept
    # gain > 0 swaps); the upper bound on swaps is therefore the initial total
    # deficit-atom count. `*4` is a paranoid headroom factor.
    max_swaps = max(int(n_unmet_initial * 4), len(top_set))
    try:
        for _ in range(max_swaps):
            sub = out_counts[floor_keys]
            unmet_mask = sub < floor_vals
            n_unmet = int(np.sum(unmet_mask))
            if n_unmet == 0:
                break
            deficit_elems = floor_keys[unmet_mask]
            min_floor_or_curr = np.minimum(floor_vals, sub)

            donor_scores = struct_elem_counts[np.ix_(outside, deficit_elems)].sum(
                axis=1
            )
            if int(donor_scores.max()) == 0:
                log.warning(
                    "stratification repair: %d unmet element(s) cannot be filled "
                    "from any unselected structure (floor too high for the "
                    "available pool); leaving residual deficits.",
                    n_unmet,
                )
                break
            # Donors in best-score-first order so a less-than-best donor can
            # still drive progress when the best one has no compatible victim.
            donor_order = np.argsort(donor_scores)[::-1]

            # Pre-rank victims by deficit-element atom contribution ascending —
            # lower contribution = larger swap gain for the same donor.
            top_arr = np.fromiter(top_set, dtype=np.int64)
            victim_scores = struct_elem_counts[np.ix_(top_arr, deficit_elems)].sum(
                axis=1
            )
            victim_order = np.argsort(victim_scores)

            swap_done = False
            for d_pos in donor_order:
                d_score = int(donor_scores[d_pos])
                if d_score == 0:
                    break
                donor = int(outside[d_pos])
                donor_counts = struct_elem_counts[donor].astype(np.int64)
                for v_pos in victim_order:
                    if int(victim_scores[v_pos]) >= d_score:
                        break  # remaining victims yield gain ≤ 0
                    sid = int(top_arr[v_pos])
                    cand_sub = (
                        out_counts - struct_elem_counts[sid] + donor_counts
                    )[floor_keys]
                    if np.all(cand_sub >= floor_vals) or np.all(
                        cand_sub >= min_floor_or_curr
                    ):
                        out_counts = (
                            out_counts - struct_elem_counts[sid] + donor_counts
                        )
                        top[top.index(sid)] = donor
                        top_set.discard(sid)
                        top_set.add(donor)
                        outside = np.setdiff1d(
                            all_struct_ids, np.fromiter(top_set, dtype=np.int64)
                        )
                        swap_done = True
                        break
                if swap_done:
                    break

            if not swap_done:
                log.warning(
                    "stratification repair: no strictly-improving swap available; "
                    "leaving residual deficits."
                )
                break

            new_unmet = int(np.sum(out_counts[floor_keys] < floor_vals))
            filled = n_unmet - new_unmet
            if filled > 0:
                bar.update(filled)
            bar.set_postfix_str(f"unmet={new_unmet}", refresh=False)
        else:
            log.warning(
                "stratification repair: hit max_swaps=%d without convergence.",
                max_swaps,
            )
    finally:
        bar.close()

    return top


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------


def _infer_D_from_features(df: pd.DataFrame) -> int | None:
    """Read the per-atom feature dimension D from the first non-empty `features`
    array in a predicted dataset. Returns None if no usable features found.
    """
    if uq_constants.FEATURES not in df.columns:
        return None
    for feats in df[uq_constants.FEATURES]:
        if feats is None:
            continue
        arr = np.asarray(feats)
        if arr.ndim == 2 and arr.shape[0] > 0:
            return int(arr.shape[1])
    return None


def _maybe_run_predict(args):
    """If --dataset given without --predicted, call predict_main and use its output."""
    from tensorpotential.uq.cli.predict import predict_main

    if not args.model:
        raise SystemExit(
            "--model is required when running prediction inside select "
            "(no --predicted given)."
        )
    output = args.predicted or os.path.join(
        os.path.dirname(args.output) or ".", ".select_predicted.pkl.gz"
    )
    pred_argv = [
        "--model",
        args.model,
        "--dataset",
        args.dataset,
        "--output",
        output,
        "--save-features",  # always needed for fps-*; harmless for random-*
        "--n-workers",
        str(args.n_workers),
        "--gpus",
        args.gpus,
        "--threads-per-worker",
        str(args.threads_per_worker),
    ]
    if args.checkpoint:
        pred_argv += ["--checkpoint", args.checkpoint]
    if args.artifact:
        pred_argv += ["--artifact", args.artifact]
    if args.verbose:
        pred_argv += ["--verbose"]
    log.info("Running prediction in select; output → %s", output)
    rc = predict_main(pred_argv)
    if rc != 0:
        raise RuntimeError("prediction step failed")
    return output


def select_main(argv=None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)

    if args.uq:
        args.uq = os.path.abspath(args.uq)
    args.output = os.path.abspath(args.output)
    if args.predicted:
        args.predicted = os.path.abspath(args.predicted)
    if args.dataset:
        args.dataset = os.path.abspath(args.dataset)
    if args.model:
        args.model = os.path.abspath(args.model)
    if args.checkpoint:
        args.checkpoint = os.path.abspath(args.checkpoint)
    if args.artifact:
        args.artifact = os.path.abspath(args.artifact)

    threads_int = resolve_threads_per_worker(args.threads_per_worker, args.n_workers)
    apply_master_thread_caps(threads_int)

    # Resolve predicted path
    predicted_path = args.predicted
    if predicted_path is None:
        if not args.dataset:
            raise SystemExit("either --predicted or --dataset must be given.")
        predicted_path = _maybe_run_predict(args)

    log.info("Loading predicted dataset: %s", predicted_path)
    df = load_dataset_any(predicted_path)

    # yaml-mode users pass --artifact for predict; it IS the UQ npz, so reuse it.
    uq_path = args.uq or args.artifact
    if uq_path:
        if not args.uq:
            log.info("--uq not given; reusing --artifact (%s) as the UQ source.", uq_path)
        meta = read_artifact_metadata(uq_path)
        D = meta["D"]
        element_map = meta["element_map"]
        n_elements = meta["n_elements"]
    elif args.model:
        element_map = load_element_map_from_savedmodel(args.model)
        if element_map is None:
            raise SystemExit(
                f"--uq not given and could not derive element_map from --model "
                f"({args.model}). Pass --uq <gmm_artifacts.npz> or use a "
                "SavedModel directory containing metadata.yaml."
            )
        D = _infer_D_from_features(df)
        n_elements = len(element_map)
        d_msg = (
            f"inferred D={D} from `features` in predicted dataset"
            if D is not None
            else "D unknown — pass --min-per-element or --no-element-stratified"
        )
        log.info(
            "Derived element_map from %s/metadata.yaml (%d elements); %s.",
            args.model,
            n_elements,
            d_msg,
        )
    else:
        raise SystemExit("either --uq or --model must be given.")

    need_features = args.strategy.startswith("fps-")
    pool = _build_atom_pool(df, element_map, need_features=need_features)
    log.info("Atom pool: %d atoms in %d structures", len(pool["struct_id"]), len(df))

    if len(pool["struct_id"]) == 0:
        raise SystemExit(
            "candidate atom pool is empty — nothing to select from. "
            "Check that --predicted/--dataset has structures with atoms and a `gamma` column."
        )
    elem = pool["element"]
    valid_mask = elem >= 0
    n_unknown = int(elem.size - valid_mask.sum())
    if n_unknown > 0:
        log.warning(
            "%d candidate atoms have an element not in the model's element_map "
            "and will be ignored by the stratification floor.",
            n_unknown,
        )

    if args.n_structures > len(df):
        log.warning(
            "--n-structures=%d exceeds candidate pool size (%d); will return at most %d.",
            args.n_structures,
            len(df),
            len(df),
        )

    floors = None
    floor_target: int | None = None
    if args.element_stratified:
        if args.min_per_element is not None:
            floor_val = args.min_per_element
        elif D is not None:
            floor_val = D + 1
        else:
            raise SystemExit(
                "Stratification is ON but D is unknown (no --uq given). "
                "Pass --min-per-element <int> or --no-element-stratified."
            )
        floor_target = floor_val
        # Per-element clamp: an element's floor cannot exceed the atoms
        # actually available in the candidate pool. Without this clamp an
        # unreachable floor would force `_collect_top_n` to exhaust the entire
        # iterator (then all strategies converge to the same answer).
        atoms_per_e = np.bincount(elem[valid_mask], minlength=n_elements)
        present = np.flatnonzero(atoms_per_e > 0)
        eff = np.minimum(floor_val, atoms_per_e[present])
        floors = {int(e): int(v) for e, v in zip(present, eff)}
        clamped_idx = present[eff < floor_val]
        log.info(
            "Stratification ON: floor=%d atoms/element across %d present element(s) "
            "(skipped %d absent element(s))",
            floor_val,
            len(floors),
            n_elements - len(present),
        )
        if len(clamped_idx) > 0:
            clamped = [
                f"{format_elem_label(int(e), element_map)}:{int(atoms_per_e[e])}"
                for e in clamped_idx
            ]
            log.warning(
                "Clamped floor for %d element(s) below %d due to limited supply "
                "in the candidate pool: %s",
                len(clamped),
                floor_val,
                ", ".join(clamped),
            )

    rng = np.random.default_rng(args.seed)
    strat_kwargs = dict(
        gamma_min=args.gamma_min,
        gamma_max=args.gamma_max,
        fps_max_pool=args.fps_max_pool,
        rng=rng,
    )
    strategy_fn = get_strategy(args.strategy)
    atom_iter = strategy_fn(pool, **strat_kwargs)

    struct_elem_counts = _structure_element_counts(df, element_map)
    top, pick_counts, _ = _collect_top_n(
        atom_iter,
        pool,
        struct_elem_counts,
        n_structures=args.n_structures,
        floors=floors,
    )

    if floors is not None:
        top = _repair_stratification(
            top, pick_counts, struct_elem_counts, floors, len(df)
        )

    # Top-up to N if we still don't have enough (e.g. pool too small).
    if len(top) < args.n_structures:
        log.warning(
            "Only %d/%d structures available; topping up from unselected pool by gamma_max.",
            len(top),
            args.n_structures,
        )
        outside = [s for s in range(len(df)) if s not in top]
        gamma_per = df["gamma"].apply(lambda g: float(np.max(g)) if len(g) else 0.0).to_numpy()
        outside_sorted = sorted(outside, key=lambda s: -gamma_per[s])
        for s in outside_sorted:
            if len(top) >= args.n_structures:
                break
            top.append(s)

    top = top[: args.n_structures]
    out_df = df.iloc[top].copy()
    out_df["n_atoms_selected"] = [int(pick_counts.get(s, 0)) for s in top]

    if is_extxyz_path(args.output) and "ase_atoms" not in out_df.columns:
        raise SystemExit(
            "extxyz output requires the ase_atoms column on the predicted dataset."
        )
    save_df = out_df.drop(
        columns=[c for c in _PREDICTION_COLUMNS if c in out_df.columns]
    )
    save_dataset_any(save_df, args.output, drop_ase_atoms=False)

    log.info(
        "Wrote %s (%d structures, total picked atoms=%d)",
        args.output,
        len(out_df),
        int(out_df["n_atoms_selected"].sum()),
    )
    _print_selection_summary(
        out_df,
        top,
        struct_elem_counts,
        element_map,
        floors,
        strategy=args.strategy,
        floor_target=floor_target,
    )
    return 0


def _print_selection_summary(
    out_df: pd.DataFrame,
    top: list[int],
    struct_elem_counts: np.ndarray,
    element_map: list[str] | None,
    floors: dict[int, int] | None,
    *,
    strategy: str,
    floor_target: int | None = None,
):
    """Human-readable summary of the selected structures (post-write).

    ``floor_target`` is the user's *requested* uniform floor (e.g. ``D+1``);
    ``floors`` is the per-element clamp (capped at the available supply).
    A clamped element where ``sel == floors[e] < floor_target`` is reported
    as ``DATA-LIMITED`` rather than ``OK`` so data-scarcity isn't hidden.
    """
    n_structs = len(out_df)
    if n_structs == 0:
        print()
        print("=" * 78)
        print(f"  Selection summary — strategy={strategy!r}: no structures selected.")
        print("=" * 78)
        print()
        return
    n_atoms_per_struct = struct_elem_counts[top].sum(axis=1)
    total_atoms = int(n_atoms_per_struct.sum())
    n_picked = int(out_df["n_atoms_selected"].sum())

    n_cand_structs = struct_elem_counts.shape[0]
    n_cand_atoms = int(struct_elem_counts.sum())

    cand_frac = f" ({total_atoms / n_cand_atoms:.1%} of candidate atoms)" if n_cand_atoms else ""
    pick_frac = f" ({n_picked / total_atoms:.1%} of selected-structure atoms)" if total_atoms else ""

    LBL_W = _SUMMARY_LBL_W
    NUM_W = _SUMMARY_NUM_W
    LABEL_COL_W = _SUMMARY_LABEL_COL_W

    print()
    print("=" * 78)
    print(
        f"  Selection summary — strategy={strategy!r}  "
        f"({n_structs:,} / {n_cand_structs:,} structures selected)"
    )
    print("=" * 78)
    print(
        f"  {'candidate pool':<{LABEL_COL_W}}: "
        f"{n_cand_structs:>{NUM_W},d} structures, {n_cand_atoms:>{NUM_W},d} atoms"
    )
    print(
        f"  {'selected':<{LABEL_COL_W}}: "
        f"{n_structs:>{NUM_W},d} structures, {total_atoms:>{NUM_W},d} atoms{cand_frac}"
    )
    print(
        f"  {'selected atoms/structure':<{LABEL_COL_W}}: "
        f"min={int(n_atoms_per_struct.min()):>{NUM_W},d}, "
        f"mean={n_atoms_per_struct.mean():>{NUM_W},.1f}, "
        f"median={int(np.median(n_atoms_per_struct)):>{NUM_W},d}, "
        f"max={int(n_atoms_per_struct.max()):>{NUM_W},d}"
    )
    print(
        f"  {'atoms picked by strategy':<{LABEL_COL_W}}: {n_picked:>{NUM_W},d}{pick_frac}"
    )

    cand_totals = struct_elem_counts.sum(axis=0)
    sel_totals = struct_elem_counts[top].sum(axis=0)
    present = np.flatnonzero(cand_totals > 0)
    if len(present) > 0:
        print()
        print("  per-element atom counts (selected / candidate, fraction selected):")
        rows = sorted(
            ((int(cand_totals[e]), int(sel_totals[e]), e) for e in present),
            reverse=True,
        )
        floor_lookup = floors or {}
        data_limited: list[tuple[str, int, int]] = []  # (label, available, target)
        unmet: list[tuple[str, int, int]] = []  # (label, selected, floor_eff)
        for cand, sel, e in rows:
            label = format_elem_label(e, element_map)
            frac = sel / cand if cand else 0.0
            bar_len = int(round(_SUMMARY_BAR_WIDTH * frac))
            bar = "█" * bar_len + "·" * (_SUMMARY_BAR_WIDTH - bar_len)
            floor_str = ""
            if e in floor_lookup:
                f_eff = floor_lookup[e]
                f_show = floor_target if floor_target is not None else f_eff
                if sel >= f_show:
                    status = "OK"
                elif sel >= f_eff:
                    status = "DATA-LIMITED"
                    data_limited.append((label, cand, f_show))
                else:
                    status = "UNMET"
                    unmet.append((label, sel, f_eff))
                floor_str = f"  [floor={f_show:>6,d} {status}]"
            print(
                f"    {label:>{LBL_W}s}  "
                f"{sel:>{NUM_W},d} / {cand:<{NUM_W},d}  "
                f"{bar}  {frac:>6.1%}{floor_str}"
            )

        if data_limited:
            print()
            print(
                "  Recommendation: the candidate pool is short on the following "
                "element(s);"
            )
            print(
                "  add more structures containing them to reach the "
                f"floor={floor_target} target:"
            )
            for label, avail, target in data_limited:
                print(f"    - {label}: only {avail} atoms available (need {target})")
        if unmet:
            print()
            print(
                "  WARNING: the following element(s) are below their (clamped) "
                "floor — strategy"
            )
            print(
                "  iterator was exhausted before they could be filled; consider "
                "lowering --min-per-element"
            )
            print("  or running --no-element-stratified:")
            for label, sel, f_eff in unmet:
                print(f"    - {label}: selected {sel} < {f_eff}")

    # Gamma summary (extrapolation strength of selected structures); use
    # nan-safe reducers so empty-row / NaN-laden gammas don't crash the print.
    if "gamma" in out_df.columns:
        gamma_max_per = np.empty(len(out_df))
        gamma_mean_per = np.empty(len(out_df))
        for i, g in enumerate(out_df["gamma"]):
            if g is not None and len(g):
                gamma_max_per[i] = float(np.nanmax(g))
                gamma_mean_per[i] = float(np.nanmean(g))
            else:
                gamma_max_per[i] = gamma_mean_per[i] = np.nan
        if np.any(np.isfinite(gamma_max_per)):
            print()
            print(
                f"  {'gamma_max per structure':<{LABEL_COL_W}}: "
                f"min={np.nanmin(gamma_max_per):>8.2f}, "
                f"median={np.nanmedian(gamma_max_per):>8.2f}, "
                f"max={np.nanmax(gamma_max_per):>8.2f}"
            )
            print(
                f"  {'gamma_mean per structure':<{LABEL_COL_W}}: "
                f"min={np.nanmin(gamma_mean_per):>8.2f}, "
                f"median={np.nanmedian(gamma_mean_per):>8.2f}, "
                f"max={np.nanmax(gamma_mean_per):>8.2f}"
            )
    print("=" * 78)
    print()


_SELECT_EPILOG = """\
Examples
--------
  # 1. Default: 200 structures, random extrap (gamma >= 1), per-element
  #    stratification ON (floor = D+1 from artifact):
  grace_uq select --uq UQ/gmm_artifacts.npz \\
                  --predicted predicted.pkl.gz \\
                  -n 200 \\
                  --output selected.pkl.gz

  # 2. Diverse extrapolatory picks via Mini-batch FPS in feature space
  #    (requires --save-features in the predicted dataset):
  grace_uq select --uq UQ/gmm_artifacts.npz \\
                  --predicted predicted.pkl.gz \\
                  -n 200 --strategy fps-extrap \\
                  --output selected.pkl.gz

  # 3. Bound the gamma window (skip wildly out-of-distribution atoms):
  grace_uq select --uq UQ/gmm_artifacts.npz \\
                  --predicted predicted.pkl.gz \\
                  -n 200 --strategy fps-extrap \\
                  --gamma-min 1.0 --gamma-max 5.0 \\
                  --output selected.pkl.gz

  # 4. Disable stratification (raw strategy ranking only):
  grace_uq select --uq UQ/gmm_artifacts.npz \\
                  --predicted predicted.pkl.gz \\
                  -n 200 --strategy fps-extrap \\
                  --no-element-stratified \\
                  --output selected.pkl.gz

  # 5. Run prediction inline on a raw dataset, then select (4 GPUs):
  grace_uq select --uq UQ/gmm_artifacts.npz \\
                  --dataset candidates.pkl.gz \\
                  --model saved_model/ \\
                  -n 200 --strategy fps-extrap \\
                  --n-workers 4 --gpus 0,1,2,3 \\
                  --output selected.pkl.gz

  # 6. FPS over all atoms (no gamma filter), capped pool size for memory:
  grace_uq select --uq UQ/gmm_artifacts.npz \\
                  --predicted predicted.pkl.gz \\
                  -n 500 --strategy fps-all \\
                  --fps-max-pool 20000 \\
                  --output selected.pkl.gz

  # 7. Custom stratification floor (atoms per element in output):
  grace_uq select --uq UQ/gmm_artifacts.npz \\
                  --predicted predicted.pkl.gz \\
                  -n 200 --strategy fps-extrap \\
                  --min-per-element 100 \\
                  --output selected.pkl.gz

  # 8. Save as extxyz instead of pkl.gz (extension-driven dispatch):
  grace_uq select --uq UQ/gmm_artifacts.npz \\
                  --predicted predicted.pkl.gz \\
                  -n 200 --strategy fps-extrap \\
                  --output selected.xyz

  # 9. Skip --uq: element_map is read from <model>/metadata.yaml. D is unknown
  #    so stratification floor must be supplied explicitly (or disabled):
  grace_uq select --model saved_model/ \\
                  --predicted predicted.pkl.gz \\
                  -n 200 --strategy fps-extrap \\
                  --min-per-element 100 \\
                  --output selected.pkl.gz
"""


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="grace_uq select",
        description="Select N structures from a candidate pool by an extrapolation/diversity strategy.",
        epilog=_SELECT_EPILOG,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument(
        "--uq",
        default=None,
        help="UQ artifact .npz. Provides D (default stratification floor = D+1) "
        "and the element_map. Optional fallback chain: (1) --artifact (yaml-mode "
        "predict already needs it); (2) <model>/metadata.yaml for element_map "
        "plus D inferred from `features` if --save-features was used. Otherwise "
        "pass --min-per-element or --no-element-stratified.",
    )
    p.add_argument(
        "--predicted",
        help="Pre-predicted dataset (output of `grace_uq predict`).",
    )
    p.add_argument(
        "--dataset",
        help="Raw dataset; if --predicted not given, predict is run first.",
    )
    p.add_argument(
        "--model",
        help="SavedModel dir or model.yaml — required iff --dataset given.",
    )
    p.add_argument("--checkpoint", help="Required iff --model is a model.yaml.")
    p.add_argument("--artifact", help="Required iff --model is a model.yaml.")
    p.add_argument(
        "--output",
        default="selected.pkl.gz",
        help="Output path (extension drives format).",
    )
    p.add_argument(
        "-n",
        "--n-structures",
        type=int,
        required=True,
        help="Number of structures to select (hard guarantee).",
    )
    p.add_argument(
        "--strategy",
        default="random-extrap",
        choices=list_strategies(),
        help="Selection strategy (default: random-extrap).",
    )
    p.add_argument("--gamma-min", type=float, default=1.0)
    p.add_argument(
        "--gamma-max",
        type=float,
        default=None,
        help="Upper gamma threshold (None = no cap).",
    )
    p.add_argument(
        "--no-element-stratified",
        dest="element_stratified",
        action="store_false",
        help="Disable per-element stratification (default: ON).",
    )
    p.set_defaults(element_stratified=True)
    p.add_argument(
        "--min-per-element",
        type=int,
        default=None,
        help="Stratification floor (atoms/element). Default = D+1 from artifact "
        "(strict floor for non-degenerate per-element covariance).",
    )
    p.add_argument(
        "--fps-max-pool",
        type=int,
        default=50_000,
        help="Mini-batch FPS pool cap (controls memory).",
    )
    p.add_argument("--n-workers", type=int, default=4)
    p.add_argument("--gpus", default="0")
    p.add_argument("--threads-per-worker", default="auto")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--verbose", action="store_true")
    return p


if __name__ == "__main__":
    raise SystemExit(select_main())
