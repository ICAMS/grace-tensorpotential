"""Select subcommand: pick N structures from a candidate pool.

Default strategy calm-greedy selects frames directly (max-first greedy with an
additive selection tower, see ``strategies.calm_greedy``) and may return fewer than
N structures: its natural set, plus any frames the element floors add
(:func:`_repair_floors_greedy`: appended below N, replacing the latest picks at N).
fps-extrap takes frames in FPS first-hit order. The other strategies yield atoms in
preferred order; the master loop walks that iterator, accumulating per-structure
pick counts.

Per-element stratification floor over the selected structures' atoms (default ON,
floor = D+1 from the UQ artifact — strict floor for non-degenerate per-element
covariance). If --uq is omitted, D is inferred from the per-atom `features` column
when present (e.g. from `grace_uq predict --save-features`) or from the HAL
sidecars; otherwise pass --min-per-element or --no-element-stratified.

All strategies except calm-greedy return N structures when the pool has them
(top-up by gamma_max). Every --strategy, one part or several, runs through
:func:`_run_parts`; the floors are applied once, to the whole selection, by
:func:`_repair_floors_greedy`.
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

from tensorpotential.uq import constants as uq_constants
from tensorpotential.uq.cli._common import (
    apply_master_thread_caps,
    format_elem_label,
    is_extxyz_path,
    load_dataset_any,
    load_element_map_from_savedmodel,
    parse_n_workers,
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

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")


# ---------------------------------------------------------------------------
# Atom pool construction (vectorize per-structure data into flat arrays)
# ---------------------------------------------------------------------------


def _z_to_index(numbers: np.ndarray, element_map: list[str] | None) -> np.ndarray:
    """Map atomic numbers to model element indices via the artifact's element_map.

    Unknown elements map to ``-1`` (filtered out of the stratification floors); without
    an element_map the atomic numbers are returned unchanged.
    """
    numbers = np.asarray(numbers)
    if element_map is None:
        return numbers.astype(np.int32)
    from ase.data import atomic_numbers

    lut = np.full(119, -1, dtype=np.int32)
    for idx, sym in enumerate(element_map):
        if sym in atomic_numbers:
            lut[atomic_numbers[sym]] = idx
    return lut[numbers.astype(np.int64)]


def _build_atom_pool(
    df: pd.DataFrame,
    element_map: list[str] | None,
    *,
    need_features: bool,
    gamma_window: tuple[float | None, float | None] | None = None,
    strategy: str = "fps-*",
) -> dict:
    """Flatten per-row arrays into per-atom struct-of-arrays.

    Required columns on `df`:
      - ase_atoms                       (for atomic numbers)
      - gamma                           (for *-extrap strategies)
    Optional:
      - features                        (for fps-* strategies)

    ``gamma_window=(gamma_min, gamma_max)`` keeps only the atoms inside the window
    (either bound may be None), filtered per row before the features are stacked, so
    the *-extrap strategies never hold every atom's features.
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
    for sid, row in enumerate(df.itertuples(index=False)):
        at = getattr(row, "ase_atoms")
        gamma = np.asarray(getattr(row, "gamma"))
        nat = len(at)
        if len(gamma) != nat:
            raise SystemExit(
                f"row {sid}: len(gamma)={len(gamma)} but n_atoms={nat}"
            )
        keep = (
            slice(None)
            if gamma_window is None
            else np.flatnonzero(_in_window(gamma, *gamma_window))
        )
        gamma = gamma[keep]
        struct_id_chunks.append(np.full(len(gamma), sid, dtype=np.int64))
        elem_chunks.append(
            _z_to_index(at.get_atomic_numbers()[keep], element_map)
        )
        gamma_chunks.append(gamma.astype(np.float32, copy=False))
        if need_features:
            feats = getattr(row, uq_constants.FEATURES, None)
            if feats is None:
                raise SystemExit(f"row {sid}: " + _features_needed_msg(strategy))
            feats = np.asarray(feats)
            if feats.shape[0] != nat:
                raise SystemExit(
                    f"row {sid}: features shape {feats.shape} but n_atoms={nat}"
                )
            feat_chunks.append(feats[keep].astype(np.float32, copy=False))
    pool = {
        "struct_id": np.concatenate(struct_id_chunks),
        "element": np.concatenate(elem_chunks),
        "gamma": np.concatenate(gamma_chunks),
    }
    if need_features:
        pool[uq_constants.FEATURES] = np.vstack(feat_chunks)
    return pool


def _features_needed_msg(strategy: str) -> str:
    return (
        f"--strategy {strategy} needs per-atom `features`: re-run `grace_uq predict "
        "--save-features`, or use `--strategy random-extrap` (the 0.6.1 default), "
        "which needs only gamma."
    )


# strategies that select whole frames uniformly at random (need no features)


def _parse_strategy_parts(args):
    """``--strategy name[:N] ...`` -> ``args.parts`` [(name, N or None)], in order.

    One part: N from ``name:N`` or ``-n`` (not both). Several parts: N per part, ``-n``
    refused. N is required for fps-* and random-*; calm-greedy without N selects its
    natural set.
    """
    specs = args.strategy if isinstance(args.strategy, list) else [args.strategy]
    known = list_strategies()
    parts = []
    for spec in specs:
        name, sep, num = str(spec).partition(":")
        if name not in known:
            raise SystemExit(f"--strategy: unknown strategy {name!r} in {spec!r}; "
                             f"available: {known}")
        n = None
        if sep:
            try:
                n = int(num)
            except ValueError:
                raise SystemExit(f"--strategy {spec}: N must be an integer") from None
            if n < 1:
                raise SystemExit(f"--strategy {spec}: N must be >= 1")
        parts.append((name, n))
    if args.n_structures is not None and args.n_structures < 1:
        raise SystemExit(f"-n/--n-structures must be >= 1, got {args.n_structures}")
    if len(parts) > 1:
        if args.n_structures is not None:
            raise SystemExit(
                "-n applies to a single --strategy; with several parts give N per part "
                "(e.g. --strategy calm-greedy fps-extrap:25 random-all:25).")
    else:
        name, n = parts[0]
        if n is not None and args.n_structures is not None:
            raise SystemExit(f"give N once: --strategy {name}:{n} or -n {args.n_structures}")
        if n is None:
            n = args.n_structures
        parts = [(name, n)]
        args.n_structures = n
    for name, n in parts:
        if n is None and name != "calm-greedy":
            raise SystemExit(
                f"--strategy {name} needs N ({name}:N, or -n/--n-structures with a single "
                "strategy); only calm-greedy, which stops at its natural set, runs "
                "without it.")
    args.parts = parts
    args.strategy = (parts[0][0] if len(parts) == 1
                     else " ".join(f"{p}:{n}" if n else p for p, n in parts))


def _check_strategy_prerequisites(args, uq_path):
    """What the strategies need that no prediction can supply: checked before any
    prediction runs."""
    if any(p == "calm-greedy" for p, _ in args.parts) and not uq_path:
        raise SystemExit(
            "--strategy calm-greedy (the default) needs --artifact or --uq: the pristine "
            "CALM artifact, which the selection tower starts as a copy of. Pass it, or "
            "use `--strategy random-extrap` (the 0.6.1 default)."
        )


def _top_n_supply(counts: np.ndarray, n: int) -> np.ndarray:
    """Per column: the sum of the n largest entries (atoms of an element that n frames
    can hold at most)."""
    if n >= len(counts):
        return counts.sum(axis=0)
    return -np.partition(-counts, n - 1, axis=0)[:n].sum(axis=0)


def _in_window(gamma, gamma_min, gamma_max):
    mask = np.ones(len(gamma), dtype=bool)
    if gamma_min is not None:
        mask &= gamma >= gamma_min
    if gamma_max is not None:
        mask &= gamma <= gamma_max
    return mask


# ---------------------------------------------------------------------------
# Candidate input (HAL sidecars / pooled candidate columns): no model prediction
# ---------------------------------------------------------------------------

# pool_frames columns holding the candidate-atom data of hal_features.npz
_CAND_COLUMNS = (
    "cand_atom_index",
    "cand_features",
    "cand_gamma",
    "cand_feature_gamma_min",
    "cand_artifact_sha256",
)
# strategies that only look at atoms inside the gamma window (candidate input works)
_WINDOW_STRATEGIES = ("random-extrap", "fps-extrap", "calm-greedy")


def _is_candidate_df(df: pd.DataFrame) -> bool:
    """Candidate columns and no all-atom features: a pooled HAL dataset, or a
    `grace_uq predict --features-gamma-min` output (which also has per-atom gamma).
    With all-atom ``features`` too (--save-features as well), the predicted path is
    used."""
    return "cand_features" in df.columns and uq_constants.FEATURES not in df.columns


def _check_candidate_meta(fgms, shas, *, gamma_min, artifact_path, source):
    """Refuse a gamma window the sidecars cannot cover, candidates of different CALM
    artifacts, or an --artifact other than theirs."""
    fgm = max(float(f) for f in fgms) if len(fgms) else None
    if fgm is not None and (gamma_min is None or gamma_min < fgm):
        raise SystemExit(
            f"--gamma-min {gamma_min} is below feature_gamma_min={fgm:g} of {source}: "
            "atoms below it have no stored features. Re-run `grace_uq predict "
            "--save-features` on the dataset instead and select with --predicted."
        )
    shas = sorted({str(s) for s in shas})
    if len(shas) > 1:
        raise SystemExit(
            f"{source} come from different CALM artifacts (sha256 "
            f"{', '.join((s[:12] + '…') if s else '<none>' for s in shas)}): their "
            "features are not comparable; select from each artifact's walkers separately."
        )
    if artifact_path is None:
        return
    if shas == [""]:
        raise SystemExit(
            f"{source} carry no artifact sha256 (a hand-composed HALCollector whose "
            "pristine model was built in memory, a sidecar written before the collector "
            "hashed GMMUQModel.source_path, or `grace_uq predict --features-gamma-min` "
            "without --artifact), so they cannot be checked against --artifact. HAL "
            "walkers: re-collect with a file-loaded pristine tower (or run_hal / grace_uq "
            "hal). A predict file: re-run `grace_uq predict … --features-gamma-min X "
            "--artifact <the CALM artifact>`. Or select without --artifact "
            "(random-extrap, raw fps-extrap)."
        )
    from tensorpotential.uq.hal.features import file_sha256

    sha = file_sha256(artifact_path)
    bad = sorted({s for s in shas if s != sha})
    if bad:
        raise SystemExit(
            f"--artifact {artifact_path} (sha256 {sha[:12]}…) is not the CALM artifact the "
            f"features of {source} were collected with (sha256 "
            f"{', '.join(b[:12] + '…' for b in bad)}): pass that artifact."
        )


def _elem_count_matrix(species_idx, counts, n_frames, n_elem):
    out = np.zeros((n_frames, n_elem), dtype=np.int64)
    for j, e in enumerate(species_idx):
        if 0 <= e < n_elem:
            out[:, e] += counts[:, j]
    return out


def _candidate_pool_from_hal_dirs(dirs, element_map, gamma_min, gamma_max, artifact_path):
    """Atom pool of the candidate atoms in the gamma window, read from the sidecars only.

    Returns (pool, struct_elem_counts, frame_source [(dir, collection_index)], D,
    {dir: frame byte offsets}). Dirs without a sidecar are skipped with a warning.
    Two passes over the sidecars: gammas first, then the window's features straight
    into one preallocated array (peak memory: the window's features + one walker's).
    """
    from tensorpotential.uq.hal.features import FEATURES_FILE, frame_offsets, read_features

    n_elem = len(element_map) if element_map is not None else 119
    metas, masks, used, offsets = [], [], [], {}
    for d in dirs:
        path = os.path.join(d, FEATURES_FILE)
        frames_path = os.path.join(d, "hal_collected.extxyz")
        if not os.path.exists(frames_path):
            log.warning("%s: no hal_collected.extxyz (the walker collected no frames), "
                        "skipped.", d)
            continue
        if not os.path.exists(path):
            log.warning(
                "%s: no %s, skipped (walker stopped before its first sidecar save, or "
                "run with --feature-gamma-min none: pool it with `grace_uq hal --pool` "
                "and select with --dataset POOL, which runs the prediction).",
                d, FEATURES_FILE)
            continue
        m = read_features(path, keys=("gamma_pristine", "collection", "n_atoms",
                                      "species_Z", "elem_counts"))
        n_coll = len(m["n_atoms"])
        offsets[d] = frame_offsets(frames_path)  # reused when reading the selection
        n_file = len(offsets[d])
        if n_coll > n_file:
            raise SystemExit(f"{d}: {FEATURES_FILE} has {n_coll} collections but "
                             f"hal_collected.extxyz only {n_file} frames: inconsistent walker")
        if n_file > n_coll:
            log.warning("%s: frames %d..%d are not covered by %s (walker stopped between "
                        "two sidecar saves): not selectable", d, n_coll, n_file - 1,
                        FEATURES_FILE)
        metas.append(m)
        used.append(d)
        # The window filters on the sidecar's gamma_pristine, computed in the walker by
        # GMMUQModel (the TF path). Its nearest-centroid search runs over all K_max
        # slots; padded slots hold sentinel centroids (never nearest), so it agrees with
        # the selection scorers, which use the real K like GMMUQArtifactBuilder.update.
        masks.append(np.flatnonzero(_in_window(m["gamma_pristine"], gamma_min, gamma_max)))
    if not used:
        raise SystemExit("--hal-dirs: no walker directory with a hal_features.npz sidecar.")
    dirs = used
    _check_candidate_meta(
        [m["feature_gamma_min"] for m in metas], [m["artifact_sha256"] for m in metas],
        gamma_min=gamma_min, artifact_path=artifact_path, source=f"the {FEATURES_FILE} sidecars",
    )
    dims = {int(m["feature_dim"]) for m in metas if int(m["feature_dim"]) > 0}
    if len(dims) > 1:
        raise SystemExit(f"--hal-dirs: sidecars disagree on the feature dimension {sorted(dims)}")
    D = dims.pop() if dims else None
    n_cand = sum(len(k) for k in masks)
    feats = np.empty((n_cand, D or 0), dtype=np.float32)
    struct_id = np.empty(n_cand, dtype=np.int64)
    element = np.empty(n_cand, dtype=np.int32)
    gamma = np.empty(n_cand, dtype=np.float32)
    counts, frame_source = [], []
    pos = offset = 0
    for d, m, keep in zip(dirs, metas, masks):
        n_coll = len(m["n_atoms"])
        sc = read_features(d, keys=("features", "Z")) if len(keep) else None
        sl = slice(pos, pos + len(keep))
        if len(keep):
            feats[sl] = sc["features"][keep]
            element[sl] = _z_to_index(sc["Z"][keep], element_map)
        struct_id[sl] = offset + m["collection"][keep]
        gamma[sl] = m["gamma_pristine"][keep]
        species_idx = _z_to_index(m["species_Z"], element_map)
        counts.append(_elem_count_matrix(species_idx, m["elem_counts"], n_coll, n_elem))
        frame_source += [(d, k) for k in range(n_coll)]
        pos += len(keep)
        offset += n_coll
        del sc
    pool = {"struct_id": struct_id, "element": element, "gamma": gamma,
            uq_constants.FEATURES: feats}
    struct_elem_counts = (np.concatenate(counts) if counts
                          else np.zeros((0, n_elem), dtype=np.int64))
    return pool, struct_elem_counts, frame_source, D, offsets


def _candidate_pool_from_df(df, element_map, gamma_min, gamma_max, artifact_path):
    """Atom pool of the candidate atoms in the gamma window from pool_frames columns."""
    have = df["cand_features"].map(lambda x: x is not None and not (
        isinstance(x, float) and np.isnan(x)))
    if (~have).any():
        # a predict file has per-atom gamma; a HAL pool does not
        why = ("the prediction failed for them" if "gamma" in df.columns
               else "their walker's sidecar did not cover them yet")
        log.warning("%d row(s) have no candidate data (%s): they can only enter through "
                    "the element floors or the top-up. Rows with no atom above the "
                    "threshold do have (empty) candidate data.", int((~have).sum()), why)
    sub = df[have]
    _check_candidate_meta(
        sub["cand_feature_gamma_min"].unique(), sub["cand_artifact_sha256"].unique(),
        gamma_min=gamma_min, artifact_path=artifact_path, source="the candidate columns",
    )
    sid_c, el_c, g_c, f_c = [], [], [], []
    D = None
    for sid, (at, idx, f, g, ok) in enumerate(zip(
            df["ase_atoms"], df["cand_atom_index"], df["cand_features"], df["cand_gamma"],
            have)):
        if not ok:
            continue
        f = np.asarray(f)
        D = f.shape[1] if f.ndim == 2 else D
        keep = np.flatnonzero(_in_window(np.asarray(g), gamma_min, gamma_max))
        idx = np.asarray(idx)[keep]
        sid_c.append(np.full(len(keep), sid, dtype=np.int64))
        el_c.append(_z_to_index(at.numbers[idx], element_map))
        g_c.append(np.asarray(g, dtype=np.float32)[keep])
        f_c.append(f[keep].astype(np.float32, copy=False))
    pool = {
        "struct_id": np.concatenate(sid_c) if sid_c else np.zeros(0, np.int64),
        "element": np.concatenate(el_c) if el_c else np.zeros(0, np.int32),
        "gamma": np.concatenate(g_c) if g_c else np.zeros(0, np.float32),
        uq_constants.FEATURES: (np.concatenate(f_c) if f_c
                                else np.zeros((0, D or 0), np.float32)),
    }
    return pool, D


def _read_selected_hal_frames(frame_source, top, offsets=None):
    """Atoms of the selected frames only, read by index from each walker's extxyz."""
    from tensorpotential.uq.hal import features as hal_features

    by_dir: dict[str, list[int]] = {}
    for s in top:
        d, k = frame_source[s]
        by_dir.setdefault(d, []).append(k)
    got = {}
    for d, ks in by_dir.items():
        frames = hal_features.read_frames(os.path.join(d, "hal_collected.extxyz"), ks,
                                          offsets=(offsets or {}).get(d))
        for k, at in zip(ks, frames):
            if int(at.info.get("collection_index", k)) != k:
                raise SystemExit(f"{d}: frame {k} has collection_index "
                                 f"{at.info['collection_index']}: frames and sidecar disagree")
            at.calc = None
            got[(d, k)] = at
    rows = []
    for s in top:
        d, k = frame_source[s]
        at = got[(d, k)]
        rows.append({"ase_atoms": at, **at.info, "hal_dir": d})
    return pd.DataFrame(rows) if rows else pd.DataFrame({"ase_atoms": [], "hal_dir": []})


def _structure_element_counts(df: pd.DataFrame, element_map: list[str] | None):
    """Per-structure per-element atom counts. Returns int matrix [N_structs, n_elem]."""
    n_elem = len(element_map) if element_map is not None else 119
    counts = np.zeros((len(df), n_elem), dtype=np.int64)
    for sid, at in enumerate(df["ase_atoms"]):
        idx = _z_to_index(at.get_atomic_numbers(), element_map)
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
    *,
    n_structures: int,
) -> tuple[list[int], dict[int, int], dict[int, float]]:
    """Walk the strategy iterator until N distinct structures are hit; rank them by
    pick count (ties: max gamma).

    The strategy generator (e.g. ``fps-all``) decides the order in which atoms are
    yielded; this function only counts and stops. Element floors are applied later,
    to the whole selection.

    Returns (top_struct_ids, pick_counts, gamma_max_per_struct).
    """
    struct_ids = atom_pool["struct_id"]
    gammas = atom_pool["gamma"]

    pick_counts: dict[int, int] = Counter()
    gamma_max: dict[int, float] = {}

    bar = tqdm(total=n_structures, desc="select", unit="struct", dynamic_ncols=True)
    atoms_seen = 0
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
            if len(pick_counts) >= n_structures:
                break
        bar.set_postfix_str(f"atoms={atoms_seen}", refresh=True)
    finally:
        bar.close()
    return _topk_from_counts(pick_counts, gamma_max, n_structures), dict(pick_counts), gamma_max


def _collect_first_hits(
    atom_iter: Iterator[int], struct_ids: np.ndarray, n_structures: int
) -> tuple[list[int], dict[int, int]]:
    """Structures in the order the strategy first yields one of their atoms, until
    ``n_structures`` distinct ones (no pick-count ranking). Returns (top,
    pick_counts)."""
    top: list[int] = []
    picks: dict[int, int] = Counter()
    for atom_idx in atom_iter:
        sid = int(struct_ids[atom_idx])
        if sid not in picks:
            top.append(sid)
        picks[sid] += 1
        if len(top) >= n_structures:
            break
    return top, dict(picks)


def _repair_floors_greedy(
    top: list[int],
    struct_elem_counts: np.ndarray,
    floors: dict[int, int],
    n_structures: int,
    n_total_structs: int,
    max_donors: int = 64,
):
    """Element floors for an ordered greedy selection (calm-greedy).

    While fewer than ``n_structures`` frames are selected, donor frames (most atoms
    of the deficit elements) are appended; at ``n_structures`` a donor replaces the
    latest greedy pick whose swap lowers the total deficit. Greedy picks keep their
    order, donors follow. Returns (top, added, evicted).
    """
    keys = np.array(sorted(floors), dtype=np.int64)
    vals = np.array([floors[int(e)] for e in keys], dtype=np.int64)
    top, greedy = list(top), list(top)
    in_top = set(top)
    out = struct_elem_counts[top].sum(axis=0).astype(np.int64)
    added: list[int] = []
    evicted: list[int] = []

    def deficit(o):
        return np.maximum(vals - o[keys], 0)

    while True:
        dfc = deficit(out)
        if not dfc.any():
            break
        need = dfc > 0
        outside = np.setdiff1d(np.arange(n_total_structs), np.fromiter(in_top, np.int64))
        if not len(outside):
            break
        useful = np.minimum(struct_elem_counts[np.ix_(outside, keys[need])], dfc[need]).sum(1)
        order = np.argsort(-useful, kind="stable")[:max_donors]
        order = order[useful[order] > 0]
        if not len(order):
            log.warning("element floors: no unselected frame has atoms of the deficit "
                        "element(s); leaving residual deficits.")
            break
        if len(top) < n_structures:
            donor = int(outside[order[0]])
            top.append(donor)
            added.append(donor)
        else:
            swap = None
            for d in outside[order]:
                for v in reversed(greedy):  # latest greedy pick first
                    if deficit(out - struct_elem_counts[v] + struct_elem_counts[d]).sum() < dfc.sum():
                        swap = (int(d), v)
                        break
                if swap:
                    break
            if swap is None:
                log.warning("element floors: no swap lowers the deficit; leaving residual "
                            "deficits.")
                break
            donor, victim = swap
            top.remove(victim)
            greedy.remove(victim)
            in_top.discard(victim)
            out -= struct_elem_counts[victim]
            top.append(donor)
            added.append(donor)
            evicted.append(victim)
        in_top.add(donor)
        out += struct_elem_counts[donor]
    return top, added, evicted


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
    # With the artifact known and a gamma-window strategy, predict keeps only the
    # candidate atoms' features (and hashes the artifact, so select can check it);
    # the all-atom strategies (and runs without an artifact) need every feature.
    artifact = args.artifact or args.uq
    candidates_only = bool(artifact) and all(
        p in _WINDOW_STRATEGIES or p == "random-all" for p, _ in args.parts)
    if candidates_only:
        features = ["--features-gamma-min", repr(float(args.gamma_min))]
    else:
        features = ["--save-features"]
    pred_argv = [
        "--model",
        args.model,
        "--dataset",
        args.dataset,
        "--output",
        output,
        *features,
        "--n-workers",
        str(args.n_workers),
        "--threads-per-worker",
        str(args.threads_per_worker),
    ]
    if args.gpus:
        pred_argv += ["--gpus", args.gpus]
    if args.checkpoint:
        pred_argv += ["--checkpoint", args.checkpoint]
    if artifact:
        # yaml model: the model's artifact; SavedModel: recorded only, for the hash
        pred_argv += ["--artifact", artifact]
    if args.dump_elements:
        pred_argv += ["--dump-elements", *args.dump_elements]
    if args.verbose:
        pred_argv += ["--verbose"]
    log.info("Running prediction in select (%s); output → %s",
             "candidate features only" if candidates_only else "all-atom features", output)
    rc = predict_main(pred_argv)
    if rc != 0:
        raise RuntimeError("prediction step failed")
    return output


def _may_have_candidate_columns(path: str, chunk: int = 1 << 22) -> bool:
    """Whether a pickled dataset can be a HAL pool with candidate columns, without
    unpickling it: a streaming byte search for the ``cand_features`` column name
    (decompression only, chunked memory). False means certainly not; True is
    confirmed by loading. Keeps a raw --dataset from being loaded twice (once here,
    once by the prediction)."""
    import bz2
    import gzip
    import lzma

    needle = b"cand_features"
    opener = open
    for ext, fn in ((".gz", gzip.open), (".gzip", gzip.open), (".bz2", bz2.open),
                    (".xz", lzma.open)):
        if path.endswith(ext):
            opener = fn
    tail = b""
    with opener(path, "rb") as fh:
        while True:
            block = fh.read(chunk)
            if not block:
                return False
            buf = tail + block
            if needle in buf:
                return True
            tail = buf[-len(needle):]


def _dataset_may_be_pool(args) -> bool:
    """:func:`_may_have_candidate_columns` of --dataset, scanned once per run (the scan
    decompresses the whole file; --model-dir and the input resolution both ask)."""
    if not hasattr(args, "_dataset_may_be_pool"):
        args._dataset_may_be_pool = (not is_extxyz_path(args.dataset)
                                     and _may_have_candidate_columns(args.dataset))
    return args._dataset_may_be_pool


def _apply_model_dir(args):
    """Fill --model / --checkpoint / --artifact from --model-dir (the HAL resolver;
    explicit flags win). Selecting from HAL sidecars or a predicted file needs only
    the artifact; running the prediction needs all three."""
    from tensorpotential.uq.hal.model_dir import resolve_model_dir

    # model.yaml and checkpoint only when a prediction will run with a yaml model: not
    # for HAL sidecars, a predicted file, a --dataset with candidate columns, or an
    # explicit SavedModel --model (which carries its own weights)
    predicts = (bool(args.dataset) and not args.hal_dirs and not args.predicted
                and not _dataset_may_be_pool(args))
    savedmodel = bool(args.model) and os.path.isdir(args.model)
    need = (("model_yaml", "checkpoint", "artifact") if predicts and not savedmodel
            else ("artifact",))
    explicit_artifact = args.artifact or args.uq
    try:
        model, ckpt, art = resolve_model_dir(
            args.model_dir, model_yaml=args.model, checkpoint=args.checkpoint,
            artifact_path=explicit_artifact, require=need,
            flags=("--model", "--checkpoint", "--artifact"))
    except (FileNotFoundError, ValueError) as exc:
        raise SystemExit(str(exc)) from None
    args.model = os.path.abspath(model) if model else None
    if not savedmodel:  # a SavedModel needs no checkpoint
        args.checkpoint = os.path.abspath(ckpt) if ckpt else None
    if not explicit_artifact and art:
        args.artifact = os.path.abspath(art)
    log.info("select model from %s: model=%s | checkpoint=%s | artifact=%s",
             args.model_dir, args.model, args.checkpoint, args.uq or args.artifact)


def _resolve_input(args):
    """(mode, df): mode 'hal' (sidecars), 'candidates' (pooled candidate columns) or
    'predicted' (per-atom gamma/features from `grace_uq predict`)."""
    if args.hal_dirs:
        if args.predicted or args.dataset:
            raise SystemExit("--hal-dirs replaces --predicted / --dataset: pass one of them.")
        return "hal", None
    if args.predicted:
        log.info("Loading predicted dataset: %s", args.predicted)
        df = load_dataset_any(args.predicted)
        return ("candidates" if _is_candidate_df(df) else "predicted"), df
    if not args.dataset:
        raise SystemExit("either --predicted, --dataset or --hal-dirs must be given.")
    if _dataset_may_be_pool(args):
        df = load_dataset_any(args.dataset)
        if _is_candidate_df(df):
            log.info("%s carries HAL candidate-atom columns: selecting without a model "
                     "prediction.", args.dataset)
            return "candidates", df
        del df
    predicted_path = _maybe_run_predict(args)
    log.info("Loading predicted dataset: %s", predicted_path)
    df = load_dataset_any(predicted_path)
    return ("candidates" if _is_candidate_df(df) else "predicted"), df


def select_main(argv=None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    _parse_strategy_parts(args)

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
    if args.hal_dirs:
        from tensorpotential.uq.hal.tools import find_walker_dirs

        try:
            args.hal_dirs = find_walker_dirs(args.hal_dirs)
        except ValueError as e:
            raise SystemExit(f"--hal-dirs: {e}")
        log.info("--hal-dirs: %d walker director%s", len(args.hal_dirs),
                 "y" if len(args.hal_dirs) == 1 else "ies")
    if args.model_dir:
        _apply_model_dir(args)

    threads_int = resolve_threads_per_worker(args.threads_per_worker, args.n_workers)
    apply_master_thread_caps(threads_int)

    # yaml-mode users pass --artifact for predict; it IS the UQ npz, so reuse it.
    uq_path = args.uq or args.artifact
    _check_strategy_prerequisites(args, uq_path)
    mode, df = _resolve_input(args)

    meta = None
    if uq_path:
        if not args.uq:
            log.info("--uq not given; reusing --artifact (%s) as the UQ source.", uq_path)
        meta = read_artifact_metadata(uq_path)
        D = meta["D"]
        element_map = meta["element_map"]
        n_elements = meta["n_elements"]
    elif mode != "predicted":
        # candidate input without an artifact: atomic numbers stand in for element indices
        element_map, n_elements, D = None, 119, None
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

    window = (args.gamma_min, args.gamma_max)
    names = [p for p, _ in args.parts]
    # the pool holds only the gamma-window atoms unless an all-atom strategy (fps-all)
    # runs; random-all needs no atoms (frame ids and element counts only)
    in_window = "fps-all" not in names
    if mode != "predicted" and not in_window:
        raise SystemExit(
            "--strategy fps-all needs every atom's gamma/features; HAL candidate input "
            f"holds only the candidate atoms. Use one of {list(_WINDOW_STRATEGIES)} or "
            "random-all, or run `grace_uq predict` on the frames."
        )
    frame_source = None
    if mode == "hal":
        pool, struct_elem_counts, frame_source, D_sc, hal_offsets = (
            _candidate_pool_from_hal_dirs(args.hal_dirs, element_map, *window, uq_path))
        n_frames = len(frame_source)
    elif mode == "candidates":
        pool, D_sc = _candidate_pool_from_df(df, element_map, *window, uq_path)
        struct_elem_counts = _structure_element_counts(df, element_map)
        n_frames = len(df)
    else:
        needing = [p for p in names if p.startswith("fps-") or p == "calm-greedy"]
        if needing and uq_constants.FEATURES not in df.columns:
            raise SystemExit(_features_needed_msg(needing[0]))
        pool = _build_atom_pool(df, element_map, need_features=bool(needing),
                                gamma_window=window if in_window else None,
                                strategy=needing[0] if needing else args.strategy)
        struct_elem_counts = _structure_element_counts(df, element_map)
        n_frames = len(df)
        D_sc = None
    if D is None:
        D = D_sc
    log.info("Atom pool: %d atoms in %d structures%s", len(pool["struct_id"]), n_frames,
             " (candidate atoms in the gamma window)" if in_window else "")

    if n_frames == 0 or (mode == "predicted" and not in_window and len(pool["struct_id"]) == 0):
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

    # calm-greedy without -n: the natural set, at most every frame; several parts: the
    # sum of their N (every frame if a calm-greedy part has none)
    if len(args.parts) == 1:
        n_cap = args.n_structures if args.n_structures is not None else n_frames
    else:
        n_cap = (min(sum(n for _, n in args.parts), n_frames)
                 if all(n is not None for _, n in args.parts) else n_frames)
    if args.n_structures is not None and args.n_structures > n_frames:
        log.warning(
            "--n-structures=%d exceeds candidate pool size (%d); will return at most %d.",
            n_cap,
            n_frames,
            n_frames,
        )

    floors = None
    floor_target: int | None = None
    if args.element_stratified is None:
        # calm-greedy covers every element in gamma terms: floors only on request
        # (--element-stratified or --min-per-element); the other strategies keep them
        args.element_stratified = (any(p != "calm-greedy" for p in names)
                                   or args.min_per_element is not None)
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
        # Per-element clamp: an element's floor cannot exceed what the selection can
        # hold, the sum of its n largest per-frame counts (n = -n, or every frame).
        # Without this clamp an unreachable floor would force `_collect_top_n` to
        # exhaust the entire iterator (then all strategies converge to the same answer).
        atoms_per_e = struct_elem_counts.sum(axis=0)
        present = np.flatnonzero(atoms_per_e > 0)
        supply = _top_n_supply(struct_elem_counts[:, present], n_cap)
        eff = np.minimum(floor_val, supply)
        floors = {int(e): int(v) for e, v in zip(present, eff)}
        clamped_idx = np.flatnonzero(eff < floor_val)
        log.info(
            "Stratification ON: floor=%d atoms/element across %d present element(s) "
            "(skipped %d absent element(s))",
            floor_val,
            len(floors),
            n_elements - len(present),
        )
        if len(clamped_idx) > 0:
            clamped = [
                f"{format_elem_label(int(present[i]), element_map)}:{int(supply[i])}"
                for i in clamped_idx
            ]
            log.warning(
                "Clamped the per-element floor %d for %d element(s) to what %d "
                "frame(s) of the candidate pool can supply: %s",
                floor_val,
                len(clamped),
                min(n_cap, n_frames),
                ", ".join(clamped),
            )

    rng = np.random.default_rng(args.seed)
    # one part or several: in order, each part on the frames not yet picked; the floors
    # are applied once, to the selection as a whole
    top, label, greedy, part_counts, pick_counts = _run_parts(
        args, pool, n_frames, struct_elem_counts, uq_path, rng, in_window)
    floor_added, floor_evicted = [], []
    # a single non-greedy part fills its N, floors included, even from nothing; an
    # empty calm-greedy natural set or empty combined parts select nothing: floor donors
    # would be frames without a single atom above --gamma-min
    fill = len(args.parts) == 1 and greedy is None
    if floors is not None and (top or fill):
        top, floor_added, floor_evicted = _repair_floors_greedy(
            top, struct_elem_counts, floors, n_cap, n_frames)
    elif floors is not None:
        log.info("no frame selected: element floors not applied.")

    for f in top:  # what the strategy parts did not pick, the floors added
        label.setdefault(int(f), "floors")
    # Top-up to N if we still don't have enough (e.g. pool too small). Not for
    # calm-greedy (fewer than N frames is its natural set) or several parts.
    if len(top) < n_cap and greedy is None and len(args.parts) == 1:
        log.warning(
            "Only %d/%d structures available; topping up from unselected pool by gamma_max.",
            len(top),
            n_cap,
        )
        picked = set(top)
        outside = [s for s in range(n_frames) if s not in picked]
        if mode == "predicted":
            gamma_per = df["gamma"].apply(
                lambda g: float(np.max(g)) if len(g) else 0.0).to_numpy()
        else:  # max candidate gamma per frame (0 without candidates in the window)
            gamma_per = np.zeros(n_frames)
            np.maximum.at(gamma_per, pool["struct_id"], pool["gamma"])
        outside_sorted = sorted(outside, key=lambda s: -gamma_per[s])
        for s in outside_sorted:
            if len(top) >= n_cap:
                break
            top.append(s)
            label[int(s)] = "top-up"

    top = top[: n_cap]
    if mode == "hal":
        out_df = _read_selected_hal_frames(frame_source, top, hal_offsets)
    else:
        out_df = df.iloc[top].copy()
    out_df["n_atoms_selected"] = [int(pick_counts.get(s, 0)) for s in top]
    out_df["strategy"] = [label[int(s)] for s in top]
    if greedy is not None:
        at_pick = dict(zip(greedy.frames, greedy.gamma_at_pick))
        rank = {f: i for i, f in enumerate(greedy.frames)}
        # NaN / -1: added by the element-floor repair, not by the greedy
        out_df["gamma_at_pick"] = [at_pick.get(s, np.nan) for s in top]
        out_df["pick_order"] = [rank.get(s, -1) for s in top]

    if is_extxyz_path(args.output) and len(out_df) == 0:
        raise SystemExit(
            f"no structures selected (no candidate above --gamma-min {args.gamma_min:g}?): "
            f"nothing to write to {args.output}"
        )
    if is_extxyz_path(args.output) and "ase_atoms" not in out_df.columns:
        raise SystemExit(
            "extxyz output requires the ase_atoms column on the predicted dataset."
        )
    drop = _PREDICTION_COLUMNS + _CAND_COLUMNS
    save_df = out_df.drop(columns=[c for c in drop if c in out_df.columns])
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
    if len(args.parts) > 1 or greedy is None:  # calm-greedy alone: its own summary below
        selected_counts = struct_elem_counts[top].sum(axis=0)
        floors_met = floors is not None and all(selected_counts[e] >= v for e, v in floors.items())
        _print_parts_summary(part_counts, floor_added, floor_evicted, floors is not None,
                             floors_met=floors_met, show_parts=len(args.parts) > 1)
    if greedy is not None:
        if len(args.parts) == 1:
            _print_greedy_summary(greedy, args.n_structures, floor_added, floor_evicted,
                                  floors_on=floors is not None)
        else:
            n_greedy = next(n for p, n in args.parts if p == "calm-greedy")
            _print_greedy_summary(greedy, n_greedy, floors_on=False)
    return 0


def _subpool(pool: dict, mask: np.ndarray) -> dict:
    return {k: v[mask] for k, v in pool.items()}


def _run_parts(args, pool, n_frames, struct_elem_counts, uq_path, rng, in_window):
    """Strategy parts in order; each part sees only the frames not yet picked.

    random-all: uniform over the remaining frames; random-extrap: uniform over the
    remaining frames with at least one candidate atom in the gamma window; calm-greedy:
    its tower first learns the frames already picked (``seed_frames``); fps-extrap:
    whitened, element-aware FPS over the remaining candidates, first-hit order; fps-all:
    mini-batch FPS over the remaining frames' atoms, pick-count ranking. Both fps parts
    start their distances from the atoms of the frames already picked (``seed``:
    candidates for fps-extrap, every atom for fps-all), so they do not pick near them.
    Returns (frames in pick order, {frame: part name}, calm-greedy result or None,
    [(name, N, n picked)], {frame: n_atoms_selected}): the atoms an fps part picked in a
    frame, else the frame's atoms in the pool (its candidates).
    """
    sid = np.asarray(pool["struct_id"])
    window_mask = (np.ones(len(sid), dtype=bool) if in_window
                   else _in_window(np.asarray(pool["gamma"]), args.gamma_min, args.gamma_max))
    fps_kw = {} if args.fps_max_pool is None else {"fps_max_pool": args.fps_max_pool}
    picked: list[int] = []
    label: dict[int, str] = {}
    greedy = None
    counts = []
    n_atoms = dict(enumerate(np.bincount(sid, minlength=n_frames).tolist()))
    for name, n in args.parts:
        taken = np.zeros(n_frames, dtype=bool)
        taken[picked] = True
        free_atoms = ~taken[sid] if len(sid) else np.zeros(0, dtype=bool)
        want = n if n is not None else n_frames
        if name == "random-all":
            frames = rng.permutation(np.flatnonzero(~taken))[:want].tolist()
        elif name == "random-extrap":
            cand = np.unique(sid[window_mask & free_atoms])
            frames = rng.permutation(cand)[:want].tolist()
        elif name == "calm-greedy":
            greedy = _run_calm_greedy(get_strategy(name), _subpool(pool, window_mask),
                                      uq_path, args, want, seed_frames=list(picked))
            frames = list(greedy.frames)
        elif name == "fps-extrap":
            # the picked frames' candidates stay in (same whitened space) as the FPS seed
            sub = _subpool(pool, window_mask)  # a copy: whitened in place
            _whiten_pool(sub, uq_path)
            frames, hits = _collect_first_hits(
                get_strategy(name)(sub, rng=rng, gamma_min=None, gamma_max=None,
                                   seed=~free_atoms[window_mask], **fps_kw),
                sub["struct_id"], want)
            n_atoms.update(hits)
        elif name == "fps-all":
            frames, hits, _ = _collect_top_n(
                get_strategy(name)(pool, rng=rng, seed=~free_atoms, **fps_kw), pool,
                n_structures=want)
            n_atoms.update(hits)
        else:
            raise SystemExit(f"--strategy {name} cannot be combined with other parts")
        frames = [int(f) for f in frames]
        picked += frames
        label.update({f: name for f in frames})
        counts.append((name, n, len(frames)))
        log.info("strategy part %s%s: %d frame(s)", name, f":{n}" if n else "", len(frames))
    return picked, label, greedy, counts, n_atoms


def _print_parts_summary(part_counts, floor_added, floor_evicted, floors_on, *, floors_met,
                         show_parts=True):
    if show_parts:
        items = [f"{name} {got}" + (f"/{n}" if n is not None else " (natural set)")
                 for name, n, got in part_counts]
        print("  strategy parts (in order, each on the frames not yet picked): "
              + " | ".join(items))
    if floor_added:
        print(f"  element floors on the combined set: {len(floor_added)} frame(s) added"
              + (f", replacing the {len(floor_evicted)} latest pick(s)" if floor_evicted
                 else ""))
    elif floors_on and floors_met:
        print("  element floors on the combined set: met (no frames added)")
    if floors_on and not floors_met:
        print("  element floors on the combined set: unmet (see per-element counts)")
    print()


def _whiten_pool(pool, uq_path):
    """fps-extrap: whiten the candidate features with the pristine CALM clusters."""
    if not uq_path:
        log.warning(
            "fps-extrap without --artifact: FPS runs on raw features (scale-dominated); "
            "pass --artifact to whiten with the pristine CALM clusters."
        )
        return
    import tensorflow as tf

    from tensorpotential.uq.cli.strategies import whiten_features
    from tensorpotential.uq.gmmuq import GMMUQModel

    whiten_features(pool[uq_constants.FEATURES], pool["element"],
                    GMMUQModel(uq_path, param_dtype=tf.float64))
    log.info("fps-extrap: %d candidate features whitened with the pristine CALM clusters",
             len(pool["element"]))


def _run_calm_greedy(strategy_fn, pool, uq_path, args, n_cap, seed_frames=()):
    """calm-greedy on a fresh float64 copy of the pristine tower (the artifact file);
    ``uq_path`` was checked by :func:`_check_strategy_prerequisites`."""
    from tensorpotential.uq.cli.strategies import selection_tower

    known = pool["element"] >= 0
    if not known.all():
        pool = _subpool(pool, known)
    tower = selection_tower(uq_path, elements=np.unique(pool["element"]))
    log.info("calm-greedy: %d candidate atoms in %d frames, update target %g",
             len(pool["struct_id"]), len(np.unique(pool["struct_id"])), args.update_target)
    return strategy_fn(pool, gmm=tower, n_structures=n_cap,
                       gamma_min=args.gamma_min, update_target=args.update_target,
                       seed_frames=seed_frames)


def _print_greedy_summary(res, n_structures, floor_added=(), floor_evicted=(),
                          floors_on=True):
    n = len(res.frames)
    print(f"  calm-greedy: {n} frame(s) picked, gamma at pick "
          + (f"{res.gamma_at_pick[0]:.2f} -> {res.gamma_at_pick[-1]:.2f}" if n else "-"))
    if res.natural or (n_structures is None and res.covered):
        print(f"  natural set size: {n} (every remaining frame has max gamma <= "
              f"{res.gamma_min:g} under the selection tower)")
    elif res.covered:
        print(f"  -n {n_structures} reached; pool covered (remaining max gamma "
              f"{res.remaining_max_gamma:.2f} <= {res.gamma_min:g}): natural set size {n}")
    else:
        print(f"  -n {n_structures} reached before the natural set: remaining max gamma "
              f"{res.remaining_max_gamma:.2f} > {res.gamma_min:g} (pool not covered)")
    if floor_added:
        kept = n - len(floor_evicted)
        print(f"  element floors: {len(floor_added)} frame(s) added for the per-element "
              f"floors" + (f", replacing the {len(floor_evicted)} latest greedy pick(s)"
                           if floor_evicted else "")
              + f"; output = {kept} greedy pick(s) + {len(floor_added)} floor frame(s)")
    elif not floors_on:
        pass
    elif not n:
        print("  element floors: not applied (the natural set is empty: no frame has an "
              "atom above --gamma-min, so nothing is selected)")
    else:
        print("  element floors: met by the greedy picks (no frames added)")
    print()


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
    n_cand_atoms = int(struct_elem_counts.sum())  # every atom of the pool's structures

    cand_frac = f" ({total_atoms / n_cand_atoms:.1%} of all pool atoms)" if n_cand_atoms else ""
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
  # 1. Default: up to 200 structures by calm-greedy (max-first greedy with an
  #    additive copy of the CALM tower; stops early at the natural set). Per-element
  #    floors are OFF for calm-greedy; add --element-stratified to turn them on:
  grace_uq select --artifact UQ/gmm_artifacts.npz \\
                  --predicted predicted.pkl.gz \\
                  -n 200 \\
                  --output selected.pkl.gz

  # 1b. HAL walkers: select from their hal_features.npz sidecars, no prediction;
  #     only the selected frames are read from hal_collected.extxyz. Without -n,
  #     calm-greedy returns its whole natural set (-n caps it):
  grace_uq select --artifact UQ/gmm_artifacts.npz \\
                  --hal-dirs hal_w0 hal_w1 hal_w2 \\
                  --output selected.pkl.gz

  # 1c. Several strategies in one call, applied in order, each on the frames not
  #     yet picked: calm-greedy's natural set plus 25 uniform reference frames
  #     (column `strategy` says which part picked a frame):
  grace_uq select --artifact UQ/gmm_artifacts.npz \\
                  --hal-dirs hal_w0 hal_w1 hal_w2 \\
                  --strategy calm-greedy random-all:25 \\
                  --output selected.pkl.gz

  # 2. Diverse extrapolatory picks: FPS on whitened features, frames in
  #    first-hit order (needs the candidate atoms' features: predict with
  #    --save-features or --features-gamma-min, or select from --hal-dirs):
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
        help="Raw dataset (.pkl.gz, .xyz/.extxyz or LAMMPS text dump); if --predicted "
        "not given, predict is run first. A HAL pool (`grace_uq hal --pool`, pkl.gz) "
        "with candidate columns is selected from directly, without a prediction.",
    )
    p.add_argument(
        "--dump-elements",
        nargs="+",
        metavar="EL",
        help="LAMMPS dump --dataset without an element column: element of atom type 1, "
        "2, ... (e.g. --dump-elements Al Li). Ignored when the dump has the column.",
    )
    p.add_argument(
        "--hal-dirs",
        nargs="+",
        default=None,
        metavar="DIR",
        help="HAL walker directories: select from their hal_features.npz sidecars "
        "(no model prediction) and read only the selected frames' Atoms.",
    )
    p.add_argument(
        "--model",
        help="SavedModel dir or model.yaml — required iff --dataset given.",
    )
    p.add_argument(
        "--model-dir",
        default=None,
        help="directory holding the model, searched like grace_uq hal --model-dir "
        "(model.yaml; checkpoint.index, checkpoints/checkpoint.best_test_loss.index, "
        "checkpoints/checkpoint.index; gmm_artifacts.npz, UQ/gmm_artifacts.npz); fills "
        "--model/--checkpoint/--artifact (explicit flags win). With --hal-dirs or "
        "--predicted only the artifact is needed.",
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
        default=None,
        help="Number of structures to select with a single --strategy: exactly N "
        "(pool permitting) for fps-extrap, fps-all and random-* (required there). "
        "calm-greedy: optional budget cap; without -n it selects the whole natural "
        "set, with -n it stops at N at most and the summary says whether the pool is "
        "covered. With several --strategy parts, give N per part (name:N) instead.",
    )
    p.add_argument(
        "--strategy",
        nargs="+",
        default=["calm-greedy"],
        metavar="NAME[:N]",
        help="Selection strategy, or several parts applied in order, each on the frames "
        "not yet picked, e.g. `--strategy calm-greedy fps-extrap:25 random-all:25`. "
        f"Names: {', '.join(list_strategies())}. Default calm-greedy: max-first greedy "
        "over frames with an additive copy of the CALM tower (needs --artifact); stops "
        "at the natural set, when no remaining frame has gamma > --gamma-min, or at N. "
        "fps-extrap: whitened FPS over the candidate atoms, frames in first-hit order. "
        "random-all: uniform over all frames (an unbiased reference sample); "
        "random-extrap: uniform over frames with candidates. N is required for fps-* "
        "and random-* (name:N, or -n with a single strategy). A later calm-greedy part "
        "first updates its tower with the frames already picked; a later fps part starts "
        "its distances from their atoms (fps-extrap: their candidates, whitened alike), "
        "so it does not pick near them. Several parts: -n is "
        "refused; the per-element floors are applied once to the combined set, on by "
        "default if any part is not calm-greedy, clamped to what the total N can hold.",
    )
    p.add_argument(
        "--update-target",
        type=float,
        default=1.25,
        help="calm-greedy: alpha search target of each selection-tower update (as HAL).",
    )
    p.add_argument("--gamma-min", type=float, default=1.0)
    p.add_argument(
        "--gamma-max",
        type=float,
        default=None,
        help="Upper gamma threshold (None = no cap).",
    )
    p.add_argument(
        "--element-stratified",
        dest="element_stratified",
        action="store_true",
        help="Per-element floors on (default: off for calm-greedy, which covers every "
        "element in gamma terms; on for fps-extrap / random-*).",
    )
    p.add_argument(
        "--no-element-stratified",
        dest="element_stratified",
        action="store_false",
        help="Per-element floors off.",
    )
    p.set_defaults(element_stratified=None)
    p.add_argument(
        "--min-per-element",
        type=int,
        default=None,
        help="Per-element floor (atoms/element); also turns the floors on for "
        "calm-greedy. Default = D+1 from artifact (strict floor for non-degenerate "
        "per-element covariance), clamped per element to what -n frames can supply.",
    )
    p.add_argument(
        "--fps-max-pool",
        type=int,
        default=None,
        help="Run FPS in shards of this many atoms (mini-batch, bounds memory). "
        "Default: exact FPS over all candidates for fps-extrap, 50,000 for fps-all.",
    )
    p.add_argument(
        "--n-workers",
        type=parse_n_workers,
        default="auto",
        help="workers of the prediction run inside select (see grace_uq predict "
        "--n-workers; auto: min(4, ceil(n_structures / 200)) over the visible GPUs, "
        "in-process for one).",
    )
    p.add_argument("--gpus", default=None, help="GPU ids for that prediction.")
    p.add_argument("--threads-per-worker", default="auto")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--verbose", action="store_true")
    return p


if __name__ == "__main__":
    raise SystemExit(select_main())
