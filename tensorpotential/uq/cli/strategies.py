"""Selection strategy registry for ``grace_uq select``.

Two kinds of strategy:

- atom generators (``random-all``, ``random-extrap``, ``fps-all``, ``fps-extrap``)
  yield atom indices (into the flat atom pool) in preferred order; select ranks the
  structures by pick count (``random-*``, ``fps-all``) or, for strategies marked
  ``first_hit`` (``fps-extrap``), in the order of their first picked atom;
- frame selectors (``selects_frames``; ``calm-greedy``, the default) return the
  chosen structures directly (:class:`GreedyResult`) and may stop before N.

Adding a new strategy: decorate a generator function with
``@register_strategy("name")`` — it gets picked up by :func:`get_strategy`.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Callable, Iterator

import numpy as np

from tensorpotential.uq import constants as uq_constants
from tensorpotential.uq.calm_numpy import (
    SIGMA_EPS,
    cholesky_psd,
    cluster_groups,
    nearest_cluster,
    sigma,
)

log = logging.getLogger("grace_uq.select")

# atom pool fields used by all strategies
# - struct_id  : int array [N]
# - element    : int array [N]
# - gamma      : float array [N]  (optional; required for *-extrap)
# - features   : float array [N, D] (required for fps-* and calm-greedy)


_STRATEGIES: dict[str, Callable[..., Iterator[int]]] = {}


def register_strategy(name: str):
    """Decorator: register a strategy generator under ``name``."""

    def deco(fn):
        _STRATEGIES[name] = fn
        return fn

    return deco


def get_strategy(name: str):
    if name not in _STRATEGIES:
        raise KeyError(
            f"unknown selection strategy: {name!r}. "
            f"Available: {sorted(_STRATEGIES)}"
        )
    return _STRATEGIES[name]


def list_strategies() -> list[str]:
    return sorted(_STRATEGIES)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _gamma_mask(atom_pool: dict, gamma_min: float | None, gamma_max: float | None):
    if "gamma" not in atom_pool:
        raise KeyError(
            "this strategy requires `gamma` in the atom pool — run `grace_uq predict` first."
        )
    gamma = np.asarray(atom_pool["gamma"])
    mask = np.ones_like(gamma, dtype=bool)
    if gamma_min is not None:
        mask &= gamma >= gamma_min
    if gamma_max is not None:
        mask &= gamma <= gamma_max
    return mask


# ---------------------------------------------------------------------------
# Random strategies
# ---------------------------------------------------------------------------


@register_strategy("random-all")
def random_all(atom_pool: dict, *, rng, **kwargs) -> Iterator[int]:
    n = len(atom_pool["struct_id"])
    perm = rng.permutation(n)
    for i in perm:
        yield int(i)


@register_strategy("random-extrap")
def random_extrap(
    atom_pool: dict,
    *,
    rng,
    gamma_min: float = 1.0,
    gamma_max: float | None = None,
    **kwargs,
) -> Iterator[int]:
    mask = _gamma_mask(atom_pool, gamma_min, gamma_max)
    candidates = np.flatnonzero(mask)
    if len(candidates) == 0:
        log.warning(
            "random-extrap: no atoms in gamma window [%s, %s] — yielding nothing",
            gamma_min,
            gamma_max,
        )
        return
    perm = rng.permutation(candidates)
    for i in perm:
        yield int(i)


# ---------------------------------------------------------------------------
# Mini-batch FPS
# ---------------------------------------------------------------------------


def _chunked_min_dist(feats: np.ndarray, ref: np.ndarray, chunk: int = 4096):
    """Min euclidean distance from each row of `feats` to any row of `ref`.

    `ref` may be empty; in that case returns +inf. `feats` is processed in
    chunks of `chunk` rows to bound the working memory regardless of |ref|.
    """
    if ref.size == 0:
        return np.full(len(feats), np.inf, dtype=feats.dtype)
    chunk = max(1, min(chunk, (1 << 24) // len(ref)))  # block of <= 16M distances
    out = np.empty(len(feats), dtype=feats.dtype)
    for s in range(0, len(feats), chunk):
        e = min(s + chunk, len(feats))
        # ||a-b||^2 = ||a||^2 + ||b||^2 - 2 a.b — kept squared, sqrt at end
        block = feats[s:e]
        d2 = (
            (block ** 2).sum(axis=1, keepdims=True)
            + (ref ** 2).sum(axis=1)
            - 2.0 * block @ ref.T
        )
        out[s:e] = np.sqrt(np.maximum(d2.min(axis=1), 0.0))
    return out


def _fps_iter(
    candidate_idx: np.ndarray,
    features_all: np.ndarray,
    *,
    rng,
    fps_max_pool: int,
    elements=None,
    seed_idx=None,
):
    """Mini-batch FPS over `candidate_idx` rows of `features_all`.

    Shards are random subsets of size ≤ fps_max_pool. Within each shard we
    greedily yield the farthest atom from the running selected set, refresh
    distances, and repeat until the shard is exhausted. The running selected
    set persists across shards, so distances accumulate correctly.

    ``elements`` (per row of `features_all`): distances only between atoms of the
    same element, i.e. one FPS per element, interleaved by distance.
    ``seed_idx`` (rows of `features_all`, not candidates): already selected; the
    running selected set starts with them, so FPS moves away from them.
    """
    if len(candidate_idx) == 0:
        return
    el_all = None if elements is None else np.asarray(elements)
    empty = np.empty((0, features_all.shape[1]), dtype=features_all.dtype)
    selected: dict = {}  # element (or 0) -> selected features
    if seed_idx is not None and len(seed_idx):
        seed_el = np.zeros(len(seed_idx), np.int64) if el_all is None else el_all[seed_idx]
        for e in np.unique(seed_el):
            selected[int(e)] = np.asarray(features_all[seed_idx[seed_el == e]])
    perm = rng.permutation(candidate_idx)
    for shard_start in range(0, len(perm), fps_max_pool):
        shard_idx = perm[shard_start : shard_start + fps_max_pool]
        feats = features_all[shard_idx]
        el = np.zeros(len(shard_idx), np.int64) if el_all is None else el_all[shard_idx]
        d = np.empty(len(shard_idx), dtype=feats.dtype)
        for e in np.unique(el):
            m = el == e
            d[m] = _chunked_min_dist(feats[m], selected.get(int(e), empty))
        for _ in range(len(shard_idx)):
            i = int(np.argmax(d))
            if d[i] == -np.inf:  # shard exhausted
                break
            yield int(shard_idx[i])
            e = int(el[i])
            selected[e] = np.vstack([selected.get(e, empty), feats[i : i + 1]])
            same = el == e
            new_d = np.sqrt(np.sum((feats[same] - feats[i]) ** 2, axis=1))
            d[same] = np.minimum(d[same], new_d)
            d[i] = -np.inf  # pop


def _require_features(atom_pool: dict):
    if uq_constants.FEATURES not in atom_pool:
        raise KeyError(
            "fps-* and calm-greedy require per-atom `features` — re-run "
            "`grace_uq predict --save-features`."
        )
    return np.asarray(atom_pool[uq_constants.FEATURES])


@register_strategy("fps-all")
def fps_all(
    atom_pool: dict, *, rng, fps_max_pool: int = 50_000, seed=None, **kwargs
) -> Iterator[int]:
    """Mini-batch FPS over every atom. ``seed`` (bool per atom, optional): atoms
    already selected (a later part of a combined ``--strategy``); never yielded,
    FPS starts its distances from them."""
    features = _require_features(atom_pool)
    cand, seed_idx = _split_seed(len(features), seed)
    yield from _fps_iter(cand, features, rng=rng, fps_max_pool=fps_max_pool,
                         seed_idx=seed_idx)


def _split_seed(n, seed, mask=None):
    """(candidate rows, seed rows): ``mask`` (default all) without the ``seed`` rows."""
    seed = np.zeros(n, dtype=bool) if seed is None else np.asarray(seed, dtype=bool)
    mask = np.ones(n, dtype=bool) if mask is None else mask
    return np.flatnonzero(mask & ~seed), np.flatnonzero(seed)


def _fps_full(candidate_idx: np.ndarray, features_all: np.ndarray, *, rng, elements=None,
              chunk=65536, seed_idx=None):
    """Exact FPS over all of ``candidate_idx`` (no shards), in float32.

    ``elements`` (per row of `features_all`): a pick only updates the distances of
    the candidates of its own element, so every element runs its own FPS (no
    cross-element distances); the elements are interleaved by distance, each
    starting from a random atom (elements in random order). Without ``elements``,
    one FPS from a random atom. Distance updates run in chunks; no copy of the
    features when ``candidate_idx`` is every row in order. ``seed_idx`` (rows of
    `features_all`, not candidates): already selected; the distances start from them
    (per element), and an element with seed atoms gets no random start.
    """
    n = len(candidate_idx)
    if n == 0:
        return
    whole = n == len(features_all) and np.array_equal(candidate_idx, np.arange(n))
    feats = features_all if whole else features_all[candidate_idx]
    if elements is None:
        groups = [np.arange(n)]
        group_of = np.zeros(n, dtype=np.int64)
    else:
        el = np.asarray(elements)[candidate_idx]
        uniq, group_of = np.unique(el, return_inverse=True)
        groups = [np.flatnonzero(group_of == g) for g in range(len(uniq))]
    d = np.full(n, np.inf, dtype=np.float32)
    seeded = set()
    if seed_idx is not None and len(seed_idx):
        seed_el = (np.zeros(len(seed_idx), np.int64) if elements is None
                   else np.asarray(elements)[seed_idx])
        keys = [0] if elements is None else uniq
        for g, rows in enumerate(groups):
            ref = np.asarray(features_all[seed_idx[seed_el == keys[g]]], dtype=np.float32)
            if len(ref):
                d[rows] = _chunked_min_dist(np.asarray(feats[rows], dtype=np.float32), ref)
                seeded.add(g)

    def take(i):
        d[i] = -np.inf
        rows = groups[group_of[i]]
        x = np.asarray(feats[i], dtype=np.float32)
        contiguous = len(rows) == n
        for s in range(0, len(rows), chunk):
            sl = slice(s, s + chunk) if contiguous else rows[s : s + chunk]
            blk = np.asarray(feats[sl], dtype=np.float32) - x
            d[sl] = np.minimum(d[sl], np.sqrt((blk * blk).sum(1)))

    started = 0
    for g in rng.permutation(len(groups)):  # one random start per element without seed
        if g in seeded:
            continue
        i = int(rng.choice(groups[g]))
        yield int(candidate_idx[i])
        take(i)
        started += 1
    for _ in range(n - started):
        i = int(np.argmax(d))
        if d[i] == -np.inf:
            break
        yield int(candidate_idx[i])
        take(i)


@register_strategy("fps-extrap")
def fps_extrap(
    atom_pool: dict,
    *,
    rng,
    gamma_min: float = 1.0,
    gamma_max: float | None = None,
    fps_max_pool: int | None = None,
    seed=None,
    **kwargs,
) -> Iterator[int]:
    """FPS over the atoms in the gamma window (``grace_uq select`` whitens their
    features with the pristine CALM clusters first, see :func:`whiten_features`).

    Distances are element-aware: a pick only updates the distances of candidates of
    its own element, so each element runs its own FPS in its own whitened space (no
    cross-element distances, which carry no CALM meaning); the elements interleave by
    distance. Within an element, atoms of different clusters are whitened about
    their own centroids and do share one space.

    The caller ranks frames by first hit (``first_hit``): a frame enters when FPS
    first picks one of its atoms, until N distinct frames. ``fps_max_pool=None``
    (default) runs exact FPS over every candidate; an integer runs mini-batch FPS
    over shards of that size.

    ``seed`` (bool per atom, optional): atoms already selected (the candidates of
    the frames an earlier part of a combined ``--strategy`` picked, in the same
    whitened space); never yielded, and FPS starts its distances from them.
    """
    features = _require_features(atom_pool)
    mask = _gamma_mask(atom_pool, gamma_min, gamma_max)
    cand, seed_idx = _split_seed(len(features), seed, mask)
    if len(cand) == 0:
        log.warning(
            "fps-extrap: no atoms in gamma window [%s, %s] — yielding nothing",
            gamma_min,
            gamma_max,
        )
        return
    elements = atom_pool.get("element")
    if fps_max_pool is None:
        yield from _fps_full(cand, features, rng=rng, elements=elements, seed_idx=seed_idx)
    else:
        yield from _fps_iter(cand, features, rng=rng, fps_max_pool=fps_max_pool,
                             elements=elements, seed_idx=seed_idx)


fps_extrap.first_hit = True


def whiten_features(features: np.ndarray, elements, gmm, chunk=65536) -> np.ndarray:
    """In place: f -> C^T (f - mu_k), Lambda_k = C C^T of the atom's nearest pristine
    cluster k, so |z| is the atom's Mahalanobis distance sigma (gamma * threshold) and
    Euclidean FPS on z runs in the CALM metric. Atoms of elements < 0 stay raw."""
    for sl, z, _, _ in CandidateGamma(gmm, features, elements).whitened_blocks(chunk):
        features[sl] = z.astype(features.dtype, copy=False)
    return features


# ---------------------------------------------------------------------------
# calm-greedy: max-first greedy over frames with an additive selection tower
# ---------------------------------------------------------------------------


def selection_tower(artifact_path, elements=None):
    """Fresh float64 copy of the pristine CALM tower for calm-greedy.

    Every cluster of ``elements`` (default: all; the pool's elements are the only ones
    scored) is re-inverted once (:meth:`GMMUQModel.reinvert_all`), so the initial
    scores come from the same float64 inverses that later updates produce; additive
    updates then never raise gamma, which the lazy heap relies on.
    """
    import tensorflow as tf

    from tensorpotential.uq.gmmuq import GMMUQModel

    tower = GMMUQModel(artifact_path, param_dtype=tf.float64)
    tower.reinvert_all(elements)
    return tower


class CandidateGamma:
    """NumPy gamma of the candidate atoms under a :class:`GMMUQModel` tower.

    Centroids and thresholds never change under an update, so each atom's cluster
    (:func:`~tensorpotential.uq.calm_numpy.nearest_cluster` over the element's real
    clusters, as in the artifact update) is computed once; the inverse covariances are
    re-read per element after each update (:meth:`refresh`).
    gamma = :func:`~tensorpotential.uq.calm_numpy.sigma` / threshold.
    """

    def __init__(self, gmm, features, elements, chunk=8192):
        self.gmm, self.features = gmm, features
        self.elements = np.asarray(elements, dtype=np.int64)
        self.thr = np.asarray(gmm.interp_thresholds_dense.numpy(), dtype=np.float64)
        self.n_clusters = np.asarray(gmm.n_clusters_per_elem.numpy())
        self.cent, self.inv = {}, {}
        self.cluster = np.zeros(len(self.elements), dtype=np.int64)
        for e in np.unique(self.elements[self.elements >= 0]):
            e = int(e)
            c = np.asarray(gmm.centroids[e].numpy(), dtype=np.float64)[: self.n_clusters[e]]
            self.cent[e] = c
            idx = np.flatnonzero(self.elements == e)
            self.cluster[idx] = nearest_cluster(
                np.asarray(features[idx], dtype=np.float64), c, chunk=chunk)
        self.refresh(self.cent)

    def refresh(self, elements):
        for e in elements:
            e = int(e)
            self.inv[e] = np.asarray(self.gmm.inv_covs[e].numpy(),
                                     dtype=np.float64)[: self.n_clusters[e]]

    def gamma(self, idx) -> np.ndarray:
        """Exact gamma of atoms ``idx`` under the tower's current inverse covariances."""
        idx = np.asarray(idx)
        out = np.empty(len(idx), dtype=np.float64)
        el, k = self.elements[idx], self.cluster[idx]
        x = np.asarray(self.features[idx], dtype=np.float64)
        for e, kk, m in cluster_groups(el, k):  # a frame has few (e, k) pairs
            out[m] = sigma(x[m] - self.cent[e][kk], self.inv[e][kk]) / self.thr[e, kk]
        return out

    def whitened_blocks(self, chunk=65536):
        """``(atom indices, z, element, cluster)`` blocks over all atoms of known
        elements, with z = C^T (f - mu) for Lambda = C C^T of the atom's cluster, so
        |z| is the atom's sigma."""
        known = self.elements >= 0
        rows = np.flatnonzero(known)
        for e, k, m in cluster_groups(self.elements[known], self.cluster[known]):
            c_fac = cholesky_psd(self.inv[e][k])
            idx = rows[m]
            for s in range(0, len(idx), chunk):
                sl = idx[s : s + chunk]
                yield sl, (np.asarray(self.features[sl], dtype=np.float64) - self.cent[e][k]) @ c_fac, e, k

    def gamma_all(self, chunk=65536) -> np.ndarray:
        """Gamma of every atom through the whitened blocks (|C^T delta|^2): fast, equal
        to :meth:`gamma` up to round-off."""
        out = np.empty(len(self.elements), dtype=np.float64)
        for sl, z, e, k in self.whitened_blocks(chunk):
            out[sl] = np.sqrt((z * z).sum(1) + SIGMA_EPS) / self.thr[e, k]
        return out


@dataclass
class GreedyResult:
    frames: list  # struct ids in pick order
    gamma_at_pick: list  # frame max gamma under the tower when picked
    alphas: list  # alpha of each tower update
    natural: bool  # stopped because every remaining frame had max gamma <= gamma_min
    remaining_max_gamma: float  # max gamma of the unpicked frames at the end (0: none)
    gamma_min: float
    n_scored: int = 0  # frame scorings (lazy: re-scorings of stale heap tops)

    @property
    def covered(self) -> bool:
        """No unpicked frame left above gamma_min: the selection is the natural set."""
        return self.remaining_max_gamma <= self.gamma_min


@register_strategy("calm-greedy")
def calm_greedy(
    atom_pool: dict,
    *,
    gmm,
    n_structures: int,
    gamma_min: float = 1.0,
    update_target: float = 1.25,
    alphas=None,
    lazy: bool = True,
    progress: bool = True,
    seed_frames=(),
    **kwargs,
) -> GreedyResult:
    """Frame-level max-first greedy with an additive selection tower (``gmm``, mutated).

    Repeat: pick the frame whose candidate atoms have the largest max gamma under the
    tower; update the tower with that frame's candidate atoms by the HAL alpha search
    (smallest alpha of ``alphas`` whose trial max gamma is below ``update_target``,
    else the last; ``mode="additive"``). Stop after ``n_structures`` frames or when
    every remaining frame has max gamma <= ``gamma_min`` (the natural set).

    Lazy evaluation: additive updates never raise an atom's gamma, so a frame's last
    score bounds its current one. Frames sit in a max-heap of (possibly stale) scores;
    the top is re-scored until a fresh score is on top, which is then the exact argmax
    (ties: lowest struct id), the same pick as re-scoring every frame (``lazy=False``).
    Atom pool: ``struct_id``, ``element`` (model indices, >= 0) and ``features`` of the
    candidate atoms (already restricted to the gamma window).

    ``seed_frames``: frames already selected (by earlier parts of a combined selection):
    the tower is first updated with their candidate atoms, in order, by the same alpha
    search, and they are never picked again.
    """
    import heapq

    from tqdm import tqdm

    from tensorpotential.uq.hal.collector import DEFAULT_ALPHAS

    alphas = tuple(DEFAULT_ALPHAS if alphas is None else alphas)
    feats = _require_features(atom_pool)
    sid = np.asarray(atom_pool["struct_id"])
    order = np.argsort(sid, kind="stable")
    frames, starts = np.unique(sid[order], return_index=True)
    bounds = np.append(starts, len(order))
    atoms_of = {int(f): order[bounds[i] : bounds[i + 1]] for i, f in enumerate(frames)}
    scorer = CandidateGamma(gmm, feats, atom_pool["element"])

    def score(f):
        return float(scorer.gamma(atoms_of[f]).max())

    def update(f):
        idx = atoms_of[f]
        x = np.asarray(feats[idx], dtype=np.float64)
        el = np.asarray(atom_pool["element"])[idx]
        alpha, _ = gmm.alpha_search_update(x, el, alphas, update_target)
        scorer.refresh(np.unique(el))
        return alpha

    seeded = [int(f) for f in seed_frames if int(f) in atoms_of]
    for f in seeded:  # the tower learns what is already selected
        update(f)
    if seeded:
        keep = ~np.isin(frames, seeded)
        frames = frames[keep]
        sid_keep = ~np.isin(sid, seeded)
        sid = sid[sid_keep]
    else:
        sid_keep = None

    picks, g_pick, used = [], [], []
    natural, remaining, n_scored = False, 0.0, 0
    bar = tqdm(total=min(n_structures, len(frames)), desc="calm-greedy", unit="frame", dynamic_ncols=True,
               disable=not progress)
    if lazy:
        # initial bounds: bulk gamma through a Cholesky factor, which differs from the
        # per-frame quadratic form by round-off (~1e-15 relative); the 1e-9 inflation
        # keeps it a bound. Every entry is stale (version -1) until re-scored, so each
        # pick is made on an exact per-frame score.
        g0 = scorer.gamma_all()
        if sid_keep is not None:
            g0 = g0[sid_keep]
        fmax = np.zeros(len(frames))
        np.maximum.at(fmax, np.searchsorted(frames, sid), g0)
        heap = [(-(m * (1 + 1e-9) + 1e-12), int(f), -1) for f, m in zip(frames, fmax)]
        heapq.heapify(heap)
        version = 0

        def fresh_top():
            nonlocal n_scored
            while heap:
                negs, f, ver = heap[0]
                if ver == version:
                    return -negs, f
                heapq.heapreplace(heap, (-score(f), f, version))
                n_scored += 1
            return None

        while len(picks) < n_structures:
            top = fresh_top()
            if top is None or top[0] <= gamma_min:
                natural, remaining = True, (top[0] if top else 0.0)
                break
            heapq.heappop(heap)
            picks.append(top[1])
            g_pick.append(top[0])
            used.append(update(top[1]))
            version += 1
            bar.update(1)
        if not natural:
            top = fresh_top()
            remaining = top[0] if top else 0.0
    else:
        left = [int(f) for f in frames]
        while len(picks) < n_structures:
            if not left:
                natural = True
                break
            s = np.array([score(f) for f in left])
            n_scored += len(left)
            i = int(np.argmax(s))
            if s[i] <= gamma_min:
                natural, remaining = True, float(s[i])
                break
            picks.append(left.pop(i))
            g_pick.append(float(s[i]))
            used.append(update(picks[-1]))
            bar.update(1)
        if not natural and left:
            remaining = max(score(f) for f in left)
    bar.close()
    return GreedyResult(picks, g_pick, used, natural, float(remaining), gamma_min, n_scored)


calm_greedy.selects_frames = True
