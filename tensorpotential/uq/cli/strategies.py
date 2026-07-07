"""Selection strategy registry for ``grace_uq select``.

Each strategy is a generator that yields atom indices (into the flat atom
pool) in preferred order. The select main loop pulls atoms lazily until the
hard guarantees on N structures and per-element stratification are met.

Adding a new strategy: decorate a generator function with
``@register_strategy("name")`` — it gets picked up by
:func:`get_strategy`.
"""

from __future__ import annotations

from typing import Callable, Iterator
import logging

import numpy as np

from tensorpotential.uq import constants as uq_constants

log = logging.getLogger("grace_uq.select")

# atom pool fields used by all strategies
# - struct_id  : int array [N]
# - element    : int array [N]
# - gamma      : float array [N]  (optional; required for *-extrap)
# - features   : float array [N, D] (required for fps-*)


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
):
    """Mini-batch FPS over `candidate_idx` rows of `features_all`.

    Shards are random subsets of size ≤ fps_max_pool. Within each shard we
    greedily yield the farthest atom from the running selected set, refresh
    distances, and repeat until the shard is exhausted. The running selected
    set persists across shards, so distances accumulate correctly.
    """
    if len(candidate_idx) == 0:
        return
    selected_feats = np.empty((0, features_all.shape[1]), dtype=features_all.dtype)
    perm = rng.permutation(candidate_idx)
    for shard_start in range(0, len(perm), fps_max_pool):
        shard_idx = perm[shard_start : shard_start + fps_max_pool]
        feats = features_all[shard_idx]
        d = _chunked_min_dist(feats, selected_feats)
        for _ in range(len(shard_idx)):
            i = int(np.argmax(d))
            if not np.isfinite(d[i]) and selected_feats.size > 0:
                break
            yield int(shard_idx[i])
            selected_feats = np.vstack([selected_feats, feats[i : i + 1]])
            new_d = np.sqrt(np.sum((feats - feats[i]) ** 2, axis=1))
            d = np.minimum(d, new_d)
            d[i] = -np.inf  # pop


def _require_features(atom_pool: dict):
    if uq_constants.FEATURES not in atom_pool:
        raise KeyError(
            "fps-* strategies require per-atom `features` — re-run "
            "`grace_uq predict --save-features`."
        )
    return np.asarray(atom_pool[uq_constants.FEATURES])


@register_strategy("fps-all")
def fps_all(
    atom_pool: dict, *, rng, fps_max_pool: int = 50_000, **kwargs
) -> Iterator[int]:
    features = _require_features(atom_pool)
    cand = np.arange(len(features))
    yield from _fps_iter(cand, features, rng=rng, fps_max_pool=fps_max_pool)


@register_strategy("fps-extrap")
def fps_extrap(
    atom_pool: dict,
    *,
    rng,
    gamma_min: float = 1.0,
    gamma_max: float | None = None,
    fps_max_pool: int = 50_000,
    **kwargs,
) -> Iterator[int]:
    features = _require_features(atom_pool)
    mask = _gamma_mask(atom_pool, gamma_min, gamma_max)
    cand = np.flatnonzero(mask)
    if len(cand) == 0:
        log.warning(
            "fps-extrap: no atoms in gamma window [%s, %s] — yielding nothing",
            gamma_min,
            gamma_max,
        )
        return
    yield from _fps_iter(cand, features, rng=rng, fps_max_pool=fps_max_pool)
