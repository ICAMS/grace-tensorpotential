"""Train-data resolution, filter callable loading, shard subsampling, prefetch."""

from __future__ import annotations

import logging

import argparse
import importlib
import inspect
import os
import queue
import threading
from typing import Callable

import pandas as pd

from tensorpotential.data.process_df import ENERGY, ENERGY_CORRECTED_COL

log = logging.getLogger("grace_uq")

DEFAULT_TRAIN_DATA: list[str] = ["training_set.pkl.gz"]


def _load_filter_fn(spec: str | None) -> Callable | None:
    """Load a ``module.path:function_name`` callable, or return None.

    The user supplies a Python dotted module path and a function name
    (separated by a colon). The module must be importable on the worker's
    ``PYTHONPATH``. Two callable signatures are accepted:

    * ``f(atoms) -> bool`` — classic per-atom filter.
    * ``f(atoms, file_path) -> bool`` — receives the source shard path so the
      filter can scope itself to a subset of inputs (e.g. apply only to OMAT
      shards in a mixed OMAT+SMAX build).

    The returned callable always has the 2-arg form ``(atoms, file_path)``;
    1-arg user functions are transparently wrapped. Returns ``True`` to keep
    the structure, ``False`` to drop it.
    """
    if spec is None:
        return None
    if ":" not in spec:
        raise ValueError(
            f"--filter-fn spec must be 'module.path:function_name'; got {spec!r}"
        )
    module_path, _, fn_name = spec.partition(":")
    module_path, fn_name = module_path.strip(), fn_name.strip()
    if not module_path or not fn_name:
        raise ValueError(
            f"--filter-fn spec is malformed: {spec!r} (need 'module.path:function_name')"
        )
    try:
        mod = importlib.import_module(module_path)
    except ImportError as exc:
        raise ImportError(
            f"--filter-fn: cannot import module {module_path!r} (PYTHONPATH issue?): {exc}"
        ) from exc
    try:
        fn = getattr(mod, fn_name)
    except AttributeError as exc:
        raise AttributeError(
            f"--filter-fn: module {module_path!r} has no attribute {fn_name!r}"
        ) from exc

    positional = [
        p for p in inspect.signature(fn).parameters.values()
        if p.kind in (
            inspect.Parameter.POSITIONAL_ONLY,
            inspect.Parameter.POSITIONAL_OR_KEYWORD,
        )
    ]
    if len(positional) >= 2:
        return fn
    return lambda atoms, file_path: fn(atoms)


def _resolve_weighted_train_data(
    args: argparse.Namespace,
) -> tuple[list[str], dict[str, float]]:
    """Flatten ``args.train_data_weighted`` groups into a (paths, weight_map).

    Returns
    -------
    paths : list of absolute file paths (sorted by first-occurrence order)
    weight_map : dict[abspath -> weight]. Empty when no weighted groups were
        provided (back-compat: all atoms then get weight 1.0).
    """
    groups = getattr(args, "train_data_weighted", None)
    if not groups:
        return list(args.train_data), {}
    weight_map: dict[str, float] = {}
    paths: list[str] = []
    for weight, files in groups:
        for raw in files:
            p = os.path.abspath(raw)
            if p in weight_map:
                if abs(weight_map[p] - weight) > 1e-12:
                    raise ValueError(
                        f"--train-data-weighted: file {p} appears in multiple "
                        f"groups with different weights ({weight_map[p]} vs {weight})"
                    )
                continue
            weight_map[p] = weight
            paths.append(p)
    return paths, weight_map


def _resolve_train_data_from_input_yaml(
    yaml_path: str = "../../input.yaml",
) -> list[str] | None:
    """Look up `data.filename` in a sibling ``input.yaml`` and use it as train data.

    Used when the defaulted ``--train-data training_set.pkl.gz`` is not on
    disk (e.g. the user runs `grace_uq build` inside `seed/{N}/` after a fit
    that didn't dump the resolved train split). The function refuses to
    proceed if `data.train_size` or `data.test_size` are set, because those
    fields imply that the actual training set is a *subset* of
    `data.filename` and using the full file would silently include test data
    into the GMM fit.

    Returns the resolved [path] list, or None if the yaml is missing / has
    no usable filename.
    """
    yaml_path = os.path.abspath(yaml_path)
    if not os.path.exists(yaml_path):
        return None
    try:
        import yaml  # PyYAML, already a project dependency
    except ImportError:
        return None
    with open(yaml_path) as f:
        cfg = yaml.safe_load(f) or {}
    data = cfg.get("data", {}) or {}

    splitting_fields = [
        f for f in ("train_size", "test_size", "test_fraction") if f in data
    ]
    if splitting_fields:
        raise SystemExit(
            f"input.yaml at {yaml_path} contains data field(s) "
            f"{splitting_fields} that subset the training data at fit time. "
            "UQ artifacts must be built on the EXACT atoms used during "
            "training; using data.filename here would silently include test "
            "structures.\nPass --train-data explicitly with the resolved "
            "training split (typically dumped by gracemaker as "
            "seed/{N}/training_set.pkl.gz)."
        )

    filename = data.get("filename")
    if not filename:
        return None
    if not os.path.isabs(filename):
        filename = os.path.join(os.path.dirname(yaml_path), filename)
    if not os.path.exists(filename):
        log.warning(
            "input.yaml referenced data.filename=%s but the file does not "
            "exist; ignoring fallback.",
            filename,
        )
        return None
    return [filename]


def _prefetch(iterator, buffer_size=2):
    """Run *iterator* in a background thread, yielding up to *buffer_size* items ahead.

    This keeps the GPU busy producing the next batch while the CPU consumes the
    current one (e.g. running KMeans partial_fit).
    """
    q = queue.Queue(maxsize=buffer_size)
    _sentinel = object()
    _stop = threading.Event()

    def _producer():
        try:
            for item in iterator:
                if _stop.is_set():
                    break
                q.put(item)
        except Exception as exc:
            q.put(exc)
        q.put(_sentinel)

    t = threading.Thread(target=_producer, daemon=True)
    t.start()
    try:
        while True:
            item = q.get()
            if item is _sentinel:
                break
            if isinstance(item, Exception):
                raise item
            yield item
    finally:
        _stop.set()


def _load_and_subsample(file_path, rng, frac, filter_fn=None):
    """Load a shard as a DataFrame, optionally filter rows, then subsample.

    Returns a ``pd.DataFrame`` with at least an ``ase_atoms`` column and an
    ``energy_corrected`` column. ``filter_fn`` is applied before ``--frac`` so
    the random thinner doesn't waste picks on structures that the user wanted
    dropped. ``filter_fn(atoms, file_path)`` is invoked per row; the
    ``_load_filter_fn`` helper normalizes 1-arg user callables to this shape.
    """
    loaded = pd.read_pickle(file_path)
    df = (
        loaded
        if isinstance(loaded, pd.DataFrame)
        else pd.DataFrame({"ase_atoms": list(loaded)})
    )
    # Some shards carry only `energy`; the fundamental builder reads
    # `energy_corrected` — back-fill so heterogeneous shard families align.
    if ENERGY_CORRECTED_COL not in df.columns and ENERGY in df.columns:
        df[ENERGY_CORRECTED_COL] = df[ENERGY]
    if filter_fn is not None:
        before = len(df)
        keep = df["ase_atoms"].apply(lambda a: filter_fn(a, file_path))
        df = df[keep].reset_index(drop=True)
        log.info(
            "filter %s: %d → %d structures",
            os.path.basename(file_path), before, len(df),
        )
    if frac is not None:
        n_select = int(len(df) * frac)
        selected_indices = rng.choice(len(df), n_select, replace=False)
        selected_indices.sort()
        df = df.iloc[selected_indices].reset_index(drop=True)
    return df
