"""Shared helpers for the grace_uq CLI subcommands."""

from __future__ import annotations

import logging
import os
import subprocess
import time

import numpy as np
import pandas as pd

from tensorpotential.uq import constants as uq_constants
from tensorpotential.uq.artifact_builder import GMMUQArtifactBuilder

log = logging.getLogger("grace_uq")


PKL_EXTS = (".pkl.gz", ".pkl.gzip", ".pckl.gz", ".pckl.gzip", ".pkl")
XYZ_EXTS = (".xyz", ".extxyz")


# Sentinel slots in step-1 / step-2 artifacts: pad rows whose centroid
# components are written as ``SENTINEL_CENTROID_VALUE`` so atoms never
# bind there at inference time. Detection uses a slightly looser threshold
# so any large-magnitude centroid (incl. accidental drift) is also caught.
SENTINEL_CENTROID_VALUE: float = 1e10
SENTINEL_DETECTION_THRESHOLD: float = 1e9


def is_sentinel_mask(centroids: np.ndarray) -> np.ndarray:
    """Boolean mask over a [K, D] centroid tensor: True for sentinel rows."""
    return np.any(np.abs(centroids) > SENTINEL_DETECTION_THRESHOLD, axis=1)


def is_extxyz_path(path: str) -> bool:
    return path.lower().endswith(XYZ_EXTS)


def is_pkl_path(path: str) -> bool:
    return any(path.lower().endswith(ext) for ext in PKL_EXTS)


def load_dataset_any(path: str) -> pd.DataFrame:
    """Load a dataframe from pkl.gz (default) or extxyz.

    Returns a DataFrame with at least an `ase_atoms` column.
    """
    if is_extxyz_path(path):
        from tensorpotential.cli.data import load_extxyz
        return load_extxyz(path)
    compression = "gzip" if path.endswith((".gz", ".gzip")) else "infer"
    return pd.read_pickle(path, compression=compression)


def save_dataset_any(
    df: pd.DataFrame,
    path: str,
    drop_ase_atoms: bool = False,
    energy_col: str = "energy_predicted",
    forces_col: str = "forces_predicted",
    stress_col: str = "stress_predicted",
):
    """Save DataFrame to pkl.gz or extxyz, dispatching by extension.

    For extxyz output, attaches a SinglePointCalculator with the predicted
    energy/forces/stress and stores `gamma` in atoms.arrays if present.
    Per-atom features are NOT preserved by extxyz output; pkl.gz preserves
    everything.
    """
    if is_extxyz_path(path):
        _save_extxyz(df, path, energy_col, forces_col, stress_col)
        return
    out = df.drop(columns=["ase_atoms"]) if drop_ase_atoms else df
    compression = "gzip" if path.endswith((".gz", ".gzip")) else None
    out.to_pickle(path, compression=compression)


def _save_extxyz(
    df: pd.DataFrame,
    path: str,
    energy_col: str,
    forces_col: str,
    stress_col: str,
):
    from ase.calculators.singlepoint import SinglePointCalculator
    from ase.io import write

    atoms_list = []
    for _, row in df.iterrows():
        at = row["ase_atoms"].copy()
        e = row.get(energy_col)
        f = row.get(forces_col)
        s = row.get(stress_col)
        if "gamma" in row and row["gamma"] is not None:
            at.arrays["gamma"] = np.asarray(row["gamma"])
        if e is not None:
            at.info["energy_predicted"] = float(e)
        at.calc = SinglePointCalculator(at, energy=e, forces=f, stress=s)
        atoms_list.append(at)
    write(path, atoms_list, format="extxyz")


def slice_dataset_for_worker(
    df: pd.DataFrame, worker_id: int, n_workers: int
) -> pd.DataFrame:
    """Round-robin row split: worker_id keeps rows where idx % n_workers == worker_id."""
    if n_workers <= 1:
        return df.reset_index(drop=True)
    return df.iloc[worker_id::n_workers].reset_index(drop=False).rename(
        columns={"index": "_orig_index"}
    )


def alignment_perm_from_hist(hist: np.ndarray, counts: np.ndarray):
    """Compute permutation ``inv`` such that ``hist[inv[k]].sum() ≈ counts[k]``.

    Some build pipelines save the per-cluster histograms (and the thresholds
    computed from them) in a different cluster order than the post-finalize
    centroids/counts: the sort in ``GMMUQArtifactBuilder.finalize`` is not
    propagated to step-3's histograms/thresholds. Apply this permutation to
    both histograms AND the corresponding row of ``interp_thresholds`` /
    ``eff_interp_thresholds`` so the displayed values line up with counts.

    Per-cluster row-sums need not match ``counts`` exactly: step-3 reassigns
    atoms by Mahalanobis distance using the finalized inv_cov, while ``counts``
    come from step-2's earlier assignment, so boundary atoms can drift between
    adjacent clusters. The thresholds remain correctly indexed for inference
    (which uses the same Mahalanobis assignment as step-3), so we treat any
    arrangement whose descending-by-count rank order already matches as
    identity. Only a genuine row permutation flips the rank order, and that's
    what we actually need to fix.

    Returns
    -------
    (inv, is_identity) : (np.ndarray | None, bool)
        ``inv`` is None when alignment cannot be recovered within tolerance
        (caller should suppress per-cluster diagnostics derived from these
        arrays). ``is_identity`` is True when the arrays are already aligned
        and no permutation is needed.
    """
    if (
        hist is None
        or counts is None
        or getattr(hist, "ndim", 0) != 2
        or hist.shape[0] != len(counts)
    ):
        return None, False
    counts_arr = np.asarray(counts, dtype=np.int64)
    hist_sums = hist.sum(axis=1).astype(np.int64)
    K = len(counts_arr)
    if np.array_equal(
        np.argsort(-hist_sums, kind="stable"),
        np.argsort(-counts_arr, kind="stable"),
    ):
        return np.arange(K, dtype=np.int64), True
    inv = np.full(K, -1, dtype=np.int64)
    used = np.zeros(K, dtype=bool)
    # For each k in counts order (largest first), find the unused storage
    # index whose row-sum best matches counts[k].
    for k in np.argsort(-counts_arr):
        diffs = np.where(used, np.iinfo(np.int64).max, np.abs(hist_sums - int(counts_arr[k])))
        j = int(np.argmin(diffs))
        tol = max(50, int(0.05 * max(int(counts_arr[k]), 1)))
        if int(diffs[j]) > tol:
            return None, False
        inv[int(k)] = j
        used[j] = True
    return inv, np.array_equal(inv, np.arange(K, dtype=np.int64))


def read_artifact_metadata(artifact_path: str) -> dict:
    """Extract summary metadata from a UQ .npz artifact.

    Returns a dict with: D (feature_dim), n_elements, K_max, element_map (or None),
    has_thresholds, has_histograms, artifacts (full per-element dict), interp_thresholds,
    eff_interp_thresholds, hist_bins, hist_arrays / eff_hist_arrays
    ({elem_idx: np.ndarray[K, n_bins]}, or None if not stored).
    """
    artifacts = GMMUQArtifactBuilder.load(artifact_path)
    with np.load(artifact_path, allow_pickle=True) as data:
        element_map = (
            list(data["element_map"]) if "element_map" in data.files else None
        )
        if element_map is not None:
            element_map = [
                e.decode("utf-8") if isinstance(e, (bytes, np.bytes_)) else str(e)
                for e in element_map
            ]
        interp_thresholds = (
            np.asarray(data["interp_thresholds"])
            if "interp_thresholds" in data.files
            else None
        )
        eff_interp_thresholds = (
            np.asarray(data["eff_interp_thresholds"])
            if "eff_interp_thresholds" in data.files
            else None
        )
        hist_bins = (
            np.asarray(data["hist_bins"]) if "hist_bins" in data.files else None
        )
        hist_arrays: dict[int, np.ndarray] = {}
        eff_hist_arrays: dict[int, np.ndarray] = {}
        for key in data.files:
            if key == "hist_bins":
                continue
            if key.startswith("hist_"):
                try:
                    hist_arrays[int(key[len("hist_"):])] = np.asarray(data[key])
                except ValueError:
                    pass
            elif key.startswith("eff_hist_"):
                try:
                    eff_hist_arrays[int(key[len("eff_hist_"):])] = np.asarray(data[key])
                except ValueError:
                    pass

    if not artifacts:
        raise ValueError(f"Empty artifact: {artifact_path}")
    first = next(iter(artifacts.values()))
    D = int(first[uq_constants.CENTROIDS].shape[1])
    K_max = max(a[uq_constants.CENTROIDS].shape[0] for a in artifacts.values())

    # Realign histograms AND thresholds to the final centroid order when the
    # build pipeline left them in a different permutation. The bug: step-3
    # builds histograms/thresholds keyed by the step-2 cluster order at the
    # time the workers ran, but the saved ``interp_thresholds`` and per-cluster
    # ``hist_*`` arrays don't get permuted alongside the post-``finalize``
    # centroids/counts. Inference is silently affected (each cluster's atoms
    # get the wrong threshold). Detect by matching hist row sums to counts,
    # then apply the same permutation to hist/eff_hist AND to the row of
    # interp_thresholds/eff_interp_thresholds. Caller is told which elements
    # were realigned so a warning can be printed.
    hist_realigned: list[int] = []
    hist_unaligned: list[int] = []
    if hist_arrays:
        if interp_thresholds is not None:
            interp_thresholds = np.array(interp_thresholds, copy=True)
        if eff_interp_thresholds is not None:
            eff_interp_thresholds = np.array(eff_interp_thresholds, copy=True)
        for e in list(hist_arrays.keys()):
            if e not in artifacts:
                continue
            counts = artifacts[e][uq_constants.COUNTS]
            inv, is_identity = alignment_perm_from_hist(hist_arrays[e], counts)
            if inv is None:
                hist_unaligned.append(int(e))
                continue
            if is_identity:
                continue
            hist_realigned.append(int(e))
            hist_arrays[e] = hist_arrays[e][inv]
            if eff_hist_arrays and e in eff_hist_arrays:
                eff_hist_arrays[e] = eff_hist_arrays[e][inv]
            if interp_thresholds is not None and e < interp_thresholds.shape[0]:
                interp_thresholds[e] = interp_thresholds[e][inv]
            if eff_interp_thresholds is not None and e < eff_interp_thresholds.shape[0]:
                eff_interp_thresholds[e] = eff_interp_thresholds[e][inv]

    return {
        "D": D,
        "K_max": K_max,
        "n_elements": len(artifacts),
        "element_map": element_map,
        "artifacts": artifacts,
        "interp_thresholds": interp_thresholds,
        "eff_interp_thresholds": eff_interp_thresholds,
        "hist_bins": hist_bins,
        "hist_arrays": hist_arrays or None,
        "eff_hist_arrays": eff_hist_arrays or None,
        "hist_realigned_elems": sorted(set(hist_realigned)),
        "hist_unaligned_elems": sorted(set(hist_unaligned)),
        "has_thresholds": interp_thresholds is not None,
        "has_eff_thresholds": eff_interp_thresholds is not None,
        "has_histograms": bool(hist_arrays),
        "has_eff_histograms": bool(eff_hist_arrays),
    }


# ---------------------------------------------------------------------------
# Worker process supervision (mirrors grace_uq build's spawn_workers)
# ---------------------------------------------------------------------------

_PROGRESS_KEYWORDS = ("progress", "completed", "starting", "finalizing", "eta", "elapsed")

# Worker→master progress channel. The format is intentionally minimal so it
# survives line-buffered pipes without ANSI/CR conflicts from in-worker tqdm.
PROGRESS_PREFIX = "__PROGRESS__"


def format_progress(done: int, total: int) -> str:
    return f"{PROGRESS_PREFIX} done={done} total={total}"


def parse_progress(line: str) -> tuple[int, int] | None:
    """Inverse of :func:`format_progress`. Returns ``(done, total)`` or None."""
    if not line.startswith(PROGRESS_PREFIX):
        return None
    try:
        parts = dict(p.split("=", 1) for p in line.split()[1:] if "=" in p)
        return int(parts["done"]), int(parts["total"])
    except (KeyError, ValueError):
        return None


def spawn_and_monitor(
    cmds: list[tuple[int, list[str]]],
    *,
    gpus: list[str],
    threads_per_worker: int,
    verbose: bool,
    label: str = "Worker",
    progress_handler=None,
) -> list[int]:
    """Spawn each (worker_id, cmd) tuple as a subprocess, stream their stdout, return failed ids.

    GPU assignment is round-robin via CUDA_VISIBLE_DEVICES env var.

    If ``progress_handler`` is given, it is called as
    ``progress_handler(worker_id, line)`` for every line starting with
    :data:`PROGRESS_PREFIX`; such lines are then suppressed from normal stdout
    forwarding so the master can render a single combined bar.
    """
    procs = []
    for i, cmd in cmds:
        env = os.environ.copy()
        env["TF_CPP_MIN_LOG_LEVEL"] = "3"
        env["CUDA_VISIBLE_DEVICES"] = gpus[i % len(gpus)]
        if threads_per_worker > 0:
            t = str(threads_per_worker)
            for var in (
                "OMP_NUM_THREADS",
                "MKL_NUM_THREADS",
                "OPENBLAS_NUM_THREADS",
                "NUMEXPR_MAX_THREADS",
                "TF_NUM_INTEROP_THREADS",
                "TF_NUM_INTRAOP_THREADS",
            ):
                env[var] = t
        p = subprocess.Popen(
            cmd,
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        procs.append((i, p))

    def should_print(line: str) -> bool:
        return verbose or any(k in line.lower() for k in _PROGRESS_KEYWORDS)

    def _handle_line(i: int, s: str):
        if not s:
            return
        if progress_handler is not None and s.startswith(PROGRESS_PREFIX):
            progress_handler(i, s)
            return
        if should_print(s):
            print(f"  [{label} {i}] {s}", flush=True)

    active = procs[:]
    failed: list[int] = []
    while active:
        for i, p in active[:]:
            line = p.stdout.readline()
            if line:
                _handle_line(i, line.strip())
            if p.poll() is not None:
                for line in p.stdout.readlines():
                    _handle_line(i, line.strip())
                if p.returncode != 0:
                    print(
                        f"  [{label} {i}] ERROR: exited with code {p.returncode}",
                        flush=True,
                    )
                    failed.append(i)
                active.remove((i, p))
        time.sleep(0.05)
    return failed


def resolve_threads_per_worker(value, n_workers: int) -> int:
    """Resolve `--threads-per-worker auto|<int>` to an int."""
    if isinstance(value, str) and value == "auto":
        cores = os.cpu_count() or 1
        return max(1, cores // max(n_workers, 1))
    return int(value)


def apply_master_thread_caps(threads_per_worker: int):
    """Limit master process thread pools so it doesn't fight workers."""
    if threads_per_worker <= 0:
        return
    t = str(threads_per_worker)
    for var in (
        "OMP_NUM_THREADS",
        "MKL_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "NUMEXPR_MAX_THREADS",
    ):
        os.environ[var] = t
    try:
        import tensorflow as tf

        tf.config.threading.set_intra_op_parallelism_threads(threads_per_worker)
        tf.config.threading.set_inter_op_parallelism_threads(
            max(1, threads_per_worker // 2)
        )
    except (ImportError, RuntimeError):
        pass


def detect_model_kind(path: str) -> str:
    """Return 'savedmodel' if `path` is a directory, 'yaml' if it ends in .yaml/.yml."""
    if os.path.isdir(path):
        return "savedmodel"
    if path.endswith((".yaml", ".yml")):
        return "yaml"
    raise ValueError(
        f"--model must be a SavedModel directory or a model.yaml file, got: {path}"
    )


def load_element_map_from_savedmodel(model_path: str) -> list[str] | None:
    """Read `chemical_symbols` from a SavedModel's metadata.yaml/json.

    Returns the list of element symbols in the model's internal index order,
    or None if the path is not a SavedModel directory or the metadata is absent.
    """
    if not os.path.isdir(model_path):
        return None
    yaml_path = os.path.join(model_path, "metadata.yaml")
    json_path = os.path.join(model_path, "metadata.json")
    syms = None
    if os.path.exists(yaml_path):
        import yaml

        with open(yaml_path) as f:
            meta = yaml.safe_load(f) or {}
        syms = meta.get("chemical_symbols")
    elif os.path.exists(json_path):
        import json

        with open(json_path) as f:
            meta = json.load(f)
        syms = meta.get("chemical_symbols")
    if syms is None:
        return None
    return [str(s) for s in syms]


def format_elem_label(idx: int, element_map=None) -> str:
    """Format an element index as ``Sym(idx)``, or ``str(idx)`` when no symbol
    is known. ``element_map`` may be:

    - ``None``                 → no symbol, returns ``str(idx)``
    - ``list[str]``            → positional, ``element_map[idx]`` is the symbol
    - ``dict[int, str]``       → ``element_map.get(idx)`` is the symbol
    """
    if element_map is None:
        return str(idx)
    if isinstance(element_map, list):
        if 0 <= idx < len(element_map):
            return f"{element_map[idx]}({idx})"
        return str(idx)
    name = element_map.get(idx)
    return f"{name}({idx})" if name else str(idx)
