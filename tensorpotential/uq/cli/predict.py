"""Predict subcommand: run E/F/S + per-atom gamma over a dataset.

Multi-worker via ``subprocess.Popen``; each worker handles a round-robin shard
of the input dataframe. Results are written to ``.predict_w{i}.pkl.gz`` files
next to ``--output`` and concatenated by the master.
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
import time

# Suppress TensorFlow noise before any TF import.
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
os.environ.setdefault("TF_ENABLE_ONEDNN_OPTS", "0")

import numpy as np
import pandas as pd
from tqdm import tqdm

from tensorpotential.calculator import predict_structures
from tensorpotential.uq import constants as uq_constants
from tensorpotential.uq.cli._common import (
    apply_master_thread_caps,
    detect_model_kind,
    format_progress,
    is_extxyz_path,
    auto_n_workers,
    load_dataset_any,
    parse_n_workers,
    parse_progress,
    resolve_threads_per_worker,
    save_dataset_any,
    slice_dataset_for_worker,
    spawn_and_monitor,
    visible_gpu_ids,
)

log = logging.getLogger("grace_uq.predict")
logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")


# --------------------------------------------------------------------------
# Calculator factories
# --------------------------------------------------------------------------


def _build_calculator(args):
    """Build a TPCalculator for either a SavedModel or model.yaml + checkpoint + artifact."""
    kind = detect_model_kind(args.model)
    pad_kwargs = dict(
        pad_atoms_number=args.pad_atoms_number,
        pad_neighbors_fraction=args.pad_neighbors_fraction,
        max_number_reduction_recompilation=args.max_number_reduction_recompilation,
    )
    if kind == "savedmodel":
        from tensorpotential.calculator import TPCalculator

        calc = TPCalculator(model=args.model, **pad_kwargs)
        if not calc._saved_model_uq_sigs:
            raise RuntimeError(
                f"SavedModel at {args.model} has no `compute_uq` signature.\n"
                "Re-export with `grace_uq build` (without --no-export), or pass\n"
                "--model model.yaml --checkpoint <ckpt> --artifact <gmm.npz> instead."
            )
        return calc
    # yaml mode
    if not args.checkpoint or not args.artifact:
        raise SystemExit(
            "When --model is a model.yaml file, --checkpoint and --artifact are required."
        )
    from tensorpotential.uq.factories import get_gmm_uq_calculator

    return get_gmm_uq_calculator(
        model_yaml=args.model,
        checkpoint=args.checkpoint,
        gmm_artifact_path=args.artifact,
        **pad_kwargs,
    )


# --------------------------------------------------------------------------
# Worker
# --------------------------------------------------------------------------


# per-row candidate columns written with --features-gamma-min (the HAL pool schema)
CAND_KEYS = ("cand_atom_index", "cand_features", "cand_gamma")


def _candidate_postprocess(features_gamma_min, keep_features):
    """Per-structure: keep the features of the atoms with gamma >= features_gamma_min as
    ``cand_*`` arrays; drop the all-atom features unless ``keep_features``."""

    def post(values):
        g = values.get(uq_constants.ATOMIC_GAMMA)
        f = values.get(uq_constants.FEATURES)
        if g is not None and f is not None:
            g = np.asarray(g)
            idx = np.flatnonzero(g >= features_gamma_min).astype(np.int32)
            values["cand_atom_index"] = idx
            values["cand_features"] = np.asarray(f)[idx].astype(np.float32)
            values["cand_gamma"] = g[idx].astype(np.float32)
        if not keep_features:
            values.pop(uq_constants.FEATURES, None)
        return values

    return post


def _predict_columns(
    atoms, calc, save_features, raise_errors, sort_by_natoms=True, progress=None,
    features_gamma_min=None,
):
    """Dataframe columns for a list of structures, in input order.

    Thin adapter over `predict_structures`: that helper owns the largest-first
    ordering (so the widest XLA shape compiles once) and the copy/detach
    discipline; this only renames its output to the `*_predicted` schema the CLI
    writes. Optional columns are omitted entirely when the model produced none
    for any structure, because downstream code tests for column PRESENCE (see
    `select.py`: `if "gamma" not in df.columns`).

    ``features_gamma_min``: the ``cand_*`` columns (features of the atoms with
    gamma >= it), reduced per structure, so the all-atom features never outlive
    their structure; the ``features`` column only with ``save_features``.
    """
    extra = [uq_constants.ATOMIC_GAMMA, uq_constants.ATOMIC_SIGMA]
    if save_features or features_gamma_min is not None:
        extra.append(uq_constants.FEATURES)
    post = (None if features_gamma_min is None
            else _candidate_postprocess(features_gamma_min, save_features))
    out = predict_structures(
        atoms,
        calc,
        properties=("energy", "forces", "stress"),
        extra=tuple(extra),
        sort_by_natoms=sort_by_natoms,
        on_error="raise" if raise_errors else "warn",
        progress=progress,
        postprocess=post,
    )
    cols = {
        "energy_predicted": out["energy"],
        "forces_predicted": out["forces"],
        "stress_predicted": out["stress"],
    }
    renamed = {uq_constants.ATOMIC_GAMMA: "gamma", uq_constants.ATOMIC_SIGMA: "sigma"}
    for key in extra + (list(CAND_KEYS) if features_gamma_min is not None else []):
        if key in out and any(v is not None for v in out[key]):
            cols[renamed.get(key, key)] = out[key]
    return cols


def _artifact_sha256(args) -> str:
    """sha256 of --artifact (the CALM artifact of a yaml-mode model, or the one a
    SavedModel was exported with, passed for the record); "" without one."""
    if not args.artifact:
        return ""
    from tensorpotential.uq.hal.features import file_sha256

    return file_sha256(args.artifact)


def _emit_progress_line(done: int, total: int):
    """Default progress sink: emit a parser-friendly line on stdout for the
    subprocess master to aggregate."""
    print(format_progress(done, total), flush=True)


def run_worker(args, *, progress_emit=None) -> int:
    log.info(
        "Worker %d/%d: starting (model=%s, dataset=%s)",
        args.worker_id,
        args.n_workers,
        args.model,
        args.dataset,
    )
    df = getattr(args, "preloaded_df", None)
    if df is None:
        df = load_dataset_any(args.dataset, args.dump_elements)
    shard = slice_dataset_for_worker(df, args.worker_id, args.n_workers)
    log.info("Worker %d: %d structures assigned", args.worker_id, len(shard))

    calc = _build_calculator(args)

    total = len(shard)
    emit = progress_emit if progress_emit is not None else _emit_progress_line
    emit_interval = 1.0
    emit(0, total)
    last_emit = time.monotonic()

    def _progress(done, n):
        nonlocal last_emit
        now = time.monotonic()
        if done == n or (now - last_emit) >= emit_interval:
            emit(done, n)
            last_emit = now

    # `predict_structures` evaluates largest-first internally when asked, and
    # returns results in the shard's own order — so unlike the previous local
    # sort, the output rows line up with the input rows.
    pred_df = pd.DataFrame(
        _predict_columns(
            shard["ase_atoms"].tolist(),
            calc,
            args.save_features,
            args.raise_errors,
            sort_by_natoms=args.sort_by_natoms,
            progress=_progress,
            features_gamma_min=args.features_gamma_min,
        ),
        index=shard.index,
    )
    if args.features_gamma_min is not None:
        # what grace_uq select checks before using the candidate columns
        pred_df["cand_feature_gamma_min"] = float(args.features_gamma_min)
        pred_df["cand_artifact_sha256"] = _artifact_sha256(args)
    # Input columns the prediction writes again (a HAL pool's cand_*, or an earlier
    # prediction's gamma, ...) are replaced, never duplicated.
    replaced = [c for c in shard.columns if c in pred_df.columns]
    if replaced:
        if args.worker_id == 0:
            log.info("input columns replaced by this prediction: %s", ", ".join(replaced))
        shard = shard.drop(columns=replaced)
    out_df = pd.concat([shard, pred_df], axis=1)

    out_path = os.path.join(
        os.path.dirname(args.output) or ".",
        f".predict_w{args.worker_id}.pkl.gz",
    )
    out_df.to_pickle(out_path, compression="gzip")
    log.info("Worker %d: wrote %s (%d rows)", args.worker_id, out_path, len(out_df))
    return 0


# --------------------------------------------------------------------------
# Master
# --------------------------------------------------------------------------


def _advance_bar(bar, done: int, total: int):
    """Grow `bar.total` if needed and advance to `done` (no-op if already there)."""
    if bar.total != total:
        bar.total = total
        bar.refresh()
    delta = done - bar.n
    if delta > 0:
        bar.update(delta)


def _make_progress_aggregator(n_workers: int):
    """Return ``(handler, close)``: ``handler(worker_id, line)`` parses worker
    progress lines and updates a single tqdm bar; ``close()`` finalizes it."""
    state = [(0, 0)] * n_workers
    bar: tqdm | None = None

    def handler(worker_id: int, line: str):
        nonlocal bar
        parsed = parse_progress(line)
        if parsed is None:
            return
        state[worker_id] = parsed
        agg_total = sum(t for _, t in state)
        agg_done = sum(d for d, _ in state)
        if bar is None and agg_total > 0:
            bar = tqdm(
                total=agg_total,
                desc="predict",
                unit="struct",
                dynamic_ncols=True,
            )
        if bar is not None:
            _advance_bar(bar, agg_done, agg_total)

    def close():
        if bar is not None:
            bar.close()

    return handler, close


def _spawn_workers(args):
    cmds = []
    for i in range(args.n_workers):
        cmd = [
            sys.executable,
            "-m",
            "tensorpotential.uq.cli.predict",
            "--worker-id",
            str(i),
            "--n-workers",
            str(args.n_workers),
            "--model",
            args.model,
            "--dataset",
            args.dataset,
            "--output",
            args.output,
            "--pad-atoms-number",
            str(args.pad_atoms_number),
            "--pad-neighbors-fraction",
            str(args.pad_neighbors_fraction),
            "--max-number-reduction-recompilation",
            str(args.max_number_reduction_recompilation),
        ]
        if args.checkpoint:
            cmd += ["--checkpoint", args.checkpoint]
        if args.artifact:
            cmd += ["--artifact", args.artifact]
        if args.dump_elements:
            cmd += ["--dump-elements", *args.dump_elements]
        if args.save_features:
            cmd += ["--save-features"]
        if args.features_gamma_min is not None:
            cmd += ["--features-gamma-min", repr(float(args.features_gamma_min))]
        if args.sort_by_natoms:
            cmd += ["--sort-by-natoms"]
        if args.raise_errors:
            cmd += ["--raise-errors"]
        if args.verbose:
            cmd += ["--verbose"]
        cmds.append((i, cmd))

    gpus = visible_gpu_ids(args.gpus) or ["-1"]  # CUDA_VISIBLE_DEVICES respected; CPU
    progress_handler, progress_close = _make_progress_aggregator(args.n_workers)
    try:
        failed = spawn_and_monitor(
            cmds,
            gpus=gpus,
            threads_per_worker=args.threads_per_worker_int,
            verbose=args.verbose,
            label="Predict",
            progress_handler=progress_handler,
        )
    finally:
        progress_close()
    if failed:
        raise RuntimeError(f"predict workers {failed} failed")


def _merge_shards(args) -> pd.DataFrame:
    parts = []
    for i in range(args.n_workers):
        path = os.path.join(
            os.path.dirname(args.output) or ".", f".predict_w{i}.pkl.gz"
        )
        if not os.path.exists(path):
            raise RuntimeError(f"missing worker shard: {path}")
        parts.append(pd.read_pickle(path, compression="gzip"))
    merged = pd.concat(parts, axis=0)
    # Worker shards carry shard-local indices (0, 1, ... in every shard), so the input
    # order is only recoverable from the ``_orig_index`` column written by
    # ``slice_dataset_for_worker``; a single worker has no such column.
    if "_orig_index" in merged.columns:
        merged = merged.sort_values("_orig_index", kind="stable").drop(
            columns="_orig_index"
        )
    return merged.reset_index(drop=True)


def _cleanup_shards(args):
    for i in range(args.n_workers):
        path = os.path.join(
            os.path.dirname(args.output) or ".", f".predict_w{i}.pkl.gz"
        )
        try:
            os.remove(path)
        except OSError:
            pass


def run_master(args):
    args.threads_per_worker_int = resolve_threads_per_worker(
        args.threads_per_worker, args.n_workers
    )
    apply_master_thread_caps(args.threads_per_worker_int)

    if args.n_workers == 1:
        # Inline path: no subprocess overhead for single-worker runs.
        wargs = argparse.Namespace(**vars(args))
        wargs.worker_id = 0
        bar = tqdm(total=0, desc="predict", unit="struct", dynamic_ncols=True)
        try:
            run_worker(
                wargs,
                progress_emit=lambda done, total: _advance_bar(bar, done, total),
            )
        finally:
            bar.close()
            args.preloaded_df = wargs.preloaded_df = None  # free before the merge
    else:
        _spawn_workers(args)

    print(f"  [Master] merging {args.n_workers} shard(s)...", flush=True)
    df = _merge_shards(args)

    if args.no_ase_atoms and "ase_atoms" in df.columns:
        df = df.drop(columns=["ase_atoms"])
    elif is_extxyz_path(args.output) and "ase_atoms" not in df.columns:
        raise SystemExit(
            "extxyz output requires the ase_atoms column; remove --no-ase-atoms."
        )

    save_dataset_any(df, args.output, drop_ase_atoms=False)
    _cleanup_shards(args)
    print(f"  [Master] wrote {args.output} ({len(df)} structures)", flush=True)


# --------------------------------------------------------------------------
# Argument parsing
# --------------------------------------------------------------------------


_PREDICT_EPILOG = """\
Examples
--------
  # 1. Predict using a UQ-enabled SavedModel (output of `grace_uq build`):
  grace_uq predict --model saved_model/ \\
                   --dataset candidates.pkl.gz \\
                   --output predicted.pkl.gz

  # 2. Predict using model.yaml + checkpoint + artifact (no SavedModel needed):
  grace_uq predict --model model.yaml \\
                   --checkpoint checkpoint.best_test_loss \\
                   --artifact UQ/gmm_artifacts.npz \\
                   --dataset candidates.pkl.gz \\
                   --output predicted.pkl.gz

  # 3. Multi-worker (4 GPUs), include per-atom features [nat, D]:
  grace_uq predict --model saved_model/ \\
                   --dataset candidates.pkl.gz \\
                   --output predicted.pkl.gz \\
                   --n-workers 4 --gpus 0,1,2,3 \\
                   --save-features

  # 3b. Candidate-only features (atoms with gamma >= 1), the input of
  #     `grace_uq select` without all-atom features in memory:
  grace_uq predict --model model.yaml --checkpoint checkpoint.best_test_loss \\
                   --artifact UQ/gmm_artifacts.npz \\
                   --dataset candidates.pkl.gz --output predicted.pkl.gz \\
                   --features-gamma-min 1.0

  # 4. Save as extxyz (per-atom features are dropped — extxyz cannot store them):
  grace_uq predict --model saved_model/ \\
                   --dataset candidates.pkl.gz \\
                   --output predicted.xyz

  # 5. Drop ase_atoms column from the pkl.gz output (smaller file):
  grace_uq predict --model saved_model/ \\
                   --dataset candidates.pkl.gz \\
                   --output predicted.pkl.gz \\
                   --no-ase-atoms

  # 6. Tune XLA padding for a heterogeneous dataset (fewer recompiles):
  grace_uq predict --model saved_model/ \\
                   --dataset candidates.pkl.gz \\
                   --output predicted.pkl.gz \\
                   --pad-atoms-number 32 \\
                   --pad-neighbors-fraction 0.40
"""


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="grace_uq predict",
        description="Predict E/F/S and per-atom gamma using a UQ-enabled GRACE model.",
        epilog=_PREDICT_EPILOG,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument(
        "--model",
        required=True,
        help="SavedModel directory (with compute_uq) OR model.yaml file.",
    )
    p.add_argument(
        "--checkpoint",
        help="Required iff --model is a model.yaml.",
    )
    p.add_argument(
        "--artifact",
        help="UQ .npz artifact. Required if --model is a model.yaml. With a SavedModel "
        "and --features-gamma-min, pass it too: its sha256 is recorded in the output "
        "and `grace_uq select --uq/--artifact` checks it.",
    )
    p.add_argument(
        "--dataset",
        required=True,
        help="Input dataset (.pkl.gz default; .xyz/.extxyz; LAMMPS text dump).",
    )
    p.add_argument(
        "--dump-elements",
        nargs="+",
        metavar="EL",
        help="LAMMPS dump --dataset without an element column: element of atom type 1, "
        "2, ... (e.g. --dump-elements Al Li). Ignored when the dump has the column.",
    )
    p.add_argument(
        "--output",
        default="predicted.pkl.gz",
        help="Output path (extension drives format).",
    )
    p.add_argument(
        "--n-workers",
        type=parse_n_workers,
        default="auto",
        help="worker processes, or auto: min(4, ceil(n_structures / 200)), round robin "
        "over the visible GPUs (CUDA_VISIBLE_DEVICES respected; CPU without one); a "
        "single worker runs in this process.",
    )
    p.add_argument(
        "--gpus",
        default=None,
        help="comma-separated GPU ids for the workers (round robin); default: the "
        "visible GPUs with --n-workers auto, GPU 0 with an explicit --n-workers.",
    )
    p.add_argument("--threads-per-worker", default="auto")
    p.add_argument(
        "--no-ase-atoms",
        action="store_true",
        help="Drop ase_atoms column from the saved output.",
    )
    p.add_argument(
        "--save-features",
        action="store_true",
        help="Add per-atom features [nat, D] column (model-native dtype).",
    )
    p.add_argument(
        "--features-gamma-min",
        type=float,
        default=None,
        metavar="X",
        help="Store only the features of atoms with gamma >= X, as the per-row "
        "cand_atom_index / cand_features (float32) / cand_gamma columns of a HAL pool "
        "(plus cand_feature_gamma_min, cand_artifact_sha256 of --artifact); "
        "`grace_uq select` then selects from them like from a HAL pool, without "
        "all-atom features in memory. With a SavedModel, pass --artifact (the npz it "
        "was exported with) to record its hash for select --artifact. Combine with "
        "--save-features to also keep the all-atom `features` column.",
    )
    p.add_argument(
        "--sort-by-natoms",
        action="store_true",
        default=True,
        help="Evaluate each worker's shard largest-structure-first so the widest "
        "XLA shape is compiled once (default ON). Performance only — output rows "
        "keep the input order either way.",
    )
    p.add_argument(
        "--no-sort-by-natoms",
        dest="sort_by_natoms",
        action="store_false",
    )
    p.add_argument("--pad-atoms-number", type=int, default=20)
    p.add_argument("--pad-neighbors-fraction", type=float, default=0.30)
    p.add_argument("--max-number-reduction-recompilation", type=int, default=10)
    p.add_argument(
        "--raise-errors",
        action="store_true",
        help="Stop on the first prediction error.",
    )
    p.add_argument("--verbose", action="store_true")

    # Internal worker flags
    p.add_argument("--worker-id", type=int, default=None)
    return p


def _resolve_paths(args):
    args.model = os.path.abspath(args.model)
    args.dataset = os.path.abspath(args.dataset)
    args.output = os.path.abspath(args.output)
    if args.checkpoint:
        args.checkpoint = os.path.abspath(args.checkpoint)
    if args.artifact:
        args.artifact = os.path.abspath(args.artifact)


def predict_main(argv=None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    _resolve_paths(args)
    if args.features_gamma_min is not None and is_extxyz_path(args.output):
        parser.error("--features-gamma-min needs a .pkl.gz --output (extxyz cannot store "
                     "the candidate columns)")
    if args.features_gamma_min is not None and args.no_ase_atoms:
        parser.error("--features-gamma-min cannot be combined with --no-ase-atoms: "
                     "grace_uq select needs the structures (element counts, output) "
                     "next to the candidate columns")
    if (args.worker_id is None and args.features_gamma_min is not None
            and not args.artifact):
        log.warning("--features-gamma-min without --artifact: the candidate columns carry "
                    "no artifact hash, so `grace_uq select --artifact` will refuse them; "
                    "pass the CALM artifact the model was exported with.")

    if args.worker_id is not None:
        return run_worker(args)
    resolve_n_workers(args)
    run_master(args)
    return 0


def single_worker_device(gpus_arg, gpus, environ):
    """CUDA_VISIBLE_DEVICES for the one in-process auto worker, or None to leave it.

    ``--gpus`` given: its first GPU, or "-1" (CPU) when it names none. Otherwise a
    CUDA_VISIBLE_DEVICES already set is left alone; with nothing set, the first GPU
    found (so a multi-GPU host does not hand TF every device), or None without one.
    """
    if gpus_arg is not None:
        return gpus[0] if gpus else "-1"
    if environ.get("CUDA_VISIBLE_DEVICES") is not None:
        return None
    return gpus[0] if gpus else None


def resolve_n_workers(args):
    """--n-workers auto -> min(4, ceil(n_structures / 200)) workers, round robin over the
    visible GPUs (on CPU without one); a single worker runs in this process and reuses
    the dataset read here for its size."""
    if args.n_workers != "auto":
        return
    df = load_dataset_any(args.dataset, args.dump_elements)
    n = auto_n_workers(len(df))
    gpus = visible_gpu_ids(args.gpus)
    args.n_workers = n
    if n == 1:
        args.preloaded_df = df  # the in-process worker does not read the file again
        device = single_worker_device(args.gpus, gpus, os.environ)
        if device is not None:  # before TF creates its devices
            os.environ["CUDA_VISIBLE_DEVICES"] = device
    else:
        args.gpus = ",".join(gpus) if gpus else "-1"  # "-1": CPU workers
    del df
    where = ("in this process" if n == 1
             else f"on GPUs {args.gpus}" if gpus else "on CPU")
    log.info("--n-workers auto: %d worker(s) %s (%d visible GPU(s))", n, where, len(gpus))


if __name__ == "__main__":
    raise SystemExit(predict_main())
