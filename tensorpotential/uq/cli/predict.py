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

from tensorpotential.uq import constants as uq_constants
from tensorpotential.uq.cli._common import (
    apply_master_thread_caps,
    detect_model_kind,
    format_progress,
    is_extxyz_path,
    load_dataset_any,
    parse_progress,
    resolve_threads_per_worker,
    save_dataset_any,
    slice_dataset_for_worker,
    spawn_and_monitor,
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


def _predict_one(at, calc, save_features, raise_errors):
    at = at.copy()
    at.calc = calc
    try:
        e = at.get_potential_energy()
        f = at.get_forces()
        try:
            s = at.get_stress()
        except Exception:
            s = None
    except Exception as exc:
        if raise_errors:
            raise
        log.warning("predict failed for n_atoms=%d: %s", len(at), exc)
        return {}
    out = {
        "energy_predicted": float(e),
        "forces_predicted": np.asarray(f),
        "stress_predicted": np.asarray(s) if s is not None else None,
    }
    res = getattr(calc, "results", {}) or {}
    if uq_constants.ATOMIC_GAMMA in res:
        out["gamma"] = np.asarray(res[uq_constants.ATOMIC_GAMMA])
    elif "gamma" in res:
        out["gamma"] = np.asarray(res["gamma"])
    if uq_constants.ATOMIC_SIGMA in res:
        out["sigma"] = np.asarray(res[uq_constants.ATOMIC_SIGMA])
    if save_features and uq_constants.FEATURES in res:
        out[uq_constants.FEATURES] = np.asarray(res[uq_constants.FEATURES])
    return out


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
    df = load_dataset_any(args.dataset)
    shard = slice_dataset_for_worker(df, args.worker_id, args.n_workers)
    if args.sort_by_natoms and "ase_atoms" in shard.columns:
        shard["_n_atoms"] = shard["ase_atoms"].map(len)
        shard = shard.sort_values("_n_atoms", ascending=False).drop(columns="_n_atoms")
        shard = shard.reset_index(drop=True)
    log.info("Worker %d: %d structures assigned", args.worker_id, len(shard))

    calc = _build_calculator(args)

    total = len(shard)
    pred_rows = []
    emit = progress_emit if progress_emit is not None else _emit_progress_line
    emit_interval = 1.0
    emit(0, total)
    last_emit = time.monotonic()
    for done, (_, row) in enumerate(shard.iterrows(), start=1):
        out = _predict_one(
            row["ase_atoms"], calc, args.save_features, args.raise_errors
        )
        pred_rows.append(out)
        now = time.monotonic()
        if done == total or (now - last_emit) >= emit_interval:
            emit(done, total)
            last_emit = now

    pred_df = pd.DataFrame(pred_rows, index=shard.index)
    out_df = pd.concat([shard.drop(columns=[]), pred_df], axis=1)

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
        if args.save_features:
            cmd += ["--save-features"]
        if args.sort_by_natoms:
            cmd += ["--sort-by-natoms"]
        if args.raise_errors:
            cmd += ["--raise-errors"]
        if args.verbose:
            cmd += ["--verbose"]
        cmds.append((i, cmd))

    gpus = args.gpus.split(",")
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
    merged = pd.concat(parts, axis=0).sort_index()
    return merged


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
        help="UQ .npz artifact. Required iff --model is a model.yaml.",
    )
    p.add_argument(
        "--dataset", required=True, help="Input dataset (.pkl.gz default; .xyz/.extxyz)."
    )
    p.add_argument(
        "--output",
        default="predicted.pkl.gz",
        help="Output path (extension drives format).",
    )
    p.add_argument("--n-workers", type=int, default=4)
    p.add_argument("--gpus", default="0")
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
        "--sort-by-natoms",
        action="store_true",
        default=True,
        help="Sort each worker's shard by descending atom count (default ON).",
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

    if args.worker_id is not None:
        return run_worker(args)
    run_master(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(predict_main())
