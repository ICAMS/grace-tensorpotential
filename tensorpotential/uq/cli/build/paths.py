"""Filesystem path helpers for grace_uq build (per-worker and per-step file layout)."""

from __future__ import annotations

import os


def _worker_output_paths(
    work_dir: str,
    step_idx: int,
    worker_id: int,
    n_clusters: list[int],
) -> list[str]:
    """Paths a worker writes for the given step.

    Step 1 writes one file per candidate K (``.step1_w{i}_k{K}.npz``).
    Steps 2 and 3 each write a single ``.step{step}_w{i}.npz`` file.
    """
    if step_idx == 1:
        return [
            os.path.join(work_dir, f".step1_w{worker_id}_k{k}.npz")
            for k in n_clusters
        ]
    return [os.path.join(work_dir, f".step{step_idx}_w{worker_id}.npz")]


def _worker_dir(args):
    """Directory for intermediate files, derived from --artifact-path."""
    return os.path.dirname(args.artifact_path) or "."


def _iter_worker_files(files, worker_id, n_workers):
    """Yield (file_path, my_rel_idx, num_workers_sharing) for this worker."""
    n_files = len(files)
    if n_files >= n_workers:
        for file_path in files[worker_id::n_workers]:
            yield file_path, 0, 1
    else:
        for f_idx, file_path in enumerate(files):
            w_start = (f_idx * n_workers) // n_files
            w_end = ((f_idx + 1) * n_workers) // n_files
            workers = list(range(w_start, w_end))
            if worker_id not in workers:
                continue
            yield file_path, workers.index(worker_id), len(workers)
