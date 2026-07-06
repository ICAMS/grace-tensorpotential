"""Utilities for discovering and distributing TF dataset shards."""

import glob
import os

SHARD_GLOB_PATTERN = "shard_*-of-*"


def discover_shards(dataset_path: str) -> list[str]:
    """Return sorted list of shard directories under dataset_path."""
    return sorted(glob.glob(os.path.join(dataset_path, SHARD_GLOB_PATTERN)))


def is_sharded_dataset(paths: list[str]) -> bool:
    """Check if paths point to a single directory containing TF dataset shards."""
    if len(paths) != 1 or not os.path.isdir(paths[0]):
        return False
    return len(discover_shards(paths[0])) > 0


def distribute_shards(dataset_path: str, worker_id: int, n_workers: int) -> list[str]:
    """Return the shard paths assigned to a given worker (round-robin)."""
    all_shards = discover_shards(dataset_path)
    return all_shards[worker_id::n_workers]
