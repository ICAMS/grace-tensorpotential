"""argparse helpers for grace_uq build (custom Action + epilog text)."""

from __future__ import annotations

import argparse


class _WeightedTrainDataAction(argparse.Action):
    """Append-style action for ``--train-data-weighted W FILE [FILE ...]``.

    Each invocation collects one ``(weight: float, files: list[str])`` group
    onto ``namespace.train_data_weighted``. Repeating the flag accumulates
    multiple groups, so the user can assign different weights to different
    source datasets in a single CLI call:

        --train-data-weighted 20.0 /OMAT/shard_*.pkl.gz \
        --train-data-weighted 1.0  /SMAX/shard_*.pkl.gz

    Shell glob expansion happens before argparse sees the tokens; we do not
    re-glob inside Python.
    """

    def __call__(self, parser, namespace, values, option_string=None):
        if len(values) < 2:
            parser.error(
                f"{option_string} requires at least one weight followed by one "
                f"or more file paths; got {values!r}"
            )
        try:
            weight = float(values[0])
        except (TypeError, ValueError):
            parser.error(
                f"{option_string}: first token must be a numeric weight; got {values[0]!r}"
            )
        if weight <= 0:
            parser.error(
                f"{option_string}: weight must be > 0; got {weight}"
            )
        files = list(values[1:])
        prior = getattr(namespace, self.dest, None) or []
        prior.append((weight, files))
        setattr(namespace, self.dest, prior)


_BUILD_EPILOG = """\
Examples
--------
  # 1. Single GPU, single worker (default cluster scan):
  grace_uq build --model-yaml model.yaml \\
                 --checkpoint checkpoint.best_test_loss \\
                 --train-data training_set.pkl.gz \\
                 --artifact-path UQ/gmm_artifacts.npz

  # 2. 4 workers across 4 GPUs, elbow scan over k = 1,2,4,8,16:
  grace_uq build --model-yaml model.yaml \\
                 --checkpoint checkpoint \\
                 --train-data training_set.pkl.gz \\
                 --n-workers 4 --gpus 0,1,2,3 \\
                 --n-clusters 1 2 4 8 16 \\
                 --artifact-path UQ/gmm_artifacts.npz

  # 3. Sharded TF dataset (auto-detected by --train-data being a directory):
  grace_uq build --model-yaml model.yaml \\
                 --checkpoint checkpoint \\
                 --train-data dataset_shards/ \\
                 --n-workers 8 --gpus 0,1,2,3 \\
                 --artifact-path UQ/gmm_artifacts.npz

  # 4. Re-export a SavedModel from existing artifacts (no training data):
  grace_uq build --model-yaml model.yaml \\
                 --checkpoint checkpoint \\
                 --artifact-path UQ/gmm_artifacts.npz

  # 5. Skip SavedModel export (faster; useful when iterating on artifacts):
  grace_uq build --model-yaml model.yaml \\
                 --checkpoint checkpoint \\
                 --train-data training_set.pkl.gz \\
                 --artifact-path UQ/gmm_artifacts.npz \\
                 --no-export

  # 6. Restart a partial run from scratch (deletes intermediate .step*.npz):
  grace_uq build --model-yaml model.yaml \\
                 --checkpoint checkpoint \\
                 --train-data training_set.pkl.gz \\
                 --artifact-path UQ/gmm_artifacts.npz \\
                 --restart

  # 7. Subsample 20% of training data for a faster artifact build:
  grace_uq build --model-yaml model.yaml \\
                 --checkpoint checkpoint \\
                 --train-data training_set.pkl.gz \\
                 --frac 0.2 \\
                 --artifact-path UQ/gmm_artifacts_quick.npz
"""
