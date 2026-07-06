"""GRACE-UQ build subcommand (split package).

This package replaces the former single-file ``tensorpotential.uq.cli.build``
module. The public entry point is :func:`build_main`; tests and the
``grace_uq`` wrapper script import symbols off this package (re-exported
below) so the external import surface is unchanged.

Worker subprocesses re-enter via ``python -m tensorpotential.uq.cli.build``.
"""

from __future__ import annotations

# absl is the only TF-noise gate that still works at this point: by the time
# ``tensorpotential.uq.cli.build`` is imported, the parent ``tensorpotential``
# package has already imported TensorFlow, so any ``TF_CPP_MIN_LOG_LEVEL``-style
# env vars set here would be no-ops. ``absl.logging`` is configurable at runtime.
try:
    from absl import logging as _absl_logging

    _absl_logging.set_verbosity(_absl_logging.ERROR)
except ImportError:
    pass

from tensorpotential.uq.cli.build.cli_args import (  # noqa: E402
    _BUILD_EPILOG,
    _WeightedTrainDataAction,
)
from tensorpotential.uq.cli.build.clusters import (  # noqa: E402
    _compact_step2_artifacts,
    _merge_small_clusters,
    _merge_step2_clusters,
    _pick_centroids_for_element,
)
from tensorpotential.uq.cli.build.data_resolve import (  # noqa: E402
    DEFAULT_TRAIN_DATA,
    _load_and_subsample,
    _load_filter_fn,
    _prefetch,
    _resolve_train_data_from_input_yaml,
    _resolve_weighted_train_data,
)
from tensorpotential.uq.cli.build.master import build_main, run_master  # noqa: E402
from tensorpotential.uq.cli.build.paths import (  # noqa: E402
    _iter_worker_files,
    _worker_dir,
    _worker_output_paths,
)
from tensorpotential.uq.cli.build.streaming import (  # noqa: E402
    StreamingEstimate,
    stream_atoms,
)
from tensorpotential.uq.cli.build.thresholds import (  # noqa: E402
    _compute_dual_thresholds,
    _normalize_inertia,
    _print_covariance_diagnostics,
    _save_elbow_plot,
    _save_elbow_report,
    select_optimal_clusters,
)
from tensorpotential.uq.cli.build.workers import (  # noqa: E402
    export_savedmodel,
    run_worker_step1,
    run_worker_step2,
    run_worker_step3,
)

__all__ = [
    "DEFAULT_TRAIN_DATA",
    "StreamingEstimate",
    "_BUILD_EPILOG",
    "_WeightedTrainDataAction",
    "_compact_step2_artifacts",
    "_compute_dual_thresholds",
    "_iter_worker_files",
    "_load_and_subsample",
    "_load_filter_fn",
    "_merge_small_clusters",
    "_merge_step2_clusters",
    "_normalize_inertia",
    "_pick_centroids_for_element",
    "_prefetch",
    "_print_covariance_diagnostics",
    "_resolve_train_data_from_input_yaml",
    "_resolve_weighted_train_data",
    "_save_elbow_plot",
    "_save_elbow_report",
    "_worker_dir",
    "_worker_output_paths",
    "build_main",
    "export_savedmodel",
    "run_master",
    "run_worker_step1",
    "run_worker_step2",
    "run_worker_step3",
    "select_optimal_clusters",
    "stream_atoms",
]
