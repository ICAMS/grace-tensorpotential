"""Per-worker structure-streaming generator + progressive count estimate."""

from __future__ import annotations

import logging

import os

import numpy as np

from tensorpotential.uq import constants as uq_constants
from tensorpotential.uq.cli.build.data_resolve import _load_and_subsample
from tensorpotential.uq.cli.build.paths import _iter_worker_files

log = logging.getLogger("grace_uq")

class StreamingEstimate:
    """Progressively refined structure-count estimate for progress bars.

    Starts with an equal-size assumption (total_files / my_files * 0 = 0) and
    refines as each file is loaded by the generator.  The ProgressPrinter reads
    ``.total`` on each emit, so the ETA improves over time.
    """

    def __init__(self, my_n_files):
        self.my_n_files = my_n_files
        self.files_loaded = 0
        self.structs_loaded = 0
        self.total = my_n_files  # conservative initial estimate (1 struct/file)

    def file_loaded(self, n_structs):
        self.files_loaded += 1
        self.structs_loaded += n_structs
        remaining = self.my_n_files - self.files_loaded
        avg = self.structs_loaded / self.files_loaded
        self.total = self.structs_loaded + int(avg * remaining)


def stream_atoms(
    files,
    worker_id,
    n_workers,
    frac=None,
    seed=123,
    estimate=None,
    file_weight_map=None,
    filter_fn=None,
):
    """Yield per-structure ``pd.Series`` rows assigned to this worker.

    Each row carries an ``ase_atoms`` column (with ``.info[UQ_WEIGHT]`` set
    so the downstream ``PerAtomWeightBuilder`` can attach the per-atom weight
    array inside ``batched_feature_iterator``) and the columns the
    ``ReferenceEnergyForcesStressesDataBuilder`` needs (``energy_corrected``,
    ``forces``). The default weight is 1.0 (back-compat with unweighted
    builds).

    Parameters
    ----------
    file_weight_map : dict[str, float], optional
        Map of *absolute* file path → weight. Files not in the map default to
        1.0. ``_resolve_weighted_train_data`` produces the right shape.
    filter_fn : callable, optional
        ``filter_atoms(atoms, file_path) -> bool``. Applied per shard before
        subsampling.
    estimate : StreamingEstimate, optional
        Updated as each file is loaded so external progress bars refine ETA.
    """
    rng = np.random.RandomState(seed) if frac is not None else None
    weight_map = file_weight_map or {}
    for file_path, my_rel_idx, num_here in _iter_worker_files(files, worker_id, n_workers):
        df = _load_and_subsample(file_path, rng, frac, filter_fn=filter_fn)
        if num_here > 1:
            df = df.iloc[my_rel_idx::num_here].reset_index(drop=True)
        if estimate is not None:
            estimate.file_loaded(len(df))
        # abspath to match the keys _resolve_weighted_train_data produces.
        weight = float(weight_map.get(os.path.abspath(file_path), 1.0))
        for _, row in df.iterrows():
            atom = row["ase_atoms"]
            # Some pickled/custom-built Atoms come in with ``.info = None``;
            # promote to dict before writing the weight.
            if atom.info is None:
                atom.info = {}
            atom.info[uq_constants.UQ_WEIGHT] = weight
            yield row
