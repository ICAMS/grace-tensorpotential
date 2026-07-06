"""Feature extraction from trained GRACE models for UQ pipelines."""

from __future__ import annotations

import time
from typing import Iterator

import numpy as np
import tensorflow as tf

from tensorpotential.tpmodel import ComputeStructureEnergyAndForcesAndVirial
from tensorpotential.calculator.asecalculator import TPCalculator
from tensorpotential import constants
from tensorpotential.data.databuilder import AbstractDataBuilder, symbols_to_indices
from tensorpotential.uq import constants as uq_constants


class PerAtomWeightBuilder(AbstractDataBuilder):
    """Emits a per-atom weight array, sourced from ``atoms.info[UQ_WEIGHT]``.

    Used by :func:`batched_feature_iterator` to carry the UQ build weight
    assigned to each structure (one weight per source shard) through the
    StreamingDatasetWrapper bucketing reorder. The per-atom array travels
    alongside ATOMIC_MU_I and stays aligned across batches.

    Structures with no ``UQ_WEIGHT`` info default to 1.0.
    """

    def extract_from_ase_atoms(self, ase_atoms, **kwarg):
        info = ase_atoms.info or {}
        w = float(info.get(uq_constants.UQ_WEIGHT, 1.0))
        return {uq_constants.UQ_WEIGHT: np.full(len(ase_atoms), w, dtype=np.float64)}

    def extract_from_row(self, row, **kwarg):
        return self.extract_from_ase_atoms(row["ase_atoms"])

    def join_to_batch(self, pre_batch_list):
        arrs = [d[uq_constants.UQ_WEIGHT] for d in pre_batch_list]
        return {uq_constants.UQ_WEIGHT: np.concatenate(arrs, axis=0)}

    def pad_batch(self, batch, max_pad_dict):
        max_nat = int(max_pad_dict[constants.PAD_MAX_N_ATOMS])
        cur = batch[uq_constants.UQ_WEIGHT]
        pad = max_nat - cur.shape[0]
        if pad > 0:
            batch[uq_constants.UQ_WEIGHT] = np.pad(
                cur, (0, pad), mode="constant", constant_values=0.0
            )


class ProgressPrinter:
    """Pipe-safe progress reporter that prints one status line per interval.

    Drop-in replacement for tqdm in contexts where stdout is piped
    (e.g. subprocess workers) and carriage-return updates would flood
    the parent process.
    """

    def __init__(self, total=None, desc="", unit="items", interval=10.0):
        self.total = total
        self.desc = desc
        self.unit = unit
        self.interval = interval
        self._n = 0
        self._t0 = time.time()
        self._last_print = 0.0

    def update(self, n=1):
        self._n += n
        now = time.time()
        if now - self._last_print >= self.interval:
            self._emit(now)
            self._last_print = now

    def close(self):
        self._emit(final=True)

    def _emit(self, now=None, final=False):
        now = now or time.time()
        elapsed = now - self._t0
        parts = [self.desc]
        if self.total is not None and self.total > 0:
            pct = 100.0 if final else min(100, 100 * self._n / self.total)
            parts.append(f"{pct:.0f}% ({self._n}/{self.total} {self.unit})")
            parts.append(f"elapsed {self._fmt(elapsed)}")
            if not final and pct > 0:
                eta = elapsed * (100 / pct - 1)
                parts.append(f"ETA ~{self._fmt(eta)}")
        else:
            parts.append(f"{self._n} {self.unit}")
            parts.append(f"elapsed {self._fmt(elapsed)}")
        if final:
            parts.append("done")
        print(" | ".join(parts), flush=True)

    @staticmethod
    def _fmt(seconds):
        m, s = divmod(int(seconds), 60)
        h, m = divmod(m, 60)
        if h:
            return f"{h}h{m:02d}m{s:02d}s"
        if m:
            return f"{m}m{s:02d}s"
        return f"{s}s"


# Canonical dtype map for converting compute_specs dtype strings to TF dtypes.
# Used by batched_feature_iterator and tf_dataset_feature_iterator.
COMPUTE_SPEC_DTYPE_MAP = {
    "int": tf.int32,
    "int32": tf.int32,
    "int64": tf.int64,
    "float": tf.float64,
    "float32": tf.float32,
    "float64": tf.float64,
}


def setup_feature_calculator(
    model_yaml_path: str,
    checkpoint_path: str,
    hidden_target_name: str = uq_constants.FEATURES,
    param_dtype=None,
    feature_spec: dict | None = None,
) -> TPCalculator:
    """Load a trained GRACE model and return a TPCalculator that extracts the
    basis-RP UQ feature.

    Parameters
    ----------
    model_yaml_path : str
        Path to model.yaml file.
    checkpoint_path : str
        Path to checkpoint file (e.g. checkpoint.best_test_loss.index).
    hidden_target_name : str
        Key under which the feature is stored in results (canonical FEATURES).
    param_dtype : tf.DType, optional
        Parameter dtype for the model. If None (default), inferred from
        model.yaml metadata — matches the dtype the model was trained with.
    feature_spec : dict, optional
        basis-RP projection spec ``{out_dim, seed, matrix}`` forwarded to
        ``load_uq_model``. Defaults to the canonical 128-D / seed-42 projection.

    Returns
    -------
    TPCalculator configured to extract the basis-RP feature.
    """
    from tensorpotential.uq.factories import load_uq_model

    compute_fn = ComputeStructureEnergyAndForcesAndVirial(
        extra_return_keys=[hidden_target_name]
    )
    tp, _ = load_uq_model(
        model_yaml=model_yaml_path,
        checkpoint=checkpoint_path,
        model_compute_function=compute_fn,
        param_dtype=param_dtype,
        feature_spec=feature_spec,
    )
    return TPCalculator(
        tp.model,
        extra_properties=[hidden_target_name],
        truncate_extras_by_natoms=True,
    )


def _build_element_map(atoms_iterable):
    """Build {symbol: idx} map from a collection of atoms objects."""
    all_symbols = set()
    for at in atoms_iterable:
        all_symbols.update(at.get_chemical_symbols())
    sorted_symbols = sorted(all_symbols)
    return {s: i for i, s in enumerate(sorted_symbols)}


def extract_features(
    calc: TPCalculator,
    atoms_iterable,
    element_map: dict = None,
) -> Iterator[tuple[np.ndarray, np.ndarray]]:
    """Generator yielding (features [N_atoms, D], element_indices [N_atoms]) per structure.

    Parameters
    ----------
    calc : TPCalculator
        Calculator with hidden feature extraction enabled.
    atoms_iterable : iterable of ase.Atoms
        Structures to extract features from.
    element_map : dict, optional
        {symbol: element_type_index}, e.g. {"Cu": 0, "Zn": 1}.
        If None, built from first pass (requires atoms_iterable to be a list/sequence).
    """
    if element_map is None:
        atoms_iterable = list(atoms_iterable)
        element_map = _build_element_map(atoms_iterable)

    for at in atoms_iterable:
        at.calc = calc
        at.get_potential_energy()
        features = calc.results[uq_constants.FEATURES]
        elem_idx = symbols_to_indices(at.get_chemical_symbols(), element_map)
        yield features, elem_idx


def extract_features_bulk(
    calc: TPCalculator,
    atoms_iterable,
    element_map: dict = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Collect all features into single arrays.

    Returns
    -------
    (all_features [total_atoms, D], all_element_indices [total_atoms])
    """
    all_feats = []
    all_elems = []
    for feats, elems in extract_features(calc, atoms_iterable, element_map):
        all_feats.append(feats)
        all_elems.append(elems)
    return np.vstack(all_feats), np.concatenate(all_elems)


class FeatureBuffer:
    """Pre-allocated numpy buffer that accumulates features contiguously in memory."""

    def __init__(self, feature_dim: int = 129, capacity: int = 1_000_000):
        self._features = np.empty((capacity, feature_dim), dtype=np.float32)
        self._elements = np.empty(capacity, dtype=np.int32)
        self._size = 0
        self._capacity = capacity

    def append(self, features: np.ndarray, element_indices: np.ndarray):
        n = len(features)
        if self._size + n > self._capacity:
            self._capacity = max(self._capacity * 2, self._size + n)
            new_f = np.empty(
                (self._capacity, self._features.shape[1]), dtype=self._features.dtype
            )
            new_f[: self._size] = self._features[: self._size]
            self._features = new_f
            new_e = np.empty(self._capacity, dtype=self._elements.dtype)
            new_e[: self._size] = self._elements[: self._size]
            self._elements = new_e
        self._features[self._size : self._size + n] = features
        self._elements[self._size : self._size + n] = element_indices
        self._size += n

    @property
    def features(self) -> np.ndarray:
        return self._features[: self._size]

    @property
    def elements(self) -> np.ndarray:
        return self._elements[: self._size]

    def __len__(self):
        return self._size

    def iter_chunks(self, chunk_size: int = 4096):
        """Yields (features, elements) slices. Re-iterable."""
        for i in range(0, self._size, chunk_size):
            end = min(i + chunk_size, self._size)
            yield self._features[i:end], self._elements[i:end]


def batch_feature_chunks(
    per_structure_iterator: Iterator[tuple[np.ndarray, np.ndarray]],
    chunk_size: int = 4096,
) -> Iterator[tuple[np.ndarray, np.ndarray]]:
    """Collect per-structure (features, elements) yields into fixed-size chunks.

    Use this to batch the small per-structure outputs of extract_features()
    into larger chunks suitable for efficient GMM evaluation or artifact building.

    Parameters
    ----------
    per_structure_iterator : iterable of (np.ndarray [N_i, D], np.ndarray [N_i])
        E.g. from extract_features().
    chunk_size : int
        Target number of atoms per yielded chunk.

    Yields
    ------
    (features [~chunk_size, D], element_indices [~chunk_size])
    """
    feat_acc = []
    elem_acc = []
    n_acc = 0

    for feats, elems in per_structure_iterator:
        feat_acc.append(feats)
        elem_acc.append(elems)
        n_acc += len(feats)
        if n_acc >= chunk_size:
            yield np.vstack(feat_acc), np.concatenate(elem_acc)
            feat_acc.clear()
            elem_acc.clear()
            n_acc = 0

    if n_acc > 0:
        yield np.vstack(feat_acc), np.concatenate(elem_acc)


def batched_feature_iterator(
    atoms_list,
    model,
    element_map: dict,
    cutoff: float,
    cutoff_dict: dict = None,
    max_num_neighbours_per_batch: int = 3000,
    hidden_target_name: str = uq_constants.FEATURES,
    verbose: bool = True,
    desc: str = "Batched features",
    total_num_structures=None,
) -> Iterator[tuple[np.ndarray, np.ndarray, np.ndarray]]:
    """Yield (features, element_indices, weights) per batch using batched inference.

    Each output is a flat array of length N_real (the real-atom count of the batch):
    features [N_real, D], element_indices [N_real], weights [N_real] (float64).

    Per-atom weights come from ``atoms.info[UQ_WEIGHT]`` via PerAtomWeightBuilder;
    structures with no such entry default to weight 1.0.

    Uses StreamingDatasetWrapper for elastic metric batching — many structures per
    forward pass — minimizing XLA recompilations and padding overhead.

    Parameters
    ----------
    atoms_list : list of ase.Atoms
    model : TPModel
        Model with the UQ feature enabled: the basis-RP instruction
        (RandomProjectedBasisFeatures) appended so it writes the canonical
        FEATURES key, with ``hidden_target_name`` in extra_return_keys.
        Must have decorate_compute_function() already called.
    element_map : dict
        {symbol: element_type_index} — must match the model's training element map.
    cutoff : float
        Neighbor cutoff in Angstrom. Use extract_cutoff_and_elements() to get from model.
    max_num_neighbours_per_batch : int
        Target number of neighbours per batch (controls batch size).
    hidden_target_name : str
        Key for hidden features in model output.
    verbose : bool
        Show progress bar.
    """
    from tensorpotential.data.databuilder import GeometricalDataBuilder
    from tensorpotential.data.streaming import StreamingDatasetWrapper, StreamingConfig

    geom_builder = GeometricalDataBuilder(
        elements_map=element_map, cutoff=cutoff, cutoff_dict=cutoff_dict
    )
    weight_builder = PerAtomWeightBuilder()
    data_builders = [geom_builder, weight_builder]

    config = StreamingConfig(
        target_metric=max_num_neighbours_per_batch,
        metric_strategy="neighbours",
        num_bins=10,
    )
    wrapper = StreamingDatasetWrapper(
        atoms_list,
        data_builders=data_builders,
        config=config,
        shuffle=False,
        name=desc,
    )

    # Build model key → dtype map for manual conversion (same logic as decorate_compute_function).
    # We do NOT call set_model_signatures so that non-model keys (N_STRUCTURES_BATCH_REAL
    # etc.) remain available in each batch for progress tracking.
    model_keys = {
        k: COMPUTE_SPEC_DTYPE_MAP[v["dtype"]] for k, v in model.compute_specs.items()
    }

    try:
        total_structs = len(atoms_list)
    except TypeError:
        total_structs = total_num_structures

    # StreamingEstimate has a .total attribute that refines as files load;
    # plain int is used directly.
    _streaming_estimate = hasattr(total_structs, "total")
    pbar_total = total_structs.total if _streaming_estimate else total_structs
    pbar = (
        ProgressPrinter(total=pbar_total, desc=desc, unit="structs")
        if verbose
        else None
    )
    try:
        for batch in wrapper:
            batch_tf = {
                k: tf.constant(batch[k], dtype=dt)
                for k, dt in model_keys.items()
                if k in batch
            }
            output = model.compute(batch_tf)
            n_real = int(batch[constants.N_ATOMS_BATCH_REAL])
            n_structs = int(batch[constants.N_STRUCTURES_BATCH_REAL])
            features = output[hidden_target_name][:n_real].numpy()
            elem_idx = np.asarray(batch[constants.ATOMIC_MU_I])[:n_real]
            weights = np.asarray(batch[uq_constants.UQ_WEIGHT])[:n_real].astype(
                np.float64
            )
            if pbar is not None:
                if _streaming_estimate:
                    pbar.total = total_structs.total
                pbar.update(n_structs)
            yield features, elem_idx, weights
    finally:
        if pbar is not None:
            pbar.close()


def tf_dataset_feature_iterator(
    shard_paths: list,
    model,
    hidden_target_name: str = uq_constants.FEATURES,
    frac: float = None,
    seed: int = 42,
    verbose: bool = True,
    desc: str = "TF dataset features",
    total_num_batches: int = None,
) -> Iterator[tuple[np.ndarray, np.ndarray, np.ndarray]]:
    """Yield (features, element_indices, weights) from pre-computed TF dataset shards.

    Reads pre-padded batch dicts directly from stage3 shards, skipping
    geometry building entirely. Weights are always 1.0 — the TF-dataset
    path does not carry per-source weights; ``grace_uq build`` errors out if
    ``--train-data-weighted`` is combined with sharded TF datasets.

    Parameters
    ----------
    shard_paths : list of str
        Shard directories assigned to this worker (already sliced).
    model : TPModel
        Model with hidden feature extraction enabled.
    hidden_target_name : str
        Key for hidden features in model output.
    frac : float, optional
        If set, randomly keep this fraction of batches (seeded).
    seed : int
        RNG seed for ``frac`` sampling.
    verbose : bool
        Show progress bar.
    total_num_batches : int, optional
        Total number of batches across the assigned shards (for progress bar).
        If None, progress bar shows no total.
    """
    if not shard_paths:
        return

    model_keys = {
        k: COMPUTE_SPEC_DTYPE_MAP[v["dtype"]] for k, v in model.compute_specs.items()
    }

    rng = np.random.RandomState(seed) if frac is not None else None

    pbar = (
        ProgressPrinter(total=total_num_batches, desc=desc, unit="batches")
        if verbose
        else None
    )
    try:
        for shard_path in shard_paths:
            ds = tf.data.Dataset.load(shard_path, compression="GZIP").prefetch(2)
            for batch in ds:
                if rng is not None and rng.random() > frac:
                    if pbar is not None:
                        pbar.update(1)
                    continue

                batch_tf = {
                    k: tf.cast(batch[k], dtype=dt)
                    for k, dt in model_keys.items()
                    if k in batch
                }
                output = model.compute(batch_tf)
                n_real = int(batch[constants.N_ATOMS_BATCH_REAL])
                features = output[hidden_target_name][:n_real].numpy()
                elem_idx = batch[constants.ATOMIC_MU_I][:n_real].numpy()
                weights = np.ones(n_real, dtype=np.float64)
                if pbar is not None:
                    pbar.update(1)
                yield features, elem_idx, weights
    finally:
        if pbar is not None:
            pbar.close()
