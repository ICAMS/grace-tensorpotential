"""
On-the-fly streaming data pipeline with elastic metric batching.

Adapted from jax_grace elastic batching (``/fscratch/AW/jax_grace/src/jax_grace/data/elastic.py``).
No pre-scan needed: structures are extracted lazily, batched via multi-bin metric routing,
and padded with dynamically growing buckets to minimize XLA recompilations.

Usage in ``input.yaml``::

    data:
      pipeline: streaming
      streaming:
        target_metric: 3000
        metric_strategy: neighbours
        num_bins: 10
        growth_fraction: 0.1
"""

from __future__ import annotations

import dataclasses
import logging
import queue
import threading
from dataclasses import dataclass
from typing import Any, Iterator, List, Optional

import numpy as np
import tensorflow as tf

from tensorpotential import constants
from tensorpotential.data.databuilder import (
    AbstractDataBuilder,
)

log = logging.getLogger(__name__)


def _normalize_metric_strategy(value: str) -> str:
    """Accept ``"neighbors"`` (US) as an alias for the canonical UK
    ``"neighbours"`` so users aren't tripped up by the spelling choice."""
    if value == "neighbors":
        return "neighbours"
    return value


def _validate_growth_fraction(value):
    """Validate ``growth_fraction`` semantics at construction time.

    ``growth_fraction`` overloads two meanings: a fractional multiplier when
    ``0 < frac < 1`` (e.g. 0.1 → grow by +10 %), and an absolute additive
    step when ``frac >= 1`` (e.g. ``2`` → grow by exactly 2 along the axis).
    Float values in ``[1, 2)`` were silently truncated to ``+1`` via
    ``int(frac)``, which almost always indicates user confusion (was 1.5
    meant to be 150 % growth or a +1 step?). Reject this ambiguous range
    explicitly so callers learn at construction rather than after several
    rounds of confusing growth steps.

    Accepts a single scalar or a per-axis dict; each axis value is checked.
    """

    def _check(v, label: str):
        if not isinstance(v, (int, float)):
            raise TypeError(
                f"{label} must be int or float, got {type(v).__name__}: {v!r}"
            )
        if v <= 0:
            raise ValueError(f"{label} must be > 0, got {v!r}")
        if 1 <= v < 2 and not isinstance(v, int):
            raise ValueError(
                f"{label}={v!r} is ambiguous: floats in [1.0, 2.0) get "
                "truncated to +1 step. Use 0 < frac < 1 for a fractional "
                "multiplier (e.g. 0.5 = +50 %), or an int >= 1 for an "
                "absolute additive step (e.g. 2 = +2)."
            )

    if isinstance(value, dict):
        for axis, v in value.items():
            _check(v, f"growth_fraction[{axis!r}]")
    else:
        _check(value, "growth_fraction")
    return value


# ---------------------------------------------------------------------------
# PreBatch: accumulation container for one bin
# ---------------------------------------------------------------------------


@dataclasses.dataclass
class PreBatch:
    """Accumulates structures until a metric target is reached."""

    target_metric: int
    metric_strategy: str  # "atoms", "neighbours", "structures"

    current_metric: int = 0
    structures: list[Any] = dataclasses.field(default_factory=list)

    def can_fit(self, structure_metric: int) -> bool:
        return self.current_metric + structure_metric <= self.target_metric

    def add(self, structure: Any, metric: int):
        self.structures.append(structure)
        self.current_metric += metric

    @property
    def is_empty(self) -> bool:
        return len(self.structures) == 0

    def reset(self):
        self.current_metric = 0
        self.structures = []


# ---------------------------------------------------------------------------
# MultiBinMetricBatcher: routes structures to bins for balanced batching
# ---------------------------------------------------------------------------


class MultiBinMetricBatcher:
    """Routes structures into multiple bins based on a target metric.

    Accumulates structures in a sorting buffer, sorts them periodically,
    and interleaves small/large structures for load balancing across bins.
    """

    def __init__(
        self,
        num_bins: int = 10,
        target_metric: int = 3000,
        metric_strategy: str = "neighbours",
        sorting_buffer_size: int = 32,
    ):
        metric_strategy = _normalize_metric_strategy(metric_strategy)
        self.bins = [PreBatch(target_metric, metric_strategy) for _ in range(num_bins)]
        self.sorting_buffer: list = []
        self.sorting_buffer_size = sorting_buffer_size
        self.metric_strategy = metric_strategy
        self.target_metric = target_metric

    def _get_metric(self, structure: Any) -> int:
        if self.metric_strategy == "structures":
            return 1
        elif self.metric_strategy == "atoms":
            return int(structure[constants.N_ATOMS_BATCH_REAL])
        elif self.metric_strategy == "neighbours":
            return int(structure[constants.N_NEIGHBORS_REAL])
        else:
            raise ValueError(f"Unknown metric strategy: {self.metric_strategy}")

    def add_to_buffer(self, structure: Any) -> Iterator[list[Any]]:
        """Add structure to buffer; yield complete batches when buffer is full."""
        self.sorting_buffer.append(structure)

        if len(self.sorting_buffer) >= self.sorting_buffer_size:
            self.sorting_buffer.sort(key=self._get_metric)

            while self.sorting_buffer:
                if self.sorting_buffer:
                    yield from self._process_structure(self.sorting_buffer.pop(0))
                if self.sorting_buffer:
                    yield from self._process_structure(self.sorting_buffer.pop(-1))

    def flush(self) -> Iterator[list[Any]]:
        """Flush remaining buffer and bins at end of epoch."""
        self.sorting_buffer.sort(key=self._get_metric)
        while self.sorting_buffer:
            yield from self._process_structure(self.sorting_buffer.pop(0))
            if self.sorting_buffer:
                yield from self._process_structure(self.sorting_buffer.pop(-1))

        for b in self.bins:
            if not b.is_empty:
                yield b.structures
                b.reset()

    def _process_structure(self, structure: Any) -> Iterator[list[Any]]:
        metric = self._get_metric(structure)

        # Single structure exceeds target → yield as its own batch
        if metric > self.target_metric:
            yield [structure]
            return

        valid_bin_indices = []
        for i, b in enumerate(self.bins):
            if b.can_fit(metric):
                valid_bin_indices.append(i)

        if not valid_bin_indices:
            # No bin fits → flush fullest bin
            fullest_bin = max(self.bins, key=lambda b: b.current_metric)
            yield fullest_bin.structures
            fullest_bin.reset()
            fullest_bin.add(structure, metric)
            return

        # Route to bin with highest utilization (best fit)
        best_idx = max(valid_bin_indices, key=lambda i: self.bins[i].current_metric)
        self.bins[best_idx].add(structure, metric)


# ---------------------------------------------------------------------------
# ElasticBatchIterator: joins/pads batches with dynamic bucket growth
# ---------------------------------------------------------------------------


class ElasticBatchIterator:
    """Iterator that assembles padded batches with elastic bucket growth.

    Pulls per-structure dicts from ``structure_iter``, groups them via the
    ``batcher``, then joins and pads each group using tensorpotential's
    ``AbstractDataBuilder`` interface.  Bucket shapes grow dynamically when
    a batch exceeds all known buckets.
    """

    def __init__(
        self,
        structure_iter: Iterator,
        batcher: MultiBinMetricBatcher,
        data_builders: list[AbstractDataBuilder],
        buckets: list[dict],
        outlier_strategy: str = "expand",
        growth_fraction: float | int | dict = 0.1,
        verbose: bool = False,
        metric_strategy: str = "neighbours",
        target_metric: int = 3000,
        is_first_epoch: bool = False,
        name: str = "train",
    ):
        self._parent_iter = structure_iter
        self._batcher = batcher
        self._builders = data_builders
        self._buckets = buckets
        self._outlier_strategy = outlier_strategy
        self._growth_fraction = _validate_growth_fraction(growth_fraction)
        self._verbose = verbose
        self._metric_strategy = _normalize_metric_strategy(metric_strategy)
        self._target_metric = target_metric
        self._is_first_epoch = is_first_epoch
        self._name = name
        self._batch_buffer: list[dict] = []
        # Padding statistics accumulators
        self._padding_stats = {
            "nreal_struc": 0,
            "nreal_atoms": 0,
            "nreal_neigh": 0,
            "pad_nstruct": 0,
            "pad_nat": 0,
            "pad_nneigh": 0,
        }

    def __iter__(self):
        return self

    def __next__(self):
        if self._batch_buffer:
            return self._batch_buffer.pop(0)

        while True:
            try:
                structure = next(self._parent_iter)
            except StopIteration:
                for batch_structures in self._batcher.flush():
                    batch = self._pad_batch(batch_structures)
                    if batch:
                        self._batch_buffer.append(batch)

                if self._batch_buffer:
                    return self._batch_buffer.pop(0)
                else:
                    raise StopIteration

            found_batches = False
            for batch_structures in self._batcher.add_to_buffer(structure):
                batch = self._pad_batch(batch_structures)
                if batch:
                    self._batch_buffer.append(batch)
                    found_batches = True

            if found_batches:
                return self._batch_buffer.pop(0)

    @property
    def buckets(self) -> list[dict]:
        return list(self._buckets)

    @property
    def padding_stats(self) -> dict:
        return dict(self._padding_stats)

    def reset_padding_stats(self):
        for k in self._padding_stats:
            self._padding_stats[k] = 0

    def _pad_batch(self, batch_structures: list[Any]) -> dict[str, Any] | None:
        """Join and pad a list of per-structure dicts into a single batch."""
        if not batch_structures:
            return None

        real_atoms = sum(int(s[constants.N_ATOMS_BATCH_REAL]) for s in batch_structures)
        real_bonds = sum(int(s[constants.N_NEIGHBORS_REAL]) for s in batch_structures)
        real_structs = len(batch_structures)

        # Find smallest existing bucket that fits
        best_bucket = None
        for bucket in self._buckets:
            if (
                bucket[constants.PAD_MAX_N_ATOMS] >= real_atoms
                and bucket[constants.PAD_MAX_N_NEIGHBORS] >= real_bonds
                and bucket[constants.PAD_MAX_N_STRUCTURES] >= real_structs
            ):
                best_bucket = bucket
                break

        # Dynamic growth when no bucket fits
        if best_bucket is None:
            if self._outlier_strategy not in ("expand", "warn_skip", "error"):
                raise ValueError(f"Unknown outlier strategy: {self._outlier_strategy}")

            if self._outlier_strategy != "expand":
                max_bucket = (
                    self._buckets[-1]
                    if self._buckets
                    else {
                        constants.PAD_MAX_N_ATOMS: 0,
                        constants.PAD_MAX_N_NEIGHBORS: 0,
                        constants.PAD_MAX_N_STRUCTURES: 0,
                    }
                )

                def exceeds(s):
                    return (
                        int(s[constants.N_ATOMS_BATCH_REAL])
                        > max_bucket[constants.PAD_MAX_N_ATOMS]
                        or int(s[constants.N_NEIGHBORS_REAL])
                        > max_bucket[constants.PAD_MAX_N_NEIGHBORS]
                    )

                outliers = [i for i, s in enumerate(batch_structures) if exceeds(s)]

                if outliers:
                    if self._outlier_strategy == "error":
                        s = batch_structures[outliers[0]]
                        raise ValueError(
                            f"Structure (atoms={s[constants.N_ATOMS_BATCH_REAL]}, "
                            f"bonds={s[constants.N_NEIGHBORS_REAL]}) exceeds largest bucket "
                            f"with outlier_strategy='error'."
                        )
                    elif self._outlier_strategy == "warn_skip":
                        log.warning(
                            f"[ElasticBatcher({self._name})] Skipping {len(outliers)} structures "
                            f"that exceed largest bucket."
                        )
                        batch_structures = [
                            s
                            for i, s in enumerate(batch_structures)
                            if i not in outliers
                        ]
                        if not batch_structures:
                            return None
                        real_atoms = sum(
                            int(s[constants.N_ATOMS_BATCH_REAL])
                            for s in batch_structures
                        )
                        real_bonds = sum(
                            int(s[constants.N_NEIGHBORS_REAL]) for s in batch_structures
                        )
                        real_structs = len(batch_structures)

                        for bucket in self._buckets:
                            if (
                                bucket[constants.PAD_MAX_N_ATOMS] >= real_atoms
                                and bucket[constants.PAD_MAX_N_NEIGHBORS] >= real_bonds
                                and bucket[constants.PAD_MAX_N_STRUCTURES]
                                >= real_structs
                            ):
                                best_bucket = bucket
                                break

            if best_bucket is None:
                ref_bucket = (
                    self._buckets[-1]
                    if self._buckets
                    else {
                        constants.PAD_MAX_N_ATOMS: 0,
                        constants.PAD_MAX_N_NEIGHBORS: 0,
                        constants.PAD_MAX_N_STRUCTURES: 0,
                    }
                )

                def grow(req, curr, axis):
                    if axis == self._metric_strategy:
                        # Hard-cap at target_metric for the metric strategy axis
                        return max(req, self._target_metric)
                    if req > curr:
                        frac = (
                            self._growth_fraction.get(axis, 0.1)
                            if isinstance(self._growth_fraction, dict)
                            else self._growth_fraction
                        )
                        if frac >= 1:
                            return max(req, curr + int(frac))
                        return max(req, int(curr * (1 + frac)))
                    return curr

                # Pad at least +1 atom and +1 structure for the padding slot
                new_atoms = grow(
                    real_atoms + 1,
                    ref_bucket.get(constants.PAD_MAX_N_ATOMS, 0),
                    "atoms",
                )
                new_bonds = grow(
                    real_bonds,
                    ref_bucket.get(constants.PAD_MAX_N_NEIGHBORS, 0),
                    "neighbours",
                )
                new_structs = grow(
                    real_structs + 1,
                    ref_bucket.get(constants.PAD_MAX_N_STRUCTURES, 0),
                    "structures",
                )

                best_bucket = {
                    constants.PAD_MAX_N_ATOMS: new_atoms,
                    constants.PAD_MAX_N_NEIGHBORS: new_bonds,
                    constants.PAD_MAX_N_STRUCTURES: new_structs,
                }
                self._buckets.append(best_bucket)
                self._buckets.sort(
                    key=lambda b: (
                        b[constants.PAD_MAX_N_ATOMS] + b[constants.PAD_MAX_N_NEIGHBORS]
                    )
                )
                # New bucket log is ALWAYS printed (not verbose-gated)
                n = len(batch_structures)
                ref_a = ref_bucket.get(constants.PAD_MAX_N_ATOMS, 0)
                ref_b = ref_bucket.get(constants.PAD_MAX_N_NEIGHBORS, 0)
                ref_s = ref_bucket.get(constants.PAD_MAX_N_STRUCTURES, 0)
                axes = []
                if real_atoms + 1 > ref_a:
                    axes.append(f"atoms {ref_a}->{new_atoms}")
                if real_bonds > ref_b:
                    axes.append(f"neighbours {ref_b}->{new_bonds}")
                if real_structs + 1 > ref_s:
                    axes.append(f"structures {ref_s}->{new_structs}")
                reason = ", ".join(axes) if axes else "first bucket"
                log.info(
                    f"[ElasticBatcher({self._name})] New bucket (total: {len(self._buckets)}): "
                    f"atom={new_atoms}, bond={new_bonds}, structure={new_structs} "
                    f"| {n} structure(s) in batch | grew: {reason}"
                )
                # Intermediate padding stats during first epoch, gated on verbose
                if self._verbose and self._is_first_epoch:
                    ps = self._padding_stats
                    if ps["nreal_atoms"] > 0:
                        log.info(
                            f"[ElasticBatcher({self._name})] Intermediate padding stats: "
                            f"num. real structures: {ps['nreal_struc']} "
                            f"(+{ps['pad_nstruct'] / ps['nreal_struc'] * 1e2:.1f}%) | "
                            f"num. real atoms: {ps['nreal_atoms']} "
                            f"(+{ps['pad_nat'] / ps['nreal_atoms'] * 1e2:.1f}%) | "
                            f"num. real neighbours: {ps['nreal_neigh']} "
                            f"(+{ps['pad_nneigh'] / max(ps['nreal_neigh'], 1) * 1e2:.1f}%)"
                        )

        # Join and pad via each builder
        batch = {}
        for builder in self._builders:
            batch.update(builder.join_to_batch(batch_structures))

        for builder in self._builders:
            builder.pad_batch(batch, best_bucket)

        # Accumulate padding statistics
        pad_atoms = best_bucket[constants.PAD_MAX_N_ATOMS] - real_atoms
        pad_bonds = best_bucket[constants.PAD_MAX_N_NEIGHBORS] - real_bonds
        pad_structs = best_bucket[constants.PAD_MAX_N_STRUCTURES] - real_structs
        self._padding_stats["nreal_struc"] += real_structs
        self._padding_stats["nreal_atoms"] += real_atoms
        self._padding_stats["nreal_neigh"] += real_bonds
        self._padding_stats["pad_nstruct"] += pad_structs
        self._padding_stats["pad_nat"] += pad_atoms
        self._padding_stats["pad_nneigh"] += pad_bonds

        return batch


# ---------------------------------------------------------------------------
# StreamingConfig
# ---------------------------------------------------------------------------


@dataclass
class StreamingConfig:
    """Configuration for the streaming elastic batching pipeline."""

    target_metric: int = 3000
    """Target metric value per batch (e.g. total neighbours)."""

    metric_strategy: str = "neighbours"
    """Metric to use: ``"atoms"``, ``"neighbours"``, or ``"structures"``.
    ``"neighbors"`` (US spelling) is accepted as an alias for ``"neighbours"``."""

    num_bins: int = 10
    """Number of parallel accumulation bins."""

    sorting_buffer_size: int = 32
    """Sort structures every N arrivals for load balancing."""

    growth_fraction: float | int | dict = 0.1
    """Bucket growth when overflow occurs.
    ``< 1``: relative (e.g. 0.1 = 10% growth).
    ``>= 1``: absolute (e.g. 5 = +5 slots).
    Can be per-axis dict: ``{"atoms": 0.1, "neighbours": 0.05, "structures": 5}``.
    """

    outlier_strategy: str = "expand"
    """How to handle batches exceeding all buckets: ``"expand"``, ``"warn_skip"``, ``"error"``."""

    initial_buckets: Optional[list[dict]] = None
    """Pre-seed buckets from a previous run to avoid first-epoch recompilations."""

    verbose: bool = False
    """Log bucket discovery events and padding statistics."""

    prefetch_queue_size: int = 0
    """Number of pre-extracted structures to buffer in the background thread.
    Set to 0 to disable prefetching (serial extraction)."""


# ---------------------------------------------------------------------------
# StreamingDatasetWrapper: training-loop-compatible iterable
# ---------------------------------------------------------------------------


class StreamingDatasetWrapper:
    """Wraps the elastic streaming pipeline for use by the training loop.

    Produces padded batch dicts on-the-fly (extract -> batch -> pad).
    Compatible with the training loop via ``len()`` and ``iter()``.
    Optionally converts numpy arrays to ``tf.Tensor`` when model signatures
    are set via :meth:`set_model_signatures`.
    """

    def __init__(
        self,
        df,
        data_builders: List[AbstractDataBuilder],
        config: StreamingConfig,
        shuffle: bool = True,
        seed: int = 42,
        name: str = "train",
    ):
        self.df = df
        self.data_builders = data_builders
        self.config = config
        
        try:
            self._n_structures = len(df)
        except TypeError:
            self._n_structures = None
            
        self._buckets: list[dict] = list(config.initial_buckets or [])
        self._shuffle = shuffle
        self._seed = seed
        self._name = name
        self._model_signatures = None

        # Pre-compute total atom count for first-epoch batch estimation.
        # Falls back gracefully if the df doesn't have ase_atoms column.
        try:
            if hasattr(df, "columns") and "ase_atoms" in df.columns:
                self._total_atoms: int | None = int(df["ase_atoms"].apply(len).sum())
            else:
                self._total_atoms = None
        except Exception:
            self._total_atoms = None

        # Estimated batch count; updated after first full epoch
        self._estimated_n_batches: int | None = None
        # Rolling stats for dynamic first-epoch estimate (live updated during iter)
        self._atoms_seen: int = 0
        self._batches_seen: int = 0

    def set_model_signatures(self, sigs: dict):
        """Set model gradient signatures for key filtering and dtype conversion."""
        self._model_signatures = sigs

    def __len__(self) -> int:
        if self._estimated_n_batches is not None:
            # Exact count from previous epoch
            return self._estimated_n_batches
        # Rolling estimate during first epoch based on avg atoms per batch seen so far
        if self._total_atoms is not None and self._batches_seen > 0:
            avg_atoms_per_batch = self._atoms_seen / self._batches_seen
            return max(1, round(self._total_atoms / avg_atoms_per_batch))
        # Rough static estimate before any batches have been produced
        if self._n_structures is not None:
            return max(1, self._n_structures // 10)
        raise TypeError("Object of type generator has no len()")

    def __iter__(self):
        if hasattr(self.df, "iloc") and self._n_structures is not None:
            indices = np.arange(self._n_structures)
            if self._shuffle:
                # Use a seeded local RNG so iteration order is reproducible
                # regardless of global numpy state.
                np.random.default_rng(self._seed).shuffle(indices)

            def _extract_one(idx):
                row = self.df.iloc[idx]
                row_ind = self.df.index[idx]
                data_dict = {}
                for builder in self.data_builders:
                    data_dict.update(
                        builder.extract_from_row(
                            row, **{constants.DATA_STRUCTURE_ID: row_ind}
                        )
                    )
                return data_dict

            def structure_generator():
                for idx in indices:
                    yield _extract_one(idx)
        else:
            def structure_generator():
                for i, row in enumerate(self.df):
                    if not isinstance(row, dict) and not hasattr(row, "get"):
                        row = {"ase_atoms": row}
                    data_dict = {}
                    for builder in self.data_builders:
                        data_dict.update(
                            builder.extract_from_row(
                                row, **{constants.DATA_STRUCTURE_ID: i}
                            )
                        )
                    yield data_dict

        if self.config.prefetch_queue_size > 0:
            # Background thread fills a queue; main thread consumes from it.
            q: queue.Queue = queue.Queue(maxsize=self.config.prefetch_queue_size)
            _SENTINEL = object()

            def _producer():
                try:
                    for item in structure_generator():
                        q.put(item)
                except Exception as exc:
                    q.put(exc)
                finally:
                    q.put(_SENTINEL)

            t = threading.Thread(target=_producer, daemon=True)
            t.start()

            def threaded_structure_generator():
                while True:
                    item = q.get()
                    if item is _SENTINEL:
                        break
                    if isinstance(item, Exception):
                        raise item
                    yield item
                t.join()
                
            iterator_to_use = threaded_structure_generator()
        else:
            iterator_to_use = structure_generator()

        batcher = MultiBinMetricBatcher(
            num_bins=self.config.num_bins,
            target_metric=self.config.target_metric,
            metric_strategy=self.config.metric_strategy,
            sorting_buffer_size=self.config.sorting_buffer_size,
        )
        elastic_iter = ElasticBatchIterator(
            structure_iter=iterator_to_use,
            batcher=batcher,
            data_builders=self.data_builders,
            buckets=self._buckets,
            outlier_strategy=self.config.outlier_strategy,
            growth_fraction=self.config.growth_fraction,
            verbose=self.config.verbose,
            metric_strategy=self.config.metric_strategy,
            target_metric=self.config.target_metric,
            is_first_epoch=self._estimated_n_batches is None,
            name=self._name,
        )

        batch_count = 0
        is_first_epoch = self._estimated_n_batches is None
        if is_first_epoch:
            # Reset rolling stats at start of first epoch
            self._atoms_seen = 0
            self._batches_seen = 0
        for batch in elastic_iter:
            batch_count += 1
            if is_first_epoch:
                # Update rolling estimate with real atoms in this batch
                nat = int(batch.get(constants.N_ATOMS_BATCH_REAL, 0))
                self._atoms_seen += nat
                self._batches_seen += 1
            if self._model_signatures is not None:
                batch = self._convert_batch(batch)
            yield batch

        # Update estimate for next epoch
        self._estimated_n_batches = batch_count
        # Persist discovered buckets for next epoch
        self._buckets = elastic_iter.buckets

        # Log padding statistics only after first epoch (suppress on subsequent epochs)
        if is_first_epoch:
            ps = elastic_iter.padding_stats
            if ps["nreal_atoms"] > 0:
                log.info(
                    f"[Streaming({self._name})] Epoch padding stats: "
                    f"num. batches: {batch_count} | "
                    f"num. real structures: {ps['nreal_struc']} "
                    f"(+{ps['pad_nstruct'] / ps['nreal_struc'] * 1e2:.1f}%) | "
                    f"num. real atoms: {ps['nreal_atoms']} "
                    f"(+{ps['pad_nat'] / ps['nreal_atoms'] * 1e2:.1f}%) | "
                    f"num. real neighbours: {ps['nreal_neigh']} "
                    f"(+{ps['pad_nneigh'] / max(ps['nreal_neigh'], 1) * 1e2:.1f}%)"
                )

    def _convert_batch(self, batch: dict) -> dict:
        """Filter keys to model signatures and convert to tf.Tensor."""
        converted = {}
        with tf.device("CPU"):
            for k, spec in self._model_signatures.items():
                if k in batch:
                    converted[k] = tf.convert_to_tensor(batch[k], dtype=spec.dtype)
        return converted
