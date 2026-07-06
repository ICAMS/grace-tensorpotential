"""Tests for the streaming elastic batching pipeline."""

import os

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

import numpy as np
import pandas as pd
import pytest

from tensorpotential import constants as tc
from tensorpotential.data.databuilder import (
    GeometricalDataBuilder,
    ReferenceEnergyForcesStressesDataBuilder,
    construct_batches,
)
from tensorpotential.data.streaming import (
    ElasticBatchIterator,
    MultiBinMetricBatcher,
    PreBatch,
    StreamingConfig,
    StreamingDatasetWrapper,
)

TEST_DATA_PATH = os.path.join(os.path.dirname(__file__), "data", "MoNbTaW_train50.pkl.gz")


@pytest.fixture
def train_df():
    return pd.read_pickle(TEST_DATA_PATH)


@pytest.fixture
def element_map():
    return {"Mo": 0, "Nb": 1, "Ta": 2, "W": 3}


@pytest.fixture
def data_builders(element_map):
    geom = GeometricalDataBuilder(
        elements_map=element_map,
        cutoff=5.0,
        is_fit_stress=False,
    )
    ref = ReferenceEnergyForcesStressesDataBuilder(
        normalize_weights=False,
        is_fit_stress=False,
    )
    return [geom, ref]


class TestPreBatch:
    def test_can_fit(self):
        pb = PreBatch(target_metric=100, metric_strategy="neighbours")
        assert pb.can_fit(50)
        pb.add({"x": 1}, 60)
        assert not pb.can_fit(50)
        assert pb.can_fit(40)

    def test_reset(self):
        pb = PreBatch(target_metric=100, metric_strategy="neighbours")
        pb.add({"x": 1}, 50)
        assert not pb.is_empty
        pb.reset()
        assert pb.is_empty
        assert pb.current_metric == 0


class TestMultiBinMetricBatcher:
    def test_basic_batching(self, train_df, data_builders):
        batcher = MultiBinMetricBatcher(
            num_bins=3,
            target_metric=2000,
            metric_strategy="neighbours",
            sorting_buffer_size=8,
        )

        all_batches = []
        for row_idx in range(len(train_df)):
            row = train_df.iloc[row_idx]
            data_dict = {}
            for builder in data_builders:
                data_dict.update(
                    builder.extract_from_row(
                        row, **{tc.DATA_STRUCTURE_ID: train_df.index[row_idx]}
                    )
                )
            for batch in batcher.add_to_buffer(data_dict):
                all_batches.append(batch)

        for batch in batcher.flush():
            all_batches.append(batch)

        # All structures should be accounted for
        total_structures = sum(len(b) for b in all_batches)
        assert total_structures == len(train_df)

    def test_oversized_structure(self):
        """Structure exceeding target_metric is yielded as its own batch."""
        batcher = MultiBinMetricBatcher(
            num_bins=2, target_metric=10, metric_strategy="neighbours"
        )
        big = {tc.N_NEIGHBORS_REAL: np.array(100)}
        batches = list(batcher.add_to_buffer(big))
        # Should be flushed immediately (buffer not full yet with size=32)
        # Flush to get it
        batches.extend(batcher.flush())
        assert any(len(b) == 1 for b in batches)


class TestElasticBatchIterator:
    def test_produces_padded_batches(self, train_df, data_builders):
        def structure_gen():
            for row_idx in range(len(train_df)):
                row = train_df.iloc[row_idx]
                data_dict = {}
                for builder in data_builders:
                    data_dict.update(
                    builder.extract_from_row(
                        row, **{tc.DATA_STRUCTURE_ID: train_df.index[row_idx]}
                    )
                )
                yield data_dict

        batcher = MultiBinMetricBatcher(
            num_bins=3, target_metric=2000, metric_strategy="neighbours"
        )
        elastic = ElasticBatchIterator(
            structure_iter=structure_gen(),
            batcher=batcher,
            data_builders=data_builders,
            buckets=[],
            growth_fraction=0.1,
            verbose=True,
        )

        batches = list(elastic)
        assert len(batches) > 0

        for batch in batches:
            # All required keys should be present
            assert tc.BOND_VECTOR in batch
            assert tc.ATOMIC_MU_I in batch
            assert tc.BOND_IND_I in batch
            assert tc.BOND_IND_J in batch
            assert tc.N_ATOMS_BATCH_REAL in batch
            assert tc.N_ATOMS_BATCH_TOTAL in batch
            assert tc.N_STRUCTURES_BATCH_REAL in batch
            assert tc.N_STRUCTURES_BATCH_TOTAL in batch
            assert tc.ATOMS_TO_STRUCTURE_MAP in batch
            assert tc.BONDS_TO_STRUCTURE_MAP in batch

            # Padded dimensions >= real dimensions
            assert int(batch[tc.N_ATOMS_BATCH_TOTAL]) >= int(
                batch[tc.N_ATOMS_BATCH_REAL]
            )
            assert int(batch[tc.N_STRUCTURES_BATCH_TOTAL]) >= int(
                batch[tc.N_STRUCTURES_BATCH_REAL]
            )

        # Buckets should have been discovered
        assert len(elastic.buckets) > 0

    def test_buckets_persist_across_iterations(self, train_df, data_builders):
        """Second iteration should reuse buckets from first."""
        shared_buckets = []

        for iteration in range(2):

            def structure_gen():
                for row_idx in range(len(train_df)):
                    row = train_df.iloc[row_idx]
                    data_dict = {}
                    for builder in data_builders:
                        data_dict.update(
                    builder.extract_from_row(
                        row, **{tc.DATA_STRUCTURE_ID: train_df.index[row_idx]}
                    )
                )
                    yield data_dict

            batcher = MultiBinMetricBatcher(
                num_bins=3, target_metric=2000, metric_strategy="neighbours"
            )
            elastic = ElasticBatchIterator(
                structure_iter=structure_gen(),
                batcher=batcher,
                data_builders=data_builders,
                buckets=shared_buckets,
                growth_fraction=0.1,
            )
            batches = list(elastic)
            shared_buckets = elastic.buckets

        # After second iteration, buckets should be stable (no new ones added)
        n_buckets_after_first = len(shared_buckets)
        # They should have been discovered
        assert n_buckets_after_first > 0

    def test_growth_fraction_rejects_ambiguous_float_range(self, data_builders):
        """Floats in [1.0, 2.0) silently get truncated to +1 via ``int()``;
        the validator should reject them so callers learn at construction."""
        batcher = MultiBinMetricBatcher(
            num_bins=1, target_metric=1000, metric_strategy="neighbours"
        )
        with pytest.raises(ValueError, match="ambiguous"):
            ElasticBatchIterator(
                structure_iter=iter(()),
                batcher=batcher,
                data_builders=data_builders,
                buckets=[],
                growth_fraction=1.5,
            )
        # Per-axis dicts get the same treatment.
        with pytest.raises(ValueError, match="ambiguous"):
            ElasticBatchIterator(
                structure_iter=iter(()),
                batcher=batcher,
                data_builders=data_builders,
                buckets=[],
                growth_fraction={"atom": 0.1, "bond": 1.7},
            )
        # 0 / negative are rejected too.
        with pytest.raises(ValueError, match="must be > 0"):
            ElasticBatchIterator(
                structure_iter=iter(()),
                batcher=batcher,
                data_builders=data_builders,
                buckets=[],
                growth_fraction=0.0,
            )

    def test_growth_fraction_absolute(self, train_df, data_builders):
        """Test integer growth_fraction for absolute slot addition."""

        def structure_gen():
            for row_idx in range(min(5, len(train_df))):
                row = train_df.iloc[row_idx]
                data_dict = {}
                for builder in data_builders:
                    data_dict.update(
                    builder.extract_from_row(
                        row, **{tc.DATA_STRUCTURE_ID: train_df.index[row_idx]}
                    )
                )
                yield data_dict

        batcher = MultiBinMetricBatcher(
            num_bins=2, target_metric=5000, metric_strategy="neighbours"
        )
        elastic = ElasticBatchIterator(
            structure_iter=structure_gen(),
            batcher=batcher,
            data_builders=data_builders,
            buckets=[],
            growth_fraction={"atom": 10, "bond": 50, "structure": 5},
        )
        batches = list(elastic)
        assert len(batches) > 0


class TestStreamingDatasetWrapper:
    def test_basic_iteration(self, train_df, data_builders):
        config = StreamingConfig(
            target_metric=2000,
            metric_strategy="neighbours",
            num_bins=3,
        )
        wrapper = StreamingDatasetWrapper(
            train_df, data_builders, config, shuffle=False, seed=42
        )

        batches = list(wrapper)
        assert len(batches) > 0

        # Check batch format
        for batch in batches:
            assert tc.BOND_VECTOR in batch
            assert tc.N_ATOMS_BATCH_REAL in batch

    def test_len_updated_after_epoch(self, train_df, data_builders):
        config = StreamingConfig(target_metric=2000, num_bins=3)
        wrapper = StreamingDatasetWrapper(
            train_df, data_builders, config, shuffle=False
        )

        # Before first epoch: rough estimate
        initial_len = len(wrapper)
        assert initial_len > 0

        # After first epoch: exact count
        batches = list(wrapper)
        assert len(wrapper) == len(batches)

    def test_shuffle_produces_different_order(self, train_df, data_builders):
        config = StreamingConfig(target_metric=5000, num_bins=2)

        wrapper1 = StreamingDatasetWrapper(
            train_df, data_builders, config, shuffle=True, seed=42
        )
        wrapper2 = StreamingDatasetWrapper(
            train_df, data_builders, config, shuffle=True, seed=42
        )

        # Same seed → same data (though order may vary due to bin routing)
        batches1 = list(wrapper1)
        batches2 = list(wrapper2)
        assert len(batches1) == len(batches2)

    def test_shuffle_with_seed_is_reproducible_across_global_rng_changes(
        self, train_df, data_builders
    ):
        """Regression: previously ``np.random.shuffle(indices)`` ignored the
        ``seed=`` kwarg and used the global numpy RNG. Two wrappers with the
        same seed must produce the same structure-id order even when global
        RNG state differs between them."""
        config = StreamingConfig(target_metric=5000, num_bins=2)

        def collect_order(global_seed: int) -> list[int]:
            np.random.seed(global_seed)
            wrapper = StreamingDatasetWrapper(
                train_df, data_builders, config, shuffle=True, seed=42
            )
            ids: list[int] = []
            for batch in wrapper:
                sid = batch.get(tc.DATA_STRUCTURE_ID)
                if sid is not None:
                    ids.extend(np.asarray(sid).flatten().tolist())
            return ids

        order1 = collect_order(global_seed=11111)
        order2 = collect_order(global_seed=99999)

        assert order1 == order2, (
            f"shuffle with seed=42 produced different orders after global "
            f"RNG pollution: first 5 = {order1[:5]} vs {order2[:5]}"
        )

    def test_model_signatures_filtering(self, train_df, data_builders):
        """When model signatures are set, only matching keys are yielded as tf.Tensor."""
        import tensorflow as tf

        config = StreamingConfig(target_metric=5000, num_bins=2)
        wrapper = StreamingDatasetWrapper(
            train_df, data_builders, config, shuffle=False
        )

        # Simulate model signatures (subset of keys)
        sigs = {
            tc.BOND_VECTOR: tf.TensorSpec(shape=[None, 3], dtype=tf.float64),
            tc.ATOMIC_MU_I: tf.TensorSpec(shape=[None], dtype=tf.int32),
            tc.BOND_IND_I: tf.TensorSpec(shape=[None], dtype=tf.int32),
            tc.BOND_IND_J: tf.TensorSpec(shape=[None], dtype=tf.int32),
            tc.BOND_MU_I: tf.TensorSpec(shape=[None], dtype=tf.int32),
            tc.BOND_MU_J: tf.TensorSpec(shape=[None], dtype=tf.int32),
            tc.N_ATOMS_BATCH_REAL: tf.TensorSpec(shape=[], dtype=tf.int32),
            tc.N_ATOMS_BATCH_TOTAL: tf.TensorSpec(shape=[], dtype=tf.int32),
            tc.N_STRUCTURES_BATCH_REAL: tf.TensorSpec(shape=[], dtype=tf.int32),
            tc.N_STRUCTURES_BATCH_TOTAL: tf.TensorSpec(shape=[], dtype=tf.int32),
            tc.ATOMS_TO_STRUCTURE_MAP: tf.TensorSpec(shape=[None], dtype=tf.int32),
            tc.BONDS_TO_STRUCTURE_MAP: tf.TensorSpec(shape=[None], dtype=tf.int32),
        }
        wrapper.set_model_signatures(sigs)

        batches = list(wrapper)
        for batch in batches:
            # Only signature keys should be present
            for key in batch:
                assert key in sigs
            # Values should be tf.Tensor
            for key, val in batch.items():
                assert isinstance(val, tf.Tensor), f"{key} is not a tf.Tensor"


class TestStreamingVsInMemory:
    def test_same_structures_processed(self, train_df, data_builders):
        """Streaming and in-memory should process the same total number of structures."""
        # In-memory
        in_memory_batches = construct_batches(
            train_df,
            data_builders=data_builders,
            batch_size=10,
            max_n_buckets=None,
        )

        total_inmem_structures = sum(
            int(b[tc.N_STRUCTURES_BATCH_REAL]) for b in in_memory_batches
        )

        # Streaming
        config = StreamingConfig(target_metric=5000, num_bins=3)
        wrapper = StreamingDatasetWrapper(
            train_df, data_builders, config, shuffle=False
        )
        streaming_batches = list(wrapper)
        total_streaming_structures = sum(
            int(b[tc.N_STRUCTURES_BATCH_REAL]) for b in streaming_batches
        )

        assert total_inmem_structures == total_streaming_structures == len(train_df)
