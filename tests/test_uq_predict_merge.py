"""`grace_uq predict` with several workers must return rows in input order."""

import argparse

import numpy as np
import pandas as pd

from tensorpotential.uq.cli._common import slice_dataset_for_worker
from tensorpotential.uq.cli.predict import _merge_shards


def _write_shards(df, tmp_path, n_workers):
    for w in range(n_workers):
        shard = slice_dataset_for_worker(df, w, n_workers)
        shard.to_pickle(tmp_path / f".predict_w{w}.pkl.gz", compression="gzip")
    return argparse.Namespace(n_workers=n_workers, output=str(tmp_path / "out.pkl.gz"))


_INDEXES = {
    "increasing": np.arange(280) * 7 + 5,
    "duplicated": np.tile(np.arange(70), 4),  # e.g. pd.concat without ignore_index
    "shuffled": np.random.default_rng(0).permutation(280),
}


def test_merge_restores_input_order(tmp_path):
    # 280 rows: enough for the unstable sort on tied shard-local indices to scramble them.
    for index_kind, index in sorted(_INDEXES.items()):
        for n_workers in (1, 2, 3, 4):
            d = tmp_path / f"{index_kind}_{n_workers}"
            d.mkdir()
            df = pd.DataFrame({"x": np.arange(280)}, index=index)
            merged = _merge_shards(_write_shards(df, d, n_workers))
            np.testing.assert_array_equal(merged["x"].to_numpy(), np.arange(280))
            assert list(merged.columns) == ["x"]
            assert merged.index.equals(pd.RangeIndex(280)), (index_kind, n_workers)
