import pytest
import pandas as pd
import numpy as np
from tensorpotential.data.databuilder import estimate_n_buckets

def test_estimate_n_buckets_adaptive():
    # Helper to create a dummy batches_df
    def create_df(n_neigh_list):
        return pd.DataFrame({
            "n_neighbours": n_neigh_list,
            "n_atoms": [10] * len(n_neigh_list),
            "n_structures": [1] * len(n_neigh_list)
        })

    # Case 1: All identical -> 1 bucket (0% overhead)
    df1 = create_df([100, 100, 100, 100])
    assert estimate_n_buckets(df1, 0.3) == 1

    # Case 2: Highly heterogeneous
    # [100, 10] -> bucket 1 (max 100), total_real=110. Padding = (100+100)/110 - 1 = 81% overhead.
    # threshold 0.3 should force more buckets if possible.
    df2 = create_df([100, 10])
    # n=1: overhead 0.81
    # n=2: buckets [100], [10] -> overhead 0.0
    assert estimate_n_buckets(df2, 0.3) == 2

    # Case 3: Smooth distribution
    # 10 batches [110, 109, ..., 101]
    # Sum real = 1055
    # n=1: max=110, total=1100. Overhead = 4.2% -> should pick 1 bucket
    df3 = create_df(list(range(101, 111)))
    assert estimate_n_buckets(df3, 0.1) == 1

    # Case 4: Tight threshold
    # n=1: overhead ~4.2% (above 1%)
    assert estimate_n_buckets(df3, 0.01) > 1

if __name__ == "__main__":
    test_estimate_n_buckets_adaptive()
    print("Test passed!")
