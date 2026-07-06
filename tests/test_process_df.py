import logging
import os
from pathlib import Path

import numpy as np
import pandas as pd

from tensorpotential.data.process_df import (
    E_CHULL_DIST_PER_ATOM,
    compute_convexhull_dist,
)

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

log = logging.getLogger()

prefix = Path(__file__).parent.resolve()


def test_compute_convex_hull():
    fname = str(prefix / "data" / "MoNbTaW_test50.pkl.gz")
    df = pd.read_pickle(fname)
    assert E_CHULL_DIST_PER_ATOM not in df.columns
    compute_convexhull_dist(df, verbose=True)
    assert E_CHULL_DIST_PER_ATOM in df.columns
    echmin = df[E_CHULL_DIST_PER_ATOM].min()
    echmax = df[E_CHULL_DIST_PER_ATOM].max()
    print(f"ECHMIN: {echmin:.15f}, ECHMAX: {echmax:.15f}")

    assert np.allclose(echmin, 0.0)
    assert np.allclose(echmax, 0.435098487307693)
