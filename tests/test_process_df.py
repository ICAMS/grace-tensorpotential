import os

import numpy as np
import pytest

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

import logging

LOG_FMT = "%(asctime)s %(levelname).1s - %(message)s"
logging.basicConfig(level=logging.INFO, format=LOG_FMT, datefmt="%Y/%m/%d %H:%M:%S")
log = logging.getLogger()

import pandas as pd
from pathlib import Path

prefix = Path(__file__).parent.resolve()

from tensorpotential.data.process_df import *


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


# def test_compute_convex_hull_HME21():
#     fname = str(prefix / "data" / "HME21_test_clean_small.pkl.gz")
#     df = pd.read_pickle(fname)
#     df["energy_per_atom"] = df["energy"] / df["NUMBER_OF_ATOMS"]
#     assert E_CHULL_DIST_PER_ATOM not in df.columns
#     with pytest.raises(RuntimeError):
#         compute_convexhull_dist(df, verbose=True)
