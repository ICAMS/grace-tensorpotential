"""df2extxyz: default output name from the dataset suffix."""

import numpy as np
import pandas as pd
import pytest
from ase.build import bulk
from ase.io import read

from tensorpotential.scripts.df2extxyz import default_output_filename, main


@pytest.mark.parametrize(
    "name, expected",
    [
        ("train.pkl.gz", "train.xyz"),
        ("train.pckl.gz", "train.xyz"),  # was train..xyz
        ("train.pckl.gzip", "train.xyz"),
        ("train.pkl.gzip", "train.xyz"),
        ("my.pkl.gz.d/train.pkl.gz", "my.pkl.gz.d/train.xyz"),
        ("train.data", "train.data.xyz"),
    ],
)
def test_default_output_filename(name, expected):
    assert default_output_filename(name) == expected


def test_writes_xyz_next_to_the_dataset(tmp_path):
    at = bulk("Al", "fcc", a=4.05, cubic=True)
    df = pd.DataFrame({"ase_atoms": [at], "energy": [-1.0], "forces": [np.zeros((len(at), 3))]})
    src = tmp_path / "train.pckl.gz"
    df.to_pickle(src, compression="gzip")
    main([str(src), "-e", "energy"])
    out = read(tmp_path / "train.xyz", index=":")
    assert len(out) == 1 and out[0].get_potential_energy() == pytest.approx(-1.0)
