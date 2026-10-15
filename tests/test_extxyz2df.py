"""extxyz2df: the default output name never overwrites the input file."""

import os

import pandas as pd
import pytest
from ase.build import bulk
from ase.calculators.singlepoint import SinglePointCalculator
from ase.io import write

from tensorpotential.scripts.extxyz2df import default_output_filename, main


@pytest.mark.parametrize(
    "name, expected",
    [
        ("data.extxyz", "data.pkl.gz"),
        ("data.xyz", "data.pkl.gz"),
        ("my.xyz.files/data.extxyz", "my.xyz.files/data.pkl.gz"),
        ("data.txt", "data.txt.pkl.gz"),
    ],
)
def test_default_output_filename(name, expected):
    assert default_output_filename(name) == expected


def test_extxyz_input_is_kept_and_dataset_written(tmp_path):
    at = bulk("Al", "fcc", a=4.05, cubic=True)
    at.calc = SinglePointCalculator(at, energy=-1.0, forces=[[0.0, 0.0, 0.0]] * len(at))
    src = tmp_path / "data.extxyz"
    write(src, [at], format="extxyz")
    before = src.read_bytes()
    main([str(src)])
    assert src.read_bytes() == before
    df = pd.read_pickle(tmp_path / "data.pkl.gz")
    assert len(df) == 1 and df["energy"].iloc[0] == -1.0


def test_output_equal_to_input_is_refused(tmp_path):
    src = tmp_path / "data.extxyz"
    src.write_text("")
    with pytest.raises(ValueError, match="overwrite the input"):
        main([str(src), "--output-dataset-filename", str(src)])
    assert os.path.getsize(src) == 0
