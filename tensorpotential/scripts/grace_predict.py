#!/usr/bin/env python

import warnings

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)

import sys
import os
import pandas as pd
import numpy as np
import argparse

from tensorpotential.calculator import TPCalculator

import logging

LOG_FMT = "%(asctime)s %(levelname).1s - %(message)s"
logging.basicConfig(level=logging.INFO, format=LOG_FMT, datefmt="%Y/%m/%d %H:%M:%S")
logger = logging.getLogger()

from tqdm import tqdm

tqdm.pandas()

NUMBER_ASSERT_ERRORS_SHOWN = 3


def set_magmom(at, magmoms):
    magmoms = np.array(magmoms)
    assert len(at) == magmoms.shape[0]
    if magmoms.shape == (len(at), 3):
        at.arrays["initial_magmoms"] = magmoms
    elif (magmoms.shape == (len(at), 1)) or (magmoms.shape == (len(at),)):
        new_magmoms = np.zeros((len(at), 3))
        new_magmoms[:, 2] = magmoms
        at.arrays["initial_magmoms"] = new_magmoms
    else:
        raise ValueError("mag_mom shape is not recognized")
    return at


def predict(row, calc, raise_errors):
    at = row["ase_atoms"].copy()
    if "mag_mom" in row:
        at = set_magmom(at, row["mag_mom"])
    at.calc = calc
    try:
        e = at.get_potential_energy()
        f = at.get_forces()
        s = at.get_stress()
        return {"energy": e, "forces": f, "stress": s}
    except AssertionError as e:
        if raise_errors:
            raise e
        global NUMBER_ASSERT_ERRORS_SHOWN
        if NUMBER_ASSERT_ERRORS_SHOWN > 0:
            print("Error: ", e)
            NUMBER_ASSERT_ERRORS_SHOWN -= 1
            print("No more errors will be shown.")
        return {}


def main(args=None):
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "-m",
        "--model_path",
        help="provide path to the saved_model directory",
        type=str,
        default="saved_model",
        dest="model_path",
    )

    parser.add_argument(
        "-d",
        "--dataset",
        help="path to the dataset.pkl.gzip containing ase_atoms structures",
        type=str,
        default="dataset.pkl.gz",
        dest="dataset_file",
    )

    parser.add_argument(
        "-o",
        "--output",
        help="path to the OUTPUT dataset (pkl.gzip) containing energy_predicted and forces_predicted",
        type=str,
        default="predicted_dataset.pkl.gz",
        dest="output",
    )

    parser.add_argument(
        "-e",
        "--raise-errors",
        help="Whether to NOT ignore errors and stop the program.",
        action="store_true",
        default=False,
        dest="raise_errors",
    )

    args_parse = parser.parse_args(args)

    model_path = os.path.abspath(args_parse.model_path)
    dataset_file = args_parse.dataset_file
    output_file = args_parse.output
    raise_errors = args_parse.raise_errors

    logger.info(f"Loading model from: {model_path}")
    calc = TPCalculator(
        model=model_path,
        pad_atoms_number=20,
        pad_neighbors_fraction=0.30,
        # max_number_reduction_recompilation=3,
    )

    logger.info(f"Loading dataset from: {dataset_file}")
    df = pd.read_pickle(dataset_file, compression="gzip")

    logger.info(f"Starting prediction")

    df["prediction"] = df.progress_apply(predict, axis=1, args=(calc, raise_errors))
    df["energy_predicted"] = df["prediction"].map(lambda x: x.get("energy"))
    df["forces_predicted"] = df["prediction"].map(lambda x: x.get("forces"))
    df["stress_predicted"] = df["prediction"].map(lambda x: x.get("stress"))

    logger.info(f"Saving dataset to {output_file}")
    df.drop(columns=["ase_atoms", "prediction"]).to_pickle(
        output_file, compression="gzip"
    )


if __name__ == "__main__":
    main()
