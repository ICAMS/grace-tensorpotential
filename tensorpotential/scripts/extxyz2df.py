#!/usr/bin/env python

import os
import argparse
from tensorpotential.cli.data import load_extxyz

import logging

LOG_FMT = "%(asctime)s %(levelname).1s - %(message)s"
logging.basicConfig(level=logging.INFO, format=LOG_FMT, datefmt="%Y/%m/%d %H:%M:%S")
logger = logging.getLogger()


def sizeof_fmt(file_name_or_size, suffix="B"):
    if isinstance(file_name_or_size, str):
        file_name_or_size = os.path.getsize(file_name_or_size)
    for unit in ["", "Ki", "Mi", "Gi", "Ti", "Pi", "Ei", "Zi"]:
        if abs(file_name_or_size) < 1024.0:
            return "%3.1f%s%s" % (file_name_or_size, unit, suffix)
        file_name_or_size /= 1024.0
    return "%.1f%s%s" % (file_name_or_size, "Yi", suffix)



def default_output_filename(extxyz_filename: str) -> str:
    """``data.extxyz`` / ``data.xyz`` -> ``data.pkl.gz``; any other name gets ``.pkl.gz`` appended."""
    for ext in (".extxyz", ".xyz"):
        if extxyz_filename.endswith(ext):
            return extxyz_filename[: -len(ext)] + ".pkl.gz"
    return extxyz_filename + ".pkl.gz"


def main(args=None):
    ##############################################################################################
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "extxyz_filename",
        help="Name of extxyz file",
        type=str,
    )

    parser.add_argument(
        "--output-dataset-filename",
        help="pickle filename, default is inferenced from extxyz_filename",
        type=str,
        default=None,
    )

    args_parse = parser.parse_args(args)
    extxyz_filename = os.path.abspath(args_parse.extxyz_filename)
    output_dataset_filename = args_parse.output_dataset_filename
    if output_dataset_filename is None:
        output_dataset_filename = default_output_filename(extxyz_filename)
    if os.path.abspath(output_dataset_filename) == extxyz_filename:
        raise ValueError(
            f"Output file {output_dataset_filename} would overwrite the input file"
        )

    logging.info(f"Input filename: {extxyz_filename}")
    logging.info(f"Output filename: {output_dataset_filename}")

    df = load_extxyz(extxyz_filename)
    logging.info("Saving dataframe")
    df.to_pickle(output_dataset_filename, compression="gzip")
    logging.info(
        f"Saved dataframe to {output_dataset_filename} ({sizeof_fmt(output_dataset_filename)})"
    )


if __name__ == "__main__":
    main()
