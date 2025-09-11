from __future__ import annotations

import glob
import json
import logging
import os
import random
from collections import defaultdict, deque
from dataclasses import dataclass
from itertools import cycle
from ase.io import read
from ase.calculators.calculator import PropertyNotImplementedError

import numpy as np
import pandas as pd
import tensorflow as tf

from tensorpotential import constants as tc
from tensorpotential.data.databuilder import (
    GeometricalDataBuilder,
    ReferenceEnergyForcesStressesDataBuilder,
    construct_batches,
    DEFAULT_STRESS_UNITS,
)
from tensorpotential.data.dataset_plotter import (
    DatasetHistPlotter,
    DEFAULT_PLOT_TARGETS,
    DEFAULT_UNIT_TRANSFORM,
)
from tensorpotential.data.process_df import (
    compute_corrected_energy,
    compute_compositions,
    compute_convexhull_dist,
    E_CHULL_DIST_PER_ATOM,
)
from tensorpotential.data.weighting import EnergyBasedWeightingPolicy
from tensorpotential.tensorpot import get_output_dir

LOG_FMT = "%(asctime)s %(levelname).1s - %(message)s"
logging.basicConfig(level=logging.INFO, format=LOG_FMT, datefmt="%Y/%m/%d %H:%M:%S")
log = logging.getLogger()


TRAINING_SET_FNAME = "training_set.pkl.gz"
TESTING_SET_FNAME = "test_set.pkl.gz"


@dataclass
class MyInputContext:
    input_pipeline_id: int
    num_input_pipelines: int


def sizeof_fmt(file_name_or_size, suffix="B"):
    if isinstance(file_name_or_size, str):
        file_name_or_size = os.path.getsize(file_name_or_size)
    for unit in ["", "Ki", "Mi", "Gi", "Ti", "Pi", "Ei", "Zi"]:
        if abs(file_name_or_size) < 1024.0:
            return "%3.1f%s%s" % (file_name_or_size, unit, suffix)
        file_name_or_size /= 1024.0
    return "%.1f%s%s" % (file_name_or_size, "Yi", suffix)


def load_dataframe(filename: str, compression: str = "infer") -> pd.DataFrame:
    filesize = os.path.getsize(filename)
    log.info(
        "Loading dataframe from pickle file {} ({})".format(
            filename, sizeof_fmt(filesize)
        )
    )
    if filename.endswith(".gzip"):
        compression = "gzip"
    df = pd.read_pickle(filename, compression=compression)
    return df


def load_extxyz(filename: str) -> pd.DataFrame:
    logging.info(
        f"Reading extxyz file: {filename} ({sizeof_fmt(filename)})"
    )
    data = read(filename, format="extxyz", index=":")
    logging.info(f"{len(data)} structures read from {filename}")
    logging.info("Converting to dataframe")
    df = pd.DataFrame({"ase_atoms": data})
    logging.info("Extracting energy")

    def get_potential_energy(at):
        try:
            return at.get_potential_energy(force_consistent=True)
        except PropertyNotImplementedError:
            return at.get_potential_energy()

    df["energy"] = df["ase_atoms"].map(get_potential_energy)
    logging.info("Extracting forces")
    df["forces"] = df["ase_atoms"].map(lambda at: at.get_forces())

    try:
        logging.info("Trying to extract stresses")
        df["stress"] = df["ase_atoms"].map(lambda at: at.get_stress())
    except Exception as e:
        logging.error(f"Error while trying to extract stress: {e}, ignoring")

    logging.info("Extracting info")
    df["info"] = df["ase_atoms"].map(lambda at: at.info)
    return df


def load_dataset(filenames):
    """Load multiple dataframes and concatenate it into one"""
    if isinstance(filenames, str):
        files_to_load = [filenames]
    elif isinstance(filenames, list):
        files_to_load = filenames
    else:
        raise ValueError(
            f"Non-supported type of filename: `{filenames}` (type: {type(filenames)})"
        )
    log.info("Search for dataset file(s): " + str(files_to_load))
    if files_to_load is not None:
        dfs = []
        for i, fname in enumerate(files_to_load):
            log.info(f"#{i + 1}/{len(files_to_load)}: try to load {fname}")
            if os.path.splitext(fname)[-1] in [".xyz", ".extxyz"]:
                df = load_extxyz(fname)
            else:
                df = load_dataframe(fname)
            log.info(f" {len(df)} structures found")
            if "name" not in df.columns:
                df["name"] = fname + ":" + df.index.map(str)
            dfs.append(df)
        tot_df = pd.concat(dfs, axis=0).reset_index(drop=True)
    else:  # if ref_df is still not loaded, try to query from DB
        raise RuntimeError("No files to load")
    return tot_df


def load_train_test_datasets(data_config, seed=None):
    train_fname = data_config["filename"]
    log.info(f"Loading training data from {train_fname}")
    train_df = load_dataset(train_fname)
    log.info(f"Loaded dataset size: {train_df.shape}")
    if "train_size" in data_config:
        train_size = data_config["train_size"]
        if train_size < 1:
            train_df = train_df.sample(frac=train_size, random_state=seed)
        else:
            train_df = train_df.sample(n=int(train_size), random_state=seed)
        log.info(f"Reducing training size to {len(train_df)} (train_size={train_size})")

    # run train data builder

    test_df = None
    test_filename = data_config.get("test_filename")
    test_size = data_config.get("test_size")
    if test_filename is not None or test_size is not None:
        if test_filename:
            log.info(f"Loading test data: {test_filename}")
            test_df = load_dataset(test_filename)
            if test_size is not None:
                if test_size > 1:
                    test_df = test_df.sample(n=test_size, random_state=seed)
            # run train data builder
        elif test_size is not None:
            log.info(f"Split test data from train dataset (fraction = {test_size})")
            tmp_df = train_df.reset_index(drop=True)
            test_df = tmp_df.sample(frac=test_size, random_state=seed)
            train_df = tmp_df.drop(test_df.index)

    return train_df, test_df


def extract_elements(df):
    els = set()
    for at in df["ase_atoms"]:
        els.update(at.get_chemical_symbols())

    return els


def apply_weighting(fit_config, train_df, test_df):
    weighting_kwargs = fit_config["weighting"].copy()
    weighting_type = weighting_kwargs["type"]
    if weighting_type == "energy_based":
        log.info(f"Applying energy based weighting")
        weighting_kwargs.pop("type")
        weighting = EnergyBasedWeightingPolicy(**weighting_kwargs)
        train_df = weighting.generate_weights(train_df)
        if test_df is not None:
            test_df = weighting.generate_weights(test_df)
    else:
        raise ValueError(f"Unsupported weighting type {weighting_type}")

    return train_df, test_df


def check_weighting(train_df, test_df):
    # Check, if DATA_ENERGY_WEIGHTS and/or DATA_FORCE_WEIGHTS are in dataframe columns
    log.info(
        f"TRAIN WEIGHTS: energy - {tc.DATA_ENERGY_WEIGHTS in train_df.columns}, forces - {tc.DATA_FORCE_WEIGHTS in train_df.columns} "
    )
    if test_df is not None:
        log.info(
            f"TEST WEIGHTS: energy - {tc.DATA_ENERGY_WEIGHTS in test_df.columns}, forces - {tc.DATA_FORCE_WEIGHTS in test_df.columns} "
        )


def read_saved_dataset_stats(dataset_path):
    with open(os.path.join(dataset_path, "stats.json"), "r") as f:
        stats = json.load(f)
    return stats


def compute_saved_dataset_num_of_batches(dataset_paths):
    n_batches = sum([len(tf.data.Dataset.load(fname)) for fname in dataset_paths])
    return n_batches


def compute_number_of_batches(dds, key):
    n_batch = 0
    for b in dds:
        tensor = b[key]
        if hasattr(tensor, "values"):
            if all((v.shape.rank > 0 and v.shape[0] > 0) for v in tensor.values):
                n_batch += 1
        else:
            n_batch += 1
    return n_batch


def limit_dataset_size(current_size, limit_size):
    # further limit train size to train_size
    if limit_size is not None:
        # if limit_size < 1:
        #     # limit_size is a fraction
        #     limit_size = max(1, int(limit_size * current_size))
        return min(current_size, limit_size)
    return current_size


class FutureDistributedDataset:
    def __init__(self, data_config):
        self.data_config = data_config
        self.dataset_path = data_config["path"]  # point to stage3 folder
        self.train_size = data_config.get("train_size")
        self.test_dataset_path = data_config.get("test_path")
        self.test_size = data_config.get("test_size")
        self.test_shards = data_config.get("test_shards")

        if "*" not in self.dataset_path:
            self.dataset_path = os.path.join(self.dataset_path, "shard_*-of-*")

        log.info(f"Initial TRAIN dataset path: {self.dataset_path}")
        self.datasets_filenames = glob.glob(self.dataset_path)
        log.info(f"Initial TRAIN dataset: {len(self.datasets_filenames)} shards found")
        # random.shuffle(self.datasets_filenames)  # shuffle

        if self.test_dataset_path is not None:
            if "*" not in self.test_dataset_path:
                self.test_dataset_path = os.path.join(
                    self.test_dataset_path, "shard_*-of-*"
                )
            log.info(f"Initial TEST dataset path: {self.test_dataset_path}")
            self.test_datasets_filenames = glob.glob(self.test_dataset_path)
            # random.shuffle(self.test_datasets_filenames)  # shuffle
        else:
            self.test_datasets_filenames = None

        # option to have single-stream dataset
        self.distribute_values = self.data_config.get("distribute_values", False)

    def generate_dataset(self, strategy, signatures=None):
        def filter_batch_by_signature(batch):
            return {k: v for k, v in batch.items() if k in signatures}

        # imitation of single-GPU context for distribute-values fit
        single_context = MyInputContext(input_pipeline_id=0, num_input_pipelines=1)

        def dataset_fn(
            context, shard_filenames, n_take=None, n_skip=None, n_take2=None
        ):
            """Distributed loading of certain shards only"""
            total = context.num_input_pipelines
            ind = context.input_pipeline_id
            original_datasets = [
                tf.data.Dataset.load(filepath, compression="GZIP")
                for filepath in shard_filenames[ind::total]
            ]
            dataset = tf.data.Dataset.from_tensor_slices(original_datasets).interleave(
                lambda x: x,
                cycle_length=min(4, len(original_datasets)),
                num_parallel_calls=1,
            )
            if signatures:
                dataset = dataset.map(filter_batch_by_signature)
            if n_take:
                dataset = dataset.take(n_take)
            if n_skip:
                dataset = dataset.skip(n_skip)
            if n_take2:
                dataset = dataset.take(n_take2)
            # Add repeat BEFORE prefetch for training datasets
            # For training, you almost always want to repeat indefinitely, then limit with steps_per_epoch
            dataset = dataset.repeat()  # Repeat indefinitely for training
            dataset = dataset.prefetch(16)
            return dataset

        test_ds = None
        train_ds = None

        # CASE 1: exact test dataset (options train_size and test_size are possible)
        if self.test_dataset_path is not None:
            log.info("Case 1")

            train_n_batch = limit_dataset_size(
                compute_saved_dataset_num_of_batches(self.datasets_filenames),
                self.train_size,
            )

            test_n_batch = limit_dataset_size(
                compute_saved_dataset_num_of_batches(self.test_datasets_filenames),
                self.test_size,
            )

            if self.distribute_values:
                log.info(
                    f"DISTRIBUTED VALUES dataset: {len(self.datasets_filenames)} shards found in {self.dataset_path}"
                )
                train_ds = dataset_fn(
                    single_context, self.datasets_filenames, n_take=self.train_size
                )
                log.info(
                    f"DISTRIBUTED VALUES TEST dataset: {len(self.test_datasets_filenames)} shards found in {self.test_dataset_path}"
                )
                test_ds = dataset_fn(
                    single_context, self.test_datasets_filenames, n_take=self.test_size
                )
            else:
                log.info(
                    f"DISTRIBUTED TRAIN dataset: {len(self.datasets_filenames)} shards found in {self.dataset_path}"
                )
                train_ds = strategy.distribute_datasets_from_function(
                    lambda context: dataset_fn(
                        context, self.datasets_filenames, n_take=self.train_size
                    ),
                )

                log.info(
                    f"DISTRIBUTED TEST dataset: {len(self.test_datasets_filenames)} shards found in {self.test_dataset_path}"
                )
                test_ds = strategy.distribute_datasets_from_function(
                    lambda context: dataset_fn(
                        context, self.test_datasets_filenames, n_take=self.test_size
                    ),
                )

        # CASE 2/3: no test_path, but test_shards or test_size provided - split test from train
        elif self.test_shards or self.test_size:
            # CASE 2: split test from train using test_shards
            if self.test_shards:
                log.info("Case 2")  # num of shards is provided
                log.info(
                    f"TEST set: request to split {self.test_shards} shards from train set"
                )
                assert (
                    self.test_shards > 0
                ), "test_shards must be integer or float, greater than zero (number of shards)"
                if self.test_shards < 1:
                    self.test_shards = max(
                        int(np.floor(self.test_shards * len(self.datasets_filenames))),
                        1,
                    )
                assert self.test_shards < len(
                    self.datasets_filenames
                ), f"`test_shards`={self.test_shards} must be less than num of shards ({len(self.datasets_filenames)})"

                self.datasets_filenames, self.test_datasets_filenames = (
                    self.datasets_filenames[: -self.test_shards],
                    self.datasets_filenames[-self.test_shards :],
                )

                train_n_batch = limit_dataset_size(
                    compute_saved_dataset_num_of_batches(self.datasets_filenames),
                    self.train_size,
                )

                test_n_batch = limit_dataset_size(
                    compute_saved_dataset_num_of_batches(self.test_datasets_filenames),
                    self.test_size,
                )
                logging.info(
                    f"Number of train shards: {len(self.datasets_filenames)}, test shards: {len(self.test_datasets_filenames)}"
                )
                if self.distribute_values:
                    raise NotImplementedError()
                else:
                    train_ds = strategy.distribute_datasets_from_function(
                        lambda context: dataset_fn(
                            context, self.datasets_filenames, n_take=self.train_size
                        ),
                    )
                    test_ds = strategy.distribute_datasets_from_function(
                        lambda context: dataset_fn(
                            context, self.test_datasets_filenames, n_take=self.test_size
                        ),
                    )

            # CASE 3: split test from train using test_size (num of test batches)
            elif self.test_size:
                log.info("Case 3")
                assert isinstance(
                    self.test_size, int
                ), "test_size must be integer (number of batches)"

                train_n_batch = compute_saved_dataset_num_of_batches(
                    self.datasets_filenames
                )
                assert self.test_size < train_n_batch, (
                    f"Requested number of test batches ({self.test_size}) is greater or equal than total number of "
                    f"provided train batches ({train_n_batch})"
                )

                if self.distribute_values:
                    log.info("Case 3: distributed values")
                    test_ds = dataset_fn(
                        single_context, self.datasets_filenames, n_take=self.test_size
                    )

                    train_ds = dataset_fn(
                        single_context,
                        self.datasets_filenames,
                        n_skip=self.test_size,
                        n_take2=self.train_size,  # could be None
                    )
                else:
                    test_ds = strategy.distribute_datasets_from_function(
                        lambda context: dataset_fn(
                            context, self.datasets_filenames, n_take=self.test_size
                        ),
                    )
                    train_ds = strategy.distribute_datasets_from_function(
                        lambda context: dataset_fn(
                            context,
                            self.datasets_filenames,
                            n_skip=self.test_size,
                            n_take2=self.train_size,  # could be None
                        ),
                    )

                test_n_batch = self.test_size
                train_n_batch -= test_n_batch  # reduce num of available batches
                train_n_batch = limit_dataset_size(train_n_batch, self.train_size)
        # CASE 4: No test set
        else:
            log.info("Case 4")
            if self.distribute_values:
                raise NotImplementedError()
            else:
                train_n_batch = limit_dataset_size(
                    compute_saved_dataset_num_of_batches(self.datasets_filenames),
                    self.train_size,
                )

                train_ds = strategy.distribute_datasets_from_function(
                    lambda context: dataset_fn(
                        context, self.datasets_filenames, n_take=self.train_size
                    ),
                )

        # count only batches, complete over all replicas
        train_n_batch = (
            train_n_batch
            // strategy.num_replicas_in_sync
            * strategy.num_replicas_in_sync
        )
        logging.info(
            f"TRAIN number of batches (aligned): total - {train_n_batch}, per-replica - {train_n_batch//strategy.num_replicas_in_sync}"
        )
        train_ds.n_batch = train_n_batch  # TODO: fix "monkey patching"
        train_ds.distribute_values = self.distribute_values

        if test_ds is not None:
            test_n_batch = (
                test_n_batch
                // strategy.num_replicas_in_sync
                * strategy.num_replicas_in_sync
            )
            logging.info(
                f"TEST number of batches (aligned): total - {test_n_batch}, per-replica - {test_n_batch//strategy.num_replicas_in_sync}"
            )
            test_ds.n_batch = test_n_batch
            test_ds.distribute_values = self.distribute_values

        return train_ds, test_ds


def load_and_prepare_distributed_datasets(args_yaml, seed=1234, strategy=None):
    data_config = args_yaml[tc.INPUT_DATA_SECTION]
    potential_config = args_yaml[tc.INPUT_POTENTIAL_SECTION]
    dataset_path = data_config["path"]  # point to stage3 folder
    test_dataset_path = data_config.get("test_path")
    test_size = data_config.get("test_size")
    test_shards = data_config.get("test_shards")

    shift = 0
    scale = 1.0
    stats = read_saved_dataset_stats(dataset_path)
    element_map = stats["element_map"]
    scale_value = stats["scale"]
    avg_n_neigh = stats["avg_n_neigh"]

    if potential_config.get("scale", False):
        scale_opt = potential_config["scale"]
        if isinstance(scale_opt, float):
            scale = scale_opt
            log.info(f"Data scale: {scale}")
        else:
            scale = scale_value
            scale = max(1e-1, scale)  # prevent too small values
            if scale < 0.8:
                log.warning(
                    f"Constant potential::{scale=} is possibly too small. Consider setting it manually unless you know "
                    f"what you are doing"
                )
            log.info(f"Data (auto) scale: {scale}")

    data_stats = {
        "avg_n_neigh": avg_n_neigh,
        "constant_out_shift": shift,
        "constant_out_scale": scale,
        "atomic_shift_map": None,
    }

    return (
        FutureDistributedDataset(data_config),
        test_dataset_path or test_size or test_shards,  # mocking test dataset
        element_map,
        data_stats,
        None,  # train_grouping_df
        None,  # test_grouping_df
    )


def load_and_prepare_datasets(
    args_yaml,
    batch_size,
    test_batch_size=None,
    seed=1234,
    strategy=None,
    float_dtype: str = "float64",
):
    # TODO: Need to use constants module.
    #  Too many strings are in here....
    data_config = args_yaml[tc.INPUT_DATA_SECTION]
    potential_config = args_yaml[tc.INPUT_POTENTIAL_SECTION]
    fit_config = args_yaml[tc.INPUT_FIT_SECTION]
    rcut = args_yaml[tc.INPUT_CUTOFF]
    if data_config.get("distributed", False):
        log.info("Precomputed and/or distributed dataset(s) will be used")
        return load_and_prepare_distributed_datasets(
            args_yaml, seed=seed, strategy=strategy
        )

    # check if fit stress
    is_fit_stress = tc.INPUT_FIT_LOSS in fit_config and (
        tc.INPUT_FIT_LOSS_STRESS in fit_config[tc.INPUT_FIT_LOSS]
        or tc.INPUT_FIT_LOSS_VIRIAL in fit_config[tc.INPUT_FIT_LOSS]
    )

    # load data (no yet preprocessing)
    train_df, test_df = load_train_test_datasets(data_config, seed=seed)
    train_df["NUMBER_OF_ATOMS"] = train_df[tc.COLUMN_ASE_ATOMS].map(len)
    if test_df is not None:
        test_df["NUMBER_OF_ATOMS"] = test_df[tc.COLUMN_ASE_ATOMS].map(len)

    if tc.INPUT_FIT_LOSS_ENERGY in fit_config[tc.INPUT_FIT_LOSS]:
        if tc.INPUT_REFERENCE_ENERGY in data_config:
            reference_energy = data_config.get(tc.INPUT_REFERENCE_ENERGY)
            if reference_energy == 0:
                log.info(
                    "Set 'energy_corrected' to 'energy', because reference_energy=0"
                )
                train_df["energy_corrected"] = train_df["energy"]
                if test_df is not None:
                    test_df["energy_corrected"] = test_df["energy"]
            elif isinstance(reference_energy, dict):
                log.info(
                    f"Construct 'energy_corrected' from 'energy', using single-atom reference {reference_energy=}"
                )
                compute_corrected_energy(train_df, esa_dict=reference_energy)
                if test_df is not None:
                    compute_corrected_energy(test_df, esa_dict=reference_energy)
            else:
                raise RuntimeError(
                    f"'input::data::reference_energy'={reference_energy} is not supported. "
                    "Only 0 or dict are supported."
                )
        if "energy_corrected" not in train_df.columns or (
            test_df is not None and "energy_corrected" not in test_df.columns
        ):
            raise RuntimeError(
                "'energy_corrected' column is not in dataset, either provide input::data::reference_energy: 0 or dict "
                "or generate column externally"
            )

        train_df["energy_corrected_per_atom"] = (
            train_df["energy_corrected"] / train_df["NUMBER_OF_ATOMS"]
        )
        if test_df is not None:
            test_df["energy_corrected_per_atom"] = (
                test_df["energy_corrected"] / test_df["NUMBER_OF_ATOMS"]
            )

    stress_units = data_config.get("stress_units", DEFAULT_STRESS_UNITS)
    if is_fit_stress:

        def extract_stress(df):
            """Extract stress from 'results' dict"""
            if "stress" not in df.columns:
                if "results" in df.columns:
                    df["stress"] = df["results"].map(
                        lambda d: d.get("stress") if d is not None else None
                    )
                else:
                    raise ValueError(
                        "Neither 'stress' nor 'results' column is present in dataframe"
                    )

        extract_stress(train_df)

        if test_df is not None:
            extract_stress(test_df)
        log.info(f"Stress units: {stress_units}")

    shift = 0
    scale = 1.0
    esa_dict = None
    if potential_config.get("shift", False):
        # train_df["energy_corrected_orig"] = train_df["energy_corrected"]
        elements = compute_compositions(train_df)
        n_elements_cols = ["n_" + e for e in elements]
        n_elements = train_df[n_elements_cols]
        total_energy = train_df["energy_corrected"]
        res = np.linalg.lstsq(n_elements, total_energy, rcond=None)
        e0_list = res[0]
        esa_dict = {e: e0 for e, e0 in zip(elements, e0_list)}
        log.info(f"Single-atom energy shift (computed on train set): {esa_dict}")

    if potential_config.get("scale", False):
        scale_opt = potential_config["scale"]
        if isinstance(scale_opt, float):
            scale = scale_opt
            log.info(f"Data scale: {scale}")
        else:
            rms_f = np.vstack(train_df["forces"].to_numpy())
            # TODO: currently per-component, need to be per-vector or per-component
            scale = np.sqrt(np.mean(rms_f**2))
            scale = max(1e-1, scale)  # prevent too small values
            if scale < 0.8:
                log.warning(
                    f"Constant potential::{scale=} is possibly too small. Consider setting it manually unless you know "
                    f"what you are doing"
                )
            log.info(f"Data (auto) scale: {scale}")

    if fit_config.get("compute_convex_hull", False):
        # compute energy up to convex hull
        if test_df is None:
            # compute only for train set
            log.info("Computing convex hull for train dataset")
            compute_convexhull_dist(
                train_df,
                energy_per_atom_column="energy_corrected_per_atom",
                verbose=True,
            )
        else:
            # compute only for train set
            train_df["is_train"] = 1
            test_df["is_train"] = 0
            tot_df = pd.concat([train_df, test_df], axis=0, copy=False)
            log.info("Computing convex hull for joint train+test dataset")
            compute_convexhull_dist(
                tot_df, energy_per_atom_column="energy_corrected_per_atom", verbose=True
            )
            train_df = tot_df.query("is_train==1").drop(columns=["is_train"])
            test_df = tot_df.query("is_train==0").drop(columns=["is_train"])
    else:
        log.info(
            "input::fit::compute_convex_hull is False, convex hull and 'low energy' group will not be computed "
        )
    log.info(f"Train Dataset size: {len(train_df)}")

    if train_df.index.duplicated().any():
        log.warning("Duplicate indices in TRAIN dataframe found, resetting")
        train_df.reset_index(drop=True, inplace=True)

    has_test_set = test_df is not None
    if has_test_set:
        if test_df.index.duplicated().any():
            log.warning("Duplicate indices in TEST dataframe found, resetting")
            test_df.reset_index(drop=True, inplace=True)

        log.info(f"Test Dataset size: {len(test_df)}")

    # extract elements mapping from input or from dataset
    elements = potential_config.get("elements")
    if elements is None:
        elements = extract_elements(train_df)
        if has_test_set:
            elements.update(extract_elements(test_df))
        elements = list(sorted(elements))
        element_map = {e: i for i, e in enumerate(elements)}
        log.info("Extract elements from train and test datasets")
    elif isinstance(elements, list):
        log.info("Elements are provided as list")
        elements = list(elements)
        element_map = {e: i for i, e in enumerate(elements)}
    elif isinstance(elements, dict):
        log.info("Elements are provided as dict")
        element_map = elements
    log.info(f"Elements mapping: {element_map}")

    # apply E,F weights (i.e. energy-based weights)
    if fit_config.get("weighting") is not None:
        train_df, test_df = apply_weighting(fit_config, train_df, test_df)
    check_weighting(train_df, test_df)

    if data_config.get("save_dataset", True):
        training_set_full_fname = os.path.join(get_output_dir(seed), TRAINING_SET_FNAME)
        os.makedirs(os.path.dirname(training_set_full_fname), exist_ok=True)
        log.info(f"Saving current training set to {training_set_full_fname}")
        train_df.to_pickle(training_set_full_fname)
        if has_test_set:
            testing_set_full_fname = os.path.join(
                get_output_dir(seed), TESTING_SET_FNAME
            )
            os.makedirs(os.path.dirname(testing_set_full_fname), exist_ok=True)
            log.info(f"Saving current test set to {testing_set_full_fname}")
            test_df.to_pickle(testing_set_full_fname)

    user_cutoff_dict = args_yaml.get(tc.INPUT_CUTOFF_DICT)

    # instantiate DataBuilders class (with optional parameters); HARDCODED, but rarely changes
    data_builders = [
        GeometricalDataBuilder(
            element_map,
            cutoff=rcut,
            cutoff_dict=user_cutoff_dict,
            is_fit_stress=is_fit_stress,
            float_dtype=float_dtype,
        ),
    ]
    if (
        tc.INPUT_FIT_LOSS_ENERGY in fit_config[tc.INPUT_FIT_LOSS]
        or tc.INPUT_FIT_LOSS_FORCES in fit_config[tc.INPUT_FIT_LOSS]
    ):
        data_builders.append(
            ReferenceEnergyForcesStressesDataBuilder(
                normalize_weights=fit_config.get("normalize_weights", False),
                normalize_force_per_structure=fit_config.get(
                    "normalize_force_per_structure", False
                ),
                is_fit_stress=is_fit_stress,
                stress_units=stress_units,
                float_dtype=float_dtype,
            ),
        )
    extras = data_config.get("extra_components", None)
    if extras is not None:
        import importlib

        mod = importlib.import_module(
            "tensorpotential.experimental.extra_data_builders"
        )
        for db_name, db_config in extras.items():
            try:
                db = getattr(mod, db_name)
                data_builders.append(db(**db_config, float_dtype=float_dtype))
            except AttributeError as e:
                raise NameError(f"Could not find data builder {db_name}")

    # preprocess data
    log.info("Train set processing")
    max_n_buckets = fit_config.get("train_max_n_buckets", 5)
    log.info(f"Train buckets: {max_n_buckets}")
    train_batches, padding_stats = construct_batches(
        train_df,
        data_builders=data_builders,
        batch_size=batch_size,
        max_n_buckets=max_n_buckets,  # 1
        return_padding_stats=True,
        verbose=True,
        max_workers=data_config.get("max_workers"),
    )

    if padding_stats:
        logging.info(
            f"[TRAIN] dataset stats: "
            f"num. batches: {len(train_batches)} | "
            f"num. real structures: {padding_stats['nreal_struc']} (+{padding_stats['pad_nstruct'] / padding_stats['nreal_struc'] * 1e2:.2f}%) | "
            f"num. real atoms: {padding_stats['nreal_atoms']} (+{padding_stats['pad_nat'] / padding_stats['nreal_atoms'] * 1e2:.2f}%) | "
            f"num. real neighbours: {padding_stats['nreal_neigh']} (+{padding_stats['pad_nneigh'] / padding_stats['nreal_neigh'] * 1e2:.2f}%) "
        )
    else:
        logging.info(f"[TRAIN] dataset stats:  num. batches: {len(train_batches)}")
    # compute average number of neighbours over TRAIN set
    avg_n_neigh = potential_config.get("avg_n_neigh")
    if avg_n_neigh is None:
        total_number_neigh = sum(b[tc.N_NEIGHBORS_REAL] for b in train_batches)
        total_number_atoms = sum(b[tc.N_ATOMS_BATCH_REAL] for b in train_batches)
        avg_n_neigh = total_number_neigh / total_number_atoms
        logging.info(f"Average number of neighbors (computed): {avg_n_neigh}")
    else:
        logging.info(f"Average number of neighbors (provided): {avg_n_neigh}")

    if has_test_set:
        log.info("Test set processing")
        test_max_n_buckets = fit_config.get("test_max_n_buckets", 1)
        log.info(f"Test buckets: {test_max_n_buckets}")
        test_batches, test_padding_stats = construct_batches(
            test_df,
            data_builders=data_builders,
            batch_size=test_batch_size if test_batch_size else batch_size,
            max_n_buckets=test_max_n_buckets,  # 1
            return_padding_stats=True,
            verbose=True,
            max_workers=data_config.get("max_workers"),
        )
        if test_padding_stats:
            logging.info(
                f"[TEST] dataset stats: "
                f"num. batches: {len(test_batches)} | "
                f"num. real structures: {test_padding_stats['nreal_struc']} (+{test_padding_stats['pad_nstruct'] / test_padding_stats['nreal_struc'] * 1e2:.2f}%) | "
                f"num. real atoms: {test_padding_stats['nreal_atoms']} (+{test_padding_stats['pad_nat'] / test_padding_stats['nreal_atoms'] * 1e2:.2f}%) | "
                f"num. real neighbours: {test_padding_stats['nreal_neigh']} (+{test_padding_stats['pad_nneigh'] / test_padding_stats['nreal_neigh'] * 1e2:.2f}%) "
            )
        else:
            logging.info(f"[TEST] dataset stats:  num. batches: {len(test_batches)}")

    else:
        # tuple_of_test_datasets = None
        test_batches = None
    # esa_dict: el -> e0
    # element_map: el -> mu
    # atomic_shift_map :  mu -> e0
    # convert atomic_shift_map from esa_dict
    atomic_shift_map = (
        {element_map[el]: e0 for el, e0 in esa_dict.items()}
        if esa_dict is not None
        else None
    )
    # use_per_specie_n_nei = args_yaml.get(tc.INPUT_USE_PER_SPECIE_N_NEI, False)
    data_stats = {
        "avg_n_neigh": (
            avg_n_neigh  # if not use_per_specie_n_nei else avg_n_neigh_per_specie
        ),
        "constant_out_shift": shift,
        "constant_out_scale": scale,
        "atomic_shift_map": atomic_shift_map,
    }
    # label train and test dataframes with low-energy(<=1 eV/atom above conv.hull) and rest
    test_grouping_df = None
    train_grouping_df = None
    if E_CHULL_DIST_PER_ATOM in train_df.columns:
        train_df["group__low"] = train_df[E_CHULL_DIST_PER_ATOM] <= 1.0
        train_grouping_df = get_group_mapping_df(train_df)

        if test_df is not None:
            test_df["group__low"] = test_df[E_CHULL_DIST_PER_ATOM] <= 1.0
            test_grouping_df = get_group_mapping_df(test_df)

    # plotting histograms in ./seed/{current_seed}/plots
    try:
        plot_out_dir = f"{get_output_dir(seed)}/plots"
        DatasetHistPlotter.plot(
            {"train": train_df, "test": test_df},
            plot_out_dir,
            plot_targets=DEFAULT_PLOT_TARGETS,
            units_transform=DEFAULT_UNIT_TRANSFORM,
        )
        log.info(
            f"Train/test data distributions are plotted and saved to {plot_out_dir}"
        )
    except Exception as e:
        log.warning(f"Failed to make histograms plot for datasets ({e}), skipping")

    return (
        train_batches,
        test_batches,
        element_map,
        data_stats,
        train_grouping_df,
        test_grouping_df,
    )


def get_group_mapping_df(df):
    gdf = df[
        [col for col in df.columns if col.startswith("group__")] + ["NUMBER_OF_ATOMS"]
    ]
    return gdf


def key_for_batch_dict(batch_dict):
    return (
        batch_dict[tc.BOND_VECTOR].shape[0],
        batch_dict[tc.N_ATOMS_BATCH_TOTAL].numpy(),
        batch_dict[tc.N_STRUCTURES_BATCH_TOTAL].numpy(),
    )


def regroup_dataset_to_iterator(ds, n_group=1, regroup_window_factor=32):
    if n_group == 1:
        return ds

    def sorting_generator(batched_dataset, window_size, sort_key_fn=key_for_batch_dict):
        """Sorts batches within a window and returns a new generator with partially sorted batches."""

        buffer = []
        for batch in batched_dataset:
            buffer.append(batch)
            if len(buffer) == window_size:
                # Sort the current chunk of batches
                buffer.sort(key=sort_key_fn)
                for sorted_batch in buffer:
                    yield sorted_batch
                buffer = []

        # Yield any remaining batches
        if buffer:
            buffer.sort(key=sort_key_fn)
            for sorted_batch in buffer:
                yield sorted_batch

    # deferred generator
    return sorting_generator(ds, window_size=n_group * regroup_window_factor)


def regroup_similar_batches(ds, n_group=1):
    if n_group == 1:
        return ds

    if is_tf_distr_dataset(ds):
        return ds

    # this is actually a sorting algorithm:
    # Step 1: Create buckets_dict
    buckets_dict = defaultdict(deque)
    for batch_dict in ds:
        key = key_for_batch_dict(batch_dict)
        buckets_dict[key].append(batch_dict)

    # Step 2: Shuffle the keys and the contents of each bucket
    keys = list(buckets_dict.keys())
    random.shuffle(keys)

    # Step 3: Create interleaved chunks
    grouped_ds = []
    buckets_iter = cycle(keys)  # Cycle through keys repeatedly
    non_empty_keys = set(keys)  # Track keys with remaining elements

    while non_empty_keys:
        key = next(buckets_iter)  # Get the next key in a round-robin manner
        if key in non_empty_keys:
            bucket = buckets_dict[key]
            # Extract a chunk of size n_group
            if len(bucket) >= n_group:
                chunk = [bucket.popleft() for _ in range(n_group)]
                grouped_ds.extend(chunk)
            else:
                non_empty_keys.remove(key)
    # add "leftovers"
    for key in keys:
        if buckets_dict[key]:
            grouped_ds.extend(buckets_dict[key])

    # for batch_dict in grouped_ds:
    #     print(key_for_batch_dict(batch_dict))
    assert len(grouped_ds) == len(ds)
    return grouped_ds


def is_tf_distr_dataset(dataset):
    from tensorflow.python.distribute.input_lib import DistributedDatasetsFromFunction

    return isinstance(dataset, (tf.data.Dataset, DistributedDatasetsFromFunction))
