from __future__ import annotations

import logging
import numpy as np
import os
import pandas as pd

from tensorpotential.data.weighting import EnergyBasedWeightingPolicy

from tensorpotential import constants as tc
from tensorpotential.data.databuilder import (
    GeometricalDataBuilder,
    ReferenceEnergyForcesStressesDataBuilder,
    construct_batches,
    DEFAULT_STRESS_UNITS,
)
from tensorpotential.data.process_df import (
    compute_corrected_energy,
    compute_compositions,
    compute_convexhull_dist,
    E_CHULL_DIST_PER_ATOM,
)

from tensorpotential.tensorpot import get_output_dir
from collections import defaultdict


LOG_FMT = "%(asctime)s %(levelname).1s - %(message)s"
logging.basicConfig(level=logging.INFO, format=LOG_FMT, datefmt="%Y/%m/%d %H:%M:%S")
log = logging.getLogger()


TRAINING_SET_FNAME = "training_set.pkl.gz"
TESTING_SET_FNAME = "test_set.pkl.gz"


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


def load_and_prepare_datasets(args_yaml, batch_size, test_batch_size=None, seed=1234):
    # TODO: Need to use constants module.
    #  Too many strings are in here....
    data_config = args_yaml[tc.INPUT_DATA_SECTION]
    potential_config = args_yaml[tc.INPUT_POTENTIAL_SECTION]
    fit_config = args_yaml[tc.INPUT_FIT_SECTION]
    rcut = args_yaml[tc.INPUT_CUTOFF]

    # check if fit stress
    is_fit_stress = tc.INPUT_FIT_LOSS in fit_config and (
        tc.INPUT_FIT_LOSS_STRESS in fit_config[tc.INPUT_FIT_LOSS]
        or tc.INPUT_FIT_LOSS_VIRIAL in fit_config[tc.INPUT_FIT_LOSS]
    )

    # load data (no yet preprocessing)
    train_df, test_df = load_train_test_datasets(data_config, seed=seed)
    train_df["NUMBER_OF_ATOMS"] = train_df["ase_atoms"].map(len)
    if test_df is not None:
        test_df["NUMBER_OF_ATOMS"] = test_df["ase_atoms"].map(len)

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
        log.info("Extract elements from train and test datasets")
    elements = list(sorted(elements))
    element_map = {e: i for i, e in enumerate(elements)}
    log.info(f"Elements mapping: {element_map}")

    # apply E,F weights (i.e. energy-based weights)
    if fit_config.get("weighting") is not None:
        train_df, test_df = apply_weighting(fit_config, train_df, test_df)
    check_weighting(train_df, test_df)

    if data_config.get("save_dataset", True):
        training_set_full_fname = os.path.join(get_output_dir(seed), TRAINING_SET_FNAME)
        log.info(f"Saving current training set to {training_set_full_fname}")
        train_df.to_pickle(training_set_full_fname)
        if has_test_set:
            testing_set_full_fname = os.path.join(
                get_output_dir(seed), TESTING_SET_FNAME
            )
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
                data_builders.append(db(**db_config))
            except AttributeError as e:
                raise NameError(f"Could not find data builder {db_name}")

    # if tc.INPUT_FIT_LOSS_EFG in fit_config[tc.INPUT_FIT_LOSS]:
    #     from tensorpotential.experimental.efg.databuilder import ReferenceEFGDataBuilder
    #
    #     data_builders.append(ReferenceEFGDataBuilder())

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
    total_number_neigh = sum(b[tc.N_NEIGHBORS_REAL] for b in train_batches)
    total_number_atoms = sum(b[tc.N_ATOMS_BATCH_REAL] for b in train_batches)
    # compute avg number of neighbors per element
    nat_per_specie = defaultdict(int)
    total_nei_per_specie = defaultdict(int)
    for b in train_batches:
        nps = b["nat_per_specie"]
        tnps = b["total_nei_per_specie"]
        for k, v in nps.items():
            nat_per_specie[k] += v
        for k, v in tnps.items():
            total_nei_per_specie[k] += v
    avg_n_neigh_per_specie = {}
    for k, v in nat_per_specie.items():
        val = v if v > 0 else 1.0
        avg_n_neigh_per_specie[k] = total_nei_per_specie[k] / val
    # log.info(f"Average per specie nnei: {avg_n_neigh_per_specie}")

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
    avg_n_neigh = total_number_neigh / total_number_atoms
    logging.info(f"Average number of neighbors: {avg_n_neigh}")
    # esa_dict: el -> e0
    # element_map: el -> mu
    # atomic_shift_map :  mu -> e0
    # convert atomic_shift_map from esa_dict
    atomic_shift_map = (
        {element_map[el]: e0 for el, e0 in esa_dict.items()}
        if esa_dict is not None
        else None
    )
    use_per_specie_n_nei = args_yaml.get(tc.INPUT_USE_PER_SPECIE_N_NEI, False)
    data_stats = {
        "avg_n_neigh": (
            avg_n_neigh if not use_per_specie_n_nei else avg_n_neigh_per_specie
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
