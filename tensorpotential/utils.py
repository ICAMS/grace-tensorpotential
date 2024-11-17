from __future__ import annotations

import os
import re

import os.path as p
import pandas as pd

from yaml import safe_load


def load_metrics(fname):
    """Load metrics from YAML file, written by gracemaker"""
    with open(fname, "r") as f:
        df = pd.json_normalize(safe_load(f))
    return df


def load_fit(folder):
    """
    Load and compile the metrics and status from a specified folder.

    This function loads metrics from YAML files located in the given folder,
    specifically "test_metrics.yaml" and "train_metrics.yaml". It also checks
    for the presence of a "finished.txt" file to determine if the process is
    finalized.

    Args:
        folder (str): The path to the folder containing the metrics files and
                      the optional "finished.txt" file.

    Returns:
        dict: A dictionary containing the following keys:
            - "test": The loaded test metrics from "test_metrics.yaml".
            - "train": The loaded train metrics from "train_metrics.yaml".
            - "name": The name of the folder provided.
            - "final": A boolean indicating if the "finished.txt" file exists
                       in the folder, signifying that the process is complete.
    """
    res = {
        "test": load_metrics(os.path.join(folder, "test_metrics.yaml")),
        "train": load_metrics(os.path.join(folder, "train_metrics.yaml")),
        "name": folder,
        "final": False,
    }
    if os.path.isfile(os.path.join(folder, "finished.txt")):
        res["final"] = True
    return res


def plot_fit(fit, k="rmse/f_comp", name=None, plot_test=True, plot_train=True, ax=None):
    """
    Plot training and test metrics over epochs.

    This function visualizes the specified metric from both the training and test
    data over the course of training epochs, using Matplotlib for plotting.

    Args:
    fit (dict): A dictionary containing training and test metrics. Expected to
        have keys "train" and "test", each of which should be a DataFrame with metrics.

    k (str, optional): The key representing the metric to be plotted. Defaults to "rmse/f_comp".

    name (str, optional): The name to be used in the plot legend. If not provided, fit["shortname"] is used.

    plot_test (bool, optional): Whether to plot the test metric. Defaults to True.

    plot_train (bool, optional): Whether to plot the train metric. Defaults to True.

    ax (matplotlib.axes.Axes, optional): The axes on which to plot. If not provided, the current axes will be used.

    Returns:
    None

    Notes:
    - The function plots the test metric with a solid line and includes the
    minimum value of the test metric in the legend.
    - The train metric is plotted with a dashed line and uses the same color
    as the test metric if both are plotted.
    - The x-axis is labeled as "Epoch" and the y-axis is labeled with the metric key k.
    """
    from matplotlib import pyplot as plt

    if ax is None:
        ax = plt.gca()
    if name is None:
        name = fit["shortname"]
    c = None
    if plot_test:
        y = fit["test"][k]
        y_min = y.min()
        short_name = name
        p = ax.plot(fit["test"].index, y, label=f"{short_name}({y_min * 1e3:.1f})")
        c = p[0].get_color()
    if plot_train:
        ax.plot(fit["train"].index, fit["train"][k], c=c, ls="--")

    ax.set_ylabel(k)
    ax.set_xlabel("Epoch")


def plot_many_fits(fits, k="rmse/f_comp", plot_test=True, plot_train=False):
    """
    Plot metrics for multiple training fits.

    This function visualizes the specified metric from multiple training fits,
    ordering them by the minimum test metric value in descending order. It can plot
    both training and test metrics.

    Args:
    fits (Union[dict, list]): A collection of training fits. If a dictionary is
    provided, its values are used. If a list is provided, it is used directly.

    k (str, optional): The key representing the metric to be plotted. Defaults to "rmse/f_comp".

    plot_test (bool, optional): Whether to plot the test metric for each fit. Defaults to True.

    plot_train (bool, optional): Whether to plot the train metric for each fit. Defaults to False.

    Returns:
    None

    Notes:
    - The function sorts the fits in descending order based on the minimum value
    of the test metric.
    - For each fit, the function prints the name of the fit and then calls
    plot_fit to create the plot.
    """
    if isinstance(fits, dict):
        list_of_fits = list(fits.values())
    elif isinstance(fits, list):
        list_of_fits = fits.copy()

    list_of_fits = sorted(
        list_of_fits, key=lambda fd: fd["test"][k].min(), reverse=True
    )
    for f in list_of_fits:
        print(f["name"])
        plot_fit(f, k=k, plot_test=plot_test, plot_train=plot_train)


def update_fit_metrics(fit_dict, folders):
    """
    Update INPLACE fit metrics for multiple folders.

    This function updates a dictionary of fit metrics by loading metrics from a
    list of specified folders. It skips folders that already have finished fits
    unless the metrics need updating.

    Args:
    fit_dict (dict): A dictionary where keys are folder names and values are fit metrics dictionaries.

    folders (list): A list of folder names from which to load and update fit metrics.

    Returns:
    None

    Notes:
    - The function skips updating fits that are already marked as finished in
    the fit_dict.
    - For each folder, it attempts to load the fit metrics using load_fit.
    If successful, the metrics are updated in the fit_dict.
    - If an error occurs during loading, it catches the exception and prints an error message.
    """
    for f in folders:
        if f in fit_dict and fit_dict[f].get("finished", False):
            continue
        try:
            fd = load_fit(f)
            print(f, len(fd["test"]))
            fit_dict[f] = fd
        except Exception as e:
            print("Error:", e)


def get_common_prefix(names, align_folder=True):
    """
    Return common string prefix for given list of names (str)
    """
    common_prefix = p.commonprefix(list(names))

    if align_folder and not common_prefix.endswith("/"):
        common_prefix = "/".join(common_prefix.split("/")[:-1])

    return common_prefix


def process_fit_dict(fit_dict, fkey="rmse/f_comp", align_folder=True):
    """
    Process fit dict from gracemaker, return common prefix, dataframe of best fits and
    grouped by common prefix (different seeds) dataframe
    """
    for k, fd in fit_dict.items():
        short_name = k.split("/seed/")[0]
        fd["interim_name"] = short_name

    common_prefix = get_common_prefix(
        [fd["interim_name"] for k, fd in fit_dict.items()], align_folder=align_folder
    )

    for k, fd in fit_dict.items():
        fd["shortname"] = fd["interim_name"][len(common_prefix) :]
        fd["shortname_and_seed"] = k[len(common_prefix) :]

    row_list = []
    for name, fd in fit_dict.items():
        df = fd["test"]
        row = df.sort_values(fkey, ascending=True).iloc[0]
        row["name"] = name
        row["shortname"] = fd["shortname"]
        row_list.append(row)

    df = pd.DataFrame(row_list).sort_values(fkey)
    gdf = (
        df.drop(columns=["name"]).groupby("shortname").agg(["mean", "std", "min", list])
    )
    return (common_prefix, df, gdf)


def discovery_fit_folders(root):
    """
    Automatically discover fit folders

    Parameters:
        root (str): Root directory of fit folders

    Return:
        List of fit folders
    """
    folders = []
    for dirpath, dirnames, filenames in os.walk(root):
        if "test_metrics.yaml" in filenames and "train_metrics.yaml" in filenames:
            folders.append(dirpath)
    return folders


def plot_dashboard(
    fit_dict,
    metric="rmse",
    plot_train=False,
    ax_e=None,
    ax_f=None,
    include_list=None,
    exclude_list=None,
    label="shortname",
):
    """Plot dashboard metrics

    Usage:

    fit_dict={}
    folders = discovery_fit_folders('/path/to/root/')
    update_fit_metrics(fit_dict, folders)
    common_prefix, df, gdf=process_fit_dict(fit_dict, fkey=fkey, align_folder=True)

    fig, ax_e, ax_f = plot_dashboard(fit_dict)
    ax_e.legend(ncol=2, bbox_to_anchor=(0.5,-0.15), loc="upper center")
    ax_f.legend(ncol=2, bbox_to_anchor=(0.5,-0.15), loc="upper center")

    ax_e.set_xscale('log')
    fig.suptitle(common_prefix)
    fig.tight_layout()


    Parameters:
        fit_dict (dict): Dictionary of fit metrics
        metric (str): Metric to plot. Default - rmse
        plot_train (bool): Plot training data. Default - False
        ax_e (Axes): Axes on which to plot
        ax_f (Axes): Axes on which to plot
        include_list (list): List of names/fits to include
        exclude_list (list): List of names/fits to exclude
        label: "shortname" or "shortname_and_seed" or "name"
    """
    from matplotlib import pyplot as plt

    if ax_f is None and ax_e is None:
        fig, (ax_f, ax_e) = plt.subplots(
            nrows=2,
            ncols=1,
            sharex=True,
            squeeze=True,
            figsize=(7, 12),
            facecolor="white",
        )
    else:
        fig = plt.gcf()

    for name, fd in fit_dict.items():
        if exclude_list is not None and name in exclude_list:
            continue

        if include_list is not None and name not in include_list:
            continue

        plot_fit(fd, k=metric + "/depa", ax=ax_e, plot_train=plot_train, name=fd[label])
        plot_fit(
            fd,
            k=metric + "/f_comp",
            ax=ax_f,
            plot_train=plot_train,
            name=fd[label],
        )

    return fig, ax_e, ax_f


CUTOFF_PRESETS = {
    # equilibrium nn-dist + 3 A, crop(4,8).round(1)
    "DEFAULT_CUTOFF_1L": {
        "H": 4.0,
        "N": 4.1,
        "O": 4.2,
        "C": 4.4,
        "F": 4.4,
        "B": 4.7,
        "Cl": 5.0,
        "S": 5.1,
        "Mn": 5.2,
        "P": 5.2,
        "Be": 5.2,
        "Se": 5.4,
        "Si": 5.4,
        "Br": 5.4,
        "Fe": 5.4,
        "Co": 5.5,
        "Cr": 5.5,
        "Ni": 5.5,
        "Ge": 5.5,
        "Ga": 5.5,
        "As": 5.6,
        "Cu": 5.6,
        "V": 5.6,
        "Ti": 5.6,
        "Zn": 5.7,
        "Ru": 5.7,
        "Os": 5.7,
        "Tc": 5.7,
        "Rh": 5.7,
        "Mo": 5.7,
        "Ir": 5.7,
        "Re": 5.7,
        "W": 5.8,
        "Pd": 5.8,
        "Pt": 5.8,
        "I": 5.8,
        "Al": 5.9,
        "Nb": 5.9,
        "Ta": 5.9,
        "Sn": 5.9,
        "Te": 5.9,
        "He": 5.9,
        "Au": 5.9,
        "Ag": 5.9,
        "Sb": 5.9,
        "Cd": 6.0,
        "Li": 6.0,
        "Ne": 6.0,
        "Bi": 6.1,
        "Hf": 6.1,
        "Mg": 6.2,
        "Zr": 6.2,
        "Sc": 6.2,
        "Po": 6.3,
        "Ce": 6.3,
        "In": 6.4,
        "Lu": 6.4,
        "Tm": 6.4,
        "Er": 6.5,
        "Ho": 6.5,
        "Y": 6.5,
        "Dy": 6.5,
        "Tl": 6.5,
        "Hg": 6.5,
        "Tb": 6.5,
        "Pb": 6.6,
        "Gd": 6.6,
        "Sm": 6.6,
        "Pm": 6.6,
        "Nd": 6.7,
        "Pr": 6.7,
        "La": 6.7,
        "Na": 6.7,
        "Yb": 6.8,
        "Ca": 6.9,
        "Eu": 6.9,
        "Ar": 7.0,
        "Sr": 7.2,
        "Ba": 7.4,
        "Ra": 7.5,
        "Kr": 7.5,
        "K": 7.7,
        "Xe": 7.9,
        "Rb": 8.0,
        "Fr": 8.0,
        "Cs": 8.0,
        "Rn": 8.0,
    },
    # equilibrium nn-dist + 1.75 A, crop(3.5,7.5).round(1)
    "DEFAULT_CUTOFF_2L": {
        "H": 3.5,
        "N": 3.5,
        "O": 3.5,
        "C": 3.5,
        "F": 3.5,
        "B": 3.5,
        "Cl": 3.8,
        "S": 3.8,
        "Mn": 3.9,
        "P": 4.0,
        "Be": 4.0,
        "Se": 4.1,
        "Si": 4.1,
        "Br": 4.1,
        "Fe": 4.2,
        "Co": 4.2,
        "Cr": 4.2,
        "Ni": 4.2,
        "Ge": 4.2,
        "Ga": 4.3,
        "As": 4.3,
        "Cu": 4.3,
        "V": 4.3,
        "Ti": 4.4,
        "Zn": 4.4,
        "Ru": 4.4,
        "Os": 4.4,
        "Tc": 4.5,
        "Rh": 4.5,
        "Mo": 4.5,
        "Ir": 4.5,
        "Re": 4.5,
        "W": 4.5,
        "Pd": 4.5,
        "Pt": 4.6,
        "I": 4.6,
        "Al": 4.6,
        "Nb": 4.6,
        "Ta": 4.6,
        "Sn": 4.6,
        "Te": 4.6,
        "He": 4.6,
        "Au": 4.7,
        "Ag": 4.7,
        "Sb": 4.7,
        "Cd": 4.8,
        "Li": 4.8,
        "Ne": 4.8,
        "Bi": 4.8,
        "Hf": 4.9,
        "Mg": 4.9,
        "Zr": 4.9,
        "Sc": 5.0,
        "Po": 5.0,
        "Ce": 5.1,
        "In": 5.1,
        "Lu": 5.2,
        "Tm": 5.2,
        "Er": 5.2,
        "Ho": 5.3,
        "Y": 5.3,
        "Dy": 5.3,
        "Tl": 5.3,
        "Hg": 5.3,
        "Tb": 5.3,
        "Pb": 5.3,
        "Gd": 5.3,
        "Sm": 5.4,
        "Pm": 5.4,
        "Nd": 5.4,
        "Pr": 5.5,
        "La": 5.5,
        "Na": 5.5,
        "Yb": 5.6,
        "Ca": 5.6,
        "Eu": 5.7,
        "Ar": 5.7,
        "Sr": 6.0,
        "Ba": 6.1,
        "Ra": 6.3,
        "Kr": 6.3,
        "K": 6.5,
        "Xe": 6.6,
        "Rb": 6.8,
        "Fr": 7.0,
        "Cs": 7.2,
        "Rn": 7.4,
    },
}


def process_cutoff_dict(pair_cutoff_map: str | dict, element_map: dict):
    """

    Expand user defined cutoff_dict (cut_dict={"AlLi":1}) into dict of tuples -> float
    Example: cut_dict={"AlLi":1}
    """

    # to prevent side effects, work on copy of pair_cutoff_map:
    if isinstance(pair_cutoff_map, dict):
        pair_cutoff_map = pair_cutoff_map.copy()

    # algorithmically re-expand cutoff dict
    all_elements = sorted(set(element_map.keys()))
    if isinstance(pair_cutoff_map, str) and pair_cutoff_map in CUTOFF_PRESETS:
        pair_cutoff_map = CUTOFF_PRESETS[pair_cutoff_map]
        pair_cutoff_map = {
            k: v for k, v in pair_cutoff_map.items() if k in all_elements
        }
        binary_cutoff_dict = {}

        for e1, r1 in pair_cutoff_map.items():
            binary_cutoff_dict[(e1, e1)] = r1
            for e2, r2 in pair_cutoff_map.items():
                binary_cutoff_dict[(e1, e2)] = (r1 + r2) / 2

        binary_cutoff_dict = {"".join(k): v for k, v in binary_cutoff_dict.items()}

        pair_cutoff_map = binary_cutoff_dict

    # pre-expand "*" to all elements:
    upd_cutoff_dict = {}
    key_to_drop = []
    for k, v in pair_cutoff_map.items():
        if "*" in k:
            assert (
                all_elements is not None
            ), f"Can not expand * in {k} if :elements: is not provided"
            for e in all_elements:
                new_k = k.replace("*", str(e))
                upd_cutoff_dict[new_k] = v
            key_to_drop.append(k)
    for k in key_to_drop:
        del pair_cutoff_map[k]
    # upd_cutoff_dict.update(cutoff_dict)
    pair_cutoff_map.update(upd_cutoff_dict)

    pattern = re.compile(r"[A-Z][a-z]*")

    def extract_element_pairs_from_str(s):
        if isinstance(s, str):
            res = tuple(pattern.findall(s))
            if len(res) == 1:
                res = (res[0], res[0])
            return res
        elif isinstance(s, tuple):
            assert len(s) == 2
            assert isinstance(s[0], str)
            assert isinstance(s[1], str)
            return s

    cut_dict = {
        extract_element_pairs_from_str(k): v for k, v in pair_cutoff_map.items()
    }

    #
    sym_upd_cut_d = {}
    for (e1, e2), v in cut_dict.items():
        sym_upd_cut_d[(e2, e1)] = v
    cut_dict.update(sym_upd_cut_d)

    # ensure that only elements from element_map are in cut_dict
    cut_dict = {
        (e1, e2): v
        for (e1, e2), v in cut_dict.items()
        if e1 in element_map and e2 in element_map
    }

    return cut_dict


default_input_dict = {
    "cutoff": 6,
    "seed": 1,
    "data": {
        "filename": "path",
        "test_filename": "path",
        "reference_energy": 0.0,
        "save_dataset": False,
    },
    "potential": {"preset": "GRACE_1L", "scale": False, "shift": False},
    "fit": {
        "loss": {
            "energy": {"type": "square", "weight": 1.0},
            "forces": {"type": "square", "weight": 5.0},
        },
        "maxiter": 500,
        "optimizer": "Adam",
        "opt_params": {
            "learning_rate": 0.01,
            "amsgrad": True,
            "use_ema": True,
            "ema_momentum": 0.99,
            "weight_decay": None,
            "clipvalue": 1.0,
        },
        "learning_rate_reduction": {
            "patience": 5,
            "factor": 0.98,
            "min": 0.0005,
            "stop_at_min": False,
        },
        "loss_norm_by_batch_size": True,
        "batch_size": 20,
        "test_batch_size": 200,
        "jit_compile": True,
        "train_max_n_buckets": 8,
        "test_max_n_buckets": 2,
        "progressbar": True,
        "train_shuffle": True,
    },
}
