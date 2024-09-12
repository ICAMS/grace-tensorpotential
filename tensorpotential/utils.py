import os
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
