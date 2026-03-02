import os
import shutil
from contextlib import contextmanager
from pathlib import Path

import pandas as pd
import pytest

from tensorpotential.cli.gracemaker import main
from tensorpotential.utils import load_metrics


def print_full(x):

    pd.set_option("display.max_rows", None)
    pd.set_option("display.max_columns", None)
    pd.set_option("display.width", 2000)
    pd.set_option("display.float_format", "{:20,.2f}".format)
    pd.set_option("display.max_colwidth", None)
    print(x)
    pd.reset_option("display.max_rows")
    pd.reset_option("display.max_columns")
    pd.reset_option("display.width")
    pd.reset_option("display.float_format")
    pd.reset_option("display.max_colwidth")


test_location_folder = Path(__file__).parent.resolve()
keep_only = [
    "input.yaml",
    "model.py",
]
keep_only_after = keep_only + ["log.txt"]


def general_integration_test(
    folder,
    train_ref_metrics,
    test_ref_metrics,
    ref_n_epochs=2,
    ref_n_init_epoch=0,
    input="input.yaml",
    many_runs=None,
    seed=42,
    rel=None,
    abs=None,
    top_folder=None,
):
    print(f"Current folder: {os.getcwd()}")
    many_runs = many_runs or [[input]]
    ref_n_epochs = ref_n_epochs + len(many_runs) * ref_n_init_epoch
    if top_folder is not None:
        prefix = top_folder
    else:
        prefix = test_location_folder
    path = str(prefix / folder)
    inp_fname = ""
    for arg in many_runs:
        inp_fname = inp_fname + "_" + arg[0].split(".")[0]
    tmp_path = str(prefix / ("tmp__" + folder + "_" + inp_fname))

    print(f"Temp path: {tmp_path}")
    if os.path.isdir(tmp_path):
        shutil.rmtree(tmp_path)

    os.makedirs(tmp_path, exist_ok=True)

    for arg in many_runs:
        src, dst = os.path.join(path, arg[0]), os.path.join(tmp_path, arg[0])
        print(f"Copying {src} to {dst}")
        shutil.copy(src, dst)

    with change_directory(tmp_path):
        for inp in many_runs:
            main(inp)
        train_metrics = load_metrics(f"seed/{seed}/train_metrics.yaml")
        test_metrics = load_metrics(f"seed/{seed}/test_metrics.yaml")
        assert len(train_metrics) == len(
            test_metrics
        ), "len(train_metrics) != len(test_metrics)"
        assert len(test_metrics) == ref_n_epochs, "len(test_metrics) != ref_n_epochs"

        last_train_row = train_metrics.iloc[-1]
        print("TRAIN metrics:", last_train_row.to_dict())

        last_test_row = test_metrics.iloc[-1]
        print("TEST metrics:", last_test_row.to_dict())

        # 4. Assert on final metric values using the helper
        _compare_metrics(
            train_metrics,
            train_ref_metrics,
            label="TRAIN",
            rel=rel,
            abs=abs,
        )
        _compare_metrics(
            test_metrics,
            test_ref_metrics,
            label="TEST",
            rel=rel,
            abs=abs,
        )

        if os.path.isdir(tmp_path):
            shutil.rmtree(tmp_path)

    clean_folder_except(path=path, keep_only=keep_only_after)


@contextmanager
def change_directory(new_path):
    """
    Context manager to change the current working directory to a specified path,
    and then change it back to the original directory upon exiting the context.
    """
    # Store the current working directory
    original_path = os.getcwd()

    try:
        # Change to the specified directory
        os.chdir(new_path)
        yield  # Allow code inside the 'with' block to run

    finally:
        # Change back to the original directory
        os.chdir(original_path)


def clean_folder_except(path, keep_only):
    """
    Removes all files and folders in the specified folder except for those whose names are in the keep_only list.

    Parameters:
        path (str): The path to the folder to clean.
        keep_only (list): A list of file and folder names to keep.
    """
    # Iterate over all files and folders in the specified folder
    for item in os.listdir(path):
        item_path = os.path.join(path, item)

        # Check if the item should be kept
        if "model" in item and ".py" in item:
            continue
        elif "input" in item and ".yaml" in item:
            continue
        elif item not in keep_only:
            # Check if the item is a file or a symbolic link
            if (
                os.path.isfile(item_path)
                or os.path.islink(item_path)
                and not item_path.startswith("input")
            ):
                # Remove the file
                os.remove(item_path)
            elif os.path.isdir(item_path):
                # Remove the folder and its contents
                shutil.rmtree(item_path)


def _compare_metrics(
    actual_metrics: pd.DataFrame,
    reference_metrics: dict,
    label: str,
    rel=None,
    abs=None,
):
    """
    Compares the last row of a metrics DataFrame with reference values.

    Args:
        actual_metrics: DataFrame containing metrics from the run.
        reference_metrics: Dictionary of expected metric values.
        label: A string label (e.g., "TRAIN") for error messages.
    """
    assert not actual_metrics.empty, f"{label} metrics DataFrame is empty."
    last_row = actual_metrics.iloc[-1]
    print(f"Comparing final {label} metrics:", last_row.to_dict())

    for key, expected_value in reference_metrics.items():
        # Skip non-deterministic values like time
        if "time" in key.lower():
            continue

        actual_value = last_row.get(key)
        assert actual_value is not None, f"Metric '{key}' not found in {label} results."

        # Using pytest.approx is idiomatic for floating-point comparisons
        assert actual_value == pytest.approx(
            expected_value, rel=rel, abs=abs
        ), f"{label} metric '{key}' mismatch: Got {actual_value}, expected {expected_value}"
