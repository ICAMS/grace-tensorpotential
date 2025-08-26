import os

import pandas as pd
import pytest

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

import logging

LOG_FMT = "%(asctime)s %(levelname).1s - %(message)s"
logging.basicConfig(level=logging.INFO, format=LOG_FMT, datefmt="%Y/%m/%d %H:%M:%S")
log = logging.getLogger()


import numpy as np
import shutil

from contextlib import contextmanager
from pathlib import Path

from tensorpotential.cli.gracemaker import main
from tensorpotential.utils import load_metrics


prefix = Path(__file__).parent.resolve()

keep_only = [
    "input.yaml",
    "model.py",
]
keep_only_after = keep_only + ["log.txt"]


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
):
    print(f"Current folder: {os.getcwd()}")
    many_runs = many_runs or [[input]]
    ref_n_epochs = ref_n_epochs + len(many_runs) * ref_n_init_epoch
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


def test_ETHANOL_LINEAR():
    ref_n_epochs = 2

    train_ref_metrics = {
        "total_loss/train": 24.340401023024047,
        "mae/depa": 4.3826865636208945,
        "mae/de": 39.44417907258804,
        "rmse/depa": 4.384674723491186,
        "rmse/de": 39.46207251142067,
        "mae/f_comp": 0.7558215777663255,
        "rmse/f_comp": 1.0384177947662916,
        "loss_component/energy/train": 1.906359271694059,
        "loss_component/forces/train": 0.5276808306083456,
        "total_time/train/per_atom": 0.0007002009360056713,
        "epoch": 2.0,
    }

    test_ref_metrics = {
        "total_loss/test": 21.672018899655374,
        "loss_component/energy/test": 0.7959055278022663,
        "loss_component/forces/test": 0.28769541718050246,
        "mae/depa": 3.9897482090517142,
        "mae/de": 35.907733881465425,
        "rmse/depa": 3.989750688457278,
        "rmse/de": 35.9077561961155,
        "mae/f_comp": 0.8495929478453962,
        "rmse/f_comp": 1.0727449224871726,
        "total_time/test/per_atom": 0.0005316375496072902,
        "epoch": 2.0,
    }

    general_integration_test(
        "Ethanol-LINEAR",
        train_ref_metrics=train_ref_metrics,
        test_ref_metrics=test_ref_metrics,
        ref_n_epochs=ref_n_epochs,
    )


def test_MoNbTaW_LINEAR_ef_switch():
    ref_n_epochs = 2

    train_ref_metrics = {
        "total_loss/train": 683.3989801826644,
        "mae/depa": 11.56719958154282,
        "mae/de": 209.68804077030117,
        "rmse/depa": 11.593332837149774,
        "rmse/de": 252.23402939209956,
        "mae/f_comp": 0.16319468020316,
        "rmse/f_comp": 0.4483997907015461,
        "loss_component/energy/train": 67.20268313646761,
        "loss_component/forces/train": 1.1372148817988201,
        "total_time/train/per_atom": 0.0011054716495614823,
        "epoch": 2.0,
        "step": 4.0,
        "lr_epoch_begin": 0.0010000000474974513,
        "lr_epoch_end": 0.0010000000474974513,
    }

    test_ref_metrics = {
        "total_loss/test": 1317.2043610113417,
        "loss_component/energy/test": 65.18023809607078,
        "loss_component/forces/test": 0.6799799544963073,
        "mae/depa": 11.396925857337175,
        "mae/de": 195.70966360574266,
        "rmse/depa": 11.41755123448726,
        "rmse/de": 226.52070825349068,
        "mae/f_comp": 0.13688989876417798,
        "rmse/f_comp": 0.3154061568813392,
        "total_time/test/per_atom": 0.00044925431446994054,
        "epoch": 2.0,
        "step": 4.0,
        "lr_epoch_begin": 0.0010000000474974513,
        "lr_epoch_end": 0.0010000000474974513,
    }

    general_integration_test(
        "MoNbTaW-LINEAR",
        train_ref_metrics=train_ref_metrics,
        test_ref_metrics=test_ref_metrics,
        ref_n_epochs=ref_n_epochs,
    )


def test_MoNbTaW_LINEAR_energy_weighting():
    ref_n_epochs = 2

    train_ref_metrics = {
        "total_loss/train": 133.30666161821685,
        "mae/depa": 11.557162430995922,
        "mae/de": 209.5240052050945,
        "rmse/depa": 11.58323391825446,
        "rmse/de": 252.03842964495828,
        "mae/f_comp": 0.1641053366076806,
        "rmse/f_comp": 0.45440274077124043,
        "loss_component/energy/train": 13.227775354328013,
        "loss_component/forces/train": 0.10289080749367076,
        "total_time/train/per_atom": 0.0012785117939094325,
        "epoch": 2.0,
        "per_group_metrics.low.mae/depa": 11.55716243099592,
        "per_group_metrics.low.mae/de": 209.5240052050945,
        "per_group_metrics.low.rmse/depa": 11.58323391825446,
        "per_group_metrics.low.rmse/de": 252.03842964495828,
        "per_group_metrics.low.mae/f_comp": 0.16410533660768054,
        "per_group_metrics.low.rmse/f_comp": 0.45440274077124043,
        "per_group_metrics.low.num_struct": 20.0,
        "per_group_metrics.low.num_atoms": 368.0,
    }

    test_ref_metrics = {
        "total_loss/test": 128.40370386133847,
        "loss_component/energy/test": 6.371613316175242,
        "loss_component/forces/test": 0.04857187689168158,
        "mae/depa": 11.376474354818964,
        "mae/de": 195.35316780329532,
        "rmse/depa": 11.397192187552223,
        "rmse/de": 226.07323907854402,
        "mae/f_comp": 0.1359766320137802,
        "rmse/f_comp": 0.31440518442077775,
        "total_time/test/per_atom": 0.0004796265857294202,
        "epoch": 2.0,
        "per_group_metrics.low.mae/depa": 11.376474354818964,
        "per_group_metrics.low.mae/de": 195.35316780329532,
        "per_group_metrics.low.rmse/depa": 11.397192187552223,
        "per_group_metrics.low.rmse/de": 226.07323907854402,
        "per_group_metrics.low.mae/f_comp": 0.1359766320137802,
        "per_group_metrics.low.rmse/f_comp": 0.31440518442077775,
        "per_group_metrics.low.num_struct": 20.0,
        "per_group_metrics.low.num_atoms": 340.0,
    }

    general_integration_test(
        "MoNbTaW-LINEAR",
        train_ref_metrics=train_ref_metrics,
        test_ref_metrics=test_ref_metrics,
        input="input_wei.yaml",
        ref_n_epochs=ref_n_epochs,
    )


def test_MoNbTaW_LINEAR_huber():
    ref_n_epochs = 2

    train_ref_metrics = {
        "total_loss/train": 1.3155029191603829,
        "mae/depa": 11.574667595078688,
        "mae/de": 209.8024897981416,
        "rmse/depa": 11.6006960038097,
        "rmse/de": 252.33652574468326,
        "mae/f_comp": 0.16153416845817606,
        "rmse/f_comp": 0.4473390701757568,
        "loss_component/energy/train": 0.11569667595078689,
        "loss_component/forces/train": 0.01585361596525141,
        "total_time/train/per_atom": 0.0011152562036168883,
        "epoch": 2.0,
        "step": 4.0,
        "lr_epoch_begin": 0.009999999776482582,
        "lr_epoch_end": 0.009999999776482582,
    }

    test_ref_metrics = {
        "total_loss/test": 1.2733759861539238,
        "loss_component/energy/test": 0.056996698367154555,
        "loss_component/forces/test": 0.006672100940541634,
        "mae/depa": 11.40433967343091,
        "mae/de": 195.83610518167228,
        "rmse/depa": 11.424754597538652,
        "rmse/de": 226.71162836283924,
        "mae/f_comp": 0.13650621314228473,
        "rmse/f_comp": 0.31622108410922367,
        "total_time/test/per_atom": 0.0004379787913742749,
        "epoch": 2.0,
        "step": 4.0,
        "lr_epoch_begin": 0.009999999776482582,
        "lr_epoch_end": 0.009999999776482582,
    }

    general_integration_test(
        "MoNbTaW-LINEAR",
        train_ref_metrics=train_ref_metrics,
        test_ref_metrics=test_ref_metrics,
        ref_n_epochs=ref_n_epochs,
        input="input_huber.yaml",
    )


def test_MoNbTaW_LINEAR_LBFGS():
    ref_n_epochs = 2

    train_ref_metrics = {
        "total_loss/train": 75.62424761902926,
        "loss_component/energy/train": 6.608621349392428,
        "loss_component/forces/train": 0.9538034125104979,
        "mae/depa": 11.468743261993646,
        "mae/de": 207.74642673826352,
        "rmse/depa": 11.496626765614709,
        "rmse/de": 249.64602963084369,
        "mae/f_comp": 0.06832822158571318,
        "rmse/f_comp": 0.13402069610767856,
        "total_time/train/per_atom": 0.0014904249198835992,
        "epoch": 2.0,
    }

    test_ref_metrics = {
        "total_loss/test": 199.5860301961596,
        "loss_component/energy/test": 6.396529596950224,
        "loss_component/forces/test": 3.582771912857758,
        "mae/depa": 11.287727997339633,
        "mae/de": 193.90614776504592,
        "rmse/depa": 11.31064065113044,
        "rmse/de": 224.62784010688543,
        "mae/f_comp": 0.1468529630102565,
        "rmse/f_comp": 0.3229294479836321,
        "total_time/test/per_atom": 0.0006448650581981329,
        "epoch": 2.0,
    }

    general_integration_test(
        "MoNbTaW-LINEAR",
        train_ref_metrics=train_ref_metrics,
        test_ref_metrics=test_ref_metrics,
        ref_n_epochs=ref_n_epochs,
        input="input_lbfgs.yaml",
    )


def test_MoNbTaW_FS_ef_switch():

    train_ref_metrics = {
        "total_loss/train": 1293.983502919536,
        "mae/depa": 11.297882410206364,
        "mae/de": 204.61222077841956,
        "rmse/depa": 11.32841264708162,
        "rmse/de": 246.13498847051895,
        "mae/f_comp": 0.12389143151079433,
        "rmse/f_comp": 0.32640728996068924,
        "loss_component/energy/train": 128.3329331025588,
        "loss_component/forces/train": 1.0654171893948143,
        "total_time/train/per_atom": 0.0015550825736009638,
        "epoch": 2.0,
        "step": 4.0,
        "lr_epoch_begin": 0.10000000149011612,
        "lr_epoch_end": 0.10000000149011612,
    }

    test_ref_metrics = {
        "total_loss/test": 1152.6070689291557,
        "loss_component/energy/test": 56.600688257212994,
        "loss_component/forces/test": 1.0296651892447926,
        "mae/depa": 10.614938049711775,
        "mae/de": 181.7574466441644,
        "rmse/depa": 10.639613550990749,
        "rmse/de": 209.56715461694537,
        "mae/f_comp": 0.2610332123351273,
        "rmse/f_comp": 0.45379845509758904,
        "total_time/test/per_atom": 0.0006433775853913496,
        "epoch": 2.0,
        "step": 4.0,
        "lr_epoch_begin": 0.10000000149011612,
        "lr_epoch_end": 0.10000000149011612,
    }

    general_integration_test(
        "MoNbTaW-FS",
        train_ref_metrics=train_ref_metrics,
        test_ref_metrics=test_ref_metrics,
        ref_n_epochs=2,
        input="input_3fs.yaml",
    )


def test_MoNbTaW_FS_HEA25():

    train_ref_metrics = {
        "total_loss/train": 573.1646492542718,
        "mae/depa": 10.65400206784554,
        "mae/de": 195.5324177269531,
        "rmse/depa": 10.703660087778598,
        "rmse/de": 237.58126023213475,
        "mae/f_comp": 0.18523098356128112,
        "rmse/f_comp": 0.4018413124288532,
        "loss_component/energy/train": 57.28416963735228,
        "loss_component/forces/train": 0.03229528807490863,
        "total_time/train/per_atom": 0.0010472961357268302,
        "epoch": 2.0,
        "step": 4.0,
        "lr_epoch_begin": 0.10000000149011612,
        "lr_epoch_end": 0.10000000149011612,
    }

    test_ref_metrics = {
        "total_loss/test": 416.5112840084474,
        "loss_component/energy/test": 20.80543232070299,
        "loss_component/forces/test": 0.02013187971937931,
        "mae/depa": 8.590189267561094,
        "mae/de": 160.95132491079954,
        "rmse/depa": 9.122594438141594,
        "rmse/de": 197.00596558789414,
        "mae/f_comp": 0.2778082607624295,
        "rmse/f_comp": 0.44868563292554076,
        "total_time/test/per_atom": 0.000425660255474641,
        "epoch": 2.0,
        "step": 4.0,
        "lr_epoch_begin": 0.10000000149011612,
        "lr_epoch_end": 0.10000000149011612,
    }

    general_integration_test(
        "MoNbTaW-FS",
        train_ref_metrics=train_ref_metrics,
        test_ref_metrics=test_ref_metrics,
        ref_n_epochs=2,
        input="input_hea25.yaml",
    )


# def test_MoNbTaW_MLP_switch_ef_lr_reduction_early_stop():
#
#     train_ref_metrics = {
#         "total_loss/train": 5578.826576320846,
#         "mae/depa": 7.187415285790789,
#         "mae/de": 145.97724052434677,
#         "rmse/depa": 7.469021288311188,
#         "rmse/de": 197.49932605119832,
#         "mae/f_comp": 0.22217471247868814,
#         "rmse/f_comp": 0.44573063196695795,
#         "loss_component/energy/train": 557.8627900524573,
#         "loss_component/forces/train": 0.019867579627366377,
#         "total_time/train/per_atom": 0.0025716584864114893,
#         "epoch": 6.0,
#     }
#
#     test_ref_metrics = {
#         "total_loss/test": 4585.197685613167,
#         "loss_component/energy/test": 229.18536522279803,
#         "loss_component/forces/test": 0.07451905786035148,
#         "mae/depa": 5.78591830236815,
#         "mae/de": 99.04727185679245,
#         "rmse/depa": 6.7703081942079715,
#         "rmse/de": 140.0968228100447,
#         "mae/f_comp": 0.4299139237404923,
#         "rmse/f_comp": 1.2208116796652257,
#         "total_time/test/per_atom": 0.00044671241467928184,
#         "epoch": 6.0,
#     }
#
#     general_integration_test(
#         "MoNbTaW-MLP",
#         train_ref_metrics=train_ref_metrics,
#         test_ref_metrics=test_ref_metrics,
#         ref_n_epochs=6,
#     )


def test_MoNbTaW_GRACE_1L():
    train_ref_metrics = {
        "total_loss/train": 341.0286832126184,
        "mae/depa": 7.517684210036725,
        "mae/de": 127.68033908577445,
        "rmse/depa": 8.25162165018171,
        "rmse/de": 150.9496955355586,
        "mae/f_comp": 0.2638054624297858,
        "rmse/f_comp": 0.5396220547201438,
        "loss_component/energy/train": 34.04462992887376,
        "loss_component/forces/train": 0.05823839238807797,
        "total_time/train/per_atom": 0.0007821729807841147,
        "epoch": 3.0,
        "step": 6.0,
        "lr_epoch_begin": 0.10000000149011612,
        "lr_epoch_end": 0.10000000149011612,
    }

    test_ref_metrics = {
        "total_loss/test": 478.79950145966114,
        "loss_component/energy/test": 23.809925080874116,
        "loss_component/forces/test": 0.13004999210894358,
        "mae/depa": 8.499836783501985,
        "mae/de": 172.1127518570671,
        "rmse/depa": 9.759082965294253,
        "rmse/de": 303.5295455214987,
        "mae/f_comp": 0.46661525254971137,
        "rmse/f_comp": 1.1403946339269737,
        "total_time/test/per_atom": 8.065275341162787e-05,
        "epoch": 3.0,
        "step": 6.0,
        "lr_epoch_begin": 0.10000000149011612,
        "lr_epoch_end": 0.10000000149011612,
    }

    general_integration_test(
        "MoNbTaW-GRACE",
        train_ref_metrics=train_ref_metrics,
        test_ref_metrics=test_ref_metrics,
        ref_n_epochs=3,
        input="input_1L.yaml",
    )


def test_MoNbTaW_GRACE_1L_bond_cutoff_and_zbl():

    train_ref_metrics = {
        "total_loss/train": 582.784971102689,
        "mae/depa": 10.684758838715997,
        "mae/de": 190.10366314752955,
        "rmse/depa": 10.769700535693952,
        "rmse/de": 225.633006621588,
        "mae/f_comp": 0.6135412497668952,
        "rmse/f_comp": 1.194303763716777,
        "loss_component/energy/train": 57.99322481426329,
        "loss_component/forces/train": 0.2852722960056119,
        "total_time/train/per_atom": 0.0007587777502561474,
        "epoch": 3.0,
        "step": 6.0,
        "lr_epoch_begin": 0.10000000149011612,
        "lr_epoch_end": 0.10000000149011612,
    }

    test_ref_metrics = {
        "total_loss/test": 358.53571030506197,
        "loss_component/energy/test": 17.797274094721505,
        "loss_component/forces/test": 0.12951142053159348,
        "mae/depa": 7.783049905839991,
        "mae/de": 110.6148845068985,
        "rmse/depa": 8.437363117638473,
        "rmse/de": 124.2492789463465,
        "mae/f_comp": 0.6587658754923219,
        "rmse/f_comp": 1.1380308455028514,
        "total_time/test/per_atom": 6.386162918608854e-05,
        "epoch": 3.0,
        "step": 6.0,
        "lr_epoch_begin": 0.10000000149011612,
        "lr_epoch_end": 0.10000000149011612,
    }

    general_integration_test(
        "MoNbTaW-GRACE",
        train_ref_metrics=train_ref_metrics,
        test_ref_metrics=test_ref_metrics,
        ref_n_epochs=3,
        input="input_1L_bond_cutoff.yaml",
    )


def test_MoNbTaW_GRACE_2L():

    train_ref_metrics = {
        "total_loss/train": 807.7923433432779,
        "mae/depa": 9.199613514676637,
        "mae/de": 142.96315972748394,
        "rmse/depa": 12.70357387456406,
        "rmse/de": 180.22139648723103,
        "mae/f_comp": 0.32015314664097694,
        "rmse/f_comp": 0.6664823368796916,
        "loss_component/energy/train": 80.69039459325327,
        "loss_component/forces/train": 0.08883974107452296,
        "total_time/train/per_atom": 0.0011897105217465887,
        "epoch": 3.0,
        "step": 6.0,
        "lr_epoch_begin": 0.10000000149011612,
        "lr_epoch_end": 0.10000000149011612,
    }

    test_ref_metrics = {
        "total_loss/test": 104.90613347722318,
        "loss_component/energy/test": 5.227199164965625,
        "loss_component/forces/test": 0.018107508895534797,
        "mae/depa": 3.951369647030762,
        "mae/de": 69.06011550836573,
        "rmse/depa": 4.5726137667490026,
        "rmse/de": 87.53365579346236,
        "mae/f_comp": 0.19262850009155408,
        "rmse/f_comp": 0.42552918696059844,
        "total_time/test/per_atom": 0.00011232122025616905,
        "epoch": 3.0,
        "step": 6.0,
        "lr_epoch_begin": 0.10000000149011612,
        "lr_epoch_end": 0.10000000149011612,
    }

    general_integration_test(
        "MoNbTaW-GRACE",
        train_ref_metrics=train_ref_metrics,
        test_ref_metrics=test_ref_metrics,
        ref_n_epochs=3,
        input="input_2L.yaml",
        seed=42,
    )


# def test_MoNbTaW_GRACE_2L_MP():
#
#     train_ref_metrics = {
#         "total_loss/train": 344.5705114254165,
#         "mae/depa": 7.482369866313792,
#         "mae/de": 113.6795078606464,
#         "rmse/depa": 8.291080354988068,
#         "rmse/de": 129.01724319410621,
#         "mae/f_comp": 0.35231065536380285,
#         "rmse/f_comp": 0.6559131653927708,
#         "loss_component/energy/train": 34.37100672643454,
#         "loss_component/forces/train": 0.08604441610711286,
#         "total_time/train/per_atom": 0.0020787830188688213,
#         "epoch": 3.0,
#     }
#
#     test_ref_metrics = {
#         "total_loss/test": 199.2727212695371,
#         "loss_component/energy/test": 9.909817972891886,
#         "loss_component/forces/test": 0.053818090584969,
#         "mae/depa": 4.7623331357791585,
#         "mae/de": 78.17992468551805,
#         "rmse/depa": 6.295972672396821,
#         "rmse/de": 116.2505975669391,
#         "mae/f_comp": 0.3062874000505317,
#         "rmse/f_comp": 0.73360814189163,
#         "total_time/test/per_atom": 0.0001875245519091978,
#         "epoch": 3.0,
#     }
#
#     general_integration_test(
#         "MoNbTaW-GRACE",
#         train_ref_metrics=train_ref_metrics,
#         test_ref_metrics=test_ref_metrics,
#         ref_n_epochs=3,
#         input="input_2L_MP.yaml",
#     )


@pytest.mark.skip(reason="skip custom model")
def test_MoNbTaW_CUSTOM_switch_ef_ema():

    train_ref_metrics = {
        "total_loss/train": 2.1716064211221107,
        "mae/depa": 0.10643320385933963,
        "mae/de": 1.5048795721041994,
        "rmse/depa": 0.14552660157733377,
        "rmse/de": 2.1043438739612728,
        "mae/f_comp": 0.15971624941433762,
        "rmse/f_comp": 0.4417355799348502,
        "loss_component/energy/train": 0.1058899588332402,
        "loss_component/forces/train": 0.11127068327897087,
        "total_time/train/per_atom": 0.001399472983473021,
        "epoch": 2.0,
    }

    test_ref_metrics = {
        "total_loss/test": 4.663940382682648,
        "loss_component/energy/test": 0.16472797742800233,
        "loss_component/forces/test": 0.06846904170613011,
        "mae/depa": 0.11826169834236133,
        "mae/de": 2.6328969175460974,
        "rmse/depa": 0.18150921597979663,
        "rmse/de": 4.866076088767953,
        "mae/f_comp": 0.13372188836891163,
        "rmse/f_comp": 0.3139186430622856,
        "total_time/test/per_atom": 0.00015732100000604988,
        "epoch": 2.0,
    }

    general_integration_test(
        "MoNbTaW-CUSTOM",
        train_ref_metrics=train_ref_metrics,
        test_ref_metrics=test_ref_metrics,
        ref_n_epochs=2,
    )


@pytest.mark.skip(reason="skip custom model")
def test_MoNbTaW_CUSTOM_mlp_emb():

    train_ref_metrics = {
        "total_loss/train": 12526.25160185599,
        "mae/depa": 10.948964532181693,
        "mae/de": 201.9527117081654,
        "rmse/depa": 11.191204806568912,
        "rmse/de": 249.14284284551678,
        "mae/f_comp": 0.15880805468319745,
        "rmse/f_comp": 0.4410328331173345,
        "loss_component/energy/train": 1252.4306502257116,
        "loss_component/forces/train": 0.19450995988750266,
        "total_time/train/per_atom": 0.0011867424131512803,
        "epoch": 2.0,
    }

    test_ref_metrics = {
        "total_loss/test": 13786.440637133539,
        "loss_component/energy/test": 689.2716697031861,
        "loss_component/forces/test": 0.0503621534908115,
        "mae/depa": 11.690077081686086,
        "mae/de": 203.29138883030566,
        "rmse/depa": 11.741138528296021,
        "rmse/de": 239.52060799979654,
        "mae/f_comp": 0.13613908328714555,
        "rmse/f_comp": 0.31737092964167807,
        "total_time/test/per_atom": 0.00010057043777230908,
        "epoch": 2.0,
    }

    general_integration_test(
        "MoNbTaW-CUSTOM",
        input="input_mlp_emb.yaml",
        train_ref_metrics=train_ref_metrics,
        test_ref_metrics=test_ref_metrics,
        ref_n_epochs=2,
    )


def test_MoNbTaW_FS_restart():

    train_ref_metrics = {
        "total_loss/train": 2346.790071871758,
        "mae/depa": 10.270012425829322,
        "mae/de": 185.5614589512267,
        "rmse/depa": 10.329625074465616,
        "rmse/de": 221.97591231688975,
        "mae/f_comp": 1.1315222644031013,
        "rmse/f_comp": 3.57739923698973,
        "loss_component/energy/train": 106.70115417902878,
        "loss_component/forces/train": 127.97785300814701,
        "total_time/train/per_atom": 0.0010155451031017076,
        "epoch": 2.0,
        "step": 4.0,
        "lr_epoch_begin": 0.10000000149011612,
        "lr_epoch_end": 0.10000000149011612,
    }

    test_ref_metrics = {
        "total_loss/test": 1042.9742465019149,
        "loss_component/energy/test": 50.82680173489472,
        "loss_component/forces/test": 1.3219105902010293,
        "mae/depa": 10.011840519961364,
        "mae/de": 174.16634196204438,
        "rmse/depa": 10.082341170075004,
        "rmse/de": 204.49113812936898,
        "mae/f_comp": 0.3217373889104534,
        "rmse/f_comp": 0.5141810168026489,
        "total_time/test/per_atom": 0.00040583938551957114,
        "epoch": 2.0,
        "step": 4.0,
        "lr_epoch_begin": 0.10000000149011612,
        "lr_epoch_end": 0.10000000149011612,
    }

    general_integration_test(
        "MoNbTaW-FS",
        train_ref_metrics=train_ref_metrics,
        test_ref_metrics=test_ref_metrics,
        ref_n_epochs=6,
        many_runs=[
            ["input.yaml"],
            ["input_lbfgs.yaml", "-r"],
            ["input.yaml", "-rs", ".epoch_2", "--reset-epoch-and-step"],
        ],
    )


def test_MoNbTaW_LINEAR_virial():

    train_ref_metrics = {
        "total_loss/train": 1830.643762287907,
        "mae/depa": 11.55763491386988,
        "mae/de": 209.46806504914738,
        "rmse/depa": 11.584519644787989,
        "rmse/de": 251.97123684880543,
        "mae/f_comp": 0.1648631385886434,
        "rmse/f_comp": 0.4552404624467939,
        "mae/virial": 4.0931044530178555,
        "rmse/virial": 12.623373800428038,
        "mae/stress": 0.017058936286511377,
        "rmse/stress": 0.05078442559360883,
        "loss_component/energy/train": 26.256257261555138,
        "loss_component/forces/train": 0.2241134588000551,
        "loss_component/virial/train": 339.64838173722626,
        "total_time/train/per_atom": 0.0014206461972840455,
        "epoch": 2.0,
        "per_group_metrics.low.mae/depa": 11.55763491386988,
        "per_group_metrics.low.mae/de": 209.46806504914738,
        "per_group_metrics.low.rmse/depa": 11.584519644787987,
        "per_group_metrics.low.rmse/de": 251.97123684880546,
        "per_group_metrics.low.mae/f_comp": 0.1648631385886434,
        "per_group_metrics.low.rmse/f_comp": 0.45524046244679395,
        "per_group_metrics.low.mae/virial": 4.093104453017857,
        "per_group_metrics.low.rmse/virial": 12.623373800428036,
        "per_group_metrics.low.mae/stress": 0.01705893628651138,
        "per_group_metrics.low.rmse/stress": 0.05078442559360883,
        "per_group_metrics.low.num_struct": 20.0,
        "per_group_metrics.low.num_atoms": 368.0,
    }

    test_ref_metrics = {
        "total_loss/test": 7177.270354734258,
        "loss_component/energy/test": 25.175054536378177,
        "loss_component/forces/test": 0.17849864547448788,
        "loss_component/virial/test": 1410.100517764999,
        "mae/depa": 11.373338418146414,
        "mae/de": 195.320264345046,
        "rmse/depa": 11.394568984039024,
        "rmse/de": 226.0404708312383,
        "mae/f_comp": 0.13505513536019015,
        "rmse/f_comp": 0.3132069431867746,
        "mae/virial": 6.039900574522764,
        "rmse/virial": 24.784262102816122,
        "mae/stress": 0.012491686340599191,
        "rmse/stress": 0.03490745036023367,
        "total_time/test/per_atom": 0.00056493255063234,
        "epoch": 2.0,
        "per_group_metrics.low.mae/depa": 11.373338418146414,
        "per_group_metrics.low.mae/de": 195.32026434504598,
        "per_group_metrics.low.rmse/depa": 11.394568984039024,
        "per_group_metrics.low.rmse/de": 226.04047083123834,
        "per_group_metrics.low.mae/f_comp": 0.13505513536019012,
        "per_group_metrics.low.rmse/f_comp": 0.3132069431867746,
        "per_group_metrics.low.mae/virial": 6.039900574522764,
        "per_group_metrics.low.rmse/virial": 24.784262102816122,
        "per_group_metrics.low.mae/stress": 0.01249168634059919,
        "per_group_metrics.low.rmse/stress": 0.03490745036023366,
        "per_group_metrics.low.num_struct": 20.0,
        "per_group_metrics.low.num_atoms": 340.0,
    }

    general_integration_test(
        "MoNbTaW-LINEAR",
        train_ref_metrics=train_ref_metrics,
        test_ref_metrics=test_ref_metrics,
        ref_n_epochs=2,
        many_runs=[
            ["input_virial.yaml"],
        ],
    )


def test_MoNbTaW_LINEAR_stress():

    train_ref_metrics = {
        "total_loss/train": 131.93521304085127,
        "mae/depa": 11.536582895984672,
        "mae/de": 209.19659735119848,
        "rmse/depa": 11.563043014548306,
        "rmse/de": 251.71664273976103,
        "mae/f_comp": 0.16371773236009757,
        "rmse/f_comp": 0.45436549775862894,
        "mae/virial": 4.126886828405503,
        "rmse/virial": 12.874311004094798,
        "mae/stress": 0.017073288279429676,
        "rmse/stress": 0.05099524889506432,
        "loss_component/energy/train": 26.170549398962002,
        "loss_component/forces/train": 0.21535180517154817,
        "loss_component/stress/train": 0.001141404036700951,
        "total_time/train/per_atom": 0.0023203688855890346,
        "epoch": 2.0,
        "per_group_metrics.low.mae/depa": 11.536582895984672,
        "per_group_metrics.low.mae/de": 209.19659735119848,
        "per_group_metrics.low.rmse/depa": 11.563043014548308,
        "per_group_metrics.low.rmse/de": 251.71664273976106,
        "per_group_metrics.low.mae/f_comp": 0.16371773236009754,
        "per_group_metrics.low.rmse/f_comp": 0.45436549775862894,
        "per_group_metrics.low.mae/virial": 4.126886828405503,
        "per_group_metrics.low.rmse/virial": 12.874311004094798,
        "per_group_metrics.low.mae/stress": 0.017073288279429673,
        "per_group_metrics.low.rmse/stress": 0.05099524889506432,
        "per_group_metrics.low.num_struct": 20.0,
        "per_group_metrics.low.num_atoms": 368.0,
    }

    test_ref_metrics = {
        "total_loss/test": 126.23292522455105,
        "loss_component/energy/test": 25.063209317657638,
        "loss_component/forces/test": 0.18146276499826341,
        "loss_component/stress/test": 0.001912962254316595,
        "mae/depa": 11.347749233642999,
        "mae/de": 194.85961421951163,
        "rmse/depa": 11.369175656315079,
        "rmse/de": 225.41038839533374,
        "mae/f_comp": 0.13574302307119399,
        "rmse/f_comp": 0.313136865849777,
        "mae/virial": 6.093159174800654,
        "rmse/virial": 24.90997902167582,
        "mae/stress": 0.012660762075161288,
        "rmse/stress": 0.035037390675468946,
        "total_time/test/per_atom": 0.0009524628643275184,
        "epoch": 2.0,
        "per_group_metrics.low.mae/depa": 11.347749233643,
        "per_group_metrics.low.mae/de": 194.85961421951166,
        "per_group_metrics.low.rmse/depa": 11.369175656315079,
        "per_group_metrics.low.rmse/de": 225.41038839533377,
        "per_group_metrics.low.mae/f_comp": 0.13574302307119399,
        "per_group_metrics.low.rmse/f_comp": 0.313136865849777,
        "per_group_metrics.low.mae/virial": 6.093159174800654,
        "per_group_metrics.low.rmse/virial": 24.909979021675827,
        "per_group_metrics.low.mae/stress": 0.012660762075161286,
        "per_group_metrics.low.rmse/stress": 0.035037390675468946,
        "per_group_metrics.low.num_struct": 20.0,
        "per_group_metrics.low.num_atoms": 340.0,
    }

    general_integration_test(
        "MoNbTaW-LINEAR",
        train_ref_metrics=train_ref_metrics,
        test_ref_metrics=test_ref_metrics,
        ref_n_epochs=2,
        many_runs=[
            ["input_stress.yaml"],
        ],
    )


def test_MoNbTaW_LINEAR_f32():
    train_ref_metrics = {
        "total_loss/train": 75.80302429199219,
        "loss_component/energy/train": 6.577875137329102,
        "loss_component/forces/train": 1.0024268627166748,
        "mae/depa": 11.437673187255859,
        "mae/de": 206.737109375,
        "rmse/depa": 11.469851573459332,
        "rmse/de": 248.105471523705,
        "mae/f_comp": 0.06886006092679674,
        "rmse/f_comp": 0.1404136322360318,
        "total_time/train/per_atom": 0.0007808934770645978,
        "epoch": 2.0,
    }

    test_ref_metrics = {
        "total_loss/test": 189.30538940429688,
        "loss_component/energy/test": 6.353288173675537,
        "loss_component/forces/test": 3.1119816303253174,
        "mae/depa": 11.247532653808594,
        "mae/de": 193.2059814453125,
        "rmse/depa": 11.27234488814744,
        "rmse/de": 223.9928919073103,
        "mae/f_comp": 0.136882048962163,
        "rmse/f_comp": 0.2974778234689519,
        "total_time/test/per_atom": 0.0003391962238203953,
        "epoch": 2.0,
    }

    general_integration_test(
        "MoNbTaW-LINEAR",
        train_ref_metrics=train_ref_metrics,
        test_ref_metrics=test_ref_metrics,
        ref_n_epochs=2,
        many_runs=[
            ["input_f32.yaml"],
        ],
        rel=1e-5,
    )


def test_MoNbTaW_LINEAR_lr_reduce_on_plateau():
    train_ref_metrics = {
        "total_loss/train": 86388.96089855464,
        "mae/depa": 293.91999064125366,
        "mae/de": 587.8399812825073,
        "rmse/depa": 293.91999064125366,
        "rmse/de": 587.8399812825073,
        "mae/f_comp": 1.709743457922741e-14,
        "rmse/f_comp": 2.5537382093454438e-14,
        "loss_component/energy/train": 17277.792179710927,
        "loss_component/forces/train": 6.521578841870874e-28,
        "total_time/train/per_atom": 0.003750124989892356,
        "epoch": 5,
    }

    test_ref_metrics = {
        "total_loss/test": 1078.717976030194,
        "loss_component/energy/test": 34.13630846923014,
        "loss_component/forces/test": 19.799590332279568,
        "mae/depa": 18.419813604139947,
        "mae/de": 318.0004536704866,
        "rmse/depa": 23.75680161302725,
        "rmse/de": 417.32888026700016,
        "mae/f_comp": 1.9076265481110548,
        "rmse/f_comp": 6.903826426461479,
        "total_time/test/per_atom": 0.0008932093492380762,
        "epoch": 5,
    }

    general_integration_test(
        "MoNbTaW-LINEAR",
        train_ref_metrics=train_ref_metrics,
        test_ref_metrics=test_ref_metrics,
        ref_n_epochs=5,
        input="input_lr_reduce_on_plateau.yaml",
    )


def test_MoNbTaW_LINEAR_lr_reduce_on_plateau_new_api():
    train_ref_metrics = {
        "total_loss/train": 86388.96089855464,
        "mae/depa": 293.91999064125366,
        "mae/de": 587.8399812825073,
        "rmse/depa": 293.91999064125366,
        "rmse/de": 587.8399812825073,
        "mae/f_comp": 1.709743457922741e-14,
        "rmse/f_comp": 2.5537382093454438e-14,
        "loss_component/energy/train": 17277.792179710927,
        "loss_component/forces/train": 6.521578841870874e-28,
        "total_time/train/per_atom": 0.003750124989892356,
        "epoch": 5,
    }

    test_ref_metrics = {
        "total_loss/test": 1078.717976030194,
        "loss_component/energy/test": 34.13630846923014,
        "loss_component/forces/test": 19.799590332279568,
        "mae/depa": 18.419813604139947,
        "mae/de": 318.0004536704866,
        "rmse/depa": 23.75680161302725,
        "rmse/de": 417.32888026700016,
        "mae/f_comp": 1.9076265481110548,
        "rmse/f_comp": 6.903826426461479,
        "total_time/test/per_atom": 0.0008932093492380762,
        "epoch": 5,
    }

    general_integration_test(
        "MoNbTaW-LINEAR",
        train_ref_metrics=train_ref_metrics,
        test_ref_metrics=test_ref_metrics,
        ref_n_epochs=5,
        input="input_lr_reduce_on_plateau_new_api.yaml",
    )


def test_MoNbTaW_LINEAR_lr_exponential_decay():
    train_ref_metrics = {
        "total_loss/train": 200.99824377262027,
        "mae/depa": 14.177384941258394,
        "mae/de": 28.354769882516788,
        "rmse/depa": 14.177384941258394,
        "rmse/de": 28.354769882516788,
        "mae/f_comp": 6.649773324577761e-18,
        "rmse/f_comp": 1.2327526502681117e-17,
        "loss_component/energy/train": 40.199648754524056,
        "loss_component/forces/train": 1.5196790967430533e-34,
        "total_time/train/per_atom": 0.010084604498842964,
        "epoch": 5,
    }

    test_ref_metrics = {
        "total_loss/test": 130.34017323870668,
        "loss_component/energy/test": 6.480867512123688,
        "loss_component/forces/test": 0.03614114981164616,
        "mae/depa": 11.378284122938348,
        "mae/de": 204.70260740817477,
        "rmse/depa": 11.406753808776225,
        "rmse/de": 249.11856906998474,
        "mae/f_comp": 0.20539834351378242,
        "rmse/f_comp": 0.4236164600096448,
        "total_time/test/per_atom": 0.0009967192804823271,
        "epoch": 5,
    }

    general_integration_test(
        "MoNbTaW-LINEAR",
        train_ref_metrics=train_ref_metrics,
        test_ref_metrics=test_ref_metrics,
        ref_n_epochs=5,
        input="input_lr_exponential_decay.yaml",
    )


def test_MoNbTaW_LINEAR_lr_cosine_decay():
    train_ref_metrics = {
        "total_loss/train": 90.97337294606098,
        "mae/depa": 9.537996275217399,
        "mae/de": 19.075992550434798,
        "rmse/depa": 9.537996275217399,
        "rmse/de": 19.075992550434798,
        "mae/f_comp": 5.66676335485757e-17,
        "rmse/f_comp": 8.017350677993573e-17,
        "loss_component/energy/train": 18.194674589212195,
        "loss_component/forces/train": 6.4277911893924e-33,
        "total_time/train/per_atom": 0.0067948545001854654,
        "epoch": 5,
    }

    test_ref_metrics = {
        "total_loss/test": 120.97643764549225,
        "loss_component/energy/test": 5.849987864107047,
        "loss_component/forces/test": 0.1988340181675662,
        "mae/depa": 10.707245647052012,
        "mae/de": 194.0414184477568,
        "rmse/depa": 10.863451339096327,
        "rmse/de": 246.01613710092678,
        "mae/f_comp": 0.3786722186496125,
        "rmse/f_comp": 0.8020594648553879,
        "total_time/test/per_atom": 0.0010997455199588848,
        "epoch": 5,
    }

    general_integration_test(
        "MoNbTaW-LINEAR",
        train_ref_metrics=train_ref_metrics,
        test_ref_metrics=test_ref_metrics,
        ref_n_epochs=5,
        input="input_lr_cosine_decay.yaml",
    )


def test_MoNbTaW_LINEAR_lr_linear_decay():
    train_ref_metrics = {
        "total_loss/train": 4.161953987445907,
        "mae/depa": 2.040086759783982,
        "mae/de": 4.080173519567964,
        "rmse/depa": 2.040086759783982,
        "rmse/de": 4.080173519567964,
        "mae/f_comp": 4.440892098500626e-16,
        "rmse/f_comp": 5.733167046599011e-16,
        "loss_component/energy/train": 0.8323907974891814,
        "loss_component/forces/train": 3.286920438420883e-31,
        "total_time/train/per_atom": 0.006057833499653498,
        "epoch": 5,
    }

    test_ref_metrics = {
        "total_loss/test": 214.44839673423044,
        "loss_component/energy/test": 5.615171197827999,
        "loss_component/forces/test": 5.107248638883526,
        "mae/depa": 10.560843996749204,
        "mae/de": 192.61692842216544,
        "rmse/depa": 10.74257307074689,
        "rmse/de": 243.20408234470116,
        "mae/f_comp": 1.0145503408016143,
        "rmse/f_comp": 3.309602211253348,
        "total_time/test/per_atom": 0.000914615483371843,
        "epoch": 5,
    }

    general_integration_test(
        "MoNbTaW-LINEAR",
        train_ref_metrics=train_ref_metrics,
        test_ref_metrics=test_ref_metrics,
        ref_n_epochs=5,
        input="input_lr_linear_decay.yaml",
    )


def test_MoNbTaW_LINEAR_lr_linear_decay_no_warmup():
    train_ref_metrics = {
        "total_loss/train": 11.30631065652632,
        "mae/depa": 3.362485785327028,
        "mae/de": 6.724971570654056,
        "rmse/depa": 3.362485785327028,
        "rmse/de": 6.724971570654056,
        "mae/f_comp": 3.3306690738754696e-16,
        "rmse/f_comp": 5.623501550354407e-16,
        "loss_component/energy/train": 2.261262131305264,
        "loss_component/forces/train": 3.1623769686838413e-31,
        "total_time/train/per_atom": 0.004507999999987078,
        "epoch": 2,
    }

    test_ref_metrics = {
        "total_loss/test": 116.24667463281212,
        "loss_component/energy/test": 5.380090753095851,
        "loss_component/forces/test": 0.4322429785447546,
        "mae/depa": 10.366235830664959,
        "mae/de": 188.84269955885392,
        "rmse/depa": 10.48768658808825,
        "rmse/de": 239.54270092148286,
        "mae/f_comp": 0.46060556364186217,
        "rmse/f_comp": 1.1016298206069393,
        "total_time/test/per_atom": 0.0008858609090927435,
        "epoch": 2,
    }

    general_integration_test(
        "MoNbTaW-LINEAR",
        train_ref_metrics=train_ref_metrics,
        test_ref_metrics=test_ref_metrics,
        ref_n_epochs=2,
        input="input_lr_linear_decay_no_warmup.yaml",
    )
