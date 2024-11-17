import os

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


def general_integration_test(
    folder,
    train_ref_metrics,
    test_ref_metrics,
    ref_n_epochs=2,
    ref_n_init_epoch=0,
    input="input.yaml",
    many_runs=None,
):
    print("Current folder: {}".format(os.getcwd()))
    path = str(prefix / folder)
    clean_folder_except(path=path, keep_only=keep_only)
    if many_runs is not None:
        ref_n_epochs = ref_n_epochs + len(many_runs) * ref_n_init_epoch
    else:
        ref_n_epochs = ref_n_epochs + ref_n_init_epoch
    with change_directory(path):
        if many_runs is not None:
            for inp in many_runs:
                main(inp)
        else:
            main([input])  #
        train_metrics = load_metrics("seed/42/train_metrics.yaml")
        test_metrics = load_metrics("seed/42/test_metrics.yaml")
        assert len(train_metrics) == len(
            test_metrics
        ), "len(train_metrics) != len(test_metrics)"
        assert len(test_metrics) == ref_n_epochs, "len(test_metrics) != ref_n_epochs"

        last_train_row = train_metrics.iloc[-1]
        print("TRAIN metrics:", last_train_row.to_dict())

        last_test_row = test_metrics.iloc[-1]
        print("TEST metrics:", last_test_row.to_dict())

        # train
        for k, ref_value in train_ref_metrics.items():
            if "total_time" in k:
                continue
            assert np.allclose(
                last_train_row[k], ref_value
            ), f"TRAIN Value for {k} should be {ref_value}"

        # test

        for k, ref_value in test_ref_metrics.items():
            if "total_time" in k:
                continue
            assert np.allclose(
                last_test_row[k], ref_value
            ), f"TEST Value for {k} should be {ref_value}"

    clean_folder_except(path=path, keep_only=keep_only_after)


def test_ETHANOL_LINEAR():
    ref_n_epochs = 2

    train_ref_metrics = {
        "total_loss/train": 25.265191364288736,
        "mae/depa": 4.464919770682446,
        "mae/de": 40.18427793614201,
        "rmse/depa": 4.466748779061823,
        "rmse/de": 40.20073901155641,
        "mae/f_comp": 0.7726077623536036,
        "rmse/f_comp": 1.0561416876842278,
        "loss_component/energy/train": 1.979536992239534,
        "loss_component/forces/train": 0.5469821441893401,
        "total_time/train/per_atom": 0.0006190356669119663,
        "epoch": 2.0,
    }

    test_ref_metrics = {
        "total_loss/test": 23.382834652110525,
        "loss_component/energy/test": 0.8299310884320736,
        "loss_component/forces/test": 0.33921064417345276,
        "mae/depa": 4.0740389810382,
        "mae/de": 36.6663508293438,
        "rmse/depa": 4.07414061719051,
        "rmse/de": 36.66726555471459,
        "mae/f_comp": 0.8898176702379256,
        "rmse/f_comp": 1.164835858262361,
        "total_time/test/per_atom": 0.000429680777920617,
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
        "total_loss/train": 676.5535345351668,
        "mae/depa": 11.514666043805633,
        "mae/de": 208.81606926183014,
        "rmse/depa": 11.539496168383767,
        "rmse/de": 251.05475780866385,
        "mae/f_comp": 0.1574557170901338,
        "rmse/f_comp": 0.4361906026069163,
        "loss_component/energy/train": 66.57998591007183,
        "loss_component/forces/train": 1.0753675434448526,
        "total_time/train/per_atom": 0.0011192695737522824,
        "epoch": 2.0,
    }

    test_ref_metrics = {
        "total_loss/test": 1308.5432123562243,
        "loss_component/energy/test": 64.74804185448698,
        "loss_component/forces/test": 0.6791187633242272,
        "mae/depa": 11.358954947827602,
        "mae/de": 195.13494889262978,
        "rmse/depa": 11.379634603491185,
        "rmse/de": 226.02453572005913,
        "mae/f_comp": 0.13669363894036962,
        "rmse/f_comp": 0.3124677903583572,
        "total_time/test/per_atom": 0.00043221186736927314,
        "epoch": 2.0,
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
        "total_loss/train": 131.92549820495174,
        "mae/depa": 11.500843206991288,
        "mae/de": 208.60078625074135,
        "rmse/depa": 11.525652607221545,
        "rmse/de": 250.81076795626964,
        "mae/f_comp": 0.1581745444418258,
        "rmse/f_comp": 0.4413963221363568,
        "loss_component/energy/train": 13.102678231345273,
        "loss_component/forces/train": 0.08987158914990329,
        "total_time/train/per_atom": 0.0010975795549988422,
        "epoch": 2.0,
        "per_group_metrics.low.mae/depa": 11.50084320699129,
        "per_group_metrics.low.mae/de": 208.60078625074135,
        "per_group_metrics.low.rmse/depa": 11.525652607221545,
        "per_group_metrics.low.rmse/de": 250.81076795626964,
        "per_group_metrics.low.mae/f_comp": 0.1581745444418258,
        "per_group_metrics.low.rmse/f_comp": 0.4413963221363568,
        "per_group_metrics.low.num_struct": 20.0,
        "per_group_metrics.low.num_atoms": 368.0,
    }

    test_ref_metrics = {
        "total_loss/test": 127.290606117801,
        "loss_component/energy/test": 6.3207465149840605,
        "loss_component/forces/test": 0.04378379090598997,
        "mae/depa": 11.32960196128851,
        "mae/de": 194.6487701603897,
        "rmse/depa": 11.350400454123092,
        "rmse/de": 225.47818417409826,
        "mae/f_comp": 0.13429138018278663,
        "rmse/f_comp": 0.3129521331125388,
        "total_time/test/per_atom": 0.0004308301503496135,
        "epoch": 2.0,
        "per_group_metrics.low.mae/depa": 11.32960196128851,
        "per_group_metrics.low.mae/de": 194.6487701603897,
        "per_group_metrics.low.rmse/depa": 11.350400454123092,
        "per_group_metrics.low.rmse/de": 225.47818417409826,
        "per_group_metrics.low.mae/f_comp": 0.13429138018278663,
        "per_group_metrics.low.rmse/f_comp": 0.3129521331125388,
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
        "total_loss/train": 1.3058846587308803,
        "mae/depa": 11.52769274985471,
        "mae/de": 209.00529571327723,
        "rmse/depa": 11.552682005325867,
        "rmse/de": 251.25220749048702,
        "mae/f_comp": 0.1566515970965017,
        "rmse/f_comp": 0.4350452393081435,
        "loss_component/energy/train": 0.1152269274985471,
        "loss_component/forces/train": 0.015361538374540917,
        "total_time/train/per_atom": 0.001182957255640877,
        "epoch": 2.0,
    }

    test_ref_metrics = {
        "total_loss/test": 1.2677846073629697,
        "loss_component/energy/test": 0.056805414004316136,
        "loss_component/forces/test": 0.006583816363832352,
        "mae/depa": 11.366082800863229,
        "mae/de": 195.22335381987028,
        "rmse/depa": 11.386500743311327,
        "rmse/de": 226.10391622638866,
        "mae/f_comp": 0.1347732880447214,
        "rmse/f_comp": 0.3130614800727851,
        "total_time/test/per_atom": 0.0004460171378655907,
        "epoch": 2.0,
    }

    general_integration_test(
        "MoNbTaW-LINEAR",
        train_ref_metrics=train_ref_metrics,
        test_ref_metrics=test_ref_metrics,
        ref_n_epochs=ref_n_epochs,
        input="input_huber.yaml",
    )


def test_MoNbTaW_LINEAR_L2_reg():
    ref_n_epochs = 2

    train_ref_metrics = {
        "total_loss/train": 1442.2297634441989,
        "mae/depa": 11.515335487979439,
        "mae/de": 208.80839589506687,
        "rmse/depa": 11.540448142139972,
        "rmse/de": 251.0486479500305,
        "mae/f_comp": 0.15715885698183887,
        "rmse/f_comp": 0.4353302900580819,
        "loss_component/energy/train": 133.18194332142193,
        "loss_component/forces/train": 1.8951246144205376,
        "total_time/train/per_atom": 0.0018339558636121776,
        "epoch": 2.0,
    }

    test_ref_metrics = {
        "total_loss/test": 1302.1027851409156,
        "loss_component/energy/test": 64.62160950985677,
        "loss_component/forces/test": 0.4835297471890111,
        "mae/depa": 11.347681951823976,
        "mae/de": 194.94918278417094,
        "rmse/depa": 11.36851876981841,
        "rmse/de": 225.80506779030264,
        "mae/f_comp": 0.1356772368246979,
        "rmse/f_comp": 0.3109758020132792,
        "total_time/test/per_atom": 0.0008793184088597842,
        "epoch": 2.0,
    }

    general_integration_test(
        "MoNbTaW-LINEAR",
        train_ref_metrics=train_ref_metrics,
        test_ref_metrics=test_ref_metrics,
        ref_n_epochs=ref_n_epochs,
        input="input_reg.yaml",
    )


def test_MoNbTaW_LINEAR_LBFGS():
    ref_n_epochs = 2

    train_ref_metrics = {
        "total_loss/train": 75.48683618839391,
        "loss_component/energy/train": 6.419800526289803,
        "loss_component/forces/train": 1.1288830925495874,
        "mae/depa": 11.304744995033513,
        "mae/de": 204.84173758111154,
        "rmse/depa": 11.331196341331133,
        "rmse/de": 245.84408084398115,
        "mae/f_comp": 0.07470524098624834,
        "rmse/f_comp": 0.141124124875399,
        "total_time/train/per_atom": 0.00106610117401705,
        "epoch": 2.0,
    }

    test_ref_metrics = {
        "total_loss/test": 187.32807165910026,
        "loss_component/energy/test": 6.258810946825872,
        "loss_component/forces/test": 3.1075926361291417,
        "mae/depa": 11.165157607138141,
        "mae/de": 191.92591728904614,
        "rmse/depa": 11.188217862399599,
        "rmse/de": 222.8066757652101,
        "mae/f_comp": 0.14704485150114724,
        "rmse/f_comp": 0.302200219133964,
        "total_time/test/per_atom": 0.0004324870916795643,
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
        "total_loss/train": 1298.648549259646,
        "mae/depa": 11.31177956259926,
        "mae/de": 205.0262586302169,
        "rmse/depa": 11.338289904097424,
        "rmse/de": 246.76939155005826,
        "mae/f_comp": 0.14461364106304483,
        "rmse/f_comp": 0.36166793839197375,
        "loss_component/energy/train": 128.5568179493576,
        "loss_component/forces/train": 1.308036976607005,
        "total_time/train/per_atom": 0.001107626402264704,
        "epoch": 2.0,
    }

    test_ref_metrics = {
        "total_loss/test": 1154.0660902348204,
        "loss_component/energy/test": 57.33991889008877,
        "loss_component/forces/test": 0.36338562165225763,
        "mae/depa": 10.687848415958253,
        "mae/de": 183.32707099985606,
        "rmse/depa": 10.708867250095947,
        "rmse/de": 211.88352550549567,
        "mae/f_comp": 0.15936079227037753,
        "rmse/f_comp": 0.26958695133565264,
        "total_time/test/per_atom": 0.0004276011823950445,
        "epoch": 2.0,
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
        "total_loss/train": 580.0551744073834,
        "mae/depa": 10.709803585047473,
        "mae/de": 193.7871289769518,
        "rmse/depa": 10.746940437097564,
        "rmse/de": 233.68629284338675,
        "mae/f_comp": 0.44363444991773454,
        "rmse/f_comp": 1.1339159172464006,
        "loss_component/energy/train": 57.7483643792614,
        "loss_component/forces/train": 0.25715306147694933,
        "total_time/train/per_atom": 0.0010201903313155408,
        "epoch": 2.0,
    }

    test_ref_metrics = {
        "total_loss/test": 374.8160429752092,
        "loss_component/energy/test": 16.128034428714074,
        "loss_component/forces/test": 2.6127677200463855,
        "mae/depa": 7.8364490671234845,
        "mae/de": 129.41339211659093,
        "rmse/depa": 8.03194482767756,
        "rmse/de": 145.0444923800858,
        "mae/f_comp": 2.7599641730390574,
        "rmse/f_comp": 5.111523960666119,
        "total_time/test/per_atom": 0.0004013978999436778,
        "epoch": 2.0,
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
        "total_loss/train": 587.969559425464,
        "mae/depa": 10.78256887100019,
        "mae/de": 199.86724571333974,
        "rmse/depa": 10.840024816710168,
        "rmse/de": 246.0331858845646,
        "mae/f_comp": 0.20022271657372614,
        "rmse/f_comp": 0.46843851837906536,
        "loss_component/energy/train": 58.753069013446165,
        "loss_component/forces/train": 0.043886929100234784,
        "total_time/train/per_atom": 0.0007823827504382833,
        "epoch": 3.0,
    }

    test_ref_metrics = {
        "total_loss/test": 473.8415039842953,
        "loss_component/energy/test": 23.681262976236837,
        "loss_component/forces/test": 0.0108122229779265,
        "mae/depa": 9.136589804013706,
        "mae/de": 170.8180299747671,
        "rmse/depa": 9.732679585034502,
        "rmse/de": 205.33937023058223,
        "mae/f_comp": 0.14729941464421478,
        "rmse/f_comp": 0.32881944860251955,
        "total_time/test/per_atom": 6.718820180086529e-05,
        "epoch": 3.0,
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
        "total_loss/train": 522.751506521031,
        "mae/depa": 9.529935763419795,
        "mae/de": 177.10344136560047,
        "rmse/depa": 10.194072774258931,
        "rmse/de": 227.8318444965785,
        "mae/f_comp": 0.649746315688714,
        "rmse/f_comp": 1.256166367682887,
        "loss_component/energy/train": 51.95955986344359,
        "loss_component/forces/train": 0.3155907886595237,
        "total_time/train/per_atom": 0.0007121811226091307,
        "epoch": 3.0,
    }

    test_ref_metrics = {
        "total_loss/test": 675.9221835997215,
        "loss_component/energy/test": 33.645163835346445,
        "loss_component/forces/test": 0.15094534463963263,
        "mae/depa": 10.851812111191185,
        "mae/de": 172.99152290032993,
        "rmse/depa": 11.600890282275142,
        "rmse/de": 209.33259782129326,
        "mae/f_comp": 0.7131755862733037,
        "rmse/f_comp": 1.2285981631096177,
        "total_time/test/per_atom": 7.459771655061666e-05,
        "epoch": 3.0,
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
        "total_loss/train": 430.09694344608204,
        "mae/depa": 8.821535182362613,
        "mae/de": 166.78945723633944,
        "rmse/depa": 9.268644246796566,
        "rmse/de": 221.7653198383413,
        "mae/f_comp": 0.2655290381454106,
        "rmse/f_comp": 0.5282577863631868,
        "loss_component/energy/train": 42.95388308683754,
        "loss_component/forces/train": 0.05581125777066687,
        "total_time/train/per_atom": 0.0012185943424296768,
        "epoch": 3.0,
    }

    test_ref_metrics = {
        "total_loss/test": 400.2635116401342,
        "loss_component/energy/test": 19.97311630389138,
        "loss_component/forces/test": 0.0400592781153317,
        "mae/depa": 7.666058960458322,
        "mae/de": 107.47972679629882,
        "rmse/depa": 8.938258511341317,
        "rmse/de": 124.84818212964348,
        "mae/f_comp": 0.2690065806684933,
        "rmse/f_comp": 0.632923993188216,
        "total_time/test/per_atom": 0.00015437550952329356,
        "epoch": 3.0,
    }

    general_integration_test(
        "MoNbTaW-GRACE",
        train_ref_metrics=train_ref_metrics,
        test_ref_metrics=test_ref_metrics,
        ref_n_epochs=3,
        input="input_2L.yaml",
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
        "total_loss/train": 1616.7761302667877,
        "mae/depa": 10.28875108476347,
        "mae/de": 186.02746511728796,
        "rmse/depa": 10.324624148981506,
        "rmse/de": 223.61061872061956,
        "mae/f_comp": 0.7610195777303886,
        "rmse/f_comp": 2.346907522868055,
        "loss_component/energy/train": 106.5978638177321,
        "loss_component/forces/train": 55.07974920894671,
        "total_time/train/per_atom": 0.0010240561665420462,
        "epoch": 2.0,
    }

    test_ref_metrics = {
        "total_loss/test": 1155.8256010659359,
        "loss_component/energy/test": 45.82083832654222,
        "loss_component/forces/test": 11.970441726754578,
        "mae/depa": 9.543851239048434,
        "mae/de": 160.8440033640785,
        "rmse/depa": 9.572965927709365,
        "rmse/de": 180.66106366720783,
        "mae/f_comp": 0.8480840037694665,
        "rmse/f_comp": 1.5472841837719777,
        "total_time/test/per_atom": 0.0004106889554189847,
        "epoch": 2.0,
    }

    general_integration_test(
        "MoNbTaW-FS",
        train_ref_metrics=train_ref_metrics,
        test_ref_metrics=test_ref_metrics,
        ref_n_epochs=6,
        many_runs=[
            ["input.yaml"],
            ["input_lbfgs.yaml", "-r"],
            ["input.yaml", "-rs", ".epoch_2"],
        ],
    )


def test_MoNbTaW_LINEAR_virial():

    train_ref_metrics = {
        "total_loss/train": 1303.9940398329318,
        "mae/depa": 11.502825314993315,
        "mae/de": 208.57981458794274,
        "rmse/depa": 11.528062856205235,
        "rmse/de": 250.7913423081873,
        "mae/f_comp": 0.15913471409898913,
        "rmse/f_comp": 0.4423051353904123,
        "mae/virial": 20.02185715778421 / 6,
        "rmse/virial": 26.286705828407506 / np.sqrt(6),
        "mae/stress": 0.09289493544303949 / 6,
        "rmse/stress": 0.12495106559133001 / np.sqrt(6),
        "loss_component/energy/train": 26.020550771015284,
        "loss_component/forces/train": 0.2020098481304599,
        "loss_component/virial/train": 234.5762473474406,
        "total_time/train/per_atom": 0.0013482222642304132,
        "epoch": 2.0,
        "per_group_metrics.low.mae/depa": 11.502825314993315,
        "per_group_metrics.low.mae/de": 208.57981458794274,
        "per_group_metrics.low.rmse/depa": 11.528062856205233,
        "per_group_metrics.low.rmse/de": 250.79134230818727,
        "per_group_metrics.low.mae/f_comp": 0.15913471409898908,
        "per_group_metrics.low.rmse/f_comp": 0.4423051353904123,
        "per_group_metrics.low.mae/virial": 20.02185715778421 / 6,
        "per_group_metrics.low.rmse/virial": 26.286705828407506 / np.sqrt(6),
        "per_group_metrics.low.mae/stress": 0.09289493544303949 / 6,
        "per_group_metrics.low.rmse/stress": 0.12495106559133001 / np.sqrt(6),
        "per_group_metrics.low.num_struct": 20.0,
        "per_group_metrics.low.num_atoms": 368.0,
    }

    test_ref_metrics = {
        "total_loss/test": 7152.245905851851,
        "loss_component/energy/test": 24.983992874678588,
        "loss_component/forces/test": 0.1856877459630084,
        "loss_component/virial/test": 1405.2795005497285,
        "mae/depa": 11.330878184360502,
        "mae/de": 194.60536324984474,
        "rmse/depa": 11.35174414948893,
        "rmse/de": 225.31548120509444,
        "mae/f_comp": 0.13613679686450758,
        "rmse/f_comp": 0.31333163025317556,
        "mae/virial": 36.52838226023615 / 6,
        "rmse/virial": 60.52494912576713 / np.sqrt(6),
        "mae/stress": 0.07688909183428375 / 6,
        "rmse/stress": 0.08586847762138143 / np.sqrt(6),
        "total_time/test/per_atom": 0.0005342044641592485,
        "epoch": 2.0,
        "per_group_metrics.low.mae/depa": 11.330878184360502,
        "per_group_metrics.low.mae/de": 194.60536324984471,
        "per_group_metrics.low.rmse/depa": 11.35174414948893,
        "per_group_metrics.low.rmse/de": 225.31548120509444,
        "per_group_metrics.low.mae/f_comp": 0.13613679686450755,
        "per_group_metrics.low.rmse/f_comp": 0.3133316302531755,
        "per_group_metrics.low.mae/virial": 36.52838226023615 / 6,
        "per_group_metrics.low.rmse/virial": 60.52494912576713 / np.sqrt(6),
        "per_group_metrics.low.mae/stress": 0.07688909183428375 / 6,
        "per_group_metrics.low.rmse/stress": 0.08586847762138143 / np.sqrt(6),
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
        "total_loss/train": 130.1095051500973,
        "mae/depa": 11.459735678199728,
        "mae/de": 207.93886066369905,
        "rmse/depa": 11.484709077506212,
        "rmse/de": 250.13629560924804,
        "mae/f_comp": 0.15762830787117205,
        "rmse/f_comp": 0.43945699027017066,
        "mae/virial": 21.56246063584931 / 6,
        "rmse/virial": 27.593456663663172 / np.sqrt(6),
        "mae/stress": 0.09670518653073343 / 6,
        "rmse/stress": 0.1265534886036844 / np.sqrt(6),
        "loss_component/energy/train": 25.833186041290624,
        "loss_component/forces/train": 0.18786268406124973,
        "loss_component/stress/train": 0.0008523046675855876,
        "total_time/train/per_atom": 0.0013555795107411382,
        "epoch": 2.0,
        "per_group_metrics.low.mae/depa": 11.459735678199726,
        "per_group_metrics.low.mae/de": 207.93886066369905,
        "per_group_metrics.low.rmse/depa": 11.484709077506214,
        "per_group_metrics.low.rmse/de": 250.136295609248,
        "per_group_metrics.low.mae/f_comp": 0.15762830787117202,
        "per_group_metrics.low.rmse/f_comp": 0.43945699027017066,
        "per_group_metrics.low.mae/virial": 21.562460635849305 / 6,
        "per_group_metrics.low.rmse/virial": 27.593456663663172 / np.sqrt(6),
        "per_group_metrics.low.mae/stress": 0.09670518653073343 / 6,
        "per_group_metrics.low.rmse/stress": 0.1265534886036844 / np.sqrt(6),
        "per_group_metrics.low.num_struct": 20.0,
        "per_group_metrics.low.num_atoms": 368.0,
    }

    test_ref_metrics = {
        "total_loss/test": 124.469971293127,
        "loss_component/energy/test": 24.737518839473054,
        "loss_component/forces/test": 0.15471204638556327,
        "loss_component/stress/test": 0.0017633727667865197,
        "mae/depa": 11.273563420336162,
        "mae/de": 193.74105659526649,
        "rmse/depa": 11.294935131267389,
        "rmse/de": 224.45670238924956,
        "mae/f_comp": 0.13249115897610664,
        "rmse/f_comp": 0.3081812658968042,
        "mae/virial": 35.14878224972145 / 6,
        "rmse/virial": 58.20357789999212 / np.sqrt(6),
        "mae/stress": 0.07332355442468 / 6,
        "rmse/stress": 0.08370579261170506 / np.sqrt(6),
        "total_time/test/per_atom": 0.0005240003550973009,
        "epoch": 2.0,
        "per_group_metrics.low.mae/depa": 11.273563420336162,
        "per_group_metrics.low.mae/de": 193.7410565952665,
        "per_group_metrics.low.rmse/depa": 11.294935131267389,
        "per_group_metrics.low.rmse/de": 224.45670238924956,
        "per_group_metrics.low.mae/f_comp": 0.13249115897610667,
        "per_group_metrics.low.rmse/f_comp": 0.30818126589680417,
        "per_group_metrics.low.mae/virial": 35.14878224972145 / 6,
        "per_group_metrics.low.rmse/virial": 58.20357789999211 / np.sqrt(6),
        "per_group_metrics.low.mae/stress": 0.07332355442468 / 6,
        "per_group_metrics.low.rmse/stress": 0.08370579261170506 / np.sqrt(6),
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


def test_MoNbTaW_LINEAR_loss_explosion():

    train_ref_metrics = {
        "total_loss/train": 408.12918184171724,
        "mae/depa": 6.679261737057627,
        "mae/de": 129.0668404380702,
        "rmse/depa": 6.914615224464614,
        "rmse/de": 159.8973819562407,
        "mae/f_comp": 0.29838333096742475,
        "rmse/f_comp": 0.8489019709475522,
        "loss_component/energy/train": 4.781190370239782,
        "loss_component/forces/train": 36.03172781393194,
        "total_time/train/per_atom": 0.0011190956555883927,
        "epoch": 5.0,
    }

    test_ref_metrics = {
        "total_loss/test": 97.79900682105732,
        "loss_component/energy/test": 2.1482403446602554,
        "loss_component/forces/test": 2.741709996392611,
        "mae/depa": 6.22956375631627,
        "mae/de": 109.03204076630132,
        "rmse/depa": 6.554754525777842,
        "rmse/de": 124.8577822706804,
        "mae/f_comp": 0.12659812947522917,
        "rmse/f_comp": 0.3311621956922385,
        "total_time/test/per_atom": 0.00043635737931574966,
        "epoch": 5.0,
    }

    general_integration_test(
        "MoNbTaW-LINEAR",
        train_ref_metrics=train_ref_metrics,
        test_ref_metrics=test_ref_metrics,
        ref_n_epochs=5,
        many_runs=[
            ["input_loss_explosion.yaml"],
        ],
    )


def test_MoNbTaW_LINEAR_f32():
    train_ref_metrics = {
        "total_loss/train": 76.09556579589844,
        "loss_component/energy/train": 6.613865852355957,
        "loss_component/forces/train": 0.995691180229187,
        "mae/depa": 11.472644805908203,
        "mae/de": 207.6016357421875,
        "rmse/depa": 11.501187735879173,
        "rmse/de": 249.19645864257382,
        "mae/f_comp": 0.07174675015435703,
        "rmse/f_comp": 0.1558739601480873,
        "total_time/train/per_atom": 0.0011840399674047544,
        "epoch": 2.0,
    }

    test_ref_metrics = {
        "total_loss/test": 200.94918823242188,
        "loss_component/energy/test": 6.408252716064453,
        "loss_component/forces/test": 3.6392064094543457,
        "mae/depa": 11.297935485839844,
        "mae/de": 194.24755859375,
        "rmse/depa": 11.320999914428937,
        "rmse/de": 225.39544416425102,
        "mae/f_comp": 0.14082810645009958,
        "rmse/f_comp": 0.32013464011877724,
        "total_time/test/per_atom": 0.0006189632735186023,
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
    )
