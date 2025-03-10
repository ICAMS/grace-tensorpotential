import os

import pytest
from tensorpotential.tpmodel import TPModel
import tensorflow as tf

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

from tensorpotential.potentials.presets import FS, GRACE_2LAYER
import logging

LOG_FMT = "%(asctime)s %(levelname).1s - %(message)s"
logging.basicConfig(level=logging.INFO, format=LOG_FMT, datefmt="%Y/%m/%d %H:%M:%S")
log = logging.getLogger()


# TODO: maybe parameterise with @pytest.mark.parametrize
def test_tp_model_repr_verbose_0():
    list_of_instructions = GRACE_2LAYER(element_map={"Be": 0, "Li": 1}, lmax=2)

    tp = TPModel(list_of_instructions)
    tp.build(tf.float64)

    res = tp.summary(verbose=0)
    print("Your summary (verbose = 0) is:\n", res, sep="\n")
    # assert trainable param table
    assert "Layer" in res
    assert "Output Shape" in res
    assert "Param #" in res
    assert "Trainable Params:" in res

    # assert Radial Instruction is included
    assert "Radial" in res

    # assert kwargs are NOT present
    assert "name=" not in res
    assert "l=" not in res

    # assert Coupling functions table is NOT present
    assert "Coupling functions" not in res
    assert " parity " not in res

    # assert Coupling data table is NOT present
    assert "left_inds" not in res
    assert "right_inds" not in res
    assert "cg_list" not in res


def test_tp_model_repr_verbose_1():
    list_of_instructions = GRACE_2LAYER(element_map={"Be": 0, "Li": 1}, lmax=2)

    tp = TPModel(list_of_instructions)
    tp.build(tf.float64)

    res = tp.summary(verbose=1)
    print("Your summary (verbose=1) is:\n", res, sep="\n")

    # assert trainable param table
    assert "Layer" in res
    assert "Output Shape" in res
    assert "Param #" in res
    assert "Trainable Params:" in res

    # assert Radial Instruction is included
    assert "Radial" in res

    # assert kwargs are present
    assert "name=" in res
    assert "l=" in res

    # assert Coupling functions table
    assert "Coupling Functions" in res
    assert " parity " in res

    # assert Coupling data columns are NOT present
    assert "left_inds" not in res
    assert "right_inds" not in res
    assert "cg_list" not in res


def test_tp_model_repr_verbose_2():
    list_of_instructions = GRACE_2LAYER(element_map={"Be": 0, "Li": 1}, lmax=2)

    tp = TPModel(list_of_instructions)
    tp.build(tf.float64)

    res = tp.summary(verbose=2)
    print("Your summary (verbose=2) is:\n", res, sep="\n")

    # assert trainable param table
    assert "Layer" in res
    assert "Output Shape" in res
    assert "Param #" in res
    assert "Trainable Params:" in res

    # assert Radial Instruction is included
    assert "Radial" in res

    # assert kwargs are present
    assert "name=" in res
    assert "l=" in res

    # assert Coupling functions table is not included
    assert "Coupling Functions" not in res

    # assert Coupling data table
    assert "Coupling Table" in res
    assert " parity " in res
    assert "left_inds" in res
    assert "right_inds" in res
    assert "cg_list" in res
