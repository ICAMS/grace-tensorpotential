import os

import numpy as np
import pytest
from ase.build import bulk

from tensorpotential.instructions.base import LORAInstructionMixin

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

import logging

import tensorflow as tf
import shutil

from tensorpotential.utils import convert_model_reduce_elements
from tensorpotential.tensorpot import TensorPotential
from tensorpotential.tpmodel import TPModel
from tensorpotential.potentials.presets import (
    GRACE_2LAYER,
    save_instructions_dict,
    load_instructions,
)
from tensorpotential.calculator import TPCalculator

from ase import Atoms

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


def test_set_trainable_variables():

    list_of_instructions = GRACE_2LAYER(element_map={"Be": 0, "Li": 1}, lmax=2)
    tp = TPModel(list_of_instructions)
    tp.build(tf.float64)

    Z = tp.instructions["Z"]
    assert len(Z.trainable_variables) > 0
    trainable_names = []
    tp.set_trainable_variables(trainable_names)

    assert len(Z.trainable_variables) == 0


def test_convert_model_reduce_elements():
    TEST_PATH = "test_checkpoints"
    if os.path.isdir(TEST_PATH):
        shutil.rmtree(TEST_PATH)
    os.makedirs(TEST_PATH, exist_ok=True)

    float_dtype = tf.float64
    # stage 1: convert_model_reduce_elements
    instructions = GRACE_2LAYER(
        element_map={"Mo": 0, "Nb": 1, "Ta": 2, "W": 3},
        lmax=0,
        basis_type="Cheb",
        cutoff_dict={"Mo": 4, "Nb": 5, "Ta": 6, "W": 7},
    )
    tp = TensorPotential(instructions, float_dtype=float_dtype)

    potential_file_name = os.path.join(TEST_PATH, "model.MoNbTaW.yaml")
    checkpoint_name = os.path.join(TEST_PATH, "checkpoint.MoNbTaW")

    save_instructions_dict(potential_file_name, instructions)
    tp.save_checkpoint(checkpoint_name=checkpoint_name)

    assert os.path.isfile(potential_file_name)
    assert os.path.isfile(checkpoint_name + ".index")

    new_potential_file_name = os.path.join(TEST_PATH, "model.MoTa.yaml")
    new_checkpoint_name = os.path.join(TEST_PATH, "checkpoint.MoTa")

    assert not os.path.isfile(new_potential_file_name)
    assert not os.path.isfile(new_checkpoint_name + ".index")

    convert_model_reduce_elements(
        element_map={"Mo": 0, "Ta": 1},
        potential_file_name=potential_file_name,
        checkpoint_name=checkpoint_name,
        new_potential_file_name=new_potential_file_name,
        new_checkpoint_name=new_checkpoint_name,
    )

    assert os.path.isfile(new_potential_file_name)
    assert os.path.isfile(new_checkpoint_name + ".index")

    # stage 2: compare predictions
    instructions1 = load_instructions(potential_file_name)
    instructions2 = load_instructions(new_potential_file_name)
    model1 = TensorPotential(instructions1)
    model1.load_checkpoint(checkpoint_name=checkpoint_name)

    model2 = TensorPotential(instructions2)
    model2.load_checkpoint(checkpoint_name=new_checkpoint_name)

    calc1 = TPCalculator(model1.model)
    calc2 = TPCalculator(model2.model)

    at = Atoms("MoTa", positions=[[0, 0, 0], [0, 0, 2]], pbc=False)

    at.calc = calc1
    e1 = at.get_potential_energy()
    print("e1=", e1)

    at.calc = calc2
    e2 = at.get_potential_energy()
    print("e2=", e2)

    assert np.allclose(e1, e2)

    # structure that does not work for potential 2
    at2 = Atoms("MoNb", positions=[[0, 0, 0], [0, 0, 2]], pbc=False)

    at2.calc = calc1
    e1 = at2.get_potential_energy()
    print("e1=", e1)

    at2.calc = calc2
    with pytest.raises(AssertionError):
        e2 = at2.get_potential_energy()

    shutil.rmtree(TEST_PATH)


@pytest.mark.xfail
def test_activate_reduce_lora():
    LORA_CONFIG = {"rank": 4, "alpha": 1}
    float_dtype = tf.float64
    instructions = GRACE_2LAYER(
        element_map={"Mo": 0, "Nb": 1, "Ta": 2, "W": 3}, lmax=0, basis_type="SBessel"
    )
    tp = TensorPotential(instructions, float_dtype=float_dtype)
    assert not tp.is_lora_enabled()
    tp.save_checkpoint(checkpoint_name="test_checkpoints/checkpoint_no_lora")

    at = bulk("Mo")

    calc1 = TPCalculator(tp.model)
    at.calc = calc1
    e1 = at.get_potential_energy()
    print("e1=", e1)

    instructions = tp.model.instructions
    Z = instructions["Z"]
    assert not Z.lora
    assert not hasattr(Z, "lora_tensors")

    Z_trainable_vars = [var.name for var in Z.trainable_variables]
    print("Z_trainable_vars=", Z_trainable_vars)
    assert Z_trainable_vars == ["Z/ChemicalEmbedding:0"]

    lora_config = {
        ins_name: LORA_CONFIG
        for ins_name, ins in instructions.items()
        if isinstance(ins, LORAInstructionMixin)
    }
    print("lora_config=", lora_config)
    for k, v in lora_config.items():
        print(" -", k)

    tp.enable_lora_adaptation(lora_config)

    assert tp.is_lora_enabled()
    assert Z.lora
    assert hasattr(Z, "lora_tensors")

    tp.save_checkpoint(checkpoint_name="test_checkpoints/checkpoint_with_lora")

    calc2 = TPCalculator(tp.model)
    at.calc = calc2
    e2 = at.get_potential_energy()
    print("e2=", e2)
    assert np.allclose(e1, e2)

    Z_trainable_vars_after = [var.name for var in Z.trainable_variables]

    print("Z_trainable_vars_after=", Z_trainable_vars_after)
    assert Z_trainable_vars_after == ["Z/w/LORA/A:0", "Z/w/LORA/B:0"]

    for t in Z.lora_tensors:
        t.assign(t + tf.ones_like(t))

    calc3 = TPCalculator(tp.model)
    at.calc = calc3
    e2b = at.get_potential_energy()
    print("e2b=", e2b)

    assert not np.allclose(e1, e2b)

    ### stage 2 : reduce LORA
    tp.finalize_lora_update()

    assert not tp.is_lora_enabled()
    assert not Z.lora
    assert not hasattr(Z, "lora_tensors")

    Z_trainable_vars_2 = [var.name for var in Z.trainable_variables]
    print("Z_trainable_vars_2=", Z_trainable_vars_2)
    assert Z_trainable_vars_2 == ["Z/ChemicalEmbedding:0"]

    calc4 = TPCalculator(tp.model)
    at.calc = calc4
    e4 = at.get_potential_energy()
    print("e4=", e4)

    assert np.allclose(e4, e2b)


@pytest.mark.xfail
def test_activate_reduce_additive():
    LORA_CONFIG = {"mode": "full_additive"}
    float_dtype = tf.float64
    instructions = GRACE_2LAYER(
        element_map={"Mo": 0, "Nb": 1, "Ta": 2, "W": 3}, lmax=0, basis_type="SBessel"
    )
    tp = TensorPotential(instructions, float_dtype=float_dtype)
    assert not tp.is_lora_enabled()
    tp.save_checkpoint(checkpoint_name="test_checkpoints/checkpoint_no_lora")

    at = bulk("Mo")

    calc1 = TPCalculator(tp.model)
    at.calc = calc1
    e1 = at.get_potential_energy()
    print("e1=", e1)

    instructions = tp.model.instructions
    Z = instructions["Z"]
    assert not Z.lora
    assert not hasattr(Z, "lora_tensors")

    Z_trainable_vars = [var.name for var in Z.trainable_variables]
    print("Z_trainable_vars=", Z_trainable_vars)
    assert Z_trainable_vars == ["Z/ChemicalEmbedding:0"]

    lora_config = {
        ins_name: LORA_CONFIG
        for ins_name, ins in instructions.items()
        if isinstance(ins, LORAInstructionMixin)
    }
    print("lora_config=", lora_config)
    for k, v in lora_config.items():
        print(" -", k)

    tp.enable_lora_adaptation(lora_config)

    assert tp.is_lora_enabled()
    assert Z.lora
    assert hasattr(Z, "lora_tensors")

    tp.save_checkpoint(checkpoint_name="test_checkpoints/checkpoint_with_lora")

    calc2 = TPCalculator(tp.model)
    at.calc = calc2
    e2 = at.get_potential_energy()
    print("e2=", e2)
    assert np.allclose(e1, e2)

    Z_trainable_vars_after = [var.name for var in Z.trainable_variables]

    print("Z_trainable_vars_after=", Z_trainable_vars_after)
    assert Z_trainable_vars_after == ["Z/w/ADDITIVE/delta_W:0"]

    for t in Z.lora_tensors:
        t.assign(t + tf.ones_like(t))

    calc3 = TPCalculator(tp.model)
    at.calc = calc3
    e2b = at.get_potential_energy()
    print("e2b=", e2b)

    assert not np.allclose(e1, e2b)

    ### stage 2 : reduce LORA
    tp.finalize_lora_update()

    assert not tp.is_lora_enabled()
    assert not Z.lora
    assert not hasattr(Z, "lora_tensors")

    Z_trainable_vars_2 = [var.name for var in Z.trainable_variables]
    print("Z_trainable_vars_2=", Z_trainable_vars_2)
    assert Z_trainable_vars_2 == ["Z/ChemicalEmbedding:0"]

    calc4 = TPCalculator(tp.model)
    at.calc = calc4
    e4 = at.get_potential_energy()
    print("e4=", e4)

    assert np.allclose(e4, e2b)
