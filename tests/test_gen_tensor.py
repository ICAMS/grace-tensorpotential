import os

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

import numpy as np
import pandas as pd
import tensorflow as tf
from ase import Atoms
from tensorpotential.potentials import get_preset
from tensorpotential.extra.gen_tensor import constants as cc
from tensorpotential import constants


def test_tensor_1l_preset_builds():
    preset_fn = get_preset("TENSOR_1L")
    instructions = preset_fn(
        element_map={"Si": 0, "O": 1},
        rcut=5.0,
        tensor_components=[1, 2],
        lmax=2,
        n_rad_base=4,
        n_rad_max=8,
        embedding_size=8,
        n_mlp_dens=4,
        max_order=2,
    )
    assert len(instructions) > 0


def test_tensor_2l_preset_builds():
    preset_fn = get_preset("TENSOR_2L")
    instructions = preset_fn(
        element_map={"Si": 0},
        rcut=5.0,
        tensor_components=[2],
        lmax=2,
        n_rad_base=4,
        n_rad_max=[8, 12],
        embedding_size=8,
        n_mlp_dens=4,
        max_order=2,
    )
    assert len(instructions) > 0


def test_reference_tensor_databuilder_rank2():
    from tensorpotential.extra.gen_tensor.databuilder import ReferenceTensorDataBuilder

    db = ReferenceTensorDataBuilder(tensor_rank=2, per_structure=False)
    n_atoms = 3
    atoms = Atoms("SiO2", positions=[[0, 0, 0], [1, 0, 0], [0, 1, 0]], cell=[5, 5, 5], pbc=True)
    tensor_data = np.random.randn(n_atoms, 9)
    row = pd.Series(
        {
            constants.COLUMN_ASE_ATOMS: atoms,
            cc.COLUMN_REFERENCE_TENSOR: tensor_data.tolist(),
        }
    )
    result = db.extract_from_row(row, **{constants.DATA_STRUCTURE_ID: 0})
    assert result[cc.DATA_REFERENCE_TENSOR].shape == (n_atoms, 9)
    assert result[cc.DATA_REFERENCE_TENSOR_WEIGHT].shape == (n_atoms, 1)


def test_weighted_tensor_loss():
    from tensorpotential.extra.gen_tensor.loss import WeightedTensorLoss

    loss = WeightedTensorLoss(loss_component_weight=1.0, type="huber", delta=0.01)
    loss.build(tf.float64)
    n = 5
    input_data = {
        cc.DATA_REFERENCE_TENSOR: tf.constant(np.random.randn(n, 9), dtype=tf.float64),
        cc.DATA_REFERENCE_TENSOR_WEIGHT: tf.constant(np.ones((n, 1)), dtype=tf.float64),
    }
    predictions = {cc.PREDICT_TENSOR: tf.constant(np.random.randn(n, 9), dtype=tf.float64)}
    loss_val = loss(input_data, predictions)
    assert tf.math.is_finite(loss_val) and loss_val > 0


