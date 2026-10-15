"""TensorMetrics: mean over real tensor components, padding rows left out,
per-atom and per-structure tensors alike."""

import os

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

import numpy as np
import tensorflow as tf

from tensorpotential import constants
from tensorpotential.cli.metrics import aggregate_metrics, process_accumulated_metrics
from tensorpotential.extra.gen_tensor import constants as cc
from tensorpotential.extra.gen_tensor.metrics import TensorMetrics


def _metrics(ref, pred, weight, nat_per_struct):
    nat = int(sum(nat_per_struct))
    batch = {
        constants.N_ATOMS_BATCH_REAL: tf.constant(nat),
        constants.N_STRUCTURES_BATCH_REAL: tf.constant(len(nat_per_struct)),
        constants.ATOMS_TO_STRUCTURE_MAP: tf.constant(
            np.repeat(np.arange(len(nat_per_struct)), nat_per_struct), dtype=tf.int32
        ),
        cc.DATA_REFERENCE_TENSOR: tf.constant(ref),
        cc.DATA_REFERENCE_TENSOR_WEIGHT: tf.constant(weight),
    }
    m = TensorMetrics()
    out = {k: np.asarray(v) for k, v in m(batch, {cc.PREDICT_TENSOR: tf.constant(pred)}).items()}
    return process_accumulated_metrics(aggregate_metrics(out), 1, m.normalization_spec)


def test_per_atom_vector_is_the_mean_over_real_components():
    rng = np.random.default_rng(0)
    ref = rng.standard_normal((6, 3))  # 5 real atoms + 1 padding atom
    pred = ref + rng.standard_normal((6, 3))
    weight = np.array([[1.0]] * 5 + [[0.0]])
    pred[5] += 100.0  # padding prediction must not count
    d = (ref - pred)[:5]
    res = _metrics(ref, pred, weight, [2, 3])
    np.testing.assert_allclose(res["mae/d_tensor"], np.abs(d).mean())
    np.testing.assert_allclose(res["rmse/d_tensor"], np.sqrt((d**2).mean()))


def test_per_structure_tensor_is_divided_by_structures_not_atoms():
    rng = np.random.default_rng(1)
    ref = rng.standard_normal((3, 9))  # 2 real structures + 1 padding structure
    pred = ref + rng.standard_normal((3, 9))
    weight = np.array([[1.0], [1.0], [0.0]])
    d = (ref - pred)[:2]
    res = _metrics(ref, pred, weight, [4, 7])  # 11 atoms in 2 structures
    np.testing.assert_allclose(res["mae/d_tensor"], np.abs(d).mean())
    np.testing.assert_allclose(res["rmse/d_tensor"], np.sqrt((d**2).mean()))
