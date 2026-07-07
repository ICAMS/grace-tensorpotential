"""Parity test for the matmul path in tensorpotential/functions/nn.py.

Linear and DenseLayer apply a plain 2-D weight via tf.matmul instead of
tf.einsum("...k,...kn->...n", x, w). The two are mathematically identical; matmul
is used because einsum's gradient lowers the weight-gradient as a reduce_sum over
the batch axis, which XLA >= 2.21 mis-tiles into large ptxas register spills
(matmul keeps it a cuBLAS GEMM). This test pins the equivalence: forward +
gradients (wrt input and weight) must match an explicit einsum reference to fp64.
Run on CPU.
"""
import os

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

import numpy as np
import tensorflow as tf
from tensorflow import float64

from tensorpotential.functions.nn import Linear, DenseLayer


def _reference(x, w, norm):
    # the math the matmul path implements, written as the explicit einsum
    return tf.einsum("...k,kn->...n", x, w * norm)


def _parity(layer, xshape):
    np.random.seed(0)
    tf.random.set_seed(0)
    layer.build(float64)
    x0 = tf.constant(np.random.randn(*xshape), dtype=float64)

    with tf.GradientTape() as t:
        t.watch(x0)
        y = layer(x0)
        loss = tf.reduce_sum(y * y)
    g = [tf.convert_to_tensor(v).numpy() for v in t.gradient(loss, [x0, layer.w])]

    with tf.GradientTape() as tr:
        tr.watch(x0)
        yr = _reference(x0, layer.w, layer.norm)
        lossr = tf.reduce_sum(yr * yr)
    gr = [tf.convert_to_tensor(v).numpy() for v in tr.gradient(lossr, [x0, layer.w])]

    assert y.shape == yr.shape
    assert np.allclose(y.numpy(), yr.numpy(), atol=1e-12, rtol=0), (
        f"forward differs, max {np.max(np.abs(y.numpy() - yr.numpy()))}"
    )
    for a, b in zip(g, gr):
        assert np.allclose(a, b, atol=1e-10, rtol=0), (
            f"grad differs, max {np.max(np.abs(a - b))}"
        )


def test_linear_matmul_parity_2d():
    _parity(Linear(n_in=5, n_out=7, name="L2", use_bias=False), (64, 5))


def test_linear_matmul_parity_3d():
    _parity(Linear(n_in=5, n_out=7, name="L3", use_bias=False), (8, 16, 5))


def test_denselayer_matmul_parity_2d():
    _parity(DenseLayer(n_in=5, n_out=7, name="D2", use_bias=False), (64, 5))


def test_denselayer_matmul_parity_3d():
    _parity(DenseLayer(n_in=5, n_out=7, name="D3", use_bias=False), (8, 16, 5))
