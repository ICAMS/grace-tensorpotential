"""Unit tests for piecewise_linear kernel and WeightedPiecewiseLinearForceLoss."""
from __future__ import annotations

import os

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

import numpy as np
import pytest
import tensorflow as tf

from tensorpotential import constants
from tensorpotential.loss import (
    WeightedPiecewiseLinearEnergyPerAtomLoss,
    WeightedPiecewiseLinearForceLoss,
    WeightedPiecewiseLinearStressLoss,
    piecewise_linear,
)


# Defaults used throughout
D1, D2 = 0.1, 0.5
B1, B2, B3 = 1.0, 0.1, 0.01

# c1 = beta_1 * delta_1 = 0.1
# c2 = c1 + beta_2 * (delta_2 - delta_1) = 0.1 + 0.1 * 0.4 = 0.14


def _kernel(x):
    return float(
        piecewise_linear(
            tf.constant([x], dtype=tf.float64), D1, D2, B1, B2, B3
        ).numpy()[0]
    )


@pytest.mark.parametrize(
    "x, expected",
    [
        (0.0, 0.0),
        (0.05, 0.05),                           # region 1: beta_1 * x
        (0.1, 0.1),                             # boundary delta_1: c1
        (0.3, 0.1 + 0.1 * (0.3 - 0.1)),         # region 2: c1 + beta_2 * (x - delta_1) = 0.12
        (0.5, 0.14),                            # boundary delta_2: c2
        (1.0, 0.14 + 0.01 * (1.0 - 0.5)),       # region 3: c2 + beta_3 * (x - delta_2) = 0.145
        (10.0, 0.14 + 0.01 * (10.0 - 0.5)),     # large outlier: still grows linearly with beta_3
    ],
)
def test_piecewise_linear_values(x, expected):
    assert _kernel(x) == pytest.approx(expected, rel=1e-12, abs=1e-12)


def test_piecewise_linear_continuity_at_boundaries():
    # approach from below and from above should give the same value
    eps = 1e-9
    assert _kernel(D1 - eps) == pytest.approx(_kernel(D1 + eps), abs=1e-7)
    assert _kernel(D2 - eps) == pytest.approx(_kernel(D2 + eps), abs=1e-7)


def test_piecewise_linear_gradient_per_region():
    # Gradient w.r.t. abs_x should be beta_1, beta_2, beta_3 per region.
    xs = tf.Variable([0.05, 0.3, 1.0], dtype=tf.float64)
    with tf.GradientTape() as tape:
        y = piecewise_linear(xs, D1, D2, B1, B2, B3)
        loss = tf.reduce_sum(y)
    grads = tape.gradient(loss, xs).numpy()
    np.testing.assert_allclose(grads, [B1, B2, B3], atol=1e-12)


def _build_loss(use_l2_norm: bool):
    loss = WeightedPiecewiseLinearForceLoss(
        loss_component_weight=1.0,
        delta_1=D1,
        delta_2=D2,
        beta_1=B1,
        beta_2=B2,
        beta_3=B3,
        use_l2_norm=use_l2_norm,
        normalize_by_samples=True,
    )
    loss.build(tf.float64)
    return loss


def test_force_loss_componentwise_known_residuals():
    # 3 atoms with deliberately one residual in each region
    f_true = tf.constant([[0.0, 0.0, 0.0]] * 3, dtype=tf.float64)
    f_pred = tf.constant(
        [[0.05, 0.0, 0.0], [0.3, 0.0, 0.0], [1.0, 0.0, 0.0]], dtype=tf.float64
    )
    f_weight = tf.ones((3, 1), dtype=tf.float64)

    loss = _build_loss(use_l2_norm=False)
    out = loss.compute_loss_component(
        input_data={
            constants.DATA_REFERENCE_FORCES: f_true,
            constants.DATA_FORCE_WEIGHTS: f_weight,
        },
        predictions={constants.PREDICT_FORCES: f_pred},
    )
    # per-element loss: 0.05 (region 1) + 0.12 (region 2) + 0.145 (region 3)
    # plus 6 zero-residual entries -> contribute 0
    # divide by sum(f_weight) * 3 = 3 * 3 = 9
    expected = (0.05 + 0.12 + 0.145) / 9.0
    assert float(out.numpy()) == pytest.approx(expected, rel=1e-10)


def test_force_loss_l2_norm_known_residuals():
    f_true = tf.constant([[0.0, 0.0, 0.0]] * 3, dtype=tf.float64)
    # residuals chosen so ||delta||_2 lands in each region
    f_pred = tf.constant(
        [[0.05, 0.0, 0.0], [0.3, 0.0, 0.0], [1.0, 0.0, 0.0]], dtype=tf.float64
    )
    f_weight = tf.ones((3, 1), dtype=tf.float64)

    loss = _build_loss(use_l2_norm=True)
    out = loss.compute_loss_component(
        input_data={
            constants.DATA_REFERENCE_FORCES: f_true,
            constants.DATA_FORCE_WEIGHTS: f_weight,
        },
        predictions={constants.PREDICT_FORCES: f_pred},
    )
    # one piecewise value per atom; divide by sum(f_weight) = 3 (no x3)
    expected = (0.05 + 0.12 + 0.145) / 3.0
    assert float(out.numpy()) == pytest.approx(expected, rel=1e-6)


def test_invalid_thresholds_raise():
    with pytest.raises(AssertionError):
        WeightedPiecewiseLinearForceLoss(
            loss_component_weight=1.0, delta_1=0.5, delta_2=0.1
        )


def test_invalid_slopes_raise():
    with pytest.raises(AssertionError):
        WeightedPiecewiseLinearForceLoss(
            loss_component_weight=1.0, beta_1=0.1, beta_2=1.0
        )
    with pytest.raises(AssertionError):
        WeightedPiecewiseLinearForceLoss(
            loss_component_weight=1.0, beta_3=0.0
        )


# --- Energy-per-atom variant ---

def _build_epa_loss(delta_1=D1, delta_2=D2, beta_1=B1, beta_2=B2, beta_3=B3):
    loss = WeightedPiecewiseLinearEnergyPerAtomLoss(
        loss_component_weight=1.0,
        delta_1=delta_1,
        delta_2=delta_2,
        beta_1=beta_1,
        beta_2=beta_2,
        beta_3=beta_3,
    )
    loss.build(tf.float64)
    return loss


def test_epa_loss_known_residuals():
    # 3 structures with N_atoms = 1 each so per-atom error == total energy error.
    # E_pred - E_true chosen to land in each region: 0.05, 0.3, 1.0
    e_true = tf.constant([[0.0], [0.0], [0.0]], dtype=tf.float64)
    e_pred = tf.constant([[0.05], [0.3], [1.0]], dtype=tf.float64)
    e_weight = tf.ones((3, 1), dtype=tf.float64)
    # one atom per structure -> map = [0, 1, 2]
    atom_map = tf.constant([0, 1, 2], dtype=tf.int32)
    n_struct = tf.constant(3, dtype=tf.int32)

    loss = _build_epa_loss()
    out = loss.compute_loss_component(
        input_data={
            constants.DATA_REFERENCE_ENERGY: e_true,
            constants.DATA_ENERGY_WEIGHTS: e_weight,
            constants.ATOMS_TO_STRUCTURE_MAP: atom_map,
            constants.N_STRUCTURES_BATCH_TOTAL: n_struct,
        },
        predictions={constants.PREDICT_TOTAL_ENERGY: e_pred},
    )
    # piecewise values: 0.05 (region 1) + 0.12 (region 2) + 0.145 (region 3)
    # divided by sum(e_weight) = 3
    expected = (0.05 + 0.12 + 0.145) / 3.0
    assert float(out.numpy()) == pytest.approx(expected, rel=1e-10)


def test_epa_loss_per_atom_normalization():
    # Two structures, one with 1 atom and one with 4 atoms.
    # Set total energies so |E/N| are 0.3 and 0.05 respectively.
    e_true = tf.constant([[0.0], [0.0]], dtype=tf.float64)
    e_pred = tf.constant([[0.3], [0.2]], dtype=tf.float64)  # 0.2 / 4 atoms = 0.05
    e_weight = tf.ones((2, 1), dtype=tf.float64)
    atom_map = tf.constant([0, 1, 1, 1, 1], dtype=tf.int32)  # 1 + 4 atoms
    n_struct = tf.constant(2, dtype=tf.int32)

    loss = _build_epa_loss()
    out = loss.compute_loss_component(
        input_data={
            constants.DATA_REFERENCE_ENERGY: e_true,
            constants.DATA_ENERGY_WEIGHTS: e_weight,
            constants.ATOMS_TO_STRUCTURE_MAP: atom_map,
            constants.N_STRUCTURES_BATCH_TOTAL: n_struct,
        },
        predictions={constants.PREDICT_TOTAL_ENERGY: e_pred},
    )
    # per-atom errors: 0.3 (region 2 -> 0.12), 0.05 (region 1 -> 0.05)
    expected = (0.12 + 0.05) / 2.0
    assert float(out.numpy()) == pytest.approx(expected, rel=1e-10)


def test_epa_loss_invalid_thresholds_raise():
    with pytest.raises(AssertionError):
        WeightedPiecewiseLinearEnergyPerAtomLoss(
            loss_component_weight=1.0, delta_1=0.1, delta_2=0.05
        )


def test_epa_loss_invalid_slopes_raise():
    with pytest.raises(AssertionError):
        WeightedPiecewiseLinearEnergyPerAtomLoss(
            loss_component_weight=1.0, beta_1=0.1, beta_2=1.0
        )


# --- Stress variant ---

def _build_stress_loss(use_frobenius_norm: bool):
    loss = WeightedPiecewiseLinearStressLoss(
        loss_component_weight=1.0,
        delta_1=D1,                # reuse force defaults so test math matches
        delta_2=D2,
        beta_1=B1,
        beta_2=B2,
        beta_3=B3,
        use_frobenius_norm=use_frobenius_norm,
    )
    loss.build(tf.float64)
    return loss


def test_stress_loss_componentwise_known_residuals():
    # 3 structures, virial deliberately spans all 3 piecewise regions when
    # divided by volume=1.0. Residual placed in the first Voigt component;
    # other 5 components are zero.
    v_true = tf.zeros((3, 6), dtype=tf.float64)
    v_pred = tf.constant(
        [
            [0.05, 0.0, 0.0, 0.0, 0.0, 0.0],   # region 1: |x|=0.05
            [0.30, 0.0, 0.0, 0.0, 0.0, 0.0],   # region 2: |x|=0.30
            [1.00, 0.0, 0.0, 0.0, 0.0, 0.0],   # region 3: |x|=1.00
        ],
        dtype=tf.float64,
    )
    v_weight = tf.ones((3, 1), dtype=tf.float64)
    volume = tf.ones((3, 1), dtype=tf.float64)

    loss = _build_stress_loss(use_frobenius_norm=False)
    out = loss.compute_loss_component(
        input_data={
            constants.DATA_REFERENCE_VIRIAL: v_true,
            constants.DATA_VIRIAL_WEIGHTS: v_weight,
            constants.DATA_VOLUME: volume,
        },
        predictions={constants.PREDICT_VIRIAL: v_pred},
    )
    # piecewise values per nonzero residual: 0.05, 0.12, 0.145
    # plus 15 zero residuals contributing 0
    # divisor = sum(weight) * 6 = 3 * 6 = 18
    expected = (0.05 + 0.12 + 0.145) / 18.0
    assert float(out.numpy()) == pytest.approx(expected, rel=1e-10)


def test_stress_loss_frobenius_norm_known_residuals():
    # Same residual magnitudes as above, but Frobenius reduces each row to a
    # single per-structure scalar |x|.
    v_true = tf.zeros((3, 6), dtype=tf.float64)
    v_pred = tf.constant(
        [
            [0.05, 0.0, 0.0, 0.0, 0.0, 0.0],
            [0.30, 0.0, 0.0, 0.0, 0.0, 0.0],
            [1.00, 0.0, 0.0, 0.0, 0.0, 0.0],
        ],
        dtype=tf.float64,
    )
    v_weight = tf.ones((3, 1), dtype=tf.float64)
    volume = tf.ones((3, 1), dtype=tf.float64)

    loss = _build_stress_loss(use_frobenius_norm=True)
    out = loss.compute_loss_component(
        input_data={
            constants.DATA_REFERENCE_VIRIAL: v_true,
            constants.DATA_VIRIAL_WEIGHTS: v_weight,
            constants.DATA_VOLUME: volume,
        },
        predictions={constants.PREDICT_VIRIAL: v_pred},
    )
    # one piecewise value per structure; divisor = sum(weight) = 3 (no x6)
    expected = (0.05 + 0.12 + 0.145) / 3.0
    assert float(out.numpy()) == pytest.approx(expected, rel=1e-6)


def test_stress_loss_volume_division():
    # Residual is large (1.0) in virial but volume=10 → stress error 0.1 → region 2
    v_true = tf.zeros((1, 6), dtype=tf.float64)
    v_pred = tf.constant([[1.0, 0.0, 0.0, 0.0, 0.0, 0.0]], dtype=tf.float64)
    v_weight = tf.ones((1, 1), dtype=tf.float64)
    volume = tf.constant([[10.0]], dtype=tf.float64)

    loss = _build_stress_loss(use_frobenius_norm=False)
    out = loss.compute_loss_component(
        input_data={
            constants.DATA_REFERENCE_VIRIAL: v_true,
            constants.DATA_VIRIAL_WEIGHTS: v_weight,
            constants.DATA_VOLUME: volume,
        },
        predictions={constants.PREDICT_VIRIAL: v_pred},
    )
    # |x| = 1.0/10 = 0.1 = delta_1 -> piecewise value = beta_1 * delta_1 = 0.1
    # divisor = 1 * 6
    expected = 0.1 / 6.0
    assert float(out.numpy()) == pytest.approx(expected, rel=1e-10)


def test_stress_loss_invalid_thresholds_raise():
    with pytest.raises(AssertionError):
        WeightedPiecewiseLinearStressLoss(
            loss_component_weight=1.0, delta_1=0.1, delta_2=0.05
        )


def test_stress_loss_invalid_slopes_raise():
    with pytest.raises(AssertionError):
        WeightedPiecewiseLinearStressLoss(
            loss_component_weight=1.0, beta_3=0.0
        )
