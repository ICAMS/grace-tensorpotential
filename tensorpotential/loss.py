from __future__ import annotations

from abc import ABC, abstractmethod

import tensorflow as tf

from tensorpotential import constants
from tensorpotential.tpmodel import TPModel
from tensorpotential.metrics import ForceMetrics, VirialMetrics


def energy_offset_error(e_true, e_pred, e_weight=None):
    n_vals = tf.shape(e_true)[0]
    # true energy composition
    etr_1 = tf.reshape(tf.tile(e_true, [1, n_vals]), [-1, 1])
    etr_2 = tf.tile(e_true, [n_vals, 1])
    etr_off = tf.abs(etr_1 - etr_2)
    ##################################################################################
    # Or stricktly upper part
    # etr_1 = tf.tile(e_true, [1, n_vals])
    # etr_2 = tf.reshape(tf.tile(e_true, [tf.shape(e_true)[0], 1]), [n_vals, n_vals])
    # etr_off = tf.abs(etr_1 - etr_2)
    # ones = tf.ones_like(etr_1)
    # mask_a = tf.linalg.band_part(
    #     ones, 0, -1
    # )  # Upper triangular matrix of 0s and 1s
    # mask_b = tf.linalg.band_part(ones, 0, 0)  # Diagonal matrix of 0s and 1s
    # mask = tf.cast(mask_a - mask_b, dtype=tf.bool)  # Make a bool mask
    # etr_off = tf.boolean_mask(tf.abs(etr_1 - etr_2), mask)
    ###################################################################################
    # pred energy composition
    epr_1 = tf.reshape(tf.tile(e_pred, [1, n_vals]), [-1, 1])
    epr_2 = tf.tile(e_pred, [n_vals, 1])
    epr_off = tf.abs(epr_1 - epr_2)
    # weights composition
    if e_weight is not None:
        w_1 = tf.reshape(tf.tile(e_weight, [1, n_vals]), [-1, 1])
        w_2 = tf.tile(e_weight, [n_vals, 1])
        w_off = w_1 * w_2
        w_off = tf.where(w_off != 0, tf.sqrt(w_off + 1e-16), w_off)
    else:
        w_off = None

    return etr_off, epr_off, w_off


def huber(error: tf.Tensor, delta=1.0):
    """Computes Huber loss value.

    Args:
        error: tensor of true targets minus predicted targets.
        delta: A float, the point where the Huber loss function changes from a
            quadratic to linear. Defaults to `1.0`.

    Returns:
        Tensor with one scalar loss entry per sample.
    """

    delta = tf.convert_to_tensor(delta, dtype=error.dtype)
    abs_error = tf.abs(error)
    half = tf.convert_to_tensor(0.5, dtype=abs_error.dtype)
    return tf.where(
        abs_error <= delta,
        half * tf.math.square(error),
        delta * abs_error - half * tf.math.square(delta),
    )


def compute_nat_per_structure(input_data):
    nat_per_struc = tf.math.unsorted_segment_sum(
        tf.ones_like(input_data[constants.ATOMS_TO_STRUCTURE_MAP]),
        input_data[constants.ATOMS_TO_STRUCTURE_MAP],
        num_segments=input_data[constants.N_STRUCTURES_BATCH_TOTAL],
    )
    nat_per_struc = tf.reshape(nat_per_struc, [-1, 1])
    # Replace zeros with ones
    nat_per_struc = tf.where(
        tf.equal(nat_per_struc, 0), tf.ones_like(nat_per_struc), nat_per_struc
    )
    return nat_per_struc


def compute_energy_per_atom_error(input_data, predictions):
    e_true = input_data[constants.DATA_REFERENCE_ENERGY]
    e_pred = predictions[constants.PREDICT_TOTAL_ENERGY]

    nat_per_struc = tf.cast(compute_nat_per_structure(input_data), dtype=e_true.dtype)
    e_true /= nat_per_struc
    e_pred /= nat_per_struc

    return e_pred - e_true


class LossComponent(tf.Module, ABC):
    """
    Base class for computing loss components
    """

    input_tensor_spec = {}

    def __init__(
        self, loss_component_weight: float, name: str, normalize_by_samples: bool = True
    ):
        super(LossComponent, self).__init__(name=name)
        self._loss_component_weight = loss_component_weight
        self.normalize_by_samples = normalize_by_samples
        self.corresponding_metrics = None

    def build(self, float_dtype):
        self.loss_component_weight = tf.Variable(
            self._loss_component_weight,
            dtype=float_dtype,
            trainable=False,
            name=f"{self.name}_component_weight",
        )
        self.epsilon = tf.constant(1e-10, dtype=float_dtype)

    # def get_corresponding_metrics(self):
    #     if hasattr(self, "corresponding_metrics"):
    #         return self.corresponding_metrics
    #     else:
    #         return None

    def set_loss_component_weight(self, loss_component_weight: float):
        self.loss_component_weight.assign(loss_component_weight)

    def get_loss_component_weight(self):
        return self.loss_component_weight.numpy()

    def set_loss_component_params(self, params: dict):
        if "weight" in params:
            self.loss_component_weight.assign(params["weight"])

    def __repr__(self):
        if hasattr(self, "loss_component_weight"):
            weight = float(self.loss_component_weight.numpy())
        else:
            weight = float(self._loss_component_weight)
        return f"{weight}*{self.__class__.__name__}"

    @abstractmethod
    def compute_loss_component(self, **kwargs):
        raise NotImplementedError()

    def __call__(
        self,
        input_data: dict[str, tf.Tensor],
        predictions: dict[str, tf.Tensor],
    ) -> tf.Tensor:
        loss_component = self.compute_loss_component(
            input_data=input_data,
            predictions=predictions,
        )

        return loss_component * self.loss_component_weight


class HuberLoss(LossComponent, ABC):
    def __init__(
        self,
        loss_component_weight,
        delta=1.0,
        name="HuberLoss",
        normalize_by_samples: bool = True,
    ):
        super().__init__(
            loss_component_weight=loss_component_weight,
            name=name,
            normalize_by_samples=normalize_by_samples,
        )
        self.delta = delta

    def build(self, float_dtype):
        super().build(float_dtype)
        self.delta = tf.Variable(
            self.delta, dtype=float_dtype, trainable=False, name=self.name + "_delta"
        )

    @abstractmethod
    def compute_loss_component(self, **kwargs):
        raise NotImplementedError()

    def set_loss_component_params(self, params: dict):
        super().set_loss_component_params(params)
        if "delta" in params:
            self.delta.assign(params["delta"])

    def __repr__(self):
        try:
            val = self.delta.numpy()
        except:
            val = self.delta
        return super().__repr__() + f"(delta={val})"


class WeightedOffsetEnergyLoss(LossComponent):
    input_tensor_spec = {
        constants.DATA_REFERENCE_ENERGY: {"shape": [None, 1], "dtype": "float"},
        constants.DATA_ENERGY_WEIGHTS: {"shape": [None, 1], "dtype": "float"},
        constants.N_STRUCTURES_BATCH_TOTAL: {"shape": [], "dtype": "int"},
        constants.ATOMS_TO_STRUCTURE_MAP: {"shape": [None], "dtype": "int"},
    }

    def __init__(
        self,
        loss_component_weight,
        name="WeightedOffsetEnergyLoss",
        normalize_by_samples: bool = True,
    ):
        super(WeightedOffsetEnergyLoss, self).__init__(
            loss_component_weight=loss_component_weight,
            name=name,
            normalize_by_samples=normalize_by_samples,
        )
        self.corresponding_metrics = None

    def compute_loss_component(
        self,
        input_data: dict[str, tf.Tensor],
        predictions: dict[str, tf.Tensor],
        **kwargs,
    ) -> tf.Tensor:
        e_true = input_data[constants.DATA_REFERENCE_ENERGY]
        e_weight = input_data[constants.DATA_ENERGY_WEIGHTS]
        e_pred = predictions[constants.PREDICT_TOTAL_ENERGY]

        nat_per_struc = tf.cast(
            compute_nat_per_structure(input_data), dtype=e_true.dtype
        )
        e_true /= nat_per_struc
        e_pred /= nat_per_struc

        etr_off, epr_off, w_off = energy_offset_error(e_true, e_pred, e_weight)

        delta_e2 = tf.square(etr_off - epr_off) * 0.5
        loss_e = tf.reduce_sum(w_off * delta_e2)
        if self.normalize_by_samples:
            loss_e /= tf.reduce_sum(w_off)

        return loss_e


class WeightedOffsetEnergyHuberLoss(HuberLoss):
    input_tensor_spec = {
        constants.DATA_REFERENCE_ENERGY: {"shape": [None, 1], "dtype": "float"},
        constants.DATA_ENERGY_WEIGHTS: {"shape": [None, 1], "dtype": "float"},
        constants.N_STRUCTURES_BATCH_TOTAL: {"shape": [], "dtype": "int"},
        constants.ATOMS_TO_STRUCTURE_MAP: {"shape": [None], "dtype": "int"},
    }

    def __init__(
        self,
        loss_component_weight,
        delta=1.0,
        name="WeightedOffsetEnergyHuberLoss",
        normalize_by_samples: bool = True,
    ):
        super().__init__(
            loss_component_weight=loss_component_weight,
            delta=delta,
            name=name,
            normalize_by_samples=normalize_by_samples,
        )

    def compute_loss_component(
        self,
        input_data: dict[str, tf.Tensor],
        predictions: dict[str, tf.Tensor],
        **kwargs,
    ) -> tf.Tensor:
        e_true = input_data[constants.DATA_REFERENCE_ENERGY]
        e_pred = predictions[constants.PREDICT_TOTAL_ENERGY]
        e_weight = input_data[constants.DATA_ENERGY_WEIGHTS]

        nat_per_struc = tf.cast(
            compute_nat_per_structure(input_data), dtype=e_true.dtype
        )
        e_true /= nat_per_struc
        e_pred /= nat_per_struc

        etr_off, epr_off, w_off = energy_offset_error(
            e_true=e_true, e_pred=e_pred, e_weight=e_weight
        )

        error = tf.math.subtract(etr_off, epr_off)
        h_loss = huber(error, delta=self.delta)

        loss_e = tf.reduce_sum(0.5 * w_off * h_loss)
        if self.normalize_by_samples:
            loss_e /= tf.reduce_sum(w_off)  # divide by real number of structures
        return loss_e


class WeightedSSEEnergyLoss(LossComponent):
    input_tensor_spec = {
        constants.DATA_REFERENCE_ENERGY: {"shape": [None, 1], "dtype": "float"},
        constants.DATA_ENERGY_WEIGHTS: {"shape": [None, 1], "dtype": "float"},
    }

    def __init__(
        self,
        loss_component_weight,
        name="WeightedSSEEnergyLoss",
        normalize_by_samples: bool = True,
    ):
        super(WeightedSSEEnergyLoss, self).__init__(
            loss_component_weight=loss_component_weight,
            name=name,
            normalize_by_samples=normalize_by_samples,
        )

    def compute_loss_component(
        self,
        input_data: dict[str, tf.Tensor],
        predictions: dict[str, tf.Tensor],
        **kwargs,
    ) -> tf.Tensor:
        e_true = input_data[constants.DATA_REFERENCE_ENERGY]
        e_weight = input_data[constants.DATA_ENERGY_WEIGHTS]
        e_pred = predictions[constants.PREDICT_TOTAL_ENERGY]
        delta_e2 = tf.square(e_pred - e_true)
        loss_e = tf.reduce_sum(e_weight * delta_e2)
        if self.normalize_by_samples:
            loss_e /= tf.reduce_sum(
                e_weight
            )  # divide by sum of weights #=real number of structures

        return loss_e


class WeightedMAEEPALoss(LossComponent):
    input_tensor_spec = {
        constants.DATA_REFERENCE_ENERGY: {"shape": [None, 1], "dtype": "float"},
        constants.DATA_ENERGY_WEIGHTS: {"shape": [None, 1], "dtype": "float"},
        constants.N_STRUCTURES_BATCH_TOTAL: {"shape": [], "dtype": "int"},
        constants.ATOMS_TO_STRUCTURE_MAP: {"shape": [None], "dtype": "int"},
    }

    def __init__(
        self,
        loss_component_weight,
        name="WeightedMAEEPALoss",
        normalize_by_samples: bool = True,
    ):
        super(WeightedMAEEPALoss, self).__init__(
            loss_component_weight=loss_component_weight,
            name=name,
            normalize_by_samples=normalize_by_samples,
        )

    def compute_loss_component(
        self,
        input_data: dict[str, tf.Tensor],
        predictions: dict[str, tf.Tensor],
        **kwargs,
    ) -> tf.Tensor:
        e_weight = input_data[constants.DATA_ENERGY_WEIGHTS]

        delta_epa = compute_energy_per_atom_error(input_data, predictions)
        loss_e = tf.reduce_sum(e_weight * tf.abs(delta_epa + self.epsilon))
        if self.normalize_by_samples:
            loss_e /= tf.reduce_sum(e_weight)
        return loss_e


class WeightedSSEEnergyPerAtomLoss(LossComponent):
    input_tensor_spec = {
        constants.DATA_REFERENCE_ENERGY: {"shape": [None, 1], "dtype": "float"},
        constants.DATA_ENERGY_WEIGHTS: {"shape": [None, 1], "dtype": "float"},
        constants.N_STRUCTURES_BATCH_TOTAL: {"shape": [], "dtype": "int"},
        constants.ATOMS_TO_STRUCTURE_MAP: {"shape": [None], "dtype": "int"},
    }

    def __init__(
        self,
        loss_component_weight,
        name="WeightedSSEEnergyPerAtomLoss",
        normalize_by_samples: bool = True,
    ):
        super(WeightedSSEEnergyPerAtomLoss, self).__init__(
            loss_component_weight=loss_component_weight,
            name=name,
            normalize_by_samples=normalize_by_samples,
        )

    def compute_loss_component(
        self,
        input_data: dict[str, tf.Tensor],
        predictions: dict[str, tf.Tensor],
        **kwargs,
    ) -> tf.Tensor:
        e_true = input_data[constants.DATA_REFERENCE_ENERGY]
        e_weight = input_data[constants.DATA_ENERGY_WEIGHTS]

        nat_per_struc = tf.math.unsorted_segment_sum(
            tf.ones_like(input_data[constants.ATOMS_TO_STRUCTURE_MAP]),
            input_data[constants.ATOMS_TO_STRUCTURE_MAP],
            num_segments=input_data[constants.N_STRUCTURES_BATCH_TOTAL],
        )
        nat_per_struc = tf.cast(tf.reshape(nat_per_struc, [-1, 1]), dtype=e_true.dtype)
        # Replace zeros with ones
        nat_per_struc = tf.where(
            tf.equal(nat_per_struc, 0), tf.ones_like(nat_per_struc), nat_per_struc
        )

        e_pred = predictions[constants.PREDICT_TOTAL_ENERGY]

        delta_e2 = tf.square((e_pred - e_true) / nat_per_struc)
        loss_e = tf.reduce_sum(e_weight * delta_e2)
        if self.normalize_by_samples:
            loss_e /= tf.reduce_sum(e_weight)
        return loss_e


class WeightedSSEForceLoss(LossComponent):
    input_tensor_spec = {
        constants.DATA_REFERENCE_FORCES: {"shape": [None, 3], "dtype": "float"},
        constants.DATA_FORCE_WEIGHTS: {"shape": [None, 1], "dtype": "float"},
    }

    def __init__(
        self,
        loss_component_weight,
        name="WeightedSSEForceLoss",
        normalize_by_samples: bool = True,
        compute_norm: bool = False,
    ):
        super(WeightedSSEForceLoss, self).__init__(
            loss_component_weight=loss_component_weight,
            name=name,
            normalize_by_samples=normalize_by_samples,
        )
        self.compute_norm = compute_norm

    def compute_loss_component(
        self,
        input_data: dict[str, tf.Tensor],
        predictions: dict[str, tf.Tensor],
        **kwargs,
    ) -> tf.Tensor:
        f_true = input_data[constants.DATA_REFERENCE_FORCES]
        f_weight = input_data[constants.DATA_FORCE_WEIGHTS]
        f_pred = predictions[constants.PREDICT_FORCES]
        delta_f2 = tf.square(f_pred - f_true)
        loss_f = tf.reduce_sum(f_weight * delta_f2)
        if self.compute_norm:
            v_true = tf.sqrt(
                tf.reduce_sum(f_true**2, axis=-1, keepdims=True) + self.epsilon
            )
            v_pred = tf.sqrt(
                tf.reduce_sum(f_pred**2, axis=-1, keepdims=True) + self.epsilon
            )
            v_err = v_pred - v_true
            v_h_loss = tf.square(v_err)
            loss_f += tf.reduce_mean(3.0 * f_weight * v_h_loss)

        if self.normalize_by_samples:
            loss_f /= tf.reduce_sum(f_weight) * 3  # divide by real number of atoms * 3
        return loss_f


class WSMAEForceLoss(LossComponent):
    input_tensor_spec = {
        constants.DATA_REFERENCE_FORCES: {"shape": [None, 3], "dtype": "float"},
        constants.DATA_FORCE_WEIGHTS: {"shape": [None, 1], "dtype": "float"},
    }

    def __init__(
        self,
        loss_component_weight,
        name="WSMAEForceLoss",
        normalize_by_samples: bool = True,
        compute_norm: bool = False,
    ):
        super(WSMAEForceLoss, self).__init__(
            loss_component_weight=loss_component_weight,
            name=name,
            normalize_by_samples=normalize_by_samples,
        )
        self.compute_norm = compute_norm
        self.corresponding_metrics = ForceMetrics

    def compute_loss_component(
        self,
        input_data: dict[str, tf.Tensor],
        predictions: dict[str, tf.Tensor],
        **kwargs,
    ) -> tf.Tensor:
        f_true = input_data[constants.DATA_REFERENCE_FORCES]
        f_weight = input_data[constants.DATA_FORCE_WEIGHTS]
        f_pred = predictions[constants.PREDICT_FORCES]
        err = f_pred - f_true
        loss_f = tf.reduce_sum(f_weight * (err * tf.nn.tanh(err / 2)))
        if self.compute_norm:
            v_true = tf.sqrt(
                tf.reduce_sum(f_true**2, axis=-1, keepdims=True) + self.epsilon
            )
            v_pred = tf.sqrt(
                tf.reduce_sum(f_pred**2, axis=-1, keepdims=True) + self.epsilon
            )
            v_err = v_pred - v_true
            v_h_loss = v_err * tf.nn.tanh(v_err / 2)
            loss_f += tf.reduce_mean(3.0 * f_weight * v_h_loss)

        if self.normalize_by_samples:
            loss_f /= tf.reduce_sum(f_weight) * 3  # divide by real number of atoms * 3
        return loss_f


class WeightedSSEVirialLoss(LossComponent):
    input_tensor_spec = {
        constants.DATA_REFERENCE_VIRIAL: {"shape": [None, 6], "dtype": "float"},
        constants.DATA_VIRIAL_WEIGHTS: {"shape": [None, 1], "dtype": "float"},
        constants.DATA_VOLUME: {"shape": [None, 1], "dtype": "float"},
    }

    def __init__(
        self,
        loss_component_weight,
        name="WeightedSSEVirialLoss",
        normalize_by_samples: bool = True,
    ):
        super(WeightedSSEVirialLoss, self).__init__(
            loss_component_weight=loss_component_weight,
            name=name,
            normalize_by_samples=normalize_by_samples,
        )

    def compute_loss_component(
        self,
        input_data: dict[str, tf.Tensor],
        predictions: dict[str, tf.Tensor],
        **kwargs,
    ) -> tf.Tensor:
        v_true = input_data[constants.DATA_REFERENCE_VIRIAL]
        v_weight = input_data[constants.DATA_VIRIAL_WEIGHTS]
        v_pred = predictions[constants.PREDICT_VIRIAL]
        delta_v2 = tf.square(v_pred - v_true)
        loss_v = tf.reduce_sum(v_weight * delta_v2)
        if self.normalize_by_samples:
            loss_v /= (
                tf.reduce_sum(v_weight) * 6
            )  # divide by sum of weights * 6 (Voigt comps)
        return loss_v


class WeightedSSEStressLoss(LossComponent):
    input_tensor_spec = {
        constants.DATA_REFERENCE_VIRIAL: {"shape": [None, 6], "dtype": "float"},
        constants.DATA_VIRIAL_WEIGHTS: {"shape": [None, 1], "dtype": "float"},
        constants.DATA_VOLUME: {"shape": [None, 1], "dtype": "float"},
    }

    def __init__(
        self,
        loss_component_weight,
        name="WeightedSSEStressLoss",
        normalize_by_samples: bool = True,
    ):
        super(WeightedSSEStressLoss, self).__init__(
            loss_component_weight=loss_component_weight,
            name=name,
            normalize_by_samples=normalize_by_samples,
        )

    def compute_loss_component(
        self,
        input_data: dict[str, tf.Tensor],
        predictions: dict[str, tf.Tensor],
        **kwargs,
    ) -> tf.Tensor:
        volume = input_data[constants.DATA_VOLUME]
        v_true = input_data[constants.DATA_REFERENCE_VIRIAL]
        v_weight = input_data[constants.DATA_VIRIAL_WEIGHTS]
        v_pred = predictions[constants.PREDICT_VIRIAL]
        delta_stress_sqr = tf.square((v_pred - v_true) / volume)
        loss_v = tf.reduce_sum(v_weight * delta_stress_sqr)
        if self.normalize_by_samples:
            loss_v /= (
                tf.reduce_sum(v_weight) * 6
            )  # divide by sum of weights * 6 (Voigt comps)
        return loss_v


class WeightedMAEStressLoss(LossComponent):
    input_tensor_spec = {
        constants.DATA_REFERENCE_VIRIAL: {"shape": [None, 6], "dtype": "float"},
        constants.DATA_VIRIAL_WEIGHTS: {"shape": [None, 1], "dtype": "float"},
        constants.DATA_VOLUME: {"shape": [None, 1], "dtype": "float"},
    }

    def __init__(
        self,
        loss_component_weight,
        name="WeightedMAEStressLoss",
        normalize_by_samples: bool = True,
    ):
        super(WeightedMAEStressLoss, self).__init__(
            loss_component_weight=loss_component_weight,
            name=name,
            normalize_by_samples=normalize_by_samples,
        )

    def compute_loss_component(
        self,
        input_data: dict[str, tf.Tensor],
        predictions: dict[str, tf.Tensor],
        **kwargs,
    ) -> tf.Tensor:
        volume = input_data[constants.DATA_VOLUME]
        v_true = input_data[constants.DATA_REFERENCE_VIRIAL]
        v_weight = input_data[constants.DATA_VIRIAL_WEIGHTS]
        v_pred = predictions[constants.PREDICT_VIRIAL]
        delta_stress = tf.abs((v_pred - v_true) / volume + self.epsilon)
        loss_v = tf.reduce_sum(v_weight * delta_stress)
        if self.normalize_by_samples:
            loss_v /= (
                tf.reduce_sum(v_weight) * 6
            )  # divide by sum of weights * 6 (Voigt comps)
        return loss_v


class WeightedHuberEnergyPerAtomLoss(HuberLoss):
    input_tensor_spec = {
        constants.DATA_REFERENCE_ENERGY: {"shape": [None, 1], "dtype": "float"},
        constants.DATA_ENERGY_WEIGHTS: {"shape": [None, 1], "dtype": "float"},
        constants.N_STRUCTURES_BATCH_TOTAL: {"shape": [], "dtype": "int"},
        constants.ATOMS_TO_STRUCTURE_MAP: {"shape": [None], "dtype": "int"},
    }

    def __init__(
        self,
        loss_component_weight,
        delta=1.0,
        name="HuberEnergyLoss",
        normalize_by_samples: bool = True,
    ):
        super().__init__(
            loss_component_weight=loss_component_weight,
            delta=delta,
            name=name,
            normalize_by_samples=normalize_by_samples,
        )

    def compute_loss_component(
        self,
        input_data: dict[str, tf.Tensor],
        predictions: dict[str, tf.Tensor],
        **kwargs,
    ) -> tf.Tensor:
        e_true = input_data[constants.DATA_REFERENCE_ENERGY]
        e_pred = predictions[constants.PREDICT_TOTAL_ENERGY]
        e_weight = input_data[constants.DATA_ENERGY_WEIGHTS]

        nat_per_struc = tf.math.unsorted_segment_sum(
            tf.ones_like(input_data[constants.ATOMS_TO_STRUCTURE_MAP]),
            input_data[constants.ATOMS_TO_STRUCTURE_MAP],
            num_segments=input_data[constants.N_STRUCTURES_BATCH_TOTAL],
        )
        nat_per_struc = tf.cast(tf.reshape(nat_per_struc, [-1, 1]), dtype=e_true.dtype)
        # Replace zeros with ones
        nat_per_struc = tf.where(
            tf.equal(nat_per_struc, 0), tf.ones_like(nat_per_struc), nat_per_struc
        )
        error = tf.math.subtract(e_pred, e_true) / nat_per_struc
        h_loss = huber(error, delta=self.delta)

        loss_e = tf.reduce_sum(e_weight * h_loss)
        if self.normalize_by_samples:
            loss_e /= tf.reduce_sum(e_weight)  # divide by real number of structures
        return loss_e


class WeightedHuberForceLoss(HuberLoss):
    input_tensor_spec = {
        constants.DATA_REFERENCE_FORCES: {"shape": [None, 3], "dtype": "float"},
        constants.DATA_FORCE_WEIGHTS: {"shape": [None, 1], "dtype": "float"},
    }

    def __init__(
        self,
        loss_component_weight,
        delta=1.0,
        name="HuberForceLoss",
        normalize_by_samples: bool = True,
        compute_norm: bool = False,
        scale_w_by_norm: bool = False,
        norm_threshold: float = 25.0,
    ):
        super().__init__(
            loss_component_weight=loss_component_weight,
            delta=delta,
            name=name,
            normalize_by_samples=normalize_by_samples,
        )
        self.compute_norm = compute_norm
        self.scale_w_by_norm = scale_w_by_norm
        self.norm_threshold = norm_threshold

    def build(self, float_dtype):
        super().build(float_dtype)
        if self.scale_w_by_norm:
            self.norm_threshold = tf.Variable(
                self.norm_threshold,
                dtype=float_dtype,
                trainable=False,
                name=self.name + "_norm_threshold",
            )

    def compute_loss_component(
        self,
        input_data: dict[str, tf.Tensor],
        predictions: dict[str, tf.Tensor],
        **kwargs,
    ) -> tf.Tensor:
        f_true = input_data[constants.DATA_REFERENCE_FORCES]
        f_pred = predictions[constants.PREDICT_FORCES]
        f_weight = input_data[constants.DATA_FORCE_WEIGHTS]

        if self.scale_w_by_norm:
            v_true = tf.sqrt(
                tf.reduce_sum(f_true**2, axis=-1, keepdims=True) + self.epsilon
            )
            f_weight = tf.where(
                v_true < self.norm_threshold, f_weight, f_weight / v_true
            )

        error = tf.math.subtract(f_pred, f_true)
        h_loss = huber(error, delta=self.delta)
        loss_f = tf.reduce_sum(f_weight * h_loss)

        if self.compute_norm:
            if not self.scale_w_by_norm:
                v_true = tf.sqrt(
                    tf.reduce_sum(f_true**2, axis=-1, keepdims=True) + self.epsilon
                )
            v_pred = tf.sqrt(
                tf.reduce_sum(f_pred**2, axis=-1, keepdims=True) + self.epsilon
            )
            v_err = v_pred - v_true
            v_h_loss = huber(v_err, delta=self.delta)
            loss_f += tf.reduce_mean(3.0 * f_weight * v_h_loss)

        if self.normalize_by_samples:
            loss_f /= tf.reduce_sum(f_weight) * 3  # divide by real number of atoms * 3
        return loss_f


class WHuberForceCompNormLoss(HuberLoss):
    input_tensor_spec = {
        constants.DATA_REFERENCE_FORCES: {"shape": [None, 3], "dtype": "float"},
        constants.DATA_FORCE_WEIGHTS: {"shape": [None, 1], "dtype": "float"},
    }

    def __init__(
        self,
        loss_component_weight,
        delta=1.0,
        name="HuberForceLoss",
        normalize_by_samples: bool = True,
        compute_norm: bool = False,
    ):
        super().__init__(
            loss_component_weight=loss_component_weight,
            delta=delta,
            name=name,
            normalize_by_samples=normalize_by_samples,
        )
        self.compute_norm = compute_norm
        self.corresponding_metrics = ForceMetrics

    def compute_loss_component(
        self,
        input_data: dict[str, tf.Tensor],
        predictions: dict[str, tf.Tensor],
        **kwargs,
    ) -> tf.Tensor:
        f_true = input_data[constants.DATA_REFERENCE_FORCES]
        f_pred = predictions[constants.PREDICT_FORCES]
        f_weight = input_data[constants.DATA_FORCE_WEIGHTS]

        error = tf.math.subtract(f_pred, f_true)
        h_loss = huber(error, delta=self.delta)
        loss_f = tf.reduce_sum(f_weight * h_loss)
        if self.compute_norm:
            v_true = tf.sqrt(
                tf.reduce_sum(f_true**2, axis=-1, keepdims=True) + self.epsilon
            )
            v_pred = tf.sqrt(
                tf.reduce_sum(f_pred**2, axis=-1, keepdims=True) + self.epsilon
            )
            v_err = v_pred - v_true

            v_h_loss = huber(v_err, delta=self.delta)
            loss_f += tf.reduce_mean(3.0 * f_weight * v_h_loss)
        if self.normalize_by_samples:
            loss_f /= tf.reduce_sum(f_weight) * 3  # divide by real number of atoms * 3
        return loss_f


class WeightedHuberVirialLoss(HuberLoss):
    input_tensor_spec = {
        constants.DATA_REFERENCE_VIRIAL: {"shape": [None, 6], "dtype": "float"},
        constants.DATA_VIRIAL_WEIGHTS: {"shape": [None, 1], "dtype": "float"},
        constants.DATA_VOLUME: {"shape": [None, 1], "dtype": "float"},
    }

    def __init__(
        self,
        loss_component_weight,
        delta=1.0,
        name="HuberVirialLoss",
        normalize_by_samples: bool = True,
    ):
        super().__init__(
            loss_component_weight=loss_component_weight,
            delta=delta,
            name=name,
            normalize_by_samples=normalize_by_samples,
        )

    def compute_loss_component(
        self,
        input_data: dict[str, tf.Tensor],
        predictions: dict[str, tf.Tensor],
        **kwargs,
    ) -> tf.Tensor:
        v_true = input_data[constants.DATA_REFERENCE_VIRIAL]
        v_pred = predictions[constants.PREDICT_VIRIAL]
        v_weight = input_data[constants.DATA_VIRIAL_WEIGHTS]

        error = tf.math.subtract(v_pred, v_true)
        h_loss = huber(error, delta=self.delta)

        loss_v = tf.reduce_sum(v_weight * h_loss)
        if self.normalize_by_samples:
            loss_v /= (
                tf.reduce_sum(v_weight) * 6
            )  # divide  sum of weights * 6 (Voigt comps)
        return loss_v


class WeightedHuberStressLoss(HuberLoss):
    input_tensor_spec = {
        constants.DATA_REFERENCE_VIRIAL: {"shape": [None, 6], "dtype": "float"},
        constants.DATA_VIRIAL_WEIGHTS: {"shape": [None, 1], "dtype": "float"},
        constants.DATA_VOLUME: {"shape": [None, 1], "dtype": "float"},
    }

    def __init__(
        self,
        loss_component_weight,
        delta=1.0,
        name="HuberStressLoss",
        normalize_by_samples: bool = True,
        compute_invariants: bool = False,
    ):
        super().__init__(
            loss_component_weight=loss_component_weight,
            delta=delta,
            name=name,
            normalize_by_samples=normalize_by_samples,
        )
        self.compute_invariants = compute_invariants

    def compute_loss_component(
        self,
        input_data: dict[str, tf.Tensor],
        predictions: dict[str, tf.Tensor],
        **kwargs,
    ) -> tf.Tensor:
        v_true = input_data[constants.DATA_REFERENCE_VIRIAL]
        # xx,yy,zz, xy,xz,yz
        v_pred = predictions[constants.PREDICT_VIRIAL]
        v_weight = input_data[constants.DATA_VIRIAL_WEIGHTS]
        volume = input_data[constants.DATA_VOLUME]

        v_true /= volume
        v_pred /= volume

        error = tf.math.subtract(v_pred, v_true)
        h_loss = huber(error, delta=self.delta)
        loss_v = tf.reduce_sum(v_weight * h_loss)

        if self.compute_invariants:
            I_1_t = tf.reduce_sum(v_true[:, :3], axis=1, keepdims=True)
            I_1_p = tf.reduce_sum(v_pred[:, :3], axis=1, keepdims=True)
            I_1_loss = huber(I_1_t - I_1_p, delta=self.delta)

            I_2_t = tf.reshape(
                (
                    v_true[:, 0] * v_true[:, 1]
                    + v_true[:, 1] * v_true[:, 2]
                    + v_true[:, 0] * v_true[:, 2]
                    - v_true[:, 3] ** 2
                    - v_true[:, 4] ** 2
                    - v_true[:, 5] ** 2
                ),
                [-1, 1],
            )
            I_2_p = tf.reshape(
                (
                    v_pred[:, 0] * v_pred[:, 1]
                    + v_pred[:, 1] * v_pred[:, 2]
                    + v_pred[:, 0] * v_pred[:, 2]
                    - v_pred[:, 3] ** 2
                    - v_pred[:, 4] ** 2
                    - v_pred[:, 5] ** 2
                ),
                [-1, 1],
            )
            I_2_loss = huber(I_2_t - I_2_p, delta=self.delta)

            I_3_t = tf.reshape(
                (
                    v_true[:, 0] * v_true[:, 1] * v_true[:, 2]
                    + 2 * v_true[:, 3] * v_true[:, 4] * v_true[:, 5]
                    - v_true[:, 1] * v_true[:, 4] ** 2
                    - v_true[:, 2] * v_true[:, 3] ** 2
                    - v_true[:, 0] * v_true[:, 5] ** 2
                ),
                [-1, 1],
            )
            I_3_p = tf.reshape(
                (
                    v_pred[:, 0] * v_pred[:, 1] * v_pred[:, 2]
                    + 2 * v_pred[:, 3] * v_pred[:, 4] * v_pred[:, 5]
                    - v_pred[:, 1] * v_pred[:, 4] ** 2
                    - v_pred[:, 2] * v_pred[:, 3] ** 2
                    - v_pred[:, 0] * v_pred[:, 5] ** 2
                ),
                [-1, 1],
            )
            I_3_loss = huber(I_3_t - I_3_p, delta=self.delta)

            loss_v += tf.reduce_sum(v_weight * I_1_loss) / 9.0
            loss_v += tf.reduce_sum(v_weight * I_2_loss) / 18.0
            loss_v += tf.reduce_sum(v_weight * I_3_loss) / 18.0

        if self.normalize_by_samples:
            loss_v /= (
                tf.reduce_sum(v_weight) * 6
            )  # divide  sum of weights * 6 (Voigt comps)
        return loss_v


class RegularizationLossComponent(LossComponent):
    def __call__(self, model: TPModel) -> tf.Tensor:
        loss_component = self.compute_loss_component(model=model)

        return loss_component * self.loss_component_weight


class L2Loss(RegularizationLossComponent):
    def __init__(self, loss_component_weight, name="L2Loss"):
        super(L2Loss, self).__init__(
            loss_component_weight=loss_component_weight, name=name
        )

    def compute_loss_component(self, model: TPModel) -> tf.Tensor:
        return model.compute_l2_regularization_loss()

    def __repr__(self):
        return f"{self.loss_component_weight}*L2Loss"


class LossFunction:
    def __init__(
        self,
        loss_components: dict[str, LossComponent | RegularizationLossComponent],
        name: str = "total_loss",
    ):
        self.name = name
        self.loss_components = loss_components
        self.input_signatures = {}

    def build(self, float_dtype):
        for _, loss_component in self.loss_components.items():
            loss_component.build(float_dtype)

    def get_input_signatures(self):
        for _, loss_component in self.loss_components.items():
            self.input_signatures.update(loss_component.input_tensor_spec)

        return self.input_signatures

    def set_loss_component_weights(self, loss_component_weights: dict[str, float]):
        for k, loss_component_weight in loss_component_weights.items():
            assert (
                k in self.loss_components
            ), f"{k} not in loss_components ({self.loss_components.keys()})"
            self.loss_components[k].set_loss_component_weight(loss_component_weight)

    def get_loss_component_weights(self):
        return {
            k: self.loss_components[k].get_loss_component_weight()
            for k, loss_component in self.loss_components.items()
        }

    def set_loss_component_params(self, loss_component_params: dict[str, dict]):
        for k, kwarg in loss_component_params.items():
            assert (
                k in self.loss_components
            ), f"{k} not in loss_components ({self.loss_components.keys()})"
            self.loss_components[k].set_loss_component_params(kwarg)

    def __call__(
        self,
        input_data: dict[str, tf.Tensor] = None,
        predictions: dict[str, tf.Tensor] = None,
        model: TPModel = None,
    ):
        result = {self.name: 0.0}
        for k, loss_component in self.loss_components.items():
            if isinstance(loss_component, RegularizationLossComponent):
                lc = loss_component(model)
                result[k] = lc
                result[self.name] += lc
            elif isinstance(loss_component, LossComponent):
                lc = loss_component(
                    input_data=input_data,
                    predictions=predictions,
                )
                result[k] = lc
                result[self.name] += lc
            else:
                raise ValueError(f"Unknown loss type {loss_component}")

        return result

    def __repr__(self):
        names = [
            loss_component.__repr__() + f"({k})"
            for k, loss_component in self.loss_components.items()
        ]
        name = " + ".join(names)
        return f"Loss({name})"
