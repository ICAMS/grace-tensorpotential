from __future__ import annotations

from abc import ABC, abstractmethod

import tensorflow as tf

from tensorpotential import constants
from tensorpotential.tpmodel import TPModel


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
    ):
        super(WeightedSSEForceLoss, self).__init__(
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
        f_true = input_data[constants.DATA_REFERENCE_FORCES]
        f_weight = input_data[constants.DATA_FORCE_WEIGHTS]
        f_pred = predictions[constants.PREDICT_FORCES]
        delta_f2 = tf.square(f_pred - f_true)
        loss_f = tf.reduce_sum(f_weight * delta_f2)
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


class HuberLoss(LossComponent):
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
        f_true = input_data[constants.DATA_REFERENCE_FORCES]
        f_pred = predictions[constants.PREDICT_FORCES]
        f_weight = input_data[constants.DATA_FORCE_WEIGHTS]

        error = tf.math.subtract(f_pred, f_true)
        h_loss = huber(error, delta=self.delta)

        loss_f = tf.reduce_sum(f_weight * h_loss)
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
        volume = input_data[constants.DATA_VOLUME]

        error = tf.math.subtract(v_pred, v_true) / volume
        h_loss = huber(error, delta=self.delta)

        loss_v = tf.reduce_sum(v_weight * h_loss)
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
