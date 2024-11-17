from __future__ import annotations

from abc import ABC

import tensorflow as tf

from tensorpotential import constants


class AbstractMetrics(ABC):
    input_tensor_spec = {}


class EnergyMetrics(AbstractMetrics):
    input_tensor_spec = {
        constants.N_ATOMS_BATCH_REAL: {"shape": [], "dtype": "int"},
        constants.N_STRUCTURES_BATCH_REAL: {"shape": [], "dtype": "int"},
        constants.ATOMS_TO_STRUCTURE_MAP: {"shape": [None], "dtype": "int"},
        constants.DATA_REFERENCE_ENERGY: {"shape": [None, 1], "dtype": "float"},
    }

    def __call__(
        self, input_data: dict[str, tf.Tensor], predictions: dict[str, tf.Tensor]
    ) -> dict[str, tf.Tensor]:
        """Compute energy related metrics."""
        tot_nat_real = input_data[constants.N_ATOMS_BATCH_REAL]
        tot_struc_real = input_data[constants.N_STRUCTURES_BATCH_REAL]
        map_at2struc = input_data[constants.ATOMS_TO_STRUCTURE_MAP][:tot_nat_real]

        nat_per_struct = tf.math.unsorted_segment_sum(
            tf.reshape(tf.ones_like(map_at2struc), [-1, 1]),
            map_at2struc,
            num_segments=tot_struc_real,
        )

        de = (
            input_data[constants.DATA_REFERENCE_ENERGY]
            - predictions[constants.PREDICT_TOTAL_ENERGY]
        )[:tot_struc_real, :]
        depa = de / tf.cast(nat_per_struct, dtype=de.dtype)

        return {
            "abs/depa/per_struct": tf.math.abs(depa),
            "abs/de/per_struct": tf.math.abs(de),
            "sqr/depa/per_struct": depa**2,
            "sqr/de/per_struct": de**2,
            "nat/per_struct": nat_per_struct,
        }


class ForceMetrics(AbstractMetrics):

    input_tensor_spec = {
        constants.N_ATOMS_BATCH_REAL: {"shape": [], "dtype": "int"},
        constants.N_STRUCTURES_BATCH_REAL: {"shape": [], "dtype": "int"},
        constants.ATOMS_TO_STRUCTURE_MAP: {"shape": [None], "dtype": "int"},
        constants.DATA_REFERENCE_FORCES: {"shape": [None, 3], "dtype": "float"},
    }

    def __call__(
        self, input_data: dict[str, tf.Tensor], predictions: dict[str, tf.Tensor]
    ) -> dict[str, tf.Tensor]:
        """Compute force related metrics."""
        tot_nat_real = input_data[constants.N_ATOMS_BATCH_REAL]
        tot_struc_real = input_data[constants.N_STRUCTURES_BATCH_REAL]
        map_at2struc = input_data[constants.ATOMS_TO_STRUCTURE_MAP][:tot_nat_real]

        df = (
            input_data[constants.DATA_REFERENCE_FORCES]
            - predictions[constants.PREDICT_FORCES]
        )[:tot_nat_real, :]
        abs_f_reduce = tf.math.unsorted_segment_sum(
            tf.reduce_sum(tf.math.abs(df), axis=1),  # reduce by x,y,z comps
            map_at2struc,
            num_segments=tot_struc_real,
        )
        sqr_f_reduce = tf.math.unsorted_segment_sum(
            tf.reduce_sum(df**2, axis=1),  # reduce by x,y,z comps
            map_at2struc,
            num_segments=tot_struc_real,
        )

        return {
            "abs/df/per_struct": abs_f_reduce,
            "sqr/df/per_struct": sqr_f_reduce,
        }


class VirialMetrics(AbstractMetrics):

    input_tensor_spec = {
        constants.N_STRUCTURES_BATCH_REAL: {"shape": [], "dtype": "int"},
        constants.DATA_REFERENCE_VIRIAL: {"shape": [None, 6], "dtype": "float"},
        constants.DATA_VOLUME: {"shape": [None, 1], "dtype": "float"},
    }

    def __call__(
        self, input_data: dict[str, tf.Tensor], predictions: dict[str, tf.Tensor]
    ) -> dict[str, tf.Tensor]:
        """Compute force related metrics."""
        tot_struc_real = input_data[constants.N_STRUCTURES_BATCH_REAL]
        # map_at2struc = input_data[constants.ATOMS_TO_STRUCTURE_MAP][:tot_nat_real]

        delta_virials = (
            input_data[constants.DATA_REFERENCE_VIRIAL]
            - predictions[constants.PREDICT_VIRIAL]
        )[:tot_struc_real, :]
        delta_stresses = (
            delta_virials / input_data[constants.DATA_VOLUME][:tot_struc_real, :]
        )

        # reduce by [xx,yy,zz,xy,xz,yz] comps, /6 to normalize by 6 components
        abs_v_reduce = tf.reduce_sum(tf.math.abs(delta_virials), axis=1, keepdims=True)
        abs_stress_reduce = tf.reduce_sum(
            tf.math.abs(delta_stresses), axis=1, keepdims=True
        )

        # reduce by [xx,yy,zz,xy,xz,yz] comps, /6 to normalize by 6 components
        sqr_v_reduce = tf.reduce_sum(delta_virials**2, axis=1, keepdims=True)
        sqr_stress_reduce = tf.reduce_sum(delta_stresses**2, axis=1, keepdims=True)

        return {
            "abs/dv/per_struct": abs_v_reduce,
            "sqr/dv/per_struct": sqr_v_reduce,
            "abs/stress/per_struct": abs_stress_reduce,
            "sqr/stress/per_struct": sqr_stress_reduce,
        }


class ComputeMetrics:
    input_tensor_spec = {
        constants.DATA_STRUCTURE_ID: {"shape": [None, 1], "dtype": "int"},
    }

    def __init__(
        self,
        metrics: list[callable],
    ):
        self._metrics = metrics
        self.input_signatures = {}

    def get_input_signatures(self):
        self.input_signatures.update(self.input_tensor_spec)
        for metric in self._metrics:
            self.input_signatures.update(metric.input_tensor_spec)
        return self.input_signatures

    def __call__(
        self, input_data: dict[str, tf.Tensor], predictions: dict[str, tf.Tensor]
    ):
        result = {}
        for metric in self._metrics:
            result.update(metric(input_data, predictions))

        return result
