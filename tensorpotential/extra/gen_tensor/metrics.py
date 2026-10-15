from tensorpotential.metrics import AbstractMetrics
from tensorpotential.extra.gen_tensor import constants as cc
from tensorpotential import constants

import tensorflow as tf


class TensorMetrics(AbstractMetrics):

    input_tensor_spec = {
        constants.N_ATOMS_BATCH_REAL: {"shape": [], "dtype": "int"},
        constants.N_STRUCTURES_BATCH_REAL: {"shape": [], "dtype": "int"},
        constants.ATOMS_TO_STRUCTURE_MAP: {"shape": [None], "dtype": "int"},
        cc.DATA_REFERENCE_TENSOR: {
            "shape": [None, None],
            "dtype": "float",
        },
        cc.DATA_REFERENCE_TENSOR_WEIGHT: {
            "shape": [None, 1],
            "dtype": "float",
        },
    }

    def __call__(
        self, input_data: dict[str, tf.Tensor], predictions: dict[str, tf.Tensor]
    ) -> dict[str, tf.Tensor]:
        """Compute tensor related metrics."""
        tot_nat_real = input_data[constants.N_ATOMS_BATCH_REAL]
        tot_struc_real = input_data[constants.N_STRUCTURES_BATCH_REAL]
        map_at2struc = input_data[constants.ATOMS_TO_STRUCTURE_MAP][:tot_nat_real]

        nat_per_struct = tf.math.unsorted_segment_sum(
            tf.reshape(tf.ones_like(map_at2struc), [-1, 1]),
            map_at2struc,
            num_segments=tot_struc_real,
        )

        ref_tensor = input_data[cc.DATA_REFERENCE_TENSOR]
        # rows are atoms (per-atom tensors) or structures (per-structure tensors);
        # padding rows have zero weight and are left out
        mask = tf.cast(input_data[cc.DATA_REFERENCE_TENSOR_WEIGHT] > 0, ref_tensor.dtype)
        d_tensor = (ref_tensor - predictions[cc.PREDICT_TENSOR]) * mask
        n_components = tf.reduce_sum(mask) * tf.cast(
            tf.shape(ref_tensor)[1], ref_tensor.dtype
        )

        return {
            "abs/d_tensor/per_struct": tf.reduce_sum(tf.math.abs(d_tensor)),
            "sqr/d_tensor/per_struct": tf.reduce_sum(d_tensor**2),
            "n/d_tensor/per_struct": n_components,
            "nat/per_struct": nat_per_struct,
        }

    @property
    def normalization_spec(self) -> dict[str, dict]:
        # mean over all real tensor components (per atom or per structure)
        return {
            "abs/d_tensor/per_struct": {"norm": "n/d_tensor/per_struct"},
            "sqr/d_tensor/per_struct": {"norm": "n/d_tensor/per_struct"},
        }
