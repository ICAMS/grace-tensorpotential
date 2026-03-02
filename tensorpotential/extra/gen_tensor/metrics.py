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

        d_tensor = (
            input_data[cc.DATA_REFERENCE_TENSOR] - predictions[cc.PREDICT_TENSOR]
        )

        abs_tensor_reduce = tf.reduce_sum(tf.math.abs(d_tensor))

        sqr_tensor_reduce = tf.reduce_sum(d_tensor**2)

        return {
            "abs/d_tensor/per_struct": abs_tensor_reduce,
            "sqr/d_tensor/per_struct": sqr_tensor_reduce,
            "nat/per_struct": nat_per_struct,
        }

    @property
    def normalization_spec(self) -> dict[str, dict]:
        return {
            "abs/d_tensor/per_struct": {"norm": "n_atoms", "factor": 1.0},
            "sqr/d_tensor/per_struct": {"norm": "n_atoms", "factor": 1.0},
        }
