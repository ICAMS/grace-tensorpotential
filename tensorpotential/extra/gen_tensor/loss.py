from tensorpotential.loss import LossComponent, huber
from tensorpotential.extra.gen_tensor import constants as tensor_constants
from tensorpotential.extra.gen_tensor.metrics import TensorMetrics
from tensorflow import reduce_sum, reduce_mean, Tensor
import tensorflow as tf


class WeightedTensorLoss(LossComponent):
    input_tensor_spec = {
        tensor_constants.DATA_REFERENCE_TENSOR: {
            "shape": [None, None],
            "dtype": "float",
        },
        tensor_constants.DATA_REFERENCE_TENSOR_WEIGHT: {
            "shape": [None, 1],
            "dtype": "float",
        },
    }

    def __init__(
        self,
        loss_component_weight,
        name="WeightedTensorLoss",
        type: str = "huber",
        delta: float = 0.01,
        normalize_by_samples: bool = True,
        **kwargs,
    ):
        super(WeightedTensorLoss, self).__init__(
            loss_component_weight=loss_component_weight,
            name=name,
            normalize_by_samples=normalize_by_samples,
        )
        # TODO: Metric
        self.corresponding_metrics = TensorMetrics
        assert type in ["huber", "square"]
        self.type = type
        self.delta = delta

    def compute_loss_component(
        self,
        input_data: dict[str, Tensor],
        predictions: dict[str, Tensor],
        **kwargs,
    ) -> Tensor:
        tensor_true = input_data[tensor_constants.DATA_REFERENCE_TENSOR]
        tensor_weight = input_data[tensor_constants.DATA_REFERENCE_TENSOR_WEIGHT]

        tensor_pred = predictions[tensor_constants.PREDICT_TENSOR]
        error = tensor_true - tensor_pred
        if self.type == "huber":
            error = huber(error, self.delta)
        else:
            error = error**2

        loss = reduce_sum(tensor_weight * reduce_mean(error, axis=[-1], keepdims=True))
        if self.normalize_by_samples:
            loss /= reduce_sum(tensor_weight)
        return loss
