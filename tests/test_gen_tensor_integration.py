import os

import pytest

from .utils import general_integration_test

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

import logging

LOG_FMT = "%(asctime)s %(levelname).1s - %(message)s"
logging.basicConfig(level=logging.INFO, format=LOG_FMT, datefmt="%Y/%m/%d %H:%M:%S")
log = logging.getLogger()


import tensorflow as tf

tf.config.experimental.enable_tensor_float_32_execution(False)
tf.experimental.numpy.experimental_enable_numpy_behavior(dtype_conversion_mode="all")


def test_TENSOR_1L():
    ref_n_epochs = 2

    train_ref_metrics = {
        "total_loss/train": 0.2632034153183147,
        "mae/d_tensor": 0.8581834814793257,
        "rmse/d_tensor": 1.0690765762664578,
        "loss_component/WeightedTensorLoss/train": 0.026320341531831472,
        "epoch": 2.0,
    }

    test_ref_metrics = {
        "total_loss/test": 0.2631642545260056,
        "loss_component/WeightedTensorLoss/test": 0.026316425452600562,
        "mae/d_tensor": 0.8580116481888477,
        "rmse/d_tensor": 1.0689738383206966,
        "epoch": 2.0,
    }

    general_integration_test(
        "MoNbTaW-TENSOR",
        train_ref_metrics=train_ref_metrics,
        test_ref_metrics=test_ref_metrics,
        ref_n_epochs=ref_n_epochs,
    )
