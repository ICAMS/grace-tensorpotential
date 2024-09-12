import gc
import logging
import os
import sys
from dataclasses import dataclass

from typing import Dict, List

from tqdm import tqdm

from tensorpotential.potentials.presets import (
    LINEAR,
    MLP,
    FS,
    GRACE_1LAYER,
    GRACE_2LAYER,
)
from tensorpotential.loss import *
from tensorpotential.metrics import *
from tensorpotential.instructions.base import (
    str_to_class,
)

from tensorpotential.tpmodel import (
    ComputeBatchEnergyAndForces,
    ComputeBatchEnergyForcesVirials,
    ComputeStructureEnergyAndForcesAndVirial,
    TrainFunction,
    ComputeFunction,
)

LOG_FMT = "%(asctime)s %(levelname).1s - %(message)s"
logging.basicConfig(level=logging.INFO, format=LOG_FMT, datefmt="%Y/%m/%d %H:%M:%S")
log = logging.getLogger()


@dataclass
class ModelFunctions:
    loss_fn: LossFunction
    regularization_loss_fn: LossFunction
    train_fn: TrainFunction
    compute_fn: ComputeFunction
    metrics_fn: ComputeMetrics


def construct_model(
    potential_config: Dict,
    element_map: Dict,
    rcut: float,
    avg_n_neigh=1.0,
    constant_out_shift: float = 0.0,
    constant_out_scale: float = 1.0,
    atomic_shift_map=None,
) -> List:
    preset = potential_config.get("preset")
    kwargs = potential_config.get("kwargs", {})
    if kwargs:
        log.info(f"Model kwargs: {kwargs}")
    if preset is not None:
        # TODO - need better names
        ALLOWED_PRESETS = {
            "LINEAR": LINEAR,
            "FS": FS,
            "MLP": MLP,
            "GRACE_1LAYER": GRACE_1LAYER,
            "GRACE_2LAYER": GRACE_2LAYER,
        }
        assert (
            preset in ALLOWED_PRESETS
        ), f"Preset `{preset}` must be in {ALLOWED_PRESETS.keys()}"
        build_fn = ALLOWED_PRESETS[preset]
        instructions_list = build_fn(
            element_map=element_map,
            rcut=rcut,
            avg_n_neigh=avg_n_neigh,
            constant_out_shift=constant_out_shift,
            constant_out_scale=constant_out_scale,
            atomic_shift_map=atomic_shift_map,
            **kwargs,
        )
    elif potential_config.get("custom"):
        model = potential_config.get("custom")
        sys.path.append(os.getcwd())
        log.info(f"Importing model from {model} (added to PYTHON_PATH: {os.getcwd()})")
        build_model_fn = str_to_class(model)
        instructions_list = build_model_fn(
            element_map=element_map,
            rcut=rcut,
            avg_n_neigh=avg_n_neigh,
            constant_out_shift=constant_out_shift,
            constant_out_scale=constant_out_scale,
            atomic_shift_map=atomic_shift_map,
            **kwargs,
        )
    else:
        raise ValueError(
            f"Neither `preset` nor `custom` specified in input.yaml::potential"
        )
    return instructions_list


def convert_to_tensors_for_model(
    tp,
    train_data,
    test_data,
):
    """
    Converts a train and test data into a tf.Tensor using only keys from tp.model's signature

    :param tp: TensorPotential instance
    :param train_data:
    :param test_data:
    :return: tuple of train data and test data
    """
    sigs = tp.get_model_grad_signatures()
    train_batches = []
    with tf.device("CPU"):
        # use deque for train_data in order to efficiently popleft and reduce immediate memory usage
        # spawn gc.collect every 100 batches
        log.info("[TRAIN] Convert to tensors")
        n = len(train_data)
        for it in tqdm(range(n), total=n, mininterval=2):
            b = train_data.popleft()
            converted_batch = {
                k: tf.convert_to_tensor(b[k], dtype=v.dtype) for k, v in sigs.items()
            }
            train_batches.append(converted_batch)
            if (it + 1) % 100 == 0:
                gc.collect()
    # train_batches = tuple(train_batches)
    test_batches = None
    if test_data is not None:
        test_batches = []
        with tf.device("CPU"):
            # use deque for test_data in order to efficiently popleft and reduce immediate memory usage
            # spawn gc.collect every 100 batches
            log.info("[TEST] Convert to tensors")
            n = len(test_data)
            for it in tqdm(range(n), total=n, mininterval=2):
                b = test_data.popleft()
                converted_batch = {
                    k: tf.convert_to_tensor(b[k], dtype=v.dtype)
                    for k, v in sigs.items()
                }
                test_batches.append(converted_batch)
                if (it + 1) % 100 == 0:
                    gc.collect()

        # test_batches = tuple(test_batches)
    return train_batches, test_batches


def build_reg_loss(fit_config):
    """
    Build regularization loss
    Params:
        fit_config (dict)
    Return :
        regularization_loss (LossFunction)
    """
    l2_reg = fit_config["loss"].get("l2_reg", None)
    regularization_loss = None
    if l2_reg:
        regularization_loss = LossFunction({"l2": L2Loss(l2_reg, name="l2_loss")})
        log.info(f"Regularization loss L2={l2_reg}")
    return regularization_loss


def build_loss_function(fit_config):
    """
    Build loss function from fit_config (support backward compat with old format)
    Params:
        fit_config (dict)

    Return:
        loss_function (LossFunction)
    """
    if "loss" not in fit_config:
        energy_loss_weight = fit_config.get("energy_loss_weight", 1.0)
        forces_loss_weight = fit_config.get("forces_loss_weight", 100.0)
        fit_config["loss"] = {
            "energy": {"type": "square", "weight": energy_loss_weight},
            "forces": {"type": "square", "weight": forces_loss_weight},
        }
        # old-style configuration
        log.warning(
            "DEPRECATION WARNING! Instead of `energy_loss_weight` and `forces_loss_weight`"
            " use input.yaml::fit::loss dict (see doc)"
        )
    fit_loss = fit_config["loss"]

    def get_loss_component(key, loss_types_dict):
        loss_config = fit_loss[key].copy()
        loss_weight = loss_config.pop("weight")
        loss_type = loss_config.pop("type", "square")

        if loss_type not in loss_types_dict:
            raise ValueError(f"Unknown type of loss function {loss_type}")
        return loss_types_dict[loss_type](loss_weight, **loss_config)

    loss_components = {}
    loss_comp_name = "energy"
    if loss_comp_name in fit_loss:
        loss_components[loss_comp_name] = get_loss_component(
            loss_comp_name,
            {
                "square": WeightedSSEEnergyPerAtomLoss,
                "huber": WeightedHuberEnergyPerAtomLoss,
            },
        )

    loss_comp_name = "forces"
    if loss_comp_name in fit_loss:
        loss_components[loss_comp_name] = get_loss_component(
            loss_comp_name,
            {
                "square": WeightedSSEForceLoss,
                "huber": WeightedHuberForceLoss,
            },
        )

    loss_comp_name = "virial"
    if loss_comp_name in fit_loss:
        loss_components[loss_comp_name] = get_loss_component(
            loss_comp_name,
            {
                "square": WeightedSSEVirialLoss,
                "huber": WeightedHuberVirialLoss,
            },
        )

    loss_comp_name = "stress"
    if loss_comp_name in fit_loss:
        loss_components[loss_comp_name] = get_loss_component(
            loss_comp_name,
            {
                "square": WeightedSSEStressLoss,
                "huber": WeightedHuberStressLoss,
            },
        )

    loss_func: LossFunction = LossFunction(loss_components=loss_components)

    reg_loss: LossFunction = build_reg_loss(fit_config)
    return loss_func, reg_loss


def construct_model_functions(fit_config):
    loss_func, regularization_loss = build_loss_function(fit_config)
    is_fit_stress = (
        "stress" in loss_func.loss_components or "virial" in loss_func.loss_components
    )

    model_train_function: TrainFunction = (
        ComputeBatchEnergyForcesVirials
        if is_fit_stress
        else ComputeBatchEnergyAndForces
    )
    model_compute_function: ComputeFunction = ComputeStructureEnergyAndForcesAndVirial

    # hardcoded for most common case
    metrics = [EnergyMetrics(), ForceMetrics()]
    if is_fit_stress:
        metrics.append(VirialMetrics())
    compute_metrics = ComputeMetrics(metrics=metrics)

    model_functions = ModelFunctions(
        loss_fn=loss_func,
        regularization_loss_fn=regularization_loss,
        train_fn=model_train_function,
        compute_fn=model_compute_function,
        metrics_fn=compute_metrics,
    )

    return model_functions
