from __future__ import annotations

import gc
import importlib
import logging
import os
import re
import sys

from tensorpotential.cli.data import FutureDistributedDataset
from tensorpotential.cli.wizard import generate_template_input  # re-export

from dataclasses import dataclass
from typing import Dict, Any

from tqdm import tqdm

from tensorpotential import constants as tc

from tensorpotential.potentials import (
    get_preset,
)

from tensorpotential.loss import *
from tensorpotential.metrics import *
from tensorpotential.instructions.base import (
    str_to_class,
    TPInstruction,
)

from tensorpotential.tpmodel import (
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
    cutoff_dict: dict = None,
    avg_n_neigh: float | dict = 1.0,
    constant_out_shift: float = 0.0,
    constant_out_scale: float = 1.0,
    atomic_shift_map=None,
) -> tuple[dict[str, TPInstruction], list[str] | None]:
    """
    Construct a model as a dict of TPInstructions.
    """
    preset = potential_config.get("preset")
    kwargs = potential_config.get("kwargs", {})
    if kwargs:
        log.info(f"Model kwargs: {kwargs}")

    if preset is not None:
        build_fn = get_preset(preset)

        instructions = build_fn(
            element_map=element_map,
            rcut=rcut,
            cutoff_dict=cutoff_dict,
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
        instructions = build_model_fn(
            element_map=element_map,
            rcut=rcut,
            cutoff_dict=cutoff_dict,
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

    communicated_keys = None
    if hasattr(instructions, "get_instructions"):
        communicated_keys = getattr(instructions, "communicated_keys", None)
        instructions = instructions.get_instructions()

    return instructions, communicated_keys


def convert_to_tensors_for_model(tp, train_data, test_data, strategy):
    """
    Converts a train and test data into a tf.Tensor using only keys from tp.model's signature

    :param tp: TensorPotential instance
    :param train_data:
    :param test_data:
    :return: tuple of train data and test data
    """
    sigs = tp.get_model_grad_signatures()

    if isinstance(train_data, FutureDistributedDataset):
        train_data, test_data = train_data.generate_dataset(
            strategy=strategy, signatures=sigs
        )
        return train_data, test_data

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


def build_metrics_function(fit_config, extra_loss_components=None):
    fit_loss = fit_config["loss"]
    metrics_list = []
    if tc.INPUT_FIT_LOSS_ENERGY in fit_loss:
        metrics_list.append(EnergyMetrics())

    if tc.INPUT_FIT_LOSS_FORCES in fit_loss:
        metrics_list.append(ForceMetrics())

    if tc.INPUT_FIT_LOSS_STRESS in fit_loss or tc.INPUT_FIT_LOSS_VIRIAL in fit_loss:
        metrics_list.append(VirialMetrics())

    if extra_loss_components is not None:
        for loss_component in extra_loss_components:
            extra_metric = loss_component.corresponding_metrics
            if extra_metric is not None:
                metrics_list.append(extra_metric())

    return metrics_list


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
            tc.INPUT_FIT_LOSS_ENERGY: {"type": "square", "weight": energy_loss_weight},
            tc.INPUT_FIT_LOSS_FORCES: {"type": "square", "weight": forces_loss_weight},
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
    loss_comp_name = tc.INPUT_FIT_LOSS_ENERGY
    if loss_comp_name in fit_loss:
        loss_components[loss_comp_name] = get_loss_component(
            loss_comp_name,
            {
                "square": WeightedSSEEnergyPerAtomLoss,
                "huber": WeightedHuberEnergyPerAtomLoss,
                "mae": WeightedMAEEPALoss,
            },
        )

    loss_comp_name = tc.INPUT_FIT_LOSS_FORCES
    if loss_comp_name in fit_loss:
        loss_components[loss_comp_name] = get_loss_component(
            loss_comp_name,
            {
                "square": WeightedSSEForceLoss,
                "huber": WeightedHuberForceLoss,
            },
        )

    loss_comp_name = tc.INPUT_FIT_LOSS_VIRIAL
    if loss_comp_name in fit_loss:
        loss_components[loss_comp_name] = get_loss_component(
            loss_comp_name,
            {
                "square": WeightedSSEVirialLoss,
                "huber": WeightedHuberVirialLoss,
            },
        )

    loss_comp_name = tc.INPUT_FIT_LOSS_STRESS
    if loss_comp_name in fit_loss:
        loss_components[loss_comp_name] = get_loss_component(
            loss_comp_name,
            {
                "square": WeightedSSEStressLoss,
                "huber": WeightedHuberStressLoss,
                "mae": WeightedMAEStressLoss,
            },
        )

    # extra loss components
    extra_loss_componens = fit_loss.get("extra_components", None)

    extras = None
    if extra_loss_componens is not None:
        extras = []
        # mod = importlib.import_module("tensorpotential.experimental.extra_losses")
        from tensorpotential.experimental import extra_losses

        from tensorpotential.extra import extra_losses as extra_losses_extra

        for loss_comp_name, loss_config in extra_loss_componens.items():
            loss_func = getattr(extra_losses, loss_comp_name, None) or getattr(
                extra_losses_extra, loss_comp_name, None
            )
            if loss_func is None:
                raise NameError(f"Could not find loss function {loss_comp_name}")
            loss_weight = loss_config.pop("weight")
            loss_comp = loss_func(loss_weight, **loss_config)
            loss_components[loss_comp_name] = loss_comp
            extras.append(loss_comp)

    loss_func: LossFunction = LossFunction(loss_components=loss_components)

    reg_loss: LossFunction = build_reg_loss(fit_config)
    return loss_func, reg_loss, extras


def construct_model_functions(fit_config):
    loss_func, regularization_loss, extra_components = build_loss_function(fit_config)
    is_fit_stress = (
        tc.INPUT_FIT_LOSS_STRESS in loss_func.loss_components
        or tc.INPUT_FIT_LOSS_VIRIAL in loss_func.loss_components
    )
    if is_fit_stress:
        train_function_name = fit_config.get(
            "train_function", "ComputeBatchEnergyForcesVirials"
        )
    else:
        train_function_name = fit_config.get(
            "train_function", "ComputeBatchEnergyAndForces"
        )
    compute_function_name = fit_config.get(
        "compute_function", "ComputeStructureEnergyAndForcesAndVirial"
    )

    # Try import from default tpmodel.py, then experimental, then extra package
    # Code should work even if experimental package is removed!
    mod = importlib.import_module("tensorpotential.tpmodel")
    mod_exp = importlib.import_module("tensorpotential.experimental.model_computes")
    mod_extra = importlib.import_module("tensorpotential.extra.model_computes")
    compute_function_config = fit_config.get("compute_function_config", {})

    def _resolve_compute_fn(name):
        for m in (mod, mod_exp, mod_extra):
            if hasattr(m, name):
                return getattr(m, name)(compute_function_config=compute_function_config)
        raise NameError(f"Could not find compute function {name}")

    train_fn: TrainFunction = _resolve_compute_fn(train_function_name)
    compute_fn: ComputeFunction = _resolve_compute_fn(compute_function_name)

    metrics = build_metrics_function(fit_config, extra_loss_components=extra_components)
    compute_metrics = ComputeMetrics(metrics=metrics)

    model_functions = ModelFunctions(
        loss_fn=loss_func,
        regularization_loss_fn=regularization_loss,
        train_fn=train_fn,
        compute_fn=compute_fn,
        metrics_fn=compute_metrics,
    )

    return model_functions
