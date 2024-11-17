from __future__ import annotations

import gc
import importlib
import logging
import os
import sys
import re

try:
    import readline

    no_readline = False
except ImportError:
    no_readline = True

from dataclasses import dataclass
from typing import Dict, List, Any

from tqdm import tqdm
from importlib import resources
import json

from tensorpotential import constants as tc
from tensorpotential.potentials.presets import (
    LINEAR,
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
    TrainFunction,
    ComputeFunction,
)


LOG_FMT = "%(asctime)s %(levelname).1s - %(message)s"
logging.basicConfig(level=logging.INFO, format=LOG_FMT, datefmt="%Y/%m/%d %H:%M:%S")
log = logging.getLogger()

# TODO - need better names
ALLOWED_PRESETS = {
    "LINEAR": LINEAR,
    "FS": FS,
    "GRACE_1LAYER": GRACE_1LAYER,
    "GRACE_2LAYER": GRACE_2LAYER,
}
default_preset = "GRACE_1LAYER"
allowed_preset_complexities = {
    "FS": {
        "small": {
            "max_order": 3,
            "lmax": [5, 5, 4],
            "n_rad_max": [20, 15, 10],
            "embedding_size": 32,
            "fs_parameters": [[1.0, 1.0], [1.0, 0.5]],
        },
        "medium": {
            "lmax": [5, 5, 4, 3],
            "n_rad_max": [20, 15, 10, 5],
            "embedding_size": 64,
            "max_order": 4,
            "fs_parameters": [[1.0, 1.0], [1.0, 0.5], [1.0, 2], [1.0, 0.75]],
        },
        "large": {
            "lmax": [5, 5, 4, 3],
            "n_rad_max": [20, 20, 15, 10],
            "embedding_size": 72,
            "max_order": 4,
            "fs_parameters": [
                [1.0, 1.0],
                [1.0, 0.5],
                [1.0, 2],
                [1.0, 0.75],
                [1.0, 1.5],
            ],
        },
    },
    "GRACE_1LAYER": {
        "small": {"lmax": 3, "n_rad_max": 20, "max_order": 3, "n_mlp_dens": 10},
        "medium": {"lmax": 4, "n_rad_max": 32, "max_order": 4, "n_mlp_dens": 12},
        "large": {"lmax": 4, "n_rad_max": 48, "max_order": 4, "n_mlp_dens": 16},
    },
    "GRACE_2LAYER": {
        "small": {
            "lmax": [3, 2],
            "max_order": 3,
            "n_rad_max": [20, 32],
            "n_mlp_dens": 8,
        },
        "medium": {
            "lmax": [3, 3],
            "max_order": 4,
            "n_rad_max": [32, 42],
            "n_mlp_dens": 10,
        },
        "large": {
            "lmax": [4, 3],
            "max_order": 4,
            "n_rad_max": [32, 48],
            "n_mlp_dens": 12,
        },
    },
}
default_cutoff = {"FS": 7.0, "GRACE_1LAYER": 6.0, "GRACE_2LAYER": 5.0}


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
) -> List:
    preset = potential_config.get("preset")
    kwargs = potential_config.get("kwargs", {})
    if kwargs:
        log.info(f"Model kwargs: {kwargs}")

    if preset is not None:
        # assert (
        #     preset in ALLOWED_PRESETS
        # ), f"Preset `{preset}` must be in {ALLOWED_PRESETS.keys()}"
        if preset in ALLOWED_PRESETS:
            build_fn = ALLOWED_PRESETS[preset]
        else:
            try:
                extra_presets = importlib.import_module(
                    "tensorpotential.experimental.presets"
                )
                build_fn = getattr(extra_presets, preset)
            except ModuleNotFoundError:
                build_fn = None
        if build_fn is not None:
            instructions_list = build_fn(
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
            raise ValueError(f"Unknown preset {preset}")
    elif potential_config.get("custom"):
        model = potential_config.get("custom")
        sys.path.append(os.getcwd())
        log.info(f"Importing model from {model} (added to PYTHON_PATH: {os.getcwd()})")
        build_model_fn = str_to_class(model)
        instructions_list = build_model_fn(
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
            extra_metric = loss_component.corresponding_metrics()
            if extra_metric is not None:
                metrics_list.append(extra_metric)
    # if tc.INPUT_FIT_LOSS_EFG in fit_loss:
    #     from tensorpotential.experimental.efg.metrics import AtomicEFGMetrics
    #
    #     metrics_list.append(AtomicEFGMetrics())

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
            },
        )

    # extra loss components
    extra_loss_componens = fit_loss.get("extra_components", None)

    extras = None
    if extra_loss_componens is not None:
        extras = []
        # mod = importlib.import_module("tensorpotential.experimental.extra_losses")
        from tensorpotential.experimental import extra_losses

        for loss_comp_name, loss_config in extra_loss_componens.items():
            try:
                # loss_func = getattr(mod, loss_comp_name)
                loss_func = getattr(extra_losses, loss_comp_name)
                loss_weight = loss_config.pop("weight")
                loss_comp = loss_func(loss_weight, **loss_config)
                loss_components[loss_comp_name] = loss_comp
                extras.append(loss_comp)
            except AttributeError as e:
                raise NameError(f"Could not find loss function {loss_comp_name}")

    # loss_comp_name = tc.INPUT_FIT_LOSS_EFG
    # if loss_comp_name in fit_loss:
    #     from tensorpotential.experimental.efg.loss import WeightedMSEEFGLoss
    #
    #     loss_components[loss_comp_name] = get_loss_component(
    #         loss_comp_name,
    #         {
    #             "square": WeightedMSEEFGLoss,
    #             # "huber": None,
    #         },
    #     )

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

    # Try import from default tpmodel.py, otherwise try experimental package
    # Code should work even if experimental package is removed!
    mod = importlib.import_module("tensorpotential.tpmodel")
    if hasattr(mod, train_function_name):
        train_fn = getattr(mod, train_function_name)
    else:
        mod_exp = importlib.import_module("tensorpotential.experimental.model_computes")
        train_fn: TrainFunction = getattr(mod_exp, train_function_name)

    if hasattr(mod, compute_function_name):
        compute_fn: ComputeFunction = getattr(mod, compute_function_name)
    else:
        mod_exp = importlib.import_module("tensorpotential.experimental.model_computes")
        compute_fn: ComputeFunction = getattr(mod_exp, compute_function_name)

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


def input_choice(query, choices, default_choice):
    """Input from stdin with list of available choices and default choice."""
    choices = list(choices)
    assert default_choice in choices
    while True:
        choice = (
            input(
                query
                + f", available options: {', '.join(choices)} (default = {default_choice}): "
            )
            or default_choice
        )
        if choice in choices:
            break
    return choice


def input_with_default(query, default_choice: Any = None):
    """Input from stdin with default value."""
    choice = input(query + ": ") or default_choice
    return choice


def input_no_default(query):
    """Input from stdin with default value."""
    while True:
        choice = input(query)
        if choice:
            break
    return choice


def generate_template_input():

    print("Generating 'input.yaml'")
    if not no_readline:
        readline.parse_and_bind("tab: complete")

    with resources.open_text("tensorpotential.resources", "input_template.yaml") as f:
        input_yaml_text = f.read()

    # 1. Training set and size
    train_filename = input_no_default(
        "Enter training dataset filename (ex.: data.pkl.gz, [TAB] - autocompletion): "
    )
    input_yaml_text = input_yaml_text.replace("{{TRAIN_FILENAME}}", train_filename)

    test_filename = input_with_default(
        "Enter test dataset filename (ex.: test.pkl.gz, [ENTER] - no separate test dataset)",
    )
    test_size = (
        float(
            input_with_default(
                "Enter test set fraction (default = 0.05)", default_choice=0.05
            )
        )
        if not test_filename
        else None
    )

    if test_filename:
        input_yaml_text = input_yaml_text.replace(
            "{{TEST_DATA}}", f"""test_filename: {test_filename}"""
        )
    elif test_size is not None and test_size > 0:
        input_yaml_text = input_yaml_text.replace(
            "{{TEST_DATA}}", f"""test_size: {test_size}"""
        )
    else:
        raise ValueError("Either test_filename or test_size must be specified.")

    # 2. Elements
    determine_elements_from_dataset = False
    elements_str = input(
        """Please enter list of elements (ex.: "Cu", "AlNi", [ENTER] - determine from dataset): """
    )
    if elements_str:
        patt = re.compile("([A-Z][a-z]?)")
        elements = patt.findall(elements_str)
        elements = sorted(elements)
        determine_elements_from_dataset = False
        print("Number of elements: ", len(elements))
        print("Elements: ", elements)
        elements = f"elements: [{elements}]"
    else:
        # determine from training set
        determine_elements_from_dataset = True
        elements = ""
    input_yaml_text = input_yaml_text.replace("{{ELEMENTS}}", elements)

    preset_name = input_choice(
        f"Enter model preset",
        choices=ALLOWED_PRESETS.keys(),
        default_choice=default_preset,
    )
    input_yaml_text = input_yaml_text.replace("{{PRESET_NAME}}", str(preset_name))

    preset_complexity = input_choice(
        "Model complexity",
        choices=allowed_preset_complexities[preset_name].keys(),
        default_choice="medium",
    )
    kwargs_str = json.dumps(
        allowed_preset_complexities[preset_name][preset_complexity],
    ).strip()
    kwargs_str = kwargs_str.replace('"', "").replace("'", "")
    input_yaml_text = input_yaml_text.replace("{{KWARGS}}", kwargs_str)

    def_cutoff = default_cutoff[preset_name]
    cutoff = float(
        input_with_default(
            f"Enter cutoff (default={def_cutoff})",
            default_choice=def_cutoff,
        )
    )
    print("Cutoff: ", cutoff)
    input_yaml_text = input_yaml_text.replace("{{CUTOFF}}", str(cutoff))

    ####### loss function type ###
    loss_type = input_choice(
        "Loss function type", choices=["square", "huber"], default_choice="huber"
    )
    input_yaml_text = input_yaml_text.replace("{{LOSS_TYPE}}", loss_type)

    huber_delta = (
        input_with_default(
            "For huber loss, please enter delta (default = 0.01)", default_choice=0.01
        )
        if loss_type == "huber"
        else None
    )
    extra_loss_args = f", delta: {huber_delta}" if loss_type == "huber" else ""
    input_yaml_text = input_yaml_text.replace("{{EXTRA_E_ARGS}}", extra_loss_args)

    ####### force loss weight ###
    print("Energy loss weight is equal to 1")
    force_loss_weight = str(
        input_with_default("Enter force loss weight (default = 5)", default_choice=5)
    )
    input_yaml_text = input_yaml_text.replace(
        "{{FORCE_LOSS_WEIGHT}}", force_loss_weight
    )

    ####### stress loss weight ###
    stress_loss_weight = input_with_default(
        "Enter stress loss weight (default = None)", default_choice=None
    )
    if stress_loss_weight is not None:
        stress_loss_str = f"stress: {{ weight: {stress_loss_weight}, type: {loss_type} {extra_loss_args} }},"
        input_yaml_text = input_yaml_text.replace("{{STRESS_LOSS}}", stress_loss_str)
    else:
        input_yaml_text = input_yaml_text.replace("{{STRESS_LOSS}}", "")

    switch_after = input_with_default(
        "Switch loss function E:F:S weights after certain number of epochs (total number of epochs = 500)? If yes - enter number of epochs (default = None)",
        default_choice=None,
    )
    if switch_after is not None:
        new_learning_rate = input_with_default(
            "Learning rate after switching (default = 0.001)", default_choice=0.001
        )
        new_energy_weight = input_with_default(
            "Energy weight after switching (old value = 1, default = 5)",
            default_choice=5,
        )
        new_force_weight = input_with_default(
            f"Force weight after switching (old value = {force_loss_weight}, default = 2)",
            default_choice=2,
        )

        switch_expression_str = (
            f"""switch: {{ after_iter: {switch_after}, learning_rate: {new_learning_rate}, """
            + f"""energy: {{ weight: {new_energy_weight} }}, """
            + f"""forces: {{ weight: {new_force_weight} }}, """
        )

        if stress_loss_weight is not None:
            new_stress_weight = input_with_default(
                f"Stress weight after switching (old value = {stress_loss_weight}, default = same value)",
                default_choice=stress_loss_weight,
            )
            switch_expression_str += f"""stress: {{ weight: {new_stress_weight} }}, """
        switch_expression_str += "}"
    else:
        switch_expression_str = ""
    input_yaml_text = input_yaml_text.replace("{{SWITCH_LOSS}}", switch_expression_str)

    # weighting scheme
    weighting_inp = input_choice(
        "Enter weighting scheme type",
        choices=["uniform", "energy"],
        default_choice="uniform",
    )

    if weighting_inp == "energy":
        weighting = """weighting: { type: energy_based, DElow: 1.0, DEup: 10.0, DFup: 50.0, DE: 1.0, DF: 1.0, wlow: 0.75, energy: convex_hull, seed: 42}"""
        print("Use energy-based sample weighting: ", weighting)
        # also enable compute_convex_hull: True option
        input_yaml_text = input_yaml_text.replace("{{COMPUTE_CONVEX_HULL}}", "True")
    else:
        weighting = ""
        print("Use uniform-based sample weighting")
        input_yaml_text = input_yaml_text.replace("{{COMPUTE_CONVEX_HULL}}", "False")

    input_yaml_text = input_yaml_text.replace("{{WEIGHTING_SCHEME}}", weighting)

    batch_size = input_with_default(
        "Enter batch size (default = 32)", default_choice=32
    )
    input_yaml_text = input_yaml_text.replace("{{BATCH_SIZE}}", str(batch_size))
    input_yaml_text = input_yaml_text.replace(
        "{{TEST_BATCH_SIZE}}", str(4 * int(batch_size))
    )

    with open("input.yaml", "w") as f:
        print(input_yaml_text, file=f)
    print("Input file is written into `input.yaml`")
    sys.exit(0)
