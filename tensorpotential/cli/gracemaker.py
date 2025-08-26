from __future__ import annotations

import argparse
import gc
import logging
import os
import sys

import numpy as np
import yaml
from ase.data import chemical_symbols

from tensorpotential import constants as tc
from tensorpotential.calculator.foundation_models import get_or_download_checkpoint
from tensorpotential.cli.data import load_and_prepare_datasets
from tensorpotential.cli.prepare import (
    construct_model,
    convert_to_tensors_for_model,
    construct_model_functions,
    generate_template_input,
)
from tensorpotential.cli.train import try_load_checkpoint, train_adam, train_bfgs
from tensorpotential.instructions.base import (
    load_instructions,
    save_instructions_dict,
)
from tensorpotential.loss import *
from tensorpotential.metrics import *
from tensorpotential.tensorpot import TensorPotential, get_output_dir
from tensorpotential.utils import convert_model_reduce_elements

MODEL_CONFIG_YAML = "model.yaml"

LOG_FMT = "%(asctime)s %(levelname).1s - %(message)s"
logging.basicConfig(level=logging.INFO, format=LOG_FMT, datefmt="%Y/%m/%d %H:%M:%S")
log = logging.getLogger()


def build_parser():
    parser = argparse.ArgumentParser(
        prog="gracemaker",
        description="Fitting utility for (graph) atomic cluster expansion "
        "potentials.\n",
    )

    parser.add_argument(
        "input",
        help="input YAML file, default: input.yaml",
        nargs="?",
        type=str,
        default="input.yaml",
    )

    parser.add_argument(
        "-l",
        "--log",
        help="log filename, default: log.txt",
        type=str,
        default="log.txt",
    )

    parser.add_argument(
        "-m",
        "--multigpu",
        action="store_true",
        default=False,
        help="Single host/multi GPU distributed fit",
    )

    parser.add_argument(
        "-rl",
        "--restart-latest",
        action="store_true",
        default=False,
        help="Restart from latest checkpoint (use separately from -r/-rs)",
    )

    parser.add_argument(
        "-r",
        "--restart-best-test",
        action="store_true",
        default=False,
        help="Restart from latest best test checkpoint (use separately from -rs/-rl)",
    )

    parser.add_argument(
        "-rs",
        "--restart-suffix",
        default=None,
        type=str,
        dest="restart_suffix",
        help="Suffix of checkpoint to restart from, i.e. .epoch_10  (use separately from -r/-rl)",
    )

    parser.add_argument(
        "-p",
        "--potential",
        type=str,
        help="Potential configuration to load, model.yaml file",
        default=None,
    )

    parser.add_argument(
        "-s",
        "--save-model",
        action="store_true",
        default=False,
        help="Export model as TF saved model",
    )

    parser.add_argument(
        "-sf",
        "--save--fs",
        action="store_true",
        default=False,
        dest="save_fs",
        help="Export FS model as yaml to be loaded in CPP",
    )

    parser.add_argument(
        "-e",
        "--eager",
        action="store_true",
        default=False,
        help="Eager graph execution",
    )

    parser.add_argument(
        "-nj", "--no-jit", action="store_true", default=False, help="No JIT"
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Random seed (will overwrite value from input.yaml)",
    )

    parser.add_argument(
        "-cm",
        "--check-model",
        action="store_true",
        default=False,
        help="Check model consistency, without performing fit",
    )

    parser.add_argument(
        "-t",
        "--template",
        help="Generate a template 'input.yaml' file by dialog",
        dest="template",
        action="store_true",
        default=False,
    )

    parser.add_argument(
        "-cn",
        "--checkpoint-name",
        dest="checkpoint_name",
        type=str,
        default=None,
        help="Explicit name of the checkpoint (omit .index suffix)",
    )

    parser.add_argument(
        "--reset-epoch-and-step",
        dest="reset_epoch_and_step",
        action="store_true",
        default=False,
        help="Reset epoch and step counters from prev. runs",
    )

    return parser


def add_loaded_model_parameter(potential_file_name, args_yaml):
    log.info(f"Loading model config from `{potential_file_name}`")
    with open(potential_file_name, "rt") as f:
        list_of_dict_instructions = yaml.safe_load(f)
    if isinstance(list_of_dict_instructions, dict):
        list_of_dict_instructions = list(list_of_dict_instructions.values())

    for ins_dict in list_of_dict_instructions:
        if "ScalarChemicalEmbedding" in ins_dict["__cls__"]:
            # if not "reduce_elements" in args_yaml::potential,
            # then set original elements map
            if not args_yaml[tc.INPUT_POTENTIAL_SECTION].get(
                tc.INPUT_POTENTIAL_REDUCE_ELEMENTS, False
            ):
                element_map = ins_dict["element_map"]
                assert isinstance(element_map, dict), "Element map is not a dict"

                log.info(
                    f"Setting {tc.INPUT_POTENTIAL_SECTION}::elements to {element_map}"
                )

                if "elements" not in args_yaml[tc.INPUT_POTENTIAL_SECTION]:
                    args_yaml[tc.INPUT_POTENTIAL_SECTION]["elements"] = element_map
                else:
                    old_elements = args_yaml[tc.INPUT_POTENTIAL_SECTION]["elements"]
                    old_element_map = {e: i for i, e in enumerate(old_elements)}
                    if element_map != old_element_map:
                        raise RuntimeError(
                            f"Provided input.yaml::{tc.INPUT_POTENTIAL_SECTION}::elements  ({old_elements}) "
                            + f"differs from that read from model ({element_map})"
                        )
            # otherwise skip until load_and_prepare_datasets
        if "rcut" in ins_dict:
            args_yaml[tc.INPUT_CUTOFF] = ins_dict["rcut"]
            log.info(f"Setting data::{tc.INPUT_CUTOFF} to {args_yaml[tc.INPUT_CUTOFF]}")
        if tc.INPUT_CUTOFF_DICT in ins_dict:
            args_yaml[tc.INPUT_CUTOFF_DICT] = ins_dict[tc.INPUT_CUTOFF_DICT]
            log.info(
                f"Setting data::{tc.INPUT_CUTOFF_DICT} to {args_yaml[tc.INPUT_CUTOFF_DICT]}"
            )
        if "ConstantScaleShiftTarget" in ins_dict["__cls__"]:
            scale = ins_dict["scale"]
            args_yaml[tc.INPUT_POTENTIAL_SECTION]["scale"] = scale
            log.info(f"Setting {tc.INPUT_POTENTIAL_SECTION}::scale to {scale}")
            # shift = ins_dict["shift"]
            # if shift != 0:
            #     args_yaml[tc.INPUT_POTENTIAL_SECTION]["shift"] = shift
        if "avg_n_neigh" in ins_dict:
            args_yaml[tc.INPUT_POTENTIAL_SECTION]["avg_n_neigh"] = ins_dict[
                "avg_n_neigh"
            ]
            log.info(
                f"Setting {tc.INPUT_POTENTIAL_SECTION}::avg_n_neigh to {ins_dict['avg_n_neigh']}"
            )

    return args_yaml


def main(argv=None, strategy=None, strategy_desc=""):
    if argv is None:
        argv = []
    # Stages:
    parser = build_parser()
    args_parse = parser.parse_args(argv)
    input_yaml_filename = args_parse.input

    if args_parse.template:
        generate_template_input()

    with open(input_yaml_filename) as f:
        args_yaml = yaml.safe_load(f)
    seed = args_parse.seed or args_yaml.get("seed", 1)
    output_dir = get_output_dir(seed=seed)
    if "log" in args_parse:
        log_file_name = os.path.join(output_dir, args_parse.log)
        os.makedirs(output_dir, exist_ok=True)
        log.info("Redirecting log into file {}".format(log_file_name))
        fileh = logging.FileHandler(log_file_name, "a")
        formatter = logging.Formatter(LOG_FMT)
        fileh.setFormatter(formatter)
        log.addHandler(fileh)

    log.info("=" * 40)
    log.info(" " * 12 + "Start GRACEmaker")
    log.info("=" * 40)
    log.info(f"Tensorflow version: {tf.__version__}")
    log.info("Loaded {}... ".format(input_yaml_filename))
    assert isinstance(args_yaml, dict)

    potential_config = args_yaml["potential"]
    fit_config = args_yaml["fit"]

    log.info(f"Set seed to {seed}")
    np.random.seed(seed)
    tf.random.set_seed(seed)

    # distributed strategy
    # global strategy, strategy_desc
    if strategy is None:
        if fit_config.get("strategy") == "mirrored" or args_parse.multigpu:
            strategy = tf.distribute.MirroredStrategy()
            strategy_desc = "Single host/multi GPU"
        else:
            # default - single GPU mode
            strategy = tf.distribute.get_strategy()
            strategy_desc = "Single GPU"
    else:
        log.info("Multi host/multi GPU distributed strategy is already initialized")

    num_replicas_in_sync = strategy.num_replicas_in_sync
    current_ctx = tf.distribute.get_replica_context()
    replica_id_in_sync_group = current_ctx.replica_id_in_sync_group
    log.info(f"Data distribution strategy: {strategy_desc}")
    log.info(f"Number of replicas: {num_replicas_in_sync}")
    log.info(f"Replica ID in sync group: {replica_id_in_sync_group}")

    rcut = args_yaml["cutoff"]
    batch_size = fit_config.get(
        "batch_size", 8
    )  # TODO: get batch_size from stats.json for distributed dataset
    global_batch_size = batch_size * strategy.num_replicas_in_sync
    log.info(
        f"Global TRAIN batch size (= minibatch size * num GPUS): {global_batch_size}, "
        f"num. GPUS: {strategy.num_replicas_in_sync}, minibatch size: {batch_size}"
    )

    test_batch_size = fit_config.get("test_batch_size", 1)
    if test_batch_size:
        global_test_batch_size = test_batch_size * strategy.num_replicas_in_sync
        log.info(
            f"Global TEST batch size (= minibatch size * num GPUS): {global_test_batch_size}, "
            f"num. GPUS: {strategy.num_replicas_in_sync}, minibatch size: {test_batch_size}"
        )

    float_dtype_param = potential_config.get("float_dtype", "float64")
    float_dtype = {
        "float64": tf.float64,
        "float32": tf.float32,
        "tfloat32": tf.float32,
    }[float_dtype_param]
    if float_dtype_param == "tfloat32":
        tf.config.experimental.enable_tensor_float_32_execution(True)
    log.info(f"Float dtype: {float_dtype_param}")

    train_data = None
    test_data = None
    element_map = None
    data_stats = None

    if args_parse.check_model:
        check_model(args_yaml, float_dtype)
        log.info(f"Exiting...")
        sys.exit(0)

    # finetuning of foundation models
    finetune_model = potential_config.get(tc.INPUT_POTENTIAL_FINETUNE_FOUNDATION_MODEL)
    if finetune_model and not (
        args_parse.restart_best_test or args_parse.restart_latest
    ):
        log.info(f"FINETUNING {finetune_model} foundation model")
        checkpoint_path = get_or_download_checkpoint(finetune_model)
        potential_config["filename"] = os.path.join(checkpoint_path, "model.yaml")
        potential_config[tc.INPUT_POTENTIAL_CHECKPOINT_NAME] = os.path.join(
            checkpoint_path, "checkpoint"
        )
        log.info(
            f'FINETUNING: set model path to {potential_config["filename"]} and '
            + f"checkpoint to {potential_config[tc.INPUT_POTENTIAL_CHECKPOINT_NAME]}"
        )

    potential_file_name = args_parse.potential or potential_config.get("filename")
    if potential_file_name:
        # model will be loaded from file,
        # need inject some parameters (cutoff, elements, scale, shift) into args_yaml
        log.info("Model YAML file is provided, try to adjust input")
        args_yaml = add_loaded_model_parameter(potential_file_name, args_yaml)

    # no need to perform expensive data processing if mode
    if not args_parse.save_model:
        # TODO: create a class or named tuple?
        (
            train_data,
            test_data,
            element_map,
            data_stats,
            train_grouping_df,
            test_grouping_df,
        ) = load_and_prepare_datasets(
            args_yaml,
            batch_size,
            test_batch_size=test_batch_size,
            seed=seed,
            strategy=strategy,
            float_dtype=float_dtype_param,
        )
        has_test_set = test_data is not None

    if args_parse.save_model:
        potential_file_name = potential_file_name or os.path.join(
            output_dir, MODEL_CONFIG_YAML
        )

    checkpoint_name = args_parse.checkpoint_name or potential_config.get(
        tc.INPUT_POTENTIAL_CHECKPOINT_NAME
    )
    # clean checkpoint_name from .index suffix if needed
    if checkpoint_name and checkpoint_name.endswith(".index"):
        checkpoint_name = checkpoint_name.replace(".index", "")

    # do model reduction / element selection
    is_reduce_elements = args_yaml[tc.INPUT_POTENTIAL_SECTION].get(
        tc.INPUT_POTENTIAL_REDUCE_ELEMENTS, False
    )
    if is_reduce_elements and not (
        args_parse.restart_best_test or args_parse.restart_latest
    ):
        new_potential_file_name = os.path.join(output_dir, MODEL_CONFIG_YAML)
        new_checkpoint_name = os.path.join(output_dir, "checkpoints", "checkpoint")

        log.info(
            f"Selecting elements from model {potential_file_name} with checkpoint at {checkpoint_name}"
            + f" and saving to {new_potential_file_name} with checkpoint at {new_checkpoint_name}"
        )

        convert_model_reduce_elements(
            element_map=element_map,
            potential_file_name=potential_file_name,
            checkpoint_name=checkpoint_name,
            new_potential_file_name=new_potential_file_name,
            new_checkpoint_name=new_checkpoint_name,
        )
        potential_file_name = new_potential_file_name
        checkpoint_name = new_checkpoint_name

    target_potential_file_name = os.path.join(output_dir, MODEL_CONFIG_YAML)
    if potential_file_name:
        # load from potential_file_name
        log.info(f"Loading model config from `{potential_file_name}`")
        pot = load_instructions(potential_file_name)
        # enforce to save model.yaml into output_dir
    elif not args_parse.save_model:
        cut_dict = args_yaml.get(tc.INPUT_CUTOFF_DICT)
        if cut_dict:
            log.info(f"User-defined cutoff dict: {cut_dict}")
        log.info(f"Constructing model from config")
        pot = construct_model(
            potential_config,
            element_map=element_map,
            rcut=rcut,
            cutoff_dict=cut_dict,
            **data_stats,
        )
    else:  # save model, but initialized from scratch
        # TODO: should be possible
        raise ValueError("Cannot save just-initialized model")

    log.info(f"Saving model config to {target_potential_file_name}")
    save_instructions_dict(target_potential_file_name, pot)

    # loss function spec from fit_config
    model_fns = construct_model_functions(fit_config)
    log.info(f"Loss function: {model_fns.loss_fn}")

    if args_parse.eager:
        log.warning("Eager execution")

    jit_compile = fit_config.get("jit_compile", True)
    if args_parse.no_jit:
        log.info("--no-jit option is provided")
        jit_compile = False
    log.info(f"JIT compilation: {jit_compile}")
    opt = fit_config.get("optimizer", "Adam")
    log.info(f"Optimization options: {fit_config.get('opt_params')}")

    tp = TensorPotential(
        potential=pot,
        fit_config=fit_config,
        global_batch_size=global_batch_size,
        global_test_batch_size=global_test_batch_size,
        loss_function=model_fns.loss_fn,
        regularization_loss=model_fns.regularization_loss_fn,
        compute_metrics=model_fns.metrics_fn,
        model_train_function=model_fns.train_fn,
        model_compute_function=model_fns.compute_fn,
        float_dtype=float_dtype,  # TODO: set up from config
        eager_mode=args_parse.eager,
        jit_compile=jit_compile,
        seed=seed,
        strategy=strategy,
        loss_norm_by_batch_size=fit_config.get("loss_norm_by_batch_size", False),
    )

    try_load_checkpoint(
        tp,
        restart_best_test=args_parse.restart_best_test,
        restart_latest=args_parse.restart_latest,
        restart_suffix=args_parse.restart_suffix,
        checkpoint_name=checkpoint_name,
        expect_partial=False,  # True for save_model, False otherwise
        verbose=True,
        assert_consumed=False,
    )

    if args_parse.save_model:
        # if tp.is_lora_enabled():
        #     log.info("LORA enabled, finalizing it.")
        #     tp.finalize_lora_update()

        if args_parse.save_fs:
            log.info("Exporting FS-model, please wait...")
            tp.export_to_yaml("FS_model.yaml")
            log.info("Exporting FS-model done")
        log.info("Saving model to `saved_model` and exit.")
        # TODO: first save with jit will convert function to JIT forever!
        # tp.save_model("saved_model_no_jit", jit_compile=False)  # first - no-jit
        tp.save_model("saved_model", jit_compile=True)
        sys.exit(0)

    # if potential_config.get("lora"):
    #     tp.enable_lora_adaptation(potential_config.get("lora"))
    #     log.info(
    #         f"Saving model config to {target_potential_file_name} after LoRA activation"
    #     )
    #     save_instructions_dict(target_potential_file_name, tp.model.instructions)
    #
    # if potential_config.get("reduce_lora"):
    #     tp.finalize_lora_update()
    #     log.info(
    #         f"Saving model config to {target_potential_file_name} after LoRA reduction"
    #     )
    #     save_instructions_dict(target_potential_file_name, tp.model.instructions)

    if fit_config.get("trainable_variable_names"):
        trainable_names = fit_config["trainable_variable_names"]
        assert isinstance(trainable_names, list), "trainable_names must be a list"
        log.info(f"Set trainable variable names: {trainable_names}")
        tp.set_trainable_variables(trainable_names, verbose=True)

    def load_checkpoint_if_test_available(
        expect_partial=False, verbose=True, assert_consumed=False
    ):
        if has_test_set:
            log.info("Loading best test loss model")
            tp.load_checkpoint(
                suffix=".best_test_loss",
                expect_partial=expect_partial,
                verbose=verbose,
                assert_consumed=assert_consumed,
            )

    def save_tp_model(name):
        log.info(f"Saving model to `{name}`")
        tp.save_model(name, jit_compile=True)

    log.info("Convert data to tensors")
    train_batches, test_batches = convert_to_tensors_for_model(
        tp, train_data, test_data, strategy=strategy
    )
    del train_data
    del test_data
    gc.collect()

    if fit_config.get("reset_epoch_and_step") or args_parse.reset_epoch_and_step:
        log.info(f"Reset epochs ({tp.epoch} -> 0) and steps ({tp.step} -> 0) counters")
        tp.reset_epoch_and_step()

    #  Run training
    try:
        if opt in ["L-BFGS-B", "BFGS"]:
            log.info(f"Start {opt} optimization")
            train_bfgs(
                tp,
                fit_config=fit_config,
                train_ds=train_batches,
                test_ds=test_batches,
                strategy=strategy,
                seed=seed,
                train_grouping_df=train_grouping_df,
                test_grouping_df=test_grouping_df,
            )
        elif opt == "Adam":
            log.info("Start Adam optimization")
            train_adam(
                tp,
                fit_config=fit_config,
                train_ds=train_batches,
                test_ds=test_batches,
                strategy=strategy,
                seed=seed,
                train_grouping_df=train_grouping_df,
                test_grouping_df=test_grouping_df,
            )
        else:
            raise ValueError(
                f"Unknown optimizer: {opt}. Can be 'Adam' or 'L-BFGS-B', 'BFGS'"
            )
        load_checkpoint_if_test_available(assert_consumed=True, verbose=True)

        # if tp.is_lora_enabled():
        #     log.info("LORA enabled, finalizing it.")
        #     tp.finalize_lora_update()

        save_tp_model(name="final_model")
        if args_parse.save_fs:
            log.info("Exporting FS-model, please wait...")
            tp.export_to_yaml("FS_model.yaml")
            log.info("Exporting FS-model done")
    except KeyboardInterrupt as e:
        log.info("Keyboard interruption is captured")
        sys.exit(0)


def check_model(args_yaml, float_dtype):
    rcut = args_yaml["cutoff"]
    potential_config = args_yaml["potential"]

    element_map = {s: i for i, s in enumerate(chemical_symbols[1:90])}
    log.info(f"Checking model with {len(element_map)} elements")
    cut_dict = args_yaml.get(tc.INPUT_CUTOFF_DICT)
    if cut_dict:
        log.info(f"User-defined cutoff dict: {cut_dict}")
    pot = construct_model(
        potential_config, element_map=element_map, rcut=rcut, cutoff_dict=cut_dict
    )
    model = TPModel(pot)
    model.build(float_dtype=float_dtype)
    init_flat_vars = model.get_flat_trainable_variables()
    log.info("Model is constructed")
    log.info(f"Number of trainable parameters: {len(init_flat_vars)}")
