#!/usr/bin/env python
import argparse
import logging
import sys

import numpy as np
from ase import Atoms

from pathlib import Path

from tensorpotential.instructions import SingleParticleBasisFunctionEquivariantInd

LOG_FMT = "%(asctime)s %(levelname).1s - %(message)s"
logging.basicConfig(level=logging.INFO, format=LOG_FMT, datefmt="%Y/%m/%d %H:%M:%S")
logger = logging.getLogger()


def load_checkpoint(tp, checkpoint_path):
    tp.load_checkpoint(
        checkpoint_name=checkpoint_path,
        expect_partial=True,
        verbose=True,
        raise_errors=True,
        assert_consumed=False,
        assert_existing_objects_matched=False,
    )


def preprocess_args(args):
    if not args.output_suffix.startswith("-"):
        args.output_suffix = "-" + args.output_suffix

    if args.checkpoint_path and args.checkpoint_path.endswith(".index"):
        args.checkpoint_path = args.checkpoint_path.replace(".index", "")


def update_model(args):
    preprocess_args(args)

    """Convert model.yaml and corresponding checkpoint to model as dict-of-instructions and corresponding checkpoint"""
    from tensorpotential import TensorPotential
    from tensorpotential.instructions.base import (
        load_instructions,
        save_instructions_dict,
    )

    model_path, checkpoint_path, output_suffix = (
        args.potential,
        args.checkpoint_path,
        args.output_suffix,
    )

    # clean checkpoint_name from .index suffix if needed
    if checkpoint_path.endswith(".index"):
        checkpoint_path = checkpoint_path.replace(".index", "")

    logger.info(
        f"Converting model {model_path} with checkpoint at {checkpoint_path} to dict-like model"
    )

    logger.info(f"Loading model from {model_path}")
    instructions = load_instructions(model_path)
    if isinstance(instructions, dict):
        logger.info(f"Model in {model_path} is already in new format (dict-like)")
        sys.exit(0)

    # convert to dict-of-instructions
    instructions_dict = {ins.name: ins for ins in instructions}
    assert len(instructions_dict) == len(instructions)

    tp = TensorPotential(potential=instructions)
    load_checkpoint(tp, checkpoint_path)

    tp.model.instructions = instructions_dict

    new_checkpoint_path = checkpoint_path + output_suffix
    logger.info(f"Saving converted checkpoint to {new_checkpoint_path}")
    tp.save_checkpoint(checkpoint_name=new_checkpoint_path, verbose=True)

    new_model_path = model_path
    for ext in [".yaml", ".yml"]:
        new_model_path = new_model_path.replace(ext, output_suffix + ext)

    logger.info(f"Saving converted model to {new_model_path}")
    save_instructions_dict(new_model_path, instructions_dict)

    # try to load again
    logger.info(f"Trying to load converted model from {new_model_path}")
    new_instructions = load_instructions(new_model_path)
    tp_new = TensorPotential(potential=new_instructions)
    load_checkpoint(tp_new, new_checkpoint_path)

    logger.info(f"Finished")


def resave_checkpoint(args):
    preprocess_args(args)

    """Convert model.yaml and corresponding checkpoint to model as dict-of-instructions and corresponding checkpoint"""
    from tensorpotential import TensorPotential
    from tensorpotential.instructions.base import (
        load_instructions,
    )

    model_path, checkpoint_path, output_suffix = (
        args.potential,
        args.checkpoint_path,
        args.output_suffix,
    )
    new_checkpoint_path = checkpoint_path + output_suffix

    # clean checkpoint_name from .index suffix if needed
    if checkpoint_path.endswith(".index"):
        checkpoint_path = checkpoint_path.replace(".index", "")

    logger.info(
        f"Resaving model {model_path} with checkpoint at {checkpoint_path} to new checkpoint {new_checkpoint_path}"
    )

    logger.info(f"Loading model from {model_path}")
    instructions = load_instructions(model_path)

    tp = TensorPotential(potential=instructions)

    load_checkpoint(tp, checkpoint_path)

    logger.info(f"Saving converted checkpoint to {new_checkpoint_path}")
    tp.setup_checkpoint(with_optimizer=False)  # will reset al lexcept model's weights
    tp.save_checkpoint(checkpoint_name=new_checkpoint_path, verbose=True)

    # try to load again
    logger.info(f"Trying to load again model from {model_path}")
    new_instructions = load_instructions(model_path)
    tp_new = TensorPotential(potential=new_instructions)
    load_checkpoint(tp_new, new_checkpoint_path)

    logger.info(f"Finished")


def reduce_elements(args):

    preprocess_args(args)

    from tensorpotential import TensorPotential
    from tensorpotential.instructions.base import load_instructions
    from tensorpotential.calculator.asecalculator import TPCalculator
    from tensorpotential.utils import convert_model_reduce_elements

    model_path, checkpoint_path, output_suffix, elements = (
        args.potential,
        args.checkpoint_path,
        args.output_suffix,
        args.elements,
    )

    # clean checkpoint_name from .index suffix if needed
    if checkpoint_path.endswith(".index"):
        checkpoint_path = checkpoint_path.replace(".index", "")

    logger.info(
        f"Selecting {elements} elements from model {model_path}, with checkpoint {checkpoint_path}"
    )

    logger.info(f"Loading model from {model_path}")
    instructions_dict = load_instructions(model_path)
    if not isinstance(instructions_dict, dict):
        logger.info(
            f"Model in {model_path} is NOT in new format (dict-like). Convert it using `update_model` option"
        )
        sys.exit(0)

    tp = TensorPotential(potential=instructions_dict)
    load_checkpoint(tp, checkpoint_path)

    new_checkpoint_path = checkpoint_path + output_suffix
    new_model_path = model_path
    for ext in [".yaml", ".yml"]:
        new_model_path = new_model_path.replace(ext, output_suffix + ext)

    # generate test structures
    test_atoms = {}
    calc = TPCalculator(model=tp.model)
    for el in elements:
        at = Atoms(positions=[[0, 0, 0], [2, 0, 0]], symbols=[el, el], pbc=False)
        at.calc = calc
        en = at.get_potential_energy()
        logger.info(f"{el} dimer: energy = {en}")
        test_atoms[el] = {"atoms": at, "e": en}

    convert_model_reduce_elements(
        elements,
        potential_file_name=model_path,
        checkpoint_name=checkpoint_path,
        new_potential_file_name=new_model_path,
        new_checkpoint_name=new_checkpoint_path,
    )

    # try to load again
    logger.info("==========================")
    logger.info(f"Trying to load converted model from {new_model_path}")
    new_instructions = load_instructions(new_model_path)
    tp_new = TensorPotential(potential=new_instructions)
    load_checkpoint(tp_new, new_checkpoint_path)

    calc_new = TPCalculator(model=tp_new.model)
    for el, info in test_atoms.items():
        at = info["atoms"]
        en_ref = info["e"]
        at.calc = calc_new
        en = at.get_potential_energy()
        if np.allclose(en, en_ref):
            diff = en - en_ref
            logger.info(f"{el} dimer: OK, diff = {diff:.5g} eV")
        assert np.allclose(
            en, en_ref
        ), f"Energy not equal for element {el}: {en} != {en_ref}"

    logger.info(f"Finished")


def cast_model(args):
    from tensorpotential import TensorPotential
    from tensorpotential.instructions.base import load_instructions
    from tensorflow import float32, float64, Variable, Module

    def recursive_cast_instruction_vars(
        instruction, from_dtype, to_dtype, max_depth=None, current_depth=0
    ):
        if max_depth is not None and current_depth >= max_depth:
            return

        next_depth = current_depth + 1

        if isinstance(instruction, Module):
            for attr_name in instruction.__dict__:
                attr_value = getattr(instruction, attr_name)
                if isinstance(attr_value, Variable) and attr_value.dtype == from_dtype:
                    setattr(
                        instruction,
                        attr_name,
                        Variable(
                            attr_value.numpy(),
                            dtype=to_dtype,
                            name=attr_value.name.replace(":0", ""),
                        ),
                    )
                elif isinstance(attr_value, list) or isinstance(attr_value, tuple):
                    for item in attr_value:
                        recursive_cast_instruction_vars(
                            item, from_dtype, to_dtype, max_depth, next_depth
                        )
                elif isinstance(attr_value, dict):
                    for key, value in attr_value.items():
                        recursive_cast_instruction_vars(
                            value, from_dtype, to_dtype, max_depth, next_depth
                        )

    preprocess_args(args)
    dtypes = {"fp32": float32, "fp64": float64}

    model_path, checkpoint_path, output_suffix, from_dtype, to_dtype = (
        args.potential,
        args.checkpoint_path,
        args.output_suffix,
        args.curr,
        args.to,
    )
    assert from_dtype in dtypes, f"Invalid dtype, {from_dtype} not in {dtypes.keys()}"
    assert to_dtype in dtypes, f"Invalid dtype, {to_dtype} not in {dtypes.keys()}"
    logger.info(
        f"Trying to cast model at model_path={model_path},"
        f" with checkpoint_path={checkpoint_path},"
        f" from dtype={from_dtype} to dtype={to_dtype}"
    )

    d_from = dtypes[from_dtype]
    d_to = dtypes[to_dtype]

    pot = load_instructions(model_path)
    tp = TensorPotential(potential=pot, fit_config={}, float_dtype=d_from)
    load_checkpoint(tp, checkpoint_path)
    assert isinstance(
        tp.model.instructions, dict
    ), "Model is not dict format, cannot cast it. Use `update_model` option to convert it to dict format."
    for iname, ins in tp.model.instructions.items():
        recursive_cast_instruction_vars(ins, d_from, d_to, max_depth=5)
    logger.info(f"Successfully casted model at {checkpoint_path} to {to_dtype} dtype")
    tp.save_checkpoint(checkpoint_name=checkpoint_path + output_suffix, verbose=True)
    logger.info(f"Saved casted checkpoint to {checkpoint_path + output_suffix}")

    current_directory = Path.cwd()
    full_path = current_directory / "casted_model"
    instructions = load_instructions(model_path)
    tp = TensorPotential(potential=instructions, float_dtype=d_to)
    load_checkpoint(tp, checkpoint_path + output_suffix)
    tp.save_model(exact_path=full_path, jit_compile=True)
    logger.info(f"Saved casted model to {full_path}")


def export_model(args):
    """Convert model.yaml and corresponding checkpoint to model as dict-of-instructions and corresponding checkpoint"""
    from tensorpotential import TensorPotential
    from tensorpotential.instructions.base import load_instructions

    preprocess_args(args)

    model_path, checkpoint_path, output_suffix = (
        args.potential,
        args.checkpoint_path,
        args.output_suffix,
    )
    save_fs = args.sf

    # clean checkpoint_name from .index suffix if needed
    if checkpoint_path.endswith(".index"):
        checkpoint_path = checkpoint_path.replace(".index", "")

    logger.info(
        f"Converting model {model_path} with checkpoint at {checkpoint_path} to dict-like model"
    )

    logger.info(f"Loading model from {model_path}")
    instructions = load_instructions(model_path)
    tp = TensorPotential(potential=instructions)
    logger.info(f"Loading checkpoint from {checkpoint_path}")
    load_checkpoint(tp, checkpoint_path)

    if not save_fs:
        saved_model_name = args.saved_model_name
        logger.info(f"Exporting model to {saved_model_name}")
        tp.save_model(exact_path=saved_model_name, jit_compile=True)
    else:
        saved_model_name = args.saved_model_name
        if not saved_model_name.endswith(".yaml") or saved_model_name.endswith(".yml"):
            saved_model_name += ".yaml"
        logger.info(f"Exporting FS model to {saved_model_name}")
        tp.export_to_yaml(exact_filename=saved_model_name)

    logger.info(f"Finished")


def model_summary(args):
    preprocess_args(args)

    from tensorpotential.instructions.base import load_instructions
    from tensorpotential.tpmodel import TPModel
    import tensorflow as tf

    model_path = args.potential
    verbose = args.verbose

    logger.info(f"Loading model from {model_path}")
    instructions_dict = load_instructions(model_path)
    tp = TPModel(instructions_dict)
    tp.build(tf.float64)
    res = tp.summary(verbose=verbose)
    logger.info(f"Summary (verbose={verbose}) is:\n{res}")


def aux_model(args):
    # Add several auxiliary compute functions and saved model to SavedModel format

    from tensorpotential.instructions import load_instructions
    from tensorpotential.tpmodel import (
        ComputeEnergy,
        ComputeStructureEnergyAndForcesAndVirial,
        TPModel,
    )
    from tensorpotential.tensorpot import TensorPotential
    from tensorpotential.instructions.instruction_graph_utils import (
        build_split_tpmodel,
        build_dependency_graph,
        find_non_local_keys,
    )

    import tensorflow as tf

    model_path, checkpoint_path, output_path, communicated_keys = (
        args.potential,
        args.checkpoint_path,
        args.output_path,
        args.communicated_keys,
    )
    # aux_options = args.aux

    # clean checkpoint_name from .index suffix if needed
    if checkpoint_path.endswith(".index"):
        checkpoint_path = checkpoint_path.replace(".index", "")

    logger.info(f"Upgrading model {model_path} with checkpoint at {checkpoint_path}")

    instr = load_instructions(model_path)
    float_dtype = tf.float64

    extra_aux_computes = {"compute_energy": ComputeEnergy()}

    #     extra_aux_computes["compute_local"] = ComputeStructureEnergyAndForcesAndVirial(
    #         local=True
    #     )

    has_spbfei = False
    for ins_name, ins in instr.items():
        if isinstance(ins, SingleParticleBasisFunctionEquivariantInd):
            has_spbfei = True
            logger.info(
                f"GRACE-2L model is identified (SingleParticleBasisFunctionEquivariantInd instruction found)"
            )
            break
    if has_spbfei:
        dep_graph = build_dependency_graph(instr)
        non_local_comm_keys = find_non_local_keys(instr, communicated_keys)
        # pretty print dep_graph
        for k, v in dep_graph.items():
            if k in communicated_keys:
                if k in non_local_comm_keys:
                    logger.info(f"  [NON-LOCAL COMMUNICATED] {k} : {v}")
                else:
                    logger.info(f"  [COMMUNICATED] {k} : {v}")
            else:
                logger.info(f"  {k}: {v}")

        logger.info(f"Communicated keys: {communicated_keys}")

        # assert that communicated_keys in dep_graph
        for ck in communicated_keys:
            assert ck in dep_graph, f"Key {ck} not found in dependency graph"

        m = build_split_tpmodel(
            instr,
            communicated_keys=communicated_keys,
            float_dtype=float_dtype,
            jit_compile=True,
            extra_aux_computes=extra_aux_computes,
        )
    else:
        m = TPModel(instructions=instr, aux_compute=extra_aux_computes)
        m.build(float_dtype, jit_compile=True)
        m.decorate_compute_function(float_dtype, jit_compile=True)

    tp = TensorPotential(m.instructions)
    tp.model = m
    tp.load_checkpoint(
        checkpoint_name=checkpoint_path,
        assert_consumed=False,
        assert_existing_objects_matched=True,
    )

    logger.info(f"Saving upgraded model to {output_path}")
    tp.save_model(exact_path=output_path)

    logger.info("Finished")


def main():
    parser = argparse.ArgumentParser(
        prog="grace_utils",
        description="CLI tool for model conversions and summarization",
    )
    parser.add_argument("-p", "--potential", required=True, help="Path to model.yaml")
    parser.add_argument(
        "-c", "--checkpoint-path", required=False, help="Path to checkpoint"
    )
    parser.add_argument(
        "-os",
        "--output-suffix",
        type=str,
        default="converted",
        help="Output suffix for converted",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    # update_models
    parser_update = subparsers.add_parser(
        "update_model", help="Update model (model.yaml) and corresponding checkpoint."
    )
    parser_update.set_defaults(func=lambda args: update_model(args))

    # resave checkpoint
    parser_resave_checkpoint = subparsers.add_parser(
        "resave_checkpoint",
        help="Resave model's (model.yaml) checkpoint (no optimizer)",
    )
    parser_resave_checkpoint.set_defaults(func=lambda args: resave_checkpoint(args))

    # select_elements
    parser_reduce = subparsers.add_parser(
        "reduce_elements", help="Reduce elements from the model."
    )
    parser_reduce.add_argument(
        "-e", "--elements", nargs="+", required=True, help="Elements to select"
    )
    parser_reduce.set_defaults(func=lambda args: reduce_elements(args))

    # precision
    parser_precision = subparsers.add_parser(
        "cast_model", help="Change model's floating point precision."
    )
    parser_precision.add_argument(
        "-curr",
        required=True,
        choices=["fp32", "fp64"],
        help="Current precision type to cast from",
    )
    parser_precision.add_argument(
        "-to",
        required=True,
        choices=["fp32", "fp64"],
        help="New precision type to cast into",
    )
    parser_precision.set_defaults(func=lambda args: cast_model(args))

    # export model
    parser_export = subparsers.add_parser(
        "export", help="Export model to saved_model or FS/C++ format."
    )
    parser_export.add_argument(
        "-sf",
        action="store_true",
        default=False,
        help="Save to GRACE-FS/C++ YAML model format",
    )
    parser_export.add_argument(
        "-n",
        "--saved-model-name",
        dest="saved_model_name",
        type=str,
        default="saved_model",
        help="Save to GRACE-FS/C++ YAML model format",
    )
    parser_export.set_defaults(func=lambda args: export_model(args))

    # info
    parser_summary = subparsers.add_parser("summary", help="Show info about the model")
    parser_summary.add_argument(
        "-v",
        "--verbose",
        default=0,
        type=int,
        choices=[0, 1, 2],
        help="Verbosity level: 0, 1 or 2",
    )

    parser_summary.set_defaults(func=lambda args: model_summary(args))

    # upgrade model
    parser_aux_model = subparsers.add_parser(
        "aux_model",
        help="Upgrade model with different compute functions: parallel, compute energy only, compute local",
    )
    parser_aux_model.add_argument(
        "-o", "--output-path", required=True, help="Path to save the upgraded model"
    )
    parser_aux_model.add_argument(
        "-ck",
        "--communicated-keys",
        nargs="+",
        default=["I_nl_LN", "I"],
        help="List of communicated keys, used for parallelization of GRACE-2L model.",
    )
    # parser_aux_model.add_argument(
    #     "--aux",
    #     nargs="+",
    #     default=["parallel_2L", "energy_only", "compute_local"],
    #     help="List of aux functions to add: parallel_2L, energy_only, compute_local",
    # )
    parser_aux_model.set_defaults(func=lambda args: aux_model(args))

    # Parse arguments and execute the corresponding function
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
