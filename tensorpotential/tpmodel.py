from __future__ import annotations

import logging
import os
from abc import ABC, abstractmethod
from collections import defaultdict
from typing import Dict

import numpy as np
import tensorflow as tf

tf.config.experimental.enable_tensor_float_32_execution(False)
tf.experimental.numpy.experimental_enable_numpy_behavior(dtype_conversion_mode="all")

import yaml
import json

from tensorpotential import constants
from tensorpotential.export import export_to_yaml
from tensorpotential.instructions.base import summary_verbose_type, LORAInstructionMixin
from tensorpotential.instructions.compute import TPInstruction, FunctionReduceN


def extract_cutoff_and_elements(instructions):
    cuts = [0.0]
    element_map_symbols = None
    element_map_index = None

    if isinstance(instructions, dict):
        instructions = list(instructions.values())
    for instruction in instructions:
        if hasattr(instruction, "basis_function"):
            cuts.append(instruction.basis_function.rc.numpy())
        if hasattr(instruction, "element_map_symbols"):
            element_map_symbols = instruction.element_map_symbols.numpy().astype(str)
            element_map_index = instruction.element_map_index.numpy()

    cutoff = np.max(cuts)
    return cutoff, element_map_symbols, element_map_index


def extract_cutoff_matrix(instructions) -> np.array:
    """
    Aggregate cutoff matrix over all instructions using cumulative max
    Reshape it into square matrix
    """
    cutoff_matrix = None
    if isinstance(instructions, dict):
        instructions = list(instructions.values())
    for instruction in instructions:
        if hasattr(instruction, "bond_cutoff_map"):
            if cutoff_matrix is None:
                # init
                cutoff_matrix = instruction.bond_cutoff_map.numpy()
            else:
                # cumulative max
                current_cutoff_matrix = instruction.bond_cutoff_map.numpy()
                for i in range(current_cutoff_matrix.shape[0]):
                    for j in range(current_cutoff_matrix.shape[1]):
                        cutoff_matrix[i, j] = np.max(
                            current_cutoff_matrix[i, j], cutoff_matrix[i, j]
                        )

    if cutoff_matrix is not None:
        # infer number of elements
        nels = int(np.round(np.sqrt(len(cutoff_matrix))))
        assert nels**2 == len(cutoff_matrix)
        return np.array(cutoff_matrix).reshape(nels, nels)


def compute_batch_virials_from_pair_forces(pair_f, input_data):
    # virials xx,yy,zz
    virial_012 = tf.reshape(
        tf.math.unsorted_segment_sum(
            pair_f * input_data[constants.BOND_VECTOR],
            segment_ids=input_data[constants.BONDS_TO_STRUCTURE_MAP],
            num_segments=input_data[constants.N_STRUCTURES_BATCH_TOTAL],
        ),
        (-1, 3),
    )

    # xy
    virial_3 = tf.reshape(
        tf.math.unsorted_segment_sum(
            pair_f[:, 1] * input_data[constants.BOND_VECTOR][:, 0],
            segment_ids=input_data[constants.BONDS_TO_STRUCTURE_MAP],
            num_segments=input_data[constants.N_STRUCTURES_BATCH_TOTAL],
        ),
        (-1, 1),
    )
    # xz
    virial_4 = tf.reshape(
        tf.math.unsorted_segment_sum(
            pair_f[:, 2] * input_data[constants.BOND_VECTOR][:, 0],
            segment_ids=input_data[constants.BONDS_TO_STRUCTURE_MAP],
            num_segments=input_data[constants.N_STRUCTURES_BATCH_TOTAL],
        ),
        (-1, 1),
    )
    # yz
    virial_5 = tf.reshape(
        tf.math.unsorted_segment_sum(
            pair_f[:, 2] * input_data[constants.BOND_VECTOR][:, 1],
            segment_ids=input_data[constants.BONDS_TO_STRUCTURE_MAP],
            num_segments=input_data[constants.N_STRUCTURES_BATCH_TOTAL],
        ),
        (-1, 1),
    )

    # order: xx,yy,zz, xy,xz,yz
    virial = tf.concat([virial_012, virial_3, virial_4, virial_5], axis=1)
    return virial


def compute_structure_virials_from_pair_forces(pair_f, input_data):
    virial_012 = tf.reshape(
        tf.reduce_sum(pair_f * input_data[constants.BOND_VECTOR], axis=0),
        [
            -1,
        ],
    )
    virial_3 = tf.reshape(
        tf.reduce_sum(pair_f[:, 1] * input_data[constants.BOND_VECTOR][:, 0], axis=0),
        [
            1,
        ],
    )
    virial_4 = tf.reshape(
        tf.reduce_sum(pair_f[:, 2] * input_data[constants.BOND_VECTOR][:, 0], axis=0),
        [
            1,
        ],
    )
    virial_5 = tf.reshape(
        tf.reduce_sum(pair_f[:, 2] * input_data[constants.BOND_VECTOR][:, 1], axis=0),
        [
            1,
        ],
    )
    virial = tf.concat([virial_012, virial_3, virial_4, virial_5], axis=0)
    return virial


class TrainFunction(ABC):
    specs: dict[str, dict] = {}

    def __init__(self, **kwargs):
        pass

    @abstractmethod
    def __call__(self, *args, **kwargs):
        raise NotImplementedError()


class ComputeFunction(ABC):
    specs: dict[str, dict] = {}

    def __init__(self, **kwargs):
        pass

    @abstractmethod
    def __call__(self, *args, **kwargs):
        raise NotImplementedError()


def execute_instructions(input_data, instructions, training=False, local=False):
    """
    Function that loop over instructions (either as list or dict) and execute them.

    Parameters:
        input_data (dict): input data
        instructions (dict): instructions
        training (bool): training flag (default False)
        local (bool): local flag (default False)

    Return: none,  modifications happen in input_data inplace
    """
    if isinstance(instructions, list):
        for ins in instructions:
            ins(input_data, training=training, local=local)
    elif isinstance(instructions, dict):
        for name, ins in instructions.items():
            ins(input_data, training=training, local=local)
    else:
        raise ValueError(f"Invalid instructions container type {type(instructions)}")


class ComputeBatchEnergyAndForces(TrainFunction):
    specs = {
        constants.BOND_IND_I: {"shape": [None], "dtype": "int"},
        constants.BOND_IND_J: {"shape": [None], "dtype": "int"},
        constants.ATOMS_TO_STRUCTURE_MAP: {"shape": [None], "dtype": "int"},
        constants.N_STRUCTURES_BATCH_TOTAL: {"shape": [], "dtype": "int"},
        constants.BOND_VECTOR: {"shape": [None, 3], "dtype": "float"},
        constants.N_ATOMS_BATCH_TOTAL: {"shape": [], "dtype": "int"},
    }

    # @staticmethod
    def __call__(
        self,
        instructions: list[TPInstruction],
        input_data: dict,
        training: bool = False,
    ):
        with tf.GradientTape() as tape:
            tape.watch(input_data[constants.BOND_VECTOR])
            execute_instructions(input_data, instructions, training)
            e_atomic = tf.reshape(input_data[constants.PREDICT_ATOMIC_ENERGY], [-1, 1])
        pair_f = tf.negative(tape.gradient(e_atomic, input_data[constants.BOND_VECTOR]))

        total_energy = tf.math.unsorted_segment_sum(
            e_atomic,
            input_data[constants.ATOMS_TO_STRUCTURE_MAP],
            num_segments=input_data[constants.N_STRUCTURES_BATCH_TOTAL],
        )

        nat = tf.reshape(input_data[constants.N_ATOMS_BATCH_TOTAL], [])
        total_f = tf.math.unsorted_segment_sum(
            pair_f, input_data[constants.BOND_IND_J], num_segments=nat
        ) - tf.math.unsorted_segment_sum(
            pair_f, input_data[constants.BOND_IND_I], num_segments=nat
        )

        return {
            constants.PREDICT_TOTAL_ENERGY: total_energy,
            constants.PREDICT_FORCES: total_f,
            constants.PREDICT_ATOMIC_ENERGY: e_atomic,
        }


class ComputeBatchEnergyForcesVirials(TrainFunction):
    specs = {
        constants.BOND_IND_I: {"shape": [None], "dtype": "int"},
        constants.BOND_IND_J: {"shape": [None], "dtype": "int"},
        constants.ATOMS_TO_STRUCTURE_MAP: {"shape": [None], "dtype": "int"},
        constants.BONDS_TO_STRUCTURE_MAP: {"shape": [None], "dtype": "int"},
        constants.N_STRUCTURES_BATCH_TOTAL: {"shape": [], "dtype": "int"},
        constants.BOND_VECTOR: {"shape": [None, 3], "dtype": "float"},
        # constants.N_ATOMS_BATCH_TOTAL: {"shape": [], "dtype": "int"},
    }

    # @staticmethod
    def __call__(
        self,
        instructions: list[TPInstruction],
        input_data: dict,
        training: bool = False,
    ):
        with tf.GradientTape() as tape:
            tape.watch(input_data[constants.BOND_VECTOR])
            execute_instructions(input_data, instructions, training)
            e_atomic = tf.reshape(input_data[constants.PREDICT_ATOMIC_ENERGY], [-1, 1])
        pair_f = tf.negative(tape.gradient(e_atomic, input_data[constants.BOND_VECTOR]))

        total_energy = tf.math.unsorted_segment_sum(
            e_atomic,
            input_data[constants.ATOMS_TO_STRUCTURE_MAP],
            num_segments=input_data[constants.N_STRUCTURES_BATCH_TOTAL],
        )

        # nat = tf.reshape(input_data[constants.N_ATOMS_BATCH_TOTAL], [])
        nat = tf.shape(input_data[constants.ATOMIC_MU_I])[0]
        total_f = tf.math.unsorted_segment_sum(
            pair_f, input_data[constants.BOND_IND_J], num_segments=nat
        ) - tf.math.unsorted_segment_sum(
            pair_f, input_data[constants.BOND_IND_I], num_segments=nat
        )
        virial = compute_batch_virials_from_pair_forces(pair_f, input_data)

        return {
            constants.PREDICT_TOTAL_ENERGY: total_energy,
            constants.PREDICT_FORCES: total_f,
            constants.PREDICT_ATOMIC_ENERGY: e_atomic,
            constants.PREDICT_VIRIAL: virial,
        }


class ComputeStructureEnergyAndForcesAndVirial(ComputeFunction):
    specs = {
        constants.BOND_IND_I: {"shape": [None], "dtype": "int"},
        constants.BOND_IND_J: {"shape": [None], "dtype": "int"},
        constants.BOND_VECTOR: {"shape": [None, 3], "dtype": "float"},
        constants.ATOMIC_MU_I: {"shape": [None], "dtype": "int"},
    }

    def __init__(self, local=False, **kwargs):
        super().__init__(**kwargs)
        self.local = local
        if self.local:
            self.specs[constants.ATOMIC_MU_I_LOCAL] = {"shape": [None], "dtype": "int"}

    def __call__(
        self,
        instructions: dict | list[TPInstruction],
        input_data: dict,
        training: bool = False,
    ):
        with tf.GradientTape() as tape:
            tape.watch(input_data[constants.BOND_VECTOR])
            execute_instructions(input_data, instructions, training, local=self.local)
            e_atomic = tf.reshape(input_data[constants.PREDICT_ATOMIC_ENERGY], [-1, 1])
        pair_f = tf.negative(tape.gradient(e_atomic, input_data[constants.BOND_VECTOR]))

        total_energy = tf.reduce_sum(e_atomic, axis=0, keepdims=True)

        # nat = tf.reshape(input_data[constants.N_ATOMS_BATCH_TOTAL], [])
        nat = tf.shape(input_data[constants.ATOMIC_MU_I])[0]
        total_f = tf.math.unsorted_segment_sum(
            pair_f, input_data[constants.BOND_IND_J], num_segments=nat
        ) - tf.math.unsorted_segment_sum(
            pair_f, input_data[constants.BOND_IND_I], num_segments=nat
        )
        virial = compute_structure_virials_from_pair_forces(pair_f, input_data)

        return {
            constants.PREDICT_TOTAL_ENERGY: total_energy,
            constants.PREDICT_FORCES: total_f,
            constants.PREDICT_VIRIAL: virial,
            constants.PREDICT_ATOMIC_ENERGY: e_atomic,
            # just quick hack to have it at last position, since all outputs are **alphabetically** sorted
            "z_" + constants.PREDICT_PAIR_FORCES: pair_f,
        }


class ComputeEnergy(ComputeFunction):
    specs = {
        constants.BOND_IND_I: {"shape": [None], "dtype": "int"},
        constants.BOND_IND_J: {"shape": [None], "dtype": "int"},
        constants.BOND_VECTOR: {"shape": [None, 3], "dtype": "float"},
        constants.ATOMIC_MU_I: {"shape": [None], "dtype": "int"},
    }

    def __init__(self, local=False, **kwargs):
        super().__init__(**kwargs)
        self.local = local
        if self.local:
            self.specs[constants.ATOMIC_MU_I_LOCAL] = {"shape": [None], "dtype": "int"}

    def __call__(
        self,
        instructions: dict | list[TPInstruction],
        input_data: dict,
        training: bool = False,
    ):

        execute_instructions(input_data, instructions, training, local=self.local)
        e_atomic = tf.reshape(input_data[constants.PREDICT_ATOMIC_ENERGY], [-1, 1])

        return {
            constants.PREDICT_ATOMIC_ENERGY: e_atomic,
        }


class ComputeEquivariantForces(ComputeFunction):
    specs = {}

    def __call__(
        self,
        instructions: list[TPInstruction],
        input_data: dict,
        training: bool = False,
    ):
        execute_instructions(input_data, instructions, training)
        atomic_f = tf.reshape(input_data[constants.PREDICT_FORCES], [-1, 3])

        return {
            constants.PREDICT_TOTAL_ENERGY: tf.constant([[0.0]], dtype=atomic_f.dtype),
            constants.PREDICT_FORCES: atomic_f,
        }


def extract_basis_functions(
    basis_reduce_instruction: TPInstruction, input_data: Dict[str, tf.Tensor]
):
    """
    Extract basis functions from computed data.

    Parameters
    ----------
    basis_reduce_instruction: TPInstruction - instruction that reduces basis functions
    input_data: Dict[str,tf.Tensor] - dictionary with computed tensors

    Returns
    -------
    basis: tf.Tensor - tensor with basis functions

    """
    basis = []
    for b_name in basis_reduce_instruction.instructions:
        instruction_collection = basis_reduce_instruction.collector[b_name.name]
        b = tf.gather(
            input_data[b_name.name],
            instruction_collection["func_collect_ind"],
            axis=2,
        )
        b = tf.reshape(b, (-1, b.shape[1] * b.shape[2]))
        basis.append(b)
    basis = tf.concat(basis, axis=-1)

    return basis


class ExtractBasisFunctions(ComputeFunction):

    specs = {
        constants.BOND_IND_I: {"shape": [None], "dtype": "int"},
        constants.BOND_IND_J: {"shape": [None], "dtype": "int"},
        constants.BOND_VECTOR: {"shape": [None, 3], "dtype": "float"},
    }

    def __init__(
        self,
        reduce_1L_instruction_name="AUTO",
        reduce_2L_instruction_name="AUTO",
        extract_1L_basis=True,
        extract_2L_basis=False,
    ):
        super().__init__()
        self.extract_1L_basis = extract_1L_basis
        self.extract_2L_basis = extract_2L_basis
        self.reduce_1L_instruction_name = reduce_1L_instruction_name
        self.reduce_2L_instruction_name = reduce_2L_instruction_name

    def find_function_reduce_name(self, instructions: Dict[str, TPInstruction]):
        """This function will find all namees of the FunctionReduceN"""
        function_reduce_ins_names = []
        for name, ins in instructions.items():
            if isinstance(ins, FunctionReduceN):
                ls_max = ins.ls_max
                if all(l == 0 for l in ls_max):
                    function_reduce_ins_names.append(name)
        return function_reduce_ins_names

    def __call__(
        self,
        instructions: dict[str, TPInstruction],
        input_data: dict[str, tf.Tensor],
        training: bool = False,
    ):
        execute_instructions(input_data, instructions, training)
        e_atomic = tf.reshape(input_data[constants.PREDICT_ATOMIC_ENERGY], [-1, 1])

        total_energy = tf.reduce_sum(e_atomic, axis=0, keepdims=True)

        function_reduce_ins_names = self.find_function_reduce_name(instructions)
        if self.reduce_1L_instruction_name == "AUTO":
            self.reduce_1L_instruction_name = function_reduce_ins_names[0]

        if self.reduce_2L_instruction_name == "AUTO" and self.extract_2L_basis:
            assert (
                len(function_reduce_ins_names) > 1
            ), "If extract_2L_basis is True, there should be at least 2 FunctionReduceN instructions"
            self.reduce_2L_instruction_name = function_reduce_ins_names[-1]

        result = {
            constants.PREDICT_TOTAL_ENERGY: total_energy,
            constants.PREDICT_ATOMIC_ENERGY: e_atomic,
        }
        if self.extract_1L_basis:
            result["1L_basis"] = extract_basis_functions(
                instructions[self.reduce_1L_instruction_name], input_data
            )

        if self.extract_2L_basis:
            result["2L_basis"] = extract_basis_functions(
                instructions[self.reduce_2L_instruction_name], input_data
            )

        return result


class ComputePlaceholder(ComputeFunction):
    specs = {}

    def __call__(
        self,
        instructions: list[TPInstruction],
        input_data: dict,
        training: bool = False,
    ):
        execute_instructions(input_data, instructions, training)
        placeholder = input_data[constants.PLACEHOLDER]

        return {
            constants.PLACEHOLDER: placeholder,
        }


class ComputeBlock(ComputeFunction):
    specs = {
        constants.ATOMIC_MU_I_LOCAL: {"shape": [None], "dtype": "int"},
    }

    def __init__(self, instructions, output_keys=None, specs=None):
        self.local_instructions = instructions
        self.output_keys = output_keys
        self.specs = specs or self.specs

    def __call__(self, instructions, input_data, training=False):
        execute_instructions(
            input_data, self.local_instructions, training=training, local=True
        )
        res = {}
        if self.output_keys:
            for k in self.output_keys:
                res[k] = input_data[k]
        return res


class ComputeBlockInputGradient(ComputeFunction):
    specs = {
        constants.ATOMIC_MU_I_LOCAL: {"shape": [None], "dtype": "int"},
    }

    def __init__(
        self, instructions, wrt_keys, target_key, output_keys=None, specs=None
    ):
        self.local_instructions = instructions
        self.wrt_keys = wrt_keys
        self.target_key = target_key
        self.output_keys = output_keys
        if specs:
            self.specs.update(specs)

    def __call__(self, instructions, input_data, training=False):
        wrt_tensors = [input_data[k] for k in self.wrt_keys]

        with tf.GradientTape() as tape:
            for t in wrt_tensors:
                tape.watch(t)
            execute_instructions(
                input_data, self.local_instructions, training, local=True
            )
            target = input_data[self.target_key]
            target = input_data[self.target_key]
        # print("target", target)
        grads = tape.gradient(target, wrt_tensors)
        # Convert IndexedSlices to dense tensors and handle None
        converted_grads = []
        for g, t in zip(grads, wrt_tensors):
            if g is None:
                converted_grads.append(tf.zeros_like(t))
            elif isinstance(g, tf.IndexedSlices):
                # Convert sparse IndexedSlices to dense tensor
                converted_grads.append(tf.convert_to_tensor(g))
            else:
                converted_grads.append(g)
        grads = converted_grads

        res = {f"grad_{k}": g for k, g in zip(self.wrt_keys, grads)}
        res[self.target_key] = target
        if self.output_keys:
            for k in self.output_keys:
                res[k] = input_data[k]
        return res


class ComputeBlockOutputGradient(ComputeFunction):
    specs = {
        constants.ATOMIC_MU_I_LOCAL: {"shape": [None], "dtype": "int"},
    }

    def __init__(self, instructions, wrt_keys, output_keys, specs=None):
        self.local_instructions = instructions
        self.wrt_keys = wrt_keys
        self.output_keys = output_keys
        if specs:
            self.specs.update(specs)

    def __call__(self, instructions, input_data, training=False):
        wrt_tensors = [input_data[k] for k in self.wrt_keys]
        output_grads = [input_data[f"grad_{k}"] for k in self.output_keys]

        with tf.GradientTape() as tape:
            for t in wrt_tensors:
                tape.watch(t)
            execute_instructions(
                input_data, self.local_instructions, training, local=True
            )
            outputs = [input_data[k] for k in self.output_keys]

        grads = tape.gradient(outputs, wrt_tensors, output_gradients=output_grads)
        # Convert IndexedSlices to dense tensors and handle None
        converted_grads = []
        for g, t in zip(grads, wrt_tensors):
            if g is None:
                converted_grads.append(tf.zeros_like(t))
            elif isinstance(g, tf.IndexedSlices):
                # Convert sparse IndexedSlices to dense tensor
                converted_grads.append(tf.convert_to_tensor(g))
            else:
                converted_grads.append(g)
        grads = converted_grads

        res = {f"grad_{k}": g for k, g in zip(self.wrt_keys, grads)}
        return res


def to_instructions_list(instructions):
    if isinstance(instructions, dict):
        return list(instructions.values())
    elif isinstance(instructions, list):
        return instructions
    else:
        raise ValueError(
            f"`instructions` must be list or dict, but {type(instructions)} given"
        )


class TPModel(tf.Module):
    def __init__(
        self,
        instructions,
        compute_function: ComputeFunction = ComputeStructureEnergyAndForcesAndVirial(),
        train_function: TrainFunction = ComputeBatchEnergyAndForces(),
        aux_compute: dict = None,
        name="TPModel",
    ):
        super(TPModel, self).__init__(name=name)
        self.instructions = instructions
        self.compute_function = compute_function
        self.train_function = train_function
        self.aux_compute = aux_compute
        self._compute_specs = {}
        self._train_specs = {}
        self.instructions_specs = {}

        self._variables_to_train = None
        self._aux_compute_sigs = {}
        self._aux_tf_funcs = {}

    def _decorate_aux_computes(self, float_dtype, jit_compile=True, input_dtype=None):
        if input_dtype is None:
            input_dtype = float_dtype

        dtypes = {
            "int": tf.int32,
            "int32": tf.int32,
            "int64": tf.int64,
            "float": input_dtype,
            "float32": tf.float32,
            "float64": tf.float64,
        }
        input_signature = {}
        for k, v in self.instructions_specs.items():
            input_signature[k] = tf.TensorSpec(
                shape=v["shape"], dtype=dtypes[v["dtype"]], name=k
            )

        if self.aux_compute is not None:
            for funcname, func in self.aux_compute.items():

                total_signatures = input_signature.copy()
                for k, v in func.specs.items():
                    total_signatures[k] = tf.TensorSpec(
                        shape=v["shape"], dtype=dtypes[v["dtype"]], name=k
                    )

                def make_implementation(captured_func):
                    def _impl(input_data):
                        copy_data = input_data.copy()
                        return captured_func(
                            self.instructions, copy_data, training=False
                        )

                    return _impl

                implementation = make_implementation(func)
                implementation.__name__ = funcname
                self._aux_compute_sigs[funcname] = total_signatures.copy()

                tf_decorated_func = tf.function(
                    implementation,
                    input_signature=[total_signatures.copy()],
                    jit_compile=jit_compile,
                )

                # Store the actual tf.function for saving
                self._aux_tf_funcs[funcname] = tf_decorated_func

                # Filter input dict to match signature keys
                sig_keys = set(total_signatures.keys())

                def create_wrapper(f, keys):
                    def input_filtering_wrapper(input_data):
                        filtered_data = {
                            k: input_data[k] for k in keys if k in input_data
                        }
                        return f(filtered_data)

                    return input_filtering_wrapper

                setattr(self, funcname, create_wrapper(tf_decorated_func, sig_keys))

    @property
    def variables_to_train(self):
        self._variables_to_train = []
        for var in self.trainable_variables:
            if not hasattr(var, "_trainable") or (var._trainable):
                self._variables_to_train.append(var)
        return self._variables_to_train

    @property
    def train_specs(self):
        self._train_specs.update(self.instructions_specs)
        self._train_specs.update(self.train_function.specs.copy())
        return self._train_specs

    @property
    def compute_specs(self):
        self._compute_specs.update(self.instructions_specs)
        self._compute_specs.update(self.compute_function.specs.copy())
        return self._compute_specs

    def build(self, float_dtype, jit_compile=True, input_dtype=None):
        self.float_dtype = float_dtype
        for sm in self.submodules:
            if hasattr(sm, "build"):
                if not hasattr(sm, "is_built") or not sm.is_built:
                    sm.build(float_dtype)
            if hasattr(sm, "input_tensor_spec"):
                self.instructions_specs.update(sm.input_tensor_spec.copy())

        self._decorate_aux_computes(
            float_dtype, jit_compile=jit_compile, input_dtype=input_dtype
        )

    def get_flat_trainable_variables(self):
        # ctx = tf.distribute.get_replica_context()
        self.count_coefs = 0
        self.slices = [0]
        flat_val = []
        for i, var in enumerate(self.variables_to_train):
            v = tf.reshape(var, [-1])
            self.count_coefs += tf.shape(v)[0]
            self.slices.append(self.count_coefs)
            flat_val += [v]
        flat_vars = tf.concat(flat_val, axis=0)

        return flat_vars

    def set_flat_trainable_variables(self, flat_vars):
        for i, var in enumerate(self.variables_to_train):
            new_values = flat_vars[self.slices[i] : self.slices[i + 1]]
            new_values = tf.reshape(new_values, tf.shape(var))
            if new_values.dtype != self.float_dtype:
                new_values = tf.cast(new_values, self.float_dtype)
            var.assign(new_values)

    def decorate_compute_function(self, float_dtype, jit_compile, input_dtype=None):
        if input_dtype is None:
            input_dtype = float_dtype
        dtypes = {
            "int": tf.int32,
            "int32": tf.int32,
            "int64": tf.int64,
            "float": input_dtype,
            "float32": tf.float32,
            "float64": tf.float64,
        }
        input_signature = {}
        for k, v in self.compute_specs.items():
            input_signature[k] = tf.TensorSpec(
                shape=v["shape"], dtype=dtypes[v["dtype"]], name=k
            )
        self.compute = tf.function(
            self.compute, input_signature=[input_signature], jit_compile=jit_compile
        )

    def save_model(
        self,
        path: str,
        jit_compile: bool = True,
        float_dtype=tf.float64,
        input_dtype=None,
    ):
        self.decorate_compute_function(
            float_dtype, jit_compile, input_dtype=input_dtype
        )
        # TODO: rename to "compute"
        signatures = {
            "serving_default": self.compute,
        }
        if self.aux_compute is not None:
            signatures.update(
                # Use the underlying tf.functions, not the python wrappers
                {name: self._aux_tf_funcs[name] for name in self.aux_compute.keys()}
            )
        tf.saved_model.save(self, path, signatures=signatures)

        cutoff, element_map_symbols, element_map_index = extract_cutoff_and_elements(
            self.instructions
        )
        element_map_symbols = element_map_symbols[element_map_index]
        cutoff_matrix = extract_cutoff_matrix(self.instructions)
        if cutoff_matrix is not None:
            cutoff = np.max(cutoff_matrix)
        metadata = {
            "chemical_symbols": list(map(str, element_map_symbols)),
            "cutoff": float(cutoff),
        }
        if cutoff_matrix is not None:
            # convert to list for YAML-friendly format
            metadata["cutoff_matrix"] = np.array(cutoff_matrix).tolist()

        # if there are 'forward_layer_1' in aux_compute, then add to metadata.yaml
        if self.aux_compute is not None and "forward_layer_1" in self.aux_compute:
            parallel_comm = {}
            # logger.info("Extracting shapes for forward_layer_1")
            tf_func = self._aux_tf_funcs["forward_layer_1"]
            sig = self._aux_compute_sigs["forward_layer_1"]
            concrete_func = tf_func.get_concrete_function(sig)

            output_specs = concrete_func.structured_outputs
            for key, spec in output_specs.items():
                shape = spec.shape.as_list()
                # Skip batch dimension (None)
                if shape and shape[0] is None:
                    shape = shape[1:]
                parallel_comm[key] = {
                    "shape": shape,
                    "non_local": key in self.non_local_communication_keys,
                }

            metadata["parallel_communication"] = parallel_comm

        with open(os.path.join(path, "metadata.yaml"), "w") as f:
            yaml.dump(metadata, f)

        with open(os.path.join(path, "metadata.json"), "w") as f:
            json.dump(metadata, f, indent=4)

    def compute_regularization_loss(self):
        total_reg_loss = defaultdict(lambda: 0.0)
        instructions = to_instructions_list(self.instructions)
        for ins in instructions:
            if ins.compute_l2_regularization:
                total_reg_loss[
                    constants.L2_LOSS_COMPONENT
                ] += ins.compute_l2_regularization_loss()
        return total_reg_loss

    def compute_l2_regularization_loss(self):
        l2_loss = 0.0
        instructions = to_instructions_list(self.instructions)
        for ins in instructions:
            if hasattr(ins, "compute_l2_regularization_loss"):
                l2_loss += ins.compute_l2_regularization_loss()
        return l2_loss

    def compute(self, input_data: dict) -> dict[str, tf.Tensor]:
        copy_data = input_data.copy()

        output = self.compute_function(self.instructions, copy_data, training=False)

        return output

    @tf.Module.with_name_scope
    def __call__(self, input_data, training: bool = False):
        copy_data = input_data.copy()

        output = self.train_function(self.instructions, copy_data, training)

        return output

    def export_to_yaml(self, filename):
        export_to_yaml(self.instructions, filename)

    def summary(
        self,
        verbose: summary_verbose_type = 1,
        separator: str = "=",
        separator_length: int = 50,
        **kwargs,
    ) -> str:
        """Generate a formatted summary string of all instructions."""
        break_line = "\n" + separator * separator_length + "\n"

        vars_counter: Dict[str, int] = dict()
        lines = [
            f"{i + 1}. {ins.summary(verbose, vars_counter, **kwargs)}"
            for i, ins in enumerate(self.instructions.values())
        ]

        res = break_line.join(lines)

        total_vars = sum(vars_counter.values())
        res += break_line + f"\tTotal Trainable Params: {total_vars}\n" + break_line

        return res

    def __repr__(self):
        # TODO: consider make it more __repr__-related
        return "\n".join(
            f"{i + 1}. {repr(ins)}" for i, ins in enumerate(self.instructions)
        )

    def set_trainable_variables(self, only_trainable_names: list[str], verbose=False):
        for var in self.variables:
            # check if any of only_trainable_names in var.name
            if any(
                train_var_name in var.name for train_var_name in only_trainable_names
            ):
                var._trainable = True
                if verbose:
                    logging.info(f"Setting {var.name} as trainable")
            else:
                var._trainable = False
                # if verbose:
                #     logging.info(f"Setting {var.name} as non-trainable")

    def enable_lora_adaptation(self, lora_config=None):
        if lora_config is None:
            return

        logging.info(f"Activating LoRA (config = {lora_config})")
        if "all" in lora_config:
            all_config = lora_config.pop("all")
            new_lora_config = {
                ins_name: all_config
                for ins_name, ins in self.instructions.items()
                if isinstance(ins, LORAInstructionMixin)
            }
            logging.info(
                f"LORA all available instructions: {', '.join(new_lora_config.keys())}"
            )
            new_lora_config.update(lora_config)
            lora_config = new_lora_config

        # # TODO: make all non-trainable ??
        # for var in self.trainable_variables:
        #     var._trainable = False

        for ins_name, ins in self.instructions.items():
            if ins_name in lora_config and isinstance(ins, LORAInstructionMixin):
                ins_lora_config = lora_config[ins_name]
                if ins_lora_config:
                    logging.info(
                        f" - activating LoRA for {ins_name}: {ins_lora_config}"
                    )
                    ins.enable_lora_adaptation(ins_lora_config)

    def finalize_lora_update(self):
        logging.info(f"Reducing LoRA")
        for ins_name, ins in self.instructions.items():
            if isinstance(ins, LORAInstructionMixin) and ins.lora:
                logging.info(f" - reducing LoRA for {ins_name}")
                ins.finalize_lora_update()

    def is_lora_enabled(self):
        for ins_name, ins in self.instructions.items():
            if isinstance(ins, LORAInstructionMixin) and ins.lora:
                return True
        return False
