from __future__ import annotations

from abc import ABC, abstractmethod
import os

import numpy as np
import tensorflow as tf
import yaml

from tensorpotential.instructions.compute import TPInstruction
from tensorpotential import constants
from tensorpotential.export import export_to_yaml
from collections import defaultdict


def extract_cutoff_and_elements(instructions):
    cuts = [0.0]
    element_map_symbols = None
    element_map_index = None

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
    specs: dict[str, dict]

    @abstractmethod
    def __call__(self, *args, **kwargs):
        raise NotImplementedError()


class ComputeFunction(ABC):
    specs: dict[str, dict] = {}

    @abstractmethod
    def __call__(self, *args, **kwargs):
        raise NotImplementedError()


# TODO: maybe name better
class ComputeBatchEnergyAndForces(TrainFunction):
    specs = {
        constants.BOND_IND_I: {"shape": [None], "dtype": "int"},
        constants.BOND_IND_J: {"shape": [None], "dtype": "int"},
        constants.ATOMS_TO_STRUCTURE_MAP: {"shape": [None], "dtype": "int"},
        constants.N_STRUCTURES_BATCH_TOTAL: {"shape": [], "dtype": "int"},
        constants.BOND_VECTOR: {"shape": [None, 3], "type": "float"},
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
            for ins in instructions:
                ins(input_data, training=training)
            # TODO: make sure it will have the atomic energy
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
        constants.BOND_VECTOR: {"shape": [None, 3], "type": "float"},
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
            for ins in instructions:
                ins(input_data, training=training)
            # TODO: make sure it will have the atomic energy
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
        constants.BOND_VECTOR: {"shape": [None, 3], "type": "float"},
        # "map_bonds_to_structure": {"shape": [None], "dtype": "int"},
        # constants.ATOMS_TO_STRUCTURE_MAP: {"shape": [None], "dtype": "int"},
        # constants.N_STRUCTURES_BATCH_TOTAL: {"shape": [], "dtype": "int"},
    }

    def __call__(
        self,
        instructions: list[TPInstruction],
        input_data: dict,
        training: bool = False,
    ):
        with tf.GradientTape() as tape:
            tape.watch(input_data[constants.BOND_VECTOR])
            for ins in instructions:
                ins(input_data, training=training)
            e_atomic = tf.reshape(input_data[constants.PREDICT_ATOMIC_ENERGY], [-1, 1])
        pair_f = tf.negative(tape.gradient(e_atomic, input_data[constants.BOND_VECTOR]))

        total_energy = tf.reduce_sum(e_atomic, axis=0, keepdims=True)

        nat = tf.reshape(input_data[constants.N_ATOMS_BATCH_TOTAL], [])
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


class ComputeEquivariantForces(ComputeFunction):
    specs = {}

    def __call__(
        self,
        instructions: list[TPInstruction],
        input_data: dict,
        training: bool = False,
    ):
        for ins in instructions:
            ins(input_data, training=training)
        atomic_f = tf.reshape(input_data[constants.PREDICT_FORCES], [-1, 3])

        return {
            constants.PREDICT_TOTAL_ENERGY: tf.constant([[0.0]], dtype=atomic_f.dtype),
            constants.PREDICT_FORCES: atomic_f,
        }


class ComputePlaceholder(ComputeFunction):
    specs = {}

    def __call__(
        self,
        instructions: list[TPInstruction],
        input_data: dict,
        training: bool = False,
    ):
        for ins in instructions:
            ins(input_data, training=training)
        placeholder = input_data[constants.PLACEHOLDER]

        return {
            constants.PLACEHOLDER: placeholder,
        }


class TPModel(tf.Module):
    def __init__(
        self,
        instructions,
        compute_function: ComputeFunction = ComputeStructureEnergyAndForcesAndVirial,
        train_function: TrainFunction = ComputeBatchEnergyAndForces,
        name="TPModel",
    ):
        super(TPModel, self).__init__(name=name)
        self.instructions = instructions
        self.compute_function = compute_function()  # instantiate functor
        self.train_function = train_function()  # instantiate functor

        self.compute_specs = compute_function.specs or {}
        self.train_specs = train_function.specs or {}

    def build(self, float_dtype):
        self.float_dtype = float_dtype
        for sm in self.submodules:
            if hasattr(sm, "build"):
                if not hasattr(sm, "is_built") or not sm.is_built:
                    sm.build(float_dtype)
            if hasattr(sm, "input_tensor_spec"):
                self.compute_specs.update(sm.input_tensor_spec)
                self.train_specs.update(sm.input_tensor_spec)

    def extract_compute_tensor_specs(self, float_dtype):
        pass

    def get_flat_trainable_variables(self):
        # ctx = tf.distribute.get_replica_context()
        self.count_coefs = 0
        self.slices = [0]
        flat_val = []
        for i, var in enumerate(self.trainable_variables):
            v = tf.reshape(var, [-1])
            self.count_coefs += tf.shape(v)[0]
            self.slices.append(self.count_coefs)
            flat_val += [v]
        flat_vars = tf.concat(flat_val, axis=0)

        return flat_vars

    def set_flat_trainable_variables(self, flat_vars):
        for i, var in enumerate(self.trainable_variables):
            new_values = flat_vars[self.slices[i] : self.slices[i + 1]]
            new_values = tf.reshape(new_values, tf.shape(var))
            if new_values.dtype != self.float_dtype:
                new_values = tf.cast(new_values, self.float_dtype)
            var.assign(new_values)

    def save_model(self, path: str, jit_compile: bool = False, float_dtype=tf.float64):
        dtypes = {"int": tf.int32, "float": float_dtype}
        input_signature = {}
        for k, v in self.compute_specs.items():
            input_signature[k] = tf.TensorSpec(
                shape=v["shape"], dtype=dtypes[v["dtype"]], name=k
            )

        self.compute = tf.function(
            self.compute,
            input_signature=[input_signature],
            jit_compile=jit_compile,
        )
        tf.saved_model.save(self, path)

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

        with open(os.path.join(path, "metadata.yaml"), "w") as f:
            yaml.dump(metadata, f)

    def compute_regularization_loss(self):
        total_reg_loss = defaultdict(lambda: 0.0)
        for ins in self.instructions:
            if ins.compute_l2_regularization:
                total_reg_loss[
                    constants.L2_LOSS_COMPONENT
                ] += ins.compute_l2_regularization_loss()
        return total_reg_loss

    def compute_l2_regularization_loss(self):
        l2_loss = 0.0
        for ins in self.instructions:
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
