from __future__ import annotations

import logging
from typing import Literal, Any

import numpy as np
import pandas as pd
import tensorflow as tf
from ase.data import atomic_numbers
from ase.units import _eps0, _e

from tensorpotential import constants
from tensorpotential.functions.couplings import (
    real_coupling_metainformation,
)
from tensorpotential.functions.lora import (
    initialize_lora_tensors,
    lora_reconstruction,
    apply_lora_update,
)
from tensorpotential.functions.nn import (
    FullyConnectedMLP,
    DenseLayer,
    Linear,
    ACTIVATION_DICT,
)
from tensorpotential.functions.radial import (
    SinBesselRadialBasisFunction,
    SimplifiedBesselRadialBasisFunction,
    GaussianRadialBasisFunction,
    ChebSqrRadialBasisFunction,
    compute_cheb_radial_basis,
    compute_sin_bessel_radial_basis,
    cutoff_func_p_order_poly,
)
from tensorpotential.functions.spherical_harmonics import SphericalHarmonics
from tensorpotential.instructions.base import (
    TPInstruction,
    TPEquivariantInstruction,
    capture_init_args,
    ElementsReduceInstructionMixin,
    LORAInstructionMixin,
)
from tensorpotential.poly import (
    Monomial,
    Polynomial,
    init_coupling_symbols,
    get_symbol,
    normalize_poly,
)
from tensorpotential.utils import process_cutoff_dict


class Parity:
    FULL_PARITY = [
        [0, 1],
        [1, -1],
        [1, 1],
        [2, -1],
        [2, 1],
        [3, -1],
        [3, 1],
        [4, -1],
        [4, 1],
        [5, -1],
        [5, 1],
        [6, -1],
        [6, 1],
    ]

    REAL_PARITY = [
        [0, 1],
        [1, -1],
        [2, 1],
        [3, -1],
        [4, 1],
        [5, -1],
        [6, 1],
    ]

    PSEUDO_PARITY = [
        [0, -1],
        [1, 1],
        [2, -1],
        [3, 1],
        [4, -1],
        [5, 1],
        [6, -1],
    ]

    SCALAR = [[0, 1]]

    VECTOR = [[1, -1]]

    TENSOR = [[2, 1]]


@capture_init_args
class BondLength(TPInstruction):
    """
    Takes bond vectors from the input data. Computes the bond length and
    puts it into the data dict.
    """

    input_tensor_spec = {constants.BOND_VECTOR: {"shape": [None, 3], "dtype": "float"}}

    def __init__(
        self, instruction_with_bonds: TPInstruction | str = None, name="BondLength"
    ):
        super().__init__(name=name)

        if instruction_with_bonds is not None:
            if isinstance(instruction_with_bonds, str):
                self._instruction_with_bonds_name = instruction_with_bonds
            elif isinstance(instruction_with_bonds, TPInstruction):
                self._instruction_with_bonds_name = instruction_with_bonds.name
            else:
                raise TypeError(f"Unexpected type {type(instruction_with_bonds)=}")
        else:
            self._instruction_with_bonds_name = constants.BOND_VECTOR

    @tf.Module.with_name_scope
    def build(self, float_dtype):
        self.is_built = True

    def frwrd(self, input_data: dict, training=False):

        return tf.sqrt(
            tf.reduce_sum(
                input_data[self._instruction_with_bonds_name] ** 2,
                axis=1,
                keepdims=True,
            )
            + 1e-10,
        )
        # return tf.linalg.norm(
        #     input_data[self._instruction_with_bonds_name],
        #     axis=1,
        #     keepdims=True,
        # )


@capture_init_args
class ScaledBondVector(TPInstruction):
    """
    Computes scaled bond vectors

    Parameters
    ----------
    bond_length: Union[TPInstruction, str]
        Instructions that computes bond lengths of name of the corresponding entry
    """

    input_tensor_spec = {constants.BOND_VECTOR: {"shape": [None, 3], "dtype": "float"}}

    # data_spec = {"bond_vector": GeometricalDataBuilder}

    def __init__(
        self,
        bond_length: TPInstruction | str,
        bonds: TPInstruction | str = None,
        name="ScaledBondVector",
    ):
        super().__init__(name=name)
        if isinstance(bond_length, TPInstruction):
            self.bond_length = bond_length.name
        elif isinstance(bond_length, str):
            self.bond_length = bond_length
        if bonds is not None:
            if isinstance(bonds, TPInstruction):
                self.bonds = bonds.name
            elif isinstance(bonds, str):
                self.bonds = bonds
        else:
            self.bonds = constants.BOND_VECTOR
        self.epsilon = None

    @tf.Module.with_name_scope
    def build(self, float_dtype):
        if not self.is_built:
            self.epsilon = tf.constant(1e-10, dtype=float_dtype)
            self.is_built = True

    def frwrd(self, input_data: dict, training=False):
        r_ij = input_data[self.bonds]
        d_ij = input_data[self.bond_length]
        return r_ij / (d_ij + self.epsilon)


@capture_init_args
class SphericalHarmonic(TPEquivariantInstruction):
    """
    Parameters
    ----------
    vhat: Union[TPInstruction, str]
        Instruction that computes the scaled vectors or the name of the corresponding entry

    lmax: int
        Max l for the spherical harmonics

    name: str
        Name of the resulting spherical harmonics tensor

    """

    def __init__(self, vhat: TPInstruction | str, name: str, lmax: int, **kwargs):
        super().__init__(name=name, lmax=lmax)
        if isinstance(vhat, TPInstruction):
            self.vhat = vhat.name
        elif isinstance(vhat, str):
            self.vhat = vhat
        else:
            raise ValueError(f"Unknown entry for {vhat=}")

        self.sg = SphericalHarmonics(self.lmax, **kwargs)
        self.coupling_meta_data = self.init_uncoupled_meta_data()
        self.coupling_origin = None

    @tf.Module.with_name_scope
    def build(self, float_dtype):
        if not self.sg.is_built:
            self.sg.build(float_dtype)

    def frwrd(self, input_data: dict, training=False):
        vhat = input_data[self.vhat]
        return self.sg(vhat)


@capture_init_args
class BondAvgSphericalHarmonic(TPEquivariantInstruction):
    input_tensor_spec = {
        constants.BOND_IND_I: {"shape": [None], "dtype": "int"},
        # constants.BOND_MU_J: {"shape": [None], "dtype": "int"},
        constants.N_ATOMS_BATCH_TOTAL: {"shape": [], "dtype": "int"},
    }

    def __init__(
        self,
        spherical_harmonics: SphericalHarmonic | str,
        bonds: BondLength | str,
        rcut: float,
        name: str,
        n_out: int = 1,
        p: int = 5,
        avg_n_neigh: float = 1.0,
    ):
        from tensorpotential.functions.radial import cutoff_func_p_order_poly

        super().__init__(name=name, lmax=spherical_harmonics.lmax)
        self.sg = spherical_harmonics
        self.bonds = bonds
        self.rcut = rcut
        self.p = p
        self.cutoff_func = cutoff_func_p_order_poly

        self.coupling_meta_data = self.sg.coupling_meta_data
        if isinstance(self.sg, str):
            raise NotImplementedError()
        if isinstance(self.bonds, str):
            raise NotImplementedError()

        self.inv_avg_n_neigh = 1.0 / avg_n_neigh
        self.n_out = n_out

    def build(self, float_dtype):
        if not self.is_built:
            self.inv_avg_n_neigh = tf.constant(
                self.inv_avg_n_neigh,
                dtype=float_dtype,
            )
            self.rcut = tf.constant(self.rcut, dtype=float_dtype)
            self.is_built = True

    def frwrd(self, input_data: dict, training=False):
        y = input_data[self.sg.name]
        r = input_data[self.bonds.name]
        # cut_func = self.cutoff_func(r / self.rcut, self.p)
        # cut_func = tf.where(r > self.rcut, tf.zeros_like(r, dtype=r.dtype), cut_func)

        y = tf.where(r > self.rcut, tf.zeros_like(y), y)
        # zy = tf.einsum("bn,bl->bnl", cut_func, y)
        zy = tf.math.unsorted_segment_sum(
            y,
            segment_ids=input_data[constants.BOND_IND_I],
            num_segments=input_data[constants.N_ATOMS_BATCH_TOTAL],
        )
        if self.inv_avg_n_neigh is not None:
            zy *= self.inv_avg_n_neigh

        return zy[:, tf.newaxis, :]


@capture_init_args
class RadialBasis(TPInstruction):
    """
    Selects particular type of the radial basis
    Computes it and stores into the container

    Parameters
    ----------
    bonds: Union[TPInstruction, str]
        Instruction that computes bond distances or key
        for accessing already computed bond distance values
    basis_type: str
        Type of the radial basis. Available options:
        "RadSinBessel", "SBessel"
    name: str
        Names used to store and access the output
    kwargs:
        Parameters of the radial basis
    """

    def __init__(
        self,
        bonds: TPInstruction | str,
        basis_type: str,
        name: str = "RadialBasis",
        **kwargs,
    ):
        super().__init__(name=name)
        if isinstance(bonds, TPInstruction):
            self.input_name = bonds.name
        elif isinstance(bonds, str):
            self.input_name = bonds
        else:
            raise ValueError("Unknown entry for bonds")
        self.kwargs = kwargs
        self.basis_type = basis_type

        if basis_type == "RadSinBessel":
            self.basis_function = SinBesselRadialBasisFunction(**kwargs)
        elif basis_type == "SBessel":
            self.basis_function = SimplifiedBesselRadialBasisFunction(**kwargs)
        elif basis_type == "Gaussian":
            self.basis_function = GaussianRadialBasisFunction(**kwargs)
        elif basis_type == "Cheb":
            self.basis_function = ChebSqrRadialBasisFunction(**kwargs)
        else:
            raise ValueError(f"Unknown type of the radial basis, {basis_type}")
        self.nfunc = self.basis_function.nfunc

    @tf.Module.with_name_scope
    def build(self, float_dtype):
        if hasattr(self.basis_function, "build"):
            self.basis_function.build(float_dtype)
            # self.norm = tf.cast(np.sqrt(self.basis_function.nfunc), dtype=float_dtype)

    @tf.Module.with_name_scope
    def frwrd(self, input_data: dict, training=False):
        r = input_data[self.input_name]
        basis = self.basis_function(r)  # * self.norm

        return basis


@capture_init_args
class BondSpecificRadialBasisFunction(TPInstruction, ElementsReduceInstructionMixin):
    """
    Radial Basis function with a cutoff defined specifically for each
    i) element:
        for each type mu_i of a central atom i, a cutoff radius r_c(mu_i) is defined.
        An atom j of any type mu_j inside r_c(mu_i) around atom i,
         counts as a neighbor of atom i.
         specified as {('mu_1',): r_c(mu_1), ('mu_2',): r_c(mu_2), ...}
    ii) symmetric bond:
        for each atomic pair ij of types mu_i, mu_j, a cutoff radius r_c(mu_i, mu_j) is defined,
        such that r_c(mu_i, mu_j) == r_c(mu_j, mu_i)
        specified as {('mu_1', 'mu_2'): r_c(mu_1, mu_2), ...}
    iii) bond
        for each atomic pair ij of types mu_i, mu_j, a cutoff radius r_c(mu_i, mu_j) is defined,
        such that r_c(mu_i, mu_j) != r_c(mu_j, mu_i)
        specified as {('mu_1', 'mu_2'): r_c(mu_1, mu_2), ...}

    Parameters
    ----------
    :cutoff_dict: dict - cutoff radius of each element pair, {'AB': 3.22, ...}
    :cutoff_type: str - one of the following: 'element', 'bond', 'symmetric_bond' (default)

    """

    input_tensor_spec = {
        constants.BOND_IND_I: {"shape": [None], "dtype": "int"},
        constants.BOND_MU_I: {"shape": [None], "dtype": "int"},
        constants.BOND_MU_J: {"shape": [None], "dtype": "int"},
        constants.N_ATOMS_BATCH_TOTAL: {"shape": [], "dtype": "int"},
    }

    def __init__(
        self,
        bonds: TPInstruction | str,
        element_map: dict,
        cutoff_dict: dict | str,
        cutoff: float = 5.0,
        cutoff_type: str = "symmetric_bond",  # or bond
        cutoff_function_param: float = 5,
        basis_type: str = "Cheb",
        nfunc: int = 8,
        name: str = "BondSpecificRadialBasisFunction",
        normalize: bool = False,
        init_gamma: float = 1.0,
        trainable: bool = False,
        **kwargs,
    ):
        super().__init__(name=name)
        if isinstance(bonds, TPInstruction):
            self.input_name = bonds.name
        elif isinstance(bonds, str):
            self.input_name = bonds
        else:
            raise ValueError("Unknown entry for bonds")

        self.nelem = len(element_map)

        self.cutoff_type = cutoff_type
        assert self.cutoff_type in ["bond", "symmetric_bond"]
        self.cutoff_dict = process_cutoff_dict(cutoff_dict, element_map)
        self.default_cutoff = cutoff
        self.cutoff_function_param = cutoff_function_param
        self.normalize = normalize

        # by construction, this includes ALL combinations, initialized with default cutoff
        bond_ind_cut = np.ones((self.nelem, self.nelem)) * self.default_cutoff

        for (el0, el1), cut in self.cutoff_dict.items():
            i0 = element_map[el0]
            i1 = element_map[el1]
            bond_ind_cut[i0, i1] = cut
            if self.cutoff_type == "symmetric_bond":
                bond_ind_cut[i1, i0] = cut

        self.bond_cutoff_map = bond_ind_cut.flatten().reshape(-1, 1).astype(np.float64)
        self.nfunc = nfunc

        self.basis_type = basis_type
        if self.basis_type == "Gaussian":
            max_cut = np.max(self.bond_cutoff_map)
            self.grid = np.linspace(0, max_cut, self.nfunc).reshape(1, -1)
            self.scales = np.ones_like(self.grid) * init_gamma
            self.trainable = trainable

    def build(self, float_dtype):
        if not self.is_built:
            self.bond_cutoff_map = tf.Variable(
                self.bond_cutoff_map,
                dtype=float_dtype,
                trainable=False,
                name="RBF_cutoff",
            )
            if self.basis_type == "Gaussian":
                self.grid = tf.Variable(
                    self.grid, dtype=float_dtype, trainable=self.trainable
                )
                self.scales = tf.Variable(
                    self.scales, dtype=float_dtype, trainable=self.trainable
                )
            self.is_built = True

    def prepare_variables_for_selected_elements(self, index_to_select):
        bond_cutoff_map = np.array(self.bond_cutoff_map).reshape(self.nelem, self.nelem)
        index_to_select = np.array(index_to_select)
        new_bond_cutoff_map = bond_cutoff_map[
            index_to_select[:, np.newaxis], index_to_select
        ]
        return {
            "bond_cutoff_map": tf.Variable(
                new_bond_cutoff_map.reshape(-1, 1).astype(np.float64)
            )
        }

    def upd_init_args_new_elements(self, new_element_map):
        els = list(new_element_map.keys())
        new_nelem = len(els)
        bond_cutoff_map = np.array(self.bond_cutoff_map).reshape((new_nelem, new_nelem))

        new_cutoff_dict = {}

        for el1 in els:
            for el2 in els:
                new_cutoff_dict["".join((el1, el2))] = bond_cutoff_map[
                    new_element_map[el1], new_element_map[el2]
                ]

        self._init_args["cutoff_dict"] = new_cutoff_dict
        self._init_args["element_map"] = new_element_map

    def frwrd(self, input_data: dict, training: bool = False):
        d = input_data[self.input_name]
        mu_i = input_data[constants.BOND_MU_I]
        mu_j = input_data[constants.BOND_MU_J]
        mu_ij = mu_j + mu_i * tf.constant(self.nelem, dtype=mu_i.dtype)
        cutoff = tf.gather(self.bond_cutoff_map, mu_ij)
        if self.basis_type == "Cheb":
            basis = compute_cheb_radial_basis(
                d, self.nfunc, cutoff, self.cutoff_function_param
            )
        elif self.basis_type == "RadSinBessel":
            basis = compute_sin_bessel_radial_basis(
                d,
                self.nfunc,
                cutoff,
                self.cutoff_function_param,
                normalized=self.normalize,
            )
        elif self.basis_type == "Gaussian":
            basis = tf.math.exp(-(self.scales**2) * (d - self.grid) ** 2)
            basis *= cutoff_func_p_order_poly(d / cutoff, self.cutoff_function_param)
        else:
            raise NotImplementedError("Cheb or RadSinBessel basis only for now")
        return tf.where(d >= cutoff, tf.zeros_like(basis, dtype=d.dtype), basis)


@capture_init_args
class LinearRadialFunction(TPInstruction):
    input_tensor_spec = {
        constants.BOND_MU_I: {"shape": [None], "dtype": "int"},
        constants.BOND_MU_J: {"shape": [None], "dtype": "int"},
    }

    def __init__(
        self,
        n_rad_max: int,
        lmax: int,
        basis: RadialBasis | TPInstruction | str = None,
        input_shape: int = None,
        init: str = "random",
        name="LinearRadialFunction",
        no_weight_decay: bool = True,
        **kwargs,
    ):
        super().__init__(name=name)
        self.n_rad_max = n_rad_max
        self.lmax = lmax
        if isinstance(basis, RadialBasis):
            self.basis_name = basis.name
            self.input_shape = basis.basis_function.nfunc
        elif isinstance(basis, TPInstruction):
            self.basis_name = basis.name
            self.input_shape = basis.nfunc
        elif isinstance(basis, str):
            self.basis_name = basis
            assert (
                input_shape is not None
            ), f"Need to provide shape if basis is not TPInstruction"
            self.input_shape = input_shape
        elif basis is not None:
            raise ValueError(f"Unknown {basis = }")

        self.l_tile = tf.cast(
            tf.concat([tf.ones((2 * l + 1)) * l for l in range(self.lmax + 1)], axis=0),
            tf.int32,
        )
        self.init = init
        self.n_out = self.n_rad_max * (self.lmax + 1)
        if no_weight_decay:
            self.no_decay = "no_decay"
        else:
            self.no_decay = "_"

    @tf.Module.with_name_scope
    def build(self, float_dtype):
        if not self.is_built:
            self.norm = tf.convert_to_tensor(1, dtype=float_dtype)
            if self.init == "random":
                limit = np.sqrt(2 / float(self.n_rad_max + self.input_shape))
                # limit = 1.
                self.crad = tf.Variable(
                    tf.random.normal(
                        [self.n_rad_max, self.lmax + 1, self.input_shape],
                        stddev=limit,
                        dtype=float_dtype,
                    ),
                    name=f"crad_{self.no_decay}",
                )
            elif self.init == "delta":
                crad = np.zeros([self.n_rad_max, self.lmax + 1, self.input_shape])
                for c in range(min(self.n_rad_max, self.input_shape)):
                    crad[c, :, c] = 1.0
                self.crad = tf.Variable(
                    crad, name=f"crad_{self.no_decay}", dtype=float_dtype
                )
            else:
                raise NotImplementedError(
                    f"LinearRadialFunction.init={self.init}  is not implemented"
                )
        self.is_built = True

    def frwrd(self, input_data, training=False):
        basis = input_data[self.basis_name]

        y = tf.einsum("nlk,ak->anl", self.crad, basis)
        y *= self.norm
        y_l = tf.gather(y, self.l_tile, axis=-1)

        return y_l


@capture_init_args
class MLPRadialFunction(TPInstruction, LORAInstructionMixin):
    input_tensor_spec = {
        constants.BOND_MU_I: {"shape": [None], "dtype": "int"},
        constants.BOND_MU_J: {"shape": [None], "dtype": "int"},
    }

    def __init__(
        self,
        n_rad_max: int,
        lmax: int,
        basis: RadialBasis | TPInstruction | str = None,
        input_shape: int = None,
        hidden_layers: list[int] = None,
        norm: bool = False,
        name="MLPRadialFunction",
        activation: str = None,
        no_weight_decay: bool = True,
        chemical_embedding_i: ScalarChemicalEmbedding = None,
        chemical_embedding_j: ScalarChemicalEmbedding = None,
        lora_config: dict[str, Any] = None,
        **kwargs,
    ):
        super().__init__(name=name)
        LORAInstructionMixin.__init__(self, lora_config)  # explicitly call

        self.n_rad_max = n_rad_max
        self.lmax = lmax
        if hidden_layers is None:
            self.hidden_layers = [64, 64, 64]
        else:
            self.hidden_layers = hidden_layers

        if isinstance(basis, RadialBasis):
            self.basis_name = basis.name
            self.input_shape = basis.basis_function.nfunc
        elif isinstance(basis, str):
            self.basis_name = basis
            assert (
                input_shape is not None
            ), f"Need to provide shape if basis is not TPInstruction"
            self.input_shape = input_shape
        elif isinstance(basis, TPInstruction):
            self.basis_name = basis.name
            self.input_shape = basis.nfunc
        elif basis is not None:
            raise ValueError(f"Unknown {basis = }")
        elif basis is None:
            assert (
                input_shape is not None
            ), f"Need to provide shape if basis is not TPInstruction"
            self.input_shape = input_shape
        self.chemical_embedding_i = chemical_embedding_i
        self.chemical_embedding_j = chemical_embedding_j

        if self.chemical_embedding_i is not None:
            self.input_shape += self.chemical_embedding_i.embedding_size
        if self.chemical_embedding_j is not None:
            self.input_shape += self.chemical_embedding_j.embedding_size

        self.l_tile = tf.cast(
            tf.concat([tf.ones((2 * l + 1)) * l for l in range(self.lmax + 1)], axis=0),
            tf.int32,
        )
        if activation is None:
            self.mlp = FullyConnectedMLP(
                input_size=self.input_shape,
                hidden_layers=self.hidden_layers,
                output_size=self.n_rad_max * (self.lmax + 1),
                no_weight_decay=no_weight_decay,
                lora_config=lora_config,
                name=self.name + "_MLP",
            )
        elif isinstance(activation, str):
            self.mlp = FullyConnectedMLP(
                input_size=self.input_shape,
                hidden_layers=self.hidden_layers,
                output_size=self.n_rad_max * (self.lmax + 1),
                activation=activation,
                no_weight_decay=no_weight_decay,
                lora_config=lora_config,
                name=self.name + "_MLP",
            )
        else:
            raise ValueError("MLP activation must be predefined str or None")
        self.n_out = self.n_rad_max * (self.lmax + 1)
        self.norm = norm

    @tf.Module.with_name_scope
    def build(self, float_dtype):
        if not self.is_built:
            if not self.mlp.is_built:
                self.mlp.build(float_dtype)
            if self.norm:
                self.gamma = tf.Variable(
                    tf.random.normal(shape=[1, self.n_out], dtype=float_dtype)
                )
                self.epsilon = tf.convert_to_tensor(1e-5, dtype=float_dtype)

            if self.lora_config:
                self.enable_lora_adaptation(self.lora_config)

            self.is_built = True

    @tf.Module.with_name_scope
    def enable_lora_adaptation(self, lora_config: dict[str, Any]):
        super().enable_lora_adaptation(lora_config)
        self.mlp.enable_lora_adaptation(lora_config)

    def finalize_lora_update(self):
        # common part
        super().finalize_lora_update()
        self.mlp.finalize_lora_update()

    def frwrd(self, input_data, training=False):
        basis = input_data[self.basis_name]

        if self.chemical_embedding_i is not None:
            z = input_data[self.chemical_embedding_i.name]
            mu_i = input_data[constants.BOND_MU_I]
            bond_z_i = tf.gather(z, mu_i, axis=0)
            basis = tf.concat([basis, bond_z_i], axis=1)

        if self.chemical_embedding_j is not None:
            z = input_data[self.chemical_embedding_j.name]
            mu_j = input_data[constants.BOND_MU_J]
            bond_z_j = tf.gather(z, mu_j, axis=0)
            basis = tf.concat([basis, bond_z_j], axis=1)

        y = self.mlp(basis)
        if self.norm:
            variance = tf.math.reduce_variance(y, axis=-1, keepdims=True, name=None)
            inv = tf.math.rsqrt(variance + self.epsilon)

            y = y * inv * self.gamma

        y = tf.reshape(y, [-1, self.n_rad_max, self.lmax + 1])
        y_l = tf.gather(y, self.l_tile, axis=-1)
        return y_l


@capture_init_args
class MLPRadialFunction_v2(TPInstruction, LORAInstructionMixin):
    input_tensor_spec = {
        constants.BOND_MU_I: {"shape": [None], "dtype": "int"},
        constants.BOND_MU_J: {"shape": [None], "dtype": "int"},
    }

    def __init__(
        self,
        n_rad_max: int,
        lmax: int,
        basis: RadialBasis | TPInstruction | str = None,
        input_shape: int = None,
        hidden_layers: list[int] = None,
        name="MLPRadialFunction",
        activation: list[str] | str = None,
        no_weight_decay: bool = True,
        init_type: str = "normal",
        normalize: bool = True,
        chem_embedding: ScalarChemicalEmbedding = None,
        embed_i: bool = False,
        embed_j: bool = True,
        lora_config: dict[str, Any] = None,
        **kwargs,
    ):
        super().__init__(name=name)
        LORAInstructionMixin.__init__(self, lora_config)  # explicitly call

        self.n_rad_max = n_rad_max
        self.lmax = lmax
        if hidden_layers is None:
            self.hidden_layers = [64, 64]
        else:
            self.hidden_layers = hidden_layers
        if activation is None:
            self.activation = ["silu" for _ in self.hidden_layers]
        else:
            if isinstance(activation, str):
                activation = [activation] * len(self.hidden_layers)
            assert len(activation) == len(self.hidden_layers)
            self.activation = activation

        if isinstance(basis, RadialBasis):
            self.basis_name = basis.name
            self.input_shape = basis.basis_function.nfunc
        elif isinstance(basis, str):
            self.basis_name = basis
            assert (
                input_shape is not None
            ), f"Need to provide shape if basis is not TPInstruction"
            self.input_shape = input_shape
        elif isinstance(basis, TPInstruction):
            self.basis_name = basis.name
            self.input_shape = basis.nfunc
        elif basis is not None:
            raise ValueError(f"Unknown {basis = }")
        elif basis is None:
            assert (
                input_shape is not None
            ), f"Need to provide shape if basis is not TPInstruction"
            self.input_shape = input_shape

        self.l_tile = tf.cast(
            tf.concat([tf.ones((2 * l + 1)) * l for l in range(self.lmax + 1)], axis=0),
            tf.int32,
        )
        total_shapes = (
            [self.input_shape] + self.hidden_layers + [self.n_rad_max * (self.lmax + 1)]
        )
        self.layers = []
        for i, (n_in, n_out) in enumerate(zip(total_shapes, total_shapes[1:])):
            self.layers.append(
                Linear(
                    n_in,
                    n_out,
                    name=f"{self.name}_Linear_{i}",
                    no_weight_decay=no_weight_decay,
                    init_type=init_type,
                    normalize=normalize,
                    lora_config=lora_config,
                )
            )
        self.n_out = self.n_rad_max * (self.lmax + 1)

        self.chem_embedding = chem_embedding
        if self.chem_embedding is not None:
            self.embed_transform = Linear(
                self.chem_embedding.embedding_size,
                self.n_out,
                name=f"{self.name}_ChemEmb_Linear",
                lora_config=lora_config,
            )
            self.embed_i = embed_i
            self.embed_j = embed_j

    @tf.Module.with_name_scope
    def build(self, float_dtype):
        if not self.is_built:
            for layer in self.layers:
                layer.build(float_dtype)
            if self.chem_embedding is not None:
                self.embed_transform.build(float_dtype)

            if self.lora_config:
                self.enable_lora_adaptation(self.lora_config)

            self.is_built = True

    @tf.Module.with_name_scope
    def enable_lora_adaptation(self, lora_config: dict[str, Any]):
        super().enable_lora_adaptation(lora_config)
        for layer in self.layers:
            layer.enable_lora_adaptation(lora_config)

        if self.chem_embedding is not None:
            self.embed_transform.enable_lora_adaptation(lora_config)

    def finalize_lora_update(self):
        # common part
        super().finalize_lora_update()

        for layer in self.layers:
            layer.finalize_lora_update()

        if self.chem_embedding is not None:
            self.embed_transform.finalize_lora_update()

    def frwrd(self, input_data, training=False):
        basis = input_data[self.basis_name]

        for act, layer in zip(self.activation, self.layers):
            act = ACTIVATION_DICT[act]
            basis = act(layer(basis))
        y = self.layers[-1](basis)

        if self.chem_embedding is not None:
            z = self.embed_transform(input_data[self.chem_embedding.name])
            embedding = 1.0
            if self.embed_j:
                mu_j = tf.gather(z, input_data[constants.BOND_MU_J], axis=0)
                embedding *= mu_j
            if self.embed_i:
                mu_i = tf.gather(z, input_data[constants.BOND_MU_I], axis=0)
                embedding *= mu_i
                embedding = tf.nn.tanh(embedding)
            y *= embedding

        y = tf.reshape(y, [-1, self.n_rad_max, self.lmax + 1])
        y_l = tf.gather(y, self.l_tile, axis=-1)
        return y_l


@capture_init_args
class ScalarChemicalEmbedding(
    TPInstruction, LORAInstructionMixin, ElementsReduceInstructionMixin
):
    """
    Vector chemical embedding

    Parameters
    ----------
    element_map: dict elem -> int

    embedding_size: int

    """

    def __init__(
        self,
        name: str,
        element_map: dict,
        embedding_size: int,
        is_trainable: bool = True,
        init: str = "random",
        lora_config: dict[str, Any] = None,
    ):
        super().__init__(name=name)
        LORAInstructionMixin.__init__(self, lora_config)  # explicitly call

        self.element_map_symbols = tf.Variable(
            list(element_map.keys()), trainable=False, name="element_map_symbols"
        )
        self.element_map_index = tf.Variable(
            list(element_map.values()), trainable=False, name="element_map_index"
        )
        self.number_of_elements = len(element_map)
        self.embedding_size = embedding_size
        self.is_trainable = is_trainable
        self.init = init

    @tf.Module.with_name_scope
    def build(self, float_dtype):
        if not self.is_built:
            shape = [self.number_of_elements, self.embedding_size]
            if self.init == "random":
                self.w = tf.Variable(
                    tf.random.normal(
                        shape=shape,
                        dtype=float_dtype,
                    ),
                    trainable=self.is_trainable,
                    name="ChemicalEmbedding",
                )
            elif self.init == "zero" or self.init == "zeros":
                self.w = tf.Variable(
                    tf.zeros(
                        shape=shape,
                        dtype=float_dtype,
                    ),
                    trainable=self.is_trainable,
                    name="ChemicalEmbedding",
                )
            elif self.init == "delta":
                w = np.zeros(shape)
                for c in range(min(shape)):
                    w[c, c] = 1.0
                self.w = tf.Variable(
                    w,
                    dtype=float_dtype,
                    trainable=self.is_trainable,
                    name="ChemicalEmbedding",
                )
            else:
                raise NotImplementedError(
                    f"Unknown initialization method: ScalarChemicalEmbedding.init={self.init}"
                )

            if self.lora_config:
                self.enable_lora_adaptation(self.lora_config)

            self.is_built = True

    def frwrd(self, input_data: dict, training=False):
        if not self.lora:
            return self.w
        else:
            return self.w + lora_reconstruction(
                *self.lora_tensors, lora_config=self.lora_config
            )

    def get_index_to_select(self, elements_to_select):
        index_to_select = []
        for ei, es in zip(
            self.element_map_index.numpy(), self.element_map_symbols.numpy()
        ):
            if es.decode() in elements_to_select:
                index_to_select.append(ei)

        return tf.constant(index_to_select, dtype=tf.int32)

    def prepare_variables_for_selected_elements(self, index_to_select):
        return {
            "element_map_symbols": tf.Variable(
                tf.gather(self.element_map_symbols, index_to_select, axis=0),
                trainable=False,
            ),
            "element_map_index": tf.Variable(
                tf.range(0, len(index_to_select), dtype=self.element_map_index.dtype),
                trainable=False,
            ),
            "w": tf.Variable(tf.gather(self.w, index_to_select, axis=0)),
        }

    def upd_init_args_new_elements(self, new_element_map):
        self._init_args["element_map"] = new_element_map

    @tf.Module.with_name_scope
    def enable_lora_adaptation(self, lora_config: dict[str, Any]):
        super().enable_lora_adaptation(lora_config)
        self.lora_tensors = initialize_lora_tensors(self.w, lora_config, name="w")

    def finalize_lora_update(self):
        apply_lora_update(self.w, *self.lora_tensors, lora_config=self.lora_config)
        del self.lora_tensors
        super().finalize_lora_update()


@capture_init_args
class SingleParticleBasisFunctionScalarInd(
    TPEquivariantInstruction, LORAInstructionMixin
):
    """
    Compute ACE single particle basis function with scalar indicator or without any

    """

    input_tensor_spec = {
        constants.BOND_IND_I: {"shape": [None], "dtype": "int"},
        constants.BOND_MU_J: {"shape": [None], "dtype": "int"},
        constants.ATOMIC_MU_I: {"shape": [None], "dtype": "int"},
        constants.N_ATOMS_BATCH_TOTAL: {"shape": [], "dtype": "int"},
    }

    def __init__(
        self,
        name: str,
        radial: TPInstruction,
        angular: SphericalHarmonic,
        indicator: ScalarChemicalEmbedding = None,
        indicator_l_depend: bool = False,
        sum_neighbors: bool = True,
        avg_n_neigh: float | dict = 1.0,
        lora_config: dict[str, Any] = None,
        lmax: int = None,
    ):
        if lmax is not None:
            assert (
                angular.lmax >= lmax
            ), f"{angular.lmax=} is too small for specified {lmax=}"
        else:
            lmax = angular.lmax
        super().__init__(name=name, lmax=lmax)
        LORAInstructionMixin.__init__(self, lora_config)  # explicitly call
        self.radial = radial
        self.angular = angular
        self.indicator = indicator
        self.sum_neighbors = sum_neighbors
        if isinstance(avg_n_neigh, float):
            self.per_specie_n_neigh = False
            self.inv_avg_n_neigh = 1.0 / avg_n_neigh
        elif isinstance(avg_n_neigh, dict):
            self.per_specie_n_neigh = True
            self.inv_avg_n_neigh = np.zeros((len(avg_n_neigh), 1))
            for k, v in avg_n_neigh.items():
                val = v if v > 0 else 1.0
                self.inv_avg_n_neigh[k] = 1.0 / val
        else:
            raise TypeError("avg_n_neigh must be float or dict")

        self.indicator_l_depend = indicator_l_depend
        if self.indicator is None:
            self.lin_transform = None
        else:
            n_out = (
                self.radial.n_rad_max
                if not self.indicator_l_depend
                else self.radial.n_rad_max * (self.lmax + 1)
            )
            self.lin_transform = DenseLayer(
                n_in=self.indicator.embedding_size,
                n_out=n_out,
                activation=None,
                name=self.name + "_ChemIndTransf",
                lora_config=lora_config,
            )

        self.n_out = self.radial.n_rad_max

        if self.angular.lmax > self.lmax:
            self.slice_angular = (self.lmax + 1) ** 2
        else:
            self.slice_angular = None
        assert (
            self.radial.lmax == self.lmax
        ), "Radial part and angular part lmax do not match"

        if self.slice_angular is not None:
            self.coupling_meta_data = self.angular.coupling_meta_data.query(
                f"l <= {self.lmax}"
            )
        else:
            self.coupling_meta_data = self.angular.coupling_meta_data
        self.coupling_origin = self.angular.coupling_origin

        init_coupling_symbols(self)

    @tf.Module.with_name_scope
    def build(self, float_dtype):
        if not self.is_built:
            if self.lin_transform is not None and not self.lin_transform.is_built:
                self.lin_transform.build(float_dtype)
            self.inv_avg_n_neigh = tf.constant(
                self.inv_avg_n_neigh,
                dtype=float_dtype,
            )
            self.is_built = True

    def frwrd(self, input_data: dict, training=False):
        r = input_data[self.radial.name]
        y = input_data[self.angular.name]
        if self.slice_angular is not None:
            y = y[:, : self.slice_angular]

        # y = y[:, tf.newaxis, :]

        a_nl = tf.einsum("jnl,jl->jnl", r, y)
        if self.indicator is not None:
            z = input_data[self.indicator.name]
            mu_j = input_data[constants.BOND_MU_J]

            z_tr = self.lin_transform(z)
            if self.indicator_l_depend:
                z_tr = tf.reshape(z_tr, [-1, self.radial.n_rad_max, self.lmax + 1])
                z_tr = tf.gather(z_tr, self.radial.l_tile, axis=2)
                bond_z_tr = tf.gather(z_tr, mu_j, axis=0)
                a_nl = tf.einsum("jnl,jnl->jnl", a_nl, bond_z_tr)
            else:
                bond_z_tr = tf.gather(z_tr, mu_j, axis=0)
                a_nl = tf.einsum("jnl,jn->jnl", a_nl, bond_z_tr)

        if self.sum_neighbors:
            ind_i = input_data[constants.BOND_IND_I]
            batch_tot_nat = input_data[constants.N_ATOMS_BATCH_TOTAL]
            a_nl = tf.math.unsorted_segment_sum(
                a_nl, segment_ids=ind_i, num_segments=batch_tot_nat
            )
            if self.inv_avg_n_neigh is not None:
                if self.per_specie_n_neigh:
                    nneigh_norm = tf.gather(
                        self.inv_avg_n_neigh,
                        input_data[constants.ATOMIC_MU_I],
                        axis=0,
                    )
                    a_nl *= nneigh_norm[:, :, tf.newaxis]
                    pass
                else:
                    a_nl *= self.inv_avg_n_neigh

        return a_nl

    @tf.Module.with_name_scope
    def enable_lora_adaptation(self, lora_config: dict[str, Any]):
        super().enable_lora_adaptation(lora_config)
        if self.lin_transform is not None:
            assert not self.lin_transform.use_bias
            self.lin_transform.enable_lora_adaptation(lora_config)

    def finalize_lora_update(self):
        super().finalize_lora_update()
        if self.lin_transform is not None:
            self.lin_transform.finalize_lora_update()


@capture_init_args
class SingleParticleBasisFunctionEquivariantInd(TPEquivariantInstruction):
    """ """

    input_tensor_spec = {
        constants.BOND_IND_I: {"shape": [None], "dtype": "int"},
        constants.BOND_IND_J: {"shape": [None], "dtype": "int"},
        constants.N_ATOMS_BATCH_TOTAL: {"shape": [], "dtype": "int"},
    }

    def __init__(
        self,
        angular: SphericalHarmonic,
        indicator: TPEquivariantInstruction,
        name: str,
        lmax: int,
        Lmax: int,
        radial: TPInstruction = None,
        radial_basis: RadialBasis | TPInstruction = None,
        hidden_layers: list[int] = None,
        keep_parity: list[list] = None,
        history_drop_list: list = None,
        l_max_ind: int = None,
        max_sum_l: int = None,
        sum_neighbors: bool = True,
        avg_n_neigh: float | dict = 1.0,
        normalize: bool = False,
        **kwargs,
    ):
        super().__init__(name=name, lmax=Lmax)
        self.angular = angular
        self.indicator = indicator

        self.radial = radial
        self.internal_radial = False
        if self.radial is None:
            assert (
                radial_basis is not None
            ), f"{self.__class__.__name__}_{self.name}: If :radial: is None, :radial_basis: should be provided"
            if hidden_layers is None:
                self.hidden_layers = [64, 64]
            else:
                self.hidden_layers = hidden_layers
            self.radial_basis = radial_basis
            self.internal_radial = True

        if self.internal_radial:
            self.n_out = self.indicator.n_out
        else:
            self.n_out = self.radial.n_rad_max

        self.sum_neighbors = sum_neighbors
        if isinstance(avg_n_neigh, float):
            self.per_specie_n_neigh = False
            self.inv_avg_n_neigh = 1.0 / avg_n_neigh
        elif isinstance(avg_n_neigh, dict):
            self.per_specie_n_neigh = True
            self.inv_avg_n_neigh = np.zeros((len(avg_n_neigh), 1))
            for k, v in avg_n_neigh.items():
                val = v if v > 0 else 1.0
                self.inv_avg_n_neigh[k] = 1.0 / val
        else:
            raise TypeError("avg_n_neigh must be float or dict")

        assert self.angular.coupling_meta_data is not None
        assert self.indicator.coupling_meta_data is not None

        if self.angular.lmax > lmax:
            if self.radial is not None:
                assert self.radial.lmax == lmax, (
                    f"Radial function is prepared for lmax={self.radial.lmax},"
                    f" while currently specified value is {lmax=}"
                )
            self.slice_angular = (lmax + 1) ** 2
        else:
            self.slice_angular = None
            if self.radial is not None:
                assert (
                    self.radial.lmax == self.angular.lmax
                ), "Radial part and angular part lmax do not match"

        if keep_parity is None:
            plist = []
            for l in range(Lmax + 1):
                p = 1 if l % 2 == 0 else -1
                plist.append([l, p])
        else:
            plist = keep_parity

        self.coupling_meta_data = real_coupling_metainformation(
            A=self.angular.coupling_meta_data,
            B=self.indicator.coupling_meta_data,
            lmax=lmax,
            lmax_B=l_max_ind,
            Lmax=Lmax,
            history_drop_list=history_drop_list,
            max_sum_l=max_sum_l,
            keep_parity=plist,
            normalize=normalize,
        )
        self.coupling_origin = [self.angular.name, self.indicator.name]
        if self.internal_radial:
            left = np.concatenate(self.coupling_meta_data["left_inds"]).reshape(-1, 1)
            right = np.concatenate(self.coupling_meta_data["right_inds"]).reshape(-1, 1)
            il = []
            for i, row in self.coupling_meta_data.iterrows():
                il.append([row["l"]] * len(row["left_inds"]))
            il = np.concatenate(il).reshape(-1, 1)
            self.lr_inds = tf.constant(
                np.concatenate([il, left, right], axis=1), dtype=tf.int32
            )
        else:
            self.lr_inds = tf.constant(
                np.concatenate(
                    [
                        np.concatenate(self.coupling_meta_data["left_inds"]).reshape(
                            -1, 1
                        ),
                        np.concatenate(self.coupling_meta_data["right_inds"]).reshape(
                            -1, 1
                        ),
                    ],
                    axis=1,
                ),
                dtype=tf.int32,
            )

        cgs = self.coupling_meta_data["cg_list"]
        sum_ind = []
        for i, cg in enumerate(cgs):
            sum_ind.append([i] * len(cg))
        sum_ind = np.concatenate(sum_ind)
        self.m_sum_ind = tf.constant(sum_ind, dtype=tf.int32)
        nfunc = np.max(sum_ind) + 1
        self.nfunc = tf.constant(nfunc, dtype=tf.int32)

        self.cg = np.concatenate(cgs).reshape(-1, 1, 1)
        if self.internal_radial:
            self.mlp = FullyConnectedMLP(
                input_size=radial_basis.nfunc,
                hidden_layers=self.hidden_layers,
                output_size=self.indicator.n_out
                * (self.angular.lmax + 1)
                * (self.lmax + 1),
                name=self.name + "_MLP",
            )
            self.l_tile = tf.cast(
                tf.concat(
                    [tf.ones((2 * l + 1)) * l for l in range(self.angular.lmax + 1)],
                    axis=0,
                ),
                tf.int32,
            )
        else:
            self.mlp = None

        init_coupling_symbols(self)

    @tf.Module.with_name_scope
    def build(self, float_dtype):
        if not self.is_built:
            if self.mlp is not None and not self.mlp.is_built:
                self.mlp.build(float_dtype)

            self.inv_avg_n_neigh = tf.constant(
                self.inv_avg_n_neigh,
                dtype=float_dtype,
            )

            self.cg = tf.constant(self.cg, dtype=float_dtype)
            self.is_built = True

    def frwrd(self, input_data: dict, training=False):
        ind_i = input_data[constants.BOND_IND_I]
        ind_j = input_data[constants.BOND_IND_J]
        batch_tot_nat = input_data[constants.N_ATOMS_BATCH_TOTAL]

        I = input_data[self.indicator.name]
        y = input_data[self.angular.name]
        bond_I = tf.gather(I, ind_j, axis=0)

        if self.internal_radial:
            rbasis = input_data[self.radial_basis.name]
            r = self.mlp(rbasis)
            r = tf.reshape(r, [-1, self.n_out, self.lmax + 1, self.angular.lmax + 1])
            r = tf.gather(r, self.l_tile, axis=-1)

            prod = tf.einsum("jl,jnol->jnol", y, r)
            prod = tf.einsum("jnol,jnr->jnolr", prod, bond_I)
        else:
            r = input_data[self.radial.name]
            if self.slice_angular is not None:
                y = y[:, : self.slice_angular]
            bond_RY_nl = tf.einsum("jnl,jl->jnl", r, y)
            prod = tf.einsum("jnl,jnr->jnlr", bond_RY_nl, bond_I)

        if self.sum_neighbors:
            prod = tf.math.unsorted_segment_sum(
                prod,
                segment_ids=ind_i,
                num_segments=batch_tot_nat,
                name=f"sum_nei_{self.name}",
            )
            if self.inv_avg_n_neigh is not None:
                if self.per_specie_n_neigh:
                    nneigh_norm = tf.gather(
                        self.inv_avg_n_neigh,
                        input_data[constants.ATOMIC_MU_I],
                        axis=0,
                    )
                    prod *= nneigh_norm[:, :, tf.newaxis]
                else:
                    prod *= self.inv_avg_n_neigh
        if self.internal_radial:
            tnsr = tf.transpose(prod, [2, 3, 4, 0, 1])  # tf.roll
        else:
            tnsr = tf.transpose(prod, [2, 3, 0, 1])  # tf.roll

        prod_g = tf.gather_nd(params=tnsr, indices=self.lr_inds)
        prod_g = prod_g * self.cg
        prod_g = tf.math.unsorted_segment_sum(
            prod_g,
            self.m_sum_ind,
            num_segments=self.nfunc,
            name=f"sum_cg_{self.name}",
        )
        prod = tf.transpose(prod_g, [1, 2, 0], name=f"trans_120{self.name}")

        return prod


@capture_init_args
class ProductFunction(TPEquivariantInstruction):
    input_tensor_spec = {
        constants.ATOMIC_MU_I: {"shape": [None], "dtype": "int"},
    }

    def __init__(
        self,
        left: TPEquivariantInstruction,
        right: TPEquivariantInstruction,
        name: str,
        lmax: int,
        Lmax: int,
        is_left_right_equal: bool = None,
        lmax_left: int = None,
        lmax_right: int = None,
        lmax_hist: int = None,
        lmax_hist_left: int = None,
        lmax_hist_right: int = None,
        history_drop_list: list = None,
        max_sum_l: int = None,
        keep_parity: list[list] = None,
        normalize: bool = False,
        **kwargs,
    ):
        super().__init__(name=name, lmax=Lmax)
        self.left = left
        self.right = right

        assert self.left.coupling_meta_data is not None
        assert self.right.coupling_meta_data is not None
        if is_left_right_equal is None:
            is_left_right_equal = left.name == right.name
            self.is_left_right_equal = is_left_right_equal
        else:
            self.is_left_right_equal = is_left_right_equal

        assert (
            self.left.n_out == self.right.n_out
        ), "n_out of the product is None but shapes of left and right do not match"
        self.n_out = self.left.n_out
        self.do_reshape = False

        if keep_parity is None:
            plist = []
            for l in range(Lmax + 1):
                p = 1 if l % 2 == 0 else -1
                plist.append([l, p])
        else:
            plist = keep_parity
        self.plist = plist
        self.normalize = normalize
        self.max_sum_l = max_sum_l

        # TODO: This must not be done
        self.lmax = lmax

        self.lmax_A = lmax_left
        self.lmax_B = lmax_right
        self.lmax_hist = lmax_hist
        self.lmax_hist_A = lmax_hist_left
        self.lmax_hist_B = lmax_hist_right
        self.Lmax = Lmax
        self.coupling_origin = [self.left.name, self.right.name]
        # to be able to re-write it later, otherwise checkpoint will fail
        # if history_drop_list == "DEFAULT":
        #     history_drop_list = EXCLUSION_HIST_LIST
        # self.history_drop_list = NoDependency(history_drop_list or [])
        self.history_drop_list = history_drop_list
        self.init_coupling()

    def init_coupling(self):
        self.coupling_meta_data = real_coupling_metainformation(
            A=self.left.coupling_meta_data,
            B=self.right.coupling_meta_data,
            lmax=self.lmax,
            lmax_A=self.lmax_A,
            lmax_B=self.lmax_B,
            lmax_hist=self.lmax_hist,
            lmax_hist_A=self.lmax_hist_A,
            lmax_hist_B=self.lmax_hist_B,
            Lmax=self.Lmax,
            is_A_B_equal=self.is_left_right_equal,
            history_drop_list=self.history_drop_list,
            max_sum_l=self.max_sum_l,
            keep_parity=self.plist,
            normalize=self.normalize,
        )
        assert len(
            self.coupling_meta_data
        ), f"No coupling channels found between {self.left.name} and {self.right.name}"

        self.left_ind = tf.constant(
            np.concatenate(self.coupling_meta_data["left_inds"]), dtype=tf.int32
        )
        self.right_ind = tf.constant(
            np.concatenate(self.coupling_meta_data["right_inds"]), dtype=tf.int32
        )

        cgs = self.coupling_meta_data["cg_list"]
        sum_ind = []
        for i, cg in enumerate(cgs):
            sum_ind.append([i] * len(cg))
        sum_ind = np.concatenate(sum_ind)
        self.m_sum_ind = tf.constant(sum_ind, dtype=tf.int32)

        self.cg = np.concatenate(cgs).reshape(1, 1, -1)

        nfunc = np.max(sum_ind) + 1
        self.nfunc = tf.constant(nfunc, dtype=tf.int32)

    @tf.Module.with_name_scope
    def build(self, float_dtype):
        if not self.is_built:
            self.cg = tf.constant(self.cg, dtype=float_dtype)
            self.is_built = True

    def frwrd(self, input_data, training=False):
        left = input_data[self.left.name]
        right = input_data[self.right.name]

        lft = tf.gather(left, self.left_ind, axis=2)
        rght = tf.gather(right, self.right_ind, axis=2)

        prod = lft * rght

        prod = prod * self.cg
        prod = tf.transpose(prod, [2, 0, 1])
        prod = tf.math.unsorted_segment_sum(
            prod, self.m_sum_ind, num_segments=self.nfunc
        )

        return tf.transpose(prod, [1, 2, 0])

    def __repr__(self):
        return f"ComputeProductFunction(name={self.name}, origin={' x '.join(self.coupling_origin)}, l_max={self.lmax})"


@capture_init_args
class CropProductFunction(TPEquivariantInstruction):
    input_tensor_spec = {
        constants.ATOMIC_MU_I: {"shape": [None], "dtype": "int"},
    }

    def __init__(
        self,
        left: TPEquivariantInstruction,
        right: TPEquivariantInstruction,
        name: str,
        lmax: int,
        Lmax: int,
        n_crop: int = None,
        is_left_right_equal: bool = None,
        lmax_left: int = None,
        lmax_right: int = None,
        lmax_hist: int = None,
        lmax_hist_left: int = None,
        lmax_hist_right: int = None,
        history_drop_list: list = None,
        max_sum_l: int = None,
        keep_parity: list[list] = None,
        normalize: bool = True,
    ):
        super().__init__(name=name, lmax=Lmax)
        self.left = left
        self.right = right

        assert self.left.coupling_meta_data is not None
        assert self.right.coupling_meta_data is not None
        if is_left_right_equal is None:
            is_left_right_equal = left.name == right.name
            self.is_left_right_equal = is_left_right_equal
        else:
            self.is_left_right_equal = is_left_right_equal

        if n_crop is not None:
            assert n_crop <= self.left.n_out
            assert n_crop <= self.right.n_out
            self.n_out = n_crop
        else:
            assert self.left.n_out == self.right.n_out
            self.n_out = self.left.n_out

        if keep_parity is None:
            plist = []
            for l in range(Lmax + 1):
                p = 1 if l % 2 == 0 else -1
                plist.append([l, p])
        else:
            plist = keep_parity
        self.plist = plist
        self.normalize = normalize
        self.max_sum_l = max_sum_l
        self.lmax = lmax
        self.lmax_A = lmax_left
        self.lmax_B = lmax_right
        self.lmax_hist = lmax_hist
        self.lmax_hist_A = lmax_hist_left
        self.lmax_hist_B = lmax_hist_right
        self.Lmax = Lmax
        self.coupling_origin = [self.left.name, self.right.name]
        # to be able to re-write it later, otherwise checkpoint will fail
        # if history_drop_list == "DEFAULT":
        #     history_drop_list = EXCLUSION_HIST_LIST
        # self.history_drop_list = NoDependency(history_drop_list or [])
        self.history_drop_list = history_drop_list
        self.init_coupling()

    def init_coupling(self):
        self.coupling_meta_data = real_coupling_metainformation(
            A=self.left.coupling_meta_data,
            B=self.right.coupling_meta_data,
            lmax=self.lmax,
            lmax_A=self.lmax_A,
            lmax_B=self.lmax_B,
            lmax_hist=self.lmax_hist,
            lmax_hist_A=self.lmax_hist_A,
            lmax_hist_B=self.lmax_hist_B,
            Lmax=self.Lmax,
            is_A_B_equal=self.is_left_right_equal,
            history_drop_list=self.history_drop_list,
            max_sum_l=self.max_sum_l,
            keep_parity=self.plist,
            normalize=self.normalize,
        )
        assert len(
            self.coupling_meta_data
        ), f"No coupling channels found between {self.left.name} and {self.right.name}"

        self.left_ind = tf.constant(
            np.concatenate(self.coupling_meta_data["left_inds"]), dtype=tf.int32
        )
        self.right_ind = tf.constant(
            np.concatenate(self.coupling_meta_data["right_inds"]), dtype=tf.int32
        )

        cgs = self.coupling_meta_data["cg_list"]
        sum_ind = []
        for i, cg in enumerate(cgs):
            sum_ind.append([i] * len(cg))
        sum_ind = np.concatenate(sum_ind)
        self.m_sum_ind = tf.constant(sum_ind, dtype=tf.int32)

        self.cg = np.concatenate(cgs).reshape(1, 1, -1)

        nfunc = np.max(sum_ind) + 1
        self.nfunc = tf.constant(nfunc, dtype=tf.int32)

    @tf.Module.with_name_scope
    def build(self, float_dtype):
        if not self.is_built:
            self.cg = tf.constant(self.cg, dtype=float_dtype)
            self.is_built = True

    def frwrd(self, input_data, training=False):
        left = input_data[self.left.name]
        left = left[:, : self.n_out, :]

        if self.is_left_right_equal:
            right = left
        else:
            right = input_data[self.right.name]
            right = right[:, : self.n_out, :]

        lft = tf.gather(left, self.left_ind, axis=2)
        rght = tf.gather(right, self.right_ind, axis=2)
        prod = lft * rght

        prod = prod * self.cg
        prod = tf.transpose(prod, [2, 0, 1])
        prod = tf.math.unsorted_segment_sum(
            prod, self.m_sum_ind, num_segments=self.nfunc
        )

        return tf.transpose(prod, [1, 2, 0])


@capture_init_args
class FunctionReduce(TPEquivariantInstruction, ElementsReduceInstructionMixin):
    input_tensor_spec = {
        constants.N_ATOMS_BATCH_TOTAL: {"shape": [], "dtype": "int"},
        constants.ATOMIC_MU_I: {"shape": [None], "dtype": "int"},
    }

    def __init__(
        self,
        instructions: list[TPEquivariantInstruction],
        name: str,
        ls_max: list[int] | int,
        n_out: int,
        allowed_l_p: list[list],
        is_central_atom_type_dependent: bool = False,
        number_of_atom_types: int = None,
        init_vars: Literal["random", "zeros"] = "random",
        init_target_value: Literal["zeros", "ones"] = "zeros",
        simplify: bool = False,
        **kwargs,
    ):
        super().__init__(name=name, lmax=np.max(ls_max))
        self.instructions = instructions
        if isinstance(ls_max, int):
            ls_max = [ls_max] * len(instructions)
        self.ls_max = ls_max
        self.n_out = n_out
        self.allowed_l_p = [list(lp) for lp in allowed_l_p]
        self.is_central_atom_type_dependent = is_central_atom_type_dependent
        self.number_of_atom_types = number_of_atom_types

        self.n_instr = len(self.instructions)
        assert init_vars in [
            "random",
            "zeros",
        ], f'Unknown variable initialization "{init_vars}"'
        self.init_vars = init_vars
        assert init_target_value in [
            "zeros",
            "ones",
        ], f'Unknown target initialization "{init_target_value}"'
        self.init_collection = init_target_value

        instr_names = [instr.name for instr in self.instructions]
        assert len(instr_names) == len(set(instr_names)), "duplicate instruction names"
        assert len(instr_names) == len(
            self.ls_max
        ), f"provide lmax to collect for every instruction, error in {self.__class__.__name__}_{self.name}"

        collector_data = []
        for p in [-1, 1]:
            for l in range(self.lmax + 1):
                if [l, p] in self.allowed_l_p:
                    for m in range(-l, l + 1):
                        lbl = 0 if p > 0 else 1
                        # TODO: rethink how to define history here. Then, possibly move to the base class method
                        collector_data.append([l, m, f"", p, l])
        cdf = pd.DataFrame(
            collector_data, columns=["l", "m", "hist", "parity", "sum_of_ls"]
        )
        self.coupling_meta_data = cdf.sort_values(
            ["l", "parity", "hist", "m"]
        ).reset_index(drop=True)

        # TODO: Special case when only one instruction to collect from
        self.simplify = simplify
        if self.simplify:
            self.simplify_collected_tensors()
            self.drop_unused()

        self.collector = {}
        for instr, instr_lmax in zip(self.instructions, self.ls_max):
            instruction_collection = instr.collect_functions(
                max_l=instr_lmax, l_p_list=self.allowed_l_p
            )
            instruction_collection["total_sum_ind"] = []
            for index, row in instruction_collection["collect_meta_df"].iterrows():
                for idx, rw in self.coupling_meta_data.iterrows():
                    if (
                        (row["l"] == rw["l"])
                        & (row["m"] == rw["m"])
                        & (row["parity"] == rw["parity"])
                    ):
                        instruction_collection["total_sum_ind"].append(idx)
            instruction_collection["total_sum_ind"] = tf.constant(
                np.array(instruction_collection["total_sum_ind"]).reshape(-1, 1),
                dtype=tf.int32,
            )
            instruction_collection["n_out"] = instr.n_out
            self.collector[instr.name] = instruction_collection

    def simplify_collected_tensors(self):
        collector_dict = {}

        A_ins_dict = {ins.name: ins for ins in self.instructions}
        for instr, instr_lmax in zip(self.instructions, self.ls_max):
            instruction_collection = instr.collect_functions(
                max_l=instr_lmax, l_p_list=self.allowed_l_p
            )
            col_index = instruction_collection["func_collect_ind"]
            collector_dict[instr.name] = instruction_collection
            for ind in col_index:
                get_symbol(instr.name, ind, A_ins_dict)

    def drop_unused(self):
        ZERO = Polynomial([Monomial(0)])
        incl_hist_list = []
        excl_hist_list = []
        for A_ins in self.instructions:
            cmd = A_ins.coupling_meta_data
            if "cg_list" in cmd:
                cmd["n_op"] = cmd["cg_list"].map(len)
                cmd_clean = cmd[cmd["symbol"] != ZERO]
                cmd_clean["symbol"] = cmd_clean["symbol"].map(normalize_poly)

                cmd_clean = (
                    cmd_clean.dropna(subset=["symbol"])
                    .sort_values("n_op")
                    .drop_duplicates("symbol", keep="first")
                )
                cmd_clean = cmd_clean.sort_index()
                incl_hist_list += [h for h in cmd_clean["hist"]]
                excl_hist_list += [
                    h for h in cmd["hist"] if h not in cmd_clean["hist"].values
                ]
        incl_hist_list = sorted(set(incl_hist_list))
        excl_hist_list = sorted(set(excl_hist_list))
        excl_hist_list = [e for e in excl_hist_list if e not in incl_hist_list]
        logging.info(f"Following exclusion hist list was discovered: {excl_hist_list}")

        # re-initialize collectable instructions with new drop_list
        for inst in self.instructions:
            # duck typing
            if hasattr(inst, "history_drop_list") and hasattr(inst, "init_coupling"):
                inst.history_drop_list = list(inst.history_drop_list) + excl_hist_list
                inst.init_coupling()

    @tf.Module.with_name_scope
    def build(self, float_dtype):
        if not self.is_built:
            for k, v in self.collector.items():
                w_shape = v["w_shape"]
                n_in = v["n_out"]
                if self.is_central_atom_type_dependent:
                    c_shape = [self.number_of_atom_types, self.n_out, n_in, w_shape]
                else:
                    c_shape = [self.n_out, n_in, w_shape]

                if self.init_vars == "random":
                    limit = (2 / (n_in * w_shape)) ** 0.5
                    setattr(
                        self,
                        f"reducing_{k}",
                        tf.Variable(
                            tf.random.normal(c_shape, stddev=limit, dtype=float_dtype),
                            name=f"reducing_{k}",
                        ),
                    )
                    setattr(
                        self,
                        f"norm_{k}",
                        tf.constant(1.0, dtype=float_dtype),
                    )
                elif self.init_vars == "zeros":
                    coeff = np.zeros(c_shape)
                    setattr(
                        self,
                        f"reducing_{k}",
                        tf.Variable(
                            coeff,
                            dtype=float_dtype,
                            name=f"reducing_{k}",
                        ),
                    )
                    setattr(
                        self,
                        f"norm_{k}",
                        tf.constant(1.0, dtype=float_dtype),
                    )
                else:
                    raise NotImplementedError(
                        f"FunctionCollector.init = {self.init_vars} is unknown"
                    )

            self.float_dtype = float_dtype
            self.n_instr = tf.constant(1 / self.n_instr, dtype=float_dtype)
            self.is_built = True

    def compute_l2_regularization_loss(self):
        total_l2_regularization = 0.0
        for var in self.trainable_variables:
            total_l2_regularization += tf.reduce_sum(tf.square(var))
        return total_l2_regularization

    def frwrd(self, input_data, training=False):
        if self.init_collection == "zeros":
            init_func = tf.zeros
        else:
            init_func = tf.ones
        collection = init_func(
            [
                self.coupling_meta_data.shape[0],
                input_data[constants.N_ATOMS_BATCH_TOTAL],
                self.n_out,
            ],
            dtype=self.float_dtype,
        )
        for instr in self.instructions:
            instruction_collection = self.collector[instr.name]
            A_r = tf.gather(
                input_data[instr.name],
                instruction_collection["func_collect_ind"],
                axis=2,
            )
            if self.is_central_atom_type_dependent:
                w = getattr(self, f"reducing_{instr.name}")
                eq = "aknw,anw->wak"
            else:
                w = getattr(self, f"reducing_{instr.name}")
                eq = "knw,anw->wak"
            w = tf.gather(w, instruction_collection["w_l_tile"], axis=-1)
            # For performance
            if self.is_central_atom_type_dependent:
                w = tf.gather(w, input_data[constants.ATOMIC_MU_I], axis=0)

            norm = getattr(self, f"norm_{instr.name}")
            pr = tf.einsum(eq, w, A_r, name=f"ein_{instr.name}") * norm

            collection = tf.tensor_scatter_nd_add(
                collection, instruction_collection["total_sum_ind"], pr
            )

        collection = tf.transpose(collection, [1, 2, 0])

        return collection  # * self.n_instr

    def prepare_variables_for_selected_elements(self, index_to_select):
        if self.is_central_atom_type_dependent:
            reducing_tensor_names = [s for s in dir(self) if s.startswith("reducing_")]
            new_tensors = {}
            for tn in reducing_tensor_names:
                var = getattr(self, tn)
                # print(tn, var.shape)
                new_tensors[tn] = tf.Variable(tf.gather(var, index_to_select, axis=0))

            return new_tensors

    def upd_init_args_new_elements(self, new_element_map):
        self._init_args["number_of_atom_types"] = len(new_element_map)


@capture_init_args
class FunctionReduceN(
    TPEquivariantInstruction, ElementsReduceInstructionMixin, LORAInstructionMixin
):
    input_tensor_spec = {
        constants.N_ATOMS_BATCH_TOTAL: {"shape": [], "dtype": "int"},
        constants.ATOMIC_MU_I: {"shape": [None], "dtype": "int"},
    }

    def __init__(
        self,
        instructions: list[TPEquivariantInstruction],
        name: str,
        ls_max: list[int] | int,
        n_out: int,
        allowed_l_p: list[list],
        out_norm: bool = False,
        is_central_atom_type_dependent: bool = False,
        number_of_atom_types: int = None,
        init_vars: str = "random",
        normalize: bool = True,
        init_target_value: str = "zeros",
        simplify: bool = False,
        scale=1.0,
        lora_config: dict[str, Any] = None,
        **kwargs,
    ):
        super().__init__(name=name, lmax=np.max(ls_max))
        LORAInstructionMixin.__init__(self, lora_config)  # explicitly call
        self.instructions = instructions
        if isinstance(ls_max, int):
            ls_max = [ls_max] * len(instructions)
        self.ls_max = ls_max
        self.n_out = n_out
        self.out_norm = out_norm
        # enforce conversion to list of lists
        self.allowed_l_p = [list(lp) for lp in allowed_l_p]
        self.is_central_atom_type_dependent = is_central_atom_type_dependent
        self.number_of_atom_types = number_of_atom_types
        self.n_instr = len(self.instructions)
        assert init_vars in [
            "random",  # normal
            "uniform",
            "zeros",
        ], f'Unknown variable initialization "{init_vars}"'
        self.init_vars = init_vars
        assert init_target_value in [
            "zeros",
            "ones",
        ], f'Unknown target initialization "{init_target_value}"'
        self.normalize = normalize
        self.init_collection = init_target_value
        self.scale = scale
        if self.is_central_atom_type_dependent:
            assert self.number_of_atom_types is not None

        instr_names = [instr.name for instr in self.instructions]
        assert len(instr_names) == len(set(instr_names)), "duplicate instruction names"
        assert len(instr_names) == len(
            self.ls_max
        ), f"provide lmax to collect for every instruction, error in {self.__class__.__name__}_{self.name}"

        collector_data = []
        for p in [-1, 1]:
            for l in range(self.lmax + 1):
                if [l, p] in self.allowed_l_p:
                    for m in range(-l, l + 1):
                        lbl = 0 if p > 0 else 1
                        # TODO:  possibly move to the base class method
                        collector_data.append([l, m, f"", p, l])
                        # collector_data.append([l, m, f"({lbl},0)", p, l])
        cdf = pd.DataFrame(
            collector_data, columns=["l", "m", "hist", "parity", "sum_of_ls"]
        )
        self.coupling_meta_data = cdf.sort_values(
            ["l", "parity", "hist", "m"]
        ).reset_index(drop=True)
        norms = np.zeros((self.coupling_meta_data.shape[0]))

        # TODO: Special case when only one instruction to collect from
        self.simplify = simplify
        if self.simplify:
            self.simplify_collected_tensors()
            self.drop_unused()

        # TODO: Special case when only one instruction to collect from
        self.collector = {}
        for instr, instr_lmax in zip(self.instructions, self.ls_max):
            instruction_collection = instr.collect_functions(
                max_l=instr_lmax, l_p_list=self.allowed_l_p
            )
            instruction_collection["total_sum_ind"] = []
            for index, row in instruction_collection["collect_meta_df"].iterrows():
                for idx, rw in self.coupling_meta_data.iterrows():
                    if (
                        (row["l"] == rw["l"])
                        & (row["m"] == rw["m"])
                        & (row["parity"] == rw["parity"])
                    ):
                        instruction_collection["total_sum_ind"].append(idx)
                        norms[idx] += 1
            instruction_collection["total_sum_ind"] = tf.constant(
                np.array(instruction_collection["total_sum_ind"]).reshape(-1, 1),
                dtype=tf.int32,
            )
            instruction_collection["n_out"] = instr.n_out
            self.collector[instr.name] = instruction_collection
        norms[norms == 0] = 1
        self.norm_map = 1 / norms**0.5

    def simplify_collected_tensors(self):
        collector_dict = {}

        A_ins_dict = {ins.name: ins for ins in self.instructions}
        for instr, instr_lmax in zip(self.instructions, self.ls_max):
            instruction_collection = instr.collect_functions(
                max_l=instr_lmax, l_p_list=self.allowed_l_p
            )
            col_index = instruction_collection["func_collect_ind"]
            collector_dict[instr.name] = instruction_collection
            for ind in col_index:
                get_symbol(instr.name, ind, A_ins_dict)

    def drop_unused(self):
        ZERO = Polynomial([Monomial(0)])
        incl_hist_list = []
        excl_hist_list = []
        for A_ins in self.instructions:
            cmd = A_ins.coupling_meta_data
            if "cg_list" in cmd:
                cmd["n_op"] = cmd["cg_list"].map(len)
                cmd_clean = cmd[cmd["symbol"] != ZERO]
                cmd_clean["symbol"] = cmd_clean["symbol"].map(normalize_poly)

                cmd_clean = (
                    cmd_clean.dropna(subset=["symbol"])
                    .sort_values("n_op")
                    .drop_duplicates("symbol", keep="first")
                )
                cmd_clean = cmd_clean.sort_index()
                incl_hist_list += [h for h in cmd_clean["hist"]]
                excl_hist_list += [
                    h for h in cmd["hist"] if h not in cmd_clean["hist"].values
                ]
        incl_hist_list = sorted(set(incl_hist_list))
        excl_hist_list = sorted(set(excl_hist_list))
        excl_hist_list = [e for e in excl_hist_list if e not in incl_hist_list]
        logging.info(f"Following exclusion hist list was discovered: {excl_hist_list}")

        # re-initialize collectable instructions with new drop_list
        for inst in self.instructions:
            # duck typing
            if hasattr(inst, "history_drop_list") and hasattr(inst, "init_coupling"):
                inst.history_drop_list = list(inst.history_drop_list) + excl_hist_list
                inst.init_coupling()

    @tf.Module.with_name_scope
    def build(self, float_dtype):
        if not self.is_built:
            for k, v in self.collector.items():
                w_shape = v["w_shape"]
                n_in = v["n_out"]
                if self.is_central_atom_type_dependent:
                    c_shape = [self.number_of_atom_types, self.n_out, n_in, w_shape]
                else:
                    c_shape = [self.n_out, n_in, w_shape]

                if self.normalize:
                    init_value = self.scale
                    norm_vlue = self.scale / n_in**0.5
                else:
                    init_value = 1.0 / n_in**0.5
                    norm_vlue = 1.0

                if self.init_vars == "random":
                    setattr(
                        self,
                        f"reducing_{k}",
                        tf.Variable(
                            tf.random.normal(
                                c_shape, stddev=init_value, dtype=float_dtype
                            ),
                            name=f"reducing_{k}",
                        ),
                    )
                elif self.init_vars == "uniform":
                    setattr(
                        self,
                        f"reducing_{k}",
                        tf.Variable(
                            tf.random.uniform(
                                minval=-init_value,
                                maxval=init_value,
                                shape=c_shape,
                                dtype=float_dtype,
                            ),
                            name=f"reducing_{k}",
                        ),
                    )
                elif self.init_vars == "zeros":
                    coeff = np.zeros(c_shape)
                    setattr(
                        self,
                        f"reducing_{k}",
                        tf.Variable(
                            coeff,
                            dtype=float_dtype,
                            name=f"reducing_{k}",
                        ),
                    )
                else:
                    raise NotImplementedError(
                        f"FunctionCollector.init = {self.init_vars} is unknown"
                    )
                setattr(
                    self,
                    f"norm_{k}",
                    tf.constant(norm_vlue, dtype=float_dtype),
                )

            self.float_dtype = float_dtype
            if self.out_norm:
                self.norm_map = tf.reshape(
                    tf.constant(self.norm_map, dtype=float_dtype), [-1, 1, 1]
                )
            else:
                self.norm_map = tf.constant(1, dtype=float_dtype)

            if self.lora_config:
                self.enable_lora_adaptation(self.lora_config)

            self.is_built = True

    @tf.Module.with_name_scope
    def enable_lora_adaptation(self, lora_config):
        super().enable_lora_adaptation(lora_config)
        for ins_name in self.collector:
            w = getattr(self, f"reducing_{ins_name}")
            lora_tensors = initialize_lora_tensors(
                w, lora_config=lora_config, name=f"reducing_{ins_name}"
            )
            setattr(self, f"reducing_{ins_name}_lora_tensors", lora_tensors)

    def finalize_lora_update(self):
        # common part

        for ins_name in self.collector:
            w = getattr(self, f"reducing_{ins_name}")
            lora_tensors = getattr(self, f"reducing_{ins_name}_lora_tensors")

            apply_lora_update(w, *lora_tensors, lora_config=self.lora_config)
            delattr(self, f"reducing_{ins_name}_lora_tensors")

        super().finalize_lora_update()

    def compute_l2_regularization_loss(self):
        total_l2_regularization = 0.0
        for var in self.trainable_variables:
            total_l2_regularization += tf.reduce_sum(tf.square(var))
        return total_l2_regularization

    def frwrd(self, input_data, training=False):
        init_func = tf.zeros if self.init_collection == "zeros" else tf.ones
        collection = init_func(
            [
                self.coupling_meta_data.shape[0],
                input_data[constants.N_ATOMS_BATCH_TOTAL],
                self.n_out,
            ],
            dtype=self.float_dtype,
        )
        for instr in self.instructions:
            instruction_collection = self.collector[instr.name]
            A_r = tf.gather(
                input_data[instr.name],
                instruction_collection["func_collect_ind"],
                axis=2,
            )
            w = getattr(self, f"reducing_{instr.name}")
            # lora
            if self.lora:
                lora_tensors = getattr(self, f"reducing_{instr.name}_lora_tensors")
                w = w + lora_reconstruction(*lora_tensors, lora_config=self.lora_config)
            if self.is_central_atom_type_dependent:
                eq = "aknw,anw->wak"
            else:
                eq = "knw,anw->wak"
            w = tf.gather(w, instruction_collection["w_l_tile"], axis=-1)
            # For performance
            if self.is_central_atom_type_dependent:
                w = tf.gather(w, input_data[constants.ATOMIC_MU_I], axis=0)

            norm = getattr(self, f"norm_{instr.name}")
            pr = tf.einsum(eq, w, A_r, name=f"ein_{instr.name}") * norm

            collection = tf.tensor_scatter_nd_add(
                collection, instruction_collection["total_sum_ind"], pr
            )
        collection *= self.norm_map
        collection = tf.transpose(collection, [1, 2, 0])

        return collection

    def prepare_variables_for_selected_elements(self, index_to_select):
        if self.is_central_atom_type_dependent:
            reducing_tensor_names = [s for s in dir(self) if s.startswith("reducing_")]
            new_tensors = {}
            for tn in reducing_tensor_names:
                var = getattr(self, tn)
                # print(tn, var.shape)
                new_tensors[tn] = tf.Variable(tf.gather(var, index_to_select, axis=0))

            return new_tensors

    def upd_init_args_new_elements(self, new_element_map):
        self._init_args["number_of_atom_types"] = len(new_element_map)


@capture_init_args
class CollectInvarBasis(TPEquivariantInstruction, ElementsReduceInstructionMixin):
    input_tensor_spec = {
        constants.N_ATOMS_BATCH_TOTAL: {"shape": [], "dtype": "int"},
        constants.ATOMIC_MU_I: {"shape": [None], "dtype": "int"},
    }

    def __init__(
        self,
        instructions: list[TPEquivariantInstruction],
        name: str,
        ls_max: list[int] | int,
        # n_out: int,
        # allowed_l_p: list[list],
        # out_norm: bool = False,
        # is_central_atom_type_dependent: bool = False,
        # number_of_atom_types: int = None,
        # init_vars: Literal["random", "zeros"] = "random",
        # scale=1.0,
    ):
        super().__init__(name=name, lmax=np.max(ls_max))
        self.instructions = instructions

        if isinstance(ls_max, int):
            ls_max = [ls_max] * len(instructions)
        assert np.max(ls_max) == 0

        self.ls_max = ls_max
        # self.n_out = n_out
        # self.out_norm = out_norm
        # enforce conversion to list of lists
        allowed_l_p = [[0, 1]]
        self.allowed_l_p = [list(lp) for lp in allowed_l_p]
        # self.is_central_atom_type_dependent = is_central_atom_type_dependent
        # self.number_of_atom_types = number_of_atom_types
        self.n_instr = len(self.instructions)

        # assert init_vars in [
        #     "random",
        #     "zeros",
        # ], f'Unknown variable initialization "{init_vars}"'
        # self.init_vars = init_vars
        #
        # self.scale = scale
        # if self.is_central_atom_type_dependent:
        #     assert self.number_of_atom_types is not None

        instr_names = [instr.name for instr in self.instructions]
        assert len(instr_names) == len(set(instr_names)), "duplicate instruction names"
        assert len(instr_names) == len(
            self.ls_max
        ), f"provide lmax to collect for every instruction, error in {self.__class__.__name__}_{self.name}"

        collector_data = []
        for p in [-1, 1]:
            for l in range(self.lmax + 1):
                if [l, p] in self.allowed_l_p:
                    for m in range(-l, l + 1):
                        lbl = 0 if p > 0 else 1
                        collector_data.append([l, m, f"", p, l])
        self.coupling_meta_data = pd.DataFrame(
            collector_data, columns=["l", "m", "hist", "parity", "sum_of_ls"]
        )

        self.collector = {}
        for instr, instr_lmax in zip(self.instructions, self.ls_max):
            instruction_collection = instr.collect_functions(
                max_l=instr_lmax, l_p_list=self.allowed_l_p
            )
            instruction_collection["total_sum_ind"] = []
            for index, row in instruction_collection["collect_meta_df"].iterrows():
                for idx, rw in self.coupling_meta_data.iterrows():
                    if (
                        (row["l"] == rw["l"])
                        & (row["m"] == rw["m"])
                        & (row["parity"] == rw["parity"])
                    ):
                        instruction_collection["total_sum_ind"].append(idx)

            instruction_collection["total_sum_ind"] = tf.constant(
                np.array(instruction_collection["total_sum_ind"]).reshape(-1, 1),
                dtype=tf.int32,
            )
            instruction_collection["n_out"] = instr.n_out
            self.collector[instr.name] = instruction_collection

    @tf.Module.with_name_scope
    def build(self, float_dtype):
        if not self.is_built:
            # size = 0
            # for k, v in self.collector.items():
            #     w_shape = v["w_shape"]
            #     n_in = v["n_out"]
            #     size += w_shape * n_in
            # if self.is_central_atom_type_dependent:
            #     c_shape = [self.number_of_atom_types, self.n_out, size]
            # else:
            #     c_shape = [self.n_out, size]
            #
            # name_v = "full"
            # if self.init_vars == "random":
            #     limit = 1
            #     setattr(
            #         self,
            #         f"reducing_{name_v}",
            #         tf.Variable(
            #             tf.random.normal(
            #                 c_shape,
            #                 stddev=self.scale * limit,
            #                 dtype=float_dtype,
            #             ),
            #             name=f"reducing_{name_v}",
            #         ),
            #     )
            # elif self.init_vars == "zeros":
            #     coeff = np.zeros(c_shape)
            #     setattr(
            #         self,
            #         f"reducing_{name_v}",
            #         tf.Variable(
            #             coeff,
            #             dtype=float_dtype,
            #             name=f"reducing_{name_v}",
            #         ),
            #     )
            # else:
            #     raise NotImplementedError(
            #         f"FunctionCollector.init = {self.init_vars} is unknown"
            #     )
            #
            # self.norm = tf.constant(1 / size, dtype=float_dtype)
            self.float_dtype = float_dtype
            self.is_built = True

    def frwrd(self, input_data, training=False):
        collection = []
        for instr in self.instructions:
            instruction_collection = self.collector[instr.name]
            A = tf.gather(
                input_data[instr.name],
                instruction_collection["func_collect_ind"],
                axis=2,
            )
            shp = tf.shape(A)
            A = tf.reshape(A, [-1, shp[1] * shp[2]])
            # rms = tf.math.rsqrt(tf.reduce_mean(A**2, axis=-1, keepdims=True) + 1e-16)
            # collection += [A * rms]
            collection += [A]
        basis = tf.concat(collection, axis=1)

        # rms = tf.math.rsqrt(tf.reduce_mean(basis**2, axis=-1, keepdims=True) + 1e-16)
        # basis *= rms

        # w = getattr(self, f"reducing_full")
        # if self.is_central_atom_type_dependent:
        #     w = tf.gather(w, input_data[constants.ATOMIC_MU_I], axis=0)
        #     eq = "akn,an->ak"
        # else:
        #     eq = "kn,an->ak"
        # pr = tf.einsum(eq, w, basis, name=f"ein_basis") * self.norm
        #
        # return pr[:, :, tf.newaxis]
        return basis

    def prepare_variables_for_selected_elements(self, index_to_select):
        if self.is_central_atom_type_dependent:
            raise NotImplementedError()


@capture_init_args
class FCRight2Left(
    TPEquivariantInstruction, ElementsReduceInstructionMixin, LORAInstructionMixin
):
    input_tensor_spec = {
        constants.N_ATOMS_BATCH_TOTAL: {"shape": [], "dtype": "int"},
        constants.ATOMIC_MU_I: {"shape": [None], "dtype": "int"},
    }

    def __init__(
        self,
        left: TPEquivariantInstruction,
        right: TPEquivariantInstruction,
        name: str,
        n_out: int,
        left_coefs: bool = True,
        is_central_atom_type_dependent: list[bool] | bool = None,
        number_of_atom_types: int = None,
        init_vars: str = "random",
        normalize: bool = True,
        norm_out: bool = False,
        lora_config: dict[str, Any] = None,
    ):
        super().__init__(name=name, lmax=left.lmax)
        LORAInstructionMixin.__init__(self, lora_config)  # explicitly call

        self.left = left
        self.right = right
        self.left_coefs = left_coefs

        # if not self.left_coefs:
        # assert left.n_out == n_out
        self.n_out = n_out

        if is_central_atom_type_dependent is None:
            self.is_central_atom_type_dependent = [False, False]
        else:
            if isinstance(is_central_atom_type_dependent, list):
                self.is_central_atom_type_dependent = is_central_atom_type_dependent
            elif isinstance(is_central_atom_type_dependent, bool):
                self.is_central_atom_type_dependent = [
                    is_central_atom_type_dependent,
                    is_central_atom_type_dependent,
                ]
            else:
                raise ValueError(
                    f"Unexpected type for :is_central_atom_type_dependent:"
                    f" {type(is_central_atom_type_dependent)} in"
                    f" {self.__class__.__name__}_{self.name}"
                )
        self.number_of_atom_types = number_of_atom_types
        self.norm_out = norm_out
        assert init_vars in [
            "random",  # normal
            "uniform",
            "zeros",
        ], f'Unknown variable initialization "{init_vars}"'
        self.init_vars = init_vars
        self.normalize = normalize

        # if self.is_central_atom_type_dependent:
        #     assert self.number_of_atom_types is not None
        #     self.eq = "aknw,anw->wak"
        # else:
        #     self.eq = "knw,anw->wak"
        self.eq_elem = "aknw,anw->wak"
        self.eq = "knw,anw->wak"

        self.coupling_meta_data = self.left.coupling_meta_data.copy()

        # TILE LEFT
        collect_ind = self.coupling_meta_data.groupby(["l", "parity", "hist"]).indices
        w_shape = len(collect_ind)
        w_l_tile = np.zeros(len(self.coupling_meta_data))
        w_l_tile[np.concatenate([v for k, v in collect_ind.items()])] = np.concatenate(
            [[i] * len(v) for i, (k, v) in enumerate(collect_ind.items())]
        )
        self.w_tile_left = w_l_tile
        self.w_shape_left = w_shape

        d_l = self.coupling_meta_data.groupby(["l", "parity", "hist"]).indices
        d_r = self.right.coupling_meta_data.groupby(["l", "parity", "hist"]).indices
        collect_to = []
        collect_from = []
        w_tile_right = []
        count = 0
        norms = np.ones((len(d_l.items())))
        for ind, ((l_left, p_left, h_left), index_left) in enumerate(d_l.items()):
            for (l_right, p_right, h_right), index_right in d_r.items():
                if l_right == l_left and p_right == p_left:
                    collect_to += [index_left]
                    collect_from += [index_right]
                    w_tile_right += [count] * len(index_right)
                    count += 1
                    norms[ind] += 1
        # norms[norms == 0] = 1
        nnorms = np.take(norms, w_l_tile.astype(np.int32))
        self.norm_map = 1 / nnorms**0.5

        self.collect_to = tf.constant(np.concatenate(collect_to), dtype=tf.int32)
        self.collect_from = tf.constant(np.concatenate(collect_from), dtype=tf.int32)
        self.w_tile_right = tf.constant(w_tile_right, dtype=tf.int32)
        self.w_shape_right = np.max(w_tile_right) + 1

    @tf.Module.with_name_scope
    def build(self, float_dtype):
        if not self.is_built:
            if self.left_coefs:
                self.w_tile_left = tf.constant(self.w_tile_left, dtype=tf.int32)
                if self.is_central_atom_type_dependent[0]:
                    c_shape_left = [
                        self.number_of_atom_types,
                        self.n_out,
                        self.left.n_out,
                        self.w_shape_left,
                    ]
                else:
                    c_shape_left = [
                        self.n_out,
                        self.left.n_out,
                        self.w_shape_left,
                    ]
            if self.is_central_atom_type_dependent[1]:
                c_shape_right = [
                    self.number_of_atom_types,
                    self.n_out,
                    self.right.n_out,
                    self.w_shape_right,
                ]
            else:
                c_shape_right = [self.n_out, self.right.n_out, self.w_shape_right]

            if self.normalize:
                init_value = 1.0
                norm_value = 1 / self.left.n_out**0.5
            else:
                init_value = 1 / self.left.n_out**0.5
                norm_value = 1.0

            if self.left_coefs:
                if self.init_vars == "random":
                    self.w_left = tf.Variable(
                        tf.random.normal(
                            c_shape_left, stddev=init_value, dtype=float_dtype
                        ),
                        name=f"w_left_FC",
                    )
                elif self.init_vars == "uniform":
                    self.w_left = tf.Variable(
                        tf.random.uniform(
                            minval=-init_value,
                            maxval=init_value,
                            shape=c_shape_left,
                            dtype=float_dtype,
                        ),
                        name=f"w_left_FC",
                    )
                elif self.init_vars == "zeros":
                    self.w_left = tf.Variable(
                        tf.zeros(shape=c_shape_left, dtype=float_dtype),
                        name=f"w_left_FC",
                    )
                else:
                    raise NotImplementedError(
                        f"FunctionCollector.init = {self.init_vars} is unknown"
                    )
                self.norm_left = tf.constant(norm_value, dtype=float_dtype)
            if self.normalize:
                init_value = 1.0
                norm_value = 1 / self.right.n_out**0.5
            else:
                init_value = 1 / self.right.n_out**0.5
                norm_value = 1.0

            if self.init_vars == "random":
                self.w_right = tf.Variable(
                    tf.random.normal(
                        c_shape_right, stddev=init_value, dtype=float_dtype
                    ),
                    name=f"w_right_FC",
                )
            elif self.init_vars == "uniform":
                self.w_right = tf.Variable(
                    tf.random.uniform(
                        minval=-init_value,
                        maxval=init_value,
                        shape=c_shape_right,
                        dtype=float_dtype,
                    ),
                    name=f"w_right_FC",
                )
            elif self.init_vars == "zeros":
                # TODO: split left right initializers
                self.w_right = tf.Variable(
                    tf.random.normal(
                        c_shape_right, stddev=init_value, dtype=float_dtype
                    ),
                    name=f"w_right_FC",
                )
            else:
                raise NotImplementedError(
                    f"FunctionCollector.init = {self.init_vars} is unknown"
                )

            self.norm_right = tf.constant(norm_value, dtype=float_dtype)

            if self.norm_out:
                self.norm_out_factor = tf.reshape(
                    tf.constant(self.norm_map, dtype=float_dtype), [-1, 1, 1]
                )

            if self.lora_config:
                self.enable_lora_adaptation(self.lora_config)

            self.is_built = True

    @tf.Module.with_name_scope
    def enable_lora_adaptation(self, lora_config):
        # 1. upd _init_args,
        # 2. add new trainable LORA weights,
        # 3. set main weights as non-trainable
        super().enable_lora_adaptation(lora_config)
        # LORA
        if self.left_coefs:
            self.w_left_lora_tensors = initialize_lora_tensors(
                self.w_left, lora_config, name="w_left"
            )

        # lora for self.w_right
        self.w_right_lora_tensors = initialize_lora_tensors(
            self.w_right, lora_config, name="w_right"
        )

    def finalize_lora_update(self):

        if self.left_coefs:
            apply_lora_update(
                self.w_left, *self.w_left_lora_tensors, lora_config=self.lora_config
            )
            del self.w_left_lora_tensors

        apply_lora_update(
            self.w_right, *self.w_right_lora_tensors, lora_config=self.lora_config
        )
        del self.w_right_lora_tensors
        # common part
        super().finalize_lora_update()

    def compute_l2_regularization_loss(self):
        total_l2_regularization = 0.0
        for var in self.trainable_variables:
            total_l2_regularization += tf.reduce_sum(tf.square(var))
        return total_l2_regularization

    def frwrd(self, input_data, training=False):
        left = input_data[self.left.name]
        if self.left_coefs:
            w_left = self.w_left
            # LORA
            if self.lora:
                w_left = w_left + lora_reconstruction(
                    *self.w_left_lora_tensors, lora_config=self.lora_config
                )
            w_left = tf.gather(w_left, self.w_tile_left, axis=-1)
            if self.is_central_atom_type_dependent[0]:
                w_left = tf.gather(w_left, input_data[constants.ATOMIC_MU_I], axis=0)
                left = (
                    tf.einsum(self.eq_elem, w_left, left, name=f"ein_left")
                    * self.norm_left
                )
            else:
                left = (
                    tf.einsum(self.eq, w_left, left, name=f"ein_left") * self.norm_left
                )
        else:
            left = tf.transpose(left, [2, 0, 1])

        right = tf.gather(input_data[self.right.name], self.collect_from, axis=-1)
        w_right = self.w_right
        # LORA
        if self.lora:
            w_right = w_right + lora_reconstruction(
                *self.w_right_lora_tensors, lora_config=self.lora_config
            )
        w_right = tf.gather(w_right, self.w_tile_right, axis=-1)
        if self.is_central_atom_type_dependent[1]:
            w_right = tf.gather(w_right, input_data[constants.ATOMIC_MU_I], axis=0)
            right = (
                tf.einsum(self.eq_elem, w_right, right, name=f"ein_right")
                * self.norm_right
            )
        else:
            right = (
                tf.einsum(self.eq, w_right, right, name=f"ein_right") * self.norm_right
            )

        left = tf.tensor_scatter_nd_add(
            left, tf.reshape(self.collect_to, [-1, 1]), right, name=f"add_right_to_left"
        )
        if self.norm_out:
            left *= self.norm_out_factor

        return tf.transpose(left, [1, 2, 0])

    def prepare_variables_for_selected_elements(self, index_to_select):
        if np.any(self.is_central_atom_type_dependent):
            raise NotImplementedError()

    def upd_init_args_new_elements(self, new_element_map):
        if np.any(self.is_central_atom_type_dependent):
            raise NotImplementedError()


@capture_init_args
class InvariantLayerRMSNorm(TPInstruction):
    input_tensor_spec = {
        constants.N_ATOMS_BATCH_REAL: {"shape": [], "dtype": "int"},
        constants.N_ATOMS_BATCH_TOTAL: {"shape": [], "dtype": "int"},
    }

    def __init__(
        self,
        inpt: TPInstruction,
        name: str,
        type: str = "only_nonlin",
        init: str = "zeros",
    ):
        super().__init__(name=name)
        self.input = inpt
        self.n_out = inpt.n_out
        assert init in ["zeros", "ones", "random", "near_zero"]
        self.init = init
        self.type = type
        assert self.type in [
            "full",
            "only_nonlin",
            "sep_lin_gate",
        ], f"Unknown type {self.type}"

    def build(self, float_dtype):
        if not self.is_built:
            if self.type == "full":
                shape = [1, self.n_out, 1]
            elif self.type == "only_nonlin" or self.type == "sep_lin_gate":
                shape = [1, self.n_out - 1, 1]
            else:
                raise ValueError(f"Unknown type {self.type}")

            if self.init == "zeros":
                self.scale = tf.Variable(tf.zeros(shape, dtype=float_dtype))
                if self.type == "sep_lin_gate":
                    self.lin_scale = tf.Variable(tf.zeros([1, 1], dtype=float_dtype))
            elif self.init == "ones":
                self.scale = tf.Variable(tf.ones(shape, dtype=float_dtype))
                if self.type == "sep_lin_gate":
                    self.lin_scale = tf.Variable(tf.ones([1, 1], dtype=float_dtype))
            elif self.init == "random":
                self.scale = tf.Variable(tf.random.normal(shape, dtype=float_dtype))
                if self.type == "sep_lin_gate":
                    self.lin_scale = tf.Variable(
                        tf.random.normal([1, 1], dtype=float_dtype)
                    )
            elif self.init == "near_zero":
                self.scale = tf.Variable(
                    tf.random.normal(shape, stddev=1e-8, dtype=float_dtype)
                )
                if self.type == "sep_lin_gate":
                    self.lin_scale = tf.Variable(
                        tf.random.normal([1, 1], stddev=1e-8, dtype=float_dtype)
                    )

            self.epsilon = tf.constant(1e-10, dtype=float_dtype)
        self.is_built = True

    def frwrd(self, input_data: dict, training: bool = False):
        x = input_data[self.input.name]
        n_at_b_real = input_data[constants.N_ATOMS_BATCH_REAL]
        n_at_b_total = input_data[constants.N_ATOMS_BATCH_TOTAL]
        r_map = tf.reshape(
            tf.range(n_at_b_total, delta=1, dtype=tf.int32, name="r_map"), [-1, 1, 1]
        )
        if self.type == "full":
            rms = tf.math.rsqrt(
                tf.reduce_mean(x**2, axis=1, keepdims=True) + self.epsilon
            )
            rms = tf.where(r_map < n_at_b_real, rms, tf.zeros_like(rms))
            return x * rms * self.scale
        elif self.type == "only_nonlin":
            lin = x[:, 0, :]
        elif self.type == "sep_lin_gate":
            lin = x[:, 0, :] * self.lin_scale
        nonlin = x[:, 1:, :]
        nl_rms = tf.math.rsqrt(
            tf.reduce_mean(nonlin**2, axis=1, keepdims=True) + self.epsilon
        )
        nl_rms = tf.where(
            r_map < n_at_b_real, nonlin * nl_rms * self.scale, tf.zeros_like(nl_rms)
        )
        return tf.concat([lin[:, tf.newaxis, :], nl_rms], axis=1)


@capture_init_args
class FunctionReduceParticular(
    TPEquivariantInstruction, ElementsReduceInstructionMixin
):
    input_tensor_spec = {
        constants.N_ATOMS_BATCH_TOTAL: {"shape": [], "dtype": "int"},
        constants.ATOMIC_MU_I: {"shape": [None], "dtype": "int"},
    }

    def __init__(
        self,
        instructions: list[TPEquivariantInstruction],
        name: str,
        selected_l: int,
        selected_p: -1 | 1,
        n_out: int,
        is_central_atom_type_dependent: bool = False,
        number_of_atom_types: int = None,
        **kwargs,
    ):
        super(FunctionReduceParticular, self).__init__(name=name, lmax=selected_l)
        self.instructions = instructions
        self.selected_l = selected_l
        self.selected_p = selected_p
        self.n_out = n_out
        self.is_central_atom_type_dependent = is_central_atom_type_dependent
        self.number_of_atom_types = number_of_atom_types

        if self.is_central_atom_type_dependent:
            assert self.number_of_atom_types is not None

        instr_names = [instr.name for instr in self.instructions]
        assert len(instr_names) == len(set(instr_names)), "duplicate instruction names"
        assert (
            np.min([instr.lmax for instr in self.instructions]) >= self.selected_l
        ), f"Some of the instructions do not have required lmax {self.selected_l}"

        collector_data = []
        for m in range(-self.selected_l, self.selected_l + 1):
            lbl = 0 if self.selected_p > 0 else 1
            collector_data.append(
                [self.selected_l, m, f"", self.selected_p, self.selected_l]
            )
        self.coupling_meta_data = pd.DataFrame(
            collector_data, columns=["l", "m", "hist", "parity", "sum_of_ls"]
        )
        self.selector = {}
        for instr in self.instructions:
            instruction_collection = instr.select_functions(
                selected_l=self.selected_l, selected_p=self.selected_p
            )
            instruction_collection["total_sum_ind"] = []
            for index, row in instruction_collection["collect_meta_df"].iterrows():
                for idx, rw in self.coupling_meta_data.iterrows():
                    if (
                        (row["l"] == rw["l"])
                        & (row["m"] == rw["m"])
                        & (row["parity"] == rw["parity"])
                    ):
                        instruction_collection["total_sum_ind"].append(idx)
            instruction_collection["total_sum_ind"] = tf.constant(
                np.array(instruction_collection["total_sum_ind"]).reshape(-1, 1),
                dtype=tf.int32,
            )
            instruction_collection["n_out"] = instr.n_out
            self.selector[instr.name] = instruction_collection

    @tf.Module.with_name_scope
    def build(self, float_dtype):
        if not self.is_built:
            for k, v in self.selector.items():
                w_shape = v["w_shape"]
                n_in = v["n_out"]
                if self.is_central_atom_type_dependent:
                    c_shape = [self.number_of_atom_types, self.n_out, n_in, w_shape]
                else:
                    c_shape = [self.n_out, n_in, w_shape]

                limit = 1.0
                self.norm = tf.constant(1.0, dtype=float_dtype)
                setattr(
                    self,
                    f"reducing_{k}",
                    tf.Variable(
                        tf.random.normal(c_shape, stddev=limit, dtype=float_dtype),
                        name=f"reducing_{k}",
                    ),
                )
                setattr(
                    self,
                    f"norm_{k}",
                    tf.constant(1 / n_in**0.5, dtype=float_dtype),
                )
            self.float_dtype = float_dtype
            self.is_built = True

    def frwrd(self, input_data, training=False):
        collection = tf.zeros(
            [
                self.coupling_meta_data.shape[0],
                input_data[constants.N_ATOMS_BATCH_TOTAL],
                self.n_out,
            ],
            dtype=self.float_dtype,
        )
        for instr in self.instructions:
            instruction_collection = self.selector[instr.name]
            A_r = tf.gather(
                input_data[instr.name],
                instruction_collection["func_collect_ind"],
                axis=2,
            )

            w = tf.gather(
                getattr(self, f"reducing_{instr.name}"),
                instruction_collection["w_l_tile"],
                axis=-1,
            )
            if self.is_central_atom_type_dependent:
                w = tf.gather(w, input_data[constants.ATOMIC_MU_I], axis=0)
                eq = "aknw,anw->wak"
            else:
                eq = "knw,anw->wak"
            # w_al = tf.gather(w, instruction_collection["w_l_tile"], axis=-1)
            norm = getattr(self, f"norm_{instr.name}")
            pr = tf.einsum(eq, w, A_r, name=f"ein_{instr.name}") * norm

            collection = tf.tensor_scatter_nd_add(
                collection, instruction_collection["total_sum_ind"], pr
            )

        collection = tf.transpose(collection, [1, 2, 0])

        return collection

    def prepare_variables_for_selected_elements(self, index_to_select):
        if self.is_central_atom_type_dependent:

            reducing_tensor_names = [s for s in dir(self) if s.startswith("reducing_")]
            new_tensors = {}
            for tn in reducing_tensor_names:
                var = getattr(self, tn)
                new_tensors[tn] = tf.Variable(tf.gather(var, index_to_select, axis=0))

            return new_tensors

    def upd_init_args_new_elements(self, new_element_map):
        self._init_args["number_of_atom_types"] = len(new_element_map)


@capture_init_args
class ZBLPotential(TPInstruction, ElementsReduceInstructionMixin):
    """
    Compute ZBL pair potential

    """

    input_tensor_spec = {
        constants.BOND_IND_I: {"shape": [None], "dtype": "int"},
        constants.BOND_MU_I: {"shape": [None], "dtype": "int"},
        constants.BOND_MU_J: {"shape": [None], "dtype": "int"},
        constants.N_ATOMS_BATCH_TOTAL: {"shape": [], "dtype": "int"},
    }

    def __init__(
        self,
        bonds: TPInstruction | str,
        cutoff: dict | float,
        element_map: dict,
        delta_cutoff: float = 0.25,
        name: str = "ZBL",
    ):
        super().__init__(name=name)
        if isinstance(bonds, TPInstruction):
            self.input_name = bonds.name
        elif isinstance(bonds, str):
            self.input_name = bonds
        else:
            raise ValueError("Unknown entry for bonds")

        self.element_map_symbols = tf.Variable(
            list(element_map.keys()), trainable=False, name="element_map_symbols"
        )
        self.element_map_index = tf.Variable(
            list(element_map.values()), trainable=False, name="element_map_index"
        )

        self.delta_cutoff = np.array(delta_cutoff).reshape(1, 1)
        if isinstance(cutoff, (float, int)):
            self.cutoff = np.array(cutoff).reshape(1, 1)
            self.bond_zbl_cutoff = False
        elif isinstance(cutoff, dict):
            self.bond_zbl_cutoff = True
            self.nelem = len(element_map)

            expanded_cutoff_dict = process_cutoff_dict(cutoff, element_map)
            bond_ind_cut = np.zeros((self.nelem, self.nelem))
            for (el0, el1), cut in expanded_cutoff_dict.items():
                i0 = element_map[el0]
                i1 = element_map[el1]
                bond_ind_cut[i0, i1] = cut
                bond_ind_cut[i1, i0] = cut

            self.bond_zbl_cutoff_map = (
                bond_ind_cut.flatten().reshape(-1, 1).astype(np.float64)
            )
        else:
            raise ValueError(f"Unsupported cutoff type {type(cutoff)}")

        self.at_nums = np.array(
            [atomic_numbers[sym] for sym, ind in element_map.items()]
        ).reshape(-1, 1)

        # coefficients of ZBL potential
        self.phi_coefs = np.array([0.18175, 0.50986, 0.28022, 0.02817]).reshape(1, -1)
        self.phi_exps = np.array([-3.19980, -0.94229, -0.40290, -0.20162]).reshape(
            1, -1
        )
        # transformation coefficients to eV
        self.K = _e**2 / (4 * np.pi * _eps0) / 1e-10 / _e
        self.eps = 1e-12

    def __repr__(self):
        return f"{self.__class__.__name__}_{self.name}"

    @tf.Module.with_name_scope
    def build(self, float_dtype):
        if not self.is_built:
            self.delta_cutoff = tf.Variable(
                self.delta_cutoff,
                dtype=float_dtype,
                trainable=False,
                name="ZBL_delta_cutoff",
            )
            if self.bond_zbl_cutoff:
                self.bond_zbl_cutoff_map = tf.Variable(
                    self.bond_zbl_cutoff_map,
                    dtype=float_dtype,
                    trainable=False,
                    name="ZBL_cutoff",
                )
            else:
                self.cutoff = tf.Variable(
                    self.cutoff,
                    dtype=float_dtype,
                    trainable=False,
                    name="ZBL_cutoff",
                )
            self.at_nums = tf.constant(self.at_nums, dtype=float_dtype)
            self.phi_coefs = tf.constant(self.phi_coefs, dtype=float_dtype)
            self.phi_exps = tf.constant(self.phi_exps, dtype=float_dtype)
            self.K = tf.constant(self.K, dtype=float_dtype)
            self.eps = tf.constant(self.eps, dtype=float_dtype)

            self.is_built = True

    def phi(self, x):
        return tf.reduce_sum(
            self.phi_coefs * tf.exp(x * self.phi_exps), axis=1, keepdims=True
        )

    def dphi(self, x):
        return tf.reduce_sum(
            self.phi_coefs * self.phi_exps * tf.exp(x * self.phi_exps),
            axis=1,
            keepdims=True,
        )

    def d2phi(self, x):
        return tf.reduce_sum(
            self.phi_coefs * self.phi_exps * self.phi_exps * tf.exp(x * self.phi_exps),
            axis=1,
            keepdims=True,
        )

    # common factor: K*Zi*Zj*
    def fun_E_ij(self, nl_dist, a):
        return 1 / (nl_dist + self.eps) * self.phi(nl_dist / a)

    # common factor: K*Zi*Zj*
    def fun_dE_ij(self, nl_dist, a):
        return (-1 / (nl_dist**2 + self.eps)) * self.phi(nl_dist / a) + 1 / (
            nl_dist + self.eps
        ) * self.dphi(nl_dist / a) / a

    # common factor: K*Zi*Zj*
    def fun_d2E_ij(self, nl_dist, a):
        return (
            (+2 / (nl_dist**3 + self.eps)) * self.phi(nl_dist / a)
            + 2 * (-1 / (nl_dist**2 + self.eps)) * self.dphi(nl_dist / a) / a
            + (1 / (nl_dist + self.eps)) * self.d2phi(nl_dist / a) / (a**2)
        )

    # def remap_bond_index(self, mu_i, mu_j):
    #     mu_ij = tf.cast(
    #         tf.concat([tf.expand_dims(mu_i, 1), tf.expand_dims(mu_j, 1)], axis=1),
    #         dtype=tf.float32,
    #     )
    #     s = self.nelem * (self.nelem - 1) / 2
    #     mu_ij = (
    #         s
    #         - (self.nelem - tf.reduce_min(mu_ij, axis=1))
    #         * (self.nelem - tf.reduce_min(mu_ij, axis=1) - 1)
    #         / 2
    #         + tf.reduce_max(mu_ij, axis=1)
    #     )
    #     return tf.cast(mu_ij, dtype=tf.int32)

    def frwrd(self, input_data: dict, training=False):
        d = input_data[self.input_name]
        mu_i = input_data[constants.BOND_MU_I]
        mu_j = input_data[constants.BOND_MU_J]
        if self.bond_zbl_cutoff:
            mu_ij = mu_j + mu_i * tf.constant(self.nelem, dtype=mu_i.dtype)
            cutoff = tf.gather(self.bond_zbl_cutoff_map, mu_ij)
        else:
            cutoff = self.cutoff

        inner_cutoff = cutoff - self.delta_cutoff
        Zi = tf.gather(self.at_nums, mu_i, axis=0)
        Zj = tf.gather(self.at_nums, mu_j, axis=0)

        a = 0.46850 / (Zi**0.23 + Zj**0.23)
        E_ij = self.fun_E_ij(d, a)

        Ec = self.fun_E_ij(cutoff, a)
        dEc = self.fun_dE_ij(cutoff, a)
        d2Ec = self.fun_d2E_ij(cutoff, a)

        drcut = cutoff - inner_cutoff

        A = (-3 * dEc + drcut * d2Ec) / drcut**2
        B = (2 * dEc - drcut * d2Ec) / drcut**3
        C = -Ec + 1 / 2 * drcut * dEc - 1 / 12 * drcut**2 * d2Ec
        S = (
            A / 3 * (d - inner_cutoff) ** 3 + B / 4 * (d - inner_cutoff) ** 4 + C
        )  # S(r)
        S = tf.where(d < inner_cutoff, C, S)
        e_at = tf.where(d > cutoff, tf.zeros_like(S), Zi * Zj * (E_ij + S))

        ind_i = input_data[constants.BOND_IND_I]
        batch_tot_nat = input_data[constants.N_ATOMS_BATCH_TOTAL]
        energy = tf.math.unsorted_segment_sum(
            e_at, segment_ids=ind_i, num_segments=batch_tot_nat
        )

        energy = tf.reshape(self.K / 2 * energy, [-1, 1, 1])

        return energy

    def prepare_variables_for_selected_elements(self, index_to_select):
        raise NotImplementedError()

    def upd_init_args_new_elements(self, new_element_map):
        raise NotImplementedError()


##### BACKWARD COMPATIBILITY ######
BondSphericalHarmonic = SphericalHarmonic
