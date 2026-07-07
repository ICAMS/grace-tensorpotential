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
    chebvander,
)
from tensorpotential.functions.spherical_harmonics import SphericalHarmonics
from tensorpotential.instructions.base import (
    TPInstruction,
    TPEquivariantInstruction,
    capture_init_args,
    ElementsReduceInstructionMixin,
    LORAInstructionMixin,
    active_dense_nbr,
)
from tensorpotential.poly import (
    Monomial,
    Polynomial,
    init_coupling_symbols,
    get_symbol,
    normalize_poly,
)
from tensorpotential.utils import process_cutoff_dict

# Compute equivariant SPBF CG couple as dense matmul instead of sparce elemwise.
# Wastes FLOPS, but forces XLA to a better layout of the backward pass.
# The larger the model, the better the trade off
_USE_GEMM_COUPLE = True


def _dense_reshape_einsum(a_nl, bond_I, n_atoms):
    """Dense neighbor aggregation with NO in-graph gather: ``a_nl``/``bond_I`` are
    already in per-atom-uniform ``[n_atoms*max_neigh, n_rad, lm]`` order (data prep
    pre-pads each atom's bonds to ``max_neigh``; padded slots zeroed by the cutoff
    envelope). The dense layout is a ``reshape``, the sum a batched matmul over the
    neighbor axis. Requires ``n_bonds == n_atoms * max_neigh``.
    """
    nb = tf.shape(a_nl)[0]
    mn = nb // n_atoms
    n_rad = tf.shape(a_nl)[1]
    a_d = tf.reshape(a_nl, [n_atoms, mn, n_rad, tf.shape(a_nl)[2]])
    b_d = tf.reshape(bond_I, [n_atoms, mn, n_rad, tf.shape(bond_I)[2]])
    return tf.einsum("amnl,amnr->anlr", a_d, b_d)  # [atoms, n_rad, lm_y, lm_ind]


def _resolve_dense_nbr(dense_nbr):
    """Resolve the `dense_nbr` bool (None -> the active InstructionManager default)."""
    raw = active_dense_nbr() if dense_nbr is None else dense_nbr
    return bool(raw)


def _equiv_cg_couple(prod, lr_inds, cg, m_sum_ind, nfunc, lm_first, name):
    """Clebsch-Gordan couple the spherical-harmonic ⊗ indicator product.

    ``prod`` is ``[atoms, n, lm_y, lm_ind]``. The coupling combines ``n_cg``
    ``(lm_y, lm_ind)`` pairs weighted by ``cg``, summed into ``nfunc`` output
    channels (``m_sum_ind`` maps each pair to its channel). Returns
    ``[atoms, n, nfunc]`` (or ``[nfunc, atoms, n]`` if ``lm_first``).

    Default (``_USE_GEMM_COUPLE``): a dense GEMM ``prod_flat @ W`` where
    ``W[lm_y*lm_ind, nfunc]`` is the (constant, XLA-folded) CG matrix. No gather
    indexes the angular axes, so XLA keeps ``prod`` atom/bond-major instead of
    forcing the lm-major layout that otherwise propagates into the upstream
    einsum backward. Fallback: the original ``transpose([2,3,0,1]) + gather_nd +
    segment_sum`` lm-major couple (bit-identical math).
    """
    if _USE_GEMM_COUPLE:
        lm_y = int(prod.shape[2])
        lm_ind = int(prod.shape[3])
        P = lm_y * lm_ind
        nf = int(nfunc) if not tf.is_tensor(nfunc) else nfunc
        # row-major flat index into (lm_y, lm_ind): left * lm_ind + right
        lr_flat = lr_inds[:, 0] * lm_ind + lr_inds[:, 1]  # [n_cg]
        # Dense CG matrix W[lm_y*lm_ind, nfunc]; constant -> XLA folds it.
        W = tf.scatter_nd(
            tf.stack([lr_flat, m_sum_ind], axis=1),
            tf.reshape(tf.cast(cg, prod.dtype), [-1]),
            tf.stack([P, tf.cast(nf, tf.int32)]),
        )
        batch = tf.shape(prod)[:2]
        prod_flat = tf.reshape(prod, tf.concat([batch, [P]], axis=0))
        out = tf.einsum("anp,pf->anf", prod_flat, W)  # [atoms, n, nfunc]
        if lm_first:
            return tf.transpose(out, [2, 0, 1], name=f"trans_201{name}")
        return out

    # Fallback: original lm-major gather couple.
    tnsr = tf.transpose(prod, [2, 3, 0, 1])  # [lm_y, lm_ind, atoms, n]
    prod_g = tf.gather_nd(params=tnsr, indices=lr_inds)  # [n_cg, atoms, n]
    prod_g = prod_g * tf.cast(cg, prod_g.dtype)
    prod_g = tf.math.unsorted_segment_sum(
        prod_g, m_sum_ind, num_segments=nfunc, name=f"sum_cg_{name}"
    )
    if lm_first:
        return prod_g  # already [lm, atoms, n]
    return tf.transpose(prod_g, [1, 2, 0], name=f"trans_120{name}")


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

    def frwrd(self, input_data: dict, training=False, local=False):

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
        # self.epsilon = None

    @tf.Module.with_name_scope
    def build(self, float_dtype):
        if not self.is_built:
            # self.epsilon = tf.constant(1e-10, dtype=tf.float32)
            self.is_built = True

    def frwrd(self, input_data: dict, training=False, local=False):
        r_ij = input_data[self.bonds]
        d_ij = input_data[self.bond_length]
        # return r_ij / (d_ij + self.epsilon)
        return r_ij / d_ij


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

        self.sh = SphericalHarmonics(self.lmax, **kwargs)
        self.coupling_meta_data = self.init_uncoupled_meta_data()
        self.coupling_origin = None

    @tf.Module.with_name_scope
    def build(self, float_dtype):
        if not self.sh.is_built:
            self.sh.build(float_dtype)

    def frwrd(self, input_data: dict, training=False, local=False):
        vhat = input_data[self.vhat]
        return self.sh(vhat)


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

    def get_cutoff(self) -> float | None:
        return float(np.array(self.rcut))

    def build(self, float_dtype):
        if not self.is_built:
            self.inv_avg_n_neigh = tf.constant(
                self.inv_avg_n_neigh,
                dtype=float_dtype,
            )
            self.rcut = tf.constant(self.rcut, dtype=float_dtype)
            self.rc = tf.Variable(
                self._init_args["rcut"], dtype=tf.float64, trainable=False, name="rc"
            )
            self.is_built = True

    def frwrd(self, input_data: dict, training=False, local=False):
        y = input_data[self.sg.name]
        r = input_data[self.bonds.name]
        # cut_func = self.cutoff_func(r / self.rcut, self.p)
        # cut_func = tf.where(r > self.rcut, tf.zeros_like(r, dtype=r.dtype), cut_func)
        rcut = tf.cast(self.rcut, dtype=r.dtype)
        y = tf.where(r > rcut, tf.zeros_like(y), y)
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

    def get_cutoff(self) -> float | None:
        # `rc` (a tf.Variable) is only created in basis_function.build(); fall back
        # to the plain `rcut` float set in __init__ when the basis isn't built yet
        # (e.g. when grace_uq inspects instructions loaded from model.yaml).
        bf = self.basis_function
        if hasattr(bf, "rc"):
            return float(bf.rc.numpy())
        return float(np.array(bf.rcut))

    @tf.Module.with_name_scope
    def build(self, float_dtype):
        if hasattr(self.basis_function, "build"):
            self.basis_function.build(float_dtype)
        self.rc = self.basis_function.rc

    @tf.Module.with_name_scope
    def frwrd(self, input_data: dict, training=False, local=False):
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
        constants.BOND_MU_I: {"shape": [None], "dtype": "int"},
        constants.BOND_MU_J: {"shape": [None], "dtype": "int"},
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

    def get_cutoff(self) -> float | None:
        return float(np.max(np.array(self.bond_cutoff_map)))

    def get_bond_cutoff_map(self) -> np.ndarray | None:
        return np.array(self.bond_cutoff_map)

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
                new_bond_cutoff_map.reshape(-1, 1)  # .astype(np.float32)
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

    def frwrd(self, input_data: dict, training: bool = False, local: bool = False):
        d = input_data[self.input_name]
        mu_i = input_data[constants.BOND_MU_I]
        mu_j = input_data[constants.BOND_MU_J]
        mu_ij = mu_j + mu_i * tf.constant(self.nelem, dtype=mu_i.dtype)
        cutoff = tf.gather(self.bond_cutoff_map, mu_ij)
        cutoff = tf.cast(cutoff, dtype=d.dtype)
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
            if self.scales.dtype != d.dtype:
                scales = tf.cast(self.scales, dtype=d.dtype)
            else:
                scales = self.scales
            if self.grid.dtype != d.dtype:
                grid = tf.cast(self.grid, dtype=d.dtype)
            else:
                grid = self.grid
            basis = tf.math.exp(-(scales**2) * (d - grid) ** 2)
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
            ), "Need to provide shape if basis is not TPInstruction"
            self.input_shape = input_shape
        elif basis is not None:
            raise ValueError(f"Unknown {basis = }")

        self.l_tile = tf.cast(
            tf.concat(
                [tf.ones((2 * l_idx + 1)) * l_idx for l_idx in range(self.lmax + 1)],
                axis=0,
            ),
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
            # self.norm = tf.convert_to_tensor(1, dtype=float_dtype)
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

    def frwrd(self, input_data, training=False, local=False):
        basis = input_data[self.basis_name]

        crad = tf.cast(self.crad, dtype=basis.dtype)
        y = tf.einsum("nlk,ak->anl", crad, basis)
        # y *= self.norm
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
            ), "Need to provide shape if basis is not TPInstruction"
            self.input_shape = input_shape
        elif isinstance(basis, TPInstruction):
            self.basis_name = basis.name
            self.input_shape = basis.nfunc
        elif basis is not None:
            raise ValueError(f"Unknown {basis = }")
        elif basis is None:
            assert (
                input_shape is not None
            ), "Need to provide shape if basis is not TPInstruction"
            self.input_shape = input_shape
        self.chemical_embedding_i = chemical_embedding_i
        self.chemical_embedding_j = chemical_embedding_j
        self.chem_i_is_per_atom = getattr(chemical_embedding_i, "is_per_atom", False)
        self.chem_j_is_per_atom = getattr(chemical_embedding_j, "is_per_atom", False)

        if self.chemical_embedding_i is not None:
            self.input_shape += self.chemical_embedding_i.embedding_size
        if self.chemical_embedding_j is not None:
            self.input_shape += self.chemical_embedding_j.embedding_size

        self.l_tile = tf.cast(
            tf.concat(
                [tf.ones((2 * l_idx + 1)) * l_idx for l_idx in range(self.lmax + 1)],
                axis=0,
            ),
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

    def frwrd(self, input_data, training=False, local=False):
        basis = input_data[self.basis_name]

        if self.chemical_embedding_i is not None:
            z = input_data[self.chemical_embedding_i.name]
            z = tf.cast(z, basis.dtype)
            if self.chem_i_is_per_atom:
                bond_z_i = tf.gather(z, input_data[constants.BOND_IND_I], axis=0)
            else:
                bond_z_i = tf.gather(z, input_data[constants.BOND_MU_I], axis=0)
            basis = tf.concat([basis, bond_z_i], axis=1)

        if self.chemical_embedding_j is not None:
            z = input_data[self.chemical_embedding_j.name]
            z = tf.cast(z, basis.dtype)
            if self.chem_j_is_per_atom:
                bond_z_j = tf.gather(z, input_data[constants.BOND_IND_J], axis=0)
            else:
                bond_z_j = tf.gather(z, input_data[constants.BOND_MU_J], axis=0)
            basis = tf.concat([basis, bond_z_j], axis=1)

        y = self.mlp(basis)
        if self.norm:
            variance = tf.math.reduce_variance(y, axis=-1, keepdims=True, name=None)
            epsilon = tf.cast(self.epsilon, y.dtype)
            inv = tf.math.rsqrt(variance + epsilon)
            gamma = tf.cast(self.gamma, y.dtype)
            y = y * inv * gamma

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
        name: str = "MLPRadialFunction",
        basis: RadialBasis | TPInstruction | str = None,
        input_shape: int = None,
        hidden_layers: list[int] = None,
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
            ), "Need to provide shape if basis is not TPInstruction"
            self.input_shape = input_shape
        elif isinstance(basis, TPInstruction):
            self.basis_name = basis.name
            self.input_shape = basis.nfunc
        elif basis is not None:
            raise ValueError(f"Unknown {basis = }")
        elif basis is None:
            assert (
                input_shape is not None
            ), "Need to provide shape if basis is not TPInstruction"
            self.input_shape = input_shape

        self.l_tile = tf.cast(
            tf.concat(
                [tf.ones((2 * l_idx + 1)) * l_idx for l_idx in range(self.lmax + 1)],
                axis=0,
            ),
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

    def frwrd(self, input_data, training=False, local=False):
        basis = input_data[self.basis_name]
        for act, layer in zip(self.activation, self.layers):
            act = ACTIVATION_DICT[act]
            basis = act(layer(basis))
        y = self.layers[-1](basis)

        if self.chem_embedding is not None:
            if local:
                raise NotImplementedError
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

    def get_element_map(self) -> tuple[np.ndarray, np.ndarray] | None:
        return (
            self.element_map_symbols.numpy().astype(str),
            self.element_map_index.numpy(),
        )

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

    def frwrd(self, input_data: dict, training=False, local=False):
        if not self.lora:
            return self.w
        else:
            return self.w + lora_reconstruction(
                *self.lora_tensors, lora_config=self.lora_config
            )

    def get_index_to_select(self, elements_to_select):
        # 1. Create a dictionary mapping {symbol: index} for fast O(1) lookup.
        # We decode the bytes to string here to match the format of elements_to_select.
        symbol_to_index_map = {
            sym.decode(): idx
            for idx, sym in zip(
                self.element_map_index.numpy(), self.element_map_symbols.numpy()
            )
        }

        # 2. Iterate through the input list (elements_to_select) to preserve its order.
        index_to_select = []
        for element in elements_to_select:
            # Only append if the element exists in our map
            if element in symbol_to_index_map:
                index_to_select.append(symbol_to_index_map[element])
            else:
                raise ValueError(
                    f"Element {element} not found in the map ({symbol_to_index_map})."
                )

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
        lm_first: bool = False,
    ):
        self.lm_first = lm_first
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

        self.indicator_is_per_atom = getattr(indicator, "is_per_atom", False)
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
        ), f"Radial part and angular part lmax do not match ({self.radial.lmax} vs {self.lmax})"

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

    def frwrd(self, input_data: dict, training=False, local=False):
        r = input_data[self.radial.name]
        y = input_data[self.angular.name]
        if self.slice_angular is not None:
            y = y[:, : self.slice_angular]

        if r.dtype != y.dtype:
            y = tf.cast(y, r.dtype)
        a_nl = tf.einsum("jnl,jl->jnl", r, y)
        if self.indicator is not None:
            z = input_data[self.indicator.name]
            z = tf.cast(z, r.dtype)

            z_tr = self.lin_transform(z)
            if z_tr.dtype != a_nl.dtype:
                a_nl = tf.cast(a_nl, z_tr.dtype)

            # Per-atom indicators: gather by neighbour atom index.
            # Per-element indicators: gather by type.
            if self.indicator_is_per_atom:
                gather_idx = input_data[constants.BOND_IND_J]
            else:
                gather_idx = input_data[constants.BOND_MU_J]

            if self.indicator_l_depend:
                z_tr = tf.reshape(z_tr, [-1, self.radial.n_rad_max, self.lmax + 1])
                z_tr = tf.gather(z_tr, self.radial.l_tile, axis=2)
                bond_z_tr = tf.gather(z_tr, gather_idx, axis=0)
                a_nl = tf.einsum("jnl,jnl->jnl", a_nl, bond_z_tr)
            else:
                bond_z_tr = tf.gather(z_tr, gather_idx, axis=0)
                a_nl = tf.einsum("jnl,jn->jnl", a_nl, bond_z_tr)

        if self.sum_neighbors:
            ind_i = input_data[constants.BOND_IND_I]
            batch_tot_nat = (
                tf.shape(input_data[constants.ATOMIC_MU_I_LOCAL])[0]
                if local
                else tf.shape(input_data[constants.ATOMIC_MU_I])[0]
            )
            a_nl = tf.math.unsorted_segment_sum(
                a_nl, segment_ids=ind_i, num_segments=batch_tot_nat
            )
            if self.inv_avg_n_neigh is not None:
                inv_avg_n_neigh = tf.cast(self.inv_avg_n_neigh, a_nl.dtype)
                if self.per_specie_n_neigh:
                    if local:
                        raise NotImplementedError()
                    nneigh_norm = tf.gather(
                        inv_avg_n_neigh,
                        input_data[constants.ATOMIC_MU_I],
                        axis=0,
                    )
                    a_nl *= nneigh_norm[:, :, tf.newaxis]
                    pass
                else:
                    a_nl *= inv_avg_n_neigh

        if self.lm_first:
            a_nl = tf.transpose(a_nl, [2, 0, 1])  # [lm, atoms, n]
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
    }

    def __init__(
        self,
        angular: SphericalHarmonic,
        indicator: TPEquivariantInstruction,
        name: str,
        lmax: int,
        Lmax: int,
        radial: TPInstruction,
        keep_parity: list[list] = None,
        history_drop_list: list = None,
        l_max_ind: int = None,
        max_sum_l: int = None,
        sum_neighbors: bool = True,
        avg_n_neigh: float = 1.0,
        normalize: bool = False,
        radial_basis: RadialBasis | TPInstruction = None,
        hidden_layers: list[int] = None,
        chemical_embedding: ScalarChemicalEmbedding = None,
        lm_first: bool = False,
        dense_nbr: bool = None,
        **kwargs,
    ):
        self.lm_first = lm_first
        super().__init__(name=name, lmax=Lmax)
        self.angular = angular
        self.indicator = indicator
        self.radial = radial
        self.chemical_embedding = chemical_embedding
        self.chem_emb_is_per_atom = getattr(chemical_embedding, "is_per_atom", False)

        self.n_out = self.radial.n_rad_max

        self.sum_neighbors = sum_neighbors

        self.per_specie_n_neigh = False
        self.inv_avg_n_neigh = 1.0 / avg_n_neigh

        assert self.angular.coupling_meta_data is not None
        assert self.indicator.coupling_meta_data is not None

        if self.chemical_embedding is not None:
            self.input_tensor_spec = {
                **self.input_tensor_spec,
                constants.BOND_MU_J: {"shape": [None], "dtype": "int"},
            }
            # Find L=0, m=0, parity=+1 index in indicator's coupling metadata
            meta = self.indicator.coupling_meta_data
            l0_mask = (meta["l"] == 0) & (meta["m"] == 0) & (meta["parity"] == 1)
            l0_idx = meta.index[l0_mask].tolist()
            assert len(l0_idx) == 1, (
                "Expected exactly one L=0,m=0,p=+1 component in indicator, "
                f"found {len(l0_idx)}"
            )
            self.chem_l0_idx = l0_idx[0]
            self.n_lm_indicator = len(meta)
            self.chem_linear = Linear(
                n_in=self.chemical_embedding.embedding_size,
                n_out=self.indicator.n_out,
                name=f"{self.name}_ChemProj",
            )

        if self.angular.lmax > lmax:
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
            for l_val in range(Lmax + 1):
                p = 1 if l_val % 2 == 0 else -1
                plist.append([l_val, p])
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
            optimize_ms_comb=False,
        )
        self.coupling_origin = [self.angular.name, self.indicator.name]

        self.lr_inds = tf.constant(
            np.concatenate(
                [
                    np.concatenate(self.coupling_meta_data["left_inds"]).reshape(-1, 1),
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

        init_coupling_symbols(self)

        # Dense neighbor aggregation (reshape): default mode of this equivariant SPBF.
        # True -> the dense reshape compute (needs the per-atom-uniform bond layout);
        # False -> segment_sum. Resolve `dense_nbr=None` against the active
        # InstructionManager default; record it for serialization. Adds NO input
        # (same signature as segment_sum); `dense_capable` flags it for the dual
        # `compute`/`compute_dense` SavedModel export.
        self.dense_nbr = _resolve_dense_nbr(dense_nbr)
        self.dense_capable = True
        self._init_args["dense_nbr"] = self.dense_nbr

    @tf.Module.with_name_scope
    def build(self, float_dtype):
        if not self.is_built:
            self.inv_avg_n_neigh = tf.constant(
                self.inv_avg_n_neigh,
                dtype=float_dtype,
            )

            self.cg = tf.constant(self.cg, dtype=float_dtype)
            if self.chemical_embedding is not None:
                self.chem_linear.build(float_dtype)
                self.chem_l0_mask = tf.one_hot(
                    self.chem_l0_idx,
                    depth=self.n_lm_indicator,
                    dtype=float_dtype,
                )
            self.is_built = True

    def frwrd(self, input_data: dict, training=False, local=False):
        ind_i = input_data[constants.BOND_IND_I]
        ind_j = input_data[constants.BOND_IND_J]
        y = input_data[self.angular.name]

        indicator_data = input_data[self.indicator.name]
        # Indicator may be in [lm, atoms, n] layout — convert to [atoms, n, lm]
        # for bond-level operations
        if self.lm_first:
            indicator_data = tf.transpose(indicator_data, [1, 2, 0])
        bond_I = tf.gather(indicator_data, ind_j, axis=0)

        if self.chemical_embedding is not None:
            z = input_data[self.chemical_embedding.name]
            z = tf.cast(z, bond_I.dtype)
            z_proj = self.chem_linear(z)  # [n_types or n_atoms, n_channels]
            if self.chem_emb_is_per_atom:
                bond_z = tf.gather(z_proj, ind_j, axis=0)
            else:
                mu_j = input_data[constants.BOND_MU_J]
                bond_z = tf.gather(z_proj, mu_j, axis=0)  # [n_bonds, n_channels]
            chem_l0_mask = tf.cast(self.chem_l0_mask, bond_I.dtype)
            bond_I = (
                bond_I
                + bond_z[:, :, tf.newaxis] * chem_l0_mask[tf.newaxis, tf.newaxis, :]
            )

        r = input_data[self.radial.name]
        if self.slice_angular is not None:
            y = y[:, : self.slice_angular]
        if r.dtype != y.dtype:
            y = tf.cast(y, dtype=r.dtype)
        bond_RY_nl = tf.einsum("jnl,jl->jnl", r, y)

        if self.sum_neighbors:
            batch_tot_nat = (
                tf.shape(input_data[constants.ATOMIC_MU_I_LOCAL])[0]
                if local
                else tf.shape(input_data[constants.ATOMIC_MU_I])[0]
            )
            if self.dense_nbr:
                prod = _dense_reshape_einsum(bond_RY_nl, bond_I, batch_tot_nat)
            else:
                prod = tf.einsum("jnl,jnr->jnlr", bond_RY_nl, bond_I)
                prod = tf.math.unsorted_segment_sum(
                    prod,
                    segment_ids=ind_i,
                    num_segments=batch_tot_nat,
                    name=f"sum_nei_{self.name}",
                )
            if self.inv_avg_n_neigh is not None:
                inv_avg_n_neigh = tf.cast(self.inv_avg_n_neigh, dtype=prod.dtype)
                prod *= inv_avg_n_neigh
        else:
            prod = tf.einsum("jnl,jnr->jnlr", bond_RY_nl, bond_I)

        # CG coupling: prod is [atoms, n, lm_y, lm_ind]
        return _equiv_cg_couple(
            prod,
            self.lr_inds,
            self.cg,
            self.m_sum_ind,
            self.nfunc,
            self.lm_first,
            self.name,
        )


@capture_init_args
class SPBF(TPEquivariantInstruction):
    """
    Single Particle Basis Function.

    Combines Chebyshev radial basis, MLP radial function, and angular coupling
    into a single instruction. The cutoff envelope is applied after the
    radial x angular product, right before neighbor reduction, which allows
    using biases in the MLP hidden layers.

    The MLP radial function is always conditioned on chemistry via concatenation:

        R_{nl}(r, mu_i, mu_j) = MLP([cheb(r), Z(mu_i), Z(mu_j)])[n, l]

    where Z is a learnable per-species embedding provided by ``chemical_embedding``.

    Additionally, when the indicator is not a ScalarChemicalEmbedding (i.e.,
    indicator is None or a TPEquivariantInstruction), the chemical embedding
    also provides a skip-add species-dependent modulation of the radial-angular
    product (scalar mode) or L=0 injection into the indicator (equivariant mode).

    Supports two modes based on the indicator type:
    - Scalar indicator (None or ScalarChemicalEmbedding)
    - Equivariant indicator (TPEquivariantInstruction)
    """

    input_tensor_spec = {
        constants.BOND_IND_I: {"shape": [None], "dtype": "int"},
        constants.ATOMIC_MU_I: {"shape": [None], "dtype": "int"},
        constants.BOND_MU_I: {"shape": [None], "dtype": "int"},
        constants.BOND_MU_J: {"shape": [None], "dtype": "int"},
    }

    def __init__(
        self,
        name: str,
        bonds: TPInstruction | str,
        angular: SphericalHarmonic,
        chemical_embedding: ScalarChemicalEmbedding,
        n_rad_max: int,
        n_rad_basis: int,
        rcut: float,
        lmax: int = None,
        p: int = 5,
        hidden_layers: list[int] = None,
        activation: str | list[str] = "silu",
        # Indicator — determines scalar vs equivariant mode
        indicator: ScalarChemicalEmbedding | TPEquivariantInstruction = None,
        # Equivariant indicator parameters
        Lmax: int = None,
        keep_parity: list[list] = None,
        history_drop_list: list = None,
        l_max_ind: int = None,
        max_sum_l: int = None,
        normalize_cg: bool = False,
        # Neighbor averaging
        avg_n_neigh: float | dict = 1.0,
        sum_neighbors: bool = True,
        lm_first: bool = False,
        dense_nbr: bool = None,
        **kwargs,
    ):
        # Determine mode
        self.equivariant_mode = isinstance(indicator, TPEquivariantInstruction)
        self.scalar_mode = not self.equivariant_mode

        # Angular lmax for radial-angular coupling
        if lmax is None:
            lmax = angular.lmax
        else:
            assert angular.lmax >= lmax
        self._lmax = lmax

        # Output lmax
        if self.equivariant_mode:
            assert Lmax is not None, "Lmax must be specified for equivariant indicator"
            output_lmax = Lmax
        else:
            output_lmax = lmax

        super().__init__(name=name, lmax=output_lmax)

        # Bond distances reference
        if isinstance(bonds, TPInstruction):
            self.bonds_name = bonds.name
        elif isinstance(bonds, str):
            self.bonds_name = bonds
        else:
            raise ValueError(f"Unknown entry for bonds: {bonds}")

        self.angular = angular
        self.indicator = indicator
        self.chemical_embedding = chemical_embedding
        self.n_rad_max = n_rad_max
        self.n_rad_basis = n_rad_basis
        self._rcut = rcut
        self.p = p
        self.sum_neighbors = sum_neighbors
        self.n_out = n_rad_max
        self.lm_first = lm_first

        # MLP setup
        if hidden_layers is None:
            hidden_layers = [64, 64]
        self.hidden_layers = hidden_layers

        if isinstance(activation, str):
            self.activation = [activation] * len(self.hidden_layers)
        else:
            assert len(activation) == len(self.hidden_layers)
            self.activation = activation

        # l_tile: maps (lmax+1) radial channels to (lmax+1)^2 via l index
        self.l_tile = tf.cast(
            tf.concat(
                [tf.ones((2 * l + 1)) * l for l in range(self._lmax + 1)],
                axis=0,
            ),
            tf.int32,
        )

        # MLP input: chebyshev basis + Z(mu_i) + Z(mu_j) (always chemistry-conditioned)
        self.mlp_emb_size = chemical_embedding.embedding_size
        mlp_input_size = n_rad_basis + 2 * self.mlp_emb_size

        # MLP layers: hidden layers have bias, output layer does not
        layer_sizes = (
            [mlp_input_size] + self.hidden_layers + [n_rad_max * (self._lmax + 1)]
        )
        n_layers = len(layer_sizes) - 1
        self.mlp_layers = []
        for i in range(n_layers):
            is_output_layer = i == n_layers - 1
            self.mlp_layers.append(
                Linear(
                    layer_sizes[i],
                    layer_sizes[i + 1],
                    name=f"{self.name}_Linear_{i}",
                    use_bias=not is_output_layer,
                    init_type="normal",
                    normalize=True,
                )
            )

        # Neighbor averaging
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

        # Slice angular if angular.lmax > coupling lmax
        if self.angular.lmax > self._lmax:
            self.slice_angular = (self._lmax + 1) ** 2
        else:
            self.slice_angular = None

        # CG coupling metadata
        if keep_parity is None:
            plist = []
            for l_val in range(output_lmax + 1):
                p_val = 1 if l_val % 2 == 0 else -1
                plist.append([l_val, p_val])
        else:
            plist = keep_parity
        self.plist = plist
        # --- Scalar mode setup ---
        if self.scalar_mode:
            if isinstance(self.indicator, ScalarChemicalEmbedding):
                # R * Y * Z: indicator provides species-dependent modulation
                self.lin_transform = Linear(
                    n_in=self.indicator.embedding_size,
                    n_out=n_rad_max,
                    name=self.name + "_ChemIndTransf",
                )
            else:
                # R * Y only (no indicator, chemistry is in the MLP)
                self.lin_transform = None

            # Coupling metadata from angular
            if self.slice_angular is not None:
                self.coupling_meta_data = self.angular.coupling_meta_data.query(
                    f"l <= {self._lmax}"
                )
            else:
                self.coupling_meta_data = self.angular.coupling_meta_data
            self.coupling_origin = self.angular.coupling_origin

        # --- Equivariant mode setup ---
        else:
            self.chem_emb_is_per_atom = getattr(
                chemical_embedding, "is_per_atom", False
            )

            # Need BOND_IND_J to gather indicator at neighbor sites
            self.input_tensor_spec = {
                **self.input_tensor_spec,
                constants.BOND_IND_J: {"shape": [None], "dtype": "int"},
            }

            # L=0 injection: project chemical_embedding → indicator n_out,
            # inject into all L=0,m=0,p=+1 components of the indicator
            meta = self.indicator.coupling_meta_data
            l0_mask = (meta["l"] == 0) & (meta["m"] == 0) & (meta["parity"] == 1)
            l0_idx = meta.index[l0_mask].tolist()
            assert len(l0_idx) >= 1, (
                "Expected at least one L=0,m=0,p=+1 component in indicator, "
                f"found {len(l0_idx)}"
            )
            self.chem_l0_indices = l0_idx
            self.n_lm_indicator = len(meta)
            self.chem_linear = Linear(
                n_in=self.chemical_embedding.embedding_size,
                n_out=self.indicator.n_out,
                name=f"{self.name}_ChemProj",
            )

            angular_meta = (
                self.angular.coupling_meta_data.query(f"l <= {self._lmax}")
                if self.slice_angular is not None
                else self.angular.coupling_meta_data
            )
            self.coupling_meta_data = real_coupling_metainformation(
                A=angular_meta,
                B=self.indicator.coupling_meta_data,
                lmax=self._lmax,
                lmax_B=l_max_ind,
                Lmax=output_lmax,
                history_drop_list=history_drop_list,
                max_sum_l=max_sum_l,
                keep_parity=self.plist,
                normalize=normalize_cg,
                optimize_ms_comb=False,
            )
            self.coupling_origin = [self.angular.name, self.indicator.name]

            # Pre-compute CG indices and coefficients
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

        init_coupling_symbols(self)

        # Dense neighbor aggregation (reshape): only the equivariant path forms the
        # per-atom 4-D product, so dense_nbr is a no-op (masked off) in scalar mode.
        # True -> dense reshape compute; False -> segment_sum. Adds NO input (same
        # signature); `dense_capable` flags the equivariant case for the dual
        # compute/compute_dense SavedModel export.
        self.dense_nbr = _resolve_dense_nbr(dense_nbr) and self.equivariant_mode
        self.dense_capable = self.equivariant_mode
        self._init_args["dense_nbr"] = self.dense_nbr

    def get_cutoff(self) -> float | None:
        return float(self._rcut)

    @tf.Module.with_name_scope
    def build(self, float_dtype):
        if not self.is_built:
            # MLP layers
            for layer in self.mlp_layers:
                layer.build(float_dtype)

            # Cutoff radius as non-trainable variable
            self.rc = tf.Variable(
                self._rcut, dtype=tf.float64, trainable=False, name="rcut"
            )

            self.inv_avg_n_neigh = tf.constant(
                self.inv_avg_n_neigh,
                dtype=float_dtype,
            )

            if self.scalar_mode:
                if self.lin_transform is not None:
                    self.lin_transform.build(float_dtype)
            else:
                self.cg = tf.constant(self.cg, dtype=float_dtype)
                self.chem_linear.build(float_dtype)
                self.chem_l0_mask = tf.reduce_sum(
                    tf.one_hot(
                        self.chem_l0_indices,
                        depth=self.n_lm_indicator,
                        dtype=float_dtype,
                    ),
                    axis=0,
                )

            self.is_built = True

    def _compute_radial_basis(self, r):
        """Chebyshev basis (kind=1, reversed=False) without envelope."""
        rc = tf.cast(self.rc, r.dtype)
        r_rescale = 2.0 * (1.0 - tf.abs(1.0 - r / rc)) - 1.0
        # Clamp to [-1, 1] so padded bonds (r >> rc) don't blow up Chebyshev
        r_rescale = tf.clip_by_value(r_rescale, -1.0, 1.0)
        basis = chebvander(r_rescale, self.n_rad_basis + 1, kind=1)[:, 1:]
        return basis

    def _compute_envelope(self, r):
        """Polynomial cutoff envelope, zeroed for r >= rc."""
        rc = tf.cast(self.rc, r.dtype)
        env = cutoff_func_p_order_poly(r / rc, self.p)
        # Hard zero for bonds beyond cutoff (padding safety)
        env = tf.where(r < rc, env, tf.zeros_like(env))
        return env

    def _mlp_forward(self, basis, z_i, z_j):
        """MLP: [n_bonds, n_rad_basis + 2*emb] -> [n_bonds, n_rad_max, (lmax+1)^2]."""
        x = tf.concat([basis, z_i, z_j], axis=-1)
        for act_name, layer in zip(self.activation, self.mlp_layers[:-1]):
            act_fn = ACTIVATION_DICT[act_name]
            x = act_fn(layer(x))
        x = self.mlp_layers[-1](x)
        # [n_bonds, n_rad_max * (lmax+1)] -> [n_bonds, n_rad_max, lmax+1]
        x = tf.reshape(x, [-1, self.n_rad_max, self._lmax + 1])
        # Gather by l: [n_bonds, n_rad_max, (lmax+1)^2]
        x = tf.gather(x, self.l_tile, axis=-1)
        return x

    def frwrd(self, input_data: dict, training=False, local=False):
        # Bond distances
        r = input_data[self.bonds_name]  # [n_bonds, 1]

        # Chebyshev basis (no envelope)
        basis = self._compute_radial_basis(r)  # [n_bonds, n_rad_basis]

        # Chemical embeddings for MLP: [gk, Z(mu_i), Z(mu_j)]
        z = input_data[self.chemical_embedding.name]
        z = tf.cast(z, basis.dtype)
        z_i = tf.gather(z, input_data[constants.BOND_MU_I], axis=0)
        z_j = tf.gather(z, input_data[constants.BOND_MU_J], axis=0)

        # MLP radial function
        R = self._mlp_forward(basis, z_i, z_j)  # [n_bonds, n_rad_max, (lmax+1)^2]

        # Spherical harmonics
        y = input_data[self.angular.name]  # [n_bonds, (angular_lmax+1)^2]
        if self.slice_angular is not None:
            y = y[:, : self.slice_angular]

        if R.dtype != y.dtype:
            y = tf.cast(y, R.dtype)

        # Radial x Angular product
        a_nl = tf.einsum("jnl,jl->jnl", R, y)

        # Envelope (applied before neighbor reduction)
        envelope = self._compute_envelope(r)  # [n_bonds, 1]
        envelope = tf.cast(envelope, a_nl.dtype)

        if self.scalar_mode:
            return self._frwrd_scalar(input_data, a_nl, envelope, local=local)
        else:
            return self._frwrd_equivariant(input_data, a_nl, envelope, local=local)

    def _frwrd_scalar(self, input_data, a_nl, envelope, local=False):
        # R * Y * Z when indicator is ScalarChemicalEmbedding
        if self.lin_transform is not None:
            z = input_data[self.indicator.name]
            z = tf.cast(z, a_nl.dtype)
            mu_j = input_data[constants.BOND_MU_J]
            z_tr = self.lin_transform(z)
            if z_tr.dtype != a_nl.dtype:
                a_nl = tf.cast(a_nl, z_tr.dtype)
            bond_z_tr = tf.gather(z_tr, mu_j, axis=0)
            a_nl = tf.einsum("jnl,jn->jnl", a_nl, bond_z_tr)

        # Apply envelope: [n_bonds, 1, 1] broadcasts with [n_bonds, n_rad, nlm]
        a_nl = a_nl * envelope[:, :, tf.newaxis]

        # Sum over neighbors
        if self.sum_neighbors:
            ind_i = input_data[constants.BOND_IND_I]
            batch_tot_nat = (
                tf.shape(input_data[constants.ATOMIC_MU_I_LOCAL])[0]
                if local
                else tf.shape(input_data[constants.ATOMIC_MU_I])[0]
            )
            a_nl = tf.math.unsorted_segment_sum(
                a_nl, segment_ids=ind_i, num_segments=batch_tot_nat
            )

            if self.inv_avg_n_neigh is not None:
                inv_avg_n_neigh = tf.cast(self.inv_avg_n_neigh, a_nl.dtype)
                if self.per_specie_n_neigh:
                    if local:
                        raise NotImplementedError()
                    nneigh_norm = tf.gather(
                        inv_avg_n_neigh,
                        input_data[constants.ATOMIC_MU_I],
                        axis=0,
                    )
                    a_nl *= nneigh_norm[:, :, tf.newaxis]
                else:
                    a_nl *= inv_avg_n_neigh
        if self.lm_first:
            a_nl = tf.transpose(a_nl, [2, 0, 1])  # [lm, atoms, n]
        return a_nl

    def _frwrd_equivariant(self, input_data, a_nl, envelope, local=False):
        ind_i = input_data[constants.BOND_IND_I]
        ind_j = input_data[constants.BOND_IND_J]

        # Gather indicator at neighbor sites
        indicator_data = input_data[self.indicator.name]
        # Indicator may be in [lm, atoms, n] layout — convert to [atoms, n, lm]
        # for bond-level operations
        if self.lm_first:
            indicator_data = tf.transpose(indicator_data, [1, 2, 0])
        bond_I = tf.gather(indicator_data, ind_j, axis=0)

        # L=0 injection from chemical_embedding
        z = input_data[self.chemical_embedding.name]
        z = tf.cast(z, bond_I.dtype)
        z_proj = self.chem_linear(z)
        if self.chem_emb_is_per_atom:
            bond_z = tf.gather(z_proj, ind_j, axis=0)
        else:
            mu_j = input_data[constants.BOND_MU_J]
            bond_z = tf.gather(z_proj, mu_j, axis=0)
        chem_l0_mask = tf.cast(self.chem_l0_mask, bond_I.dtype)
        bond_I = (
            bond_I + bond_z[:, :, tf.newaxis] * chem_l0_mask[tf.newaxis, tf.newaxis, :]
        )

        # Apply envelope before 4-tensor product
        a_nl = a_nl * envelope[:, :, tf.newaxis]

        batch_tot_nat = (
            tf.shape(input_data[constants.ATOMIC_MU_I_LOCAL])[0]
            if local
            else tf.shape(input_data[constants.ATOMIC_MU_I])[0]
        )

        # Form the per-atom 4-D product summed over neighbors.
        if self.sum_neighbors:
            if self.dense_nbr:
                prod = _dense_reshape_einsum(a_nl, bond_I, batch_tot_nat)
            else:
                prod = tf.einsum("jnl,jnr->jnlr", a_nl, bond_I)
                prod = tf.math.unsorted_segment_sum(
                    prod,
                    segment_ids=ind_i,
                    num_segments=batch_tot_nat,
                    name=f"sum_nei_{self.name}",
                )
            if self.inv_avg_n_neigh is not None:
                inv_avg_n_neigh = tf.cast(self.inv_avg_n_neigh, dtype=prod.dtype)
                prod *= inv_avg_n_neigh
        else:
            prod = tf.einsum("jnl,jnr->jnlr", a_nl, bond_I)

        # CG coupling: prod is [atoms, n, lm_y, lm_ind]
        return _equiv_cg_couple(
            prod,
            self.lr_inds,
            self.cg,
            self.m_sum_ind,
            self.nfunc,
            self.lm_first,
            self.name,
        )


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
        lm_first: bool = False,
        **kwargs,
    ):
        super().__init__(name=name, lmax=Lmax)
        self.lm_first = lm_first
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
            for l_val in range(Lmax + 1):
                p = 1 if l_val % 2 == 0 else -1
                plist.append([l_val, p])
        else:
            plist = keep_parity
        self.plist = plist
        self.normalize = normalize
        self.max_sum_l = max_sum_l

        # TODO: This must not be done
        # self.lmax = lmax
        input_lmax = lmax
        output_Lmax = Lmax
        self.lmax_A = lmax_left
        self.lmax_B = lmax_right
        self.lmax_hist = lmax_hist
        self.lmax_hist_A = lmax_hist_left
        self.lmax_hist_B = lmax_hist_right
        # self.Lmax = Lmax
        self.coupling_origin = [self.left.name, self.right.name]
        # to be able to re-write it later, otherwise checkpoint will fail
        # if history_drop_list == "DEFAULT":
        #     history_drop_list = EXCLUSION_HIST_LIST
        # self.history_drop_list = NoDependency(history_drop_list or [])
        self.history_drop_list = history_drop_list
        self.init_coupling(input_lmax, output_Lmax)

    def init_coupling(self, input_lmax, output_Lmax):
        self.coupling_meta_data = real_coupling_metainformation(
            A=self.left.coupling_meta_data,
            B=self.right.coupling_meta_data,
            lmax=input_lmax,
            lmax_A=self.lmax_A,
            lmax_B=self.lmax_B,
            lmax_hist=self.lmax_hist,
            lmax_hist_A=self.lmax_hist_A,
            lmax_hist_B=self.lmax_hist_B,
            Lmax=output_Lmax,
            is_A_B_equal=self.is_left_right_equal,
            history_drop_list=self.history_drop_list,
            max_sum_l=self.max_sum_l,
            keep_parity=self.plist,
            normalize=self.normalize,
            legacy_format=True,
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

        # cg broadcasts against the gathered product: [n_cg,1,1] for lm_first
        # ([n_cg, atoms, n]) or [1,1,n_cg] for standard ([atoms, n, n_cg]).
        cg_flat = np.concatenate(cgs)
        self.cg = (
            cg_flat.reshape(-1, 1, 1) if self.lm_first else cg_flat.reshape(1, 1, -1)
        )

        nfunc = np.max(sum_ind) + 1
        self.nfunc = tf.constant(nfunc, dtype=tf.int32)

    @tf.Module.with_name_scope
    def build(self, float_dtype):
        if not self.is_built:
            self.cg = tf.constant(self.cg, dtype=float_dtype)
            self.is_built = True

    def frwrd(self, input_data, training=False, local=False):
        left = input_data[self.left.name]
        right = input_data[self.right.name]

        ax = 0 if self.lm_first else 2
        lft = tf.gather(left, self.left_ind, axis=ax)
        rght = tf.gather(right, self.right_ind, axis=ax)

        prod = lft * rght * self.cg

        if self.lm_first:
            # prod is [n_cg, atoms, n]; reduce m on axis 0 -> [nfunc, atoms, n]
            return tf.math.unsorted_segment_sum(
                prod, self.m_sum_ind, num_segments=self.nfunc
            )
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
        lm_first: bool = False,
    ):
        super().__init__(name=name, lmax=Lmax)
        self.lm_first = lm_first
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
            for l_val in range(Lmax + 1):
                p = 1 if l_val % 2 == 0 else -1
                plist.append([l_val, p])
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
            legacy_format=True,
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

        # cg broadcasts against the gathered product: [n_cg,1,1] for lm_first
        # ([n_cg, atoms, n]) or [1,1,n_cg] for standard ([atoms, n, n_cg]).
        cg_flat = np.concatenate(cgs)
        self.cg = (
            cg_flat.reshape(-1, 1, 1) if self.lm_first else cg_flat.reshape(1, 1, -1)
        )

        nfunc = np.max(sum_ind) + 1
        self.nfunc = tf.constant(nfunc, dtype=tf.int32)

    @tf.Module.with_name_scope
    def build(self, float_dtype):
        if not self.is_built:
            self.cg = tf.constant(self.cg, dtype=float_dtype)
            self.is_built = True

    def _crop(self, x):
        # crop the n_out (feature) axis: axis 2 for lm_first [lm, atoms, n],
        # axis 1 for standard [atoms, n, lm].
        return x[:, :, : self.n_out] if self.lm_first else x[:, : self.n_out, :]

    def frwrd(self, input_data, training=False, local=False):
        left = self._crop(input_data[self.left.name])

        if self.is_left_right_equal:
            right = left
        else:
            right = self._crop(input_data[self.right.name])

        ax = 0 if self.lm_first else 2
        lft = tf.gather(left, self.left_ind, axis=ax)
        rght = tf.gather(right, self.right_ind, axis=ax)
        prod = lft * rght * self.cg

        if self.lm_first:
            # prod is [n_cg, atoms, n]; reduce m on axis 0 -> [nfunc, atoms, n]
            return tf.math.unsorted_segment_sum(
                prod, self.m_sum_ind, num_segments=self.nfunc
            )
        prod = tf.transpose(prod, [2, 0, 1])
        prod = tf.math.unsorted_segment_sum(
            prod, self.m_sum_ind, num_segments=self.nfunc
        )
        return tf.transpose(prod, [1, 2, 0])


@capture_init_args
class GeneralProductFunction(TPEquivariantInstruction):
    """
    Generalized tensor product with optional trainable n-channel coupling via CP decomposition.

    Extends ProductFunction to support full tensor product on n-channels
    (not just element-wise), using low-rank (CP) decomposition to keep
    the parameter count tractable.

    Four modes are supported:

    mode="elementwise":
        Numerically equivalent to ProductFunction. Element-wise product on
        n-channels, CG coupling on angular channels. No trainable parameters.
        Requires left.n_out == right.n_out. The internal coupling table
        differs from ProductFunction's: symmetric (m1, m2) pairs are kept as
        separate rows (optimize_ms_comb=False) because merging them is only
        valid when the product commutes, which does not hold for CP modes.

    mode="cp":
        CP decomposition with global U, V matrices (shared across all angular channels).
        U: [R, n_L], V: [R, n_R], optional S: [n_out, R].

    mode="cp_l":
        CP decomposition where U depends on (l, hist, parity) of the left input channel,
        and V depends on (l, hist, parity) of the right input channel.
        U: [G_L, R, n_L], V: [G_R, R, n_R], optional S: [n_out, R].

    mode="cp_lL":
        Like cp_l but U also depends on output L, and V also depends on output L.
        U: [G_U, R, n_L], V: [G_V, R, n_R], optional S: [n_out, R].

    Parameters
    ----------
    left, right : TPEquivariantInstruction
        Input equivariant tensors of shape [atoms, n_left, l_m] and [atoms, n_right, l_m].
    name : str
        Name of this instruction.
    lmax : int
        Maximum l for input angular channels.
    Lmax : int
        Maximum L for output angular channels.
    mode : str
        One of "elementwise", "cp", "cp_l", "cp_lL".
    n_out : int, optional
        Output n-channel dimension. Required for cp/cp_l/cp_lL modes.
        If None in cp modes, n_out = rank.
    rank : int, optional
        CP decomposition rank R. Required for cp/cp_l/cp_lL modes.
    use_S : bool
        Whether to use the output projection matrix S. If False, rank must equal n_out.
    """

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
        mode: str = "elementwise",
        n_out: int = None,
        rank: int = None,
        use_S: bool = True,
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
        lm_first: bool = False,
        **kwargs,
    ):
        self.lm_first = lm_first
        super().__init__(name=name, lmax=Lmax)
        self.left = left
        self.right = right
        self.mode = mode

        assert self.left.coupling_meta_data is not None
        assert self.right.coupling_meta_data is not None
        if is_left_right_equal is None:
            is_left_right_equal = left.name == right.name
        self.is_left_right_equal = is_left_right_equal

        # Validate mode and set n_out
        assert mode in (
            "elementwise",
            "cp",
            "cp_l",
            "cp_lL",
        ), f"Unknown mode '{mode}', expected one of: elementwise, cp, cp_l, cp_lL"

        if mode == "elementwise":
            assert (
                self.left.n_out == self.right.n_out
            ), "elementwise mode requires left.n_out == right.n_out"
            self.n_out = self.left.n_out
            self.rank = None
            self.use_S = False
        else:
            assert rank is not None, f"rank is required for mode='{mode}'"
            self.rank = rank
            self.use_S = use_S
            if n_out is None:
                n_out = rank
            self.n_out = n_out
            if not use_S:
                assert (
                    rank == n_out
                ), f"When use_S=False, rank must equal n_out, got {rank=} and {n_out=}"

        self.n_left = self.left.n_out
        self.n_right = self.right.n_out

        if keep_parity is None:
            plist = []
            for l_val in range(Lmax + 1):
                p = 1 if l_val % 2 == 0 else -1
                plist.append([l_val, p])
        else:
            plist = keep_parity
        self.plist = plist
        self.normalize = normalize
        self.max_sum_l = max_sum_l

        input_lmax = lmax
        output_Lmax = Lmax
        self.lmax_A = lmax_left
        self.lmax_B = lmax_right
        self.lmax_hist = lmax_hist
        self.lmax_hist_A = lmax_hist_left
        self.lmax_hist_B = lmax_hist_right
        self.coupling_origin = [self.left.name, self.right.name]
        self.history_drop_list = history_drop_list
        self.init_coupling(input_lmax, output_Lmax)

        # Build group assignment tensors for cp_l and cp_lL modes
        if mode in ("cp_l", "cp_lL"):
            self._build_group_assignments()

    def init_coupling(self, input_lmax=None, output_Lmax=None):
        """Initialize CG coupling metadata. Same as ProductFunction."""
        if input_lmax is None:
            input_lmax = self.lmax
        if output_Lmax is None:
            output_Lmax = self.lmax

        self.coupling_meta_data = real_coupling_metainformation(
            A=self.left.coupling_meta_data,
            B=self.right.coupling_meta_data,
            lmax=input_lmax,
            lmax_A=self.lmax_A,
            lmax_B=self.lmax_B,
            lmax_hist=self.lmax_hist,
            lmax_hist_A=self.lmax_hist_A,
            lmax_hist_B=self.lmax_hist_B,
            Lmax=output_Lmax,
            is_A_B_equal=self.is_left_right_equal,
            history_drop_list=self.history_drop_list,
            max_sum_l=self.max_sum_l,
            keep_parity=self.plist,
            normalize=self.normalize,
            optimize_ms_comb=False,
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

        cg_flat = np.concatenate(cgs)
        if self.lm_first:
            self.cg = cg_flat.reshape(-1, 1, 1)
        else:
            self.cg = cg_flat.reshape(1, 1, -1)

        nfunc = np.max(sum_ind) + 1
        self.nfunc = tf.constant(nfunc, dtype=tf.int32)

    def _build_group_assignments(self):
        """Build group index mappings for cp_l and cp_lL modes.

        For cp_l:
            - group_left: [l_m_left] -> group index based on (l, hist, parity) of left input
            - group_right: [l_m_right] -> group index based on (l, hist, parity) of right input

        For cp_lL:
            - cg_u_group: [n_cg] -> group index based on (l1, hist1, parity1, L) per CG pair
            - cg_v_group: [n_cg] -> group index based on (l2, hist2, parity2, L) per CG pair
        """
        left_cmd = self.left.coupling_meta_data
        right_cmd = self.right.coupling_meta_data

        # Build left input channel -> (l, hist, parity) group mapping
        left_groups = left_cmd.groupby(["l", "hist", "parity"]).indices
        left_group_keys = list(left_groups.keys())
        left_group_map = {}  # channel_index -> group_id
        for gid, (key, indices) in enumerate(left_groups.items()):
            for idx in indices:
                left_group_map[idx] = gid

        # Build right input channel -> (l, hist, parity) group mapping
        right_groups = right_cmd.groupby(["l", "hist", "parity"]).indices
        right_group_keys = list(right_groups.keys())
        right_group_map = {}
        for gid, (key, indices) in enumerate(right_groups.items()):
            for idx in indices:
                right_group_map[idx] = gid

        if self.mode == "cp_l":
            # For cp_l: assign each input angular channel to its (l, hist, parity) group
            n_lm_left = len(left_cmd)
            n_lm_right = len(right_cmd)
            group_left = np.array(
                [left_group_map[i] for i in range(n_lm_left)], dtype=np.int32
            )
            group_right = np.array(
                [right_group_map[i] for i in range(n_lm_right)], dtype=np.int32
            )
            self.group_left = tf.constant(group_left, dtype=tf.int32)
            self.group_right = tf.constant(group_right, dtype=tf.int32)
            self.n_groups_left = len(left_group_keys)
            self.n_groups_right = len(right_group_keys)

        elif self.mode == "cp_lL":
            # For cp_lL: assign each CG pair to its (l, hist, parity, L) group
            # We need to walk through coupling_meta_data to find what (l1, hist1, p1, L)
            # each CG pair belongs to
            left_inds_list = self.coupling_meta_data["left_inds"].values
            right_inds_list = self.coupling_meta_data["right_inds"].values
            output_L_list = self.coupling_meta_data["l"].values

            u_group_keys = []  # unique (l1, hist1, p1, L) keys
            u_group_key_to_id = {}
            v_group_keys = []
            v_group_key_to_id = {}

            cg_u_groups = []
            cg_v_groups = []

            for row_idx in range(len(self.coupling_meta_data)):
                l_inds = left_inds_list[row_idx]
                r_inds = right_inds_list[row_idx]
                out_L = output_L_list[row_idx]

                for li, ri in zip(l_inds, r_inds):
                    # Left: (l1, hist1, parity1) group + output L
                    left_base_gid = left_group_map[li]
                    u_key = (left_group_keys[left_base_gid], out_L)
                    if u_key not in u_group_key_to_id:
                        u_group_key_to_id[u_key] = len(u_group_keys)
                        u_group_keys.append(u_key)
                    cg_u_groups.append(u_group_key_to_id[u_key])

                    # Right: (l2, hist2, parity2) group + output L
                    right_base_gid = right_group_map[ri]
                    v_key = (right_group_keys[right_base_gid], out_L)
                    if v_key not in v_group_key_to_id:
                        v_group_key_to_id[v_key] = len(v_group_keys)
                        v_group_keys.append(v_key)
                    cg_v_groups.append(v_group_key_to_id[v_key])

            self.cg_u_group = tf.constant(cg_u_groups, dtype=tf.int32)
            self.cg_v_group = tf.constant(cg_v_groups, dtype=tf.int32)
            self.n_groups_u = len(u_group_keys)
            self.n_groups_v = len(v_group_keys)

            # The per-CG-pair weight depends on
            # the pair only via (group, lm-channel), so project on the D distinct
            # (group, lm) pairs then gather D->n_cg. cg_u_groups[c] aligns with
            # left_inds flattened in the same row-major order (both walk the
            # coupling rows in order), so they pair element-wise.
            left_flat = np.concatenate(self.coupling_meta_data["left_inds"]).astype(
                np.int64
            )
            right_flat = np.concatenate(self.coupling_meta_data["right_inds"]).astype(
                np.int64
            )
            ug = np.asarray(cg_u_groups, dtype=np.int64)
            vg = np.asarray(cg_v_groups, dtype=np.int64)
            uniq_u, dinv_u = np.unique(
                np.stack([ug, left_flat], 1), axis=0, return_inverse=True
            )
            uniq_v, dinv_v = np.unique(
                np.stack([vg, right_flat], 1), axis=0, return_inverse=True
            )
            self.cpll_pg_u = tf.constant(uniq_u[:, 0].astype(np.int32))
            self.cpll_pc_u = tf.constant(uniq_u[:, 1].astype(np.int32))
            self.cpll_dinv_u = tf.constant(
                np.asarray(dinv_u).reshape(-1).astype(np.int32)
            )
            self.cpll_pg_v = tf.constant(uniq_v[:, 0].astype(np.int32))
            self.cpll_pc_v = tf.constant(uniq_v[:, 1].astype(np.int32))
            self.cpll_dinv_v = tf.constant(
                np.asarray(dinv_v).reshape(-1).astype(np.int32)
            )

    @tf.Module.with_name_scope
    def build(self, float_dtype):
        if not self.is_built:
            self.cg = tf.constant(self.cg, dtype=float_dtype)

            # Runtime normalization factors (1/sqrt of contraction dim)
            self.norm_u = tf.constant(1.0 / np.sqrt(self.n_left), dtype=float_dtype)
            self.norm_v = tf.constant(1.0 / np.sqrt(self.n_right), dtype=float_dtype)
            if self.use_S:
                self.norm_s = tf.constant(1.0 / np.sqrt(self.rank), dtype=float_dtype)

            if self.mode == "cp":
                self.U = tf.Variable(
                    tf.random.normal([self.rank, self.n_left], dtype=float_dtype),
                    name="U_cp",
                )
                self.V = tf.Variable(
                    tf.random.normal([self.rank, self.n_right], dtype=float_dtype),
                    name="V_cp",
                )
                if self.use_S:
                    self.S = tf.Variable(
                        tf.random.normal(
                            [self.n_out, self.rank],
                            dtype=float_dtype,
                        ),
                        name="S_cp",
                    )

            elif self.mode == "cp_l":
                self.U = tf.Variable(
                    tf.random.normal(
                        [self.n_groups_left, self.rank, self.n_left],
                        dtype=float_dtype,
                    ),
                    name="U_cp_l",
                )
                self.V = tf.Variable(
                    tf.random.normal(
                        [self.n_groups_right, self.rank, self.n_right],
                        dtype=float_dtype,
                    ),
                    name="V_cp_l",
                )
                if self.use_S:
                    self.S = tf.Variable(
                        tf.random.normal(
                            [self.n_out, self.rank],
                            dtype=float_dtype,
                        ),
                        name="S_cp_l",
                    )

            elif self.mode == "cp_lL":
                self.U = tf.Variable(
                    tf.random.normal(
                        [self.n_groups_u, self.rank, self.n_left],
                        dtype=float_dtype,
                    ),
                    name="U_cp_lL",
                )
                self.V = tf.Variable(
                    tf.random.normal(
                        [self.n_groups_v, self.rank, self.n_right],
                        dtype=float_dtype,
                    ),
                    name="V_cp_lL",
                )
                if self.use_S:
                    self.S = tf.Variable(
                        tf.random.normal(
                            [self.n_out, self.rank],
                            dtype=float_dtype,
                        ),
                        name="S_cp_lL",
                    )

            self.is_built = True

    def _cg_coupling(self, lft, rght):
        """Apply CG coupling: element-wise product, CG weighting, and reduction.

        Args:
            lft: gathered left tensor, shape depends on layout:
                standard: [atoms, n, n_cg]
                lm_first: [n_cg, atoms, n]
            rght: gathered right tensor, same shape as lft

        Returns:
            Coupled tensor:
                standard: [atoms, n, n_output_lm]
                lm_first: [n_output_lm, atoms, n]
        """
        prod = lft * rght * tf.cast(self.cg, lft.dtype)
        if self.lm_first:
            # prod is [n_cg, atoms, n], segment_sum on axis 0
            return tf.math.unsorted_segment_sum(
                prod, self.m_sum_ind, num_segments=self.nfunc
            )
        else:
            prod = tf.transpose(prod, [2, 0, 1])
            prod = tf.math.unsorted_segment_sum(
                prod, self.m_sum_ind, num_segments=self.nfunc
            )
            return tf.transpose(prod, [1, 2, 0])

    def _gather_axis(self):
        """Return the axis for angular gathers: 0 for lm_first, 2 for standard."""
        return 0 if self.lm_first else 2

    def _apply_S(self, prod):
        """Optional output projection S [n_out, R] over the rank axis."""
        if not self.use_S:
            return prod
        eq = "kr,war->wak" if self.lm_first else "kr,arL->akL"
        return tf.einsum(eq, self.S, prod) * self.norm_s

    def _frwrd_elementwise(self, left, right):
        """Mode 'elementwise': identical to ProductFunction."""
        ax = self._gather_axis()
        lft = tf.gather(left, self.left_ind, axis=ax)
        rght = tf.gather(right, self.right_ind, axis=ax)
        return self._cg_coupling(lft, rght)

    def _frwrd_cp(self, left, right):
        """Mode 'cp': global CP decomposition."""
        U = self.U
        V = self.V
        if self.lm_first:
            # left: [lm, atoms, n_L] → project: [lm, atoms, R]
            left_proj = tf.einsum("rn,wan->war", U, left) * self.norm_u
            right_proj = tf.einsum("rn,wan->war", V, right) * self.norm_v
        else:
            left_proj = tf.einsum("rn,anl->arl", U, left) * self.norm_u
            right_proj = tf.einsum("rn,anl->arl", V, right) * self.norm_v

        ax = self._gather_axis()
        lft = tf.gather(left_proj, self.left_ind, axis=ax)
        rght = tf.gather(right_proj, self.right_ind, axis=ax)
        prod = self._cg_coupling(lft, rght)
        return self._apply_S(prod)

    def _frwrd_cp_l(self, left, right):
        """Mode 'cp_l': CP with U depending on (l1, hist1), V on (l2, hist2)."""
        U_tiled = tf.gather(self.U, self.group_left, axis=0)  # [lm, R, n]
        V_tiled = tf.gather(self.V, self.group_right, axis=0)

        if self.lm_first:
            left_proj = tf.einsum("wrn,wan->war", U_tiled, left) * self.norm_u
            right_proj = tf.einsum("wrn,wan->war", V_tiled, right) * self.norm_v
        else:
            left_proj = tf.einsum("wrn,anw->arw", U_tiled, left) * self.norm_u
            right_proj = tf.einsum("wrn,anw->arw", V_tiled, right) * self.norm_v

        ax = self._gather_axis()
        lft = tf.gather(left_proj, self.left_ind, axis=ax)
        rght = tf.gather(right_proj, self.right_ind, axis=ax)
        prod = self._cg_coupling(lft, rght)
        return self._apply_S(prod)

    def _frwrd_cp_lL(self, left, right):
        """Mode 'cp_lL': CP with U depending on (l1, hist1, L), V on (l2, hist2, L)."""
        # Project on the D distinct (group, lm) channels, then expand D -> n_cg.
        # ax = lm/coupling axis (0 lm_first, 2 standard); subscript picks the layout.
        ax = self._gather_axis()
        eq = "drn,dan->dar" if self.lm_first else "drn,and->ard"

        def project(x, pc, W, pg, dinv, norm):
            xd = tf.gather(x, pc, axis=ax)  # lm -> D distinct channels
            Wd = tf.gather(W, pg, axis=0)  # [D, R, n]
            proj = tf.einsum(eq, Wd, xd) * norm
            return tf.gather(proj, dinv, axis=ax)  # expand D -> n_cg

        lft_proj = project(
            left, self.cpll_pc_u, self.U, self.cpll_pg_u, self.cpll_dinv_u, self.norm_u
        )
        rght_proj = project(
            right, self.cpll_pc_v, self.V, self.cpll_pg_v, self.cpll_dinv_v, self.norm_v
        )
        prod = self._cg_coupling(lft_proj, rght_proj)
        return self._apply_S(prod)

    def frwrd(self, input_data, training=False, local=False):
        left = input_data[self.left.name]
        right = input_data[self.right.name]

        if self.mode == "elementwise":
            return self._frwrd_elementwise(left, right)
        elif self.mode == "cp":
            return self._frwrd_cp(left, right)
        elif self.mode == "cp_l":
            return self._frwrd_cp_l(left, right)
        elif self.mode == "cp_lL":
            return self._frwrd_cp_lL(left, right)

    def __repr__(self):
        return (
            f"GeneralProductFunction(name={self.name}, "
            f"origin={' x '.join(self.coupling_origin)}, "
            f"l_max={self.lmax}, mode={self.mode})"
        )


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
        lm_first: bool = False,
        **kwargs,
    ):
        super().__init__(name=name, lmax=np.max(ls_max))
        self.lm_first = lm_first
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
            for l_idx in range(self.lmax + 1):
                if [l_idx, p] in self.allowed_l_p:
                    for m in range(-l_idx, l_idx + 1):
                        # TODO: rethink how to define history here. Then, possibly move to the base class method
                        collector_data.append([l_idx, m, "", p, l_idx])
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

    def frwrd(self, input_data, training=False, local=False):
        if self.init_collection == "zeros":
            init_func = tf.zeros
        else:
            init_func = tf.ones
        # lm_first: equivariant inputs are [lm, atoms, n] (gather lm on axis 0,
        # einsum reads A_r as "wan"); standard: [atoms, n, lm] (axis 2, "anw").
        gather_ax = 0 if self.lm_first else 2
        collection = None
        for instr in self.instructions:
            instruction_collection = self.collector[instr.name]
            A_r = tf.gather(
                input_data[instr.name],
                instruction_collection["func_collect_ind"],
                axis=gather_ax,
            )
            w = getattr(self, f"reducing_{instr.name}")
            if A_r.dtype != w.dtype:
                A_r = tf.cast(A_r, w.dtype)
            if collection is None:
                collection = init_func(
                    [
                        self.coupling_meta_data.shape[0],
                        input_data[constants.N_ATOMS_BATCH_TOTAL],
                        self.n_out,
                    ],
                    dtype=w.dtype,
                )
            if self.is_central_atom_type_dependent:
                eq = "aknw,wan->wak" if self.lm_first else "aknw,anw->wak"
            else:
                eq = "knw,wan->wak" if self.lm_first else "knw,anw->wak"
            w = tf.gather(w, instruction_collection["w_l_tile"], axis=-1)
            # For performance
            if self.is_central_atom_type_dependent:
                w = tf.gather(w, input_data[constants.ATOMIC_MU_I], axis=0)

            norm = getattr(self, f"norm_{instr.name}")
            pr = tf.einsum(eq, w, A_r, name=f"ein_{instr.name}") * norm

            collection = tf.tensor_scatter_nd_add(
                collection, instruction_collection["total_sum_ind"], pr
            )

        # collection is [n_cg_out, atoms, n_out] (already lm-first); standard
        # consumers want [atoms, n_out, lm].
        if self.lm_first:
            return collection  # [lm, atoms, n_out]  # * self.n_instr
        return tf.transpose(collection, [1, 2, 0])  # * self.n_instr

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
        lm_first: bool = False,
        **kwargs,
    ):
        self.lm_first = lm_first
        super().__init__(name=name, lmax=np.max(ls_max))
        LORAInstructionMixin.__init__(self, lora_config)  # explicitly call
        self.instructions = instructions
        if isinstance(ls_max, int):
            ls_max = [ls_max] * len(instructions)
        self.ls_max = ls_max
        self.only_invar = False
        if np.all(np.array(ls_max) == 0):
            self.only_invar = True
        self.n_out = n_out
        self.out_norm = out_norm
        # enforce conversion to list of lists
        self.allowed_l_p = [list(lp) for lp in allowed_l_p]
        self.plist = self.allowed_l_p
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
            for l_idx in range(self.lmax + 1):
                if [l_idx, p] in self.allowed_l_p:
                    for m in range(-l_idx, l_idx + 1):
                        # TODO:  possibly move to the base class method
                        collector_data.append([l_idx, m, "", p, l_idx])
                        # collector_data.append([l_idx, m, f"({lbl},0)", p, l_idx])
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

    def frwrd(self, input_data, training=False, local=False):
        init_func = tf.zeros if self.init_collection == "zeros" else tf.ones
        collection = None

        atomic_mu_i = (
            input_data[constants.ATOMIC_MU_I_LOCAL]
            if local
            else input_data[constants.ATOMIC_MU_I]
        )

        # Determine gather axis and einsum based on layout
        if self.lm_first:
            gather_ax = 0
            eq_base = "knw,wan->wak"
            eq_elem_base = "aknw,wan->wak"
        else:
            gather_ax = 2
            eq_base = "knw,anw->wak"
            eq_elem_base = "aknw,anw->wak"

        for instr in self.instructions:
            instruction_collection = self.collector[instr.name]
            A_r = tf.gather(
                input_data[instr.name],
                instruction_collection["func_collect_ind"],
                axis=gather_ax,
            )
            w = getattr(self, f"reducing_{instr.name}")

            A_r = tf.cast(A_r, w.dtype)
            if collection is None:
                if self.only_invar:
                    collect_shape = [
                        1,
                        tf.shape(atomic_mu_i)[0],
                        self.n_out,
                    ]
                else:
                    collect_shape = [
                        self.coupling_meta_data.shape[0],
                        tf.shape(atomic_mu_i)[0],
                        self.n_out,
                    ]
                collection = init_func(
                    collect_shape,
                    dtype=A_r.dtype,
                )
            # lora
            if self.lora:
                lora_tensors = getattr(self, f"reducing_{instr.name}_lora_tensors")
                w = w + lora_reconstruction(*lora_tensors, lora_config=self.lora_config)

            w = tf.gather(w, instruction_collection["w_l_tile"], axis=-1)
            if self.is_central_atom_type_dependent:
                eq = eq_elem_base
                w = tf.gather(w, atomic_mu_i, axis=0)
                if not self.lm_first:
                    A_r = A_r[: tf.shape(atomic_mu_i)[0]]
                else:
                    A_r = A_r[:, : tf.shape(atomic_mu_i)[0], :]
            else:
                eq = eq_base

            norm = getattr(self, f"norm_{instr.name}")
            pr = tf.einsum(eq, w, A_r, name=f"ein_{instr.name}") * norm

            if self.only_invar:
                collection += tf.reduce_sum(pr, axis=0, keepdims=True)
            else:
                collection = tf.tensor_scatter_nd_add(
                    collection, instruction_collection["total_sum_ind"], pr
                )
        collection *= self.norm_map

        # lm_first consumers get [lm, atoms, n_out] regardless of only_invar.
        # Scalar readout consumers (InvariantLayerRMSNorm, LinMLPOut2ScalarTarget)
        # must detect lm_first on the upstream and transpose themselves.
        if self.lm_first:
            return collection  # [lm, atoms, n_out]
        return tf.transpose(collection, [1, 2, 0])

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
        lm_first: bool = False,
        # n_out: int,
        # allowed_l_p: list[list],
        # out_norm: bool = False,
        # is_central_atom_type_dependent: bool = False,
        # number_of_atom_types: int = None,
        # init_vars: Literal["random", "zeros"] = "random",
        # scale=1.0,
    ):
        super().__init__(name=name, lmax=np.max(ls_max))
        self.lm_first = lm_first
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
            for l_idx in range(self.lmax + 1):
                if [l_idx, p] in self.allowed_l_p:
                    for m in range(-l_idx, l_idx + 1):
                        collector_data.append([l_idx, m, "", p, l_idx])
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

    def frwrd(self, input_data, training=False, local=False):
        # lm_first inputs are [lm, atoms, n]; gather lm on axis 0, then transpose
        # to [atoms, n, lm] so the flatten order matches the standard layout.
        gather_ax = 0 if self.lm_first else 2
        collection = []
        for instr in self.instructions:
            instruction_collection = self.collector[instr.name]
            A = tf.gather(
                input_data[instr.name],
                instruction_collection["func_collect_ind"],
                axis=gather_ax,
            )
            if self.lm_first:
                A = tf.transpose(
                    A, [1, 2, 0]
                )  # [collected, atoms, n]->[atoms, n, collected]
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
        constants.ATOMIC_MU_I: {"shape": [None], "dtype": "int"},
    }

    def __init__(
        self,
        left: TPEquivariantInstruction,
        right: TPEquivariantInstruction,
        name: str,
        n_out: int = None,
        left_coefs: bool = True,
        is_central_atom_type_dependent: list[bool] | bool = None,
        number_of_atom_types: int = None,
        init_vars: str = "random",
        normalize: bool = True,
        norm_out: bool = False,
        lora_config: dict[str, Any] = None,
        lm_first: bool = False,
        **kwargs,
    ):
        self.lm_first = lm_first
        super().__init__(name=name, lmax=left.lmax)
        LORAInstructionMixin.__init__(self, lora_config)  # explicitly call

        self.left = left
        self.right = right
        self.left_coefs = left_coefs
        if n_out is not None:
            self.n_out = n_out
        else:
            self.n_out = self.left.n_out
        if not self.left_coefs:
            assert left.n_out == self.n_out

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
        if any(self.is_central_atom_type_dependent):
            assert number_of_atom_types is not None, (
                "number_of_atom_types cannot be None"
                " if is_central_atom_type_dependent is True"
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
                        name="w_left_FC",
                    )
                elif self.init_vars == "uniform":
                    self.w_left = tf.Variable(
                        tf.random.uniform(
                            minval=-init_value,
                            maxval=init_value,
                            shape=c_shape_left,
                            dtype=float_dtype,
                        ),
                        name="w_left_FC",
                    )
                elif self.init_vars == "zeros":
                    self.w_left = tf.Variable(
                        tf.zeros(shape=c_shape_left, dtype=float_dtype),
                        name="w_left_FC",
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
                    name="w_right_FC",
                )
            elif self.init_vars == "uniform":
                self.w_right = tf.Variable(
                    tf.random.uniform(
                        minval=-init_value,
                        maxval=init_value,
                        shape=c_shape_right,
                        dtype=float_dtype,
                    ),
                    name="w_right_FC",
                )
            elif self.init_vars == "zeros":
                self.w_right = tf.Variable(
                    tf.random.normal(
                        c_shape_right, stddev=init_value, dtype=float_dtype
                    ),
                    name="w_right_FC",
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

    def frwrd(self, input_data, training=False, local=False):
        left = input_data[self.left.name]

        atomic_mu_i = (
            input_data[constants.ATOMIC_MU_I_LOCAL]
            if local
            else input_data[constants.ATOMIC_MU_I]
        )

        if self.lm_first:
            eq = "knw,wan->wak"
            eq_elem = "aknw,wan->wak"
            gather_ax = 0  # angular is axis 0
        else:
            eq = self.eq  # "knw,anw->wak"
            eq_elem = self.eq_elem  # "aknw,anw->wak"
            gather_ax = -1  # angular is last axis

        if self.left_coefs:
            w_left = self.w_left
            # LORA
            if self.lora:
                w_left = w_left + lora_reconstruction(
                    *self.w_left_lora_tensors, lora_config=self.lora_config
                )
            w_left = tf.gather(w_left, self.w_tile_left, axis=-1)
            if self.is_central_atom_type_dependent[0]:
                w_left = tf.gather(w_left, atomic_mu_i, axis=0)
                left = (
                    tf.einsum(eq_elem, w_left, left, name="ein_left") * self.norm_left
                )
            else:
                left = tf.einsum(eq, w_left, left, name="ein_left") * self.norm_left
        else:
            if not self.lm_first:
                left = tf.transpose(left, [2, 0, 1])
            # lm_first: left is already [lm, atoms, n], no transpose needed

        right = tf.gather(
            input_data[self.right.name], self.collect_from, axis=gather_ax
        )
        w_right = self.w_right
        # LORA
        if self.lora:
            w_right = w_right + lora_reconstruction(
                *self.w_right_lora_tensors, lora_config=self.lora_config
            )
        w_right = tf.gather(w_right, self.w_tile_right, axis=-1)
        if self.is_central_atom_type_dependent[1]:
            w_right = tf.gather(w_right, atomic_mu_i, axis=0)
            right = (
                tf.einsum(eq_elem, w_right, right, name="ein_right") * self.norm_right
            )
        else:
            right = tf.einsum(eq, w_right, right, name="ein_right") * self.norm_right

        left = tf.tensor_scatter_nd_add(
            left, tf.reshape(self.collect_to, [-1, 1]), right, name="add_right_to_left"
        )
        if self.norm_out:
            left *= self.norm_out_factor

        if self.lm_first:
            return left  # already [lm, atoms, n_out]
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
    }

    def __init__(
        self,
        inpt: TPInstruction,
        name: str,
        type: str = "only_nonlin",
        init: str = "zeros",
        **kwargs,
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

    def frwrd(self, input_data: dict, training: bool = False, local: bool = False):
        x = input_data[self.input.name]
        if getattr(self.input, "lm_first", False):
            # [lm, atoms, n_out] -> [atoms, n_out, lm]
            x = tf.transpose(x, [1, 2, 0])

        n_at_b_total = (
            tf.shape(input_data[constants.ATOMIC_MU_I_LOCAL])[0]
            if local
            else tf.shape(input_data[constants.ATOMIC_MU_I])[0]
        )
        n_at_b_real = input_data[constants.N_ATOMS_BATCH_REAL]

        r_map = tf.reshape(
            tf.range(n_at_b_total, delta=1, dtype=tf.int32, name="r_map"), [-1, 1, 1]
        )
        if self.type == "full":
            epsilon = tf.cast(self.epsilon, x.dtype)
            rms = tf.math.rsqrt(tf.reduce_mean(x**2, axis=1, keepdims=True) + epsilon)
            rms = tf.where(r_map < n_at_b_real, rms, tf.zeros_like(rms))
            return x * rms * tf.cast(self.scale, x.dtype)
        elif self.type == "only_nonlin":
            lin = x[:, 0, :]
        elif self.type == "sep_lin_gate":
            lin = x[:, 0, :] * tf.cast(self.lin_scale, x.dtype)
        nonlin = x[:, 1:, :]
        nl_rms = tf.math.rsqrt(
            tf.reduce_mean(nonlin**2, axis=1, keepdims=True)
            + tf.cast(self.epsilon, x.dtype)
        )
        nl_rms = tf.where(
            r_map < n_at_b_real,
            nonlin * nl_rms * tf.cast(self.scale, x.dtype),
            tf.zeros_like(nl_rms),
        )
        return tf.concat([lin[:, tf.newaxis, :], nl_rms], axis=1)


@capture_init_args
class EquivariantRMSNorm(TPEquivariantInstruction):
    """
    Equivariant RMS normalization with degree-balanced weighting.

    Computes a single per-atom RMS across all angular channels, weighting each
    degree l equally regardless of its (2l+1) multiplicity or number of history
    groups. Applies per-(l, parity, hist) learnable scale.

    Preserves equivariance because the degree-balanced sum of squares is
    rotationally invariant, and the per-(l, parity, hist) scale is shared
    across all m-components within each group.

    Parameters
    ----------
    input : TPEquivariantInstruction
        The equivariant instruction to normalize.
    name : str
        Name of this instruction.
    center_l0 : bool
        If True, subtract the feature-mean from L=0 channels before computing
        the norm (mean over the n_features axis, per angular position).
    center_l0_bias : bool
        If True and center_l0 is True, add a learnable per-channel bias to L=0
        features after normalization. Off by default: a bias shifts all atoms'
        L=0 features uniformly, which can corrupt meaningfully-zero features.
    balance_degrees : bool
        If True, weight each degree l equally in the norm computation
        (each m-component gets weight 1 / (count_of_channels_with_same_l * n_degrees)).
        If False, simple mean over all angular channels.
    normalize_l0_only : bool
        If True, compute RMS only from l=0 channels and apply normalization
        only to l=0 features, leaving higher-l channels unchanged.
    split_norm : bool
        If True, compute separate RMS for l=0 and l>0 channels, so each group
        is normalized by its own scale. Preserves equivariance (both norms are
        rotationally invariant). Off by default to keep a single global norm.
    init : str
        Initialization for affine scale: "zeros" or "ones".
    """

    input_tensor_spec = {
        constants.N_ATOMS_BATCH_REAL: {"shape": [], "dtype": "int"},
    }

    def __init__(
        self,
        input: TPEquivariantInstruction,
        name: str,
        center_l0: bool = False,
        center_l0_bias: bool = False,
        balance_degrees: bool = True,
        normalize_l0_only: bool = False,
        split_norm: bool = False,
        init: str = "zeros",
        lm_first: bool = False,
        **kwargs,
    ):
        self.lm_first = lm_first
        super().__init__(name=name, lmax=input.lmax)
        self.input = input
        self.n_out = input.n_out
        self.coupling_meta_data = input.coupling_meta_data.copy()
        if hasattr(input, "coupling_origin"):
            self.coupling_origin = input.coupling_origin

        self.center_l0 = center_l0
        self.center_l0_bias = center_l0_bias
        self.balance_degrees = balance_degrees
        self.normalize_l0_only = normalize_l0_only
        self.split_norm = split_norm
        assert init in ["zeros", "ones"]
        self.init = init

        # Precompute degree-balanced weights from coupling_meta_data
        l_values = self.coupling_meta_data["l"].values
        n_angular = len(l_values)

        if self.balance_degrees:
            unique_ls = np.unique(l_values)
            n_degrees = len(unique_ls)
            weights = np.zeros(n_angular, dtype=np.float64)
            for l_val in unique_ls:
                mask = l_values == l_val
                count = mask.sum()
                weights[mask] = 1.0 / (count * n_degrees)
        else:
            weights = np.ones(n_angular, dtype=np.float64) / n_angular
        self._degree_weights_np = weights

        # Build (l, parity, hist) group index for affine weight
        lph_groups = self.coupling_meta_data.groupby(["l", "parity", "hist"]).indices
        lph_keys = list(lph_groups.keys())
        self.n_lph_groups = len(lph_keys)
        expand_idx = np.zeros(n_angular, dtype=np.int32)
        for gid, (key, indices) in enumerate(lph_groups.items()):
            for idx in indices:
                expand_idx[idx] = gid
        self._expand_index_np = expand_idx

        # Split norm: separate weights for l=0 and l>0
        if self.split_norm:
            l0_mask = l_values == 0
            lgt0_mask = ~l0_mask
            # l=0 weights: uniform (all l=0 components are m=0 scalars)
            n_l0 = l0_mask.sum()
            n_lgt0 = lgt0_mask.sum()
            self._split_l0_weights_np = np.zeros(n_angular, dtype=np.float64)
            self._split_lgt0_weights_np = np.zeros(n_angular, dtype=np.float64)
            if n_l0 > 0:
                self._split_l0_weights_np[l0_mask] = 1.0 / n_l0
            if n_lgt0 > 0:
                if self.balance_degrees:
                    # Balance within l>0 channels only
                    unique_lgt0 = np.unique(l_values[lgt0_mask])
                    n_deg_gt0 = len(unique_lgt0)
                    for l_val in unique_lgt0:
                        mask = l_values == l_val
                        count = mask.sum()
                        self._split_lgt0_weights_np[mask] = 1.0 / (count * n_deg_gt0)
                else:
                    self._split_lgt0_weights_np[lgt0_mask] = 1.0 / n_lgt0
            self._split_l0_mask_np = l0_mask
            self._split_lgt0_mask_np = lgt0_mask

        # L=0 mask (used by center_l0 and normalize_l0_only)
        if self.center_l0 or self.normalize_l0_only:
            l0_mask = l_values == 0
            self._l0_mask_np = l0_mask

        # Precompute l0-only weights for norm computation
        if self.normalize_l0_only:
            l0_mask = l_values == 0
            n_l0 = l0_mask.sum()
            self._l0_norm_weights_np = np.zeros(n_angular, dtype=np.float64)
            self._l0_norm_weights_np[l0_mask] = 1.0 / n_l0
            # Non-l0 mask for passthrough
            self._non_l0_mask_np = ~l0_mask

    @tf.Module.with_name_scope
    def build(self, float_dtype):
        if not self.is_built:
            if self.lm_first:
                dw_shape = (-1, 1, 1)
            else:
                dw_shape = (1, 1, -1)
            self.degree_weights = tf.constant(
                self._degree_weights_np.reshape(dw_shape), dtype=float_dtype
            )
            self.expand_index = tf.constant(self._expand_index_np, dtype=tf.int32)
            eps_val = 1e-8 if float_dtype == tf.float32 else 1e-12
            self.epsilon = tf.constant(eps_val, dtype=float_dtype)

            # Per-(l, parity, hist) affine scale
            shape = [self.n_lph_groups, self.n_out]
            if self.init == "zeros":
                self.affine_weight = tf.Variable(
                    tf.zeros(shape, dtype=float_dtype), name="affine_weight"
                )
            elif self.init == "ones":
                self.affine_weight = tf.Variable(
                    tf.ones(shape, dtype=float_dtype), name="affine_weight"
                )

            # Split norm tensors
            if self.split_norm:
                self.split_l0_weights = tf.constant(
                    self._split_l0_weights_np.reshape(dw_shape), dtype=float_dtype
                )
                self.split_lgt0_weights = tf.constant(
                    self._split_lgt0_weights_np.reshape(dw_shape), dtype=float_dtype
                )
                self.split_l0_mask = tf.constant(
                    self._split_l0_mask_np.reshape(dw_shape)
                )
                self.split_lgt0_mask = tf.constant(
                    self._split_lgt0_mask_np.reshape(dw_shape)
                )

            # L=0 mask (shared by center_l0 and normalize_l0_only)
            if self.center_l0 or self.normalize_l0_only:
                self.l0_mask_f = tf.constant(
                    self._l0_mask_np.reshape(dw_shape).astype(np.float64),
                    dtype=float_dtype,
                )

            # L=0-only normalization weights and masks
            if self.normalize_l0_only:
                self.l0_norm_weights = tf.constant(
                    self._l0_norm_weights_np.reshape(dw_shape), dtype=float_dtype
                )
                self.non_l0_mask = tf.constant(self._non_l0_mask_np.reshape(dw_shape))

            # L=0 centering bias (optional)
            # Variable shape always [1, n_out, n_angular] for checkpoint compat;
            # transposed at runtime when lm_first=True.
            if self.center_l0 and self.center_l0_bias:
                n_angular = len(self._expand_index_np)
                self.l0_bias = tf.Variable(
                    tf.zeros([1, self.n_out, n_angular], dtype=float_dtype),
                    name="l0_bias",
                )

            self.is_built = True

    def frwrd(self, input_data, training=False, local=False):
        x = input_data[self.input.name]
        # x shape: [atoms, n_features, n_angular] (standard)
        #      or: [n_angular, atoms, n_features] (lm_first)

        n_at_b_total = (
            tf.shape(input_data[constants.ATOMIC_MU_I_LOCAL])[0]
            if local
            else tf.shape(input_data[constants.ATOMIC_MU_I])[0]
        )
        n_at_b_real = input_data[constants.N_ATOMS_BATCH_REAL]

        # Axis assignments based on layout
        if self.lm_first:
            angular_ax = 0
            feat_ax = 2
        else:
            angular_ax = 2
            feat_ax = 1

        # 1. Optional L=0 centering
        if self.center_l0:
            l0_mask_f = tf.cast(self.l0_mask_f, x.dtype)
            l0_mean = tf.reduce_mean(x, axis=feat_ax, keepdims=True) * l0_mask_f
            x = x - l0_mean

        # 2. Degree-balanced RMS
        x_sq = x**2
        eps = tf.cast(self.epsilon, x.dtype)

        if self.normalize_l0_only:
            l0_weights = tf.cast(self.l0_norm_weights, x.dtype)
            weighted = x_sq * l0_weights
            norm = tf.reduce_sum(weighted, axis=angular_ax, keepdims=True)
            norm = tf.reduce_mean(norm, axis=feat_ax, keepdims=True)
            rms = tf.math.rsqrt(norm + eps)
        elif self.split_norm:
            l0_w = tf.cast(self.split_l0_weights, x.dtype)
            lgt0_w = tf.cast(self.split_lgt0_weights, x.dtype)
            norm_l0 = tf.reduce_sum(x_sq * l0_w, axis=angular_ax, keepdims=True)
            norm_l0 = tf.reduce_mean(norm_l0, axis=feat_ax, keepdims=True)
            norm_lgt0 = tf.reduce_sum(x_sq * lgt0_w, axis=angular_ax, keepdims=True)
            norm_lgt0 = tf.reduce_mean(norm_lgt0, axis=feat_ax, keepdims=True)
            rms_l0 = tf.math.rsqrt(norm_l0 + eps)
            rms_lgt0 = tf.math.rsqrt(norm_lgt0 + eps)
            l0_m = tf.cast(self.split_l0_mask, x.dtype)
            lgt0_m = tf.cast(self.split_lgt0_mask, x.dtype)
            rms = rms_l0 * l0_m + rms_lgt0 * lgt0_m
        else:
            degree_weights = tf.cast(self.degree_weights, x.dtype)
            weighted = x_sq * degree_weights
            norm = tf.reduce_sum(weighted, axis=angular_ax, keepdims=True)
            norm = tf.reduce_mean(norm, axis=feat_ax, keepdims=True)
            rms = tf.math.rsqrt(norm + eps)

        # 3. Zero out padding atoms
        if self.lm_first:
            r_map = tf.reshape(
                tf.range(n_at_b_total, delta=1, dtype=tf.int32), [1, -1, 1]
            )
        else:
            r_map = tf.reshape(
                tf.range(n_at_b_total, delta=1, dtype=tf.int32), [-1, 1, 1]
            )
        rms = tf.where(r_map < n_at_b_real, rms, tf.zeros_like(rms))

        # 4. Per-(l, parity, hist) affine scale
        scale = tf.gather(
            tf.cast(self.affine_weight, x.dtype), self.expand_index, axis=0
        )  # [n_angular, n_out]
        if self.lm_first:
            # scale: [n_angular, n_out] → [n_angular, 1, n_out]
            scale = scale[:, tf.newaxis, :]
        else:
            # scale: [n_angular, n_out] → [n_out, n_angular] → [1, n_out, n_angular]
            scale = tf.transpose(scale)[tf.newaxis, :, :]

        if self.normalize_l0_only:
            normalized = x * rms * scale
            non_l0 = tf.cast(self.non_l0_mask, x.dtype)
            out = normalized * (1.0 - non_l0) + x * non_l0
        else:
            out = x * rms * scale

        # 5. Optional L=0 bias
        if self.center_l0 and self.center_l0_bias:
            bias_mask = tf.cast(r_map < n_at_b_real, out.dtype)
            l0_bias = tf.cast(self.l0_bias, out.dtype)  # [1, n_out, n_angular]
            if self.lm_first:
                l0_bias = tf.transpose(l0_bias, [2, 0, 1])  # [n_angular, 1, n_out]
            out = out + l0_bias * l0_mask_f * bias_mask

        return out


@capture_init_args
class EquivariantGate(TPEquivariantInstruction):
    """
    Equivariant gating using L=0 (scalar) channels to gate all angular channels.

    Extracts L=0 components from the input, projects them through a learnable
    linear layer (or optional MLP with SiLU activation), applies sigmoid, and
    multiplies with all channels. Gate values are computed per (l, parity, hist)
    group and broadcast over m within each group, preserving equivariance.

    Parameters
    ----------
    input : TPEquivariantInstruction
        The equivariant instruction whose output will be gated.
    name : str
        Name of this instruction.
    hidden_dim : int or None
        If None, use a single linear projection. If int, use a two-layer MLP
        with this hidden dimension and a SiLU activation.
    use_bias : bool
        If True, add learnable bias to the gate projection(s). Default True.
    mix_channels : int or None
        If None (default), gate values are computed per-channel independently
        (same projection applied to each channel's L=0 features). If int,
        first mix across n_channels via a bottleneck of this dimension, then
        project to gate values. This allows the gate to make decisions based
        on cross-channel information.
    """

    input_tensor_spec = {}

    def __init__(
        self,
        input: TPEquivariantInstruction,
        name: str,
        hidden_dim: int = None,
        use_bias: bool = True,
        mix_channels: int = None,
        lm_first: bool = False,
        **kwargs,
    ):
        self.lm_first = lm_first
        super().__init__(name=name, lmax=input.lmax)
        self.input = input
        self.n_out = input.n_out
        self.coupling_meta_data = input.coupling_meta_data.copy()
        if hasattr(input, "coupling_origin"):
            self.coupling_origin = input.coupling_origin

        self.hidden_dim = hidden_dim
        self.use_bias = use_bias
        self.mix_channels = mix_channels

        # Identify L=0 indices in the angular dimension
        l_values = self.coupling_meta_data["l"].values
        self._l0_indices_np = np.where(l_values == 0)[0].astype(np.int32)
        self.n_l0 = len(self._l0_indices_np)
        assert self.n_l0 > 0, "No L=0 channels found — cannot construct gate signal"

        # Runtime normalization constant for the linear projection
        self._norm_l0 = 1.0 / np.sqrt(self.n_l0)

        # Build (l, parity, hist) group mapping
        lph_groups = self.coupling_meta_data.groupby(["l", "parity", "hist"]).indices
        self.n_groups = len(lph_groups)

        n_angular = len(l_values)
        expand_idx = np.zeros(n_angular, dtype=np.int32)
        for gid, (key, indices) in enumerate(lph_groups.items()):
            for idx in indices:
                expand_idx[idx] = gid
        self._expand_index_np = expand_idx

        if self.hidden_dim is not None:
            self._norm_hidden = 1.0 / np.sqrt(self.hidden_dim)

        if self.mix_channels is not None:
            self._norm_mix = 1.0 / np.sqrt(self.input.n_out)
            self._norm_mix_back = 1.0 / np.sqrt(self.mix_channels)

    @tf.Module.with_name_scope
    def build(self, float_dtype):
        if not self.is_built:
            self.l0_indices = tf.constant(self._l0_indices_np, dtype=tf.int32)
            self.expand_index = tf.constant(self._expand_index_np, dtype=tf.int32)
            self.norm_l0 = tf.constant(self._norm_l0, dtype=float_dtype)

            # Optional cross-channel mixing bottleneck
            if self.mix_channels is not None:
                self.norm_mix = tf.constant(self._norm_mix, dtype=float_dtype)
                self.norm_mix_back = tf.constant(self._norm_mix_back, dtype=float_dtype)
                self.mix_weight = tf.Variable(
                    tf.random.normal(
                        [self.input.n_out, self.mix_channels], dtype=float_dtype
                    ),
                    name="mix_weight",
                )
                self.mix_weight_back = tf.Variable(
                    tf.random.normal(
                        [self.mix_channels, self.input.n_out], dtype=float_dtype
                    ),
                    name="mix_weight_back",
                )

            if self.hidden_dim is None:
                # Linear: [n_l0] -> [n_groups]
                self.gate_weight = tf.Variable(
                    tf.random.normal([self.n_l0, self.n_groups], dtype=float_dtype),
                    name="gate_weight",
                )
                if self.use_bias:
                    self.gate_bias = tf.Variable(
                        tf.zeros([self.n_groups], dtype=float_dtype),
                        name="gate_bias",
                    )
            else:
                # MLP: [n_l0] -> [hidden_dim] -> [n_groups]
                self.norm_hidden = tf.constant(self._norm_hidden, dtype=float_dtype)
                self.gate_w1 = tf.Variable(
                    tf.random.normal([self.n_l0, self.hidden_dim], dtype=float_dtype),
                    name="gate_w1",
                )
                self.gate_w2 = tf.Variable(
                    tf.random.normal(
                        [self.hidden_dim, self.n_groups], dtype=float_dtype
                    ),
                    name="gate_w2",
                )
                if self.use_bias:
                    self.gate_b1 = tf.Variable(
                        tf.zeros([self.hidden_dim], dtype=float_dtype),
                        name="gate_b1",
                    )
                    self.gate_b2 = tf.Variable(
                        tf.zeros([self.n_groups], dtype=float_dtype),
                        name="gate_b2",
                    )

            self.is_built = True

    def frwrd(self, input_data, training=False, local=False):
        x = input_data[self.input.name]
        # x shape: [atoms, n_channels, n_angular] (standard)
        #      or: [n_angular, atoms, n_channels] (lm_first)

        if self.lm_first:
            # Gather L=0 from axis 0, then transpose to [atoms, n_channels, n_l0]
            # for gate computation (which is layout-agnostic scalar math)
            x_l0 = tf.gather(x, self.l0_indices, axis=0)  # [n_l0, atoms, n_channels]
            x_l0 = tf.transpose(x_l0, [1, 2, 0])  # [atoms, n_channels, n_l0]
        else:
            x_l0 = tf.gather(x, self.l0_indices, axis=2)  # [atoms, n_channels, n_l0]

        # Optional cross-channel mixing bottleneck
        if self.mix_channels is not None:
            mix_w = tf.cast(self.mix_weight, x.dtype)
            mix_w_back = tf.cast(self.mix_weight_back, x.dtype)
            norm_mix = tf.cast(self.norm_mix, x.dtype)
            norm_mix_back = tf.cast(self.norm_mix_back, x.dtype)
            # [atoms, n_channels, n_l0] -> [atoms, mix_channels, n_l0]
            x_l0 = tf.einsum("anl,nk->akl", x_l0, mix_w) * norm_mix
            x_l0 = tf.nn.silu(x_l0)
            # [atoms, mix_channels, n_l0] -> [atoms, n_channels, n_l0]
            x_l0 = tf.einsum("akl,kn->anl", x_l0, mix_w_back) * norm_mix_back

        # Compute gate values
        if self.hidden_dim is None:
            gate_w = tf.cast(self.gate_weight, x.dtype)
            norm = tf.cast(self.norm_l0, x.dtype)
            gate = tf.einsum("anl,lg->ang", x_l0, gate_w) * norm
            if self.use_bias:
                gate = gate + tf.cast(self.gate_bias, x.dtype)
        else:
            w1 = tf.cast(self.gate_w1, x.dtype)
            w2 = tf.cast(self.gate_w2, x.dtype)
            norm_l0 = tf.cast(self.norm_l0, x.dtype)
            norm_h = tf.cast(self.norm_hidden, x.dtype)
            h = tf.einsum("anl,lh->anh", x_l0, w1) * norm_l0
            if self.use_bias:
                h = h + tf.cast(self.gate_b1, x.dtype)
            h = tf.nn.silu(h)
            gate = tf.einsum("anh,hg->ang", h, w2) * norm_h
            if self.use_bias:
                gate = gate + tf.cast(self.gate_b2, x.dtype)

        gate = tf.sigmoid(gate)  # [atoms, n_out_channels, n_groups]

        # Expand gate to full angular dimension
        if self.lm_first:
            # gate: [atoms, n_channels, n_groups]
            # → transpose to [n_groups, atoms, n_channels]
            # → gather to [n_angular, atoms, n_channels]
            gate_t = tf.transpose(gate, [2, 0, 1])
            gate_expanded = tf.gather(gate_t, self.expand_index, axis=0)
        else:
            gate_expanded = tf.gather(gate, self.expand_index, axis=2)

        return x * gate_expanded


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
        out_norm: bool = False,
        lm_first: bool = False,
        **kwargs,
    ):
        super(FunctionReduceParticular, self).__init__(name=name, lmax=selected_l)
        self.lm_first = lm_first
        self.instructions = instructions
        self.selected_l = selected_l
        self.selected_p = selected_p
        self.n_out = n_out
        self.is_central_atom_type_dependent = is_central_atom_type_dependent
        self.number_of_atom_types = number_of_atom_types
        self.out_norm = out_norm

        if self.is_central_atom_type_dependent:
            assert self.number_of_atom_types is not None

        instr_names = [instr.name for instr in self.instructions]
        assert len(instr_names) == len(set(instr_names)), "duplicate instruction names"
        assert (
            np.min([instr.lmax for instr in self.instructions]) >= self.selected_l
        ), f"Some of the instructions do not have required lmax {self.selected_l}"

        collector_data = []
        for m in range(-self.selected_l, self.selected_l + 1):
            collector_data.append(
                [self.selected_l, m, "", self.selected_p, self.selected_l]
            )
        self.coupling_meta_data = pd.DataFrame(
            collector_data, columns=["l", "m", "hist", "parity", "sum_of_ls"]
        )
        norms = np.zeros(self.coupling_meta_data.shape[0])
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
                        norms[idx] += 1
            instruction_collection["total_sum_ind"] = tf.constant(
                np.array(instruction_collection["total_sum_ind"]).reshape(-1, 1),
                dtype=tf.int32,
            )
            instruction_collection["n_out"] = instr.n_out
            self.selector[instr.name] = instruction_collection
        norms[norms == 0] = 1
        self._out_norm_map = 1 / norms**0.5

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
            if self.out_norm:
                self.norm_map = tf.reshape(
                    tf.constant(self._out_norm_map, dtype=float_dtype), [-1, 1, 1]
                )
            else:
                self.norm_map = tf.constant(1, dtype=float_dtype)
            self.float_dtype = float_dtype
            self.is_built = True

    def frwrd(self, input_data, training=False, local=False):
        # lm_first: equivariant inputs are [lm, atoms, n] (gather lm on axis 0,
        # einsum reads A_r as "wan"); standard: [atoms, n, lm] (axis 2, "anw").
        gather_ax = 0 if self.lm_first else 2
        collection = None
        for instr in self.instructions:
            instruction_collection = self.selector[instr.name]
            A_r = tf.gather(
                input_data[instr.name],
                instruction_collection["func_collect_ind"],
                axis=gather_ax,
            )
            w = tf.gather(
                getattr(self, f"reducing_{instr.name}"),
                instruction_collection["w_l_tile"],
                axis=-1,
            )
            if collection is None:
                collection = tf.zeros(
                    [
                        self.coupling_meta_data.shape[0],
                        input_data[constants.N_ATOMS_BATCH_TOTAL],
                        self.n_out,
                    ],
                    dtype=w.dtype,
                )
            if self.is_central_atom_type_dependent:
                w = tf.gather(w, input_data[constants.ATOMIC_MU_I], axis=0)
                eq = "aknw,wan->wak" if self.lm_first else "aknw,anw->wak"
            else:
                eq = "knw,wan->wak" if self.lm_first else "knw,anw->wak"
            # w_al = tf.gather(w, instruction_collection["w_l_tile"], axis=-1)
            norm = getattr(self, f"norm_{instr.name}")
            if A_r.dtype != w.dtype:
                A_r = tf.cast(A_r, w.dtype)
            pr = tf.einsum(eq, w, A_r, name=f"ein_{instr.name}") * norm

            collection = tf.tensor_scatter_nd_add(
                collection, instruction_collection["total_sum_ind"], pr
            )

        # norm_map is [n_cg_out,1,1], broadcasts over [n_cg_out, atoms, n_out].
        collection *= self.norm_map
        # collection is [n_cg_out, atoms, n_out] (already lm-first); standard
        # consumers want [atoms, n_out, lm].
        if self.lm_first:
            return collection
        return tf.transpose(collection, [1, 2, 0])

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

        self.at_nums = (
            np.array([atomic_numbers[sym] for sym, ind in element_map.items()])
            .astype(np.float64)
            .reshape(-1, 1)
        )

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

    def get_element_map(self) -> tuple[np.ndarray, np.ndarray] | None:
        return (
            self.element_map_symbols.numpy().astype(str),
            self.element_map_index.numpy(),
        )

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

    def frwrd(self, input_data: dict, training=False, local=False):
        if local:
            raise NotImplementedError
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
