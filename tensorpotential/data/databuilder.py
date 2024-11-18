from __future__ import annotations

import gc
import logging
import os
import signal
import threading
import time

import numpy as np
import pandas as pd

from abc import ABC, abstractmethod
from ase import Atoms
from collections import defaultdict, deque
from concurrent.futures import ProcessPoolExecutor
from tqdm import tqdm
from itertools import combinations_with_replacement

from tensorpotential import constants
from tensorpotential.data.process_df import ENERGY_CORRECTED_COL, FORCES_COL
from tensorpotential.utils import process_cutoff_dict

from matscipy.neighbours import neighbour_list as nl

# from ase.neighborlist import neighbor_list as nl

from ase.stress import full_3x3_to_voigt_6_stress

DEFAULT_STRESS_UNITS = "eV/A3"
MININTERVAL = 2  # min interval for progress bar


### from https://stackoverflow.com/questions/71300294/how-to-terminate-pythons-processpoolexecutor-when-parent-process-dies
def start_thread_to_terminate_when_parent_process_dies(ppid):
    pid = os.getpid()

    def f():
        while True:
            try:
                os.kill(ppid, 0)
            except OSError:
                os.kill(pid, signal.SIGTERM)
            time.sleep(1)

    thread = threading.Thread(target=f, daemon=True)
    thread.start()


def enforce_pbc(atoms, cutoff):
    """Enforce periodic boundary conditions for a given cutoff."""
    pos = atoms.get_positions()
    if (atoms.get_pbc() == 0).all():
        max_d = np.max(np.linalg.norm(pos - pos[0], axis=1))
        cell = np.eye(3) * ((max_d + cutoff) * 2)
        atoms.set_cell(cell)
        atoms.center()
    atoms.set_pbc(True)

    return atoms


def transparent_iterator(iterator, *arg, **kwarg):
    return iterator


def get_number_of_real_atoms(data_dict):
    return int(data_dict[constants.N_ATOMS_BATCH_REAL])


def get_number_of_real_neigh(data_dict):
    return int(data_dict[constants.N_NEIGHBORS_REAL])


def get_number_of_real_struc(data_dict):
    return int(data_dict[constants.N_STRUCTURES_BATCH_REAL])


def generate_batches_df(structures_df, batch_size):
    structures_df["bid"] = np.arange(len(structures_df)) // batch_size
    structures_df["structure_ind"] = structures_df.index
    batch_df = structures_df.groupby("bid").agg(
        {"nat": "sum", "nneigh": "sum", "nstruct": "sum", "structure_ind": list}
    )
    assert batch_df["nstruct"].sum() == len(structures_df)
    return batch_df


def direct_split(data_dict_list, batch_size):
    n_structs = len(data_dict_list)

    splits_groups = np.array_split(np.arange(n_structs), n_structs // batch_size)

    data_batches = []
    for split in splits_groups:
        batch = [data_dict_list[i] for i in split]
        data_batches.append(batch)
    return data_batches


def get_padding_dims(batch, max_pad_dict):
    nreal_struc = get_number_of_real_struc(batch)
    nreal_atoms = get_number_of_real_atoms(batch)
    nreal_neigh = get_number_of_real_neigh(batch)
    pad_nstruct = max_pad_dict[constants.PAD_MAX_N_STRUCTURES] - nreal_struc
    pad_nat = max_pad_dict[constants.PAD_MAX_N_ATOMS] - nreal_atoms
    pad_nneigh = max_pad_dict[constants.PAD_MAX_N_NEIGHBORS] - nreal_neigh
    return pad_nat, pad_nneigh, pad_nstruct


def bucketing_split(data_dict_list, batch_size, max_n_buckets, verbose=False):
    """Split data into buckets based on number of atoms and number of neighbors."""
    nat = [get_number_of_real_atoms(d) for d in data_dict_list]
    nneighbors = [get_number_of_real_neigh(d) for d in data_dict_list]

    structures_df = pd.DataFrame({"nat": nat, "nneigh": nneighbors})
    structures_df["nstruct"] = 1
    try:
        structures_df["nneigh"] = structures_df["nneigh"].map(sum)
    except TypeError:
        pass
    # index: bid,   "nat", "nneigh", "nstruct", "structure_ind"
    batches_df = generate_batches_df(structures_df, batch_size)
    batches_df = batches_df.sort_values(
        ["nneigh", "nat", "nstruct"], ascending=False
    ).reset_index(drop=True)
    buckets_list = np.array_split(batches_df, max_n_buckets)

    data_batches = []
    max_pad_batches = []

    iterator_func = tqdm if verbose else transparent_iterator

    for bucket in iterator_func(
        buckets_list, total=len(buckets_list), mininterval=MININTERVAL
    ):
        # bucket = collection of batches (repr as pd.DataFrame)
        max_nstruct = bucket["nstruct"].max()
        max_nat = bucket["nat"].max()
        max_nneigh = bucket["nneigh"].max()

        # pad at least one atom and structure if has to pad nneigh
        is_pad_nneigh = np.any(bucket["nneigh"] != max_nneigh)
        is_pad_atoms = np.any(bucket["nat"] != max_nat) or is_pad_nneigh
        is_pad_struct = is_pad_atoms

        if is_pad_atoms:
            max_nat += 1
        if is_pad_struct:
            max_nstruct += 1

        for _, row in bucket.iterrows():
            batch = [data_dict_list[i] for i in row["structure_ind"]]

            data_batches.append(batch)

            max_pad_batches.append(
                {
                    constants.PAD_MAX_N_STRUCTURES: max_nstruct,
                    constants.PAD_MAX_N_ATOMS: max_nat,
                    constants.PAD_MAX_N_NEIGHBORS: max_nneigh,
                }
            )

    return data_batches, max_pad_batches


class AbstractDataBuilder(ABC):
    def __init__(self, float_dtype: np.float32 | np.float64 = np.float64):
        self.float_dtype = float_dtype

    @abstractmethod
    def extract_from_ase_atoms(self, ase_atoms, **kwarg):
        raise NotImplementedError()

    @abstractmethod
    def extract_from_row(self, row, **kwarg):
        raise NotImplementedError()

    @abstractmethod
    def join_to_batch(self, pre_batch_list: list):
        """Implement logic of joining list of dicts(per-structure) into single batch"""
        raise NotImplementedError()

    @abstractmethod
    def pad_batch(self, batch, max_pad_dict):
        """Inplace pad of batch"""
        raise NotImplementedError()

    def postprocess_dataset(self, batches):
        pass


class GeometricalDataBuilder(AbstractDataBuilder):
    """
    Default data builder for geometrical data:

    atomic_mu_i
    bond_vector
    mu_i
    mu_j
    ind_i
    ind_j

    n_atoms_real
    n_struct_real
    n_neigh_real

    batch_tot_nat
    map_atoms_to_structure
    batch_total_num_structures
    """

    def __init__(
        self,
        elements_map: dict,
        cutoff: float,
        cutoff_dict: dict = None,
        is_fit_stress=False,
        bond_type: str = "symmetric_bond",
        **kwargs,
    ):
        super().__init__(**kwargs)
        """
        elements_map: mapping of elements to index
        """
        if isinstance(elements_map, (list, tuple, set)):
            elements_map = sorted(elements_map)
            elements_map = {s: i for i, s in enumerate(elements_map)}
        assert isinstance(
            elements_map, dict
        ), "`elements_map` should be list of elements of dict[element->index] mapping"
        self.elements_map = elements_map
        self.cutoff = float(cutoff)

        self.cutoff_dict = (
            process_cutoff_dict(cutoff_dict, self.elements_map)
            if cutoff_dict is not None
            else None
        )

        assert bond_type == "symmetric_bond", ValueError(
            "Only one option available now"
        )
        self.bond_type = bond_type

        if self.cutoff_dict is not None:
            max_cutoff = np.max([v for k, v in self.cutoff_dict.items()])
            self.max_cutoff = np.max([max_cutoff, self.cutoff])
        else:
            self.max_cutoff = self.cutoff
            self.nl_cutoffs = self.cutoff

        self.is_fit_stress = is_fit_stress

    def extract_from_ase_atoms(self, ase_atoms, **kwarg):
        """
        Construction of per-structure data dictionary
        """
        ase_atoms = enforce_pbc(ase_atoms, cutoff=self.max_cutoff)

        if self.cutoff_dict is not None:
            cutoff = {}
            for e1, e2 in combinations_with_replacement(
                set(ase_atoms.get_chemical_symbols()), 2
            ):
                cutoff[(e1, e2)] = self.cutoff_dict.get((e1, e2), self.cutoff)
        else:
            cutoff = self.cutoff

        ind_i, ind_j, bond_vector = nl("ijD", ase_atoms, cutoff=cutoff)

        atomic_mu_i = np.array(
            [self.elements_map[s] for s in ase_atoms.get_chemical_symbols()]
        )

        nat_per_specie = defaultdict(int)
        total_nei_per_specie = defaultdict(int)
        u_i, n_j = np.unique(ind_i, return_counts=True)

        for at_i, nb_i in zip(u_i, n_j):
            specie_i = atomic_mu_i[at_i]
            nat_per_specie[specie_i] += 1
            total_nei_per_specie[specie_i] += nb_i

        all_atom_ind = np.arange(len(ase_atoms))
        if np.unique(ind_i).shape[0] < all_atom_ind.shape[0]:
            # print(
            #     f"Found an atom with no neighbors within cutoff."
            #     f" Adding a fictitious neighbor beyond cutoff. Structure id: id"
            # )
            missing_ind = all_atom_ind[~np.isin(all_atom_ind, np.unique(ind_i))]
            ind_j_to_add = np.zeros(len(missing_ind)).astype(int)
            dv_j_to_add = (
                np.dot(
                    ase_atoms.cell,
                    np.array([[1, 1, 1] for _ in missing_ind]).reshape(3, -1),
                ).reshape(-1, 3)
                + self.max_cutoff
            )
            ind_i = np.append(ind_i, missing_ind)
            ind_j = np.append(ind_j, ind_j_to_add)
            bond_vector = np.append(bond_vector, dv_j_to_add, axis=0)

            sort = np.argsort(ind_i)
            ind_i = ind_i[sort]
            ind_j = ind_j[sort]
            bond_vector = bond_vector[sort]

        mu_i = np.array([atomic_mu_i[i] for i in ind_i])
        mu_j = np.array([atomic_mu_i[j] for j in ind_j])

        return {
            constants.ATOMIC_MU_I: atomic_mu_i.astype(np.int32),
            constants.BOND_VECTOR: bond_vector.astype(self.float_dtype),
            constants.BOND_MU_I: mu_i.astype(np.int32),
            constants.BOND_MU_J: mu_j.astype(np.int32),
            constants.BOND_IND_I: ind_i.astype(np.int32),
            constants.BOND_IND_J: ind_j.astype(np.int32),
            constants.N_ATOMS_BATCH_REAL: np.array(len(ase_atoms)).astype(np.int32),
            constants.N_STRUCTURES_BATCH_REAL: np.array(1).astype(np.int32),
            constants.N_NEIGHBORS_REAL: np.array(len(bond_vector)).astype(np.int32),
            "nat_per_specie": nat_per_specie,
            "total_nei_per_specie": total_nei_per_specie,
        }

    def extract_from_row(self, row, **kwarg):
        return self.extract_from_ase_atoms(row["ase_atoms"])

    def join_to_batch(self, pre_batch_list: list):
        """Implement logic of joining list of dicts(per-structure) into single batch"""

        res_dict = {}
        for key in [
            constants.BOND_VECTOR,
            constants.BOND_MU_I,
            constants.BOND_MU_J,
            constants.ATOMIC_MU_I,
        ]:
            data_list = [data_dict[key] for data_dict in pre_batch_list]
            res_dict[key] = np.concatenate(data_list, axis=0)

        for key in [constants.BOND_IND_I, constants.BOND_IND_J]:
            data_list = []
            index_shift = 0
            for data_dict in pre_batch_list:
                data_list.append(data_dict[key] + index_shift)
                index_shift += data_dict[constants.N_ATOMS_BATCH_REAL]

            res_dict[key] = np.concatenate(data_list, axis=0)

        for key in [
            constants.N_ATOMS_BATCH_REAL,
            constants.N_STRUCTURES_BATCH_REAL,
            constants.N_NEIGHBORS_REAL,
        ]:
            sum_val = sum([data_dict[key] for data_dict in pre_batch_list])
            res_dict[key] = np.array(sum_val)

        # batch_tot_nat, batch_total_num_structures
        res_dict[constants.N_ATOMS_BATCH_REAL] = np.sum(
            [data_dict[constants.N_ATOMS_BATCH_REAL] for data_dict in pre_batch_list]
        ).astype(np.int32)
        res_dict[constants.N_STRUCTURES_BATCH_REAL] = np.sum(
            [
                data_dict[constants.N_STRUCTURES_BATCH_REAL]
                for data_dict in pre_batch_list
            ]
        ).astype(np.int32)

        # In case no padding will follow
        res_dict[constants.N_ATOMS_BATCH_TOTAL] = res_dict[constants.N_ATOMS_BATCH_REAL]
        res_dict[constants.N_STRUCTURES_BATCH_TOTAL] = res_dict[
            constants.N_STRUCTURES_BATCH_REAL
        ]

        # map_atoms_to_structure
        map_atoms_to_structure = []

        for struct_ind, data_dict in enumerate(pre_batch_list):
            map_atoms_to_structure += [struct_ind] * get_number_of_real_atoms(data_dict)
        res_dict[constants.ATOMS_TO_STRUCTURE_MAP] = np.array(
            map_atoms_to_structure
        ).astype(np.int32)

        if self.is_fit_stress:
            # map bonds to structure
            map_bonds_to_structure = []
            for struct_ind, data_dict in enumerate(pre_batch_list):
                map_bonds_to_structure += [struct_ind] * get_number_of_real_neigh(
                    data_dict
                )
            res_dict[constants.BONDS_TO_STRUCTURE_MAP] = np.array(
                map_bonds_to_structure
            ).astype(np.int32)

        nat_per_specie = defaultdict(int)
        total_nei_per_specie = defaultdict(int)
        for data_dict in pre_batch_list:
            nps = data_dict["nat_per_specie"]
            tnps = data_dict["total_nei_per_specie"]
            for k, v in nps.items():
                nat_per_specie[k] += v
            for k, v in tnps.items():
                total_nei_per_specie[k] += v
        res_dict["nat_per_specie"] = nat_per_specie
        res_dict["total_nei_per_specie"] = total_nei_per_specie

        return res_dict

    def pad_batch(self, batch, max_pad_dict):
        """Inplace pad of batch"""
        pad_nat, pad_nneigh, pad_nstruct = get_padding_dims(batch, max_pad_dict)

        max_nat = np.array(max_pad_dict[constants.PAD_MAX_N_ATOMS]).astype(np.int32)
        max_structs = np.array(max_pad_dict[constants.PAD_MAX_N_STRUCTURES]).astype(
            np.int32
        )

        batch[constants.N_ATOMS_BATCH_TOTAL] = max_nat
        batch[constants.N_STRUCTURES_BATCH_TOTAL] = max_structs

        # pad atoms:
        if pad_nat > 0:
            key = constants.ATOMIC_MU_I
            a = batch[key]
            batch[key] = np.pad(a, (0, pad_nat), mode="constant", constant_values=a[0])

            key = constants.ATOMS_TO_STRUCTURE_MAP
            a = batch[key]
            batch[key] = np.pad(
                a,
                (0, pad_nat),
                mode="constant",
                constant_values=max_structs - 1,  # fake structure
            )

        # pad neigh
        if pad_nneigh > 0:
            for key in [
                constants.BOND_MU_I,
                constants.BOND_MU_J,
            ]:
                a = batch[key]
                batch[key] = np.pad(
                    a,
                    (0, pad_nneigh),
                    mode="constant",
                    constant_values=a[0],
                    # TODO: pad BOND_IND_I BOND_IND_J with fake atom?
                )

            # pad BOND_IND_I BOND_IND_J with fake atom
            for key in [
                constants.BOND_IND_I,
                constants.BOND_IND_J,
            ]:
                a = batch[key]
                batch[key] = np.pad(
                    a,
                    (0, pad_nneigh),
                    mode="constant",
                    constant_values=max_nat - 1,  # fake atom
                )

            # special case: bond vector
            key = constants.BOND_VECTOR
            a = batch[key]
            batch[key] = np.pad(
                a, ((0, pad_nneigh), (0, 0)), mode="constant", constant_values=1e6
            )

            if self.is_fit_stress:
                key = constants.BONDS_TO_STRUCTURE_MAP
                a = batch[key]
                batch[key] = np.pad(
                    a,
                    (0, pad_nneigh),
                    mode="constant",
                    constant_values=max_structs - 1,  # fake structure
                )


class ReferenceEnergyForcesStressesDataBuilder(AbstractDataBuilder):
    def __init__(
        self,
        normalize_weights=True,
        normalize_force_per_structure=True,
        is_fit_stress=False,
        stress_units=None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.energy_weight: str = constants.DATA_ENERGY_WEIGHTS
        self.force_weight: str = constants.DATA_FORCE_WEIGHTS
        self.virial_weight: str = constants.DATA_VIRIAL_WEIGHTS
        self.normalize_weights = normalize_weights
        self.normalize_force_per_structure = normalize_force_per_structure
        self.is_fit_stress = is_fit_stress

        self.stress_conversion_factor = {
            None: 1.0,  # default
            DEFAULT_STRESS_UNITS: 1.0,  # eV/A3 -> eV/A3
            "GPa": 1 / 160.2176621,  # GPa -> eV/A3
            "kbar": 1 / 160.2176621 / 10,  # kbar -> eV/A3
            "-kbar": -1 / 160.2176621 / 10,  # kbar -> eV/A3
        }[stress_units]

    def extract_from_ase_atoms(self, ase_atoms, **kwarg):
        # ref E, F
        res = {
            constants.DATA_REFERENCE_ENERGY: np.array(
                ase_atoms.get_potential_energy()
            ).reshape(-1, 1),
            constants.DATA_REFERENCE_FORCES: np.array(ase_atoms.get_forces()),
            constants.DATA_STRUCTURE_ID: np.array([-1]).reshape(-1, 1),
        }

        # weights
        e_w = ase_atoms.info.get(
            self.energy_weight, np.array(1.0).reshape(-1, 1)
        ).reshape(-1, 1)
        if self.normalize_force_per_structure:
            default_f_w = (
                np.ones((len(ase_atoms), 1)).reshape(-1, 1) / len(ase_atoms) / 3
            )
        else:
            default_f_w = np.ones((len(ase_atoms), 1)).reshape(-1, 1)

        f_w = ase_atoms.info.get(self.force_weight, default_f_w).reshape(-1, 1)

        res.update({self.energy_weight: e_w, self.force_weight: f_w})

        # if fit stress
        if self.is_fit_stress:
            if np.all(ase_atoms.pbc):
                virials = -np.array(ase_atoms.get_stress() * ase_atoms.get_volume())
                # convert from Voigt to straight notations: xx,yy,zz, xy,xz,yz:
                virials = virials[[0, 1, 2, 5, 4, 3]]
                res[constants.DATA_REFERENCE_VIRIAL] = virials.reshape(1, 6)
                res[self.virial_weight] = np.array(1.0).reshape(-1, 1)
            else:
                res[constants.DATA_REFERENCE_VIRIAL] = np.zeros((1, 6))
                res[self.virial_weight] = np.array(0.0).reshape(-1, 1)
            res[constants.DATA_REFERENCE_VIRIAL] *= self.stress_conversion_factor

        return res

    def extract_from_row(self, row, **kwarg):
        res = {
            constants.DATA_REFERENCE_ENERGY: np.array(
                row[ENERGY_CORRECTED_COL]
            ).reshape(-1, 1),
            constants.DATA_REFERENCE_FORCES: np.array(row[FORCES_COL]),
        }
        if constants.DATA_STRUCTURE_ID in kwarg:
            res[constants.DATA_STRUCTURE_ID] = kwarg[constants.DATA_STRUCTURE_ID]

        e_w = (
            np.array(row[self.energy_weight]).reshape(-1, 1)
            if self.energy_weight in row.index
            else np.array(1.0).reshape(-1, 1)
        )
        ase_atoms = row["ase_atoms"]
        if self.normalize_force_per_structure:
            default_f_w = (
                np.ones((len(ase_atoms), 1)).reshape(-1, 1) / len(ase_atoms) / 3
            )
        else:
            default_f_w = np.ones((len(ase_atoms), 1)).reshape(-1, 1)

        f_w = (
            np.array(row[self.force_weight]).reshape(-1, 1)
            if self.force_weight in row.index
            else default_f_w
        )
        res.update({self.energy_weight: e_w, self.force_weight: f_w})

        # if fit stress
        if self.is_fit_stress:
            default_stress = np.zeros((1, 6))
            default_stress_weight = np.array(0.0).reshape(-1, 1)

            if np.all(ase_atoms.pbc):
                if "stress" in row.index:
                    stress = row["stress"]
                else:
                    raise ValueError(
                        "Fit stress is requested, but no stress was found."
                    )

                if stress is None:
                    stress = default_stress
                if np.shape(stress) == (3, 3):
                    stress = full_3x3_to_voigt_6_stress(stress)

                # virials in Voigt notations from ASE: xx,yy,zz,yz,xz,xy
                virials = -np.array(stress * ase_atoms.get_volume())
                # convert from Voigt to straight notations: xx,yy,zz, xy,xz,yz:
                virials = virials[[0, 1, 2, 5, 4, 3]]
                res[constants.DATA_REFERENCE_VIRIAL] = np.array(virials).reshape(1, 6)

                res[self.virial_weight] = (
                    np.array(row[self.virial_weight]).reshape(-1, 1)
                    if self.virial_weight in row.index
                    else np.array(1.0).reshape(-1, 1)
                )
                res[constants.DATA_VOLUME] = np.array(ase_atoms.get_volume()).reshape(
                    -1, 1
                )
            else:
                # non-periodic structure
                res[constants.DATA_REFERENCE_VIRIAL] = default_stress
                res[self.virial_weight] = default_stress_weight

                # largest possible value
                max_val = np.finfo(np.float64).max
                res[constants.DATA_VOLUME] = np.array(max_val).reshape(-1, 1)
            res[constants.DATA_REFERENCE_VIRIAL] *= self.stress_conversion_factor
        return res

    def join_to_batch(self, pre_batch_list: list):
        res_dict = {}
        for key in [constants.DATA_REFERENCE_ENERGY]:
            data_list = [float(data_dict[key]) for data_dict in pre_batch_list]
            res_dict[key] = np.array(data_list).reshape(-1, 1).astype(self.float_dtype)

        for key in [constants.DATA_STRUCTURE_ID]:
            data_list = [int(data_dict[key]) for data_dict in pre_batch_list]
            res_dict[key] = np.array(data_list).reshape(-1, 1).astype(int)

        for key in [constants.DATA_REFERENCE_FORCES]:
            data_list = [data_dict[key] for data_dict in pre_batch_list]
            res_dict[key] = np.concatenate(data_list, axis=0).astype(self.float_dtype)

        for key in [self.energy_weight, self.force_weight]:
            data_list = [data_dict[key] for data_dict in pre_batch_list]
            res_dict[key] = (
                np.concatenate(data_list, axis=0)
                .reshape(-1, 1)
                .astype(self.float_dtype)
            )

        if self.is_fit_stress:
            key = constants.DATA_REFERENCE_VIRIAL
            data_list = [data_dict[key] for data_dict in pre_batch_list]
            # TODO: think about shape (-1, 6) or (-1)
            res_dict[key] = np.concatenate(data_list, axis=0).astype(
                self.float_dtype
            )  # .reshape(-1,6)

            # weights
            data_list = [data_dict[self.virial_weight] for data_dict in pre_batch_list]
            res_dict[self.virial_weight] = (
                np.concatenate(data_list, axis=0)
                .reshape(-1, 1)
                .astype(self.float_dtype)
            )

            # volumes
            data_list = [
                data_dict[constants.DATA_VOLUME] for data_dict in pre_batch_list
            ]
            res_dict[constants.DATA_VOLUME] = (
                np.concatenate(data_list, axis=0)
                .reshape(-1, 1)
                .astype(self.float_dtype)
            )

        return res_dict

    def pad_batch(self, batch, max_pad_dict):
        pad_nat, pad_nneigh, pad_nstruct = get_padding_dims(batch, max_pad_dict)

        if pad_nat > 0:
            k = constants.DATA_REFERENCE_FORCES
            batch[k] = np.pad(
                batch[k], ((0, pad_nat), (0, 0)), mode="constant", constant_values=0
            )

            k = self.force_weight
            batch[k] = np.pad(
                batch[k], ((0, pad_nat), (0, 0)), mode="constant", constant_values=0
            )

        if pad_nstruct > 0:
            k = constants.DATA_REFERENCE_ENERGY
            batch[k] = np.pad(
                batch[k], ((0, pad_nstruct), (0, 0)), mode="constant", constant_values=0
            )

            k = constants.DATA_STRUCTURE_ID
            batch[k] = np.pad(
                batch[k],
                ((0, pad_nstruct), (0, 0)),
                mode="constant",
                constant_values=-1,
            )

            k = self.energy_weight
            batch[k] = np.pad(
                batch[k], ((0, pad_nstruct), (0, 0)), mode="constant", constant_values=0
            )

            if self.is_fit_stress:
                k = constants.DATA_REFERENCE_VIRIAL
                batch[k] = np.pad(
                    batch[k],
                    ((0, pad_nstruct), (0, 0)),
                    mode="constant",
                    constant_values=0,
                )

                k = self.virial_weight
                batch[k] = np.pad(
                    batch[k],
                    ((0, pad_nstruct), (0, 0)),
                    mode="constant",
                    constant_values=0,
                )

                # volumes
                max_val = np.finfo(np.float64).max
                k = constants.DATA_VOLUME
                batch[k] = np.pad(
                    batch[k],
                    ((0, pad_nstruct), (0, 0)),
                    mode="constant",
                    constant_values=max_val,
                )

    def postprocess_dataset(self, batches):
        if self.normalize_weights:
            energy_weight_sum = np.sum(np.sum(b[self.energy_weight]) for b in batches)
            forces_weight_sum = np.sum(np.sum(b[self.force_weight]) for b in batches)

            for b in batches:
                b[self.energy_weight] /= energy_weight_sum
                b[self.force_weight] /= forces_weight_sum

            if self.is_fit_stress:
                virial_weight_sum = np.sum(
                    np.sum(b[self.virial_weight]) for b in batches
                )
                for b in batches:
                    b[self.virial_weight] /= virial_weight_sum


def construct_batches(
    df_or_ase_atoms_list: list[Atoms] | pd.DataFrame,
    data_builders: list[AbstractDataBuilder],
    batch_size: int,
    max_n_buckets: int = None,
    return_padding_stats: bool = False,
    verbose: bool = True,
    external_max_nneigh: int = None,
    external_max_nat: int = None,
    gc_collect=True,
    max_workers=None,
):
    """
    Return:    batches, padding_stats (optional)
    """
    iterator_func = tqdm if verbose else transparent_iterator
    data_dict_list = []

    if isinstance(df_or_ase_atoms_list, list) and isinstance(
        df_or_ase_atoms_list[0], Atoms
    ):
        # Stage 1. Build List of (data dict)
        if verbose:
            logging.info("1/3. Converting atomic structures to batch data")
        for ase_atoms in iterator_func(
            df_or_ase_atoms_list,
            total=len(df_or_ase_atoms_list),
            mininterval=MININTERVAL,
        ):
            data_dict = {}
            for data_builder in data_builders:
                data_dict.update(data_builder.extract_from_ase_atoms(ase_atoms))
            data_dict_list.append(data_dict)
    elif isinstance(df_or_ase_atoms_list, pd.DataFrame):
        # Stage 1. Build List of (data dict)
        if max_workers is None:
            # SERIAL PROCESSING
            if verbose:
                logging.info("1/4. Converting pd.DataFrame to batch data")
            data_dict_list = process_dataframe(
                df_or_ase_atoms_list, data_builders, iterator_func
            )
        else:
            # PARALLEL PROCESSING
            if verbose:
                logging.info(
                    f"1/4. Converting pd.DataFrame to batch data (parallel, max_workers={max_workers})"
                )
            data_dict_list = parallel_process_dataframe(
                df_or_ase_atoms_list, data_builders, iterator_func, max_workers
            )
    else:
        raise NotImplementedError()

    assert len(data_dict_list) == len(df_or_ase_atoms_list)
    padding_stats = None
    # Stage 2:
    if max_n_buckets is None:
        # direct_split
        pre_batch_groups = direct_split(data_dict_list, batch_size=batch_size)
        batches = deque()  # for more memory-efficient batches conversion later
        for batch_group in iterator_func(
            pre_batch_groups, total=len(pre_batch_groups), mininterval=MININTERVAL
        ):
            batch = {}
            for data_builder in data_builders:
                batch.update(data_builder.join_to_batch(batch_group))
            if batch:
                batches.append(batch)
    else:
        # bucketing
        if verbose:
            logging.info("2/4. Splitting batches into per-batch groups.")
        pre_batch_groups, max_pad_batches = bucketing_split(
            data_dict_list,
            batch_size=batch_size,
            max_n_buckets=max_n_buckets,
            verbose=verbose,
        )
        del data_dict_list
        if gc_collect:
            gc.collect()

        batches = deque()  # for more memory-efficient batches conversion later
        if verbose:
            logging.info("3/4. Joining groups to batches")
        for batch_group in iterator_func(
            pre_batch_groups, total=len(pre_batch_groups), mininterval=MININTERVAL
        ):
            batch = {}
            for data_builder in data_builders:
                batch.update(data_builder.join_to_batch(batch_group))
            if batch:
                batches.append(batch)
        del pre_batch_groups
        if gc_collect:
            gc.collect()

        # Stage 4. Padding batches
        if verbose:
            logging.info("4/4. Padding batches")
        if external_max_nneigh is not None:
            current_max_nneigh = np.max(
                [d[constants.PAD_MAX_N_NEIGHBORS] for d in max_pad_batches]
            )
            if current_max_nneigh > external_max_nneigh:
                external_max_nneigh = current_max_nneigh
            for d in max_pad_batches:
                d[constants.PAD_MAX_N_NEIGHBORS] = external_max_nneigh
        if external_max_nat is not None:
            current_max_nat = np.max(
                [d[constants.PAD_MAX_N_ATOMS] for d in max_pad_batches]
            )
            if current_max_nat > external_max_nat:
                external_max_nat = current_max_nat
            for d in max_pad_batches:
                d[constants.PAD_MAX_N_ATOMS] = external_max_nat

        for batch, batch_max_pad_dict in iterator_func(
            zip(batches, max_pad_batches), total=len(batches), mininterval=MININTERVAL
        ):
            for data_builder in data_builders:
                data_builder.pad_batch(batch, batch_max_pad_dict)

        # accumulate stats
        if return_padding_stats:
            padding_stats = defaultdict(int)
            for batch, max_pad_dict in iterator_func(
                zip(batches, max_pad_batches), mininterval=MININTERVAL
            ):
                nreal_struc = get_number_of_real_struc(batch)
                nreal_atoms = get_number_of_real_atoms(batch)
                nreal_neigh = get_number_of_real_neigh(batch)

                pad_nstruct = max_pad_dict[constants.PAD_MAX_N_STRUCTURES] - nreal_struc
                pad_nat = max_pad_dict[constants.PAD_MAX_N_ATOMS] - nreal_atoms
                pad_nneigh = max_pad_dict[constants.PAD_MAX_N_NEIGHBORS] - nreal_neigh

                for k, v in {
                    "pad_nstruct": pad_nstruct,
                    "pad_nat": pad_nat,
                    "pad_nneigh": pad_nneigh,
                    "nreal_struc": nreal_struc,
                    "nreal_atoms": nreal_atoms,
                    "nreal_neigh": nreal_neigh,
                }.items():
                    padding_stats[k] += v

    # post-processing stage
    for data_builder in data_builders:
        data_builder.postprocess_dataset(batches)

    if return_padding_stats:
        return batches, padding_stats
    else:
        return batches


def process_dataframe(df_or_ase_atoms_list, data_builders, iterator_func):
    data_dict_list = []
    for row_ind, row in iterator_func(
        df_or_ase_atoms_list.iterrows(),
        total=len(df_or_ase_atoms_list),
        mininterval=MININTERVAL,
    ):
        data_dict = process_row(row_ind, row, data_builders)
        data_dict_list.append(data_dict)

    return data_dict_list


def parallel_process_dataframe(
    df_or_ase_atoms_list, data_builders, iterator_func, max_workers
):
    data_dict_list = []
    # Use ProcessPoolExecutor to parallelize row processing
    with ProcessPoolExecutor(
        max_workers=max_workers,
        initializer=start_thread_to_terminate_when_parent_process_dies,  # +
        initargs=(os.getpid(),),
    ) as executor:
        # Submit tasks for each row in df_or_ase_atoms_list
        futures = [
            executor.submit(process_row, row_ind, row, data_builders)
            for row_ind, row in df_or_ase_atoms_list.iterrows()
        ]

        # Collect results as they are completed
        for future in iterator_func(
            futures, total=len(futures), mininterval=MININTERVAL
        ):
            data_dict = future.result()
            data_dict_list.append(data_dict)

    return data_dict_list


def process_row(row_ind, row, data_builders):
    data_dict = {}  # managed by reference
    for data_builder in data_builders:
        data_dict.update(
            data_builder.extract_from_row(row, **{constants.DATA_STRUCTURE_ID: row_ind})
        )
    return data_dict
