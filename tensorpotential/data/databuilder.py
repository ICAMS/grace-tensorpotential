from __future__ import annotations

import gc
import itertools
import logging
import os
import signal
import threading
import time
import warnings
from abc import ABC, abstractmethod
from collections import defaultdict, deque
from concurrent.futures import ProcessPoolExecutor
from itertools import combinations_with_replacement

import numpy as np
import pandas as pd
from ase import Atoms
from ase.stress import full_3x3_to_voigt_6_stress
from matscipy.neighbours import neighbour_list as nl
from tqdm import tqdm

from tensorpotential import constants
from tensorpotential.data.process_df import ENERGY_CORRECTED_COL, FORCES_COL, STRESS_COL
from tensorpotential.utils import process_cutoff_dict, enforce_pbc


def symbols_to_indices(symbols, sym_to_idx, *, default=None) -> np.ndarray:
    """Map ASE chemical symbols to model element indices.

    ``sym_to_idx`` is a ``{symbol: index}`` mapping. If ``default`` is None
    (the strict default), unknown symbols raise ``KeyError`` — match the
    strict behavior already used by the data pipeline. If ``default`` is an
    int sentinel (e.g. ``-1``), unknowns map to that value instead.
    """
    if default is None:
        return np.array([sym_to_idx[s] for s in symbols], dtype=np.int32)
    sentinel = int(default)
    return np.array(
        [sym_to_idx.get(s, sentinel) for s in symbols], dtype=np.int32
    )

# from ase.neighborlist import neighbor_list as nl

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
        {
            "n_atoms": "sum",
            "n_neighbours": "sum",
            "n_structures": "sum",
            "structure_ind": list,
        }
    )
    assert batch_df["n_structures"].sum() == len(structures_df)
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


def estimate_n_buckets(batches_df, max_padding_fraction=0.3):
    """Estimate optimal number of buckets based on padding overhead.

    Sweeps n_buckets from 1 to 32, returns the smallest fulfilling the threshold.
    """
    batches_df = batches_df.sort_values(
        ["n_neighbours", "n_atoms", "n_structures"], ascending=False
    ).reset_index(drop=True)
    num_batches = len(batches_df)
    if num_batches <= 1:
        return 1

    total_real_neigh = batches_df["n_neighbours"].sum()
    if total_real_neigh == 0:
        return 1

    for n_buckets in range(1, min(num_batches, 32) + 1):
        buckets = np.array_split(batches_df, n_buckets)
        total_padded_neigh = sum(b["n_neighbours"].max() * len(b) for b in buckets)
        overhead = total_padded_neigh / total_real_neigh - 1.0
        if overhead <= max_padding_fraction:
            return n_buckets

    return min(num_batches, 32)


def bucketing_split(
    data_dict_list, batch_size, max_n_buckets, verbose=False, max_padding_fraction=0.3
):
    """Split data into buckets based on number of atoms and number of neighbors."""
    n_atoms = [get_number_of_real_atoms(d) for d in data_dict_list]
    nneighbors = [get_number_of_real_neigh(d) for d in data_dict_list]

    structures_df = pd.DataFrame({"n_atoms": n_atoms, "n_neighbours": nneighbors})
    structures_df["n_structures"] = 1
    try:
        structures_df["n_neighbours"] = structures_df["n_neighbours"].map(sum)
    except TypeError:
        pass
    # index: bid,   "n_atoms", "n_neighbours", "n_structures", "structure_ind"
    batches_df = generate_batches_df(structures_df, batch_size)
    batches_df = batches_df.sort_values(
        ["n_neighbours", "n_atoms", "n_structures"], ascending=False
    ).reset_index(drop=True)

    if max_n_buckets == "auto":
        max_n_buckets = estimate_n_buckets(
            batches_df, max_padding_fraction=max_padding_fraction
        )
        if verbose:
            logging.info(f"Auto-estimated max_n_buckets: {max_n_buckets}")

    buckets_list = split_batches_into_buckets(batches_df, max_n_buckets)

    data_batches = []
    max_pad_batches = []

    iterator_func = tqdm if verbose else transparent_iterator

    for bucket in iterator_func(
        buckets_list, total=len(buckets_list), mininterval=MININTERVAL
    ):
        # bucket = collection of batches (repr as pd.DataFrame)
        max_nstruct = bucket["n_structures"].max()
        max_nat = bucket["n_atoms"].max()
        max_nneigh = bucket["n_neighbours"].max()

        # pad at least one atom and structure if has to pad n_neighbours
        is_pad_nneigh = np.any(bucket["n_neighbours"] != max_nneigh)
        is_pad_atoms = np.any(bucket["n_atoms"] != max_nat) or is_pad_nneigh
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


def split_batches_into_buckets(batches_df, max_n_buckets):
    # dynamic bucket splitting, where split boundary is determined based on increase of nneigh
    # Assumes batches_df is already sorted
    buckets_list = np.array_split(batches_df, max_n_buckets)  # naive static splitting
    return buckets_list


def _struct_max_neigh(data_dict):
    """Max per-atom neighbor count for one (per-structure) data dict, from its real bond
    list. Every atom is guaranteed >=1 neighbor by the fictitious-neighbor handling in
    GeometricalDataBuilder.extract_from_ase_atoms."""
    nat = get_number_of_real_atoms(data_dict)
    n_real = get_number_of_real_neigh(data_dict)
    ind_i = np.asarray(data_dict[constants.BOND_IND_I])[:n_real]
    counts = np.bincount(ind_i, minlength=nat)[:nat]
    return int(counts.max()) if counts.size else 0


def bucketing_split_dense(
    data_dict_list,
    batch_size,
    max_n_buckets,
    slot_budget="auto",
    n_neigh_buckets="auto",
    net_padding=0.15,
    max_shapes=64,
    max_neigh_cap=None,
    verbose=False,
):
    """max_neigh-aware elastic bucketing for the dense (reshape) layout. Same return contract
    as ``bucketing_split`` — ``(data_batches, max_pad_batches)`` — but each ``max_pad`` dict
    also carries ``PAD_MAX_NEIGH`` (the per-batch reshape width) and sets
    ``PAD_MAX_N_NEIGHBORS = max_nat * max_neigh`` (the derived dense bond count).

    ``net_padding`` is the UNIFIED target on the net neighbor-padding fraction (the dense
    ``auto_bucket_max_padding``). Because the dense atom/neighbor axes are bound, an adaptive split
    drives BOTH the ``max_neigh`` width bucketing AND the within-width ``nat`` (fake-atom) bucketing
    to meet that target with the FEWEST distinct shapes. ``max_shapes`` is the HARD cap on distinct
    ``(max_nat, max_neigh)`` shapes (== XLA recompiles) — the compile budget. ``n_neigh_buckets``
    (explicit width K) and ``max_n_buckets`` (explicit nat-bucket cap) override the auto search.

    ``verbose`` is accepted for API symmetry with ``bucketing_split`` but is currently
    unused — ``plan_dense_batches`` takes no ``verbose`` argument."""
    from tensorpotential.data.dense_nbr import plan_dense_batches

    nat = np.array([get_number_of_real_atoms(d) for d in data_dict_list])
    max_neigh = np.array([_struct_max_neigh(d) for d in data_dict_list])
    n_neigh = np.array([get_number_of_real_neigh(d) for d in data_dict_list])

    plan, dropped = plan_dense_batches(
        nat,
        max_neigh,
        n_neigh,
        batch_size=batch_size,
        slot_budget=slot_budget,
        n_neigh_buckets=n_neigh_buckets,
        net_padding=net_padding,
        max_shapes=max_shapes,
        max_neigh_cap=max_neigh_cap,
        max_n_buckets=max_n_buckets,
    )
    if dropped:
        logging.warning(
            f"dense_max_neigh_cap={max_neigh_cap}: dropped {len(dropped)} structure(s) with "
            f"per-atom max_neigh above the cap (treated as broken cells)."
        )

    # Report the achieved compile budget (distinct shapes) and net padding; warn if the compile
    # cap forced the net padding above the target.
    if plan:
        real_total = sum(int(pb_n) for pb_n in n_neigh)
        slots = sum(pb["max_nat"] * pb["max_neigh"] for pb in plan)
        achieved_net = 1.0 - real_total / slots if slots else 0.0
        n_shapes = len({(pb["max_nat"], pb["max_neigh"]) for pb in plan})
        logging.info(
            f"dense bucketing: {len(plan)} batches, {n_shapes}/{max_shapes} distinct shapes "
            f"(XLA recompiles), net neighbor padding {achieved_net * 1e2:.1f}% "
            f"(target {net_padding * 1e2:.0f}%)"
        )
        if achieved_net > net_padding + 0.02:
            logging.warning(
                f"dense net padding {achieved_net * 1e2:.1f}% exceeds target {net_padding * 1e2:.0f}% "
                f"because the compile cap max_shapes={max_shapes} binds. Raise dense_max_shapes "
                f"(more XLA recompiles) or batch_size to lower padding."
            )

    data_batches = []
    max_pad_batches = []
    for pb in plan:
        data_batches.append([data_dict_list[i] for i in pb["structure_ind"]])
        max_pad_batches.append(
            {
                constants.PAD_MAX_N_STRUCTURES: pb["max_nstruct"],
                constants.PAD_MAX_N_ATOMS: pb["max_nat"],
                constants.PAD_MAX_NEIGH: pb["max_neigh"],
                constants.PAD_MAX_N_NEIGHBORS: pb["max_nat"] * pb["max_neigh"],
            }
        )
    return data_batches, max_pad_batches


class AbstractDataBuilder(ABC):
    def __init__(self, float_dtype: str = "float64"):
        dtypes = {"float64": np.float64, "float32": np.float32}
        assert float_dtype in dtypes, f"float_dtype must be in {list(dtypes.keys())}"
        if float_dtype == "float32":
            warnings.warn(
                f"DataBuiilder {self.__class__.__name__} wants to build a float32 data"
                f"This is likely undesired."
            )
        float_dtype = dtypes[float_dtype]
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
        dense_nbr: bool = False,
        dense_slot_budget="auto",
        dense_n_neigh_buckets="auto",
        dense_net_padding=0.15,
        dense_max_shapes=64,
        dense_max_neigh_cap=None,
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
        # Mirrors input.yaml::potential::dense_nbr (the model's dense reshape mode).
        # When True, pad_batch emits the per-atom-uniform reshape bond layout
        # (_emit_dense_reshape_layout) and construct_batches routes batching through
        # bucketing_split_dense; single-structure inference uses the same layout via
        # DensePaddingManager in the calculator.
        self.dense_nbr = bool(dense_nbr)
        # Dense (reshape) batching knobs (mode 1, in-RAM). Used by bucketing_split_dense via
        # construct_batches. See knowledge/superpowers/specs/2026-06-24-dense-nbr-bucketing-design.md.
        # dense_net_padding is the UNIFIED net neighbor-padding target (dense auto_bucket_max_padding):
        # it drives BOTH the adaptive max_neigh width bucketing and the within-width nat (fake-atom)
        # bucketing so the combined net lands at the target (the axes are bound in dense). The
        # cli sources it from fit::auto_bucket_max_padding (default 0.15 in dense). dense_max_shapes
        # is the HARD cap on distinct (max_nat, max_neigh) shapes (== XLA recompiles) — the compile
        # budget. dense_n_neigh_buckets (explicit width K) overrides the adaptive search;
        # dense_max_neigh_cap drops over-coordinated (broken) cells.
        self.dense_slot_budget = dense_slot_budget
        self.dense_n_neigh_buckets = dense_n_neigh_buckets
        self.dense_net_padding = dense_net_padding
        self.dense_max_shapes = dense_max_shapes
        self.dense_max_neigh_cap = dense_max_neigh_cap

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

        atomic_mu_i = symbols_to_indices(
            ase_atoms.get_chemical_symbols(), self.elements_map
        )

        # nat_per_specie = defaultdict(int)
        # total_nei_per_specie = defaultdict(int)
        # u_i, n_j = np.unique(ind_i, return_counts=True)

        # for at_i, nb_i in zip(u_i, n_j):
        #     specie_i = atomic_mu_i[at_i]
        #     nat_per_specie[specie_i] += 1
        #     total_nei_per_specie[specie_i] += nb_i

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
            # "nat_per_specie": nat_per_specie,
            # "total_nei_per_specie": total_nei_per_specie,
        }

    def get_sample_dtypes(self):
        return {
            constants.ATOMIC_MU_I: np.int32,
            constants.BOND_VECTOR: self.float_dtype,
            constants.BOND_MU_I: np.int32,
            constants.BOND_MU_J: np.int32,
            constants.BOND_IND_I: np.int32,
            constants.BOND_IND_J: np.int32,
            constants.N_ATOMS_BATCH_REAL: np.int32,
            constants.N_STRUCTURES_BATCH_REAL: np.int32,
            constants.N_NEIGHBORS_REAL: np.int32,
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
            sum_val = sum((int(data_dict[key]) for data_dict in pre_batch_list))
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

        # if self.is_fit_stress:
        # map bonds to structure
        map_bonds_to_structure = []
        for struct_ind, data_dict in enumerate(pre_batch_list):
            map_bonds_to_structure += [struct_ind] * get_number_of_real_neigh(data_dict)
        res_dict[constants.BONDS_TO_STRUCTURE_MAP] = np.array(
            map_bonds_to_structure
        ).astype(np.int32)

        # nat_per_specie = defaultdict(int)
        # total_nei_per_specie = defaultdict(int)
        # for data_dict in pre_batch_list:
        #     nps = data_dict["nat_per_specie"]
        #     tnps = data_dict["total_nei_per_specie"]
        #     for k, v in nps.items():
        #         nat_per_specie[k] += v
        #     for k, v in tnps.items():
        #         total_nei_per_specie[k] += v
        # res_dict["nat_per_specie"] = nat_per_specie
        # res_dict["total_nei_per_specie"] = total_nei_per_specie

        return res_dict

    def get_batch_dtypes(self):
        sample_dtypes = self.get_sample_dtypes()
        order = [
            constants.BOND_VECTOR,
            constants.BOND_MU_I,
            constants.BOND_MU_J,
            constants.ATOMIC_MU_I,
            constants.BOND_IND_I,
            constants.BOND_IND_J,
            constants.N_ATOMS_BATCH_REAL,
            constants.N_STRUCTURES_BATCH_REAL,
            constants.N_NEIGHBORS_REAL,
        ]

        batch_dtypes_dict = {k: sample_dtypes[k] for k in order}

        batch_dtypes_dict[constants.N_ATOMS_BATCH_TOTAL] = np.int32
        batch_dtypes_dict[constants.N_STRUCTURES_BATCH_TOTAL] = np.int32
        batch_dtypes_dict[constants.ATOMS_TO_STRUCTURE_MAP] = np.int32

        # if self.is_fit_stress:
        batch_dtypes_dict[constants.BONDS_TO_STRUCTURE_MAP] = np.int32

        return batch_dtypes_dict

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

        if self.dense_nbr:
            # Dense (reshape) layout: replace the flat bond arrays with the per-atom-uniform
            # [max_nat * max_neigh] layout the reshape compute consumes. Each atom is padded
            # to `width` slots; padded slots get a dummy bond > cutoff (zeroed by the
            # envelope). The fake atom (if any) gets `width` all-dummy slots.
            self._emit_dense_reshape_layout(
                batch, int(max_nat), int(max_pad_dict[constants.PAD_MAX_NEIGH])
            )
            return

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
                a, ((0, pad_nneigh), (0, 0)), mode="constant", constant_values=52.0
            )

            # if self.is_fit_stress:
            key = constants.BONDS_TO_STRUCTURE_MAP
            a = batch[key]
            batch[key] = np.pad(
                a,
                (0, pad_nneigh),
                mode="constant",
                constant_values=max_structs - 1,  # fake structure
            )

    def _emit_dense_reshape_layout(self, batch, max_nat, width):
        """In-place: rebuild the bond arrays in the per-atom-uniform [max_nat * width] reshape
        layout from the REAL bonds. Padded slots (incl. the fake atom's whole block) get a
        dummy bond > cutoff so the envelope zeros them; BOND_IND dummies point at atom 0 (a
        zero-contribution scatter target). Requires n_bonds == max_nat * width exactly."""
        from tensorpotential.data.dense_nbr import (
            build_dense_reshape_perm,
            reorder_bonds_for_reshape,
        )

        n_real = get_number_of_real_neigh(batch)
        ind_i = np.asarray(batch[constants.BOND_IND_I])[:n_real]
        perm, _ = build_dense_reshape_perm(
            ind_i, n_atoms=max_nat, sentinel=n_real, force_max_neigh=width
        )
        dummy = {
            constants.BOND_VECTOR: 52.0,  # > cutoff -> envelope zeros padded slots
            constants.BOND_MU_I: 0,
            constants.BOND_MU_J: 0,
            constants.BOND_IND_I: 0,
            constants.BOND_IND_J: 0,
            constants.BONDS_TO_STRUCTURE_MAP: 0,
        }
        present = {k: np.asarray(batch[k])[:n_real] for k in dummy if k in batch}
        reordered = reorder_bonds_for_reshape(
            present, perm, {k: dummy[k] for k in present}
        )
        batch.update(reordered)


class ReferenceEnergyForcesStressesDataBuilder(AbstractDataBuilder):
    def __init__(
        self,
        normalize_weights=True,
        normalize_force_per_structure=True,
        is_fit_stress=False,
        stress_units=None,
        energy_col=ENERGY_CORRECTED_COL,
        forces_col=FORCES_COL,
        stress_col=STRESS_COL,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.energy_weight: str = constants.DATA_ENERGY_WEIGHTS
        self.force_weight: str = constants.DATA_FORCE_WEIGHTS
        self.virial_weight: str = constants.DATA_VIRIAL_WEIGHTS
        self.normalize_weights = normalize_weights
        self.normalize_force_per_structure = normalize_force_per_structure
        self.is_fit_stress = is_fit_stress
        self.energy_col = energy_col
        self.forces_col = forces_col
        self.stress_col = stress_col

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
            constants.DATA_REFERENCE_ENERGY: np.array(row[self.energy_col]).reshape(
                -1, 1
            ),
            constants.DATA_REFERENCE_FORCES: np.array(row[self.forces_col]),
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
                if self.stress_col in row.index:
                    stress = np.array(row[self.stress_col])
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
                max_val = np.finfo(self.float_dtype).max
                res[constants.DATA_VOLUME] = np.array(max_val).reshape(-1, 1)
            res[constants.DATA_REFERENCE_VIRIAL] *= self.stress_conversion_factor
        return res

    def get_sample_dtypes(self):
        sample_dtypes = {
            constants.DATA_REFERENCE_ENERGY: self.float_dtype,
            constants.DATA_REFERENCE_FORCES: self.float_dtype,
            constants.DATA_STRUCTURE_ID: np.int32,
            self.energy_weight: self.float_dtype,
            self.force_weight: self.float_dtype,
        }

        if self.is_fit_stress:
            sample_dtypes[constants.DATA_REFERENCE_VIRIAL] = self.float_dtype
            sample_dtypes[self.virial_weight] = self.float_dtype
            sample_dtypes[constants.DATA_VOLUME] = self.float_dtype

        return sample_dtypes

    def join_to_batch(self, pre_batch_list: list):
        res_dict = {}
        for key in [constants.DATA_REFERENCE_ENERGY]:
            data_list = [np.float64(data_dict[key]) for data_dict in pre_batch_list]
            res_dict[key] = np.array(data_list).reshape(-1, 1).astype(self.float_dtype)

        for key in [constants.DATA_STRUCTURE_ID]:
            data_list = [np.int32(data_dict[key]) for data_dict in pre_batch_list]
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

    def get_batch_dtypes(self):
        return self.get_sample_dtypes()

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
                max_val = np.finfo(self.float_dtype).max
                k = constants.DATA_VOLUME
                batch[k] = np.pad(
                    batch[k],
                    ((0, pad_nstruct), (0, 0)),
                    mode="constant",
                    constant_values=max_val,
                )

    def postprocess_dataset(self, batches):
        if self.normalize_weights:
            energy_weight_sum = np.sum([np.sum(b[self.energy_weight]) for b in batches])
            forces_weight_sum = np.sum([np.sum(b[self.force_weight]) for b in batches])

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
    max_padding_fraction: float = 0.3,
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
        dense_builder = next(
            (db for db in data_builders if getattr(db, "dense_nbr", False)), None
        )
        if dense_builder is not None:
            pre_batch_groups, max_pad_batches = bucketing_split_dense(
                data_dict_list,
                batch_size=batch_size,
                max_n_buckets=max_n_buckets,
                slot_budget=dense_builder.dense_slot_budget,
                n_neigh_buckets=dense_builder.dense_n_neigh_buckets,
                net_padding=dense_builder.dense_net_padding,
                max_shapes=dense_builder.dense_max_shapes,
                max_neigh_cap=dense_builder.dense_max_neigh_cap,
                verbose=verbose,
            )
        else:
            pre_batch_groups, max_pad_batches = bucketing_split(
                data_dict_list,
                batch_size=batch_size,
                max_n_buckets=max_n_buckets,
                verbose=verbose,
                max_padding_fraction=max_padding_fraction,
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
        if dense_builder is None and external_max_nneigh is not None:
            current_max_nneigh = np.max(
                [d[constants.PAD_MAX_N_NEIGHBORS] for d in max_pad_batches]
            )
            if current_max_nneigh > external_max_nneigh:
                external_max_nneigh = current_max_nneigh
            for d in max_pad_batches:
                d[constants.PAD_MAX_N_NEIGHBORS] = external_max_nneigh
        if dense_builder is None and external_max_nat is not None:
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
    df_or_ase_atoms_list,
    data_builders,
    iterator_func,
    max_workers,
    chunksize=2500,
):
    logging.info(f"Parallel data processing: {max_workers=}, {chunksize=}")
    chunks = (
        df_or_ase_atoms_list.iloc[i * chunksize : (i + 1) * chunksize]
        for i in range(len(df_or_ase_atoms_list) // chunksize)
    )

    # Add a tail chunk if there's a remainder
    if len(df_or_ase_atoms_list) % chunksize != 0:
        tail_start = (len(df_or_ase_atoms_list) // chunksize) * chunksize
        tail_chunk = [
            df_or_ase_atoms_list.iloc[tail_start:]
        ]  # Wrap in a list to make it iterable
        chunks = itertools.chain(
            chunks, tail_chunk
        )  # Lazily chain the main chunks with the tail chunk

    data_dict_list = []
    # Use ProcessPoolExecutor to parallelize row processing
    with ProcessPoolExecutor(
        max_workers=max_workers,
        initializer=start_thread_to_terminate_when_parent_process_dies,
        initargs=(os.getpid(),),
    ) as executor:
        with tqdm(total=len(df_or_ase_atoms_list), mininterval=MININTERVAL) as pbar:
            for chunk in chunks:
                # Submit tasks for each row in df_or_ase_atoms_list
                futures = [
                    executor.submit(process_row, row_ind, row, data_builders)
                    for row_ind, row in chunk.iterrows()
                ]

                # Collect results as they are completed
                for future in futures:
                    data_dict = future.result()
                    data_dict_list.append(data_dict)
                    pbar.update(1)
    assert len(data_dict_list) == len(
        df_or_ase_atoms_list
    ), "Data loss during parallel processing"
    return data_dict_list


def process_row(row_ind, row, data_builders):
    data_dict = {}  # managed by reference
    for data_builder in data_builders:
        data_dict.update(
            data_builder.extract_from_row(row, **{constants.DATA_STRUCTURE_ID: row_ind})
        )
    return data_dict
