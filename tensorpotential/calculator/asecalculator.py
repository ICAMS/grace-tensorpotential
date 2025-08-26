from __future__ import annotations

import numpy as np
import time
import bisect

from tensorflow.data import Dataset

from typing import Any, Dict, Tuple, Optional, List
from itertools import combinations_with_replacement
from ase.calculators.calculator import Calculator, all_changes

from tensorpotential.tensorpot import TensorPotential
from tensorpotential.tpmodel import extract_cutoff_and_elements, extract_cutoff_matrix
from tensorpotential.data.databuilder import (
    construct_batches,
    GeometricalDataBuilder,
    get_number_of_real_atoms,
    get_number_of_real_neigh,
    AbstractDataBuilder,
)
from tensorpotential import constants


class PaddingManager:
    """
    Manages padding for batches of atomic structures to ensure consistent input sizes for models.

    This class handles padding based on either a fraction of neighbors or a fixed number of atoms,
    or a combination of both. It maintains a history of padding bounds to reuse existing bounds
    if possible and to potentially reduce padding if it becomes excessive.

    Args:
        data_builders (List[DataBuilderProtocol]): A list of DataBuilder objects that handle
            data extraction, batching, and padding for different aspects of the atomic structures.
        pad_neighbors_fraction (Optional[float]):  Fraction of real neighbors to add as padding.
            Must be between 0 and 1 inclusive if provided. Defaults to 0.10. If None, neighbor padding
            is not based on a fraction.
        pad_atoms_number (Optional[int]): Number of atoms to add as padding. Must be a positive integer
            if provided. Defaults to 10. If None, atom padding is not based on a fixed number.
        max_number_reduction_recompilation (Optional[int]): Maximum number of times padding can be
            reduced and recompilation triggered. If None, padding reduction is disabled. Defaults to 3.
        debug_padding (bool): Enables debug logging for padding operations. Defaults to False.
    """

    def __init__(
        self,
        data_builders: List[AbstractDataBuilder],
        pad_neighbors_fraction: Optional[float] = 0.10,
        pad_atoms_number: Optional[int] = 10,
        max_number_reduction_recompilation: Optional[int] = 3,
        debug_padding: bool = False,
    ):

        self.padding_fraction_history = []
        if pad_neighbors_fraction is not None:
            if not 0 < pad_neighbors_fraction <= 1:
                raise ValueError(
                    f"pad_neighbors_fraction must be a fraction between 0 and 1, but got {pad_neighbors_fraction}"
                )

        if pad_atoms_number is not None:
            if not isinstance(pad_atoms_number, int):
                raise TypeError(
                    f"pad_atoms_number must be an integer, but got {type(pad_atoms_number)}"
                )
            if not pad_atoms_number > 0:
                raise ValueError(
                    f"pad_atoms_number must be larger than 0, but got {pad_atoms_number}"
                )

        self.data_builders: List[AbstractDataBuilder] = data_builders
        self.pad_neighbors_fraction: Optional[float] = pad_neighbors_fraction
        self.pad_atoms_number: Optional[int] = pad_atoms_number
        self.max_number_reduction_recompilation: Optional[int] = (
            max_number_reduction_recompilation
        )
        self.number_reduction_recompilation: int = 0
        self.debug_padding: bool = debug_padding

        # List of tuples (max_atoms, max_neighbors)
        self.padding_bounds: List[Tuple[int, int]] = []

    def find_upper_padding_bound(
        self, max_nat: int, max_nneigh: int
    ) -> Optional[Tuple[int, int]]:
        """
        Finds the smallest existing padding bound in history that is greater than or equal to
        the current required bounds (max_nat, max_nneigh).

        Args:
            max_nat (int): The maximum number of atoms in the current batch.
            max_nneigh (int): The maximum number of neighbors in the current batch.

        Returns:
            Optional[Tuple[int, int]]: An existing upper padding bound (max_atoms, max_neighbors)
                                        from history if found, otherwise None.
        """
        current_bound: Tuple[int, int] = (max_nat, max_nneigh)
        for upper_bound_tuple in self.padding_bounds:
            if (
                upper_bound_tuple[0] >= current_bound[0]
                and upper_bound_tuple[1] >= current_bound[1]
            ):  # Component-wise comparison (atoms and neighbors)
                return upper_bound_tuple
        return None  # Explicitly return None if no suitable bound is found

    def get_padded_bound(self, nreal_atoms: int, nreal_neigh: int) -> Tuple[int, int]:
        """
        Calculates the padded bounds for atoms and neighbors based on the configured padding parameters.

        Args:
            nreal_atoms (int): The number of real atoms in the current structure.
            nreal_neigh (int): The number of real neighbors in the current structure.

        Returns:
            Tuple[int, int]: The calculated padded bounds as a tuple (max_atoms, max_neighbors).
        """
        current_max_neighbors: int
        if self.pad_neighbors_fraction is not None:
            current_max_neighbors = max(
                256,
                int(np.ceil(nreal_neigh + nreal_neigh * self.pad_neighbors_fraction)),
            )
        else:
            # No padding for neighbors if fraction is None
            current_max_neighbors = nreal_neigh

        current_max_atoms: int
        if self.pad_atoms_number is not None:
            current_max_atoms = int(nreal_atoms + self.pad_atoms_number)
        else:
            current_max_atoms = nreal_atoms  # No padding for atoms if number is None

        return current_max_atoms, current_max_neighbors

    def get_data(self, atoms) -> Dict[str, Any]:
        """
        Processes the input atomic structure (atoms) to extract data, batch it, and apply padding.

        This method orchestrates the data processing pipeline:
        1. Extracts data features from the atomic structure using DataBuilders.
        2. Joins the extracted data into a batch format.
        3. Determines and applies padding to the batch based on the current padding strategy
           and history.

        Args:
            atoms: An atomic structure object (e.g., ASE Atoms object).

        Returns:
            Dict[str, Any]: The processed and padded batch of data.
        """
        # Stage 1: Data Extraction
        data_dict: Dict[str, Any] = {}
        for data_builder in self.data_builders:
            data_dict.update(data_builder.extract_from_ase_atoms(atoms))

        # Stage 2: Batching
        batch_group: List[Dict[str, Any]] = [
            data_dict
        ]  # Consider if batch_group is always of size 1
        batch: Dict[str, Any] = {}
        for data_builder in self.data_builders:
            batch.update(data_builder.join_to_batch(batch_group))

        # Stage 3: Determine Real Number of Atoms and Neighbors
        nreal_atoms: int = get_number_of_real_atoms(batch)
        nreal_neigh: int = get_number_of_real_neigh(batch)

        if self.debug_padding:
            print(f"nreal_atoms={nreal_atoms}, nreal_neigh={nreal_neigh}")

        # Stage 4: Padding
        if self.pad_atoms_number and self.pad_neighbors_fraction:
            if not self.padding_bounds:
                # FIRST TIME - NO current_max_neighbors and current_max_atoms, setup with padded threshold
                upper_bound = self.get_padded_bound(nreal_atoms, nreal_neigh)
                # insert into ordered list
                if self.debug_padding:
                    print(f"Adding initial padding bound: {upper_bound}")
                bisect.insort(self.padding_bounds, upper_bound)
            else:
                # extract padding bounds from self.padding_bounds
                upper_bound: Optional[Tuple[int, int]] = self.find_upper_padding_bound(
                    nreal_atoms, nreal_neigh
                )
                if self.debug_padding:
                    print(f"extract padding bounds: {upper_bound}")
                if upper_bound is None:
                    upper_bound = self.get_padded_bound(nreal_atoms, nreal_neigh)
                    # insert into ordered list
                    if self.debug_padding:
                        print(f"Adding new maximum padding bound: {upper_bound}")
                    # if upper_bound not in self.padding_bounds:
                    bisect.insort(self.padding_bounds, upper_bound)
                else:
                    # check if too large padding
                    if self.is_large_padding(nreal_atoms, nreal_neigh, upper_bound):
                        if self.max_number_reduction_recompilation is not None and (
                            self.number_reduction_recompilation
                            < self.max_number_reduction_recompilation
                        ):
                            self.number_reduction_recompilation += 1
                            upper_bound = self.get_padded_bound(
                                nreal_atoms, nreal_neigh
                            )
                            # insert into ordered list
                            if self.debug_padding:
                                print(
                                    f"Reducing padding and adding new bound: {upper_bound}"
                                )
                            # if upper_bound not in self.padding_bounds:
                            bisect.insort(self.padding_bounds, upper_bound)
                        elif self.debug_padding:
                            print(
                                f"Padding reduction skipped or max reductions reached. Keeping bound: {upper_bound}"
                            )
                    elif self.debug_padding:
                        print(f"Using existing padding bound: {upper_bound}")

            pad_max_n_atoms, pad_max_n_neighbors = upper_bound

            batch_max_pad_dict: Dict[str, int] = {
                constants.PAD_MAX_N_STRUCTURES: 1,
                constants.PAD_MAX_N_ATOMS: pad_max_n_atoms,
                constants.PAD_MAX_N_NEIGHBORS: pad_max_n_neighbors,
            }

            padding_fraction = (pad_max_n_atoms + pad_max_n_neighbors) / (
                nreal_atoms + nreal_neigh
            ) - 1
            if self.debug_padding:
                print("padding_fraction={:.2f}".format(padding_fraction))
                self.padding_fraction_history.append(padding_fraction)

            for data_builder in self.data_builders:
                data_builder.pad_batch(batch, batch_max_pad_dict)

        return batch

    def is_large_padding(self, nreal_atoms, nreal_neigh, upper_bound):
        npad_atoms = upper_bound[0] - nreal_atoms
        npad_neigh = upper_bound[1] - nreal_neigh
        padding_fraction = (npad_atoms + npad_neigh) / (nreal_atoms + nreal_neigh)
        return (
            npad_atoms + npad_neigh > 1000
        ) and padding_fraction > self.pad_neighbors_fraction


class TPCalculator(Calculator):
    """
    Atomic Simulation Environment (ASE) calculator for TensorPotential models.

    Args:
        model (Any): The TensorPotential model. This can be:
            - A path to a tf.saved_model file (string).
            - An instance of a TPModel
        cutoff (float, optional): The cutoff radius for potential interactions (default: 5).
        pad_neighbors_fraction (float, optional): Fraction by which to extend
            the neighbor list with fake neighbors (between 0 and 1, for XLA compiled models).
        pad_atoms_number (int, optional): number of  fake atoms to pad ( for XLA compiled models)
        **kwargs: Additional keyword arguments passed to the base Calculator class.
    """

    implemented_properties = ["energy", "forces", "stress", "free_energy"]

    def __init__(
        self,
        model: list[Any] | Any,
        cutoff: float = None,
        pad_neighbors_fraction: float | None = 0.05,
        pad_atoms_number: int | None = 1,
        min_dist=None,
        extra_properties: list[str] = None,
        truncate_extras_by_natoms: bool = False,
        max_number_reduction_recompilation: int | None = 2,
        debug_padding=False,
        float_dtype: str = "float64",
        **kwargs,
    ):
        Calculator.__init__(self, **kwargs)
        self.data = None
        self.cutoff = cutoff

        self.eval_time = 0
        self.basis = None

        self.min_dist = min_dist  # minimal distance
        self.compute_properties = ["energy", "forces", "free_energy", "stress"]

        self.extra_properties = None
        if extra_properties is not None:
            self.extra_properties = extra_properties
        self.truncate_extras_by_natoms = truncate_extras_by_natoms

        assert model is not None, ValueError(f'"model" parameter is not provided')
        self.models = []
        self.data_keys = []
        if isinstance(model, list):
            for modeli in model:
                if isinstance(modeli, str):
                    modeli = TensorPotential.load_model(modeli)
                    self.models.append(modeli)
                    i_data_keys = modeli.signatures["serving_default"]._arg_keywords
                elif hasattr(modeli, "compute"):
                    assert callable(modeli.compute)
                    self.models.append(modeli)
                    i_data_keys = [k for k, v in modeli.compute_specs.items()]
                else:
                    raise ValueError(f"model type is not recognized")
                if len(self.data_keys) == 0:
                    self.data_keys = i_data_keys
                else:
                    assert len(set(self.data_keys).intersection(i_data_keys)) == len(
                        self.data_keys
                    ), "Models have inconsistent data keys"
        else:
            if isinstance(model, str):
                model = TensorPotential.load_model(model)
                self.data_keys = model.signatures["serving_default"]._arg_keywords
            elif hasattr(model, "compute"):
                assert callable(model.compute)
                self.data_keys = [k for k, v in model.compute_specs.items()]
            else:
                raise ValueError(f"model type is not recognized")
            self.models.append(model)

        cutoffs, element_maps, cutoff_matrices = [], [], []
        for model in self.models:
            cutoff, element_map_symbols, element_map_index = (
                extract_cutoff_and_elements(model.instructions)
            )
            cutoffs.append(cutoff)
            element_maps.append(
                {k: v for k, v in zip(element_map_symbols, element_map_index)}
            )
            cutoff_matrix = extract_cutoff_matrix(model.instructions)
            if cutoff_matrix is not None:
                cutoff_matrices.append(cutoff_matrix)
        cutoff = np.max(cutoffs)
        assert all(
            [ems == element_maps[0]] for ems in element_maps
        )  # check that all maps are identical
        self.element_map = element_maps[0]

        actual_cutoff_matrices = [m for m in cutoff_matrices if m is not None]

        if len(actual_cutoff_matrices) == 0:
            self.cutoff_dict = None
            if cutoff > 0:
                if self.cutoff is not None and self.cutoff != cutoff:
                    print(
                        f"Cutoff of the potential {cutoff} A is different from calculator's {self.cutoff} A. "
                        f"Using the value from the potential."
                    )
                self.cutoff = cutoff
            else:
                print(
                    f"Couldn't extract cutoff value from the model. Using the value from calculator: {self.cutoff}A"
                )
        else:
            assert len(set([m.shape for m in actual_cutoff_matrices])) == 1
            # get one max matrix over possibly many
            nelems = len(self.element_map)
            matrix = np.max(actual_cutoff_matrices, axis=0).reshape(nelems, nelems)
            # get single max value
            self.cutoff = np.max(matrix)
            # construct dict
            all_bond_comb = combinations_with_replacement(np.arange(nelems), 2)
            inv_element_map = {v: k for k, v in self.element_map.items()}

            self.cutoff_dict = {}
            for comb in all_bond_comb:
                el0, el1 = inv_element_map[comb[0]], inv_element_map[comb[1]]
                self.cutoff_dict[(el0, el1)] = matrix[comb[0], comb[1]]

        self.geom_data_builder = GeometricalDataBuilder(
            elements_map=self.element_map,
            cutoff=self.cutoff,
            cutoff_dict=self.cutoff_dict,
            float_dtype=float_dtype,
        )
        self.data_builders = [self.geom_data_builder]
        if constants.ATOMIC_MAGMOM in self.data_keys:
            from tensorpotential.experimental.mag.databuilder import MagMomDataBuilder

            self.data_builders.append(MagMomDataBuilder())

        self.padding_manager = PaddingManager(
            data_builders=self.data_builders,
            pad_neighbors_fraction=pad_neighbors_fraction,
            pad_atoms_number=pad_atoms_number,
            max_number_reduction_recompilation=max_number_reduction_recompilation,
            debug_padding=debug_padding,
        )

    def get_data(self, atoms):
        current_symbs = atoms.symbols.species()
        assert all([x in self.element_map for x in current_symbs]), (
            f"This model is configured to process "
            f"the following elements only: {list(self.element_map.keys())}, but the structure "
            f"contains {current_symbs}"
        )

        data = self.padding_manager.get_data(atoms)

        self.data = data

        # filtering
        data = {k: v for k, v in data.items() if k in self.data_keys}
        self.current_min_dist = np.min(
            np.linalg.norm(data[constants.BOND_VECTOR], axis=1)
        )
        if self.min_dist is not None and self.current_min_dist < self.min_dist:
            raise RuntimeError(
                f"Minimal bond distance {self.current_min_dist} is smaller than {self.min_dist}"
            )
        self.data = Dataset.from_tensors(data).get_single_element()

    def find_upper_padding_bound(self, max_nat):
        # find first entry, when max_nat<=
        for ind, upper_bound_tuple in enumerate(self.padding_history):
            if upper_bound_tuple[0] >= max_nat:
                break
        # shift further, until same max_nat
        while (
            ind + 1 < len(self.padding_history)
            and self.padding_history[ind + 1][0] == upper_bound_tuple[0]
        ):
            ind += 1
        upper_bound_tuple = self.padding_history[ind]
        return upper_bound_tuple

    def calculate(
        self,
        atoms=None,
        properties=("energy", "forces", "stress"),
        system_changes=all_changes,
    ):
        Calculator.calculate(self, atoms, properties, system_changes)

        self.forces = np.empty((len(atoms), 3))
        self.stress = np.empty((1, 3, 3))
        self.energy = 0.0
        results = {}

        self.get_data(atoms)

        t0 = time.perf_counter()

        self.outputs = []
        energy_list = []
        forces_list = []
        stress_list = []
        if self.extra_properties is not None:
            extras = {prop: [] for prop in self.extra_properties}
        else:
            extras = {}
        n_model = len(self.models)

        for model in self.models:
            output = model.compute(self.data)
            self.outputs.append(output)

        for output in self.outputs:
            if "energy" in self.compute_properties:
                e = output.get(constants.PREDICT_TOTAL_ENERGY, None)
                if e is None:
                    energy_list.append(np.zeros((1, 1)))
                else:
                    energy_list.append(e.numpy())

            if "forces" in self.compute_properties:
                forces = output.get(constants.PREDICT_FORCES, None)
                if forces is None:
                    forces = np.zeros((len(atoms), 3))
                else:
                    forces = forces.numpy()
                forces_list.append(forces)

            if "stress" in self.compute_properties:
                stress = output.get(constants.PREDICT_VIRIAL, None)
                if stress is None or atoms.get_cell().rank == 0:
                    stress = np.zeros((6,))
                else:
                    stress = (
                        -stress.numpy().reshape((6,))[[0, 1, 2, 5, 4, 3]]
                        / atoms.get_volume()
                    )
                stress_list.append(stress)
            for prop in extras:
                res = output.get(prop)
                if res is None:
                    extras[prop].append(np.zeros((1, 1)))
                else:
                    res = res.numpy()
                    if self.truncate_extras_by_natoms:
                        extras[prop].append(res[: len(atoms)])
                    else:
                        extras[prop].append(res)
        self.eval_time = time.perf_counter() - t0

        self.energy = np.mean(energy_list, axis=0).flatten()[0]

        # ensure only real atoms have forces
        self.forces = np.mean(forces_list, axis=0)[: len(atoms)]
        self.stress = np.mean(stress_list, axis=0)

        results["energy"] = self.energy
        results["free_energy"] = results["energy"]
        results["forces"] = self.forces
        results["stress"] = self.stress

        for k, v in extras.items():
            results[k] = np.mean(v, axis=0)

        if n_model > 1:
            results["energy_std"] = np.std(energy_list, axis=0).flatten()[0]
            results["forces_std"] = np.std(forces_list, axis=0)[: len(atoms)]
            results["stress_std"] = np.std(stress_list, axis=0)

        self.results = results
