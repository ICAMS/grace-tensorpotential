from __future__ import annotations

import numpy as np
import time

from tensorflow.data import Dataset

from typing import Any
from itertools import combinations_with_replacement
from ase.calculators.calculator import Calculator, all_changes

from tensorpotential.tensorpot import TensorPotential
from tensorpotential.tpmodel import extract_cutoff_and_elements, extract_cutoff_matrix
from tensorpotential.data.databuilder import construct_batches, GeometricalDataBuilder
from tensorpotential import constants


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
        **kwargs,
    ):
        Calculator.__init__(self, **kwargs)
        self.cutoff = cutoff
        # self.model_type = None
        if pad_neighbors_fraction is not None:
            assert (
                0 < pad_neighbors_fraction <= 1
            ), f"pad_neighbors_fraction must be a fraction between 0 and 1"
        self.pad_neighbors_fraction = pad_neighbors_fraction
        self.current_max_neighbors = None

        if pad_atoms_number is not None:
            assert 0 < pad_atoms_number, f"pad_atoms_number must be larger than 0"
            assert isinstance(
                pad_atoms_number, int
            ), f"pad_atoms_number must be an integer"
        self.pad_atoms_number = pad_atoms_number
        self.current_max_atoms = None
        if self.pad_atoms_number is not None:
            assert (
                self.pad_neighbors_fraction is not None
            ), f"Padding natoms only is not supported"

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

        self.data_builders = [
            GeometricalDataBuilder(
                elements_map=self.element_map,
                cutoff=self.cutoff,
                cutoff_dict=self.cutoff_dict,
            )
        ]


    def get_data(self, atoms):
        current_symbs = atoms.symbols.species()
        assert all([x in self.element_map for x in current_symbs]), (
            f"This model is configured to process "
            f"the following elements only: {list(self.element_map.keys())}, but the structure "
            f"contains {current_symbs}"
        )

        if self.pad_neighbors_fraction is not None or self.pad_atoms_number is not None:
            if self.current_max_neighbors is None or self.current_max_atoms is None:
                data, stats = construct_batches(
                    [atoms],
                    self.data_builders,
                    verbose=False,
                    batch_size=1,
                    max_n_buckets=1,
                    return_padding_stats=True,
                    gc_collect=False,
                )
                max_nneigh = stats["nreal_neigh"]
                max_at = stats["nreal_atoms"]

                self.current_max_neighbors = np.ceil(
                    max_nneigh + max_nneigh * self.pad_neighbors_fraction
                ).astype(int)

                if self.pad_atoms_number is not None:
                    self.current_max_atoms = int(max_at + self.pad_atoms_number)
                else:
                    self.current_max_atoms = max_at

            # if self.current_max_neighbors is not None or self.current_max_atoms is not None:
            data, stats = construct_batches(
                [atoms],
                self.data_builders,
                verbose=False,
                batch_size=1,
                max_n_buckets=1,
                return_padding_stats=True,
                external_max_nneigh=self.current_max_neighbors,
                external_max_nat=self.current_max_atoms,
                gc_collect=False,
            )
            max_nneigh = stats["nreal_neigh"]
            if max_nneigh > self.current_max_neighbors:
                self.current_max_neighbors = np.ceil(
                    max_nneigh + max_nneigh * self.pad_neighbors_fraction
                ).astype(int)

            max_at = stats["nreal_atoms"]
            if max_at > self.current_max_atoms:
                self.current_max_atoms = np.array(
                    max_at + self.pad_atoms_number
                ).astype(int)
        else:
            data = construct_batches(
                [atoms],
                self.data_builders,
                verbose=False,
                batch_size=1,
                max_n_buckets=None,
                gc_collect=False,
            )

        data = {k: v for k, v in data[0].items() if k in self.data_keys}
        self.current_min_dist = np.min(
            np.linalg.norm(data[constants.BOND_VECTOR], axis=1)
        )
        if self.min_dist is not None and self.current_min_dist < self.min_dist:
            raise RuntimeError(
                f"Minimal bond distance {self.current_min_dist} is smaller than {self.min_dist}"
            )
        self.data = Dataset.from_tensors(data).get_single_element()

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

        outputs = []
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
            outputs.append(output)

        for output in outputs:
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
                    stress = -stress.numpy()[[0, 1, 2, 5, 4, 3]] / atoms.get_volume()
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
            results["forces_std"] = np.std(forces_list, axis=0)
            results["stress_std"] = np.std(stress_list, axis=0)

        self.results = results
