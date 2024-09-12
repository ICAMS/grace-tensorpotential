from __future__ import annotations


import numpy as np
import time

from tensorflow.data import Dataset

from typing import Any
from ase.calculators.calculator import Calculator, all_changes

from tensorpotential.tensorpot import TensorPotential
from tensorpotential.tpmodel import extract_cutoff_and_elements, TPModel
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
        pad_neighbors_fraction: float = None,
        pad_atoms_number: int = None,
        min_dist=None,
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

        cutoffs, element_maps = [], []
        for model in self.models:
            cutoff, element_map_symbols, element_map_index = (
                extract_cutoff_and_elements(model.instructions)
            )
            cutoffs.append(cutoff)
            element_maps.append(
                {k: v for k, v in zip(element_map_symbols, element_map_index)}
            )
        cutoff = np.max(cutoffs)
        assert all(
            [ems == element_maps[0]] for ems in element_maps
        )  # check that all maps are identical
        self.element_map = element_maps[0]

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
        self.data_builders = [
            GeometricalDataBuilder(
                elements_map=self.element_map,
                cutoff=self.cutoff,
            )
        ]
        if constants.ATOMIC_MAGMOM in self.data_keys:
            from tensorpotential.experimental.mag.databuilder import MagMomDataBuilder

            self.data_builders.append(MagMomDataBuilder())

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
        properties=["energy", "forces", "stress"],
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
        torques_list = []
        n_model = len(self.models)

        for model in self.models:
            output = model.compute(self.data)
            outputs.append(output)

        for output in outputs:
            if "energy" in self.compute_properties:
                energy_list.append(output.get(constants.PREDICT_TOTAL_ENERGY).numpy())

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
            if "torques" in self.compute_properties:
                from tensorpotential.experimental.mag import constants as constants_mag

                torques = output.get(constants_mag.PREDICT_TORQUES)
                torques_list.append(torques)
        self.eval_time = time.perf_counter() - t0

        # energy_list = np.array(energy_list)
        # forces_list = np.array(forces_list)
        # stress_list = np.array(stress_list)

        self.energy = np.mean(energy_list, axis=0).flatten()[0]

        # ensure only real atoms have forces
        self.forces = np.mean(forces_list, axis=0)[: len(atoms)]
        self.stress = np.mean(stress_list, axis=0)

        results["energy"] = self.energy
        results["free_energy"] = results["energy"]
        results["forces"] = self.forces
        results["stress"] = self.stress

        if "torques" in self.compute_properties:
            self.torques = np.mean(torques_list, axis=0)
            results["torques"] = self.torques

        if n_model > 1:
            results["energy_std"] = np.std(energy_list, axis=0).flatten()[0]
            results["forces_std"] = np.std(forces_list, axis=0)
            results["stress_std"] = np.std(stress_list, axis=0)

        self.results = results
