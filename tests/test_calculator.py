import os
import shutil

import pytest

from tensorpotential.calculator.asecalculator import PaddingManager
from tensorpotential.data.databuilder import GeometricalDataBuilder
from tensorflow.data import Dataset

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

import numpy as np

from tensorpotential.calculator import TPCalculator
from tensorpotential.instructions import load_instructions
from tensorpotential import TPModel
from tensorflow import float64

from ase.build import bulk
from ase import Atoms


def test_calculator_model():
    ins = load_instructions("model_grace.yaml")
    model = TPModel(ins)
    model.build(float64)
    calc = TPCalculator(model=model)

    np.random.seed(322)
    s = bulk("W", cubic=True) * (2, 2, 2)
    s.positions += np.random.uniform(-0.2, 0.2, size=(len(s), 3))
    s.calc = calc
    e0 = s.get_potential_energy()
    print(e0)
    f0 = s.get_forces()
    print(f0)
    st0 = s.get_stress()
    print(st0)
    st0_num = calc.calculate_numerical_stress(s)
    assert np.allclose(st0, st0_num, rtol=1e-5)

    shutil.rmtree("./test_calculator_model_tmp", ignore_errors=True)
    model.save_model("./test_calculator_model_tmp")

    calc1 = TPCalculator(model="./test_calculator_model_tmp")

    np.random.seed(322)
    s = bulk("W", cubic=True) * (2, 2, 2)
    s.positions += np.random.uniform(-0.2, 0.2, size=(len(s), 3))
    s.calc = calc1
    e1 = s.get_potential_energy()
    print(e1)
    f1 = s.get_forces()
    print(f1)
    st1 = s.get_stress()
    print(st1)
    st1_num = calc.calculate_numerical_stress(s)
    assert np.allclose(st1, st1_num, rtol=1e-5)

    assert np.allclose(e0, e1)
    assert np.allclose(f0, f1)
    assert np.allclose(st0, st1)
    shutil.rmtree("./test_calculator_model_tmp", ignore_errors=True)


def test_calculator_with_fake_neighbors():
    calc = TPCalculator(
        model="./test_calculator_model", pad_neighbors_fraction=0.25, pad_atoms_number=1
    )
    calc1 = TPCalculator(
        model="./test_calculator_model",
        pad_neighbors_fraction=None,
        pad_atoms_number=None,
    )
    # print(f"{calc.padding_manager.current_max_neighbors=} at the start")
    np.random.seed(322)
    s = bulk("W", cubic=True) * (2, 2, 2)
    s.positions += np.random.uniform(-0.2, 0.2, size=(len(s), 3))
    s.calc = calc
    e = s.get_potential_energy()
    print(e)
    f = s.get_forces()
    print(f)
    assert np.shape(f) == (len(s), 3)
    st = s.get_stress()
    print(st)
    # print(f"{calc.padding_manager.current_max_neighbors=} at the end")
    st_num = calc.calculate_numerical_stress(s)
    assert np.allclose(st, st_num, rtol=1e-5)

    s.calc = calc1
    e1 = s.get_potential_energy()
    f1 = s.get_forces()
    st1 = s.get_stress()
    assert np.shape(f1) == (len(s), 3)

    assert np.allclose(e, e1, rtol=1e-5)
    assert np.allclose(f, f1, rtol=1e-5)
    assert np.allclose(st, st1, rtol=1e-5)


def test_dynamic_padding():
    calc = TPCalculator(
        model="./test_calculator_model",
        pad_neighbors_fraction=0.25,
        pad_atoms_number=1,
        debug_padding=True,
    )

    assert len(calc.padding_manager.padding_bounds) == 0
    s1 = bulk("W", a=3.165, cubic=True)
    s1.calc = calc
    e1 = s1.get_potential_energy()
    print(f"{e1=}")
    assert len(calc.padding_manager.padding_bounds) == 1
    # print(
    #     calc.padding_manager.current_max_atoms,
    #     calc.padding_manager.current_max_neighbors,
    # )

    s2 = bulk("W", a=3.165, cubic=True) * (2, 2, 2)
    s2.calc = calc
    e2 = s2.get_potential_energy()
    print(f"{e2=}")
    assert e1 != e2
    assert len(calc.padding_manager.padding_bounds) == 2
    # print(calc.current_max_atoms, calc.current_max_neighbors)

    s3 = bulk("W", a=2.5, cubic=True)
    s3.calc = calc
    e3 = s3.get_potential_energy()
    print(f"{e3=}")
    assert e1 != e3
    assert len(calc.padding_manager.padding_bounds) == 2
    # print(calc.current_max_atoms, calc.current_max_neighbors)

    s4 = bulk("W", a=2.75, cubic=True)
    s4.calc = calc
    e4 = s4.get_potential_energy()
    print(f"{e4=}")
    assert e4 != e3
    assert len(calc.padding_manager.padding_bounds) == 2
    # print(calc.current_max_atoms, calc.current_max_neighbors)


def test_dynamic_padding_reducing():
    calc = TPCalculator(
        model="./test_calculator_model", pad_neighbors_fraction=0.25, pad_atoms_number=5
    )

    assert len(calc.padding_manager.padding_bounds) == 0

    s2 = bulk("W", a=3.165, cubic=True) * (2, 2, 2)
    s2.calc = calc
    e2 = s2.get_potential_energy()
    print(f"{e2=}")
    assert len(calc.padding_manager.padding_bounds) == 1
    # print(calc.current_max_atoms, calc.current_max_neighbors)
    print("padding history=", calc.padding_manager.padding_bounds)

    s1 = bulk("W", a=3.165, cubic=True)
    s1.calc = calc
    e1 = s1.get_potential_energy()
    print(f"{e1=}")
    assert e1 != e2
    assert len(calc.padding_manager.padding_bounds) == 2
    # print(calc.current_max_atoms, calc.current_max_neighbors)
    print("padding history=", calc.padding_manager.padding_bounds)


def test_dynamic_padding_reducing_limit():
    calc = TPCalculator(
        model="./test_calculator_model",
        pad_neighbors_fraction=0.25,
        pad_atoms_number=5,
        max_number_reduction_recompilation=0,
    )

    assert len(calc.padding_manager.padding_bounds) == 0

    s2 = bulk("W", a=3.165, cubic=True) * (2, 2, 2)
    s2.calc = calc
    e2 = s2.get_potential_energy()
    print(f"{e2=}")
    assert len(calc.padding_manager.padding_bounds) == 1
    # print(calc.current_max_atoms, calc.current_max_neighbors)
    print("padding history=", calc.padding_manager.padding_bounds)

    s1 = bulk("W", a=3.165, cubic=True)
    s1.calc = calc
    e1 = s1.get_potential_energy()
    print(f"{e1=}")
    assert e1 != e2
    assert len(calc.padding_manager.padding_bounds) == 1
    # print(calc.current_max_atoms, calc.current_max_neighbors)
    print("padding history=", calc.padding_manager.padding_bounds)


def test_calculator_with_fake_atoms():
    calc = TPCalculator(
        model="./test_calculator_model",
        pad_neighbors_fraction=0.25,
        pad_atoms_number=10,
    )
    calc1 = TPCalculator(
        model="./test_calculator_model",
    )
    # print(f"{calc.current_max_neighbors=} at the start")
    # print(f"{calc.current_max_atoms=} at the start")
    np.random.seed(322)
    s = bulk("W", cubic=True) * (2, 2, 2)
    s.positions += np.random.uniform(-0.2, 0.2, size=(len(s), 3))
    s.calc = calc
    e = s.get_potential_energy()
    print(e)
    f = s.get_forces()
    assert np.shape(f) == (len(s), 3)
    print(f)
    st = s.get_stress()
    print(st)
    # print(f"{calc.padding_manager.current_max_neighbors=} at the end")
    # print(f"{calc.padding_manager.current_max_atoms=} at the end")
    st_num = calc.calculate_numerical_stress(s)
    assert np.allclose(st, st_num, rtol=1e-5)

    s.calc = calc1
    e1 = s.get_potential_energy()
    f1 = s.get_forces()
    assert np.shape(f1) == (len(s), 3)

    st1 = s.get_stress()

    assert np.allclose(e, e1, rtol=1e-5)
    assert np.allclose(f, f1, rtol=1e-5)
    assert np.allclose(st, st1, rtol=1e-5)


def test_ensemble_calculator_model():
    ins = load_instructions("model_grace.yaml")
    model = TPModel(ins)
    model.build(float64)
    calc = TPCalculator(model=[model, model])

    np.random.seed(322)
    s = bulk("W", cubic=True) * (2, 2, 2)
    s.positions += np.random.uniform(-0.2, 0.2, size=(len(s), 3))
    s.calc = calc
    e0 = s.get_potential_energy()
    print(e0)
    f0 = s.get_forces()
    print(f0)
    st0 = s.get_stress()
    print(st0)
    st0_num = calc.calculate_numerical_stress(s)
    assert np.allclose(st0, st0_num, rtol=1e-5)

    assert "energy_std" in calc.results
    assert "forces_std" in calc.results
    assert "stress_std" in calc.results

    assert np.allclose(calc.results["energy_std"], 0)
    assert np.allclose(calc.results["forces_std"], 0)
    assert np.allclose(calc.results["stress_std"], 0)


def test_calculator_min_dist():
    calc = TPCalculator(
        model="./test_calculator_model",
        pad_neighbors_fraction=0.25,
        min_dist=1.0,
    )

    at = Atoms("W2", positions=[[0, 0, 0], [0, 0, 2]], cell=(10, 10, 10), pbc=True)
    at.calc = calc

    at.get_potential_energy()

    at2 = Atoms("W2", positions=[[0, 0, 0], [0, 0, 0.5]], cell=(10, 10, 10), pbc=True)
    at2.calc = calc
    with pytest.raises(RuntimeError):
        at2.get_potential_energy()


def test_calculator_model_with_custom_cutoff():
    calc = TPCalculator(
        model="./test_model_custom_cutoff",
        pad_neighbors_fraction=0.25,
        min_dist=1.0,
    )

    def compute_e(symbs, z):
        at = Atoms(
            symbs, positions=[[0, 0, 0], [0, 0, z]], cell=(100, 100, 100), pbc=True
        )
        desc = f"{symbs} (z={z})"
        at.calc = calc
        e = at.get_potential_energy()
        print(f"{desc} e=", e)
        return e

    # Ta-*: 7
    assert compute_e("Ta2", 7.5) == 0
    assert compute_e("Ta2", 5.5) != 0

    # Ta-Mo: 7
    assert compute_e("TaMo", 7.5) == 0
    assert compute_e("MoTa", 5.5) != 0

    # Mo: 4
    assert compute_e("Mo2", 5.5) == 0
    assert compute_e("Mo2", 3.5) != 0

    # W: 5
    assert compute_e("W2", 5.5) == 0
    assert compute_e("W2", 3.5) != 0

    # Mo-Nb: 3
    assert compute_e("MoNb", 5.5) == 0
    assert compute_e("MoNb", 2.5) != 0


def test_padding_manager():
    print()
    element_map = {"W": 0}
    cutoff = 6
    geom_data_builder = GeometricalDataBuilder(
        elements_map=element_map,
        cutoff=cutoff,
        cutoff_dict=None,
    )
    data_builders = [geom_data_builder]

    pm = PaddingManager(
        data_builders=data_builders,
        pad_neighbors_fraction=0.25,
        pad_atoms_number=5,
        max_number_reduction_recompilation=0,
        debug_padding=True,
    )

    assert len(pm.padding_bounds) == 0

    atoms = bulk("W", a=3.165, cubic=True) * (2, 2, 2)
    data = pm.get_data(atoms)
    assert len(pm.padding_bounds) == 1

    del atoms[0]
    del atoms[1]
    data = pm.get_data(atoms)
    assert len(pm.padding_bounds) == 1

    atoms = bulk("W", a=3.165, cubic=True)
    data = pm.get_data(atoms)
    assert len(pm.padding_bounds) == 1

    print("set max_number_reduction_recompilation=1")
    pm.max_number_reduction_recompilation = 1
    data = pm.get_data(atoms)
    assert len(pm.padding_bounds) == 2

    atoms = bulk("W", a=3.165, cubic=True) * (1, 2, 2)
    data = pm.get_data(atoms)
    assert len(pm.padding_bounds) == 2

    # print("data=", data)
    data = {k: v for k, v in data.items()}

    ins = load_instructions("model_grace.yaml")
    model = TPModel(ins)
    model.build(float64)

    data_keys = [k for k, v in model.compute_specs.items()]
    data = {k: v for k, v in data.items() if k in data_keys}

    tf_data = Dataset.from_tensors(data).get_single_element()
    output = model.compute(tf_data)

    # print("output=", output)


def test_padding_manager_no_padding():
    print()
    element_map = {"W": 0}
    cutoff = 6
    geom_data_builder = GeometricalDataBuilder(
        elements_map=element_map,
        cutoff=cutoff,
        cutoff_dict=None,
    )
    data_builders = [geom_data_builder]

    pm = PaddingManager(
        data_builders=data_builders,
        pad_neighbors_fraction=None,
        pad_atoms_number=None,
        debug_padding=True,
    )

    assert len(pm.padding_bounds) == 0

    atoms = bulk("W", a=3.165, cubic=True) * (2, 2, 2)
    data = pm.get_data(atoms)
    assert len(pm.padding_bounds) == 0

    del atoms[0]
    del atoms[1]
    data = pm.get_data(atoms)
    assert len(pm.padding_bounds) == 0

    atoms = bulk("W", a=3.165, cubic=True)
    data = pm.get_data(atoms)
    assert len(pm.padding_bounds) == 0

    print("set max_number_reduction_recompilation=1")
    pm.max_number_reduction_recompilation = 1
    data = pm.get_data(atoms)
    assert len(pm.padding_bounds) == 0

    atoms = bulk("W", a=3.165, cubic=True) * (1, 2, 2)
    data = pm.get_data(atoms)
    assert len(pm.padding_bounds) == 0

    # print("data=", data)
    data = {k: v for k, v in data.items()}

    ins = load_instructions("model_grace.yaml")
    model = TPModel(ins)
    model.build(float64)

    data_keys = [k for k, v in model.compute_specs.items()]
    data = {k: v for k, v in data.items() if k in data_keys}

    tf_data = Dataset.from_tensors(data).get_single_element()
    output = model.compute(tf_data)
