"""Tests for FM-based automatic energy shift correction (potential::shift: auto).

Uses tiny GRACE_1LAYER_latest and GRACE_2LAYER_latest models as "foundation models"
and ASE EMT calculator as reference "DFT" energies — no mocking required.

Covers:
- select_representative_structures
- build_composition_matrix
- compute_fm_energy_shift_auto (parametrized over 1L / 2L)
- inject_or_update_atomic_shift_in_model (parametrized over 1L / 2L)
"""
import os

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

import logging

import numpy as np
import pandas as pd
import pytest
import tensorflow as tf
from ase import Atoms
from ase.build import bulk
from ase.calculators.emt import EMT

from tensorpotential.cli.data import (
    build_composition_matrix,
    compute_fm_energy_shift_auto,
    inject_or_update_atomic_shift_in_model,
    select_representative_structures,
)

LOG_FMT = "%(asctime)s %(levelname).1s - %(message)s"
logging.basicConfig(level=logging.INFO, format=LOG_FMT, datefmt="%Y/%m/%d %H:%M:%S")


# ---------------------------------------------------------------------------
# Model preset configs
# ---------------------------------------------------------------------------

ELEMENT_MAP = {"Al": 0, "Cu": 1}
RCUT = 4.0

PRESET_CONFIGS = {
    "GRACE_1LAYER_latest": dict(
        lmax=1,
        n_rad_base=4,
        n_rad_max=8,
        prod_func_n_max=8,
        embedding_size=8,
        n_mlp_dens=4,
        max_order=2,
    ),
    "GRACE_2LAYER_latest": dict(
        lmax=[1, 1],
        n_rad_base=4,
        n_rad_max=[8, 8],
        prod_func_n_max=[8, 8],
        embedding_size=8,
        n_mlp_dens=4,
        max_order=2,
        indicator_lmax=1,
    ),
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_bulk_structures():
    """Return a few bulk Cu/Al structures with EMT energies as 'DFT' reference."""
    structs = [
        bulk("Cu", cubic=True) * (2, 2, 2),  # 32 Cu atoms
        bulk("Al", cubic=True) * (2, 2, 2),  # 32 Al atoms
        bulk("Cu", cubic=True) * (2, 1, 1),  # 8  Cu atoms
        bulk("Al", cubic=True) * (2, 1, 1),  # 8  Al atoms
        bulk("Cu", cubic=True) + bulk("Al", cubic=True),  # mixed
    ]
    return structs


def _emt_energy(atoms):
    at = atoms.copy()
    at.calc = EMT()
    return at.get_potential_energy()


def _make_train_df(structs):
    energies = [_emt_energy(at) for at in structs]
    return pd.DataFrame(
        {
            "ase_atoms": structs,
            "energy": energies,
            "energy_corrected": energies,
        }
    )


def _build_preset(preset_name):
    from tensorpotential.potentials import get_preset

    build_fn = get_preset(preset_name)
    manager = build_fn(
        element_map=ELEMENT_MAP,
        rcut=RCUT,
        avg_n_neigh=12.0,
        **PRESET_CONFIGS[preset_name],
    )
    return manager.get_instructions()


def _build_checkpoint(preset_name, folder):
    """Build a tiny model, save model.yaml + checkpoint, return folder path."""
    from tensorpotential.tensorpot import TensorPotential
    from tensorpotential.instructions.base import save_instructions_dict
    from tensorpotential.calculator.asecalculator import TPCalculator

    model_yaml = str(folder / "model.yaml")
    ckpt_prefix = str(folder / "checkpoint")

    instructions = _build_preset(preset_name)
    save_instructions_dict(model_yaml, instructions)

    tp = TensorPotential(potential=instructions)
    calc = TPCalculator(tp.model)
    # trigger variable initialization via a forward pass
    calc.get_potential_energy(bulk("Cu", cubic=True).copy())
    tp.save_checkpoint(checkpoint_name=ckpt_prefix)
    return str(folder)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def train_df():
    return _make_train_df(_make_bulk_structures())


@pytest.fixture(scope="module", params=list(PRESET_CONFIGS.keys()))
def fm_checkpoint_folder(request, tmp_path_factory):
    """Build a tiny model checkpoint for each preset."""
    preset_name = request.param
    folder = tmp_path_factory.mktemp(f"fm_ckpt_{preset_name}")
    return _build_checkpoint(preset_name, folder)


@pytest.fixture(params=list(PRESET_CONFIGS.keys()))
def preset_name(request):
    return request.param


# ---------------------------------------------------------------------------
# select_representative_structures (preset-independent)
# ---------------------------------------------------------------------------


def test_select_structures_covers_all_elements(train_df):
    selected_atoms, selected_idx = select_representative_structures(train_df, max_structures=50)

    selected_elements = set()
    for at in selected_atoms:
        selected_elements.update(at.get_chemical_symbols())

    assert "Cu" in selected_elements
    assert "Al" in selected_elements
    assert len(selected_atoms) > 0


def test_select_structures_returns_valid_indices(train_df):
    _, selected_idx = select_representative_structures(train_df, max_structures=50)
    assert all(i in train_df.index for i in selected_idx)


def test_select_structures_warns_when_over_max(train_df, caplog):
    with caplog.at_level(logging.WARNING):
        selected_atoms, _ = select_representative_structures(train_df, max_structures=1)

    if len(selected_atoms) > 1:
        assert any("Using all" in msg for msg in caplog.messages)


def test_select_structures_reproducible(train_df):
    _, idx1 = select_representative_structures(train_df, max_structures=50, seed=7)
    _, idx2 = select_representative_structures(train_df, max_structures=50, seed=7)
    np.testing.assert_array_equal(idx1, idx2)


# ---------------------------------------------------------------------------
# build_composition_matrix (preset-independent)
# ---------------------------------------------------------------------------


def test_build_composition_matrix_shape():
    atoms_list = [
        Atoms("Cu2Al", positions=np.random.rand(3, 3) * 3, pbc=False),
        Atoms("Al3",   positions=np.random.rand(3, 3) * 3, pbc=False),
    ]
    mat = build_composition_matrix(atoms_list, ["Al", "Cu"])
    assert mat.shape == (2, 2)


def test_build_composition_matrix_values():
    atoms_list = [
        Atoms("Cu2Al", positions=np.random.rand(3, 3) * 3, pbc=False),
        Atoms("Al3",   positions=np.random.rand(3, 3) * 3, pbc=False),
    ]
    mat = build_composition_matrix(atoms_list, ["Al", "Cu"])
    np.testing.assert_array_equal(mat[0], [1, 2])
    np.testing.assert_array_equal(mat[1], [3, 0])


def test_build_composition_matrix_row_sums_equal_natoms():
    structs = _make_bulk_structures()
    mat = build_composition_matrix(structs, ["Al", "Cu"])
    row_sums = mat.sum(axis=1)
    n_atoms = np.array([len(at) for at in structs])
    np.testing.assert_array_equal(row_sums, n_atoms)


# ---------------------------------------------------------------------------
# compute_fm_energy_shift_auto (parametrized over 1L / 2L)
# ---------------------------------------------------------------------------


def test_compute_fm_energy_shift_returns_all_elements(fm_checkpoint_folder, train_df):
    result = compute_fm_energy_shift_auto(
        train_df=train_df,
        checkpoint_folder=fm_checkpoint_folder,
        seed=42,
    )
    assert isinstance(result, dict)
    assert "Cu" in result
    assert "Al" in result


def test_compute_fm_energy_shift_values_are_finite(fm_checkpoint_folder, train_df):
    result = compute_fm_energy_shift_auto(
        train_df=train_df,
        checkpoint_folder=fm_checkpoint_folder,
        seed=42,
    )
    for el, shift in result.items():
        assert np.isfinite(shift), f"shift for {el} is not finite: {shift}"


def test_compute_fm_energy_shift_reduces_residual(fm_checkpoint_folder, train_df):
    """After applying computed shifts, per-structure energy residual should decrease."""
    from tensorpotential.calculator.asecalculator import TPCalculator
    from tensorpotential.tensorpot import TensorPotential
    from tensorpotential.instructions.base import load_instructions

    shift_dict = compute_fm_energy_shift_auto(
        train_df=train_df,
        checkpoint_folder=fm_checkpoint_folder,
        seed=42,
    )

    model_yaml = os.path.join(fm_checkpoint_folder, "model.yaml")
    ckpt_prefix = os.path.join(fm_checkpoint_folder, "checkpoint")
    instructions = load_instructions(model_yaml)
    tp = TensorPotential(potential=instructions)
    tp.load_checkpoint(checkpoint_name=ckpt_prefix, expect_partial=True, verbose=False)
    calc = TPCalculator(tp.model)

    structs = train_df["ase_atoms"].tolist()
    e_dft = train_df["energy_corrected"].values
    e_fm = np.array([calc.get_potential_energy(at.copy()) for at in structs])

    residual_before = np.mean((e_dft - e_fm) ** 2)

    e_fm_corrected = e_fm.copy()
    for i, at in enumerate(structs):
        syms = at.get_chemical_symbols()
        correction = sum(shift_dict.get(s, 0.0) for s in syms)
        e_fm_corrected[i] += correction

    residual_after = np.mean((e_dft - e_fm_corrected) ** 2)
    assert residual_after < residual_before, (
        f"Shift correction did not reduce residual: before={residual_before:.4f}, after={residual_after:.4f}"
    )


# ---------------------------------------------------------------------------
# inject_or_update_atomic_shift_in_model (parametrized over 1L / 2L)
# ---------------------------------------------------------------------------


def test_inject_adds_shifts_when_none_exist(preset_name):
    from tensorpotential.instructions.output import ConstantScaleShiftTarget

    instructions = _build_preset(preset_name)
    fm_shift_dict = {"Al": -1.0, "Cu": -0.5}

    inject_or_update_atomic_shift_in_model(instructions, fm_shift_dict, ELEMENT_MAP)

    inst = next(v for v in instructions.values() if isinstance(v, ConstantScaleShiftTarget))
    saved = inst._init_args["atomic_shift_map"]
    assert saved[ELEMENT_MAP["Al"]] == pytest.approx(-1.0)
    assert saved[ELEMENT_MAP["Cu"]] == pytest.approx(-0.5)
    assert inst.apply_shift is True


def test_inject_accumulates_with_existing_shifts(preset_name):
    from tensorpotential.instructions.output import ConstantScaleShiftTarget

    instructions = _build_preset(preset_name)

    inject_or_update_atomic_shift_in_model(
        instructions, {"Al": -0.3, "Cu": -0.2}, ELEMENT_MAP
    )
    inject_or_update_atomic_shift_in_model(
        instructions, {"Al": -0.7, "Cu": -0.3}, ELEMENT_MAP
    )

    inst = next(v for v in instructions.values() if isinstance(v, ConstantScaleShiftTarget))
    saved = inst._init_args["atomic_shift_map"]
    assert saved[ELEMENT_MAP["Al"]] == pytest.approx(-1.0)
    assert saved[ELEMENT_MAP["Cu"]] == pytest.approx(-0.5)


def test_inject_numpy_array_consistent_with_init_args(preset_name):
    from tensorpotential.instructions.output import ConstantScaleShiftTarget

    instructions = _build_preset(preset_name)
    inject_or_update_atomic_shift_in_model(instructions, {"Al": -1.0, "Cu": -0.5}, ELEMENT_MAP)

    inst = next(v for v in instructions.values() if isinstance(v, ConstantScaleShiftTarget))
    saved = inst._init_args["atomic_shift_map"]
    arr = inst.atomic_shift_map

    for el, mu in ELEMENT_MAP.items():
        assert arr[mu] == pytest.approx(saved[mu])


def test_inject_model_predictions_change_after_surgery(fm_checkpoint_folder):
    """TPCalculator energies must shift by exactly the injected per-atom correction."""
    from tensorpotential.calculator.asecalculator import TPCalculator
    from tensorpotential.tensorpot import TensorPotential
    from tensorpotential.instructions.base import load_instructions

    model_yaml = os.path.join(fm_checkpoint_folder, "model.yaml")
    ckpt_prefix = os.path.join(fm_checkpoint_folder, "checkpoint")

    at = bulk("Cu", cubic=True) * (2, 2, 2)  # 32 Cu atoms
    n_cu = len(at)

    # Energy before surgery
    instructions_before = load_instructions(model_yaml)
    tp_before = TensorPotential(potential=instructions_before)
    tp_before.load_checkpoint(checkpoint_name=ckpt_prefix, expect_partial=True, verbose=False)
    calc_before = TPCalculator(tp_before.model)
    e_before = calc_before.get_potential_energy(at.copy())

    # Apply surgery and rebuild
    cu_shift = -0.25
    inject_or_update_atomic_shift_in_model(
        instructions_before, {"Cu": cu_shift, "Al": -1.0}, ELEMENT_MAP
    )
    tp_after = TensorPotential(potential=instructions_before)
    tp_after.load_checkpoint(checkpoint_name=ckpt_prefix, expect_partial=True, verbose=False)
    calc_after = TPCalculator(tp_after.model)
    e_after = calc_after.get_potential_energy(at.copy())

    expected_delta = n_cu * cu_shift
    actual_delta = e_after - e_before
    assert actual_delta == pytest.approx(expected_delta, abs=1e-4)


def test_inject_skips_unknown_elements(preset_name):
    from tensorpotential.instructions.output import ConstantScaleShiftTarget

    instructions = _build_preset(preset_name)
    inject_or_update_atomic_shift_in_model(instructions, {"Al": -1.0, "Zr": -99.0}, ELEMENT_MAP)

    inst = next(v for v in instructions.values() if isinstance(v, ConstantScaleShiftTarget))
    saved = inst._init_args["atomic_shift_map"]
    assert ELEMENT_MAP["Al"] in saved
    assert -99.0 not in saved.values()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])