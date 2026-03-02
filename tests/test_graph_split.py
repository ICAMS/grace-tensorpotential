"""
Tests for domain decomposition / graph splitting functionality.

Tests the ComputeBlock, ComputeBlockInputGradient, and ComputeBlockOutputGradient
classes used for distributed computation of neural network potentials.

Based on: pgs/graph_split_v2.py
"""

import os
import shutil

import numpy as np
import pytest
from ase.build import bulk

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

import tensorflow as tf

tf.config.experimental.enable_tensor_float_32_execution(False)

from tensorpotential import constants, TPModel
from tensorpotential.tpmodel import (
    ComputeBlock,
    ComputeBlockInputGradient,
    ComputeBlockOutputGradient,
)
from tensorpotential.potentials import get_preset
from tensorpotential.data.databuilder import GeometricalDataBuilder
from tensorpotential.instructions.instruction_graph_utils import build_split_tpmodel


# --- Fixtures ---


@pytest.fixture
def grace_2layer_config():
    """Configuration for GRACE 2-layer model."""
    element_dict = {"Na": 0, "Cl": 1}
    cutoff = 6.01
    lmax = [4, 3]
    return {
        "element_map": element_dict,
        "rcut": cutoff,
        "constant_out_scale": 10.0,
        "atomic_shift_map": {0: -3, 1: 2},
        "cutoff_dict": None,
        "max_order": 3,
        "lmax": lmax,
        "avg_n_neigh": 33.0,
    }


@pytest.fixture
def nacl_structure():
    """Simple NaCl structure for testing."""
    np.random.seed(322)
    at = bulk("NaCl", "rocksalt", a=3.0)
    at.rattle(0.5)
    return at


@pytest.fixture
def communication_keys():
    """Keys communicated between layers."""
    return ["I", "I_out_0_LN"]


@pytest.fixture
def l1_out_specs():
    """Output specs for Layer 1."""
    return {
        "I": {"shape": [None, 32, 16], "dtype": "float64"},
        "I_out_0_LN": {"shape": [None, 17, 1], "dtype": "float64"},
    }


def prepare_data(element_dict, cutoff, at, float_dtype="float64"):
    """Prepare input data from ASE atoms."""
    db = GeometricalDataBuilder(
        elements_map=element_dict, cutoff=cutoff, float_dtype=float_dtype
    )
    data = db.extract_from_ase_atoms(at)
    ds = tf.data.Dataset.from_tensors(data)
    el = ds.get_single_element()
    el[constants.ATOMIC_MU_I_LOCAL] = el[constants.ATOMIC_MU_I]
    el[constants.N_ATOMS_BATCH_TOTAL] = el[constants.N_ATOMS_BATCH_REAL]
    return el


# --- Unit Tests ---


class TestComputeBlock:
    """Unit tests for ComputeBlock class."""

    def test_compute_block_basic(
        self, grace_2layer_config, nacl_structure, communication_keys
    ):
        """Test that ComputeBlock executes instructions and returns expected output keys."""
        np.random.seed(322)
        tf.random.set_seed(322)

        preset = get_preset("GRACE_2LAYER_latest")
        instructor = preset(**grace_2layer_config)
        block_layer_1 = instructor.get_block("Layer_1")

        # Create ComputeBlock
        compute_block = ComputeBlock(
            instructions=block_layer_1, output_keys=communication_keys
        )

        # Verify specs
        assert compute_block.output_keys == communication_keys

        # Build model with aux compute
        aux_computes = {"forward_layer_1": compute_block}
        instructions = instructor.get_instructions()
        m = TPModel(instructions=instructions, aux_compute=aux_computes)
        m.build(tf.float64, jit_compile=True, input_dtype=tf.float64)

        # Prepare input data
        inpt_tf = prepare_data(
            grace_2layer_config["element_map"],
            grace_2layer_config["rcut"],
            nacl_structure,
        )

        # Execute forward layer 1
        l1_outputs = m.forward_layer_1(inpt_tf)

        # Verify output keys
        assert set(l1_outputs.keys()) == set(communication_keys)

        # Verify output shapes
        for key in communication_keys:
            assert l1_outputs[key] is not None
            assert l1_outputs[key].shape[0] is not None  # batch dimension


class TestComputeBlockInputGradient:
    """Unit tests for ComputeBlockInputGradient class."""

    def test_compute_block_input_gradient(
        self, grace_2layer_config, nacl_structure, communication_keys, l1_out_specs
    ):
        """Test that ComputeBlockInputGradient computes gradients correctly."""
        np.random.seed(322)
        tf.random.set_seed(322)

        preset = get_preset("GRACE_2LAYER_latest")
        instructor = preset(**grace_2layer_config)
        block_layer_2 = instructor.get_block("Layer_2")

        # Create ComputeBlockInputGradient
        compute_block_grad = ComputeBlockInputGradient(
            instructions=block_layer_2,
            wrt_keys=communication_keys + [constants.BOND_VECTOR],
            target_key=constants.PREDICT_ATOMIC_ENERGY,
            output_keys=[constants.PREDICT_ATOMIC_ENERGY],
            specs=l1_out_specs,
        )

        # Prepare input data
        inpt_tf = prepare_data(
            grace_2layer_config["element_map"],
            grace_2layer_config["rcut"],
            nacl_structure,
        )
        # Mock l1 outputs as inputs
        # We need to ensure types match
        for k, v in l1_out_specs.items():
            inpt_tf[k] = tf.zeros(
                (
                    v["shape"]
                    if v["shape"][0] is not None
                    else [inpt_tf[constants.ATOMIC_MU_I].shape[0]] + v["shape"][1:]
                ),
                dtype=getattr(tf, v["dtype"]),
            )

        # Build model with aux compute
        aux_computes = {"backward_layer_2": compute_block_grad}
        instructions = instructor.get_instructions()
        m = TPModel(instructions=instructions, aux_compute=aux_computes)
        m.build(tf.float64, jit_compile=True, input_dtype=tf.float64)

        # Execute backward layer 2
        l2_results = m.backward_layer_2(inpt_tf)

        # Verify output keys
        expected_keys = {
            f"grad_{k}" for k in communication_keys + [constants.BOND_VECTOR]
        } | {constants.PREDICT_ATOMIC_ENERGY}
        assert expected_keys.issubset(l2_results.keys())


class TestComputeBlockOutputGradient:
    """Unit tests for ComputeBlockOutputGradient class."""

    def test_compute_block_output_gradient(
        self, grace_2layer_config, nacl_structure, communication_keys, l1_out_specs
    ):
        """Test that ComputeBlockOutputGradient is configured correctly."""
        np.random.seed(322)
        tf.random.set_seed(322)

        preset = get_preset("GRACE_2LAYER_latest")
        instructor = preset(**grace_2layer_config)
        block_layer_1 = instructor.get_block("Layer_1")

        l2_grad_specs = {f"grad_{k}": v for k, v in l1_out_specs.items()}

        # Create ComputeBlockOutputGradient
        compute_block_out_grad = ComputeBlockOutputGradient(
            instructions=block_layer_1,
            wrt_keys=[constants.BOND_VECTOR],
            output_keys=communication_keys,
            specs=l2_grad_specs,
        )

        # Prepare input data
        inpt_tf = prepare_data(
            grace_2layer_config["element_map"],
            grace_2layer_config["rcut"],
            nacl_structure,
        )
        # Mock l2 gradients as inputs
        for k, v in l1_out_specs.items():
            inpt_tf[f"grad_{k}"] = tf.zeros(
                (
                    v["shape"]
                    if v["shape"][0] is not None
                    else [inpt_tf[constants.ATOMIC_MU_I].shape[0]] + v["shape"][1:]
                ),
                dtype=getattr(tf, v["dtype"]),
            )

        # Build model with aux compute
        aux_computes = {"backward_layer_1": compute_block_out_grad}
        instructions = instructor.get_instructions()
        m = TPModel(instructions=instructions, aux_compute=aux_computes)
        m.build(tf.float64, jit_compile=True, input_dtype=tf.float64)

        # Execute backward layer 1
        l1_grads = m.backward_layer_1(inpt_tf)

        # Verify output keys
        assert f"grad_{constants.BOND_VECTOR}" in l1_grads


# --- Integration Tests ---


class TestGraphSplitIntegration:
    """Integration tests comparing split vs reference model execution."""

    def test_split_vs_reference_forces(
        self, grace_2layer_config, nacl_structure, communication_keys, l1_out_specs
    ):
        """Compare pair forces from split execution vs reference model."""
        np.random.seed(322)
        tf.random.set_seed(322)

        preset = get_preset("GRACE_2LAYER_latest")
        instructor = preset(**grace_2layer_config)
        instructions = instructor.get_instructions()

        # Use build_split_tpmodel instead of manual construction
        m_split = build_split_tpmodel(
            instructions,
            communicated_keys=communication_keys,
            float_dtype=tf.float64,
            jit_compile=True,
        )

        # Prepare data
        inpt_tf = prepare_data(
            grace_2layer_config["element_map"],
            grace_2layer_config["rcut"],
            nacl_structure,
        )

        # --- Split execution ---
        # A. Forward Layer 1
        l1_outputs = m_split.forward_layer_1(inpt_tf)

        # B. Forward & Backward Layer 2
        inpt_for_l2 = {**inpt_tf, **l1_outputs}
        l2_results = m_split.backward_layer_2(inpt_for_l2)
        atomic_energy_split = l2_results[constants.PREDICT_ATOMIC_ENERGY]

        # C. Backward Layer 1
        inpt_for_l1_back = {**inpt_tf}
        for k in communication_keys:
            inpt_for_l1_back[f"grad_{k}"] = l2_results[f"grad_{k}"]

        l1_grads = m_split.backward_layer_1(inpt_for_l1_back)

        # Total pair forces
        grad_bond_vector_l2 = l2_results[f"grad_{constants.BOND_VECTOR}"]
        grad_bond_vector_l1 = l1_grads[f"grad_{constants.BOND_VECTOR}"]
        pair_forces_split = -grad_bond_vector_l2 - grad_bond_vector_l1

        # --- Reference execution ---
        m_ref = TPModel(instructions=instructions)
        m_ref.build(tf.float64, jit_compile=True, input_dtype=tf.float64)
        m_ref.decorate_compute_function(
            tf.float64, jit_compile=True, input_dtype=tf.float64
        )

        # Get reference input data
        db = GeometricalDataBuilder(
            elements_map=grace_2layer_config["element_map"],
            cutoff=grace_2layer_config["rcut"],
            float_dtype="float64",
        )
        data = db.extract_from_ase_atoms(nacl_structure)
        data[constants.N_ATOMS_BATCH_TOTAL] = data[constants.N_ATOMS_BATCH_REAL]
        inpt_dict = {k: data[k] for k in m_ref.compute_specs.keys()}
        output_ref = m_ref.compute(inpt_dict)

        # Compare atomic energies
        atomic_energy_diff = (
            atomic_energy_split - output_ref[constants.PREDICT_ATOMIC_ENERGY]
        )
        assert np.allclose(
            atomic_energy_diff.numpy(), 0, atol=1e-6
        ), f"Atomic energy mismatch: max diff = {np.max(np.abs(atomic_energy_diff.numpy()))}"

        # Compare pair forces
        pair_forces_ref = output_ref["z_pair_f"]
        pair_forces_diff = pair_forces_split - pair_forces_ref
        assert np.allclose(
            pair_forces_diff.numpy(), 0, atol=1e-6
        ), f"Pair forces mismatch: max diff = {np.max(np.abs(pair_forces_diff.numpy()))}"


class TestGraphSplitSaveReload:
    """Tests for saving and reloading models with aux_compute."""

    def test_split_model_save_reload(
        self, grace_2layer_config, nacl_structure, communication_keys, l1_out_specs
    ):
        """Test saving/loading model with aux_compute and verify output consistency."""
        np.random.seed(322)
        tf.random.set_seed(322)

        model_path = "temp_saved_model_test"

        try:
            preset = get_preset("GRACE_2LAYER_latest")
            instructor = preset(**grace_2layer_config)
            instructions = instructor.get_instructions()

            # Use build_split_tpmodel which sets non_local_communication_keys
            m_aux = build_split_tpmodel(
                instructions,
                communicated_keys=communication_keys,
                float_dtype=tf.float64,
                jit_compile=True,
            )

            # Save model
            m_aux.save_model(
                model_path,
                jit_compile=True,
                float_dtype=tf.float64,
                input_dtype=tf.float64,
            )

            # Get original outputs
            inpt_tf = prepare_data(
                grace_2layer_config["element_map"],
                grace_2layer_config["rcut"],
                nacl_structure,
            )

            l1_outputs_orig = m_aux.forward_layer_1(inpt_tf)
            inpt_for_l2 = {**inpt_tf, **l1_outputs_orig}
            l2_results_orig = m_aux.backward_layer_2(inpt_for_l2)

            inpt_for_l1_back = {**inpt_tf}
            for k in communication_keys:
                inpt_for_l1_back[f"grad_{k}"] = l2_results_orig[f"grad_{k}"]
            l1_grads_orig = m_aux.backward_layer_1(inpt_for_l1_back)

            grad_bv_l2_orig = l2_results_orig[f"grad_{constants.BOND_VECTOR}"]
            grad_bv_l1_orig = l1_grads_orig[f"grad_{constants.BOND_VECTOR}"]
            pair_forces_orig = -(grad_bv_l2_orig + grad_bv_l1_orig)

            # Load model
            loaded_model = tf.saved_model.load(model_path)

            # Helper function to run loaded signature
            def run_loaded_signature(signature_name, input_data):
                func = loaded_model.signatures[signature_name]
                # Filter input dict keys to match what signature expects
                # Signature inputs are flattened in SavedModel signatures
                # Using [1] because [0] is usually empty for signatures
                if (
                    hasattr(func, "structured_input_signature")
                    and func.structured_input_signature
                ):
                    keys = func.structured_input_signature[1].keys()
                    filtered_data = {k: v for k, v in input_data.items() if k in keys}
                    return func(**filtered_data)
                return func(**input_data)

            # Run loaded model
            l1_outputs_loaded = run_loaded_signature("forward_layer_1", inpt_tf)

            inpt_for_l2_loaded = {**inpt_tf, **l1_outputs_loaded}
            l2_results_loaded = run_loaded_signature(
                "backward_layer_2", inpt_for_l2_loaded
            )

            inpt_for_l1_back_loaded = {**inpt_tf}
            for k in communication_keys:
                inpt_for_l1_back_loaded[f"grad_{k}"] = l2_results_loaded[f"grad_{k}"]
            l1_grads_loaded = run_loaded_signature(
                "backward_layer_1", inpt_for_l1_back_loaded
            )

            grad_bv_l2_loaded = l2_results_loaded[f"grad_{constants.BOND_VECTOR}"]
            grad_bv_l1_loaded = l1_grads_loaded[f"grad_{constants.BOND_VECTOR}"]
            pair_forces_loaded = -(grad_bv_l2_loaded + grad_bv_l1_loaded)

            # Compare outputs
            diff = np.max(np.abs(pair_forces_orig.numpy() - pair_forces_loaded.numpy()))
            assert np.allclose(
                pair_forces_orig, pair_forces_loaded, atol=1e-6
            ), f"Mismatch after reload! Max diff: {diff}"

        finally:
            # Clean up
            if os.path.exists(model_path):
                shutil.rmtree(model_path)
