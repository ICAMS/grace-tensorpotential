import os
import pytest
import tensorflow as tf
import numpy as np
from tensorpotential.potentials import get_preset
from tensorpotential import constants
from tensorpotential.instructions.instruction_graph_utils import (
    get_dependencies,
    build_dependency_graph,
    print_dependency_tree,
    get_communication_keys,
    split_2layer_instructions,
    build_split_tpmodel,
    infer_communicated_keys,
)
from tensorpotential.instructions import (
    BondLength,
    ScaledBondVector,
    SphericalHarmonic,
    ScalarChemicalEmbedding,
    InstructionManager,
    load_instructions,
)
from tensorpotential.tpmodel import ComputeEnergy


@pytest.fixture
def simple_instructions():
    with InstructionManager() as instructor:
        d_ij = BondLength(name="d_ij")
        rhat = ScaledBondVector(bond_length=d_ij, name="rhat")
        Y = SphericalHarmonic(vhat=rhat, lmax=2, name="Y")
        z = ScalarChemicalEmbedding(element_map={"H": 0}, embedding_size=8, name="Z")
    return instructor.get_instructions()


def test_get_dependencies(simple_instructions):
    deps = get_dependencies(simple_instructions["rhat"])
    assert deps == ["d_ij"]
    
    deps = get_dependencies(simple_instructions["Y"])
    assert deps == ["rhat"]
    
    deps = get_dependencies(simple_instructions["Z"])
    assert deps == []


def test_build_dependency_graph(simple_instructions):
    graph = build_dependency_graph(simple_instructions)
    assert graph["d_ij"] == []
    assert graph["rhat"] == ["d_ij"]
    assert graph["Y"] == ["rhat"]
    assert graph["Z"] == []


def test_print_dependency_tree(simple_instructions):
    tree_str = print_dependency_tree(simple_instructions)
    assert "Y (SphericalHarmonic)" in tree_str
    assert "└── rhat (ScaledBondVector)" in tree_str
    assert "    └── d_ij (BondLength)" in tree_str
    assert "Z (ScalarChemicalEmbedding)" in tree_str


def test_get_communication_keys():
    with InstructionManager() as instructor:
        d_ij = BondLength(name="d_ij")
        rhat = ScaledBondVector(bond_length=d_ij, name="rhat")
        
    l1 = instructor.get_instructions()
    
    with InstructionManager() as instructor2:
        # Mocking l2 which depends on l1
        Y = SphericalHarmonic(vhat=rhat, lmax=2, name="Y")
        
    l2 = instructor2.get_instructions()
    
    comm_keys = get_communication_keys(l1, l2)
    assert comm_keys == ["rhat"]


def test_split_2layer_grace():
    preset = get_preset("GRACE_2LAYER_latest")
    config = {
        "element_map": {"Na": 0, "Cl": 1},
        "rcut": 6.0,
        "max_order": 2,
        "lmax": [2, 1],
    }
    instructor = preset(**config)
    instructions = instructor.get_instructions()
    
    # Standard comm keys for GRACE 2-layer
    comm_keys = ["I", "I_out_0_LN"]
    l1, l2 = split_2layer_instructions(instructions, comm_keys)
    
    # Communication keys should NOT be in L2 (they are inputs)
    assert "I" not in l2
    assert "I_out_0_LN" not in l2
    
    # Roots of L2 should be the targets
    assert constants.PREDICT_ATOMIC_ENERGY in l2
    
    # Common dependencies like BondLength should be in both
    assert "BondLength" in l1
    assert "BondLength" in l2
    
    # Verify order is preserved
    original_names = list(instructions.keys())
    l1_names = list(l1.keys())
    l2_names = list(l2.keys())
    
    # Names in L1 should appear in the same relative order as original
    expected_l1_order = [n for n in original_names if n in l1]
    assert l1_names == expected_l1_order
    
    # Names in L2 should appear in the same relative order as original
    expected_l2_order = [n for n in original_names if n in l2]
    assert l2_names == expected_l2_order


def test_build_split_tpmodel():
    preset = get_preset("GRACE_2LAYER_latest")
    config = {
        "element_map": {"Na": 0, "Cl": 1},
        "rcut": 6.0,
        "max_order": 2,
        "lmax": [2, 1],
    }
    instructor = preset(**config)
    instructions = instructor.get_instructions()

    # Standard comm keys for GRACE 2-layer
    comm_keys = ["I", "I_out_0_LN"]
    
    # Build split model automatically, matching usage in grace_utils.py
    m = build_split_tpmodel(
        instructions, 
        comm_keys, 
        float_dtype=tf.float64, 
        input_dtype=tf.float64, 
        jit_compile=True,
        extra_aux_computes={"compute_energy": ComputeEnergy()}
    )

    assert hasattr(m, "forward_layer_1")
    assert hasattr(m, "backward_layer_2")
    assert hasattr(m, "backward_layer_1")
    assert hasattr(m, "compute_energy")

    # Basic verification of aux compute existence
    assert "forward_layer_1" in m.aux_compute
    assert "backward_layer_2" in m.aux_compute
    assert "backward_layer_1" in m.aux_compute
    assert "compute_energy" in m.aux_compute
    
    # Verify that ComputeEnergy was correctly added
    assert isinstance(m.aux_compute["compute_energy"], ComputeEnergy)

    # Verify that it built successfully
    assert m.float_dtype == tf.float64
    
    # Verify non-local keys are present (even if empty list)
    assert hasattr(m, "non_local_communication_keys")
    assert isinstance(m.non_local_communication_keys, list)


def test_infer_communicated_keys_grace_2l_latest():
    preset = get_preset("GRACE_2LAYER_latest")
    config = {
        "element_map": {"Na": 0, "Cl": 1},
        "rcut": 6.0,
        "max_order": 2,
        "lmax": [2, 1],
    }
    instructor = preset(**config)
    instructions = instructor.get_instructions()
    
    comm_keys = infer_communicated_keys(instructions)
    print(f"Detected comm keys: {comm_keys}")
    
    # In GRACE_2LAYER_latest, it should detect 'I' and 'I_out_0_LN' (or similar)
    assert "I" in comm_keys
    assert "I_out_0_LN" in comm_keys
    assert len(comm_keys) == 2

    


def test_infer_communicated_keys_omat_large_base():
    # Load from the copied YAML
    yaml_path = os.path.join(os.path.dirname(__file__), "model_grace_2L_omat_large_base.yaml")
    assert os.path.exists(yaml_path), f"Test file {yaml_path} not found"
    
    instructions = load_instructions(yaml_path)
    
    comm_keys = infer_communicated_keys(instructions)
    print(f"Detected comm keys for OMAT: {comm_keys}")
    
    # User stated it should be ['I', 'I_nl_LN']
    assert "I" in comm_keys
    assert "I_nl_LN" in comm_keys
    assert len(comm_keys) == 2
