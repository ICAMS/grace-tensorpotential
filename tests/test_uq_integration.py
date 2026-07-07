import os

import numpy as np
import pandas as pd
import pytest
import tensorflow as tf
from tensorpotential.uq.artifact_builder import GMMUQArtifactBuilder
from tensorpotential.uq.factories import (
    get_gmm_uq_calculator,
    make_basis_rp_spec,
    patch_instructions_for_basis_rp_features,
)
from tensorpotential.uq.compute import ComputeStructureEnergyAndForcesAndVirialAndUncertainty
from tensorpotential.uq.feature_extraction import (
    extract_features_bulk,
    setup_feature_calculator,
)
from tensorpotential.uq.gmmuq import GMMUQModel
from tensorpotential.instructions import load_instructions
from tensorpotential.tensorpot import TensorPotential
from tensorpotential import constants as tc
from tensorpotential.uq import constants as uqc
from tensorpotential.data.databuilder import GeometricalDataBuilder

_TRAINED_MODEL_DIR = os.path.join(
    os.path.dirname(__file__), "MoNbTaW-GRACE", "seed", "42"
)
_TRAINED_MODEL_AVAILABLE = os.path.exists(
    os.path.join(_TRAINED_MODEL_DIR, "model.yaml")
) and os.path.exists(
    os.path.join(_TRAINED_MODEL_DIR, "checkpoints", "checkpoint.best_test_loss.index")
)


@pytest.fixture(scope="module")
def trained_uq_setup(tmp_path_factory):
    """UQ artifact built from the *trained* MoNbTaW-GRACE seed/42 model.

    Distinct from the random-init ``uq_setup`` fixture in conftest: real
    weights yield features that occupy a structured subspace, so this exercises
    the centroid/covariance pipeline against signal rather than noise.
    """
    if not _TRAINED_MODEL_AVAILABLE:
        pytest.skip("MoNbTaW-GRACE seed/42 trained model not available")

    model_yaml = os.path.join(_TRAINED_MODEL_DIR, "model.yaml")
    checkpoint = os.path.join(
        _TRAINED_MODEL_DIR, "checkpoints", "checkpoint.best_test_loss"
    )
    df = pd.read_pickle(os.path.join(_TRAINED_MODEL_DIR, "training_set.pkl.gz"))
    atoms = df["ase_atoms"].tolist()
    element_map = {"Mo": 0, "Nb": 1, "Ta": 2, "W": 3}

    feat_calc = setup_feature_calculator(
        model_yaml, checkpoint, param_dtype=tf.float64
    )
    features, element_indices = extract_features_bulk(
        feat_calc, atoms, element_map=element_map
    )

    # K=2 keeps it fast even with ~20 train structures.
    builder = GMMUQArtifactBuilder(n_clusters=2, feature_dim=features.shape[1])
    stream = [(features, element_indices, np.ones(len(features)))]
    builder.fit_centroids(stream, verbose=False)
    builder.accumulate_scatter(stream, verbose=False)

    tmp_dir = tmp_path_factory.mktemp("trained_uq")
    artifact_path = str(tmp_dir / "uq_artifacts.npz")
    builder.save(
        artifact_path,
        element_map=np.array(list(element_map.keys())),
        **make_basis_rp_spec(model_yaml),
    )

    return {
        "artifact_path": artifact_path,
        "model_yaml": model_yaml,
        "checkpoint": checkpoint,
        "element_map": element_map,
        "atoms": atoms,
    }


def test_trained_model_uq_calculator_produces_finite_sigma(trained_uq_setup):
    """End-to-end: trained checkpoint + builder + calculator → finite σ.

    Catches regressions in the trained-model code path that the random-init
    fixture misses: real activations can blow up Mahalanobis distance if the
    feature normalization or per-cluster covariance regularization is broken.
    """
    calc = get_gmm_uq_calculator(
        model_yaml=trained_uq_setup["model_yaml"],
        checkpoint=trained_uq_setup["checkpoint"],
        gmm_artifact_path=trained_uq_setup["artifact_path"],
        param_dtype="float64",
    )

    atoms = trained_uq_setup["atoms"][1].copy()
    atoms.calc = calc

    energy = atoms.get_potential_energy()
    forces = atoms.get_forces()
    assert np.isfinite(energy)
    assert forces.shape == (len(atoms), 3)
    assert np.all(np.isfinite(forces))

    atomic_sigma = np.asarray(calc.results[uqc.ATOMIC_SIGMA])
    total_sigma = float(calc.results[uqc.TOTAL_SIGMA])
    assert atomic_sigma.shape == (len(atoms),)
    assert np.all(np.isfinite(atomic_sigma))
    assert np.all(atomic_sigma >= 0)
    assert np.isfinite(total_sigma) and total_sigma > 0

    # Cluster assignments must be valid indices in [0, K).
    clusters = np.asarray(calc.results[uqc.GMM_CLUSTER]).astype(int)
    assert clusters.shape == (len(atoms),)
    assert clusters.min() >= 0
    assert clusters.max() < 2

def test_get_gmm_uq_calculator(uq_setup):
    """Test the high-level factory for creating UQ-enabled calculators."""
    calc = get_gmm_uq_calculator(
        model_yaml=uq_setup["model_yaml"],
        checkpoint=uq_setup["checkpoint"],
        gmm_artifact_path=uq_setup["artifact_path"],
        param_dtype="float64"
    )
    
    # Use one of the test atoms
    atoms = uq_setup["atoms"][0].copy()
    atoms.calc = calc
    
    # 1. Energy and Forces should still work
    energy = atoms.get_potential_energy()
    forces = atoms.get_forces()
    
    assert isinstance(energy, float)
    assert forces.shape == (len(atoms), 3)
    
    # 2. gmm_uq_model must be accessible for incremental updates
    assert calc.gmm_uq_model is not None
    assert isinstance(calc.gmm_uq_model, GMMUQModel)

    # 3. Uncertainty should be present in calc results
    assert uqc.ATOMIC_SIGMA in calc.results
    assert uqc.TOTAL_SIGMA in calc.results
    
    atomic_uq = calc.results[uqc.ATOMIC_SIGMA]
    total_uq = calc.results[uqc.TOTAL_SIGMA]
    
    assert atomic_uq.shape == (len(atoms),)
    assert total_uq > 0
    assert np.all(atomic_uq >= 0)

def test_hal_compute_integration(uq_setup):
    """Test the HAL compute function directly within a TPModel."""
    instructions = load_instructions(uq_setup["model_yaml"])

    # Features must be available for the GMM model: append the basis-RP feature
    # instruction (writes the canonical FEATURES key), matching the artifact.
    patch_instructions_for_basis_rp_features(
        instructions, out_dim=uq_setup["feature_spec"]["out_dim"]
    )

    uq_model = GMMUQModel(uq_setup["artifact_path"], param_dtype=tf.float64)
    
    # Create the specialized HAL compute function
    hal_compute = ComputeStructureEnergyAndForcesAndVirialAndUncertainty(
        gmm_uq_model=uq_model,
        # We need features key to be present in model output
        extra_return_keys=[uqc.FEATURES]
    )
    
    tp = TensorPotential(
        instructions, 
        model_compute_function=hal_compute,
        param_dtype=tf.float64
    )
    tp.load_checkpoint(uq_setup["checkpoint"])
    # Important: decorate to enable optimized paths
    tp.model.decorate_compute_function()
    
    atoms = uq_setup["atoms"][0]
    db = GeometricalDataBuilder(uq_setup["element_map"], cutoff=6.0)
    # Use join_to_batch to ensure all mapping tensors (map_atoms_to_structure, etc.) are present
    inputs = db.join_to_batch([db.extract_from_ase_atoms(atoms)])
    
    # Select only keys required by the model
    tf_inputs = {}
    for k, v in inputs.items():
        if k in tp.model.compute_specs:
            spec = tp.model.compute_specs[k]
            # Convert to correct dtype as expected by model
            dtype = tf.int32 if spec["dtype"] == "int" else tf.float64
            tf_inputs[k] = tf.convert_to_tensor(v, dtype=dtype)
            
    # Run inference using the compute function (not the train function)
    results = tp.model.compute(tf_inputs)
    
    # Verify both physical and UQ properties
    assert tc.PREDICT_TOTAL_ENERGY in results
    assert tc.PREDICT_FORCES in results
    assert uqc.TOTAL_SIGMA in results
    assert uqc.ATOMIC_SIGMA in results
    
    # Check values sanity
    assert results[uqc.TOTAL_SIGMA].numpy() > 0
    assert results[uqc.ATOMIC_SIGMA].shape == (len(atoms),)


def test_gamma_only_mode_drops_dsigma_dr_keys(uq_setup):
    """When ``compute_dsigma_dr=False`` the calculator must still expose
    energy/forces/sigma but should NOT compute DSIGMA_DR/VIRIAL_SIGMA, and
    energies and atomic sigmas must agree with the full-mode computation."""
    common = dict(
        model_yaml=uq_setup["model_yaml"],
        checkpoint=uq_setup["checkpoint"],
        gmm_artifact_path=uq_setup["artifact_path"],
        param_dtype="float64",
    )
    calc_full = get_gmm_uq_calculator(**common, compute_dsigma_dr=True)
    calc_gamma = get_gmm_uq_calculator(**common, compute_dsigma_dr=False)

    atoms_full = uq_setup["atoms"][0].copy()
    atoms_gamma = uq_setup["atoms"][0].copy()
    atoms_full.calc = calc_full
    atoms_gamma.calc = calc_gamma

    e_full = atoms_full.get_potential_energy()
    f_full = atoms_full.get_forces()
    e_gamma = atoms_gamma.get_potential_energy()
    f_gamma = atoms_gamma.get_forces()

    np.testing.assert_allclose(e_gamma, e_full, rtol=1e-10, atol=1e-10)
    np.testing.assert_allclose(f_gamma, f_full, rtol=1e-10, atol=1e-10)

    np.testing.assert_allclose(
        calc_gamma.results[uqc.ATOMIC_SIGMA],
        calc_full.results[uqc.ATOMIC_SIGMA],
        rtol=1e-10,
        atol=1e-10,
    )
    assert uqc.DSIGMA_DR not in calc_gamma.results
    assert uqc.VIRIAL_SIGMA not in calc_gamma.results
    assert uqc.DSIGMA_DR in calc_full.results
    assert uqc.VIRIAL_SIGMA in calc_full.results

    # Cluster index must be present in both modes and identical
    assert uqc.GMM_CLUSTER in calc_full.results
    assert uqc.GMM_CLUSTER in calc_gamma.results
    np.testing.assert_array_equal(
        np.asarray(calc_full.results[uqc.GMM_CLUSTER]).astype(int),
        np.asarray(calc_gamma.results[uqc.GMM_CLUSTER]).astype(int),
    )


def test_savedmodel_exports_both_uq_signatures(uq_setup, tmp_path):
    """Round-trip: a SavedModel exported with ``gmm_uq_model=`` must carry
    BOTH ``compute_uq`` (full) and ``compute_uq_gamma_only`` signatures,
    and a TPCalculator built from that path must be able to switch
    between modes via ``enable_uq(mode=...)``."""
    from tensorpotential.uq.gmmuq import GMMUQModel
    from tensorpotential.tensorpot import TensorPotential
    from tensorpotential.uq.compute import (
        ComputeStructureEnergyAndForcesAndVirialAndUncertainty,
    )
    from tensorpotential.calculator.asecalculator import TPCalculator
    from tensorpotential.instructions import load_instructions

    # Inject synthetic interp_thresholds so save_model() accepts the artifact.
    raw = dict(np.load(uq_setup["artifact_path"], allow_pickle=True))
    n_elem = len(uq_setup["element_map"])
    # Centroids are stored under "centroids_<elem>"; pull one to get n_clusters
    one_elem = int(raw["elements"][0])
    n_clusters = raw[f"{uqc.CENTROIDS}_{one_elem}"].shape[0]
    raw["interp_thresholds"] = np.full((n_elem, n_clusters), 1.0, dtype=np.float64)
    artifact_with_thresh = str(tmp_path / "artifact_with_thresh.npz")
    np.savez(artifact_with_thresh, **raw)

    gmm_uq = GMMUQModel(artifact_with_thresh, param_dtype=tf.float64)
    instructions = load_instructions(uq_setup["model_yaml"])
    # save_model() bakes the basis-RP feature into the exported graph from the
    # artifact's stored R, so no manual feature patch is needed here.
    tp = TensorPotential(
        instructions,
        model_compute_function=ComputeStructureEnergyAndForcesAndVirialAndUncertainty(gmm_uq),
        param_dtype=tf.float64,
    )
    tp.load_checkpoint(uq_setup["checkpoint"])

    export_path = str(tmp_path / "saved_model_with_uq")
    tp.model.save_model(
        export_path,
        gmm_uq_model=gmm_uq,
        input_signature_float_dtype=tf.float64,
    )

    # SavedModel itself carries both signatures.
    loaded = tf.saved_model.load(export_path)
    assert "compute_uq" in loaded.signatures
    assert "compute_uq_gamma_only" in loaded.signatures

    # Calculator surfaces both modes and can switch.
    calc = TPCalculator(export_path, enable_uq_if_available=True)
    assert set(calc.available_uq_modes) == {"full", "gamma_only"}
    # Default should pick "full" when both are available
    assert calc._uq_state["mode"] == "full"
    calc.enable_uq(mode="gamma_only")
    assert calc._uq_state["mode"] == "gamma_only"
    calc.enable_uq(mode="full")
    assert calc._uq_state["mode"] == "full"
