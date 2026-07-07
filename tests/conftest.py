import pytest
import os
os.environ["TF_USE_LEGACY_KERAS"] = "1"
import pandas as pd
import numpy as np
import tensorflow as tf
from tensorpotential.potentials import get_preset
from tensorpotential.tensorpot import TensorPotential
from tensorpotential.uq.artifact_builder import GMMUQArtifactBuilder
from tensorpotential.uq.feature_extraction import setup_feature_calculator, extract_features_bulk
from tensorpotential.uq.factories import make_basis_rp_spec
from tensorpotential.instructions import save_instructions_dict

@pytest.fixture(scope="session")
def uq_setup(tmp_path_factory):
    """Session-level fixture to provide a real UQ artifact and model for testing.
    
    Uses GRACE-2L-latest randomly initialized and MoNbTaW test data.
    """
    tmp_dir = tmp_path_factory.mktemp("uq_test")
    model_dir = tmp_dir / "model"
    model_dir.mkdir()
    
    # 1. Load data
    df_path = os.path.join(os.path.dirname(__file__), "data/MoNbTaW_test50.pkl.gz")
    df = pd.read_pickle(df_path)
    atoms = df["ase_atoms"].tolist()
    element_map = {"Mo": 0, "Nb": 1, "Ta": 2, "W": 3}
    
    # 2. Create model
    # Seed the weight init so the (random-init) fixture is reproducible
    # run-to-run; otherwise the covariance conditioning — and hence which
    # clusters are degenerate — drifts and makes the GMM math tests flaky.
    tf.random.set_seed(42)
    GRACE_2LAYER = get_preset("GRACE_2LAYER_latest")
    # Using lmax=1 to keep it relatively fast for tests while being representative
    instructions = GRACE_2LAYER(element_map=element_map, lmax=1).get_instructions()
    
    model_yaml = str(model_dir / "model.yaml")
    checkpoint = str(model_dir / "checkpoint")
    
    save_instructions_dict(model_yaml, instructions, param_dtype=tf.float64)
    tp = TensorPotential(instructions, param_dtype=tf.float64)
    tp.save_checkpoint(checkpoint_name=checkpoint)
    
    # 3. Generate artifacts. A reduced basis-RP dim (16) + few clusters (4) keep
    # each per-element covariance full-rank and well-conditioned on this tiny
    # fixture (the 128-D default is rank-deficient here). The feature_spec used to
    # extract `features` must match the spec stamped below so the stored R
    # reproduces these features byte-for-byte.
    RP_DIM = 16
    N_CLUSTERS = 4
    feature_spec = {"out_dim": RP_DIM, "seed": 42}
    calc = setup_feature_calculator(
        model_yaml, checkpoint, param_dtype=tf.float64, feature_spec=feature_spec
    )
    features, element_indices = extract_features_bulk(calc, atoms, element_map=element_map)

    builder = GMMUQArtifactBuilder(n_clusters=N_CLUSTERS, feature_dim=features.shape[1])
    # Pass a single-batch iterator as expected by the builder
    stream = [(features, element_indices, np.ones(len(features)))]
    builder.fit_centroids(stream, verbose=False)
    builder.accumulate_scatter(stream, verbose=False)
    
    artifact_path = str(tmp_dir / "uq_artifacts.npz")
    # element_map in .npz is used by GMMUQModel to retrieve symbols.
    # Stamp the self-describing basis-RP spec so the eval / SavedModel paths that
    # require a basis-RP artifact accept this fixture. rp_dim/seed must match the
    # feature_spec used above so the stored R is byte-identical to the one that
    # produced `features`. store_fp32=False keeps the GMM stats in float64: the
    # GMM math tests compare against exact double precision and
    # test_gmm_model_dtype_autodetect asserts this fixture auto-detects float64.
    # (The reduced RP_DIM already makes the covariance full-rank; the float32
    # production path is covered by the trained_uq_setup fixture.)
    builder.save(
        artifact_path,
        element_map=np.array(list(element_map.keys())),
        store_fp32=False,
        **make_basis_rp_spec(model_yaml, rp_dim=RP_DIM),
    )
    
    return {
        "artifact_path": artifact_path,
        "model_yaml": model_yaml,
        "checkpoint": checkpoint,
        "element_map": element_map,
        "atoms": atoms,
        "feature_dim": features.shape[1],
        "features": features,
        "element_indices": element_indices,
        "feature_spec": feature_spec,
    }
