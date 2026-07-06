import numpy as np
import pytest
import tensorflow as tf
from tensorpotential.uq.gmmuq import GMMUQModel, _MISSING_ELEM_INV_COV_DIAG
from tensorpotential.uq.feature_extraction import setup_feature_calculator

def test_gmm_model_init(uq_setup):
    """Test GMMUQModel loading and metadata inspection."""
    artifact_path = uq_setup["artifact_path"]
    model = GMMUQModel(artifact_path)

    assert model.n_elements == 4
    assert model.D == uq_setup["feature_dim"]
    assert model.K_max == 4
    assert 0 in model.element_to_idx
    assert model.element_symbols == ["Mo", "Nb", "Ta", "W"]
    assert model.__repr__().startswith("GMMUQModel")

def test_gmm_model_dtype_autodetect(uq_setup, tmp_path):
    """GMMUQModel with no param_dtype should infer dtype from the npz centroids."""
    artifact_path = uq_setup["artifact_path"]
    # This fixture is saved float64 (store_fp32=False, see conftest), so
    # auto-detect must yield float64.
    model = GMMUQModel(artifact_path)
    assert model.param_dtype == tf.float64
    assert model.centroids.dtype == tf.float64

    # Explicit override must still work
    model_f32 = GMMUQModel(artifact_path, param_dtype=tf.float32)
    assert model_f32.param_dtype == tf.float32
    assert model_f32.centroids.dtype == tf.float32

    # A float32-stored artifact (the production default) must auto-detect float32.
    from tensorpotential.uq.artifact_builder import GMMUQArtifactBuilder
    fp32_path = str(tmp_path / "uq_artifacts_fp32.npz")
    GMMUQArtifactBuilder.save_artifacts(
        fp32_path,
        GMMUQArtifactBuilder.load(artifact_path),
        store_fp32=True,
        element_map=np.array(list(uq_setup["element_map"].keys())),
    )
    model_auto = GMMUQModel(fp32_path)
    assert model_auto.param_dtype == tf.float32
    assert model_auto.centroids.dtype == tf.float32

def test_gmm_model_compute_correctness(uq_setup):
    """Verify TF compute matches naive numpy Mahalanobis implementation."""
    artifact_path = uq_setup["artifact_path"]
    model = GMMUQModel(artifact_path, param_dtype=tf.float64)
    
    # Sample some data
    features = uq_setup["features"][:10]
    element_indices = uq_setup["element_indices"][:10]
    
    # TF compute
    sigma_tf, total_sigma_tf, _ = model.compute(features, element_indices)
    
    # Manual Numpy compute for validation
    sigma_np = []
    for i in range(len(features)):
        feat = features[i]
        elem_idx = element_indices[i]
        
        # Get centroids and inv_covs for this element
        centroids = model.centroids[elem_idx].numpy()
        inv_covs = model.inv_covs[elem_idx].numpy()
        
        # Find nearest centroid (Euclidean)
        diffs = feat[None, :] - centroids
        dists_sq = np.sum(diffs**2, axis=1)
        assign = np.argmin(dists_sq)
        
        # Mahalanobis
        delta = feat - centroids[assign]
        s2 = delta @ inv_covs[assign] @ delta
        sigma_np.append(np.sqrt(s2 + 1e-8))
    
    sigma_np = np.array(sigma_np, dtype=np.float64)
    assert np.allclose(sigma_tf.numpy(), sigma_np, atol=1e-6)
    assert np.allclose(total_sigma_tf.numpy(), np.sum(sigma_np), atol=1e-6)

def test_gmm_model_chunking(uq_setup):
    """Ensure chunked evaluation via __call__ matches full evaluation."""
    artifact_path = uq_setup["artifact_path"]
    # Force small chunk size to trigger internal loop
    model = GMMUQModel(artifact_path, chunk_size=5)
    
    features = uq_setup["features"][:20]
    element_indices = uq_setup["element_indices"][:20]
    
    # Full compute (via non-chunked core)
    sigma_full, total_full, assign_full = model.compute(features, element_indices)

    # Chunked compute (via __call__ entry point)
    sigma_chunked, total_chunked, assign_chunked = model(features, element_indices, verbose=False)

    # Cluster assignment (argmin of Euclidean distance) is well-conditioned and
    # must be bit-identical regardless of chunking.
    assert np.array_equal(assign_full.numpy(), assign_chunked.numpy())
    # Chunking is exact; with the reduced, well-conditioned fixture covariance the
    # per-atom Mahalanobis matches the full reduction to the float tolerance.
    assert np.allclose(sigma_full.numpy(), sigma_chunked.numpy(), rtol=1e-5)
    assert np.allclose(total_full.numpy(), total_chunked.numpy(), rtol=1e-5)

def test_gmm_model_incremental_update(uq_setup):
    """Test incremental artifact update and tensor rebuild."""
    artifact_path = uq_setup["artifact_path"]
    model = GMMUQModel(artifact_path, regularization=1e-6)

    # Feed a contiguous (non-representative) half of the features spanning every
    # element. Updating with a subset shifts the per-cluster scatter/count ratio
    # of any element that formed real clusters, so inv_cov must change. (Feeding
    # the first few atoms is not robust: the tiny fixture leaves some elements
    # with only sentinel clusters, which an update leaves untouched.)
    half = len(uq_setup["features"]) // 2
    new_features = uq_setup["features"][:half]
    new_elements = uq_setup["element_indices"][:half]

    orig_inv_covs = model.inv_covs.numpy().copy()

    # Perform update
    model.update_one(new_features, new_elements)

    # Check that inv_covs changed (a real cluster's covariance must shift).
    # atol=0.0 makes this scale-invariant: the shifted cluster's inv_cov entries
    # can be ~1e-20, far below the default atol=1e-8, yet are a real change.
    assert not np.allclose(model.inv_covs.numpy(), orig_inv_covs, atol=0.0)

def test_gmm_eval_from_atoms(uq_setup):
    """Test full pipeline: Extract features from ASE atoms -> GMM evaluation."""
    artifact_path = uq_setup["artifact_path"]
    model = GMMUQModel(artifact_path)

    calc = setup_feature_calculator(
        uq_setup["model_yaml"],
        uq_setup["checkpoint"],
        feature_spec=uq_setup["feature_spec"],
    )
    atoms = uq_setup["atoms"][:2]

    sigma, total, cluster_assign = model.eval_from_atoms(
        calc,
        atoms,
        element_map=uq_setup["element_map"],
        verbose=False
    )

    total_atoms = sum(len(at) for at in atoms)
    assert len(sigma) == total_atoms
    assert len(cluster_assign) == total_atoms
    assert isinstance(total, float)
    assert total > 0


def test_gmm_missing_elements_fictitious_inv_cov(uq_setup):
    """Elements absent from artifacts get a large isotropic inv_cov and fire a UserWarning."""
    D = uq_setup["feature_dim"]
    K = 8
    # Build artifacts for only 2 of the 4 model elements (Mo=0, Nb=1).
    # Ta=2 and W=3 are intentionally left out.
    from tensorpotential.uq.artifact_builder import GMMUQArtifactBuilder
    builder = GMMUQArtifactBuilder(n_clusters=K, feature_dim=D)
    feats_01 = uq_setup["features"][uq_setup["element_indices"] < 2]
    elems_01 = uq_setup["element_indices"][uq_setup["element_indices"] < 2]
    stream = [(feats_01, elems_01, np.ones(len(feats_01)))]
    builder.fit_centroids(stream, verbose=False)
    builder.accumulate_scatter(stream, verbose=False)
    artifacts = builder.finalize()

    # element_map covers all 4 elements so the model knows the full index space
    element_map = np.array(["Mo", "Nb", "Ta", "W"])
    with pytest.warns(UserWarning) as caught:
        model = GMMUQModel.from_artifacts(
            artifacts, extra_data={"element_map": element_map}
        )

    # Warning must name the two absent elements
    msg = str(caught[0].message)
    assert "Ta" in msg and "W" in msg

    # Absent rows must have the large diagonal inv_cov
    for absent in (2, 3):
        row = model.inv_covs[absent, 0].numpy()
        assert np.allclose(np.diag(row), _MISSING_ELEM_INV_COV_DIAG)

    # Absent elements must produce much higher sigma than fitted ones.
    # Use features that belong to fitted elements (0 or 1) for a clean comparison.
    mask_fitted = uq_setup["element_indices"] < 2
    feats_f = tf.constant(uq_setup["features"][mask_fitted][:20], dtype=tf.float64)
    elems_f = tf.constant(uq_setup["element_indices"][mask_fitted][:20], dtype=tf.int32)
    elems_m = tf.constant(np.full(len(elems_f), 2, dtype=np.int32))  # Ta — no training data
    sigma_fitted, _, _ = model.compute(feats_f, elems_f)
    sigma_missing, _, _ = model.compute(feats_f, elems_m)
    assert sigma_missing.numpy().mean() > sigma_fitted.numpy().mean() * 2
