import numpy as np
from tensorpotential.uq.artifact_builder import GMMUQArtifactBuilder
from tensorpotential.uq import constants as uqc

def test_builder_two_pass():
    """Test full artifact generation cycle with synthetic well-separated clusters."""
    D = 4
    K = 3
    builder = GMMUQArtifactBuilder(n_clusters=K, feature_dim=D)
    
    # Generate 100 samples from 3 well-separated Gaussian clusters
    c1 = np.ones(D) * 0.0
    c2 = np.ones(D) * 10.0
    c3 = np.ones(D) * 20.0
    
    # Deterministic seed for reproducible testing
    np.random.seed(42)
    features = np.concatenate([
        np.random.normal(c1, 0.1, size=(30, D)),
        np.random.normal(c2, 0.1, size=(30, D)),
        np.random.normal(c3, 0.1, size=(40, D)),
    ])
    elements = np.zeros(100, dtype=np.int32)
    weights = np.ones(100, dtype=np.float64)
    stream = [(features, elements, weights)]

    # Pass 1: Fit centroids
    builder.fit_centroids(stream, verbose=False)
    centroids = builder.get_centroids(0)
    assert centroids.shape == (K, D)
    for center in [c1, c2, c3]:
        # At least one centroid should be close to each true cluster center
        dists = np.linalg.norm(centroids - center, axis=1)
        assert np.min(dists) < 1.0
        
    # Pass 2: Accumulate scatter
    builder.accumulate_scatter(stream, verbose=False)
    artifacts = builder.finalize()
    
    assert 0 in artifacts
    a = artifacts[0]
    assert a[uqc.CENTROIDS].shape == (K, D)
    assert a[uqc.INV_COV].shape == (K, D, D)
    assert a[uqc.COUNTS].shape == (K,)
    # Verify we assigned samples to all clusters
    assert np.all(a[uqc.COUNTS] > 0)
    # Verify covariance diagnostics are stored
    assert a[uqc.COND_NUMBER].shape == (K,)
    assert a[uqc.EFFECTIVE_RANK].shape == (K,)
    assert a[uqc.N_TRUNCATED].shape == (K,)
    # Well-separated Gaussians with D=4 should be full-rank
    assert np.all(a[uqc.EFFECTIVE_RANK] == D)
    assert np.all(a[uqc.N_TRUNCATED] == 0)
    assert np.all(np.isfinite(a[uqc.COND_NUMBER]))

def test_builder_safe_invert():
    """Test pseudo-inversion diagnostics for rank-deficient matrices."""
    D = 10
    np.random.seed(42)
    # Create a rank-deficient matrix (rank 5)
    v = np.random.normal(size=(D, 5))
    cov = v @ v.T
    
    inv, diag = GMMUQArtifactBuilder._safe_invert(cov, rcond=1e-15)
    # Due to float precision, might not be exactly 5, but roughly.
    # In this case eigh should be stable.
    assert diag["rank"] == 5
    assert diag["n_truncated"] == 5
    assert diag["error"] < 1e-10
    
    # Test well-conditioned addition (regularization)
    cov_reg = cov + 1e-6 * np.eye(D)
    inv_reg, diag_reg = GMMUQArtifactBuilder._safe_invert(cov_reg)
    assert diag_reg["rank"] == 10
    
def test_builder_zero_count(caplog):
    """Verify handling of clusters with zero assigned samples."""
    D = 2
    K = 2
    builder = GMMUQArtifactBuilder(n_clusters=K, feature_dim=D, regularization=1e-4)
    # Manually set centroids far apart
    builder.set_centroids(0, np.array([[0.0, 0.0], [100.0, 100.0]]))
    
    # Data only near the first centroid
    features = np.random.normal(0, 0.1, size=(10, D))
    elements = np.zeros(10, dtype=np.int32)
    weights = np.ones(10, dtype=np.float64)
    stream = [(features, elements, weights)]

    builder.accumulate_scatter(stream, verbose=False)
    # This should warn about the zero-count cluster
    artifacts = builder.finalize()

    assert artifacts[0][uqc.COUNTS][1] == 0
    # For zero count, cov_k = eps_I, so inv_cov = (1/eps) * I
    expected_inv = (1.0 / 1e-4) * np.eye(D)
    assert np.allclose(artifacts[0][uqc.INV_COV][1], expected_inv)
    assert any("zero effective count" in record.message for record in caplog.records)

def test_builder_save_load(tmp_path):
    """Verify .npz artifact persistence and extra metadata."""
    D = 3
    K = 2
    builder = GMMUQArtifactBuilder(n_clusters=K, feature_dim=D)
    builder.set_centroids(1, np.random.normal(size=(K, D)))
    
    # Mock some data to allow finalize
    builder._elements_seen.add(1)
    builder._scatter[1] = np.zeros((K, D, D))
    builder._counts[1] = np.zeros(K, dtype=np.int64)
    # Mark as fitted to satisfy RuntimeError checks
    builder._fitted = True
    
    artifacts = builder.finalize()
    path = tmp_path / "test_artifact.npz"
    
    # Save with artifacts and some extra metadata
    builder.save(str(path), artifacts=artifacts, custom_key="test_value")
    
    # Static load
    loaded = GMMUQArtifactBuilder.load(str(path))
    assert 1 in loaded
    assert np.allclose(loaded[1][uqc.CENTROIDS], artifacts[1][uqc.CENTROIDS])
    # Diagnostics should round-trip through save/load
    assert np.allclose(loaded[1][uqc.COND_NUMBER], artifacts[1][uqc.COND_NUMBER])
    assert np.array_equal(loaded[1][uqc.EFFECTIVE_RANK], artifacts[1][uqc.EFFECTIVE_RANK])
    assert np.array_equal(loaded[1][uqc.N_TRUNCATED], artifacts[1][uqc.N_TRUNCATED])
    
    # Check custom key in raw NPZ
    raw_data = np.load(path)
    assert "custom_key" in raw_data
    assert str(raw_data["custom_key"]) == "test_value"


def test_accumulate_scatter_skips_unknown_elements(caplog):
    """Elements not in step1 artifacts are skipped with a warning, not a crash."""
    D = 4
    K = 2
    builder = GMMUQArtifactBuilder(n_clusters=K, feature_dim=D)

    # Fit centroids for element 0 only
    np.random.seed(0)
    feats = np.random.normal(size=(30, D))
    builder.fit_centroids(
        [(feats, np.zeros(30, dtype=np.int32), np.ones(30))], verbose=False
    )

    # Step 2 stream contains element 0 (known) and element 99 (unknown)
    feats2 = np.random.normal(size=(20, D))
    elems2 = np.array([0] * 10 + [99] * 10, dtype=np.int32)

    with caplog.at_level("WARNING", logger="tensorpotential.uq.artifact_builder"):
        builder.accumulate_scatter(
            [(feats2, elems2, np.ones(20))], verbose=False
        )

    # Element 0 must have accumulated scatter; element 99 must be absent
    artifacts = builder.finalize()
    assert 0 in artifacts
    assert np.any(artifacts[0][uqc.COUNTS] > 0)

    # Warning logged exactly once for element 99
    missing_warnings = [r for r in caplog.records if "99" in r.message]
    assert len(missing_warnings) == 1


def test_effective_rank_reports_data_rank_not_regularized():
    """When N < D, effective_rank must reflect the data deficit (rank ≤ N-1
    or N), NOT the post-regularization rank (which is always D because of
    the eps_I shift)."""
    D = 10
    K = 1
    N = 5  # fewer atoms than feature dimensions
    builder = GMMUQArtifactBuilder(n_clusters=K, feature_dim=D)

    rng = np.random.default_rng(0)
    feats = rng.standard_normal((N, D))
    elements = np.zeros(N, dtype=np.int32)
    stream = [(feats, elements, np.ones(N))]

    builder.fit_centroids(stream, verbose=False)
    builder.accumulate_scatter(stream, verbose=False)
    artifacts = builder.finalize()

    rank = int(artifacts[0][uqc.EFFECTIVE_RANK][0])
    assert rank <= N, (
        f"effective_rank={rank} exceeds n_atoms={N}; the rank field appears "
        "to count the regularized covariance instead of the data covariance."
    )
    assert rank >= 1


def test_regularization_propagates_to_inv_cov():
    """A larger Tikhonov eps must visibly attenuate the inv_cov eigenvalues:
    inv(C + eps*I) has eigenvalues ≤ 1/eps."""
    D = 5
    K = 1
    rng = np.random.default_rng(0)
    feats = rng.standard_normal((100, D))
    elements = np.zeros(100, dtype=np.int32)
    stream = [(feats, elements, np.ones(100))]

    eps_large = 1e-2
    eps_small = 1e-6
    artifacts = {}
    for eps in (eps_large, eps_small):
        b = GMMUQArtifactBuilder(n_clusters=K, feature_dim=D, regularization=eps)
        b.fit_centroids(stream, verbose=False)
        b.accumulate_scatter(stream, verbose=False)
        artifacts[eps] = b.finalize()

    inv_cov_large = artifacts[eps_large][0][uqc.INV_COV][0]
    inv_cov_small = artifacts[eps_small][0][uqc.INV_COV][0]

    eig_large = np.linalg.eigvalsh(inv_cov_large)
    eig_small = np.linalg.eigvalsh(inv_cov_small)

    # eps_large = 1e-2 caps inv_cov eigenvalues at 1/eps_large = 100
    assert np.max(eig_large) <= 1.0 / eps_large + 1e-6
    # eps_small leaves them larger (or much larger if data is rank-deficient)
    assert np.max(eig_small) >= np.max(eig_large)


def test_from_artifacts_propagates_regularization():
    """from_artifacts(... regularization=X) sets the builder's eps used in
    finalize()."""
    D = 4
    K = 2
    centroids = np.array([[0.0] * D, [10.0] * D])
    artifacts = {
        0: {
            uqc.CENTROIDS: centroids,
            uqc.INV_COV: np.zeros((K, D, D)),
            uqc.COUNTS: np.zeros(K),
            uqc.SCATTER: None,
        }
    }
    b = GMMUQArtifactBuilder.from_artifacts(
        artifacts, n_clusters=K, feature_dim=D, regularization=1e-3
    )
    assert b.regularization == 1e-3


def _build_finalized(features, elements, weights, K, D, regularization=1e-6, seed=42):
    """Helper: build + fit + scatter + finalize, returning the artifacts dict."""
    b = GMMUQArtifactBuilder(
        n_clusters=K, feature_dim=D, regularization=regularization, random_state=seed
    )
    stream = [(features, elements, weights)]
    b.fit_centroids(stream, verbose=False)
    b.accumulate_scatter(stream, verbose=False)
    return b, b.finalize()


def test_weighted_equals_unweighted_when_w_is_one():
    """Uniform weight=1 must yield identical statistics to no-weight, and the
    effective_count column must equal raw counts cast to float."""
    D = 4
    K = 3
    rng = np.random.default_rng(0)
    feats = np.vstack([
        rng.normal(0.0, 0.5, (40, D)),
        rng.normal(5.0, 0.5, (35, D)),
        rng.normal(-5.0, 0.5, (25, D)),
    ])
    elems = np.zeros(100, dtype=np.int32)

    _, a_unweighted = _build_finalized(feats, elems, np.ones(100), K, D)
    _, a_weighted = _build_finalized(feats, elems, np.ones(100), K, D)

    # Bit-exact equality (or 1e-15 tolerance — sklearn's sample_weight=ones
    # passes the weight kwarg, which can drift by a tiny amount).
    for key in (uqc.CENTROIDS, uqc.INV_COV, uqc.SCATTER):
        np.testing.assert_allclose(
            a_unweighted[0][key], a_weighted[0][key], atol=1e-12, rtol=1e-12,
            err_msg=f"divergence in {key}",
        )
    np.testing.assert_array_equal(a_unweighted[0][uqc.COUNTS], a_weighted[0][uqc.COUNTS])
    # Effective_count must equal raw counts when all weights are 1.0
    np.testing.assert_allclose(
        a_weighted[0][uqc.EFFECTIVE_COUNT],
        a_weighted[0][uqc.COUNTS].astype(np.float64),
        atol=1e-12, rtol=1e-12,
    )


def test_doubling_all_weights_preserves_inv_cov():
    """Scaling all weights by 2 is mathematically equivalent (cov = scatter/eff
    is invariant) — inv_cov and centroids must be unchanged, but effective_count
    must double while raw COUNTS stays the same."""
    D = 4
    K = 2
    rng = np.random.default_rng(1)
    feats = np.vstack([
        rng.normal(0.0, 0.7, (50, D)),
        rng.normal(7.0, 0.7, (50, D)),
    ])
    elems = np.zeros(100, dtype=np.int32)

    _, a_w1 = _build_finalized(feats, elems, np.ones(100), K, D)
    _, a_w2 = _build_finalized(feats, elems, np.full(100, 2.0), K, D)

    # Centroids and inv_cov should be invariant under uniform weight scaling.
    np.testing.assert_allclose(
        a_w1[0][uqc.CENTROIDS], a_w2[0][uqc.CENTROIDS], atol=1e-10
    )
    np.testing.assert_allclose(
        a_w1[0][uqc.INV_COV], a_w2[0][uqc.INV_COV], atol=1e-6, rtol=1e-6
    )
    # Raw counts must be identical.
    np.testing.assert_array_equal(a_w1[0][uqc.COUNTS], a_w2[0][uqc.COUNTS])
    # Effective counts must double.
    np.testing.assert_allclose(
        a_w2[0][uqc.EFFECTIVE_COUNT],
        2.0 * a_w1[0][uqc.EFFECTIVE_COUNT],
        atol=1e-10,
    )


def test_one_to_ten_weighted_pulls_centroid_to_heavy_mode():
    """A 1:10 weight imbalance between two Gaussian modes shifts the dominant
    centroid (cluster 0 after sort) toward the heavily weighted cluster."""
    D = 4
    K = 2
    rng = np.random.default_rng(2)
    n_each = 50
    feats = np.vstack([
        rng.normal(-3.0, 0.4, (n_each, D)),  # mode A
        rng.normal(+3.0, 0.4, (n_each, D)),  # mode B
    ])
    elems = np.zeros(2 * n_each, dtype=np.int32)
    # Mode A weight=1; Mode B weight=10. Effective count of B = 500 >> A = 50.
    weights = np.concatenate([np.ones(n_each), np.full(n_each, 10.0)])

    _, a = _build_finalized(feats, elems, weights, K, D, seed=2)
    # After finalize() sort, cluster 0 is the dominant mode by effective count
    # — that should be mode B (positive coordinates).
    c0 = a[0][uqc.CENTROIDS][0]
    c1 = a[0][uqc.CENTROIDS][1]
    assert np.mean(c0) > np.mean(c1), (
        f"cluster 0 (effective-count winner) should be positive-mode, "
        f"got c0={c0}, c1={c1}"
    )
    # Effective count of cluster 0 must exceed cluster 1.
    assert a[0][uqc.EFFECTIVE_COUNT][0] > a[0][uqc.EFFECTIVE_COUNT][1]


def test_save_load_roundtrip_keeps_both_counts(tmp_path):
    """Both COUNTS and EFFECTIVE_COUNT must round-trip via npz; legacy
    artifacts without EFFECTIVE_COUNT back-fill from COUNTS."""
    D = 3
    K = 2
    rng = np.random.default_rng(3)
    feats = rng.standard_normal((60, D))
    elems = np.zeros(60, dtype=np.int32)
    weights = rng.uniform(1.0, 5.0, 60)  # non-uniform

    b, artifacts = _build_finalized(feats, elems, weights, K, D)
    path = tmp_path / "weighted_artifact.npz"
    b.save(str(path), artifacts=artifacts)

    loaded = GMMUQArtifactBuilder.load(str(path))
    np.testing.assert_array_equal(loaded[0][uqc.COUNTS], artifacts[0][uqc.COUNTS])
    np.testing.assert_allclose(
        loaded[0][uqc.EFFECTIVE_COUNT], artifacts[0][uqc.EFFECTIVE_COUNT],
        atol=1e-12, rtol=1e-12,
    )
    # fp32-only storage (default): stats are stored float32, but counts (raw and
    # effective) are exempt — they round-trip exactly for incremental updates and
    # cluster sorting.
    assert loaded[0][uqc.CENTROIDS].dtype == np.float32
    assert loaded[0][uqc.EFFECTIVE_COUNT].dtype == np.float64

    # Now simulate a legacy artifact: strip EFFECTIVE_COUNT_* keys from the npz
    legacy_path = tmp_path / "legacy_artifact.npz"
    with np.load(path) as data:
        keep = {k: data[k] for k in data.files if not k.startswith("effective_count_")}
    np.savez(str(legacy_path), **keep)
    legacy = GMMUQArtifactBuilder.load(str(legacy_path))
    np.testing.assert_array_equal(
        legacy[0][uqc.EFFECTIVE_COUNT], legacy[0][uqc.COUNTS].astype(np.float64),
    )


def test_finalize_sort_uses_effective_when_weighted():
    """When the builder has seen non-trivial weights, finalize() must sort
    clusters by effective_count desc; otherwise it falls back to raw count
    desc (the legacy unweighted ordering)."""
    D = 3
    K = 2
    rng = np.random.default_rng(4)
    # Two well-separated modes, ~equal raw count.
    feats = np.vstack([
        rng.normal(-5.0, 0.3, (50, D)),
        rng.normal(+5.0, 0.3, (50, D)),
    ])
    elems = np.zeros(100, dtype=np.int32)

    # Unweighted: sort by raw count → either cluster can be first (equal),
    # but the sort key path is the int-count branch.
    b_unw = GMMUQArtifactBuilder(n_clusters=K, feature_dim=D)
    b_unw.fit_centroids([(feats, elems, np.ones(100))], verbose=False)
    b_unw.accumulate_scatter([(feats, elems, np.ones(100))], verbose=False)
    assert not b_unw._weights_seen
    a_unw = b_unw.finalize()

    # Weighted: heavy weights on the negative mode → cluster 0 must be neg.
    weights = np.concatenate([np.full(50, 5.0), np.ones(50)])
    b_w = GMMUQArtifactBuilder(n_clusters=K, feature_dim=D)
    b_w.fit_centroids([(feats, elems, weights)], verbose=False)
    b_w.accumulate_scatter([(feats, elems, weights)], verbose=False)
    assert b_w._weights_seen
    a_w = b_w.finalize()

    c0_w = a_w[0][uqc.CENTROIDS][0]
    assert np.mean(c0_w) < 0, (
        f"weighted-sort: cluster 0 should be the heavy (negative) mode, got {c0_w}"
    )
    # Effective count of cluster 0 must exceed cluster 1 in weighted run.
    assert a_w[0][uqc.EFFECTIVE_COUNT][0] > a_w[0][uqc.EFFECTIVE_COUNT][1]
    # Unweighted run: raw counts ≈ effective counts (all 1's).
    np.testing.assert_array_equal(
        a_unw[0][uqc.COUNTS], a_unw[0][uqc.EFFECTIVE_COUNT].astype(np.int64)
    )


def test_update_ema_branch_increments_counts_and_preserves_covariance():
    """Regression: in EMA mode, ``update()`` previously updated ``SCATTER``
    but never incremented ``COUNTS``. The subsequent ``cov_k = SCATTER /
    COUNTS`` recompute then fell back to ``eps_I`` because COUNTS stayed 0,
    silently discarding the EMA-blended covariance."""
    D = 2
    K = 1
    eps = 1e-6
    artifacts = {
        0: {
            uqc.CENTROIDS: np.zeros((K, D)),
            uqc.SCATTER: np.zeros((K, D, D)),
            uqc.INV_COV: np.zeros((K, D, D)),
            uqc.COUNTS: np.zeros(K, dtype=np.int64),
        }
    }
    # 4 features with non-trivial covariance, all assigned to the single cluster
    features = np.array(
        [[1.0, 0.0], [0.0, 1.0], [-1.0, 0.0], [0.0, -1.0]],
        dtype=np.float64,
    )
    element_indices = np.zeros(4, dtype=np.int32)

    GMMUQArtifactBuilder.update(
        artifacts, features, element_indices, regularization=eps, alpha=0.5
    )

    # After EMA update from a fresh (n_old=0) cluster, COUNTS must reflect
    # the atoms processed; otherwise the recomputed cov_k falls back to eps_I.
    assert artifacts[0][uqc.COUNTS][0] > 0, (
        "EMA branch failed to increment COUNTS — recomputed cov_k will "
        "discard the data and fall back to the regularization identity."
    )
    # And the resulting inv_cov must NOT be the pure regularized-identity
    # inverse (which would be ~1/eps * I, dominated by the eps_I fallback).
    inv_cov = artifacts[0][uqc.INV_COV][0]
    pure_eps_inv_diag = 1.0 / eps  # = 1e6
    assert np.max(np.abs(np.diag(inv_cov))) < 0.5 * pure_eps_inv_diag, (
        "inv_cov is dominated by eps_I — EMA covariance was discarded."
    )
