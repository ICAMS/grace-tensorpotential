"""The NumPy CALM kernel (tensorpotential.uq.calm_numpy) agrees with the TensorFlow path,
and GMMUQModel.alpha_search_update equals the trial_gamma + update_one procedure."""

import copy

import numpy as np
import pytest
import tensorflow as tf

from tensorpotential.uq import constants as uqc
from tensorpotential.uq.calm_numpy import nearest_cluster
from tensorpotential.uq.cli.strategies import CandidateGamma, whiten_features
from tensorpotential.uq.gmmuq import GMMUQModel

ALPHAS = (0.0005, 0.001, 0.002, 0.005, 0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 1.0)


def test_nearest_cluster_matches_brute_force_with_chunks_and_ties():
    rng = np.random.default_rng(0)
    c = rng.standard_normal((5, 3))
    f = rng.standard_normal((23, 3))
    ref = np.argmin(((f[:, None, :] - c[None]) ** 2).sum(-1), axis=1)
    np.testing.assert_array_equal(nearest_cluster(f, c, chunk=4), ref)
    # exactly equidistant from clusters 1 and 3 (squared distance 1): lowest k wins
    tie = np.array([[3.0, 3.0], [1.0, 0.0], [0.0, 5.0], [-1.0, 0.0]])
    assert nearest_cluster(np.zeros((1, 2)), tie)[0] == 1


@pytest.fixture
def tower(hal_uq_setup):
    return GMMUQModel(hal_uq_setup["artifact_path"], param_dtype=tf.float64)


def test_numpy_gamma_equals_tensorflow(hal_uq_setup, tower):
    feats = np.asarray(hal_uq_setup["features"], dtype=np.float64)
    eidx = np.asarray(hal_uq_setup["element_indices"])
    g_tf, _ = tower.gamma_from_features(feats, eidx)
    scorer = CandidateGamma(tower, feats, eidx)
    np.testing.assert_allclose(scorer.gamma(np.arange(len(feats))), g_tf, rtol=1e-9)
    np.testing.assert_allclose(scorer.gamma_all(), g_tf, rtol=1e-9)
    _, _, assign_tf = tower(feats, eidx, verbose=False)
    np.testing.assert_array_equal(scorer.cluster, np.asarray(assign_tf))


def test_whitened_norm_is_sigma(hal_uq_setup, tower):
    feats = np.asarray(hal_uq_setup["features"], dtype=np.float64)
    eidx = np.asarray(hal_uq_setup["element_indices"])
    _, sigma_tf = tower.gamma_from_features(feats, eidx)
    z = whiten_features(feats.copy(), eidx, tower)
    np.testing.assert_allclose(np.sqrt((z * z).sum(1) + 1e-8), sigma_tf, rtol=1e-9)


def _frame(setup, start, n=12):
    return (np.asarray(setup["features"], dtype=np.float64)[start:start + n],
            np.asarray(setup["element_indices"])[start:start + n])


@pytest.mark.parametrize("target", [1.25, 1e-6])  # reached at some alpha / never reached
def test_alpha_search_update_equals_trial_then_update(hal_uq_setup, target):
    ref = GMMUQModel(hal_uq_setup["artifact_path"], param_dtype=tf.float64)
    new = GMMUQModel(hal_uq_setup["artifact_path"], param_dtype=tf.float64)
    for start in (0, 12, 24):  # three collections in a row
        x, el = _frame(hal_uq_setup, start)
        trial = np.asarray(ref.trial_gamma(x, el, ALPHAS, target=target, mode="additive"))
        ok = np.nonzero(trial < target)[0]
        alpha_ref = ALPHAS[ok[0]] if len(ok) else ALPHAS[-1]
        ref.update_one(x, el, alpha=alpha_ref, mode="additive")

        alpha, reached = new.alpha_search_update(x, el, ALPHAS, target)
        assert alpha == alpha_ref
        assert reached == bool(len(ok))
    for e in ref._artifacts:
        for key in (uqc.INV_COV, uqc.SCATTER, uqc.COUNTS, uqc.EFFECTIVE_COUNT):
            np.testing.assert_array_equal(new._artifacts[e][key], ref._artifacts[e][key])
    np.testing.assert_array_equal(new.inv_covs.numpy(), ref.inv_covs.numpy())


def test_alpha_search_update_leaves_model_unchanged_on_error(hal_uq_setup):
    gmm = GMMUQModel(hal_uq_setup["artifact_path"], param_dtype=tf.float64)
    gmm.interp_thresholds = None
    before = copy.deepcopy(gmm._artifacts)
    x, el = _frame(hal_uq_setup, 0)
    with pytest.raises(ValueError, match="thresholds"):
        gmm.alpha_search_update(x, el, ALPHAS, 1.25)
    for e, a in before.items():
        np.testing.assert_array_equal(gmm._artifacts[e][uqc.INV_COV], a[uqc.INV_COV])


def test_reinvert_all_limited_to_given_elements(tmp_path):
    """HAL and calm-greedy re-invert only the elements they evaluate: the others keep
    their stored inverse."""
    from tests.test_uq_select_candidates import _write_artifact

    path = _write_artifact(tmp_path / "a.npz", store_fp32=True)  # elements 0 and 1
    gmm = GMMUQModel(path)
    stored = {e: np.array(a[uqc.INV_COV]) for e, a in gmm._artifacts.items()}
    gmm.reinvert_all([0])
    assert not np.array_equal(gmm._artifacts[0][uqc.INV_COV], stored[0])  # float64 now
    np.testing.assert_array_equal(gmm._artifacts[1][uqc.INV_COV], stored[1])
    full = GMMUQModel(path)
    full.reinvert_all()
    np.testing.assert_array_equal(full._artifacts[0][uqc.INV_COV], gmm._artifacts[0][uqc.INV_COV])


def test_selection_tower_reinverts_only_the_pool_elements(tmp_path):
    from tensorpotential.uq.cli.strategies import selection_tower
    from tests.test_uq_select_candidates import _write_artifact

    path = _write_artifact(tmp_path / "a.npz", store_fp32=True)
    stored = GMMUQModel(path)._artifacts[1][uqc.INV_COV]
    tower = selection_tower(path, elements=[0])
    np.testing.assert_array_equal(tower._artifacts[1][uqc.INV_COV], stored)
