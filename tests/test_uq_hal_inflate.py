"""GMMUQModel.inflate: bounded covariance inflation of the clusters a frame occupies."""

import copy

import numpy as np
import pytest

from tensorpotential.uq import constants as uqc
from tensorpotential.uq.gmmuq import GMMUQModel

DENSE = ("inv_covs", "centroids", "n_clusters_per_elem", "interp_thresholds")


def _frame(setup, n=40):
    return (
        np.asarray(setup["features"])[:n],
        np.asarray(setup["element_indices"])[:n],
    )


def _gmax(gmm, feats, eidx):
    return float(np.max(gmm.gamma_from_features(feats, eidx)[0]))


def _touched(gmm, feats, eidx):
    out = {}
    for e in np.unique(eidx):
        c = gmm._artifacts[int(e)][uqc.CENTROIDS]
        f = feats[eidx == e]
        k = np.argmin(((f[:, None, :] - c[None]) ** 2).sum(-1), axis=1)
        out[int(e)] = set(np.unique(k).tolist())
    return out


def test_inflate_reaches_target_within_bounds(hal_uq_setup):
    feats, eidx = _frame(hal_uq_setup)
    gmm = GMMUQModel(hal_uq_setup["artifact_path"])
    g0 = _gmax(gmm, feats, eidx)
    target = g0 / 3.0
    res = gmm.inflate(feats, eidx, target, max_rounds=3, max_factor=4.0)
    assert 1 <= res.n_rounds <= 3 and len(res.factors) == res.n_rounds
    assert all(1.0 < f <= 4.0 for f in res.factors)
    assert res.gamma_max_after < target
    np.testing.assert_allclose(res.gamma_max_after, _gmax(gmm, feats, eidx), rtol=1e-12)


def test_inflate_factor_formula_and_cap(hal_uq_setup):
    feats, eidx = _frame(hal_uq_setup)
    gmm = GMMUQModel(hal_uq_setup["artifact_path"])
    g0 = _gmax(gmm, feats, eidx)
    # modest overshoot: factor = (g/(0.95 target))^2 < max_factor
    target = g0 / 1.2
    res = gmm.inflate(feats, eidx, target, max_rounds=1, max_factor=4.0)
    np.testing.assert_allclose(res.factors[0], (g0 / (0.95 * target)) ** 2)
    # huge overshoot: capped
    gmm2 = GMMUQModel(hal_uq_setup["artifact_path"])
    res2 = gmm2.inflate(feats, eidx, g0 / 50.0, max_rounds=2, max_factor=2.5)
    assert res2.factors == [2.5, 2.5]


def test_inflate_touches_only_occupied_clusters(hal_uq_setup):
    feats, eidx = _frame(hal_uq_setup, n=6)
    gmm = GMMUQModel(hal_uq_setup["artifact_path"])
    touched = _touched(gmm, feats, eidx)
    before = copy.deepcopy(gmm._artifacts)
    dense_before = gmm.inv_covs.numpy().copy()
    res = gmm.inflate(feats, eidx, _gmax(gmm, feats, eidx) / 3.0)
    assert res.n_rounds >= 1
    total = float(np.prod(res.factors))
    n_untouched = 0
    for e, art in before.items():
        K = art[uqc.CENTROIDS].shape[0]
        for k in range(K):
            now = gmm._artifacts[e]
            if k in touched.get(e, set()):
                np.testing.assert_allclose(
                    now[uqc.SCATTER][k], art[uqc.SCATTER][k] * total, rtol=1e-6
                )
                assert not np.array_equal(now[uqc.INV_COV][k], art[uqc.INV_COV][k])
                assert np.array_equal(now[uqc.COUNTS][k], art[uqc.COUNTS][k])
            else:
                n_untouched += 1
                for key in (uqc.SCATTER, uqc.INV_COV, uqc.COUNTS, uqc.CENTROIDS):
                    assert np.array_equal(now[key][k], art[key][k]), (e, k, key)
                assert np.array_equal(gmm.inv_covs.numpy()[e, k], dense_before[e, k])
    assert n_untouched > 0


def test_inflate_dense_rows_match_full_rebuild(hal_uq_setup, monkeypatch):
    feats, eidx = _frame(hal_uq_setup)
    gmm = GMMUQModel(hal_uq_setup["artifact_path"])
    build = GMMUQModel._build_tensors
    monkeypatch.setattr(
        GMMUQModel, "_build_tensors",
        lambda *a, **k: pytest.fail("inflate must use _refresh_element_rows"),
    )
    gmm.inflate(feats, eidx, _gmax(gmm, feats, eidx) / 3.0)
    monkeypatch.setattr(GMMUQModel, "_build_tensors", build)
    ref = GMMUQModel.from_artifacts(
        copy.deepcopy(gmm._artifacts), extra_data=gmm.extra_data,
        param_dtype=gmm.param_dtype, regularization=gmm.regularization,
    )
    for name in DENSE:
        assert np.array_equal(getattr(gmm, name).numpy(), getattr(ref, name).numpy()), name


def test_inflate_noop_cases(hal_uq_setup):
    feats, eidx = _frame(hal_uq_setup)
    gmm = GMMUQModel(hal_uq_setup["artifact_path"])
    g0 = _gmax(gmm, feats, eidx)
    before = copy.deepcopy(gmm._artifacts)
    dense = gmm.inv_covs.numpy().copy()
    res = gmm.inflate(feats, eidx, g0 / 3.0, max_rounds=0)
    assert res.n_rounds == 0 and res.factors == []
    np.testing.assert_allclose(res.gamma_max_after, g0, rtol=1e-12)
    assert np.array_equal(gmm.inv_covs.numpy(), dense)
    for e, art in before.items():
        for key, val in art.items():
            assert np.array_equal(gmm._artifacts[e][key], val), (e, key)
    res = gmm.inflate(feats, eidx, g0 * 1.5)  # already below the target
    assert res.n_rounds == 0 and res.factors == []
    assert np.array_equal(gmm.inv_covs.numpy(), dense)


def test_trial_gamma_and_inflate_require_thresholds(uq_setup):
    gmm = GMMUQModel(uq_setup["artifact_path"])
    if gmm.interp_thresholds is not None:
        pytest.skip("base uq_setup artifact already has thresholds")
    feats, eidx = _frame(uq_setup)
    with pytest.raises(ValueError, match="thresholds"):
        gmm.inflate(feats, eidx, 1.25)
    with pytest.raises(ValueError, match="thresholds"):
        gmm.trial_gamma(feats, eidx, [0.1])
