"""Additive live-tower update: C_new = C_old + alpha * C_batch never forgets the training set.

With fixed centroids and thresholds and C_live >= C_train in the PSD order, every atom's
gamma under the live tower is at most its pristine gamma.
"""

import copy

import numpy as np
import pytest

from tensorpotential.uq import constants as uqc
from tensorpotential.uq.artifact_builder import GMMUQArtifactBuilder
from tensorpotential.uq.gmmuq import GMMUQModel

REG = 1e-6


def _synthetic(K, D=6, n=300, seed=0):
    """One-element artifacts with K clusters built from random training features."""
    rng = np.random.default_rng(seed)
    centroids = rng.normal(scale=3.0, size=(K, D))
    lab = rng.integers(0, K, n)
    train = centroids[lab] + rng.normal(size=(n, D)) @ rng.normal(size=(D, D)) * 0.5
    k_near = np.argmin(((train[:, None] - centroids[None]) ** 2).sum(-1), axis=1)
    scatter = np.zeros((K, D, D))
    counts = np.zeros(K, dtype=np.int64)
    for k in range(K):
        d = train[k_near == k] - centroids[k]
        scatter[k] = d.T @ d
        counts[k] = len(d)
    a = {uqc.CENTROIDS: centroids, uqc.SCATTER: scatter, uqc.COUNTS: counts,
         uqc.EFFECTIVE_COUNT: counts.astype(float), uqc.INV_COV: np.zeros((K, D, D))}
    for k in range(K):
        GMMUQArtifactBuilder.reinvert_cluster(a, k, REG * np.eye(D))
    return {0: a}, train, rng


def _sigma(arts, x):
    a = arts[0]
    k = np.argmin(((x[:, None] - a[uqc.CENTROIDS][None]) ** 2).sum(-1), axis=1)
    d = x - a[uqc.CENTROIDS][k]
    return np.sqrt(np.einsum("nd,nde,ne->n", d, a[uqc.INV_COV][k], d))


def _walk(arts, train, rng, mode, n_updates=8, inflate=True):
    """Several live updates with narrow, shifted frames; optional bounded inflation."""
    probes = [train]
    D = train.shape[1]
    for i in range(n_updates):
        base = train[rng.integers(0, len(train), 5)]
        frame = base + rng.normal(scale=0.05, size=base.shape) + rng.normal(scale=2.0, size=D)
        probes.append(frame)
        GMMUQArtifactBuilder.update(arts, frame, np.zeros(len(frame), np.int32), REG,
                                    alpha=float(rng.choice([0.05, 0.2, 0.7, 1.0])), mode=mode)
        if inflate and i % 2:
            a = arts[0]
            k_touched = np.unique(np.argmin(((frame[:, None] - a[uqc.CENTROIDS][None]) ** 2).sum(-1), 1))
            for k in k_touched:  # same operation as GMMUQModel.inflate: SCATTER *= factor > 1
                a[uqc.SCATTER][k] *= float(rng.uniform(1.0, 4.0))
                GMMUQArtifactBuilder.reinvert_cluster(a, k, REG * np.eye(D))
    probes.append(rng.normal(scale=4.0, size=(500, D)))
    return np.concatenate(probes)


def test_additive_live_sigma_never_exceeds_pristine():
    for K in (1, 2):
        for seed in (0, 1, 2):
            pristine, train, rng = _synthetic(K, seed=seed)
            live = copy.deepcopy(pristine)
            probes = _walk(live, train, rng, "additive")
            s_live, s_pri = _sigma(live, probes), _sigma(pristine, probes)
            assert np.all(s_live <= s_pri * (1 + 1e-12) + 1e-12), (K, seed, float(np.max(s_live - s_pri)))
    # control: the property is not trivial, the convex blend does exceed pristine
    pristine, train, rng = _synthetic(2, seed=0)
    live = copy.deepcopy(pristine)
    probes = _walk(live, train, rng, "convex", inflate=False)
    assert np.max(_sigma(live, probes) - _sigma(pristine, probes)) > 0.1


def test_additive_scatter_bookkeeping():
    arts, train, _ = _synthetic(1, seed=3)
    a = arts[0]
    cov_old = a[uqc.SCATTER][0] / a[uqc.EFFECTIVE_COUNT][0]
    frame = train[:7] + 1.0
    d = frame - a[uqc.CENTROIDS][0]
    cov_batch = d.T @ d / len(frame)
    w_old, n_old = a[uqc.EFFECTIVE_COUNT][0], int(a[uqc.COUNTS][0])
    GMMUQArtifactBuilder.update(arts, frame, np.zeros(7, np.int32), REG, alpha=0.3, mode="additive")
    np.testing.assert_allclose(a[uqc.SCATTER][0] / a[uqc.EFFECTIVE_COUNT][0],
                               cov_old + 0.3 * cov_batch, rtol=1e-12)
    assert a[uqc.EFFECTIVE_COUNT][0] == w_old + 7 and a[uqc.COUNTS][0] == n_old + 7


def test_update_mode_validation():
    arts, train, _ = _synthetic(1)
    for kw, match in [({"alpha": None, "mode": "additive"}, "alpha"),
                      ({"alpha": 0.3, "mode": "sum"}, "mode")]:
        with pytest.raises(ValueError, match=match):
            GMMUQArtifactBuilder.update(arts, train[:3], np.zeros(3, np.int32), REG, **kw)


# -- GMMUQModel (real artifact with thresholds) ------------------------------------

def _frame(setup, n=12):
    return np.asarray(setup["features"])[:n], np.asarray(setup["element_indices"])[:n]


@pytest.mark.parametrize("mode", ["convex", "additive"])
def test_trial_gamma_matches_committed_update(hal_uq_setup, mode):
    feats, eidx = _frame(hal_uq_setup)
    gmm = GMMUQModel(hal_uq_setup["artifact_path"])
    alphas = [0.05, 0.4, 1.0]
    trial = gmm.trial_gamma(feats, eidx, alphas, mode=mode)
    for i, alpha in enumerate(alphas):
        real = GMMUQModel(hal_uq_setup["artifact_path"])
        real.update_one(feats, eidx, alpha=alpha, mode=mode)
        gamma, _ = real.gamma_from_features(feats, eidx)
        np.testing.assert_allclose(trial[i], gamma.max(), rtol=1e-6)


def test_model_additive_with_inflation_never_exceeds_pristine(hal_uq_setup):
    feats_all = np.asarray(hal_uq_setup["features"], dtype=np.float64)
    eidx_all = np.asarray(hal_uq_setup["element_indices"])
    pristine = GMMUQModel(hal_uq_setup["artifact_path"])
    live = GMMUQModel(hal_uq_setup["artifact_path"])
    rng = np.random.default_rng(0)
    probes, probe_e = [feats_all], [eidx_all]
    for i in range(6):
        sel = rng.integers(0, len(feats_all), 10)
        frame = feats_all[sel] + rng.normal(scale=0.3 * feats_all.std(), size=(10, feats_all.shape[1]))
        live.update_one(frame, eidx_all[sel], alpha=[0.05, 0.4, 1.0][i % 3], mode="additive")
        if i % 2:
            live.inflate(frame, eidx_all[sel], target=1e-3, max_rounds=1, max_factor=4.0)
        probes.append(frame)
        probe_e.append(eidx_all[sel])
    x, e = np.concatenate(probes), np.concatenate(probe_e)
    g_live, _ = live.gamma_from_features(x, e)
    g_pri, _ = pristine.gamma_from_features(x, e)
    g_live, g_pri = np.asarray(g_live), np.asarray(g_pri)
    assert np.all(g_live <= g_pri * (1 + 1e-9) + 1e-12), float(np.max(g_live - g_pri))
    assert np.mean(g_live < g_pri * (1 - 1e-6)) > 0.05  # the updates did something



@pytest.mark.parametrize("reinvert", [True, False])
def test_fp32_artifact_live_never_exceeds_pristine_after_setup(tmp_path, reinvert):
    """run_hal set-up: both towers reinvert_all() from a float32-stored artifact; then
    gamma_live <= gamma_pristine to 1e-12 relative after additive updates and inflation,
    for random features. Control (no reinversion): a cluster's first update replaces
    the stored float32 inverse by a re-inverted one and gamma_live exceeds pristine.

    The artifact is full rank (condition number 1e5). A float32 scatter of a
    rank-deficient cluster (fewer atoms than D) can make cov + eps*I indefinite; its
    pseudo-inverse then drops those directions and no ordering holds (the
    hal_uq_setup fixture has such clusters, so it is not used here)."""
    import tensorflow as tf

    from tests.test_uq_select_candidates import D as D_SYN
    from tests.test_uq_select_candidates import _write_artifact

    art = _write_artifact(tmp_path / "fp32.npz", seed=2, store_fp32=True, cond=1e7)
    pristine = GMMUQModel(art, param_dtype=tf.float64)
    live = GMMUQModel(art, param_dtype=tf.float64)
    for tower in (live, pristine) if reinvert else ():  # as run_hal does at set-up
        tower.reinvert_all()
    rng = np.random.default_rng(0)
    centroids = np.stack([a[uqc.CENTROIDS] for a in pristine._artifacts.values()])  # [E, K, D]
    eidx = rng.integers(0, 2, 400)
    probes = centroids[eidx, rng.integers(0, 2, 400)] + rng.normal(scale=2.0, size=(400, D_SYN))
    g_pri = np.asarray(pristine.gamma_from_features(probes, eidx)[0])
    worst = 0.0
    for i in range(6):
        sel = rng.integers(0, 400, 3)
        frame = probes[sel] + rng.normal(scale=0.5, size=(3, D_SYN))
        live.update_one(frame, eidx[sel], alpha=[0.0005, 0.01, 1.0][i % 3], mode="additive")
        if i % 2:
            live.inflate(frame, eidx[sel], target=1e-3, max_rounds=1, max_factor=4.0)
        g_live = np.asarray(live.gamma_from_features(probes, eidx)[0])
        worst = max(worst, float(np.max(g_live / g_pri)))
    if not reinvert:
        assert worst > 1 + 1e-4, worst
        return
    assert worst <= 1 + 1e-12, worst
    assert np.mean(g_live < g_pri * (1 - 1e-6)) > 0.05  # the updates did something
