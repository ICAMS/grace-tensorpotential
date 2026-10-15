import copy

import numpy as np

from tensorpotential.uq import constants as uqc
from tensorpotential.uq.gmmuq import GMMUQModel

ALPHAS = [0.05, 0.4, 1.0]


def _frame(setup, n=12):
    return (
        np.asarray(setup["features"])[:n],
        np.asarray(setup["element_indices"])[:n],
    )


def test_trial_gamma_is_read_only(hal_uq_setup):
    feats, eidx = _frame(hal_uq_setup)
    gmm = GMMUQModel(hal_uq_setup["artifact_path"])
    before = copy.deepcopy(gmm._artifacts)
    gamma_before, _ = gmm.gamma_from_features(feats, eidx)
    gmm.trial_gamma(feats, eidx, ALPHAS)
    for e, a in before.items():
        np.testing.assert_array_equal(a[uqc.INV_COV], gmm._artifacts[e][uqc.INV_COV])
        np.testing.assert_array_equal(a[uqc.SCATTER], gmm._artifacts[e][uqc.SCATTER])
    gamma_after, _ = gmm.gamma_from_features(feats, eidx)
    np.testing.assert_array_equal(gamma_before, gamma_after)


def test_trial_gamma_target_early_exit(hal_uq_setup):
    feats, eidx = _frame(hal_uq_setup)
    gmm = GMMUQModel(hal_uq_setup["artifact_path"])
    alphas = [0.05, 0.1, 0.2, 0.4, 0.7, 1.0]
    full = gmm.trial_gamma(feats, eidx, alphas)
    assert np.all(np.isfinite(full))
    # a target just above the 3rd value: the early exit happens at that alpha at the latest
    target = float(full[2]) + 1e-9 * abs(float(full[2])) + 1e-12
    first = int(np.nonzero(full < target)[0][0])
    assert first <= 2
    cut = gmm.trial_gamma(feats, eidx, alphas, target=target)
    np.testing.assert_array_equal(cut[: first + 1], full[: first + 1])
    assert np.all(np.isnan(cut[first + 1 :]))
    # same alpha selected as from the full curve
    assert np.nonzero(cut < target)[0][0] == np.nonzero(full < target)[0][0]
    # target never reached: full curve, no NaN
    np.testing.assert_array_equal(
        gmm.trial_gamma(feats, eidx, alphas, target=-1.0), full
    )
    # target=None unchanged
    np.testing.assert_array_equal(gmm.trial_gamma(feats, eidx, alphas, target=None), full)
