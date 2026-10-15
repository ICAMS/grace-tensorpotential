"""update_one / update_many rewrite only the touched rows of ``inv_covs``.

Reference ("old") path: apply the same ``GMMUQArtifactBuilder.update`` to a second
model's artifacts and then rebuild every dense tensor with ``_build_tensors``.
The fast path must give bit-identical variables.
"""

import copy
import warnings

import numpy as np
import pytest
import tensorflow as tf

from tensorpotential.uq import constants as uqc
from tensorpotential.uq.artifact_builder import GMMUQArtifactBuilder
from tensorpotential.uq.factories import get_gmm_uq_calculator
from tensorpotential.uq.gmmuq import GMMUQModel, _MISSING_ELEM_INV_COV_DIAG

DENSE = ("inv_covs", "centroids", "n_clusters_per_elem", "interp_thresholds")


def _frame(setup, n=12):
    return (
        np.asarray(setup["features"])[:n],
        np.asarray(setup["element_indices"])[:n],
    )


def _reference(path, feats, eidx, alpha, mode="convex", **kw):
    """Old path: builder update, then full dense rebuild."""
    ref = GMMUQModel(path, **kw)
    GMMUQArtifactBuilder.update(
        ref._artifacts, feats, eidx, ref.regularization, alpha=alpha, mode=mode
    )
    ref._build_tensors(ref._artifacts)
    return ref


def _assert_dense_identical(a, b):
    for name in DENSE:
        va, vb = getattr(a, name), getattr(b, name)
        assert isinstance(va, tf.Variable), name
        assert np.array_equal(va.numpy(), vb.numpy()), name


@pytest.mark.parametrize("mode", ["convex", "additive"])  # additive: the HAL path
def test_update_one_matches_full_rebuild(hal_uq_setup, monkeypatch, mode):
    feats, eidx = _frame(hal_uq_setup, n=40)
    fast = GMMUQModel(hal_uq_setup["artifact_path"])
    before = {n: getattr(fast, n) for n in DENSE}

    def boom(*a, **k):
        raise AssertionError("_build_tensors must not be called by update")

    monkeypatch.setattr(fast, "_build_tensors", boom)
    fast.update_one(feats, eidx, alpha=0.3, mode=mode)
    for n, v in before.items():  # variables assigned in place
        assert getattr(fast, n) is v
    ref = _reference(hal_uq_setup["artifact_path"], feats, eidx, 0.3, mode=mode)
    _assert_dense_identical(fast, ref)
    g_fast, _ = fast.gamma_from_features(feats, eidx)
    g_ref, _ = ref.gamma_from_features(feats, eidx)
    assert np.array_equal(g_fast, g_ref)


def test_update_one_matches_full_rebuild_float32(hal_uq_setup):
    feats, eidx = _frame(hal_uq_setup, n=40)
    fast = GMMUQModel(hal_uq_setup["artifact_path"], param_dtype=tf.float32)
    fast.update_one(feats, eidx, alpha=0.3)
    ref = _reference(
        hal_uq_setup["artifact_path"], feats, eidx, 0.3, param_dtype=tf.float32
    )
    _assert_dense_identical(fast, ref)
    assert fast.inv_covs.dtype == tf.float32


def test_update_many_matches_full_rebuild(hal_uq_setup):
    alpha = None  # accumulate: the grace_uq update path
    feats = np.asarray(hal_uq_setup["features"])
    eidx = np.asarray(hal_uq_setup["element_indices"])
    half = len(feats) // 2
    fast = GMMUQModel(hal_uq_setup["artifact_path"])
    fast.update_many([feats[:half], feats[half:]], [eidx[:half], eidx[half:]],
                     chunk_size=64, alpha=alpha)
    ref = GMMUQModel(hal_uq_setup["artifact_path"])
    for s in range(0, len(feats), 64):
        GMMUQArtifactBuilder.update(
            ref._artifacts, feats[s : s + 64], eidx[s : s + 64],
            ref.regularization, alpha=alpha,
        )
    ref._build_tensors(ref._artifacts)
    _assert_dense_identical(fast, ref)


def test_untouched_clusters_keep_inv_cov(hal_uq_setup):
    """Features of one element, few atoms: other elements and unhit clusters stay."""
    eidx_all = np.asarray(hal_uq_setup["element_indices"])
    feats_all = np.asarray(hal_uq_setup["features"])
    elem = 1
    sel = np.where(eidx_all == elem)[0][:3]
    feats, eidx = feats_all[sel], eidx_all[sel]

    gmm = GMMUQModel(hal_uq_setup["artifact_path"])
    art = gmm._artifacts[elem]
    d = ((feats[:, None, :] - art[uqc.CENTROIDS][None]) ** 2).sum(-1)
    hit = set(np.argmin(d, axis=1).tolist())
    K = art[uqc.CENTROIDS].shape[0]
    untouched = [k for k in range(K) if k not in hit]
    assert untouched, "need at least one cluster that receives no atoms"

    # Make the test discriminating: re-inverting an untouched cluster would
    # reproduce its stored inv_cov bit for bit, so perturb the stored inverse of
    # every untouched cluster first.  Only a re-inversion could undo the perturbation.
    for k in untouched:
        art[uqc.INV_COV][k] = 2.0 * art[uqc.INV_COV][k]
    gmm._refresh_element_rows([elem])

    before_dense = gmm.inv_covs.numpy().copy()
    before_art = copy.deepcopy(gmm._artifacts)
    gmm.update_one(feats, eidx, alpha=0.5)
    after_dense = gmm.inv_covs.numpy()

    for e in range(gmm.n_elements):
        if e != elem:
            assert np.array_equal(after_dense[e], before_dense[e])
    for k in untouched:
        assert np.array_equal(after_dense[elem, k], before_dense[elem, k])
        for key in (uqc.INV_COV, uqc.SCATTER, uqc.COUNTS):
            assert np.array_equal(gmm._artifacts[elem][key][k], before_art[elem][key][k])
    for k in hit:
        assert not np.array_equal(after_dense[elem, k], before_dense[elem, k])
        assert gmm._artifacts[elem][uqc.COUNTS][k] > before_art[elem][uqc.COUNTS][k]


def test_absent_elements_keep_fictitious_rows(hal_uq_setup):
    """Elements missing from the artifacts are skipped, rows stay isotropic."""
    base = GMMUQModel(hal_uq_setup["artifact_path"])
    arts = copy.deepcopy(base._artifacts)
    missing = 3
    del arts[missing]
    extra = {
        "element_map": np.array(["Mo", "Nb", "Ta", "W"]),
        "interp_thresholds": hal_uq_setup["thresholds"],
    }

    def make():
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            return GMMUQModel.from_artifacts(copy.deepcopy(arts), extra_data=dict(extra))

    feats, eidx = _frame(hal_uq_setup, n=len(hal_uq_setup["features"]))
    assert (eidx == missing).any() and (eidx != missing).any()

    fast = make()
    iso = fast.inv_covs.numpy()[missing].copy()
    assert np.all(iso[:, np.arange(fast.D), np.arange(fast.D)] == _MISSING_ELEM_INV_COV_DIAG)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        fast.update_one(feats, eidx, alpha=0.3)
    assert np.array_equal(fast.inv_covs.numpy()[missing], iso)

    ref = make()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        GMMUQArtifactBuilder.update(ref._artifacts, feats, eidx, ref.regularization, alpha=0.3)
        ref._build_tensors(ref._artifacts)
    _assert_dense_identical(fast, ref)

    # an update made only of absent elements is a no-op
    only_missing = eidx == missing
    before = fast.inv_covs.numpy().copy()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        fast.update_one(feats[only_missing], eidx[only_missing], alpha=0.3)
    assert np.array_equal(fast.inv_covs.numpy(), before)


def test_compiled_graph_sees_update(hal_uq_setup):
    """The traced calculator graph reads the updated variable (no stale constant)."""
    calc = get_gmm_uq_calculator(
        model_yaml=hal_uq_setup["model_yaml"],
        checkpoint=hal_uq_setup["checkpoint"],
        gmm_artifact_path=hal_uq_setup["artifact_path"],
        param_dtype="float64",
    )
    atoms = max(hal_uq_setup["atoms"], key=len).copy()

    calc.calculate(atoms, ["energy", "forces", "stress"], ["positions"])
    gamma0 = np.asarray(calc.results[uqc.ATOMIC_GAMMA])[: len(atoms)].copy()
    feats = np.asarray(calc.results[uqc.FEATURES])[: len(atoms)].copy()
    eidx = np.array([calc.element_map[s] for s in atoms.get_chemical_symbols()])

    calc.gmm_uq_model.update_one(feats, eidx, alpha=0.5)

    calc.calculate(atoms, ["energy", "forces", "stress"], ["positions"])
    gamma1 = np.asarray(calc.results[uqc.ATOMIC_GAMMA])[: len(atoms)]
    assert not np.allclose(gamma1, gamma0, rtol=1e-6, atol=0.0)

    ref = _reference(hal_uq_setup["artifact_path"], feats, eidx, 0.5)
    g_ref, _ = ref.gamma_from_features(feats, eidx)
    np.testing.assert_allclose(gamma1, g_ref, rtol=1e-6)
