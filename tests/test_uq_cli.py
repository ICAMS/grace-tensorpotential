"""Unit tests for the grace_uq CLI: dispatcher, info, build (K-selection /
merging / compaction), select (greedy + stratification repair), and the
strategy registry.

This file consolidates what previously lived in:
  - test_uq_cli_dispatch.py
  - test_uq_cli_info.py
  - test_uq_cli_build_pick_k.py
  - test_uq_cli_select.py
  - test_uq_cli_strategies.py
"""

import inspect
import os
import subprocess
import sys

os.environ.setdefault("TF_USE_LEGACY_KERAS", "1")

import numpy as np
import pandas as pd
import pytest
from ase import Atoms
from sklearn.cluster import KMeans

from tensorpotential.scripts import grace_uq as dispatcher
from tensorpotential.uq import constants as uq_constants
from tensorpotential.uq.artifact_builder import GMMUQArtifactBuilder
from tensorpotential.uq.cli import select as select_mod
from tensorpotential.uq.cli._common import read_artifact_metadata
from tensorpotential.uq.cli.build import (
    _compact_step2_artifacts,
    _load_and_subsample,
    _load_filter_fn,
    _merge_small_clusters,
    _merge_step2_clusters,
    _pick_centroids_for_element,
    _resolve_train_data_from_input_yaml,
    _resolve_weighted_train_data,
    _WeightedTrainDataAction,
    _worker_output_paths,
    build_main,
    select_optimal_clusters,
)
from tensorpotential.uq.cli.info import info_main
from tensorpotential.uq.cli.select import (
    _build_atom_pool,
    _collect_top_n,
    _repair_stratification,
    _structure_element_counts,
)
from tensorpotential.uq.cli.strategies import (
    _chunked_min_dist,
    _fps_iter,
    get_strategy,
    list_strategies,
    register_strategy,
)


# ===========================================================================
# Shared fixtures and helpers
# ===========================================================================


@pytest.fixture
def picker():
    return _pick_centroids_for_element


@pytest.fixture
def merger():
    return _merge_small_clusters


@pytest.fixture
def step2_merger():
    return _merge_step2_clusters


def _fit_global_km(c_all, w_all, k, seed=0):
    """Fit a sklearn KMeans on weighted meta-centroids."""
    return KMeans(n_clusters=k, n_init=10, random_state=seed).fit(
        c_all, sample_weight=w_all
    )


def _make_atoms(symbols, n=None):
    """ASE Atoms with `n` atoms (optionally repeating `symbols`)."""
    if n is not None:
        symbols = (symbols * ((n + len(symbols) - 1) // len(symbols)))[:n]
    return Atoms(symbols=symbols, positions=np.zeros((len(symbols), 3)))


@pytest.fixture
def synthetic_df():
    """5 structures × 4 atoms each. Mix of Mo/Nb/Ta/W with known per-atom
    gamma/features."""
    rng = np.random.default_rng(0)
    rows = []
    elements = ["Mo", "Nb", "Ta", "W"]
    for sid in range(5):
        atoms = _make_atoms(elements, n=4)
        gamma = rng.uniform(0.5, 5.0, size=4).astype(np.float32)
        feats = rng.standard_normal((4, 8)).astype(np.float32)
        rows.append({"ase_atoms": atoms, "gamma": gamma, "features": feats})
    return pd.DataFrame(rows)


def _make_step2_artifact(D, atoms_per_cluster, centroid_offsets, K_max):
    """Build a synthetic step-2 (centroids, scatter, counts) tensor with
    sentinel padding up to K_max."""
    SENTINEL = 1e10
    K_real = len(atoms_per_cluster)
    centroids = np.full((K_max, D), SENTINEL, dtype=np.float64)
    scatter = np.zeros((K_max, D, D), dtype=np.float64)
    counts = np.zeros(K_max, dtype=np.float64)

    rng = np.random.default_rng(0)
    for k in range(K_real):
        n = atoms_per_cluster[k]
        offset = np.asarray(centroid_offsets[k], dtype=np.float64)
        if n > 0:
            atoms = rng.standard_normal((n, D)) * 0.1 + offset
            c = atoms.mean(axis=0)
            diffs = atoms - c
            S = diffs.T @ diffs
        else:
            c = offset
            S = np.zeros((D, D))
        centroids[k] = c
        scatter[k] = S
        counts[k] = n
    return centroids, scatter, counts


def _imbalanced_features(D=10, n_big=200, n_small=5, seed=0):
    """A pool that KMeans-K=2 will cleanly split into big + small cluster."""
    rng = np.random.default_rng(seed)
    big = rng.standard_normal((n_big, D)) + 0.0
    small = rng.standard_normal((n_small, D)) * 0.1 + 100.0
    return np.vstack([big, small])


def _make_pool(n_atoms=200, n_struct=20, D=4, seed=0):
    rng = np.random.default_rng(seed)
    return {
        "struct_id": rng.integers(0, n_struct, size=n_atoms),
        "element": rng.integers(0, 2, size=n_atoms),
        "gamma": rng.uniform(0.0, 5.0, size=n_atoms).astype(np.float32),
        "features": rng.standard_normal((n_atoms, D)).astype(np.float32),
    }


def _write_synthetic_info_artifact(path, n_elements=2, K=3, D=4):
    """Minimal valid UQ npz with finalized centroids/inv_cov + thresholds."""
    rng = np.random.default_rng(0)
    artifacts = {}
    for e in range(n_elements):
        centroids = rng.standard_normal((K, D))
        inv_cov = np.tile(np.eye(D), (K, 1, 1))
        counts = np.full(K, 100, dtype=np.int64)
        artifacts[e] = {
            uq_constants.CENTROIDS: centroids,
            uq_constants.INV_COV: inv_cov,
            uq_constants.COUNTS: counts,
            uq_constants.SCATTER: None,
            uq_constants.COND_NUMBER: np.ones(K),
            uq_constants.EFFECTIVE_RANK: np.full(K, D, dtype=np.int64),
            uq_constants.N_TRUNCATED: np.zeros(K, dtype=np.int64),
        }
    interp_thresholds = np.full((n_elements, K), 1.0, dtype=np.float64)
    element_map = np.array(["Mo", "Nb"][:n_elements])
    GMMUQArtifactBuilder.save_artifacts(
        path,
        artifacts,
        interp_thresholds=interp_thresholds,
        element_map=element_map,
    )


# ===========================================================================
# Dispatcher (grace_uq <subcommand>)
# ===========================================================================


def test_dispatch_help_works(capsys):
    rc = dispatcher._dispatch(["--help"])
    out = capsys.readouterr().out
    assert rc == 0
    assert "info" in out and "predict" in out and "select" in out and "build" in out


def test_dispatch_unknown_subcommand(capsys):
    rc = dispatcher._dispatch(["definitely-not-a-cmd"])
    assert rc == 2


def test_dispatch_leading_flag_is_unknown_subcommand(capsys):
    """Without the legacy auto-`build` shim, leading flags are rejected."""
    rc = dispatcher._dispatch(["--model-yaml", "x.yaml"])
    assert rc == 2


def test_dispatch_no_args_prints_help(capsys):
    rc = dispatcher._dispatch([])
    out = capsys.readouterr().out
    assert rc == 0
    assert "Subcommands" in out


def test_input_yaml_fallback_resolves_data_filename(tmp_path, monkeypatch):
    """When training_set.pkl.gz is missing, build should consult ../../input.yaml."""
    parent = tmp_path / "fit"
    seed = parent / "seed" / "1"
    seed.mkdir(parents=True)
    (parent / "input.yaml").write_text("data:\n  filename: train.pkl.gz\n")
    (parent / "train.pkl.gz").write_text("dummy")

    monkeypatch.chdir(seed)
    resolved = _resolve_train_data_from_input_yaml()
    assert resolved is not None
    assert resolved[0].endswith("train.pkl.gz")


def test_input_yaml_fallback_rejects_train_size(tmp_path, monkeypatch):
    """Refuse the fallback when train_size/test_size would make data.filename
    a superset of the actual training set."""
    parent = tmp_path / "fit"
    seed = parent / "seed" / "1"
    seed.mkdir(parents=True)
    (parent / "input.yaml").write_text(
        "data:\n  filename: full.pkl.gz\n  train_size: 0.8\n"
    )
    (parent / "full.pkl.gz").write_text("dummy")

    monkeypatch.chdir(seed)
    with pytest.raises(SystemExit) as exc:
        _resolve_train_data_from_input_yaml()
    assert "train_size" in str(exc.value)


def test_input_yaml_fallback_missing_yaml_returns_none(tmp_path, monkeypatch):
    """No input.yaml, no error — just None so the caller can fall through."""
    seed = tmp_path / "seed" / "1"
    seed.mkdir(parents=True)
    monkeypatch.chdir(seed)
    assert _resolve_train_data_from_input_yaml() is None


# ===========================================================================
# grace_uq info
# ===========================================================================


def test_read_artifact_metadata(tmp_path):
    path = str(tmp_path / "art.npz")
    _write_synthetic_info_artifact(path, n_elements=2, K=3, D=4)
    meta = read_artifact_metadata(path)
    assert meta["D"] == 4
    assert meta["K_max"] == 3
    assert meta["n_elements"] == 2
    assert meta["element_map"] == ["Mo", "Nb"]
    assert meta["has_thresholds"] is True
    assert meta["has_histograms"] is False


def test_info_main_prints_summary(tmp_path, capsys):
    path = str(tmp_path / "art.npz")
    _write_synthetic_info_artifact(path)
    rc = info_main([path])
    assert rc == 0
    out = capsys.readouterr().out
    assert "feature_dim D" in out
    assert "Mo" in out and "Nb" in out
    assert "thr_min" in out


# ===========================================================================
# grace_uq build — _worker_output_paths
# ===========================================================================


def test_worker_output_paths_step1_lists_one_file_per_k():
    paths = _worker_output_paths(
        "/tmp/run", step_idx=1, worker_id=2, n_clusters=[4, 8, 16]
    )
    assert paths == [
        "/tmp/run/.step1_w2_k4.npz",
        "/tmp/run/.step1_w2_k8.npz",
        "/tmp/run/.step1_w2_k16.npz",
    ]


def test_worker_output_paths_step2_returns_single_file_ignoring_n_clusters():
    paths = _worker_output_paths(
        "/tmp/run", step_idx=2, worker_id=0, n_clusters=[4, 8]
    )
    assert paths == ["/tmp/run/.step2_w0.npz"]


def test_worker_output_paths_step3_returns_single_file():
    paths = _worker_output_paths(
        "/tmp/run", step_idx=3, worker_id=5, n_clusters=[]
    )
    assert paths == ["/tmp/run/.step3_w5.npz"]


# ===========================================================================
# grace_uq build — _merge_small_clusters (step-1 merging)
# ===========================================================================


def test_merge_drops_smallest_cluster_into_nearest(merger):
    """Small cluster merges into closest neighbor by Euclidean distance,
    with weighted-average centroid and summed counts."""
    centroids = np.array(
        [
            [0.0, 0.0],
            [1.0, 0.0],   # closest to the small cluster at [0.5, 0]
            [10.0, 0.0],  # far away
        ]
    )
    counts = np.array([3.0, 100.0, 100.0])
    centroids[0] = [0.5, 0.0]

    new_c, new_n, n_merged = merger(centroids, counts, min_atoms=10)
    assert n_merged == 1
    assert new_c.shape == (2, 2)
    assert new_n.shape == (2,)
    cluster_with_103 = np.where(np.isclose(new_n, 103.0))[0]
    assert cluster_with_103.size == 1
    survivor = new_c[cluster_with_103[0]]
    expected_x = (0.5 * 3.0 + 1.0 * 100.0) / 103.0
    assert np.isclose(survivor[0], expected_x)


def test_merge_iterates_until_all_clusters_pass(merger):
    centroids = np.array(
        [
            [0.0, 0.0],
            [1.0, 0.0],
            [2.0, 0.0],
            [10.0, 0.0],
        ]
    )
    counts = np.array([2.0, 5.0, 200.0, 200.0])
    new_c, new_n, n_merged = merger(centroids, counts, min_atoms=10)
    assert n_merged >= 2
    assert (new_n >= 10).all()
    assert int(new_n.sum()) == int(counts.sum())


def test_merge_stops_at_k1(merger):
    """When all clusters are too small, merging continues until K=1."""
    centroids = np.array([[0.0, 0.0], [5.0, 0.0], [10.0, 0.0]])
    counts = np.array([3.0, 4.0, 5.0])
    new_c, new_n, n_merged = merger(centroids, counts, min_atoms=100)
    assert new_c.shape == (1, 2)
    assert int(new_n[0]) == 12
    assert n_merged == 2


def test_merge_noop_when_all_pass(merger):
    centroids = np.array([[0.0, 0.0], [10.0, 0.0]])
    counts = np.array([100.0, 100.0])
    new_c, new_n, n_merged = merger(centroids, counts, min_atoms=10)
    assert n_merged == 0
    assert new_c.shape == (2, 2)


# ===========================================================================
# grace_uq build — _merge_step2_clusters (parallel-axis merge)
# ===========================================================================


def test_step2_merge_demotes_underpopulated_to_sentinel(step2_merger):
    D = 4
    K_max = 8
    centroids, scatter, counts = _make_step2_artifact(
        D=D,
        atoms_per_cluster=[100, 3, 100, 100],
        centroid_offsets=[
            [0.0, 0.0, 0.0, 0.0],
            [0.5, 0.0, 0.0, 0.0],
            [10.0, 0.0, 0.0, 0.0],
            [0.0, 10.0, 0.0, 0.0],
        ],
        K_max=K_max,
    )
    total_atoms_before = int(counts.sum())
    new_c, new_S, new_n, new_eff, n_merged = step2_merger(
        centroids, scatter, counts, min_atoms=D + 1
    )
    assert n_merged == 1
    is_real = ~np.any(np.abs(new_c) > 1e9, axis=1)
    assert int(is_real.sum()) == 3
    assert int(new_n.sum()) == total_atoms_before
    assert np.any(np.abs(new_c[1]) > 1e9)
    assert int(new_n[1]) == 0
    assert np.allclose(new_S[1], 0)


def test_step2_merge_parallel_axis_recovers_full_scatter(step2_merger):
    """Parallel-axis must reconstruct the scatter you'd get computing it
    over the union directly (exact when each cluster's centroid is its
    own sample mean)."""
    D = 3
    rng = np.random.default_rng(42)
    atoms_a = rng.standard_normal((50, D)) * 1.0 + np.array([0.0, 0.0, 0.0])
    atoms_b = rng.standard_normal((5, D)) * 0.2 + np.array([3.0, 0.0, 0.0])

    c_A = atoms_a.mean(axis=0)
    c_B = atoms_b.mean(axis=0)
    S_A = (atoms_a - c_A).T @ (atoms_a - c_A)
    S_B = (atoms_b - c_B).T @ (atoms_b - c_B)

    centroids = np.array([c_A, c_B], dtype=np.float64)
    scatter = np.stack([S_A, S_B], axis=0)
    counts = np.array([50.0, 5.0])

    new_c, new_S, new_n, new_eff, n_merged = step2_merger(
        centroids, scatter, counts, min_atoms=10
    )
    assert n_merged == 1

    all_atoms = np.vstack([atoms_a, atoms_b])
    c_union = all_atoms.mean(axis=0)
    S_union = (all_atoms - c_union).T @ (all_atoms - c_union)

    assert int(new_n[0]) == 55
    assert int(new_n[1]) == 0
    assert np.allclose(new_c[0], c_union, atol=1e-10)
    assert np.allclose(new_S[0], S_union, atol=1e-8)


def test_step2_merge_stops_at_one_real_cluster(step2_merger):
    """Every cluster too small → merge until one real cluster remains."""
    D = 3
    centroids, scatter, counts = _make_step2_artifact(
        D=D,
        atoms_per_cluster=[2, 2, 2],
        centroid_offsets=[[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [2.0, 0.0, 0.0]],
        K_max=4,
    )
    new_c, new_S, new_n, new_eff, n_merged = step2_merger(
        centroids, scatter, counts, min_atoms=D + 1
    )
    assert n_merged == 2
    is_real = ~np.any(np.abs(new_c) > 1e9, axis=1)
    assert int(is_real.sum()) == 1
    assert int(new_n[is_real].sum()) == 6


def test_step2_merge_noop_when_all_pass(step2_merger):
    D = 4
    centroids, scatter, counts = _make_step2_artifact(
        D=D,
        atoms_per_cluster=[100, 100],
        centroid_offsets=[[0.0] * D, [10.0] + [0.0] * (D - 1)],
        K_max=4,
    )
    _, _, new_n, _, n_merged = step2_merger(
        centroids, scatter, counts, min_atoms=D + 1
    )
    assert n_merged == 0
    assert int((new_n > 0).sum()) == 2


# ===========================================================================
# grace_uq build — _compact_step2_artifacts
# ===========================================================================


def test_compact_shrinks_to_max_kEff():
    """Every element with K_eff <= 4 → compaction shrinks K_max from 8 to 4."""
    D = 3
    K_max = 8

    def make_elem(real_counts):
        c, S, n = _make_step2_artifact(
            D=D,
            atoms_per_cluster=real_counts,
            centroid_offsets=[[float(i), 0.0, 0.0] for i in range(len(real_counts))],
            K_max=K_max,
        )
        return {
            uq_constants.CENTROIDS: c,
            uq_constants.SCATTER: S,
            uq_constants.COUNTS: n,
        }

    base = {
        0: make_elem([100, 100, 100, 100]),  # K_eff = 4
        1: make_elem([100, 100]),            # K_eff = 2
    }
    K_new = _compact_step2_artifacts(base)
    assert K_new == 4
    assert base[0][uq_constants.CENTROIDS].shape == (4, D)
    assert base[1][uq_constants.CENTROIDS].shape == (4, D)
    is_real_e1 = ~np.any(np.abs(base[1][uq_constants.CENTROIDS]) > 1e9, axis=1)
    assert is_real_e1.tolist() == [True, True, False, False]


def test_compact_packs_interior_sentinels_to_back():
    """An element with sentinels interleaved (post-merge) must come out with
    real-first ordering and no spurious zero-count holes."""
    D = 3
    K_max = 4
    SENTINEL = 1e10

    centroids = np.array(
        [
            [0.0, 0.0, 0.0],
            [SENTINEL] * D,
            [10.0, 0.0, 0.0],
            [SENTINEL] * D,
        ],
        dtype=np.float64,
    )
    scatter = np.zeros((K_max, D, D))
    counts = np.array([100.0, 0.0, 50.0, 0.0])

    base = {
        0: {
            uq_constants.CENTROIDS: centroids,
            uq_constants.SCATTER: scatter,
            uq_constants.COUNTS: counts,
        }
    }
    K_new = _compact_step2_artifacts(base)
    assert K_new == 2
    new_c = base[0][uq_constants.CENTROIDS]
    new_n = base[0][uq_constants.COUNTS]
    assert new_c.shape == (2, D)
    assert int(new_n[0]) == 100
    assert int(new_n[1]) == 50


# ===========================================================================
# grace_uq build — _pick_centroids_for_element
# ===========================================================================


def test_pick_merges_imbalanced_to_k1(picker):
    """If K=2 fits but yields a cluster smaller than min_atoms, the picker
    must merge — landing on K=1 here because only two clusters exist."""
    D = 10
    feats = _imbalanced_features(D=D, n_big=200, n_small=5, seed=0)
    weights = np.ones(len(feats))

    candidates = [1, 2]
    km1 = _fit_global_km(feats, weights, k=1)
    km2 = _fit_global_km(feats, weights, k=2)
    global_kms_by_k = {1: {0: km1}, 2: {0: km2}}
    all_centroids_by_k = {
        1: {0: [(feats, weights)]},
        2: {0: [(feats, weights)]},
    }

    real_centroids, best_k = picker(
        e=0,
        n_atoms_e=len(feats),
        n_clusters_candidates=candidates,
        elem_optimal_k=2,
        global_kms_by_k=global_kms_by_k,
        all_centroids_by_k=all_centroids_by_k,
        min_atoms=D + 1,
    )
    assert best_k == 1
    assert real_centroids.shape == (1, D)


def test_pick_keeps_k_when_balanced(picker):
    D = 4
    rng = np.random.default_rng(0)
    feats = np.vstack(
        [
            rng.standard_normal((50, D)) + 0.0,
            rng.standard_normal((50, D)) + 10.0,
        ]
    )
    weights = np.ones(len(feats))

    candidates = [1, 2]
    km1 = _fit_global_km(feats, weights, k=1)
    km2 = _fit_global_km(feats, weights, k=2)
    global_kms_by_k = {1: {0: km1}, 2: {0: km2}}
    all_centroids_by_k = {
        1: {0: [(feats, weights)]},
        2: {0: [(feats, weights)]},
    }

    real_centroids, best_k = picker(
        e=0,
        n_atoms_e=100,
        n_clusters_candidates=candidates,
        elem_optimal_k=2,
        global_kms_by_k=global_kms_by_k,
        all_centroids_by_k=all_centroids_by_k,
        min_atoms=D + 1,
    )
    assert best_k == 2
    assert real_centroids.shape == (2, D)


def test_pick_partial_merge_preserves_higher_k(picker):
    """K=4 with 3 healthy clusters + 1 tiny one → pick K=3 (merge tiny only)."""
    D = 4
    rng = np.random.default_rng(0)
    big1 = rng.standard_normal((100, D)) * 0.05 + np.array([0.0, 0.0, 0.0, 0.0])
    big2 = rng.standard_normal((100, D)) * 0.05 + np.array([20.0, 0.0, 0.0, 0.0])
    big3 = rng.standard_normal((100, D)) * 0.05 + np.array([0.0, 20.0, 0.0, 0.0])
    tiny = rng.standard_normal((3, D)) * 0.05 + np.array([0.0, 0.0, 20.0, 0.0])
    feats = np.vstack([big1, big2, big3, tiny])
    weights = np.ones(len(feats))

    km4 = _fit_global_km(feats, weights, k=4)
    candidates = [1, 4]
    global_kms_by_k = {1: {0: _fit_global_km(feats, weights, k=1)}, 4: {0: km4}}
    all_centroids_by_k = {
        1: {0: [(feats, weights)]},
        4: {0: [(feats, weights)]},
    }

    real_centroids, best_k = picker(
        e=0,
        n_atoms_e=len(feats),
        n_clusters_candidates=candidates,
        elem_optimal_k=4,
        global_kms_by_k=global_kms_by_k,
        all_centroids_by_k=all_centroids_by_k,
        min_atoms=D + 1,
    )
    assert best_k == 3, f"expected K=3, got K={best_k}"
    assert real_centroids.shape == (3, D)


def test_pick_falls_back_to_k1_when_kmeans_missing(picker, caplog):
    """No KMeans fit at any candidate K → synthesize K=1 from weighted mean."""
    D = 10
    N = 5
    rng = np.random.default_rng(0)
    feats = rng.standard_normal((N, D))
    weights = np.ones(N)

    global_kms_by_k = {1: {}, 2: {}}
    all_centroids_by_k = {
        1: {0: [(feats, weights)]},
        2: {0: [(feats, weights)]},
    }

    with caplog.at_level("WARNING", logger="tensorpotential.uq.cli.build"):
        real_centroids, best_k = picker(
            e=0,
            n_atoms_e=N,
            n_clusters_candidates=[1, 2],
            elem_optimal_k=2,
            global_kms_by_k=global_kms_by_k,
            all_centroids_by_k=all_centroids_by_k,
            min_atoms=D + 1,
            lbl="Pt(0)",
            regularization=1e-6,
        )
    assert best_k == 1
    assert real_centroids.shape == (1, D)
    msgs = [r.message for r in caplog.records]
    assert any("Pt(0)" in m and "no fitted KMeans" in m for m in msgs)


def test_pick_warns_when_underpopulated_to_k1(picker, caplog):
    """n_atoms < min_atoms → K=1 with Tikhonov-fallback warning."""
    D = 10
    N = 5
    rng = np.random.default_rng(0)
    feats = rng.standard_normal((N, D))
    weights = np.ones(N)

    km1 = _fit_global_km(feats, weights, k=1)
    global_kms_by_k = {1: {0: km1}}
    all_centroids_by_k = {1: {0: [(feats, weights)]}}

    with caplog.at_level("WARNING", logger="tensorpotential.uq.cli.build"):
        real_centroids, best_k = picker(
            e=0,
            n_atoms_e=N,
            n_clusters_candidates=[1],
            elem_optimal_k=1,
            global_kms_by_k=global_kms_by_k,
            all_centroids_by_k=all_centroids_by_k,
            min_atoms=D + 1,
            lbl="Pt(0)",
            regularization=1e-6,
        )
    assert best_k == 1
    assert real_centroids.shape == (1, D)
    msgs = [r.message for r in caplog.records]
    assert any("Pt(0)" in m and "Tikhonov" in m for m in msgs)


# ===========================================================================
# grace_uq select — greedy loop + stratification repair
# ===========================================================================


def test_build_atom_pool_shapes(synthetic_df):
    pool = _build_atom_pool(synthetic_df, ["Mo", "Nb", "Ta", "W"], need_features=True)
    assert pool["struct_id"].shape == (20,)
    assert pool["features"].shape == (20, 8)
    assert (np.unique(pool["struct_id"]) == np.arange(5)).all()
    assert pool["element"].max() == 3


def test_structure_element_counts(synthetic_df):
    elem_map = ["Mo", "Nb", "Ta", "W"]
    counts = _structure_element_counts(synthetic_df, elem_map)
    assert counts.shape == (5, 4)
    assert (counts == 1).all()


def test_collect_top_n_n_structures_only(synthetic_df):
    elem_map = ["Mo", "Nb", "Ta", "W"]
    pool = _build_atom_pool(synthetic_df, elem_map, need_features=True)
    elem_counts = _structure_element_counts(synthetic_df, elem_map)
    rng = np.random.default_rng(0)
    atom_iter = get_strategy("random-all")(pool, rng=rng)
    top, picks, _ = _collect_top_n(
        atom_iter, pool, elem_counts, n_structures=3, floors=None
    )
    assert len(top) == 3
    assert all(0 <= s < 5 for s in top)


def test_collect_top_n_meets_stratification(synthetic_df):
    elem_map = ["Mo", "Nb", "Ta", "W"]
    pool = _build_atom_pool(synthetic_df, elem_map, need_features=True)
    elem_counts = _structure_element_counts(synthetic_df, elem_map)
    floors = {0: 2, 1: 2, 2: 2, 3: 2}
    rng = np.random.default_rng(0)
    atom_iter = get_strategy("random-all")(pool, rng=rng)
    top, _, _ = _collect_top_n(
        atom_iter, pool, elem_counts, n_structures=2, floors=floors
    )
    out_counts = elem_counts[top].sum(axis=0)
    for e, f in floors.items():
        assert out_counts[e] >= f


def test_repair_stratification_preserves_input_order_when_repair_impossible():
    """Regression: when no donor can satisfy any unmet floor, the function
    reaches its tail return path. It must return the input list with its
    original ordering, not a hash-ordered ``list(set(top))`` conversion —
    downstream consumers (``out_df.iloc[top]``, ``n_atoms_selected``) rely
    on the pick-count ranking being preserved."""
    # 4 structures, 1 element type. No structure has any atoms of element 0,
    # so no swap can ever improve the selection — function falls through to
    # the tail return.
    elem_counts = np.zeros((4, 1), dtype=np.int64)
    top = [3, 1, 0]  # deliberately unsorted; set order would be [0, 1, 3]
    picks = {3: 5, 1: 5, 0: 5}
    floors = {0: 5}  # unmet, impossible to fill

    result = _repair_stratification(
        top, picks, elem_counts, floors, n_total_structs=4
    )

    assert result == top, f"expected {top} (input order), got {result}"


def test_repair_stratification_swaps_in_donor():
    """When the strategy's top-N misses an element, repair swaps in a donor."""
    elem_counts = np.array(
        [
            [4, 0, 0, 0],  # 0: pure Mo
            [4, 0, 0, 0],  # 1: pure Mo
            [4, 0, 0, 0],  # 2: pure Mo
            [0, 0, 0, 4],  # 3: pure W (only donor for element 3)
        ],
        dtype=np.int64,
    )
    top = [0, 1, 2]
    picks = {0: 5, 1: 5, 2: 5}
    floors = {0: 2, 3: 2}
    repaired = _repair_stratification(top, picks, elem_counts, floors, n_total_structs=4)
    assert 3 in repaired
    out = elem_counts[repaired].sum(axis=0)
    assert out[0] >= 2 and out[3] >= 2


def test_print_selection_summary_handles_empty_top(capsys):
    """Empty `top` must not crash on .min()/.max() of an empty array."""
    from tensorpotential.uq.cli.select import _print_selection_summary

    elem_counts = np.array([[1, 0], [0, 1]], dtype=np.int64)
    out_df = pd.DataFrame({"ase_atoms": [], "n_atoms_selected": []})
    _print_selection_summary(
        out_df, [], elem_counts, ["A", "B"], floors=None, strategy="random-all"
    )
    text = capsys.readouterr().out
    assert "no structures selected" in text


def test_bincount_skips_unknown_element_indices():
    """pool['element'] may contain -1 for unknown atoms; clamp must not crash."""
    pool_elems = np.array([0, 0, 1, -1, -1, 2], dtype=np.int64)
    n_elements = 3
    elem = pool_elems
    counts = np.bincount(elem[elem >= 0], minlength=n_elements)
    assert counts.tolist() == [2, 1, 1]


def test_extrap_filter_zero_pool_warns(synthetic_df, caplog):
    pool = _build_atom_pool(synthetic_df, ["Mo", "Nb", "Ta", "W"], need_features=True)
    rng = np.random.default_rng(0)
    out = list(
        get_strategy("random-extrap")(
            pool, rng=rng, gamma_min=100.0, gamma_max=200.0
        )
    )
    assert out == []


def test_default_min_per_element_floor_is_d_plus_one():
    """grace_uq select must default the per-element stratification floor to
    D + 1. Source-level check avoids a full CLI invocation."""
    src = inspect.getsource(select_mod.select_main)
    assert "D + 1" in src or "(D + 1)" in src, (
        "select_main must default min_per_element floor to D+1"
    )


def test_select_uq_is_optional(tmp_path):
    """--uq must be optional; element_map can come from model metadata.yaml."""
    parser = select_mod._build_parser()
    # Argparse should accept the form without --uq
    args = parser.parse_args(
        [
            "--model",
            str(tmp_path),
            "--predicted",
            str(tmp_path / "p.pkl.gz"),
            "-n",
            "10",
            "--no-element-stratified",
        ]
    )
    assert args.uq is None
    assert args.element_stratified is False


def test_load_element_map_from_savedmodel(tmp_path):
    """metadata.yaml under a SavedModel dir feeds element_map into select."""
    import yaml

    from tensorpotential.uq.cli._common import load_element_map_from_savedmodel

    model_dir = tmp_path / "saved_model"
    model_dir.mkdir()
    with open(model_dir / "metadata.yaml", "w") as f:
        yaml.safe_dump({"chemical_symbols": ["H", "C", "O"]}, f)
    em = load_element_map_from_savedmodel(str(model_dir))
    assert em == ["H", "C", "O"]

    # Non-existent directory → None
    assert load_element_map_from_savedmodel(str(tmp_path / "nope")) is None

    # Directory without metadata.yaml → None
    empty = tmp_path / "empty"
    empty.mkdir()
    assert load_element_map_from_savedmodel(str(empty)) is None


def test_infer_D_from_features():
    """_infer_D_from_features reads D from the first usable features array."""
    from tensorpotential.uq.cli.select import _infer_D_from_features

    # No features column → None
    df0 = pd.DataFrame({"ase_atoms": [None]})
    assert _infer_D_from_features(df0) is None

    # All-None features → None
    df1 = pd.DataFrame({"features": [None, None]})
    assert _infer_D_from_features(df1) is None

    # First row None, second row valid → D from second row
    df2 = pd.DataFrame({"features": [None, np.zeros((3, 17), dtype=np.float32)]})
    assert _infer_D_from_features(df2) == 17

    # Empty (0-row) features should be skipped
    df3 = pd.DataFrame(
        {"features": [np.zeros((0, 21)), np.zeros((4, 21), dtype=np.float32)]}
    )
    assert _infer_D_from_features(df3) == 21


def test_select_requires_either_uq_or_model(tmp_path):
    """select_main rejects invocations missing both --uq and --model."""
    pkl = tmp_path / "p.pkl.gz"
    # Empty DataFrame just so load_dataset_any does not pre-fail; we expect
    # the SystemExit before that anyway because args.predicted check still
    # passes (file just needs to be addressable).
    pd.DataFrame({"ase_atoms": [], "gamma": []}).to_pickle(pkl, compression="gzip")
    with pytest.raises(SystemExit, match="either --uq or --model"):
        select_mod.select_main(
            [
                "--predicted",
                str(pkl),
                "-n",
                "1",
                "--no-element-stratified",
            ]
        )


# ===========================================================================
# grace_uq select — strategy registry & FPS
# ===========================================================================


def test_registry_lists_builtins():
    names = list_strategies()
    assert "random-all" in names
    assert "random-extrap" in names
    assert "fps-all" in names
    assert "fps-extrap" in names


def test_random_all_yields_permutation():
    pool = _make_pool()
    rng = np.random.default_rng(123)
    out = list(get_strategy("random-all")(pool, rng=rng))
    assert sorted(out) == list(range(len(pool["struct_id"])))


def test_random_extrap_filters_by_gamma():
    pool = _make_pool()
    rng = np.random.default_rng(7)
    out = list(get_strategy("random-extrap")(pool, rng=rng, gamma_min=2.0, gamma_max=4.0))
    g = pool["gamma"][out]
    assert ((g >= 2.0) & (g <= 4.0)).all()


def test_fps_all_yields_unique_atoms():
    pool = _make_pool()
    rng = np.random.default_rng(0)
    out = list(get_strategy("fps-all")(pool, rng=rng, fps_max_pool=50))
    assert len(set(out)) == len(out)
    assert set(out).issubset(set(range(len(pool["struct_id"]))))


def test_fps_first_picks_are_far_apart():
    """First 5 FPS picks should have larger pairwise distances than 5 random
    picks (in expectation)."""
    rng = np.random.default_rng(42)
    feats = rng.standard_normal((400, 8)).astype(np.float32)
    pool = {
        "struct_id": np.arange(400),
        "element": np.zeros(400, dtype=np.int32),
        "gamma": np.ones(400, dtype=np.float32),
        "features": feats,
    }
    out = []
    for i, idx in enumerate(
        get_strategy("fps-all")(
            pool, rng=np.random.default_rng(0), fps_max_pool=200
        )
    ):
        out.append(idx)
        if i >= 4:
            break
    fps_pts = feats[out]
    fps_min = np.min(
        [
            np.linalg.norm(fps_pts[i] - fps_pts[j])
            for i in range(len(fps_pts))
            for j in range(i + 1, len(fps_pts))
        ]
    )

    rand_pts = feats[rng.choice(len(feats), 5, replace=False)]
    rand_min = np.min(
        [
            np.linalg.norm(rand_pts[i] - rand_pts[j])
            for i in range(len(rand_pts))
            for j in range(i + 1, len(rand_pts))
        ]
    )
    assert fps_min >= rand_min


def test_fps_iter_all_unique_under_minibatch():
    """Mini-batch FPS must not yield duplicates across shards."""
    rng = np.random.default_rng(0)
    feats = rng.standard_normal((300, 4)).astype(np.float32)
    cand = np.arange(300)
    picks = list(_fps_iter(cand, feats, rng=np.random.default_rng(1), fps_max_pool=80))
    assert len(picks) == len(set(picks)) == 300


def test_register_strategy_decorator():
    @register_strategy("test-noop")
    def _noop(pool, **kwargs):
        return iter([])

    assert "test-noop" in list_strategies()


def test_chunked_min_dist_matches_brute_force():
    rng = np.random.default_rng(0)
    feats = rng.standard_normal((50, 6)).astype(np.float32)
    ref = rng.standard_normal((20, 6)).astype(np.float32)

    d_chunk = _chunked_min_dist(feats, ref, chunk=8)
    d_ref = np.array(
        [np.min(np.linalg.norm(ref - x, axis=1)) for x in feats], dtype=np.float32
    )
    np.testing.assert_allclose(d_chunk, d_ref, atol=1e-4)


def test_fps_extrap_requires_features():
    rng = np.random.default_rng(0)
    pool = {
        "struct_id": np.arange(10),
        "element": np.zeros(10, dtype=np.int32),
        "gamma": np.ones(10, dtype=np.float32),
        # no features
    }
    with pytest.raises(KeyError):
        list(get_strategy("fps-all")(pool, rng=rng))


# ===========================================================================
# grace_uq build — --train-data-weighted / --filter-fn
# ===========================================================================


def test_weighted_train_data_action_parses_groups():
    """Two --train-data-weighted blocks accumulate as (weight, files) tuples."""
    import argparse as _ap
    parser = _ap.ArgumentParser()
    parser.add_argument(
        "--train-data-weighted",
        nargs="+",
        action=_WeightedTrainDataAction,
        default=None,
    )
    ns = parser.parse_args([
        "--train-data-weighted", "20.0", "a.pkl.gz", "b.pkl.gz",
        "--train-data-weighted", "1.0", "c.pkl.gz",
    ])
    assert ns.train_data_weighted == [
        (20.0, ["a.pkl.gz", "b.pkl.gz"]),
        (1.0, ["c.pkl.gz"]),
    ]


def test_weighted_train_data_action_rejects_nonnumeric_weight(capsys):
    import argparse as _ap
    parser = _ap.ArgumentParser()
    parser.add_argument(
        "--train-data-weighted",
        nargs="+",
        action=_WeightedTrainDataAction,
        default=None,
    )
    with pytest.raises(SystemExit):
        parser.parse_args(["--train-data-weighted", "notanumber", "a.pkl.gz"])


def test_resolve_weighted_train_data_builds_weight_map(tmp_path):
    """_resolve_weighted_train_data flattens groups into (paths, weight_map)."""
    import argparse as _ap
    f_omat1 = tmp_path / "omat1.pkl.gz"
    f_omat2 = tmp_path / "omat2.pkl.gz"
    f_smax = tmp_path / "smax.pkl.gz"
    for f in (f_omat1, f_omat2, f_smax):
        f.write_bytes(b"")
    ns = _ap.Namespace(
        train_data=["should-be-ignored.pkl.gz"],
        train_data_weighted=[
            (20.0, [str(f_omat1), str(f_omat2)]),
            (1.0, [str(f_smax)]),
        ],
    )
    paths, wmap = _resolve_weighted_train_data(ns)
    assert paths == [
        os.path.abspath(str(f_omat1)),
        os.path.abspath(str(f_omat2)),
        os.path.abspath(str(f_smax)),
    ]
    assert wmap[os.path.abspath(str(f_omat1))] == 20.0
    assert wmap[os.path.abspath(str(f_omat2))] == 20.0
    assert wmap[os.path.abspath(str(f_smax))] == 1.0


def test_resolve_weighted_train_data_conflicting_weights_raises(tmp_path):
    import argparse as _ap
    f = tmp_path / "shared.pkl.gz"
    f.write_bytes(b"")
    ns = _ap.Namespace(
        train_data=[],
        train_data_weighted=[
            (5.0, [str(f)]),
            (1.0, [str(f)]),
        ],
    )
    with pytest.raises(ValueError, match="multiple groups"):
        _resolve_weighted_train_data(ns)


def test_resolve_weighted_train_data_empty_returns_train_data(tmp_path):
    import argparse as _ap
    ns = _ap.Namespace(
        train_data=["/x/y.pkl.gz"],
        train_data_weighted=None,
    )
    paths, wmap = _resolve_weighted_train_data(ns)
    assert paths == ["/x/y.pkl.gz"]
    assert wmap == {}


def test_cli_weighted_and_train_data_mutually_exclusive(tmp_path, capsys):
    """build_main must SystemExit when both --train-data (non-default) and
    --train-data-weighted are supplied."""
    f = tmp_path / "x.pkl.gz"
    f.write_bytes(b"")
    argv = [
        "--model-yaml", str(tmp_path / "model.yaml"),
        "--checkpoint", str(tmp_path / "ckpt.index"),
        "--train-data", str(f),
        "--train-data-weighted", "2.0", str(f),
        "--artifact-path", str(tmp_path / "out.npz"),
    ]
    with pytest.raises(SystemExit):
        build_main(argv)
    err = capsys.readouterr().err
    assert "mutually exclusive" in err


def test_load_filter_fn_returns_none_for_none():
    assert _load_filter_fn(None) is None


def test_load_filter_fn_loads_callable(tmp_path, monkeypatch):
    """A valid module.path:fn spec loads the callable; 1-arg user fns are
    transparently wrapped to the canonical (atoms, file_path) shape."""
    mod_path = tmp_path / "tmp_filter_mod.py"
    mod_path.write_text(
        "def keep_large(atoms):\n"
        "    return len(atoms) >= 5\n"
    )
    monkeypatch.syspath_prepend(str(tmp_path))
    fn = _load_filter_fn("tmp_filter_mod:keep_large")
    # Small atoms → False; large atoms → True
    small = Atoms("Mo" * 2)
    big = Atoms("Mo" * 7)
    assert fn(small, "/some/shard.pkl.gz") is False
    assert fn(big, "/some/shard.pkl.gz") is True


def test_load_filter_fn_missing_colon_errors():
    with pytest.raises(ValueError, match="module.path:function_name"):
        _load_filter_fn("no_colon_here")


def test_load_filter_fn_empty_parts_errors():
    with pytest.raises(ValueError, match="malformed"):
        _load_filter_fn(":fn")
    with pytest.raises(ValueError, match="malformed"):
        _load_filter_fn("module:")


def test_load_filter_fn_bad_module_errors():
    with pytest.raises(ImportError, match="cannot import module"):
        _load_filter_fn("definitely_not_a_real_module_xxx:fn")


def test_load_filter_fn_missing_attribute_errors(tmp_path, monkeypatch):
    mod_path = tmp_path / "tmp_filter_mod2.py"
    mod_path.write_text("def some_other_fn(a): return True\n")
    monkeypatch.syspath_prepend(str(tmp_path))
    with pytest.raises(AttributeError, match="no attribute"):
        _load_filter_fn("tmp_filter_mod2:nonexistent")


def test_load_and_subsample_applies_filter(tmp_path):
    """_load_and_subsample applies the filter before --frac subsampling."""
    df = pd.DataFrame({
        "ase_atoms": [
            _make_atoms(["Mo"] * 2),
            _make_atoms(["Mo"] * 10),
            _make_atoms(["Mo"] * 6),
            _make_atoms(["Mo"] * 3),
        ]
    })
    p = tmp_path / "data.pkl.gz"
    df.to_pickle(str(p), compression="gzip")

    # Keep only structures with >= 5 atoms → 2 of 4 survive.
    def keep_large(atoms, file_path):
        return len(atoms) >= 5
    rng = np.random.RandomState(0)
    survivors = _load_and_subsample(str(p), rng, frac=None, filter_fn=keep_large)
    assert len(survivors) == 2
    for a in survivors["ase_atoms"]:
        assert len(a) >= 5


def test_weighted_train_data_action_requires_at_least_two_tokens(capsys):
    """A bare --train-data-weighted 5.0 with no file paths must error out."""
    import argparse as _ap
    parser = _ap.ArgumentParser()
    parser.add_argument(
        "--train-data-weighted",
        nargs="+",
        action=_WeightedTrainDataAction,
        default=None,
    )
    with pytest.raises(SystemExit):
        parser.parse_args(["--train-data-weighted", "5.0"])


# ---------------------------------------------------------------------------
# select_optimal_clusters — deterministic tiebreaker
# ---------------------------------------------------------------------------


def test_select_optimal_clusters_obvious_elbow_picks_knee():
    """Clear elbow at k=4 (cliff drop, then plateau) → returns 4."""
    k_values = [1, 2, 4, 8, 16]
    inertias = [10.0, 8.0, 1.0, 0.9, 0.85]
    assert select_optimal_clusters(k_values, inertias) == 4


def test_select_optimal_clusters_ties_pick_smallest_k(caplog):
    """When several k candidates tie on max distance to chord, pick the
    smallest and emit an INFO log so the close call is visible."""
    # Hand-crafted three-way tie: k=[1,3,5,7], inertias=[12,4,2,0] place
    # k=3, k=5, k=7 all at distance 0.5 to the chord (smallest-k wins).
    k_values = [1, 3, 5, 7]
    inertias = [12.0, 4.0, 2.0, 0.0]
    with caplog.at_level("INFO", logger="grace_uq"):
        chosen = select_optimal_clusters(k_values, inertias)
    assert chosen == 3
    assert any("Elbow tiebreak" in r.getMessage() for r in caplog.records)


def test_select_optimal_clusters_flat_falls_back_to_first():
    """All-constant inertias hit the flat-plateau guard → first k."""
    assert select_optimal_clusters([1, 2, 4, 8], [5.0, 5.0, 5.0, 5.0]) == 1


# ---------------------------------------------------------------------------
# Subprocess smoke test: end-to-end ``python -m tensorpotential.uq.cli.build``
# ---------------------------------------------------------------------------
_TRAINED_DIR = os.path.join(
    os.path.dirname(__file__), "MoNbTaW-GRACE", "seed", "42"
)
_TRAINED_AVAILABLE = (
    os.path.exists(os.path.join(_TRAINED_DIR, "model.yaml"))
    and os.path.exists(
        os.path.join(_TRAINED_DIR, "checkpoints", "checkpoint.best_test_loss.index")
    )
    and os.path.exists(os.path.join(_TRAINED_DIR, "training_set.pkl.gz"))
)


@pytest.mark.skipif(
    not _TRAINED_AVAILABLE, reason="MoNbTaW-GRACE seed/42 trained model not available"
)
def test_grace_uq_build_subprocess_smoke(tmp_path):
    """Smoke-test ``python -m tensorpotential.uq.cli.build`` end-to-end.

    Spawning the CLI in a real subprocess (rather than calling ``build_main``
    in-process) catches packaging regressions — missing ``__main__.py``,
    broken worker-spawn argv, missing imports under a clean interpreter — that
    in-process tests can't see.
    """
    artifact_path = tmp_path / "gmm_artifacts.npz"
    proc = subprocess.run(
        [
            sys.executable,
            "-m",
            "tensorpotential.uq.cli.build",
            "--n-workers", "1",
            "--n-clusters", "2",
            "--no-export",
            "--model-yaml", os.path.join(_TRAINED_DIR, "model.yaml"),
            "--checkpoint",
            os.path.join(_TRAINED_DIR, "checkpoints", "checkpoint.best_test_loss.index"),
            "--train-data", os.path.join(_TRAINED_DIR, "training_set.pkl.gz"),
            "--artifact-path", str(artifact_path),
        ],
        cwd=str(tmp_path),
        capture_output=True,
        text=True,
        timeout=240,
        env={**os.environ, "TF_USE_LEGACY_KERAS": "1"},
    )
    assert proc.returncode == 0, (
        f"grace_uq build failed (rc={proc.returncode}).\n"
        f"--- stdout ---\n{proc.stdout[-2000:]}\n"
        f"--- stderr ---\n{proc.stderr[-2000:]}"
    )
    assert artifact_path.exists()

    # Sanity-check artifact contents: schema-versioned, per-element centroids
    # and inverse covariances present for at least one element.
    with np.load(str(artifact_path)) as data:
        assert uq_constants.SCHEMA_VERSION_KEY in data.files
        assert int(data[uq_constants.SCHEMA_VERSION_KEY]) == uq_constants.SCHEMA_VERSION
        elements = data["elements"]
        assert len(elements) > 0
        e0 = int(elements[0])
        assert f"{uq_constants.CENTROIDS}_{e0}" in data.files
        assert f"{uq_constants.INV_COV}_{e0}" in data.files
        assert "interp_thresholds" in data.files


@pytest.mark.skipif(
    not _TRAINED_AVAILABLE, reason="MoNbTaW-GRACE seed/42 trained model not available"
)
def test_load_uq_model_tolerates_checkpoint_without_intra_epoch_save(monkeypatch):
    """``load_uq_model`` loads a checkpoint lacking the ``intra_epoch_save`` flag.

    UQ artifacts are built on foundation / older checkpoints that predate the
    mid-epoch-checkpoint bookkeeping variable (``intra_epoch_save``, a
    training-only scalar bool). UQ needs only the model weights, so the load
    must succeed under the default strict object match — it must not trip over
    auxiliary training trackables that are absent from the file. Regression for
    ``AssertionError: Found 1 Python objects that were not bound to checkpointed
    values ... dtype=bool``.
    """
    from tensorpotential.uq.factories import load_uq_model
    from tensorpotential.tpmodel import ComputeEnergy

    # Exercise the strict default path, not the GRACE_UQ_LENIENT_LOAD hatch.
    monkeypatch.delenv("GRACE_UQ_LENIENT_LOAD", raising=False)

    # The MoNbTaW-GRACE fixture checkpoint predates intra_epoch_save (verified:
    # its trackables are step/epoch/optimizer/model only).
    checkpoint = os.path.join(_TRAINED_DIR, "checkpoints", "checkpoint.best_test_loss")
    tp, instructions = load_uq_model(
        model_yaml=os.path.join(_TRAINED_DIR, "model.yaml"),
        checkpoint=checkpoint,
        model_compute_function=ComputeEnergy(extra_return_keys=[uq_constants.FEATURES]),
    )
    assert tp is not None
    # The basis-RP UQ feature was patched in under the canonical FEATURES key.
    assert uq_constants.FEATURES in instructions
