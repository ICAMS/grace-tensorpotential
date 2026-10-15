"""grace_uq select from HAL candidate data (hal_features.npz / pooled cand_* columns).

Model-free: walker directories are written by HALCollector with fake calculators, the
CALM artifact is synthetic.
"""

import os
from collections import Counter

import numpy as np
import pandas as pd
import pytest
from ase.build import bulk
from ase.io import read

from tensorpotential.uq import constants as uqc
from tensorpotential.uq.artifact_builder import GMMUQArtifactBuilder
from tensorpotential.uq.cli import select as select_mod
from tensorpotential.uq.hal import features as hal_features
from tests.hal_fakes import make_walker_dir

D = 4


def _write_artifact(path, seed=0, element_map=("Cu", "Al"), K=2, store_fp32=False,
                    cond=None):
    """Synthetic CALM artifact with scatter (updatable) and unit thresholds.

    ``store_fp32=True`` stores it like a production artifact; ``cond`` gives every
    covariance eigenvalues log-spaced over that condition number (real artifacts:
    ~1e5 with the 1e-6 regulariser)."""
    rng = np.random.default_rng(seed)
    arts = {}
    for e in range(len(element_map)):
        a = rng.standard_normal((K, D, D))
        cov = np.einsum("kij,klj->kil", a, a) / D + 0.5 * np.eye(D)
        if cond is not None:
            q = np.linalg.qr(rng.standard_normal((K, D, D)))[0]
            w = np.logspace(np.log10(2.0 / cond), np.log10(2.0), D)
            cov = np.einsum("kij,j,klj->kil", q, w, q)
        counts = np.full(K, 50, dtype=np.int64)
        arts[e] = {
            uqc.CENTROIDS: 3.0 * rng.standard_normal((K, D)),
            uqc.INV_COV: np.linalg.inv(cov + 1e-6 * np.eye(D)),  # as the builder (eps 1e-6)
            uqc.COUNTS: counts,
            uqc.EFFECTIVE_COUNT: counts.astype(np.float64),
            uqc.SCATTER: cov * 50.0,
        }
    GMMUQArtifactBuilder.save_artifacts(
        str(path), arts, store_fp32=store_fp32,
        interp_thresholds=np.ones((len(element_map), K)),
        element_map=np.array(element_map),
    )
    return str(path)


def _walkers(tmp_path, artifact, n_frames=(5, 4), seed=1, fgm=1.0):
    """Cu walkers; frame k of walker w: pristine gamma 1.0-3.0 on two atoms, else 0.5."""
    rng = np.random.default_rng(seed)
    sha = hal_features.file_sha256(artifact)
    atoms = bulk("Cu", "fcc", a=3.6, cubic=True)
    dirs = []
    for w, nf in enumerate(n_frames):
        feats = [rng.standard_normal((4, D)) * 3 for _ in range(nf)]
        gp = []
        for _ in range(nf):
            g = np.full(4, 0.5)
            g[rng.choice(4, 2, replace=False)] = rng.uniform(1.0, 3.0, 2)
            gp.append(g)
        d = str(tmp_path / f"w{w}")
        make_walker_dir(d, atoms, np.zeros(4, int), feats, gp, feature_gamma_min=fgm,
                        feature_meta={"artifact_sha256": sha, "model_yaml": "m.yaml"})
        dirs.append(d)
    return dirs


@pytest.fixture
def no_model(monkeypatch):
    """Selection from candidate data must never predict or build a model calculator."""
    from tensorpotential.uq import factories

    def boom(*a, **k):
        raise AssertionError("no model prediction expected")

    monkeypatch.setattr(factories, "get_gmm_uq_calculator", boom)
    monkeypatch.setattr(select_mod, "_maybe_run_predict", boom)


@pytest.fixture
def reader_spy(monkeypatch):
    calls = []
    real = hal_features.read_frames
    parsed = []
    real_ase_read = hal_features.ase_read

    def spy(path, indices, offsets=None):
        calls.append((os.path.basename(os.path.dirname(path)), list(indices)))
        return real(path, indices, offsets=offsets)

    def ase_spy(*a, **k):
        parsed.append(1)
        return real_ase_read(*a, **k)

    monkeypatch.setattr(hal_features, "read_frames", spy)
    monkeypatch.setattr(hal_features, "ase_read", ase_spy)
    return calls, parsed


def _select(*argv):
    return select_mod.select_main([str(a) for a in argv])


def test_frame_reader_matches_ase(tmp_path):
    art = _write_artifact(tmp_path / "a.npz")
    (d,) = _walkers(tmp_path, art, n_frames=(6,))
    path = os.path.join(d, "hal_collected.extxyz")
    ref = read(path, ":")
    got = hal_features.read_frames(path, [4, 0, 5])
    for k, at in zip([4, 0, 5], got):
        np.testing.assert_array_equal(at.positions, ref[k].positions)
        np.testing.assert_array_equal(at.arrays["gamma_pristine"], ref[k].arrays["gamma_pristine"])
        assert at.info.keys() == ref[k].info.keys()
        assert at.info["collection_index"] == ref[k].info["collection_index"] == k
    assert len(hal_features.frame_offsets(path)) == 6
    with pytest.raises(IndexError):
        hal_features.read_frames(path, [6])


def test_select_hal_dirs_without_model_reads_only_selected(tmp_path, no_model, reader_spy):
    calls, parsed = reader_spy
    art = _write_artifact(tmp_path / "a.npz")
    dirs = _walkers(tmp_path, art)
    out = tmp_path / "sel.pkl.gz"
    assert _select("--hal-dirs", *dirs, "--artifact", art, "-n", 3, "--strategy",
                   "random-extrap", "--no-element-stratified", "--output", out) == 0
    sel = pd.read_pickle(out, compression="gzip")
    assert len(sel) == 3 and len(parsed) == 3  # only the selected frames were parsed
    assert sum(len(ks) for _, ks in calls) == 3
    assert not [c for c in sel.columns if c.startswith("cand_") or c == "n_atoms_selected"]
    for _, r in sel.iterrows():  # the frame the sidecar row pointed at
        ref = read(os.path.join(r["hal_dir"], "hal_collected.extxyz"), r["collection_index"])
        np.testing.assert_array_equal(r["ase_atoms"].positions, ref.positions)
        assert r["ase_atoms"].calc is None


def test_select_hal_dirs_refusals(tmp_path, no_model):
    art = _write_artifact(tmp_path / "a.npz")
    other = _write_artifact(tmp_path / "b.npz", seed=7)
    dirs = _walkers(tmp_path, art)
    base = ("--hal-dirs", *dirs, "-n", 2, "--no-element-stratified",
            "--output", tmp_path / "s.pkl.gz")
    with pytest.raises(SystemExit, match="not the CALM artifact"):
        _select(*base, "--artifact", other, "--strategy", "random-extrap")
    with pytest.raises(SystemExit, match="feature_gamma_min"):
        _select(*base, "--artifact", art, "--strategy", "random-extrap", "--gamma-min", 0.5)
    with pytest.raises(SystemExit, match="fps-all needs every atom"):
        _select(*base, "--artifact", art, "--strategy", "fps-all")
    os.remove(os.path.join(dirs[1], "hal_features.npz"))
    with pytest.raises(SystemExit, match="no walker directory"):
        os.remove(os.path.join(dirs[0], "hal_features.npz"))
        _select(*base, "--artifact", art, "--strategy", "random-extrap")


def test_select_pool_with_candidate_columns_skips_predict(tmp_path, no_model):
    from tensorpotential.uq.hal import pool_frames

    art = _write_artifact(tmp_path / "a.npz")
    dirs = _walkers(tmp_path, art)
    pool = str(tmp_path / "pool.pkl.gz")
    pool_frames(dirs).to_pickle(pool, compression="gzip")
    out = tmp_path / "sel.pkl.gz"
    assert _select("--dataset", pool, "--artifact", art, "-n", 4, "--strategy",
                   "random-extrap", "--no-element-stratified", "--output", out) == 0
    sel = pd.read_pickle(out, compression="gzip")
    assert len(sel) == 4 and not [c for c in sel.columns if c.startswith("cand_")]
    with pytest.raises(SystemExit, match="not the CALM artifact"):
        _select("--dataset", pool, "--artifact", _write_artifact(tmp_path / "b.npz", seed=3),
                "-n", 2, "--strategy", "random-extrap", "--no-element-stratified",
                "--output", out)


def test_hal_dirs_element_counts_and_floor_supply(tmp_path, no_model):
    """Per-frame element counts come from the sidecar; floors clamp to that supply."""
    art = _write_artifact(tmp_path / "a.npz")
    dirs = _walkers(tmp_path, art)
    pool, counts, src, d, offsets = select_mod._candidate_pool_from_hal_dirs(
        dirs, ["Cu", "Al"], 1.0, None, art)
    assert [len(offsets[x]) for x in dirs] == [5, 4]
    assert d == D and counts.shape == (9, 2) and (counts[:, 0] == 4).all()
    assert (counts[:, 1] == 0).all() and src[5] == (dirs[1], 0)
    assert len(pool["struct_id"]) == 18 and (pool["element"] == 0).all()
    assert pool[uqc.FEATURES].dtype == np.float32
    # gamma window applies to the stored pristine gamma
    pool2, *_ = select_mod._candidate_pool_from_hal_dirs(dirs, ["Cu", "Al"], 2.0, None, art)
    assert (pool2["gamma"] >= 2.0).all() and len(pool2["gamma"]) < 18


# -- calm-greedy ----------------------------------------------------------------------


def _tower(art):
    import tensorflow as tf

    from tensorpotential.uq.gmmuq import GMMUQModel

    return GMMUQModel(art, param_dtype=tf.float64)


def _random_pool(art, n_frames=40, seed=0, n_dirs=3, spread=0.3):
    """Candidate atoms of two elements along a few shared directions per element."""
    from tensorpotential.uq.artifact_builder import GMMUQArtifactBuilder

    rng = np.random.default_rng(seed)
    arts = GMMUQArtifactBuilder.load(art)
    dirs = rng.standard_normal((2, n_dirs, D)) * 2.5
    sid, el, f = [], [], []
    for s in range(n_frames):
        for _ in range(rng.integers(1, 4)):
            e = int(rng.integers(0, 2))
            c = arts[e][uqc.CENTROIDS][rng.integers(0, 2)]
            sid.append(s)
            el.append(e)
            f.append(c + dirs[e, rng.integers(n_dirs)] * rng.uniform(0.6, 1.4)
                     + spread * rng.standard_normal(D))
    return {"struct_id": np.array(sid), "element": np.array(el, dtype=np.int32),
            "gamma": np.full(len(sid), 2.0, np.float32),
            uqc.FEATURES: np.array(f, dtype=np.float32)}


def test_candidate_gamma_matches_gmm(tmp_path):
    from tensorpotential.uq.cli.strategies import CandidateGamma

    art = _write_artifact(tmp_path / "a.npz")
    pool = _random_pool(art)
    tower = _tower(art)
    sc = CandidateGamma(tower, pool[uqc.FEATURES], pool["element"])
    ref = tower.gamma_from_features(pool[uqc.FEATURES].astype(np.float64), pool["element"])[0]
    idx = np.arange(len(ref))
    np.testing.assert_allclose(sc.gamma(idx), ref, rtol=1e-10)
    np.testing.assert_allclose(sc.gamma_all(), ref, rtol=1e-8)
    m = pool["struct_id"] == 3
    tower.update_one(pool[uqc.FEATURES][m].astype(np.float64), pool["element"][m], alpha=0.5,
                     mode="additive")
    sc.refresh([0, 1])
    ref2 = tower.gamma_from_features(pool[uqc.FEATURES].astype(np.float64), pool["element"])[0]
    np.testing.assert_allclose(sc.gamma(idx), ref2, rtol=1e-10)
    assert (ref2 <= ref + 1e-12).all()  # additive: gamma never rises


@pytest.mark.parametrize("seed,n", [(0, 1000), (1, 1000), (2, 7), (3, 1000)])
def test_calm_greedy_lazy_equals_eager(tmp_path, seed, n):
    from tensorpotential.uq.cli.strategies import CandidateGamma, get_strategy

    art = _write_artifact(tmp_path / "a.npz", seed=seed)
    pool = _random_pool(art, seed=seed)
    greedy = get_strategy("calm-greedy")
    lazy = greedy(pool, gmm=_tower(art), n_structures=n, progress=False)
    eager = greedy(pool, gmm=_tower(art), n_structures=n, lazy=False, progress=False)
    assert lazy.frames == eager.frames and lazy.alphas == eager.alphas
    np.testing.assert_array_equal(lazy.gamma_at_pick, eager.gamma_at_pick)
    assert (lazy.natural, lazy.covered) == (eager.natural, eager.covered)
    assert lazy.remaining_max_gamma == pytest.approx(eager.remaining_max_gamma, rel=1e-12)
    assert lazy.n_scored < eager.n_scored
    assert np.all(np.diff(lazy.gamma_at_pick) <= 1e-12)  # max-first: non-increasing
    if n == 1000:  # natural stop: every unpicked frame at or below gamma_min
        tower = _tower(art)
        for f in lazy.frames:
            m = pool["struct_id"] == f
            tower.update_one(pool[uqc.FEATURES][m].astype(np.float64), pool["element"][m],
                             alpha=lazy.alphas[lazy.frames.index(f)], mode="additive")
        sc = CandidateGamma(tower, pool[uqc.FEATURES], pool["element"])
        rest = ~np.isin(pool["struct_id"], lazy.frames)
        assert lazy.natural and 0 < len(lazy.frames) < 40
        assert sc.gamma(np.flatnonzero(rest)).max() <= 1.0
    else:
        assert len(lazy.frames) == n and not lazy.natural and not lazy.covered


def test_select_hal_dirs_calm_greedy_default(tmp_path, no_model, reader_spy, capsys):
    calls, parsed = reader_spy
    art = _write_artifact(tmp_path / "a.npz")
    dirs = _walkers(tmp_path, art, n_frames=(12, 10))
    out = tmp_path / "sel.pkl.gz"
    with pytest.raises(SystemExit, match="calm-greedy.*needs --artifact.*random-extrap"):
        _select("--hal-dirs", *dirs, "-n", 30, "--no-element-stratified", "--output", out)
    assert _select("--hal-dirs", *dirs, "--artifact", art, "-n", 30,
                   "--no-element-stratified", "--output", out) == 0
    sel = pd.read_pickle(out, compression="gzip")
    text = capsys.readouterr().out
    assert "strategy='calm-greedy'" in text and "natural set size" in text
    assert 0 < len(sel) <= 22 and len(parsed) == len(sel)
    assert list(sel["pick_order"]) == list(range(len(sel)))
    assert (np.diff(sel["gamma_at_pick"]) <= 1e-12).all() and (sel["gamma_at_pick"] > 1.0).all()


def test_calm_greedy_element_floor_repair(tmp_path, no_model, capsys):
    """An element without candidates enters through the floors (D+1 atoms): at -n the
    latest greedy picks make room; the summary says so."""
    art = _write_artifact(tmp_path / "a.npz")
    dirs = _walkers(tmp_path, art, n_frames=(8,))
    al = bulk("Al", "fcc", a=4.05, cubic=True)
    d_al = str(tmp_path / "al")
    make_walker_dir(d_al, al, np.ones(4, int), [np.zeros((4, D))] * 3, [[0.5] * 4] * 3,
                    feature_meta={"artifact_sha256": hal_features.file_sha256(art)})
    out = tmp_path / "sel.pkl.gz"
    assert _select("--hal-dirs", *dirs, d_al, "--artifact", art, "-n", 4,
                   "--element-stratified", "--output", out) == 0
    sel = pd.read_pickle(out, compression="gzip")
    syms = [set(a.get_chemical_symbols()) for a in sel["ase_atoms"]]
    n_al = sum(4 for s in syms if s == {"Al"})
    assert n_al >= D + 1 and len(sel) == 4  # floor = D + 1 = 5 -> two Al frames
    al_rows = sel[[s == {"Al"} for s in syms]]
    assert (al_rows["pick_order"] == -1).all() and al_rows["gamma_at_pick"].isna().all()
    assert sorted(sel["pick_order"][sel["pick_order"] >= 0]) == [0, 1]  # earliest kept
    text = capsys.readouterr().out
    assert "2 frame(s) added" in text and "replacing the 2 latest greedy pick(s)" in text


def test_repair_floors_greedy_appends_below_n_and_evicts_latest_at_n():
    counts = np.array([[4, 0]] * 5 + [[0, 4], [1, 3]])  # frames 5, 6 carry element 1
    floors = {0: 2, 1: 5}
    top, added, evicted = select_mod._repair_floors_greedy([3, 0, 1], counts, floors, 6, 7)
    assert top == [3, 0, 1, 5, 6] and added == [5, 6] and evicted == []
    top, added, evicted = select_mod._repair_floors_greedy([3, 0, 1], counts, floors, 3, 7)
    assert top == [3, 5, 6] and evicted == [1, 0]  # latest greedy picks go first
    top, added, evicted = select_mod._repair_floors_greedy([3, 5], counts, floors, 3, 7)
    assert top == [3, 5, 6] and added == [6] and evicted == []


# -- fps-extrap (whitened, first-hit order) -------------------------------------------


def _aniso_artifact(path):
    """One element, one cluster at 0, covariance diag(100, 1, 1, 1)."""
    cov = np.diag([100.0, 1.0, 1.0, 1.0])
    arts = {0: {uqc.CENTROIDS: np.zeros((1, D)), uqc.INV_COV: np.linalg.inv(cov)[None],
                uqc.COUNTS: np.array([50]), uqc.EFFECTIVE_COUNT: np.array([50.0]),
                uqc.SCATTER: cov[None] * 50}}
    GMMUQArtifactBuilder.save_artifacts(str(path), arts, store_fp32=False,
                                        interp_thresholds=np.ones((1, 1)),
                                        element_map=np.array(["Cu"]))
    return str(path)


def _two_cluster_pool():
    """Frames 0-9 spread along the high-variance axis (raw +-10, whitened +-1); frames
    10-19 along a unit-variance axis (+-3 both ways)."""
    rng = np.random.default_rng(0)
    f = np.zeros((40, D))
    f[:20, 0] = rng.uniform(-10, 10, 20)
    f[20:, 1] = rng.uniform(-3, 3, 20)
    return {"struct_id": np.repeat(np.arange(20), 2), "element": np.zeros(40, np.int32),
            "gamma": np.full(40, 2.0, np.float32), uqc.FEATURES: f.astype(np.float32)}


def test_whiten_features_gives_mahalanobis_norm(tmp_path):
    from tensorpotential.uq.cli.strategies import whiten_features

    art = _write_artifact(tmp_path / "a.npz")
    pool = _random_pool(art)
    tower = _tower(art)
    sigma = tower.gamma_from_features(pool[uqc.FEATURES].astype(np.float64), pool["element"])[1]
    z = whiten_features(pool[uqc.FEATURES].copy(), pool["element"], tower)
    np.testing.assert_allclose(np.linalg.norm(z.astype(np.float64), axis=1), sigma, rtol=1e-5)


def test_fps_extrap_whitened_first_hit_order(tmp_path):
    from tensorpotential.uq.cli.strategies import get_strategy, whiten_features

    art = _aniso_artifact(tmp_path / "a.npz")
    fps = get_strategy("fps-extrap")
    assert fps.first_hit
    raw = _two_cluster_pool()
    top_raw, _ = select_mod._collect_first_hits(
        fps(raw, rng=np.random.default_rng(1)), raw["struct_id"], 4)
    white = _two_cluster_pool()
    whiten_features(white[uqc.FEATURES], white["element"], _tower(art))
    top_w, picks = select_mod._collect_first_hits(
        fps(white, rng=np.random.default_rng(1)), white["struct_id"], 4)
    # raw: the +-10 axis dominates; whitened: the +-3 unit-variance axis is the far one
    assert sum(s < 10 for s in top_raw[1:3]) == 2
    assert sum(s >= 10 for s in top_w[1:3]) == 2
    assert len(top_w) == len(set(top_w)) == 4 and sum(picks.values()) >= 4
    # mini-batch FPS (sharded) still yields every candidate once
    out = list(fps(white, rng=np.random.default_rng(0), fps_max_pool=7))
    assert sorted(out) == list(range(40))


def test_collect_first_hits_ignores_pick_counts():
    sid = np.array([5, 7, 7, 7, 9, 9, 9, 9])
    top, picks = select_mod._collect_first_hits(iter([4, 0, 1, 2, 3, 5, 6, 7]), sid, 2)
    assert top == [9, 5] and picks == {9: 1, 5: 1}


def test_select_hal_dirs_fps_extrap(tmp_path, no_model, caplog):
    art = _write_artifact(tmp_path / "a.npz")
    dirs = _walkers(tmp_path, art)
    out = tmp_path / "sel.pkl.gz"
    assert _select("--hal-dirs", *dirs, "--artifact", art, "-n", 3, "--strategy",
                   "fps-extrap", "--no-element-stratified", "--output", out) == 0
    assert len(pd.read_pickle(out, compression="gzip")) == 3
    assert "whitened" in caplog.text
    caplog.clear()
    assert _select("--hal-dirs", *dirs, "-n", 3, "--strategy", "fps-extrap",
                   "--no-element-stratified", "--output", out) == 0
    assert "raw features" in caplog.text


# -- float32-stored artifacts: float64 working statistics --------------------------


def _fp32_art(tmp_path, seed=0):
    return _write_artifact(tmp_path / f"fp32_{seed}.npz", seed=seed, store_fp32=True, cond=1e5)


def test_additive_update_monotone_on_fp32_artifact(tmp_path):
    """An additive update never raises gamma (beyond 1e-12 relative) on an
    fp32-stored artifact: the model works on float64 statistics."""
    from tensorpotential.uq.cli.strategies import selection_tower

    art = _fp32_art(tmp_path)
    pool = _random_pool(art, n_frames=60, spread=1.0)
    x, el = pool[uqc.FEATURES].astype(np.float64), pool["element"]
    tower = selection_tower(art)
    for key in (uqc.SCATTER, uqc.INV_COV, uqc.CENTROIDS):
        assert tower._artifacts[0][key].dtype == np.float64, key
    g = tower.gamma_from_features(x, el)[0]
    for f in range(0, 60, 6):
        m = pool["struct_id"] == f
        tower.update_one(x[m], el[m], alpha=0.0005, mode="additive")
        g_new = tower.gamma_from_features(x, el)[0]
        assert (g_new <= g * (1 + 1e-12)).all(), np.max(g_new / g)
        g = g_new


@pytest.mark.parametrize("seed", [0, 1])
def test_calm_greedy_lazy_equals_eager_fp32(tmp_path, seed):
    from tensorpotential.uq.cli.strategies import get_strategy, selection_tower

    art = _fp32_art(tmp_path, seed)
    pool = _random_pool(art, seed=seed, spread=1.0)
    greedy = get_strategy("calm-greedy")
    lazy = greedy(pool, gmm=selection_tower(art), n_structures=1000, progress=False)
    eager = greedy(pool, gmm=selection_tower(art), n_structures=1000, lazy=False,
                   progress=False)
    assert len(lazy.frames) > 3
    assert lazy.frames == eager.frames and lazy.alphas == eager.alphas
    np.testing.assert_array_equal(lazy.gamma_at_pick, eager.gamma_at_pick)
    assert lazy.natural == eager.natural


def test_prerequisites_checked_before_predict(tmp_path, monkeypatch):
    """0.6.1-style calls fail before any prediction, naming the strategy."""
    def boom(args):
        raise AssertionError("prediction must not start")

    monkeypatch.setattr(select_mod, "_maybe_run_predict", boom)
    raw = tmp_path / "raw.pkl.gz"
    pd.DataFrame({"ase_atoms": [bulk("Cu")]}).to_pickle(raw, compression="gzip")
    with pytest.raises(SystemExit, match="calm-greedy.*--strategy random-extrap"):
        _select("--dataset", raw, "--model", tmp_path, "-n", 2, "--output",
                tmp_path / "s.pkl.gz")
    # a prediction without --save-features: the error names the actual strategy
    art = _write_artifact(tmp_path / "a.npz")
    pred = tmp_path / "pred.pkl.gz"
    pd.DataFrame({"ase_atoms": [bulk("Cu")], "gamma": [np.array([2.0])]}).to_pickle(
        pred, compression="gzip")
    for strategy in ("calm-greedy", "fps-extrap"):
        with pytest.raises(SystemExit, match=f"--strategy {strategy} needs per-atom"):
            _select("--predicted", pred, "--artifact", art, "-n", 1, "--strategy", strategy,
                    "--no-element-stratified", "--output", tmp_path / "s.pkl.gz")


@pytest.mark.parametrize("fps_max_pool", [None, 5])
def test_fps_extrap_element_aware(fps_max_pool):
    """Two elements whose features coincide: element-aware FPS still starts each
    element and never measures across elements; every candidate is yielded once."""
    from tensorpotential.uq.cli.strategies import get_strategy

    rng = np.random.default_rng(0)
    f = rng.standard_normal((10, D)).astype(np.float32)
    pool = {"struct_id": np.arange(20), "element": np.repeat([0, 1], 10).astype(np.int32),
            "gamma": np.full(20, 2.0, np.float32), uqc.FEATURES: np.vstack([f, f])}
    out = list(get_strategy("fps-extrap")(pool, rng=np.random.default_rng(1),
                                          fps_max_pool=fps_max_pool))
    assert sorted(out) == list(range(20))
    if fps_max_pool is None:
        # both elements are seeded first; the element-1 twins of element-0 picks are
        # not at distance 0, so FPS reaches both elements' far atoms early
        assert {pool["element"][i] for i in out[:2]} == {0, 1}
        firsts = [i for i in out[:6]]
        assert sum(pool["element"][firsts] == 0) >= 2 and sum(pool["element"][firsts] == 1) >= 2


def test_hal_dirs_killed_walkers_and_mixed_artifacts(tmp_path, no_model, caplog):
    art = _write_artifact(tmp_path / "a.npz")
    dirs = _walkers(tmp_path, art)
    write_frames = read(os.path.join(dirs[0], "hal_collected.extxyz"), ":")
    from ase.io import write

    os.makedirs(tmp_path / "killed")  # frames written, stopped before its first sidecar save
    write(os.path.join(tmp_path / "killed", "hal_collected.extxyz"), write_frames[0])
    os.makedirs(tmp_path / "empty")  # collected nothing: no frames file at all
    (tmp_path / "empty" / "hal_settings.json").write_text("{}")  # every walker writes it
    write(os.path.join(dirs[1], "hal_collected.extxyz"), write_frames[0], append=True)
    out = tmp_path / "s.pkl.gz"
    base = ("-n", 2, "--strategy", "random-extrap", "--no-element-stratified", "--output", out)
    assert _select("--hal-dirs", *dirs, tmp_path / "killed", tmp_path / "empty",
                   "--artifact", art, *base) == 0
    assert "killed: no hal_features.npz, skipped" in caplog.text
    assert "empty: no hal_collected.extxyz (the walker collected no frames)" in caplog.text
    assert "frames 4..4 are not covered" in caplog.text
    # different artifacts: refused even without --artifact
    other = _write_artifact(tmp_path / "b.npz", seed=5)
    d_other = _walkers(tmp_path / "o", other, n_frames=(3,))
    with pytest.raises(SystemExit, match="different CALM artifacts"):
        _select("--hal-dirs", dirs[0], *d_other, *base)
    # a sidecar without a hash cannot be checked against --artifact
    nohash = str(tmp_path / "nohash")
    with pytest.warns(UserWarning, match="no artifact hash"):  # in-memory pristine
        make_walker_dir(nohash, bulk("Cu", "fcc", a=3.6, cubic=True), np.zeros(4, int),
                        [np.ones((4, D))], [[2.0] * 4], feature_meta={"artifact_sha256": ""})
    with pytest.raises(SystemExit, match="carry no artifact sha256"):
        _select("--hal-dirs", nohash, "--artifact", art, *base)


def test_empty_selection_to_extxyz_and_corrupt_frames(tmp_path, no_model):
    art = _write_artifact(tmp_path / "a.npz")
    dirs = _walkers(tmp_path, art)
    with pytest.raises(SystemExit, match="no structures selected"):
        _select("--hal-dirs", *dirs, "--artifact", art, "-n", 2, "--gamma-min", 100,
                "--no-element-stratified", "--output", tmp_path / "s.extxyz")
    path = os.path.join(dirs[0], "hal_collected.extxyz")
    text = open(path).read()
    with open(tmp_path / "trunc.extxyz", "w") as fh:
        fh.write("".join(text.splitlines(True)[:-2]))  # last two atom lines lost
    with pytest.warns(UserWarning, match="last frame truncated"):
        assert len(hal_features.frame_offsets(str(tmp_path / "trunc.extxyz"))) == 4
    with open(tmp_path / "bad.extxyz", "w") as fh:
        fh.write(text.replace("\n4\n", "\nfour\n", 1))
    with pytest.raises(ValueError, match="line .* atom count"):
        hal_features.frame_offsets(str(tmp_path / "bad.extxyz"))


def test_calm_greedy_empty_natural_set_adds_no_floor_frames(tmp_path, no_model, capsys):
    """Nothing above --gamma-min: no greedy pick, and the element floors add no frames
    (they would be DFT frames without an extrapolative atom)."""
    art = _write_artifact(tmp_path / "a.npz")
    dirs = _walkers(tmp_path, art)
    out = tmp_path / "sel.pkl.gz"
    assert _select("--hal-dirs", *dirs, "--artifact", art, "-n", 5, "--gamma-min", 100,
                   "--element-stratified", "--output", out) == 0
    assert len(pd.read_pickle(out, compression="gzip")) == 0
    text = capsys.readouterr().out
    assert "natural set size: 0" in text and "element floors: not applied" in text


def test_fp32_artifact_load_save_round_trip_is_bitwise(tmp_path):
    """float64 working stats in memory; saving a float32 artifact gives it back
    bit for bit (same keys, dtypes and values)."""
    from tensorpotential.uq.gmmuq import GMMUQModel

    art = _write_artifact(tmp_path / "a.npz", store_fp32=True, cond=1e5)
    model = GMMUQModel(art)
    assert model._artifacts[0][uqc.SCATTER].dtype == np.float64
    model.save(str(tmp_path / "b.npz"))
    with np.load(art) as a, np.load(tmp_path / "b.npz") as b:
        assert sorted(a.files) == sorted(b.files)
        for k in a.files:
            assert a[k].dtype == b[k].dtype, k
            assert np.array_equal(a[k], b[k]), k
    reloaded = GMMUQModel(str(tmp_path / "b.npz"))
    for e, a in model._artifacts.items():
        for key in (uqc.SCATTER, uqc.INV_COV, uqc.CENTROIDS):
            assert np.array_equal(reloaded._artifacts[e][key], a[key]), (e, key)


def test_from_artifacts_leaves_the_callers_dict_alone(tmp_path):
    from tensorpotential.uq.gmmuq import GMMUQModel

    art = _write_artifact(tmp_path / "a.npz", store_fp32=True)
    arts = GMMUQArtifactBuilder.load(art)
    before = {e: {k: v for k, v in a.items()} for e, a in arts.items()}
    model = GMMUQModel.from_artifacts(arts, extra_data={"interp_thresholds": np.ones((2, 2))})
    assert model._artifacts[0][uqc.SCATTER].dtype == np.float64
    for e, a in arts.items():
        for k, v in a.items():
            assert v is before[e][k], (e, k)  # same objects, float32 untouched
    assert arts[0][uqc.SCATTER].dtype == np.float32


def test_raw_dataset_detected_without_unpickling(tmp_path):
    from tensorpotential.uq.hal import pool_frames

    art = _write_artifact(tmp_path / "a.npz")
    dirs = _walkers(tmp_path, art)
    pool = str(tmp_path / "pool.pkl.gz")
    pool_frames(dirs).to_pickle(pool, compression="gzip")
    raw = str(tmp_path / "raw.pkl.gz")
    pd.DataFrame({"ase_atoms": [bulk("Cu")] * 3}).to_pickle(raw, compression="gzip")
    assert select_mod._may_have_candidate_columns(pool, chunk=64)  # needle across chunks
    assert not select_mod._may_have_candidate_columns(raw)
    plain = str(tmp_path / "pool.pkl")
    pool_frames(dirs).to_pickle(plain)
    assert select_mod._may_have_candidate_columns(plain)


def test_n_optional_for_calm_greedy_only(tmp_path, no_model, capsys):
    art = _write_artifact(tmp_path / "a.npz")
    dirs = _walkers(tmp_path, art, n_frames=(12, 10))
    out = tmp_path / "sel.pkl.gz"
    assert _select("--hal-dirs", *dirs, "--artifact", art, "--no-element-stratified",
                   "--output", out) == 0  # no -n: the whole natural set
    sel = pd.read_pickle(out, compression="gzip")
    text = capsys.readouterr().out
    assert f"natural set size: {len(sel)}" in text and 0 < len(sel) <= 22
    assert _select("--hal-dirs", *dirs, "--artifact", art, "--no-element-stratified",
                   "-n", 2, "--output", out) == 0  # -n: a budget cap
    assert len(pd.read_pickle(out, compression="gzip")) == 2
    assert "-n 2 reached" in capsys.readouterr().out
    for strategy in ("fps-extrap", "random-extrap"):
        with pytest.raises(SystemExit, match=f"--strategy {strategy} needs N"):
            _select("--hal-dirs", *dirs, "--artifact", art, "--strategy", strategy,
                    "--output", out)


def test_floors_off_by_default_for_calm_greedy(tmp_path, no_model, capsys):
    art = _write_artifact(tmp_path / "a.npz")
    dirs = _walkers(tmp_path, art, n_frames=(8,))
    d_al = str(tmp_path / "al")
    make_walker_dir(d_al, bulk("Al", "fcc", a=4.05, cubic=True), np.ones(4, int),
                    [np.zeros((4, D))] * 3, [[0.5] * 4] * 3,
                    feature_meta={"artifact_sha256": hal_features.file_sha256(art)})
    out = tmp_path / "sel.pkl.gz"
    assert _select("--hal-dirs", *dirs, d_al, "--artifact", art, "--output", out) == 0
    sel = pd.read_pickle(out, compression="gzip")
    assert all(set(a.get_chemical_symbols()) == {"Cu"} for a in sel["ase_atoms"])
    assert "element floors" not in capsys.readouterr().out
    # --min-per-element turns them on
    assert _select("--hal-dirs", *dirs, d_al, "--artifact", art, "--min-per-element", 4,
                   "--output", out) == 0
    sel = pd.read_pickle(out, compression="gzip")
    assert any(set(a.get_chemical_symbols()) == {"Al"} for a in sel["ase_atoms"])


def test_floors_clamped_to_what_n_frames_supply(tmp_path, no_model, caplog):
    """random-extrap keeps floors on; with -n 1 a 4-atom Cu frame can hold 4 Cu atoms
    at most, so the D+1 = 5 floor is clamped to 4, with one warning naming it."""
    art = _write_artifact(tmp_path / "a.npz")
    dirs = _walkers(tmp_path, art)
    out = tmp_path / "sel.pkl.gz"
    assert _select("--hal-dirs", *dirs, "--artifact", art, "--strategy", "random-extrap",
                   "-n", 1, "--output", out) == 0
    msgs = [r.getMessage() for r in caplog.records if "Clamped" in r.getMessage()]
    assert len(msgs) == 1 and "Cu(0):4" in msgs[0] and "1 frame(s)" in msgs[0]
    np.testing.assert_array_equal(
        select_mod._top_n_supply(np.array([[4, 0], [1, 3], [2, 2]]), 2), [6, 5])


def test_select_model_dir(tmp_path, no_model, monkeypatch):
    import shutil

    art = _write_artifact(tmp_path / "a.npz")
    dirs = _walkers(tmp_path, art)
    md = tmp_path / "model"
    md.mkdir()
    out = tmp_path / "sel.pkl.gz"
    with pytest.raises(SystemExit, match="no CALM artifact found; tried .*--artifact"):
        _select("--hal-dirs", *dirs, "--model-dir", md, "--output", out)
    shutil.copy(art, md / "gmm_artifacts.npz")  # sidecars only need the artifact
    assert _select("--hal-dirs", *dirs, "--model-dir", md, "-n", 2, "--output", out) == 0
    assert len(pd.read_pickle(out, compression="gzip")) == 2
    # predict path: model.yaml + checkpoint + artifact from the dir, explicit flags win
    for f in ("model.yaml", "checkpoint.index"):
        (md / f).write_text("")
    seen = {}

    def capture(args):
        seen.update(model=args.model, checkpoint=args.checkpoint, artifact=args.artifact)
        raise SystemExit("stop")

    monkeypatch.setattr(select_mod, "_maybe_run_predict", capture)
    raw = tmp_path / "raw.pkl.gz"
    pd.DataFrame({"ase_atoms": [bulk("Cu")]}).to_pickle(raw, compression="gzip")
    with pytest.raises(SystemExit, match="stop"):
        _select("--dataset", raw, "--model-dir", md, "--artifact", art, "--output", out)
    assert seen == {"model": str(md / "model.yaml"), "checkpoint": str(md / "checkpoint"),
                    "artifact": art}
    (md / "model.yaml").unlink()
    with pytest.raises(SystemExit, match="no model.yaml found.*--model"):
        _select("--dataset", raw, "--model-dir", md, "--output", out)


def test_select_model_dir_requires_model_only_when_predicting(tmp_path, no_model, monkeypatch):
    """--model-dir with only the artifact: fine for a --dataset with candidate columns and
    for an explicit SavedModel --model; a raw --dataset with a yaml model needs all."""
    import shutil

    from tensorpotential.uq.hal import pool_frames

    art = _write_artifact(tmp_path / "a.npz")
    md = tmp_path / "model"
    md.mkdir()
    shutil.copy(art, md / "gmm_artifacts.npz")
    pool = tmp_path / "pool.pkl.gz"
    pool_frames(_walkers(tmp_path, art)).to_pickle(pool, compression="gzip")
    out = tmp_path / "sel.pkl.gz"
    assert _select("--dataset", pool, "--model-dir", md, "-n", 2, "--output", out) == 0
    raw = tmp_path / "raw.pkl.gz"
    pd.DataFrame({"ase_atoms": [bulk("Cu")]}).to_pickle(raw, compression="gzip")
    seen = {}

    def capture(args):
        seen.update(model=args.model, checkpoint=args.checkpoint, artifact=args.artifact)
        raise SystemExit("stop")

    monkeypatch.setattr(select_mod, "_maybe_run_predict", capture)
    saved = tmp_path / "saved_model"
    saved.mkdir()
    with pytest.raises(SystemExit, match="stop"):
        _select("--dataset", raw, "--model", saved, "--model-dir", md, "--output", out)
    assert seen == {"model": str(saved), "checkpoint": None,
                    "artifact": str(md / "gmm_artifacts.npz")}
    with pytest.raises(SystemExit, match="no model.yaml found"):
        _select("--dataset", raw, "--model-dir", md, "--output", out)


def test_clamp_warning_names_at_most_the_pool_size(tmp_path, no_model, caplog):
    art = _write_artifact(tmp_path / "a.npz")
    dirs = _walkers(tmp_path, art, n_frames=(1,))  # one 4-atom Cu frame, floor 5
    assert _select("--hal-dirs", *dirs, "--artifact", art, "--strategy", "random-extrap",
                   "-n", 500, "--output", tmp_path / "s.pkl.gz") == 0
    msgs = [r.getMessage() for r in caplog.records if "Clamped" in r.getMessage()]
    assert len(msgs) == 1 and "what 1 frame(s)" in msgs[0]


# -- combined strategies --------------------------------------------------------------


def _parts(strategy, n=None):
    import argparse

    args = argparse.Namespace(strategy=strategy, n_structures=n)
    select_mod._parse_strategy_parts(args)
    return args


def test_strategy_parts_parsing_and_validation():
    a = _parts(["calm-greedy", "fps-extrap:25", "random-all:25"])
    assert a.parts == [("calm-greedy", None), ("fps-extrap", 25), ("random-all", 25)]
    assert _parts(["fps-extrap:5"]).n_structures == 5
    assert _parts(["fps-extrap"], 7).parts == [("fps-extrap", 7)]
    assert _parts(["calm-greedy"]).parts == [("calm-greedy", None)]
    for bad, n, match in ((["calm-greedy", "random-all:5"], 3, "-n applies to a single"),
                          (["calm-greedy", "fps-extrap"], None, "fps-extrap needs N"),
                          (["fps-extrap:5"], 3, "give N once"),
                          (["nope:3"], None, "unknown strategy"),
                          (["random-all:0"], None, ">= 1"),
                          (["random-all:x"], None, "integer"),
                          (["random-all"], None, "random-all needs N")):
        with pytest.raises(SystemExit, match=match):
            _parts(bad, n)


def test_combined_parts_never_overlap_in_order(tmp_path, no_model, capsys):
    art = _write_artifact(tmp_path / "a.npz")
    dirs = _walkers(tmp_path, art, n_frames=(12, 10))
    out = tmp_path / "sel.pkl.gz"
    assert _select("--hal-dirs", *dirs, "--artifact", art, "--strategy", "calm-greedy:3",
                   "fps-extrap:4", "random-all:5", "--no-element-stratified",
                   "--output", out) == 0
    sel = pd.read_pickle(out, compression="gzip")
    assert list(sel["strategy"]) == ["calm-greedy"] * 3 + ["fps-extrap"] * 4 + ["random-all"] * 5
    keys = list(zip(sel["hal_dir"], sel["collection_index"]))
    assert len(set(keys)) == 12  # parts never overlap
    assert list(sel["pick_order"][:3]) == [0, 1, 2] and (sel["pick_order"][3:] == -1).all()
    text = capsys.readouterr().out
    assert "calm-greedy 3/3 | fps-extrap 4/4 | random-all 5/5" in text


def test_random_all_from_hal_dirs_takes_frames_without_candidates(tmp_path, no_model):
    art = _write_artifact(tmp_path / "a.npz")
    d_al = str(tmp_path / "al")
    make_walker_dir(d_al, bulk("Al", "fcc", a=4.05, cubic=True), np.ones(4, int),
                    [np.zeros((4, D))] * 6, [[0.5] * 4] * 6,
                    feature_meta={"artifact_sha256": hal_features.file_sha256(art)})
    out = tmp_path / "sel.pkl.gz"
    assert _select("--hal-dirs", d_al, "--artifact", art, "--strategy", "random-all:3",
                   "--no-element-stratified", "--output", out) == 0
    sel = pd.read_pickle(out, compression="gzip")
    assert len(sel) == 3 and (sel["strategy"] == "random-all").all()
    assert len(set(sel["collection_index"])) == 3
    # random-extrap: no frame has a candidate, nothing to pick
    assert _select("--hal-dirs", d_al, "--artifact", art, "--strategy", "random-extrap:3",
                   "--no-element-stratified", "--output", out) == 0
    assert len(pd.read_pickle(out, compression="gzip")) == 3  # single part: top-up
    assert (pd.read_pickle(out, compression="gzip")["strategy"] == "top-up").all()


def test_combined_floors_on_the_combined_set(tmp_path, no_model, capsys):
    """A mixed call turns the floors on by default (a random part); they are applied
    once, to the combined set: Al frames replace the latest picks."""
    art = _write_artifact(tmp_path / "a.npz")
    dirs = _walkers(tmp_path, art, n_frames=(8,))
    d_al = str(tmp_path / "al")
    make_walker_dir(d_al, bulk("Al", "fcc", a=4.05, cubic=True), np.ones(4, int),
                    [np.zeros((4, D))] * 3, [[0.5] * 4] * 3,
                    feature_meta={"artifact_sha256": hal_features.file_sha256(art)})
    out = tmp_path / "sel.pkl.gz"
    assert _select("--hal-dirs", *dirs, d_al, "--artifact", art, "--strategy",
                   "calm-greedy:2", "random-extrap:2", "--output", out) == 0
    sel = pd.read_pickle(out, compression="gzip")
    assert len(sel) == 4
    al = [set(a.get_chemical_symbols()) == {"Al"} for a in sel["ase_atoms"]]
    assert sum(al) == 2 and (sel["strategy"][al] == "floors").all()
    assert list(sel["strategy"][[not x for x in al]]) == ["calm-greedy"] * 2  # latest replaced
    assert "element floors on the combined set: 2 frame(s) added" in capsys.readouterr().out


@pytest.mark.parametrize("strategy, has_candidates", [
    (["random-extrap:3"], True),
    (["random-extrap:2", "random-extrap:1"], True),
    (["random-extrap:3"], False),
])
def test_random_extrap_floors_use_remaining_budget(tmp_path, no_model, capsys, strategy,
                                                  has_candidates):
    art = _write_artifact(tmp_path / "a.npz")
    meta = {"artifact_sha256": hal_features.file_sha256(art)}
    cu, al = str(tmp_path / "cu"), str(tmp_path / "al")
    make_walker_dir(cu, bulk("Cu", "fcc", a=3.6, cubic=True), np.zeros(4, int),
                    [np.ones((4, D))] * 3,
                    [[2.0 if has_candidates else 0.5] * 4, [0.5] * 4, [0.5] * 4],
                    feature_meta=meta)
    make_walker_dir(al, bulk("Al", "fcc", a=4.05, cubic=True), np.ones(4, int),
                    [np.zeros((4, D))], [[0.5] * 4], feature_meta=meta)
    out = tmp_path / "sel.pkl.gz"
    assert _select("--hal-dirs", cu, al, "--artifact", art, "--strategy", *strategy,
                   "--min-per-element", 4, "--output", out) == 0
    sel = pd.read_pickle(out, compression="gzip")
    assert 2 <= len(sel) <= 3
    counts = Counter(s for a in sel["ase_atoms"] for s in a.get_chemical_symbols())
    assert counts["Cu"] >= 4 and counts["Al"] >= 4
    assert "floors" in sel["strategy"].values
    assert "UNMET" not in capsys.readouterr().out

    # A one-frame budget cannot cover both pure-element frames. Report the deficit.
    assert _select("--hal-dirs", cu, al, "--artifact", art, "--strategy", "random-extrap:1",
                   "--min-per-element", 4, "--output", out) == 0
    summary = capsys.readouterr().out
    assert "element floors on the combined set: unmet" in summary
    assert "element floors on the combined set: met" not in summary


def test_later_calm_greedy_learns_earlier_picks(tmp_path, no_model):
    art = _write_artifact(tmp_path / "a.npz")
    dirs = _walkers(tmp_path, art, n_frames=(12, 10))
    out = tmp_path / "sel.pkl.gz"
    assert _select("--hal-dirs", *dirs, "--artifact", art, "--strategy", "calm-greedy",
                   "--no-element-stratified", "--output", out) == 0
    alone = len(pd.read_pickle(out, compression="gzip"))
    assert _select("--hal-dirs", *dirs, "--artifact", art, "--strategy", "random-extrap:4",
                   "calm-greedy", "--no-element-stratified", "--output", out) == 0
    sel = pd.read_pickle(out, compression="gzip")
    keys = list(zip(sel["hal_dir"], sel["collection_index"]))
    assert len(set(keys)) == len(keys)
    n_greedy = int((sel["strategy"] == "calm-greedy").sum())
    assert n_greedy <= alone  # the seeded tower needs no more frames than alone


def _line_pool(xs, elements=None):
    """One candidate atom per frame, features on a line at ``xs``."""
    f = np.zeros((len(xs), D), np.float32)
    f[:, 0] = xs
    el = np.zeros(len(xs), np.int32) if elements is None else np.asarray(elements, np.int32)
    return {"struct_id": np.arange(len(xs)), "element": el,
            "gamma": np.full(len(xs), 2.0, np.float32), uqc.FEATURES: f}


@pytest.mark.parametrize("name, fps_max_pool", [("fps-extrap", None), ("fps-extrap", 10),
                                                ("fps-all", 10)])
def test_fps_seeded_starts_from_the_seed_atoms(name, fps_max_pool):
    """Seed atoms are never yielded; FPS distances start from them (no random start)."""
    from tensorpotential.uq.cli.strategies import get_strategy

    pool = _line_pool([0.0, 0.05, 0.1, 5.0, 10.0])
    seed = np.array([True, False, False, False, False])
    kw = {} if fps_max_pool is None else {"fps_max_pool": fps_max_pool}
    for s in range(5):
        out = list(get_strategy(name)(pool, rng=np.random.default_rng(s), seed=seed, **kw))
        assert out[:2] == [4, 3] and sorted(out) == [1, 2, 3, 4]


@pytest.mark.parametrize("fps_max_pool", [None, 10])
def test_fps_seed_is_element_aware(fps_max_pool):
    """A seed of element 0 moves only element 0's distances; element 1 still starts
    from a random atom of its own."""
    from tensorpotential.uq.cli.strategies import get_strategy

    pool = _line_pool([0.0, 0.1, 8.0, 0.0, 0.1, 8.0], elements=[0, 0, 0, 1, 1, 1])
    seed = np.array([True, False, False, False, False, False])
    kw = {} if fps_max_pool is None else {"fps_max_pool": fps_max_pool}
    starts1 = set()
    for s in range(12):
        out = list(get_strategy("fps-extrap")(pool, rng=np.random.default_rng(s), seed=seed,
                                              **kw))
        assert sorted(out) == [1, 2, 3, 4, 5]
        assert [i for i in out if i < 3][0] == 2  # farthest from the element-0 seed
        starts1.add([i for i in out if i >= 3][0])
    assert len(starts1) > 1  # element 1: random start, not driven by the element-0 seed


@pytest.mark.parametrize("second", ["fps-extrap", "fps-all"])
def test_fps_part_avoids_frames_near_earlier_picks(second):
    """Frames come in twins; an fps part after another part never takes the twin of an
    earlier pick, and its first pick is the frame farthest from the earlier picks."""
    import argparse

    xs = [0.0, 0.01, 10.0, 10.01, 5.0, 20.0]
    twin = {0: 1, 1: 0, 2: 3, 3: 2}
    for s in range(10):
        pool = _line_pool(xs)
        args = argparse.Namespace(parts=[("fps-extrap", 1), (second, 2)], gamma_min=1.0,
                                  gamma_max=None, fps_max_pool=None if second == "fps-extrap"
                                  else 50)
        counts = np.ones((len(xs), 1), np.int64)
        picked, label, _, _, _ = select_mod._run_parts(args, pool, len(xs), counts, None,
                                                    np.random.default_rng(s), True)
        first, later = picked[0], picked[1:]
        assert len(later) == 2 and first not in later
        assert twin.get(first) not in later
        assert later[0] == int(np.argmax(np.abs(np.asarray(xs) - xs[first])))


def test_select_help_n_names_fps_all_and_has_a_combined_example():
    p = select_mod._build_parser()
    assert "--strategy calm-greedy random-all:25" in p.epilog
    assert "requires --save-features" not in p.epilog
    n_help = next(a.help for a in p._actions if "-n" in a.option_strings)
    assert "fps-all" in n_help and "name:N" in n_help
