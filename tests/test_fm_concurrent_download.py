"""Concurrency and failure-cleanup tests for foundation model, checkpoint, and kokkos downloads.

Regression tests for https://github.com/ICAMS/grace-tensorpotential/issues/44:
concurrent ``grace_fm()`` / ``get_or_download_*`` calls into a shared cache must
not share temporary archives or ``.part`` files, swap models across directories,
or expose half-extracted directories before validation and sidecar stamping.
"""

from concurrent.futures import ThreadPoolExecutor
import os
import tarfile
import threading

import pytest
import requests

from tensorpotential.calculator import foundation_models as fm


def _make_tgz(tmp_path, archive_name, top_folder, files):
    """Build a real .tar.gz archive containing ``top_folder`` and ``files``."""
    src_root = tmp_path / f"src_{archive_name}"
    top_dir = src_root / top_folder
    top_dir.mkdir(parents=True)
    for rel_path, content in files.items():
        target = top_dir / rel_path
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(content if isinstance(content, bytes) else content.encode())
    tgz_path = tmp_path / f"{archive_name}.tar.gz"
    with tarfile.open(tgz_path, "w:gz") as tar:
        tar.add(top_dir, arcname=top_folder)
    return tgz_path.read_bytes()


class _FakeResponse:
    """Minimal streaming requests.Response mock yielding multi-chunk payloads."""

    def __init__(self, payload: bytes, *, barrier=None, fail_after_first=False):
        self._payload = payload
        self._barrier = barrier
        self._fail_after_first = fail_after_first
        self.headers = {
            "Content-Type": "application/octet-stream",
            "Content-Length": str(len(payload)),
        }

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return False

    def raise_for_status(self):
        return None

    def iter_content(self, chunk_size=1 << 20):
        split = max(1, len(self._payload) // 2)
        yield self._payload[:split]
        if self._fail_after_first:
            raise requests.ConnectionError("simulated mid-stream failure")
        if self._barrier is not None:
            self._barrier.wait(timeout=10)
        if split < len(self._payload):
            yield self._payload[split:]


def _install_fake_requests_get(monkeypatch, url_to_bytes, *, barrier=None, fail_urls=()):
    def fake_get(url, stream=True, timeout=None, allow_redirects=True):
        return _FakeResponse(
            url_to_bytes[url],
            barrier=barrier,
            fail_after_first=url in fail_urls,
        )

    monkeypatch.setattr(fm.requests, "get", fake_get)


def _run_concurrent(fn, items):
    with ThreadPoolExecutor(max_workers=len(items)) as pool:
        futures = [pool.submit(fn, item) for item in items]
        return [fut.result(timeout=15) for fut in futures]


@pytest.mark.parametrize("shared_top_dir", [False, True])
def test_concurrent_different_models_do_not_swap_or_collide(tmp_path, monkeypatch, shared_top_dir):
    """Concurrent downloads of distinct models must never cross-contaminate cache dirs."""
    cache = tmp_path / "cache"
    cache.mkdir()
    monkeypatch.setattr(fm, "FOUNDATION_CACHE_DIR", str(cache))

    names = [f"_concurrent_model_{i}" for i in range(8)]
    url_to_bytes = {}
    for name in names:
        url = f"https://huggingface.co/x/y/resolve/model-v5-uq-v6/models/{name}-model.tar.gz"
        top = "exported_model" if shared_top_dir else name
        url_to_bytes[url] = _make_tgz(
            tmp_path,
            name,
            top,
            {
                "saved_model.pb": f"pb-{name}",
                "variables/variables.index": f"idx-{name}",
                "variables/variables.data-00000-of-00001": f"weights-{name}",
            },
        )
        monkeypatch.setitem(fm.MODELS_METADATA, name, {fm.MODEL_URL_KEY: url})

    barrier = threading.Barrier(len(names))
    _install_fake_requests_get(monkeypatch, url_to_bytes, barrier=barrier)

    paths = _run_concurrent(fm.get_or_download_model, names)

    assert paths == [str(cache / n) for n in names]
    assert sorted(os.listdir(cache)) == sorted(names)
    for name in names:
        model_dir = cache / name
        assert (model_dir / "saved_model.pb").read_text() == f"pb-{name}"
        assert (model_dir / "variables" / "variables.data-00000-of-00001").read_text() == f"weights-{name}"
        sidecar = fm.read_origin_sidecar(str(model_dir))
        assert sidecar is not None
        assert sidecar["model_name"] == name
        assert sidecar["url"] == fm.MODELS_METADATA[name][fm.MODEL_URL_KEY]


def test_concurrent_same_model_cold_cache(tmp_path, monkeypatch):
    """Concurrent downloads of the same uncached model must converge cleanly with no leftovers."""
    cache = tmp_path / "cache"
    cache.mkdir()
    monkeypatch.setattr(fm, "FOUNDATION_CACHE_DIR", str(cache))

    name = "_concurrent_same_model"
    url = f"https://huggingface.co/x/y/resolve/model-v5-uq-v6/models/{name}-model.tar.gz"
    payload = _make_tgz(
        tmp_path,
        name,
        "exported_model",
        {
            "saved_model.pb": "pb-same",
            "variables/variables.data-00000-of-00001": "weights-same",
        },
    )
    monkeypatch.setitem(fm.MODELS_METADATA, name, {fm.MODEL_URL_KEY: url})

    n_workers = 6
    barrier = threading.Barrier(n_workers)
    _install_fake_requests_get(monkeypatch, {url: payload}, barrier=barrier)

    paths = _run_concurrent(fm.get_or_download_model, [name] * n_workers)

    assert paths == [str(cache / name)] * n_workers
    assert os.listdir(cache) == [name]
    model_dir = cache / name
    assert sorted(os.listdir(model_dir)) == [".grace_origin.json", "saved_model.pb", "variables"]
    assert (model_dir / "saved_model.pb").read_text() == "pb-same"
    assert (model_dir / "variables" / "variables.data-00000-of-00001").read_text() == "weights-same"
    assert fm.read_origin_sidecar(str(model_dir))["url"] == url


def test_concurrent_same_model_with_existing_kokkos(tmp_path, monkeypatch):
    """Concurrent model downloads into a kokkos-only dir must preserve kokkos.npz."""
    cache = tmp_path / "cache"
    name = "_concurrent_kokkos_merge"
    model_dir = cache / name
    model_dir.mkdir(parents=True)
    (model_dir / "kokkos.npz").write_text("kokkos-payload")
    monkeypatch.setattr(fm, "FOUNDATION_CACHE_DIR", str(cache))

    url = f"https://huggingface.co/x/y/resolve/model-v5-uq-v6/models/{name}-model.tar.gz"
    payload = _make_tgz(
        tmp_path,
        name,
        "different_top_dir",
        {
            "saved_model.pb": "pb-merged",
            "variables/variables.index": "idx-merged",
        },
    )
    monkeypatch.setitem(fm.MODELS_METADATA, name, {fm.MODEL_URL_KEY: url})

    n_workers = 4
    barrier = threading.Barrier(n_workers)
    _install_fake_requests_get(monkeypatch, {url: payload}, barrier=barrier)

    paths = _run_concurrent(fm.get_or_download_model, [name] * n_workers)

    assert paths == [str(model_dir)] * n_workers
    assert os.listdir(cache) == [name]
    assert sorted(os.listdir(model_dir)) == [
        ".grace_origin.json",
        "kokkos.npz",
        "saved_model.pb",
        "variables",
    ]
    assert (model_dir / "kokkos.npz").read_text() == "kokkos-payload"
    assert (model_dir / "saved_model.pb").read_text() == "pb-merged"
    assert (model_dir / "variables" / "variables.index").read_text() == "idx-merged"
    assert fm.read_origin_sidecar(str(model_dir))["url"] == url


def test_concurrent_model_and_kokkos_same_dir(tmp_path, monkeypatch):
    """Concurrent get_or_download_model and get_or_download_kokkos on cold cache must both succeed."""
    cache = tmp_path / "cache"
    cache.mkdir()
    monkeypatch.setattr(fm, "FOUNDATION_CACHE_DIR", str(cache))

    name = "_concurrent_model_plus_kokkos"
    model_url = f"https://huggingface.co/x/y/resolve/model-v5-uq-v6/models/{name}-model.tar.gz"
    kokkos_url = f"https://huggingface.co/x/y/resolve/kk-v3-uq-v6/kokkos/{name}-kokkos.npz"
    model_bytes = _make_tgz(
        tmp_path,
        name,
        name,
        {
            "saved_model.pb": "pb-both",
            "variables/variables.index": "idx-both",
        },
    )
    kokkos_bytes = b"NPZ-KOKKOS-BINARY-DATA" * 100
    monkeypatch.setitem(
        fm.MODELS_METADATA,
        name,
        {fm.MODEL_URL_KEY: model_url, fm.KOKKOS_URL_KEY: kokkos_url},
    )

    barrier = threading.Barrier(2)
    _install_fake_requests_get(
        monkeypatch,
        {model_url: model_bytes, kokkos_url: kokkos_bytes},
        barrier=barrier,
    )

    def _run_one(kind):
        if kind == "model":
            return fm.get_or_download_model(name)
        return fm.get_or_download_kokkos(name)

    model_path, kokkos_path = _run_concurrent(_run_one, ["model", "kokkos"])

    model_dir = cache / name
    assert model_path == str(model_dir)
    assert kokkos_path == str(model_dir / "kokkos.npz")
    assert os.listdir(cache) == [name]
    assert sorted(os.listdir(model_dir)) == [
        ".grace_origin.json",
        "kokkos.npz",
        "saved_model.pb",
        "variables",
    ]
    assert (model_dir / "kokkos.npz").read_bytes() == kokkos_bytes
    assert (model_dir / "saved_model.pb").read_text() == "pb-both"
    assert fm.read_origin_sidecar(str(model_dir))["url"] == model_url


def test_concurrent_same_kokkos_download(tmp_path, monkeypatch):
    """Concurrent get_or_download_kokkos calls must not share .part files or truncate output."""
    cache = tmp_path / "cache"
    cache.mkdir()
    monkeypatch.setattr(fm, "FOUNDATION_CACHE_DIR", str(cache))

    name = "_concurrent_same_kokkos"
    kokkos_url = f"https://huggingface.co/x/y/resolve/kk-v3-uq-v6/kokkos/{name}-kokkos.npz"
    kokkos_bytes = b"0123456789ABCDEF" * 512
    monkeypatch.setitem(fm.MODELS_METADATA, name, {fm.KOKKOS_URL_KEY: kokkos_url})

    n_workers = 4
    barrier = threading.Barrier(n_workers)
    _install_fake_requests_get(monkeypatch, {kokkos_url: kokkos_bytes}, barrier=barrier)

    paths = _run_concurrent(fm.get_or_download_kokkos, [name] * n_workers)

    expected = str(cache / name / "kokkos.npz")
    assert paths == [expected] * n_workers
    assert (cache / name / "kokkos.npz").read_bytes() == kokkos_bytes
    assert os.listdir(cache / name) == ["kokkos.npz"]


def test_concurrent_different_checkpoints(tmp_path, monkeypatch):
    """Concurrent downloads of distinct checkpoints must keep each checkpoint isolated."""
    ckpt_cache = tmp_path / "checkpoints"
    ckpt_cache.mkdir()
    monkeypatch.setattr(fm, "FOUNDATION_CHECKPOINTS_CACHE_DIR", str(ckpt_cache))

    names = [f"_concurrent_ckpt_{i}" for i in range(6)]
    url_to_bytes = {}
    for name in names:
        url = f"https://huggingface.co/x/y/resolve/checkpoint-v3-uq-v6/checkpoints/{name}-checkpoint.tar.gz"
        url_to_bytes[url] = _make_tgz(
            tmp_path,
            f"ckpt_{name}",
            "checkpoint",
            {
                "model.yaml": f"model: {name}\n",
                "checkpoint.data-00000-of-00001": f"ckpt-weights-{name}",
            },
        )
        monkeypatch.setitem(fm.MODELS_METADATA, name, {fm.CHECKPOINT_URL_KEY: url})

    barrier = threading.Barrier(len(names))
    _install_fake_requests_get(monkeypatch, url_to_bytes, barrier=barrier)

    paths = _run_concurrent(fm.get_or_download_checkpoint, names)

    assert paths == [str(ckpt_cache / n) for n in names]
    assert sorted(os.listdir(ckpt_cache)) == sorted(names)
    for name in names:
        ckpt_dir = ckpt_cache / name
        assert (ckpt_dir / "model.yaml").read_text() == f"model: {name}\n"
        assert (ckpt_dir / "checkpoint.data-00000-of-00001").read_text() == f"ckpt-weights-{name}"
        sidecar = fm.read_origin_sidecar(str(ckpt_dir))
        assert sidecar is not None
        assert sidecar["payload"] == "checkpoint"
        assert sidecar["model_name"] == name
        assert sidecar["url"] == fm.MODELS_METADATA[name][fm.CHECKPOINT_URL_KEY]


def test_concurrent_same_checkpoint_cold_cache(tmp_path, monkeypatch):
    """Concurrent downloads of the same uncached checkpoint must converge cleanly."""
    ckpt_cache = tmp_path / "checkpoints"
    ckpt_cache.mkdir()
    monkeypatch.setattr(fm, "FOUNDATION_CHECKPOINTS_CACHE_DIR", str(ckpt_cache))

    name = "_concurrent_same_ckpt"
    url = f"https://huggingface.co/x/y/resolve/checkpoint-v3-uq-v6/checkpoints/{name}-checkpoint.tar.gz"
    payload = _make_tgz(
        tmp_path,
        name,
        "checkpoint",
        {
            "model.yaml": "model: same\n",
            "checkpoint.index": "idx-same",
        },
    )
    monkeypatch.setitem(fm.MODELS_METADATA, name, {fm.CHECKPOINT_URL_KEY: url})

    n_workers = 4
    barrier = threading.Barrier(n_workers)
    _install_fake_requests_get(monkeypatch, {url: payload}, barrier=barrier)

    paths = _run_concurrent(fm.get_or_download_checkpoint, [name] * n_workers)

    assert paths == [str(ckpt_cache / name)] * n_workers
    assert os.listdir(ckpt_cache) == [name]
    ckpt_dir = ckpt_cache / name
    assert sorted(os.listdir(ckpt_dir)) == [".grace_origin.json", "checkpoint.index", "model.yaml"]
    assert (ckpt_dir / "model.yaml").read_text() == "model: same\n"
    assert fm.read_origin_sidecar(str(ckpt_dir))["url"] == url


def test_failed_validation_leaves_no_cache_dir_or_staging(tmp_path, monkeypatch):
    """Invalid archives fail in staging without creating target dirs or leaving staging debris."""
    cache = tmp_path / "cache"
    ckpt_cache = tmp_path / "checkpoints"
    cache.mkdir()
    ckpt_cache.mkdir()
    monkeypatch.setattr(fm, "FOUNDATION_CACHE_DIR", str(cache))
    monkeypatch.setattr(fm, "FOUNDATION_CHECKPOINTS_CACHE_DIR", str(ckpt_cache))

    bad_model = "_bad_model_archive"
    bad_ckpt = "_bad_ckpt_archive"
    model_url = "https://huggingface.co/x/y/resolve/model-v5-uq-v6/models/bad.tar.gz"
    ckpt_url = "https://huggingface.co/x/y/resolve/checkpoint-v3-uq-v6/checkpoints/bad.tar.gz"
    model_bytes = _make_tgz(tmp_path, "bad_m", bad_model, {"variables/variables.index": "idx"})
    ckpt_bytes = _make_tgz(tmp_path, "bad_c", bad_ckpt, {"checkpoint.index": "idx"})

    monkeypatch.setitem(fm.MODELS_METADATA, bad_model, {fm.MODEL_URL_KEY: model_url})
    monkeypatch.setitem(fm.MODELS_METADATA, bad_ckpt, {fm.CHECKPOINT_URL_KEY: ckpt_url})
    _install_fake_requests_get(monkeypatch, {model_url: model_bytes, ckpt_url: ckpt_bytes})

    with pytest.raises(AssertionError, match="No SavedModel"):
        fm.get_or_download_model(bad_model)
    assert os.listdir(cache) == []

    with pytest.raises(AssertionError, match="No model.yaml"):
        fm.get_or_download_checkpoint(bad_ckpt)
    assert os.listdir(ckpt_cache) == []


def test_failed_download_cleans_up_part_and_staging(tmp_path, monkeypatch):
    """Interrupted downloads leave zero .part files or staging directories behind."""
    cache = tmp_path / "cache"
    cache.mkdir()
    monkeypatch.setattr(fm, "FOUNDATION_CACHE_DIR", str(cache))

    name = "_failing_download"
    model_url = "https://huggingface.co/x/y/resolve/model-v5-uq-v6/models/fail.tar.gz"
    kokkos_url = "https://huggingface.co/x/y/resolve/kk-v3-uq-v6/kokkos/fail-kokkos.npz"
    monkeypatch.setitem(
        fm.MODELS_METADATA,
        name,
        {fm.MODEL_URL_KEY: model_url, fm.KOKKOS_URL_KEY: kokkos_url},
    )
    _install_fake_requests_get(
        monkeypatch,
        {model_url: b"partial-archive", kokkos_url: b"partial-kokkos"},
        fail_urls={model_url, kokkos_url},
    )

    with pytest.raises(RuntimeError, match="failed after"):
        fm.get_or_download_model(name)
    assert os.listdir(cache) == []

    with pytest.raises(RuntimeError, match="failed after"):
        fm.get_or_download_kokkos(name)
    assert os.listdir(cache / name) == []


def test_sidecar_and_payload_present_before_publish(tmp_path, monkeypatch):
    """Payload validation and .grace_origin.json stamping finish inside staging before publish."""
    cache = tmp_path / "cache"
    cache.mkdir()
    monkeypatch.setattr(fm, "FOUNDATION_CACHE_DIR", str(cache))

    name = "_staged_before_publish"
    url = f"https://huggingface.co/x/y/resolve/model-v5-uq-v6/models/{name}-model.tar.gz"
    payload = _make_tgz(
        tmp_path,
        name,
        name,
        {"saved_model.pb": "pb", "variables/variables.index": "idx"},
    )
    monkeypatch.setitem(fm.MODELS_METADATA, name, {fm.MODEL_URL_KEY: url})
    _install_fake_requests_get(monkeypatch, {url: payload})

    observations = []
    real_publish = fm._publish_extracted_dir

    def spy_publish(src_dir, dst_dir, *, sentinel=None):
        observations.append(
            {
                "dst_exists_before": os.path.exists(dst_dir),
                "src_entries": sorted(os.listdir(src_dir)),
                "is_saved_model": fm.is_saved_model(src_dir),
                "sidecar": fm.read_origin_sidecar(src_dir),
            }
        )
        return real_publish(src_dir, dst_dir, sentinel=sentinel)

    monkeypatch.setattr(fm, "_publish_extracted_dir", spy_publish)

    fm.get_or_download_model(name)

    assert len(observations) == 1
    obs = observations[0]
    assert obs["dst_exists_before"] is False
    assert obs["is_saved_model"] is True
    assert obs["src_entries"] == [".grace_origin.json", "saved_model.pb", "variables"]
    assert obs["sidecar"]["model_name"] == name
    assert obs["sidecar"]["url"] == url
