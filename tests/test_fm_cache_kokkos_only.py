"""A cache dir that holds only side-loaded files (``kokkos.npz`` fetched by
``get_or_download_kokkos`` or by grace-native) is not a downloaded model.

The model getter used to short-circuit on ``os.path.isdir``: once ``kokkos.npz``
had created ``<cache>/<MODEL>/``, ``grace_fm(MODEL)`` never downloaded the
SavedModel and TPCalculator failed to load the directory.
"""

import argparse
import os
import shutil
import tarfile

import pytest

from tensorpotential.calculator import foundation_models as fm

URL = "https://huggingface.co/x/y/resolve/model-v5-uq-v6/models/M-model.tar.gz"
NAME = "_kokkos_only_model"


def _archive(tmp_path, top):
    """A model .tar.gz with one top-level folder ``top`` holding a SavedModel."""
    src = tmp_path / "src" / top
    (src / "variables").mkdir(parents=True)
    (src / "saved_model.pb").write_text("pb")
    (src / "variables" / "variables.index").write_text("idx")
    tgz = tmp_path / "model.tar.gz"
    with tarfile.open(tgz, "w:gz") as tar:
        tar.add(src, arcname=top)
    return tgz


@pytest.fixture
def kokkos_only(tmp_path, monkeypatch):
    cache = tmp_path / "cache"
    d = cache / NAME
    d.mkdir(parents=True)
    (d / "kokkos.npz").write_text("side-loaded")
    monkeypatch.setattr(fm, "FOUNDATION_CACHE_DIR", str(cache))
    monkeypatch.setitem(fm.MODELS_METADATA, NAME, {fm.MODEL_URL_KEY: URL})
    return d


@pytest.mark.parametrize("top", [NAME, "M-model-v5"])
def test_kokkos_only_dir_gets_the_saved_model(tmp_path, kokkos_only, monkeypatch, top):
    """Archive top folder named as the cache dir (extracted in place) or not (moved in)."""
    tgz = _archive(tmp_path, top)
    calls = []

    def fake_stream(url, dest, desc=None):
        calls.append(url)
        shutil.copy(tgz, dest)
        return dest

    monkeypatch.setattr(fm, "_stream_download", fake_stream)
    path = fm.get_or_download_model(NAME)

    assert calls == [URL], "the SavedModel was not downloaded"
    assert path == str(kokkos_only)
    assert (kokkos_only / "saved_model.pb").read_text() == "pb"
    assert (kokkos_only / "variables" / "variables.index").read_text() == "idx"
    assert (kokkos_only / "kokkos.npz").read_text() == "side-loaded"
    assert fm.read_origin_sidecar(str(kokkos_only))["url"] == URL
    assert sorted(os.listdir(kokkos_only.parent)) == [NAME], "leftovers next to the cache dir"


def test_complete_cache_is_not_downloaded_again(kokkos_only, monkeypatch):
    (kokkos_only / "saved_model.pb").write_text("pb")
    monkeypatch.setattr(
        fm, "_stream_download", lambda *a, **k: pytest.fail("cached model downloaded again")
    )
    assert fm.get_or_download_model(NAME) == str(kokkos_only)


def test_cli_reports_kokkos_only_dir_as_not_downloaded(kokkos_only):
    from tensorpotential.scripts.grace_models import _HEADERS, _download_hints, _row_cells

    md = fm.MODELS_METADATA[NAME]
    hints = _download_hints(NAME, md, str(kokkos_only), str(kokkos_only) + "-ckpt")
    assert ("model", f"grace_models download {NAME}") in hints
    model_cell = _row_cells(NAME, md, ascii_mode=True)[_HEADERS.index("MODEL")]
    assert model_cell == ".", f"MODEL column {model_cell!r}: shown as downloaded"


def test_update_skips_kokkos_only_dir(kokkos_only, tmp_path, monkeypatch, capsys):
    """Nothing to update: the model was never downloaded (and kokkos.npz must survive)."""
    from tensorpotential.scripts.grace_models import update_models

    monkeypatch.setattr(fm, "FOUNDATION_CHECKPOINTS_CACHE_DIR", str(tmp_path / "ckpt"))
    fetched = []    # update_models catches whatever fetch raises: record instead of pytest.fail
    monkeypatch.setattr(fm, "get_or_download_model", fetched.append)
    update_models(argparse.Namespace(model_name=[NAME], all=False, force=False, dry_run=False))
    assert fetched == [], "update fetched a model that was never downloaded"
    assert (kokkos_only / "kokkos.npz").read_text() == "side-loaded"


def test_checkpoint_download_needs_no_saved_model(tmp_path, monkeypatch):
    """A checkpoint archive holds model.yaml + checkpoint files, no SavedModel.

    0.6.2 checked every download (models and checkpoints) for saved_model.pb, so
    downloading any foundation checkpoint failed: fine-tuning from a foundation
    model on a machine without a cached checkpoint stopped with
    "AssertionError: No SavedModel in .../checkpoints/<MODEL>".
    """
    ckpt_url = "https://huggingface.co/x/y/resolve/checkpoint-v3/checkpoints/M-checkpoint.tar.gz"
    src = tmp_path / "src" / NAME
    src.mkdir(parents=True)
    (src / "model.yaml").write_text("yaml")
    (src / "checkpoint.index").write_text("idx")
    tgz = tmp_path / "checkpoint.tar.gz"
    with tarfile.open(tgz, "w:gz") as tar:
        tar.add(src, arcname=NAME)

    ckpt_cache = tmp_path / "checkpoints"
    monkeypatch.setattr(fm, "FOUNDATION_CHECKPOINTS_CACHE_DIR", str(ckpt_cache))
    monkeypatch.setitem(fm.MODELS_METADATA, NAME, {fm.CHECKPOINT_URL_KEY: ckpt_url})
    monkeypatch.setattr(
        fm, "_stream_download", lambda url, dest, desc=None: shutil.copy(tgz, dest) and dest
    )

    path = fm.get_or_download_checkpoint(NAME)

    assert path == str(ckpt_cache / NAME)
    assert (ckpt_cache / NAME / "model.yaml").read_text() == "yaml"
    assert fm.read_origin_sidecar(path)["url"] == ckpt_url
