"""Candidate-atom feature sidecar of a HAL walker (``hal_features.npz``).

At each collection the walker keeps the pristine-model features of the atoms with
``gamma_pristine >= feature_gamma_min`` (the candidates for DFT selection). Row ``k`` of
the per-collection arrays belongs to frame ``k`` of ``hal_collected.extxyz``
(``frame.info["collection_index"] == k``), so ``grace_uq select --hal-dirs`` can pick
frames from the sidecars alone and read only the selected frames' Atoms.

Arrays (candidate rows, M of them): ``collection`` int32, ``atom`` int32 (index in the
frame), ``Z`` int16, ``element_index`` int16 (model element index), ``gamma_pristine``
float32, ``features`` float32 [M, D]. Per collection: ``collection_step`` int64,
``n_atoms`` int32, ``elem_counts`` int32 [n_collections, n_species] over ``species_Z``.
Metadata (0-d): ``feature_gamma_min``, ``feature_dim``, ``artifact_sha256``, ``model_yaml``,
``walker_id``.
"""

from __future__ import annotations

import hashlib
import io
import math
import os
import warnings

import numpy as np
from ase.io import read as ase_read

from tensorpotential.uq.artifact_builder import atomic_savez

FEATURES_FILE = "hal_features.npz"


def check_feature_gamma_min(value):
    """``None`` (no sidecar) or a finite number >= 0."""
    if value is None:
        return
    if isinstance(value, bool) or not isinstance(value, (int, float, np.integer, np.floating)):
        raise ValueError(f"feature_gamma_min must be a number >= 0 or None, got {value!r}")
    if not (math.isfinite(float(value)) and float(value) >= 0):
        raise ValueError(f"feature_gamma_min must be a finite number >= 0 or None, got {value!r}")


def file_sha256(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


class CandidateFeatures:
    """Accumulates candidate rows in memory; :meth:`save` writes the sidecar atomically."""

    def __init__(self, path, feature_gamma_min, numbers, *, artifact_sha256="",
                 model_yaml="", walker_id=0):
        check_feature_gamma_min(feature_gamma_min)
        self.path = path
        self.feature_gamma_min = float(feature_gamma_min)
        self.species_Z = np.unique(np.asarray(numbers)).astype(np.int16)
        self.meta = dict(artifact_sha256=str(artifact_sha256 or ""),
                         model_yaml=str(model_yaml or ""), walker_id=int(walker_id))
        self._rows = {k: [] for k in ("collection", "atom", "Z", "element_index",
                                      "gamma_pristine", "features")}
        self._per_coll = {k: [] for k in ("collection_step", "n_atoms", "elem_counts")}
        self.feature_dim = -1

    @property
    def n_collections(self) -> int:
        return len(self._per_coll["n_atoms"])

    def add(self, collection_index, step, numbers, element_indices, features, gamma_pristine):
        if collection_index != self.n_collections:
            raise ValueError(f"collection {collection_index} out of order "
                             f"(expected {self.n_collections})")
        features = np.asarray(features)
        g = np.asarray(gamma_pristine, dtype=np.float64)
        numbers = np.asarray(numbers)
        self.feature_dim = int(features.shape[1])
        sel = np.flatnonzero(g >= self.feature_gamma_min)
        r = self._rows
        r["collection"].append(np.full(len(sel), collection_index, dtype=np.int32))
        r["atom"].append(sel.astype(np.int32))
        r["Z"].append(numbers[sel].astype(np.int16))
        r["element_index"].append(np.asarray(element_indices)[sel].astype(np.int16))
        r["gamma_pristine"].append(g[sel].astype(np.float32))
        r["features"].append(features[sel].astype(np.float32))
        pc = self._per_coll
        pc["collection_step"].append(int(step))
        pc["n_atoms"].append(len(numbers))
        pc["elem_counts"].append(
            np.array([np.count_nonzero(numbers == z) for z in self.species_Z], dtype=np.int32))

    def save(self):
        r, pc = self._rows, self._per_coll
        d = max(self.feature_dim, 0)
        out = {
            "collection": np.concatenate(r["collection"] or [np.zeros(0, np.int32)]),
            "atom": np.concatenate(r["atom"] or [np.zeros(0, np.int32)]),
            "Z": np.concatenate(r["Z"] or [np.zeros(0, np.int16)]),
            "element_index": np.concatenate(r["element_index"] or [np.zeros(0, np.int16)]),
            "gamma_pristine": np.concatenate(r["gamma_pristine"] or [np.zeros(0, np.float32)]),
            "features": (np.concatenate(r["features"]) if r["features"]
                         else np.zeros((0, d), np.float32)),
            "collection_step": np.asarray(pc["collection_step"], dtype=np.int64),
            "n_atoms": np.asarray(pc["n_atoms"], dtype=np.int32),
            "species_Z": self.species_Z,
            "elem_counts": (np.stack(pc["elem_counts"]) if pc["elem_counts"]
                            else np.zeros((0, len(self.species_Z)), np.int32)),
            "feature_gamma_min": np.float64(self.feature_gamma_min),
            "feature_dim": np.int64(self.feature_dim),
            "artifact_sha256": np.str_(self.meta["artifact_sha256"]),
            "model_yaml": np.str_(self.meta["model_yaml"]),
            "walker_id": np.int64(self.meta["walker_id"]),
        }
        atomic_savez(self.path, **out)


_META = ("feature_gamma_min", "feature_dim", "artifact_sha256", "model_yaml", "walker_id")


def read_features(path: str, keys=None) -> dict:
    """Sidecar of a walker dir (or the ``.npz`` path itself) as a dict; 0-d metadata as
    Python scalars. ``keys`` loads only those arrays (plus the metadata)."""
    if os.path.isdir(path):
        path = os.path.join(path, FEATURES_FILE)
    with np.load(path, allow_pickle=False) as z:
        names = z.files if keys is None else [k for k in z.files if k in set(keys) | set(_META)]
        out = {k: z[k] for k in names}
    for k in _META:
        if k in out:
            out[k] = out[k].item()
    return out


def frame_offsets(path: str) -> np.ndarray:
    """Byte offset of every frame of an extxyz file (one pass, no parsing beyond the
    atom-count lines).

    A line where an atom count belongs that is not an integer raises ValueError
    (corrupt or not extxyz). A truncated last frame (walker killed while writing it)
    is dropped with a warning.
    """
    offsets, pos, skip = [], 0, 0
    with open(path, "rb") as fh:
        for lineno, line in enumerate(fh, 1):
            if skip:
                skip -= 1
            elif line.strip():
                try:
                    n = int(line)
                except ValueError:
                    raise ValueError(
                        f"{path}: line {lineno} should hold the atom count of a frame, "
                        f"got {line[:60]!r}: corrupt file or not extxyz"
                    ) from None
                offsets.append(pos)
                skip = n + 1  # comment line + atom lines
            pos += len(line)
    if skip and offsets:
        warnings.warn(f"{path}: last frame truncated ({skip} line(s) missing), ignored",
                      stacklevel=2)
        offsets.pop()
    return np.asarray(offsets, dtype=np.int64)


def read_frames(path: str, indices, offsets=None) -> list:
    """Frames ``indices`` (0-based, any order) of an extxyz file; parses only those.
    ``offsets``: :func:`frame_offsets` of the file, if already computed."""
    indices = [int(i) for i in indices]
    if not indices:
        return []
    if offsets is None:
        offsets = frame_offsets(path)
    bad = [i for i in indices if not 0 <= i < len(offsets)]
    if bad:
        raise IndexError(f"{path}: frames {bad} out of range (file has {len(offsets)})")
    out = []
    with open(path, "rb") as fh:
        for i in indices:
            fh.seek(offsets[i])
            n = int(fh.readline())
            text = b"".join([str(n).encode() + b"\n"] + [fh.readline() for _ in range(n + 1)])
            out.append(ase_read(io.StringIO(text.decode()), format="extxyz"))
    return out
