"""Campaign helpers around ``run_hal``: pooling collected frames and scanning tau."""

from __future__ import annotations

import os
import warnings

import numpy as np
import pandas as pd



_WALKER_MARKERS = ("hal_settings.json", "hal_collected.extxyz", "hal_features.npz")


def find_walker_dirs(paths) -> list[str]:
    """Walker directories in ``paths``, each path searched at any depth.

    A walker directory holds one of ``hal_settings.json`` (written by every walker,
    also one that collected nothing), ``hal_collected.extxyz`` or ``hal_features.npz``;
    the search does not descend into it. Absolute paths, sorted per given path, without
    duplicates. Raises ValueError for a path that contains no walker directory.
    """
    found = []
    for path in paths:
        path = os.path.abspath(path)
        hits = []
        for root, subdirs, files in os.walk(path):
            if any(m in files for m in _WALKER_MARKERS):
                hits.append(root)
                subdirs[:] = []
            else:
                subdirs.sort()
        if not hits:
            raise ValueError(f"{path}: no HAL walker directory (with hal_settings.json, "
                             "hal_collected.extxyz or hal_features.npz) in it or below it")
        found += [h for h in sorted(hits) if h not in found]
    return found


def pool_frames(dirs, gamma_pristine_min=None) -> pd.DataFrame:
    """Collected frames of several walker directories as one DataFrame.

    Columns: ``ase_atoms``, every ``frame.info`` field, and ``hal_dir``. Frames carry no
    calculator (model predictions stay in ``grace_*`` keys), so the result can go straight
    to ``grace_uq select --dataset`` as a pkl.gz. ``gamma_pristine_min`` keeps only frames
    with ``gamma_max_pristine >= gamma_pristine_min``. Directories without
    ``hal_collected.extxyz`` are skipped with a warning.

    Walkers with a candidate-feature sidecar (``hal_features.npz``) add per-row
    ``cand_atom_index`` (int32 [m]), ``cand_features`` (float32 [m, D]), ``cand_gamma``
    (pristine gamma, float32 [m]) and the sidecar's ``cand_feature_gamma_min`` and
    ``cand_artifact_sha256``: ``grace_uq select --dataset POOL`` then selects without a
    model prediction. Frames the sidecar does not cover yet (walker killed between two
    sidecar writes) get None.
    """
    from ase.io import read

    from tensorpotential.uq.hal.features import FEATURES_FILE, read_features

    rows = []
    for d in dirs:
        path = os.path.join(d, "hal_collected.extxyz")
        if not os.path.exists(path):
            warnings.warn(f"{d}: no hal_collected.extxyz, skipped", stacklevel=2)
            continue
        sc_path = os.path.join(d, FEATURES_FILE)
        sc = read_features(sc_path) if os.path.exists(sc_path) else None
        if sc is not None:
            bounds = np.searchsorted(sc["collection"], np.arange(len(sc["n_atoms"]) + 1))
        for pos, at in enumerate(read(path, index=":")):
            if gamma_pristine_min is not None and not at.info["gamma_max_pristine"] >= gamma_pristine_min:
                continue
            at.calc = None
            row = {"ase_atoms": at, **at.info, "hal_dir": d}
            if sc is not None:
                k = int(at.info.get("collection_index", pos))
                covered = k < len(sc["n_atoms"])
                sl = slice(bounds[k], bounds[k + 1]) if covered else None
                row.update(
                    cand_atom_index=sc["atom"][sl] if covered else None,
                    cand_features=sc["features"][sl] if covered else None,
                    cand_gamma=sc["gamma_pristine"][sl] if covered else None,
                    cand_feature_gamma_min=sc["feature_gamma_min"],
                    cand_artifact_sha256=sc["artifact_sha256"],
                )
            rows.append(row)
    return pd.DataFrame(rows) if rows else pd.DataFrame({"ase_atoms": [], "hal_dir": []})


def tau_label(tau) -> str:
    """Directory label of one tau value, e.g. ``tau_0.35`` (tau scan and walker grid)."""
    return f"tau_{float(tau):g}"


def check_unique(name, values) -> None:
    """Refuse repeated values of a scanned setting: each value gets its own directory."""
    dup = sorted({v for v in values if values.count(v) > 1}, key=str)
    if dup:
        raise ValueError(f"duplicate {name} values {dup}: each value gets its own directory")


def scan_tau(taus, atoms, *, output_dir="hal_tau_scan", walkers_per_gpu=None,
             **run_hal_kwargs) -> pd.DataFrame:
    """A tau scan: one walker per tau, otherwise identical settings.

    The same as :func:`~tensorpotential.uq.hal.walkers.run_walkers` with one repeat and
    the list of tau values: walker folders ``<output_dir>/tau_<tau>/hal_w0/``, at most
    ``walkers_per_gpu`` (default 4) at a time per GPU, and the table (one row per tau:
    collected, rejected by reason, max gamma_live, max gamma_pristine at collection,
    fragmented frames, stop reason, mean T, ...) returned and written to
    ``<output_dir>/walkers.csv``. Duplicate tau values are refused up front.
    """
    from tensorpotential.uq.hal.walkers import WALKERS_PER_GPU, run_walkers

    return run_walkers(atoms, 1, tau=[float(t) for t in taus], output_dir=output_dir,
                       walkers_per_gpu=walkers_per_gpu or WALKERS_PER_GPU, **run_hal_kwargs)
