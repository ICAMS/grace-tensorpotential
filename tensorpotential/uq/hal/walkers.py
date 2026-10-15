"""Several independent HAL walkers, each in its own subprocess, oversubscribing the GPUs.

Walker ``i`` runs :func:`~tensorpotential.uq.hal.run.run_hal` in a subprocess
(``python -m tensorpotential.uq.hal.walkers``; the job is pickled to its stdin) and writes
to ``<output_dir>/hal_w{i}`` (``grace_uq select --hal-dirs <output_dir>/hal_w*``), with
``seed + i``, ``walker_id = i`` and starting frame ``(structure_offset + i) mod n_frames``.
Several values of ``tau``, ``ensemble`` or ``temperature_K`` run ``repeats`` walkers for
every combination, in ``<output_dir>/<label>/hal_w{r}`` (label: the varying settings, e.g.
``npt_tau_0.5``); repeat ``r`` gets ``seed + r`` and frame ``(structure_offset + r)``, so the
combinations differ only in the scanned settings, and ``walker_id`` counts over the grid.
At most ``walkers_per_gpu`` walkers run on each visible GPU at the same time (on a CPU-only
machine: ``walkers_per_gpu`` processes on the CPU); the others wait for a free slot. Each
subprocess sees one GPU (``CUDA_VISIBLE_DEVICES``), allocates GPU memory on demand
(``TF_FORCE_GPU_ALLOW_GROWTH=true``, so several walkers share a GPU) and gets
``cores // concurrent walkers`` CPU threads.
"""

from __future__ import annotations

import contextlib
import inspect
import itertools
import json
import os
import pickle
import subprocess
import sys
import time
import warnings
from collections import deque

import numpy as np
import pandas as pd

WALKERS_PER_GPU = 4
WALKER_LOG = "walker.log"
WALKERS_CSV = "walkers.csv"
# the subprocess; a test swaps it for a fake walker
CHILD_CMD = (sys.executable, "-m", "tensorpotential.uq.hal.walkers")
# run_walkers sets these per walker
_PER_WALKER = ("seed", "walker_id", "output_dir", "overwrite", "keep_frames")
# run_hal arguments that may take several values: `repeats` walkers per combination
GRID_KEYS = ("ensemble", "temperature_K", "tau")


def _grid_axes(run_hal_kwargs, defaults) -> dict:
    """{key: [values]} for each of GRID_KEYS: the given value(s), else run_hal's default."""
    from tensorpotential.uq.hal.tools import check_unique

    axes = {}
    for key in GRID_KEYS:
        v = run_hal_kwargs.get(key, defaults[key])
        axes[key] = list(v) if isinstance(v, (list, tuple, np.ndarray)) else [v]
        check_unique(key, axes[key])
    return axes


def _label(combo, varying) -> str:
    """Directory label of a combination: only the settings that vary over the grid."""
    from tensorpotential.uq.hal.tools import tau_label

    fmt = {"ensemble": str, "temperature_K": lambda t: f"T_{t:g}", "tau": tau_label}
    return "_".join(fmt[k](combo[k]) for k in varying)


def _check(frames, overwrite, run_hal_kwargs, combos, outs):
    """Every check that needs no model, before any subprocess starts."""
    from tensorpotential.uq.hal.model_dir import resolve_model_dir
    from tensorpotential.uq.hal.run import OUTPUT_FILES, _validate_options, run_hal

    params = inspect.signature(run_hal).parameters
    unknown = sorted(set(run_hal_kwargs) - set(params))
    if unknown:
        raise TypeError(f"run_walkers: unknown run_hal arguments {unknown}")
    fixed = sorted(set(run_hal_kwargs) & set(_PER_WALKER))
    if fixed:
        raise TypeError(f"run_walkers sets {fixed} per walker; pass seed/output_dir/overwrite "
                        "to run_walkers itself")
    missing = [n for n in ("temperature_K", "n_steps") if n not in run_hal_kwargs]
    if missing:
        raise TypeError(f"run_walkers: missing {missing}")
    kw = {n: p.default for n, p in params.items() if p.default is not inspect.Parameter.empty}
    kw.update(run_hal_kwargs)
    if kw["model_dir"] is not None:
        kw["model_yaml"], kw["checkpoint"], kw["uq_artifact"] = resolve_model_dir(
            kw["model_dir"], model_yaml=kw["model_yaml"], checkpoint=kw["checkpoint"],
            artifact_path=kw["uq_artifact"])
    for name in ("model_yaml", "uq_artifact", "live_start_artifact"):
        if kw[name] is not None and not os.path.isfile(kw[name]):
            raise FileNotFoundError(f"{name} {kw[name]} not found")
    if kw["model_yaml"] is None or kw["checkpoint"] is None or kw["uq_artifact"] is None:
        raise ValueError("pass model_dir or all of model_yaml, checkpoint, uq_artifact")
    ck = kw["checkpoint"]
    if not (os.path.exists(ck) or os.path.exists(ck + ".index")):
        raise FileNotFoundError(f"checkpoint {ck} not found (nor {ck}.index)")
    names = [n for n in inspect.signature(_validate_options).parameters if n != "atoms"]
    for combo in combos:
        opts = {n: {**kw, **combo}[n] for n in names}
        for atoms in frames:
            _validate_options(atoms, **opts)
    if not overwrite:
        held = [out for out in outs
                if any(os.path.exists(os.path.join(out, f)) for f in OUTPUT_FILES)]
        if held:
            raise FileExistsError(f"{len(held)} walker dirs already hold HAL output "
                                  f"({', '.join(held[:3])}{' ...' if len(held) > 3 else ''}); "
                                  "choose another directory (--output-dir; output_dir= in "
                                  "Python) or replace them with --overwrite (overwrite=True "
                                  "in Python)")


def _row(job, device, code, wall_s) -> dict:
    out = job["out"]
    row = {"walker": job["walker"], "repeat": job["repeat"], **job["combo"],
           "start_frame": job["frame"], "device": device or "cpu",
           "exit_code": code, "collected": np.nan, "rejected": "", "n_rejected": np.nan,
           "stop_reason": "", "mean_T_K": np.nan, "steps": np.nan, "steps_per_s": np.nan,
           "max_gamma_live": np.nan, "max_gamma_pristine_collected": np.nan,
           "n_fragmented": np.nan, "wall_s": wall_s, "output_dir": out}
    path = os.path.join(out, "hal_summary.json")
    if os.path.exists(path):  # a walker that failed during the run still finalizes
        with open(path) as fh:
            s = json.load(fh)
        rej = s.get("rejected", {})
        row.update(collected=s["n_collected"], n_rejected=int(sum(rej.values())),
                   rejected=", ".join(f"{k}={v}" for k, v in sorted(rej.items())) or "0",
                   stop_reason=s["stop_reason"], mean_T_K=s["temperature_mean_K"],
                   steps=s["n_steps"], steps_per_s=s["steps_per_s"],
                   max_gamma_live=s.get("gamma_max_live", np.nan),
                   max_gamma_pristine_collected=max(
                       (c["gamma_max_pristine"] for c in s.get("collections", [])),
                       default=np.nan),
                   n_fragmented=s.get("n_fragmented", np.nan))
        row.update({f"rejected_{k}": v for k, v in rej.items()})
    elif code != 0:
        row["stop_reason"] = "failed before the run"
    return row


def _line(row, n_done, n_total, label) -> str:
    where = f"{label}, " if label else ""
    head = (f"[{n_done}/{n_total}] walker {row['walker']} ({where}frame {row['start_frame']}, "
            f"{row['device']})")
    log = os.path.join(row["output_dir"], WALKER_LOG)
    if row["exit_code"] != 0:
        return f"{head} FAILED (exit {row['exit_code']}, {row['stop_reason']}): see {log}"
    return (f"{head}: {row['collected']} collected, rejected {row['rejected']}, "
            f"stop {row['stop_reason']}, {row['mean_T_K']:.0f} K, "
            f"{row['steps_per_s']:.1f} steps/s, {row['wall_s']:.0f} s")


def _table(rows) -> pd.DataFrame:
    df = pd.DataFrame(sorted(rows, key=lambda r: r["walker"]))
    rej = sorted(c for c in df.columns if c.startswith("rejected_"))
    df[rej] = df[rej].fillna(0).astype(int)
    return df[[c for c in df.columns if c not in rej] + rej]


def run_walkers(atoms, repeats, *, walkers_per_gpu=WALKERS_PER_GPU, output_dir="hal_walkers",
                seed=42, overwrite=False, structure_offset=0, gpus=None, verbose=True,
                **run_hal_kwargs) -> pd.DataFrame:
    """Run ``repeats`` independent HAL walkers per setting in subprocesses; one row per walker.

    ``atoms``: one structure or a list; walker ``i`` starts from frame
    ``(structure_offset + i) % len(atoms)``, with ``seed + i`` and ``walker_id = i``, and
    writes to ``<output_dir>/hal_w{i}`` (its stdout/stderr to ``walker.log`` there).
    ``tau``, ``ensemble`` and ``temperature_K`` may each be a list: then ``repeats``
    walkers of every combination run, in ``<output_dir>/<label>/hal_w{r}`` (the label
    names the varying settings, e.g. ``npt_tau_0.5``); repeat ``r`` gets ``seed + r`` and
    frame ``(structure_offset + r) % len(atoms)`` in every combination, and ``walker_id``
    counts over the whole grid. ``walkers_per_gpu`` walkers run on each visible GPU at once
    (``gpus``: ids like "0,1", default CUDA_VISIBLE_DEVICES or every GPU), the rest queue;
    without a GPU, ``walkers_per_gpu`` run on the CPU. The other ``run_hal_kwargs`` go to
    every walker unchanged. Options are checked once, before any subprocess starts. A
    failing walker does not stop the others: its row has ``exit_code != 0`` and a warning
    is issued. The table (walker, repeat, ensemble, temperature, tau, start frame, device,
    exit code, collected, rejected by reason, stop reason, mean T, steps/s, wall time) is
    written to ``<output_dir>/walkers.csv`` after every finished walker.
    """
    from tensorpotential.uq.cli._common import visible_gpu_ids, worker_env
    from tensorpotential.uq.hal.run import _is_int, run_hal

    frames = list(atoms) if isinstance(atoms, (list, tuple)) else [atoms]
    if not _is_int(repeats) or repeats < 1:
        raise ValueError(f"repeats must be an integer >= 1, got {repeats!r}")
    if not _is_int(walkers_per_gpu) or walkers_per_gpu < 1:
        raise ValueError(f"walkers_per_gpu must be an integer >= 1, got {walkers_per_gpu!r}")
    if not frames:
        raise ValueError("no starting structure")
    defaults = {n: p.default for n, p in inspect.signature(run_hal).parameters.items()}
    axes = _grid_axes(run_hal_kwargs, defaults)
    varying = [k for k in GRID_KEYS if len(axes[k]) > 1]
    combos = [dict(zip(GRID_KEYS, c)) for c in itertools.product(*axes.values())]
    jobs = []
    for g, (combo, r) in enumerate(itertools.product(combos, range(repeats))):
        label = _label(combo, varying)  # "" without a grid: OUTPUT_DIR/hal_w<i>
        jobs.append({"walker": g, "repeat": r, "combo": combo, "label": label,
                     "frame": (structure_offset + r) % len(frames),
                     "out": os.path.join(output_dir, label, f"hal_w{r}")})
    fixed = {k: v for k, v in run_hal_kwargs.items() if k not in GRID_KEYS}
    _check(frames, overwrite, run_hal_kwargs, combos, [j["out"] for j in jobs])
    n_total = len(jobs)
    devices = visible_gpu_ids(gpus) or [None]
    n_conc = min(n_total, walkers_per_gpu * len(devices))
    threads = max(1, (os.cpu_count() or 1) // n_conc)
    os.makedirs(output_dir, exist_ok=True)
    csv = os.path.join(output_dir, WALKERS_CSV)
    if verbose:
        where = ", ".join(f"GPU {d}" for d in devices) if devices[0] is not None else "CPU"
        grid = (f" ({len(combos)} combinations x {repeats})" if len(combos) > 1 else "")
        dirs = f"{output_dir}/*/hal_w*" if varying else f"{output_dir}/hal_w*"
        print(f"run_walkers: {n_total} walkers{grid}, up to {walkers_per_gpu} at a time on each "
              f"of {where} ({n_conc} concurrent, {threads} threads each) -> {dirs}", flush=True)

    pending = deque(jobs)
    running = {}  # walker -> (proc, device, log handle, job, t0)
    load = {d: 0 for d in devices}
    rows = []
    try:
        while pending or running:
            while pending and min(load.values()) < walkers_per_gpu:
                d = min(devices, key=lambda x: load[x])  # least loaded, first on ties
                job = pending.popleft()
                os.makedirs(job["out"], exist_ok=True)
                env = worker_env(d, threads)
                log = open(os.path.join(job["out"], WALKER_LOG), "w")
                payload = {"atoms": frames[job["frame"]].copy(),  # no calculator
                           "kwargs": {**fixed, **job["combo"], "seed": seed + job["repeat"],
                                      "walker_id": job["walker"], "output_dir": job["out"],
                                      "overwrite": overwrite, "keep_frames": False}}
                proc = subprocess.Popen(list(CHILD_CMD), stdin=subprocess.PIPE, stdout=log,
                                        stderr=subprocess.STDOUT, env=env)
                # a child that dies before reading its job (import or CUDA failure) is
                # one failed walker: its exit code and walker.log are recorded below
                with contextlib.suppress(BrokenPipeError):
                    proc.stdin.write(pickle.dumps(payload))
                with contextlib.suppress(BrokenPipeError):
                    proc.stdin.close()
                running[job["walker"]] = (proc, d, log, job, time.perf_counter())
                load[d] += 1
            for g, (proc, d, log, job, t0) in list(running.items()):
                code = proc.poll()
                if code is None:
                    continue
                log.close()
                del running[g]
                load[d] -= 1
                rows.append(_row(job, d, code, time.perf_counter() - t0))
                _table(rows).to_csv(csv, index=False)
                if verbose:
                    print(_line(rows[-1], len(rows), n_total, job["label"]), flush=True)
            time.sleep(0.2)
    except BaseException:
        for proc, *_ in running.values():
            proc.terminate()
        raise
    df = _table(rows)
    failed = df.loc[df["exit_code"] != 0, "walker"].tolist()
    if failed:
        warnings.warn(f"{len(failed)} of {n_total} walkers failed: {failed} "
                      f"(see {WALKER_LOG} in their directories, column output_dir)",
                      stacklevel=2)
    return df


def _child_main() -> int:
    job = pickle.load(sys.stdin.buffer)
    from tensorpotential.uq.hal.run import run_hal

    kw = job["kwargs"]
    print(f"walker {kw['walker_id']}: CUDA_VISIBLE_DEVICES="
          f"{os.environ.get('CUDA_VISIBLE_DEVICES')}, seed {kw['seed']}", flush=True)
    s = run_hal(job["atoms"], **kw).summary
    print(f"walker {kw['walker_id']}: {s['n_steps']} steps, {s['n_collected']} collected, "
          f"rejected {s['rejected']}, stop: {s['stop_reason']}, "
          f"{s['steps_per_s']:.2f} steps/s", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(_child_main())
