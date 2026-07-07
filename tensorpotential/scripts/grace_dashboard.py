#!/usr/bin/env python3
"""Interactive Flask dashboard for GRACE-style fit output folders.

Recursively scans a base directory for "fit folders" — any directory that
contains ``seed/<N>/test_metrics.yaml`` for one or more integer seeds — and
renders three panels:

  1. Training curves (metric vs epoch, per-experiment)
  2. Scatter / Pareto plot with selectable X/Y axes
  3. Table overview with all metrics at the best-test-loss epoch

Experiment display names are the deepest non-common path under the base
folder, stripped of the ``/seed/<N>/...`` suffix. Multiple seeds of the same
fit appear as separate rows by default; a checkbox averages them.

Usage:
    grace_dashboard.py [--base DIR] [--host HOST] [--port N]
    grace_dashboard.py               # scan the current directory

Open ``http://<host>:<port>/`` in a browser.
"""
import argparse
import json
import math
import os
import re
from datetime import datetime

import numpy as np
from flask import Flask, jsonify, render_template_string, request


TARGET_EPOCHS = 160

COMMON_METRIC_KEYS = [
    "mae/depa", "rmse/depa",
    "mae/f_comp", "rmse/f_comp",
    "mae/stress", "rmse/stress",
    "mae/virial", "rmse/virial",
    "mae/de", "rmse/de",
]
TRAIN_ONLY = [
    "total_loss/train",
    "loss_component/energy/train", "loss_component/forces/train", "loss_component/stress/train",
]
TEST_ONLY = [
    "total_loss/test",
    "loss_component/energy/test", "loss_component/forces/test", "loss_component/stress/test",
]

PARETO_METRIC_KEYS = [
    "mae/depa", "rmse/depa",
    "mae/f_comp", "rmse/f_comp",
    "mae/stress", "rmse/stress",
]
PARETO_TRAIN_METRIC_KEYS = ["train/" + k for k in PARETO_METRIC_KEYS]
PARETO_SCALAR_KEYS = ["n_params", "train_us_per_atom", "test_us_per_atom"]

PARETO_UNITS = {
    "mae/depa":          ("meV/atom", 1000.0),
    "rmse/depa":         ("meV/atom", 1000.0),
    "mae/f_comp":        ("meV/Å",    1000.0),
    "rmse/f_comp":       ("meV/Å",    1000.0),
    "mae/stress":        ("GPa",      160.21766),
    "rmse/stress":       ("GPa",      160.21766),
    "n_params":          ("params",   1.0),
    "train_us_per_atom": ("μs/atom",  1.0),
    "test_us_per_atom":  ("μs/atom",  1.0),
}
for _k in list(PARETO_UNITS):
    if _k in PARETO_METRIC_KEYS:
        PARETO_UNITS["train/" + _k] = PARETO_UNITS[_k]


# ---------------------------------------------------------------------------
# File parsing helpers
# ---------------------------------------------------------------------------

def load_yaml_lines(path):
    """Parse gracemaker's per-epoch JSON-line YAML files."""
    out = []
    with open(path) as f:
        for line in f:
            if "{" not in line:
                continue
            try:
                out.append(json.loads(line.lstrip("- ")))
            except json.JSONDecodeError:
                pass
    return out


_TIME_RE   = re.compile(r"Time\(mcs/at\):\s+(\d+(?:\.\d+)?)\s+\((\d+(?:\.\d+)?)\)")
_ITER_RE   = re.compile(r"Iteration #(\d+)/(\d+)")
_PARAMS_RE = re.compile(r"Number of trainable parameters:\s*([\d,]+)")
_LOG_TS_RE = re.compile(r"^(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2})")


def extract_n_params(log_path):
    if not os.path.isfile(log_path):
        return None
    try:
        with open(log_path) as f:
            for line in f:
                m = _PARAMS_RE.search(line)
                if m:
                    return int(m.group(1).replace(",", ""))
    except Exception:
        pass
    return None


def extract_times_mcs_atom(log_path):
    """Return (train_us, test_us) per atom from the last steady-state line."""
    if not os.path.isfile(log_path):
        return (None, None)
    last = None
    try:
        with open(log_path) as f:
            for line in f:
                i = _ITER_RE.search(line)
                if i and int(i.group(1)) == 1:
                    continue
                t = _TIME_RE.search(line)
                if t:
                    last = t
    except Exception:
        return (None, None)
    if last is None:
        return (None, None)
    return (float(last.group(1)), float(last.group(2)))


def extract_epoch_timing(log_path):
    """Return seconds-per-epoch from log timestamps, or None."""
    if not os.path.isfile(log_path):
        return None
    first_ts = last_ts = None
    first_ep = last_ep = None
    try:
        with open(log_path) as f:
            for line in f:
                im = _ITER_RE.search(line)
                if not im:
                    continue
                tm = _LOG_TS_RE.match(line)
                if not tm:
                    continue
                ep = int(im.group(1))
                ts = datetime.strptime(tm.group(1), "%Y-%m-%d %H:%M:%S")
                if first_ts is None:
                    first_ts, first_ep = ts, ep
                last_ts, last_ep = ts, ep
    except Exception:
        return None
    if first_ts is None or last_ts is None or last_ep <= first_ep:
        return None
    elapsed = (last_ts - first_ts).total_seconds()
    return elapsed / (last_ep - first_ep)


# ---------------------------------------------------------------------------
# Discovery
# ---------------------------------------------------------------------------

def discover_fits(base):
    """Walk ``base`` and return [{'fit_folder': path, 'seeds': [N, ...]}, ...].

    A "fit folder" is any directory that contains BOTH:
      - an ``input.yaml`` file, and
      - a ``seed/<N>/`` subdirectory for at least one integer N.

    Only seeds whose directory already has a ``test_metrics.yaml`` contribute
    to the returned seed list (so just-started fits with no metrics yet are
    discovered but yield an empty seed list). Discovered fit folders are not
    descended into.
    """
    results = []
    base = os.path.abspath(base)
    for root, dirs, files in os.walk(base):
        if "input.yaml" not in files:
            continue
        seed_dir = os.path.join(root, "seed")
        if not os.path.isdir(seed_dir):
            continue
        try:
            entries = sorted(os.listdir(seed_dir))
        except OSError:
            continue
        numeric_seed_dirs = [n for n in entries
                             if n.isdigit()
                             and os.path.isdir(os.path.join(seed_dir, n))]
        if not numeric_seed_dirs:
            continue
        seeds_with_metrics = [
            int(n) for n in numeric_seed_dirs
            if os.path.isfile(os.path.join(seed_dir, n, "test_metrics.yaml"))
        ]
        results.append({"fit_folder": root, "seeds": sorted(seeds_with_metrics)})
        dirs.clear()  # do not recurse inside a discovered fit folder
    return results


def compute_fit_names(fit_folders, base):
    """Return {folder: display_name}, where display_name is the path under
    ``base`` with the longest common prefix stripped across all folders."""
    base = os.path.abspath(base)
    if not fit_folders:
        return {}
    rel = {f: os.path.relpath(os.path.abspath(f), base) for f in fit_folders}
    split = [p.split(os.sep) for p in rel.values()]
    if len(split) <= 1:
        common_depth = 0
    else:
        common_depth = 0
        for parts in zip(*split):
            if len(set(parts)) == 1:
                common_depth += 1
            else:
                break
    out = {}
    for folder, r in rel.items():
        parts = r.split(os.sep)
        tail = os.sep.join(parts[common_depth:])
        out[folder] = tail or os.path.basename(folder)
    return out


# ---------------------------------------------------------------------------
# Extrapolation (same model as before: metric = a + b * epoch^m)
# ---------------------------------------------------------------------------

def extrapolate_series(rows, key, target_ep, min_eps=10):
    pts = [(r.get("epoch"), r.get(key)) for r in rows]
    # epoch > 0 avoids evaluating a + b * x^m at x=0 (0^m = inf for m<0,
    # triggers numpy RuntimeWarning inside curve_fit).
    pts = [(e, v) for e, v in pts
           if e is not None and v is not None and e > 0 and v > 0]
    if len(pts) < min_eps or pts[-1][0] >= target_ep:
        return None
    eps  = np.array([p[0] for p in pts], dtype=float)
    vals = np.array([p[1] for p in pts], dtype=float)
    v_min, v_max = float(vals.min()), float(vals.max())
    a0 = max(1e-9, v_min * 0.7)
    b0 = max(1e-9, v_max - a0)
    m0 = -0.5
    try:
        from scipy.optimize import curve_fit
        popt, pcov = curve_fit(
            lambda x, a, b, m: a + b * np.power(x, m),
            eps, vals, p0=[a0, b0, m0],
            bounds=([0.0, 0.0, -5.0], [v_min, np.inf, -1e-3]),
            maxfev=5000,
        )
        a, b, m = popt
    except Exception:
        return None
    pred_eps  = np.arange(pts[-1][0] + 1, target_ep + 1, dtype=int)
    if len(pred_eps) == 0:
        return None
    pred_vals = a + b * np.power(pred_eps.astype(float), m)
    last_ep, last_val = pts[-1]
    pred_vals = np.clip(pred_vals, v_min * 0.3, last_val * 1.5)
    std = None
    try:
        ep_t = float(target_ep)
        if ep_t > 0 and np.all(np.isfinite(pcov)):
            grad = np.array([1.0, ep_t ** m, b * (ep_t ** m) * math.log(ep_t)])
            var = float(grad @ pcov @ grad)
            if var > 0 and math.isfinite(var):
                std = min(math.sqrt(var), float(last_val))
    except Exception:
        pass
    return (
        [int(last_ep)] + pred_eps.tolist(),
        [float(last_val)] + pred_vals.tolist(),
        std,
    )


# ---------------------------------------------------------------------------
# Per-seed data collection
# ---------------------------------------------------------------------------

def collect_seed(fit_folder, seed_num, fit_name, target_epochs=TARGET_EPOCHS):
    seed_dir = os.path.join(fit_folder, "seed", str(seed_num))
    test_path  = os.path.join(seed_dir, "test_metrics.yaml")
    train_path = os.path.join(seed_dir, "train_metrics.yaml")
    log_path   = os.path.join(seed_dir, "log.txt")

    if not os.path.isfile(test_path):
        return None
    test_rows = load_yaml_lines(test_path)
    if not test_rows:
        return None
    train_rows = load_yaml_lines(train_path) if os.path.isfile(train_path) else []

    n_params = extract_n_params(log_path)
    train_us, test_us = extract_times_mcs_atom(log_path)

    best = min(test_rows, key=lambda r: r.get("total_loss/test", float("inf")))
    best_ep = best.get("epoch")
    best_train_row = next((r for r in train_rows if r.get("epoch") == best_ep), None)
    best_train = {}
    if best_train_row is not None:
        for k in PARETO_METRIC_KEYS:
            best_train["train/" + k] = best_train_row.get(k)

    mtime = os.path.getmtime(test_path)
    age_s = (datetime.now() - datetime.fromtimestamp(mtime)).total_seconds()

    def series(rows, key):
        return [r.get(key) for r in rows]

    test_series = {
        "epoch": series(test_rows, "epoch"),
        "step":  series(test_rows, "step"),
    }
    for k in COMMON_METRIC_KEYS + TEST_ONLY:
        test_series[k] = series(test_rows, k)
    test_series["lr"] = series(test_rows, "lr_epoch_begin")

    train_series = {
        "epoch": series(train_rows, "epoch"),
        "step":  series(train_rows, "step"),
    }
    for k in COMMON_METRIC_KEYS + TRAIN_ONLY:
        train_series[k] = series(train_rows, k)
    train_series["lr"] = series(train_rows, "lr_epoch_begin")

    n_eps = test_rows[-1].get("epoch") or 0
    complete = n_eps >= target_epochs
    secs_per_epoch = extract_epoch_timing(log_path) if not complete else None
    running_threshold = max(120, 3 * secs_per_epoch) if secs_per_epoch else 120
    is_running = age_s < running_threshold
    eta_s = secs_per_epoch * (target_epochs - n_eps) if is_running and secs_per_epoch else None
    extrap_test = extrap_train = None
    extrap_best, extrap_best_err = {}, {}
    if not complete and n_eps >= 10:
        extrap_test = {}
        for k in COMMON_METRIC_KEYS + TEST_ONLY:
            r = extrapolate_series(test_rows, k, target_epochs)
            if r:
                extrap_test["epoch_" + k] = r[0]
                extrap_test[k] = r[1]
        extrap_train = {}
        for k in COMMON_METRIC_KEYS + TRAIN_ONLY:
            r = extrapolate_series(train_rows, k, target_epochs)
            if r:
                extrap_train["epoch_" + k] = r[0]
                extrap_train[k] = r[1]
        for k in PARETO_METRIC_KEYS:
            r = extrapolate_series(test_rows, k, target_epochs)
            if r and r[1]:
                extrap_best[k] = r[1][-1]
                if r[2] is not None:
                    extrap_best_err[k] = r[2]
            r_tr = extrapolate_series(train_rows, k, target_epochs)
            if r_tr and r_tr[1]:
                extrap_best["train/" + k] = r_tr[1][-1]
                if r_tr[2] is not None:
                    extrap_best_err["train/" + k] = r_tr[2]

    return {
        "name":              fit_name,   # may get a "(seed=N)" suffix in build_payload
        "fit_name":          fit_name,
        "seed":              seed_num,
        "n_params":          n_params,
        "train_us_per_atom": train_us,
        "test_us_per_atom":  test_us,
        "n_epochs":          n_eps,
        "target_epochs":     target_epochs,
        "complete":          complete,
        "best_epoch":        best_ep,
        "best_loss":         best.get("total_loss/test"),
        "best":              {**{k: best.get(k) for k in PARETO_METRIC_KEYS}, **best_train},
        "extrap_best":       extrap_best,
        "extrap_best_err":   extrap_best_err,
        "train":             train_series,
        "test":              test_series,
        "extrap_train":      extrap_train,
        "extrap_test":       extrap_test,
        "state":             "running" if is_running else "idle",
        "age_s":             age_s,
        "eta_s":             eta_s,
    }


def build_payload(base, target_epochs=TARGET_EPOCHS):
    fits = discover_fits(base)
    folders = [f["fit_folder"] for f in fits]
    names = compute_fit_names(folders, base)

    experiments = []
    for fit in fits:
        fit_name = names[fit["fit_folder"]]
        seeds = fit["seeds"]
        multi = len(seeds) > 1
        for s in seeds:
            try:
                exp = collect_seed(fit["fit_folder"], s, fit_name, target_epochs)
            except Exception as e:
                print(f"[warn] skipping {fit['fit_folder']} seed={s}: {e}")
                continue
            if exp is None:
                continue
            if multi:
                exp["name"] = f"{fit_name} (seed={s})"
            experiments.append(exp)

    return {
        "experiments":              experiments,
        "common_metric_keys":       COMMON_METRIC_KEYS,
        "train_only":               TRAIN_ONLY,
        "test_only":                TEST_ONLY,
        "pareto_metric_keys":       PARETO_METRIC_KEYS,
        "pareto_train_metric_keys": PARETO_TRAIN_METRIC_KEYS,
        "pareto_scalar_keys":       PARETO_SCALAR_KEYS,
        "pareto_units":             PARETO_UNITS,
        "base":                     os.path.abspath(base),
        "base_name":                os.path.basename(os.path.abspath(base)) or "root",
        "target_epochs":            target_epochs,
    }


# ---------------------------------------------------------------------------
# HTML template
# ---------------------------------------------------------------------------

TEMPLATE = r"""
<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>GRACE dashboard — {{ base_name }}</title>
<link rel="icon" href="data:image/svg+xml;utf8,<svg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 64 64'><circle cx='32' cy='32' r='5' fill='%231f77b4'/><ellipse cx='32' cy='32' rx='26' ry='10' fill='none' stroke='%231f77b4' stroke-width='3'/><ellipse cx='32' cy='32' rx='26' ry='10' fill='none' stroke='%23d62728' stroke-width='3' transform='rotate(60 32 32)'/><ellipse cx='32' cy='32' rx='26' ry='10' fill='none' stroke='%232ca02c' stroke-width='3' transform='rotate(-60 32 32)'/></svg>">
<script src="https://cdn.plot.ly/plotly-2.35.2.min.js"></script>
<style>
  body { font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
         margin: 0 1.5rem 2rem; color: #222; }
  h1 { margin-top: 1rem; font-size: 1.3rem; display: flex; align-items: center; gap: 0.45rem; }
  h1 .grace-atom { display: inline-block; font-size: 1.3rem; }
  h1 .folder { color: #1f77b4; font-family: ui-monospace, Menlo, monospace;
               font-size: 1.05rem; background: #eef5ff; padding: 1px 8px;
               border-radius: 4px; }
  h2 { margin-top: 1.5rem; font-size: 1.05rem; border-bottom: 1px solid #ddd;
       padding-bottom: 0.15rem; }
  .row { display: flex; gap: 1rem; flex-wrap: wrap; align-items: center;
         margin: 0.4rem 0; font-size: 0.9rem; }
  .row label { font-weight: 600; }
  .row select, .row input { font-size: 0.9rem; padding: 2px 4px; }
  .exp-grid { display: grid; grid-template-columns: repeat(auto-fill, minmax(220px, 1fr));
              gap: 0 0.75rem; font-size: 0.85rem; }
  .exp-grid label { display: block; white-space: nowrap;
                    overflow: hidden; text-overflow: ellipsis; font-weight: normal; }
  .panel { border: 1px solid #ddd; border-radius: 6px; padding: 0.6rem;
           margin-bottom: 1rem; background: #fafafa; }
  .chart { height: 540px; background: white; border-radius: 4px; }
  .scatter { height: 560px; background: white; border-radius: 4px; }
  .muted { color: #888; font-size: 0.8rem; }
  .pareto-summary { font-family: ui-monospace, Menlo, monospace; font-size: 0.75rem;
                    overflow-x: auto; white-space: nowrap; max-width: 100%; }
  table { border-collapse: collapse; font-size: 0.8rem; }
  th, td { padding: 2px 8px; border-bottom: 1px solid #eee; text-align: right; }
  th:first-child, td:first-child { text-align: left; }
  tr.running td { color: #b35c00; font-weight: 600; }
  tr.clickable { cursor: pointer; }
  tr.clickable:hover td { background: #eef5ff; }
  th.sortable { cursor: pointer; user-select: none; }
  th.sortable:hover { background: #f0f0f0; }
  th.sort-asc::after  { content: " ▲"; color: #888; }
  th.sort-desc::after { content: " ▼"; color: #888; }
</style>
</head>
<body>

<h1>
  <span class="grace-atom">⚛️</span>
  <span>GRACE dashboard</span>
  <span class="folder">{{ base_name }}</span>
  <button id="refresh-btn" onclick="refresh()" style="margin-left: 0.6rem;
          font-size: 0.85rem; padding: 3px 10px; cursor: pointer;">↻ refresh</button>
  <span class="muted" id="meta"></span>
</h1>

<div class="panel">
  <h2>1. Training curves</h2>
  <div class="row">
    <label for="curve-metric">Metric:</label>
    <select id="curve-metric"></select>
    <label for="curve-split">Split:</label>
    <select id="curve-split">
      <option value="both">train + test</option>
      <option value="test" selected>test only</option>
      <option value="train">train only</option>
    </select>
    <label for="curve-xaxis">X axis:</label>
    <select id="curve-xaxis">
      <option value="epoch">epoch</option>
      <option value="step" selected>step (batch)</option>
    </select>
    <label><input type="checkbox" id="curve-logx"> log x</label>
    <label><input type="checkbox" id="curve-logy" checked> log y</label>
    <label><input type="checkbox" id="avg-seeds"> average seeds</label>
    <label><input type="checkbox" id="extrap-on"> extrapolate</label>
    <span id="extrap-target-wrap" style="display: none;">
      <label for="extrap-target"><span id="extrap-unit-label">to ep</span>:
        <input type="number" id="extrap-target" value="160" min="10" max="100000" style="width: 80px;">
      </label>
    </span>
  </div>
  <div class="row">
    <label>Experiments:</label>
    <button onclick="toggleAllExperiments(true)">all</button>
    <button onclick="toggleAllExperiments(false)">none</button>
    <label for="exp-filter" style="margin-left: 0.6rem;">filter:</label>
    <input type="text" id="exp-filter" placeholder="substring" style="width: 260px;">
    <button onclick="clearFilter()">clear</button>
    <span class="muted" id="filter-count"></span>
  </div>
  <div class="exp-grid" id="exp-list"></div>
  <div id="curve-chart" class="chart"></div>
</div>

<div class="panel">
  <h2>2. Scatter / Pareto (values at best-test-loss epoch)</h2>
  <div class="row">
    <label for="scatter-x">X axis:</label>
    <select id="scatter-x"></select>
    <label for="scatter-y">Y axis:</label>
    <select id="scatter-y"></select>
    <label><input type="checkbox" id="scatter-logx" checked> log x</label>
    <label><input type="checkbox" id="scatter-logy" checked> log y</label>
    <label><input type="checkbox" id="scatter-pareto" checked> draw Pareto front</label>
    <span class="muted">· click a point to solo it on plot 1 · shift-click to toggle</span>
  </div>
  <div id="scatter-chart" class="scatter"></div>
</div>

<div class="panel">
  <h2>3. Table overview</h2>
  <div style="margin: 0.4rem 0;">
    <button id="copy-md-btn" onclick="copyTableAsMarkdown()">copy table as markdown</button>
    <span class="muted" id="copy-status"></span>
  </div>
  <div class="pareto-summary" id="scatter-table"></div>
</div>

<script>
let DATA = null;
let HIGHLIGHTED_EXP = null;
let FILTER = "";
let SORT_COL = null;
let SORT_DIR = 1;
let EXTRAP_MODE = "epoch";  // tracks whether extrap target is in epochs or steps

function globalStepsPerEpoch() {
  if (!DATA || !DATA.experiments) return null;
  for (const e of DATA.experiments) {
    const f = stepsPerEpoch(e, "test") || stepsPerEpoch(e, "train");
    if (f && f > 0) return f;
  }
  return null;
}

function syncExtrapLabel() {
  const xKey = document.getElementById("curve-xaxis").value;
  const newMode = xKey === "step" ? "step" : "epoch";
  const label = document.getElementById("extrap-unit-label");
  const inp = document.getElementById("extrap-target");
  if (!label || !inp) return;
  if (newMode !== EXTRAP_MODE) {
    const f = globalStepsPerEpoch();
    const cur = parseInt(inp.value, 10) || 160;
    if (f && f > 0) {
      inp.value = newMode === "step" ? Math.round(cur * f) : Math.round(cur / f);
    }
    EXTRAP_MODE = newMode;
  }
  label.textContent = newMode === "step" ? "to step" : "to ep";
}

function sortByColumn(col) {
  if (SORT_COL === col) SORT_DIR = -SORT_DIR;
  else { SORT_COL = col; SORT_DIR = 1; }
  applySort();
}

function applySort() {
  const table = document.querySelector("#scatter-table table");
  if (!table) return;
  const ths = table.querySelectorAll("thead th");
  ths.forEach((th, i) => {
    th.classList.remove("sort-asc", "sort-desc");
    if (i === SORT_COL) th.classList.add(SORT_DIR > 0 ? "sort-asc" : "sort-desc");
  });
  if (SORT_COL == null) return;
  const tbody = table.querySelector("tbody");
  const trs = [...tbody.querySelectorAll("tr")];
  trs.sort((a, b) => {
    const ac = a.children[SORT_COL]; const bc = b.children[SORT_COL];
    const av = ac ? (ac.dataset.sortKey ?? ac.innerText) : "";
    const bv = bc ? (bc.dataset.sortKey ?? bc.innerText) : "";
    const an = (av === "" || av == null) ? Infinity : parseFloat(av);
    const bn = (bv === "" || bv == null) ? Infinity : parseFloat(bv);
    let r;
    if (!isNaN(an) && !isNaN(bn) && !(an === Infinity && bn === Infinity)) r = an - bn;
    else r = String(av).localeCompare(String(bv));
    return r * SORT_DIR;
  });
  trs.forEach(tr => tbody.appendChild(tr));
}

function passesFilter(name) {
  if (!FILTER) return true;
  return name.toLowerCase().includes(FILTER);
}

function updateFilterVisibility() {
  let visible = 0, total = 0;
  document.querySelectorAll("#exp-list > label").forEach(lbl => {
    total += 1;
    const name = lbl.querySelector("input").dataset.name;
    const show = passesFilter(name);
    lbl.style.display = show ? "" : "none";
    if (show) visible += 1;
  });
  const tag = document.getElementById("filter-count");
  if (tag) tag.textContent = FILTER ? `${visible}/${total} match` : "";
}

function clearFilter() {
  FILTER = "";
  document.getElementById("exp-filter").value = "";
  updateFilterVisibility();
  drawCurves(); drawScatter();
}

function copyTableAsMarkdown() {
  const table = document.querySelector("#scatter-table table");
  const status = document.getElementById("copy-status");
  if (!table) { if (status) status.textContent = "no table"; return; }
  const headers = [...table.querySelectorAll("thead th")].map(th =>
    th.innerText.replace(/\n/g, " ").trim().replace(/\s+/g, " "));
  const rows = [...table.querySelectorAll("tbody tr")].map(tr =>
    [...tr.querySelectorAll("td")].map(td => td.innerText.trim()));
  const md = [
    "| " + headers.join(" | ") + " |",
    "|" + headers.map(() => "---").join("|") + "|",
    ...rows.map(r => "| " + r.join(" | ") + " |"),
  ].join("\n");
  const show = (msg) => {
    if (!status) return;
    status.textContent = msg;
    setTimeout(() => { status.textContent = ""; }, 3000);
  };
  if (navigator.clipboard && navigator.clipboard.writeText) {
    navigator.clipboard.writeText(md)
      .then(() => show(`copied ${rows.length} rows`))
      .catch(err => show("copy failed: " + err));
  } else {
    const ta = document.createElement("textarea");
    ta.value = md; document.body.appendChild(ta);
    ta.select(); try { document.execCommand("copy"); show(`copied ${rows.length} rows`); }
    catch (e) { show("copy failed"); }
    document.body.removeChild(ta);
  }
}

const PALETTE = [
  "#1f77b4","#d62728","#2ca02c","#9467bd","#ff7f0e","#8c564b",
  "#e377c2","#17becf","#bcbd22","#7f7f7f","#5c8df7","#b06f26",
  "#008b8b","#a52a2a","#00688b","#cd5c5c"
];
function colorFor(name) {
  if (!DATA._colorMap) DATA._colorMap = {};
  if (!(name in DATA._colorMap)) {
    DATA._colorMap[name] = PALETTE[Object.keys(DATA._colorMap).length % PALETTE.length];
  }
  return DATA._colorMap[name];
}

// ---- Seed averaging -----------------------------------------------------
// When "average seeds" is checked, groups per-seed experiments by fit_name and
// collapses each group into one averaged experiment. Curves are averaged only
// over epochs present in every seed of the group.

function averageSeries(side, es, metric) {
  const epochs0 = es[0][side].epoch || [];
  const outE = [], outV = [];
  for (let i = 0; i < epochs0.length; i++) {
    const ep = epochs0[i];
    const vals = [];
    let ok = true;
    for (const e of es) {
      const j = (e[side].epoch || []).indexOf(ep);
      if (j < 0) { ok = false; break; }
      const v = (e[side][metric] || [])[j];
      if (v == null) { ok = false; break; }
      vals.push(v);
    }
    if (ok && vals.length) {
      outE.push(ep);
      outV.push(vals.reduce((a,b)=>a+b, 0) / vals.length);
    }
  }
  return [outE, outV];
}

function averageExperiments(name, es) {
  const n = es.length;
  const keysOf = (e, field) => e[field] ? Object.keys(e[field]) : [];
  const averageObj = (getter) => {
    const keys = new Set();
    es.forEach(e => keysOf(e, getter).forEach(k => keys.add(k)));
    const out = {};
    for (const k of keys) {
      const vals = es.map(e => (e[getter] || {})[k]).filter(v => v != null);
      if (vals.length === n) out[k] = vals.reduce((a,b)=>a+b, 0) / n;
    }
    return out;
  };
  const best        = averageObj("best");
  const extrap_best = averageObj("extrap_best");

  const avgSide = (side) => {
    const keys = new Set();
    es.forEach(e => Object.keys(e[side] || {}).forEach(k => keys.add(k)));
    const out = {};
    let commonEpochs = null;
    for (const k of keys) {
      if (k === "epoch") continue;
      const [ee, vv] = averageSeries(side, es, k);
      if (commonEpochs == null) commonEpochs = ee;
      out[k] = vv;
    }
    out.epoch = commonEpochs || [];
    return out;
  };

  return {
    name: name + " (avg)",
    fit_name: name,
    seed: null,
    n_params: es[0].n_params,
    train_us_per_atom: es[0].train_us_per_atom,
    test_us_per_atom:  es[0].test_us_per_atom,
    n_epochs: Math.min(...es.map(e => e.n_epochs || 0)),
    target_epochs: es[0].target_epochs,
    complete: es.every(e => e.complete),
    best_epoch: Math.round(es.map(e => e.best_epoch || 0).reduce((a,b)=>a+b,0) / n),
    best_loss: null,
    best, extrap_best, extrap_best_err: {},
    train: avgSide("train"),
    test:  avgSide("test"),
    extrap_train: null,
    extrap_test:  null,
    state: es.some(e => e.state === "running") ? "running" : "idle",
    num_seeds: n,
  };
}

function effectiveExperiments() {
  const raw = DATA.experiments || [];
  const chk = document.getElementById("avg-seeds");
  if (!chk || !chk.checked) return raw;
  const groups = new Map();
  for (const e of raw) {
    if (!groups.has(e.fit_name)) groups.set(e.fit_name, []);
    groups.get(e.fit_name).push(e);
  }
  const out = [];
  for (const [name, es] of groups) {
    if (es.length === 1) out.push(es[0]);
    else out.push(averageExperiments(name, es));
  }
  return out;
}

// ---- Curve / scatter populators ----------------------------------------

function fmtEta(secs) {
  if (secs == null || !isFinite(secs) || secs < 0) return "";
  const h = Math.floor(secs / 3600);
  const m = Math.floor((secs % 3600) / 60);
  if (h > 24) { const d = (secs / 86400).toFixed(1); return d + "d"; }
  if (h > 0) return h + "h" + String(m).padStart(2, "0") + "m";
  if (m > 0) return m + "m";
  return "<1m";
}

function fmtUnits(key) {
  const u = DATA.pareto_units[key];
  return u ? u[0] : "";
}
function scaleValue(key, v) {
  if (v === null || v === undefined) return null;
  const u = DATA.pareto_units[key];
  return u ? v * u[1] : v;
}

function populateCurveMetric() {
  const sel = document.getElementById("curve-metric");
  const metrics = [
    ...DATA.common_metric_keys,
    ...DATA.train_only,
    ...DATA.test_only,
    "lr",
  ];
  metrics.forEach(k => {
    const o = document.createElement("option");
    o.value = k; o.textContent = k;
    sel.appendChild(o);
  });
  sel.value = "mae/f_comp";
}

function populateExpList() {
  const g = document.getElementById("exp-list");
  g.innerHTML = "";
  // Checkbox labels always use the raw per-seed list
  DATA.experiments.forEach(e => {
    const id = "exp-chk-" + e.name;
    const lbl = document.createElement("label");
    lbl.innerHTML = `<input type="checkbox" id="${id}" checked data-name="${e.name}">
                     <span style="color:${colorFor(e.name)}">●</span>
                     ${e.name}`;
    lbl.querySelector("input").addEventListener("change", drawCurves);
    g.appendChild(lbl);
  });
}

function toggleAllExperiments(on) {
  document.querySelectorAll("#exp-list input[type=checkbox]").forEach(c => c.checked = on);
  drawCurves();
}

function selectedExpSet() {
  const s = new Set();
  document.querySelectorAll("#exp-list input[type=checkbox]").forEach(c => {
    if (c.checked) s.add(c.dataset.name);
  });
  return s;
}

// Infer a steps-per-epoch factor from any experiment's existing (epoch, step)
// pairs — used to remap extrapolation continuation arrays (which live in epochs)
// onto the step axis when the user selects it.
function stepsPerEpoch(e, side) {
  const ee = (e && e[side] && e[side].epoch) || [];
  const ss = (e && e[side] && e[side].step)  || [];
  for (let i = ee.length - 1; i >= 0; i--) {
    if (ee[i] && ee[i] > 0 && ss[i] != null) return ss[i] / ee[i];
  }
  return null;
}

// Filter out (x, y) pairs where either is null — happens on the `step` axis
// for the first (INITIAL) row which has no step value yet. Optionally scales
// y by a fixed factor (used to convert e.g. eV/atom → meV/atom on the curves).
function filterXY(xs, ys, yScale) {
  const s = (yScale == null || yScale === 1) ? null : yScale;
  const xo = [], yo = [];
  const n = Math.min(xs.length, ys.length);
  for (let i = 0; i < n; i++) {
    if (xs[i] == null || ys[i] == null) continue;
    xo.push(xs[i]); yo.push(s ? ys[i] * s : ys[i]);
  }
  return [xo, yo];
}

// Curve-panel unit factors: applied in drawCurves only. Currently only the
// per-atom energy metrics are rescaled (eV/atom → meV/atom); everything else
// renders in its native units (loss components, lr, etc.).
const CURVE_SCALE_KEYS = new Set(["mae/depa", "rmse/depa"]);
function curveUnitInfo(metric) {
  if (!CURVE_SCALE_KEYS.has(metric)) return { scale: 1, unit: "" };
  const u = DATA.pareto_units && DATA.pareto_units[metric];
  return u ? { scale: u[1], unit: u[0] } : { scale: 1, unit: "" };
}

function extrapOn() {
  const c = document.getElementById("extrap-on");
  return !!(c && c.checked);
}

function drawCurves() {
  const metric = document.getElementById("curve-metric").value;
  const split = document.getElementById("curve-split").value;
  const xKey = document.getElementById("curve-xaxis").value;   // "epoch" | "step"
  const logx = document.getElementById("curve-logx").checked;
  const logy = document.getElementById("curve-logy").checked;
  const withExtrap = extrapOn();
  const selected = selectedExpSet();
  const experiments = effectiveExperiments();
  const { scale: yScale, unit: yUnit } = curveUnitInfo(metric);

  // If averaging is ON, "selected" contains raw seed names. Accept an averaged
  // exp whose underlying fit_name has *any* of its seeds selected.
  const rawByFit = new Map();
  for (const e of DATA.experiments) {
    if (!rawByFit.has(e.fit_name)) rawByFit.set(e.fit_name, []);
    rawByFit.get(e.fit_name).push(e);
  }
  const avg = document.getElementById("avg-seeds")?.checked;
  const isSelected = (e) => {
    if (!avg) return selected.has(e.name);
    const names = (rawByFit.get(e.fit_name) || []).map(x => x.name);
    return names.some(n => selected.has(n));
  };

  const opacityFor = (name) =>
    (HIGHLIGHTED_EXP == null || HIGHLIGHTED_EXP === name) ? 1.0 : 0.25;
  const widthBoost = (name, base) => (HIGHLIGHTED_EXP === name) ? base + 1 : base;

  const traces = [];
  for (const e of experiments) {
    if (!isSelected(e)) continue;
    if (!passesFilter(e.name)) continue;
    const c = colorFor(avg ? e.fit_name : e.name);
    const op = opacityFor(e.name);
    if (split === "test" || split === "both") {
      const rawX = e.test[xKey] || e.test.epoch;
      const [x, y] = filterXY(rawX, e.test[metric] || [], yScale);
      if (y && y.some(v => v !== null && v !== undefined)) {
        traces.push({
          name: e.name + (split === "both" ? " (test)" : ""),
          x, y, type: "scatter", mode: "lines",
          line: { color: c, width: widthBoost(e.name, 2) },
          opacity: op,
          legendgroup: e.name,
        });
      }
      // Extrapolation arrays are in epoch units; rescale if X is steps.
      let xe = withExtrap && e.extrap_test ? e.extrap_test["epoch_" + metric] : null;
      let ye = withExtrap && e.extrap_test ? e.extrap_test[metric] : null;
      if (xe && ye && xe.length > 1) {
        if (xKey === "step") {
          const f = stepsPerEpoch(e, "test");
          xe = f ? xe.map(ep => ep * f) : xe;
        }
        if (yScale !== 1) ye = ye.map(v => v * yScale);
        traces.push({
          name: e.name + " (extrap)",
          x: xe, y: ye, type: "scatter", mode: "lines",
          line: { color: c, width: widthBoost(e.name, 2), dash: "dash" },
          opacity: op, legendgroup: e.name, showlegend: false, hoverinfo: "skip",
        });
      }
    }
    if (split === "train" || split === "both") {
      const rawX = e.train[xKey] || e.train.epoch;
      const [x, y] = filterXY(rawX, e.train[metric] || [], yScale);
      if (y && y.some(v => v !== null && v !== undefined)) {
        traces.push({
          name: e.name + (split === "both" ? " (train)" : ""),
          x, y, type: "scatter", mode: "lines",
          line: { color: c, width: widthBoost(e.name, 1.3), dash: "dot" },
          opacity: op, legendgroup: e.name,
          showlegend: split !== "both",
        });
      }
      let xe = withExtrap && e.extrap_train ? e.extrap_train["epoch_" + metric] : null;
      let ye = withExtrap && e.extrap_train ? e.extrap_train[metric] : null;
      if (xe && ye && xe.length > 1) {
        if (xKey === "step") {
          const f = stepsPerEpoch(e, "train");
          xe = f ? xe.map(ep => ep * f) : xe;
        }
        if (yScale !== 1) ye = ye.map(v => v * yScale);
        traces.push({
          name: e.name + " (train extrap)",
          x: xe, y: ye, type: "scatter", mode: "lines",
          line: { color: c, width: widthBoost(e.name, 1.3), dash: "dashdot" },
          opacity: op, legendgroup: e.name, showlegend: false, hoverinfo: "skip",
        });
      }
    }
  }

  const layout = {
    margin: { t: 10, r: 20, b: 50, l: 65 },
    xaxis: { title: xKey === "step" ? "step (batches)" : "epoch",
             type: logx ? "log" : "linear" },
    yaxis: { title: yUnit ? `${metric} (${yUnit})` : metric, type: logy ? "log" : "linear" },
    legend: { orientation: "v", x: 1.01, y: 1 },
    hovermode: "closest",
  };
  const chart = document.getElementById("curve-chart");
  Plotly.newPlot(chart, traces, layout, { responsive: true }).then(() => {
    chart.on("plotly_click", (ev) => {
      const g = ev && ev.points && ev.points[0] && ev.points[0].data.legendgroup;
      if (!g) return;
      HIGHLIGHTED_EXP = (HIGHLIGHTED_EXP === g) ? null : g;
      drawCurves();
    });
    chart.on("plotly_doubleclick", () => { HIGHLIGHTED_EXP = null; drawCurves(); });
  });
}

function populateScatterAxes() {
  const allKeys = [
    ...DATA.pareto_metric_keys,
    ...(DATA.pareto_train_metric_keys || []),
    ...DATA.pareto_scalar_keys,
  ];
  const xSel = document.getElementById("scatter-x");
  const ySel = document.getElementById("scatter-y");
  xSel.innerHTML = ""; ySel.innerHTML = "";
  allKeys.forEach(k => {
    const label = `${k} (${fmtUnits(k)})`;
    [xSel, ySel].forEach(sel => {
      const o = document.createElement("option");
      o.value = k; o.textContent = label;
      sel.appendChild(o);
    });
  });
  xSel.value = "mae/depa";
  ySel.value = "mae/f_comp";
}

function getScatterValue(e, key) {
  if (DATA.pareto_scalar_keys.includes(key)) return e[key];
  // Only fall back to extrapolated values if the user has opted in.
  if (extrapOn() && !e.complete && e.extrap_best && (key in e.extrap_best))
    return e.extrap_best[key];
  return (e.best || {})[key];
}

function computeParetoFront(pts) {
  const sorted = [...pts].sort((a, b) => (a.x - b.x) || (a.y - b.y));
  const front = [];
  let bestY = Infinity;
  for (const p of sorted) {
    if (p.y < bestY) { front.push(p); bestY = p.y; }
  }
  return front;
}

function drawScatter() {
  const xKey = document.getElementById("scatter-x").value;
  const yKey = document.getElementById("scatter-y").value;
  const logx = document.getElementById("scatter-logx").checked;
  const logy = document.getElementById("scatter-logy").checked;
  const showPareto = document.getElementById("scatter-pareto").checked;
  const avg = document.getElementById("avg-seeds")?.checked;
  const experiments = effectiveExperiments();

  const withExtrap = extrapOn();
  const pts = [];
  for (const e of experiments) {
    if (!passesFilter(e.name)) continue;
    const xv = scaleValue(xKey, getScatterValue(e, xKey));
    const yv = scaleValue(yKey, getScatterValue(e, yKey));
    const xHas = (xv !== null && xv !== undefined);
    const yHas = (yv !== null && yv !== undefined);
    const hasPoint = xHas && yHas;
    let yerr = null, xerr = null;
    if (withExtrap && !e.complete && e.extrap_best_err && (yKey in e.extrap_best_err))
      yerr = scaleValue(yKey, e.extrap_best_err[yKey]);
    if (withExtrap && !e.complete && e.extrap_best_err && (xKey in e.extrap_best_err))
      xerr = scaleValue(xKey, e.extrap_best_err[xKey]);
    // "extrapolated" flag controls open-marker rendering. Only flag incomplete
    // runs as extrapolated when the user has opted in; otherwise render normally
    // at their current best-test-loss-epoch values.
    const extrapolated = withExtrap && !e.complete;
    pts.push({
      name: e.name, x: xv, y: yv,
      color: colorFor(avg ? e.fit_name : e.name),
      state: e.state, epoch: e.best_epoch, seed: e.seed,
      complete: !!e.complete, extrapolated,
      n_epochs: e.n_epochs, target_epochs: e.target_epochs, eta_s: e.eta_s,
      yerr, xerr, hasPoint, num_seeds: e.num_seeds,
    });
  }

  const front = computeParetoFront(pts.filter(p => p.hasPoint));
  const frontSet = new Set(front.map(p => p.name));

  const traces = pts.filter(p => p.hasPoint).map(p => {
    const baseSym = frontSet.has(p.name) ? "diamond" : "circle";
    // Open markers only when the user asked for extrapolation AND this run is incomplete.
    const sym = p.extrapolated ? (baseSym + "-open") : baseSym;
    const trace = {
      x: [p.x], y: [p.y], name: p.name,
      mode: "markers+text", type: "scatter",
      marker: {
        color: p.color,
        size: frontSet.has(p.name) ? 15 : 11,
        symbol: sym,
        line: {
          width: p.extrapolated
            ? 2
            : (p.state === "running" ? 2 : (frontSet.has(p.name) ? 1.5 : 0)),
          color: p.extrapolated ? p.color : "#000",
        },
      },
      text: [p.name], textposition: "top center",
      textfont: { size: 10, color: frontSet.has(p.name) ? "#000" : "#555" },
      hovertemplate: `<b>%{text}</b><br>x = %{x}<br>y = %{y}<br>best ep: ${p.epoch}` +
                     (p.extrapolated ? `<br><i>extrapolated to ep ${p.target_epochs} (ran to ${p.n_epochs})</i>` : "") +
                     (!p.complete && !p.extrapolated ? `<br><i>ran to ep ${p.n_epochs}</i>` : "") +
                     (p.yerr != null ? `<br>σ(y) = ${p.yerr.toExponential(2)}` : "") +
                     (p.seed != null ? `<br>seed: ${p.seed}` : "") +
                     (p.num_seeds ? `<br>averaged over ${p.num_seeds} seeds` : "") +
                     (frontSet.has(p.name) ? "<br><i>Pareto-optimal</i>" : "") + "<extra></extra>",
      showlegend: false,
    };
    if (p.yerr != null) trace.error_y = { type: "data", array: [p.yerr], visible: true,
                                          color: p.color, thickness: 1.5, width: 6 };
    if (p.xerr != null) trace.error_x = { type: "data", array: [p.xerr], visible: true,
                                          color: p.color, thickness: 1.5, width: 6 };
    return trace;
  });

  if (showPareto && front.length >= 2) {
    traces.push({
      name: "Pareto front",
      x: front.map(p => p.x), y: front.map(p => p.y),
      mode: "lines", type: "scatter",
      line: { color: "rgba(0,0,0,0.45)", width: 1.8, shape: "hv", dash: "dash" },
      hoverinfo: "skip", showlegend: false,
    });
  }

  const layout = {
    margin: { t: 10, r: 20, b: 60, l: 80 },
    xaxis: { title: `${xKey} (${fmtUnits(xKey)})`, type: logx ? "log" : "linear" },
    yaxis: { title: `${yKey} (${fmtUnits(yKey)})`, type: logy ? "log" : "linear" },
    hovermode: "closest",
  };
  const chart = document.getElementById("scatter-chart");
  Plotly.newPlot(chart, traces, layout, { responsive: true }).then(() => {
    chart.on("plotly_click", (ev) => {
      if (!ev || !ev.points || !ev.points.length) return;
      const name = ev.points[0].data.name;
      const shift = ev.event && ev.event.shiftKey;
      const boxes = document.querySelectorAll("#exp-list input[type=checkbox]");
      if (shift) boxes.forEach(c => { if (c.dataset.name === name) c.checked = !c.checked; });
      else       boxes.forEach(c => { c.checked = (c.dataset.name === name); });
      drawCurves();
      document.getElementById("curve-chart").scrollIntoView({ behavior: "smooth", block: "center" });
    });
  });

  // ---- Table ----
  const metricCols = [
    ["mae/depa",    "E-MAE"],
    ["rmse/depa",   "E-RMSE"],
    ["mae/f_comp",  "F-MAE"],
    ["rmse/f_comp", "F-RMSE"],
    ["mae/stress",  "S-MAE"],
    ["rmse/stress", "S-RMSE"],
  ];
  const fmtMetric = (e, key) => {
    const raw = e && e.best ? e.best[key] : null;
    if (raw == null) return { display: "", sort: "" };
    const unit = DATA.pareto_units[key];
    const scaled = unit ? raw * unit[1] : raw;
    return { display: Number(scaled).toPrecision(3), sort: scaled };
  };
  const makeTh = (label, idx) => `<th class="sortable" onclick="sortByColumn(${idx})">${label}</th>`;
  const headers = ["experiment", "seed", "params", "n_eps", "best_ep"];
  for (const [k, lbl] of metricCols) {
    const u = DATA.pareto_units[k] ? ` (${DATA.pareto_units[k][0]})` : "";
    headers.push(`${lbl}<br>te${u}`, `${lbl}<br>tr${u}`);
  }
  headers.push("state", "Pareto");
  const headerHtml = headers.map((h, i) => makeTh(h, i)).join("");

  const expByName = new Map(experiments.map(e => [e.name, e]));
  const rows = [...pts].sort((a, b) => (a.y ?? Infinity) - (b.y ?? Infinity));
  let html = `<table><thead><tr>${headerHtml}</tr></thead><tbody>`;
  for (const p of rows) {
    const e = expByName.get(p.name);
    const cls = p.state === "running" ? "running" : "";
    const tag = frontSet.has(p.name) ? "◆" : "";
    const seed = (p.seed == null) ? (p.num_seeds ? `avg×${p.num_seeds}` : "") : p.seed;
    const eta = fmtEta(p.eta_s);
    const state = p.complete
      ? p.state
      : (p.extrapolated
          ? `${p.state} (~extrap ${p.n_epochs}/${p.target_epochs})` + (eta ? ` ETA ${eta}` : "")
          : `${p.state} (ep ${p.n_epochs})` + (eta ? ` ETA ${eta}` : ""));
    const metricCells = metricCols.flatMap(([k]) => {
      const t = fmtMetric(e, k);
      const r = fmtMetric(e, "train/" + k);
      return [
        `<td data-sort-key="${t.sort}">${t.display}</td>`,
        `<td data-sort-key="${r.sort}">${r.display}</td>`,
      ];
    }).join("");
    html += `<tr class="${cls} clickable" data-name="${p.name}">
      <td data-sort-key="${p.name}">${p.name}</td>
      <td data-sort-key="${seed !== "" ? seed : Infinity}">${seed}</td>
      <td data-sort-key="${e && e.n_params != null ? e.n_params : ""}">${e && e.n_params != null ? e.n_params : ""}</td>
      <td data-sort-key="${e && e.n_epochs != null ? e.n_epochs : ""}">${e && e.n_epochs != null ? e.n_epochs : ""}</td>
      <td data-sort-key="${p.epoch != null ? p.epoch : ""}">${p.epoch ?? ""}</td>
      ${metricCells}
      <td data-sort-key="${p.state}">${state}</td>
      <td data-sort-key="${frontSet.has(p.name) ? 0 : 1}">${tag}</td></tr>`;
  }
  html += "</tbody></table>";
  const tbl = document.getElementById("scatter-table");
  tbl.innerHTML = html;
  tbl.querySelectorAll("tr.clickable").forEach(tr => {
    tr.addEventListener("click", (ev) => {
      const name = tr.dataset.name;
      const shift = ev.shiftKey;
      const boxes = document.querySelectorAll("#exp-list input[type=checkbox]");
      if (shift) boxes.forEach(c => { if (c.dataset.name === name) c.checked = !c.checked; });
      else       boxes.forEach(c => { c.checked = (c.dataset.name === name); });
      drawCurves();
      document.getElementById("curve-chart").scrollIntoView({ behavior: "smooth", block: "center" });
    });
  });
  applySort();
}

function apiUrl() {
  const inp = document.getElementById("extrap-target");
  const t = inp ? parseInt(inp.value, 10) : NaN;
  if (!isFinite(t) || t <= 0) return "/api/data";
  const xKey = document.getElementById("curve-xaxis").value;
  if (xKey === "step") {
    const f = globalStepsPerEpoch();
    if (f && f > 0) return `/api/data?target=${Math.round(t / f)}`;
  }
  return `/api/data?target=${t}`;
}

async function load() {
  const r = await fetch(apiUrl());
  DATA = await r.json();
  document.getElementById("meta").textContent =
      `— ${DATA.experiments.length} entries, refreshed ${new Date().toLocaleTimeString()}`;
  populateCurveMetric();
  populateExpList();
  populateScatterAxes();
  updateFilterVisibility();
  syncExtrapLabel();
  drawCurves(); drawScatter();

  ["curve-metric","curve-split","curve-logx","curve-logy","avg-seeds"].forEach(id =>
    document.getElementById(id).addEventListener("change", () => { drawCurves(); drawScatter(); }));
  document.getElementById("curve-xaxis").addEventListener("change", () => {
    syncExtrapLabel();
    drawCurves(); drawScatter();
  });
  ["scatter-x","scatter-y","scatter-logx","scatter-logy","scatter-pareto"].forEach(id =>
    document.getElementById(id).addEventListener("change", drawScatter));
  document.getElementById("exp-filter").addEventListener("input", (ev) => {
    FILTER = ev.target.value.trim().toLowerCase();
    updateFilterVisibility(); drawCurves(); drawScatter();
  });
  // Extrapolation toggle: show/hide target-epoch input, redraw both panels.
  document.getElementById("extrap-on").addEventListener("change", (ev) => {
    document.getElementById("extrap-target-wrap").style.display =
      ev.target.checked ? "" : "none";
    drawCurves(); drawScatter();
  });
  // Editing the target recomputes extrapolation server-side, then redraws.
  document.getElementById("extrap-target").addEventListener("change", () => {
    if (!extrapOn()) return;  // no-op when extrap is off
    refresh();
  });
}

async function refresh() {
  const btn = document.getElementById("refresh-btn");
  if (btn) { btn.disabled = true; btn.textContent = "↻ refreshing…"; }
  try {
    const r = await fetch(apiUrl());
    const fresh = await r.json();
    DATA = { ...fresh, _colorMap: DATA ? DATA._colorMap : undefined };
    document.getElementById("meta").textContent =
        `— ${DATA.experiments.length} entries, refreshed ${new Date().toLocaleTimeString()}`;
    populateExpList();
    updateFilterVisibility();
    drawCurves(); drawScatter();
  } catch (e) { console.error(e); }
  finally { if (btn) { btn.disabled = false; btn.textContent = "↻ refresh"; } }
}

load();
setInterval(refresh, 60000);
</script>
</body>
</html>
"""


# ---------------------------------------------------------------------------
# Flask app
# ---------------------------------------------------------------------------

def create_app(base, target_epochs=TARGET_EPOCHS):
    app = Flask(__name__)
    base = os.path.abspath(base)

    @app.route("/")
    def index():
        return render_template_string(
            TEMPLATE,
            base_name=os.path.basename(base) or "root",
        )

    @app.route("/api/data")
    def api_data():
        # Allow the frontend to override the extrapolation target per request
        # via ?target=<int>. Falls back to the server default if unset.
        t = request.args.get("target", type=int)
        return jsonify(build_payload(base, target_epochs=(t or target_epochs)))

    return app


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--base", default=os.getcwd(),
                    help="directory to scan for fit folders (default: cwd)")
    ap.add_argument("--host", default="0.0.0.0", help="bind host")
    ap.add_argument("--port", default=5000, type=int, help="bind port")
    ap.add_argument("--target-epochs", default=TARGET_EPOCHS, type=int,
                    help="target epochs used for extrapolation of incomplete runs")
    ap.add_argument("--debug", action="store_true")
    args = ap.parse_args()

    base = os.path.abspath(args.base)
    if not os.path.isdir(base):
        raise SystemExit(f"base directory does not exist: {base}")

    fits = discover_fits(base)
    print(f"[grace_dashboard] scanning {base}")
    print(f"[grace_dashboard] discovered {len(fits)} fit folder(s), "
          f"{sum(len(f['seeds']) for f in fits)} seed(s) total")
    app = create_app(base, target_epochs=args.target_epochs)
    app.run(host=args.host, port=args.port, debug=args.debug)


if __name__ == "__main__":
    main()
