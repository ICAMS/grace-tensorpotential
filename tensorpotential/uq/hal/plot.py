"""Diagnostics figure for one HAL walker."""

from __future__ import annotations

import json
import os

import numpy as np


def _hline(ax, y, label, style="--", color="k"):
    if y is not None:
        ax.axhline(y, ls=style, color=color, lw=0.8, label=label)


def _settings_lines(summary: dict, s: dict) -> list:
    """Short key-settings / outcome lines for the text panel.

    Missing keys are skipped: a killed walker's summary (``replot_dir``) has no outcome
    counters.
    """
    lines = []

    def add(label, key):
        if s.get(key) is not None:
            lines.append(f"{label}: {s[key]}")

    add("ensemble", "ensemble")
    add("temperature_K", "temperature_K")
    if summary.get("temperature_mean_K") is not None:
        lines.append(f"T_mean: {summary['temperature_mean_K']:.1f} K")
    add("n_steps", "n_steps")
    add("tau", "tau")
    if s.get("trigger_gamma") is not None:
        n_min = s.get("trigger_min_atoms") or 1
        extra = f" (>= {n_min} atoms)" if n_min > 1 else ""
        lines.append(f"trigger_gamma: {s['trigger_gamma']}{extra}")
    add("update_target", "update_target")
    add("reject_gamma", "reject_gamma")
    if s.get("update") is not None:
        upd = str(s["update"])
        lines.append(f"update: {upd}")
        if upd != "none" and s.get("update_inflate") is not None:
            lines.append(f"inflate: {s['update_inflate']}")
    lines.append(f"stop_reason: {summary.get('stop_reason', '?')}")
    lines.append(f"n_collected: {summary.get('n_collected', len(summary.get('collections', [])))}")
    rejected = summary.get("rejected") or {}
    lines.append("rejected: " + (", ".join(f"{k}={v}" for k, v in rejected.items()) or "none"))
    if summary.get("n_model_calls") is not None:
        lines.append(f"n_model_calls: {summary['n_model_calls']}")
    if summary.get("steps_per_s") is not None:
        lines.append(f"steps/s: {summary['steps_per_s']:.2f}")
    lines.append("")
    lines.append("full settings: hal_settings.json")
    return lines


def _note(ax, text):
    ax.text(0.5, 0.5, text, transform=ax.transAxes, ha="center", va="center",
            color="0.35", fontsize=10)
    ax.set_yticks([])


def plot_diagnostics(history: dict, summary: dict, path: str) -> str:
    # A bare Figure with an Agg canvas: no matplotlib.use() and no pyplot registry, so the
    # caller's backend (e.g. Jupyter inline) and open figures are left alone.
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    from matplotlib.figure import Figure

    s = summary.get("settings", {})
    step = np.asarray(history["step"])
    status = np.asarray(history["status"])
    col, rej = status == 1, status == 2
    cluster = s.get("ensemble") == "cluster" or np.all(np.isnan(history["v_ratio"]))

    # gamma spans the full top row, the settings box the full bottom row
    if cluster:  # gamma, tau_r, T, Rg(+wall), d_min, alpha (full row) + text
        fig = Figure(figsize=(14, 13), layout="constrained")
        gs = fig.add_gridspec(5, 2, height_ratios=[1, 1, 1, 0.8, 0.45])
        slots = [gs[0, :], gs[1, 0], gs[1, 1], gs[2, 0], gs[2, 1], gs[3, :]]
    else:  # gamma, tau_r, T, V, P, d_min, alpha + text
        fig = Figure(figsize=(14, 14), layout="constrained")
        gs = fig.add_gridspec(5, 2, height_ratios=[1, 1, 1, 1, 0.45])
        slots = [gs[0, :], gs[1, 0], gs[1, 1], gs[2, 0], gs[2, 1], gs[3, 0], gs[3, 1]]
    FigureCanvasAgg(fig)
    text_slot = gs[4, :]
    axes = []
    for sl in slots:
        axes.append(fig.add_subplot(sl, sharex=axes[0] if axes else None))
    g_ax, tau_ax, t_ax = axes[0], axes[1], axes[2]

    # --- gamma ---
    gmax = np.asarray(history["gamma_max"])
    g_ax.semilogy(step, gmax, lw=0.7, label="max γ (live)")
    g_ax.plot(step[col], gmax[col], "go", ms=4, mfc="none", label="collected (live)")
    g_ax.plot(step[rej], gmax[rej], "rx", ms=5, label="rejected (live)")
    cs = summary.get("collections", [])
    gp = np.asarray(history["gamma_pristine_max"], dtype=float)  # trigger steps only; NaN elsewhere
    pc, pr = col & ~np.isnan(gp), rej & ~np.isnan(gp)
    px_c, py_c, px_r, py_r = step[pc], gp[pc], step[pr], gp[pr]
    if len(px_c):
        g_ax.plot(px_c, py_c, "D", color="C2", ms=5, label="γ_pristine (collected)")
    if len(px_r):
        g_ax.plot(px_r, py_r, "v", color="C3", ms=5, label="γ_pristine (rejected)")
    n_min = s.get("trigger_min_atoms") or 1
    trigger_label = "trigger"
    if n_min > 1 and s.get("trigger_gamma") is not None:
        trigger_label = f"trigger: γ > {s['trigger_gamma']:g} on ≥ {n_min} atoms"
    _hline(g_ax, s.get("trigger_gamma"), trigger_label)
    _hline(g_ax, s.get("update_target"), "update target", ":", "g")
    _hline(g_ax, s.get("reject_gamma"), "reject", "-.", "r")
    g_ax.set_ylabel("max γ")
    g_ax.legend(fontsize=7, loc="lower left", bbox_to_anchor=(0, 1.01), ncol=4, frameon=False)

    # --- tau_r ---
    tau_ax.plot(step, history["tau_r"], lw=0.7)
    tau_ax.set_ylabel("τ_r (bias strength)")

    # --- temperature ---
    t_ax.plot(step, history["temperature"], lw=0.7, label="T")
    _hline(t_ax, s.get("temperature_K"), "target")
    t_ax.set_ylabel("T (K)")
    t_ax.legend(fontsize=7, loc="best")

    # --- volume / Rg (+ pressure or wall energy) ---
    if cluster:
        rg_ax, dmin_ax, alpha_ax = axes[3], axes[4], axes[5]
        rg_ax.plot(step, history["rg_ratio"], lw=0.7)
        rg_ax.set_ylabel("Rg/Rg0")
        wall = np.asarray(history["wall_energy"], dtype=float)
        tw = rg_ax.twinx()
        tw.plot(step, wall, "C1", lw=0.7)
        hi = max(float(np.nanmax(wall)) * 1.05, 0.05) if np.any(np.isfinite(wall)) else 0.05
        tw.set_ylim(-0.02 * hi, hi)
        tw.set_ylabel("wall energy (eV)", color="C1")
    else:
        v_ax, p_ax, dmin_ax, alpha_ax = axes[3], axes[4], axes[5], axes[6]
        v_ax.plot(step, history["v_ratio"], lw=0.7)
        if s.get("ensemble") == "nvt_ramp" and s.get("nvt_ramp") and s.get("n_steps"):
            v0, v1 = s["nvt_ramp"]
            v_ax.plot(step, v0 + (v1 - v0) * np.minimum(step, s["n_steps"]) / s["n_steps"],
                      "k--", lw=0.8)
        if s.get("ensemble") in ("npt", "npt_stress") and s.get("volume_guard"):
            for y in s["volume_guard"]:
                _hline(v_ax, y, None, ":", "r")
        v_ax.set_ylabel("V/V0")
        p_ax.plot(step, history["pressure_gpa"], lw=0.7)
        p_ax.set_ylabel("pressure (GPa)")

    # --- d_min + bond ratio ---
    dmin_ax.plot(step, history["d_min"], lw=0.7)
    dmin_ax.set_ylabel("d_min (Å)")
    tw = dmin_ax.twinx()
    tw.plot(step, history["bond_ratio_min"], "C1", lw=0.7)
    tw.axhline(1.0, color="r", ls=":", lw=0.8)
    tw.set_ylabel("min d / d_reject", color="C1")

    # --- alpha ---
    plain = [c for c in cs if not c.get("n_inflate") and c.get("step") is not None]
    infl = [c for c in cs if c.get("n_inflate") and c.get("step") is not None]
    alpha_ax.set_ylim(0, 1.05)
    alpha_ax.set_ylabel("α (update)")
    if s.get("update") == "none":
        _note(alpha_ax, "no CALM update")
    elif not plain and not infl:
        _note(alpha_ax, "no collections")
    else:
        if plain:
            alpha_ax.plot([c["step"] for c in plain], [c["alpha"] for c in plain], "o",
                          label="α search")
        if infl:
            alpha_ax.plot([c["step"] for c in infl], [c["alpha"] for c in infl], "s", color="C3",
                          label="α = 1 + inflation")
            for c in infl:
                alpha_ax.annotate(f"×{c['n_inflate']}", (c["step"], c["alpha"]),
                                  textcoords="offset points", xytext=(0, 6), ha="center",
                                  fontsize=7)
        alpha_ax.legend(fontsize=7, loc="best")

    # x tick labels / label only on the lowest plot of each column. A panel is "covered"
    # only by a panel of the same width below it (other widths have other x scales), and
    # the full-width gamma panel always keeps its own step axis, so collection steps can
    # be read on it directly.
    def _cols(a):
        return a.get_subplotspec().colspan

    for a in axes:
        ca = _cols(a)
        below = [b for b in axes if b is not a
                 and b.get_subplotspec().rowspan.start > a.get_subplotspec().rowspan.start
                 and _cols(b) == ca]
        if below and a is not g_ax:
            a.tick_params(labelbottom=False)
        else:
            a.tick_params(labelbottom=True)
            a.set_xlabel("step")

    # settings / outcome box: full width, key: value lines in several columns
    tx = fig.add_subplot(text_slot)
    tx.set_xticks([])
    tx.set_yticks([])
    lines = [ln for ln in _settings_lines(summary, s) if ln and not ln.startswith("full settings")]
    # Four columns fit lines of up to ~36 monospace characters; fall back to three
    # (e.g. several rejection reasons) so that columns never overlap.
    n_col = 4 if max(len(ln) for ln in lines) <= 36 else 3
    per_col = -(-len(lines) // n_col)
    for i in range(n_col):
        chunk = lines[i * per_col:(i + 1) * per_col]
        if chunk:
            tx.text(0.012 + i / n_col, 0.95, "\n".join(chunk), va="top", family="monospace",
                    fontsize=10, transform=tx.transAxes)
    tx.text(0.012, 0.05, "full settings: hal_settings.json", va="bottom", family="monospace",
            fontsize=9, color="0.35", transform=tx.transAxes)

    fig.savefig(path, dpi=110)
    return path


def replot_dir(output_dir: str) -> str:
    """Rebuild hal_diagnostics.png from the files of a (possibly killed) walker."""
    with np.load(os.path.join(output_dir, "hal_history.npz")) as h:
        history = {k: h[k] for k in h.files}
    summary_path = os.path.join(output_dir, "hal_summary.json")
    if os.path.exists(summary_path):
        with open(summary_path) as fh:
            summary = json.load(fh)
    else:  # killed walker: hal_settings.json is written before the run starts
        with open(os.path.join(output_dir, "hal_settings.json")) as fh:
            summary = {"stop_reason": "unfinished", "settings": json.load(fh)}
        frames_path = os.path.join(output_dir, "hal_collected.extxyz")
        if os.path.exists(frames_path):
            from ase.io import read

            summary["collections"] = [
                {"step": f.info["step"], "gamma_max_pristine": f.info["gamma_max_pristine"],
                 "alpha": f.info["alpha"], "n_inflate": f.info["n_inflate"]}
                for f in read(frames_path, ":")
            ]
    return plot_diagnostics(history, summary, os.path.join(output_dir, "hal_diagnostics.png"))
