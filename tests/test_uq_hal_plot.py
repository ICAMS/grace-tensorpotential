import os
import warnings

import numpy as np

from tensorpotential.uq.hal.collector import HISTORY_FIELDS
from tensorpotential.uq.hal.plot import plot_diagnostics

_SUMMARY = {
    "settings": {"trigger_gamma": 1.5, "update_target": 1.25, "reject_gamma": 10,
                 "temperature_K": 300, "ensemble": "nvt", "update": "alpha_search",
                 "update_inflate": True, "trigger_min_atoms": 3},
    "collections": [{"step": 10, "gamma_max_pristine": 1.7, "alpha": 0.1, "n_inflate": 0},
                    {"step": 30, "gamma_max_pristine": 1.9, "alpha": 1.0, "n_inflate": 2}],
    "rejected": {"gamma": 2, "bond": 1}, "n_model_calls": 77, "temperature_mean_K": 312.345,
    "stop_reason": "max_steps",
}


def _history(n=50, cluster=False):
    h = {k: np.random.default_rng(0).uniform(0.5, 2.0, n) for k in HISTORY_FIELDS}
    h["step"] = np.arange(n, dtype=float)
    h["status"] = np.zeros(n)
    h["status"][[10, 30]] = 1
    h["status"][20] = 2
    h["gamma_pristine_max"] = np.full(n, np.nan)
    h["gamma_pristine_max"][[10, 20, 30]] = [1.7, 12.0, 1.9]
    if cluster:
        h["v_ratio"][:] = np.nan
    return h


def _plot(tmp_path, monkeypatch, history, summary):
    """plot_diagnostics without warnings; returns the Figure (built without pyplot)."""
    from matplotlib.figure import Figure

    saved, orig = [], Figure.savefig

    def spy(self, *a, **k):
        saved.append(self)
        return orig(self, *a, **k)

    monkeypatch.setattr(Figure, "savefig", spy)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        out = plot_diagnostics(history, summary, str(tmp_path / "d.png"))
    assert os.path.getsize(out) > 10_000
    return saved[0]


def _has_content(ax):
    return bool(ax.lines or ax.collections or ax.texts or ax.patches)


def _assert_gamma_panel(fig):
    """gamma panel: own labelled step axis, gamma_pristine at collected + rejected triggers."""
    g = next(a for a in fig.axes if a.get_ylabel() == "max γ")
    fig.canvas.draw()
    assert g.get_xlabel() == "step"
    assert any(t.get_text() for t in g.get_xticklabels() if t.get_visible())
    pri = [ln for ln in g.get_lines() if str(ln.get_label()).startswith("γ_pristine")]
    assert sorted(x for ln in pri for x in ln.get_xdata()) == [10, 20, 30]


def test_plot_periodic_layout(tmp_path, monkeypatch):
    fig = _plot(tmp_path, monkeypatch, _history(), _SUMMARY)
    shown = [a for a in fig.axes if a.axison]
    assert all(_has_content(a) for a in shown)
    assert {a.get_ylabel() for a in shown} >= {"max γ", "τ_r (bias strength)", "T (K)", "V/V0",
                                                "pressure (GPa)", "d_min (Å)", "α (update)"}
    _assert_gamma_panel(fig)


def test_plot_cluster_layout(tmp_path, monkeypatch):
    summary = {**_SUMMARY, "collections": [],
               "settings": {**_SUMMARY["settings"], "ensemble": "cluster", "update": "none"}}
    fig = _plot(tmp_path, monkeypatch, _history(cluster=True), summary)
    shown = [a for a in fig.axes if a.axison]
    assert all(_has_content(a) for a in shown)
    ylabels = [a.get_ylabel() for a in shown]
    assert "wall energy (eV)" in ylabels and "Rg/Rg0" in ylabels
    assert "V/V0" not in ylabels and "pressure (GPa)" not in ylabels
    alpha = next(a for a in shown if a.get_ylabel() == "α (update)")
    assert [t.get_text() for t in alpha.texts] == ["no CALM update"]
    _assert_gamma_panel(fig)


def test_plot_settings_text(tmp_path, monkeypatch):
    fig = _plot(tmp_path, monkeypatch, _history(), _SUMMARY)
    texts = " ".join(t.get_text() for a in fig.axes for t in a.texts)
    for line in ("trigger_gamma: 1.5 (>= 3 atoms)", "rejected: gamma=2, bond=1",
                 "n_model_calls: 77", "T_mean: 312.3 K", "update: alpha_search", "inflate: True",
                 "stop_reason: max_steps", "full settings: hal_settings.json"):
        assert line in texts, line


def test_plot_leaves_backend_and_pyplot_registry_alone(tmp_path):
    import matplotlib
    import matplotlib.pyplot as plt

    before_backend = matplotlib.get_backend()
    plt.switch_backend("svg")  # any non-Agg backend
    try:
        keep = plt.figure()
        nums = plt.get_fignums()
        plot_diagnostics(_history(), _SUMMARY, str(tmp_path / "b.png"))
        assert matplotlib.get_backend() == "svg"
        assert plt.get_fignums() == nums
        assert os.path.getsize(tmp_path / "b.png") > 10_000
        plt.close(keep)
    finally:
        plt.switch_backend(before_backend)
