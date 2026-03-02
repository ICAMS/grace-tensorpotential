"""Tests for tensorpotential.cli.wizard."""
from __future__ import annotations

import os
import sys
import pytest

import tensorpotential.cli.wizard as wizard
from tensorpotential.cli.wizard import (
    _WizardState,
    _is_fs_model,
    _apply_state,
    _section_optimizer,
    _section_loss,
    _section_weighting,
)


# ---------------------------------------------------------------------------
# Minimal template for _apply_state tests
# ---------------------------------------------------------------------------

_MINIMAL_TEMPLATE = """\
cutoff: {{CUTOFF}}
data:
  filename: {{TRAIN_FILENAME}}
  {{TEST_DATA}}
  save_dataset: {{SAVE_DATASET}}
potential:
  {{POTENTIAL_SETTINGS}}
fit:
  loss: {
    energy: { weight: {{ENERGY_LOSS_WEIGHT}}, type: {{LOSS_TYPE}} {{EXTRA_E_ARGS}} },
    forces: { weight: {{FORCE_LOSS_WEIGHT}}, type: {{LOSS_TYPE}} {{EXTRA_E_ARGS}} },
    {{STRESS_LOSS}}
    {{SWITCH_LOSS}}
  }
  {{WEIGHTING_SCHEME}}
  target_total_updates: {{TARGET_TOTAL_UPDATES}}
{{OPTIMIZER_BLOCK}}
  compute_convex_hull: {{COMPUTE_CONVEX_HULL}}
  batch_size: {{BATCH_SIZE}}
  test_batch_size: {{TEST_BATCH_SIZE}}
  eval_init_stats: {{eval_init_stats}}
  reset_epoch_and_step: {{RESET_EPOCH}}
"""


def _make_state(**kwargs) -> _WizardState:
    """Create a _WizardState with sensible defaults for testing."""
    s = _WizardState()
    s.train_filename = "train.pkl.gz"
    s.preset_name = "GRACE_1LAYER_latest"
    s.cutoff = 6.0
    s.preset_kwargs_str = "{}"
    s.fit_type = "fit from scratch"
    s.finetune_model = False
    for k, v in kwargs.items():
        setattr(s, k, v)
    return s


# ---------------------------------------------------------------------------
# Unit tests: _is_fs_model
# ---------------------------------------------------------------------------


def test_is_fs_model_preset():
    s = _make_state(preset_name="FS")
    assert _is_fs_model(s) is True


def test_is_fs_model_foundation():
    s = _make_state(preset_name="", foundation_model_name="GRACE-FS-OAM")
    assert _is_fs_model(s) is True


def test_is_fs_model_non_fs():
    s = _make_state(preset_name="GRACE_1LAYER_latest")
    assert _is_fs_model(s) is False


def test_is_fs_model_non_fs_foundation():
    s = _make_state(preset_name="", foundation_model_name="GRACE-2L-OAM")
    assert _is_fs_model(s) is False


# ---------------------------------------------------------------------------
# Unit tests: _apply_state
# ---------------------------------------------------------------------------


def test_apply_state_adam():
    s = _make_state(
        optimizer="Adam",
        learning_rate=0.008,
        use_switch=False,
    )
    result = _apply_state(s, _MINIMAL_TEMPLATE)
    assert "optimizer: Adam" in result
    assert "scheduler: cosine_decay" in result
    assert "learning_rate: 0.008" in result
    # No L-BFGS-B content
    assert "maxcor" not in result


def test_apply_state_lbfgsb():
    s = _make_state(
        optimizer="L-BFGS-B",
        bfgs_maxcor=100,
    )
    result = _apply_state(s, _MINIMAL_TEMPLATE)
    assert "optimizer: L-BFGS-B" in result
    assert "maxcor" in result
    # No Adam-specific content
    assert "scheduler:" not in result
    assert "learning_rate:" not in result


def test_apply_state_bfgs():
    s = _make_state(
        optimizer="BFGS",
        bfgs_maxcor=100,
    )
    result = _apply_state(s, _MINIMAL_TEMPLATE)
    assert "optimizer: BFGS" in result
    assert "maxcor" in result
    assert "scheduler:" not in result
    assert "learning_rate:" not in result


def test_apply_state_adam_with_switch():
    s = _make_state(
        optimizer="Adam",
        learning_rate=0.001,
        use_switch=True,
        switch_after=0.75,
        lr_reduction_factor=0.1,
        new_energy_w=128,
        new_force_w=32,
    )
    result = _apply_state(s, _MINIMAL_TEMPLATE)
    assert "scheduler: reduce_on_plateau" in result
    assert "switch:" in result


def test_apply_state_no_switch():
    s = _make_state(optimizer="Adam", use_switch=False)
    result = _apply_state(s, _MINIMAL_TEMPLATE)
    # The {{SWITCH_LOSS}} placeholder should be replaced with empty string
    assert "{{SWITCH_LOSS}}" not in result
    assert "switch:" not in result


def test_apply_state_no_dangling_placeholders():
    """After applying state, no {{...}} placeholders should remain."""
    s = _make_state(optimizer="L-BFGS-B")
    result = _apply_state(s, _MINIMAL_TEMPLATE)
    assert "{{" not in result, f"Found unresolved placeholders in:\n{result}"


def test_apply_state_adam_no_dangling_placeholders():
    s = _make_state(optimizer="Adam", use_switch=False)
    result = _apply_state(s, _MINIMAL_TEMPLATE)
    assert "{{" not in result, f"Found unresolved placeholders in:\n{result}"


# ---------------------------------------------------------------------------
# Section-level tests (monkeypatched _ask_*)
# ---------------------------------------------------------------------------


@pytest.fixture
def silence_output(monkeypatch):
    """Suppress print-based output helpers."""
    monkeypatch.setattr(wizard, "_info", lambda msg: None)
    monkeypatch.setattr(wizard, "_success", lambda msg: None)
    monkeypatch.setattr(wizard, "_section", lambda title: None)


def _patch_ask_defaults(monkeypatch):
    """Patch all _ask_* to return their defaults."""
    monkeypatch.setattr(wizard, "_ask_select", lambda msg, choices, default=None: default)
    monkeypatch.setattr(wizard, "_ask_text", lambda msg, default=None: default)
    monkeypatch.setattr(wizard, "_ask_path", lambda msg: "data.pkl.gz")
    monkeypatch.setattr(wizard, "_ask_confirm", lambda msg, default=True: default)


def test_section_optimizer_fs_scratch_defaults_bfgs(monkeypatch, silence_output):
    _patch_ask_defaults(monkeypatch)
    s = _make_state(preset_name="FS", fit_type="fit from scratch")
    s = _section_optimizer(s)
    assert s.optimizer == "BFGS"


def test_section_optimizer_fs_finetune_defaults_lbfgsb(monkeypatch, silence_output):
    _patch_ask_defaults(monkeypatch)
    s = _make_state(preset_name="FS", fit_type="finetune foundation model", finetune_model=True)
    s = _section_optimizer(s)
    assert s.optimizer == "L-BFGS-B"


def test_section_optimizer_fs_continue_defaults_lbfgsb(monkeypatch, silence_output):
    _patch_ask_defaults(monkeypatch)
    s = _make_state(preset_name="FS", fit_type="continue fit", finetune_model=True)
    s = _section_optimizer(s)
    assert s.optimizer == "L-BFGS-B"


def test_section_optimizer_fs_foundation_finetune_defaults_lbfgsb(monkeypatch, silence_output):
    _patch_ask_defaults(monkeypatch)
    s = _make_state(
        preset_name="",
        foundation_model_name="GRACE-FS-OAM",
        fit_type="finetune foundation model",
        finetune_model=True,
    )
    s = _section_optimizer(s)
    assert s.optimizer == "L-BFGS-B"


def test_section_optimizer_grace_defaults_adam(monkeypatch, silence_output):
    _patch_ask_defaults(monkeypatch)
    s = _make_state(preset_name="GRACE_1LAYER_latest", fit_type="fit from scratch")
    s = _section_optimizer(s)
    assert s.optimizer == "Adam"


def test_section_loss_no_switch_for_bfgs(monkeypatch, silence_output):
    """For BFGS optimizer, switch prompt is skipped and use_switch forced False."""
    # Patch _ask_confirm to always return True — if the switch prompt is NOT skipped,
    # use_switch would be True; we verify it stays False for BFGS.
    monkeypatch.setattr(wizard, "_ask_select", lambda msg, choices, default=None: default)
    monkeypatch.setattr(wizard, "_ask_text", lambda msg, default=None: default)
    monkeypatch.setattr(wizard, "_ask_confirm", lambda msg, default=True: True)
    s = _make_state(optimizer="BFGS")
    s = _section_loss(s)
    assert s.use_switch is False


def test_section_loss_no_switch_for_lbfgsb(monkeypatch, silence_output):
    monkeypatch.setattr(wizard, "_ask_select", lambda msg, choices, default=None: default)
    monkeypatch.setattr(wizard, "_ask_text", lambda msg, default=None: default)
    monkeypatch.setattr(wizard, "_ask_confirm", lambda msg, default=True: True)
    s = _make_state(optimizer="L-BFGS-B")
    s = _section_loss(s)
    assert s.use_switch is False


def test_section_loss_switch_allowed_for_adam(monkeypatch, silence_output):
    """For Adam, the switch prompt is asked and respected."""
    monkeypatch.setattr(wizard, "_ask_select", lambda msg, choices, default=None: default)
    monkeypatch.setattr(wizard, "_ask_text", lambda msg, default=None: default)
    # Return True for the switch confirm
    monkeypatch.setattr(wizard, "_ask_confirm", lambda msg, default=True: True)
    s = _make_state(optimizer="Adam")
    s = _section_loss(s)
    assert s.use_switch is True


def test_section_weighting_bfgs_scratch_defaults(monkeypatch, silence_output):
    """For BFGS + from scratch, default iterations should be 500."""
    captured = {}

    def fake_ask_text(msg, default=None):
        captured[msg] = default
        return default

    monkeypatch.setattr(wizard, "_ask_select", lambda msg, choices, default=None: default)
    monkeypatch.setattr(wizard, "_ask_text", fake_ask_text)
    monkeypatch.setattr(wizard, "_ask_confirm", lambda msg, default=True: default)

    s = _make_state(optimizer="BFGS", fit_type="fit from scratch")
    s = _section_weighting(s)
    assert s.target_total_updates == 500


def test_section_weighting_bfgs_finetune_defaults(monkeypatch, silence_output):
    """For BFGS + finetune, default iterations should be 100."""
    monkeypatch.setattr(wizard, "_ask_select", lambda msg, choices, default=None: default)
    monkeypatch.setattr(wizard, "_ask_text", lambda msg, default=None: default)
    monkeypatch.setattr(wizard, "_ask_confirm", lambda msg, default=True: default)

    s = _make_state(optimizer="BFGS", fit_type="finetune foundation model", finetune_model=True)
    s = _section_weighting(s)
    assert s.target_total_updates == 100


def test_section_weighting_adam_scratch_defaults(monkeypatch, silence_output):
    """For Adam + from scratch, default total updates should be 50000."""
    monkeypatch.setattr(wizard, "_ask_select", lambda msg, choices, default=None: default)
    monkeypatch.setattr(wizard, "_ask_text", lambda msg, default=None: default)
    monkeypatch.setattr(wizard, "_ask_confirm", lambda msg, default=True: default)

    s = _make_state(optimizer="Adam", fit_type="fit from scratch")
    s = _section_weighting(s)
    assert s.target_total_updates == 50000


# ---------------------------------------------------------------------------
# End-to-end test
# ---------------------------------------------------------------------------


def test_e2e_fs_lbfgsb(monkeypatch, tmp_path):
    """FS from scratch with L-BFGS-B chosen → valid input.yaml without scheduler/LR."""

    def fake_select(msg, choices, default=None):
        if "Optimizer" in msg:
            return "L-BFGS-B"
        if "Fit type" in msg:
            return "fit from scratch"
        if "preset" in msg.lower() or "Model preset" in msg:
            return "FS"
        # For complexity/size selections, return default
        return default

    monkeypatch.setattr(wizard, "_ask_select", fake_select)
    monkeypatch.setattr(wizard, "_ask_text", lambda msg, default=None: default)
    monkeypatch.setattr(wizard, "_ask_path", lambda msg: "train.pkl.gz")
    monkeypatch.setattr(wizard, "_ask_confirm", lambda msg, default=True: default)
    monkeypatch.setattr(wizard, "_print_header", lambda: None)
    monkeypatch.setattr(wizard, "_show_review", lambda s: None)
    monkeypatch.setattr(wizard, "_info", lambda msg: None)
    monkeypatch.setattr(wizard, "_success", lambda msg: None)
    monkeypatch.setattr(wizard, "_section", lambda title: None)

    # Force non-questionary path for the review loop
    monkeypatch.setattr(wizard, "_HAS_QUESTIONARY", False)
    # Provide "" to the review loop input → triggers "ok" (confirm)
    monkeypatch.setattr("builtins.input", lambda prompt="": "")

    monkeypatch.chdir(tmp_path)

    with pytest.raises(SystemExit) as exc_info:
        wizard.generate_template_input()

    assert exc_info.value.code == 0
    output_file = tmp_path / "input.yaml"
    assert output_file.exists(), "input.yaml was not written"
    content = output_file.read_text()
    assert "optimizer: L-BFGS-B" in content, "Expected L-BFGS-B optimizer in output"
    assert "scheduler:" not in content, "Unexpected scheduler key for L-BFGS-B"
    assert "learning_rate:" not in content, "Unexpected learning_rate key for L-BFGS-B"


def test_e2e_adam_has_scheduler(monkeypatch, tmp_path):
    """Adam optimizer → input.yaml should contain scheduler and learning_rate."""

    monkeypatch.setattr(wizard, "_ask_select", lambda msg, choices, default=None: default)
    monkeypatch.setattr(wizard, "_ask_text", lambda msg, default=None: default)
    monkeypatch.setattr(wizard, "_ask_path", lambda msg: "train.pkl.gz")
    monkeypatch.setattr(wizard, "_ask_confirm", lambda msg, default=True: default)
    monkeypatch.setattr(wizard, "_print_header", lambda: None)
    monkeypatch.setattr(wizard, "_show_review", lambda s: None)
    monkeypatch.setattr(wizard, "_info", lambda msg: None)
    monkeypatch.setattr(wizard, "_success", lambda msg: None)
    monkeypatch.setattr(wizard, "_section", lambda title: None)

    monkeypatch.setattr(wizard, "_HAS_QUESTIONARY", False)
    monkeypatch.setattr("builtins.input", lambda prompt="": "")

    monkeypatch.chdir(tmp_path)

    with pytest.raises(SystemExit) as exc_info:
        wizard.generate_template_input()

    assert exc_info.value.code == 0
    content = (tmp_path / "input.yaml").read_text()
    assert "optimizer: Adam" in content
    assert "scheduler:" in content
    assert "learning_rate:" in content
