"""Tests for scripts.gracemaker._detect_multigpu_intent early CLI parsing."""

import os

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

import pytest

from tensorpotential.scripts.gracemaker import (
    _VALUE_TAKING_FLAGS,
    _detect_multigpu_intent,
)


def test_value_taking_flags_in_sync_with_real_parser():
    """_VALUE_TAKING_FLAGS must list exactly the value-taking options of
    cli.gracemaker.build_parser (the early detector cannot import the real
    parser without pulling in tensorflow, so the list is duplicated)."""
    from tensorpotential.cli.gracemaker import build_parser

    expected = {
        opt
        for action in build_parser()._actions
        if action.option_strings and action.nargs != 0
        for opt in action.option_strings
    }
    assert _VALUE_TAKING_FLAGS == expected


@pytest.fixture
def mirrored_yaml(tmp_path):
    p = tmp_path / "mirrored.yaml"
    p.write_text("fit:\n  strategy: mirrored\n")
    return str(p)


def test_multigpu_flag_on_argv(mirrored_yaml):
    assert _detect_multigpu_intent(["-m"])
    assert _detect_multigpu_intent(["--multigpu"])
    assert _detect_multigpu_intent([mirrored_yaml, "-m"])


def test_yaml_strategy_detected_regardless_of_option_order(mirrored_yaml):
    assert _detect_multigpu_intent([mirrored_yaml])
    assert _detect_multigpu_intent([mirrored_yaml, "--seed", "42"])
    # flag values before the positional must not be mistaken for the YAML path
    assert _detect_multigpu_intent(["--seed", "42", mirrored_yaml])
    assert _detect_multigpu_intent(["-cn", "ckpt", mirrored_yaml])
    assert _detect_multigpu_intent(["--checkpoint-name", "ckpt", mirrored_yaml])
    assert _detect_multigpu_intent(
        ["-l", "log.txt", "--seed", "42", "-cn", "ckpt", mirrored_yaml]
    )


def test_no_mirrored_strategy(tmp_path):
    p = tmp_path / "plain.yaml"
    p.write_text("fit:\n  strategy: default\n")
    assert not _detect_multigpu_intent([str(p)])
    assert not _detect_multigpu_intent(["--seed", "42", str(p)])


def test_missing_yaml_returns_false(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    assert not _detect_multigpu_intent(["--seed", "42"])
