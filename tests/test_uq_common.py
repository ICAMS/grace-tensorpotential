"""Unit tests for ``tensorpotential.uq.cli._common`` helpers.

These cover the orchestration helpers reused across grace_uq subcommands
(``spawn_and_monitor``, ``resolve_threads_per_worker``,
``apply_master_thread_caps``, ``format_progress`` / ``parse_progress``).
The helpers are the refactor destinations for build.py's local copies.
"""

from __future__ import annotations

import os
import sys


from tensorpotential.uq.cli import _common as common


# ---------------------------------------------------------------------------
# resolve_threads_per_worker
# ---------------------------------------------------------------------------


def test_resolve_threads_per_worker_explicit_int():
    assert common.resolve_threads_per_worker(4, n_workers=2) == 4


def test_resolve_threads_per_worker_explicit_string_int():
    assert common.resolve_threads_per_worker("3", n_workers=2) == 3


def test_resolve_threads_per_worker_auto_divides_cpus(monkeypatch):
    monkeypatch.setattr(os, "cpu_count", lambda: 16)
    assert common.resolve_threads_per_worker("auto", n_workers=4) == 4


def test_resolve_threads_per_worker_auto_floors_at_one(monkeypatch):
    monkeypatch.setattr(os, "cpu_count", lambda: 2)
    # 2 cores / 8 workers → would be 0, must floor to 1
    assert common.resolve_threads_per_worker("auto", n_workers=8) == 1


def test_resolve_threads_per_worker_auto_handles_none_cpu_count(monkeypatch):
    monkeypatch.setattr(os, "cpu_count", lambda: None)
    assert common.resolve_threads_per_worker("auto", n_workers=2) == 1


# ---------------------------------------------------------------------------
# apply_master_thread_caps
# ---------------------------------------------------------------------------


_THREAD_ENV_VARS = (
    "OMP_NUM_THREADS",
    "MKL_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "NUMEXPR_MAX_THREADS",
)


def _clear_thread_env(monkeypatch):
    for var in _THREAD_ENV_VARS:
        monkeypatch.delenv(var, raising=False)


def test_apply_master_thread_caps_sets_env(monkeypatch):
    _clear_thread_env(monkeypatch)
    common.apply_master_thread_caps(3)
    for var in _THREAD_ENV_VARS:
        assert os.environ[var] == "3"


def test_apply_master_thread_caps_zero_is_noop(monkeypatch):
    _clear_thread_env(monkeypatch)
    common.apply_master_thread_caps(0)
    for var in _THREAD_ENV_VARS:
        assert var not in os.environ


def test_apply_master_thread_caps_negative_is_noop(monkeypatch):
    _clear_thread_env(monkeypatch)
    common.apply_master_thread_caps(-1)
    for var in _THREAD_ENV_VARS:
        assert var not in os.environ


# ---------------------------------------------------------------------------
# format_progress / parse_progress
# ---------------------------------------------------------------------------


def test_format_progress_starts_with_prefix():
    s = common.format_progress(3, 10)
    assert s.startswith(common.PROGRESS_PREFIX)


def test_progress_roundtrip():
    s = common.format_progress(7, 42)
    assert common.parse_progress(s) == (7, 42)


def test_parse_progress_returns_none_for_non_progress_line():
    assert common.parse_progress("hello world") is None
    assert common.parse_progress("") is None


def test_parse_progress_returns_none_for_malformed_progress():
    assert common.parse_progress(common.PROGRESS_PREFIX + " done=abc total=10") is None
    assert common.parse_progress(common.PROGRESS_PREFIX + " done=1") is None  # missing total


# ---------------------------------------------------------------------------
# spawn_and_monitor
# ---------------------------------------------------------------------------


def _trivial_cmd(stdout: str = "", exit_code: int = 0) -> list[str]:
    """Build a tiny python -c command that prints once then exits."""
    body = f"import sys; sys.stdout.write({stdout!r}); sys.exit({exit_code})"
    return [sys.executable, "-c", body]


def test_spawn_and_monitor_all_succeed_returns_empty():
    cmds = [(0, _trivial_cmd()), (1, _trivial_cmd())]
    failed = common.spawn_and_monitor(
        cmds, gpus=[""], threads_per_worker=0, verbose=False
    )
    assert failed == []


def test_spawn_and_monitor_collects_failures():
    cmds = [
        (0, _trivial_cmd(exit_code=0)),
        (1, _trivial_cmd(exit_code=2)),
        (2, _trivial_cmd(exit_code=3)),
    ]
    failed = common.spawn_and_monitor(
        cmds, gpus=[""], threads_per_worker=0, verbose=False
    )
    assert sorted(failed) == [1, 2]


def test_spawn_and_monitor_progress_handler_intercepts_progress_lines(capsys):
    progress_line = common.format_progress(3, 10)
    cmds = [(0, _trivial_cmd(stdout=progress_line + "\n"))]
    received = []

    failed = common.spawn_and_monitor(
        cmds,
        gpus=[""],
        threads_per_worker=0,
        verbose=True,
        progress_handler=lambda i, line: received.append((i, line)),
    )

    assert failed == []
    assert received == [(0, progress_line)]
    # Progress lines must be suppressed from stdout when handler is given.
    out = capsys.readouterr().out
    assert progress_line not in out


def test_spawn_and_monitor_forwards_progress_lines_to_stdout_when_no_handler(capsys):
    progress_line = common.format_progress(1, 5)
    cmds = [(0, _trivial_cmd(stdout=progress_line + "\n"))]
    common.spawn_and_monitor(
        cmds, gpus=[""], threads_per_worker=0, verbose=True
    )
    out = capsys.readouterr().out
    assert progress_line in out


def test_spawn_and_monitor_round_robins_cuda_visible_devices(tmp_path):
    """Each worker should see CUDA_VISIBLE_DEVICES = gpus[i % len(gpus)]."""
    out_files = [str(tmp_path / f"w{i}.txt") for i in range(4)]
    cmds = []
    for i, out_path in enumerate(out_files):
        body = (
            "import os\n"
            f"open({out_path!r}, 'w').write(os.environ.get('CUDA_VISIBLE_DEVICES', ''))\n"
        )
        cmds.append((i, [sys.executable, "-c", body]))

    common.spawn_and_monitor(
        cmds, gpus=["0", "1"], threads_per_worker=0, verbose=False
    )

    assert open(out_files[0]).read() == "0"
    assert open(out_files[1]).read() == "1"
    assert open(out_files[2]).read() == "0"
    assert open(out_files[3]).read() == "1"


def test_spawn_and_monitor_sets_thread_env_vars_when_positive(tmp_path):
    out_path = str(tmp_path / "env.txt")
    body = (
        "import os, json\n"
        f"open({out_path!r}, 'w').write(json.dumps({{v: os.environ.get(v) for v in "
        "('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS',"
        "'NUMEXPR_MAX_THREADS','TF_NUM_INTEROP_THREADS','TF_NUM_INTRAOP_THREADS')}))\n"
    )
    cmds = [(0, [sys.executable, "-c", body])]
    common.spawn_and_monitor(
        cmds, gpus=[""], threads_per_worker=4, verbose=False
    )

    import json
    env = json.loads(open(out_path).read())
    for var in (
        "OMP_NUM_THREADS",
        "MKL_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "NUMEXPR_MAX_THREADS",
        "TF_NUM_INTEROP_THREADS",
        "TF_NUM_INTRAOP_THREADS",
    ):
        assert env[var] == "4", f"{var} not propagated: {env}"


def test_spawn_and_monitor_label_appears_in_output(capsys):
    cmds = [(7, _trivial_cmd(stdout="hello\n"))]
    common.spawn_and_monitor(
        cmds, gpus=[""], threads_per_worker=0, verbose=True, label="Step3Worker"
    )
    out = capsys.readouterr().out
    assert "[Step3Worker 7]" in out
