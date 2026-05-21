# Copyright (c) 2026 LightSeek Foundation
# MIT License

"""Phase 1 — MPS lifecycle helper unit tests.

These tests run without NVIDIA drivers by mocking ``subprocess.run`` and
``shutil.which``. They cover env-var construction, idempotency, and the
"daemon already running" race-recovery branch. The actual *behavioural*
test (does MPS really get started? do trainer + engine see each other?)
runs on Modal as part of `phase1_placement` — see
`tests/colocate/test_placement.py` (added in the next sub-task).
"""

from __future__ import annotations

import os
import subprocess

import pytest

from torchspec.colocate import mps as mps_mod

# ---------------------------------------------------------------------------
# mps_client_env
# ---------------------------------------------------------------------------


def test_mps_client_env_default_pipe_and_log():
    env = mps_mod.mps_client_env()
    assert env == {
        "CUDA_MPS_PIPE_DIRECTORY": mps_mod.DEFAULT_PIPE_DIR,
        "CUDA_MPS_LOG_DIRECTORY": mps_mod.DEFAULT_LOG_DIR,
    }


def test_mps_client_env_custom_paths():
    env = mps_mod.mps_client_env(pipe_dir="/tmp/pipe", log_dir="/tmp/log")
    assert env["CUDA_MPS_PIPE_DIRECTORY"] == "/tmp/pipe"
    assert env["CUDA_MPS_LOG_DIRECTORY"] == "/tmp/log"


# ---------------------------------------------------------------------------
# is_mps_available
# ---------------------------------------------------------------------------


def test_is_mps_available_true_when_in_path(monkeypatch):
    monkeypatch.setattr(mps_mod.shutil, "which", lambda binary: "/usr/bin/" + binary)
    assert mps_mod.is_mps_available() is True


def test_is_mps_available_false_when_missing(monkeypatch):
    monkeypatch.setattr(mps_mod.shutil, "which", lambda binary: None)
    assert mps_mod.is_mps_available() is False


# ---------------------------------------------------------------------------
# is_mps_running
# ---------------------------------------------------------------------------


def test_is_mps_running_via_pipe_file(tmp_path, monkeypatch):
    # If the named pipe ``control`` exists, we should detect a daemon
    # without invoking pgrep.
    pipe_dir = tmp_path / "nvidia-mps"
    pipe_dir.mkdir()
    (pipe_dir / "control").write_text("")  # placeholder file

    # If we even reach pgrep that's a bug — fail loudly.
    def _no_subprocess(*a, **kw):
        raise AssertionError("pgrep must not be called when pipe file exists")

    monkeypatch.setattr(mps_mod.subprocess, "run", _no_subprocess)
    assert mps_mod.is_mps_running(pipe_dir=str(pipe_dir)) is True


def test_is_mps_running_via_pgrep(tmp_path, monkeypatch):
    # No pipe file → fallback to pgrep. Return rc=0 (process found).
    pipe_dir = tmp_path / "no-pipe"
    monkeypatch.setattr(mps_mod.shutil, "which", lambda b: "/usr/bin/" + b)

    def _fake_run(args, **kwargs):
        assert args[0] == "pgrep"
        return subprocess.CompletedProcess(args=args, returncode=0, stdout=b"", stderr=b"")

    monkeypatch.setattr(mps_mod.subprocess, "run", _fake_run)
    assert mps_mod.is_mps_running(pipe_dir=str(pipe_dir)) is True


def test_is_mps_running_false_when_neither(tmp_path, monkeypatch):
    pipe_dir = tmp_path / "no-pipe"
    monkeypatch.setattr(mps_mod.shutil, "which", lambda b: "/usr/bin/" + b)

    def _fake_run(args, **kwargs):
        return subprocess.CompletedProcess(args=args, returncode=1, stdout=b"", stderr=b"")

    monkeypatch.setattr(mps_mod.subprocess, "run", _fake_run)
    assert mps_mod.is_mps_running(pipe_dir=str(pipe_dir)) is False


# ---------------------------------------------------------------------------
# start_mps_daemon
# ---------------------------------------------------------------------------


def test_start_mps_daemon_raises_when_binary_missing(monkeypatch):
    monkeypatch.setattr(mps_mod, "is_mps_available", lambda: False)
    with pytest.raises(FileNotFoundError, match="not found on PATH"):
        mps_mod.start_mps_daemon()


def test_start_mps_daemon_idempotent_when_running(tmp_path, monkeypatch):
    monkeypatch.setattr(mps_mod, "is_mps_available", lambda: True)
    monkeypatch.setattr(mps_mod, "is_mps_running", lambda pipe_dir=None: True)

    def _no_subprocess(*a, **kw):
        raise AssertionError("must not exec when daemon is already running")

    monkeypatch.setattr(mps_mod.subprocess, "run", _no_subprocess)

    handle = mps_mod.start_mps_daemon(pipe_dir=str(tmp_path / "p"))
    assert handle.started_by_us is False
    assert handle.pipe_dir == str(tmp_path / "p")


def test_start_mps_daemon_runs_subprocess(tmp_path, monkeypatch):
    pipe_dir = tmp_path / "pipe"
    log_dir = tmp_path / "log"

    monkeypatch.setattr(mps_mod, "is_mps_available", lambda: True)
    monkeypatch.setattr(mps_mod, "is_mps_running", lambda pipe_dir=None: False)

    captured = {}

    def _fake_run(args, **kwargs):
        captured["args"] = args
        captured["env"] = kwargs.get("env", {})
        # Simulate the real daemon's behaviour: it creates the control
        # pipe under pipe_dir before returning. start_mps_daemon polls
        # for this file post-spawn (see mps.py), so the unit test must
        # produce it or block on the 10-second deadline.
        pipe_dir_str = kwargs.get("env", {}).get("CUDA_MPS_PIPE_DIRECTORY", "")
        if pipe_dir_str:
            os.makedirs(pipe_dir_str, exist_ok=True)
            with open(os.path.join(pipe_dir_str, "control"), "w") as f:
                f.write("")
        return subprocess.CompletedProcess(args=args, returncode=0, stdout=b"", stderr=b"")

    monkeypatch.setattr(mps_mod.subprocess, "run", _fake_run)

    handle = mps_mod.start_mps_daemon(pipe_dir=str(pipe_dir), log_dir=str(log_dir))
    assert handle.started_by_us is True
    assert pipe_dir.exists() and log_dir.exists()
    assert captured["args"] == ["nvidia-cuda-mps-control", "-d"]
    assert captured["env"]["CUDA_MPS_PIPE_DIRECTORY"] == str(pipe_dir)
    assert captured["env"]["CUDA_MPS_LOG_DIRECTORY"] == str(log_dir)


def test_start_mps_daemon_handles_already_running_race(tmp_path, monkeypatch):
    """If is_mps_running() said False but the binary later complains about
    an existing daemon, we recover gracefully (race between detection and
    spawn)."""
    monkeypatch.setattr(mps_mod, "is_mps_available", lambda: True)
    monkeypatch.setattr(mps_mod, "is_mps_running", lambda pipe_dir=None: False)

    def _fake_run(args, **kwargs):
        raise subprocess.CalledProcessError(
            returncode=1,
            cmd=args,
            output=b"",
            stderr=b"MPS daemon already running\n",
        )

    monkeypatch.setattr(mps_mod.subprocess, "run", _fake_run)

    handle = mps_mod.start_mps_daemon(pipe_dir=str(tmp_path / "p"))
    assert handle.started_by_us is False  # didn't actually start


def test_start_mps_daemon_propagates_real_failure(tmp_path, monkeypatch):
    monkeypatch.setattr(mps_mod, "is_mps_available", lambda: True)
    monkeypatch.setattr(mps_mod, "is_mps_running", lambda pipe_dir=None: False)

    def _fake_run(args, **kwargs):
        raise subprocess.CalledProcessError(
            returncode=2,
            cmd=args,
            output=b"",
            stderr=b"permission denied\n",
        )

    monkeypatch.setattr(mps_mod.subprocess, "run", _fake_run)

    with pytest.raises(RuntimeError, match="permission denied"):
        mps_mod.start_mps_daemon(pipe_dir=str(tmp_path / "p"))


# ---------------------------------------------------------------------------
# stop_mps_daemon
# ---------------------------------------------------------------------------


def test_stop_mps_daemon_no_op_when_unavailable(monkeypatch):
    monkeypatch.setattr(mps_mod, "is_mps_available", lambda: False)
    assert mps_mod.stop_mps_daemon() is False


def test_stop_mps_daemon_no_op_when_not_running(monkeypatch):
    monkeypatch.setattr(mps_mod, "is_mps_available", lambda: True)
    monkeypatch.setattr(mps_mod, "is_mps_running", lambda pipe_dir=None: False)

    def _no_subprocess(*a, **kw):
        raise AssertionError("must not exec when no daemon is running")

    monkeypatch.setattr(mps_mod.subprocess, "run", _no_subprocess)
    assert mps_mod.stop_mps_daemon() is False


def test_stop_mps_daemon_sends_quit(monkeypatch):
    monkeypatch.setattr(mps_mod, "is_mps_available", lambda: True)
    monkeypatch.setattr(mps_mod, "is_mps_running", lambda pipe_dir=None: True)

    captured = {}

    def _fake_run(args, **kwargs):
        captured["args"] = args
        captured["input"] = kwargs.get("input")
        return subprocess.CompletedProcess(args=args, returncode=0, stdout=b"", stderr=b"")

    monkeypatch.setattr(mps_mod.subprocess, "run", _fake_run)

    assert mps_mod.stop_mps_daemon() is True
    assert captured["args"] == ["nvidia-cuda-mps-control"]
    assert captured["input"] == b"quit\n"


def test_stop_mps_daemon_swallows_timeout(monkeypatch):
    monkeypatch.setattr(mps_mod, "is_mps_available", lambda: True)
    monkeypatch.setattr(mps_mod, "is_mps_running", lambda pipe_dir=None: True)

    def _fake_run(*args, **kwargs):
        raise subprocess.TimeoutExpired(cmd="nvidia-cuda-mps-control", timeout=5)

    monkeypatch.setattr(mps_mod.subprocess, "run", _fake_run)

    # Must NOT raise — cleanup is best-effort.
    assert mps_mod.stop_mps_daemon() is False


# ---------------------------------------------------------------------------
# setup_for_colocate (one-shot convenience)
# ---------------------------------------------------------------------------


def test_setup_for_colocate_returns_handle_and_env(tmp_path, monkeypatch):
    monkeypatch.setattr(mps_mod, "is_mps_available", lambda: True)
    monkeypatch.setattr(mps_mod, "is_mps_running", lambda pipe_dir=None: True)

    # The MPS-server probe spawns a CUDA subprocess (cuInit + cuDeviceGetCount)
    # to detect hosts where the daemon comes up but the per-GPU server can't
    # actually create a CUDA context. That's runtime/integration behaviour,
    # not unit-test territory; this Mac dev box has no CUDA, so the probe
    # would fail and (correctly) cause setup_for_colocate to return
    # ``(None, {})``. Disable the probe so we exercise just the
    # daemon-bring-up + env-var construction logic this test cares about.
    handle, env = mps_mod.setup_for_colocate(
        pipe_dir=str(tmp_path / "pipe"),
        log_dir=str(tmp_path / "log"),
        probe_server=False,
    )
    assert handle is not None
    assert handle.pipe_dir == str(tmp_path / "pipe")
    assert env["CUDA_MPS_PIPE_DIRECTORY"] == str(tmp_path / "pipe")
    assert env["CUDA_MPS_LOG_DIRECTORY"] == str(tmp_path / "log")


def test_setup_for_colocate_falls_back_when_probe_fails(tmp_path, monkeypatch):
    """When the MPS server probe reports failure (Modal sandbox / no
    --ipc=host), setup returns ``(None, {})`` instead of raising."""
    monkeypatch.setattr(mps_mod, "is_mps_available", lambda: True)
    monkeypatch.setattr(mps_mod, "is_mps_running", lambda pipe_dir=None: True)
    monkeypatch.setattr(
        mps_mod,
        "_probe_mps_server_works",
        lambda pipe_dir, log_dir, **kw: (False, "operation not supported"),
    )

    handle, env = mps_mod.setup_for_colocate(
        pipe_dir=str(tmp_path / "pipe"),
        log_dir=str(tmp_path / "log"),
    )
    assert handle is None
    assert env == {}
