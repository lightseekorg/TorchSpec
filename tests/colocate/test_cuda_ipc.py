# Copyright (c) 2026 LightSeek Foundation
# MIT License

"""Unit tests for the CUDA IPC transport plumbing.

CUDA IPC is the default colocate hidden-state transport; these tests
run on a Mac dev box (no real CUDA) and exercise the env-var contract
(default-on, opt-out via TORCHSPEC_COLOCATE_IPC=0) and the fail-fast
guard — not the actual IPC handle exchange (which needs two processes
on one GPU and is covered by the colocate e2e tests).
"""

from __future__ import annotations

import os

import pytest

from torchspec.colocate import cuda_ipc


@pytest.fixture(autouse=True)
def _clean():
    saved = {
        k: os.environ.get(k) for k in ("TORCHSPEC_COLOCATE_IPC", "TORCHSPEC_COLOCATE_IPC_PIPELINE")
    }
    cuda_ipc._reset_probe_cache_for_test()
    yield
    for k, v in saved.items():
        if v is None:
            os.environ.pop(k, None)
        else:
            os.environ[k] = v
    cuda_ipc._reset_probe_cache_for_test()


@pytest.mark.parametrize(
    "value,expected",
    [
        # Default-on: any value that is not an explicit disable token
        # (including an empty string and unrecognised junk) enables IPC.
        ("1", True),
        ("true", True),
        ("YES", True),
        ("garbage", True),
        ("", True),
        ("0", False),
        ("false", False),
        ("no", False),
        ("OFF", False),
    ],
)
def test_ipc_enabled_env_toggle(value, expected):
    os.environ["TORCHSPEC_COLOCATE_IPC"] = value
    assert cuda_ipc.ipc_enabled() is expected


def test_ipc_enabled_unset_defaults_on():
    os.environ.pop("TORCHSPEC_COLOCATE_IPC", None)
    assert cuda_ipc.ipc_enabled() is True


def test_ensure_ipc_usable_raises_when_probe_fails(monkeypatch):
    monkeypatch.setattr(
        cuda_ipc,
        "probe_ipc_capability",
        lambda: (False, "expandable_segments active"),
    )
    with pytest.raises(RuntimeError, match="expandable_segments active"):
        cuda_ipc.ensure_ipc_usable()


def test_ensure_ipc_usable_passes_when_probe_ok(monkeypatch):
    monkeypatch.setattr(cuda_ipc, "probe_ipc_capability", lambda: (True, "ok"))
    cuda_ipc.ensure_ipc_usable()  # must not raise


def test_probe_cache_reset_hook():
    cuda_ipc._probe_cache = (True, "stale")
    cuda_ipc._reset_probe_cache_for_test()
    assert cuda_ipc._probe_cache is None


# ---------------------------------------------------------------------------
# Pipelined transport opt-in (TORCHSPEC_COLOCATE_IPC_PIPELINE)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "value,expected",
    [
        # Opt-in: only an explicit enable token turns the pipeline on.
        ("1", True),
        ("true", True),
        ("YES", True),
        ("on", True),
        (" On ", True),
        # Anything else — including unset, empty, junk — leaves it off.
        ("0", False),
        ("false", False),
        ("garbage", False),
        ("", False),
    ],
)
def test_ipc_pipeline_enabled_env_toggle(value, expected):
    # IPC itself on (default) — the pipeline flag then decides.
    os.environ.pop("TORCHSPEC_COLOCATE_IPC", None)
    os.environ["TORCHSPEC_COLOCATE_IPC_PIPELINE"] = value
    assert cuda_ipc.ipc_pipeline_enabled() is expected


def test_ipc_pipeline_unset_defaults_off():
    os.environ.pop("TORCHSPEC_COLOCATE_IPC", None)
    os.environ.pop("TORCHSPEC_COLOCATE_IPC_PIPELINE", None)
    assert cuda_ipc.ipc_pipeline_enabled() is False


def test_ipc_pipeline_requires_ipc_enabled():
    """The pipeline is layered on CUDA IPC — disabling IPC disables it
    even when the pipeline flag is explicitly on."""
    os.environ["TORCHSPEC_COLOCATE_IPC"] = "0"
    os.environ["TORCHSPEC_COLOCATE_IPC_PIPELINE"] = "1"
    assert cuda_ipc.ipc_enabled() is False
    assert cuda_ipc.ipc_pipeline_enabled() is False


def test_ipc_pipeline_transport_rejects_bad_role():
    with pytest.raises(ValueError, match="role must be"):
        cuda_ipc.IpcPipelineTransport(role="banana")


@pytest.mark.parametrize("role", ["engine", "trainer"])
def test_ipc_pipeline_transport_flush_is_safe_before_use(role):
    """flush() on a fresh transport (no steps run) must be a harmless
    no-op for both roles — teardown may fire before any transfer."""
    cuda_ipc.IpcPipelineTransport(role=role).flush()


def test_ipc_pipeline_wrong_role_methods_raise():
    eng = cuda_ipc.IpcPipelineTransport(role="engine")
    trn = cuda_ipc.IpcPipelineTransport(role="trainer")
    with pytest.raises(RuntimeError, match="trainer_recv called on an engine-role"):
        eng.trainer_recv({}, src=0, device=None, group=None)
    with pytest.raises(RuntimeError, match="engine_send called on a trainer-role"):
        trn.engine_send({"x": object()}, dst=0, group=None)
