"""Tests for the default environment forwarded to TorchSpec Ray actors."""

from torchspec.utils.env import get_torchspec_env_vars


def test_vllm_v2_model_runner_is_enabled_by_default(monkeypatch):
    monkeypatch.delenv("VLLM_USE_V2_MODEL_RUNNER", raising=False)

    env = get_torchspec_env_vars()

    assert env["VLLM_USE_V2_MODEL_RUNNER"] == "1"


def test_explicit_vllm_model_runner_value_is_forwarded(monkeypatch):
    monkeypatch.setenv("VLLM_USE_V2_MODEL_RUNNER", "0")

    env = get_torchspec_env_vars()

    assert env["VLLM_USE_V2_MODEL_RUNNER"] == "0"
