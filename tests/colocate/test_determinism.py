# Copyright (c) 2026 LightSeek Foundation
# MIT License

"""Unit tests for the deterministic-seed plumbing.

These run on a Mac dev box (no torch) thanks to conftest's stubs — they
exercise the env-var contract and the pure-Python seeding (random /
PYTHONHASHSEED / CUBLAS env), not the torch/CUDA path.
"""

from __future__ import annotations

import os
import random

import pytest

from torchspec.colocate.determinism import is_grad_parity_mode, seed_everything

_ENV = "TORCHSPEC_GRAD_PARITY"


@pytest.fixture(autouse=True)
def _clean_env():
    """Snapshot and restore the env vars these tests poke."""
    saved = {k: os.environ.get(k) for k in (_ENV, "PYTHONHASHSEED", "CUBLAS_WORKSPACE_CONFIG")}
    yield
    for k, v in saved.items():
        if v is None:
            os.environ.pop(k, None)
        else:
            os.environ[k] = v


@pytest.mark.parametrize(
    "value,expected",
    [
        ("1", True),
        ("true", True),
        ("TRUE", True),
        ("yes", True),
        ("0", False),
        ("false", False),
        ("", False),
        ("nope", False),
    ],
)
def test_is_grad_parity_mode_env_toggle(value, expected):
    os.environ[_ENV] = value
    assert is_grad_parity_mode() is expected


def test_is_grad_parity_mode_unset():
    os.environ.pop(_ENV, None)
    assert is_grad_parity_mode() is False


def test_seed_everything_sets_pythonhashseed():
    seed_everything(1234)
    assert os.environ["PYTHONHASHSEED"] == "1234"


def test_seed_everything_seeds_python_random():
    seed_everything(7)
    first = [random.random() for _ in range(5)]
    seed_everything(7)
    second = [random.random() for _ in range(5)]
    assert first == second, "python random not reproducibly seeded"


def test_seed_everything_strict_sets_cublas_env():
    os.environ.pop("CUBLAS_WORKSPACE_CONFIG", None)
    seed_everything(42, strict=True)
    assert os.environ.get("CUBLAS_WORKSPACE_CONFIG") == ":4096:8"


def test_seed_everything_non_strict_skips_cublas_env():
    os.environ.pop("CUBLAS_WORKSPACE_CONFIG", None)
    os.environ.pop(_ENV, None)  # strict defaults to is_grad_parity_mode()
    seed_everything(42, strict=False)
    assert "CUBLAS_WORKSPACE_CONFIG" not in os.environ


def test_seed_everything_strict_defaults_to_grad_parity_mode():
    os.environ.pop("CUBLAS_WORKSPACE_CONFIG", None)
    os.environ[_ENV] = "1"
    seed_everything(42)  # strict=None -> picks up TORCHSPEC_GRAD_PARITY
    assert os.environ.get("CUBLAS_WORKSPACE_CONFIG") == ":4096:8"


def test_seed_everything_idempotent():
    seed_everything(99)
    seed_everything(99)  # second call must not raise
    assert os.environ["PYTHONHASHSEED"] == "99"
