# Copyright (c) 2026 LightSeek Foundation
# MIT License

"""Phase 0 — config plumbing & feature flag.

These tests run on Mac dev boxes thanks to the root ``conftest.py`` torch
stubs. They cover the validator only; downstream behaviour (placement, MPS,
NCCL world) is covered by Phase 1+ smoke tests on Modal.
"""

from __future__ import annotations

import argparse

import pytest

from torchspec.colocate import (
    ColocateConfigError,
    is_colocate_enabled,
    validate_colocate_config,
)


def _baseline_disagg_args(**overrides):
    """Build a flat Namespace mirroring what ``parse_config`` produces.

    Default = today's behaviour: 4 trainer GPUs + 1 engine, mooncake transfer.
    """
    args = argparse.Namespace(
        colocate=False,
        colocate_strategy=None,
        transfer_mode="mooncake",
        train_frac=None,
        infer_frac=None,
        training_num_nodes=1,
        training_num_gpus_per_node=4,
        world_size=4,
        inference_num_gpus=1,
        inference_num_gpus_per_engine=1,
    )
    for k, v in overrides.items():
        setattr(args, k, v)
    return args


def _baseline_colocate_mps_args(**overrides):
    """Build a flat Namespace for the supported colocate=mps combination."""
    args = argparse.Namespace(
        colocate=True,
        colocate_strategy="mps",
        transfer_mode="nccl",
        train_frac=0.45,
        infer_frac=0.45,
        training_num_nodes=1,
        training_num_gpus_per_node=4,
        world_size=4,
        # 1 engine × TP=4 == 4 trainer ranks
        inference_num_gpus=4,
        inference_num_gpus_per_engine=4,
    )
    for k, v in overrides.items():
        setattr(args, k, v)
    return args


# ---------------------------------------------------------------------------
# Happy paths
# ---------------------------------------------------------------------------


def test_disagg_default_passes():
    args = _baseline_disagg_args()
    validate_colocate_config(args)
    assert not is_colocate_enabled(args)


def test_colocate_mps_supported_combination_passes():
    args = _baseline_colocate_mps_args()
    validate_colocate_config(args)
    assert is_colocate_enabled(args)


def test_legacy_colocate_true_with_mooncake_still_passes():
    """The pre-existing partial colocate path uses ``colocate=True`` without
    setting strategy. We keep it working so existing examples (and the
    upstream merged PR #81) don't regress."""
    args = _baseline_disagg_args(
        colocate=True,
        # 4 inf + 4 train would also be valid here, but we don't enforce the
        # 1:1 invariant unless strategy=mps.
        inference_num_gpus=4,
        inference_num_gpus_per_engine=4,
    )
    validate_colocate_config(args)
    assert is_colocate_enabled(args)


# ---------------------------------------------------------------------------
# Combination errors
# ---------------------------------------------------------------------------


def test_mps_with_mooncake_rejected():
    args = _baseline_colocate_mps_args(transfer_mode="mooncake")
    with pytest.raises(ColocateConfigError, match="requires transfer_mode='nccl'"):
        validate_colocate_config(args)


def test_unknown_strategy_rejected():
    args = _baseline_colocate_mps_args(colocate_strategy="bogus")
    with pytest.raises(ColocateConfigError, match="Unsupported colocate combination"):
        validate_colocate_config(args)


def test_nccl_without_strategy_rejected():
    """transfer_mode=nccl is only meaningful when strategy=mps."""
    args = _baseline_colocate_mps_args(colocate_strategy=None, colocate=True)
    with pytest.raises(ColocateConfigError, match="Unsupported colocate combination"):
        validate_colocate_config(args)


# ---------------------------------------------------------------------------
# Memory-fraction errors
# ---------------------------------------------------------------------------


def test_missing_train_frac_rejected():
    args = _baseline_colocate_mps_args(train_frac=None)
    with pytest.raises(ColocateConfigError, match="train_frac and training.infer_frac"):
        validate_colocate_config(args)


def test_missing_infer_frac_rejected():
    args = _baseline_colocate_mps_args(infer_frac=None)
    with pytest.raises(ColocateConfigError, match="train_frac and training.infer_frac"):
        validate_colocate_config(args)


def test_frac_sum_over_budget_rejected():
    args = _baseline_colocate_mps_args(train_frac=0.6, infer_frac=0.5)
    with pytest.raises(ColocateConfigError, match=r"> 1\.0"):
        validate_colocate_config(args)


def test_frac_at_budget_passes():
    """0.45 + 0.45 + 0.10 = 1.00 exactly should be accepted."""
    args = _baseline_colocate_mps_args(train_frac=0.45, infer_frac=0.45)
    validate_colocate_config(args)


@pytest.mark.parametrize("bad", [0.0, -0.1, 1.0, 1.5])
def test_frac_out_of_range_rejected(bad):
    args = _baseline_colocate_mps_args(train_frac=bad)
    with pytest.raises(ColocateConfigError, match=r"train_frac must be in \(0, 1\)"):
        validate_colocate_config(args)


# ---------------------------------------------------------------------------
# Topology errors
# ---------------------------------------------------------------------------


def test_engine_count_mismatch_rejected():
    """4 trainer ranks but 1 engine × TP=1 → 1 engine rank → mismatch."""
    args = _baseline_colocate_mps_args(
        inference_num_gpus=1,
        inference_num_gpus_per_engine=1,
    )
    with pytest.raises(ColocateConfigError, match=r"engine_count.*engine_tp_size"):
        validate_colocate_config(args)


def test_two_engines_each_tp2_matches_4_trainers():
    """2 engines × TP=2 == 4 trainer ranks should validate."""
    args = _baseline_colocate_mps_args(
        inference_num_gpus=4,
        inference_num_gpus_per_engine=2,
    )
    validate_colocate_config(args)


# ---------------------------------------------------------------------------
# Stray-field guard
# ---------------------------------------------------------------------------


def test_stray_train_frac_without_colocate_rejected():
    """If the user sets train_frac but forgets colocate, fail loudly rather
    than silently no-op."""
    args = _baseline_disagg_args(train_frac=0.4)
    with pytest.raises(ColocateConfigError, match="training.colocate=False"):
        validate_colocate_config(args)


def test_stray_strategy_without_colocate_rejected():
    args = _baseline_disagg_args(colocate_strategy="mps")
    # is_colocate_enabled returns True because strategy is set — this should
    # fall into the strategy-validation path and complain about the missing
    # fractions, not the stray-field path. Either error message is acceptable
    # for the user.
    with pytest.raises(ColocateConfigError):
        validate_colocate_config(args)
