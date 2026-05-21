# Copyright (c) 2026 LightSeek Foundation
# MIT License

"""Phase 4 — multi-tensor connector / fetcher unit tests (no NCCL required).

These exercise the small, side-effect-free pieces:

* deterministic key ordering (``sorted_tensor_names``),
* env var helpers (``export_transfer_mode_env`` / readers),
* dtype normalisation (``_normalise_dtype``).

The full NCCL P2P round-trip lives in ``tests/colocate/test_p2p_dummy.py``
(Phase 3, single-tensor) and ``tests/colocate/test_p2p_multi_tensor.py``
(Phase 4, multi-tensor) — both Modal-only.
"""

from __future__ import annotations

import os

import pytest

torch = pytest.importorskip("torch")


def _real_torch() -> bool:
    try:
        t = torch.zeros(2)
        return hasattr(t, "shape") and tuple(t.shape) == (2,)
    except Exception:
        return False


pytestmark = pytest.mark.skipif(
    not _real_torch(), reason="requires real torch (conftest stubs on Mac dev box)"
)


# ----------------------------------------------------------------------
# Key ordering
# ----------------------------------------------------------------------


def test_sorted_tensor_names_alphabetic():
    """Both sides walk sorted(keys); insertion order must not matter."""
    from torchspec.inference.engine.nccl_hidden_states_connector import (
        sorted_tensor_names,
    )

    a = sorted_tensor_names(
        {"target_logits": None, "hidden_states": None, "aux_hidden_states": None}
    )
    b = sorted_tensor_names(
        {"hidden_states": None, "aux_hidden_states": None, "target_logits": None}
    )
    assert a == b == ["aux_hidden_states", "hidden_states", "target_logits"]


def test_sorted_tensor_names_handles_singleton():
    from torchspec.inference.engine.nccl_hidden_states_connector import (
        sorted_tensor_names,
    )

    assert sorted_tensor_names({"hidden_states": None}) == ["hidden_states"]


def test_fetcher_and_connector_agree_on_order():
    """Receiver and sender must both sort by key — same fn / equivalent fn."""
    from torchspec.inference.engine.nccl_hidden_states_connector import (
        sorted_tensor_names,
    )
    from torchspec.training.nccl_data_fetcher import _sorted_tensor_names

    keys = {"z": None, "a": None, "m": None}
    assert sorted_tensor_names(keys) == _sorted_tensor_names(keys)


# ----------------------------------------------------------------------
# Env var helpers
# ----------------------------------------------------------------------


def test_export_transfer_mode_env_round_trip(monkeypatch: pytest.MonkeyPatch):
    """The patch reads the same env var the engine writes."""
    from torchspec.inference.engine.nccl_hidden_states_connector import (
        PAIRED_TRAINER_RANK_ENV,
        TRANSFER_MODE_ENV,
        export_transfer_mode_env,
        read_paired_trainer_rank_env,
        read_transfer_mode_env,
    )

    monkeypatch.delenv(TRANSFER_MODE_ENV, raising=False)
    monkeypatch.delenv(PAIRED_TRAINER_RANK_ENV, raising=False)
    assert read_transfer_mode_env() is None
    assert read_paired_trainer_rank_env() is None

    export_transfer_mode_env(transfer_mode="nccl", paired_trainer_rank=3)
    assert read_transfer_mode_env() == "nccl"
    assert read_paired_trainer_rank_env() == 3
    # Cleanup — monkeypatch can't undo direct os.environ writes.
    os.environ.pop(TRANSFER_MODE_ENV, None)
    os.environ.pop(PAIRED_TRAINER_RANK_ENV, None)


def test_paired_trainer_rank_env_unset_returns_none(monkeypatch: pytest.MonkeyPatch):
    from torchspec.inference.engine.nccl_hidden_states_connector import (
        PAIRED_TRAINER_RANK_ENV,
        read_paired_trainer_rank_env,
    )

    monkeypatch.delenv(PAIRED_TRAINER_RANK_ENV, raising=False)
    assert read_paired_trainer_rank_env() is None


# ----------------------------------------------------------------------
# Dtype normalisation
# ----------------------------------------------------------------------


def test_normalise_dtype_accepts_torch_dtype():
    from torchspec.training.nccl_data_fetcher import _normalise_dtype

    assert _normalise_dtype(torch.bfloat16) is torch.bfloat16


def test_normalise_dtype_accepts_short_string():
    from torchspec.training.nccl_data_fetcher import _normalise_dtype

    assert _normalise_dtype("bfloat16") is torch.bfloat16
    assert _normalise_dtype("float32") is torch.float32


def test_normalise_dtype_accepts_torch_prefixed_string():
    """MooncakeDataFetcher metadata sometimes carries 'torch.bfloat16'."""
    from torchspec.training.nccl_data_fetcher import _normalise_dtype

    assert _normalise_dtype("torch.bfloat16") is torch.bfloat16


def test_normalise_dtype_rejects_garbage():
    from torchspec.training.nccl_data_fetcher import _normalise_dtype

    with pytest.raises(TypeError, match="unsupported tensor dtype"):
        _normalise_dtype(42)


# ----------------------------------------------------------------------
# Connector / fetcher pre-init guards
# ----------------------------------------------------------------------


def test_connector_requires_dist_initialised(monkeypatch: pytest.MonkeyPatch):
    """Constructor refuses to build a connector when torch.distributed is
    not initialised — this catches a class of test bugs where a stale
    fixture left state across cases."""
    import torch.distributed as tdist

    from torchspec.inference.engine.nccl_hidden_states_connector import (
        NcclHiddenStatesConnector,
    )

    if tdist.is_initialized():
        pytest.skip("torch.distributed already initialised in this process")

    with pytest.raises(RuntimeError, match="torch.distributed to be"):
        NcclHiddenStatesConnector(dst_global_rank=1)


def test_multi_tensor_fetcher_requires_dist_initialised(monkeypatch: pytest.MonkeyPatch):
    import torch.distributed as tdist

    from torchspec.training.nccl_data_fetcher import NcclMultiTensorFetcher

    if tdist.is_initialized():
        pytest.skip("torch.distributed already initialised in this process")

    with pytest.raises(RuntimeError, match="torch.distributed to be"):
        NcclMultiTensorFetcher(
            src_global_rank=0,
            device=torch.device("cuda", 0) if torch.cuda.is_available() else torch.device("cpu"),
        )


def test_multi_tensor_fetcher_rejects_cpu_device():
    import torch.distributed as tdist

    from torchspec.training.nccl_data_fetcher import NcclMultiTensorFetcher

    if tdist.is_initialized():
        pytest.skip("torch.distributed already initialised; can't construct without CUDA check")

    with pytest.raises(RuntimeError):
        NcclMultiTensorFetcher(src_global_rank=0, device=torch.device("cpu"))


# ----------------------------------------------------------------------
# ColocateTrainSample shape sanity
# ----------------------------------------------------------------------


def test_colocate_train_sample_dataclass_round_trip():
    """The dataclass is what ships through the Ray queue — make sure
    the tensor-spec shape is what NcclMultiTensorFetcher consumes."""
    from torchspec.training.data_fetcher import ColocateTrainSample

    sample = ColocateTrainSample(
        step_id=7,
        tensor_specs={
            "hidden_states": ((2, 8, 4096), torch.bfloat16),
            "aux_hidden_states": ((6, 8, 4096), torch.bfloat16),
        },
        packed_loss_mask="3,5",
        last_turn_loss_only=False,
        metadata={"data_id": "x"},
    )
    assert sample.step_id == 7
    assert "hidden_states" in sample.tensor_specs
    shape, dtype = sample.tensor_specs["hidden_states"]
    assert shape == (2, 8, 4096)
    assert dtype is torch.bfloat16
