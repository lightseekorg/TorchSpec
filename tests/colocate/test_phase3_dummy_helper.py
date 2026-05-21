# Copyright (c) 2026 LightSeek Foundation
# MIT License

"""Phase 3 — dummy-tensor helper unit tests (no NCCL required).

The actual ``NcclDataFetcher.recv()`` path is exercised by the Modal
smoke test ``tests/colocate/test_p2p_dummy.py``. Here we only unit-test
the deterministic-tensor builder which does NOT touch torch.distributed.
"""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

# conftest stubs torch with MagicMock on Mac dev boxes; skip cleanly.
try:
    _has_real_torch = (
        bool(torch.cuda.is_available())
        or hasattr(torch, "arange")
        and callable(torch.arange)
        and not str(type(torch)).startswith("<class 'unittest.mock")
    )
except Exception:
    _has_real_torch = False


from torchspec.training.nccl_data_fetcher import make_dummy_tensor


def _real_torch() -> bool:
    """Detect whether torch is the real one or the conftest mock."""
    try:
        t = torch.zeros(2)
        return hasattr(t, "shape") and tuple(t.shape) == (2,)
    except Exception:
        return False


pytestmark = pytest.mark.skipif(
    not _real_torch(), reason="requires real torch (conftest stubs on Mac dev box)"
)


def test_make_dummy_tensor_shape_and_dtype():
    t = make_dummy_tensor((2, 3, 4), dtype=torch.float32, device=torch.device("cpu"))
    assert tuple(t.shape) == (2, 3, 4)
    assert t.dtype == torch.float32
    # Deterministic: arange(0..23) reshaped, no offset.
    assert t.flatten()[0].item() == 0.0
    assert t.flatten()[-1].item() == 23.0


def test_make_dummy_tensor_seed_offsets_every_element():
    a = make_dummy_tensor((4,), dtype=torch.float32, device=torch.device("cpu"), seed=0)
    b = make_dummy_tensor((4,), dtype=torch.float32, device=torch.device("cpu"), seed=7)
    # b == a + 7 elementwise
    diff = (b - a).tolist()
    assert all(abs(d - 7.0) < 1e-6 for d in diff)


def test_make_dummy_tensor_bf16_roundtrip():
    """bfloat16 has limited precision; verify we still get the documented
    values exactly for small ints (the integers up to 256 are
    representable exactly in bf16)."""
    t = make_dummy_tensor((8,), dtype=torch.bfloat16, device=torch.device("cpu"))
    expected = list(range(8))
    got = [int(x.item()) for x in t]
    assert got == expected


def test_make_dummy_tensor_total_size():
    t = make_dummy_tensor((2, 8, 4096), dtype=torch.bfloat16, device=torch.device("cpu"))
    assert tuple(t.shape) == (2, 8, 4096)
    assert t.numel() == 2 * 8 * 4096


def test_make_dummy_tensor_determinism():
    """Same args → byte-equal output (the whole point of using arange)."""
    a = make_dummy_tensor((3, 5), dtype=torch.float32, device=torch.device("cpu"), seed=42)
    b = make_dummy_tensor((3, 5), dtype=torch.float32, device=torch.device("cpu"), seed=42)
    assert torch.equal(a, b)


def test_nccl_data_fetcher_rejects_cpu_device():
    """The fetcher requires CUDA — sanity-check the precondition runs
    even on machines without CUDA, since constructing on CPU would
    silently work for a moment and then deadlock at recv time."""
    from torchspec.training.nccl_data_fetcher import NcclDataFetcher

    with pytest.raises(ValueError, match="requires a CUDA device"):
        NcclDataFetcher(
            src_rank=0,
            shape=(2, 4),
            dtype=torch.float32,
            device=torch.device("cpu"),
        )
