# Copyright (c) 2026 LightSeek Foundation
# MIT License

"""Tests for Mooncake Store force delete + hard pin refactoring.

Depends on conftest.py (project root) to stub torch and mooncake when
running on environments without GPU dependencies (e.g. Mac dev machines).
"""

import os
import sys
from unittest.mock import MagicMock, patch

import pytest

from torchspec.config.mooncake_config import MooncakeConfig
from torchspec.transfer.mooncake.buffers import AsyncPutManager
from torchspec.transfer.mooncake.eagle_store import EagleMooncakeStore
from torchspec.transfer.mooncake.store import MooncakeHiddenStateStore


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _make_eagle_store(mock_raw_store):
    """Create an EagleMooncakeStore with a mocked internal _store."""
    config = MooncakeConfig()
    store = EagleMooncakeStore(config)
    store._store = mock_raw_store
    store._initialized = True
    return store


def _make_base_store(mock_raw_store, enable_hard_pin=False):
    """Create a MooncakeHiddenStateStore subclass with a mocked internal _store."""
    config = MooncakeConfig(enable_hard_pin=enable_hard_pin)

    class ConcreteStore(MooncakeHiddenStateStore):
        pass

    store = ConcreteStore(config)
    store._store = mock_raw_store
    return store


# ---------------------------------------------------------------------------
# Test 1: enable_hard_pin env roundtrip
# ---------------------------------------------------------------------------
class TestEnableHardPinConfig:
    def test_enable_hard_pin_env_roundtrip(self):
        config = MooncakeConfig(enable_hard_pin=True)
        assert config.enable_hard_pin is True

        with patch.dict(os.environ, {}, clear=False):
            config.export_env()
            assert os.environ["MOONCAKE_ENABLE_HARD_PIN"] == "1"
            restored = MooncakeConfig.from_env()
            assert restored.enable_hard_pin is True

    def test_enable_hard_pin_default_off(self):
        config = MooncakeConfig()
        assert config.enable_hard_pin is False

        with patch.dict(os.environ, {}, clear=False):
            config.export_env()
            assert os.environ["MOONCAKE_ENABLE_HARD_PIN"] == "0"
            restored = MooncakeConfig.from_env()
            assert restored.enable_hard_pin is False


# ---------------------------------------------------------------------------
# Tests 2-3: _verify_force_delete
# ---------------------------------------------------------------------------
class TestVerifyForceDelete:
    def test_missing_batch_remove_method(self):
        mock_raw = MagicMock(spec=[])  # no batch_remove attr
        store = _make_base_store(mock_raw)
        with pytest.raises(RuntimeError, match="batch_remove.*not found"):
            store._verify_force_delete()

    def test_missing_force_param(self):
        mock_raw = MagicMock()
        mock_raw.batch_remove = MagicMock()
        mock_raw.batch_remove.__doc__ = "batch_remove(keys) -> list[int]"
        store = _make_base_store(mock_raw)
        # Patch importlib.metadata.version to force fallback to docstring check
        with patch.dict(
            sys.modules, {"importlib.metadata": MagicMock(version=MagicMock(side_effect=Exception))}
        ):
            with pytest.raises(RuntimeError, match="batch_remove.*force.*not supported"):
                store._verify_force_delete()

    def test_valid_batch_remove(self):
        mock_raw = MagicMock()
        mock_raw.batch_remove = MagicMock()
        mock_raw.batch_remove.__doc__ = "batch_remove(keys, force=False) -> list[int]"
        store = _make_base_store(mock_raw)
        store._verify_force_delete()  # should not raise


# ---------------------------------------------------------------------------
# Tests 4-9: remove_eagle3_tensors
# ---------------------------------------------------------------------------
class TestRemoveEagle3Tensors:
    def test_success(self):
        mock_raw = MagicMock()
        mock_raw.batch_remove.return_value = [0, 0]
        store = _make_eagle_store(mock_raw)
        store.remove_eagle3_tensors("k1")
        mock_raw.batch_remove.assert_called_once_with(["k1_hs", "k1_ids"], force=True)

    def test_not_found_is_success(self):
        mock_raw = MagicMock()
        mock_raw.batch_remove.return_value = [-704, 0]
        store = _make_eagle_store(mock_raw)
        store.remove_eagle3_tensors("k1")
        assert mock_raw.batch_remove.call_count == 1

    def test_retry_then_succeed(self):
        mock_raw = MagicMock()
        mock_raw.batch_remove.side_effect = [
            [-1, 0],  # first: k1_hs fails
            [0],  # second: k1_hs succeeds
        ]
        store = _make_eagle_store(mock_raw)
        store.remove_eagle3_tensors("k1")
        assert mock_raw.batch_remove.call_count == 2

    def test_exhaust_retries(self):
        mock_raw = MagicMock()
        mock_raw.batch_remove.return_value = [-1, -1]
        store = _make_eagle_store(mock_raw)
        # Should not raise
        store.remove_eagle3_tensors("k1")
        assert mock_raw.batch_remove.call_count == 3

    def test_exception_is_retried(self):
        mock_raw = MagicMock()
        mock_raw.batch_remove.side_effect = [
            RuntimeError("connection lost"),
            [0, 0],
        ]
        store = _make_eagle_store(mock_raw)
        store.remove_eagle3_tensors("k1")
        assert mock_raw.batch_remove.call_count == 2

    def test_all_exceptions(self):
        mock_raw = MagicMock()
        mock_raw.batch_remove.side_effect = RuntimeError("down")
        store = _make_eagle_store(mock_raw)
        # Should not raise
        store.remove_eagle3_tensors("k1")
        assert mock_raw.batch_remove.call_count == 3


# ---------------------------------------------------------------------------
# Tests 10-11: cleanup does not mask put error
# ---------------------------------------------------------------------------
class TestCleanupDoesNotMaskPutError:
    def test_sync_put_cleanup_does_not_mask_put_error(self):
        mock_raw = MagicMock()
        mock_raw.batch_put_from.return_value = [-1, 0]
        mock_raw.batch_remove.side_effect = RuntimeError("cleanup failed")
        store = _make_eagle_store(mock_raw)
        store._replicate_config = None

        with pytest.raises(RuntimeError, match="batch_put_from failed"):
            store._do_sync_batch_put(["k_hs", "k_ids"], [100, 200], [64, 32])

    def test_async_put_cleanup_does_not_mask_put_error(self):
        mock_raw = MagicMock()
        mock_raw.batch_put_from.return_value = [-1, 0]
        mock_raw.batch_remove.side_effect = RuntimeError("cleanup failed")
        mgr = AsyncPutManager(store=mock_raw, max_workers=1)
        with pytest.raises(RuntimeError, match="async batch_put_from failed"):
            mgr._do_put(["k_hs", "k_ids"], [100, 200], [64, 32])
        mgr.shutdown()


# ---------------------------------------------------------------------------
# Tests 12-13: _build_replicate_config
# ---------------------------------------------------------------------------
class TestBuildReplicateConfig:
    def test_supported(self):
        mock_config_instance = MagicMock()
        mock_config_instance.with_hard_pin = False
        mock_store_module = MagicMock()
        mock_store_module.ReplicateConfig.return_value = mock_config_instance

        with patch.dict(sys.modules, {"mooncake.store": mock_store_module}):
            mock_raw = MagicMock()
            store = _make_base_store(mock_raw, enable_hard_pin=True)
            store._build_replicate_config()
            assert store._replicate_config is not None
            assert store._replicate_config.with_hard_pin is True

    def test_unsupported(self):
        mock_config_instance = MagicMock(spec=[])  # no with_hard_pin attr
        mock_store_module = MagicMock()
        mock_store_module.ReplicateConfig.return_value = mock_config_instance

        with patch.dict(sys.modules, {"mooncake.store": mock_store_module}):
            mock_raw = MagicMock()
            store = _make_base_store(mock_raw, enable_hard_pin=True)
            store._build_replicate_config()
            assert store._replicate_config is None


# ---------------------------------------------------------------------------
# Tests 14-15: replicate_config passed through put paths
# ---------------------------------------------------------------------------
class TestReplicateConfigPassthrough:
    def test_sync_put_passes_replicate_config(self):
        mock_raw = MagicMock()
        mock_raw.batch_put_from.return_value = [0, 0]
        store = _make_eagle_store(mock_raw)
        mock_cfg = MagicMock()
        store._replicate_config = mock_cfg

        store._do_sync_batch_put(["k_hs", "k_ids"], [100, 200], [64, 32])
        mock_raw.batch_put_from.assert_called_once_with(
            ["k_hs", "k_ids"], [100, 200], [64, 32], config=mock_cfg
        )

    def test_sync_put_no_config_when_none(self):
        mock_raw = MagicMock()
        mock_raw.batch_put_from.return_value = [0, 0]
        store = _make_eagle_store(mock_raw)
        store._replicate_config = None

        store._do_sync_batch_put(["k_hs", "k_ids"], [100, 200], [64, 32])
        mock_raw.batch_put_from.assert_called_once_with(["k_hs", "k_ids"], [100, 200], [64, 32])

    def test_async_put_passes_replicate_config(self):
        mock_raw = MagicMock()
        mock_raw.batch_put_from.return_value = [0, 0]
        mock_cfg = MagicMock()
        mgr = AsyncPutManager(store=mock_raw, max_workers=1, replicate_config=mock_cfg)
        mgr._do_put(["k_hs", "k_ids"], [100, 200], [64, 32])
        mock_raw.batch_put_from.assert_called_once_with(
            ["k_hs", "k_ids"], [100, 200], [64, 32], config=mock_cfg
        )
        mgr.shutdown()
