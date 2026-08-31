# Copyright (c) 2026 LightSeek Foundation
# MIT License

"""Inference-side contract for asynchronous Mooncake publication."""

import sys
from concurrent.futures import Future
from types import ModuleType
from unittest.mock import MagicMock, patch

import pytest

from torchspec.config.mooncake_config import MooncakeConfig
from torchspec.inference.engine.mooncake_hidden_states_connector import (
    MooncakeHiddenStatesConnector,
)
from torchspec.transfer.mooncake.buffers import AsyncPutManager, HostBuffer
from torchspec.transfer.mooncake.eagle_store import EagleMooncakeStore
from torchspec.transfer.mooncake.store import MooncakeHiddenStateStore


class _ConcreteStore(MooncakeHiddenStateStore):
    pass


def _make_store(raw_store, **config_kwargs):
    store = _ConcreteStore(MooncakeConfig(**config_kwargs))
    store._store = raw_store
    store._initialized = True
    store._init_event.set()
    return store


def test_connector_wait_for_save_checks_errors_without_flush():
    connector = object.__new__(MooncakeHiddenStatesConnector)
    connector._mooncake_store = MagicMock()

    connector.wait_for_save()

    connector._mooncake_store.check_async_errors.assert_called_once_with()
    connector._mooncake_store.flush.assert_not_called()


def test_eagle_raw_put_validates_then_delegates():
    store = object.__new__(EagleMooncakeStore)
    store._ensure_initialized = MagicMock()
    store._put_raw_tensors = MagicMock()
    keys = ["req_layer2_hs", "req_layer2_ids"]
    tensors = [MagicMock(), MagicMock()]

    store.put_raw_tensors(keys, tensors)

    store._ensure_initialized.assert_called_once_with()
    store._put_raw_tensors.assert_called_once_with(keys, tensors)


def test_async_error_check_does_not_wait_for_running_put():
    manager = AsyncPutManager(MagicMock(), max_workers=1)
    pending = Future()
    manager._in_flight[123] = pending

    manager.check_errors()

    assert manager._in_flight[123] is pending
    pending.set_result(None)
    manager.shutdown()


def test_async_error_check_surfaces_completed_failure():
    manager = AsyncPutManager(MagicMock(), max_workers=1)
    failed = Future()
    failed.set_exception(RuntimeError("put failed"))
    manager._in_flight[123] = failed

    with pytest.raises(RuntimeError, match="put failed"):
        manager.check_errors()

    assert 123 not in manager._in_flight
    manager.shutdown()


def test_store_check_async_errors_delegates_without_drain():
    store = _make_store(MagicMock())
    store._async_put_manager = MagicMock()

    store.check_async_errors()

    store._async_put_manager.check_errors.assert_called_once_with()
    store._async_put_manager.drain.assert_not_called()


def test_batch_exists_uses_single_metadata_census():
    raw_store = MagicMock(spec=["batch_is_exist", "is_exist"])
    raw_store.batch_is_exist.return_value = [1, 0, True]
    store = _make_store(raw_store)

    assert store.batch_exists(["a", "b", "c"]) == {
        "a": True,
        "b": False,
        "c": True,
    }
    raw_store.batch_is_exist.assert_called_once_with(["a", "b", "c"])
    raw_store.is_exist.assert_not_called()


def test_wait_for_keys_retries_metadata_only_until_complete():
    raw_store = MagicMock(spec=["batch_is_exist"])
    raw_store.batch_is_exist.side_effect = [[1, 0], [1, 1]]
    store = _make_store(raw_store)

    with patch("torchspec.transfer.mooncake.store.time.sleep") as sleep:
        store.wait_for_keys(["layer0", "layer1"], timeout=1.0, poll_interval=0.01)

    assert raw_store.batch_is_exist.call_count == 2
    sleep.assert_called_once_with(0.01)


def test_wait_for_keys_timeout_names_every_missing_fragment():
    raw_store = MagicMock(spec=["batch_is_exist"])
    raw_store.batch_is_exist.return_value = [1, 0, 0]
    store = _make_store(raw_store)

    with patch(
        "torchspec.transfer.mooncake.store.time.monotonic",
        side_effect=[10.0, 10.2],
    ):
        with pytest.raises(TimeoutError, match=r"missing: layer1, layer2"):
            store.wait_for_keys(
                ["layer0", "layer1", "layer2"],
                timeout=0.1,
                poll_interval=0.01,
            )


def test_eagle_get_waits_for_all_keys_before_moving_bytes():
    trace = []
    store = object.__new__(EagleMooncakeStore)
    store._initialized = True
    store._init_event = MagicMock()
    store._init_event.is_set.return_value = True
    store._gpu_direct_available = False
    store._gpu_receive_buffer = None
    store.wait_for_keys = MagicMock(side_effect=lambda keys: trace.append(("wait", keys)))
    store._get_tensors_via_host_buffer = MagicMock(
        side_effect=lambda keys, specs, device: (
            trace.append(("get", keys)) or {"hidden_states": "hs", "input_ids": "ids"}
        )
    )

    target_module = ModuleType("torchspec.models.target.eagle3_target_model")

    class Eagle3TargetOutput:
        def __init__(self, **kwargs):
            self.__dict__.update(kwargs)

    target_module.Eagle3TargetOutput = Eagle3TargetOutput
    with patch.dict(
        sys.modules,
        {"torchspec.models.target.eagle3_target_model": target_module},
    ):
        store.get(
            "req",
            shapes={"hidden_states": (2, 3), "input_ids": (2,)},
            dtypes={},
            device="cuda",
        )

    assert trace == [
        ("wait", ["req_hs", "req_ids"]),
        ("get", ["req_hs", "req_ids"]),
    ]


def test_host_buffer_copy_is_non_blocking():
    copied = {}

    class View:
        def copy_(self, source, **kwargs):
            copied["source"] = source
            copied.update(kwargs)

    class Storage:
        def __getitem__(self, item):
            return View()

    class Tensor:
        def contiguous(self):
            return self

        def numel(self):
            return 4

        def element_size(self):
            return 2

        def view(self, *args):
            return self

    buffer = object.__new__(HostBuffer)
    buffer.size = 32
    buffer._tensor = Storage()

    assert buffer.copy_from_tensor(Tensor(), offset=4) == 8
    assert copied["non_blocking"] is True
