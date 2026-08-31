"""Completed-request publication lifecycle for the vLLM Mooncake connector."""

from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
import torch

from torchspec.inference.engine.mooncake_hidden_states_connector import (
    MooncakeConnectorMetadata,
    MooncakeHiddenStatesConnector,
    _PendingSave,
)


def _scheduler_connector() -> MooncakeHiddenStatesConnector:
    connector = MooncakeHiddenStatesConnector.__new__(MooncakeHiddenStatesConnector)
    connector._cache_layer_group_id = 1
    connector._request_token_ids = {"req": [10, 20, 30]}
    connector._req_metadata = {"req": {"mooncake_key": "req"}}
    connector._pending_saves = {}
    connector._num_training_layers = 3
    connector._hidden_size = 8
    connector.num_hidden_states = 4
    connector._pp_size = 2
    connector._layer_ids = [2, 46, 90, 93]
    connector._num_target_layers = 93
    return connector


def test_request_finished_queues_completed_save_and_delays_blocks():
    connector = _scheduler_connector()
    request = SimpleNamespace(request_id="req", prompt_token_ids=[10, 20, 30])

    delay_free, metadata = connector.request_finished(request, [7, 11])

    assert delay_free is True
    assert metadata == {"mooncake_key": "req"}
    pending = connector._pending_saves["req"]
    assert pending.req_id == "req"
    assert pending.token_ids.tolist() == [10, 20, 30]
    assert pending.block_ids == [7, 11]


def test_request_finished_all_groups_selects_hidden_state_group():
    connector = _scheduler_connector()
    request = SimpleNamespace(request_id="req", prompt_token_ids=[10, 20, 30])

    connector.request_finished_all_groups(request, ([1], [7, 11], [99]))

    assert connector._pending_saves["req"].block_ids == [7, 11]


def test_build_connector_meta_moves_pending_saves_to_worker():
    connector = _scheduler_connector()
    pending = _PendingSave("req", torch.tensor([1, 2]), [3])
    connector._pending_saves = {"req": pending}
    scheduler_output = SimpleNamespace(scheduled_new_reqs=[])

    metadata = connector.build_connector_meta(scheduler_output)

    assert metadata.pending_saves == [pending]
    assert connector._pending_saves == {}


def test_save_kv_layer_is_noop_for_latest_vllm_lifecycle():
    connector = MooncakeHiddenStatesConnector.__new__(MooncakeHiddenStatesConnector)
    connector.save_kv_layer("cache_only_layers.93", MagicMock(), MagicMock())


def test_publish_pending_saves_pack_pipeline_fragments(monkeypatch):
    connector = MooncakeHiddenStatesConnector.__new__(MooncakeHiddenStatesConnector)
    connector._block_size = 3
    connector._kv_cache = torch.arange(2 * 4 * 3 * 2, dtype=torch.float16).view(2, 4, 3, 2)
    connector._pp_size = 2
    connector._layer_ids = [2, 46, 90, 93]
    connector._mooncake_store = MagicMock()
    connector._mooncake_store.config.host_buffer_size = 1024
    monkeypatch.setattr(connector, "_ensure_mooncake_store", lambda: True)
    monkeypatch.setattr(connector, "_local_layer_positions", lambda: [0, 1])
    pending = _PendingSave("req", torch.tensor([10, 20, 30]), [1])

    connector._publish_pending_saves([pending])

    connector._mooncake_store.put_raw_tensors.assert_called_once()
    keys, tensors = connector._mooncake_store.put_raw_tensors.call_args.args
    assert keys == [
        "req_layer2_hs",
        "req_layer2_ids",
        "req_layer46_hs",
        "req_layer46_ids",
    ]
    assert tensors[0].shape == (3, 2)
    assert tensors[0].dtype == torch.bfloat16
    assert tensors[1].tolist() == [10, 20, 30]
    assert tensors[2].dtype == torch.bfloat16


def test_pending_save_tensors_normalizes_tp_hidden_states():
    connector = MooncakeHiddenStatesConnector.__new__(MooncakeHiddenStatesConnector)
    connector._block_size = 3
    connector._kv_cache = torch.arange(2 * 4 * 3 * 2, dtype=torch.float16).view(2, 4, 3, 2)
    connector._pp_size = 1
    connector._num_training_layers = 3
    connector._hidden_size = 2
    pending = _PendingSave("req", torch.tensor([10, 20, 30]), [1])

    tensors = connector._pending_save_tensors(pending, [])

    assert [key for key, _ in tensors] == ["req_hs", "req_ids", "req_lhs"]
    assert tensors[0][1].dtype == torch.bfloat16
    assert tensors[1][1].dtype == torch.int64
    assert tensors[2][1].dtype == torch.bfloat16


def test_publish_pending_saves_splits_at_host_buffer_capacity(monkeypatch):
    connector = MooncakeHiddenStatesConnector.__new__(MooncakeHiddenStatesConnector)
    connector._block_size = 3
    connector._kv_cache = torch.randn(4, 4, 3, 2, dtype=torch.bfloat16)
    connector._pp_size = 2
    connector._layer_ids = [2, 46, 90, 93]
    connector._mooncake_store = MagicMock()
    connector._mooncake_store.config.host_buffer_size = 60
    monkeypatch.setattr(connector, "_ensure_mooncake_store", lambda: True)
    monkeypatch.setattr(connector, "_local_layer_positions", lambda: [0, 1])
    pending = [
        _PendingSave("req0", torch.tensor([1, 2, 3]), [1]),
        _PendingSave("req1", torch.tensor([4, 5, 6]), [2]),
    ]

    connector._publish_pending_saves(pending)

    calls = connector._mooncake_store.put_raw_tensors.call_args_list
    assert len(calls) == 3
    assert calls[0].args[0] == [
        "req0_layer2_hs",
        "req0_layer2_ids",
        "req0_layer46_hs",
    ]
    assert calls[1].args[0] == [
        "req0_layer46_ids",
        "req1_layer2_hs",
        "req1_layer2_ids",
    ]
    assert calls[2].args[0] == [
        "req1_layer46_hs",
        "req1_layer46_ids",
    ]


def test_get_finished_checks_errors_without_drain_before_pipeline_barrier(monkeypatch):
    connector = MooncakeHiddenStatesConnector.__new__(MooncakeHiddenStatesConnector)
    connector._pp_size = 2
    connector._mooncake_store = MagicMock()
    pending = _PendingSave("req", torch.tensor([1]), [2])
    metadata = MooncakeConnectorMetadata(pending_saves=[pending])
    monkeypatch.setattr(connector, "has_connector_metadata", lambda: True)
    monkeypatch.setattr(connector, "_get_connector_metadata", lambda: metadata)
    events = []
    monkeypatch.setattr(connector, "_publish_pending_saves", lambda items: events.append("put"))
    connector._mooncake_store.check_async_errors.side_effect = lambda: events.append("check")
    pp_group = MagicMock()
    pp_group.barrier.side_effect = lambda: events.append("barrier")
    monkeypatch.setattr("vllm.distributed.get_pp_group", lambda: pp_group)

    finished_sending, finished_receiving = connector.get_finished(set())

    assert events == ["put", "check", "barrier"]
    connector._mooncake_store.flush.assert_not_called()
    assert finished_sending == {"req"}
    assert finished_receiving is None


def test_get_finished_reaches_barrier_then_propagates_put_failure(monkeypatch):
    connector = MooncakeHiddenStatesConnector.__new__(MooncakeHiddenStatesConnector)
    connector._pp_size = 2
    connector._mooncake_store = MagicMock()
    pending = _PendingSave("req", torch.tensor([1]), [2])
    metadata = MooncakeConnectorMetadata(pending_saves=[pending])
    monkeypatch.setattr(connector, "has_connector_metadata", lambda: True)
    monkeypatch.setattr(connector, "_get_connector_metadata", lambda: metadata)
    monkeypatch.setattr(
        connector,
        "_publish_pending_saves",
        MagicMock(side_effect=RuntimeError("put failed")),
    )
    pp_group = MagicMock()
    monkeypatch.setattr("vllm.distributed.get_pp_group", lambda: pp_group)

    with pytest.raises(RuntimeError, match="put failed"):
        connector.get_finished(set())

    connector._mooncake_store.check_async_errors.assert_not_called()
    connector._mooncake_store.flush.assert_not_called()
    pp_group.barrier.assert_called_once_with()


def test_latest_vllm_hidden_state_layout_is_requested():
    assert MooncakeHiddenStatesConnector.get_required_kvcache_layout(MagicMock()) == "LBNHC"
