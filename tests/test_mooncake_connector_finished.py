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


def test_publish_pending_save_writes_only_pipeline_local_layers(monkeypatch):
    connector = MooncakeHiddenStatesConnector.__new__(MooncakeHiddenStatesConnector)
    connector._block_size = 3
    connector._kv_cache = torch.arange(2 * 4 * 3 * 2, dtype=torch.bfloat16).view(2, 4, 3, 2)
    connector._pp_size = 2
    connector._layer_ids = [2, 46, 90, 93]
    connector._mooncake_store = MagicMock()
    monkeypatch.setattr(connector, "_ensure_mooncake_store", lambda: True)
    monkeypatch.setattr(connector, "_local_layer_positions", lambda: [0, 1])
    pending = _PendingSave("req", torch.tensor([10, 20, 30]), [1])

    connector._publish_pending_save(pending)

    assert [call.kwargs["key"] for call in connector._mooncake_store.put.call_args_list] == [
        "req_layer2",
        "req_layer46",
    ]
    first = connector._mooncake_store.put.call_args_list[0].kwargs
    assert first["hidden_states"].shape == (3, 2)
    assert first["input_ids"].tolist() == [10, 20, 30]
    assert first["last_hidden_states"] is None


def test_get_finished_flushes_before_pipeline_barrier(monkeypatch):
    connector = MooncakeHiddenStatesConnector.__new__(MooncakeHiddenStatesConnector)
    connector._pp_size = 2
    connector._mooncake_store = MagicMock()
    pending = _PendingSave("req", torch.tensor([1]), [2])
    metadata = MooncakeConnectorMetadata(pending_saves=[pending])
    monkeypatch.setattr(connector, "has_connector_metadata", lambda: True)
    monkeypatch.setattr(connector, "_get_connector_metadata", lambda: metadata)
    events = []
    monkeypatch.setattr(connector, "_publish_pending_save", lambda item: events.append("put"))
    connector._mooncake_store.flush.side_effect = lambda: events.append("flush")
    pp_group = MagicMock()
    pp_group.barrier.side_effect = lambda: events.append("barrier")
    monkeypatch.setattr("vllm.distributed.get_pp_group", lambda: pp_group)

    finished_sending, finished_receiving = connector.get_finished(set())

    assert events == ["put", "flush", "barrier"]
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
        "_publish_pending_save",
        MagicMock(side_effect=RuntimeError("put failed")),
    )
    pp_group = MagicMock()
    monkeypatch.setattr("vllm.distributed.get_pp_group", lambda: pp_group)

    with pytest.raises(RuntimeError, match="put failed"):
        connector.get_finished(set())

    connector._mooncake_store.flush.assert_not_called()
    pp_group.barrier.assert_called_once_with()


def test_latest_vllm_hidden_state_layout_is_requested():
    assert MooncakeHiddenStatesConnector.get_required_kvcache_layout(MagicMock()) == "LBNHC"
