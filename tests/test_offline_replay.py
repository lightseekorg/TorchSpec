"""Tests for the offline dataset, saver, and replay contract."""

from argparse import Namespace
from queue import Queue
from unittest.mock import MagicMock, patch

import pytest
import torch

from torchspec.offline.dataset import (
    OFFLINE_SCHEMA_VERSION,
    OfflineDataset,
)


def _args(tmp_path, **overrides):
    values = {
        "offline_data_path": str(tmp_path),
        "target_model_path": "target/model",
        "target_model_backend": "sglang",
        "inference_engine_type": "sgl",
        "last_hidden_states_prenorm": False,
        "chat_template": "llama3",
        "max_seq_length": 128,
        "aux_hidden_states_layers": [1, 2, 3],
        "attention_backend": "sdpa",
        "sp_ulysses_size": 1,
        "sp_ring_size": 1,
        "ttt_length": 7,
    }
    values.update(overrides)
    return Namespace(**values)


def _tensors(seq_len=4):
    return {
        "input_ids": torch.arange(seq_len, dtype=torch.int64).unsqueeze(0),
        "hidden_states": torch.randn(1, seq_len, 6, dtype=torch.bfloat16),
        "last_hidden_states": torch.randn(1, seq_len, 2, dtype=torch.bfloat16),
        "target": None,
    }


def _write_dataset(tmp_path):
    args = _args(tmp_path)
    writer = OfflineDataset(tmp_path, create=True, last_hidden_states_prenorm=False)
    writer.append(
        "train",
        data_id="train-1",
        tensors=_tensors(),
        packed_loss_mask="2,2",
        metadata={"source": "test"},
    )
    writer.append(
        "eval",
        data_id="eval-1",
        tensors=_tensors(3),
        packed_loss_mask="1,2",
    )
    return args, writer


def test_offline_dataset_round_trip_and_resume(tmp_path):
    _args_obj, writer = _write_dataset(tmp_path)

    assert writer.count("train") == 1
    assert not writer.append(
        "train",
        data_id="train-1",
        tensors=_tensors(),
        packed_loss_mask="2,2",
    )

    resumed = OfflineDataset(tmp_path)
    assert resumed.count("train") == 1
    assert resumed.metadata["version"] == OFFLINE_SCHEMA_VERSION
    assert (tmp_path / "manifest.jsonl").is_file()
    assert not (tmp_path / "train" / "manifest.jsonl").exists()
    record = resumed.load(resumed.rows("train")[0])
    assert record["data_id"] == "train-1"
    assert record["packed_loss_mask"] == "2,2"
    assert set(record) >= {"input_ids", "hidden_states", "last_hidden_states"}
    assert "input_ids_cpu" not in record


def test_offline_dataset_accepts_dflash_hidden_states_only(tmp_path):
    writer = OfflineDataset(tmp_path, create=True, last_hidden_states_prenorm=False)
    tensors = _tensors()
    tensors["last_hidden_states"] = None

    assert writer.append(
        "train",
        data_id="dflash-1",
        tensors=tensors,
        packed_loss_mask=None,
    )

    record = OfflineDataset(tmp_path).load("dflash-1")
    assert set(record) >= {"input_ids", "hidden_states"}
    assert "last_hidden_states" not in record
    assert "target" not in record


def test_offline_dataset_rejects_cross_split_ids_and_missing_tensors(tmp_path):
    _args_obj, writer = _write_dataset(tmp_path)

    with pytest.raises(ValueError, match="already exists in another split"):
        writer.append(
            "eval",
            data_id="train-1",
            tensors=_tensors(),
            packed_loss_mask="2,2",
        )

    with pytest.raises(ValueError, match="missing tensors"):
        writer.append(
            "train",
            data_id="broken",
            tensors={"input_ids": torch.ones(1, 2)},
            packed_loss_mask=None,
        )


class _FakeMooncakeStore:
    def __init__(self):
        self.put_calls = []
        self.flush_count = 0

    @staticmethod
    def _meta(kwargs):
        tensors = {
            key: value
            for key, value in kwargs.items()
            if key in ("input_ids", "hidden_states", "target", "last_hidden_states")
            and isinstance(value, torch.Tensor)
        }
        return {
            "shapes": {key: tuple(value.shape) for key, value in tensors.items()},
            "dtypes": {key: value.dtype for key, value in tensors.items()},
        }

    def put(self, **kwargs):
        self.put_calls.append(kwargs)
        return self._meta(kwargs)

    def flush(self):
        self.flush_count += 1


def _make_replay_engine(tmp_path, args):
    from torchspec.inference.engine.offline_replay_engine import OfflineReplayEngine

    with patch("torchspec.inference.engine.offline_replay_engine.setup_file_logging"):
        engine = OfflineReplayEngine(args, rank=0)
    dataset = OfflineDataset(tmp_path)
    engine._dataset = dataset
    rows = dataset.rows("train") + dataset.rows("eval")
    engine._rows_by_id = {row["data_id"]: row for row in rows}
    engine._mooncake_store = _FakeMooncakeStore()
    return engine


def test_replay_engine_returns_live_engine_contract(tmp_path):
    args, _writer = _write_dataset(tmp_path)
    engine = _make_replay_engine(tmp_path, args)

    outputs = engine.generate(data_id=["train-1"])

    assert len(outputs) == 1
    output = outputs[0]
    assert output["mooncake_key"]
    assert output["packed_loss_mask"] == "2,2"
    assert output["metadata"]["source"] == "test"
    assert output["tensor_shapes"]["input_ids"] == (1, 4)
    assert len(engine._mooncake_store.put_calls) == 1
    assert engine._mooncake_store.flush_count == 1


def test_replay_engine_rejects_unknown_data_id(tmp_path):
    args, _writer = _write_dataset(tmp_path)
    engine = _make_replay_engine(tmp_path, args)

    with pytest.raises(KeyError, match="missing"):
        engine.generate(data_id="missing")


def test_saving_actor_consumes_normal_train_sample(tmp_path):
    from torchspec.offline.saving_actor import OfflineSavingActor
    from torchspec.training.data_fetcher import TrainSample

    actor_class = OfflineSavingActor.__ray_metadata__.modified_class
    actor = object.__new__(actor_class)
    actor.dataset = MagicMock()
    actor.dataset.append.return_value = True
    actor.store = MagicMock()
    actor.store.get.return_value.to_tensor_dict.return_value = _tensors()
    actor.queues = {"train": Queue()}
    actor.queues["train"].put(
        TrainSample(
            data_id="train-1",
            mooncake_key="key",
            tensor_shapes={"input_ids": (1, 4), "hidden_states": (1, 4, 6)},
            tensor_dtypes={"input_ids": "int64"},
            packed_loss_mask="2,2",
            metadata={"source": "test"},
        )
    )

    assert actor.save_from_queue("train") == 1
    assert actor.dataset.append.call_args.kwargs["data_id"] == "train-1"
    actor.store.remove_eagle3_tensors.assert_called_once()


def test_saving_actor_rejects_vllm_pp_sample():
    """PP writes per-layer fragments, never the base key the actor reads."""
    from torchspec.offline.saving_actor import OfflineSavingActor
    from torchspec.training.data_fetcher import TrainSample

    actor_class = OfflineSavingActor.__ray_metadata__.modified_class
    actor = object.__new__(actor_class)
    actor.dataset = MagicMock()
    actor.store = MagicMock()
    actor.queues = {"train": Queue()}
    actor.queues["train"].put(
        TrainSample(
            data_id="train-1",
            mooncake_key="key",
            tensor_shapes={"input_ids": (1, 4), "hidden_states": (1, 4, 6)},
            tensor_dtypes={"input_ids": "int64"},
            metadata={
                "vllm_pp_complete": True,
                "vllm_pp_layer_manifest": [
                    {"layer_id": 4, "mooncake_key": "key_layer4", "role": "last_hidden_states"}
                ],
            },
        )
    )

    with pytest.raises(NotImplementedError, match="pipeline parallelism"):
        actor.save_from_queue("train")

    actor.store.get.assert_not_called()
    actor.store.remove_eagle3_tensors.assert_not_called()


def test_controller_sources_manifest_ids_without_retokenizing(tmp_path):
    args, _writer = _write_dataset(tmp_path)
    args.inference_engine_type = "offline"
    args.per_dp_rank_batch_size = 1
    args.shuffle_dataset = False
    args.seed = 0

    from torchspec.controller.training_controller import AsyncTrainingController

    controller_class = AsyncTrainingController.__ray_metadata__.modified_class
    with patch("torchspec.controller.training_controller.Queue", side_effect=lambda: MagicMock()):
        controller = controller_class(args, dp_size=1)

    assert controller.load_dataset(args) == 1
    assert controller.load_eval_dataset(args) == 1
    assert controller._stored_dataset == [
        {"data_id": "train-1", "metadata": {"offline_replay": True}, "seq_len": 4}
    ]

    controller.submit_eval_chunk(0, 1)
    entry = controller.prompt_buffer.popleft()
    assert entry.data_id == "eval-1"
    assert entry.input_ids is None


def test_controller_common_loader_handles_online_train_and_eval(tmp_path):
    from torchspec.controller.training_controller import AsyncTrainingController

    args = _args(
        tmp_path,
        train_data_path="train.jsonl",
        eval_data_path="eval.jsonl",
        eval_prompt_key="eval_prompt",
        per_dp_rank_batch_size=1,
    )
    controller_class = AsyncTrainingController.__ray_metadata__.modified_class
    with patch("torchspec.controller.training_controller.Queue", side_effect=lambda: MagicMock()):
        controller = controller_class(args, dp_size=2)

    train_rows = [{"data_id": "train-1"}]
    eval_rows = [{"data_id": f"eval-{i}"} for i in range(3)]
    with patch(
        "torchspec.data.dataset.load_conversation_dataset",
        side_effect=[train_rows, eval_rows],
    ) as load_conversation_dataset:
        assert controller.load_dataset(args) == 1
        assert controller.load_eval_dataset(args) == 2

    train_args = load_conversation_dataset.call_args_list[0].args[0]
    eval_args = load_conversation_dataset.call_args_list[1].args[0]
    assert train_args is args
    assert eval_args is not args
    assert eval_args.train_data_path == "eval.jsonl"
    assert eval_args.prompt_key == "eval_prompt"


def test_controller_preserves_data_id_in_train_queue(tmp_path):
    from torchspec.controller.training_controller import AsyncTrainingController
    from torchspec.utils.types import InferenceOutput

    args, _writer = _write_dataset(tmp_path)
    args.per_dp_rank_batch_size = 1
    controller_class = AsyncTrainingController.__ray_metadata__.modified_class
    with patch("torchspec.controller.training_controller.Queue", side_effect=lambda: MagicMock()):
        controller = controller_class(args, dp_size=1)

    controller._dispatch_to_queues(
        [
            InferenceOutput(
                data_id="source-id",
                mooncake_key="key",
                tensor_shapes={},
                tensor_dtypes={},
            )
        ],
        controller.train_queues,
    )

    sample = controller.train_queues[0].put.call_args.args[0]
    assert sample.data_id == "source-id"


def test_factory_creates_cpu_replay_actors(tmp_path):
    args, _writer = _write_dataset(tmp_path)
    args.offline_num_engines = 2
    actor_class = MagicMock()
    actor_class.options.return_value.remote.side_effect = [MagicMock(), MagicMock()]

    with (
        patch("torchspec.inference.factory.ray.remote", return_value=actor_class),
        patch("torchspec.inference.factory.get_torchspec_env_vars", return_value={}),
    ):
        from torchspec.inference.factory import _prepare_offline_replay_engines

        engines, refs = _prepare_offline_replay_engines(args, MagicMock())

    assert len(engines) == 2
    assert len(refs) == 2
    for call in actor_class.options.call_args_list:
        assert call.kwargs["num_gpus"] == 0
        assert call.kwargs["num_cpus"] == 1


def test_offline_training_is_selected_by_engine_type(tmp_path):
    from torchspec.config.train_config import config_to_flat_args, load_config

    config = load_config(
        cli_args=[
            "inference.inference_engine_type=offline",
            f"inference.offline.data_path={tmp_path}",
            "inference.offline.num_engines=3",
            "dataset.defer_tokenization=true",
        ]
    )
    args = config_to_flat_args(config)

    assert args.inference_engine_type == "offline"
    assert args.offline_data_path == str(tmp_path)
    assert args.offline_num_engines == 3
    assert not hasattr(args, "offline_enabled")
    assert args.last_hidden_states_prenorm is None
    assert args.defer_tokenization is False
    assert args.dynamic_loss_mask is False


def test_inference_manager_uses_its_normal_input_path_for_offline(tmp_path):
    from torchspec.controller.inference_manager import AsyncInferenceManager
    from torchspec.utils.types import InferenceInput

    args = _args(tmp_path, inference_engine_type="offline", defer_tokenization=False)
    manager_class = AsyncInferenceManager.__ray_metadata__.modified_class
    manager = manager_class(
        args,
        controller=MagicMock(),
        inference_engines=[MagicMock()],
    )
    entry = InferenceInput(data_id="train-1")

    with patch(
        "torchspec.controller.inference_manager.ray.put", return_value="input-ref"
    ) as ray_put:
        inputs = manager._prepare_engine_inputs([entry])

    ray_put.assert_called_once_with([None])
    assert inputs["input_ids_ref"] == "input-ref"


def test_offline_rejects_usp(tmp_path):
    from torchspec.config.train_config import load_config

    with pytest.raises(ValueError, match="usp is not supported offline"):
        load_config(
            cli_args=[
                "inference.inference_engine_type=offline",
                f"inference.offline.data_path={tmp_path}",
                "training.attention_backend=usp",
            ]
        )


def test_mooncake_put_accepts_cpu_replay_tensors_without_cuda_events():
    from torchspec.transfer.mooncake.eagle_store import EagleMooncakeStore

    store = object.__new__(EagleMooncakeStore)
    store._gpu_direct_available = False
    store._gpu_send_buffer = None
    buffer = MagicMock(ptr=123)
    store._host_buffer_pool = MagicMock()
    store._host_buffer_pool.get_buffer.return_value = buffer
    store._async_put_manager = MagicMock()
    store._stage_tensors_into_buffer = MagicMock(return_value=([123], [8]))

    store._put_raw_tensors(["key"], [torch.ones(2)])

    store._async_put_manager.submit.assert_called_once_with(["key"], [123], [8], 123)


def test_mooncake_cpu_client_does_not_create_cuda_stream():
    from torchspec.config.mooncake_config import MooncakeConfig
    from torchspec.transfer.mooncake.store import MooncakeHiddenStateStore

    client = MagicMock()
    client.setup.return_value = 0
    client.batch_remove.__doc__ = "batch_remove(keys, force=False)"
    config = MooncakeConfig(async_put_pool_size=0, enable_gpu_direct=False)
    store = MooncakeHiddenStateStore(config)

    with (
        patch(
            "torchspec.transfer.mooncake.store.MooncakeDistributedStore",
            return_value=client,
        ),
        patch("torchspec.transfer.mooncake.store.torch.cuda.is_available", return_value=True),
        patch("torchspec.transfer.mooncake.store.torch.cuda.Stream") as cuda_stream,
    ):
        store.setup(device=torch.device("cpu"))

    cuda_stream.assert_not_called()
    store.close()
