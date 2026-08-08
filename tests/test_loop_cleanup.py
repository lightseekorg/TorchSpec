from types import SimpleNamespace
from unittest import mock

import pytest
from omegaconf import OmegaConf

from torchspec.controller import eval as eval_utils
from torchspec.controller import loop


class TestTrainingLoopCleanup:
    def test_safe_cleanup_stops_inference_and_shutdowns_mooncake_actor(self):
        args = SimpleNamespace(_mooncake_master_actor=mock.MagicMock())
        inference_manager = mock.MagicMock()
        inference_future = object()

        stop_ref = inference_manager.stop.remote.return_value
        shutdown_ref = args._mooncake_master_actor.shutdown.remote.return_value

        with mock.patch("torchspec.controller.loop.ray.get") as mock_ray_get:
            loop._safe_training_cleanup(args, inference_manager, inference_future)

        mock_ray_get.assert_any_call(stop_ref)
        mock_ray_get.assert_any_call(inference_future)
        mock_ray_get.assert_any_call(shutdown_ref, timeout=10)

    def test_safe_cleanup_skips_mooncake_shutdown_when_actor_missing(self):
        args = SimpleNamespace()
        inference_manager = mock.MagicMock()
        inference_future = object()

        with mock.patch("torchspec.controller.loop.ray.get"):
            loop._safe_training_cleanup(args, inference_manager, inference_future)

        inference_manager.stop.remote.assert_called_once()

    def test_run_training_loop_finally_runs_cleanup_on_success(self):
        args = SimpleNamespace(_mooncake_master_actor=mock.MagicMock())
        inference_manager = mock.MagicMock()
        inference_future = inference_manager.run.remote.return_value

        with (
            mock.patch("torchspec.controller.loop.training_loop", return_value="ok") as mock_impl,
            mock.patch("torchspec.controller.loop._safe_training_cleanup") as mock_cleanup,
        ):
            result = loop.run_training_loop(args, object(), inference_manager, object())

        assert result == "ok"
        mock_impl.assert_called_once()
        mock_cleanup.assert_called_once_with(
            args=args,
            inference_manager=inference_manager,
            inference_future=inference_future,
            inference_engines=None,
        )

    def test_run_training_loop_finally_runs_cleanup_on_exception(self):
        args = SimpleNamespace(_mooncake_master_actor=mock.MagicMock())
        inference_manager = mock.MagicMock()
        inference_future = inference_manager.run.remote.return_value

        with (
            mock.patch(
                "torchspec.controller.loop.training_loop",
                side_effect=RuntimeError("training failed"),
            ),
            mock.patch("torchspec.controller.loop._safe_training_cleanup") as mock_cleanup,
        ):
            with pytest.raises(RuntimeError, match="training failed"):
                loop.run_training_loop(args, object(), inference_manager, object())

        mock_cleanup.assert_called_once_with(
            args=args,
            inference_manager=inference_manager,
            inference_future=inference_future,
            inference_engines=None,
        )


def _run_mock_training_loop(
    *,
    prefetch_depth: int,
    num_steps: int,
    steps_per_epoch: int,
    dispatch_results=None,
):
    args = SimpleNamespace(
        training_num_nodes=1,
        training_num_gpus_per_node=1,
        num_train_steps=num_steps,
        draft_accumulation_steps=1,
        steps_per_epoch=steps_per_epoch,
        num_epochs=(num_steps + steps_per_epoch - 1) // steps_per_epoch,
        global_batch_size=1,
        per_dp_rank_batch_size=1,
        prefetch_depth=prefetch_depth,
        enable_perf_metrics=False,
        save_interval=0,
        checkpoint_dir=None,
        save_per_epoch=False,
        train_with_decode=False,
    )
    events = []
    results = iter(dispatch_results) if dispatch_results is not None else None

    controller = mock.MagicMock()
    controller.submit_training_dataset.remote.return_value = None
    controller.reload_dataset.remote.return_value = None

    def dispatch():
        result = next(results) if results is not None else True
        events.append(f"dispatch:{result}")
        return result

    controller.try_dispatch_batch.remote.side_effect = dispatch
    controller.get_full_status.remote.return_value = {
        "inference_speed": 0.0,
        "sample_pool_size": 0,
        "elapsed_seconds": 0.0,
        "avg_inference_speed": 0.0,
        "avg_training_speed": 0.0,
    }

    actor = mock.MagicMock()
    actor.get_global_step.remote.return_value = 0

    def train(*, step, num_batches):
        assert num_batches == 1
        events.append(f"train:{step}")
        return {}

    actor.train_from_queue.remote.side_effect = train
    train_group = mock.MagicMock()
    train_group._actor_handlers = [actor]

    eval_state = eval_utils.EvalSetupState(
        eval_interval=0,
        eval_enabled=False,
        eval_cache_loaded=False,
        eval_cache_path=None,
        best_eval_score=0.0,
        eval_dispatch_bs=0,
        eval_dataset_size=0,
        dp_size=1,
    )

    with (
        mock.patch("torchspec.controller.loop.setup_eval", return_value=eval_state),
        mock.patch("torchspec.controller.loop.ray.get", side_effect=lambda value: value),
        mock.patch("torchspec.controller.loop.time.sleep") as mock_sleep,
        mock.patch("torchspec.controller.loop.tqdm"),
    ):
        loop.training_loop(
            args,
            controller,
            mock.MagicMock(),
            train_group,
            dataset_size=max(num_steps, 1),
            eval_dataset_size=0,
        )

    return events, controller, mock_sleep


def test_training_loop_prefills_and_refills_controller_queue():
    events, _, _ = _run_mock_training_loop(
        prefetch_depth=2,
        num_steps=3,
        steps_per_epoch=3,
    )

    assert events == [
        "dispatch:True",
        "dispatch:True",
        "train:0",
        "dispatch:True",
        "train:1",
        "train:2",
    ]


def test_training_loop_prefetch_zero_preserves_one_step_dispatching():
    events, _, _ = _run_mock_training_loop(
        prefetch_depth=0,
        num_steps=3,
        steps_per_epoch=3,
    )

    assert events == [
        "dispatch:True",
        "train:0",
        "dispatch:True",
        "train:1",
        "dispatch:True",
        "train:2",
    ]


def test_training_loop_runs_ready_step_when_lookahead_cannot_be_filled():
    events, _, mock_sleep = _run_mock_training_loop(
        prefetch_depth=2,
        num_steps=3,
        steps_per_epoch=3,
        dispatch_results=[True, False, True, True],
    )

    assert events == [
        "dispatch:True",
        "dispatch:False",
        "train:0",
        "dispatch:True",
        "dispatch:True",
        "train:1",
        "train:2",
    ]
    mock_sleep.assert_not_called()


def test_training_loop_does_not_prefetch_across_epoch_boundary():
    events, controller, _ = _run_mock_training_loop(
        prefetch_depth=3,
        num_steps=4,
        steps_per_epoch=2,
    )

    assert events == [
        "dispatch:True",
        "dispatch:True",
        "train:0",
        "train:1",
        "dispatch:True",
        "dispatch:True",
        "train:2",
        "train:3",
    ]
    controller.reload_dataset.remote.assert_called_once_with()


def test_generate_eval_cache_dispatches_and_finalizes():
    controller = mock.MagicMock()
    train_group = mock.MagicMock()

    controller.try_dispatch_eval_batch.remote.side_effect = [True, True, True, True]

    state = eval_utils.EvalSetupState(
        eval_interval=0,
        eval_enabled=True,
        eval_cache_loaded=False,
        eval_cache_path=None,
        best_eval_score=0.0,
        eval_dispatch_bs=2,
        eval_dataset_size=8,
        dp_size=2,
        initial_eval_submit_count=8,
    )

    with (
        mock.patch("torchspec.controller.eval.ray.get", side_effect=lambda x: x),
        mock.patch("torchspec.controller.eval.time.sleep"),
    ):
        eval_utils.generate_eval_cache(controller, train_group, state)

    assert train_group.cache_eval_samples.call_count == 4
    train_group.cache_eval_samples.assert_called_with(1)
    controller.finalize_eval_dispatch.remote.assert_called_once()


def test_generate_eval_cache_interleaves_refill_and_drain_without_stalling():
    controller = mock.MagicMock()
    train_group = mock.MagicMock()

    controller.try_dispatch_eval_batch.remote.side_effect = [
        False,
        True,
        False,
        True,
        False,
        True,
        False,
        True,
    ]

    state = eval_utils.EvalSetupState(
        eval_interval=0,
        eval_enabled=True,
        eval_cache_loaded=False,
        eval_cache_path=None,
        best_eval_score=0.0,
        eval_dispatch_bs=2,
        eval_dataset_size=8,
        dp_size=2,
        initial_eval_submit_count=4,
    )

    with (
        mock.patch("torchspec.controller.eval.ray.get", side_effect=lambda x: x),
        mock.patch("torchspec.controller.eval.time.sleep") as mock_sleep,
    ):
        eval_utils.generate_eval_cache(controller, train_group, state)

    assert train_group.cache_eval_samples.call_count == 4
    assert controller.submit_eval_chunk.remote.call_args_list == [
        mock.call(4, 6),
        mock.call(6, 8),
    ]
    controller.finalize_eval_dispatch.remote.assert_called_once()
    assert mock_sleep.call_count == 4


def test_generate_eval_cache_times_out_when_eval_never_arrives():
    controller = mock.MagicMock()
    train_group = mock.MagicMock()

    state = eval_utils.EvalSetupState(
        eval_interval=0,
        eval_enabled=True,
        eval_cache_loaded=False,
        eval_cache_path=None,
        best_eval_score=0.0,
        eval_dispatch_bs=2,
        eval_dataset_size=8,
        dp_size=2,
        initial_eval_submit_count=4,
    )

    controller.try_dispatch_eval_batch.remote.return_value = False
    with (
        mock.patch("torchspec.controller.eval.ray.get", side_effect=lambda x: x),
        mock.patch("torchspec.controller.eval.time.sleep") as mock_sleep,
        mock.patch("torchspec.controller.eval.EVAL_CACHE_IDLE_TIMEOUT", 0.0),
    ):
        with pytest.raises(TimeoutError, match="Timed out while waiting for eval cache generation"):
            eval_utils.generate_eval_cache(controller, train_group, state)

    train_group.cache_eval_samples.assert_not_called()
    controller.finalize_eval_dispatch.remote.assert_not_called()
    mock_sleep.assert_not_called()


def test_setup_eval_dispatch_bs_is_dp_size():
    controller = mock.MagicMock()
    train_group = mock.MagicMock()
    train_group.load_eval_cache.return_value = [0]
    controller.submit_eval_chunk.remote.return_value = 4

    args = SimpleNamespace(
        eval_interval=50,
        dp_size=2,
        inference_batch_size=4,
        max_sample_pool_size=64,
        checkpoint_dir=None,
        cache_dir="./cache",
        eval_data_path="eval.jsonl",
        target_model_path="model",
        max_seq_length=4096,
    )

    with mock.patch("torchspec.controller.eval.ray.get", side_effect=lambda x: x):
        state = eval_utils.setup_eval(
            controller=controller,
            train_group=train_group,
            args=args,
            eval_dataset_size=16,
        )

    assert state.eval_enabled is True
    assert state.eval_dispatch_bs == 2
    assert state.eval_dataset_size == 16
    assert state.initial_eval_submit_count == 4
    controller.submit_eval_chunk.remote.assert_called_once_with(0, 4)


def test_setup_eval_dispatch_bs_caps_at_dataset_size():
    controller = mock.MagicMock()
    train_group = mock.MagicMock()
    train_group.load_eval_cache.return_value = [0]
    controller.submit_eval_chunk.remote.return_value = 4

    args = SimpleNamespace(
        eval_interval=50,
        dp_size=8,
        inference_batch_size=4,
        max_sample_pool_size=64,
        checkpoint_dir=None,
        cache_dir="./cache",
        eval_data_path="eval.jsonl",
        target_model_path="model",
        max_seq_length=4096,
    )

    with mock.patch("torchspec.controller.eval.ray.get", side_effect=lambda x: x):
        state = eval_utils.setup_eval(
            controller=controller,
            train_group=train_group,
            args=args,
            eval_dataset_size=4,
        )

    assert state.eval_dispatch_bs == 4
    assert state.initial_eval_submit_count == 4
    controller.submit_eval_chunk.remote.assert_called_once_with(0, 4)


def test_setup_eval_cache_key_includes_aux_hidden_state_layers():
    controller = mock.MagicMock()
    train_group = mock.MagicMock()
    train_group.load_eval_cache.return_value = [0]
    controller.submit_eval_chunk.remote.return_value = 1

    args = SimpleNamespace(
        eval_interval=50,
        dp_size=1,
        inference_batch_size=1,
        checkpoint_dir=None,
        cache_dir="./cache",
        eval_data_path="eval.jsonl",
        target_model_path="model",
        max_seq_length=4096,
        aux_hidden_states_layers=[1, 17, 32],
    )

    with mock.patch("torchspec.controller.eval.ray.get", side_effect=lambda x: x):
        first = eval_utils.setup_eval(
            controller=controller,
            train_group=train_group,
            args=args,
            eval_dataset_size=1,
        )
        args.aux_hidden_states_layers = [1, 8, 16, 24, 32]
        second = eval_utils.setup_eval(
            controller=controller,
            train_group=train_group,
            args=args,
            eval_dataset_size=1,
        )

    assert first.eval_cache_path != second.eval_cache_path


def test_setup_eval_cache_key_accepts_omegaconf_list():
    controller = mock.MagicMock()
    train_group = mock.MagicMock()
    train_group.load_eval_cache.return_value = [0]
    controller.submit_eval_chunk.remote.return_value = 1

    args = SimpleNamespace(
        eval_interval=50,
        dp_size=1,
        inference_batch_size=1,
        checkpoint_dir=None,
        cache_dir="./cache",
        eval_data_path="eval.jsonl",
        target_model_path="model",
        max_seq_length=4096,
        aux_hidden_states_layers=OmegaConf.create([2, 46, 90, 93]),
    )

    with mock.patch("torchspec.controller.eval.ray.get", side_effect=lambda x: x):
        state = eval_utils.setup_eval(
            controller=controller,
            train_group=train_group,
            args=args,
            eval_dataset_size=1,
        )

    assert state.eval_enabled is True
    train_group.load_eval_cache.assert_called_once()
