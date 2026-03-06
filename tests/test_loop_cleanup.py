from types import SimpleNamespace
from unittest import mock

import pytest

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
        )


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
    )

    with (
        mock.patch("torchspec.controller.eval.ray.get", side_effect=lambda x: x),
        mock.patch("torchspec.controller.eval.time.sleep"),
    ):
        eval_utils.generate_eval_cache(controller, train_group, state)

    assert train_group.cache_eval_samples.call_count == 4
    train_group.cache_eval_samples.assert_called_with(1)
    controller.finalize_eval_dispatch.remote.assert_called_once()


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
    controller.submit_eval_dataset.remote.return_value = 16

    args = SimpleNamespace(
        eval_interval=50,
        dp_size=2,
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
    controller.submit_eval_dataset.remote.assert_called_once()


def test_setup_eval_dispatch_bs_caps_at_dataset_size():
    controller = mock.MagicMock()
    train_group = mock.MagicMock()
    train_group.load_eval_cache.return_value = [0]
    controller.submit_eval_dataset.remote.return_value = 4

    args = SimpleNamespace(
        eval_interval=50,
        dp_size=8,
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
