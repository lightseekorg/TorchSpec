# Copyright (c) 2026 LightSeek Foundation
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

"""Pipeline training loop: main training loop with sync training and async inference."""

import hashlib
import os
import time
from pathlib import Path

import ray
import wandb
from tqdm import tqdm

from torchspec.training.checkpoint import (
    _read_checkpoint_metadata,
    _write_checkpoint_metadata,
)
from torchspec.utils.logging import logger


def _is_save_interval_step(step: int, interval: int) -> bool:
    return interval > 0 and step % interval == 0


def _update_checkpoint_eval_meta(
    checkpoint_dir: str | None,
    step: int,
    eval_metrics: dict,
    current_best: float,
) -> float:
    """Append eval metrics to checkpoint meta.json and track best checkpoint."""
    if not checkpoint_dir or not eval_metrics:
        return current_best

    base_dir = Path(checkpoint_dir).expanduser()
    step_id = step + 1
    meta_path = base_dir / f"iter_{step_id:07d}" / "meta.json"

    metadata = _read_checkpoint_metadata(meta_path)
    if not metadata:
        logger.warning(f"Checkpoint meta.json not found at {meta_path}, skipping eval meta update")
        return current_best

    for key in ("eval/avg_loss", "eval/avg_acc", "eval/simulated_acc_len"):
        if key in eval_metrics:
            metadata[key] = eval_metrics[key]
    _write_checkpoint_metadata(meta_path, metadata)

    score = eval_metrics.get("eval/simulated_acc_len")
    if score is not None and score > current_best:
        current_best = score
        (base_dir / "best_checkpointed_iteration.txt").write_text(str(step_id))
        _write_checkpoint_metadata(base_dir / "best_meta.json", metadata)
        logger.info(f"New best checkpoint: iter_{step_id:07d} (simulated_acc_len={score:.2f})")

    return current_best


def _generate_eval_cache(
    controller,
    train_group,
    num_dispatches: int,
    samples_per_rank: int,
    eval_cache_path: str | None,
) -> None:
    """Drain eval results from inference and cache individual samples on each trainer."""
    total_samples = num_dispatches * samples_per_rank
    logger.info(
        f"Caching eval hidden states from inference engine "
        f"({num_dispatches} dispatches, {total_samples} samples per rank)..."
    )
    dispatched = 0
    eval_progress = tqdm(total=num_dispatches, desc="Eval caching", unit="dispatch")
    while dispatched < num_dispatches:
        ok = ray.get(controller.try_dispatch_eval_batch.remote())
        if ok:
            train_group.cache_eval_samples(samples_per_rank)
            dispatched += 1
            eval_progress.update(1)
        else:
            time.sleep(0.5)
    eval_progress.close()

    ray.get(controller.finalize_eval_dispatch.remote())
    logger.info(f"Eval caching complete ({total_samples} samples per rank)")
    if eval_cache_path:
        train_group.async_save_eval_cache(eval_cache_path)
        logger.info(f"Eval cache save started (async) to {eval_cache_path}")


def run_eval(step: int, train_group, eval_enabled: bool) -> dict:
    """Run forward-only eval from cache. Assumes eval cache is already populated."""
    if not eval_enabled:
        return {}
    eval_results = train_group.run_eval()
    eval_metrics = eval_results[0] if eval_results else {}
    if eval_metrics:
        eval_metrics["eval/step"] = step
        if wandb.run is not None:
            wandb.log(eval_metrics)
        logger.info(
            f"Step {step} eval: "
            f"loss={eval_metrics.get('eval/avg_loss', 0):.4f}, "
            f"acc={eval_metrics.get('eval/avg_acc', 0):.4f}, "
            f"sim_acc_len={eval_metrics.get('eval/simulated_acc_len', 0):.2f}"
        )
    return eval_metrics


def run_training_loop(
    args,
    controller,
    inference_manager,
    train_group,
    inference_engines=None,
    dataset_size=None,
    eval_dataset_size=None,
):
    """Run the training loop with sync training and async inference.

    Training is synchronous - waits for each step to complete.
    Inference runs in background, continuously producing data.

    Each optimizer step (with draft_accumulation_steps dispatches):
      1. Controller dispatches per_dp_rank_batch_size * dp_size samples, accumulation_steps times
      2. Each DP rank receives per_dp_rank_batch_size * accumulation_steps samples total
      3. train_from_queue(num_batches=accumulation_steps) processes all micro-batches
      4. Optimizer steps after the last micro-batch

    completed_steps counts optimizer steps (consistent with lr_total_steps).

    Args:
        args: Configuration arguments.
        controller: AsyncTrainingController ray actor handle (dataset already loaded).
        inference_manager: AsyncInferenceManager ray actor handle.
        train_group: Training group with set_train_queues method.
        dataset_size: Number of training samples. If None, queried from controller.
        eval_dataset_size: Number of eval samples. If None, queried from controller. 0 means no eval.
    """
    if dataset_size is None:
        dataset_size = ray.get(controller.get_dataset_size.remote())
    if dataset_size <= 0:
        raise ValueError(
            f"Training dataset size is {dataset_size}. "
            f"Ensure controller.load_dataset() was called before run_training_loop()."
        )
    if eval_dataset_size is None:
        eval_dataset_size = ray.get(controller.get_eval_dataset_size.remote())

    # ── Eval setup ──────────────────────────────────────────────
    # eval_interval=N>0: eval runs every N steps in addition to checkpoint saves.
    eval_interval = args.eval_interval
    eval_enabled = eval_dataset_size > 0
    eval_cache_path: str | None = None
    dispatch_bs = args.per_dp_rank_batch_size * args.dp_size
    max_pool = getattr(args, "max_sample_pool_size", 0) or dispatch_bs
    eval_samples_per_rank = max_pool // args.dp_size

    if eval_enabled:
        num_dispatches = eval_dataset_size // max_pool
        if num_dispatches == 0:
            logger.warning(
                f"Eval dataset ({eval_dataset_size} samples) is smaller than "
                f"max_sample_pool_size ({max_pool}). Disabling eval."
            )
            eval_enabled = False
        else:
            dropped = eval_dataset_size - num_dispatches * max_pool
            if dropped > 0:
                logger.info(
                    f"Eval: {dropped} samples will be dropped (not enough for a full dispatch)"
                )

    best_eval_score = -float("inf")
    if eval_enabled and args.checkpoint_dir:
        best_meta_path = Path(args.checkpoint_dir).expanduser() / "best_meta.json"
        if best_meta_path.exists():
            existing = _read_checkpoint_metadata(best_meta_path)
            if "eval/simulated_acc_len" in existing:
                best_eval_score = existing["eval/simulated_acc_len"]
                logger.info(f"Resumed best eval score: {best_eval_score:.2f}")

    if eval_enabled:
        cache_dir = os.path.abspath(getattr(args, "cache_dir", "./cache"))
        cache_key = hashlib.md5(
            f"{getattr(args, 'eval_data_path', '')}|"
            f"{getattr(args, 'target_model_path', '')}|"
            f"{getattr(args, 'max_seq_length', 0)}|"
            f"{max_pool}".encode()
        ).hexdigest()[:12]
        eval_cache_path = os.path.join(cache_dir, "eval_cache", cache_key)

        eval_cache_loaded = train_group.load_eval_cache(eval_cache_path)
        if all(n > 0 for n in eval_cache_loaded):
            logger.info(
                f"Eval: loaded cached tensors from {eval_cache_path} ({eval_cache_loaded[0]} batches per rank)"
            )
        else:
            eval_cache_loaded = None
            ray.get(controller.submit_eval_dataset.remote())
            logger.info(
                f"Eval: {eval_dataset_size} samples, "
                f"{num_dispatches} dispatches, "
                f"{eval_samples_per_rank} samples/rank"
            )

    inference_future = inference_manager.run.remote()
    # Inference engine is alive.
    if eval_enabled and not eval_cache_loaded:
        _generate_eval_cache(
            controller,
            train_group,
            num_dispatches,
            eval_samples_per_rank,
            eval_cache_path,
        )

    dp_size = (
        getattr(args, "dp_size", None) or args.training_num_nodes * args.training_num_gpus_per_node
    )
    num_steps = args.num_train_steps
    accumulation_steps = getattr(args, "draft_accumulation_steps", 1)
    # steps_per_epoch in optimizer steps, pre-computed in auto_calculate_training_steps
    steps_per_epoch = getattr(args, "steps_per_epoch", dataset_size // args.global_batch_size)
    if steps_per_epoch == 0:
        steps_per_epoch = 1
    num_epochs = getattr(args, "num_epochs", 1)

    logger.info(
        f"Starting: num_steps={num_steps}, num_epochs={num_epochs}, "
        f"steps_per_epoch={steps_per_epoch}, global_batch_size={args.global_batch_size}, "
        f"accumulation_steps={accumulation_steps}, "
        f"dp_size={dp_size}, per_dp_rank_batch_size={args.per_dp_rank_batch_size}"
    )

    enable_perf = getattr(args, "enable_perf_metrics", True)

    # Restore step counter from checkpoint (actors load checkpoint in __init__)
    start_step = ray.get(train_group._actor_handlers[0].get_global_step.remote())
    completed_steps = start_step
    current_epoch = completed_steps // steps_per_epoch + 1
    steps_in_current_epoch = completed_steps % steps_per_epoch
    if start_step > 0:
        logger.info(f"Resuming from step {start_step} (epoch {current_epoch})")
    dispatch_attempts = 0
    consecutive_failures = 0
    last_saved_step: int | None = None
    progress = tqdm(total=num_steps, desc="Training", unit="step", initial=start_step)
    while completed_steps < num_steps:
        # Inner loop: dispatch accumulation_steps batches before training
        dispatches_done = 0
        if enable_perf:
            t_dispatch = time.time()
        status = None
        while dispatches_done < accumulation_steps:
            dispatch_attempts += 1

            dispatched = ray.get(controller.try_dispatch_batch.remote())
            if dispatched:
                dispatches_done += 1
                consecutive_failures = 0
            else:
                consecutive_failures += 1

                # Only fetch status when needed for logging or reload decision
                if dispatch_attempts % 100 == 0 or consecutive_failures >= 500:
                    status = ray.get(controller.get_full_status.remote())

                if dispatch_attempts % 100 == 0 and status is not None:
                    logger.debug(
                        f"Epoch {current_epoch}, Step {completed_steps}: "
                        f"dispatch failed {dispatch_attempts} times "
                        f"(consecutive={consecutive_failures}), "
                        f"pool_size={status['sample_pool_size']}, "
                        f"need={dispatch_bs}"
                    )

                should_reload = False
                if steps_in_current_epoch >= steps_per_epoch:
                    should_reload = True
                elif (
                    consecutive_failures >= 500
                    and (completed_steps > 0 or dispatches_done > 0)
                    and status is not None
                    and status["sample_pool_size"] < dispatch_bs
                    and status.get("prompt_buffer_size", 0) == 0
                ):
                    logger.warning(
                        f"Pool insufficient for dispatch "
                        f"(pool_size={status['sample_pool_size']}, "
                        f"need={dispatch_bs}, "
                        f"{dispatches_done}/{accumulation_steps} dispatches done, "
                        f"{steps_in_current_epoch}/{steps_per_epoch} steps in epoch). "
                        f"Reloading dataset."
                    )
                    should_reload = True

                if should_reload:
                    if completed_steps < num_steps:
                        current_epoch += 1
                        steps_in_current_epoch = 0
                        consecutive_failures = 0
                        logger.info(f"Dataset exhausted, reloading (epoch {current_epoch})...")
                        ray.get(controller.reload_dataset.remote())
                    else:
                        logger.info("Max steps reached, stopping")
                        break

                time.sleep(0.01)
        else:
            # All accumulation dispatches succeeded — run training
            if enable_perf:
                dispatch_wait = time.time() - t_dispatch

            train_futures = [
                actor.train_from_queue.remote(
                    step=completed_steps,
                    num_batches=accumulation_steps,
                )
                for actor in train_group._actor_handlers
            ]

            train_results = ray.get(train_futures)
            completed_steps += 1

            # Log metrics from training (use rank 0's metrics - they're already all-reduced)
            metrics = train_results[0] if train_results and train_results[0] else {}
            if metrics:
                # Add step counters for wandb x-axis (required in shared mode)
                metrics["train/step"] = completed_steps
                metrics["inference/step"] = completed_steps

                # Add inference metrics (e2e_latency, spec metrics, etc.)
                inference_metrics = ray.get(inference_manager.flush_metrics.remote())
                metrics.update(inference_metrics)

                if enable_perf:
                    metrics["perf/dispatch_wait"] = dispatch_wait
                    step_time = metrics.get("perf/step_time", 0)
                    if step_time > 0:
                        metrics["perf/train_capacity"] = args.global_batch_size / step_time

                if wandb.run is not None:
                    wandb.log(metrics)

            # ── Eval at explicit interval (if configured) ─────────
            # Skip if a checkpoint save is about to run (it will eval anyway)
            save_due = _is_save_interval_step(completed_steps, args.save_interval)
            if eval_interval > 0 and completed_steps % eval_interval == 0 and not save_due:
                run_eval(completed_steps, train_group, eval_enabled)

            steps_in_current_epoch += 1
            dispatch_attempts = 0

            status = ray.get(controller.get_full_status.remote())
            postfix = {
                "loss": f"{metrics.get('train/avg_loss', 0):.3f}",
                "acc": f"{metrics.get('train/avg_acc', 0):.3f}",
                "acc_len": f"{metrics.get('train/simulated_acc_len', 0):.2f}",
                "thru": f"{status['inference_speed']:.1f}",
            }
            if enable_perf:
                postfix["I"] = f"{metrics.get('perf/infer_capacity', 0):.1f}"
                postfix["T"] = f"{metrics.get('perf/train_capacity', 0):.1f}"
                postfix["wait"] = f"{dispatch_wait:.1f}s"
                postfix["pool"] = status["sample_pool_size"]
            postfix["epoch"] = f"{current_epoch}/{num_epochs}"
            # Set postfix before update so tqdm emits only one line
            progress.set_postfix(postfix, refresh=False)
            progress.update(1)

            if _is_save_interval_step(completed_steps, args.save_interval):
                eval_metrics = run_eval(completed_steps, train_group, eval_enabled)
                logger.info(f"Saving checkpoint at step {completed_steps}...")
                train_group.save_model(completed_steps)
                last_saved_step = completed_steps
                best_eval_score = _update_checkpoint_eval_meta(
                    args.checkpoint_dir, completed_steps, eval_metrics, best_eval_score
                )

            # Check if epoch completed
            if steps_in_current_epoch >= steps_per_epoch:
                logger.info(
                    f"Epoch {current_epoch} completed ({steps_in_current_epoch} steps). "
                    f"Total steps: {completed_steps}/{num_steps}"
                )

                if (
                    args.save_per_epoch
                    and args.checkpoint_dir
                    and last_saved_step != completed_steps
                ):
                    eval_metrics = run_eval(completed_steps, train_group, eval_enabled)
                    logger.info(
                        f"Saving checkpoint at end of epoch {current_epoch} "
                        f"(step {completed_steps})..."
                    )
                    train_group.save_model(completed_steps)
                    last_saved_step = completed_steps
                    best_eval_score = _update_checkpoint_eval_meta(
                        args.checkpoint_dir, completed_steps, eval_metrics, best_eval_score
                    )

                if completed_steps < num_steps:
                    current_epoch += 1
                    steps_in_current_epoch = 0
                    logger.info(f"Dataset exhausted, reloading (epoch {current_epoch})...")
                    ray.get(controller.reload_dataset.remote())
                else:
                    logger.info("Max steps reached")
                    break
            continue

        # Inner while broke (max steps reached during reload), break outer loop
        break

    progress.close()

    ray.get(inference_manager.stop.remote())
    ray.get(inference_future)

    # Always save a final checkpoint unless saved.
    if args.checkpoint_dir and last_saved_step != completed_steps:
        eval_metrics = run_eval(completed_steps, train_group, eval_enabled)
        logger.info(f"Saving final checkpoint at step {completed_steps}...")
        train_group.save_model(completed_steps, force_sync=True)
        best_eval_score = _update_checkpoint_eval_meta(
            args.checkpoint_dir, completed_steps, eval_metrics, best_eval_score
        )

    final_status = ray.get(controller.get_full_status.remote())
    logger.info(
        f"Training completed: {completed_steps} steps in {final_status['elapsed_seconds']:.1f}s | "
        f"avg inference={final_status['avg_inference_speed']:.1f} entries/s | "
        f"avg training={final_status['avg_training_speed']:.1f} entries/s"
    )
