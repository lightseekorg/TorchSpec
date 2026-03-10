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

import abc
import concurrent.futures
import dataclasses
import itertools
import logging
import os
import time
from argparse import Namespace
from typing import Optional

import torch
import torch.distributed as dist
from torch.distributed.checkpoint.state_dict import (
    StateDictOptions,
    get_model_state_dict,
)
from torch.distributed.device_mesh import init_device_mesh

from torchspec.config.mooncake_config import MooncakeConfig
from torchspec.data.utils import DataCollatorWithPadding
from torchspec.training import checkpoint
from torchspec.training.data_fetcher import MooncakeDataFetcher
from torchspec.training.fsdp import init_empty_weights
from torchspec.training.optimizer import BF16Optimizer
from torchspec.transfer.mooncake.eagle_store import EagleMooncakeStore
from torchspec.utils.logging import logger
from torchspec.utils.processing import get_assistant_token_ids
from torchspec.utils.profiling import TrainProfiler
from torchspec.utils.train_dump import extract_gradients, extract_model_weights


class Trainer(abc.ABC):
    """Base trainer for async training pipeline.

    Provides shared infrastructure: device mesh, data fetching, training loop
    skeleton, checkpointing, and profiling. Subclasses implement model-specific
    logic via ``init_model``, ``_train_step``, and ``_aggregate_metrics``.
    """

    def __init__(self, args: Namespace):
        self.args = args

        self._setup_device_mesh()
        torch.manual_seed(getattr(args, "seed", 42))

        self.fsdp_cpu_offload = getattr(args, "fsdp_cpu_offload", False)

        self.global_step = 0
        self.model = None
        self.draft_model = None
        self.optimizer: Optional[BF16Optimizer] = None
        self.lr_scheduler = None
        self.data_fetcher: Optional[MooncakeDataFetcher] = None
        self.train_queue = None
        self.mooncake_store: Optional[EagleMooncakeStore] = None
        self._eval_cache: list[dict] = []

        self.prof = TrainProfiler(args)

        self.dynamic_loss_mask = getattr(args, "dynamic_loss_mask", False)
        self.last_turn_loss_only = getattr(args, "last_turn_loss_only", False)
        self.assistant_header_ids, self.end_token_ids = get_assistant_token_ids(self.args)

        self.save_debug_train_data = getattr(args, "save_debug_train_data", None)
        self.max_dump_steps = getattr(args, "max_dump_steps", 5)

        self._enable_perf_metrics = getattr(args, "enable_perf_metrics", True)

        self._io_executor = concurrent.futures.ThreadPoolExecutor(max_workers=1)
        self._eval_cache_save_future: Optional[concurrent.futures.Future] = None

    # ------------------------------------------------------------------
    # Device mesh
    # ------------------------------------------------------------------

    def _setup_device_mesh(self) -> None:
        world_size = dist.get_world_size()
        rank = dist.get_rank()

        self.dp_size = world_size
        self.dp_rank = rank

        self.mesh = init_device_mesh("cuda", mesh_shape=(self.dp_size,), mesh_dim_names=("dp",))
        self.dp_group = self.mesh.get_group("dp")
        self.dp_mesh = self.mesh

        logger.info(
            f"[Rank {rank}] Device mesh (1D): world_size={world_size}, dp_size={self.dp_size}"
        )

    def _get_init_weight_context_manager(self):
        """Meta-device context for non-rank-0 processes to save memory."""

        def cpu_init_weights():
            return torch.device("cpu")

        if dist.get_rank() != 0:
            return init_empty_weights
        return cpu_init_weights

    # ------------------------------------------------------------------
    # Mooncake store
    # ------------------------------------------------------------------

    def init_mooncake_store(
        self,
        mooncake_config: Optional[MooncakeConfig] = None,
    ) -> EagleMooncakeStore:
        if mooncake_config is None:
            mooncake_config = MooncakeConfig.from_flat_args(self.args)

        mooncake_config = dataclasses.replace(
            mooncake_config,
            global_segment_size=0,
            async_put_pool_size=0,
        )

        store = EagleMooncakeStore(mooncake_config)
        store.setup(device=torch.cuda.current_device())
        self.mooncake_store = store
        logger.info(f"[Rank {self.dp_rank}] EagleMooncakeStore initialized")
        return store

    # ------------------------------------------------------------------
    # Data queue
    # ------------------------------------------------------------------

    def set_train_queue(
        self,
        queue,
        mooncake_config: Optional[MooncakeConfig] = None,
        per_dp_rank_batch_size: int = 1,
    ) -> None:
        self.train_queue = queue
        self.per_dp_rank_batch_size = per_dp_rank_batch_size
        if mooncake_config is not None and self.mooncake_store is None:
            self.init_mooncake_store(mooncake_config)

        collator = DataCollatorWithPadding()

        self.data_fetcher = MooncakeDataFetcher(
            queue=self.train_queue,
            mooncake_store=self.mooncake_store,
            collator=collator,
            device=torch.cuda.current_device(),
            batch_size=per_dp_rank_batch_size,
            assistant_header_ids=self.assistant_header_ids,
            end_token_ids=self.end_token_ids,
            dynamic_loss_mask=self.dynamic_loss_mask,
            last_turn_loss_only=self.last_turn_loss_only,
        )

        logger.info(
            f"[Rank {self.dp_rank}] Data fetcher initialized with batch_size={per_dp_rank_batch_size}"
        )

    # ------------------------------------------------------------------
    # Eval queue & CPU cache
    # ------------------------------------------------------------------

    def set_eval_queue(
        self,
        queue,
        mooncake_config: Optional[MooncakeConfig] = None,
        per_dp_rank_batch_size: int = 1,
    ) -> None:
        if mooncake_config is not None and self.mooncake_store is None:
            self.init_mooncake_store(mooncake_config)

        collator = DataCollatorWithPadding()

        self._eval_data_fetcher = MooncakeDataFetcher(
            queue=queue,
            mooncake_store=self.mooncake_store,
            collator=collator,
            device=torch.cuda.current_device(),
            batch_size=per_dp_rank_batch_size,
            assistant_header_ids=self.assistant_header_ids,
            end_token_ids=self.end_token_ids,
            dynamic_loss_mask=self.dynamic_loss_mask,
            last_turn_loss_only=self.last_turn_loss_only,
        )
        self._eval_collator = collator
        self._eval_cache: list[dict] = []
        logger.info(
            f"[Rank {self.dp_rank}] Eval data fetcher initialized "
            f"with batch_size={per_dp_rank_batch_size}"
        )

    def cache_eval_samples(self, count: int) -> int:
        for sample in itertools.islice(self._eval_data_fetcher, count):
            cpu_sample = {
                k: v.cpu() if isinstance(v, torch.Tensor) else v for k, v in sample.items()
            }
            self._eval_cache.append(cpu_sample)
        return len(self._eval_cache)

    def save_eval_cache(self, cache_dir: str) -> None:
        if not getattr(self, "_eval_cache", None):
            return
        self._wait_for_eval_cache_save()

        cache_snapshot = list(self._eval_cache)
        rank = self.dp_rank

        def _save() -> None:
            os.makedirs(cache_dir, exist_ok=True)
            path = os.path.join(cache_dir, f"eval_rank_{rank}.pt")
            tmp_path = path + ".tmp"
            torch.save(cache_snapshot, tmp_path)
            os.replace(tmp_path, path)
            logger.info(f"[Rank {rank}] Saved {len(cache_snapshot)} eval batches to {path}")

        self._eval_cache_save_future = self._io_executor.submit(_save)

    def _wait_for_eval_cache_save(self) -> None:
        fut = self._eval_cache_save_future
        if fut is not None:
            fut.result()
            self._eval_cache_save_future = None

    def load_eval_cache(self, cache_dir: str) -> int:
        # Safe guard to wait for eval cache save to complete.
        self._wait_for_eval_cache_save()
        path = os.path.join(cache_dir, f"eval_rank_{self.dp_rank}.pt")
        if not os.path.exists(path):
            return 0
        try:
            self._eval_cache = torch.load(path, weights_only=False)
        except Exception as e:
            logger.warning(f"[Rank {self.dp_rank}] Corrupt eval cache at {path}, ignoring: {e}")
            return 0
        logger.info(
            f"[Rank {self.dp_rank}] Loaded {len(self._eval_cache)} eval batches from {path}"
        )
        return len(self._eval_cache)

    # ------------------------------------------------------------------
    # Training loop
    # ------------------------------------------------------------------

    def train_from_queue(self, step: int, num_batches: int) -> dict:
        if self.data_fetcher is None:
            raise RuntimeError("Data fetcher not initialized. Call set_train_queue first.")
        perf = self._enable_perf_metrics
        if perf:
            t0 = time.time()
        metrics = self._train_core_from_queue(step=step, num_batches=num_batches)
        if perf:
            # _aggregate_metrics already synced via .item() — wall-clock is accurate
            metrics["perf/step_time"] = time.time() - t0
        self.prof.step(step=step)
        return metrics

    def _train_core_from_queue(self, step: int, num_batches: int) -> dict:
        """Training loop skeleton.

        Calls ``_train_step`` for each micro-batch, wrapped with debug logging
        and optional dump extraction.  One optimizer step is performed after
        all micro-batches.  ``global_step`` counts optimizer steps.
        """
        self.model.train()
        accumulation_steps = num_batches

        all_step_metrics: list[dict] = []
        grad_norm = None

        perf = self._enable_perf_metrics
        if perf:
            data_time = 0.0
            compute_events: list[tuple[torch.cuda.Event, torch.cuda.Event]] = []
            t_data_start = time.time()

        batches = self.prof.iterate_train_actor(self._iter_batches_from_queue(num_batches))
        for batch_idx, batch in enumerate(batches):
            is_last = batch_idx == num_batches - 1

            if perf:
                data_time += time.time() - t_data_start
                evt_start = torch.cuda.Event(enable_timing=True)
                evt_end = torch.cuda.Event(enable_timing=True)
                evt_start.record()

            if logger.isEnabledFor(logging.DEBUG):
                self._log_batch_debug(batch, step, batch_idx, num_batches)

            step_metrics = self._train_step(
                batch=batch,
                accumulation_steps=accumulation_steps,
                step=step,
                batch_idx=batch_idx,
                num_batches=num_batches,
            )

            if is_last:
                self._maybe_dump(batch, step_metrics, step, batch_idx)
                grad_norm = self.optimizer.step()

            if perf:
                evt_end.record()
                compute_events.append((evt_start, evt_end))

            all_step_metrics.append(step_metrics)

            if perf:
                t_data_start = time.time()

        self.global_step += 1

        metrics = self._aggregate_metrics(all_step_metrics, step, grad_norm=grad_norm)
        # _aggregate_metrics calls .item() which syncs CUDA —
        # all recorded events are now completed, safe to query without extra sync
        if perf:
            compute_time_ms = sum(s.elapsed_time(e) for s, e in compute_events)
            metrics["perf/data_time"] = data_time
            metrics["perf/compute_time"] = compute_time_ms / 1000.0

        return metrics

    def _iter_batches_from_queue(self, num_batches: int):
        yield from itertools.islice(self.data_fetcher, num_batches)

    # ------------------------------------------------------------------
    # Checkpointing & persistence
    # ------------------------------------------------------------------

    def save_model(self, step: int, force_sync: bool = False) -> None:
        if not self.args.checkpoint_dir:
            return
        checkpoint.save(self, step=step)

    def save_draft_model_for_serving(self, output_dir: str) -> None:
        """Save draft model in HuggingFace format for serving update."""
        os.makedirs(output_dir, exist_ok=True)

        model = self.draft_model
        if hasattr(model, "module"):
            model = model.module

        try:
            state_dict = get_model_state_dict(
                self.draft_model,
                options=StateDictOptions(full_state_dict=True, cpu_offload=True),
            )

            if self.dp_rank == 0:
                if hasattr(model, "save_pretrained"):
                    if hasattr(model, "config"):
                        model.config.save_pretrained(output_dir)
                    torch.save(state_dict, os.path.join(output_dir, "pytorch_model.bin"))
                    logger.info(f"[Rank {self.dp_rank}] Saved draft model to {output_dir}")
                else:
                    torch.save(state_dict, os.path.join(output_dir, "pytorch_model.bin"))
                    logger.info(
                        f"[Rank {self.dp_rank}] Saved draft model state dict to {output_dir}"
                    )

        except Exception as e:
            logger.warning(
                f"[Rank {self.dp_rank}] Failed to save with FSDP2 state dict, trying fallback: {e}"
            )
            if self.dp_rank == 0:
                if hasattr(model, "save_pretrained"):
                    model.save_pretrained(output_dir)
                    logger.info(
                        f"[Rank {self.dp_rank}] Saved draft model using save_pretrained to {output_dir}"
                    )
                else:
                    torch.save(model.state_dict(), os.path.join(output_dir, "pytorch_model.bin"))
                    logger.info(
                        f"[Rank {self.dp_rank}] Saved draft model state dict to {output_dir}"
                    )

        if dist.is_initialized():
            dist.barrier()

    def load_checkpoint(self) -> dict | None:
        return checkpoint.load(self)

    def close(self) -> None:
        self._wait_for_eval_cache_save()
        self._io_executor.shutdown(wait=True)
        if self.mooncake_store is not None and hasattr(self.mooncake_store, "close"):
            self.mooncake_store.close()
            logger.info(f"[Rank {self.dp_rank}] EagleMooncakeStore closed")

    # ------------------------------------------------------------------
    # Debug logging & dump helpers
    # ------------------------------------------------------------------

    def _log_batch_debug(self, batch: dict, step: int, batch_idx: int, num_batches: int) -> None:
        batch_size = batch["input_ids"].shape[0]
        seq_len = batch["input_ids"].shape[1]
        hs_shape = (
            tuple(batch["hidden_states"].shape) if batch.get("hidden_states") is not None else None
        )
        logger.debug(
            f"step={step} batch={batch_idx}/{num_batches} | "
            f"batch_size={batch_size}, seq_len={seq_len}, "
            f"input_ids={tuple(batch['input_ids'].shape)}, hidden_states={hs_shape}"
        )

    def _should_dump_step(self) -> bool:
        return bool(self.save_debug_train_data and (self.global_step + 1) <= self.max_dump_steps)

    def _maybe_dump(self, batch: dict, step_metrics: dict, step: int, batch_idx: int) -> None:
        if not self._should_dump_step():
            return

        self._save_dump_data(
            batch=batch,
            step_metrics=step_metrics,
            gradients=extract_gradients(self.model),
            model_weights=extract_model_weights(self.model),
            step=step,
            batch_idx=batch_idx,
        )

    def _save_dump_data(
        self,
        *,
        batch: dict,
        step_metrics: dict,
        gradients: dict,
        model_weights: dict,
        step: int,
        batch_idx: int,
    ) -> None:
        """Save debug dump data. Override in subclass for model-specific dumps."""

    # ------------------------------------------------------------------
    # Subclass contract
    # ------------------------------------------------------------------

    @abc.abstractmethod
    def init_model(self, *args, **kwargs) -> int:
        """Initialize model, optimizer, and load checkpoint.

        Returns:
            Start step.
        """
        ...

    @abc.abstractmethod
    def _train_step(
        self,
        batch: dict,
        accumulation_steps: int,
        step: int,
        batch_idx: int,
        num_batches: int,
    ) -> dict:
        """Run forward + backward for a single micro-batch.

        The optimizer step is handled by the base class after the last
        micro-batch — do NOT call ``self.optimizer.step()`` here.

        Returns:
            Dict of step-level metrics for later aggregation.
        """
        ...

    @abc.abstractmethod
    def _aggregate_metrics(
        self, all_step_metrics: list[dict], step: int, *, grad_norm: torch.Tensor = None
    ) -> dict:
        """Aggregate per-step metrics into a single metrics dict.

        Called once per optimizer step after all micro-batches.
        """
        ...
