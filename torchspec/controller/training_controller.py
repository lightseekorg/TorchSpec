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

"""
Async Training Controller for decoupled inference and training.

Data flow:
  load_dataset(args) → Stored Dataset → Prompt Buffer → Inference Manager
    → Sample Pool → Train Queues → Training Workers
  (Stored dataset is retained for epoch reloads and vocab mapping computation.)

Controller manages the tokenized dataset (for epoch reloads and vocab mapping),
prompt metadata, and mooncake keys. Actual inference tensor data is stored in
mooncake; the controller only tracks keys and byte sizes for backpressure.

Batch Size Design:
  micro_batch_size                   # Samples per GPU per dispatch (user config)
  per_dp_rank_batch_size             # = micro_batch_size * sp_size (derived)
  dispatch_batch_size            # = per_dp_rank_batch_size * dp_size (samples per dispatch)
  global_batch_size                  # = dispatch_batch_size * accumulation_steps (per optimizer step)

  Example with micro_batch_size=2, sp_size=1, dp_size=4, accumulation_steps=2:
    - per_dp_rank_batch_size = 2 * 1 = 2
    - dispatch_batch_size = 2 * 4 = 8
    - global_batch_size = 8 * 2 = 16

  Data flow per optimizer step (with accumulation_steps=2):
    1. Controller dispatches 8 samples per dispatch, 2 dispatches per optimizer step
    2. Each DP rank receives 2 samples per dispatch (4 total per optimizer step)
    3. Each train actor calls train_from_queue(num_batches=accumulation_steps)
    4. Forward/backward for each micro-batch, optimizer step after last one
"""

import copy
import threading
import time
from collections import deque
from dataclasses import dataclass, field

import ray
from ray.util.queue import Queue

from torchspec.training.data_fetcher import TrainSample
from torchspec.utils.logging import logger
from torchspec.utils.memory import estimate_tensor_bytes
from torchspec.utils.types import InferenceInput, InferenceOutput

_estimate_bytes = estimate_tensor_bytes


@dataclass
class SpeedMonitor:
    """Tracks throughput over a sliding time window."""

    window_seconds: float = 10.0
    _events: deque = field(default_factory=deque)
    _lock: threading.Lock = field(default_factory=threading.Lock)
    _total_count: int = 0

    def record(self, count: int = 1) -> None:
        """Record count entries at current time."""
        now = time.time()
        with self._lock:
            self._events.append((now, count))
            self._total_count += count
            self._prune_old_events(now)

    def _prune_old_events(self, now: float) -> None:
        """Remove events outside the window."""
        cutoff = now - self.window_seconds
        while self._events and self._events[0][0] < cutoff:
            self._events.popleft()

    def get_speed(self) -> float:
        """Get current speed in entries/sec over the window."""
        now = time.time()
        with self._lock:
            self._prune_old_events(now)
            if not self._events:
                return 0.0

            window_count = sum(count for _, count in self._events)
            oldest_time = self._events[0][0]
            elapsed = now - oldest_time
            if elapsed < 0.001:
                return 0.0
            return window_count / elapsed

    def get_total_count(self) -> int:
        """Get total count since start."""
        return self._total_count


@ray.remote
class AsyncTrainingController:
    """Central controller for async training pipeline.

    Responsibilities:
      - Loads and stores tokenized datasets for training and eval
      - Computes vocab mappings for draft model pruning
      - Manages prompt buffer (samples waiting for inference)
      - Manages sample pool (completed inferences waiting for training)
      - Dispatches batches to per-DP training queues when pool is full
      - Monitors inference and training throughput
    """

    def __init__(self, args, dp_size: int):
        self.args = args
        self.dp_size = dp_size

        self.prompt_buffer: deque[InferenceInput] = deque()
        self._prompt_lock = threading.Lock()

        self.sample_pool: deque[InferenceOutput] = deque()
        self._pool_lock = threading.Lock()
        self._pool_bytes = 0
        self._sample_bytes: dict[str, int] = {}

        self.train_queues = [Queue() for _ in range(dp_size)]

        # Eval: separate pool and queues so eval data never mixes with training
        self.eval_pool: deque[InferenceOutput] = deque()
        self._eval_pool_lock = threading.Lock()
        self._eval_data_ids: set[str] = set()
        self._eval_expected_count: int = 0
        self._eval_dispatched_samples: int = 0
        self.eval_queues = [Queue() for _ in range(dp_size)]

        self.batch_id = 0
        self.dispatch_batch_size = args.per_dp_rank_batch_size * dp_size
        max_pool = getattr(args, "max_sample_pool_size", 0)
        self.eval_dispatch_batch_size = max_pool or self.dispatch_batch_size
        self._data_id_counter = 0

        self._stored_dataset: list | None = None
        self._stored_eval_dataset: list | None = None

        self._start_time = time.time()
        self._inference_monitor = SpeedMonitor(window_seconds=10.0)
        self._training_monitor = SpeedMonitor(window_seconds=10.0)
        self._last_dispatch_log_time = 0.0
        self._inference_error: str | None = None

    def _generate_data_id(self) -> str:
        self._data_id_counter += 1
        return f"data_{self._data_id_counter}"

    # ─────────────────────────────────────────────────────────────
    # Dataset Loading
    # ─────────────────────────────────────────────────────────────

    def add_dataset(self, dataset: list) -> int:
        with self._prompt_lock:
            for sample in dataset:
                if isinstance(sample, dict):
                    data_id = sample.get("data_id") or self._generate_data_id()
                    input_ids = sample.get("input_ids")
                    packed_loss_mask = sample.get("packed_loss_mask")
                    if input_ids is not None and packed_loss_mask is None:
                        raise ValueError(
                            f"packed_loss_mask is required when input_ids is provided "
                            f"(data_id={data_id}). Use defer_tokenization=True to skip "
                            f"tokenization entirely."
                        )
                    entry = InferenceInput(
                        data_id=data_id,
                        prompt=sample.get("prompt", sample),
                        input_ids=input_ids,
                        packed_loss_mask=packed_loss_mask,
                        formatted_prompt=sample.get("formatted_prompt"),
                        metadata=sample.get("metadata", {}),
                        multimodal_inputs=sample.get("multimodal_inputs"),
                    )
                else:
                    entry = InferenceInput(
                        data_id=self._generate_data_id(),
                        prompt=sample,
                    )
                self.prompt_buffer.append(entry)
            return len(dataset)

    def load_dataset(self, args) -> int:
        """Load and process dataset on the controller, store for epoch reloads, and prime the prompt buffer."""
        from torchspec.data.dataset import load_conversation_dataset

        self._stored_dataset = load_conversation_dataset(args)
        if not self._stored_dataset:
            raise ValueError(
                f"Training dataset is empty after processing. "
                f"Check train_data_path='{args.train_data_path}', "
                f"max_seq_length={getattr(args, 'max_seq_length', None)}, "
                f"and chat_template settings."
            )
        count = self.add_dataset(self._stored_dataset)
        logger.info(f"Controller loaded dataset: {count} samples")
        return count

    def reload_dataset(self) -> int:
        """Re-add the stored dataset to the prompt buffer (epoch reload)."""
        assert self._stored_dataset is not None, "No stored dataset to reload"
        return self.add_dataset(self._stored_dataset)

    def load_eval_dataset(self, args) -> int:
        """Load eval dataset on the controller and store it. Returns size (0 if none)."""
        eval_data_path = getattr(args, "eval_data_path", None)
        if not eval_data_path:
            return 0

        from torchspec.data.dataset import load_conversation_dataset

        eval_args = copy.copy(args)
        eval_args.train_data_path = eval_data_path
        eval_prompt_key = getattr(args, "eval_prompt_key", None)
        if eval_prompt_key:
            eval_args.prompt_key = eval_prompt_key
        self._stored_eval_dataset = load_conversation_dataset(eval_args)
        count = len(self._stored_eval_dataset)
        logger.info(f"Controller loaded eval dataset: {count} samples from {eval_data_path}")
        return count

    def submit_eval_dataset(self) -> int:
        """Submit the stored eval dataset for inference (prepended with high priority)."""
        assert self._stored_eval_dataset is not None, "No stored eval dataset"
        return self.set_eval_dataset(self._stored_eval_dataset)

    def get_dataset_size(self) -> int:
        if self._stored_dataset is None:
            raise RuntimeError(
                "get_dataset_size() called but no dataset has been loaded. "
                "Call load_dataset() first."
            )
        return len(self._stored_dataset)

    def get_eval_dataset_size(self) -> int:
        return len(self._stored_eval_dataset) if self._stored_eval_dataset is not None else 0

    def compute_vocab_mapping(self, target_vocab_size: int, draft_vocab_size: int) -> tuple:
        """Generate vocab mapping on the controller using the stored dataset.

        Requires the dataset to have been loaded with defer_tokenization=False,
        since vocab mapping needs input_ids.
        """
        from torchspec.data.preprocessing import generate_vocab_mapping

        assert self._stored_dataset is not None, "No stored dataset for vocab mapping"
        assert "input_ids" in self._stored_dataset[0], (
            "compute_vocab_mapping requires input_ids in dataset. "
            "Set defer_tokenization=False to enable tokenization."
        )
        return generate_vocab_mapping(
            prompts=self._stored_dataset,
            target_vocab_size=target_vocab_size,
            draft_vocab_size=draft_vocab_size,
        )

    # ─────────────────────────────────────────────────────────────
    # Interface for Inference Manager
    # ─────────────────────────────────────────────────────────────

    def get_prompts(self, num_prompts: int) -> list[InferenceInput]:
        """Inference manager gets prompts with data_ids.

        Args:
            num_prompts: Maximum number of prompts to fetch.

        Returns:
            List of InferenceInput objects.
        """
        with self._prompt_lock:
            entries = []
            for _ in range(min(num_prompts, len(self.prompt_buffer))):
                entries.append(self.prompt_buffer.popleft())
            return entries

    def push_inference_results(self, results: list[InferenceOutput]) -> int:
        """Inference sends back (data_id, mooncake_key) pairs.

        Controller stores the keys and tracks exact bytes for backpressure.
        Eval results (identified by data_id) are routed to the eval pool.

        Args:
            results: List of InferenceOutput containing data_id, mooncake_key,
                    tensor_shapes, and tensor_dtypes.

        Returns:
            Current pool bytes after adding results. This allows inference manager
            to implement Mooncake backpressure.
        """
        eval_results = []
        train_results = []
        for result in results:
            if result.data_id in self._eval_data_ids:
                eval_results.append(result)
            else:
                train_results.append(result)

        if eval_results:
            with self._eval_pool_lock:
                self.eval_pool.extend(eval_results)

        pool_bytes = 0
        if train_results:
            with self._pool_lock:
                for result in train_results:
                    sample_bytes = estimate_tensor_bytes(
                        result.tensor_shapes or {},
                        result.tensor_dtypes or {},
                    )
                    self._sample_bytes[result.mooncake_key] = sample_bytes
                    self._pool_bytes += sample_bytes
                self.sample_pool.extend(train_results)
                pool_bytes = self._pool_bytes

        self._inference_monitor.record(len(results))
        return pool_bytes

    def get_prompt_buffer_size(self) -> int:
        """Get current size of prompt buffer."""
        return len(self.prompt_buffer)

    # ─────────────────────────────────────────────────────────────
    # Interface for Training
    # ─────────────────────────────────────────────────────────────

    def get_train_queues(self) -> list[Queue]:
        """Get the per-DP training queues."""
        return self.train_queues

    def get_pool_size(self) -> int:
        """Total mooncake-resident samples (training + eval) for backpressure.

        Always includes eval pool so that backpressure accounts for mooncake
        segment capacity used by eval data.  Without this, eval data occupies
        mooncake outside backpressure awareness and the segment overflows.
        """
        train_size = len(self.sample_pool)
        with self._eval_pool_lock:
            return train_size + len(self.eval_pool)

    def get_pool_bytes(self) -> int:
        """Get current bytes in sample pool (for Mooncake backpressure)."""
        with self._pool_lock:
            return self._pool_bytes

    # ─────────────────────────────────────────────────────────────
    # Dispatch Logic
    # ─────────────────────────────────────────────────────────────

    def set_inference_error(self, msg: str) -> None:
        """Called by the inference manager when a fatal error occurs."""
        self._inference_error = msg

    def try_dispatch_batch(self) -> bool:
        """Try to dispatch one batch to training queues.

        Only dispatches when sample pool has enough samples (>= dispatch_batch_size).
        Dispatches TrainSample objects that MooncakeDataFetcher can consume.
        Subtracts bytes from pool tracking when dispatching.

        Returns:
            True if a batch was dispatched, False if not enough samples.

        Raises:
            RuntimeError: If the inference manager has reported a fatal error.
        """
        if self._inference_error is not None:
            raise RuntimeError(f"Inference engine failed: {self._inference_error}")

        with self._pool_lock:
            pool_size = len(self.sample_pool)
            now = time.time()
            should_log = (now - self._last_dispatch_log_time) >= 2.0
            if pool_size < self.dispatch_batch_size:
                if should_log:
                    self._last_dispatch_log_time = now
                    logger.debug(
                        f"try_dispatch_batch: pool_size={pool_size} < dispatch_batch_size={self.dispatch_batch_size}, not dispatching"
                    )
                return False

            if should_log:
                self._last_dispatch_log_time = now
                logger.debug(
                    f"try_dispatch_batch: pool_size={pool_size} >= dispatch_batch_size={self.dispatch_batch_size}, dispatching batch {self.batch_id}"
                )

            batch_results = []
            for _ in range(self.dispatch_batch_size):
                result = self.sample_pool.popleft()
                sample_bytes = self._sample_bytes.pop(result.mooncake_key, 0)
                self._pool_bytes -= sample_bytes
                batch_results.append(result)

        self._dispatch_to_queues(batch_results, self.train_queues)

        self._training_monitor.record(self.dispatch_batch_size)
        logger.debug(
            f"Dispatched batch {self.batch_id} with {self.dispatch_batch_size} samples "
            f"to {self.dp_size} queues at t={time.time():.3f}"
        )
        self.batch_id += 1
        return True

    def _partition_results(self, results: list[InferenceOutput]) -> list[list[InferenceOutput]]:
        """Partition InferenceOutputs across DP ranks."""
        partitions: list[list[InferenceOutput]] = [[] for _ in range(self.dp_size)]
        for i, result in enumerate(results):
            partitions[i % self.dp_size].append(result)
        return partitions

    def _dispatch_to_queues(
        self,
        batch_results: list[InferenceOutput],
        queues: list[Queue],
    ) -> None:
        """Partition results across DP ranks and push TrainSamples into queues."""
        partitioned = self._partition_results(batch_results)
        for dp_rank, results in enumerate(partitioned):
            for result in results:
                queues[dp_rank].put(
                    TrainSample(
                        mooncake_key=result.mooncake_key,
                        tensor_shapes=result.tensor_shapes,
                        tensor_dtypes=result.tensor_dtypes,
                        packed_loss_mask=result.packed_loss_mask,
                    )
                )

    def push_inference_sample(self, sample: InferenceOutput) -> int:
        """Add a single inference sample to the training pool.

        This method is used by OnlineServingController to add samples
        captured from the inference engine's /generate API.

        Args:
            sample: InferenceOutput containing mooncake_key and tensor metadata.

        Returns:
            Current pool bytes after adding the sample.
        """
        return self.push_inference_results([sample])

    # ─────────────────────────────────────────────────────────────
    # Eval Pipeline
    # ─────────────────────────────────────────────────────────────

    def set_eval_dataset(self, dataset: list) -> int:
        """Push eval samples into the prompt buffer for inference.

        Eval data_ids are tracked so that inference results are routed to
        the eval pool instead of the training sample pool.

        Eval samples are prepended (high priority) so they are processed
        before queued training data — otherwise backpressure on the
        training pool can starve eval indefinitely.
        """
        self._eval_expected_count = len(dataset)
        self._eval_dispatched_samples = 0

        eval_entries: list[InferenceInput] = []
        for sample in dataset:
            if isinstance(sample, dict):
                raw_id = sample.get("data_id") or self._generate_data_id()
                data_id = f"eval_{raw_id}"
                self._eval_data_ids.add(data_id)
                entry = InferenceInput(
                    data_id=data_id,
                    prompt=sample.get("prompt", sample),
                    input_ids=sample.get("input_ids"),
                    packed_loss_mask=sample.get("packed_loss_mask"),
                    formatted_prompt=sample.get("formatted_prompt"),
                    metadata=sample.get("metadata", {}),
                    multimodal_inputs=sample.get("multimodal_inputs"),
                )
            else:
                data_id = f"eval_{self._generate_data_id()}"
                self._eval_data_ids.add(data_id)
                entry = InferenceInput(data_id=data_id, prompt=sample)
            eval_entries.append(entry)

        with self._prompt_lock:
            self.prompt_buffer.extendleft(reversed(eval_entries))
        logger.info(f"Eval: prepended {len(dataset)} samples to prompt buffer (high priority)")
        return len(dataset)

    def get_eval_pool_size(self) -> int:
        with self._eval_pool_lock:
            return len(self.eval_pool)

    def get_eval_queues(self) -> list[Queue]:
        return self.eval_queues

    def try_dispatch_eval_batch(self) -> bool:
        """Dispatch one eval batch from the pool if enough samples are available."""
        bs = self.eval_dispatch_batch_size
        with self._eval_pool_lock:
            if len(self.eval_pool) < bs:
                return False
            batch_results = [self.eval_pool.popleft() for _ in range(bs)]

        self._dispatch_to_queues(batch_results, self.eval_queues)
        self._eval_dispatched_samples += bs
        logger.debug(
            f"Eval: dispatched batch ({self._eval_dispatched_samples}/"
            f"{self._eval_expected_count} samples)"
        )
        return True

    def is_eval_dispatch_complete(self) -> bool:
        """True when all dispatchable eval batches have been sent to queues.

        Becomes true once all expected eval samples have completed inference
        AND all full batches have been dispatched.  Remainder samples (less
        than one batch) are not dispatched.
        """
        if self._eval_expected_count == 0:
            return False
        with self._eval_pool_lock:
            arrived = self._eval_dispatched_samples + len(self.eval_pool)
        if arrived < self._eval_expected_count:
            return False
        # All samples arrived. Check if any full batch remains.
        with self._eval_pool_lock:
            return len(self.eval_pool) < self.eval_dispatch_batch_size

    def finalize_eval_dispatch(self) -> None:
        """Clean up eval tracking state after all batches have been cached.

        Drops remainder samples that didn't fill a full batch and resets
        counters so the next eval cycle starts fresh.
        """
        with self._eval_pool_lock:
            remaining = len(self.eval_pool)
            if remaining > 0:
                logger.info(f"Eval: dropping {remaining} leftover samples that didn't fill a batch")
                self.eval_pool.clear()

        self._eval_data_ids.clear()
        self._eval_expected_count = 0
        self._eval_dispatched_samples = 0
        logger.info("Eval: dispatch finalized, tracking state cleared")

    # ─────────────────────────────────────────────────────────────
    # Status and Monitoring
    # ─────────────────────────────────────────────────────────────

    def get_status(self) -> dict:
        """Get current status of controller."""
        return {
            "prompt_buffer_size": len(self.prompt_buffer),
            "sample_pool_size": len(self.sample_pool),
            "batches_dispatched": self.batch_id,
            "dispatch_batch_size": self.dispatch_batch_size,
        }

    def get_speeds(self) -> dict:
        """Get current throughput speeds in entries/sec."""
        elapsed = time.time() - self._start_time
        return {
            "inference_speed": round(self._inference_monitor.get_speed(), 2),
            "training_speed": round(self._training_monitor.get_speed(), 2),
            "inference_total": self._inference_monitor.get_total_count(),
            "training_total": self._training_monitor.get_total_count(),
            "elapsed_seconds": round(elapsed, 1),
            "avg_inference_speed": round(
                self._inference_monitor.get_total_count() / max(elapsed, 0.001), 2
            ),
            "avg_training_speed": round(
                self._training_monitor.get_total_count() / max(elapsed, 0.001), 2
            ),
        }

    def get_full_status(self) -> dict:
        """Get complete status including speeds."""
        status = self.get_status()
        status.update(self.get_speeds())
        return status

    def shutdown(self) -> None:
        """Signal training workers to stop by sending None to queues."""
        for q in self.train_queues:
            q.put(None)
        logger.info("Controller shutdown: sent stop signals to training queues")
