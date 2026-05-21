# Copyright (c) 2026 LightSeek Foundation
# MIT License

"""Synchronous training loop for colocate (MPS + NCCL) mode.

This is the Phase-5 deliverable: replaces the disaggregated path's
``training_loop`` (loop.py) for colocate runs. Architectural
differences:

* No ``AsyncInferenceManager``. Engines are paired 1:1 with trainers
  on the same physical GPU; the engine writes hidden states directly
  to its paired trainer over NCCL P2P. Backpressure is implicit (the
  engine's NCCL send blocks until the trainer recvs).
* No Mooncake KV store. Trainer-side tensor recv buffers are allocated
  per-step from ``ColocateTrainSample.tensor_specs`` (CPU metadata)
  and filled via ``NcclMultiTensorFetcher.recv_step``.
* Driver fan-out: this loop pulls prompts from the controller and
  dispatches one ``engine.generate`` call per engine paired with the
  matching trainer rank. Trainers run ``train_from_queue`` in parallel
  (one Ray remote each), and the loop awaits both engine and trainer
  futures before advancing the step counter.

Out of scope here (parked for Phase 5 follow-ups):

* Multi-step accumulation (``draft_accumulation_steps > 1``). The disagg
  loop dispatches ``accumulation_steps`` batches before kicking
  ``train_from_queue(num_batches=N)``. The colocate equivalent
  requires careful sample-ordering across the metadata queue and is
  deferred — for now we hard-require ``accumulation_steps == 1``.
* USP attention. ``validate_colocate_config`` already rejects
  USP+colocate, so we don't need a guard here.
* Resume from non-zero step. The disagg loop reads
  ``trainer.get_global_step``; we follow the same pattern but never
  test the resume path because the colocate one-step bring-up runs
  from step 0.
* Eval. Eval cache generation in the colocate path is parked along
  with the rest of Phase 5's "feature parity" — first land the happy
  path, then reintroduce eval.
"""

from __future__ import annotations

import os
import time
from typing import Any

import ray
import torch
from tqdm.auto import tqdm

from torchspec.training.data_fetcher import ColocateTrainSample
from torchspec.utils.logging import logger

# Mirror the disagg path: hidden states are stored / sent in this
# storage dtype (bf16 by default). Keep in lockstep with
# `HIDDEN_STATES_STORAGE_DTYPE` in the SglEngine module.
_HIDDEN_STATES_DTYPE = torch.bfloat16


def _get_hidden_size_from_engine(engine_handle) -> int:
    """Pull the post-init hidden_size from an engine actor."""
    return ray.get(engine_handle.get_status.remote())["hidden_size"]


def _build_tensor_specs(
    seq_len: int,
    *,
    hidden_size: int,
    num_aux_layers: int,
    store_last_hidden_states: bool,
) -> dict[str, tuple[tuple[int, ...], Any]]:
    """Return the ``ColocateTrainSample.tensor_specs`` dict for one sample.

    Shape contract matches the patched sglang's
    ``_send_hidden_states_to_nccl`` (no batch dim — the trainer-side
    ``ColocateDataset`` adds it). Concretely:

      * ``hidden_states``: (seq_len, num_aux_layers * hidden_size), bf16
      * ``input_ids``: (seq_len,), int64
      * ``last_hidden_states``: (seq_len, hidden_size), bf16 [optional]

    Trainer and engine both sort by key, so insertion order is
    irrelevant.
    """
    if num_aux_layers <= 0:
        raise ValueError(f"num_aux_layers must be > 0 to size hidden_states; got {num_aux_layers}")
    concat_hidden_size = num_aux_layers * hidden_size
    specs: dict[str, tuple[tuple[int, ...], Any]] = {
        "hidden_states": ((seq_len, concat_hidden_size), _HIDDEN_STATES_DTYPE),
        "input_ids": ((seq_len,), torch.long),
    }
    if store_last_hidden_states:
        specs["last_hidden_states"] = (
            (seq_len, hidden_size),
            _HIDDEN_STATES_DTYPE,
        )
    return specs


def _seq_len_from_input_ids(input_ids) -> int:
    """Robustly extract seq_len from a possibly-2D tensor."""
    if isinstance(input_ids, torch.Tensor):
        if input_ids.dim() == 2 and input_ids.shape[0] == 1:
            return int(input_ids.shape[1])
        if input_ids.dim() == 1:
            return int(input_ids.shape[0])
        raise ValueError(
            f"unexpected input_ids shape {tuple(input_ids.shape)}; "
            f"expected (seq_len,) or (1, seq_len)"
        )
    return int(len(input_ids))


def run_colocate_training_loop(
    args,
    controller,
    train_group,
    *,
    inference_engines,
    dataset_size: int,
    eval_dataset_size: int = 0,
):
    """Run the synchronous colocate training loop.

    Pre-conditions (asserted by ``train_entry.py`` before calling):
      * Trainer + engine actors have completed init() — the union NCCL
        world is up, the engine subprocess has joined as ranks
        ``[N, 2N)``, and the trainer is sitting on its queue waiting
        for ``ColocateTrainSample`` items.
      * ``args.transfer_mode == 'nccl'`` and ``is_mps_colocate(args)``.
      * ``args.draft_accumulation_steps == 1`` (enforced below).

    The loop is intentionally minimal: one batch dispatched per step,
    no eval, no LR-warmup-aware accumulation. This is the smoke-test
    surface that ``phase4_one_step`` exercises.
    """
    accumulation_steps = int(getattr(args, "draft_accumulation_steps", 1) or 1)
    if accumulation_steps != 1:
        raise NotImplementedError(
            f"colocate loop currently requires draft_accumulation_steps=1 "
            f"(got {accumulation_steps}). Multi-step accumulation is parked."
        )

    dp_size = int(
        getattr(args, "dp_size", None) or args.training_num_nodes * args.training_num_gpus_per_node
    )
    n_engines = len(inference_engines)
    if n_engines == 0 or dp_size % n_engines != 0:
        raise RuntimeError(
            f"Colocate loop: dp_size ({dp_size}) must be a positive multiple "
            f"of the engine count ({n_engines}). Check colocate_strategy=mps "
            f"and that inference_num_gpus / inference_num_gpus_per_engine are "
            f"consistent with training_num_gpus."
        )
    # engine_tp_size: each engine actor owns this many union ranks — its
    # TP scheduler subprocesses — each paired 1:1 with a trainer rank.
    # engine_tp_size == 1 is the original 1:1 engine<->trainer topology.
    engine_tp_size = dp_size // n_engines

    per_dp_rank_batch_size = int(getattr(args, "per_dp_rank_batch_size", 1))
    if per_dp_rank_batch_size != 1:
        raise NotImplementedError(
            f"colocate loop currently requires per_dp_rank_batch_size=1 "
            f"(got {per_dp_rank_batch_size}). Multi-sample-per-rank batching "
            f"requires per-request tensor specs threaded through the controller."
        )

    # Resolve per-step tensor specs from the engine config: hidden_size
    # comes from the loaded model, num_aux_layers from args, and the
    # last-hidden-states flag mirrors what the engine was told to
    # store. We assume all engines agree (same model, same args).
    hidden_size = _get_hidden_size_from_engine(inference_engines[0])
    aux_layers = list(getattr(args, "aux_hidden_states_layers", []) or [])
    if not aux_layers:
        raise RuntimeError(
            "Colocate loop requires aux_hidden_states_layers to be set "
            "(determines hidden_states' last-dim). Use the auto-resolver "
            "in train_entry or set it explicitly in the config."
        )
    num_aux_layers = len(aux_layers)
    store_last_hidden_states = bool(getattr(args, "store_last_hidden_states", True))

    logger.info(
        "[colocate_loop] dp_size=%d engines=%d hidden_size=%d "
        "num_aux_layers=%d store_last_hidden_states=%s "
        "per_dp_rank_batch_size=%d num_train_steps=%d",
        dp_size,
        n_engines,
        hidden_size,
        num_aux_layers,
        store_last_hidden_states,
        per_dp_rank_batch_size,
        int(args.num_train_steps),
    )

    # Submit the dataset (epoch=0, skip=0). Resumption from non-zero
    # step is handled the same way as the disagg loop, but we don't
    # exercise it in tests yet.
    ray.get(controller.submit_training_dataset.remote(epoch=0, skip=0))

    train_queues = ray.get(controller.get_train_queues.remote())
    if len(train_queues) != dp_size:
        raise RuntimeError(
            f"controller.get_train_queues returned {len(train_queues)} queues but dp_size={dp_size}"
        )

    return_last_hidden_states = store_last_hidden_states
    return_logits = False

    enable_perf = bool(getattr(args, "enable_perf_metrics", True))

    completed_steps = int(ray.get(train_group._actor_handlers[0].get_global_step.remote()))
    num_steps = int(args.num_train_steps)
    # Periodic checkpointing. The colocate loop uses the same
    # `save_interval` config knob as the disagg loop (loop.py) -- the
    # previous code read a non-existent `save_steps` attr via getattr,
    # so the save path (and the dcp.save process_group= fix in
    # checkpoint.py) was unreachable dead code. save_interval<=0
    # disables saving. last_saved_step starts at the resume step so a
    # resumed run doesn't immediately re-save.
    save_interval = int(getattr(args, "save_interval", 0) or 0)
    last_saved_step = completed_steps
    progress = tqdm(
        total=num_steps,
        desc="Colocate Training",
        unit="step",
        initial=completed_steps,
    )

    while completed_steps < num_steps:
        t_step = time.time()

        # Pull dp_size prompts (one per engine/trainer pair). If the
        # controller is dry, reload the dataset (epoch boundary).
        prompts = ray.get(controller.get_prompts.remote(dp_size))
        if len(prompts) < dp_size:
            ray.get(controller.reload_dataset.remote())
            prompts = ray.get(controller.get_prompts.remote(dp_size))
            if len(prompts) < dp_size:
                logger.warning(
                    "[colocate_loop] Not enough prompts after reload "
                    "(%d < %d). Stopping at step %d.",
                    len(prompts),
                    dp_size,
                    completed_steps,
                )
                break

        # Fan out the per-rank work:
        #   1. Push ColocateTrainSample(tensor_specs, ...) to trainer queue r
        #      so trainer r's data fetcher knows shapes ahead of recv.
        #   2. Kick engine r's generate() — its spec_training callback
        #      will fire NCCL sends to trainer r once tensors are ready.
        # Steps 1 and 2 must both happen BEFORE we await on either side
        # because the NCCL P2P send/recv pair must rendezvous.
        # (1) Per trainer: announce this step's tensor specs on each
        #     trainer queue so its fetcher knows shapes before the recv.
        for r in range(dp_size):
            entry = prompts[r]
            if entry.input_ids is None:
                raise RuntimeError(
                    f"colocate loop only supports pre-tokenised input_ids "
                    f"prompts (defer_tokenization=False); got entry "
                    f"data_id={entry.data_id} with no input_ids."
                )
            seq_len = _seq_len_from_input_ids(entry.input_ids)
            specs = _build_tensor_specs(
                seq_len,
                hidden_size=hidden_size,
                num_aux_layers=num_aux_layers,
                store_last_hidden_states=store_last_hidden_states,
            )
            train_queues[r].put(
                ColocateTrainSample(
                    step_id=completed_steps,
                    tensor_specs=specs,
                    packed_loss_mask=entry.packed_loss_mask,
                )
            )

        # (2) Per engine: one generate() carrying its engine_tp_size
        #     prompts — those for trainers [e*tp, e*tp+tp). The engine's
        #     TP scheduler subprocesses process the batch together; TP
        #     rank t NCCL-sends batch item t to trainer e*tp+t (the
        #     colocate.patch _send_hidden_states_to_nccl gate enforces
        #     the per-TP-rank partition). At engine_tp_size==1 this is
        #     the original one-prompt-per-engine dispatch.
        engine_refs: list[Any] = []
        for e in range(n_engines):
            grp = prompts[e * engine_tp_size : (e + 1) * engine_tp_size]
            input_ids_ref = ray.put([p.input_ids for p in grp])
            masks = [p.packed_loss_mask for p in grp]
            engine_refs.append(
                inference_engines[e].generate.remote(
                    data_id=[p.data_id for p in grp],
                    input_ids_ref=input_ids_ref,
                    packed_loss_mask_list=masks if any(masks) else None,
                    formatted_prompts=None,
                    return_last_hidden_states=return_last_hidden_states,
                    return_logits=return_logits,
                    multimodal_inputs=None,
                )
            )

        # Both sides run concurrently. Trainer reads from queue,
        # blocks on NCCL recv; engine forwards through sglang, fires
        # spec_training callback, NCCL send unblocks the trainer recv.
        train_refs = [
            actor.train_from_queue.remote(
                step=completed_steps,
                num_batches=1,
            )
            for actor in train_group._actor_handlers
        ]

        try:
            ray.get(engine_refs)
        except Exception:
            logger.exception(
                "[colocate_loop] engine.generate failed at step %d. "
                "Cancelling outstanding trainer futures.",
                completed_steps,
            )
            for ref in train_refs:
                ray.cancel(ref, force=True)
            raise

        train_results = ray.get(train_refs)
        completed_steps += 1
        progress.update(1)

        metrics = train_results[0] if train_results and train_results[0] else {}
        if metrics:
            metrics["train/step"] = completed_steps
            metrics["inference/step"] = completed_steps

            # Optional per-step loss-curve trace, env-gated so it is
            # silent in normal runs. Consumed by the colocate-vs-disagg
            # convergence test (tests/colocate/test_convergence.py),
            # which needs an identically-formatted loss point from both
            # this colocate loop and the disaggregated loop.
            if os.environ.get("TORCHSPEC_LOSS_CURVE_LOG"):
                _lc = metrics.get("train/avg_loss")
                if _lc is not None:
                    logger.info("[loss_curve] step=%d loss=%.6f", completed_steps, float(_lc))

            if enable_perf:
                step_dt = time.time() - t_step
                metrics["perf/step_time"] = step_dt
                if step_dt > 0:
                    metrics["perf/train_capacity"] = args.global_batch_size / step_dt
                if completed_steps % 5 == 0 or completed_steps <= 5:
                    logger.info(
                        "[colocate_loop] step=%d step_time=%.3fs loss=%s lr=%s peak_alloc=%s",
                        completed_steps,
                        step_dt,
                        metrics.get("train/avg_loss"),
                        metrics.get("train/lr"),
                        metrics.get("perf/peak_bytes_allocated"),
                    )

        if save_interval > 0 and completed_steps % save_interval == 0:
            logger.info(
                "[colocate_loop] Saving checkpoint at step %d ...",
                completed_steps,
            )
            train_group.save_model(completed_steps, force_sync=True)
            last_saved_step = completed_steps

    progress.close()

    # Final save: persist the last step if periodic saving is enabled
    # and the last step wasn't already a save-interval boundary.
    if save_interval > 0 and completed_steps > 0 and completed_steps != last_saved_step:
        logger.info(
            "[colocate_loop] Saving final checkpoint at step %d ...",
            completed_steps,
        )
        train_group.save_model(completed_steps, force_sync=True)
        last_saved_step = completed_steps

    logger.info(
        "[colocate_loop] Training complete: completed_steps=%d / num_steps=%d",
        completed_steps,
        num_steps,
    )
