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

"""DSpark trainer — drives the block-anchor DSpark draft model directly.

Mirrors the DeepSpec ``Qwen3DSparkTrainer.run_batch`` (model forward →
``compute_dspark_loss``) but plugs into TorchSpec's async Ray training loop the
same way :class:`DFlashTrainer` does. Unlike DFlash, the DSpark model performs
its own anchor sampling and owns a frozen ``lm_head`` (copied from the target),
so there is no external ``target_lm_head`` and no anchor-slot to drop in the
per-position metrics — every one of the ``block_size`` positions is a draft
prediction.
"""

import glob
import json
import os
from argparse import Namespace
from typing import Optional, Tuple

import torch
import torch.distributed as dist
import torch.nn.functional as F
from safetensors import safe_open

from torchspec.models.dspark_loss import compute_dspark_loss
from torchspec.training import checkpoint
from torchspec.training.fsdp import apply_fsdp2, fsdp2_load_full_state_dict
from torchspec.training.optimizer import BF16Optimizer
from torchspec.training.trainer import Trainer
from torchspec.utils.distributed import get_gloo_group
from torchspec.utils.logging import logger


def _load_target_weight(model_path: str, key: str) -> Optional[torch.Tensor]:
    """Load a single named weight tensor from a HF checkpoint dir.

    Handles sharded (*.index.json) and single-file (safetensors/bin) layouts.
    Returns ``None`` if the key is absent (e.g. tied lm_head).
    """
    if not os.path.exists(model_path):
        from huggingface_hub import snapshot_download

        model_path = snapshot_download(repo_id=model_path)

    index_files = glob.glob(os.path.join(model_path, "*.index.json"))
    if index_files:
        with open(index_files[0], "r") as f:
            weight_map = json.load(f).get("weight_map", {})
        if key not in weight_map:
            return None
        files = [os.path.join(model_path, weight_map[key])]
    else:
        files = glob.glob(os.path.join(model_path, "*.safetensors")) or glob.glob(
            os.path.join(model_path, "*.bin")
        )

    for file_path in files:
        if file_path.endswith(".safetensors"):
            with safe_open(file_path, framework="pt") as f:
                if key in f.keys():
                    return f.get_tensor(key)
        else:
            state_dict = torch.load(file_path, map_location="cpu", weights_only=True)
            if key in state_dict:
                return state_dict[key]
    return None


class DSparkTrainer(Trainer):
    """DSpark-specific trainer (block-anchor parallel drafter)."""

    def __init__(self, args: Namespace):
        super().__init__(args)
        self.num_target_layers = getattr(args, "dspark_num_target_layers", 5)
        self.loss_decay_gamma = getattr(args, "dspark_loss_decay_gamma", 4.0)
        self.ce_loss_alpha = getattr(args, "dspark_ce_loss_alpha", 0.1)
        self.l1_loss_alpha = getattr(args, "dspark_l1_loss_alpha", 0.9)
        self.confidence_head_alpha = getattr(args, "dspark_confidence_head_alpha", 1.0)
        self.block_size = getattr(args, "dspark_block_size", 7)

    # ------------------------------------------------------------------
    # Model init
    # ------------------------------------------------------------------

    def init_model(
        self,
        draft_model_config,
        target_model_path: str,
        mooncake_config=None,
    ) -> int:
        if mooncake_config is not None:
            from torchspec.transfer.mooncake.utils import (
                check_mooncake_master_available,
            )

            check_mooncake_master_available(
                mooncake_config.master_server_address, mooncake_config.metadata_server
            )

        from transformers import AutoConfig

        from torchspec.models.draft.dspark import (
            DSparkConfig,
            Qwen3DSparkModel,
            build_dspark_draft_config,
        )

        if isinstance(draft_model_config, str):
            dspark_cfg = DSparkConfig.from_pretrained(draft_model_config)
        elif isinstance(draft_model_config, dict):
            dspark_cfg = DSparkConfig(**draft_model_config)
        elif isinstance(draft_model_config, DSparkConfig):
            dspark_cfg = draft_model_config
        else:
            raise TypeError(
                f"Unsupported draft_model_config type: {type(draft_model_config).__name__}. "
                f"Expected str, dict, or DSparkConfig."
            )

        target_config = AutoConfig.from_pretrained(
            target_model_path,
            trust_remote_code=getattr(self.args, "trust_remote_code", True),
        )
        config = build_dspark_draft_config(target_config, dspark_cfg)
        self.block_size = int(config.block_size)
        self.num_target_layers = len(config.target_layer_ids)

        init_context = self._get_init_weight_context_manager()
        with init_context():
            draft_model = Qwen3DSparkModel(config)

        if dist.get_rank() == 0:
            embed_key = getattr(self.args, "embedding_key", "model.embed_tokens.weight")
            lm_head_key = getattr(self.args, "lm_head_key", "lm_head.weight")
            embed_w = _load_target_weight(target_model_path, embed_key)
            lm_head_w = _load_target_weight(target_model_path, lm_head_key)
            if embed_w is None:
                raise ValueError(f"Could not load '{embed_key}' from {target_model_path}")
            if lm_head_w is None:
                # Tied embeddings (common for smaller Qwen3): lm_head == embed_tokens.
                logger.info(
                    f"'{lm_head_key}' not found in {target_model_path}; "
                    "assuming tied word embeddings (lm_head = embed_tokens)."
                )
                lm_head_w = embed_w
            with torch.no_grad():
                draft_model.embed_tokens.weight.copy_(embed_w)
                draft_model.lm_head.weight.copy_(lm_head_w)

        # Freeze embed + lm_head on every rank so the optimizer skips them.
        draft_model.set_embedding_head_trainable(False)
        draft_model = draft_model.to(torch.bfloat16)

        dist.barrier(group=get_gloo_group())

        frozen_count = sum(p.numel() for p in draft_model.parameters() if not p.requires_grad)
        trainable_count = sum(p.numel() for p in draft_model.parameters() if p.requires_grad)
        logger.info(
            f"[Rank {self.dp_rank}] DSpark draft model: {trainable_count:,} trainable, "
            f"{frozen_count:,} frozen (embedding + lm_head) parameters"
        )

        full_state = draft_model.state_dict() if dist.get_rank() == 0 else {}

        draft_model = apply_fsdp2(
            draft_model,
            mesh=self.dp_mesh,
            cpu_offload=self.fsdp_cpu_offload,
            args=self.args,
            modules_to_shard=list(draft_model.layers),
        )

        draft_model = fsdp2_load_full_state_dict(
            draft_model,
            full_state,
            self.dp_mesh,
            cpu_offload=True if self.fsdp_cpu_offload else None,
        )

        if getattr(self.args, "compile_model", False):
            logger.info("Compiling DSpark model with torch.compile (inductor backend)")
            draft_model = torch.compile(draft_model)

        self.model = draft_model
        _unwrapped = getattr(self.model, "_orig_mod", self.model)  # torch.compile
        self.draft_model = getattr(_unwrapped, "module", _unwrapped)  # DDP/replicate

        total_steps = self.args.lr_total_steps
        decay_style = getattr(self.args, "lr_decay_style", "cosine")
        warmup_ratio = getattr(self.args, "warmup_ratio", 0.1)

        self.optimizer = BF16Optimizer(
            self.draft_model,
            lr=self.args.learning_rate,
            weight_decay=getattr(self.args, "weight_decay", 0.0),
            max_grad_norm=self.args.max_grad_norm,
            warmup_ratio=warmup_ratio,
            total_steps=total_steps,
            decay_style=decay_style if decay_style != "WSD" else "cosine",
            min_lr=getattr(self.args, "min_lr", 0.0),
        )

        if decay_style == "WSD" and total_steps:
            from torchspec.training.lr_scheduler import LRSchedulerWithWarmup

            wsd_ratio = getattr(self.args, "wsd_decay_ratio", 0.2)
            self.optimizer.scheduler = LRSchedulerWithWarmup(
                self.optimizer.optimizer,
                max_lr=self.args.learning_rate,
                total_steps=total_steps,
                warmup_steps=int(warmup_ratio * total_steps),
                decay_style="WSD",
                min_lr=getattr(self.args, "min_lr", 0.0),
                wsd_decay_steps=int(wsd_ratio * total_steps),
                wsd_decay_style=getattr(self.args, "wsd_decay_style", "cosine"),
            )

        self.lr_scheduler = self.optimizer.lr_scheduler

        checkpoint_payload = checkpoint.load(self)
        checkpoint.finalize_load(self, checkpoint_payload)

        self.prof.on_init_end()
        logger.info(f"[Rank {self.dp_rank}] DSpark model initialized with FSDP2")
        return 0

    # ------------------------------------------------------------------
    # Forward / backward
    # ------------------------------------------------------------------

    def _forward(self, batch: dict):
        device = torch.device("cuda")
        input_ids = batch["input_ids"].to(device, non_blocking=True)
        target_hidden_states = batch["hidden_states"].to(device, non_blocking=True)

        loss_mask = batch["loss_mask"]
        if loss_mask.dim() == 3:
            loss_mask = loss_mask.squeeze(-1)
        loss_mask = loss_mask.to(device, non_blocking=True)

        last_hidden_states = batch.get("last_hidden_states", None)
        if last_hidden_states is not None:
            last_hidden_states = last_hidden_states.to(device, non_blocking=True)
        elif self.l1_loss_alpha > 0 or self.confidence_head_alpha > 0:
            raise ValueError(
                "DSpark requires 'last_hidden_states' in the batch when "
                "l1_loss_alpha > 0 or confidence_head_alpha > 0. "
                "Set inference.store_last_hidden_states=true."
            )

        outputs = self.model(
            input_ids=input_ids,
            target_hidden_states=target_hidden_states,
            loss_mask=loss_mask,
            target_last_hidden_states=last_hidden_states,
        )

        loss, components = compute_dspark_loss(
            outputs=outputs,
            loss_decay_gamma=self.loss_decay_gamma,
            ce_loss_alpha=self.ce_loss_alpha,
            l1_loss_alpha=self.l1_loss_alpha,
            confidence_head_alpha=self.confidence_head_alpha,
        )

        accuracy, loss_pp, acc_pp, count_pp = self._position_metrics(outputs)
        return loss, accuracy, loss_pp, acc_pp, count_pp, components

    @torch.no_grad()
    def _position_metrics(
        self, outputs
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Per-within-block-position loss/accuracy (no decay, binary eval mask).

        Returns ``(accuracy, loss_pp, acc_pp, count_pp)`` with the per-position
        tensors shaped ``[block_size]``. Position ``i`` predicts the token
        ``i + 1`` steps past the anchor.
        """
        draft_logits = outputs.draft_logits
        target_ids = outputs.target_ids
        bsz, num_blocks, block_size, vocab_size = draft_logits.shape

        mask = outputs.eval_mask.to(torch.float32)
        flat_logits = draft_logits.reshape(-1, vocab_size)
        flat_targets = target_ids.reshape(-1)
        ce = F.cross_entropy(flat_logits, flat_targets, reduction="none").view(
            bsz, num_blocks, block_size
        )
        pred = flat_logits.argmax(dim=-1).view(bsz, num_blocks, block_size)
        correct = (pred == target_ids).float() * mask

        count_pp = mask.sum(dim=(0, 1))
        safe_count = count_pp.clamp(min=1.0)
        loss_pp = (ce * mask).sum(dim=(0, 1)) / safe_count
        acc_pp = correct.sum(dim=(0, 1)) / safe_count

        total = count_pp.sum().clamp(min=1.0)
        accuracy = correct.sum() / total
        return accuracy, loss_pp, acc_pp, count_pp

    def _backward(self, loss: torch.Tensor, accumulation_steps: int = 1) -> torch.Tensor:
        scaled_loss = loss / accumulation_steps
        scaled_loss.backward()
        return loss

    # ------------------------------------------------------------------
    # Metric aggregation
    # ------------------------------------------------------------------

    def _reduce_position_metrics(
        self, all_step_metrics: list[dict], *, loss_key: str, acc_key: str, count_key: str
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        loss_sum_pp = torch.stack([m[loss_key] * m[count_key] for m in all_step_metrics]).sum(dim=0)
        correct_sum_pp = torch.stack([m[acc_key] * m[count_key] for m in all_step_metrics]).sum(
            dim=0
        )
        count_pp = torch.stack([m[count_key] for m in all_step_metrics]).sum(dim=0)

        dist.all_reduce(loss_sum_pp, op=dist.ReduceOp.SUM)
        dist.all_reduce(correct_sum_pp, op=dist.ReduceOp.SUM)
        dist.all_reduce(count_pp, op=dist.ReduceOp.SUM)

        safe_count_pp = count_pp.clamp(min=1.0)
        return loss_sum_pp / safe_count_pp, correct_sum_pp / safe_count_pp, count_pp

    def _compute_scalar_metrics(
        self, loss_pp: torch.Tensor, acc_pp: torch.Tensor, count_pp: torch.Tensor
    ) -> Tuple[float, float]:
        safe_total = count_pp.sum().clamp(min=1.0)
        avg_acc = ((acc_pp * count_pp).sum() / safe_total).item()

        gamma = self.loss_decay_gamma
        if gamma is not None and gamma > 0:
            k = torch.arange(loss_pp.shape[0], device=loss_pp.device)
            weights = torch.exp(-k.float() / gamma)
        else:
            weights = torch.ones_like(loss_pp)

        weighted_counts = count_pp * weights
        safe_weighted = weighted_counts.sum().clamp(min=1.0)
        avg_loss = ((loss_pp * weighted_counts).sum() / safe_weighted).item()
        return avg_loss, avg_acc

    def _reduce_loss_components(self, all_step_metrics: list[dict]) -> dict:
        """All-reduce local CE/L1/confidence numerators and denominators for logging."""
        keys = (
            "ce_loss_num",
            "ce_loss_den",
            "l1_loss_num",
            "l1_loss_den",
            "confidence_loss_num",
            "confidence_loss_den",
        )
        comps = [m["components"] for m in all_step_metrics if "components" in m]
        out: dict = {}
        if not comps:
            return out
        device = torch.device("cuda")
        has_confidence = any(c.get("has_confidence") for c in comps)
        for key in keys:
            total = torch.stack([c[key].to(device).float() for c in comps]).sum()
            dist.all_reduce(total, op=dist.ReduceOp.SUM)
            out[key] = total
        result = {"train/ce_loss": (out["ce_loss_num"] / out["ce_loss_den"].clamp(min=1e-6)).item()}
        if out["l1_loss_den"].item() > 0:
            result["train/l1_loss"] = (
                out["l1_loss_num"] / out["l1_loss_den"].clamp(min=1e-6)
            ).item()
        if has_confidence and out["confidence_loss_den"].item() > 0:
            result["train/confidence_loss"] = (
                out["confidence_loss_num"] / out["confidence_loss_den"].clamp(min=1e-6)
            ).item()
        return result

    def _summarize_positions(
        self, loss_pp: torch.Tensor, acc_pp: torch.Tensor, count_pp: torch.Tensor, *, prefix: str
    ) -> dict:
        # Simulated accepted length: acc_0 + acc_0*acc_1 + ... (expected consecutive accepts).
        cumulative = 1.0
        simulated_acc_len = 0.0
        for i in range(acc_pp.shape[0]):
            cumulative *= acc_pp[i].item()
            simulated_acc_len += cumulative

        avg_loss, avg_acc = self._compute_scalar_metrics(loss_pp, acc_pp, count_pp)
        metrics = {
            f"{prefix}/avg_loss": avg_loss,
            f"{prefix}/avg_acc": avg_acc,
            f"{prefix}/simulated_acc_len": simulated_acc_len,
        }
        for i in range(loss_pp.shape[0]):
            metrics[f"{prefix}/ploss_{i}"] = loss_pp[i].item()
            metrics[f"{prefix}/acc_{i}"] = acc_pp[i].item()
        return metrics

    # ------------------------------------------------------------------
    # Subclass contract
    # ------------------------------------------------------------------

    def _train_step(
        self,
        batch: dict,
        accumulation_steps: int,
        step: int,
        batch_idx: int,
        num_batches: int,
    ) -> dict:
        evt_fwd_s = torch.cuda.Event(enable_timing=True)
        evt_fwd_e = torch.cuda.Event(enable_timing=True)
        evt_bwd_s = torch.cuda.Event(enable_timing=True)
        evt_bwd_e = torch.cuda.Event(enable_timing=True)

        evt_fwd_s.record()
        loss, accuracy, loss_pp, acc_pp, count_pp, components = self._forward(batch)
        evt_fwd_e.record()

        evt_bwd_s.record()
        total_loss = self._backward(loss, accumulation_steps=accumulation_steps)
        evt_bwd_e.record()

        return {
            "loss": loss.detach(),
            "accuracy": accuracy.detach(),
            "loss_per_position": loss_pp.detach(),
            "acc_per_position": acc_pp.detach(),
            "count_per_position": count_pp.detach(),
            "components": components,
            "total_loss": total_loss.detach(),
            "_fwd_events": (evt_fwd_s, evt_fwd_e),
            "_bwd_events": (evt_bwd_s, evt_bwd_e),
        }

    def _aggregate_metrics(
        self, all_step_metrics: list[dict], step: int, *, grad_norm: torch.Tensor = None
    ) -> dict:
        if not all_step_metrics:
            return {}

        loss_pp, acc_pp, count_pp = self._reduce_position_metrics(
            all_step_metrics,
            loss_key="loss_per_position",
            acc_key="acc_per_position",
            count_key="count_per_position",
        )

        metrics = self._summarize_positions(loss_pp, acc_pp, count_pp, prefix="train")
        metrics.update(self._reduce_loss_components(all_step_metrics))
        metrics.update(
            {
                "train/grad_norm": grad_norm.item() if grad_norm is not None else 0.0,
                "train/global_step": self.global_step,
                "train/lr": self.optimizer.get_learning_rate(),
                "train/step": step,
            }
        )

        fwd_ms = sum(
            m["_fwd_events"][0].elapsed_time(m["_fwd_events"][1])
            for m in all_step_metrics
            if "_fwd_events" in m
        )
        bwd_ms = sum(
            m["_bwd_events"][0].elapsed_time(m["_bwd_events"][1])
            for m in all_step_metrics
            if "_bwd_events" in m
        )
        metrics["perf/forward_time"] = fwd_ms / 1000.0
        metrics["perf/backward_time"] = bwd_ms / 1000.0

        if dist.get_rank() == 0 and (step % 5 == 0 or step <= 5):
            logger.info(
                f"COMPUTE_BREAKDOWN step={step}: forward={fwd_ms:.1f}ms backward={bwd_ms:.1f}ms"
            )
        if dist.get_rank() == 0:
            logger.debug(f"step {step}: {metrics}")
        return metrics

    # ------------------------------------------------------------------
    # Eval (forward-only on CPU-cached samples) — mirrors DFlashTrainer
    # ------------------------------------------------------------------

    def eval_forward(self, batch: dict) -> dict:
        with torch.no_grad():
            _, _, loss_pp, acc_pp, count_pp, _ = self._forward(batch)
        return {
            "loss_pp": loss_pp.detach(),
            "acc_pp": acc_pp.detach(),
            "count_pp": count_pp.detach(),
        }

    def eval_from_cache(self) -> dict:
        if not getattr(self, "_eval_cache", None):
            return {}

        eval_mbs = getattr(self.args, "eval_micro_batch_size", None) or self.args.micro_batch_size
        self.model.eval()
        all_metrics: list[dict] = []
        for i in range(0, len(self._eval_cache), eval_mbs):
            chunk = self._eval_cache[i : i + eval_mbs]
            batch = self._eval_collator(chunk)
            gpu_batch = {
                k: v.cuda() if isinstance(v, torch.Tensor) else v for k, v in batch.items()
            }
            all_metrics.append(self.eval_forward(gpu_batch))
        self.model.train()

        if not all_metrics:
            return {}
        loss_pp, acc_pp, count_pp = self._reduce_position_metrics(
            all_metrics, loss_key="loss_pp", acc_key="acc_pp", count_key="count_pp"
        )
        metrics = self._summarize_positions(loss_pp, acc_pp, count_pp, prefix="eval")
        if dist.get_rank() == 0:
            logger.info(
                f"eval: loss={metrics['eval/avg_loss']:.4f}, acc={metrics['eval/avg_acc']:.4f}, "
                f"sim_acc_len={metrics['eval/simulated_acc_len']:.2f}"
            )
        return metrics
