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

"""DFlash-specific trainer.

Extends ``Trainer`` with DFlash model initialisation, forward/backward,
and metric aggregation.  Parallel to ``Eagle3Trainer`` — they share the
base class but no model-specific logic.
"""

from argparse import Namespace
from typing import Optional, Tuple

import torch
import torch.distributed as dist

from torchspec.models.dflash import DFlashModel
from torchspec.models.draft.dflash import DFlashDraftModel
from torchspec.training import checkpoint
from torchspec.training.fsdp import apply_fsdp2, fsdp2_load_full_state_dict
from torchspec.training.optimizer import BF16Optimizer
from torchspec.training.trainer import Trainer
from torchspec.utils.distributed import get_gloo_group
from torchspec.utils.logging import logger
from torchspec.utils.tensor import padding


class DFlashTrainer(Trainer):
    """DFlash-specific trainer.

    Extends ``Trainer`` with DFlash model initialisation, forward/backward,
    and metric aggregation.
    """

    def __init__(self, args: Namespace):
        super().__init__(args)
        self._target_components: Optional[torch.nn.Module] = None

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

        init_context = self._get_init_weight_context_manager()

        with init_context():
            draft_model = DFlashDraftModel(draft_model_config)
            draft_model = draft_model.to(dtype=torch.bfloat16)

        dist.barrier(group=get_gloo_group())

        trainable_count = sum(p.numel() for p in draft_model.parameters() if p.requires_grad)
        frozen_count = sum(p.numel() for p in draft_model.parameters() if not p.requires_grad)
        logger.info(
            f"[Rank {self.dp_rank}] DFlash draft model: {trainable_count:,} trainable, "
            f"{frozen_count:,} frozen parameters"
        )

        # Load frozen target embed_tokens + lm_head
        self._init_target_components(target_model_path)

        # DFlash config from draft model
        block_size = getattr(draft_model_config, "block_size", 16)
        dflash_config = getattr(draft_model_config, "dflash_config", {}) or {}
        mask_token_id = dflash_config.get("mask_token_id", None)
        if mask_token_id is None:
            raise ValueError(
                "mask_token_id must be set in draft model config's dflash_config section"
            )

        attention_backend = getattr(self.args, "attention_backend", "sdpa")
        num_anchors = getattr(self.args, "dflash_num_anchors", 512)
        loss_decay_gamma = getattr(self.args, "dflash_loss_decay_gamma", None)

        dflash_model = DFlashModel(
            draft_model=draft_model,
            target_lm_head=self._target_components.lm_head,
            target_embed_tokens=self._target_components.embed_tokens,
            mask_token_id=mask_token_id,
            block_size=block_size,
            attention_backend=attention_backend,
            num_anchors=num_anchors,
            loss_decay_gamma=loss_decay_gamma,
        )

        full_state = dflash_model.state_dict() if dist.get_rank() == 0 else {}

        dflash_model = apply_fsdp2(
            dflash_model,
            mesh=self.dp_mesh,
            cpu_offload=self.fsdp_cpu_offload,
            args=self.args,
        )

        dflash_model = fsdp2_load_full_state_dict(
            dflash_model,
            full_state,
            self.dp_mesh,
            cpu_offload=True if self.fsdp_cpu_offload else None,
        )

        self.model = dflash_model
        self.draft_model = (
            self.model.module.draft_model
            if hasattr(self.model, "module")
            else self.model.draft_model
        )

        decay_style = getattr(self.args, "lr_decay_style", "cosine")
        wsd_decay_steps = None
        wsd_decay_style = None
        if decay_style == "WSD":
            wsd_ratio = getattr(self.args, "lr_wsd_decay_ratio", 0.2)
            wsd_decay_steps = int(wsd_ratio * self.args.lr_total_steps)
            wsd_decay_style = getattr(self.args, "lr_wsd_decay_style", "cosine")
        self.optimizer = BF16Optimizer(
            self.draft_model,
            lr=self.args.learning_rate,
            max_grad_norm=self.args.max_grad_norm,
            warmup_ratio=getattr(self.args, "warmup_ratio", 0.1),
            total_steps=self.args.lr_total_steps,
            decay_style=decay_style,
            wsd_decay_steps=wsd_decay_steps,
            wsd_decay_style=wsd_decay_style,
        )
        self.lr_scheduler = self.optimizer.lr_scheduler

        checkpoint_payload = checkpoint.load(self)
        checkpoint.finalize_load(self, checkpoint_payload)

        self.prof.on_init_end()

        logger.info(f"[Rank {self.dp_rank}] DFlash model initialized with FSDP2")

        return 0

    # ------------------------------------------------------------------
    # Target component loading
    # ------------------------------------------------------------------

    def _init_target_components(self, target_model_path: str) -> None:
        """Load frozen target embed_tokens + lm_head, broadcast to all ranks."""
        from torchspec.models.target.target_utils import TargetEmbeddingsAndHead

        if dist.get_rank() == 0:
            self._target_components = TargetEmbeddingsAndHead.from_pretrained(
                model_path=target_model_path,
                embed_key=getattr(self.args, "embedding_key", "model.embed_tokens.weight"),
                lm_head_key=getattr(self.args, "lm_head_key", "lm_head.weight"),
                device="cuda",
                dtype=torch.bfloat16,
                trust_remote_code=getattr(self.args, "trust_remote_code", True),
            )
            logger.info(f"[Rank 0] TargetEmbeddingsAndHead loaded from {target_model_path}")
        else:
            from transformers import AutoConfig

            config = AutoConfig.from_pretrained(
                target_model_path,
                trust_remote_code=getattr(self.args, "trust_remote_code", True),
            )
            self._target_components = TargetEmbeddingsAndHead(config)
            self._target_components.to(device="cuda", dtype=torch.bfloat16)
            self._target_components.eval()
            self._target_components.requires_grad_(False)

        dist.barrier()

        for param in self._target_components.parameters():
            dist.broadcast(param.data, src=0)

        logger.info(f"[Rank {self.dp_rank}] TargetEmbeddingsAndHead initialized and synced")

    # ------------------------------------------------------------------
    # Forward / backward
    # ------------------------------------------------------------------

    def _forward(self, batch: dict) -> Tuple[torch.Tensor, torch.Tensor]:
        input_ids = padding(batch["input_ids"], left=False).cuda()
        hidden_states = padding(batch["hidden_states"], left=False).cuda()

        loss_mask = batch["loss_mask"]
        if loss_mask.dim() == 3:
            loss_mask = loss_mask.squeeze(-1)
        loss_mask = loss_mask.cuda()

        loss, accuracy = self.model(
            input_ids=input_ids,
            hidden_states=hidden_states,
            loss_mask=loss_mask,
        )
        return loss, accuracy

    def _backward(self, loss: torch.Tensor, accumulation_steps: int = 1) -> torch.Tensor:
        scaled_loss = loss / accumulation_steps
        scaled_loss.backward()
        return scaled_loss

    # ------------------------------------------------------------------
    # Eval
    # ------------------------------------------------------------------

    def eval_forward(self, batch: dict) -> dict:
        with torch.no_grad():
            loss, accuracy = self._forward(batch)
        return {"loss": loss.detach(), "accuracy": accuracy.detach()}

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

        return self._aggregate_eval_metrics(all_metrics)

    def _aggregate_eval_metrics(self, all_step_metrics: list[dict]) -> dict:
        if not all_step_metrics:
            return {}

        avg_loss = torch.stack([m["loss"] for m in all_step_metrics]).mean()
        avg_acc = torch.stack([m["accuracy"] for m in all_step_metrics]).mean()

        dist.all_reduce(avg_loss, op=dist.ReduceOp.AVG)
        dist.all_reduce(avg_acc, op=dist.ReduceOp.AVG)

        metrics = {
            "eval/loss": avg_loss.item(),
            "eval/accuracy": avg_acc.item(),
        }

        if dist.get_rank() == 0:
            logger.info(f"eval: loss={avg_loss.item():.4f}, accuracy={avg_acc.item():.4f}")

        return metrics

    # ------------------------------------------------------------------
    # Subclass contract implementations
    # ------------------------------------------------------------------

    def _train_step(
        self,
        batch: dict,
        accumulation_steps: int,
        step: int,
        batch_idx: int,
        num_batches: int,
    ) -> dict:
        loss, accuracy = self._forward(batch)
        total_loss = self._backward(loss, accumulation_steps=accumulation_steps)

        return {
            "loss": loss.detach(),
            "accuracy": accuracy.detach(),
            "total_loss": total_loss.detach(),
        }

    def _aggregate_metrics(
        self, all_step_metrics: list[dict], step: int, *, grad_norm: torch.Tensor = None
    ) -> dict:
        if not all_step_metrics:
            return {}

        avg_loss = torch.stack([m["loss"] for m in all_step_metrics]).mean()
        avg_acc = torch.stack([m["accuracy"] for m in all_step_metrics]).mean()

        dist.all_reduce(avg_loss, op=dist.ReduceOp.AVG)
        dist.all_reduce(avg_acc, op=dist.ReduceOp.AVG)

        metrics = {
            "train/avg_loss": avg_loss.item(),
            "train/avg_acc": avg_acc.item(),
            "train/loss": avg_loss.item(),
            "train/accuracy": avg_acc.item(),
            "train/grad_norm": grad_norm.item() if grad_norm is not None else 0.0,
            "train/global_step": self.global_step,
            "train/lr": self.optimizer.get_learning_rate(),
            "train/step": step,
        }

        if dist.get_rank() == 0:
            logger.debug(f"step {step}: {metrics}")

        return metrics
