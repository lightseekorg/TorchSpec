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

"""DFlash trainer — extends Trainer with DFlash-specific model init, forward, and metrics."""

from argparse import Namespace
from typing import List, Optional, Tuple

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


class DFlashTrainer(Trainer):
    """DFlash-specific trainer.

    Extends ``Trainer`` with DFlash model initialisation (dual-source KV draft model),
    forward/backward with anchor sampling + block-causal mask, and metric aggregation.
    """

    def __init__(self, args: Namespace):
        super().__init__(args)
        self.target_lm_head: Optional[torch.nn.Module] = None
        self.num_target_layers = getattr(args, "dflash_num_target_layers", 5)
        self.block_size = getattr(args, "dflash_block_size", 16)
        self.num_anchors = getattr(args, "dflash_num_anchors", 512)
        self.loss_decay_gamma = getattr(args, "dflash_loss_decay_gamma", 7.0)

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
            from torchspec.models.draft.dflash import DFlashConfig

            if isinstance(draft_model_config, str):
                config = DFlashConfig.from_pretrained(draft_model_config)
            elif isinstance(draft_model_config, dict):
                config = DFlashConfig(**draft_model_config)
            elif isinstance(draft_model_config, DFlashConfig):
                config = draft_model_config
            else:
                config = draft_model_config

            if not hasattr(config, "num_target_layers") or config.num_target_layers is None:
                config.num_target_layers = self.num_target_layers
            if not hasattr(config, "target_hidden_size") or config.target_hidden_size is None:
                config.target_hidden_size = config.hidden_size
            if (
                not hasattr(config, "target_num_hidden_layers")
                or config.target_num_hidden_layers is None
            ):
                from transformers import AutoConfig

                target_config = AutoConfig.from_pretrained(
                    target_model_path,
                    trust_remote_code=getattr(self.args, "trust_remote_code", True),
                )
                config.target_num_hidden_layers = target_config.num_hidden_layers

            draft_model = DFlashDraftModel(config)

        if dist.get_rank() == 0:
            draft_model.load_embedding(
                target_model_path,
                embedding_key=getattr(self.args, "embedding_key", "model.embed_tokens.weight"),
            )

        draft_model.freeze_embedding()
        draft_model = draft_model.to(torch.bfloat16)

        dist.barrier(group=get_gloo_group())

        frozen_count = sum(p.numel() for p in draft_model.parameters() if not p.requires_grad)
        trainable_count = sum(p.numel() for p in draft_model.parameters() if p.requires_grad)
        logger.info(
            f"[Rank {self.dp_rank}] DFlash draft model: {trainable_count:,} trainable, "
            f"{frozen_count:,} frozen (embedding) parameters"
        )

        dflash_model = DFlashModel(
            draft_model=draft_model,
            block_size=self.block_size,
            num_anchors=self.num_anchors,
            loss_decay_gamma=self.loss_decay_gamma,
        )

        full_state = dflash_model.state_dict() if dist.get_rank() == 0 else {}

        dflash_model = apply_fsdp2(
            dflash_model,
            mesh=self.dp_mesh,
            cpu_offload=self.fsdp_cpu_offload,
            args=self.args,
            modules_to_shard=list(draft_model.layers),
        )

        dflash_model = fsdp2_load_full_state_dict(
            dflash_model,
            full_state,
            self.dp_mesh,
            cpu_offload=True if self.fsdp_cpu_offload else None,
        )

        if getattr(self.args, "compile_model", False):
            logger.info("Compiling DFlash model with torch.compile (inductor backend)")
            dflash_model = torch.compile(dflash_model)

        self.model = dflash_model
        # Unwrap torch.compile and/or DDP module wrappers to access underlying DFlashModel
        _unwrapped = getattr(self.model, "_orig_mod", self.model)  # torch.compile
        self.dflash = getattr(_unwrapped, "module", _unwrapped)  # DDP/replicate
        self.draft_model = self.dflash.draft_model

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

        self._init_target_lm_head(target_model_path)

        if self.target_lm_head is None:
            raise ValueError(
                "target_lm_head is required but was None. Ensure _init_target_lm_head succeeded."
            )
        self.target_lm_head_weight = self.target_lm_head.lm_head.weight

        self.prof.on_init_end()

        logger.info(f"[Rank {self.dp_rank}] DFlash model initialized with FSDP2")

        return 0

    # ------------------------------------------------------------------
    # Target LM head (same as Eagle3Trainer)
    # ------------------------------------------------------------------

    def _init_target_lm_head(self, target_model_path: str) -> None:
        from torchspec.models.target.target_utils import TargetLMHead

        if dist.get_rank() == 0:
            self.target_lm_head = TargetLMHead.from_pretrained(
                model_path=target_model_path,
                lm_head_key=getattr(self.args, "lm_head_key", "lm_head.weight"),
                device="cuda",
                dtype=torch.bfloat16,
                trust_remote_code=getattr(self.args, "trust_remote_code", True),
            )
            logger.info(f"[Rank 0] TargetLMHead loaded from {target_model_path}")
        else:
            from transformers import AutoConfig

            config = AutoConfig.from_pretrained(
                target_model_path,
                trust_remote_code=getattr(self.args, "trust_remote_code", True),
            )
            self.target_lm_head = TargetLMHead(config)
            self.target_lm_head.to(device="cuda", dtype=torch.bfloat16)
            self.target_lm_head.eval()
            self.target_lm_head.requires_grad_(False)

        dist.barrier()

        for param in self.target_lm_head.parameters():
            dist.broadcast(param.data, src=0)

        logger.info(f"[Rank {self.dp_rank}] TargetLMHead initialized and synced")

    # ------------------------------------------------------------------
    # Forward / backward
    # ------------------------------------------------------------------

    def _split_hidden_states(self, hidden_states: torch.Tensor) -> List[torch.Tensor]:
        """Split concatenated hidden states [B, seq_len, num_layers*D] into per-layer list.

        The target model concatenates hidden states from `num_target_layers` layers
        along the last dimension. We split them back into a list of [B, seq_len, D] tensors.
        """
        total_dim = hidden_states.shape[-1]
        per_layer_dim = total_dim // self.num_target_layers
        return list(hidden_states.split(per_layer_dim, dim=-1))

    def _forward(self, batch: dict) -> Tuple[torch.Tensor, torch.Tensor]:
        device = torch.device("cuda")
        input_ids = batch["input_ids"].to(device, non_blocking=True)
        hidden_states = batch["hidden_states"].to(device, non_blocking=True)

        loss_mask = batch["loss_mask"]
        if loss_mask.dim() == 3:
            loss_mask = loss_mask.squeeze(-1)
        loss_mask = loss_mask.to(device, non_blocking=True)

        hidden_states_list = self._split_hidden_states(hidden_states)
        del hidden_states

        loss, accuracy = self.model(
            input_ids=input_ids,
            hidden_states_list=hidden_states_list,
            loss_mask=loss_mask,
            lm_head_weight=self.target_lm_head_weight,
        )

        return loss, accuracy

    def _backward(self, loss: torch.Tensor, accumulation_steps: int = 1) -> torch.Tensor:
        scaled_loss = loss / accumulation_steps
        scaled_loss.backward()
        return loss

    # ------------------------------------------------------------------
    # Eval — disabled for DFlash (eval hangs in colocate/SGLang mode;
    # benchmark τ separately after training via scripts/modal/modal_dflash_benchmark_sglang.py)
    # ------------------------------------------------------------------

    def eval_from_cache(self) -> dict:
        return {}

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
        evt_fwd_s = torch.cuda.Event(enable_timing=True)
        evt_fwd_e = torch.cuda.Event(enable_timing=True)
        evt_bwd_s = torch.cuda.Event(enable_timing=True)
        evt_bwd_e = torch.cuda.Event(enable_timing=True)

        evt_fwd_s.record()
        loss, accuracy = self._forward(batch)
        evt_fwd_e.record()

        evt_bwd_s.record()
        total_loss = self._backward(loss, accumulation_steps=accumulation_steps)
        evt_bwd_e.record()

        return {
            "loss": loss.detach(),
            "accuracy": accuracy.detach(),
            "total_loss": total_loss.detach(),
            "_fwd_events": (evt_fwd_s, evt_fwd_e),
            "_bwd_events": (evt_bwd_s, evt_bwd_e),
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
            "train/grad_norm": grad_norm.item() if grad_norm is not None else 0.0,
            "train/global_step": self.global_step,
            "train/lr": self.optimizer.get_learning_rate(),
            "train/step": step,
        }

        # Sub-timing breakdown (forward vs backward)
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
