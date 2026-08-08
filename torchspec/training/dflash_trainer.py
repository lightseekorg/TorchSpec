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
from torchspec.models.draft.dflash import DFlashConfig, DFlashDraftModel
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

    DSparkTrainer is a subclass that overrides the build hooks:
    - `_draft_config_class`
    - `_build_draft_model`
    - `_build_training_wrapper`
    """

    _draft_config_class = DFlashConfig
    _anchor_slot_offset = 1
    _extra_loss_component_keys: list[str] = []

    def _build_draft_model(self, config):
        """Instantiate the draft network. Overridden by subclasses."""
        return DFlashDraftModel(config)

    def _build_training_wrapper(self, draft_model):
        """Wrap the draft network with the training-objective module."""
        return DFlashModel(
            draft_model=draft_model,
            block_size=self.block_size,
            num_anchors=self.num_anchors,
            loss_objective=self.loss_objective,
            dpace_alpha=self.dpace_alpha,
            loss_decay_gamma=self.loss_decay_gamma,
            ce_loss_alpha=self.ce_loss_alpha,
            l1_loss_alpha=self.l1_loss_alpha,
        )

    def __init__(self, args: Namespace):
        super().__init__(args)
        self.target_lm_head: Optional[torch.nn.Module] = None
        self.num_target_layers = getattr(args, "dflash_num_target_layers", 5)
        self.block_size = getattr(args, "dflash_block_size", 16)
        self.num_anchors = getattr(args, "dflash_num_anchors", 512)
        self.loss_objective = getattr(args, "dflash_loss_objective", "decay")
        self.dpace_alpha = getattr(args, "dflash_dpace_alpha", 0.5)
        self.loss_decay_gamma = getattr(args, "dflash_loss_decay_gamma", 7.0)
        self.ce_loss_alpha = getattr(args, "dflash_ce_loss_alpha", 1.0)
        self.l1_loss_alpha = getattr(args, "dflash_l1_loss_alpha", 0.0)
        self._last_hs_prenorm = getattr(args, "last_hidden_states_prenorm", False)
        self.verifier_norm: Optional[torch.nn.Module] = None

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
            cfg_cls = self._draft_config_class

            if isinstance(draft_model_config, str):
                config = cfg_cls.from_pretrained(draft_model_config)
            elif isinstance(draft_model_config, dict):
                config = cfg_cls(**draft_model_config)
            elif isinstance(draft_model_config, DFlashConfig):
                config = draft_model_config
            else:
                raise TypeError(
                    f"Unsupported draft_model_config type: {type(draft_model_config).__name__}. "
                    f"Expected str, dict, or {cfg_cls.__name__}."
                )

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

            draft_model = self._build_draft_model(config)

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

        dflash_model = self._build_training_wrapper(draft_model)

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
                norm_key=getattr(self.args, "norm_key", "model.norm.weight"),
                load_norm=self._last_hs_prenorm,
                device="cuda",
                dtype=torch.bfloat16,
                trust_remote_code=getattr(self.args, "trust_remote_code", True),
            )
            if self._last_hs_prenorm and self.target_lm_head.norm is None:
                raise RuntimeError(
                    "last_hidden_states_prenorm=True requires a loaded verifier norm"
                )
            norm_status = "loaded" if self.target_lm_head.norm is not None else "not requested"
            logger.info(
                f"[Rank 0] TargetLMHead loaded from {target_model_path}"
                f" (verifier norm: {norm_status})"
            )
        else:
            from transformers import AutoConfig

            config = AutoConfig.from_pretrained(
                target_model_path,
                trust_remote_code=getattr(self.args, "trust_remote_code", True),
            )
            self.target_lm_head = TargetLMHead(config)
            if self._last_hs_prenorm:
                self.target_lm_head._init_norm_structure()
            self.target_lm_head.to(device="cuda", dtype=torch.bfloat16)
            self.target_lm_head.eval()
            self.target_lm_head.requires_grad_(False)

        # Sync norm status from rank 0 so all ranks have the same parameter count
        # before the broadcast loop (a mismatch deadlocks it).
        has_norm = torch.tensor(
            [self.target_lm_head.norm is not None], dtype=torch.int32, device="cuda"
        )
        dist.broadcast(has_norm, src=0)
        if self._last_hs_prenorm and not has_norm.item():
            raise RuntimeError(
                "Rank 0 did not load the verifier norm required by last_hidden_states_prenorm=True"
            )
        if has_norm.item():
            if self.target_lm_head.norm is None:
                logger.warning(
                    f"[Rank {self.dp_rank}] Rank 0 has norm but this rank does not — "
                    "this indicates _init_norm_structure failed; attempting recovery"
                )
                self.target_lm_head._init_norm_structure()
                self.target_lm_head.norm = self.target_lm_head.norm.to(
                    device="cuda", dtype=torch.bfloat16
                )
        elif self.target_lm_head.norm is not None:
            logger.warning(
                f"[Rank {self.dp_rank}] Rank 0 does not have norm — "
                "removing norm on this rank to match"
            )
            self.target_lm_head.norm = None

        dist.barrier()

        for param in self.target_lm_head.parameters():
            dist.broadcast(param.data, src=0)

        self.verifier_norm = self.target_lm_head.norm
        logger.info(
            f"[Rank {self.dp_rank}] TargetLMHead initialized and synced "
            f"(verifier norm loaded: {self.verifier_norm is not None})"
        )

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

    def _forward(
        self, batch: dict
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, dict]:
        device = torch.device("cuda")
        input_ids = batch["input_ids"].to(device, non_blocking=True)
        hidden_states = batch["hidden_states"].to(device, non_blocking=True)

        loss_mask = batch["loss_mask"]
        if loss_mask.dim() == 3:
            loss_mask = loss_mask.squeeze(-1)
        loss_mask = loss_mask.to(device, non_blocking=True)

        last_hidden_states = batch.get("last_hidden_states", None)
        if last_hidden_states is not None:
            last_hidden_states = last_hidden_states.to(device, non_blocking=True)
            if self.verifier_norm is not None:
                with torch.no_grad():
                    last_hidden_states = self.verifier_norm(last_hidden_states)

        hidden_states_list = self._split_hidden_states(hidden_states)
        del hidden_states

        loss, accuracy, loss_per_position, acc_per_position, count_per_position, loss_components = (
            self.model(
                input_ids=input_ids,
                hidden_states_list=hidden_states_list,
                loss_mask=loss_mask,
                lm_head_weight=self.target_lm_head_weight,
                last_hidden_states=last_hidden_states,
            )
        )

        return (
            loss,
            accuracy,
            loss_per_position,
            acc_per_position,
            count_per_position,
            loss_components,
        )

    def _backward(self, loss: torch.Tensor, accumulation_steps: int = 1) -> torch.Tensor:
        scaled_loss = loss / accumulation_steps
        scaled_loss.backward()
        return loss

    # ------------------------------------------------------------------
    # Eval (no-grad forward on CPU-cached data) — mirrors Eagle3Trainer
    # ------------------------------------------------------------------

    def eval_forward(self, batch: dict) -> dict:
        """Single forward pass without backward — returns per-position tensors."""
        with torch.no_grad():
            _, _, loss_per_position, acc_per_position, count_per_position, loss_components = (
                self._forward(batch)
            )
        return {
            "loss_pp": loss_per_position.detach(),
            "acc_pp": acc_per_position.detach(),
            "count_pp": count_per_position.detach(),
            **loss_components,
        }

    def _reduce_position_metrics(
        self,
        all_step_metrics: list[dict],
        *,
        loss_key: str,
        acc_key: str,
        count_key: str,
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
        avg_loss_pp = loss_sum_pp / safe_count_pp
        avg_acc_pp = correct_sum_pp / safe_count_pp
        return avg_loss_pp, avg_acc_pp, count_pp

    def _compute_scalar_metrics(
        self,
        pred_loss_pp: torch.Tensor,
        pred_acc_pp: torch.Tensor,
        pred_count_pp: torch.Tensor,
    ) -> Tuple[float, float]:
        safe_total_count = pred_count_pp.sum().clamp(min=1.0)
        avg_acc = ((pred_acc_pp * pred_count_pp).sum() / safe_total_count).item()

        gamma = self.loss_decay_gamma
        if gamma is not None and gamma > 0:
            k = torch.arange(pred_loss_pp.shape[0], device=pred_loss_pp.device)
            weights = torch.exp(-k.float() / gamma)
        else:
            weights = torch.ones_like(pred_loss_pp)

        weighted_counts = pred_count_pp * weights
        safe_weighted_count = weighted_counts.sum().clamp(min=1.0)
        avg_loss = ((pred_loss_pp * weighted_counts).sum() / safe_weighted_count).item()
        return avg_loss, avg_acc

    def _reduce_loss_components(self, all_step_metrics: list[dict], prefix: str) -> dict:
        """
        Reduce extra scalar loss components into ``{prefix}{key}`` global means.
        """
        out: dict = {}
        for key in self._extra_loss_component_keys:
            vals = [m[key] for m in all_step_metrics if key in m]
            if not vals:
                continue
            value = torch.stack([v.float() for v in vals]).mean()
            if dist.is_initialized() and dist.get_world_size() > 1:
                dist.all_reduce(value, op=dist.ReduceOp.SUM)
                value = value / dist.get_world_size()
            out[f"{prefix}{key}"] = value.item()
        return out

    def eval_from_cache(self) -> dict:
        """Run forward-only eval over all CPU-cached eval samples.

        Samples are stored individually (no padding). Re-collate into batches
        of ``eval_micro_batch_size`` (or ``micro_batch_size``) so the eval
        forward batch size is independent of cache generation throughput.
        """
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

        avg_loss_pp, avg_acc_pp, count_pp = self._reduce_position_metrics(
            all_step_metrics,
            loss_key="loss_pp",
            acc_key="acc_pp",
            count_key="count_pp",
        )

        # Drop anchor slot (index 0) — see _aggregate_metrics for rationale.
        pred_loss_pp = avg_loss_pp[self._anchor_slot_offset :]
        pred_acc_pp = avg_acc_pp[self._anchor_slot_offset :]
        pred_count_pp = count_pp[self._anchor_slot_offset :]

        cumulative = 1.0
        simulated_acc_len = 0.0
        for i in range(pred_acc_pp.shape[0]):
            cumulative *= pred_acc_pp[i].item()
            simulated_acc_len += cumulative

        weighted_avg_loss, avg_acc = self._compute_scalar_metrics(
            pred_loss_pp, pred_acc_pp, pred_count_pp
        )

        metrics: dict = {
            "eval/avg_loss": weighted_avg_loss,
            "eval/avg_acc": avg_acc,
            "eval/simulated_acc_len": simulated_acc_len,
        }
        for i in range(pred_loss_pp.shape[0]):
            metrics[f"eval/ploss_{i}"] = pred_loss_pp[i].item()
            metrics[f"eval/acc_{i}"] = pred_acc_pp[i].item()

        metrics.update(self._reduce_loss_components(all_step_metrics, "eval/"))

        if dist.get_rank() == 0:
            logger.info(
                f"eval: loss={weighted_avg_loss:.4f}, acc={avg_acc:.4f}, "
                f"sim_acc_len={simulated_acc_len:.2f}"
            )

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
        evt_fwd_s = torch.cuda.Event(enable_timing=True)
        evt_fwd_e = torch.cuda.Event(enable_timing=True)
        evt_bwd_s = torch.cuda.Event(enable_timing=True)
        evt_bwd_e = torch.cuda.Event(enable_timing=True)

        evt_fwd_s.record()
        loss, accuracy, loss_per_position, acc_per_position, count_per_position, loss_components = (
            self._forward(batch)
        )
        evt_fwd_e.record()

        evt_bwd_s.record()
        total_loss = self._backward(loss, accumulation_steps=accumulation_steps)
        evt_bwd_e.record()

        return {
            "loss": loss.detach(),
            "accuracy": accuracy.detach(),
            "loss_per_position": loss_per_position.detach(),
            "acc_per_position": acc_per_position.detach(),
            "count_per_position": count_per_position.detach(),
            "total_loss": total_loss.detach(),
            "_fwd_events": (evt_fwd_s, evt_fwd_e),
            "_bwd_events": (evt_bwd_s, evt_bwd_e),
            **loss_components,
        }

    def _aggregate_metrics(
        self, all_step_metrics: list[dict], step: int, *, grad_norm: torch.Tensor = None
    ) -> dict:
        if not all_step_metrics:
            return {}

        avg_loss_pp, avg_acc_pp, count_pp = self._reduce_position_metrics(
            all_step_metrics,
            loss_key="loss_per_position",
            acc_key="acc_per_position",
            count_key="count_per_position",
        )

        # Skip index 0 (anchor slot, always zero); indices 1..B-1 are the
        # predicted tokens at 1..B-1 steps past the anchor. Re-index to 0..B-2
        # so the naming matches Eagle3 (acc_0 = first predicted token).
        pred_loss_pp = avg_loss_pp[self._anchor_slot_offset :]
        pred_acc_pp = avg_acc_pp[self._anchor_slot_offset :]
        pred_count_pp = count_pp[self._anchor_slot_offset :]

        # Simulated accepted length: acc_0 + acc_0*acc_1 + ... + prod(acc_0..acc_{B-2})
        # Models the expected number of consecutively accepted draft tokens.
        cumulative = 1.0
        simulated_acc_len = 0.0
        for i in range(pred_acc_pp.shape[0]):
            cumulative *= pred_acc_pp[i].item()
            simulated_acc_len += cumulative

        avg_loss, avg_acc = self._compute_scalar_metrics(pred_loss_pp, pred_acc_pp, pred_count_pp)

        metrics = {
            "train/avg_loss": avg_loss,
            "train/avg_acc": avg_acc,
            "train/simulated_acc_len": simulated_acc_len,
            "train/grad_norm": grad_norm.item() if grad_norm is not None else 0.0,
            "train/global_step": self.global_step,
            "train/lr": self.optimizer.get_learning_rate(),
            "train/step": step,
        }

        for i in range(pred_loss_pp.shape[0]):
            metrics[f"train/ploss_{i}"] = pred_loss_pp[i].item()
            metrics[f"train/acc_{i}"] = pred_acc_pp[i].item()

        metrics.update(self._reduce_loss_components(all_step_metrics, "train/"))

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
