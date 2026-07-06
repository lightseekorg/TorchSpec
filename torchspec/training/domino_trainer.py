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

"""Domino trainer - DFlash trainer + Domino head + base-anchored curriculum.

Reuses DFlashTrainer's hook-based init/model path and adds the curriculum:
lambda is linearly annealed 1 -> 0 and pushed onto the DominoModel before each step.
"""

from argparse import Namespace

import torch
import torch.distributed as dist

from torchspec.models.domino import DominoModel
from torchspec.models.draft.domino import DominoConfig, DominoDraftModel
from torchspec.training.dflash_trainer import DFlashTrainer


class DominoTrainer(DFlashTrainer):
    """Domino-specific trainer (extends DFlashTrainer with the curriculum)."""

    _draft_config_class = DominoConfig

    def __init__(self, args: Namespace):
        super().__init__(args)
        # Steps over which lambda: 1 -> 0. Defaults to the LR schedule length.
        self.curriculum_steps = getattr(args, "domino_curriculum_steps", None) or getattr(
            args, "lr_total_steps", None
        )

    def _build_draft_model(self, config):
        return DominoDraftModel(config)

    def _build_training_wrapper(self, draft_model):
        return DominoModel(
            draft_model=draft_model,
            block_size=self.block_size,
            num_anchors=self.num_anchors,
            loss_objective=self.loss_objective,
            dpace_alpha=self.dpace_alpha,
            loss_decay_gamma=self.loss_decay_gamma,
            ce_loss_alpha=self.ce_loss_alpha,
            l1_loss_alpha=self.l1_loss_alpha,
        )

    def _compute_curriculum_lambda(self, step: int) -> float:
        """Linear anneal lambda from 1 (pure base) to 0 (pure final) over curriculum_steps."""
        total = self.curriculum_steps
        if not total or total <= 0:
            return 0.0
        return max(0.0, 1.0 - float(step) / float(total))

    def _forward(self, batch: dict):
        device = torch.device("cuda")
        input_ids = batch["input_ids"].to(device, non_blocking=True)
        hidden_states = batch["hidden_states"].to(device, non_blocking=True)

        loss_mask = batch["loss_mask"]
        if loss_mask.dim() == 3:
            loss_mask = loss_mask.squeeze(-1)
        loss_mask = loss_mask.to(device, non_blocking=True)

        hidden_states_list = self._split_hidden_states(hidden_states)
        del hidden_states

        output = self.model(
            input_ids=input_ids,
            hidden_states_list=hidden_states_list,
            loss_mask=loss_mask,
            lm_head_weight=self.target_lm_head_weight,
        )
        loss, accuracy, loss_pp, acc_pp, count_pp, aux_metrics = output
        self._last_domino_aux_metrics = aux_metrics
        return loss, accuracy, loss_pp, acc_pp, count_pp, aux_metrics

    def _train_step(
        self,
        batch: dict,
        accumulation_steps: int,
        step: int,
        batch_idx: int,
        num_batches: int,
    ) -> dict:
        # Push the current curriculum weight onto the unwrapped DominoModel so
        # forward() picks it up - avoids threading kwargs through the FSDP wrapper.
        lam = self._compute_curriculum_lambda(step)
        self.dflash.curriculum_lambda = lam
        metrics = super()._train_step(batch, accumulation_steps, step, batch_idx, num_batches)
        metrics["train/curriculum_lambda"] = lam
        for name, value in getattr(self, "_last_domino_aux_metrics", {}).items():
            metrics[f"domino/{name}"] = value.detach()
        return metrics

    def _aggregate_metrics(
        self, all_step_metrics: list[dict], step: int, *, grad_norm: torch.Tensor = None
    ) -> dict:
        metrics = super()._aggregate_metrics(all_step_metrics, step, grad_norm=grad_norm)

        aux_keys = [
            "domino/base_loss",
            "domino/final_loss",
            "domino/correction_norm",
            "domino/correction_abs_mean",
        ]
        for key in aux_keys:
            values = [m[key].float() for m in all_step_metrics if key in m]
            if not values:
                continue
            value = torch.stack(values).mean()
            dist.all_reduce(value, op=dist.ReduceOp.SUM)
            value = value / dist.get_world_size()
            metrics[f"train/{key.removeprefix('domino/')}"] = value.item()

        if all_step_metrics and "train/curriculum_lambda" in all_step_metrics[-1]:
            metrics["train/curriculum_lambda"] = all_step_metrics[-1]["train/curriculum_lambda"]

        return metrics
