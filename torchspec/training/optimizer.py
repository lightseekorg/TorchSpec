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

import torch

from torchspec.training.lr_scheduler import LRSchedulerWithWarmup
from torchspec.utils.logging import print_on_rank0


class BF16Optimizer:
    def __init__(
        self,
        model,
        lr,
        weight_decay=0.0,
        max_grad_norm=0.5,
        total_steps=800_000,
        warmup_ratio=0.015,
        decay_style="cosine",
        wsd_decay_steps=None,
        wsd_decay_style=None,
    ):
        self.model = model
        self.model_params = [p for p in model.parameters() if p.requires_grad]
        self.max_grad_norm = max_grad_norm
        self.fp32_params = [p.detach().clone().to(torch.float32) for p in self.model_params]
        for mp in self.fp32_params:
            mp.requires_grad = True
        self.optimizer = torch.optim.AdamW(self.fp32_params, lr=lr, weight_decay=weight_decay)
        self.scheduler = LRSchedulerWithWarmup(
            self.optimizer,
            max_lr=lr,
            total_steps=total_steps,
            warmup_steps=int(warmup_ratio * total_steps),
            decay_style=decay_style,
            wsd_decay_steps=wsd_decay_steps,
            wsd_decay_style=wsd_decay_style,
        )

    def step(self, closure=None):
        """Perform optimizer step with gradient clipping.

        Args:
            closure: Ignored, for compatibility with PyTorch optimizer interface.

        Returns:
            grad_norm: The gradient norm before clipping (for logging).
        """
        with torch.no_grad():
            for p, mp in zip(self.model_params, self.fp32_params):
                mp.grad = p.grad.detach().to(torch.float32) if p.grad is not None else None

        grad_norm = torch.nn.utils.clip_grad_norm_(self.fp32_params, self.max_grad_norm)
        if grad_norm > 0.0:
            self.optimizer.step()

        self.optimizer.zero_grad()
        self.scheduler.step()
        with torch.no_grad():
            for p, mp in zip(self.model_params, self.fp32_params):
                p.data.copy_(mp.data.to(p.dtype))
                p.grad = None

        return grad_norm

    def zero_grad(self, set_to_none=True):
        self.optimizer.zero_grad(set_to_none=set_to_none)
        for p in self.model_params:
            if set_to_none:
                p.grad = None
            elif p.grad is not None:
                p.grad.zero_()

    def load_state_dict(self, state_dict):
        self.optimizer.load_state_dict(state_dict)
        print_on_rank0("Successfully loaded optimizer state_dict.")

    def sync_fp32_params_from_model(self):
        """Reinitialize fp32_params from model params. Call after loading model checkpoint."""
        with torch.no_grad():
            for mp, p in zip(self.fp32_params, self.model_params):
                mp.data.copy_(p.data.to(torch.float32))

    def state_dict(self):
        return self.optimizer.state_dict()

    def get_learning_rate(self):
        return self.optimizer.param_groups[0]["lr"]

    @property
    def state(self):
        return self.optimizer.state

    @property
    def param_groups(self):
        return self.optimizer.param_groups

    @property
    def lr_scheduler(self):
        return self.scheduler
