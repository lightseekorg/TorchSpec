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

"""EAGLE-4 MoE draft MLPs against a per-token reference."""

import unittest

import torch
import torch.nn.functional as F
from transformers import LlamaConfig

from torchspec.models.draft.eagle4_moe import (
    LATENT_MOE,
    Eagle4MoE,
    split_latent_experts,
    stack_latent_experts,
)

_MOE = dict(
    hidden_size=32,
    initializer_range=0.5,
    enable_moe=True,
    n_routed_experts=8,
    num_experts_per_tok=2,
    moe_intermediate_size=16,
    n_shared_experts=1,
    n_group=4,
    topk_group=2,
    routed_scaling_factor=2.5,
    hidden_act="silu",
)
_LATENT = {
    **_MOE,
    "moe_architecture": LATENT_MOE,
    "hidden_act": "relu2",
    "moe_latent_size": 24,
    "moe_shared_expert_intermediate_size": 40,
}


def _moe(**overrides):
    torch.manual_seed(0)
    moe = Eagle4MoE(LlamaConfig(**{**_MOE, **overrides}))
    with torch.no_grad():
        for p in moe.experts.values():
            p.normal_()
        moe.gate.e_score_correction_bias.normal_(std=0.3)
    return moe


def _reference(moe, x):
    up, down = (moe.experts[name] for name in moe.expert_names)
    out = []
    for t in x:
        scores = (moe.gate.weight @ t).sigmoid()
        biased = (scores + moe.gate.e_score_correction_bias).view(moe.n_group, -1)
        groups = biased.topk(2).values.sum(-1).topk(moe.topk_group).indices
        allowed = torch.full_like(biased, float("-inf"))
        allowed[groups] = biased[groups]
        idx = allowed.flatten().topk(moe.top_k).indices
        weights = scores[idx] / scores[idx].sum() * moe.routed_scaling_factor
        h = moe.fc1_latent_proj(t) if moe.latent else t
        y = 0
        for e, w in zip(idx, weights):
            inner = up[e] @ h
            if moe.latent:
                inner = F.relu(inner) ** 2
            else:
                gate, value = inner.chunk(2)
                inner = F.silu(gate) * value
            y = y + w * (down[e] @ inner)
        y = moe.fc2_latent_proj(y) if moe.latent else y
        out.append(y + moe.shared_experts(t))
    return torch.stack(out)


class TestEagle4MoE(unittest.TestCase):
    def test_matches_reference(self):
        for config in (_MOE, _LATENT):
            with self.subTest(architecture=config.get("moe_architecture", "sparse_swiglu")):
                moe = _moe(**config)
                x = torch.randn(3, 5, 32)
                torch.testing.assert_close(
                    moe(x), _reference(moe, x.view(-1, 32)).view_as(x), rtol=1e-4, atol=1e-4
                )

    def test_gradients_match_reference(self):
        for config in (_MOE, _LATENT):
            with self.subTest(architecture=config.get("moe_architecture", "sparse_swiglu")):
                moe = _moe(**config)
                x = torch.randn(3, 5, 32, requires_grad=True)
                upstream = torch.randn(3, 5, 32)
                grads = []
                for forward in (moe, lambda x: _reference(moe, x.view(-1, 32)).view_as(x)):
                    moe.zero_grad()
                    x.grad = None
                    (forward(x) * upstream).sum().backward()
                    grads.append(
                        {"input": x.grad, **{n: p.grad for n, p in moe.named_parameters()}}
                    )
                for name, grad in grads[0].items():
                    self.assertGreater(grad.abs().sum().item(), 0, name)
                    # ReLU^2 gradients are large; fp32 sums them in a different order.
                    torch.testing.assert_close(grad, grads[1][name], rtol=1e-3, atol=1e-3)

    def test_serving_layout(self):
        swiglu, latent = _moe(), _moe(**_LATENT)
        self.assertEqual(swiglu.experts["w13_weight"].shape, (8, 32, 32))
        self.assertEqual(swiglu.experts["w2_weight"].shape, (8, 32, 16))
        state = {f"layers.0.mlp.{k}": v for k, v in latent.state_dict().items()}
        served = split_latent_experts(state)
        self.assertEqual(served["layers.0.mlp.experts.7.up_proj.weight"].shape, (16, 24))
        self.assertEqual(served["layers.0.mlp.experts.0.down_proj.weight"].shape, (24, 16))
        self.assertNotIn("layers.0.mlp.experts.up_proj", served)
        restored = stack_latent_experts(served)
        self.assertEqual(restored.keys(), state.keys())
        for key, value in state.items():
            self.assertTrue(torch.equal(restored[key], value), key)

    def test_bias_steps_towards_balanced_load(self):
        moe = _moe(n_group=1, topk_group=1).to(torch.bfloat16)
        self.assertEqual(moe.gate.e_score_correction_bias.dtype, torch.float32)
        bias = moe.gate.e_score_correction_bias.clone()
        x = torch.randn(64, 32, dtype=torch.bfloat16)
        moe.eval()(x)
        self.assertEqual(int(moe.gate.load.sum()), 0)
        moe.train()(x)
        load = moe.gate.load.clone()
        self.assertEqual(int(load.sum()), 64 * 2)
        moe.gate.update_bias()
        step = moe.gate.e_score_correction_bias - bias
        self.assertTrue(torch.equal(step.sign(), (load.float().mean() - load).sign()))
        self.assertEqual(int(moe.gate.load.sum()), 0)


if __name__ == "__main__":
    unittest.main()
