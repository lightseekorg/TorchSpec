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

"""Factorized draft lm_head (``lm_head_rank``)."""

import tempfile
import unittest
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from safetensors.torch import save_file

from torchspec.models.draft.auto import AutoDraftModelConfig, AutoEagle3DraftModel
from torchspec.models.draft.base import LowRankHead, build_lm_head
from torchspec.models.ops.loss import _project

H, V, DIVISOR = 64, 128, 8
RANK = H // DIVISOR

_EAGLE3 = {
    "architectures": ["LlamaForCausalLMEagle3"],
    "model_type": "llama",
    "hidden_size": H,
    "intermediate_size": 128,
    "num_hidden_layers": 1,
    "num_attention_heads": 4,
    "num_key_value_heads": 4,
    "vocab_size": V,
    "max_position_embeddings": 256,
    "rms_norm_eps": 1e-6,
    "rope_theta": 10000.0,
    "pad_token_id": 0,
    "tie_word_embeddings": False,
}


def _build(**overrides):
    torch.manual_seed(0)
    config = AutoDraftModelConfig.from_dict({**_EAGLE3, **overrides})
    return AutoEagle3DraftModel.from_config(config, torch_dtype=torch.float32)


def _tmpdir(test: unittest.TestCase) -> Path:
    directory = tempfile.TemporaryDirectory()
    test.addCleanup(directory.cleanup)
    return Path(directory.name)


class TestLowRankHead(unittest.TestCase):
    def test_shapes_and_parameter_count(self):
        head = LowRankHead(H, V, DIVISOR)

        self.assertEqual(tuple(head.down.weight.shape), (RANK, H))
        self.assertEqual(tuple(head.up.weight.shape), (V, RANK))
        self.assertEqual(sum(p.numel() for p in head.parameters()), RANK * (H + V))

    def test_forward_matches_the_materialized_product(self):
        head = LowRankHead(H, V, DIVISOR)
        hidden = torch.randn(5, H)

        dense = head.up.weight @ head.down.weight
        torch.testing.assert_close(head(hidden), F.linear(hidden, dense), atol=1e-5, rtol=1e-5)

    def test_divisor_must_shrink_the_head(self):
        for bad in (0, 1, -1):
            with self.assertRaisesRegex(ValueError, "at least 2"):
                LowRankHead(H, V, bad)

    def test_divisor_larger_than_hidden_size_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "leaves no rank"):
            LowRankHead(H, V, H + 1)

    def test_rank_is_hidden_size_over_divisor(self):
        self.assertEqual(LowRankHead(H, V, 4).rank, H // 4)

    def test_state_dict_uses_the_down_up_convention(self):
        keys = set(LowRankHead(H, V, DIVISOR).state_dict())

        self.assertEqual(keys, {"down.weight", "up.weight"})


class TestBuildLmHead(unittest.TestCase):
    def test_no_rank_gives_a_dense_linear(self):
        config = AutoDraftModelConfig.from_dict(dict(_EAGLE3))

        head = build_lm_head(config, H, V, V)

        self.assertIsInstance(head, nn.Linear)

    def test_rank_gives_a_factorized_head(self):
        config = AutoDraftModelConfig.from_dict({**_EAGLE3, "lm_head_rank_divisor": DIVISOR})

        head = build_lm_head(config, H, V, V)

        self.assertIsInstance(head, LowRankHead)

    def test_pruned_vocabulary_is_rejected(self):
        config = AutoDraftModelConfig.from_dict({**_EAGLE3, "lm_head_rank_divisor": DIVISOR})

        with self.assertRaisesRegex(ValueError, "pruned draft vocabulary"):
            build_lm_head(config, H, 32, V)


class TestProject(unittest.TestCase):
    """The loss path takes the two factors rather than a materialized head."""

    def test_dense_path_is_a_single_projection(self):
        hidden, weight = torch.randn(5, H), torch.randn(V, H)

        torch.testing.assert_close(_project(hidden, weight, None), F.linear(hidden, weight))

    def test_factorized_path_matches_the_dense_equivalent(self):
        hidden = torch.randn(5, H)
        down, up = torch.randn(RANK, H), torch.randn(V, RANK)

        torch.testing.assert_close(
            _project(hidden, down, up), F.linear(hidden, up @ down), atol=1e-5, rtol=1e-5
        )


class TestModelIntegration(unittest.TestCase):
    def test_draft_exposes_both_factors_to_the_loss(self):
        model = _build(lm_head_rank_divisor=DIVISOR)

        _, projection, _ = model.get_lm_head_params()
        up = model.get_lm_head_up_weight()

        self.assertTrue(model.is_lm_head_factorized)
        self.assertEqual(tuple(projection.shape), (RANK, H))
        self.assertEqual(tuple(up.shape), (V, RANK))

    def test_dense_draft_reports_no_second_factor(self):
        model = _build()

        self.assertFalse(model.is_lm_head_factorized)
        self.assertIsNone(model.get_lm_head_up_weight())

    def test_freeze_covers_both_factors(self):
        model = _build(lm_head_rank_divisor=DIVISOR)

        model.freeze_lm_head()

        self.assertFalse(model.lm_head.down.weight.requires_grad)
        self.assertFalse(model.lm_head.up.weight.requires_grad)
        self.assertTrue(model.midlayer.self_attn.q_proj.weight.requires_grad)

    def test_factorized_head_shrinks_the_trainable_set(self):
        dense = sum(p.numel() for p in _build().parameters() if p.requires_grad)
        low_rank = sum(
            p.numel() for p in _build(lm_head_rank_divisor=DIVISOR).parameters() if p.requires_grad
        )

        self.assertEqual(dense - low_rank, V * H - RANK * (H + V))


class TestSeeding(unittest.TestCase):
    def test_seeding_a_factorized_head_from_the_target_is_refused(self):
        directory = _tmpdir(self) / "target"
        directory.mkdir(parents=True, exist_ok=True)
        save_file({"lm_head.weight": torch.randn(V, H)}, str(directory / "model.safetensors"))
        model = _build(lm_head_rank_divisor=DIVISOR)

        with self.assertRaisesRegex(ValueError, "factorized lm_head"):
            model.load_lm_head(str(directory))


if __name__ == "__main__":
    unittest.main()
