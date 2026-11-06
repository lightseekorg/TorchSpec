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

"""Tests that a draft's rotary base is the one its config declares.

transformers 5.x moved `rope_theta` from a top-level config attribute into
`rope_parameters`, which silently reduced every affected draft to the library
default of 10000 while its config advertised something else. The sweep over
`configs/draft_models/` is the part that fails if a future release moves the
field again.
"""

import glob
import json
import os
import unittest

import torch
from transformers import LlamaConfig
from transformers.models.deepseek_v3.configuration_deepseek_v3 import DeepseekV3Config
from transformers.models.llama.modeling_llama import LlamaRotaryEmbedding as HFLlamaRotaryEmbedding

from torchspec.config.utils import resolve_rope_theta
from torchspec.models.draft.auto import AutoDraftModelConfig
from torchspec.models.draft.deepseek_eagle import DeepSeekMLAAttention
from torchspec.models.draft.dflash import DFlashAttention, DFlashConfig
from torchspec.models.draft.dspark import DSparkConfig
from torchspec.models.draft.llama3_eagle import LlamaAttention, build_rotary_embedding

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DRAFT_CONFIG_DIR = os.path.join(REPO_ROOT, "configs", "draft_models")

YARN = {
    "rope_type": "yarn",
    "factor": 32.0,
    "original_max_position_embeddings": 64,
    "beta_fast": 32,
    "beta_slow": 1,
    "mscale": 1.0,
    "mscale_all_dim": 1.0,
}


def _declared_theta(raw):
    """The base a config file asks for, from wherever the file happens to put it."""
    block = raw.get("rope_parameters") or raw.get("rope_scaling") or {}
    nested = block.get("rope_theta")
    return nested if nested is not None else raw.get("rope_theta")


class TestResolveRopeTheta(unittest.TestCase):
    def test_prefers_top_level_attribute(self):
        cfg = DFlashConfig(rope_theta=50000.0)
        self.assertEqual(resolve_rope_theta(cfg), 50000.0)

    def test_falls_back_to_rope_parameters(self):
        # transformers 5.x absorbs a top-level rope_theta into rope_parameters,
        # leaving no attribute for the naive read to find.
        cfg = LlamaConfig(rope_theta=1000000.0)
        self.assertIsNone(getattr(cfg, "rope_theta", None))
        self.assertEqual(resolve_rope_theta(cfg), 1000000.0)

    def test_default_when_declared_nowhere(self):
        cfg = DFlashConfig()
        del cfg.rope_theta
        self.assertEqual(resolve_rope_theta(cfg), 10000.0)
        self.assertIsNone(resolve_rope_theta(cfg, default=None))

    def test_conflicting_values_are_refused(self):
        cfg = LlamaConfig(rope_theta=1000000.0)
        cfg.rope_theta = 10000.0  # stale legacy value left beside the block
        with self.assertRaisesRegex(ValueError, "Conflicting rope_theta"):
            resolve_rope_theta(cfg)


class TestShippedDraftConfigs(unittest.TestCase):
    """Every config in configs/draft_models/ must resolve to the base it declares."""

    def test_declared_theta_is_what_resolves(self):
        paths = sorted(glob.glob(os.path.join(DRAFT_CONFIG_DIR, "*.json")))
        self.assertGreater(len(paths), 0, "no draft configs found")
        for path in paths:
            with self.subTest(config=os.path.basename(path)):
                declared = _declared_theta(json.load(open(path)))
                if declared is None:
                    self.skipTest("config declares no rope_theta")
                cfg = AutoDraftModelConfig.from_file(path)
                self.assertEqual(resolve_rope_theta(cfg), float(declared))


class TestAttentionUsesResolvedTheta(unittest.TestCase):
    """Each rotary construction path must reach the resolved base, not the default."""

    def test_llama_yarn_path(self):
        cfg = LlamaConfig(
            hidden_size=64,
            num_attention_heads=4,
            num_key_value_heads=4,
            max_position_embeddings=2048,
            rope_theta=1000000.0,
            rope_parameters=dict(YARN),
        )
        cfg.target_hidden_size = 64
        self.assertEqual(LlamaAttention(cfg).rotary_emb.base, 1000000.0)

    def test_deepseek_mla_yarn_path(self):
        cfg = DeepseekV3Config(
            hidden_size=64,
            num_attention_heads=4,
            q_lora_rank=32,
            kv_lora_rank=16,
            qk_nope_head_dim=16,
            qk_rope_head_dim=8,
            v_head_dim=16,
            max_position_embeddings=2048,
            rope_theta=50000.0,
            rope_parameters=dict(YARN),
        )
        self.assertEqual(DeepSeekMLAAttention(cfg).rotary_emb.base, 50000.0)

    def test_dflash_path(self):
        cfg = DFlashConfig(
            hidden_size=64,
            num_attention_heads=4,
            num_key_value_heads=2,
            max_position_embeddings=512,
            rope_theta=1000000.0,
        )
        self.assertEqual(DFlashAttention(cfg).rotary_emb.base, 1000000.0)


def _tiny_llama(rope_scaling, rope_theta=1000000.0, max_position_embeddings=64):
    return LlamaConfig(
        hidden_size=8,
        num_attention_heads=1,
        num_key_value_heads=1,
        head_dim=8,
        max_position_embeddings=max_position_embeddings,
        rope_theta=rope_theta,
        rope_scaling=rope_scaling,
    )


def _hf_cos(cfg, seq_len):
    cos, _ = HFLlamaRotaryEmbedding(cfg)(torch.zeros(1, seq_len, 8), torch.arange(seq_len)[None])
    return cos[0]


class TestScaledRopeMatchesTransformers(unittest.TestCase):
    """Scaled branches must reproduce transformers' tables, not only carry the base."""

    def test_linear_and_dynamic_use_declared_theta(self):
        for rope_type in ("linear", "dynamic"):
            for theta in (10000.0, 1000000.0):
                with self.subTest(rope_type=rope_type, theta=theta):
                    cfg = _tiny_llama({"rope_type": rope_type, "factor": 2.0}, theta)
                    rope = build_rotary_embedding(cfg, 8, 64)
                    self.assertEqual(rope.base, theta)
                    torch.testing.assert_close(rope.cos_cached[0, 0, :64], _hf_cos(cfg, 64))

    def test_dynamic_resets_after_a_longer_sequence(self):
        cfg = _tiny_llama({"rope_type": "dynamic", "factor": 2.0})
        rope = build_rotary_embedding(cfg, 8, 64)
        x = torch.zeros(1, 1, 128, 8)
        rope(x, seq_len=128)
        cos, _ = rope(x, seq_len=64)
        torch.testing.assert_close(cos[0, 0], _hf_cos(cfg, 64))

    def test_mrope_uses_declared_theta(self):
        cfg = _tiny_llama({"rope_type": "mrope", "mrope_section": [1, 1, 2]})
        self.assertEqual(build_rotary_embedding(cfg, 8, 64).base, 1000000.0)

    def test_yarn_optional_keys_take_transformers_defaults(self):
        cfg = _tiny_llama(
            {"rope_type": "yarn", "factor": 4.0, "original_max_position_embeddings": 16}
        )
        rope = build_rotary_embedding(cfg, 8, 64)
        torch.testing.assert_close(rope.cos_cached[0, 0, :64], _hf_cos(cfg, 64))


LLAMA3 = {
    "rope_type": "llama3",
    "factor": 8.0,
    "low_freq_factor": 1.0,
    "high_freq_factor": 4.0,
    "original_max_position_embeddings": 64,
}
YARN_MINIMAL = {"rope_type": "yarn", "factor": 4.0, "original_max_position_embeddings": 32}


class TestDFlashRopeScaling(unittest.TestCase):
    """DFlash configs accept the rope blocks LlamaConfig does, and the attention applies them."""

    def test_configs_accept_hf_rope_blocks(self):
        for cls in (DFlashConfig, DSparkConfig):
            for scaling in (LLAMA3, YARN_MINIMAL):
                for key in ("rope_scaling", "rope_parameters"):
                    with self.subTest(cls=cls.__name__, rope_type=scaling["rope_type"], key=key):
                        cfg = cls(
                            hidden_size=64,
                            num_attention_heads=4,
                            num_key_value_heads=2,
                            max_position_embeddings=128,
                            rope_theta=500000.0,
                            **{key: dict(scaling)},
                        )
                        self.assertEqual(cfg.rope_scaling["rope_type"], scaling["rope_type"])
                        self.assertEqual(DFlashAttention(cfg).rotary_emb.base, 500000.0)

    def test_draft_config_file_path_accepts_yarn(self):
        cfg = AutoDraftModelConfig.from_dict(
            {
                "architectures": ["DFlashDraftModel"],
                "max_position_embeddings": 128,
                "rope_scaling": dict(YARN_MINIMAL),
            }
        )
        self.assertEqual(cfg.rope_scaling["beta_fast"], 32.0)

    def test_attention_applies_declared_scaling(self):
        scaling = {"rope_type": "linear", "factor": 2.0}
        cfg = DFlashConfig(
            hidden_size=8,
            num_attention_heads=1,
            num_key_value_heads=1,
            head_dim=8,
            max_position_embeddings=8,
            rope_theta=1000000.0,
            rope_scaling=dict(scaling),
        )
        rope = DFlashAttention(cfg).rotary_emb
        expected = _hf_cos(_tiny_llama(dict(scaling), max_position_embeddings=8), 8)
        torch.testing.assert_close(rope.cos_cached[0, 0, :8], expected)


if __name__ == "__main__":
    unittest.main()
