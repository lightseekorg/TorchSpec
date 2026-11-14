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
from types import SimpleNamespace

import pytest
import torch
from transformers import LlamaConfig
from transformers.models.deepseek_v3.configuration_deepseek_v3 import DeepseekV3Config
from transformers.models.llama.modeling_llama import LlamaRotaryEmbedding as HFLlamaRotaryEmbedding

from torchspec.config.utils import resolve_rope_theta
from torchspec.models.dflash import DFlashModel
from torchspec.models.draft.auto import AutoDraftModelConfig
from torchspec.models.draft.deepseek_eagle import DeepSeekMLAAttention
from torchspec.models.draft.dflash import DFlashAttention, DFlashConfig, DFlashDraftModel
from torchspec.models.draft.dspark import (
    DSparkConfig,
    K3DSparkConfig,
    K3DSparkMLAAttention,
    K3DSparkModel,
)
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


def _tiny_dflash(config_cls, rope_scaling):
    return config_cls(
        hidden_size=8,
        intermediate_size=16,
        num_attention_heads=1,
        num_key_value_heads=1,
        head_dim=8,
        vocab_size=32,
        max_position_embeddings=16,
        num_target_layers=1,
        target_hidden_size=8,
        mask_token_id=31,
        rope_scaling=rope_scaling,
    )


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

    def test_mrope_uses_declared_theta(self):
        cfg = _tiny_llama({"rope_type": "mrope", "mrope_section": [1, 1, 2]})
        self.assertEqual(build_rotary_embedding(cfg, 8, 64).base, 1000000.0)

    def test_yarn_optional_keys_take_transformers_defaults(self):
        cfg = _tiny_llama(
            {"rope_type": "yarn", "factor": 4.0, "original_max_position_embeddings": 16}
        )
        rope = build_rotary_embedding(cfg, 8, 64)
        torch.testing.assert_close(rope.cos_cached[0, 0, :64], _hf_cos(cfg, 64))

    def test_yarn_falls_back_to_config_context_length(self):
        scaling = {"rope_type": "yarn", "factor": 4.0}
        cfg = _tiny_dflash(DFlashConfig, scaling)
        rope = DFlashAttention(cfg).rotary_emb
        expected = _hf_cos(
            _tiny_llama(dict(scaling), rope_theta=cfg.rope_theta, max_position_embeddings=16), 16
        )
        torch.testing.assert_close(rope.cos_cached[0, 0], expected)

    def test_yarn_honors_explicit_attention_factor(self):
        scaling = {
            "rope_type": "yarn",
            "factor": 4.0,
            "original_max_position_embeddings": 4,
            "attention_factor": 1.75,
        }
        cfg = _tiny_dflash(DFlashConfig, scaling)
        rope = DFlashAttention(cfg).rotary_emb
        ref = HFLlamaRotaryEmbedding(
            _tiny_llama(dict(scaling), rope_theta=cfg.rope_theta, max_position_embeddings=16)
        )
        cos, sin = ref(torch.zeros(1, 16, 8), torch.arange(16)[None])
        torch.testing.assert_close(rope.cos_cached[0], cos)
        torch.testing.assert_close(rope.sin_cached[0], sin)

    def test_dynamic_cache_grows_and_resets_like_transformers(self):
        cfg = _tiny_llama({"rope_type": "dynamic", "factor": 2.0})
        rope = build_rotary_embedding(cfg, 8, 64)
        ref = HFLlamaRotaryEmbedding(cfg)
        for length in (8, 96, 8):
            with self.subTest(length=length):
                x = torch.zeros(1, length, 8)
                cos, sin = rope(x, seq_len=length)
                ref_cos, ref_sin = ref(x, torch.arange(length)[None])
                torch.testing.assert_close(cos[0], ref_cos)
                torch.testing.assert_close(sin[0], ref_sin)

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA is required for bf16 autocast")
    def test_dynamic_cache_reset_preserves_phase_under_cuda_autocast(self):
        cfg = _tiny_llama({"rope_type": "dynamic", "factor": 2.0})
        with torch.device("cuda"):
            rope = build_rotary_embedding(cfg, 8, 64)
            x = torch.zeros(1, 1, 96, 8, dtype=torch.bfloat16)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            before = rope(x, seq_len=48)
            rope(x, seq_len=96)
            after = rope(x, seq_len=48)

        for initial, restored in zip(before, after):
            torch.testing.assert_close(initial, restored, atol=0, rtol=0)

    def test_dynamic_cache_can_start_below_the_scaling_threshold(self):
        cfg = _tiny_llama({"rope_type": "dynamic", "factor": 2.0})
        rope = build_rotary_embedding(cfg, 8, 8)
        self.assertEqual(rope.max_seq_len_cached, 8)
        cos, _ = rope(torch.zeros(1, 16, 8), seq_len=16)
        torch.testing.assert_close(cos[0, 0], _hf_cos(cfg, 16))

    def test_shared_builder_accepts_rope_parameters_without_legacy_alias(self):
        cfg = SimpleNamespace(
            rope_parameters={"rope_type": "linear", "factor": 2.0, "rope_theta": 50000.0}
        )
        rope = build_rotary_embedding(cfg, 8, 16)
        self.assertEqual(rope.base, 50000.0)
        self.assertEqual(rope.scaling_factor, 2.0)


LLAMA3 = {
    "rope_type": "llama3",
    "factor": 8.0,
    "low_freq_factor": 1.0,
    "high_freq_factor": 4.0,
    "original_max_position_embeddings": 64,
}
YARN_MINIMAL = {"rope_type": "yarn", "factor": 4.0, "original_max_position_embeddings": 32}


@pytest.mark.parametrize(
    "device",
    [
        "cpu",
        pytest.param(
            "cuda",
            marks=pytest.mark.skipif(
                not torch.cuda.is_bf16_supported(), reason="requires CUDA BF16"
            ),
        ),
    ],
)
@pytest.mark.parametrize(
    "rope_scaling",
    [None, {"rope_type": "linear", "factor": 2.0}, LLAMA3],
    ids=["default", "linear", "llama3"],
)
def test_bf16_eagle_attention_preserves_rope_cache_on_growth(device, rope_scaling):
    length = 257
    if rope_scaling is not None:
        rope_scaling = rope_scaling.copy()
    config = _tiny_llama(rope_scaling, max_position_embeddings=length)
    with torch.device(device):
        attention = LlamaAttention(config).to(dtype=torch.bfloat16)
        reference = build_rotary_embedding(config, config.head_dim, length + 1)
        hidden = torch.zeros(1, length, 2 * config.hidden_size, dtype=torch.bfloat16)
        positions = torch.arange(length)[None]
        mask = hidden.new_zeros(length, length).masked_fill(positions.T < positions, float("-inf"))

    rotary = attention.rotary_emb
    with torch.no_grad():
        _, keys, values = attention(
            hidden, attention_mask=mask, position_ids=positions, use_cache=True
        )
        assert rotary.max_seq_len_cached == length
        attention(hidden, keys, values, attention_mask=mask, position_ids=positions, use_cache=True)

    assert rotary.max_seq_len_cached == length + 1
    for actual, expected in (
        (rotary.cos_cached, reference.cos_cached),
        (rotary.sin_cached, reference.sin_cached),
    ):
        torch.testing.assert_close(actual, expected.to(torch.bfloat16), rtol=0, atol=0)


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

    def test_dflash_and_k3_dspark_use_position_span_in_training(self):
        scaling = {"rope_type": "dynamic", "factor": 2.0}
        for model_cls, config_cls in (
            (DFlashDraftModel, DFlashConfig),
            (K3DSparkModel, K3DSparkConfig),
        ):
            with self.subTest(model=model_cls.__name__):
                config = _tiny_dflash(config_cls, dict(scaling))
                draft = model_cls(config)
                model = DFlashModel(draft, block_size=5, num_anchors=4)
                rotary = draft.layers[0].self_attn.rotary_emb
                initial_inv_freq = rotary.inv_freq.clone()
                input_ids = torch.arange(8).unsqueeze(0)
                loss_mask = torch.ones_like(input_ids)
                hidden, _, _, _ = model._draft_backbone(
                    input_ids, [torch.randn(1, 8, 8)], loss_mask
                )
                self.assertEqual(hidden.shape, (1, 20, 8))
                self.assertEqual(rotary.max_seq_len_cached, 16)
                torch.testing.assert_close(rotary.inv_freq, initial_inv_freq)

    def test_direct_attention_supports_offset_position_ids(self):
        scaling = {"rope_type": "dynamic", "factor": 2.0}
        cfg = _tiny_dflash(DFlashConfig, scaling)
        attention = DFlashAttention(cfg)
        attention(
            torch.randn(1, 4, 8),
            torch.randn(1, 8, 8),
            torch.arange(24, 28)[None],
            torch.arange(24, 32)[None],
        )
        self.assertEqual(attention.rotary_emb.max_seq_len_cached, 32)
        ref = _tiny_llama(dict(scaling), rope_theta=cfg.rope_theta, max_position_embeddings=16)
        torch.testing.assert_close(attention.rotary_emb.cos_cached[0, 0], _hf_cos(ref, 32))

    def test_attention_rejects_mrope_without_three_axis_positions(self):
        scaling = {"rope_type": "mrope", "mrope_section": [1, 1, 2]}
        for attention_cls, config_cls in (
            (DFlashAttention, DFlashConfig),
            (K3DSparkMLAAttention, K3DSparkConfig),
        ):
            with self.subTest(attention=attention_cls.__name__):
                cfg = _tiny_dflash(config_cls, dict(scaling))
                with self.assertRaisesRegex(ValueError, "mRoPE requires three-axis position IDs"):
                    attention_cls(cfg)


if __name__ == "__main__":
    unittest.main()
