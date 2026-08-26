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

"""Tests for DSpark (DFlash backbone + Markov / confidence heads + L1 distillation).

Pins the DSpark wiring so future refactors can't silently break the objective:

1. DSparkConfig / DSparkDraftModel: head construction, subclass relationship.
2. forward returns the 6-tuple with detached per-component losses.
3. Loss-wiring invariants (no DeepSpec dependency):
   - internal identity: combined loss == ce_a*ce + l1_a*l1 + cf_a*conf  (so the
     logged loss_components are trustworthy)
   - all-masked batch -> loss 0
   - gradients reach markov + confidence + backbone; embedding stays frozen
   - next-token convention: every within-block slot is supervised (B predictions)
4. Markov / confidence head unit math.
5. Algorithm dispatch (DSparkConfig resolves from the JSON and is checked before
   DFlashConfig since it subclasses it).
"""

import math
import unittest
from pathlib import Path

import torch

from torchspec.config import load_config
from torchspec.models.draft.auto import AutoDraftModelConfig, AutoEagle3DraftModel
from torchspec.models.draft.dflash import DFlashConfig
from torchspec.models.draft.dspark import (
    AcceptRatePredictor,
    DSparkConfig,
    DSparkDraftModel,
    K3DSparkConfig,
    K3DSparkModel,
    VanillaMarkov,
)
from torchspec.models.draft.llama3_eagle import LlamaYarnRotaryEmbedding, yarn_get_mscale
from torchspec.models.dspark import DSparkModel

CE_A, L1_A, CF_A = 0.1, 0.9, 1.0
ROOT = Path(__file__).resolve().parents[1]


def _make_dspark_config(
    H=64,
    V=128,
    num_target_layers=2,
    markov_rank=16,
    enable_confidence_head=True,
    confidence_head_with_markov=True,
    fc_norm=False,
):
    return DSparkConfig(
        hidden_size=H,
        intermediate_size=256,
        num_hidden_layers=1,
        num_attention_heads=4,
        num_key_value_heads=2,
        vocab_size=V,
        rms_norm_eps=1e-6,
        max_position_embeddings=512,
        rope_theta=10000.0,
        num_target_layers=num_target_layers,
        target_hidden_size=H,
        target_num_hidden_layers=12,
        mask_token_id=V - 1,
        markov_rank=markov_rank,
        markov_head_type="vanilla",
        enable_confidence_head=enable_confidence_head,
        confidence_head_with_markov=confidence_head_with_markov,
        fc_norm=fc_norm,
    )


def _make_dspark_model(
    block_size=4,
    num_anchors=6,
    l1_loss_alpha=L1_A,
    confidence_head_alpha=CF_A,
    **cfg_kw,
):
    config = _make_dspark_config(**cfg_kw)
    draft = DSparkDraftModel(config).to(dtype=torch.float32)
    draft.freeze_embedding()
    return DSparkModel(
        draft_model=draft,
        block_size=block_size,
        num_anchors=num_anchors,
        loss_decay_gamma=4.0,
        ce_loss_alpha=CE_A,
        l1_loss_alpha=l1_loss_alpha,
        confidence_head_alpha=confidence_head_alpha,
    )


def _batch(B=2, S=24, H=64, V=128, num_target_layers=2, all_masked=False, seed=0):
    g = torch.Generator().manual_seed(seed)
    input_ids = torch.randint(0, V, (B, S), generator=g)
    hidden_states_list = [torch.randn(B, S, H, generator=g) for _ in range(num_target_layers)]
    loss_mask = torch.zeros(B, S) if all_masked else torch.ones(B, S)
    if not all_masked:
        loss_mask[:, :2] = 0  # prompt
    lm_head_weight = torch.randn(V, H, generator=g)
    last_hidden_states = torch.randn(B, S, H, generator=g)
    return dict(
        input_ids=input_ids,
        hidden_states_list=hidden_states_list,
        loss_mask=loss_mask,
        lm_head_weight=lm_head_weight,
        last_hidden_states=last_hidden_states,
    )


class TestDSparkConfig(unittest.TestCase):
    def test_subclasses_dflash_and_attrs(self):
        cfg = _make_dspark_config(markov_rank=32)
        self.assertIsInstance(cfg, DFlashConfig)  # ordering hazard: check DSpark first
        self.assertEqual(cfg.model_type, "qwen3_dspark")
        self.assertEqual(cfg.markov_rank, 32)
        self.assertTrue(cfg.enable_confidence_head)
        self.assertFalse(cfg.fc_norm)

    def test_ci_config_uses_standard_dspark_shape(self):
        config_path = ROOT / "configs" / "ci" / "vllm_qwen3_8_27b_dspark_2gpu_smoke.yaml"
        config = load_config(str(config_path))
        draft_config = AutoDraftModelConfig.from_file(config.model.draft_model_config)

        self.assertIsInstance(draft_config, DSparkConfig)
        self.assertEqual(config.training.dflash_block_size, 7)
        self.assertEqual(config.training.dspark_num_anchors, 512)
        self.assertEqual(config.dataset.min_loss_tokens, 32)
        self.assertEqual(draft_config.num_hidden_layers, 5)
        self.assertEqual(draft_config.markov_rank, 256)
        self.assertEqual(draft_config.target_num_hidden_layers, 64)
        self.assertEqual(
            list(config.inference.aux_hidden_states_layers),
            draft_config.target_layer_ids,
        )

    def test_optional_fc_norm_normalizes_each_target_layer_before_projection(self):
        cfg = _make_dspark_config(H=16, num_target_layers=3, fc_norm=True)
        model = DSparkDraftModel(cfg).to(dtype=torch.float32)
        self.assertIsNotNone(model.fc_norm)
        self.assertEqual(len(model.fc_norm), 3)

        hidden_states = [torch.randn(2, 5, 16) * scale for scale in (1.0, 15.0, 20.0)]
        projected_inputs = []
        handle = model.context_proj.register_forward_pre_hook(
            lambda _module, args: projected_inputs.append(args[0].detach())
        )
        try:
            model.extract_context_feature(hidden_states)
        finally:
            handle.remove()

        normalized_chunks = projected_inputs[0].chunk(3, dim=-1)
        for norm, raw, actual in zip(model.fc_norm, hidden_states, normalized_chunks, strict=True):
            torch.testing.assert_close(actual, norm(raw))

    def test_draft_model_heads(self):
        cfg = _make_dspark_config(H=64, markov_rank=16)
        m = DSparkDraftModel(cfg)
        self.assertIsInstance(m.markov_head, VanillaMarkov)
        self.assertIsInstance(m.confidence_head, AcceptRatePredictor)
        # confidence input = hidden + markov_rank when fused
        self.assertEqual(m.confidence_head.proj.in_features, 64 + 16)

    def test_no_heads(self):
        cfg = _make_dspark_config(
            markov_rank=0, enable_confidence_head=False, confidence_head_with_markov=False
        )
        m = DSparkDraftModel(cfg)
        self.assertIsNone(m.markov_head)
        self.assertIsNone(m.confidence_head)


class TestDSparkForward(unittest.TestCase):
    def test_returns_loss_terms_with_detached_components(self):
        m = _make_dspark_model()
        out = m(**_batch())
        self.assertEqual(len(out), 7)
        loss, acc, lpp, app, cpp, comps, loss_terms = out
        self.assertEqual(set(comps), {"ce_loss", "l1_loss", "confidence_loss"})
        for pair in comps.values():
            for v in pair:  # (numerator, denominator), pooled by the trainer
                self.assertTrue(torch.isfinite(v).all())
                self.assertFalse(v.requires_grad)  # detached for logging
        self.assertTrue(torch.isfinite(loss))
        self.assertEqual(lpp.shape[0], m.block_size)
        numerator, denominator = loss_terms
        torch.testing.assert_close(loss, numerator / denominator)

    def test_internal_loss_identity(self):
        # The reported loss is a purely local ratio now, so the combined loss must
        # equal the alpha-weighted sum of the logged components on every rank —
        # the components are a faithful decomposition of what's optimized.
        m = _make_dspark_model()
        loss, _, _, _, _, comps, _ = m(**_batch(seed=1))
        denominator = comps["ce_loss"][1]
        recomputed = (
            CE_A * comps["ce_loss"][0]
            + L1_A * comps["l1_loss"][0]
            + CF_A * comps["confidence_loss"][0]
        ) / denominator
        self.assertTrue(
            torch.allclose(loss, recomputed, atol=1e-4), f"{loss.item()} vs {recomputed.item()}"
        )

    def test_all_masked_is_zero(self):
        m = _make_dspark_model()
        loss, _, _, _, _, comps, _ = m(**_batch(all_masked=True))
        self.assertAlmostEqual(loss.item(), 0.0, places=5)
        for numerator, denominator in comps.values():
            self.assertAlmostEqual(numerator.item(), 0.0, places=5)
            # Zero denominator too, so the trainer's pooling must guard it.
            self.assertAlmostEqual(denominator.item(), 0.0, places=5)

    def test_next_token_convention_all_slots_supervised(self):
        # Fix 1: every within-block slot predicts a real token (B predictions),
        # unlike DFlash where slot 0 is the masked anchor. With a long fully
        # supervised sequence, every position should accumulate supervised tokens.
        m = _make_dspark_model(block_size=4, num_anchors=8)
        b = _batch(B=2, S=40)
        b["loss_mask"] = torch.ones(2, 40)
        _, _, _, _, count_per_position, _, _ = m(**b)
        self.assertEqual(count_per_position.shape[0], 4)
        self.assertTrue(
            (count_per_position > 0).all(), f"some slot unsupervised: {count_per_position.tolist()}"
        )

    def test_grad_flow_and_frozen_embedding(self):
        m = _make_dspark_model()
        loss, *_ = m(**_batch(seed=2))
        loss.backward()
        draft = m.draft_model
        self.assertIsNotNone(draft.markov_head.markov_w2.weight.grad)
        self.assertGreater(draft.markov_head.markov_w2.weight.grad.abs().sum().item(), 0)
        self.assertIsNotNone(draft.confidence_head.proj.weight.grad)
        self.assertGreater(draft.confidence_head.proj.weight.grad.abs().sum().item(), 0)
        self.assertIsNotNone(draft.context_proj.weight.grad)
        self.assertIsNone(draft.embed_tokens.weight.grad)  # frozen

    def test_ce_only_without_target(self):
        # ce-only (l1=0, no confidence) must run without last_hidden_states.
        m = _make_dspark_model(
            markov_rank=16, enable_confidence_head=False, confidence_head_with_markov=False
        )
        m.l1_loss_alpha = 0.0
        m.ce_loss_alpha = 1.0
        m.confidence_head_alpha = 0.0
        b = _batch()
        b["last_hidden_states"] = None
        loss, *_ = m(**b)
        self.assertTrue(torch.isfinite(loss))


class TestHeadMath(unittest.TestCase):
    def test_vanilla_markov_is_bigram_bias(self):
        torch.manual_seed(0)
        mk = VanillaMarkov(vocab_size=50, markov_rank=8)
        base = torch.randn(2, 3, 4, 50)
        prev = torch.randint(0, 50, (2, 3, 4))
        out = mk.apply_block_logits(base, token_ids=prev)
        expected = base + mk.markov_w2(mk.markov_w1(prev))
        self.assertTrue(torch.allclose(out, expected, atol=1e-6))

    def test_confidence_head_is_linear(self):
        torch.manual_seed(0)
        head = AcceptRatePredictor(20)
        feats = torch.randn(2, 3, 4, 20)
        out = head(feats)
        expected = head.proj(feats).squeeze(-1)
        self.assertTrue(torch.allclose(out, expected, atol=1e-6))
        self.assertEqual(out.shape, (2, 3, 4))


class TestDSparkTargetHiddenStates(unittest.TestCase):
    """The predicate the trainer reads to decide whether to load the verifier norm."""

    def test_confidence_head_alone_still_reads_the_target_distribution(self):
        m = _make_dspark_model(
            l1_loss_alpha=0.0, confidence_head_alpha=1.0, enable_confidence_head=True
        )
        self.assertTrue(m.uses_target_hidden_states)

    def test_l1_alone_reads_the_target_distribution(self):
        m = _make_dspark_model(
            l1_loss_alpha=0.9,
            confidence_head_alpha=0.0,
            enable_confidence_head=False,
            confidence_head_with_markov=False,
        )
        self.assertTrue(m.uses_target_hidden_states)

    def test_ce_only_does_not(self):
        m = _make_dspark_model(
            l1_loss_alpha=0.0,
            confidence_head_alpha=0.0,
            enable_confidence_head=False,
            confidence_head_with_markov=False,
        )
        self.assertFalse(m.uses_target_hidden_states)

    def test_a_weighted_confidence_head_the_draft_does_not_have_does_not(self):
        m = _make_dspark_model(
            l1_loss_alpha=0.0,
            confidence_head_alpha=1.0,
            enable_confidence_head=False,
            confidence_head_with_markov=False,
        )
        self.assertFalse(m.uses_target_hidden_states)

    def test_a_ce_only_model_runs_without_last_hidden_states(self):
        m = _make_dspark_model(
            l1_loss_alpha=0.0,
            confidence_head_alpha=0.0,
            enable_confidence_head=False,
            confidence_head_with_markov=False,
        )
        batch = _batch()
        batch["last_hidden_states"] = None
        with torch.no_grad():
            loss = m(**batch)[0]
        self.assertTrue(torch.isfinite(loss))


# yarn block in the shape published by Inferact/Kimi-K3-DSpark (rope_theta nested)
K3_ROPE_PARAMETERS = {
    "rope_type": "yarn",
    "factor": 32.0,
    "original_max_position_embeddings": 64,
    "rope_theta": 50000.0,
    "beta_fast": 32,
    "beta_slow": 1,
    "mscale": 1.0,
    "mscale_all_dim": 1.0,
}


def _make_k3_config(H=64, V=128, **over):
    base = dict(
        hidden_size=H,
        intermediate_size=128,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=4,
        q_lora_rank=32,
        kv_lora_rank=16,
        qk_nope_head_dim=16,
        qk_rope_head_dim=8,
        v_head_dim=16,
        vocab_size=V,
        rms_norm_eps=1e-6,
        max_position_embeddings=2048,
        rope_theta=10000.0,  # stale legacy value; rope_parameters must win
        num_target_layers=2,
        target_hidden_size=H,
        target_num_hidden_layers=12,
        mask_token_id=V - 1,
        markov_rank=16,
        markov_head_type="vanilla",
        enable_confidence_head=True,
        confidence_head_with_markov=True,
        rope_parameters=dict(K3_ROPE_PARAMETERS),
    )
    base.update(over)
    return K3DSparkConfig(**base)


def _make_k3_model(block_size=4, num_anchors=6, **cfg_kw):
    config = _make_k3_config(**cfg_kw)
    draft = K3DSparkModel(config).to(dtype=torch.float32)
    draft.freeze_embedding()
    return DSparkModel(
        draft_model=draft,
        block_size=block_size,
        num_anchors=num_anchors,
        loss_decay_gamma=4.0,
        ce_loss_alpha=CE_A,
        l1_loss_alpha=L1_A,
        confidence_head_alpha=CF_A,
    )


class TestK3DSparkConfig(unittest.TestCase):
    def test_rope_parameters_normalized_into_rope_scaling(self):
        cfg = _make_k3_config()
        self.assertEqual(cfg.model_type, "k3_dspark")
        self.assertEqual(cfg.rope_theta, 50000.0)  # nested value overrides legacy
        self.assertIsNotNone(cfg.rope_scaling)
        self.assertEqual(cfg.rope_scaling["rope_type"], "yarn")
        self.assertEqual(cfg.rope_scaling["factor"], 32.0)
        self.assertEqual(cfg.rope_scaling["mscale_all_dim"], 1.0)
        # transformers 5.x aliases the two names to a single attribute; the
        # nested rope_theta survives for serving-config round trips.
        self.assertIs(cfg.rope_scaling, cfg.rope_parameters)
        self.assertEqual(cfg.rope_parameters, K3_ROPE_PARAMETERS)

    def test_explicit_rope_scaling_wins(self):
        scaling = {"rope_type": "yarn", "factor": 2.0, "original_max_position_embeddings": 64}
        cfg = _make_k3_config(rope_scaling=scaling)
        self.assertEqual(cfg.rope_scaling["factor"], 2.0)

    def test_yarn_defaults_filled_for_partial_block(self):
        cfg = _make_k3_config(
            rope_parameters={
                "rope_type": "yarn",
                "factor": 32.0,
                "original_max_position_embeddings": 64,
            }
        )
        self.assertEqual(cfg.rope_scaling["beta_fast"], 32.0)
        self.assertEqual(cfg.rope_scaling["mscale_all_dim"], 0.0)


class TestK3DSparkModel(unittest.TestCase):
    def test_mla_layout_matches_published_checkpoint(self):
        m = K3DSparkModel(_make_k3_config())
        sd = m.state_dict()
        for name in (
            "layers.0.self_attn.q_a_proj.weight",
            "layers.0.self_attn.q_a_layernorm.weight",
            "layers.0.self_attn.q_b_proj.weight",
            "layers.0.self_attn.kv_a_proj_with_mqa.weight",
            "layers.0.self_attn.kv_a_layernorm.weight",
            "layers.0.self_attn.kv_b_proj.weight",
            "layers.0.self_attn.o_proj.weight",
            "layers.1.self_attn.o_proj.weight",
            "context_proj.weight",
            "markov_head.markov_w1.weight",
            "confidence_head.proj.weight",
        ):
            self.assertIn(name, sd)
        # no Qwen3-style per-head norms and no output gate in the MLA layout
        self.assertNotIn("layers.0.self_attn.q_norm.weight", sd)
        self.assertNotIn("layers.0.self_attn.k_norm.weight", sd)
        # H=4 heads, qk_head_dim=24, v_head_dim=16, hidden=64
        self.assertEqual(sd["layers.0.self_attn.q_b_proj.weight"].shape, (4 * 24, 32))
        self.assertEqual(sd["layers.0.self_attn.kv_a_proj_with_mqa.weight"].shape, (16 + 8, 64))
        self.assertEqual(sd["layers.0.self_attn.kv_b_proj.weight"].shape, (4 * (16 + 16), 16))
        self.assertEqual(sd["layers.0.self_attn.o_proj.weight"].shape, (64, 4 * 16))

    def test_yarn_rope_and_softmax_scale(self):
        m = K3DSparkModel(_make_k3_config())
        attn = m.layers[0].self_attn
        self.assertIsInstance(attn.rotary_emb, LlamaYarnRotaryEmbedding)
        self.assertEqual(attn.rotary_emb.dim, 8)  # rope side dims only
        mscale = yarn_get_mscale(32.0, 1.0)
        expected = (mscale * mscale) / math.sqrt(24)
        self.assertAlmostEqual(attn.softmax_scale, expected, places=8)

    def test_yarn_rope_uses_config_theta(self):
        # The shared yarn branch omitted base=, so it silently fell back to 10000
        # and every RoPE-enabled K3 run trained on the wrong rotary frequencies
        # even though K3DSparkConfig lifts the nested rope_theta.
        attn = K3DSparkModel(_make_k3_config()).layers[0].self_attn
        self.assertEqual(attn.rotary_emb.base, 50000.0)

        def reference(base):
            return LlamaYarnRotaryEmbedding(
                8,
                max_position_embeddings=2048,
                base=base,
                original_max_position_embeddings=64,
                scaling_factor=32.0,
                beta_fast=32,
                beta_slow=1,
                mscale=1.0,
                mscale_all_dim=1.0,
            )

        torch.testing.assert_close(attn.rotary_emb.inv_freq, reference(50000.0).inv_freq)
        self.assertFalse(torch.allclose(attn.rotary_emb.inv_freq, reference(10000.0).inv_freq))

    def test_output_gate_unsupported(self):
        with self.assertRaises(NotImplementedError):
            K3DSparkModel(_make_k3_config(mla_use_output_gate=True))

    def test_forward_backward_through_dspark_wrapper(self):
        m = _make_k3_model()
        loss, acc, lpp, app, cpp, comps, loss_terms = m(**_batch())
        self.assertTrue(torch.isfinite(loss))
        self.assertEqual(set(comps), {"ce_loss", "l1_loss", "confidence_loss"})
        numerator, denominator = loss_terms
        torch.testing.assert_close(loss, numerator / denominator)

        loss.backward()
        draft = m.draft_model
        attn = draft.layers[0].self_attn
        for p in (attn.q_a_proj.weight, attn.kv_b_proj.weight, attn.o_proj.weight):
            self.assertIsNotNone(p.grad)
            self.assertGreater(p.grad.abs().sum().item(), 0)
        self.assertIsNotNone(draft.markov_head.markov_w2.weight.grad)
        self.assertIsNone(draft.embed_tokens.weight.grad)  # frozen


class TestDispatch(unittest.TestCase):
    def test_json_resolves_to_dspark_config(self):
        cfg = AutoDraftModelConfig.from_dict(
            {
                "architectures": ["Qwen3DSparkModel"],
                "model_type": "qwen3_dspark",
                "hidden_size": 64,
                "vocab_size": 128,
                "num_hidden_layers": 1,
                "num_target_layers": 2,
                "markov_rank": 16,
                "enable_confidence_head": True,
            }
        )
        self.assertIsInstance(cfg, DSparkConfig)
        # Subclass of DFlashConfig -> any isinstance(DFlashConfig) dispatch must
        # test DSparkConfig first (trainer_actor / train_entry rely on this).
        self.assertIsInstance(cfg, DFlashConfig)

    def test_draft_class_name_resolves_to_dspark_config(self):
        cfg = AutoDraftModelConfig.from_dict(
            {
                "architectures": ["DSparkDraftModel"],
                "model_type": "qwen3_dspark",
                "hidden_size": 64,
                "intermediate_size": 128,
                "num_attention_heads": 4,
                "num_key_value_heads": 2,
                "vocab_size": 128,
                "num_hidden_layers": 1,
                "num_target_layers": 2,
                "markov_rank": 16,
                "enable_confidence_head": True,
            }
        )
        self.assertIsInstance(cfg, DSparkConfig)
        model = AutoEagle3DraftModel.from_config(cfg, torch_dtype=torch.float32)
        self.assertIsInstance(model, DSparkDraftModel)

    def test_k3_json_resolves_to_k3_config(self):
        cfg = AutoDraftModelConfig.from_dict(
            {
                "architectures": ["K3DSparkModel"],
                "model_type": "k3_dspark",
                "hidden_size": 64,
                "vocab_size": 128,
                "num_hidden_layers": 1,
                "num_attention_heads": 4,
                "q_lora_rank": 32,
                "kv_lora_rank": 16,
                "qk_nope_head_dim": 16,
                "qk_rope_head_dim": 8,
                "v_head_dim": 16,
                "num_target_layers": 2,
                "markov_rank": 16,
                "enable_confidence_head": True,
                "rope_parameters": dict(K3_ROPE_PARAMETERS),
            }
        )
        self.assertIsInstance(cfg, K3DSparkConfig)
        # trainer_actor's isinstance(DSparkConfig) dispatch must pick DSparkTrainer
        self.assertIsInstance(cfg, DSparkConfig)
        self.assertEqual(cfg.rope_scaling["rope_type"], "yarn")
        self.assertEqual(cfg.rope_theta, 50000.0)


if __name__ == "__main__":
    unittest.main()
