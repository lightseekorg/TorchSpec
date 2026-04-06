"""Tests for DeepSeek MLA Eagle3 draft model.

Verifies that:
1. Forward pass produces correct output shapes (with and without cache).
2. Backward pass computes gradients for all trainable parameters.
3. q_lora_rank=None path works correctly.
4. Config dispatch correctly routes to Eagle3DeepseekV2ForCausalLM.
5. Softmax scale is computed correctly with YaRN mscale.
6. Eagle3Model TTT loop works end-to-end with MLA draft model.
"""

import math
import unittest

import torch
import torch.nn.functional as F
from transformers.models.deepseek_v3.configuration_deepseek_v3 import DeepseekV3Config

from torchspec.models.draft.auto import AutoDraftModelConfig, AutoEagle3DraftModel
from torchspec.models.draft.deepseek_eagle import (
    DeepSeekMLAAttention,
    Eagle3DeepseekV2ForCausalLM,
)
from torchspec.models.draft.llama3_eagle import yarn_get_mscale
from torchspec.models.eagle3 import (
    Eagle3Model,
    PrecomputedTarget,
    compute_lazy_target_padded,
)


def _make_config(
    H=64,
    V=256,
    draft_V=None,
    num_heads=4,
    qk_nope=16,
    qk_rope=8,
    v_head=16,
    kv_lora=32,
    q_lora=48,
    rope_scaling=None,
):
    config = DeepseekV3Config(
        hidden_size=H,
        num_attention_heads=num_heads,
        num_key_value_heads=num_heads,
        intermediate_size=H * 4,
        max_position_embeddings=1024,
        vocab_size=V,
        hidden_act="silu",
        rms_norm_eps=1e-6,
        rope_scaling=rope_scaling,
        pretraining_tp=1,
        pad_token_id=0,
        q_lora_rank=q_lora,
        kv_lora_rank=kv_lora,
        qk_nope_head_dim=qk_nope,
        qk_rope_head_dim=qk_rope,
        v_head_dim=v_head,
        num_hidden_layers=1,
        # MoE fields (unused by draft model, use small defaults)
        n_routed_experts=1,
        n_shared_experts=0,
        first_k_dense_replace=0,
        num_experts_per_tok=1,
    )
    config.draft_vocab_size = draft_V or V
    config.target_hidden_size = H
    return config


def _make_model(config, length=3, device="cpu", attention_backend="sdpa"):
    draft_model = Eagle3DeepseekV2ForCausalLM(config, attention_backend=attention_backend)
    draft_model = draft_model.to(device=device, dtype=torch.bfloat16)
    model = Eagle3Model(
        draft_model,
        length=length,
        attention_backend=attention_backend,
    )
    model.eval()
    return model


def _make_batch(B, T, H, V, device="cpu"):
    input_ids = torch.randint(0, V, (B, T), device=device)
    attention_mask = torch.ones(B, T, dtype=torch.long, device=device)
    loss_mask = torch.zeros(B, T, device=device)
    loss_mask[:, T // 4 :] = 1.0
    hidden_states = torch.randn(B, T, H * 3, device=device, dtype=torch.bfloat16)
    target_hidden_states = torch.randn(B, T, H, device=device, dtype=torch.bfloat16)
    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "loss_mask": loss_mask,
        "hidden_states": hidden_states,
        "target_hidden_states": target_hidden_states,
    }


class TestForwardShapeNoCache(unittest.TestCase):
    """backbone() without cache returns correct shapes."""

    def test_output_shape(self):
        torch.manual_seed(42)
        H, V, B, T = 64, 256, 2, 16
        config = _make_config(H=H, V=V)
        draft = Eagle3DeepseekV2ForCausalLM(config).to(torch.bfloat16)

        input_emb = torch.randn(B, T, H, dtype=torch.bfloat16)
        hidden = torch.randn(B, T, H, dtype=torch.bfloat16)
        pos_ids = torch.arange(T).unsqueeze(0).expand(B, -1)

        out, ck, cv = draft.backbone(
            input_embeds=input_emb,
            hidden_states=hidden,
            attention_mask=None,
            position_ids=pos_ids,
            cache_keys=None,
            cache_values=None,
            use_cache=False,
        )
        self.assertEqual(out.shape, (B, T, H))
        self.assertIsNone(ck)
        self.assertIsNone(cv)


class TestForwardShapeWithCache(unittest.TestCase):
    """backbone() with cache returns correct 5D cache shapes."""

    def test_cache_shapes_across_steps(self):
        torch.manual_seed(42)
        H, V, B, T = 64, 256, 2, 8
        num_heads, qk_nope, qk_rope, v_head = 4, 16, 8, 16
        config = _make_config(
            H=H,
            V=V,
            num_heads=num_heads,
            qk_nope=qk_nope,
            qk_rope=qk_rope,
            v_head=v_head,
        )
        draft = Eagle3DeepseekV2ForCausalLM(config).to(torch.bfloat16)

        from torchspec.models.draft.base import prepare_decoder_attention_mask

        input_emb = torch.randn(B, T, H, dtype=torch.bfloat16)
        hidden = torch.randn(B, T, H, dtype=torch.bfloat16)
        pos_ids = torch.arange(T).unsqueeze(0).expand(B, -1)
        attn_mask_base = torch.ones(B, T, dtype=torch.long)

        cache_keys = None
        cache_values = None

        for step in range(3):
            attn_mask = prepare_decoder_attention_mask(attn_mask_base, (B, T), hidden, 0)
            _, cache_keys, cache_values = draft.backbone(
                input_embeds=input_emb,
                hidden_states=hidden,
                attention_mask=attn_mask,
                position_ids=pos_ids,
                cache_keys=cache_keys,
                cache_values=cache_values,
                use_cache=True,
            )

            expected_k_shape = (B, num_heads, step + 1, T, qk_nope + qk_rope)
            expected_v_shape = (B, num_heads, step + 1, T, v_head)
            self.assertEqual(cache_keys.shape, expected_k_shape, f"step {step}")
            self.assertEqual(cache_values.shape, expected_v_shape, f"step {step}")


class TestBackward(unittest.TestCase):
    """All trainable parameters receive gradients."""

    def test_gradients(self):
        torch.manual_seed(42)
        H, V, B, T = 64, 256, 2, 8
        config = _make_config(H=H, V=V)
        draft = Eagle3DeepseekV2ForCausalLM(config).to(torch.bfloat16)
        draft.freeze_embedding()
        draft.train()

        input_emb = torch.randn(B, T, H, dtype=torch.bfloat16)
        hidden = torch.randn(B, T, H, dtype=torch.bfloat16)
        pos_ids = torch.arange(T).unsqueeze(0).expand(B, -1)

        from torchspec.models.draft.base import prepare_decoder_attention_mask

        attn_mask = prepare_decoder_attention_mask(
            torch.ones(B, T, dtype=torch.long), (B, T), hidden, 0
        )

        out, _, _ = draft.backbone(
            input_embeds=input_emb,
            hidden_states=hidden,
            attention_mask=attn_mask,
            position_ids=pos_ids,
            use_cache=True,
        )
        loss = out.sum()
        loss.backward()

        # Embedding should be frozen
        self.assertIsNone(draft.embed_tokens.weight.grad)

        # All params in backbone's forward path (midlayer.*) should have gradients.
        # embed_tokens, fc, norm, lm_head are used outside backbone().
        for name, param in draft.named_parameters():
            if "midlayer" not in name:
                continue
            if param.requires_grad:
                self.assertIsNotNone(param.grad, f"Parameter {name} has no gradient")


class TestQLoraNone(unittest.TestCase):
    """q_lora_rank=None uses direct Q projection."""

    def test_forward_works(self):
        torch.manual_seed(42)
        H, V, B, T = 64, 256, 2, 8
        config = _make_config(H=H, V=V, q_lora=None)
        draft = Eagle3DeepseekV2ForCausalLM(config).to(torch.bfloat16)

        self.assertFalse(hasattr(draft.midlayer.self_attn, "q_a_proj"))
        self.assertTrue(hasattr(draft.midlayer.self_attn, "q_proj"))

        input_emb = torch.randn(B, T, H, dtype=torch.bfloat16)
        hidden = torch.randn(B, T, H, dtype=torch.bfloat16)
        pos_ids = torch.arange(T).unsqueeze(0).expand(B, -1)

        out, _, _ = draft.backbone(
            input_embeds=input_emb,
            hidden_states=hidden,
            attention_mask=None,
            position_ids=pos_ids,
            use_cache=False,
        )
        self.assertEqual(out.shape, (B, T, H))


class TestConfigDispatch(unittest.TestCase):
    """AutoDraftModelConfig + AutoEagle3DraftModel correctly dispatch to DeepSeek."""

    def test_dispatch(self):
        config_dict = {
            "architectures": ["Eagle3DeepseekV2ForCausalLM"],
            "model_type": "deepseek_v3",
            "hidden_size": 64,
            "num_attention_heads": 4,
            "num_key_value_heads": 4,
            "num_hidden_layers": 1,
            "intermediate_size": 256,
            "hidden_act": "silu",
            "rms_norm_eps": 1e-6,
            "q_lora_rank": 48,
            "kv_lora_rank": 32,
            "qk_nope_head_dim": 16,
            "qk_rope_head_dim": 8,
            "v_head_dim": 16,
            "vocab_size": 256,
            "n_routed_experts": 1,
            "n_shared_experts": 0,
            "first_k_dense_replace": 0,
            "num_experts_per_tok": 1,
        }
        config = AutoDraftModelConfig.from_dict(config_dict)
        self.assertIsInstance(config, DeepseekV3Config)

        model = AutoEagle3DraftModel.from_config(config, torch_dtype=torch.bfloat16)
        self.assertIsInstance(model, Eagle3DeepseekV2ForCausalLM)


class TestSoftmaxScale(unittest.TestCase):
    """Softmax scale computation with YaRN mscale."""

    def test_yarn_mscale(self):
        config = _make_config(
            qk_nope=128,
            qk_rope=64,
            rope_scaling={
                "type": "yarn",
                "factor": 40,
                "original_max_position_embeddings": 4096,
                "beta_fast": 32,
                "beta_slow": 1,
                "mscale": 1,
                "mscale_all_dim": 0.1,
            },
        )
        attn = DeepSeekMLAAttention(config)

        mscale = yarn_get_mscale(40, 0.1)
        expected = (mscale * mscale) / math.sqrt(128 + 64)
        self.assertAlmostEqual(attn.softmax_scale, expected, places=6)

    def test_no_rope_scaling(self):
        config = _make_config(qk_nope=16, qk_rope=8, rope_scaling=None)
        attn = DeepSeekMLAAttention(config)
        expected = 1.0 / math.sqrt(16 + 8)
        self.assertAlmostEqual(attn.softmax_scale, expected, places=6)


class TestEagle3ModelTTT(unittest.TestCase):
    """Eagle3Model TTT loop with MLA draft model: Lazy vs Precomputed target."""

    def _run_both_paths(self, device="cpu"):
        torch.manual_seed(42)
        H, V, B, T, length = 64, 256, 1, 32, 3

        config = _make_config(H=H, V=V)
        model = _make_model(config, length=length, device=device)
        batch = _make_batch(B, T, H, V, device=device)

        draft_model = model.draft_model
        _, lm_head_weight, _ = draft_model.get_lm_head_params()

        with torch.no_grad():
            target_logits = F.linear(batch["target_hidden_states"], lm_head_weight.detach())
            target_p = F.softmax(target_logits.float(), dim=-1)
        target_p_padded = F.pad(target_p, (0, 0, 0, length), value=0.0)

        precomputed = PrecomputedTarget(target_p_padded)
        with torch.no_grad():
            plosses_pre, _, acces_pre = model(
                input_ids=batch["input_ids"],
                attention_mask=batch["attention_mask"],
                target=precomputed,
                loss_mask=batch["loss_mask"],
                hidden_states=batch["hidden_states"],
            )

        lazy = compute_lazy_target_padded(
            batch["target_hidden_states"],
            lm_head_weight,
            length,
        )
        with torch.no_grad():
            plosses_lazy, _, acces_lazy = model(
                input_ids=batch["input_ids"],
                attention_mask=batch["attention_mask"],
                target=lazy,
                loss_mask=batch["loss_mask"],
                hidden_states=batch["hidden_states"],
            )

        return plosses_pre, acces_pre, plosses_lazy, acces_lazy

    def test_losses_finite_and_non_negative_cpu(self):
        plosses_pre, _, _, _ = self._run_both_paths("cpu")
        for i, loss in enumerate(plosses_pre):
            self.assertTrue(torch.isfinite(loss), f"Loss {i} is not finite: {loss}")
            self.assertGreaterEqual(loss.item(), 0.0, f"Loss {i} is negative: {loss}")

    def test_lazy_matches_precomputed_cpu(self):
        plosses_pre, acces_pre, plosses_lazy, acces_lazy = self._run_both_paths("cpu")
        for i, (pre, lazy) in enumerate(zip(plosses_pre, plosses_lazy)):
            torch.testing.assert_close(
                pre,
                lazy,
                atol=1e-4,
                rtol=1e-4,
                msg=f"Loss mismatch at position {i}",
            )
        for i, (pre, lazy) in enumerate(zip(acces_pre, acces_lazy)):
            torch.testing.assert_close(
                pre,
                lazy,
                atol=1e-4,
                rtol=1e-4,
                msg=f"Accuracy mismatch at position {i}",
            )

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA not available")
    def test_losses_match_cuda(self):
        plosses_pre, acces_pre, plosses_lazy, acces_lazy = self._run_both_paths("cuda")
        for i, (pre, lazy) in enumerate(zip(plosses_pre, plosses_lazy)):
            torch.testing.assert_close(
                pre,
                lazy,
                atol=1e-3,
                rtol=1e-3,
                msg=f"Loss mismatch at position {i}",
            )


class TestFlexAttentionTTT(unittest.TestCase):
    """Eagle3Model TTT loop with MLA + flex_attention backend."""

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA not available")
    def test_flex_losses_finite_cuda(self):
        torch.manual_seed(42)
        H, V, B, T, length = 64, 256, 1, 32, 3
        config = _make_config(H=H, V=V)
        model = _make_model(
            config, length=length, device="cuda", attention_backend="flex_attention"
        )
        batch = _make_batch(B, T, H, V, device="cuda")

        draft_model = model.draft_model
        _, lm_head_weight, _ = draft_model.get_lm_head_params()

        with torch.no_grad():
            target_logits = F.linear(batch["target_hidden_states"], lm_head_weight.detach())
            target_p = F.softmax(target_logits.float(), dim=-1)
        target_p_padded = F.pad(target_p, (0, 0, 0, length), value=0.0)

        precomputed = PrecomputedTarget(target_p_padded)
        with torch.no_grad():
            plosses, _, acces = model(
                input_ids=batch["input_ids"],
                attention_mask=batch["attention_mask"],
                target=precomputed,
                loss_mask=batch["loss_mask"],
                hidden_states=batch["hidden_states"],
            )

        for i, loss in enumerate(plosses):
            self.assertTrue(torch.isfinite(loss), f"Loss {i} is not finite: {loss}")
            self.assertGreaterEqual(loss.item(), 0.0, f"Loss {i} is negative: {loss}")

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA not available")
    def test_flex_matches_sdpa_cuda(self):
        """Flex attention and SDPA should produce similar losses."""
        torch.manual_seed(42)
        H, V, B, T, length = 64, 256, 1, 32, 3
        config = _make_config(H=H, V=V)
        batch = _make_batch(B, T, H, V, device="cuda")

        results = {}
        for backend in ("sdpa", "flex_attention"):
            torch.manual_seed(42)
            model = _make_model(config, length=length, device="cuda", attention_backend=backend)
            draft_model = model.draft_model
            _, lm_head_weight, _ = draft_model.get_lm_head_params()

            with torch.no_grad():
                target_logits = F.linear(batch["target_hidden_states"], lm_head_weight.detach())
                target_p = F.softmax(target_logits.float(), dim=-1)
            target_p_padded = F.pad(target_p, (0, 0, 0, length), value=0.0)

            precomputed = PrecomputedTarget(target_p_padded)
            with torch.no_grad():
                plosses, _, _ = model(
                    input_ids=batch["input_ids"],
                    attention_mask=batch["attention_mask"],
                    target=precomputed,
                    loss_mask=batch["loss_mask"],
                    hidden_states=batch["hidden_states"],
                )
            results[backend] = plosses

        for i, (sdpa_loss, flex_loss) in enumerate(zip(results["sdpa"], results["flex_attention"])):
            torch.testing.assert_close(
                sdpa_loss,
                flex_loss,
                atol=1e-2,
                rtol=1e-2,
                msg=f"SDPA vs Flex loss mismatch at step {i}",
            )


if __name__ == "__main__":
    unittest.main()
