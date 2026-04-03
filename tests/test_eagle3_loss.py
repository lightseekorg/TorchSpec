"""Tests for Eagle3 loss computation paths.

Verifies that:
1. compiled_forward_kl_loss matches a naive reference implementation.
2. compute_target_p_padded produces correct shapes and valid probabilities
   for both pruning and non-pruning paths.
3. The lazy target path (non-pruning, target_p_padded=None) produces identical
   losses to the pre-computed target_p_padded path.
"""

import unittest

import torch
import torch.nn.functional as F
from transformers.models.llama.configuration_llama import LlamaConfig

from torchspec.models.draft.llama3_eagle import LlamaForCausalLMEagle3
from torchspec.models.eagle3 import (
    Eagle3Model,
    PrecomputedTarget,
    compute_lazy_target_padded,
    compute_target_p_padded,
)
from torchspec.models.ops.loss import (
    compiled_forward_kl_loss,
    compiled_forward_kl_loss_from_hs,
)


def _reference_forward_kl_loss(hs_flat, target_p_flat, norm_weight, lm_head_weight, norm_eps):
    """Pure-Python reference (no torch.compile) for validation."""
    hs_f32 = hs_flat.float()
    variance = hs_f32.pow(2).mean(-1, keepdim=True)
    rstd = torch.rsqrt(variance + norm_eps)
    norm_hs = (hs_f32 * rstd).to(hs_flat.dtype) * norm_weight

    logits = F.linear(norm_hs, lm_head_weight)
    log_p = F.log_softmax(logits.float(), dim=-1)
    loss = -(target_p_flat * log_p).sum(-1).mean()
    acc = (logits.argmax(-1) == target_p_flat.argmax(-1)).float().mean()
    return loss, acc


def _make_config(H=128, V=256, draft_V=None, num_heads=4, num_kv_heads=2):
    config = LlamaConfig(
        hidden_size=H,
        num_attention_heads=num_heads,
        num_key_value_heads=num_kv_heads,
        intermediate_size=H * 4,
        max_position_embeddings=1024,
        vocab_size=V,
        hidden_act="silu",
        rms_norm_eps=1e-6,
        rope_scaling=None,
        pretraining_tp=1,
        pad_token_id=0,
    )
    config.draft_vocab_size = draft_V or V
    return config


def _make_model(config, length=3, attention_backend="sdpa", device="cpu"):
    draft_model = LlamaForCausalLMEagle3(config, attention_backend=attention_backend)
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


class TestCompiledForwardKLLoss(unittest.TestCase):
    """compiled_forward_kl_loss should match the reference implementation."""

    def test_matches_reference(self):
        torch.manual_seed(42)
        N, H, V = 32, 128, 256
        hs = torch.randn(N, H, dtype=torch.bfloat16)
        norm_weight = torch.randn(H, dtype=torch.bfloat16)
        lm_head_weight = torch.randn(V, H, dtype=torch.bfloat16)
        norm_eps = 1e-6
        valid_idx = torch.arange(N)

        raw_logits = F.linear(hs.float(), lm_head_weight.float())
        target_p = F.softmax(raw_logits + torch.randn_like(raw_logits) * 0.5, dim=-1)

        loss, acc = compiled_forward_kl_loss(
            hs, target_p, valid_idx, norm_weight, lm_head_weight, norm_eps
        )
        ref_loss, ref_acc = _reference_forward_kl_loss(
            hs, target_p, norm_weight, lm_head_weight, norm_eps
        )

        torch.testing.assert_close(loss, ref_loss, atol=1e-3, rtol=1e-3)
        torch.testing.assert_close(acc, ref_acc, atol=1e-3, rtol=1e-3)

    def test_perfect_prediction_equals_entropy(self):
        """When draft == target, cross-entropy loss equals target entropy."""
        torch.manual_seed(0)
        N, H, V = 16, 64, 32
        norm_weight = torch.ones(H, dtype=torch.float32)
        lm_head_weight = torch.randn(V, H, dtype=torch.float32)
        norm_eps = 1e-6
        valid_idx = torch.arange(N)

        hs = torch.randn(N, H, dtype=torch.float32)
        # The loss function computes H(target, draft) = H(target) + KL(target||draft).
        # When target_p is derived from the same logits, KL ≈ 0 so loss ≈ H(target).
        variance = hs.pow(2).mean(-1, keepdim=True)
        rstd = torch.rsqrt(variance + norm_eps)
        norm_hs = hs * rstd * norm_weight
        logits = F.linear(norm_hs, lm_head_weight)
        target_p = F.softmax(logits, dim=-1)
        expected_entropy = -(target_p * target_p.log()).sum(-1).mean()

        loss, acc = compiled_forward_kl_loss(
            hs, target_p, valid_idx, norm_weight, lm_head_weight, norm_eps
        )
        torch.testing.assert_close(loss, expected_entropy, atol=1e-3, rtol=1e-3)
        self.assertAlmostEqual(acc.item(), 1.0, places=2)

    def test_loss_non_negative_and_finite(self):
        torch.manual_seed(0)
        N, H, V = 16, 64, 32
        hs = torch.randn(N, H, dtype=torch.bfloat16)
        norm_weight = torch.randn(H, dtype=torch.bfloat16)
        lm_head_weight = torch.randn(V, H, dtype=torch.bfloat16)
        target_p = F.softmax(torch.randn(N, V), dim=-1)
        valid_idx = torch.arange(N)

        loss, acc = compiled_forward_kl_loss(
            hs, target_p, valid_idx, norm_weight, lm_head_weight, 1e-6
        )
        self.assertTrue(torch.isfinite(loss))
        self.assertGreaterEqual(loss.item(), 0.0)
        self.assertGreaterEqual(acc.item(), 0.0)
        self.assertLessEqual(acc.item(), 1.0)


class TestComputeTargetPPadded(unittest.TestCase):
    """compute_target_p_padded: shape, dtype, and probability correctness."""

    def test_pruning_shapes_and_position_mask(self):
        torch.manual_seed(0)
        B, T, D = 2, 16, 64
        V_target, V_draft = 128, 32
        length = 3
        hs = torch.randn(B, T, D, dtype=torch.bfloat16)
        weight = torch.randn(V_target, D, dtype=torch.bfloat16)
        loss_mask = torch.ones(B, T)

        t2d = torch.zeros(V_target, dtype=torch.bool)
        t2d[:V_draft] = True

        result = compute_target_p_padded(
            hs,
            weight,
            t2d=t2d,
            loss_mask=loss_mask,
            length=length,
        )

        self.assertIsInstance(result, PrecomputedTarget)
        self.assertEqual(result.target_p_padded.shape, (B, T + length, V_draft))
        self.assertIsNotNone(result.position_mask)
        self.assertEqual(result.position_mask.shape, (B, T))
        sums = result.target_p_padded[:, :T, :].sum(dim=-1)
        torch.testing.assert_close(sums, torch.ones_like(sums), atol=1e-4, rtol=1e-4)

    def test_loss_mask_respected_in_position_mask(self):
        """Masked positions should have position_mask == 0."""
        torch.manual_seed(0)
        B, T, D = 1, 32, 64
        V_target, V_draft = 128, 32
        hs = torch.randn(B, T, D, dtype=torch.bfloat16)
        weight = torch.randn(V_target, D, dtype=torch.bfloat16)
        loss_mask = torch.zeros(B, T)
        loss_mask[:, T // 2 :] = 1.0

        t2d = torch.zeros(V_target, dtype=torch.bool)
        t2d[:V_draft] = True

        result = compute_target_p_padded(
            hs,
            weight,
            t2d=t2d,
            loss_mask=loss_mask,
            length=3,
        )

        self.assertTrue((result.position_mask[:, : T // 2] == 0).all())


class TestLazyVsPrecomputedTarget(unittest.TestCase):
    """The lazy path (target_p_padded=None) must produce identical losses."""

    def _run_both_paths(self, device="cpu"):
        torch.manual_seed(42)
        H, V, B, T, length = 128, 256, 1, 32, 3

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

    def test_losses_match_cpu(self):
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
        for i, (pre, lazy) in enumerate(zip(acces_pre, acces_lazy)):
            torch.testing.assert_close(
                pre,
                lazy,
                atol=1e-3,
                rtol=1e-3,
                msg=f"Accuracy mismatch at position {i}",
            )


class TestRotaryConfigWiring(unittest.TestCase):
    """Model config should fully wire RoPE settings into rotary embeddings."""

    def test_yarn_uses_rope_theta_as_base(self):
        config = LlamaConfig(
            hidden_size=128,
            num_attention_heads=4,
            num_key_value_heads=4,
            intermediate_size=512,
            max_position_embeddings=262144,
            vocab_size=256,
            hidden_act="silu",
            rms_norm_eps=1e-6,
            rope_theta=50000.0,
            rope_scaling={
                "type": "yarn",
                "factor": 64.0,
                "original_max_position_embeddings": 4096,
                "beta_fast": 32.0,
                "beta_slow": 1.0,
                "mscale": 1.0,
                "mscale_all_dim": 1.0,
            },
            pretraining_tp=1,
            pad_token_id=0,
        )
        config.draft_vocab_size = 256

        model = LlamaForCausalLMEagle3(config, attention_backend="sdpa")
        rotary = model.midlayer.self_attn.rotary_emb

        self.assertEqual(rotary.base, 50000.0)
        self.assertEqual(rotary.original_max_position_embeddings, 4096)
        self.assertEqual(rotary.scaling_factor, 64.0)


def _make_mask_patterns(BT):
    """Return (name, valid_idx) pairs covering diverse masking patterns."""
    patterns = []

    # contiguous first half
    m = torch.zeros(BT, dtype=torch.bool)
    m[: BT // 2] = True
    patterns.append(("first_half", m.nonzero().squeeze(-1)))

    # contiguous second half
    m = torch.zeros(BT, dtype=torch.bool)
    m[BT // 2 :] = True
    patterns.append(("second_half", m.nonzero().squeeze(-1)))

    # every other position (strided)
    m = torch.zeros(BT, dtype=torch.bool)
    m[::2] = True
    patterns.append(("strided", m.nonzero().squeeze(-1)))

    # random sparse (~25%)
    g = torch.Generator().manual_seed(99)
    m = torch.rand(BT, generator=g) < 0.25
    patterns.append(("random_sparse", m.nonzero().squeeze(-1)))

    # single valid position
    patterns.append(("single", torch.tensor([BT // 3])))

    # all valid
    patterns.append(("all", torch.arange(BT)))

    return patterns


class TestValidIdxSubsetting(unittest.TestCase):
    """valid_idx filtering must produce the same loss as manual pre-filtering."""

    BT, H, V = 64, 128, 256

    def _check_forward_kl(self, valid_idx):
        torch.manual_seed(7)
        hs_flat = torch.randn(self.BT, self.H, dtype=torch.bfloat16)
        norm_weight = torch.randn(self.H, dtype=torch.bfloat16)
        lm_head_weight = torch.randn(self.V, self.H, dtype=torch.bfloat16)
        tp_flat = F.softmax(torch.randn(self.BT, self.V), dim=-1)
        norm_eps = 1e-6

        loss, acc = compiled_forward_kl_loss(
            hs_flat,
            tp_flat,
            valid_idx,
            norm_weight,
            lm_head_weight,
            norm_eps,
        )

        hs_valid = hs_flat[valid_idx]
        tp_valid = tp_flat[valid_idx]
        all_idx = torch.arange(hs_valid.shape[0])
        loss_ref, acc_ref = compiled_forward_kl_loss(
            hs_valid,
            tp_valid,
            all_idx,
            norm_weight,
            lm_head_weight,
            norm_eps,
        )

        torch.testing.assert_close(loss, loss_ref, atol=1e-5, rtol=1e-5)
        torch.testing.assert_close(acc, acc_ref, atol=1e-5, rtol=1e-5)

    def _check_forward_kl_from_hs(self, valid_idx):
        torch.manual_seed(7)
        hs_flat = torch.randn(self.BT, self.H, dtype=torch.bfloat16)
        ths_flat = torch.randn(self.BT, self.H, dtype=torch.bfloat16)
        norm_weight = torch.randn(self.H, dtype=torch.bfloat16)
        lm_head_weight = torch.randn(self.V, self.H, dtype=torch.bfloat16)
        target_lm_head_weight = torch.randn(self.V, self.H, dtype=torch.bfloat16)
        norm_eps = 1e-6

        loss, acc = compiled_forward_kl_loss_from_hs(
            hs_flat,
            ths_flat,
            valid_idx,
            norm_weight,
            lm_head_weight,
            target_lm_head_weight,
            norm_eps,
        )

        hs_valid = hs_flat[valid_idx]
        ths_valid = ths_flat[valid_idx]
        all_idx = torch.arange(hs_valid.shape[0])
        loss_ref, acc_ref = compiled_forward_kl_loss_from_hs(
            hs_valid,
            ths_valid,
            all_idx,
            norm_weight,
            lm_head_weight,
            target_lm_head_weight,
            norm_eps,
        )

        torch.testing.assert_close(loss, loss_ref, atol=1e-5, rtol=1e-5)
        torch.testing.assert_close(acc, acc_ref, atol=1e-5, rtol=1e-5)


# Dynamically generate one test method per mask pattern per loss function.
for _name, _vidx in _make_mask_patterns(TestValidIdxSubsetting.BT):

    def _make_kl(vidx=_vidx):
        def test(self):
            self._check_forward_kl(vidx)

        return test

    def _make_kl_from_hs(vidx=_vidx):
        def test(self):
            self._check_forward_kl_from_hs(vidx)

        return test

    setattr(TestValidIdxSubsetting, f"test_forward_kl_{_name}", _make_kl())
    setattr(TestValidIdxSubsetting, f"test_forward_kl_from_hs_{_name}", _make_kl_from_hs())


if __name__ == "__main__":
    unittest.main()
