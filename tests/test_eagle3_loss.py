"""Tests for Eagle3 loss computation paths.

Verifies that:
1. compiled_forward_kl_loss matches a naive reference implementation.
2. compiled_lk_alpha_loss and compiled_lk_lambda_loss match reference implementations.
3. compute_target_p_padded produces correct shapes and valid probabilities
   for both pruning and non-pruning paths.
4. The lazy target path (non-pruning, target_p_padded=None) produces identical
   losses to the pre-computed target_p_padded path for all loss types.
"""

import unittest
from unittest.mock import patch

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
    compiled_lk_alpha_loss,
    compiled_lk_lambda_loss,
    compiled_lk_lambda_loss_from_hs,
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


def _reference_lk_alpha_loss(
    hs_flat, target_p_flat, norm_weight, lm_head_weight, norm_eps, coverage_flat=None
):
    """Pure-Python reference for LK^alpha loss.

    ``coverage_flat`` rescales the (possibly draft-vocab-renormalized)
    ``target_p_flat`` back to true target probabilities; defaults to 1 (no
    rescaling) for the common no-vocab-pruning case.
    """
    hs_f32 = hs_flat.float()
    variance = hs_f32.pow(2).mean(-1, keepdim=True)
    rstd = torch.rsqrt(variance + norm_eps)
    norm_hs = (hs_f32 * rstd).to(hs_flat.dtype) * norm_weight

    logits = F.linear(norm_hs, lm_head_weight)
    q = F.softmax(logits.float(), dim=-1)

    if coverage_flat is None:
        coverage_flat = torch.ones(target_p_flat.shape[0])
    target_p_true = target_p_flat * coverage_flat.unsqueeze(-1)

    alpha = torch.min(target_p_true, q).sum(-1)
    loss = -torch.log(alpha.clamp(min=1e-8)).mean()
    acc = (logits.argmax(-1) == target_p_flat.argmax(-1)).float().mean()
    return loss, acc, alpha.mean()


def _reference_lk_lambda_loss(
    hs_flat,
    target_p_flat,
    norm_weight,
    lm_head_weight,
    norm_eps,
    eta,
    coverage_flat=None,
    external_mean_alpha=None,
):
    """Pure-Python reference for LK^lambda loss.

    ``lambda`` is computed once from the mean acceptance rate across the whole
    call (i.e. one EAGLE-3 depth), matching the paper's schedule rather than a
    per-token value. See ``_reference_lk_alpha_loss`` for ``coverage_flat``.
    ``external_mean_alpha`` overrides the local mean, mirroring the SP-reduced
    mean-alpha override used under USP in ``compiled_lk_lambda_loss``.
    """
    hs_f32 = hs_flat.float()
    variance = hs_f32.pow(2).mean(-1, keepdim=True)
    rstd = torch.rsqrt(variance + norm_eps)
    norm_hs = (hs_f32 * rstd).to(hs_flat.dtype) * norm_weight

    logits = F.linear(norm_hs, lm_head_weight)
    q = F.softmax(logits.float(), dim=-1)
    log_q = F.log_softmax(logits.float(), dim=-1)

    if coverage_flat is None:
        coverage_flat = torch.ones(target_p_flat.shape[0])
    target_p_true = target_p_flat * coverage_flat.unsqueeze(-1)

    alpha = torch.min(target_p_true, q).sum(-1)
    mean_alpha = external_mean_alpha if external_mean_alpha is not None else alpha.mean()
    lam = torch.exp(-eta * mean_alpha.detach())

    kl = F.kl_div(log_q, target_p_flat, reduction="none").sum(-1)
    tv = 0.5 * (target_p_true - q).abs().sum(-1)

    loss = (lam * kl + (1.0 - lam) * tv).mean()
    acc = (logits.argmax(-1) == target_p_flat.argmax(-1)).float().mean()
    return loss, acc, alpha.mean()


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


def _make_model(
    config,
    length=3,
    attention_backend="sdpa",
    device="cpu",
    loss_type="forward_kl",
    lk_eta=3.0,
):
    draft_model = LlamaForCausalLMEagle3(config, attention_backend=attention_backend)
    draft_model = draft_model.to(device=device, dtype=torch.bfloat16)
    model = Eagle3Model(
        draft_model,
        length=length,
        attention_backend=attention_backend,
        loss_type=loss_type,
        lk_eta=lk_eta,
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

        loss_sum, correct, count = compiled_forward_kl_loss(
            hs, target_p, valid_idx, norm_weight, lm_head_weight, None, norm_eps
        )
        loss = loss_sum / count
        acc = correct / count
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

        loss_sum, correct, count = compiled_forward_kl_loss(
            hs, target_p, valid_idx, norm_weight, lm_head_weight, None, norm_eps
        )
        loss = loss_sum / count
        acc = correct / count
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

        loss_sum, correct, count = compiled_forward_kl_loss(
            hs, target_p, valid_idx, norm_weight, lm_head_weight, None, 1e-6
        )
        loss = loss_sum / count
        acc = correct / count
        self.assertTrue(torch.isfinite(loss))
        self.assertGreaterEqual(loss.item(), 0.0)
        self.assertGreaterEqual(acc.item(), 0.0)
        self.assertLessEqual(acc.item(), 1.0)


class TestCompiledLkAlphaLoss(unittest.TestCase):
    """compiled_lk_alpha_loss should match the reference implementation."""

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
        coverage = torch.ones(N)

        loss_sum, correct, count, alpha_sum = compiled_lk_alpha_loss(
            hs, target_p, valid_idx, norm_weight, lm_head_weight, None, norm_eps, coverage
        )
        loss, acc, alpha = loss_sum / count, correct / count, alpha_sum / count
        ref_loss, ref_acc, ref_alpha = _reference_lk_alpha_loss(
            hs, target_p, norm_weight, lm_head_weight, norm_eps, coverage
        )

        torch.testing.assert_close(loss, ref_loss, atol=1e-3, rtol=1e-3)
        torch.testing.assert_close(acc, ref_acc, atol=1e-3, rtol=1e-3)
        torch.testing.assert_close(alpha, ref_alpha, atol=1e-3, rtol=1e-3)

    def test_coverage_rescales_alpha_for_vocab_pruning(self):
        """With coverage < 1 (vocab pruning), alpha should use the true,
        un-renormalized target probabilities rather than the draft-vocab-
        renormalized ones — see arXiv:2602.23881 §4.4."""
        torch.manual_seed(3)
        N, H, V = 8, 32, 16
        hs = torch.randn(N, H, dtype=torch.float32)
        norm_weight = torch.randn(H, dtype=torch.float32)
        lm_head_weight = torch.randn(V, H, dtype=torch.float32)
        norm_eps = 1e-6
        valid_idx = torch.arange(N)
        target_p_tilde = F.softmax(torch.randn(N, V), dim=-1)
        coverage = torch.rand(N) * 0.5 + 0.3  # in [0.3, 0.8): meaningful vocab pruning

        _, _, _, alpha_sum_no_coverage = compiled_lk_alpha_loss(
            hs,
            target_p_tilde,
            valid_idx,
            norm_weight,
            lm_head_weight,
            None,
            norm_eps,
            torch.ones(N),
        )
        _, _, _, alpha_sum_with_coverage = compiled_lk_alpha_loss(
            hs, target_p_tilde, valid_idx, norm_weight, lm_head_weight, None, norm_eps, coverage
        )
        # Rescaling by coverage < 1 can only shrink (never inflate) alpha,
        # since it shrinks every component of the min(p, q) sum.
        self.assertLess(alpha_sum_with_coverage.item(), alpha_sum_no_coverage.item())
        ref_loss, _, ref_alpha = _reference_lk_alpha_loss(
            hs, target_p_tilde, norm_weight, lm_head_weight, norm_eps, coverage
        )
        loss_sum, _, count, alpha_sum = compiled_lk_alpha_loss(
            hs, target_p_tilde, valid_idx, norm_weight, lm_head_weight, None, norm_eps, coverage
        )
        torch.testing.assert_close(loss_sum / count, ref_loss, atol=1e-4, rtol=1e-4)
        torch.testing.assert_close(alpha_sum / count, ref_alpha, atol=1e-4, rtol=1e-4)

    def test_perfect_prediction_loss_zero(self):
        """When draft == target, alpha=1 so -log(alpha)=0."""
        torch.manual_seed(0)
        N, H, V = 16, 64, 32
        norm_weight = torch.ones(H, dtype=torch.float32)
        lm_head_weight = torch.randn(V, H, dtype=torch.float32)
        norm_eps = 1e-6
        valid_idx = torch.arange(N)

        hs = torch.randn(N, H, dtype=torch.float32)
        variance = hs.pow(2).mean(-1, keepdim=True)
        rstd = torch.rsqrt(variance + norm_eps)
        norm_hs = hs * rstd * norm_weight
        logits = F.linear(norm_hs, lm_head_weight)
        target_p = F.softmax(logits, dim=-1)
        coverage = torch.ones(N)

        loss_sum, correct, count, alpha_sum = compiled_lk_alpha_loss(
            hs, target_p, valid_idx, norm_weight, lm_head_weight, None, norm_eps, coverage
        )
        loss, acc, alpha = loss_sum / count, correct / count, alpha_sum / count
        self.assertAlmostEqual(loss.item(), 0.0, places=3)
        self.assertAlmostEqual(alpha.item(), 1.0, places=3)
        self.assertAlmostEqual(acc.item(), 1.0, places=2)

    def test_loss_finite_and_alpha_in_range(self):
        torch.manual_seed(0)
        N, H, V = 16, 64, 32
        hs = torch.randn(N, H, dtype=torch.bfloat16)
        norm_weight = torch.randn(H, dtype=torch.bfloat16)
        lm_head_weight = torch.randn(V, H, dtype=torch.bfloat16)
        target_p = F.softmax(torch.randn(N, V), dim=-1)
        valid_idx = torch.arange(N)
        coverage = torch.ones(N)

        loss_sum, correct, count, alpha_sum = compiled_lk_alpha_loss(
            hs, target_p, valid_idx, norm_weight, lm_head_weight, None, 1e-6, coverage
        )
        loss, alpha = loss_sum / count, alpha_sum / count
        self.assertTrue(torch.isfinite(loss))
        self.assertGreaterEqual(alpha.item(), 0.0)
        self.assertLessEqual(alpha.item(), 1.0)


class TestCompiledLkLambdaLoss(unittest.TestCase):
    """compiled_lk_lambda_loss should match the reference implementation."""

    def test_matches_reference(self):
        torch.manual_seed(42)
        N, H, V = 32, 128, 256
        hs = torch.randn(N, H, dtype=torch.bfloat16)
        norm_weight = torch.randn(H, dtype=torch.bfloat16)
        lm_head_weight = torch.randn(V, H, dtype=torch.bfloat16)
        norm_eps = 1e-6
        eta = 3.0
        valid_idx = torch.arange(N)

        raw_logits = F.linear(hs.float(), lm_head_weight.float())
        target_p = F.softmax(raw_logits + torch.randn_like(raw_logits) * 0.5, dim=-1)
        coverage = torch.ones(N)

        loss_sum, correct, count, alpha_sum = compiled_lk_lambda_loss(
            hs,
            target_p,
            valid_idx,
            norm_weight,
            lm_head_weight,
            None,
            norm_eps,
            coverage,
            None,
            eta,
        )
        loss, acc, alpha = loss_sum / count, correct / count, alpha_sum / count
        ref_loss, ref_acc, ref_alpha = _reference_lk_lambda_loss(
            hs, target_p, norm_weight, lm_head_weight, norm_eps, eta, coverage
        )

        torch.testing.assert_close(loss, ref_loss, atol=1e-3, rtol=1e-3)
        torch.testing.assert_close(acc, ref_acc, atol=1e-3, rtol=1e-3)
        torch.testing.assert_close(alpha, ref_alpha, atol=1e-3, rtol=1e-3)

    def test_eta_sensitivity(self):
        """Different eta values should produce different losses."""
        torch.manual_seed(42)
        N, H, V = 32, 128, 256
        hs = torch.randn(N, H, dtype=torch.bfloat16)
        norm_weight = torch.randn(H, dtype=torch.bfloat16)
        lm_head_weight = torch.randn(V, H, dtype=torch.bfloat16)
        target_p = F.softmax(torch.randn(N, V), dim=-1)
        valid_idx = torch.arange(N)
        coverage = torch.ones(N)

        loss_eta3, _, count3, _ = compiled_lk_lambda_loss(
            hs, target_p, valid_idx, norm_weight, lm_head_weight, None, 1e-6, coverage, None, 3.0
        )
        loss_eta10, _, count10, _ = compiled_lk_lambda_loss(
            hs, target_p, valid_idx, norm_weight, lm_head_weight, None, 1e-6, coverage, None, 10.0
        )
        self.assertFalse(torch.allclose(loss_eta3 / count3, loss_eta10 / count10))

    def test_coverage_rescales_alpha_and_tv_but_not_kl(self):
        """With coverage < 1 (vocab pruning): alpha/TV use the true, rescaled
        target probabilities, but the KL term keeps the draft-vocab-
        renormalized ones (raw p would make KL diverge) — arXiv:2602.23881 §4.4."""
        torch.manual_seed(4)
        N, H, V = 8, 32, 16
        hs = torch.randn(N, H, dtype=torch.float32)
        norm_weight = torch.randn(H, dtype=torch.float32)
        lm_head_weight = torch.randn(V, H, dtype=torch.float32)
        norm_eps = 1e-6
        eta = 3.0
        valid_idx = torch.arange(N)
        target_p_tilde = F.softmax(torch.randn(N, V), dim=-1)
        coverage = torch.rand(N) * 0.5 + 0.3

        loss_sum, _, count, alpha_sum = compiled_lk_lambda_loss(
            hs,
            target_p_tilde,
            valid_idx,
            norm_weight,
            lm_head_weight,
            None,
            norm_eps,
            coverage,
            None,
            eta,
        )
        ref_loss, _, ref_alpha = _reference_lk_lambda_loss(
            hs, target_p_tilde, norm_weight, lm_head_weight, norm_eps, eta, coverage
        )
        torch.testing.assert_close(loss_sum / count, ref_loss, atol=1e-4, rtol=1e-4)
        torch.testing.assert_close(alpha_sum / count, ref_alpha, atol=1e-4, rtol=1e-4)

        # Coverage=1 (no pruning) reference must differ, since rescaling by
        # coverage<1 changes both the KL-vs-TV blend (via mean alpha) and TV
        # itself — otherwise this test wouldn't be exercising the rescale path.
        ref_loss_no_coverage, _, ref_alpha_no_coverage = _reference_lk_lambda_loss(
            hs, target_p_tilde, norm_weight, lm_head_weight, norm_eps, eta
        )
        self.assertGreater((ref_alpha - ref_alpha_no_coverage).abs().item(), 1e-4)
        self.assertGreater((ref_loss - ref_loss_no_coverage).abs().item(), 1e-4)

    def test_lambda_uses_mean_alpha_not_per_token_alpha(self):
        """Guards against regressing to a per-token lambda schedule: with
        per-token alphas that vary meaningfully, the loss must match a
        shared-lambda (mean-alpha) schedule and disagree with a per-token one."""
        torch.manual_seed(5)
        N, H, V = 2, 32, 24
        hs = torch.randn(N, H, dtype=torch.float32)
        norm_weight = torch.randn(H, dtype=torch.float32)
        lm_head_weight = torch.randn(V, H, dtype=torch.float32)
        norm_eps = 1e-6
        eta = 5.0
        valid_idx = torch.arange(N)
        target_p = F.softmax(torch.randn(N, V) * 2.0, dim=-1)  # spread out alphas
        coverage = torch.ones(N)

        loss_sum, _, _count, _alpha_sum = compiled_lk_lambda_loss(
            hs,
            target_p,
            valid_idx,
            norm_weight,
            lm_head_weight,
            None,
            norm_eps,
            coverage,
            None,
            eta,
        )

        variance = hs.pow(2).mean(-1, keepdim=True)
        rstd = torch.rsqrt(variance + norm_eps)
        norm_hs = hs * rstd * norm_weight
        logits = F.linear(norm_hs, lm_head_weight)
        q = F.softmax(logits, dim=-1)
        log_q = F.log_softmax(logits, dim=-1)
        alpha = torch.min(target_p, q).sum(-1)
        kl = F.kl_div(log_q, target_p, reduction="none").sum(-1)
        tv = 0.5 * (target_p - q).abs().sum(-1)

        lam_per_token = torch.exp(-eta * alpha.detach())
        per_token_loss_sum = (lam_per_token * kl + (1.0 - lam_per_token) * tv).sum()

        lam_shared = torch.exp(-eta * alpha.mean().detach())
        shared_loss_sum = (lam_shared * kl + (1.0 - lam_shared) * tv).sum()

        torch.testing.assert_close(loss_sum, shared_loss_sum, atol=1e-4, rtol=1e-4)
        self.assertGreater((loss_sum - per_token_loss_sum).abs().item(), 1e-4)

    def test_external_mean_alpha_overrides_local_mean(self):
        """Under USP, the caller SP-reduces alpha across the sequence-parallel
        group before forming lambda and passes the result in as
        ``external_mean_alpha`` — this must override this call's local mean
        alpha entirely (not just blend with it), since the local shard alone
        does not span the depth's full batch/sequence (arXiv:2602.23881)."""
        torch.manual_seed(6)
        N, H, V = 4, 32, 24
        hs = torch.randn(N, H, dtype=torch.float32)
        norm_weight = torch.randn(H, dtype=torch.float32)
        lm_head_weight = torch.randn(V, H, dtype=torch.float32)
        norm_eps = 1e-6
        eta = 4.0
        valid_idx = torch.arange(N)
        target_p = F.softmax(torch.randn(N, V) * 2.0, dim=-1)
        coverage = torch.ones(N)

        loss_local, _, count, alpha_sum = compiled_lk_lambda_loss(
            hs,
            target_p,
            valid_idx,
            norm_weight,
            lm_head_weight,
            None,
            norm_eps,
            coverage,
            None,
            eta,
        )
        local_mean_alpha = alpha_sum / count

        # A deliberately different external mean (as if peer SP shards were
        # much easier/harder) must change the loss relative to the local-only
        # computation, and must match a reference that uses it directly.
        external_mean_alpha = (local_mean_alpha + 0.3).clamp(max=0.99).detach()
        loss_external, _, _, _ = compiled_lk_lambda_loss(
            hs,
            target_p,
            valid_idx,
            norm_weight,
            lm_head_weight,
            None,
            norm_eps,
            coverage,
            external_mean_alpha,
            eta,
        )
        ref_loss, _, _ = _reference_lk_lambda_loss(
            hs,
            target_p,
            norm_weight,
            lm_head_weight,
            norm_eps,
            eta,
            coverage,
            external_mean_alpha=external_mean_alpha,
        )

        self.assertGreater((loss_local - loss_external).abs().item(), 1e-4)
        torch.testing.assert_close(loss_external / count, ref_loss, atol=1e-4, rtol=1e-4)

    def test_perfect_prediction_loss_zero(self):
        """When draft == target, KL=0 and TV=0 so loss=0."""
        torch.manual_seed(0)
        N, H, V = 16, 64, 32
        norm_weight = torch.ones(H, dtype=torch.float32)
        lm_head_weight = torch.randn(V, H, dtype=torch.float32)
        norm_eps = 1e-6
        valid_idx = torch.arange(N)

        hs = torch.randn(N, H, dtype=torch.float32)
        variance = hs.pow(2).mean(-1, keepdim=True)
        rstd = torch.rsqrt(variance + norm_eps)
        norm_hs = hs * rstd * norm_weight
        logits = F.linear(norm_hs, lm_head_weight)
        target_p = F.softmax(logits, dim=-1)
        coverage = torch.ones(N)

        loss_sum, _correct, count, alpha_sum = compiled_lk_lambda_loss(
            hs,
            target_p,
            valid_idx,
            norm_weight,
            lm_head_weight,
            None,
            norm_eps,
            coverage,
            None,
            3.0,
        )
        loss, alpha = loss_sum / count, alpha_sum / count
        self.assertAlmostEqual(loss.item(), 0.0, places=3)
        self.assertAlmostEqual(alpha.item(), 1.0, places=3)

    def test_loss_finite(self):
        torch.manual_seed(0)
        N, H, V = 16, 64, 32
        hs = torch.randn(N, H, dtype=torch.bfloat16)
        norm_weight = torch.randn(H, dtype=torch.bfloat16)
        lm_head_weight = torch.randn(V, H, dtype=torch.bfloat16)
        target_p = F.softmax(torch.randn(N, V), dim=-1)
        valid_idx = torch.arange(N)
        coverage = torch.ones(N)

        loss_sum, _correct, count, _alpha_sum = compiled_lk_lambda_loss(
            hs, target_p, valid_idx, norm_weight, lm_head_weight, None, 1e-6, coverage, None, 3.0
        )
        loss = loss_sum / count
        self.assertTrue(torch.isfinite(loss))
        self.assertGreaterEqual(loss.item(), 0.0)


class TestUspLambdaAlphaReduction(unittest.TestCase):
    """Under USP, lk_lambda's schedule must use alpha aggregated across the SP
    group (the depth's full sequence/batch), not just this rank's local
    shard — the SP all-reduce must happen before lambda is formed inside
    ``_calculate_loss``, not only in the later, separate metrics all-reduce in
    ``forward``. See arXiv:2602.23881 (the schedule is defined per depth over
    the whole batch/sequence, not per shard)."""

    def _make_usp_model_and_inputs(self, B=1, T=8, H=32, V=24, length=1, all_valid=True):
        torch.manual_seed(11)
        config = _make_config(H=H, V=V)
        model = _make_model(config, length=length, loss_type="lk_lambda", lk_eta=4.0)
        # No real distributed process group is initialized in this test; use a
        # sentinel so `self._usp_sp_group is not None` triggers the SP-reduce
        # path, and mock `dist.all_reduce` below instead of using a real group.
        model._usp_sp_group = "fake-sp-group"
        # Post-backbone hidden states (matching norm_weight/lm_head_weight's
        # input dim H) — _calculate_loss operates downstream of the backbone,
        # unlike the raw (H*3) concat fed into the full model forward.
        hidden_states = torch.randn(B, T, H, dtype=torch.bfloat16)
        target_p = F.softmax(torch.randn(B, T, V), dim=-1)
        target = PrecomputedTarget(F.pad(target_p, (0, 0, 0, length)))
        mask = torch.ones(B, T) if all_valid else torch.zeros(B, T)
        norm_weight, lm_head_weight, norm_eps = model.draft_model.get_lm_head_params()
        return model, hidden_states, target, mask, norm_weight, lm_head_weight, norm_eps

    def test_lambda_uses_sp_reduced_alpha_not_local_shard_alone(self):
        model, hidden_states, target, mask, norm_weight, lm_head_weight, norm_eps = (
            self._make_usp_model_and_inputs()
        )
        seq_length = mask.shape[1]

        # Simulate a peer SP shard with a much higher local alpha (e.g. an
        # "easy" chunk of the sequence): the mocked all-reduce adds its
        # (alpha_sum, count) on top of this rank's real local contribution.
        peer_alpha_sum = torch.tensor(50.0)
        peer_count = torch.tensor(10.0)

        def fake_all_reduce(tensor, op, group):
            self.assertEqual(group, "fake-sp-group")
            tensor[0] += peer_alpha_sum
            tensor[1] += peer_count

        with patch(
            "torchspec.models.eagle3.dist.all_reduce", side_effect=fake_all_reduce
        ) as mock_reduce:
            loss_sum, _correct_sum, _count, _alpha_sum = model._calculate_loss(
                hidden_states=hidden_states,
                target=target,
                mask=mask,
                idx=0,
                seq_length=seq_length,
                norm_weight=norm_weight,
                lm_head_weight=lm_head_weight,
                lm_head_up=None,
                norm_eps=norm_eps,
            )
        mock_reduce.assert_called_once()

        hs_flat = hidden_states.reshape(-1, hidden_states.shape[-1])
        tp_flat = target.target_p_padded[:, :seq_length, :].reshape(
            -1, target.target_p_padded.shape[-1]
        )
        valid_idx = mask.flatten().nonzero().squeeze(-1)
        coverage_flat = tp_flat.new_ones(tp_flat.shape[0])

        # Without the fix (local mean alpha only), the loss would differ,
        # since the peer's alpha materially shifts the schedule.
        local_only_loss_sum, _, local_only_count, local_only_alpha_sum = compiled_lk_lambda_loss(
            hs_flat,
            tp_flat,
            valid_idx,
            norm_weight,
            lm_head_weight,
            None,
            norm_eps,
            coverage_flat,
            None,
            model.lk_eta,
        )
        self.assertGreater((loss_sum - local_only_loss_sum).abs().item(), 1e-4)

        # The actual result must match forming lambda from the true SP-wide
        # (local + peer) mean alpha.
        expected_mean_alpha = (
            (local_only_alpha_sum + peer_alpha_sum) / (local_only_count + peer_count)
        ).detach()
        expected_loss_sum, _, _, _ = compiled_lk_lambda_loss(
            hs_flat,
            tp_flat,
            valid_idx,
            norm_weight,
            lm_head_weight,
            None,
            norm_eps,
            coverage_flat,
            expected_mean_alpha,
            model.lk_eta,
        )
        torch.testing.assert_close(loss_sum, expected_loss_sum, atol=1e-4, rtol=1e-4)

    def test_zero_valid_tokens_still_participates_in_sp_reduce(self):
        """A rank with no local valid tokens at this depth must still call the
        SP all-reduce (with a zero contribution) so peer ranks with valid
        tokens don't hang waiting for it."""
        model, hidden_states, target, mask, norm_weight, lm_head_weight, norm_eps = (
            self._make_usp_model_and_inputs(all_valid=False)
        )

        with patch("torchspec.models.eagle3.dist.all_reduce") as mock_reduce:
            loss_sum, correct_sum, count, alpha_sum = model._calculate_loss(
                hidden_states=hidden_states,
                target=target,
                mask=mask,
                idx=0,
                seq_length=mask.shape[1],
                norm_weight=norm_weight,
                lm_head_weight=lm_head_weight,
                lm_head_up=None,
                norm_eps=norm_eps,
            )

        mock_reduce.assert_called_once()
        (zero_stats_arg,), kwargs = mock_reduce.call_args
        self.assertEqual(kwargs["group"], "fake-sp-group")
        torch.testing.assert_close(zero_stats_arg, torch.zeros(2))
        self.assertEqual(loss_sum.item(), 0.0)

    def test_lazy_target_path_also_uses_sp_reduced_alpha(self):
        """Same fix, LazyTarget (non-vocab-pruned) path: compiled_lk_lambda_loss_from_hs
        must also use the SP-reduced mean alpha, not this rank's local shard alone."""
        torch.manual_seed(12)
        B, T, H, V, length = 1, 8, 32, 24, 1
        config = _make_config(H=H, V=V)
        model = _make_model(config, length=length, loss_type="lk_lambda", lk_eta=4.0)
        model._usp_sp_group = "fake-sp-group"
        norm_weight, lm_head_weight, norm_eps = model.draft_model.get_lm_head_params()

        hidden_states = torch.randn(B, T, H, dtype=torch.bfloat16)
        target_hidden_states = torch.randn(B, T, H, dtype=torch.bfloat16)
        target = compute_lazy_target_padded(target_hidden_states, lm_head_weight, length)
        mask = torch.ones(B, T)
        seq_length = T

        peer_alpha_sum = torch.tensor(50.0)
        peer_count = torch.tensor(10.0)

        def fake_all_reduce(tensor, op, group):
            tensor[0] += peer_alpha_sum
            tensor[1] += peer_count

        with patch("torchspec.models.eagle3.dist.all_reduce", side_effect=fake_all_reduce):
            loss_sum, _, _, _ = model._calculate_loss(
                hidden_states=hidden_states,
                target=target,
                mask=mask,
                idx=0,
                seq_length=seq_length,
                norm_weight=norm_weight,
                lm_head_weight=lm_head_weight,
                lm_head_up=None,
                norm_eps=norm_eps,
            )

        hs_flat = hidden_states.reshape(-1, hidden_states.shape[-1])
        ths_flat = target.hidden_states_padded[:, :seq_length, :].reshape(-1, H)
        valid_idx = mask.flatten().nonzero().squeeze(-1)
        local_only_loss_sum, _, local_only_count, local_only_alpha_sum = (
            compiled_lk_lambda_loss_from_hs(
                hs_flat,
                ths_flat,
                valid_idx,
                norm_weight,
                lm_head_weight,
                None,
                target.lm_head_weight,
                norm_eps,
                None,
                model.lk_eta,
            )
        )
        self.assertGreater((loss_sum - local_only_loss_sum).abs().item(), 1e-4)

        expected_mean_alpha = (
            (local_only_alpha_sum + peer_alpha_sum) / (local_only_count + peer_count)
        ).detach()
        expected_loss_sum, _, _, _ = compiled_lk_lambda_loss_from_hs(
            hs_flat,
            ths_flat,
            valid_idx,
            norm_weight,
            lm_head_weight,
            None,
            target.lm_head_weight,
            norm_eps,
            expected_mean_alpha,
            model.lk_eta,
        )
        torch.testing.assert_close(loss_sum, expected_loss_sum, atol=1e-4, rtol=1e-4)

    def test_non_usp_lambda_unaffected(self):
        """Without a USP SP group, _calculate_loss must not call all_reduce at
        all for lk_lambda (i.e. this fix must be a no-op outside USP)."""
        model, hidden_states, target, mask, norm_weight, lm_head_weight, norm_eps = (
            self._make_usp_model_and_inputs()
        )
        model._usp_sp_group = None

        with patch("torchspec.models.eagle3.dist.all_reduce") as mock_reduce:
            model._calculate_loss(
                hidden_states=hidden_states,
                target=target,
                mask=mask,
                idx=0,
                seq_length=mask.shape[1],
                norm_weight=norm_weight,
                lm_head_weight=lm_head_weight,
                lm_head_up=None,
                norm_eps=norm_eps,
            )
        mock_reduce.assert_not_called()


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

    def test_coverage_matches_true_full_vocab_probability_mass(self):
        """coverage_padded should equal the draft-vocab probability mass under
        the true full-vocab softmax, so that target_p_padded * coverage_padded
        recovers true (un-renormalized) target probabilities — see
        ``PrecomputedTarget.coverage_padded`` and arXiv:2602.23881 §4.4."""
        torch.manual_seed(1)
        B, T, D = 2, 12, 64
        V_target, V_draft = 96, 24
        length = 2
        hs = torch.randn(B, T, D, dtype=torch.float32)
        weight = torch.randn(V_target, D, dtype=torch.float32)
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

        self.assertIsNotNone(result.coverage_padded)
        self.assertEqual(result.coverage_padded.shape, (B, T + length))
        coverage = result.coverage_padded[:, :T]
        self.assertTrue((coverage >= 0).all())
        self.assertTrue((coverage <= 1 + 1e-4).all())

        full_p = F.softmax(F.linear(hs, weight).float(), dim=-1)
        expected_coverage = full_p[..., t2d].sum(-1)
        torch.testing.assert_close(coverage, expected_coverage, atol=1e-4, rtol=1e-4)

        true_p = result.target_p_padded[:, :T, :] * coverage.unsqueeze(-1)
        expected_true_p = full_p[..., t2d]
        torch.testing.assert_close(true_p, expected_true_p, atol=1e-4, rtol=1e-4)

    def test_loss_mask_respected_in_position_mask(self):
        """Masked positions should have position_mask == 0 and coverage == 0."""
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
        self.assertTrue((result.coverage_padded[:, : T // 2] == 0).all())


class TestLazyVsPrecomputedTarget(unittest.TestCase):
    """The lazy path (target_p_padded=None) must produce identical losses."""

    def _run_both_paths(self, device="cpu", loss_type="forward_kl", lk_eta=3.0):
        torch.manual_seed(42)
        H, V, B, T, length = 128, 256, 1, 32, 3

        config = _make_config(H=H, V=V)
        model = _make_model(
            config, length=length, device=device, loss_type=loss_type, lk_eta=lk_eta
        )
        batch = _make_batch(B, T, H, V, device=device)

        draft_model = model.draft_model
        _, lm_head_weight, _ = draft_model.get_lm_head_params()

        with torch.no_grad():
            target_logits = F.linear(batch["target_hidden_states"], lm_head_weight.detach())
            target_p = F.softmax(target_logits.float(), dim=-1)
        target_p_padded = F.pad(target_p, (0, 0, 0, length), value=0.0)

        precomputed = PrecomputedTarget(target_p_padded)
        with torch.no_grad():
            plosses_pre, _, acces_pre, _, alphas_pre = model(
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
            plosses_lazy, _, acces_lazy, _, alphas_lazy = model(
                input_ids=batch["input_ids"],
                attention_mask=batch["attention_mask"],
                target=lazy,
                loss_mask=batch["loss_mask"],
                hidden_states=batch["hidden_states"],
            )

        return plosses_pre, acces_pre, alphas_pre, plosses_lazy, acces_lazy, alphas_lazy

    def _assert_paths_match(self, device, loss_type="forward_kl", lk_eta=3.0, atol=1e-4, rtol=1e-4):
        results = self._run_both_paths(device, loss_type=loss_type, lk_eta=lk_eta)
        plosses_pre, acces_pre, alphas_pre, plosses_lazy, acces_lazy, alphas_lazy = results
        for i, (pre, lazy) in enumerate(zip(plosses_pre, plosses_lazy)):
            torch.testing.assert_close(
                pre,
                lazy,
                atol=atol,
                rtol=rtol,
                msg=f"Loss mismatch at position {i} (loss_type={loss_type})",
            )
        for i, (pre, lazy) in enumerate(zip(acces_pre, acces_lazy)):
            torch.testing.assert_close(
                pre,
                lazy,
                atol=atol,
                rtol=rtol,
                msg=f"Accuracy mismatch at position {i} (loss_type={loss_type})",
            )
        for i, (pre, lazy) in enumerate(zip(alphas_pre, alphas_lazy)):
            torch.testing.assert_close(
                pre,
                lazy,
                atol=atol,
                rtol=rtol,
                msg=f"Alpha mismatch at position {i} (loss_type={loss_type})",
            )

    def test_forward_kl_losses_match_cpu(self):
        self._assert_paths_match("cpu", loss_type="forward_kl")

    def test_lk_alpha_losses_match_cpu(self):
        self._assert_paths_match("cpu", loss_type="lk_alpha")

    def test_lk_lambda_losses_match_cpu(self):
        self._assert_paths_match("cpu", loss_type="lk_lambda")

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA not available")
    def test_forward_kl_losses_match_cuda(self):
        self._assert_paths_match("cuda", loss_type="forward_kl", atol=1e-3, rtol=1e-3)

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA not available")
    def test_lk_alpha_losses_match_cuda(self):
        self._assert_paths_match("cuda", loss_type="lk_alpha", atol=1e-3, rtol=1e-3)

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA not available")
    def test_lk_lambda_losses_match_cuda(self):
        self._assert_paths_match("cuda", loss_type="lk_lambda", atol=1e-3, rtol=1e-3)


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

        loss_sum, correct, count = compiled_forward_kl_loss(
            hs_flat,
            tp_flat,
            valid_idx,
            norm_weight,
            lm_head_weight,
            None,
            norm_eps,
        )
        loss = loss_sum / count
        acc = correct / count

        hs_valid = hs_flat[valid_idx]
        tp_valid = tp_flat[valid_idx]
        all_idx = torch.arange(hs_valid.shape[0])
        loss_sum_ref, correct_ref, count_ref = compiled_forward_kl_loss(
            hs_valid,
            tp_valid,
            all_idx,
            norm_weight,
            lm_head_weight,
            None,
            norm_eps,
        )
        loss_ref = loss_sum_ref / count_ref
        acc_ref = correct_ref / count_ref

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

        loss_sum, correct, count = compiled_forward_kl_loss_from_hs(
            hs_flat,
            ths_flat,
            valid_idx,
            norm_weight,
            lm_head_weight,
            None,
            target_lm_head_weight,
            norm_eps,
        )
        loss = loss_sum / count
        acc = correct / count

        hs_valid = hs_flat[valid_idx]
        ths_valid = ths_flat[valid_idx]
        all_idx = torch.arange(hs_valid.shape[0])
        loss_sum_ref, correct_ref, count_ref = compiled_forward_kl_loss_from_hs(
            hs_valid,
            ths_valid,
            all_idx,
            norm_weight,
            lm_head_weight,
            None,
            target_lm_head_weight,
            norm_eps,
        )
        loss_ref = loss_sum_ref / count_ref
        acc_ref = correct_ref / count_ref

        torch.testing.assert_close(loss, loss_ref, atol=1e-5, rtol=1e-5)
        torch.testing.assert_close(acc, acc_ref, atol=1e-5, rtol=1e-5)

    def _check_lk_alpha(self, valid_idx):
        torch.manual_seed(7)
        hs_flat = torch.randn(self.BT, self.H, dtype=torch.bfloat16)
        norm_weight = torch.randn(self.H, dtype=torch.bfloat16)
        lm_head_weight = torch.randn(self.V, self.H, dtype=torch.bfloat16)
        tp_flat = F.softmax(torch.randn(self.BT, self.V), dim=-1)
        coverage_flat = torch.rand(self.BT) * 0.5 + 0.5
        norm_eps = 1e-6

        loss_sum, correct, count, alpha_sum = compiled_lk_alpha_loss(
            hs_flat,
            tp_flat,
            valid_idx,
            norm_weight,
            lm_head_weight,
            None,
            norm_eps,
            coverage_flat,
        )
        loss, acc, alpha = loss_sum / count, correct / count, alpha_sum / count

        hs_valid = hs_flat[valid_idx]
        tp_valid = tp_flat[valid_idx]
        coverage_valid = coverage_flat[valid_idx]
        all_idx = torch.arange(hs_valid.shape[0])
        loss_sum_ref, correct_ref, count_ref, alpha_sum_ref = compiled_lk_alpha_loss(
            hs_valid,
            tp_valid,
            all_idx,
            norm_weight,
            lm_head_weight,
            None,
            norm_eps,
            coverage_valid,
        )
        loss_ref = loss_sum_ref / count_ref
        acc_ref = correct_ref / count_ref
        alpha_ref = alpha_sum_ref / count_ref

        torch.testing.assert_close(loss, loss_ref, atol=1e-5, rtol=1e-5)
        torch.testing.assert_close(acc, acc_ref, atol=1e-5, rtol=1e-5)
        torch.testing.assert_close(alpha, alpha_ref, atol=1e-5, rtol=1e-5)

    def _check_lk_lambda(self, valid_idx):
        torch.manual_seed(7)
        hs_flat = torch.randn(self.BT, self.H, dtype=torch.bfloat16)
        norm_weight = torch.randn(self.H, dtype=torch.bfloat16)
        lm_head_weight = torch.randn(self.V, self.H, dtype=torch.bfloat16)
        tp_flat = F.softmax(torch.randn(self.BT, self.V), dim=-1)
        coverage_flat = torch.rand(self.BT) * 0.5 + 0.5
        norm_eps = 1e-6
        eta = 3.0

        loss_sum, correct, count, alpha_sum = compiled_lk_lambda_loss(
            hs_flat,
            tp_flat,
            valid_idx,
            norm_weight,
            lm_head_weight,
            None,
            norm_eps,
            coverage_flat,
            None,
            eta,
        )
        loss, acc, alpha = loss_sum / count, correct / count, alpha_sum / count

        hs_valid = hs_flat[valid_idx]
        tp_valid = tp_flat[valid_idx]
        coverage_valid = coverage_flat[valid_idx]
        all_idx = torch.arange(hs_valid.shape[0])
        loss_sum_ref, correct_ref, count_ref, alpha_sum_ref = compiled_lk_lambda_loss(
            hs_valid,
            tp_valid,
            all_idx,
            norm_weight,
            lm_head_weight,
            None,
            norm_eps,
            coverage_valid,
            None,
            eta,
        )
        loss_ref = loss_sum_ref / count_ref
        acc_ref = correct_ref / count_ref
        alpha_ref = alpha_sum_ref / count_ref

        torch.testing.assert_close(loss, loss_ref, atol=1e-5, rtol=1e-5)
        torch.testing.assert_close(acc, acc_ref, atol=1e-5, rtol=1e-5)
        torch.testing.assert_close(alpha, alpha_ref, atol=1e-5, rtol=1e-5)


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

    def _make_lk_alpha(vidx=_vidx):
        def test(self):
            self._check_lk_alpha(vidx)

        return test

    def _make_lk_lambda(vidx=_vidx):
        def test(self):
            self._check_lk_lambda(vidx)

        return test

    setattr(TestValidIdxSubsetting, f"test_forward_kl_{_name}", _make_kl())
    setattr(TestValidIdxSubsetting, f"test_forward_kl_from_hs_{_name}", _make_kl_from_hs())
    setattr(TestValidIdxSubsetting, f"test_lk_alpha_{_name}", _make_lk_alpha())
    setattr(TestValidIdxSubsetting, f"test_lk_lambda_{_name}", _make_lk_lambda())


if __name__ == "__main__":
    unittest.main()
