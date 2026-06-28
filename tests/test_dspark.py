"""Unit tests for the DSpark training objective.

DSpark is the existing DFlash drafter run with ``loss_objective="dspark"`` plus the
Markov / confidence heads. These tests pin the DSpark wiring (so a refactor can't
silently break the objective) and check numerical faithfulness to DeepSeek's
DeepSpec reference loss.

The DeepSpec loss math is vendored verbatim below (not imported) so the test suite
has no dependency on the ``deepspec`` package -- see the permalink on
``_deepspec_reference_components``.

CPU-only and tiny by design (block_mask falls back to dense SDPA off-CUDA).
"""

import torch
import torch.nn.functional as F

from torchspec.models.dflash import DFlashModel
from torchspec.models.draft.dflash import (
    DFlashAcceptRatePredictor,
    DFlashConfig,
    DFlashDraftModel,
    DFlashMarkovHead,
)

CE_A, L1_A, CF_A, GAMMA = 0.1, 0.9, 1.0, 4.0


# --------------------------------------------------------------------------------------
# Vendored DeepSpec reference loss math (not imported).
# Copied/adapted verbatim from DeepSeek DeepSpec @ dd854392fadf053cfddcbc4dc0e6e32de46d1bd0:
#   https://github.com/deepseek-ai/DeepSpec/blob/dd854392fadf053cfddcbc4dc0e6e32de46d1bd0/deepspec/modeling/dspark/loss.py
# (single-process / world_size==1 form: loss_weight_mask = eval_mask * exp(-k/gamma);
#  per-term numerator/denominator with den + 1e-6; CE + TV-L1 + confidence BCE against the
#  accept-rate target 1 - 0.5*L1; softmax in fp32).
# --------------------------------------------------------------------------------------
def _deepspec_reference_components(
    draft_logits, target_logits, target_ids, eval_mask, decay, confidence_pred
):
    vocab = draft_logits.shape[-1]
    loss_weight_mask = eval_mask.to(torch.float32) * decay  # [B, N, bs]
    w = loss_weight_mask.reshape(-1)
    den = w.sum() + 1e-6

    ce_pt = F.cross_entropy(
        draft_logits.reshape(-1, vocab), target_ids.reshape(-1), reduction="none"
    )
    ce = (ce_pt * w).sum() / den

    draft_probs = torch.softmax(draft_logits.reshape(-1, vocab).float(), dim=-1)
    target_probs = torch.softmax(target_logits.reshape(-1, vocab).float(), dim=-1)
    l1_pt = (draft_probs - target_probs).abs().sum(dim=-1)
    l1 = (l1_pt * w).sum() / den

    accept = (1.0 - 0.5 * l1_pt).clamp(0.0, 1.0)
    bce = F.binary_cross_entropy_with_logits(
        confidence_pred.reshape(-1).float(), accept, reduction="none"
    )
    conf = (bce * w).sum() / den
    return ce, l1, conf


# --------------------------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------------------------
def _make_config(
    H=64, V=128, markov_rank=16, enable_confidence_head=True, confidence_head_with_markov=True
):
    return DFlashConfig(
        hidden_size=H,
        intermediate_size=128,
        num_hidden_layers=1,
        num_attention_heads=4,
        num_key_value_heads=2,
        vocab_size=V,
        rms_norm_eps=1e-6,
        max_position_embeddings=512,
        rope_theta=10000.0,
        num_target_layers=2,
        target_hidden_size=H,
        target_num_hidden_layers=12,
        mask_token_id=V - 1,
        markov_rank=markov_rank,
        markov_head_type="vanilla",
        enable_confidence_head=enable_confidence_head,
        confidence_head_with_markov=confidence_head_with_markov,
    )


def _make_model(block_size=4, num_anchors=8, **cfg_kw):
    config = _make_config(**cfg_kw)
    draft = DFlashDraftModel(config).to(dtype=torch.float32)
    draft.freeze_embedding()
    return DFlashModel(
        draft_model=draft,
        block_size=block_size,
        num_anchors=num_anchors,
        loss_objective="dspark",
        loss_decay_gamma=GAMMA,
        dspark_ce_alpha=CE_A,
        dspark_l1_alpha=L1_A,
        dspark_confidence_alpha=CF_A,
    )


def _make_batch(B=2, S=32, H=64, V=128, num_target_layers=2, all_masked=False, seed=0):
    g = torch.Generator().manual_seed(seed)
    input_ids = torch.randint(0, V, (B, S), generator=g)
    hidden_states_list = [torch.randn(B, S, H, generator=g) for _ in range(num_target_layers)]
    loss_mask = torch.zeros(B, S) if all_masked else torch.ones(B, S)
    if not all_masked:
        loss_mask[:, :2] = 0  # prompt prefix
    lm_head_weight = torch.randn(V, H, generator=g)
    last_hidden = torch.randn(B, S, H, generator=g)
    return dict(
        input_ids=input_ids,
        hidden_states_list=hidden_states_list,
        loss_mask=loss_mask,
        lm_head_weight=lm_head_weight,
        target_last_hidden_states=last_hidden,
    )


# --------------------------------------------------------------------------------------
# Self-consistency (no DeepSpec dependency)
# --------------------------------------------------------------------------------------
def test_forward_returns_five_tuple_and_components():
    m = _make_model()
    out = m(**_make_batch())
    assert len(out) == 5
    loss, acc, lpp, app, cpp = out
    assert torch.isfinite(loss)
    assert set(m._dspark_components) == {"dspark_ce", "dspark_l1", "dspark_conf"}
    for v in m._dspark_components.values():
        assert torch.isfinite(v).all() and not v.requires_grad
    assert lpp.shape[0] == m.block_size


def test_loss_is_alpha_weighted_sum_of_components():
    # world_size==1 -> backward loss equals the alpha-weighted sum of the logged
    # (local-mean) components, so the components faithfully decompose the objective.
    m = _make_model()
    loss, *_ = m(**_make_batch(seed=1))
    c = m._dspark_components
    recomputed = CE_A * c["dspark_ce"] + L1_A * c["dspark_l1"] + CF_A * c["dspark_conf"]
    torch.testing.assert_close(loss, recomputed, rtol=1e-5, atol=1e-4)


def test_all_masked_is_zero():
    m = _make_model()
    loss, *_ = m(**_make_batch(all_masked=True))
    assert abs(loss.item()) < 1e-4


def test_next_token_supervises_all_block_slots():
    # DSpark uses the next-token convention: slot 0 (seeded by the anchor token) is
    # supervised too, unlike DFlash infill where slot 0 is the masked anchor. With a
    # fully supervised sequence every within-block position should accumulate count.
    m = _make_model(block_size=4, num_anchors=8)
    b = _make_batch(B=2, S=40)
    b["loss_mask"] = torch.ones(2, 40)
    _, _, _, _, cpp = m(**b)
    assert cpp.shape[0] == 4
    assert (cpp > 0).all(), f"some slot unsupervised: {cpp.tolist()}"


def test_grad_flow_and_frozen_embedding():
    m = _make_model()
    loss, *_ = m(**_make_batch(seed=2))
    loss.backward()
    d = m.draft_model
    assert d.markov_head.markov_w2.weight.grad is not None
    assert d.markov_head.markov_w2.weight.grad.abs().sum() > 0
    assert d.confidence_head.proj.weight.grad is not None
    assert d.confidence_head.proj.weight.grad.abs().sum() > 0
    assert d.context_proj.weight.grad is not None
    assert d.embed_tokens.weight.grad is None  # frozen


# --------------------------------------------------------------------------------------
# Head math
# --------------------------------------------------------------------------------------
def test_vanilla_markov_is_low_rank_bigram_bias():
    torch.manual_seed(0)
    mk = DFlashMarkovHead(vocab_size=50, markov_rank=8)
    base = torch.randn(2, 3, 4, 50)
    prev = torch.randint(0, 50, (2, 3, 4))
    out = mk.apply_block_logits(base, prev)
    expected = base + mk.markov_w2(mk.markov_w1(prev))
    torch.testing.assert_close(out, expected, rtol=1e-5, atol=1e-6)


def test_confidence_head_is_linear():
    torch.manual_seed(0)
    head = DFlashAcceptRatePredictor(20)
    feats = torch.randn(2, 3, 4, 20)
    out = head(feats)
    torch.testing.assert_close(out, head.proj(feats).squeeze(-1), rtol=1e-5, atol=1e-6)
    assert out.shape == (2, 3, 4)


# --------------------------------------------------------------------------------------
# Numerical faithfulness vs the vendored DeepSpec reference
# --------------------------------------------------------------------------------------
def test_dspark_loss_components_match_deepspec_reference():
    """DFlashModel._dspark_loss components match the vendored DeepSpec formula.

    Fully-supervised eval mask (all rows valid) so the valid-row optimization in
    _dspark_loss reduces to the dense reference computation.
    """
    torch.manual_seed(0)
    B, N, bs, V, H, S = 2, 3, 4, 128, 64, 40
    m = _make_model(block_size=bs, num_anchors=8)

    flat_logits = torch.randn(B * N * bs, V)
    target_ids = torch.randint(0, V, (B, N, bs))
    label_indices = torch.randint(1, S, (B, N, bs))
    target_last_hidden = torch.randn(B, S, H)
    lm_head_weight = torch.randn(V, H)
    draft_hidden = torch.randn(B * N * bs, H)
    anchor_tok = torch.randint(0, V, (B, N, 1))
    prev_token_ids = torch.cat([anchor_tok, target_ids[:, :, :-1]], dim=-1)

    eval_mask = torch.ones(B, N, bs)  # fully supervised
    k = torch.arange(bs).view(1, 1, -1)
    decay = torch.exp(-k.float() / GAMMA)

    loss_per_token = F.cross_entropy(flat_logits, target_ids.reshape(-1), reduction="none")

    m._dspark_loss(
        flat_logits=flat_logits,
        loss_per_token=loss_per_token,
        draft_hidden=draft_hidden,
        weight_mask=eval_mask,
        decay_weights=decay,
        label_indices=label_indices,
        target_last_hidden_states=target_last_hidden,
        lm_head_weight=lm_head_weight,
        bsz=B,
        n_blocks=N,
        prev_token_ids=prev_token_ids,
    )
    ours = m._dspark_components

    # Re-derive the same target logits + confidence the way _dspark_loss does, then
    # apply the vendored DeepSpec formula.
    tgt_idx = (label_indices - 1).clamp(min=0, max=S - 1)
    gather_idx = tgt_idx.reshape(B, N * bs, 1).expand(B, N * bs, H)
    aligned = torch.gather(target_last_hidden, 1, gather_idx).reshape(B, N, bs, H)
    target_logits = F.linear(aligned, lm_head_weight)
    draft_logits_4d = flat_logits.reshape(B, N, bs, V)

    prev_emb = m.draft_model.markov_head.get_prev_embeddings(prev_token_ids)
    conf_feats = torch.cat([draft_hidden.reshape(B, N, bs, H), prev_emb], dim=-1)
    confidence_pred = m.draft_model.confidence_head(conf_feats)

    ref_ce, ref_l1, ref_conf = _deepspec_reference_components(
        draft_logits_4d, target_logits, target_ids, eval_mask, decay, confidence_pred
    )

    torch.testing.assert_close(ours["dspark_ce"], ref_ce, rtol=1e-4, atol=1e-5)
    torch.testing.assert_close(ours["dspark_l1"], ref_l1, rtol=1e-4, atol=1e-5)
    torch.testing.assert_close(ours["dspark_conf"], ref_conf, rtol=1e-4, atol=1e-5)
