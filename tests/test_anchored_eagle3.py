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

"""Anchored Eagle3: the gathered-query mask, and the model plumbing around it."""

import unittest
import unittest.mock

import torch

from torchspec.models.anchored_eagle3 import AnchoredEagle3Model, anchored_bool_mask
from torchspec.models.draft.auto import AutoDraftModelConfig, AutoEagle3DraftModel
from torchspec.models.draft.deepseek_eagle import Eagle3DeepseekV2ForCausalLM
from torchspec.models.eagle3 import Eagle3Model, LazyTarget, PrecomputedTarget
from torchspec.models.ops.anchors import sample_anchor_positions
from torchspec.models.ops.flex_attention import generate_eagle3_mask

S, N, T = 12, 4, 4
ANCHORS = torch.tensor([[2, 5, 8, 10]])
KEEP = torch.tensor([[True, True, True, True]])

_EAGLE3 = {
    "architectures": ["LlamaForCausalLMEagle3"],
    "model_type": "llama",
    "hidden_size": 64,
    "intermediate_size": 128,
    "num_hidden_layers": 1,
    "num_attention_heads": 4,
    "num_key_value_heads": 4,
    "vocab_size": 128,
    "max_position_embeddings": 256,
    "rms_norm_eps": 1e-6,
    "rope_theta": 10000.0,
    "pad_token_id": 0,
    "tie_word_embeddings": False,
}


def _mla_draft():
    """A DeepSeek-style draft: keys come from the compressed kv_a/kv_b pair."""
    from transformers.models.deepseek_v3.configuration_deepseek_v3 import DeepseekV3Config

    config = DeepseekV3Config(
        hidden_size=64,
        num_attention_heads=4,
        num_key_value_heads=4,
        intermediate_size=256,
        max_position_embeddings=1024,
        vocab_size=128,
        hidden_act="silu",
        rms_norm_eps=1e-6,
        pad_token_id=0,
        q_lora_rank=48,
        kv_lora_rank=32,
        qk_nope_head_dim=16,
        qk_rope_head_dim=8,
        v_head_dim=16,
        num_hidden_layers=1,
        n_routed_experts=1,
        n_shared_experts=0,
        first_k_dense_replace=0,
        num_experts_per_tok=1,
    )
    config.draft_vocab_size = 128
    config.target_hidden_size = 64
    return Eagle3DeepseekV2ForCausalLM(config, attention_backend="sdpa").cuda().float()


def _mask(anchors, keep, kv_len):
    """[N, KV] view of the anchored mask for one batch element."""
    return anchored_bool_mask(anchors, keep, S, N, kv_len)[0, 0]


class TestAnchoredMask(unittest.TestCase):
    def test_context_rows_match_the_dense_eagle3_mask(self):
        """Anchor a must see exactly what dense position a sees in the context block."""
        q = torch.arange(S).view(-1, 1).expand(S, 2 * S)
        kv = torch.arange(2 * S).view(1, -1).expand(S, 2 * S)
        dense = generate_eagle3_mask(Q_LEN=S, KV_LEN=2 * S, lck=1)(
            torch.tensor(0), torch.tensor(0), q, kv
        )
        anchored = _mask(ANCHORS, KEEP, S + N)

        for slot, position in enumerate(ANCHORS[0].tolist()):
            torch.testing.assert_close(anchored[slot, :S], dense[position, :S])

    def test_each_anchor_sees_only_its_own_chain_slot(self):
        depth = 2
        anchored = _mask(ANCHORS, KEEP, S + depth * N)

        chain = anchored[:, S:]
        for block in range(depth):
            torch.testing.assert_close(
                chain[:, block * N : (block + 1) * N], torch.eye(N, dtype=torch.bool)
            )

    def test_keep_mask_blanks_a_padded_anchor(self):
        keep = torch.tensor([[True, False, True, True]])
        anchored = _mask(ANCHORS, keep, S + N)

        self.assertFalse(anchored[1].any())
        self.assertTrue(anchored[0].any())


class TestAnchoredModel(unittest.TestCase):
    def _batch(self, bsz=1, seq=64, hidden=64):
        return dict(
            input_ids=torch.randint(0, 128, (bsz, seq), device="cuda"),
            attention_mask=None,
            target=PrecomputedTarget(
                target_p_padded=torch.softmax(torch.randn(bsz, seq + T, 128, device="cuda"), dim=-1)
            ),
            loss_mask=torch.ones(bsz, seq, device="cuda"),
            hidden_states=torch.randn(bsz, seq, hidden * 3, device="cuda"),
        )

    def _model(self, **kwargs):
        torch.manual_seed(0)
        config = AutoDraftModelConfig.from_dict(dict(_EAGLE3))
        draft = AutoEagle3DraftModel.from_config(config, torch_dtype=torch.float32)
        return AnchoredEagle3Model(
            draft_model=draft, length=T, attention_backend="flex_attention", **kwargs
        )

    def test_rejects_mrope_drafts(self):
        """MRoPE rotates through apply_multimodal_rotary_pos_emb, which this path does not."""
        torch.manual_seed(0)
        config = AutoDraftModelConfig.from_dict(
            dict(_EAGLE3, rope_scaling={"rope_type": "mrope", "mrope_section": [16, 24, 24]})
        )
        draft = AutoEagle3DraftModel.from_config(config, torch_dtype=torch.float32)

        with self.assertRaisesRegex(ValueError, "MRoPE"):
            AnchoredEagle3Model(draft_model=draft, length=T, attention_backend="flex_attention")

    def test_rejects_unsupported_loss_types(self):
        torch.manual_seed(0)
        config = AutoDraftModelConfig.from_dict(dict(_EAGLE3))
        draft = AutoEagle3DraftModel.from_config(config, torch_dtype=torch.float32)

        with self.assertRaisesRegex(ValueError, "forward_kl"):
            AnchoredEagle3Model(
                draft_model=draft,
                length=T,
                attention_backend="flex_attention",
                loss_type="lk_alpha",
            )

    @unittest.skipUnless(torch.cuda.is_available(), "FlexAttention path needs CUDA")
    def test_mla_draft_with_far_fewer_anchors_than_positions(self):
        """The rotary table must span the sequence, not the anchor count.

        MLA's own _apply_rope_and_assemble sizes cos/sin from its query length. Here that
        is num_anchors (4) while anchors reach position 63, so reusing it indexed off the
        end of the table -- a device-side assert in training and invisible whenever a test
        gives num_anchors and seq_len the same size.
        """
        model = AnchoredEagle3Model(
            draft_model=_mla_draft(), length=T, attention_backend="sdpa", num_anchors=N
        ).cuda()

        plosses, *_ = model(**self._batch(seq=64))

        self.assertEqual(len(plosses), T)
        for loss in plosses:
            self.assertTrue(torch.isfinite(loss))

    @unittest.skipUnless(torch.cuda.is_available(), "FlexAttention path needs CUDA")
    def test_lazy_target_path(self):
        """Full-vocab configs use LazyTarget; PrecomputedTarget only appears with pruning."""
        model = self._model(num_anchors=N).cuda()
        bsz, seq, hidden = 1, 64, 64
        ids = torch.randint(0, 128, (bsz, seq), device="cuda")
        aux = torch.randn(bsz, seq, hidden * model.draft_model.num_aux_hidden_states, device="cuda")
        target = LazyTarget(
            hidden_states_padded=torch.randn(bsz, seq + T, hidden, device="cuda"),
            lm_head_weight=torch.randn(128, hidden, device="cuda"),
        )

        plosses, *_ = model(
            input_ids=ids,
            attention_mask=None,
            target=target,
            loss_mask=torch.ones(bsz, seq, device="cuda"),
            hidden_states=aux,
        )

        self.assertEqual(len(plosses), T)
        for loss in plosses:
            self.assertTrue(torch.isfinite(loss))

    @unittest.skipUnless(torch.cuda.is_available(), "FlexAttention path needs CUDA")
    def test_every_trainable_parameter_receives_a_gradient(self):
        """A disconnected component would train silently; assert the whole graph is live."""
        model = self._model(num_anchors=N).cuda()
        model.draft_model.freeze_embedding()

        sum(model(**self._batch())[0]).backward()

        missing, nonfinite = [], []
        for name, param in model.named_parameters():
            if not param.requires_grad:
                continue
            if param.grad is None or not param.grad.any():
                missing.append(name)
            elif not torch.isfinite(param.grad).all():
                nonfinite.append(name)
        self.assertEqual(missing, [], f"no gradient reached: {missing}")
        self.assertEqual(nonfinite, [], f"non-finite gradient: {nonfinite}")

    @unittest.skipUnless(torch.cuda.is_available(), "FlexAttention path needs CUDA")
    def test_each_depth_contributes_gradient(self):
        """Every TTT depth must be differentiable on its own, not just the sum."""
        for depth in range(T):
            model = self._model(num_anchors=N).cuda()
            model.draft_model.freeze_embedding()

            model(**self._batch())[0][depth].backward()

            grad = model.draft_model.midlayer.self_attn.q_proj.weight.grad
            self.assertIsNotNone(grad, f"depth {depth} produced no gradient")
            self.assertTrue(torch.isfinite(grad).all() and grad.any(), f"depth {depth}")


class TestMatchesDense(unittest.TestCase):
    """Anchored must be a faithful reimplementation of the dense TTT unroll.

    With every supervised position selected as an anchor, the two paths supervise the
    same positions at every depth >= 1 and must agree numerically. Depth 0 is excluded
    only because dense also supervises the one trailing position that cannot be an
    anchor (an anchor needs ``loss_mask[a] & loss_mask[a + 1]``).
    """

    def _llama_draft(self):
        config = AutoDraftModelConfig.from_dict(dict(_EAGLE3))
        return AutoEagle3DraftModel.from_config(config, torch_dtype=torch.float32).cuda()

    def test_per_depth_losses_match_dense(self):
        for name in ("llama", "mla"):
            with self.subTest(draft=name):
                self._assert_matches_dense(name)

    def _assert_matches_dense(self, draft_kind):
        torch.manual_seed(0)
        seq, hidden, supervised = 64, 64, 40

        draft = self._llama_draft() if draft_kind == "llama" else _mla_draft()

        # Supervise a prefix so that the anchor set is exactly {0 .. supervised - 2}:
        # num_anchors exceeds the candidate count, so sampling is deterministic.
        loss_mask = torch.zeros(1, seq, device="cuda")
        loss_mask[:, :supervised] = 1.0
        batch = dict(
            input_ids=torch.randint(0, 128, (1, seq), device="cuda"),
            attention_mask=torch.ones(1, seq, dtype=torch.long, device="cuda"),
            target=PrecomputedTarget(
                target_p_padded=torch.softmax(torch.randn(1, seq + T, 128, device="cuda"), dim=-1)
            ),
            loss_mask=loss_mask,
            hidden_states=torch.randn(1, seq, hidden * 3, device="cuda"),
        )

        # "sdpa" so the dense path builds the additive decoder mask its eager attention
        # needs; the anchored path builds its own mask either way.
        shared = dict(draft_model=draft, length=T, attention_backend="sdpa")
        dense = Eagle3Model(**shared)
        anchored = AnchoredEagle3Model(num_anchors=seq, **shared)

        _, dense_losses, dense_acc, _, _ = dense(**batch)
        _, anch_losses, anch_acc, _, _ = anchored(**batch)

        for depth in range(1, T):
            self.assertTrue(
                torch.allclose(dense_losses[depth], anch_losses[depth], atol=1e-4),
                f"depth {depth}: dense {dense_losses[depth].item():.6f} "
                f"vs anchored {anch_losses[depth].item():.6f}",
            )
            self.assertTrue(
                torch.allclose(dense_acc[depth], anch_acc[depth], atol=1e-4),
                f"depth {depth} accuracy",
            )


def _depth_coverage(anchors, keep, loss_mask, seq, ttt):
    """(fraction of supervised tokens seen at every depth, mean depths per token)."""
    from collections import defaultdict

    seen = defaultdict(set)
    for a in anchors[0][keep[0]].tolist():
        for depth in range(ttt):
            target = min(a + depth, seq - 1)
            if loss_mask[0, target] > 0.5:
                seen[target].add(depth)
    if not seen:
        return 0.0, 0.0
    counts = [len(v) for v in seen.values()]
    return sum(c == ttt for c in counts) / len(counts), sum(counts) / len(counts)


class TestAnchorGap(unittest.TestCase):
    """``max_gap`` caps how many supervised positions sit between neighbouring anchors.

    Anchor ``a`` supervises token ``a + d`` at depth ``d``, so a token is seen at depth
    ``d`` only if the anchor ``d`` positions behind it was also picked. A gap of ``g``
    leaves each token ``ttt_length / (g + 1)`` of its depths; once ``g`` reaches
    ``ttt_length`` every token is pinned to a single depth. ``None`` means ``block_size``,
    which is what DFlash and the dense-vocabulary paths take.
    """

    TTT = 4

    def _mask(self, seq, supervised):
        m = torch.zeros(1, seq)
        m[:, :supervised] = 1.0
        return m

    def _kept(self, seq, supervised, n, ttt=None, **kw):
        mask = self._mask(seq, supervised)
        anchors, keep = sample_anchor_positions(seq, mask, n, ttt or self.TTT, "cpu", **kw)
        return mask, sorted(anchors[0][keep[0]].tolist())

    def _spacings(self, kept):
        """Gaps between neighbours, dropping the single seam where the window wraps."""
        diffs = sorted(b - a for a, b in zip(kept, kept[1:]))
        return set(diffs[:-1]) or set(diffs)

    def test_gap_sets_the_spacing_between_anchors(self):
        for gap in (0, 2):
            _, kept = self._kept(4096, 3000, 128, max_gap=gap)
            self.assertEqual(self._spacings(kept), {gap + 1}, f"max_gap={gap}")

    def test_default_gap_spreads_anchors_but_never_past_block_size(self):
        # valid // num_anchors == 2, well under the cap, so anchors span the whole sample
        _, spread = self._kept(2048, 1500, 512, ttt=16)
        self.assertEqual(self._spacings(spread), {2})

        # here the even stride would be 124, so the block_size cap binds instead
        _, capped = self._kept(8192, 8000, 64, ttt=16)
        self.assertEqual(self._spacings(capped), {17})

    def test_gap_zero_restores_depth_coverage(self):
        seq, supervised, n = 2048, 1500, 128
        mask = self._mask(seq, supervised)

        pinned = sample_anchor_positions(seq, mask, n, self.TTT, "cpu")  # None -> ttt_length
        adjacent = sample_anchor_positions(seq, mask, n, self.TTT, "cpu", max_gap=0)

        full_pinned, mean_pinned = _depth_coverage(*pinned, mask, seq, self.TTT)
        full_adjacent, mean_adjacent = _depth_coverage(*adjacent, mask, seq, self.TTT)

        self.assertEqual(full_pinned, 0.0, "the default gap pins every token to one depth")
        self.assertAlmostEqual(mean_pinned, 1.0, delta=0.05)
        self.assertGreater(full_adjacent, 0.9, f"max_gap=0 covers nearly all, got {full_adjacent}")
        self.assertGreater(mean_adjacent, mean_pinned)

    def test_every_valid_position_is_used_when_the_budget_allows(self):
        for gap in (None, 0, 4):
            _, kept = self._kept(512, 200, 512, max_gap=gap)
            self.assertEqual(kept, list(range(199)), f"max_gap={gap}")

    def test_window_wraps_so_every_candidate_stays_reachable(self):
        """The phase is drawn over every candidate and the window wraps past the end.

        Clamping the phase to the leftover slack instead would make the window a fixed
        interval whose centre is picked almost always and whose ends almost never: with
        700 candidates and 512 anchors the middle came out at 100% and the edges at 0.5%,
        against a flat 512/700 for uniform sampling. Wrapping restores the flat marginal.
        """
        seq, supervised, n = 1024, 701, 512
        mask = torch.zeros(1, seq)
        mask[:, :supervised] = 1.0
        valid = supervised - 1  # 700 candidates, stride 1, so the window spans 512 of them

        # Force the phase to the far end of the candidate list.
        with unittest.mock.patch("torch.rand", lambda *shape, **kw: torch.full(shape, 0.99)):
            anchors, keep = sample_anchor_positions(seq, mask, n, 16, "cpu")

        kept = sorted(anchors[0][keep[0]].tolist())
        self.assertEqual(len(kept), n)
        self.assertEqual(kept[-1], valid - 1, "the last candidate must be reachable")
        self.assertEqual(kept[0], 0, "and the window must wrap round to the first")

    def test_never_selects_unsupervised_positions(self):
        # two supervised spans with a hole, as in a multi-turn conversation
        mask = torch.zeros(1, 2048)
        mask[:, 100:700] = 1.0
        mask[:, 1200:1800] = 1.0

        for gap in (None, 0, 1, 4):
            anchors, keep = sample_anchor_positions(2048, mask, 128, self.TTT, "cpu", max_gap=gap)
            for a in anchors[0][keep[0]].tolist():
                self.assertTrue(
                    mask[0, a] > 0.5 and mask[0, a + 1] > 0.5, f"gap={gap}: anchor {a} invalid"
                )


if __name__ == "__main__":
    unittest.main()
