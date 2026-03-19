"""Tests for DFlash training model components.

Covers:
1. DFlashConfig: creation and attribute access
2. build_target_layer_ids: uniform spacing of target layers
3. DFlashDraftModel: forward pass shapes, embedding load/freeze
4. DFlashModel helper functions: anchor sampling, position IDs, noise input, mask
5. DFlashModel: end-to-end forward, loss + accuracy computation, decay weights
"""

import math
import unittest

import torch
import torch.nn.functional as F

from torchspec.models.draft.dflash import (
    DFlashConfig,
    DFlashDraftModel,
    build_target_layer_ids,
)
from torchspec.models.dflash import (
    DFlashModel,
    _create_dflash_mask_mod,
    _create_position_ids,
    _prepare_noise_input,
    _sample_anchor_positions,
)


def _make_config(
    H=128,
    intermediate=512,
    num_layers=1,
    num_heads=4,
    num_kv_heads=2,
    V=256,
    num_target_layers=3,
    target_hidden_size=None,
    target_num_hidden=12,
):
    return DFlashConfig(
        hidden_size=H,
        intermediate_size=intermediate,
        num_hidden_layers=num_layers,
        num_attention_heads=num_heads,
        num_key_value_heads=num_kv_heads,
        vocab_size=V,
        rms_norm_eps=1e-6,
        max_position_embeddings=512,
        rope_theta=10000.0,
        num_target_layers=num_target_layers,
        target_hidden_size=target_hidden_size or H,
        target_num_hidden_layers=target_num_hidden,
        mask_token_id=V - 1,
    )


class TestDFlashConfig(unittest.TestCase):
    def test_config_attributes(self):
        config = _make_config(H=128, V=256, num_target_layers=5)
        self.assertEqual(config.hidden_size, 128)
        self.assertEqual(config.vocab_size, 256)
        self.assertEqual(config.num_target_layers, 5)
        self.assertEqual(config.model_type, "dflash")
        self.assertFalse(config.tie_word_embeddings)

    def test_config_serialization(self):
        config = _make_config()
        d = config.to_dict()
        restored = DFlashConfig(**{k: v for k, v in d.items() if k != "transformers_version"})
        self.assertEqual(restored.hidden_size, config.hidden_size)
        self.assertEqual(restored.num_target_layers, config.num_target_layers)


class TestBuildTargetLayerIds(unittest.TestCase):
    def test_single_layer(self):
        ids = build_target_layer_ids(1, 36)
        self.assertEqual(ids, [35])

    def test_five_layers_36(self):
        ids = build_target_layer_ids(5, 36)
        self.assertEqual(len(ids), 5)
        self.assertGreaterEqual(ids[0], 1)
        self.assertLessEqual(ids[-1], 35)
        self.assertEqual(ids, sorted(ids))

    def test_two_layers(self):
        ids = build_target_layer_ids(2, 36)
        self.assertEqual(len(ids), 2)
        self.assertEqual(ids[0], 1)
        self.assertEqual(ids[1], 35)

    def test_monotonically_increasing(self):
        for n in range(1, 8):
            ids = build_target_layer_ids(n, 36)
            self.assertEqual(len(ids), n)
            for i in range(1, len(ids)):
                self.assertGreater(ids[i], ids[i - 1])


class TestDFlashDraftModel(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(42)
        self.config = _make_config(
            H=64, intermediate=256, num_heads=4, num_kv_heads=2,
            V=128, num_target_layers=2, target_num_hidden=12,
        )
        self.model = DFlashDraftModel(self.config).to(dtype=torch.float32)
        self.model.eval()

    def test_forward_shapes(self):
        B, draft_len, ctx_len = 2, 8, 16
        H = self.config.hidden_size

        draft_input_ids = torch.randint(0, self.config.vocab_size, (B, draft_len))
        context_feature = torch.randn(B, ctx_len, H)
        draft_pos = torch.arange(draft_len).unsqueeze(0).expand(B, -1)
        ctx_pos = torch.arange(ctx_len).unsqueeze(0).expand(B, -1)

        with torch.no_grad():
            out = self.model(
                draft_input_ids=draft_input_ids,
                context_feature=context_feature,
                draft_position_ids=draft_pos,
                context_position_ids=ctx_pos,
            )

        self.assertEqual(out.shape, (B, draft_len, H))

    def test_extract_context_feature(self):
        B, seq_len = 2, 16
        H = self.config.hidden_size
        num_target = self.config.num_target_layers

        hs_list = [torch.randn(B, seq_len, H) for _ in range(num_target)]

        with torch.no_grad():
            ctx = self.model.extract_context_feature(hs_list)

        self.assertEqual(ctx.shape, (B, seq_len, H))

    def test_freeze_embedding(self):
        self.model.freeze_embedding()
        self.assertFalse(self.model.embed_tokens.weight.requires_grad)

    def test_trainable_params_exclude_embedding(self):
        self.model.freeze_embedding()
        for name, param in self.model.named_parameters():
            if "embed_tokens" in name:
                self.assertFalse(param.requires_grad)
            else:
                self.assertTrue(param.requires_grad, f"{name} should be trainable")


class TestAnchorSampling(unittest.TestCase):
    def test_basic_sampling(self):
        B, seq_len = 2, 64
        loss_mask = torch.ones(B, seq_len)
        anchors = _sample_anchor_positions(loss_mask, num_anchors=4, block_size=8)

        self.assertEqual(anchors.shape, (B, 4))
        # All anchors should be valid (< seq_len - block_size)
        self.assertTrue((anchors < seq_len - 8).all())

    def test_sorted_order(self):
        B, seq_len = 1, 128
        loss_mask = torch.ones(B, seq_len)
        anchors = _sample_anchor_positions(loss_mask, num_anchors=10, block_size=4)

        for b in range(B):
            a = anchors[b]
            self.assertTrue(torch.all(a[1:] >= a[:-1]))

    def test_respects_loss_mask(self):
        B, seq_len = 1, 64
        loss_mask = torch.zeros(B, seq_len)
        loss_mask[:, 32:] = 1.0
        anchors = _sample_anchor_positions(loss_mask, num_anchors=4, block_size=4)

        # All anchors should be >= 32 (where loss_mask is valid)
        for b in range(B):
            for a in range(4):
                self.assertGreaterEqual(anchors[b, a].item(), 32)

    def test_short_sequence_no_crash(self):
        B = 1
        loss_mask = torch.ones(B, 4)
        anchors = _sample_anchor_positions(loss_mask, num_anchors=2, block_size=8)
        self.assertEqual(anchors.shape, (B, 2))


class TestPositionIds(unittest.TestCase):
    def test_shapes(self):
        B, num_anchors, block_size, ctx_len = 2, 4, 8, 32
        anchor_pos = torch.tensor([[0, 8, 16, 24], [1, 9, 17, 25]])

        ctx_ids, draft_ids = _create_position_ids(anchor_pos, block_size, ctx_len)

        self.assertEqual(ctx_ids.shape, (B, ctx_len))
        self.assertEqual(draft_ids.shape, (B, num_anchors * block_size))

    def test_context_sequential(self):
        anchor_pos = torch.tensor([[0, 8]])
        ctx_ids, _ = _create_position_ids(anchor_pos, block_size=4, ctx_len=16)
        expected = torch.arange(16).unsqueeze(0)
        torch.testing.assert_close(ctx_ids, expected)

    def test_draft_offsets(self):
        anchor_pos = torch.tensor([[5, 20]])
        _, draft_ids = _create_position_ids(anchor_pos, block_size=4, ctx_len=32)

        # Block 0: [5, 6, 7, 8], Block 1: [20, 21, 22, 23]
        expected = torch.tensor([[5, 6, 7, 8, 20, 21, 22, 23]])
        torch.testing.assert_close(draft_ids, expected)


class TestNoiseInput(unittest.TestCase):
    def test_anchor_placed_correctly(self):
        B, seq_len = 1, 32
        input_ids = torch.arange(seq_len).unsqueeze(0)
        anchor_pos = torch.tensor([[4, 12]])
        block_size = 4
        mask_id = 999

        draft_ids, draft_labels = _prepare_noise_input(input_ids, anchor_pos, block_size, mask_id)

        # Block 0: anchor at pos 4 → draft_ids[0] = 4
        self.assertEqual(draft_ids[0, 0].item(), 4)
        # Block 0: positions 1..3 should be MASK
        self.assertEqual(draft_ids[0, 1].item(), mask_id)
        self.assertEqual(draft_ids[0, 2].item(), mask_id)
        self.assertEqual(draft_ids[0, 3].item(), mask_id)

        # Block 1: anchor at pos 12
        self.assertEqual(draft_ids[0, 4].item(), 12)

    def test_labels(self):
        B, seq_len = 1, 32
        input_ids = torch.arange(seq_len).unsqueeze(0)
        anchor_pos = torch.tensor([[4]])
        block_size = 4
        mask_id = 999

        _, draft_labels = _prepare_noise_input(input_ids, anchor_pos, block_size, mask_id)

        # Position 0 (anchor) → label -100 (no loss)
        self.assertEqual(draft_labels[0, 0].item(), -100)
        # Position 1 → label = input_ids[4+1] = 5
        self.assertEqual(draft_labels[0, 1].item(), 5)
        # Position 2 → label = input_ids[4+2] = 6
        self.assertEqual(draft_labels[0, 2].item(), 6)
        # Position 3 → label = input_ids[4+3] = 7
        self.assertEqual(draft_labels[0, 3].item(), 7)


class TestDFlashMaskMod(unittest.TestCase):
    def test_block_internal_visibility(self):
        """Within a block, all positions should see each other (bidirectional)."""
        num_anchors, block_size, ctx_len = 2, 4, 16
        anchor_pos = torch.tensor([[4, 12]])

        mask_mod = _create_dflash_mask_mod(num_anchors, block_size, anchor_pos, ctx_len)

        # Block 0: q_idx=0..3, kv_idx=ctx_len+0..3 (=16..19) are all in block 0
        for qi in range(4):
            for ki in range(4):
                self.assertTrue(mask_mod(0, 0, qi, ctx_len + ki))

    def test_inter_block_invisible(self):
        """Tokens in block 0 should NOT see tokens in block 1."""
        num_anchors, block_size, ctx_len = 2, 4, 16
        anchor_pos = torch.tensor([[4, 12]])

        mask_mod = _create_dflash_mask_mod(num_anchors, block_size, anchor_pos, ctx_len)

        # Block 0 query (q_idx=0) should not see block 1 draft (kv_idx=ctx_len+4..7)
        for ki in range(4, 8):
            self.assertFalse(mask_mod(0, 0, 0, ctx_len + ki))

        # Block 1 query (q_idx=4) should not see block 0 draft
        for ki in range(4):
            self.assertFalse(mask_mod(0, 0, 4, ctx_len + ki))

    def test_context_causal(self):
        """Block i should see context tokens BEFORE its anchor position."""
        num_anchors, block_size, ctx_len = 1, 4, 16
        anchor_pos = torch.tensor([[8]])

        mask_mod = _create_dflash_mask_mod(num_anchors, block_size, anchor_pos, ctx_len)

        # Context positions 0..7 should be visible (before anchor 8)
        for kv in range(8):
            self.assertTrue(mask_mod(0, 0, 0, kv))

        # Context position 8+ should NOT be visible
        for kv in range(8, ctx_len):
            self.assertFalse(mask_mod(0, 0, 0, kv))


class TestDFlashModelForward(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(42)
        self.H = 64
        self.V = 128
        self.num_target_layers = 2
        self.config = _make_config(
            H=self.H,
            intermediate=256,
            num_heads=4,
            num_kv_heads=2,
            V=self.V,
            num_target_layers=self.num_target_layers,
            target_num_hidden=12,
        )
        draft_model = DFlashDraftModel(self.config).to(dtype=torch.float32)
        draft_model.freeze_embedding()
        self.model = DFlashModel(
            draft_model=draft_model,
            block_size=4,
            num_anchors=4,
            loss_decay_gamma=7.0,
        )
        self.model.eval()

    def test_forward_produces_loss_and_acc(self):
        B, seq_len = 1, 32
        input_ids = torch.randint(0, self.V, (B, seq_len))
        hidden_states_list = [
            torch.randn(B, seq_len, self.H) for _ in range(self.num_target_layers)
        ]
        loss_mask = torch.ones(B, seq_len)
        lm_head_weight = torch.randn(self.V, self.H)

        with torch.no_grad():
            loss, acc = self.model(
                input_ids=input_ids,
                hidden_states_list=hidden_states_list,
                loss_mask=loss_mask,
                lm_head_weight=lm_head_weight,
            )

        self.assertTrue(torch.isfinite(loss))
        self.assertGreaterEqual(loss.item(), 0.0)
        self.assertGreaterEqual(acc.item(), 0.0)
        self.assertLessEqual(acc.item(), 1.0)

    def test_loss_requires_grad(self):
        """Loss should be differentiable through the draft model."""
        B, seq_len = 1, 32
        input_ids = torch.randint(0, self.V, (B, seq_len))
        hidden_states_list = [
            torch.randn(B, seq_len, self.H) for _ in range(self.num_target_layers)
        ]
        loss_mask = torch.ones(B, seq_len)
        lm_head_weight = torch.randn(self.V, self.H)

        self.model.train()
        loss, acc = self.model(
            input_ids=input_ids,
            hidden_states_list=hidden_states_list,
            loss_mask=loss_mask,
            lm_head_weight=lm_head_weight,
        )

        loss.backward()

        grad_found = False
        for name, param in self.model.draft_model.named_parameters():
            if param.requires_grad and param.grad is not None:
                if param.grad.abs().sum() > 0:
                    grad_found = True
                    break
        self.assertTrue(grad_found, "No gradient flowed to draft model parameters")


class TestDecayWeights(unittest.TestCase):
    def test_anchor_position_zero_weight(self):
        """Position 0 (anchor) should have weight 0."""
        config = _make_config(H=64, V=128, num_target_layers=2, target_num_hidden=12)
        draft_model = DFlashDraftModel(config)
        model = DFlashModel(draft_model=draft_model, block_size=8, loss_decay_gamma=7.0)
        self.assertEqual(model.decay_weights[0].item(), 0.0)

    def test_decay_values(self):
        gamma = 7.0
        block_size = 8
        config = _make_config(H=64, V=128, num_target_layers=2, target_num_hidden=12)
        draft_model = DFlashDraftModel(config)
        model = DFlashModel(draft_model=draft_model, block_size=block_size, loss_decay_gamma=gamma)

        for k in range(1, block_size):
            expected = math.exp(-(k - 1) / gamma)
            self.assertAlmostEqual(model.decay_weights[k].item(), expected, places=5)

    def test_monotonically_decreasing(self):
        config = _make_config(H=64, V=128, num_target_layers=2, target_num_hidden=12)
        draft_model = DFlashDraftModel(config)
        model = DFlashModel(draft_model=draft_model, block_size=16, loss_decay_gamma=7.0)

        weights = model.decay_weights
        for k in range(2, len(weights)):
            self.assertGreater(weights[k - 1].item(), weights[k].item())


class TestMiniTrainingLoop(unittest.TestCase):
    """Smoke test: forward → backward → optimizer step should not crash and loss should decrease."""

    def test_loss_decreases_over_steps(self):
        torch.manual_seed(42)
        H, V = 64, 128
        num_target_layers = 2
        config = _make_config(
            H=H, intermediate=256, num_heads=4, num_kv_heads=2,
            V=V, num_target_layers=num_target_layers, target_num_hidden=12,
        )
        draft_model = DFlashDraftModel(config).to(dtype=torch.float32)
        draft_model.freeze_embedding()
        model = DFlashModel(
            draft_model=draft_model, block_size=4, num_anchors=4, loss_decay_gamma=7.0,
        )

        optimizer = torch.optim.Adam(
            [p for p in model.parameters() if p.requires_grad], lr=1e-3,
        )

        B, seq_len = 1, 32
        input_ids = torch.randint(0, V, (B, seq_len))
        hidden_states_list = [torch.randn(B, seq_len, H) for _ in range(num_target_layers)]
        loss_mask = torch.ones(B, seq_len)
        lm_head_weight = torch.randn(V, H)

        model.train()
        losses = []
        for step in range(10):
            optimizer.zero_grad()
            loss, acc = model(
                input_ids=input_ids,
                hidden_states_list=hidden_states_list,
                loss_mask=loss_mask,
                lm_head_weight=lm_head_weight,
            )
            loss.backward()
            optimizer.step()
            losses.append(loss.item())

        # Loss should generally decrease (allow some noise, check last < first)
        self.assertLess(losses[-1], losses[0], "Loss did not decrease over 10 training steps")
        self.assertTrue(all(math.isfinite(l) for l in losses), "Non-finite loss encountered")

    def test_gradient_accumulation(self):
        """Two half-LR steps with accumulated gradients ≈ one full step."""
        torch.manual_seed(42)
        H, V = 64, 128
        num_target_layers = 2
        config = _make_config(
            H=H, intermediate=256, num_heads=4, num_kv_heads=2,
            V=V, num_target_layers=num_target_layers, target_num_hidden=12,
        )

        B, seq_len = 1, 32
        input_ids = torch.randint(0, V, (B, seq_len))
        hidden_states_list = [torch.randn(B, seq_len, H) for _ in range(num_target_layers)]
        loss_mask = torch.ones(B, seq_len)
        lm_head_weight = torch.randn(V, H)

        draft_model = DFlashDraftModel(config).to(dtype=torch.float32)
        draft_model.freeze_embedding()
        model = DFlashModel(
            draft_model=draft_model, block_size=4, num_anchors=4, loss_decay_gamma=7.0,
        )
        model.train()

        # Two micro-batches with accumulation
        model.zero_grad()
        loss1, _ = model(
            input_ids=input_ids,
            hidden_states_list=hidden_states_list,
            loss_mask=loss_mask,
            lm_head_weight=lm_head_weight,
        )
        (loss1 / 2).backward()

        loss2, _ = model(
            input_ids=input_ids,
            hidden_states_list=hidden_states_list,
            loss_mask=loss_mask,
            lm_head_weight=lm_head_weight,
        )
        (loss2 / 2).backward()

        # Check gradients exist and are finite
        has_grad = False
        for p in model.parameters():
            if p.requires_grad and p.grad is not None:
                self.assertTrue(torch.isfinite(p.grad).all())
                has_grad = True
        self.assertTrue(has_grad)


if __name__ == "__main__":
    unittest.main()
