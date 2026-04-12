# Copyright (c) 2026 LightSeek Foundation
# Tests for DFlash training implementation.
# Verifies precision alignment with SpecForge and dataflow integration.

import unittest

import torch


def _reference_dflash_mask(anchor_positions, block_keep_mask, S, block_size, device):
    """Element-level reference mask using Python loops.

    Copied from SpecForge/tests/test_utils/test_dflash_mask.py for precision alignment.
    """
    B, N = anchor_positions.shape
    Q_LEN = N * block_size
    KV_LEN = S + N * block_size

    mask = torch.zeros(B, 1, Q_LEN, KV_LEN, dtype=torch.bool, device=device)
    for b in range(B):
        for q_idx in range(Q_LEN):
            q_block_id = q_idx // block_size
            anchor_pos = anchor_positions[b, q_block_id].item()
            is_valid = block_keep_mask[b, q_block_id].item()
            if not is_valid:
                continue
            for kv_idx in range(KV_LEN):
                is_context = kv_idx < S
                ctx_visible = is_context and (kv_idx < anchor_pos)
                is_draft = kv_idx >= S
                kv_block_id = (kv_idx - S) // block_size
                draft_visible = is_draft and (q_block_id == kv_block_id)
                if ctx_visible or draft_visible:
                    mask[b, 0, q_idx, kv_idx] = True
    return mask


class TestBuildTargetLayerIds(unittest.TestCase):
    def test_single_layer(self):
        from torchspec.models.draft.dflash import build_target_layer_ids

        self.assertEqual(build_target_layer_ids(36, 1), [18])
        self.assertEqual(build_target_layer_ids(28, 1), [14])

    def test_multi_layer(self):
        from torchspec.models.draft.dflash import build_target_layer_ids

        ids = build_target_layer_ids(36, 5)
        self.assertEqual(len(ids), 5)
        self.assertEqual(ids[0], 1)
        self.assertEqual(ids[-1], 33)
        # All IDs should be unique and sorted
        self.assertEqual(ids, sorted(set(ids)))

    def test_two_layers(self):
        from torchspec.models.draft.dflash import build_target_layer_ids

        ids = build_target_layer_ids(36, 2)
        self.assertEqual(len(ids), 2)
        self.assertEqual(ids[0], 1)
        self.assertEqual(ids[1], 33)


class TestDFlashSdpaMask(unittest.TestCase):
    def setUp(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def _compare_masks(self, anchor_positions, block_keep_mask, S, block_size):
        from torchspec.models.dflash import create_dflash_sdpa_mask

        anchor_positions = anchor_positions.to(self.device)
        block_keep_mask = block_keep_mask.to(self.device)

        sdpa_mask = create_dflash_sdpa_mask(
            anchor_positions=anchor_positions,
            block_keep_mask=block_keep_mask,
            S=S,
            block_size=block_size,
            device=self.device,
        )
        ref_mask = _reference_dflash_mask(
            anchor_positions=anchor_positions,
            block_keep_mask=block_keep_mask,
            S=S,
            block_size=block_size,
            device=self.device,
        )
        self.assertEqual(sdpa_mask.shape, ref_mask.shape)
        self.assertTrue(
            torch.equal(sdpa_mask, ref_mask),
            f"Mask mismatch with S={S}, block_size={block_size}",
        )

    def test_single_batch_single_block(self):
        self._compare_masks(torch.tensor([[64]]), torch.tensor([[True]]), S=128, block_size=4)

    def test_single_batch_multi_block(self):
        self._compare_masks(
            torch.tensor([[32, 64, 96]]),
            torch.tensor([[True, True, True]]),
            S=128,
            block_size=4,
        )

    def test_multi_batch(self):
        self._compare_masks(
            torch.tensor([[16, 48, 80], [32, 64, 100]]),
            torch.tensor([[True, True, True], [True, True, True]]),
            S=128,
            block_size=4,
        )

    def test_invalid_blocks(self):
        self._compare_masks(
            torch.tensor([[20, 50, 80, 110]]),
            torch.tensor([[True, False, True, False]]),
            S=128,
            block_size=4,
        )

    def test_all_blocks_invalid(self):
        self._compare_masks(
            torch.tensor([[30, 60]]),
            torch.tensor([[False, False]]),
            S=128,
            block_size=4,
        )

    def test_anchor_at_zero(self):
        self._compare_masks(
            torch.tensor([[0, 64]]),
            torch.tensor([[True, True]]),
            S=128,
            block_size=4,
        )

    def test_various_block_sizes(self):
        for block_size in [1, 2, 4, 8, 16]:
            with self.subTest(block_size=block_size):
                self._compare_masks(
                    torch.tensor([[32, 80]]),
                    torch.tensor([[True, True]]),
                    S=128,
                    block_size=block_size,
                )

    def test_random_stress(self):
        rng = torch.Generator().manual_seed(123)
        for trial in range(5):
            with self.subTest(trial=trial):
                B = torch.randint(1, 4, (1,), generator=rng).item()
                N = torch.randint(1, 8, (1,), generator=rng).item()
                S = 64 * torch.randint(1, 5, (1,), generator=rng).item()
                block_size = [1, 2, 4, 8][torch.randint(0, 4, (1,), generator=rng).item()]
                anchor_positions = torch.stack(
                    [torch.randperm(S, generator=rng)[:N].sort().values for _ in range(B)]
                )
                block_keep_mask = torch.rand(B, N, generator=rng) > 0.3
                self._compare_masks(anchor_positions, block_keep_mask, S=S, block_size=block_size)


class TestDataflowIntegration(unittest.TestCase):
    """Test that DFlash-style samples flow through the collator correctly."""

    def test_collator_accepts_dflash_samples(self):
        from torchspec.data.utils import DataCollatorWithPadding

        collator = DataCollatorWithPadding()
        hidden_size = 128
        num_layers = 3

        # DFlash samples: hidden_states + input_ids, no target/last_hidden_states
        samples = []
        for seq_len in [32, 48]:
            samples.append(
                {
                    "input_ids": torch.randint(0, 1000, (1, seq_len)),
                    "hidden_states": torch.randn(1, seq_len, num_layers * hidden_size),
                    "loss_mask": torch.ones(1, seq_len),
                }
            )

        batch = collator(samples)

        self.assertIn("input_ids", batch)
        self.assertIn("hidden_states", batch)
        self.assertIn("loss_mask", batch)
        self.assertIsNotNone(batch["hidden_states"])
        self.assertIsNone(batch["target"])
        self.assertIsNone(batch["last_hidden_states"])
        self.assertEqual(batch["input_ids"].shape[0], 2)
        self.assertEqual(batch["hidden_states"].shape[0], 2)
        # Padded to max length
        self.assertEqual(batch["input_ids"].shape[1], 48)


class TestDFlashSampleFilter(unittest.TestCase):
    """Test that DFlash minimum sample filter works with packed_loss_mask."""

    def test_filter_by_min_loss_tokens(self):
        from torchspec.data.utils import (
            pack_loss_mask,
            serialize_packed_loss_mask,
            unpack_loss_mask,
        )

        block_size = 16
        min_loss_tokens = 2 * block_size

        # Sample with enough loss tokens (40 > 32)
        mask_good = torch.zeros(100, dtype=torch.long)
        mask_good[10:50] = 1
        packed_good = serialize_packed_loss_mask(pack_loss_mask(mask_good))
        self.assertTrue(unpack_loss_mask(packed_good).sum() >= min_loss_tokens)

        # Sample with too few loss tokens (10 < 32)
        mask_bad = torch.zeros(100, dtype=torch.long)
        mask_bad[10:20] = 1
        packed_bad = serialize_packed_loss_mask(pack_loss_mask(mask_bad))
        self.assertTrue(unpack_loss_mask(packed_bad).sum() < min_loss_tokens)


@unittest.skipUnless(torch.cuda.is_available(), "CUDA required")
class TestDFlashDraftModelForward(unittest.TestCase):
    """Test DFlashDraftModel forward pass on GPU."""

    def test_forward_produces_output(self):
        from transformers.models.qwen3.configuration_qwen3 import Qwen3Config

        from torchspec.models.draft.dflash import DFlashDraftModel

        config = Qwen3Config(
            hidden_size=64,
            num_hidden_layers=2,
            num_attention_heads=4,
            num_key_value_heads=2,
            intermediate_size=128,
            vocab_size=256,
            rms_norm_eps=1e-6,
            rope_theta=10000,
            attention_bias=False,
            attention_dropout=0.0,
            max_position_embeddings=512,
            head_dim=16,
            layer_types=["full_attention", "full_attention"],
        )
        config.block_size = 8
        config.num_target_layers = 12
        config.dflash_config = {"mask_token_id": 0, "target_layer_ids": None}

        torch.manual_seed(42)
        model = DFlashDraftModel(config).cuda().bfloat16()

        bsz, seq_len, block_size = 2, 32, 8
        n_blocks = 4
        n_ctx_features = len(model.target_layer_ids)

        noise_embed = torch.randn(
            bsz, n_blocks * block_size, 64, device="cuda", dtype=torch.bfloat16
        )
        target_hidden = torch.randn(
            bsz, seq_len, n_ctx_features * 64, device="cuda", dtype=torch.bfloat16
        )
        pos_ids = (
            torch.arange(seq_len + n_blocks * block_size, device="cuda")
            .unsqueeze(0)
            .expand(bsz, -1)
        )

        output = model(
            position_ids=pos_ids,
            noise_embedding=noise_embed,
            target_hidden=target_hidden,
        )

        self.assertEqual(output.shape, (bsz, n_blocks * block_size, 64))
        self.assertFalse(torch.isnan(output).any())
        self.assertFalse(torch.isinf(output).any())


@unittest.skipUnless(torch.cuda.is_available(), "CUDA required")
class TestDFlashModelLoss(unittest.TestCase):
    """Test DFlashModel training wrapper forward + loss on GPU."""

    def test_forward_loss_and_accuracy(self):
        from transformers.models.qwen3.configuration_qwen3 import Qwen3Config

        from torchspec.models.dflash import DFlashModel
        from torchspec.models.draft.dflash import DFlashDraftModel

        config = Qwen3Config(
            hidden_size=64,
            num_hidden_layers=2,
            num_attention_heads=4,
            num_key_value_heads=2,
            intermediate_size=128,
            vocab_size=256,
            rms_norm_eps=1e-6,
            rope_theta=10000,
            attention_bias=False,
            attention_dropout=0.0,
            max_position_embeddings=512,
            head_dim=16,
            layer_types=["full_attention", "full_attention"],
        )
        config.block_size = 4
        config.num_target_layers = 12
        config.dflash_config = {"mask_token_id": 0, "target_layer_ids": None}

        torch.manual_seed(42)
        draft_model = DFlashDraftModel(config).cuda().bfloat16()
        n_ctx_features = len(draft_model.target_layer_ids)

        lm_head = torch.nn.Linear(64, 256, bias=False).cuda().bfloat16()
        embed_tokens = torch.nn.Embedding(256, 64).cuda().bfloat16()

        dflash_model = DFlashModel(
            draft_model=draft_model,
            target_lm_head=lm_head,
            target_embed_tokens=embed_tokens,
            mask_token_id=0,
            block_size=4,
            attention_backend="sdpa",
            num_anchors=8,
            loss_decay_gamma=5.0,
        )

        bsz, seq_len = 2, 64
        input_ids = torch.randint(1, 256, (bsz, seq_len), device="cuda")
        hidden_states = torch.randn(
            bsz, seq_len, n_ctx_features * 64, device="cuda", dtype=torch.bfloat16
        )
        loss_mask = torch.ones(bsz, seq_len, device="cuda")
        loss_mask[:, :4] = 0  # mask out first few tokens

        loss, accuracy = dflash_model(input_ids, hidden_states, loss_mask)

        self.assertFalse(torch.isnan(loss))
        self.assertGreater(loss.item(), 0)
        self.assertGreaterEqual(accuracy.item(), 0)
        self.assertLessEqual(accuracy.item(), 1)

        # Verify backward works
        loss.backward()
        has_grad = any(
            p.grad is not None and p.grad.abs().sum() > 0
            for p in draft_model.parameters()
            if p.requires_grad
        )
        self.assertTrue(has_grad, "Draft model should have gradients after backward")


if __name__ == "__main__":
    unittest.main(verbosity=2)
