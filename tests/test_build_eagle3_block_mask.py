"""Tests for build_eagle3_block_mask -- the analytical BlockMask builder."""
import unittest
import torch
from torch.nn.attention.flex_attention import create_block_mask, flex_attention

from torchspec.models.ops.flex_attention import (
    build_eagle3_block_mask,
    eagle3_block_mask,
    generate_eagle3_mask,
)


def _mask_to_dense_bool(Q_LEN, KV_LEN, bm, BLOCK_SIZE=128):
    """Convert a BlockMask to a full (Q_LEN, KV_LEN) bool tensor via mask_mod."""
    qi = torch.arange(Q_LEN, device="cuda").unsqueeze(1)
    ki = torch.arange(KV_LEN, device="cuda").unsqueeze(0)
    b = torch.zeros_like(qi)
    h = torch.zeros_like(qi)
    return bm.mask_mod(b, h, qi, ki).bool()


class TestBuildEagle3BlockMask(unittest.TestCase):

    def _reference_block_mask(self, Q_LEN, KV_LEN, B=1, H=1, device="cuda"):
        seq_lengths = torch.tensor([Q_LEN] * B, device=device, dtype=torch.int32)
        mask_mod = generate_eagle3_mask(seq_lengths, Q_LEN, KV_LEN, lck=0)
        return create_block_mask(mask_mod, B=B, H=H, Q_LEN=Q_LEN, KV_LEN=KV_LEN, device=device)

    def test_mask_pattern_matches_reference_small(self):
        """Element-level mask pattern must be identical for small sizes."""
        for n_rounds in [1, 2, 3, 5]:
            Q_LEN = 256
            KV_LEN = Q_LEN * n_rounds
            with self.subTest(rounds=n_rounds):
                ref = self._reference_block_mask(Q_LEN, KV_LEN)
                ours = build_eagle3_block_mask(Q_LEN, KV_LEN, B=1, H=1, device="cuda")
                ref_dense = _mask_to_dense_bool(Q_LEN, KV_LEN, ref)
                our_dense = _mask_to_dense_bool(Q_LEN, KV_LEN, ours)
                self.assertTrue(torch.equal(ref_dense, our_dense),
                    f"Mask pattern mismatch at rounds={n_rounds}")

    def test_mask_pattern_matches_reference_medium(self):
        """Test at 1024 tokens with multiple rounds."""
        for n_rounds in [1, 2, 4]:
            Q_LEN = 1024
            KV_LEN = Q_LEN * n_rounds
            with self.subTest(rounds=n_rounds):
                ref = self._reference_block_mask(Q_LEN, KV_LEN)
                ours = build_eagle3_block_mask(Q_LEN, KV_LEN, B=1, H=1, device="cuda")
                ref_dense = _mask_to_dense_bool(Q_LEN, KV_LEN, ref)
                our_dense = _mask_to_dense_bool(Q_LEN, KV_LEN, ours)
                self.assertTrue(torch.equal(ref_dense, our_dense))

    def test_batch_size_broadcast(self):
        """H=1 mask should work with multi-head attention via broadcast."""
        Q_LEN, KV_LEN = 256, 768
        bm = build_eagle3_block_mask(Q_LEN, KV_LEN, B=2, H=1, device="cuda")
        self.assertEqual(bm.kv_num_blocks.shape[0], 2)
        self.assertEqual(bm.kv_num_blocks.shape[1], 1)

    def test_flex_attention_output_matches(self):
        """flex_attention output must match between analytical and reference mask."""
        torch.manual_seed(42)
        B, H, D = 1, 4, 64
        Q_LEN = 512
        for n_rounds in [1, 2, 3]:
            KV_LEN = Q_LEN * n_rounds
            with self.subTest(rounds=n_rounds):
                q = torch.randn(B, H, Q_LEN, D, device="cuda", dtype=torch.bfloat16)
                k = torch.randn(B, H, KV_LEN, D, device="cuda", dtype=torch.bfloat16)
                v = torch.randn(B, H, KV_LEN, D, device="cuda", dtype=torch.bfloat16)

                ref_bm = self._reference_block_mask(Q_LEN, KV_LEN)
                our_bm = build_eagle3_block_mask(Q_LEN, KV_LEN, B=B, H=1, device="cuda")

                out_ref = flex_attention(q, k, v, block_mask=ref_bm, enable_gqa=False)
                out_ours = flex_attention(q, k, v, block_mask=our_bm, enable_gqa=False)

                self.assertEqual(out_ref.shape, out_ours.shape)
                self.assertFalse(out_ours.isnan().any())
                max_diff = (out_ref - out_ours).abs().max().item()
                self.assertAlmostEqual(max_diff, 0.0, places=5,
                    msg=f"Output diff={max_diff} at rounds={n_rounds}")

    def test_flex_attention_gqa(self):
        """Test with GQA (fewer KV heads than Q heads)."""
        torch.manual_seed(42)
        B, Q_HEADS, KV_HEADS, D = 1, 8, 2, 64
        Q_LEN, KV_LEN = 256, 768

        q = torch.randn(B, Q_HEADS, Q_LEN, D, device="cuda", dtype=torch.bfloat16)
        k = torch.randn(B, KV_HEADS, KV_LEN, D, device="cuda", dtype=torch.bfloat16)
        v = torch.randn(B, KV_HEADS, KV_LEN, D, device="cuda", dtype=torch.bfloat16)

        bm = build_eagle3_block_mask(Q_LEN, KV_LEN, B=B, H=1, device="cuda")
        out = flex_attention(q, k, v, block_mask=bm, enable_gqa=True)

        self.assertEqual(out.shape, (B, Q_HEADS, Q_LEN, D))
        self.assertFalse(out.isnan().any())

    def test_backward_pass(self):
        """Gradients must flow through flex_attention with analytical mask."""
        torch.manual_seed(42)
        B, H, D = 1, 4, 64
        Q_LEN, KV_LEN = 256, 768

        q = torch.randn(B, H, Q_LEN, D, device="cuda", dtype=torch.float32, requires_grad=True)
        k = torch.randn(B, H, KV_LEN, D, device="cuda", dtype=torch.float32, requires_grad=True)
        v = torch.randn(B, H, KV_LEN, D, device="cuda", dtype=torch.float32, requires_grad=True)

        bm = build_eagle3_block_mask(Q_LEN, KV_LEN, B=B, H=1, device="cuda")
        out = flex_attention(q, k, v, block_mask=bm, enable_gqa=False)
        out.sum().backward()

        for name, t in [("q", q), ("k", k), ("v", v)]:
            self.assertIsNotNone(t.grad, f"{name}.grad is None")
            self.assertFalse(t.grad.isnan().any(), f"{name}.grad has NaN")

    def test_backward_gradients_match_reference(self):
        """Gradients must match between analytical and reference masks."""
        torch.manual_seed(42)
        B, H, D = 1, 4, 64
        Q_LEN, KV_LEN = 256, 768

        q1 = torch.randn(B, H, Q_LEN, D, device="cuda", dtype=torch.float32, requires_grad=True)
        k1 = torch.randn(B, H, KV_LEN, D, device="cuda", dtype=torch.float32, requires_grad=True)
        v1 = torch.randn(B, H, KV_LEN, D, device="cuda", dtype=torch.float32, requires_grad=True)
        q2, k2, v2 = [t.clone().detach().requires_grad_(True) for t in (q1, k1, v1)]

        ref_bm = self._reference_block_mask(Q_LEN, KV_LEN)
        our_bm = build_eagle3_block_mask(Q_LEN, KV_LEN, B=B, H=1, device="cuda")

        flex_attention(q1, k1, v1, block_mask=ref_bm, enable_gqa=False).sum().backward()
        flex_attention(q2, k2, v2, block_mask=our_bm, enable_gqa=False).sum().backward()

        for name, g1, g2 in [("q", q1.grad, q2.grad), ("k", k1.grad, k2.grad), ("v", v1.grad, v2.grad)]:
            max_diff = (g1 - g2).abs().max().item()
            self.assertAlmostEqual(max_diff, 0.0, places=4,
                msg=f"Gradient mismatch for {name}: max_diff={max_diff}")

    def test_causal_only_single_round(self):
        """With KV_LEN == Q_LEN (1 round), mask should be purely causal."""
        Q_LEN = 256
        bm = build_eagle3_block_mask(Q_LEN, Q_LEN, B=1, H=1, device="cuda")
        dense = _mask_to_dense_bool(Q_LEN, Q_LEN, bm)

        # Lower triangular
        expected = torch.tril(torch.ones(Q_LEN, Q_LEN, device="cuda", dtype=torch.bool))
        self.assertTrue(torch.equal(dense, expected), "Single round should be purely causal")

    def test_suffix_diagonal_structure(self):
        """Suffix rounds should have exactly one active element per row (diagonal)."""
        Q_LEN = 256
        KV_LEN = 256 * 3  # 3 rounds: 1 causal + 2 suffix
        bm = build_eagle3_block_mask(Q_LEN, KV_LEN, B=1, H=1, device="cuda")
        dense = _mask_to_dense_bool(Q_LEN, KV_LEN, bm)

        # Check suffix region (kv_idx >= Q_LEN)
        suffix_mask = dense[:, Q_LEN:]  # (Q_LEN, 2*Q_LEN)
        for qi in range(Q_LEN):
            active_kv = suffix_mask[qi].nonzero().squeeze(-1)
            # Should have exactly 2 active positions (one per suffix round)
            self.assertEqual(active_kv.numel(), 2,
                f"Row {qi}: expected 2 suffix positions, got {active_kv.numel()}")

    def test_memory_is_negligible(self):
        """Analytical builder should use negligible memory."""
        Q_LEN, KV_LEN = 4096, 4096 * 5
        torch.cuda.reset_peak_memory_stats()
        before = torch.cuda.memory_allocated()
        bm = build_eagle3_block_mask(Q_LEN, KV_LEN, B=1, H=1, device="cuda")
        after = torch.cuda.max_memory_allocated()
        mem_mb = (after - before) / 1024**2
        self.assertLess(mem_mb, 10.0,
            f"Block mask used {mem_mb:.1f} MB -- should be negligible")

    def test_assertion_on_non_divisible(self):
        """Should raise if Q_LEN or KV_LEN not divisible by BLOCK_SIZE."""
        with self.assertRaises(AssertionError):
            build_eagle3_block_mask(100, 300, B=1, H=1, device="cuda")

    def test_assertion_on_non_round_aligned_kv(self):
        """Should raise if KV_LEN is not an integer multiple of Q_LEN."""
        # 256-aligned to BLOCK_SIZE but KV_LEN % Q_LEN != 0.
        with self.assertRaises(AssertionError):
            build_eagle3_block_mask(256, 384, B=1, H=1, device="cuda")


class TestEagle3BlockMaskDispatcher(unittest.TestCase):
    """Tests for the eagle3_block_mask dispatcher (analytical + fallback)."""

    def _ref_dense(self, Q_LEN, KV_LEN, bm):
        qi = torch.arange(Q_LEN, device="cuda").unsqueeze(1)
        ki = torch.arange(KV_LEN, device="cuda").unsqueeze(0)
        b = torch.zeros_like(qi)
        h = torch.zeros_like(qi)
        return bm.mask_mod(b, h, qi, ki).bool()

    def test_dispatcher_picks_analytical_path(self):
        """When Q_LEN aligned & KV_LEN % Q_LEN == 0, output must equal analytical builder."""
        Q_LEN, KV_LEN = 256, 256 * 3
        dispatched = eagle3_block_mask(Q_LEN, KV_LEN, B=1, H=1, device="cuda")
        analytical = build_eagle3_block_mask(Q_LEN, KV_LEN, B=1, H=1, device="cuda")
        # Same kv_indices/q_indices structure indicates analytical path was taken.
        self.assertTrue(torch.equal(dispatched.kv_num_blocks, analytical.kv_num_blocks))
        self.assertTrue(torch.equal(dispatched.kv_indices, analytical.kv_indices))
        self.assertTrue(torch.equal(dispatched.q_num_blocks, analytical.q_num_blocks))
        self.assertTrue(torch.equal(dispatched.q_indices, analytical.q_indices))

    def test_dispatcher_first_round_uses_analytical(self):
        """First round (KV_LEN == Q_LEN) is now handled by the analytical builder."""
        Q_LEN = 256
        dispatched = eagle3_block_mask(Q_LEN, Q_LEN, B=1, H=1, device="cuda")
        analytical = build_eagle3_block_mask(Q_LEN, Q_LEN, B=1, H=1, device="cuda")
        self.assertTrue(torch.equal(dispatched.kv_indices, analytical.kv_indices))

    def test_dispatcher_falls_back_when_q_too_small(self):
        """When Q_LEN < BLOCK_SIZE, must fall back to create_block_mask without raising."""
        Q_LEN, KV_LEN = 64, 64
        bm = eagle3_block_mask(Q_LEN, KV_LEN, B=1, H=1, device="cuda")
        # Should produce a causal-only mask and not raise.
        dense = self._ref_dense(Q_LEN, KV_LEN, bm)
        expected = torch.tril(torch.ones(Q_LEN, KV_LEN, device="cuda", dtype=torch.bool))
        self.assertTrue(torch.equal(dense, expected))

    def test_dispatcher_falls_back_when_kv_not_round_multiple(self):
        """When KV_LEN % Q_LEN != 0, must fall back rather than raising the assert."""
        # 256 and 384 are both multiples of 128 but 384 % 256 != 0, so analytical
        # path's diagonal layout would be wrong: dispatcher must take the fallback.
        Q_LEN, KV_LEN = 256, 384
        bm = eagle3_block_mask(Q_LEN, KV_LEN, B=1, H=1, device="cuda")
        # Mask correctness is verified element-wise against the canonical mask_mod.
        seq_lengths = torch.tensor([Q_LEN], device="cuda", dtype=torch.int32)
        ref_mod = generate_eagle3_mask(seq_lengths, Q_LEN, KV_LEN, lck=0)
        qi = torch.arange(Q_LEN, device="cuda").unsqueeze(1)
        ki = torch.arange(KV_LEN, device="cuda").unsqueeze(0)
        b = torch.zeros_like(qi)
        h = torch.zeros_like(qi)
        expected = ref_mod(b, h, qi, ki).bool()
        actual = bm.mask_mod(b, h, qi, ki).bool()
        self.assertTrue(torch.equal(actual, expected))

    def test_dispatcher_flex_attention_matches_reference(self):
        """flex_attention output via dispatcher must match the create_block_mask reference."""
        torch.manual_seed(42)
        B, H, D = 1, 4, 64
        for Q_LEN, KV_LEN in [(256, 256), (256, 256 * 3), (512, 512 * 4)]:
            with self.subTest(Q_LEN=Q_LEN, KV_LEN=KV_LEN):
                q = torch.randn(B, H, Q_LEN, D, device="cuda", dtype=torch.bfloat16)
                k = torch.randn(B, H, KV_LEN, D, device="cuda", dtype=torch.bfloat16)
                v = torch.randn(B, H, KV_LEN, D, device="cuda", dtype=torch.bfloat16)

                seq_lengths = torch.tensor([Q_LEN] * B, device="cuda", dtype=torch.int32)
                ref_bm = create_block_mask(
                    generate_eagle3_mask(seq_lengths, Q_LEN, KV_LEN, lck=0),
                    B=B, H=1, Q_LEN=Q_LEN, KV_LEN=KV_LEN, device="cuda",
                )
                dispatched_bm = eagle3_block_mask(
                    Q_LEN, KV_LEN, B=B, H=1, device="cuda",
                )

                out_ref = flex_attention(q, k, v, block_mask=ref_bm, enable_gqa=False)
                out_disp = flex_attention(q, k, v, block_mask=dispatched_bm, enable_gqa=False)
                self.assertEqual(out_ref.shape, out_disp.shape)
                max_diff = (out_ref - out_disp).abs().max().item()
                self.assertAlmostEqual(max_diff, 0.0, places=5,
                    msg=f"Output diff={max_diff} at Q={Q_LEN}, KV={KV_LEN}")

    def test_dispatcher_seq_lengths_optional(self):
        """seq_lengths is optional; analytical path ignores it, fallback synthesises a default."""
        # Analytical path: no seq_lengths required.
        bm1 = eagle3_block_mask(256, 768, B=1, H=1, device="cuda")
        self.assertEqual(bm1.kv_indices.shape[0], 1)
        # Fallback path: omitted seq_lengths must not raise.
        bm2 = eagle3_block_mask(64, 64, B=1, H=1, device="cuda")
        self.assertIsNotNone(bm2)


if __name__ == "__main__":
    unittest.main(verbosity=2)
