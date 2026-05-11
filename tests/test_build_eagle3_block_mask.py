"""Tests for build_eagle3_block_mask -- the analytical Eagle3 BlockMask builder."""

import unittest

import torch
import torch._dynamo as dynamo
from torch.nn.attention.flex_attention import create_block_mask, flex_attention

from torchspec.models.ops.flex_attention import (
    _build_eagle3_block_mask_tensors,
    build_eagle3_block_mask,
    eagle3_block_mask,
    generate_eagle3_mask,
)

DEVICE = "cuda"
BLOCK_SIZE = 128


def dense_from_mod(Q_LEN, KV_LEN, mask_mod, batch_idx=0):
    """Materialise a (Q_LEN, KV_LEN) bool grid from a mask_mod or BlockMask."""
    qi = torch.arange(Q_LEN, device=DEVICE).unsqueeze(1)
    ki = torch.arange(KV_LEN, device=DEVICE).unsqueeze(0)
    b = torch.full_like(qi, batch_idx)
    h = torch.zeros_like(qi)
    fn = mask_mod.mask_mod if hasattr(mask_mod, "mask_mod") else mask_mod
    return fn(b, h, qi, ki).bool()


def reference_block_mask(Q_LEN, KV_LEN, B=1, H=1):
    """create_block_mask using the production simplified mask_mod."""
    return create_block_mask(
        generate_eagle3_mask(Q_LEN, KV_LEN),
        B=B,
        H=H,
        Q_LEN=Q_LEN,
        KV_LEN=KV_LEN,
        device=DEVICE,
    )


# Sizes covering single round, short-multi-round, and aligned-multi-round cases.
SHAPES = [(256, 256), (256, 768), (256, 1280), (1024, 4096)]


class TestBuildEagle3BlockMask(unittest.TestCase):
    """Analytical builder must produce a mask equivalent to create_block_mask."""

    def test_dense_mask_matches_reference(self):
        for Q, KV in SHAPES:
            with self.subTest(Q=Q, KV=KV):
                ref = dense_from_mod(Q, KV, reference_block_mask(Q, KV))
                ours = dense_from_mod(Q, KV, build_eagle3_block_mask(Q, KV, device=DEVICE))
                self.assertTrue(torch.equal(ref, ours))

    def test_forward_matches_reference(self):
        torch.manual_seed(42)
        B, H, D = 1, 4, 64
        for Q, KV in SHAPES:
            with self.subTest(Q=Q, KV=KV):
                q = torch.randn(B, H, Q, D, device=DEVICE, dtype=torch.bfloat16)
                k = torch.randn(B, H, KV, D, device=DEVICE, dtype=torch.bfloat16)
                v = torch.randn(B, H, KV, D, device=DEVICE, dtype=torch.bfloat16)
                ref = flex_attention(q, k, v, block_mask=reference_block_mask(Q, KV))
                ours = flex_attention(q, k, v, block_mask=build_eagle3_block_mask(Q, KV, B=B))
                self.assertEqual(ref.shape, ours.shape)
                self.assertFalse(ours.isnan().any())
                self.assertLess((ref - ours).abs().max().item(), 1e-5)

    def test_backward_gradients_match_reference(self):
        torch.manual_seed(42)
        B, H, D, Q, KV = 1, 4, 64, 256, 768

        def grads(mask):
            q = torch.randn(B, H, Q, D, device=DEVICE, dtype=torch.float32, requires_grad=True)
            k = torch.randn(B, H, KV, D, device=DEVICE, dtype=torch.float32, requires_grad=True)
            v = torch.randn(B, H, KV, D, device=DEVICE, dtype=torch.float32, requires_grad=True)
            flex_attention(q, k, v, block_mask=mask).sum().backward()
            return q.grad, k.grad, v.grad

        torch.manual_seed(42)
        gq_r, gk_r, gv_r = grads(reference_block_mask(Q, KV))
        torch.manual_seed(42)
        gq_o, gk_o, gv_o = grads(build_eagle3_block_mask(Q, KV, B=B))
        for name, gr, go in [("q", gq_r, gq_o), ("k", gk_r, gk_o), ("v", gv_r, gv_o)]:
            self.assertLess((gr - go).abs().max().item(), 1e-4, f"grad mismatch on {name}")

    def test_gqa_broadcast(self):
        """H=1 mask broadcasts over multi-Q-head GQA without NaN."""
        torch.manual_seed(0)
        B, Qh, KVh, D, Q, KV = 1, 8, 2, 64, 256, 768
        q = torch.randn(B, Qh, Q, D, device=DEVICE, dtype=torch.bfloat16)
        k = torch.randn(B, KVh, KV, D, device=DEVICE, dtype=torch.bfloat16)
        v = torch.randn(B, KVh, KV, D, device=DEVICE, dtype=torch.bfloat16)
        bm = build_eagle3_block_mask(Q, KV, B=B, device=DEVICE)
        out = flex_attention(q, k, v, block_mask=bm, enable_gqa=True)
        self.assertEqual(out.shape, (B, Qh, Q, D))
        self.assertFalse(out.isnan().any())

    def test_memory_is_negligible(self):
        """Original create_block_mask costs ~112 GB at Q=49K; this must stay in MB."""
        Q, KV = 4096, 4096 * 5
        torch.cuda.reset_peak_memory_stats()
        before = torch.cuda.memory_allocated()
        build_eagle3_block_mask(Q, KV, device=DEVICE)
        mem_mb = (torch.cuda.max_memory_allocated() - before) / 1024**2
        self.assertLess(mem_mb, 10.0, f"used {mem_mb:.1f} MB")

    def test_assertions_on_invalid_shapes(self):
        # not divisible by BLOCK_SIZE
        with self.assertRaises(AssertionError):
            build_eagle3_block_mask(100, 300, device=DEVICE)
        # KV not a Q-multiple
        with self.assertRaises(AssertionError):
            build_eagle3_block_mask(256, 384, device=DEVICE)

    def _visited_blocks(self, num, idx):
        """Return set of (row, col) block pairs from a (B,H,rows,W) num/idx pair."""
        rows = num.shape[-1]
        seen = set()
        for r in range(rows):
            for c in range(num[0, 0, r].item()):
                seen.add((r, idx[0, 0, r, c].item()))
        return seen

    def test_full_and_partial_blocks_match_create_block_mask(self):
        """Analytical builder must emit the same {full,partial} block split as
        create_block_mask -- this is what FA4 BlockSparseTensorsTorch consumes."""
        for Q, KV, BS in [
            (256, 256, 128),
            (256, 768, 128),
            (1024, 4096, 128),
            # FA4 Blackwell SM100+ uses Q_BS=256, KV_BS=128.
            (512, 1024, (256, 128)),
            (1024, 4096, (256, 128)),
        ]:
            with self.subTest(Q=Q, KV=KV, BS=BS):
                ref = create_block_mask(
                    generate_eagle3_mask(Q, KV),
                    B=1,
                    H=1,
                    Q_LEN=Q,
                    KV_LEN=KV,
                    device=DEVICE,
                    BLOCK_SIZE=BS,
                )
                ours = build_eagle3_block_mask(Q, KV, B=1, H=1, device=DEVICE, BLOCK_SIZE=BS)
                # KV-direction (forward iteration).
                self.assertEqual(
                    self._visited_blocks(ref.kv_num_blocks, ref.kv_indices),
                    self._visited_blocks(ours.kv_num_blocks, ours.kv_indices),
                )
                self.assertEqual(
                    self._visited_blocks(ref.full_kv_num_blocks, ref.full_kv_indices),
                    self._visited_blocks(ours.full_kv_num_blocks, ours.full_kv_indices),
                )
                # Q-direction (backward iteration).
                self.assertEqual(
                    self._visited_blocks(ref.q_num_blocks, ref.q_indices),
                    self._visited_blocks(ours.q_num_blocks, ours.q_indices),
                )
                self.assertEqual(
                    self._visited_blocks(ref.full_q_num_blocks, ref.full_q_indices),
                    self._visited_blocks(ours.full_q_num_blocks, ours.full_q_indices),
                )

    def test_non_square_block_size_forward(self):
        """Q_BS != KV_BS (Blackwell case): output must match reference."""
        torch.manual_seed(123)
        B, H, D = 1, 4, 64
        for Q, KV, BS in [(512, 1024, (256, 128)), (1024, 4096, (256, 128))]:
            with self.subTest(Q=Q, KV=KV, BS=BS):
                q = torch.randn(B, H, Q, D, device=DEVICE, dtype=torch.bfloat16)
                k = torch.randn(B, H, KV, D, device=DEVICE, dtype=torch.bfloat16)
                v = torch.randn(B, H, KV, D, device=DEVICE, dtype=torch.bfloat16)
                ref = flex_attention(
                    q,
                    k,
                    v,
                    block_mask=create_block_mask(
                        generate_eagle3_mask(Q, KV),
                        B=1,
                        H=1,
                        Q_LEN=Q,
                        KV_LEN=KV,
                        device=DEVICE,
                        BLOCK_SIZE=BS,
                    ),
                )
                ours = flex_attention(
                    q,
                    k,
                    v,
                    block_mask=build_eagle3_block_mask(
                        Q, KV, B=B, H=1, device=DEVICE, BLOCK_SIZE=BS
                    ),
                )
                self.assertEqual(ref.shape, ours.shape)
                self.assertFalse(ours.isnan().any())
                self.assertLess((ref - ours).abs().max().item(), 1e-5)


class TestEagle3BlockMaskDispatcher(unittest.TestCase):
    """Dispatcher picks analytical when shapes align, otherwise falls back."""

    def test_analytical_path_when_aligned(self):
        for Q, KV in [(256, 256), (256, 768)]:
            with self.subTest(Q=Q, KV=KV):
                disp = eagle3_block_mask(Q, KV, B=1, H=1, device=DEVICE)
                ana = build_eagle3_block_mask(Q, KV, device=DEVICE)
                self.assertTrue(torch.equal(disp.kv_indices, ana.kv_indices))
                self.assertTrue(torch.equal(disp.q_indices, ana.q_indices))
                self.assertTrue(torch.equal(disp.full_kv_indices, ana.full_kv_indices))
                self.assertTrue(torch.equal(disp.full_q_indices, ana.full_q_indices))

    def test_dispatcher_with_non_square_block_size(self):
        """Tuple BLOCK_SIZE (e.g. FA4 Blackwell 256x128) routes through analytical path."""
        Q, KV, BS = 512, 2048, (256, 128)
        disp = eagle3_block_mask(Q, KV, B=1, H=1, device=DEVICE, BLOCK_SIZE=BS)
        ana = build_eagle3_block_mask(Q, KV, B=1, H=1, device=DEVICE, BLOCK_SIZE=BS)
        self.assertEqual(disp.BLOCK_SIZE, (256, 128))
        self.assertTrue(torch.equal(disp.kv_indices, ana.kv_indices))
        self.assertTrue(torch.equal(disp.full_kv_indices, ana.full_kv_indices))

    def test_fallback_path_matches_reference_mask_mod(self):
        """Fallback shapes (Q<BLOCK_SIZE, or KV%Q!=0) must produce the canonical mask."""
        for Q, KV in [(64, 64), (256, 384)]:
            with self.subTest(Q=Q, KV=KV):
                bm = eagle3_block_mask(Q, KV, B=1, H=1, device=DEVICE)
                expected = dense_from_mod(Q, KV, generate_eagle3_mask(Q, KV))
                self.assertTrue(torch.equal(dense_from_mod(Q, KV, bm), expected))

    def test_dispatcher_forward_matches_reference(self):
        torch.manual_seed(0)
        B, H, D = 1, 4, 64
        for Q, KV in [(256, 256), (256, 768)]:
            with self.subTest(Q=Q, KV=KV):
                q = torch.randn(B, H, Q, D, device=DEVICE, dtype=torch.bfloat16)
                k = torch.randn(B, H, KV, D, device=DEVICE, dtype=torch.bfloat16)
                v = torch.randn(B, H, KV, D, device=DEVICE, dtype=torch.bfloat16)
                ref = flex_attention(q, k, v, block_mask=reference_block_mask(Q, KV, B=B))
                disp = flex_attention(
                    q, k, v, block_mask=eagle3_block_mask(Q, KV, B=B, H=1, device=DEVICE)
                )
                self.assertLess((ref - disp).abs().max().item(), 1e-5)


class TestCompiledTensorBuilder(unittest.TestCase):
    """build_eagle3_block_mask routes through torch.compile -- verify behaviour."""

    def test_compiled_output_matches_eager(self):
        for Q, KV in [(256, 256), (256, 768), (1024, 4096)]:
            with self.subTest(Q=Q, KV=KV):
                eager = _build_eagle3_block_mask_tensors(
                    Q, KV, 1, 1, BLOCK_SIZE, BLOCK_SIZE, DEVICE
                )
                (
                    e_kv_num,
                    e_kv_idx,
                    e_full_kv_num,
                    e_full_kv_idx,
                    e_q_num,
                    e_q_idx,
                    e_full_q_num,
                    e_full_q_idx,
                ) = eager
                bm = build_eagle3_block_mask(Q, KV, device=DEVICE)
                self.assertTrue(torch.equal(bm.kv_num_blocks, e_kv_num))
                self.assertTrue(torch.equal(bm.kv_indices, e_kv_idx))
                self.assertTrue(torch.equal(bm.full_kv_num_blocks, e_full_kv_num))
                self.assertTrue(torch.equal(bm.full_kv_indices, e_full_kv_idx))
                self.assertTrue(torch.equal(bm.q_num_blocks, e_q_num))
                self.assertTrue(torch.equal(bm.q_indices, e_q_idx))
                self.assertTrue(torch.equal(bm.full_q_num_blocks, e_full_q_num))
                self.assertTrue(torch.equal(bm.full_q_indices, e_full_q_idx))

    def test_dynamic_true_does_not_recompile_across_growing_kv(self):
        """KV_LEN grows by Q_LEN every TTT step; dynamic=True must keep one graph."""
        Q = 512
        # Warm up to lock the compiled artifact.
        build_eagle3_block_mask(Q, Q, device=DEVICE)
        dynamo.reset()
        build_eagle3_block_mask(Q, Q, device=DEVICE)
        before = dynamo.utils.counters["stats"].get("unique_graphs", 0)
        for n_rounds in [2, 3, 4, 5]:
            build_eagle3_block_mask(Q, Q * n_rounds, device=DEVICE)
        after = dynamo.utils.counters["stats"].get("unique_graphs", 0)
        # First call after dynamo.reset() compiles once (+1); growing KV must not add more.
        self.assertLessEqual(
            after - before,
            1,
            f"dynamic=True triggered {after - before} extra graphs across growing KV",
        )


if __name__ == "__main__":
    unittest.main(verbosity=2)
