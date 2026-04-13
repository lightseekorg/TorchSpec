"""Unit tests for slot-mapping and chunked prefill in MooncakeHiddenStatesConnector.

Tests _slot_mapping_from_block_ids (the sole slot-mapping strategy, matching
upstream's design of always computing from block_ids) and _extract_from_kv_cache.
"""

import pytest
import torch

from torchspec.inference.engine.mooncake_hidden_states_connector import (
    _extract_from_kv_cache,
    _slot_mapping_from_block_ids,
)


class TestSlotMappingFromBlockIds:
    def test_single_block(self):
        result = _slot_mapping_from_block_ids(
            block_ids=[5],
            page_size=4,
            num_tokens=4,
            device=torch.device("cpu"),
        )
        expected = torch.tensor([20, 21, 22, 23], dtype=torch.int64)
        assert torch.equal(result, expected)

    def test_multiple_blocks(self):
        result = _slot_mapping_from_block_ids(
            block_ids=[2, 7],
            page_size=3,
            num_tokens=6,
            device=torch.device("cpu"),
        )
        expected = torch.tensor([6, 7, 8, 21, 22, 23], dtype=torch.int64)
        assert torch.equal(result, expected)

    def test_partial_last_block(self):
        result = _slot_mapping_from_block_ids(
            block_ids=[0, 1],
            page_size=4,
            num_tokens=5,
            device=torch.device("cpu"),
        )
        expected = torch.tensor([0, 1, 2, 3, 4], dtype=torch.int64)
        assert torch.equal(result, expected)

    def test_non_contiguous_blocks(self):
        result = _slot_mapping_from_block_ids(
            block_ids=[10, 3],
            page_size=2,
            num_tokens=4,
            device=torch.device("cpu"),
        )
        expected = torch.tensor([20, 21, 6, 7], dtype=torch.int64)
        assert torch.equal(result, expected)

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
    def test_gpu_device(self):
        result = _slot_mapping_from_block_ids(
            block_ids=[1, 2],
            page_size=3,
            num_tokens=5,
            device=torch.device("cuda:0"),
        )
        assert result.device.type == "cuda"
        expected = torch.tensor([3, 4, 5, 6, 7], dtype=torch.int64, device="cuda:0")
        assert torch.equal(result, expected)

    def test_chunked_prefill_partial_skip(self):
        """With fewer blocks than needed, num_slots < num_tokens — the
        chunked prefill guard in save_kv_layer should skip this request."""
        block_ids = [0, 1]
        page_size = 16
        num_tokens = 200
        slot_mapping = _slot_mapping_from_block_ids(
            block_ids,
            page_size,
            num_tokens,
            device=torch.device("cpu"),
        )
        assert slot_mapping.shape[0] < num_tokens

    def test_chunked_prefill_full_store(self):
        """With enough blocks, num_slots >= num_tokens — store proceeds."""
        block_ids = list(range(13))
        page_size = 16
        num_tokens = 200
        slot_mapping = _slot_mapping_from_block_ids(
            block_ids,
            page_size,
            num_tokens,
            device=torch.device("cpu"),
        )
        assert slot_mapping.shape[0] >= num_tokens

    def test_hma_page_size_vs_block_size(self):
        """On HMA models, page_size (from tensor shape) differs from
        cache_config.block_size.  Using the tensor's page_size produces
        indices within the KV cache's flat size."""
        block_ids = [3, 7]
        actual_page_size = 16
        wrong_block_size = 1056
        num_tokens = 20

        correct = _slot_mapping_from_block_ids(
            block_ids,
            actual_page_size,
            num_tokens,
            device=torch.device("cpu"),
        )
        assert correct.max().item() < 200

        wrong = _slot_mapping_from_block_ids(
            block_ids,
            wrong_block_size,
            num_tokens,
            device=torch.device("cpu"),
        )
        assert wrong.max().item() > 3000


class TestExtractFromKvCache:
    def test_basic_extraction(self):
        num_pages, page_size, num_heads, head_size = 4, 3, 2, 4
        kv_cache = torch.arange(num_pages * page_size * num_heads * head_size, dtype=torch.float32)
        kv_cache = kv_cache.view(num_pages, page_size, num_heads, head_size)

        slot_mapping = torch.tensor([0, 1, 5], dtype=torch.int64)
        result = _extract_from_kv_cache(kv_cache, slot_mapping, num_tokens=3)

        flat = kv_cache.flatten(0, 1)
        expected = flat[slot_mapping][:3]
        assert torch.equal(result, expected)

    def test_with_block_ids_mapping(self):
        """End-to-end: block_ids -> slot_mapping -> extract."""
        num_pages, page_size, num_heads, head_size = 10, 4, 2, 8
        kv_cache = torch.randn(num_pages, page_size, num_heads, head_size)

        block_ids = [3, 7]
        num_tokens = 6
        slot_mapping = _slot_mapping_from_block_ids(
            block_ids,
            page_size,
            num_tokens,
            device=torch.device("cpu"),
        )
        result = _extract_from_kv_cache(kv_cache, slot_mapping, num_tokens)
        assert result.shape == (num_tokens, num_heads, head_size)

        flat = kv_cache.flatten(0, 1)
        assert torch.equal(result[0], flat[3 * 4 + 0])
        assert torch.equal(result[3], flat[3 * 4 + 3])
        assert torch.equal(result[4], flat[7 * 4 + 0])
        assert torch.equal(result[5], flat[7 * 4 + 1])


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
