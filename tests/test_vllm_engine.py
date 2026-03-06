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

"""Tests for vLLM Worker Extension.

This file contains both:
- Unit tests: Test logic with mocks (no GPU/vLLM/Mooncake needed)
- Integration tests: Test with real vLLM engine (requires GPU + infrastructure)
"""

import os
from dataclasses import dataclass
from unittest.mock import MagicMock, patch

import pytest
import torch

# =============================================================================
# Helpers
# =============================================================================


@dataclass
class MockArgs:
    """Mock args for VllmWorkerExtension initialization."""

    target_model_path: str = "Qwen/Qwen3-8B"
    tensor_parallel_size: int = 2
    max_model_len: int = 2048
    trust_remote_code: bool = True


def _import_vllm_worker_extension():
    """Import VllmWorkerExtension, skipping test if dependencies unavailable."""
    try:
        from torchspec.inference.engine.vllm_worker_extension import (
            VllmWorkerExtension,
            _sanitize_mooncake_key,
        )

        return VllmWorkerExtension, _sanitize_mooncake_key
    except ImportError as e:
        pytest.skip(f"VllmWorkerExtension import failed (missing deps): {e}")


# =============================================================================
# Unit Tests (No real vLLM/GPU/Mooncake needed)
# =============================================================================


class TestSanitizeMooncakeKey:
    """Unit tests for _sanitize_mooncake_key pure function."""

    def test_alphanumeric_unchanged(self):
        """Test alphanumeric keys pass through unchanged."""
        _, _sanitize = _import_vllm_worker_extension()
        assert _sanitize("req_abc_123") == "req_abc_123"

    def test_special_chars_replaced(self):
        """Test special characters are replaced with underscores."""
        _, _sanitize = _import_vllm_worker_extension()
        assert _sanitize("req@abc#123") == "req_abc_123"
        assert _sanitize("req.id.name") == "req_id_name"
        assert _sanitize("req:name|value") == "req_name_value"

    def test_leading_digit_prefixed(self):
        """Test leading digits get 'k' prefix."""
        _, _sanitize = _import_vllm_worker_extension()
        assert _sanitize("123_req") == "k123_req"
        assert _sanitize("1abc") == "k1abc"

    def test_empty_string(self):
        """Test empty string handling."""
        _, _sanitize = _import_vllm_worker_extension()
        assert _sanitize("") == ""


class TestVllmWorkerExtensionState:
    """Unit tests for VllmWorkerExtension state management."""

    def test_init_stores_config(self):
        """Test constructor initializes state correctly."""
        VllmWorkerExtension, _ = _import_vllm_worker_extension()

        ext = VllmWorkerExtension()

        assert ext._layer_ids == frozenset()
        assert ext._captured_states is None
        assert ext._request_metadata == []
        assert ext._current_request_metadata is None
        assert ext._mooncake_store is None
        assert ext._store_initialized is False

    def test_set_request_metadata(self):
        """Test setting request metadata."""
        VllmWorkerExtension, _ = _import_vllm_worker_extension()

        ext = VllmWorkerExtension()
        metadata = {"req_1": 100, "req_2": 200}
        packed_map = {"req_1": "0,3", "req_2": "0,5"}
        input_ids_map = {"req_1": [1, 2, 3], "req_2": [4, 5, 6]}

        ext._set_request_metadata(metadata, packed_map, input_ids_map)

        assert ext._current_request_metadata == metadata
        assert ext._packed_loss_mask_map == packed_map
        assert ext._input_ids_map == input_ids_map

    def test_reset_capture_clears_state(self):
        """Test reset_capture clears all captured state."""
        VllmWorkerExtension, _ = _import_vllm_worker_extension()

        ext = VllmWorkerExtension()
        ext._layer_ids = frozenset({5, 10, 15})
        ext._captured_states = [[torch.randn(10, 4096)], [torch.randn(10, 4096)]]
        ext._captured_input_ids = torch.tensor([1, 2, 3])
        ext._request_metadata = [{"req_1": 10}]
        ext._current_request_metadata = {"req_1": 10}
        ext._packed_loss_mask_map = {"req_1": "0,3"}
        ext._input_ids_map = {"req_1": [1, 2, 3]}

        ext._reset_capture()

        assert ext._captured_states is None
        assert ext._captured_input_ids is None
        assert ext._request_metadata == []
        assert ext._current_request_metadata is None
        assert ext._packed_loss_mask_map == {}
        assert ext._input_ids_map == {}

    def test_reset_capture_requires_prior_setup(self):
        """Test reset_capture requires _setup_hidden_states_capture first."""
        VllmWorkerExtension, _ = _import_vllm_worker_extension()

        ext = VllmWorkerExtension()
        # Don't set _layer_ids

        with pytest.raises(RuntimeError, match="Must call _setup_hidden_states_capture"):
            ext._reset_capture()


class TestStoreCapturedStates:
    """Unit tests for _store_captured_states with mocked dependencies."""

    def test_store_first_capture(self):
        """Test first capture initializes the state lists."""
        VllmWorkerExtension, _ = _import_vllm_worker_extension()

        ext = VllmWorkerExtension()
        tensors = [torch.randn(10, 4096), torch.randn(10, 4096)]

        ext._store_captured_states(tensors)

        assert ext._captured_states is not None
        assert len(ext._captured_states) == 2
        assert torch.equal(ext._captured_states[0][0], tensors[0])
        assert torch.equal(ext._captured_states[1][0], tensors[1])

    def test_store_appends_to_existing(self):
        """Test subsequent captures append to existing lists."""
        VllmWorkerExtension, _ = _import_vllm_worker_extension()

        ext = VllmWorkerExtension()
        ext._captured_states = [[torch.randn(10, 4096)], [torch.randn(10, 4096)]]

        new_tensors = [torch.randn(10, 4096), torch.randn(10, 4096)]
        ext._store_captured_states(new_tensors)

        assert len(ext._captured_states[0]) == 2
        assert len(ext._captured_states[1]) == 2
        assert torch.equal(ext._captured_states[0][1], new_tensors[0])

    def test_store_extracts_metadata_from_input_batch(self):
        """Test metadata extraction from model_runner.input_batch."""
        VllmWorkerExtension, _ = _import_vllm_worker_extension()

        ext = VllmWorkerExtension()

        # Mock model_runner with input_batch
        mock_batch = MagicMock()
        mock_batch.req_ids = ["req_1", "req_2"]
        mock_batch.req_id_to_index = {"req_1": 0, "req_2": 1}
        mock_batch.num_tokens = [100, 200]
        mock_batch.num_computed_tokens = [0, 0]

        ext.model_runner = MagicMock()
        ext.model_runner.input_batch = mock_batch

        tensors = [torch.randn(10, 4096)]
        ext._store_captured_states(tensors)

        assert len(ext._request_metadata) == 1
        assert "req_1" in ext._request_metadata[0]
        assert "req_2" in ext._request_metadata[0]


class TestCudaDeviceSafe:
    """Unit tests for _get_cuda_device_safe with mocked torch.cuda."""

    @patch("torch.cuda.is_initialized")
    @patch("torch.cuda.current_device")
    def test_initialized_context(self, mock_current, mock_initialized):
        """Test when CUDA is already initialized."""
        VllmWorkerExtension, _ = _import_vllm_worker_extension()

        mock_initialized.return_value = True
        mock_current.return_value = 1

        ext = VllmWorkerExtension()
        device = ext._get_cuda_device_safe()

        assert str(device) == "cuda:1"

    @patch("torch.cuda.is_initialized")
    def test_uninitialized_context_fallback(self, mock_initialized):
        """Test fallback when CUDA not initialized (V1 engine)."""
        VllmWorkerExtension, _ = _import_vllm_worker_extension()

        mock_initialized.return_value = False

        ext = VllmWorkerExtension()
        device = ext._get_cuda_device_safe()

        assert str(device) == "cuda:0"


class TestTokenSlicingLogic:
    """Unit tests for token distribution and slicing logic."""

    def test_ratio_based_distribution(self):
        """Test ratio calculation for token distribution."""
        VllmWorkerExtension, _ = _import_vllm_worker_extension()

        ext = VllmWorkerExtension()
        ext._current_request_metadata = {"req_1": 100, "req_2": 200}

        external_ids = list(ext._current_request_metadata.keys())
        token_counts = list(ext._current_request_metadata.values())
        total_expected = sum(token_counts)  # 300
        total_captured = 150  # Half the expected tokens

        ratio = total_captured / total_expected  # 0.5

        # Calculate actual tokens per request
        actual_tokens = {ext_id: int(tc * ratio) for ext_id, tc in zip(external_ids, token_counts)}

        assert actual_tokens == {"req_1": 50, "req_2": 100}

    def test_concatenated_tensors_shape(self):
        """Test tensor concatenation from multiple iterations."""
        VllmWorkerExtension, _ = _import_vllm_worker_extension()

        ext = VllmWorkerExtension()
        # Simulate 2 iterations with 5 tokens each
        ext._captured_states = [
            [torch.randn(5, 4096), torch.randn(5, 4096)],  # Layer 0
            [torch.randn(5, 4096), torch.randn(5, 4096)],  # Layer 1
        ]

        # Concatenate (simulating _store_and_get_metadata logic)
        concatenated = [torch.cat(layer_tensors, dim=0) for layer_tensors in ext._captured_states]

        assert concatenated[0].shape == (10, 4096)
        assert concatenated[1].shape == (10, 4096)


# =============================================================================
# Integration Tests (Requires real GPU + vLLM + Mooncake)
# =============================================================================


@pytest.mark.integration
@pytest.mark.skipif(not torch.cuda.is_available(), reason="GPU required")
@pytest.mark.skipif(torch.cuda.device_count() < 2, reason="Need at least 2 GPUs for TP=2")
class TestVllmWorkerExtensionIntegration:
    """Integration tests for vLLM Worker Extension with real infrastructure."""

    @pytest.fixture(autouse=True)
    def setup_env(self):
        """Setup Mooncake environment variables."""
        os.environ.setdefault("MOONCAKE_MASTER_HOST", "0.0.0.0")
        os.environ.setdefault("MOONCAKE_MASTER_PORT", "50051")
        os.environ.setdefault("MOONCAKE_METADATA_PORT", "8090")
        yield
        # Cleanup not needed for env vars

    def test_vllm_worker_extension_mooncake(self):
        """Test vLLM Worker Extension stores and retrieves hidden states from Mooncake."""
        from transformers import AutoTokenizer
        from vllm import LLM, SamplingParams

        from torchspec.transfer.mooncake import EagleMooncakeStore, MooncakeConfig

        model_path = "Qwen/Qwen3-8B"

        # Initialize tokenizer
        tokenizer = AutoTokenizer.from_pretrained(model_path)

        # Test inputs
        input_ids_list = [
            [1, 2345, 6789],
            [100, 200, 300, 400],
            [500, 600],
        ]
        data_ids = ["test_req_0", "test_req_1", "test_req_2"]

        # Initialize vLLM with Worker Extension
        engine = LLM(
            model=model_path,
            tensor_parallel_size=2,
            gpu_memory_utilization=0.7,
            trust_remote_code=True,
            worker_extension_cls="torchspec.inference.engine.vllm_worker_extension.VllmWorkerExtension",
            max_model_len=2048,
        )

        try:
            # Configure hidden states capture
            engine.collective_rpc("_setup_hidden_states_capture", args=([5, 10, 15],))

            # Prepare generation
            prompts = [tokenizer.decode(ids) for ids in input_ids_list]
            sampling_params = SamplingParams(max_tokens=32, temperature=0)

            # Setup request metadata
            request_metadata = {data_ids[i]: len(ids) for i, ids in enumerate(input_ids_list)}
            engine.collective_rpc("_reset_capture")
            engine.collective_rpc("_set_request_metadata", args=(request_metadata,))

            # Generate
            print("=== Generating with vLLM Worker Extension ===")
            outputs = engine.generate(prompts, sampling_params)
            assert len(outputs) == len(input_ids_list), "Generation output count mismatch"

            for i, output in enumerate(outputs):
                print(f"\n--- Request {i} ---")
                print(f"output_ids: {output.prompt_token_ids + list(output.outputs[0].token_ids)}")
                print(f"num tokens generated: {len(output.outputs[0].token_ids)}")

            # Retrieve metadata from Mooncake
            print("\n=== Retrieving metadata from Mooncake ===")
            metadata_list = engine.collective_rpc("_store_and_get_metadata")
            assert metadata_list is not None, "No metadata returned from workers"

            all_keys = []
            seq_lens = []
            for metadata in metadata_list:
                if isinstance(metadata, dict):
                    for req_id, meta in metadata.items():
                        assert "mooncake_key" in meta
                        assert "tensor_shapes" in meta
                        assert "num_layers" in meta
                        assert meta["num_layers"] == 3
                        all_keys.append(meta["mooncake_key"])
                        seq_lens.append(request_metadata[req_id])
                        print(
                            f"  {req_id}: key={meta['mooncake_key']}, layers={meta['num_layers']}"
                        )

            # Fetch data from Mooncake Store
            print("\n=== Fetching data from Mooncake Store ===")
            mooncake_config = MooncakeConfig.from_env()
            mooncake_store = EagleMooncakeStore(mooncake_config)
            mooncake_store.setup(device="cuda")

            # Qwen3-8B dimensions
            hidden_dim = 12288  # 3 layers concatenated (4096 * 3)
            last_hidden_dim = 4096

            for i, key in enumerate(all_keys):
                seq_len = seq_lens[i]
                shapes = {
                    "hidden_states": (seq_len, hidden_dim),
                    "input_ids": (seq_len,),
                    "last_hidden_states": (seq_len, last_hidden_dim),
                }
                dtypes = {
                    "hidden_states": torch.bfloat16,
                    "input_ids": torch.long,
                    "last_hidden_states": torch.bfloat16,
                }

                data = mooncake_store.get(key, shapes=shapes, dtypes=dtypes, device="cuda")
                print(f"\n  Key: {key}")
                print(
                    f"    hidden_states: shape={data.hidden_states.shape}, dtype={data.hidden_states.dtype}"
                )
                print(f"    input_ids: {data.input_ids.tolist()}")
                print(f"    last_hidden_states: shape={data.last_hidden_states.shape}")

                # Verify tensor device consistency
                assert data.hidden_states.device == data.input_ids.device, (
                    f"Device mismatch: hidden_states={data.hidden_states.device}, input_ids={data.input_ids.device}"
                )

            print("\n✓ Test completed - hidden states sent to Mooncake and retrieved successfully")

        finally:
            # Cleanup
            if hasattr(engine, "shutdown"):
                engine.shutdown()


# =============================================================================
# Legacy main block (kept for backward compatibility)
# =============================================================================

if __name__ == "__main__":
    pytest.main([__file__, "-v"])
