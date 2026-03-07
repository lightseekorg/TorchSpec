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
# VllmEngine.generate() metadata flow tests
# =============================================================================


def _make_mock_output(request_id: str, prompt_token_ids: list[int]):
    """Create a mock vLLM RequestOutput."""
    out = MagicMock()
    out.request_id = request_id
    out.prompt_token_ids = prompt_token_ids
    return out


def _build_engine_with_mock_vllm(metadata_by_request: dict):
    """Build a VllmEngine whose _engine is a mock vLLM LLM.

    Returns (engine, mock_llm) so tests can inspect collective_rpc calls.
    """
    try:
        from torchspec.inference.engine.vllm_engine import VllmEngine
    except ImportError as e:
        pytest.skip(f"VllmEngine import failed: {e}")

    args = MagicMock()
    args.target_model_path = "mock-model"
    args.trust_remote_code = True
    engine = VllmEngine.__new__(VllmEngine)
    engine.args = args
    engine.rank = 0
    engine.base_gpu_id = 0
    engine._hidden_size = 4096
    engine.aux_hidden_state_layer_ids = [2, 4]

    mock_llm = MagicMock()

    def _collective_rpc(method, args=(), kwargs=None):
        if method == "_store_and_get_metadata":
            return [metadata_by_request]
        return [None]

    mock_llm.collective_rpc = MagicMock(side_effect=_collective_rpc)
    engine._engine = mock_llm
    return engine, mock_llm


class TestGenerateMetadataFlow:
    """Test that generate() builds and sends request_metadata for both
    the input_ids path and the formatted_prompts (defer_tokenization) path.
    """

    def test_input_ids_path_sends_metadata_twice(self):
        """input_ids path: _set_request_metadata is called both pre- and
        post-generation with correct token counts."""
        ids_a = torch.tensor([10, 20, 30])
        ids_b = torch.tensor([40, 50, 60, 70])
        data_ids = ["a", "b"]

        worker_meta = {
            "a": {
                "mooncake_key": "a",
                "tensor_shapes": {},
                "tensor_dtypes": {},
                "input_ids_list": ids_a.tolist(),
            },
            "b": {
                "mooncake_key": "b",
                "tensor_shapes": {},
                "tensor_dtypes": {},
                "input_ids_list": ids_b.tolist(),
            },
        }
        engine, mock_llm = _build_engine_with_mock_vllm(worker_meta)

        mock_llm.generate.return_value = [
            _make_mock_output("0", ids_a.tolist()),
            _make_mock_output("1", ids_b.tolist()),
        ]

        results = engine.generate(
            data_id=data_ids,
            input_ids_ref=[ids_a, ids_b],
        )

        set_meta_calls = [
            c for c in mock_llm.collective_rpc.call_args_list if c[0][0] == "_set_request_metadata"
        ]
        assert len(set_meta_calls) == 2, (
            f"Expected 2 _set_request_metadata calls, got {len(set_meta_calls)}"
        )

        # Post-gen call (last one) must carry authoritative token counts
        post_gen_args = set_meta_calls[-1][1]["args"]
        req_meta = post_gen_args[0]
        assert req_meta == {"a": 3, "b": 4}

        input_ids_map = post_gen_args[2]
        assert input_ids_map == {"a": ids_a.tolist(), "b": ids_b.tolist()}

        assert len(results) == 2
        assert results[0]["data_id"] == "a"
        assert results[1]["data_id"] == "b"

    def test_formatted_prompts_path_sends_metadata_post_gen(self):
        """formatted_prompts (defer_tokenization) path: _set_request_metadata
        is sent after generation with token counts from vLLM outputs."""
        prompt_tokens_a = [10, 20, 30, 40, 50]
        prompt_tokens_b = [60, 70, 80]
        data_ids = ["p0", "p1"]

        worker_meta = {
            "p0": {
                "mooncake_key": "p0",
                "tensor_shapes": {},
                "tensor_dtypes": {},
                "input_ids_list": prompt_tokens_a,
            },
            "p1": {
                "mooncake_key": "p1",
                "tensor_shapes": {},
                "tensor_dtypes": {},
                "input_ids_list": prompt_tokens_b,
            },
        }
        engine, mock_llm = _build_engine_with_mock_vllm(worker_meta)

        mock_llm.generate.return_value = [
            _make_mock_output("0", prompt_tokens_a),
            _make_mock_output("1", prompt_tokens_b),
        ]

        results = engine.generate(
            data_id=data_ids,
            formatted_prompts=["Hello world", "Goodbye"],
        )

        set_meta_calls = [
            c for c in mock_llm.collective_rpc.call_args_list if c[0][0] == "_set_request_metadata"
        ]
        # Only the post-gen call (pre-gen is skipped because request_metadata
        # is empty before generation).
        assert len(set_meta_calls) == 1

        post_gen_args = set_meta_calls[0][1]["args"]
        req_meta = post_gen_args[0]
        assert req_meta == {"p0": 5, "p1": 3}

        input_ids_map = post_gen_args[2]
        assert input_ids_map == {"p0": prompt_tokens_a, "p1": prompt_tokens_b}

        assert len(results) == 2
        assert results[0]["input_ids_list"] == prompt_tokens_a
        assert results[1]["input_ids_list"] == prompt_tokens_b

    def test_formatted_prompts_with_no_packed_loss_mask(self):
        """defer_tokenization path with packed_loss_mask_list=None works."""
        tokens = [1, 2, 3]
        worker_meta = {
            "d0": {
                "mooncake_key": "d0",
                "tensor_shapes": {},
                "tensor_dtypes": {},
                "input_ids_list": tokens,
            },
        }
        engine, mock_llm = _build_engine_with_mock_vllm(worker_meta)
        mock_llm.generate.return_value = [_make_mock_output("0", tokens)]

        results = engine.generate(
            data_id=["d0"],
            formatted_prompts=["test"],
            packed_loss_mask_list=None,
        )

        set_meta_calls = [
            c for c in mock_llm.collective_rpc.call_args_list if c[0][0] == "_set_request_metadata"
        ]
        assert len(set_meta_calls) == 1

        packed_map = set_meta_calls[0][1]["args"][1]
        assert packed_map == {}

        assert len(results) == 1
        assert "packed_loss_mask" not in results[0]


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
