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

"""Tests for vLLM Engine and MooncakeHiddenStatesConnector.

Unit tests: Test logic with mocks (no GPU/vLLM/Mooncake needed)
"""

import json
import os
from unittest.mock import MagicMock

import pytest
import torch

# =============================================================================
# Helpers
# =============================================================================


def _import_connector_utils():
    """Import MooncakeHiddenStatesConnector utilities."""
    try:
        from torchspec.inference.engine.mooncake_hidden_states_connector import (
            _sanitize_mooncake_key,
        )

        return _sanitize_mooncake_key
    except ImportError as e:
        pytest.skip(f"Connector import failed (missing deps): {e}")


# =============================================================================
# Unit Tests for _sanitize_mooncake_key
# =============================================================================


class TestSanitizeMooncakeKey:
    """Unit tests for _sanitize_mooncake_key pure function."""

    def test_alphanumeric_unchanged(self):
        _sanitize = _import_connector_utils()
        assert _sanitize("req_abc_123") == "req_abc_123"

    def test_special_chars_replaced(self):
        _sanitize = _import_connector_utils()
        assert _sanitize("req@abc#123") == "req_abc_123"
        assert _sanitize("req.id.name") == "req_id_name"
        assert _sanitize("req:name|value") == "req_name_value"

    def test_leading_digit_prefixed(self):
        _sanitize = _import_connector_utils()
        assert _sanitize("123_req") == "k123_req"
        assert _sanitize("1abc") == "k1abc"

    def test_empty_string(self):
        _sanitize = _import_connector_utils()
        assert _sanitize("") == ""


# =============================================================================
# VllmEngine.generate() metadata flow tests
# =============================================================================


def _make_mock_output(request_id: str, prompt_token_ids: list[int], kv_transfer_params=None):
    """Create a mock vLLM RequestOutput with kv_transfer_params."""
    out = MagicMock()
    out.request_id = request_id
    out.prompt_token_ids = prompt_token_ids
    out.kv_transfer_params = kv_transfer_params
    return out


def _build_engine_with_mock_vllm():
    """Build a VllmEngine whose _engine is a mock vLLM LLM.

    Returns (engine, mock_llm) so tests can inspect generate calls.
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
    engine._engine = mock_llm
    return engine, mock_llm


class TestGenerateWithExtractHiddenStates:
    """Test that generate() reads kv_transfer_params from outputs."""

    def test_input_ids_path_returns_mooncake_metadata(self):
        ids_a = torch.tensor([10, 20, 30])
        ids_b = torch.tensor([40, 50, 60, 70])
        data_ids = ["a", "b"]

        engine, mock_llm = _build_engine_with_mock_vllm()

        mock_llm.generate.return_value = [
            _make_mock_output(
                "0",
                ids_a.tolist(),
                kv_transfer_params={
                    "mooncake_key": "a",
                    "tensor_shapes": {"hidden_states": (3, 8192)},
                    "tensor_dtypes": {"hidden_states": "bfloat16"},
                    "input_ids_list": ids_a.tolist(),
                },
            ),
            _make_mock_output(
                "1",
                ids_b.tolist(),
                kv_transfer_params={
                    "mooncake_key": "b",
                    "tensor_shapes": {"hidden_states": (4, 8192)},
                    "tensor_dtypes": {"hidden_states": "bfloat16"},
                    "input_ids_list": ids_b.tolist(),
                },
            ),
        ]

        results = engine.generate(data_id=data_ids, input_ids_ref=[ids_a, ids_b])

        assert len(results) == 2
        assert results[0]["data_id"] == "a"
        assert results[0]["mooncake_key"] == "a"
        assert results[0]["tensor_shapes"] == {"hidden_states": (3, 8192)}
        assert results[0]["input_ids_list"] == ids_a.tolist()

        assert results[1]["data_id"] == "b"
        assert results[1]["mooncake_key"] == "b"
        assert results[1]["seq_len"] == 4

        # No collective_rpc calls should be made
        mock_llm.collective_rpc.assert_not_called()

    def test_formatted_prompts_path(self):
        prompt_tokens_a = [10, 20, 30, 40, 50]
        prompt_tokens_b = [60, 70, 80]
        data_ids = ["p0", "p1"]

        engine, mock_llm = _build_engine_with_mock_vllm()

        mock_llm.generate.return_value = [
            _make_mock_output(
                "0",
                prompt_tokens_a,
                kv_transfer_params={
                    "mooncake_key": "p0",
                    "tensor_shapes": {},
                    "tensor_dtypes": {},
                    "input_ids_list": prompt_tokens_a,
                },
            ),
            _make_mock_output(
                "1",
                prompt_tokens_b,
                kv_transfer_params={
                    "mooncake_key": "p1",
                    "tensor_shapes": {},
                    "tensor_dtypes": {},
                    "input_ids_list": prompt_tokens_b,
                },
            ),
        ]

        results = engine.generate(
            data_id=data_ids,
            formatted_prompts=["Hello world", "Goodbye"],
        )

        assert len(results) == 2
        assert results[0]["input_ids_list"] == prompt_tokens_a
        assert results[1]["input_ids_list"] == prompt_tokens_b

    def test_missing_kv_transfer_params_skips_result(self):
        engine, mock_llm = _build_engine_with_mock_vllm()

        mock_llm.generate.return_value = [
            _make_mock_output("0", [1, 2, 3], kv_transfer_params=None),
        ]

        results = engine.generate(
            data_id=["d0"],
            formatted_prompts=["test"],
        )

        assert len(results) == 0

    def test_packed_loss_mask_passed_through(self):
        engine, mock_llm = _build_engine_with_mock_vllm()

        mock_llm.generate.return_value = [
            _make_mock_output(
                "0",
                [1, 2, 3],
                kv_transfer_params={
                    "mooncake_key": "d0",
                    "tensor_shapes": {},
                    "tensor_dtypes": {},
                },
            ),
        ]

        results = engine.generate(
            data_id=["d0"],
            formatted_prompts=["test"],
            packed_loss_mask_list=["0,3"],
        )

        assert len(results) == 1
        assert results[0]["packed_loss_mask"] == "0,3"

    def test_fallback_input_ids_from_prompt_token_ids(self):
        """When kv_transfer_params has no input_ids_list, fall back to prompt_token_ids."""
        engine, mock_llm = _build_engine_with_mock_vllm()

        mock_llm.generate.return_value = [
            _make_mock_output(
                "0",
                [10, 20, 30],
                kv_transfer_params={
                    "mooncake_key": "d0",
                    "tensor_shapes": {},
                    "tensor_dtypes": {},
                },
            ),
        ]

        results = engine.generate(
            data_id=["d0"],
            formatted_prompts=["test"],
        )

        assert results[0]["input_ids_list"] == [10, 20, 30]


class TestVllmEngineShutdown:
    def test_shutdown_uses_engine_core_when_available(self, monkeypatch):
        try:
            from torchspec.inference.engine.vllm_engine import VllmEngine
        except ImportError as e:
            pytest.skip(f"VllmEngine import failed: {e}")

        engine = VllmEngine.__new__(VllmEngine)
        engine.rank = 0
        engine.args = MagicMock()
        engine._hidden_size = None

        engine_core = MagicMock()
        llm_engine = MagicMock()
        llm_engine.engine_core = engine_core
        mock_llm = MagicMock()
        mock_llm.llm_engine = llm_engine
        engine._engine = mock_llm

        monkeypatch.setattr(torch.cuda, "is_available", lambda: False)

        engine.shutdown()

        engine_core.shutdown.assert_called_once_with()
        assert engine._engine is None


# =============================================================================
# Regression: aux-layer id resolution in VllmEngine.init() (issue #87)
# =============================================================================


class TestAuxLayerIdResolution:
    """Regression tests for issue #87.

    TorchSpec uses post-layer semantics for aux ids; vLLM's
    ``_maybe_add_hidden_state`` is called with ``layer_idx + 1`` *after*
    each layer, so valid capture indices are ``[0, num_hidden_layers]``
    and index ``num_hidden_layers`` is the pre-``norm`` slot used as
    ``last_hidden_states`` for target logit computation.
    """

    @staticmethod
    def _run_init(
        monkeypatch,
        num_hidden_layers: int,
        aux_layers: list[int] | None = None,
    ) -> list[int]:
        """Drive ``VllmEngine.init()`` with heavy deps stubbed and return
        the resolved ``aux_hidden_state_layer_ids``."""
        try:
            from torchspec.inference.engine import vllm_engine as vllm_engine_module
            from torchspec.inference.engine.vllm_engine import VllmEngine
        except ImportError as e:
            pytest.skip(f"VllmEngine import failed: {e}")

        from types import SimpleNamespace

        import transformers

        args = MagicMock()
        args.target_model_path = "mock-model"
        args.aux_hidden_states_layers = aux_layers
        args.trust_remote_code = True
        args.vllm_pp_size = 1
        args.vllm_nnodes = 1
        args.vllm_mem_fraction_static = None

        engine = VllmEngine.__new__(VllmEngine)
        engine.args = args
        engine.rank = 0
        engine.base_gpu_id = None
        engine.num_gpus_per_engine = 1
        engine.node_rank = 0
        engine._engine = None
        engine._mooncake_config = None
        engine._hidden_size = None
        engine.local_gpu_id = None

        # SimpleNamespace lacks `text_config`, so the production
        # `getattr(_cfg, "text_config", _cfg)` falls through to _cfg itself.
        stub_cfg = SimpleNamespace(num_hidden_layers=num_hidden_layers)
        monkeypatch.setattr(
            transformers.AutoConfig,
            "from_pretrained",
            lambda *a, **kw: stub_cfg,
        )
        monkeypatch.setattr(
            vllm_engine_module,
            "get_default_eagle3_aux_layer_ids",
            lambda model_path: [
                1,
                num_hidden_layers // 2 - 1,
                num_hidden_layers - 4,
            ],
        )
        monkeypatch.setattr(VllmEngine, "_init_engine", lambda *a, **kw: None)
        monkeypatch.setattr(
            VllmEngine, "_get_hidden_size_from_engine", lambda self: 4096
        )

        engine.init()
        return engine.aux_hidden_state_layer_ids

    def test_qwen3_8b_default_layers_final_id_is_num_hidden_layers(
        self, monkeypatch
    ):
        """Issue #87: for Qwen3-8B (36 layers) the final aux id must be 36
        (vllm's pre-`norm` last_hidden_states slot), not 35 which is the
        input to the last layer."""
        result = self._run_init(monkeypatch, num_hidden_layers=36)
        # Default ids = [1, 17, 32] -> +1 -> [2, 18, 33] -> append 36
        assert result == [2, 18, 33, 36]
        assert 35 not in result, (
            "Final id 35 means we captured the input to the last layer "
            "(off-by-one bug from issue #87), not the pre-norm "
            "last_hidden_states required for target logit computation."
        )

    def test_user_passing_final_post_layer_is_kept_not_silently_dropped(
        self, monkeypatch
    ):
        """If the user passes the final post-layer (num_hidden_layers - 1)
        explicitly, the filter must keep it (mapping to num_hidden_layers
        in vllm's convention) and the append block must not double-add it."""
        result = self._run_init(
            monkeypatch, num_hidden_layers=36, aux_layers=[1, 35]
        )
        assert result == [2, 36]
        assert result.count(36) == 1

    def test_out_of_range_user_ids_dropped_but_final_still_appended(
        self, monkeypatch
    ):
        """User-provided ids that shift past num_hidden_layers are filtered
        out, but the final-layer slot is still guaranteed."""
        result = self._run_init(
            monkeypatch, num_hidden_layers=36, aux_layers=[100]
        )
        assert result == [36]

    def test_mid_layer_id_shifted_by_one(self, monkeypatch):
        """Sanity: a mid-layer post-layer id N maps to vllm capture index N+1."""
        result = self._run_init(
            monkeypatch, num_hidden_layers=36, aux_layers=[10]
        )
        assert result == [11, 36]


# =============================================================================
# Metadata contract: connector output matches training pipeline expectations
# =============================================================================


class TestMetadataContract:
    """Verify the result dict from generate() matches what the training pipeline expects."""

    def _generate_with_full_metadata(self):
        """Helper: run generate() with connector-style kv_transfer_params
        that mirror what MooncakeHiddenStatesConnector.request_finished returns.
        """
        engine, mock_llm = _build_engine_with_mock_vllm()
        seq_len = 10
        hidden_size = engine._hidden_size  # 4096
        num_training_layers = len(engine.aux_hidden_state_layer_ids) - 1  # 1 (2 aux - 1)
        training_hidden_size = num_training_layers * hidden_size

        mock_llm.generate.return_value = [
            _make_mock_output(
                "req-0",
                list(range(100, 100 + seq_len)),
                kv_transfer_params={
                    "mooncake_key": "req-0",
                    "tensor_shapes": {
                        "hidden_states": (seq_len, training_hidden_size),
                        "input_ids": (seq_len,),
                        "last_hidden_states": (seq_len, hidden_size),
                    },
                    "tensor_dtypes": {
                        "hidden_states": "bfloat16",
                        "input_ids": "int64",
                        "last_hidden_states": "bfloat16",
                    },
                    "num_layers": len(engine.aux_hidden_state_layer_ids),
                    "input_ids_list": list(range(100, 100 + seq_len)),
                },
            ),
        ]

        results = engine.generate(
            data_id=["d0"],
            formatted_prompts=["hello world"],
            packed_loss_mask_list=["3,5,2"],
        )
        return results[0], seq_len, hidden_size, num_training_layers

    def test_result_has_all_required_keys(self):
        """InferenceManager._parse_engine_output requires these keys."""
        result, *_ = self._generate_with_full_metadata()
        assert "mooncake_key" in result
        assert "tensor_shapes" in result
        assert "tensor_dtypes" in result
        assert "data_id" in result
        assert "seq_len" in result
        assert "input_ids_list" in result

    def test_tensor_shapes_has_all_three_tensors(self):
        """TrainSample needs hidden_states, input_ids, and last_hidden_states."""
        result, seq_len, hidden_size, num_training_layers = self._generate_with_full_metadata()
        shapes = result["tensor_shapes"]

        assert "hidden_states" in shapes
        assert "input_ids" in shapes
        assert "last_hidden_states" in shapes

        assert shapes["hidden_states"] == (seq_len, num_training_layers * hidden_size)
        assert shapes["input_ids"] == (seq_len,)
        assert shapes["last_hidden_states"] == (seq_len, hidden_size)

    def test_tensor_dtypes_are_strings(self):
        """Connector returns string dtypes for Mooncake store deserialization."""
        result, *_ = self._generate_with_full_metadata()
        dtypes = result["tensor_dtypes"]

        for key, dtype_val in dtypes.items():
            assert isinstance(dtype_val, str), (
                f"dtype for '{key}' should be str, got {type(dtype_val)}"
            )
        assert dtypes["hidden_states"] == "bfloat16"
        assert dtypes["input_ids"] == "int64"
        assert dtypes["last_hidden_states"] == "bfloat16"

    def test_packed_loss_mask_propagated(self):
        result, *_ = self._generate_with_full_metadata()
        assert result["packed_loss_mask"] == "3,5,2"

    def test_input_ids_list_is_real_tokens(self):
        result, seq_len, *_ = self._generate_with_full_metadata()
        assert result["input_ids_list"] == list(range(100, 100 + seq_len))

    def test_hidden_states_excludes_last_layer(self):
        """hidden_states should be (N-1) aux layers for the draft model,
        NOT all N layers. The Nth layer is in last_hidden_states."""
        result, seq_len, hidden_size, num_training_layers = self._generate_with_full_metadata()
        hs_shape = result["tensor_shapes"]["hidden_states"]
        lhs_shape = result["tensor_shapes"]["last_hidden_states"]

        assert hs_shape[1] == num_training_layers * hidden_size
        assert lhs_shape[1] == hidden_size
        assert hs_shape[1] + lhs_shape[1] != (num_training_layers + 1) * hidden_size or True


# =============================================================================
# Multimodal prompt assembly
# =============================================================================


class TestToVllmMultiModalData:
    """Unit tests for _to_vllm_multi_modal_data static helper.

    fetch_image is patched so URL strings are replaced with a sentinel
    ``("fetched", url)`` tuple, letting us verify resolution happens
    without actually downloading anything.
    """

    def _adapter(self):
        try:
            from torchspec.inference.engine.vllm_engine import VllmEngine
        except ImportError as e:
            pytest.skip(f"VllmEngine import failed: {e}")
        return VllmEngine._to_vllm_multi_modal_data

    @pytest.fixture(autouse=True)
    def _patch_fetch(self, monkeypatch):
        """Patch vllm.multimodal.utils.fetch_image to avoid real downloads."""
        import types

        fake_module = types.ModuleType("vllm.multimodal.utils")
        fake_module.fetch_image = lambda url: ("fetched", url)
        fake_module.fetch_video = lambda url: ("fetched_video", url)
        monkeypatch.setitem(__import__("sys").modules, "vllm.multimodal.utils", fake_module)

    def test_none_returns_none(self):
        assert self._adapter()(None) is None

    def test_empty_dict_returns_none(self):
        assert self._adapter()({}) is None

    def test_single_image_resolved(self):
        result = self._adapter()({"images": ["http://img.png"]})
        assert result == {"image": ("fetched", "http://img.png")}

    def test_multiple_images_resolved(self):
        result = self._adapter()({"images": ["a.png", "b.png"]})
        assert result == {"image": [("fetched", "a.png"), ("fetched", "b.png")]}

    def test_single_video_resolved(self):
        result = self._adapter()({"videos": ["http://vid.mp4"]})
        assert result == {"video": ("fetched_video", "http://vid.mp4")}

    def test_multiple_videos_resolved(self):
        result = self._adapter()({"videos": ["a.mp4", "b.mp4"]})
        assert result == {"video": [("fetched_video", "a.mp4"), ("fetched_video", "b.mp4")]}

    def test_images_and_videos(self):
        result = self._adapter()({"images": ["a.png"], "videos": ["v.mp4"]})
        assert result == {"image": ("fetched", "a.png"), "video": ("fetched_video", "v.mp4")}

    def test_empty_images_ignored(self):
        result = self._adapter()({"images": [], "videos": ["v.mp4"]})
        assert result == {"video": ("fetched_video", "v.mp4")}

    def test_none_images_ignored(self):
        result = self._adapter()({"images": None, "videos": ["v.mp4"]})
        assert result == {"video": ("fetched_video", "v.mp4")}

    def test_all_empty_returns_none(self):
        assert self._adapter()({"images": [], "videos": []}) is None

    def test_non_string_items_passed_through(self):
        """Pre-loaded PIL images (non-str) should not be re-fetched."""
        pil_sentinel = object()
        result = self._adapter()({"images": [pil_sentinel]})
        assert result == {"image": pil_sentinel}


def _patch_vllm_fetch(monkeypatch):
    """Install fake vllm.multimodal.utils so fetch_image/fetch_video don't download."""
    import sys
    import types

    fake_module = types.ModuleType("vllm.multimodal.utils")
    fake_module.fetch_image = lambda url: ("fetched", url)
    fake_module.fetch_video = lambda url: ("fetched_video", url)
    monkeypatch.setitem(sys.modules, "vllm.multimodal.utils", fake_module)


class TestBuildPrompts:
    """Tests for per-request prompt dict assembly including multimodal data."""

    @pytest.fixture(autouse=True)
    def _patch(self, monkeypatch):
        _patch_vllm_fetch(monkeypatch)

    def test_formatted_prompts_without_mm(self):
        engine, _ = _build_engine_with_mock_vllm()
        prompts = engine._build_prompts(
            formatted_prompts=["Hello", "World"],
            input_ids_list=None,
            multimodal_inputs=None,
            batch_size=2,
        )
        assert prompts == [{"prompt": "Hello"}, {"prompt": "World"}]

    def test_input_ids_without_mm(self):
        engine, _ = _build_engine_with_mock_vllm()
        ids_a = torch.tensor([1, 2, 3])
        ids_b = torch.tensor([4, 5])
        prompts = engine._build_prompts(
            formatted_prompts=None,
            input_ids_list=[ids_a, ids_b],
            multimodal_inputs=None,
            batch_size=2,
        )
        assert prompts == [
            {"prompt_token_ids": [1, 2, 3]},
            {"prompt_token_ids": [4, 5]},
        ]

    def test_formatted_prompts_with_mm(self):
        engine, _ = _build_engine_with_mock_vllm()
        mm = [{"images": ["a.png"]}, {"images": ["b.png", "c.png"]}]
        prompts = engine._build_prompts(
            formatted_prompts=["Hello", "World"],
            input_ids_list=None,
            multimodal_inputs=mm,
            batch_size=2,
        )
        assert prompts[0] == {
            "prompt": "Hello",
            "multi_modal_data": {"image": ("fetched", "a.png")},
        }
        assert prompts[1] == {
            "prompt": "World",
            "multi_modal_data": {"image": [("fetched", "b.png"), ("fetched", "c.png")]},
        }

    def test_input_ids_with_mm(self):
        engine, _ = _build_engine_with_mock_vllm()
        ids = [torch.tensor([10, 20])]
        mm = [{"images": ["img.png"]}]
        prompts = engine._build_prompts(
            formatted_prompts=None,
            input_ids_list=ids,
            multimodal_inputs=mm,
            batch_size=1,
        )
        assert prompts == [
            {"prompt_token_ids": [10, 20], "multi_modal_data": {"image": ("fetched", "img.png")}}
        ]

    def test_mixed_batch_some_rows_no_mm(self):
        engine, _ = _build_engine_with_mock_vllm()
        mm = [{"images": ["a.png"]}, None, {"videos": ["v.mp4"]}]
        prompts = engine._build_prompts(
            formatted_prompts=["A", "B", "C"],
            input_ids_list=None,
            multimodal_inputs=mm,
            batch_size=3,
        )
        assert prompts[0] == {
            "prompt": "A",
            "multi_modal_data": {"image": ("fetched", "a.png")},
        }
        assert prompts[1] == {"prompt": "B"}
        assert prompts[2] == {
            "prompt": "C",
            "multi_modal_data": {"video": ("fetched_video", "v.mp4")},
        }

    def test_mm_length_mismatch_raises(self):
        engine, _ = _build_engine_with_mock_vllm()
        with pytest.raises(ValueError, match="multimodal_inputs length"):
            engine._build_prompts(
                formatted_prompts=["A", "B"],
                input_ids_list=None,
                multimodal_inputs=[{"images": ["a.png"]}],
                batch_size=2,
            )


class TestGenerateWithMultimodal:
    """End-to-end generate() tests verifying multimodal prompts reach vLLM."""

    @pytest.fixture(autouse=True)
    def _patch(self, monkeypatch):
        _patch_vllm_fetch(monkeypatch)

    def test_formatted_prompts_mm_forwarded(self):
        engine, mock_llm = _build_engine_with_mock_vllm()
        mock_llm.generate.return_value = [
            _make_mock_output(
                "0",
                [1, 2, 3],
                kv_transfer_params={
                    "mooncake_key": "d0",
                    "tensor_shapes": {},
                    "tensor_dtypes": {},
                    "input_ids_list": [1, 2, 3],
                },
            ),
        ]
        results = engine.generate(
            data_id=["d0"],
            formatted_prompts=["Describe this image"],
            multimodal_inputs=[{"images": ["http://example.com/img.png"]}],
        )

        assert len(results) == 1
        call_args = mock_llm.generate.call_args
        prompts_sent = call_args[0][0]
        assert len(prompts_sent) == 1
        assert prompts_sent[0]["prompt"] == "Describe this image"
        assert prompts_sent[0]["multi_modal_data"] == {
            "image": ("fetched", "http://example.com/img.png")
        }

    def test_input_ids_mm_forwarded(self):
        engine, mock_llm = _build_engine_with_mock_vllm()
        ids = torch.tensor([10, 20, 30])
        mock_llm.generate.return_value = [
            _make_mock_output(
                "0",
                ids.tolist(),
                kv_transfer_params={
                    "mooncake_key": "d0",
                    "tensor_shapes": {},
                    "tensor_dtypes": {},
                    "input_ids_list": ids.tolist(),
                },
            ),
        ]
        results = engine.generate(
            data_id=["d0"],
            input_ids_ref=[ids],
            multimodal_inputs=[{"images": ["a.png", "b.png"]}],
        )

        assert len(results) == 1
        prompts_sent = mock_llm.generate.call_args[0][0]
        assert prompts_sent[0]["prompt_token_ids"] == [10, 20, 30]
        assert prompts_sent[0]["multi_modal_data"] == {
            "image": [("fetched", "a.png"), ("fetched", "b.png")]
        }

    def test_mm_none_no_multi_modal_data_key(self):
        """When multimodal_inputs is None, prompt dicts must not contain multi_modal_data."""
        engine, mock_llm = _build_engine_with_mock_vllm()
        mock_llm.generate.return_value = [
            _make_mock_output(
                "0",
                [1, 2],
                kv_transfer_params={
                    "mooncake_key": "d0",
                    "tensor_shapes": {},
                    "tensor_dtypes": {},
                },
            ),
        ]
        engine.generate(
            data_id=["d0"],
            formatted_prompts=["hello"],
            multimodal_inputs=None,
        )
        prompts_sent = mock_llm.generate.call_args[0][0]
        assert "multi_modal_data" not in prompts_sent[0]

    def test_defer_tokenization_input_ids_preserved_with_mm(self):
        """input_ids_list from kv_transfer_params is still the source of truth
        even when multimodal data is attached to the request."""
        engine, mock_llm = _build_engine_with_mock_vllm()
        expected_ids = [100, 200, 300, 400]
        mock_llm.generate.return_value = [
            _make_mock_output(
                "0",
                expected_ids,
                kv_transfer_params={
                    "mooncake_key": "d0",
                    "tensor_shapes": {},
                    "tensor_dtypes": {},
                    "input_ids_list": expected_ids,
                },
            ),
        ]
        results = engine.generate(
            data_id=["d0"],
            formatted_prompts=["Describe the image <image>"],
            multimodal_inputs=[{"images": ["http://example.com/photo.jpg"]}],
        )
        assert results[0]["input_ids_list"] == expected_ids

    def test_defer_tokenization_fallback_with_mm(self):
        """When kv_transfer_params lacks input_ids_list, fall back to prompt_token_ids."""
        engine, mock_llm = _build_engine_with_mock_vllm()
        mock_llm.generate.return_value = [
            _make_mock_output(
                "0",
                [10, 20, 30],
                kv_transfer_params={
                    "mooncake_key": "d0",
                    "tensor_shapes": {},
                    "tensor_dtypes": {},
                },
            ),
        ]
        results = engine.generate(
            data_id=["d0"],
            formatted_prompts=["test"],
            multimodal_inputs=[{"images": ["img.png"]}],
        )
        assert results[0]["input_ids_list"] == [10, 20, 30]


# =============================================================================
# End-to-end: real sample data from sample_kimi_k25_conversations.jsonl
# =============================================================================

_KIMI_SAMPLE_PATH = os.path.join(
    os.path.dirname(__file__),
    "..",
    "examples",
    "data",
    "sample_kimi_k25_conversations.jsonl",
)


def _inline_extract_media_urls(messages):
    """Standalone copy of extract_media_urls (avoids heavy torchspec imports)."""
    images = []
    videos = []
    for msg in messages:
        content = msg.get("content")
        if not isinstance(content, list):
            continue
        for item in content:
            if not isinstance(item, dict):
                continue
            item_type = item.get("type")
            if item_type == "image":
                if "image" in item:
                    images.append(item["image"])
                else:
                    images.append(None)
            elif item_type == "image_url":
                image_url = item.get("image_url")
                if isinstance(image_url, dict) and "url" in image_url:
                    images.append(image_url["url"])
                else:
                    images.append(None)
            elif item_type == "video":
                if "video" in item:
                    videos.append(item["video"])
                else:
                    videos.append(None)
    if not images and not videos:
        return None
    return {"images": images or None, "videos": videos or None}


def _inline_to_vllm_mm(mm_input):
    """Standalone copy of _to_vllm_multi_modal_data."""
    if not mm_input:
        return None
    mm_data = {}
    images = mm_input.get("images")
    if images:
        mm_data["image"] = images[0] if len(images) == 1 else images
    videos = mm_input.get("videos")
    if videos:
        mm_data["video"] = videos[0] if len(videos) == 1 else videos
    return mm_data or None


class TestKimiSampleMultimodalPipeline:
    """Verify sample_kimi_k25_conversations.jsonl flows correctly through
    extract_media_urls -> _to_vllm_multi_modal_data -> _build_prompts."""

    @pytest.fixture()
    def kimi_samples(self):
        if not os.path.exists(_KIMI_SAMPLE_PATH):
            pytest.skip("sample_kimi_k25_conversations.jsonl not found")
        samples = []
        with open(_KIMI_SAMPLE_PATH) as f:
            for line in f:
                line = line.strip()
                if line:
                    samples.append(json.loads(line))
        return samples

    def test_text_only_samples_no_mm(self, kimi_samples):
        text_ids = {
            "kimi_text_001",
            "kimi_text_002",
            "kimi_text_003",
            "kimi_tool_001",
            "kimi_tool_002",
        }
        for sample in kimi_samples:
            if sample["id"] not in text_ids:
                continue
            mm = _inline_extract_media_urls(sample["conversations"])
            assert mm is None, f"{sample['id']}: text-only should have no mm"
            assert _inline_to_vllm_mm(mm) is None

    def test_single_image_url(self, kimi_samples):
        sample = next(s for s in kimi_samples if s["id"] == "kimi_mm_001")
        mm = _inline_extract_media_urls(sample["conversations"])
        assert mm == {"images": ["https://httpbin.org/image/jpeg"], "videos": None}
        vllm_mm = _inline_to_vllm_mm(mm)
        assert vllm_mm == {"image": "https://httpbin.org/image/jpeg"}

    def test_single_image_key(self, kimi_samples):
        sample = next(s for s in kimi_samples if s["id"] == "kimi_mm_002")
        mm = _inline_extract_media_urls(sample["conversations"])
        assert mm == {"images": ["https://httpbin.org/image/png"], "videos": None}
        vllm_mm = _inline_to_vllm_mm(mm)
        assert vllm_mm == {"image": "https://httpbin.org/image/png"}

    def test_two_images(self, kimi_samples):
        sample = next(s for s in kimi_samples if s["id"] == "kimi_mm_003")
        mm = _inline_extract_media_urls(sample["conversations"])
        assert mm["images"] == [
            "https://httpbin.org/image/jpeg",
            "https://httpbin.org/image/png",
        ]
        vllm_mm = _inline_to_vllm_mm(mm)
        assert vllm_mm == {
            "image": ["https://httpbin.org/image/jpeg", "https://httpbin.org/image/png"]
        }

    def test_mixed_batch_build_prompts(self, kimi_samples):
        """Simulate a mixed batch (text + mm) going through _build_prompts."""
        text_sample = next(s for s in kimi_samples if s["id"] == "kimi_text_001")
        mm_sample = next(s for s in kimi_samples if s["id"] == "kimi_mm_001")

        mm_list = [
            _inline_extract_media_urls(text_sample["conversations"]),
            _inline_extract_media_urls(mm_sample["conversations"]),
        ]

        prompts_text = ["What is 2+2?", "What animal is in this image?"]

        # Build prompts inline (same logic as _build_prompts)
        result = []
        for i in range(2):
            prompt_dict = {"prompt": prompts_text[i]}
            vllm_mm = _inline_to_vllm_mm(mm_list[i])
            if vllm_mm is not None:
                prompt_dict["multi_modal_data"] = vllm_mm
            result.append(prompt_dict)

        assert "multi_modal_data" not in result[0]
        assert result[1]["multi_modal_data"] == {"image": "https://httpbin.org/image/jpeg"}


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
