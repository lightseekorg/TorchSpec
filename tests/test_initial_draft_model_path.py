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

"""Staged training off a published draft (``model.initial_draft_model_path``)."""

import tempfile
import unittest
from pathlib import Path
from unittest import mock

import torch
from omegaconf import OmegaConf
from safetensors.torch import save_file

from torchspec.config.train_config import Config, config_to_flat_args, load_config
from torchspec.models.draft.auto import AutoDraftModelConfig, AutoEagle3DraftModel
from torchspec.models.draft.keymap import to_export_keys, to_internal_keys
from torchspec.training.checkpoint import (
    load_initial_draft_weights,
    resolve_initial_draft_checkpoint,
)

_EAGLE3 = {
    "architectures": ["LlamaForCausalLMEagle3"],
    "model_type": "llama",
    "hidden_size": 64,
    "intermediate_size": 128,
    "num_hidden_layers": 1,
    "num_attention_heads": 4,
    "num_key_value_heads": 4,
    "vocab_size": 128,
    "draft_vocab_size": 64,
    "max_position_embeddings": 256,
    "rms_norm_eps": 1e-6,
    "rope_theta": 10000.0,
    "pad_token_id": 0,
    "tie_word_embeddings": False,
}

_DFLASH = {
    "architectures": ["DFlashDraftModel"],
    "model_type": "dflash",
    "hidden_size": 64,
    "intermediate_size": 128,
    "num_hidden_layers": 2,
    "num_attention_heads": 4,
    "num_key_value_heads": 2,
    "vocab_size": 128,
    "max_position_embeddings": 256,
    "rms_norm_eps": 1e-6,
    "rope_theta": 10000.0,
    "num_target_layers": 2,
    "target_hidden_size": 64,
    "target_num_hidden_layers": 12,
    "mask_token_id": 127,
}


def _tmpdir(test: unittest.TestCase) -> Path:
    directory = tempfile.TemporaryDirectory()
    test.addCleanup(directory.cleanup)
    return Path(directory.name)


def _build(config_dict, seed):
    torch.manual_seed(seed)
    config = AutoDraftModelConfig.from_dict(dict(config_dict))
    return AutoEagle3DraftModel.from_config(config, torch_dtype=torch.float32)


def _publish(model, directory: Path) -> Path:
    """Write *model* out the way ``tools/convert_to_hf.py`` does, under serving key names."""
    directory.mkdir(parents=True, exist_ok=True)
    save_file(to_export_keys(model.state_dict()), str(directory / "model.safetensors"))
    return directory


class TestResolveInitialDraftCheckpoint(unittest.TestCase):
    def setUp(self):
        self.tmpdir = _tmpdir(self)
        self.checkpoint = self.tmpdir / "model.safetensors"
        save_file({"w": torch.zeros(2)}, str(self.checkpoint))

    def test_accepts_the_file_itself(self):
        self.assertEqual(resolve_initial_draft_checkpoint(str(self.checkpoint)), self.checkpoint)

    def test_accepts_the_directory_holding_it(self):
        self.assertEqual(resolve_initial_draft_checkpoint(str(self.tmpdir)), self.checkpoint)

    def test_expands_user_home(self):
        with mock.patch.dict("os.environ", {"HOME": str(self.tmpdir)}):
            self.assertEqual(
                resolve_initial_draft_checkpoint("~/model.safetensors"), self.checkpoint
            )

    def test_rejects_a_directory_without_a_checkpoint(self):
        empty = self.tmpdir / "empty"
        empty.mkdir()
        with self.assertRaises(ValueError):
            resolve_initial_draft_checkpoint(str(empty))

    def test_rejects_a_nonexistent_file(self):
        with self.assertRaises(ValueError):
            resolve_initial_draft_checkpoint(str(self.tmpdir / "missing.safetensors"))

    def test_rejects_a_non_safetensors_file(self):
        other = self.tmpdir / "model.bin"
        other.write_bytes(b"")
        with self.assertRaises(ValueError):
            resolve_initial_draft_checkpoint(str(other))


class TestLoadInitialDraftWeights(unittest.TestCase):
    def setUp(self):
        self.tmpdir = _tmpdir(self)

    def _assert_matches(self, model, reference):
        expected = reference.state_dict()
        for key, value in model.state_dict().items():
            self.assertTrue(torch.equal(value, expected[key]), msg=f"{key} was not restored")

    def test_loads_a_published_eagle3_draft(self):
        published = _build(_EAGLE3, seed=0)
        directory = _publish(published, self.tmpdir / "eagle3")

        fresh = _build(_EAGLE3, seed=1)
        loaded_from = load_initial_draft_weights(fresh, str(directory))

        self.assertEqual(loaded_from, directory / "model.safetensors")
        self._assert_matches(fresh, published)

    def test_loads_a_published_dflash_draft(self):
        # DFlash exports `final_norm` as `norm` and `context_proj` as `fc`, both of which are
        # names an Eagle3 draft uses internally.
        published = _build(_DFLASH, seed=0)
        directory = _publish(published, self.tmpdir / "dflash")

        fresh = _build(_DFLASH, seed=1)
        load_initial_draft_weights(fresh, str(directory))

        self._assert_matches(fresh, published)

    def test_loads_a_checkpoint_that_already_uses_internal_keys(self):
        published = _build(_EAGLE3, seed=0)
        directory = self.tmpdir / "native"
        directory.mkdir()
        save_file(published.state_dict(), str(directory / "model.safetensors"))

        fresh = _build(_EAGLE3, seed=1)
        load_initial_draft_weights(fresh, str(directory))

        self._assert_matches(fresh, published)

    def test_rejects_a_checkpoint_for_a_different_draft(self):
        published = _build(_DFLASH, seed=0)
        directory = _publish(published, self.tmpdir / "wrong")

        with self.assertRaises(RuntimeError):
            load_initial_draft_weights(_build(_EAGLE3, seed=1), str(directory))

    def test_rejects_a_checkpoint_missing_a_tensor(self):
        published = _build(_EAGLE3, seed=0)
        tensors = to_export_keys(published.state_dict())
        tensors.pop("lm_head.weight")
        directory = self.tmpdir / "truncated"
        directory.mkdir()
        save_file(tensors, str(directory / "model.safetensors"))

        with self.assertRaises(RuntimeError):
            load_initial_draft_weights(_build(_EAGLE3, seed=1), str(directory))


class TestVocabMappingGuard(unittest.TestCase):
    """A seeded draft's lm_head rows are ordered by the mapping that produced them."""

    @staticmethod
    def _mapping(used_tokens, target_vocab=128):
        used = torch.tensor(used_tokens, dtype=torch.long)
        t2d = torch.zeros(target_vocab, dtype=torch.bool)
        t2d[used] = True
        return used - torch.arange(len(used)), t2d

    def _pruned_draft(self):
        return _build({**_EAGLE3, "vocab_size": 128, "draft_vocab_size": 4}, seed=0)

    def test_fresh_draft_accepts_any_mapping(self):
        draft = self._pruned_draft()
        self.assertFalse(draft.has_vocab_pruning)

        d2t, t2d = self._mapping([3, 9, 40, 77])
        draft.set_vocab_buffers(d2t, t2d)

        self.assertTrue(draft.has_vocab_pruning)
        self.assertTrue(torch.equal(draft.t2d, t2d))
        self.assertTrue(torch.equal(draft.d2t, d2t))

    def test_seeded_draft_accepts_the_same_mapping(self):
        draft = self._pruned_draft()
        d2t, t2d = self._mapping([3, 9, 40, 77])
        draft.set_vocab_buffers(d2t, t2d)
        draft.set_vocab_buffers(d2t.clone(), t2d.clone())

    def test_seeded_draft_rejects_a_different_token_set(self):
        draft = self._pruned_draft()
        draft.set_vocab_buffers(*self._mapping([3, 9, 40, 77]))

        with self.assertRaises(ValueError) as caught:
            draft.set_vocab_buffers(*self._mapping([3, 9, 40, 78]))
        self.assertIn("keep_initial_vocab_mapping", str(caught.exception))

    def test_guard_survives_a_load_from_a_pruned_published_draft(self):
        published = self._pruned_draft()
        published.set_vocab_buffers(*self._mapping([3, 9, 40, 77]))
        directory = _publish(published, _tmpdir(self) / "pruned")

        fresh = self._pruned_draft()
        load_initial_draft_weights(fresh, str(directory))
        self.assertTrue(fresh.has_vocab_pruning)

        with self.assertRaises(ValueError):
            fresh.set_vocab_buffers(*self._mapping([3, 9, 40, 78]))


class TestToInternalKeys(unittest.TestCase):
    def test_leaves_a_key_the_model_already_owns_alone(self):
        eagle3_keys = {"norm.weight", "fc.weight", "midlayer.self_attn.q_proj.weight"}
        remapped = to_internal_keys(
            {
                "norm.weight": torch.zeros(1),
                "fc.weight": torch.zeros(1),
                "layers.0.self_attn.q_proj.weight": torch.zeros(1),
            },
            eagle3_keys,
        )
        self.assertEqual(sorted(remapped), sorted(eagle3_keys))

    def test_rewrites_only_when_the_result_is_a_model_key(self):
        remapped = to_internal_keys({"norm.weight": torch.zeros(1)}, {"final_norm.weight"})
        self.assertEqual(list(remapped), ["final_norm.weight"])

    def test_passes_unknown_keys_through_untouched(self):
        remapped = to_internal_keys({"mystery.weight": torch.zeros(1)}, {"norm.weight"})
        self.assertEqual(list(remapped), ["mystery.weight"])

    def test_round_trips_the_forward_mapping_for_every_family(self):
        for name, config_dict in (("eagle3", _EAGLE3), ("dflash", _DFLASH)):
            with self.subTest(family=name):
                keys = _build(config_dict, seed=0).state_dict().keys()
                exported = to_export_keys({k: torch.zeros(1) for k in keys})
                self.assertEqual(sorted(to_internal_keys(exported, keys)), sorted(keys))


class TestConfigPlumbing(unittest.TestCase):
    def test_field_defaults_to_none_and_reaches_the_flat_args(self):
        config = OmegaConf.structured(Config)
        self.assertIsNone(config.model.initial_draft_model_path)

        config.model.initial_draft_model_path = "/drafts/published"
        args = config_to_flat_args(config)
        self.assertEqual(args.initial_draft_model_path, "/drafts/published")

    def test_relative_path_is_absolutized_like_the_other_local_paths(self):
        # Ray actors do not share the launcher's working directory. Like output_dir, this resolves
        # against the invocation CWD, not the config file's directory.
        tmpdir = _tmpdir(self)
        config_path = tmpdir / "train.yaml"
        config_path.write_text(
            "model:\n  initial_draft_model_path: drafts/published\noutput_dir: out\n"
        )
        loaded = load_config(config_path=str(config_path))

        self.assertEqual(
            loaded.model.initial_draft_model_path,
            str(Path.cwd() / "drafts" / "published"),
        )

    def test_absolute_path_is_left_alone(self):
        tmpdir = _tmpdir(self)
        config_path = tmpdir / "train.yaml"
        config_path.write_text(f"model:\n  initial_draft_model_path: {tmpdir}/published\n")
        loaded = load_config(config_path=str(config_path))

        self.assertEqual(loaded.model.initial_draft_model_path, f"{tmpdir}/published")

    def test_keep_initial_vocab_mapping_requires_loaded_weights(self):
        tmpdir = _tmpdir(self)
        config_path = tmpdir / "train.yaml"
        config_path.write_text("model:\n  keep_initial_vocab_mapping: true\n")

        with self.assertRaises(ValueError):
            load_config(config_path=str(config_path))

    def test_keep_initial_vocab_mapping_accepted_with_a_seeded_draft(self):
        tmpdir = _tmpdir(self)
        config_path = tmpdir / "train.yaml"
        config_path.write_text(
            "model:\n"
            "  keep_initial_vocab_mapping: true\n"
            f"  initial_draft_model_path: {tmpdir}/published\n"
        )
        loaded = load_config(config_path=str(config_path))

        self.assertTrue(loaded.model.keep_initial_vocab_mapping)

    def test_unset_path_stays_unset(self):
        tmpdir = _tmpdir(self)
        config_path = tmpdir / "train.yaml"
        config_path.write_text("output_dir: out\n")
        loaded = load_config(config_path=str(config_path))

        self.assertIsNone(loaded.model.initial_draft_model_path)


if __name__ == "__main__":
    unittest.main()
