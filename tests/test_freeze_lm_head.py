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

"""Frozen draft lm_head (``model.freeze_lm_head``)."""

import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import torch
from omegaconf import OmegaConf
from safetensors import safe_open
from safetensors.torch import save_file

from torchspec.config.train_config import Config, config_to_flat_args
from torchspec.models.draft.auto import AutoDraftModelConfig, AutoEagle3DraftModel
from torchspec.training.checkpoint import resolve_resume_model_dir
from torchspec.training.optimizer import BF16Optimizer

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


def _tmpdir(test: unittest.TestCase) -> Path:
    directory = tempfile.TemporaryDirectory()
    test.addCleanup(directory.cleanup)
    return Path(directory.name)


def _build(**overrides):
    torch.manual_seed(0)
    config = AutoDraftModelConfig.from_dict({**_EAGLE3, **overrides})
    return AutoEagle3DraftModel.from_config(config, torch_dtype=torch.float32)


def _publish_target(directory: Path, hidden_size: int = 64, vocab_size: int = 128) -> Path:
    """A minimal single-file HF target checkpoint carrying just the keys we read."""
    directory.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(7)
    save_file(
        {
            "lm_head.weight": torch.randn(vocab_size, hidden_size),
            "model.embed_tokens.weight": torch.randn(vocab_size, hidden_size),
        },
        str(directory / "model.safetensors"),
    )
    return directory


class TestFreezeLmHead(unittest.TestCase):
    def test_freeze_only_detaches_the_head(self):
        model = _build()
        model.freeze_lm_head()

        self.assertFalse(model.lm_head.weight.requires_grad)
        self.assertTrue(model.midlayer.self_attn.q_proj.weight.requires_grad)
        self.assertTrue(model.fc.weight.requires_grad)
        self.assertTrue(model.norm.weight.requires_grad)

    def test_frozen_head_leaves_the_trainable_set(self):
        """``BF16Optimizer`` keeps an fp32 master copy per trainable param, so the head's
        optimizer state disappears with it -- the head is the largest draft tensor."""
        model = _build()
        model.freeze_embedding()

        def trainable():
            return sum(p.numel() for p in model.parameters() if p.requires_grad)

        before = trainable()
        model.freeze_lm_head()

        self.assertEqual(before - trainable(), model.lm_head.weight.numel())

    def test_load_lm_head_copies_the_target_rows(self):
        target = _publish_target(_tmpdir(self) / "target")
        model = _build()

        model.load_lm_head(str(target))

        with safe_open(str(target / "model.safetensors"), framework="pt") as f:
            expected = f.get_tensor("lm_head.weight")
        torch.testing.assert_close(model.lm_head.weight, expected)

    def test_load_lm_head_rejects_a_shape_mismatch(self):
        target = _publish_target(_tmpdir(self) / "target", hidden_size=32)
        model = _build()

        with self.assertRaisesRegex(ValueError, "does not match the draft lm_head"):
            model.load_lm_head(str(target))

    def test_load_lm_head_rejects_a_pruned_vocabulary(self):
        """A pruned head's rows follow t2d, which is not populated until after model init."""
        target = _publish_target(_tmpdir(self) / "target")
        model = _build(draft_vocab_size=64)

        with self.assertRaisesRegex(ValueError, "pruned draft vocabulary"):
            model.load_lm_head(str(target))


def _head_inputs(model, device="cpu"):
    """Aux hidden states shaped for ``project_hidden_states``."""
    width = model.fc.in_features
    return torch.randn(2, 5, width, device=device)


class TestFrozenHeadGradients(unittest.TestCase):
    """The flag assertions above check declarations; these check autograd and the optimizer."""

    def test_backward_leaves_no_gradient_on_the_head(self):
        model = _build()
        model.freeze_embedding()
        model.freeze_lm_head()

        logits = model.compute_logits(model.project_hidden_states(_head_inputs(model)))
        logits.square().mean().backward()

        self.assertIsNone(model.lm_head.weight.grad)
        # The same graph must still reach the trainable parameters, or the test is vacuous.
        self.assertIsNotNone(model.norm.weight.grad)
        self.assertIsNotNone(model.fc.weight.grad)

    def test_unfrozen_head_does_receive_a_gradient(self):
        model = _build()

        logits = model.compute_logits(model.project_hidden_states(_head_inputs(model)))
        logits.square().mean().backward()

        self.assertIsNotNone(model.lm_head.weight.grad)

    @unittest.skipUnless(
        torch.cuda.is_available(), "BF16Optimizer builds fused AdamW, which requires CUDA"
    )
    def test_optimizer_steps_leave_the_frozen_head_bitwise_unchanged(self):
        model = _build().cuda()
        model.freeze_embedding()
        model.freeze_lm_head()
        optimizer = BF16Optimizer(model, lr=1e-2, total_steps=10)
        head_before = model.lm_head.weight.detach().clone()
        norm_before = model.norm.weight.detach().clone()

        for _ in range(3):
            logits = model.compute_logits(
                model.project_hidden_states(_head_inputs(model, device="cuda"))
            )
            logits.square().mean().backward()
            optimizer.step()

        torch.testing.assert_close(model.lm_head.weight, head_before, rtol=0, atol=0)
        self.assertFalse(torch.equal(model.norm.weight, norm_before))


class TestResolveResumeModelDir(unittest.TestCase):
    """Seeding keys off a resolved checkpoint, not off the load_path string.

    ``load()`` skips a load_path that resolves to nothing, so treating the bare string as
    "the head will arrive from a checkpoint" would freeze a randomly initialized head.
    """

    def _checkpoint(self, step: int = 10, with_model: bool = True) -> Path:
        root = _tmpdir(self) / "run"
        (root / f"iter_{step:07d}").mkdir(parents=True)
        if with_model:
            (root / f"iter_{step:07d}" / "model").mkdir()
        (root / "latest_checkpointed_iteration.txt").write_text(f"{step}\n")
        return root

    def test_unset_path(self):
        self.assertIsNone(resolve_resume_model_dir(SimpleNamespace(load_path=None)))

    def test_nonexistent_path(self):
        args = SimpleNamespace(load_path=str(_tmpdir(self) / "typo"), ckpt_step=None)

        self.assertIsNone(resolve_resume_model_dir(args))

    def test_directory_without_a_tracker(self):
        root = _tmpdir(self) / "run"
        root.mkdir()

        self.assertIsNone(
            resolve_resume_model_dir(SimpleNamespace(load_path=str(root), ckpt_step=None))
        )

    def test_tracker_pointing_at_a_step_with_no_model_dir(self):
        root = self._checkpoint(with_model=False)

        self.assertIsNone(
            resolve_resume_model_dir(SimpleNamespace(load_path=str(root), ckpt_step=None))
        )

    def test_complete_checkpoint_resolves(self):
        root = self._checkpoint(step=10)

        resolved = resolve_resume_model_dir(SimpleNamespace(load_path=str(root), ckpt_step=None))

        self.assertEqual(resolved, root / "iter_0000010" / "model")

    def test_explicit_step_bypasses_the_tracker(self):
        root = self._checkpoint(step=7)
        (root / "latest_checkpointed_iteration.txt").unlink()

        resolved = resolve_resume_model_dir(SimpleNamespace(load_path=str(root), ckpt_step=7))

        self.assertEqual(resolved, root / "iter_0000007" / "model")


class TestFreezeLmHeadConfig(unittest.TestCase):
    def test_flag_defaults_off_and_reaches_the_flat_args(self):
        config = OmegaConf.structured(Config)
        self.assertFalse(config.model.freeze_lm_head)

        config.model.freeze_lm_head = True
        self.assertTrue(config_to_flat_args(config).freeze_lm_head)


if __name__ == "__main__":
    unittest.main()
