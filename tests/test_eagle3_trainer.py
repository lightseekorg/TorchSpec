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

"""Tests for Eagle3Trainer metric aggregation and model initialisation.

Verifies that per-depth accuracy and LK acceptance-rate (alpha) metrics are
weighted by valid-token counts across gradient-accumulation steps (and, in
_aggregate_eval_metrics, eval-cache chunks) rather than averaged naively —
chunks/microbatches can carry very different numbers of loss positions, so a
mean-of-means over/under-weights small ones relative to a true token-weighted
mean.

TestInitModelInitialDraft covers ``model.initial_draft_model_path``, the
staged-training entry point.
"""

import tempfile
import unittest
from argparse import Namespace
from pathlib import Path
from unittest import mock

import torch
from safetensors.torch import save_file

from torchspec.models.draft.auto import AutoDraftModelConfig, AutoEagle3DraftModel
from torchspec.models.draft.keymap import to_export_keys
from torchspec.training import eagle3_trainer as eagle3_trainer_module
from torchspec.training.eagle3_trainer import Eagle3Trainer


class _DummyOptimizer:
    def get_learning_rate(self):
        return 1e-3


class _DummyArgs:
    ploss_weights = None


def _step_metrics() -> list[dict]:
    return [
        {
            "vlosses": torch.tensor([1.0, 2.0, 4.0]),
            "acces": torch.tensor([0.5, 0.25, 0.5]),
            "acc_counts": torch.tensor([10.0, 8.0, 2.0]),
            "alphas": torch.tensor([0.6, 0.3, 0.5]),
        },
        {
            "vlosses": torch.tensor([1.0, 2.0, 4.0]),
            "acces": torch.tensor([1.0, 0.0, 0.0]),
            "acc_counts": torch.tensor([2.0, 0.0, 0.0]),
            "alphas": torch.tensor([1.0, 0.0, 0.0]),
        },
    ]


class TestEagle3TrainerAggregation(unittest.TestCase):
    @staticmethod
    def _make_trainer():
        trainer = object.__new__(Eagle3Trainer)
        trainer.global_step = 7
        trainer.optimizer = _DummyOptimizer()
        trainer.args = _DummyArgs()
        trainer._ploss_weights = (1.0, 1.0, 1.0)
        trainer._ploss_weight_sum = sum(trainer._ploss_weights)
        trainer._metric_buffers = []
        return trainer

    @mock.patch("torchspec.training.eagle3_trainer.dist.get_rank", return_value=0)
    @mock.patch(
        "torchspec.training.eagle3_trainer.dist.all_reduce",
        side_effect=lambda tensor, op=None: None,
    )
    def test_aggregate_metrics_weights_by_token_counts(self, _mock_all_reduce, _mock_get_rank):
        trainer = self._make_trainer()
        metrics = trainer._aggregate_metrics(_step_metrics(), step=11, grad_norm=torch.tensor(3.0))

        # depth0: correct = 0.5*10 + 1.0*2 = 7, counts = 12 -> acc = 7/12
        # depth1: correct = 0.25*8 + 0.0*0 = 2, counts = 8  -> acc = 0.25
        # depth2: correct = 0.5*2  + 0.0*0 = 1, counts = 2  -> acc = 0.5
        self.assertAlmostEqual(metrics["train/acc_0"], 7.0 / 12.0, places=6)
        self.assertAlmostEqual(metrics["train/acc_1"], 0.25, places=6)
        self.assertAlmostEqual(metrics["train/acc_2"], 0.5, places=6)
        self.assertAlmostEqual(metrics["train/avg_acc"], (7.0 / 12.0 + 0.25 + 0.5) / 3, places=6)

        # alpha follows the same acc_counts weighting.
        # depth0: 0.6*10 + 1.0*2 = 8, /12 = 2/3
        # depth1: 0.3*8  + 0.0*0 = 2.4, /8 = 0.3
        # depth2: 0.5*2  + 0.0*0 = 1, /2 = 0.5
        self.assertAlmostEqual(metrics["train/alpha_0"], 8.0 / 12.0, places=6)
        self.assertAlmostEqual(metrics["train/alpha_1"], 0.3, places=6)
        self.assertAlmostEqual(metrics["train/alpha_2"], 0.5, places=6)
        self.assertAlmostEqual(metrics["train/avg_alpha"], (8.0 / 12.0 + 0.3 + 0.5) / 3, places=6)

        # vloss (already correctly count-weighted inside Eagle3Model.forward)
        # is unaffected: plain mean across accumulation steps, then AVG across ranks.
        self.assertAlmostEqual(metrics["train/ploss_0"], 1.0, places=6)
        self.assertAlmostEqual(metrics["train/ploss_1"], 2.0, places=6)
        self.assertAlmostEqual(metrics["train/ploss_2"], 4.0, places=6)
        self.assertAlmostEqual(metrics["train/avg_loss"], 7.0 / 3.0, places=6)
        self.assertAlmostEqual(metrics["train/grad_norm"], 3.0, places=6)

    @mock.patch("torchspec.training.eagle3_trainer.dist.get_rank", return_value=0)
    @mock.patch(
        "torchspec.training.eagle3_trainer.dist.all_reduce",
        side_effect=lambda tensor, op=None: None,
    )
    def test_naive_mean_of_means_would_disagree(self, _mock_all_reduce, _mock_get_rank):
        """Sanity check that the fix actually changes the reported value.

        Without count weighting, depth0's acc would be a plain mean of the two
        per-step accuracies (0.5 and 1.0) = 0.75, which over-weights the
        2-token step relative to the 10-token step.
        """
        trainer = self._make_trainer()
        metrics = trainer._aggregate_metrics(_step_metrics(), step=11, grad_norm=torch.tensor(3.0))
        naive_mean = (0.5 + 1.0) / 2
        self.assertNotAlmostEqual(metrics["train/acc_0"], naive_mean, places=3)


class TestEagle3TrainerEvalAggregation(unittest.TestCase):
    @staticmethod
    def _make_trainer():
        trainer = object.__new__(Eagle3Trainer)
        trainer.args = _DummyArgs()
        return trainer

    @mock.patch("torchspec.training.eagle3_trainer.dist.get_rank", return_value=0)
    @mock.patch(
        "torchspec.training.eagle3_trainer.dist.all_reduce",
        side_effect=lambda tensor, op=None: None,
    )
    def test_aggregate_eval_metrics_weights_by_token_counts(self, _mock_all_reduce, _mock_get_rank):
        trainer = self._make_trainer()
        metrics = trainer._aggregate_eval_metrics(_step_metrics())

        self.assertAlmostEqual(metrics["eval/acc_0"], 7.0 / 12.0, places=6)
        self.assertAlmostEqual(metrics["eval/acc_1"], 0.25, places=6)
        self.assertAlmostEqual(metrics["eval/acc_2"], 0.5, places=6)
        self.assertAlmostEqual(metrics["eval/alpha_0"], 8.0 / 12.0, places=6)
        self.assertAlmostEqual(metrics["eval/alpha_1"], 0.3, places=6)
        self.assertAlmostEqual(metrics["eval/alpha_2"], 0.5, places=6)
        self.assertAlmostEqual(metrics["eval/avg_acc"], (7.0 / 12.0 + 0.25 + 0.5) / 3, places=6)
        self.assertAlmostEqual(metrics["eval/avg_alpha"], (8.0 / 12.0 + 0.3 + 0.5) / 3, places=6)


_DRAFT_CONFIG = {
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


class TestInitModelInitialDraft(unittest.TestCase):
    """``init_model`` with ``initial_draft_model_path`` set, FSDP2 and the optimizer stubbed out."""

    def setUp(self):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.root = Path(directory.name)

    @staticmethod
    def _build_draft(seed):
        torch.manual_seed(seed)
        config = AutoDraftModelConfig.from_dict(dict(_DRAFT_CONFIG))
        return AutoEagle3DraftModel.from_config(config, torch_dtype=torch.bfloat16)

    def _publish_draft(self):
        model = self._build_draft(seed=0)
        directory = self.root / "published"
        directory.mkdir()
        save_file(to_export_keys(model.state_dict()), str(directory / "model.safetensors"))
        return model, directory

    def _write_target(self):
        directory = self.root / "target"
        directory.mkdir()
        torch.manual_seed(99)
        embedding = torch.randn(128, 64, dtype=torch.bfloat16)
        save_file({"model.embed_tokens.weight": embedding}, str(directory / "model.safetensors"))
        return embedding, directory

    def _run_init_model(self, initial_draft_model_path, target_dir):
        trainer = object.__new__(Eagle3Trainer)
        trainer.args = Namespace(
            attention_backend="sdpa",
            ttt_length=3,
            gradient_checkpointing=False,
            initial_draft_model_path=initial_draft_model_path,
            embedding_key="model.embed_tokens.weight",
            learning_rate=1e-3,
            max_grad_norm=1.0,
            lr_total_steps=4,
            compute_logits_in_trainer=False,
            max_seq_length=16,
        )
        trainer.dp_rank = 0
        trainer.grad_sync_mesh = None
        trainer.fsdp_cpu_offload = False
        trainer.target_lm_head = mock.MagicMock()
        trainer.prof = mock.MagicMock()

        module = eagle3_trainer_module
        with (
            mock.patch.object(module.dist, "get_rank", return_value=0),
            mock.patch.object(module.dist, "barrier"),
            mock.patch.object(module, "get_gloo_group", return_value=None),
            mock.patch.object(module, "apply_fsdp2", side_effect=lambda m, **kw: m),
            mock.patch.object(
                module, "fsdp2_load_full_state_dict", side_effect=lambda m, *a, **kw: m
            ),
            mock.patch.object(module, "BF16Optimizer"),
            mock.patch.object(module.checkpoint, "load", return_value=None),
            mock.patch.object(module.checkpoint, "finalize_load"),
        ):
            trainer.init_model(AutoDraftModelConfig.from_dict(dict(_DRAFT_CONFIG)), str(target_dir))
        return trainer

    def test_published_draft_is_loaded_and_target_embedding_wins(self):
        published, published_dir = self._publish_draft()
        target_embedding, target_dir = self._write_target()

        trainer = self._run_init_model(str(published_dir), target_dir)

        expected = published.state_dict()
        for key, value in trainer.draft_model.state_dict().items():
            if key == "embed_tokens.weight":
                continue
            self.assertTrue(torch.equal(value, expected[key]), msg=f"{key} was not restored")

        self.assertTrue(torch.equal(trainer.draft_model.embed_tokens.weight, target_embedding))
        self.assertFalse(trainer.draft_model.embed_tokens.weight.requires_grad)

    def test_unset_path_leaves_the_draft_freshly_initialized(self):
        published, _ = self._publish_draft()
        _, target_dir = self._write_target()

        trainer = self._run_init_model(None, target_dir)

        expected = published.state_dict()
        self.assertFalse(
            torch.equal(trainer.draft_model.lm_head.weight, expected["lm_head.weight"])
        )

    def test_bad_path_fails_before_training_starts(self):
        _, target_dir = self._write_target()

        with self.assertRaises(ValueError):
            self._run_init_model(str(self.root / "does-not-exist"), target_dir)


if __name__ == "__main__":
    unittest.main()
