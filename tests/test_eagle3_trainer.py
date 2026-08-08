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

"""Tests for Eagle3Trainer metric aggregation.

Verifies that per-depth accuracy and LK acceptance-rate (alpha) metrics are
weighted by valid-token counts across gradient-accumulation steps (and, in
_aggregate_eval_metrics, eval-cache chunks) rather than averaged naively —
chunks/microbatches can carry very different numbers of loss positions, so a
mean-of-means over/under-weights small ones relative to a true token-weighted
mean.
"""

import unittest
from unittest import mock

import torch

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


if __name__ == "__main__":
    unittest.main()
