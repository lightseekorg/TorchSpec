"""Independent training-reference masks and nonvacuous convergence criteria."""

from copy import deepcopy

import pytest

from tests.tokenizer_contract.support import expected_mask, load_cases
from tools.qualification.train_tokenizer_convergence import qualify, reference_sample


@pytest.mark.parametrize("name", ["ascii", "unicode", "whitespace", "empty"])
def test_training_reference_matches_reviewed_masks(name, reference_tokenizers):
    case = next(case for case in load_cases("qwen3") if case["name"] == name)
    for limit in range(1, len(case["input_ids"]) + 2):
        result = reference_sample(reference_tokenizers["qwen3"], case["messages"], limit)
        assert result["input_ids"] == case["input_ids"][:limit]
        assert result["loss_mask"] == expected_mask(case, True, False, limit)


def trajectories(steps=1000):
    run = {
        "initial_trainable_sha256": "same",
        "steps": [{"step": i, "loss": 10 - 8 * i / steps} for i in range(1, steps + 1)],
        "eval": [
            {"step": 0, "loss": 10, "accuracy_by_depth": [0.1, 0.05]},
            {"step": steps, "loss": 2, "accuracy_by_depth": [0.5, 0.3]},
        ],
    }
    return {"hf": deepcopy(run), "smg": deepcopy(run)}


def test_converged_matched_runs_pass():
    assert qualify(trajectories(), 1000)["qualified"]


def test_short_smoke_cannot_qualify_convergence():
    assert not qualify(trajectories(2), 2)["qualified"]


@pytest.mark.parametrize(
    "failure",
    [
        "flat_train",
        "flat_eval",
        "flat_accuracy",
        "different_init",
        "divergent_loss",
        "missing_step",
        "nan",
    ],
)
def test_invalid_training_cannot_pass(failure):
    runs = trajectories()
    smg = runs["smg"]
    if failure == "flat_train":
        for run in runs.values():
            for row in run["steps"]:
                row["loss"] = 10
    elif failure == "flat_eval":
        smg["eval"][-1]["loss"] = 10
    elif failure == "flat_accuracy":
        smg["eval"][-1]["accuracy_by_depth"][0] = 0.1
    elif failure == "different_init":
        smg["initial_trainable_sha256"] = "different"
    elif failure == "divergent_loss":
        smg["steps"][50]["loss"] *= 1.1
    elif failure == "missing_step":
        smg["steps"].pop()
    else:
        smg["steps"][50]["loss"] = float("nan")
    with pytest.raises(AssertionError):
        qualify(runs, 1000)
