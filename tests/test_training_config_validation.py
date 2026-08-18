"""Regression tests for training-config validation at config load.

Guards the empty-dispatch NCCL-hang root cause behind the issue #126 surface:
a ``training.micro_batch_size`` of 0 makes the derived ``dispatch_batch_size`` 0,
so ``try_dispatch_batch`` no-op-dispatches (returns True while queuing nothing)
and every rank blocks on the data-fetcher queue, surfacing as an NCCL
all-gather timeout. The validator rejects non-positive sizes at config load,
before any training process starts.

Extends the same fail-closed-at-load idiom (PR #171) to four more numeric
fields whose non-positive values currently fail *silently* (flat loss from
``learning_rate=0`` / ``max_grad_norm=0``, sign-flipped gradients from
``max_grad_norm<0``, inference-pool starvation from ``inference_batch_size=0``)
or crash *opaquely and late* (post-init ``ZeroDivisionError`` / bare
``total_steps`` assert from ``draft_accumulation_steps<=0``). Rejecting at
load surfaces the misconfig before Ray/mooncake/vLLM init.
"""

import pytest
from omegaconf import OmegaConf

from torchspec.config.train_config import Config, load_config


def _resolved_config(micro_batch_size: int):
    """Build a fully-resolved config from the schema defaults + a batch-size override."""
    config = OmegaConf.structured(Config)
    config.training.micro_batch_size = micro_batch_size
    return config


def test_load_config_rejects_zero_micro_batch_size():
    """micro_batch_size=0 must fail at load: empty dispatch -> NCCL hang."""
    base = _resolved_config(micro_batch_size=0)
    with pytest.raises(ValueError, match="micro_batch_size"):
        load_config(base_config=base)


def test_load_config_rejects_negative_micro_batch_size():
    """Negative sizes are rejected for the same reason."""
    base = _resolved_config(micro_batch_size=-4)
    with pytest.raises(ValueError, match="micro_batch_size"):
        load_config(base_config=base)


def test_load_config_accepts_positive_micro_batch_size():
    """Positive control: a valid size loads without raising."""
    base = _resolved_config(micro_batch_size=2)
    config = load_config(base_config=base)
    assert config.training.micro_batch_size == 2


def test_validate_training_batch_config_raises_on_zero():
    """Directly exercise the module-level validator, not just the load_config wiring."""
    from torchspec.config.train_config import _validate_training_batch_config

    config = _resolved_config(micro_batch_size=0)
    with pytest.raises(ValueError, match="micro_batch_size"):
        _validate_training_batch_config(config)


def test_validate_training_batch_config_accepts_positive():
    """Positive control for the direct validator call."""
    from torchspec.config.train_config import _validate_training_batch_config

    config = _resolved_config(micro_batch_size=8)
    _validate_training_batch_config(config)  # must not raise


# --- Numeric training-config fields (draft_accumulation_steps, learning_rate,
#     max_grad_norm) and inference_batch_size — PR #171 idiom extended. ---


def _resolved_training_config(**overrides):
    """Build a fully-resolved config from the schema defaults + training overrides."""
    config = OmegaConf.structured(Config)
    for key, value in overrides.items():
        setattr(config.training, key, value)
    return config


def _resolved_inference_config(inference_batch_size: int):
    """Build a fully-resolved config from the schema defaults + an inference-batch override."""
    config = OmegaConf.structured(Config)
    config.inference.inference_batch_size = inference_batch_size
    return config


def test_load_config_rejects_zero_draft_accumulation_steps():
    """draft_accumulation_steps=0 must fail at load: propagates to global_batch_size=0
    and crashes post-init (ZeroDivisionError or bare total_steps assert)."""
    base = _resolved_training_config(draft_accumulation_steps=0)
    with pytest.raises(ValueError, match="draft_accumulation_steps"):
        load_config(base_config=base)


def test_load_config_rejects_negative_draft_accumulation_steps():
    """Negative values propagate the same way and must be rejected at load."""
    base = _resolved_training_config(draft_accumulation_steps=-2)
    with pytest.raises(ValueError, match="draft_accumulation_steps"):
        load_config(base_config=base)


def test_load_config_rejects_zero_learning_rate():
    """learning_rate=0 yields silent flat loss (AdamW+scheduler at lr=0) and an
    untrained checkpoint; reject at load."""
    base = _resolved_training_config(learning_rate=0)
    with pytest.raises(ValueError, match="learning_rate"):
        load_config(base_config=base)


def test_load_config_rejects_negative_learning_rate():
    """Negative learning rates hit a late scheduler assert; reject at load."""
    base = _resolved_training_config(learning_rate=-1e-4)
    with pytest.raises(ValueError, match="learning_rate"):
        load_config(base_config=base)


def test_load_config_rejects_zero_max_grad_norm():
    """max_grad_norm=0 zeroes all grads via clip_grad_norm_ (silent flat loss);
    reject at load."""
    base = _resolved_training_config(max_grad_norm=0)
    with pytest.raises(ValueError, match="max_grad_norm"):
        load_config(base_config=base)


def test_load_config_rejects_negative_max_grad_norm():
    """Negative max_grad_norm sign-flips grads (silent gradient ascent); reject at load."""
    base = _resolved_training_config(max_grad_norm=-0.5)
    with pytest.raises(ValueError, match="max_grad_norm"):
        load_config(base_config=base)


def test_load_config_rejects_zero_inference_batch_size():
    """inference_batch_size=0 starves the inference pool (silent dispatch spin) and
    vLLM rejects max_num_seqs=0 late after init; reject at load."""
    base = _resolved_inference_config(inference_batch_size=0)
    with pytest.raises(ValueError, match="inference_batch_size"):
        load_config(base_config=base)


def test_load_config_rejects_negative_inference_batch_size():
    """Negative inference batch sizes starve the pool the same way; reject at load."""
    base = _resolved_inference_config(inference_batch_size=-1)
    with pytest.raises(ValueError, match="inference_batch_size"):
        load_config(base_config=base)
