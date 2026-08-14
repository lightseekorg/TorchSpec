"""Regression tests for training batch-size validation at config load.

Guards the empty-dispatch NCCL-hang root cause behind the issue #126 surface:
a ``training.micro_batch_size`` of 0 makes the derived ``dispatch_batch_size`` 0,
so ``try_dispatch_batch`` no-op-dispatches (returns True while queuing nothing)
and every rank blocks on the data-fetcher queue, surfacing as an NCCL
all-gather timeout. The validator rejects non-positive sizes at config load,
before any training process starts.
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
