# Copyright (c) 2026 LightSeek Foundation
# MIT License
#
# Public surface for the colocate (training + inference on the same GPU) mode.
# See docs/colocate/implementation.md for the phased plan and
# docs/colocate/knowledge.md for background concepts.

from torchspec.colocate.config import (
    SUPPORTED_COMBINATIONS,
    ColocateConfigError,
    is_colocate_enabled,
    is_mps_colocate,
    validate_colocate_config,
)

__all__ = [
    "ColocateConfigError",
    "SUPPORTED_COMBINATIONS",
    "is_colocate_enabled",
    "is_mps_colocate",
    "validate_colocate_config",
]
