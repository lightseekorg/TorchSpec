# Copyright (c) 2026 LightSeek Foundation
# MIT License

"""Deterministic-seed plumbing for the colocate grad-parity test.

The Phase-7 ``test_grad_parity_full`` compares per-parameter draft-model
gradients between the disaggregated (Mooncake) baseline and the colocate
(NCCL/gloo) path. For that comparison to be meaningful both arms must be
bit-reproducible given a single ``training.seed``.

The colocate engine runs the target model in *prefill-only* mode
(``max_new_tokens=0``) so there is **no sampling RNG** to worry about —
the hidden states it ships are a deterministic function of the input
ids. What remains:

* model init (draft model weights, optimizer state) — seeded by
  ``torch.manual_seed`` already, here promoted to seed numpy/random too;
* per-step kernels — pinned with ``use_deterministic_algorithms`` and
  the cuBLAS workspace env var, but only under :func:`is_grad_parity_mode`
  so production throughput is untouched;
* data order — the grad-parity test additionally passes
  ``training.shuffle_dataset=false`` so prompt order is the dataset's
  file order in both arms.

``seed_everything`` is safe to call from the driver, the controller, and
every trainer/engine actor; it is idempotent.
"""

from __future__ import annotations

import os

# Env var the grad-parity test sets on both arms. When set, seed_everything
# additionally engages the strict (slower) deterministic kernels.
_GRAD_PARITY_ENV = "TORCHSPEC_GRAD_PARITY"


def is_grad_parity_mode() -> bool:
    """True when the run is a grad-parity arm (``TORCHSPEC_GRAD_PARITY=1``)."""
    return os.environ.get(_GRAD_PARITY_ENV, "").strip().lower() in (
        "1",
        "true",
        "yes",
    )


def seed_everything(seed: int, *, strict: bool | None = None) -> None:
    """Seed every RNG that can affect draft-model gradients.

    Args:
        seed: the integer seed (``training.seed``).
        strict: when True, also pin deterministic kernels
            (``torch.use_deterministic_algorithms`` + the cuBLAS
            workspace env var). Defaults to :func:`is_grad_parity_mode`
            so normal runs keep their fast non-deterministic kernels.
    """
    if strict is None:
        strict = is_grad_parity_mode()

    seed = int(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)

    import random

    random.seed(seed)

    try:  # numpy is a hard dep in practice but guard for stub envs
        import numpy as np

        np.random.seed(seed)
    except Exception:  # pragma: no cover - numpy always present in real runs
        pass

    try:
        import torch
    except Exception:  # pragma: no cover - conftest stub / no-torch unit env
        return

    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    if strict:
        # cuBLAS needs a fixed workspace for deterministic GEMMs; this must
        # be set before the first CUDA context use, hence also exported so
        # child processes (sglang TP scheduler) inherit it.
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
        try:
            # warn_only: a handful of ops have no deterministic kernel; we
            # do not want the parity run to hard-crash on those — the
            # comparison tolerance absorbs them.
            torch.use_deterministic_algorithms(True, warn_only=True)
        except Exception:  # pragma: no cover - older torch
            pass

    try:
        from torchspec.utils.logging import logger

        logger.info("[determinism] seeded everything (seed=%d, strict=%s)", seed, strict)
    except Exception:  # pragma: no cover
        pass
