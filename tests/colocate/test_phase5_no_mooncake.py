# Copyright (c) 2026 LightSeek Foundation
# MIT License

"""Phase 5 — assert the colocate path doesn't pull in Mooncake.

The plan in [`implementation.md` §Phase 5](../../docs/colocate/implementation.md)
says: "A clean colocate run leaves no Mooncake processes alive". This
test enforces a stronger structural property: when the colocate setup
function is the only one called, **no Mooncake C++ wrapper modules end
up in ``sys.modules``**.

We can't easily check the "no Mooncake processes alive" condition in
unit-test land (the master daemon runs as a subprocess), so we check
the import-time precondition. If Mooncake-bridge modules are imported,
that's strong evidence the runtime path will spin them up. If they're
not, the runtime path can't reach the daemon either — Mooncake bridges
into Python via these modules.

The Python-side ``torchspec.transfer.mooncake.utils`` is allowed to
exist in ``sys.modules`` because it's a thin shell that doesn't load
any C++ bridge until you actually call ``launch_mooncake_master`` or
``init_mooncake_store``. We don't: we want exact zero touches.

Note: the train_entry top-level module imports ``launch_mooncake_master``,
so any test that imports ``torchspec.train_entry`` will pull in the
Python wrapper transitively. This test therefore avoids importing
``train_entry`` and instead exercises the controller setup function
directly.
"""

from __future__ import annotations

import sys

import pytest

torch = pytest.importorskip("torch")


def _real_torch() -> bool:
    try:
        t = torch.zeros(2)
        return hasattr(t, "shape") and tuple(t.shape) == (2,)
    except Exception:
        return False


pytestmark = pytest.mark.skipif(
    not _real_torch(), reason="requires real torch (conftest stubs on Mac dev box)"
)


# Modules that, if loaded, indicate Mooncake's C++ runtime bridge has
# been touched. Any of these in `sys.modules` post-setup is a fail.
_MOONCAKE_RUNTIME_MODULES = (
    "mooncake_vllm_adaptor",
    "mooncake_master",
    # Mooncake's Python package itself (the "transfer engine" wrapper):
    "mooncake.engine",
    "mooncake.config",
    # The torchspec store wrapper (Phase 5 invariant: never touched):
    "torchspec.transfer.mooncake.eagle_store",
)


def _mooncake_runtime_modules_in_sys() -> list[str]:
    return [m for m in _MOONCAKE_RUNTIME_MODULES if m in sys.modules]


def test_colocate_setup_module_does_not_import_mooncake_runtime():
    """Importing ``setup`` must not pull Mooncake's C++ bridge modules.

    The ``setup`` module unconditionally imports
    ``AsyncInferenceManager`` and ``AsyncTrainingController`` and
    ``build_mooncake_config`` (because the disagg path needs them);
    that's fine — those are pure Python and don't touch the C++
    bridge until called.
    """
    pre = _mooncake_runtime_modules_in_sys()

    import torchspec.controller.setup  # noqa: F401

    post = _mooncake_runtime_modules_in_sys()
    new = sorted(set(post) - set(pre))
    assert new == [], (
        "Importing torchspec.controller.setup pulled Mooncake runtime "
        f"modules into sys.modules: {new}. The Phase 5 invariant requires "
        "the colocate path stay free of these bridges."
    )


def test_colocate_setup_function_signature_matches_async():
    """``setup_colocate_training_with_engines`` and the async sibling
    must have the same call surface for ``train_entry`` branching to be
    a clean swap."""
    import inspect

    from torchspec.controller.setup import (
        setup_async_training_with_engines,
        setup_colocate_training_with_engines,
    )

    async_sig = inspect.signature(setup_async_training_with_engines)
    colocate_sig = inspect.signature(setup_colocate_training_with_engines)

    # Colocate intentionally drops mooncake_config (one fewer positional
    # arg). The remaining params match by name.
    async_params = set(async_sig.parameters) - {"mooncake_config"}
    colocate_params = set(colocate_sig.parameters)
    assert async_params == colocate_params, (
        f"async params {async_params} != colocate params {colocate_params}"
    )


def test_colocate_setup_returns_none_inference_manager():
    """The runtime loop has to know to skip ``inference_manager``-only
    work in colocate mode. The contract is ``(controller, None)``;
    pin that here so a future refactor can't silently change it.

    Smoke-tests the docstring contract without standing up Ray
    actors — we just call the function with a stub controller and
    train_group that report what they're called with.
    """
    from unittest.mock import MagicMock

    from torchspec.controller.setup import setup_colocate_training_with_engines

    # Stub args namespace
    class _Args:
        training_num_nodes = 1
        training_num_gpus_per_node = 2
        per_dp_rank_batch_size = 1
        dp_size = 2

    train_group = MagicMock()
    # Stub controller — we pass it as `controller=` so the function
    # doesn't try to spawn a Ray actor.
    controller = MagicMock()
    controller.get_train_queues.remote.return_value = MagicMock()
    controller.get_eval_queues.remote.return_value = MagicMock()

    # ray.get returns whatever the .remote() call returned (also stubbed)
    import ray

    real_ray_get = ray.get
    try:
        ray.get = lambda x: x  # passthrough for test
        result_controller, manager = setup_colocate_training_with_engines(
            _Args(),
            train_group,
            inference_engines=[1, 2],
            controller=controller,
        )
    finally:
        ray.get = real_ray_get

    assert result_controller is controller
    assert manager is None, "colocate setup must return None for inference_manager"

    # And: train_group.set_train_queues was called with mooncake_config=None.
    train_group.set_train_queues.assert_called_once()
    _, kwargs = train_group.set_train_queues.call_args
    assert kwargs.get("mooncake_config") is None, kwargs
    train_group.set_eval_queues.assert_called_once()
    _, kwargs = train_group.set_eval_queues.call_args
    assert kwargs.get("mooncake_config") is None, kwargs
