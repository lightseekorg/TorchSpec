import importlib.abc
import importlib.machinery
import sys
import types
from unittest.mock import MagicMock, patch

# torch is installed, but numba/wandb/omegaconf/ray are not, and the root
# conftest only installs its mock finder when torch is ABSENT. So before any
# torchspec import we install a restricted meta-path finder for the missing
# tangential deps. flash_attn/cutlass are deliberately NOT mocked: llama3_eagle's
# own try/except must see flash_attn fail so it sets _flash_attn_fwd = None, the
# symbol this test patches explicitly (not mocked at import time).
_MISSING = {"numba", "wandb", "omegaconf", "ray"}


class _TangentialFinder(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path, target=None):
        top = fullname.split(".", 1)[0]
        if top not in _MISSING:
            return None
        return importlib.machinery.ModuleSpec(fullname, self, is_package=True)

    def create_module(self, spec):
        class _Proxy(types.ModuleType):
            def __getattr__(self, name):
                return MagicMock()

        proxy = _Proxy(spec.name)
        proxy.__path__ = []
        proxy.__package__ = spec.name
        if spec.name == "numba":
            # no-op @njit passthrough: supports both @njit and @njit(...) forms.
            def _njit(*args, **kwargs):
                if len(args) == 1 and callable(args[0]) and not kwargs:
                    return args[0]
                return lambda fn: fn

            proxy.njit = _njit
        return proxy

    def exec_module(self, module):
        pass


sys.meta_path.insert(0, _TangentialFinder())

import torch  # noqa: E402

import torchspec.models.draft.llama3_eagle as _mod  # noqa: E402
from torchspec.models.draft.llama3_eagle import _EagleMaskedFlashAttnFunc  # noqa: E402


def test_flash_attn_fwd_tolerates_trailing_returns():
    """Regression for issue #170.

    flash-attn 4.0.0b22 CUTE ``_flash_attn_fwd`` returns more than two values;
    the rigid ``out, lse = _flash_attn_fwd(...)`` unpack raised
    ``ValueError: too many values to unpack``. The fix uses
    ``out, lse, *_rest = _flash_attn_fwd(...)`` so trailing returns are
    tolerated (the backward consumes only the saved ``(q, k, v, out, lse)`` via
    ``ctx.save_for_backward``).

    On master this call raises ValueError (RED); on the fix branch it returns
    ``out`` cleanly (GREEN).
    """
    q = torch.randn(1, 8, 2, 8, dtype=torch.float32)
    k = torch.randn(1, 8, 2, 8, dtype=torch.float32)
    v = torch.randn(1, 8, 2, 8, dtype=torch.float32)

    def fake_fwd(*args, **kwargs):
        # Mimic CUTE returning a trailing extra value beyond (out, lse).
        return torch.randn_like(q), torch.randn(1, 2, 8, dtype=torch.float32), "extra_return"

    with (
        patch.object(_mod, "_flash_attn_fwd", fake_fwd),
        patch.object(_mod, "_get_block_sparse", return_value=None),
    ):
        out = _EagleMaskedFlashAttnFunc.apply(
            q,
            k,
            v,
            mask_mod_cute=None,
            mask_mod_flex=None,
            softmax_scale=1.0,
            max_seq_len=8,
            aux_tensors=None,
        )

    assert isinstance(out, torch.Tensor)
    assert out.shape == q.shape
