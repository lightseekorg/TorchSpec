from unittest.mock import patch

import torch

import torchspec.models.draft.llama3_eagle as llama3_eagle
from torchspec.models.draft.llama3_eagle import _EagleMaskedFlashAttnFunc


def test_flash_attn_fwd_tolerates_trailing_returns():
    """Regression for #170.

    The CUTE ``_flash_attn_fwd`` returns more than ``(out, lse)``, which broke
    the rigid ``out, lse = _flash_attn_fwd(...)`` unpack. The fix unpacks with
    ``out, lse, *_rest`` so trailing returns are tolerated.
    """
    q = torch.randn(1, 8, 2, 8)
    k = torch.randn(1, 8, 2, 8)
    v = torch.randn(1, 8, 2, 8)

    def fake_fwd(*args, **kwargs):
        # Mimic CUTE returning a trailing value beyond (out, lse).
        return torch.randn_like(q), torch.randn(1, 2, 8), "extra_return"

    with (
        patch.object(llama3_eagle, "_flash_attn_fwd", fake_fwd),
        patch.object(llama3_eagle, "_get_block_sparse", return_value=None),
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
