"""Published DFlash backbone initialization for DFlash2 finetuning."""

import pytest
import torch
from safetensors.torch import save_file

from torchspec.models.draft.dflash2 import DFlash2Config, DFlash2DraftModel
from torchspec.models.draft.keymap import to_export_keys
from torchspec.training.checkpoint import load_initial_dflash_weights


def model():
    return DFlash2DraftModel(
        DFlash2Config(
            hidden_size=16,
            intermediate_size=32,
            num_hidden_layers=1,
            num_attention_heads=2,
            num_key_value_heads=1,
            vocab_size=32,
            max_position_embeddings=64,
            num_target_layers=4,
            target_layer_ids=[1, 3],
            block_size=16,
            conv_group_size=4,
            selector_rank=4,
            selector_top_k=4,
            mask_token_id=31,
        )
    )


def plain_weights(draft):
    return to_export_keys(
        {
            k: v.clone() + 0.125
            for k, v in draft.state_dict().items()
            if k != "embed_tokens.weight"
            and ".attention_conv." not in k
            and ".mlp_conv." not in k
            and not k.startswith("candidate_selector.")
        }
    )


def test_shared_weights_loaded_new_modules_retained(tmp_path):
    draft = model()
    before = {k: v.clone() for k, v in draft.state_dict().items()}
    weights = plain_weights(draft)
    path = tmp_path / "model.safetensors"
    save_file(weights, path)
    load_initial_dflash_weights(draft, str(path))
    after = draft.state_dict()
    for key in before:
        if (
            key == "embed_tokens.weight"
            or ".attention_conv." in key
            or ".mlp_conv." in key
            or key.startswith("candidate_selector.")
        ):
            assert torch.equal(after[key], before[key])
        else:
            assert torch.equal(after[key], before[key] + 0.125)
    conv = draft.layers[0].attention_conv
    x = torch.randn(1, 16, 16)
    prepared, kernel = conv.prepare(x)
    assert torch.equal(prepared, x)
    assert torch.equal(conv.finish(x, kernel), x)


@pytest.mark.parametrize(
    "fault", ["missing_backbone", "bad_shape", "unexpected", "partial_dflash2"]
)
def test_reject_incomplete_or_incompatible_weights(tmp_path, fault):
    draft = model()
    weights = plain_weights(draft)
    if fault == "missing_backbone":
        del weights["fc.weight"]
    elif fault == "bad_shape":
        weights["fc.weight"] = weights["fc.weight"][:1]
    elif fault == "unexpected":
        weights["unknown.weight"] = torch.ones(1)
    else:
        weights["candidate_selector.hidden_projection.weight"] = (
            draft.candidate_selector.hidden_projection.weight.detach().clone()
        )
    path = tmp_path / "model.safetensors"
    save_file(weights, path)
    with pytest.raises(ValueError):
        load_initial_dflash_weights(draft, str(path))


def test_complete_dflash2_checkpoint_loads_selector(tmp_path):
    draft = model()
    weights = to_export_keys(
        {k: v.clone() for k, v in draft.state_dict().items() if k != "embed_tokens.weight"}
    )
    path = tmp_path / "model.safetensors"
    save_file(weights, path)
    other = model()
    load_initial_dflash_weights(other, str(path))
    assert torch.equal(
        other.candidate_selector.predecessor_codebook, draft.candidate_selector.predecessor_codebook
    )
