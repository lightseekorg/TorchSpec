"""Strict weight migration from DFlash to DSpark and EAGLE3 to EAGLE3.1."""

import pytest
import torch
from safetensors.torch import save_file
from transformers import LlamaConfig
from transformers.models.deepseek_v3.configuration_deepseek_v3 import DeepseekV3Config

from torchspec.models.draft.deepseek_eagle import Eagle3DeepseekV2ForCausalLM
from torchspec.models.draft.dflash import DFlashConfig, DFlashDraftModel
from torchspec.models.draft.dspark import DSparkConfig, DSparkDraftModel
from torchspec.models.draft.keymap import to_export_keys
from torchspec.models.draft.llama3_eagle import LlamaForCausalLMEagle3
from torchspec.training.checkpoint import load_initial_dflash_weights, load_initial_draft_weights

FAMILIES = ["dspark", "eagle_gqa", "eagle_mla"]


def build(family, upgraded, seed=0, *, markov=True, confidence=True):
    torch.manual_seed(seed)
    kwargs = dict(
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=1,
        num_attention_heads=4,
        num_key_value_heads=2,
        vocab_size=64,
        max_position_embeddings=128,
        rms_norm_eps=1e-6,
        pad_token_id=0,
    )
    if family == "dspark":
        kwargs.update(
            num_target_layers=2, target_hidden_size=32, target_num_hidden_layers=8, mask_token_id=63
        )
        if not upgraded:
            return DFlashDraftModel(DFlashConfig(**kwargs))
        return DSparkDraftModel(
            DSparkConfig(
                **kwargs,
                markov_rank=8 if markov else 0,
                enable_confidence_head=confidence,
                confidence_head_with_markov=markov,
            )
        )
    kwargs.update(
        draft_vocab_size=32,
        fc_norm=upgraded,
        norm_output=upgraded,
        num_aux_hidden_states=3,
        target_hidden_size=32,
    )
    if family == "eagle_gqa":
        return LlamaForCausalLMEagle3(LlamaConfig(**kwargs))
    kwargs.update(
        q_lora_rank=16,
        kv_lora_rank=16,
        qk_nope_head_dim=8,
        qk_rope_head_dim=4,
        v_head_dim=8,
        n_routed_experts=1,
        n_shared_experts=0,
        first_k_dense_replace=0,
        num_experts_per_tok=1,
    )
    return Eagle3DeepseekV2ForCausalLM(DeepseekV3Config(**kwargs))


def upgrade_key(key):
    return key.startswith(("markov_head.", "confidence_head.", "fc_norm."))


def load(model, path, family):
    loader = load_initial_dflash_weights if family == "dspark" else load_initial_draft_weights
    return loader(model, str(path))


def publish(state, path, exported=True):
    save_file(to_export_keys(state) if exported else state, path)


@pytest.mark.parametrize("family", FAMILIES)
@pytest.mark.parametrize("exported", [False, True])
@pytest.mark.parametrize("include_embedding", [False, True])
def test_base_checkpoint_preserves_shared_weights_and_new_initialization(
    tmp_path, family, exported, include_embedding
):
    source = build(family, False)
    # Give pruned-vocabulary buffers meaningful values to catch accidental resets.
    if family != "dspark":
        source.d2t.copy_(torch.arange(32))
        source.t2d[:32] = False
    target = build(family, True, seed=1)
    before = {k: v.clone() for k, v in target.state_dict().items()}
    state = {
        k: v
        for k, v in source.state_dict().items()
        if include_embedding or k != "embed_tokens.weight"
    }
    path = tmp_path / "model.safetensors"
    publish(state, path, exported)
    assert load(target, path, family) == path
    for k, v in target.state_dict().items():
        assert torch.equal(v, state[k] if k in state else before[k]), k
    if family != "dspark":
        assert target.norm_output
        assert all(torch.equal(n.weight, torch.ones_like(n.weight)) for n in target.fc_norm)


@pytest.mark.parametrize("family", FAMILIES)
def test_complete_upgraded_checkpoint_restores_all_new_weights(tmp_path, family):
    source = build(family, True)
    with torch.no_grad():
        for k, v in source.state_dict().items():
            if upgrade_key(k):
                v.fill_(0.37)
    target = build(family, True, seed=1)
    state = {k: v for k, v in source.state_dict().items() if k != "embed_tokens.weight"}
    path = tmp_path / "model.safetensors"
    publish(state, path)
    load(target, path, family)
    for k, v in state.items():
        assert torch.equal(target.state_dict()[k], v), k


@pytest.mark.parametrize("family", FAMILIES)
@pytest.mark.parametrize("fault", ["partial_upgrade", "missing_backbone", "shape", "unexpected"])
def test_bad_checkpoint_is_rejected_without_modifying_model(tmp_path, family, fault):
    source = build(family, False)
    target = build(family, True, seed=1)
    before = {k: v.clone() for k, v in target.state_dict().items()}
    state = dict(source.state_dict())
    backbone_key = next(k for k in state if k.endswith("weight") and k != "embed_tokens.weight")
    if fault == "partial_upgrade":
        k = next(k for k in before if upgrade_key(k))
        state[k] = before[k]
    elif fault == "missing_backbone":
        del state[backbone_key]
    elif fault == "shape":
        state[backbone_key] = state[backbone_key][:1]
    else:
        state["unknown.weight"] = torch.ones(1)
    path = tmp_path / "model.safetensors"
    publish(state, path)
    with pytest.raises((RuntimeError, ValueError)):
        load(target, path, family)
    assert all(torch.equal(target.state_dict()[k], v) for k, v in before.items())


@pytest.mark.parametrize("markov,confidence", [(True, False), (False, True), (False, False)])
def test_dspark_respects_disabled_heads(tmp_path, markov, confidence):
    source = build("dspark", False)
    target = build("dspark", True, seed=1, markov=markov, confidence=confidence)
    path = tmp_path / "model.safetensors"
    publish(source.state_dict(), path)
    load(target, path, "dspark")
    for k, v in source.state_dict().items():
        assert torch.equal(target.state_dict()[k], v)


@pytest.mark.parametrize("family", ["eagle_gqa", "eagle_mla"])
@pytest.mark.parametrize("missing", ["lm_head.weight", "d2t", "t2d"])
def test_eagle_upgrade_still_requires_head_and_vocabulary_mapping(tmp_path, family, missing):
    state = dict(build(family, False).state_dict())
    del state[missing]
    path = tmp_path / "model.safetensors"
    publish(state, path)
    with pytest.raises(RuntimeError, match="missing"):
        load(build(family, True), path, family)


@pytest.mark.parametrize("family", FAMILIES)
def test_upgraded_checkpoint_cannot_silently_downgrade(tmp_path, family):
    path = tmp_path / "model.safetensors"
    publish(build(family, True).state_dict(), path)
    with pytest.raises((RuntimeError, ValueError), match="unexpected"):
        load(build(family, False), path, family)
