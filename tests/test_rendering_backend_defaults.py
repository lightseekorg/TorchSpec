"""Default selection, explicit legacy rollback and unsupported-mode boundaries."""

import json
import sys
from types import SimpleNamespace

import pytest
from omegaconf import OmegaConf

from torchspec.config.train_config import Config, config_to_flat_args, load_config, save_config
from torchspec.data.renderers.smg import effective_renderer, prepare_checkpoint


@pytest.mark.parametrize("template", ["llama3", "qwen", "qwen3-thinking", "minimax-m2"])
def test_supported_text_profiles_default_to_smg(template):
    args = SimpleNamespace(chat_template=template)
    assert effective_renderer(args) == "smg"
    args.renderer = "smg"
    assert effective_renderer(args) == "smg"
    args.rendering_backend = "huggingface"
    assert effective_renderer(args) is None


@pytest.mark.parametrize("identification", ["renderer", "model_name", "local_config"])
def test_k3_defaults_and_explicit_rollback(identification, tmp_path):
    args = SimpleNamespace()
    if identification == "renderer":
        args.renderer = "kimi-k3"
    elif identification == "model_name":
        args.target_model_path = "org/Kimi-K3"
    else:
        (tmp_path / "config.json").write_text(json.dumps({"model_type": "kimi_k3"}))
        args.target_model_path = str(tmp_path)
    assert effective_renderer(args) == "smg-kimi-k3"
    args.renderer = "smg-kimi-k3"
    assert effective_renderer(args) == "smg-kimi-k3"
    args.rendering_backend = "huggingface"
    assert effective_renderer(args) == "kimi-k3"
    args.rendering_backend = "auto"
    assert effective_renderer(args) == "smg-kimi-k3"
    args.rendering_backend = "smg"
    assert effective_renderer(args) == "smg-kimi-k3"


@pytest.mark.parametrize("mode", ["defer_tokenization", "train_with_decode"])
def test_unsupported_modes_keep_legacy_routing(mode):
    cfg = OmegaConf.structured(Config)
    cfg.model.target_model_path = "Qwen/Qwen3-8B"
    cfg.dataset.chat_template = "qwen"
    (cfg.dataset if mode == "defer_tokenization" else cfg.training)[mode] = True
    args = config_to_flat_args(cfg)
    assert args.renderer is None
    assert args.dynamic_loss_mask is (mode == "defer_tokenization")
    cfg.dataset.rendering_backend = "smg"
    with pytest.raises(ValueError, match="requires"):
        config_to_flat_args(cfg)


@pytest.mark.parametrize(
    "template", ["gpt-oss", "deepseek-r1-distill", "qwen2-vl", "kimi-k25-vlm", "custom-template"]
)
def test_nondefault_profiles_keep_legacy(template):
    assert effective_renderer(SimpleNamespace(chat_template=template)) is None


def test_custom_renderer_is_preserved_even_on_a_k3_model():
    args = SimpleNamespace(renderer="custom", target_model_path="org/Kimi-K3")
    assert effective_renderer(args) == "custom"
    args.rendering_backend = "smg"
    with pytest.raises(ValueError, match="cannot override"):
        effective_renderer(args)


def test_missing_binding_requires_install_or_explicit_legacy(monkeypatch):
    monkeypatch.setitem(sys.modules, "torchspec_smg", None)
    args = SimpleNamespace(chat_template="qwen")
    assert effective_renderer(args) == "smg"
    with pytest.raises(ImportError, match="rendering_backend=huggingface"):
        prepare_checkpoint("unused")
    args.rendering_backend = "huggingface"
    assert effective_renderer(args) is None


@pytest.mark.parametrize("backend", ["auto", "smg", "huggingface"])
def test_saved_config_preserves_backend_and_mask_routing(tmp_path, backend):
    cfg = OmegaConf.structured(Config)
    cfg.model.target_model_path = "org/Kimi-K3"
    cfg.dataset.renderer = "kimi-k3"
    cfg.dataset.rendering_backend = backend
    path = tmp_path / "config.yaml"
    save_config(cfg, str(path))
    reloaded = load_config(config_path=str(path))
    args = config_to_flat_args(reloaded)
    assert reloaded.dataset.rendering_backend == backend
    assert args.renderer == ("kimi-k3" if backend == "huggingface" else "smg-kimi-k3")
    assert args.dynamic_loss_mask is (backend == "huggingface")


def test_outdated_binding_is_not_silently_used(monkeypatch):
    monkeypatch.setitem(
        sys.modules, "torchspec_smg", SimpleNamespace(__version__="0.1.0", build_features="default")
    )
    with pytest.raises(ImportError, match="binding >=0.1.2"):
        prepare_checkpoint("unused")
