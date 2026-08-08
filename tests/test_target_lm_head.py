import json
from types import SimpleNamespace

import pytest
import torch
from safetensors.torch import save_file

from torchspec.models.target import target_utils
from torchspec.models.target.target_utils import TargetLMHead, load_synced_target_lm_head

NORM_KEY = "language_model.model.norm.weight"


def _multimodal_config(hidden_size: int = 8, vocab_size: int = 16):
    """A config whose text sub-config carries the sizes, as multimodal targets do."""
    text_config = SimpleNamespace(
        hidden_size=hidden_size,
        vocab_size=vocab_size,
        rms_norm_eps=1.0e-5,
        model_type="fake_multimodal_text",
    )
    return SimpleNamespace(text_config=text_config, model_type="fake_multimodal")


def _unbuildable_architecture(_self=None):
    raise ValueError("invalid remote module name")


def _write_norm_checkpoint(tmp_path, weight: torch.Tensor) -> None:
    shard = "model-00001-of-00001.safetensors"
    save_file({NORM_KEY: weight}, tmp_path / shard)
    (tmp_path / "model.safetensors.index.json").write_text(
        json.dumps({"weight_map": {NORM_KEY: shard}})
    )


def test_rms_norm_fallback_loads_weight_and_matches_reference(tmp_path, monkeypatch):
    config = _multimodal_config()
    target = TargetLMHead(config)
    monkeypatch.setattr(target, "_extract_norm_from_architecture", _unbuildable_architecture)
    weight = torch.linspace(0.5, 1.5, config.text_config.hidden_size)
    _write_norm_checkpoint(tmp_path, weight)

    target._init_and_load_norm(str(tmp_path), NORM_KEY)

    assert target.model_config is config
    assert target.config is config.text_config
    assert target.norm is not None
    torch.testing.assert_close(target.norm.weight, weight)

    hidden = torch.tensor(
        [[1.0, -2.0, 3.0, -4.0, 5.0, -6.0, 7.0, -8.0]],
        dtype=torch.bfloat16,
    )
    variance = hidden.float().pow(2).mean(dim=-1, keepdim=True)
    expected = weight * (
        hidden.float() * torch.rsqrt(variance + config.text_config.rms_norm_eps)
    ).to(hidden.dtype)
    torch.testing.assert_close(target.norm(hidden), expected)


def test_rms_norm_fallback_creates_matching_non_rank_zero_structure(monkeypatch):
    target = TargetLMHead(_multimodal_config())
    monkeypatch.setattr(target, "_extract_norm_from_architecture", _unbuildable_architecture)

    target._init_norm_structure()

    assert target.norm is not None
    torch.testing.assert_close(target.norm.weight, torch.ones_like(target.norm.weight))
    assert [tuple(param.shape) for param in target.parameters()] == [(16, 8), (8,)]


def test_explicit_norm_load_fails_closed_without_a_supported_norm(tmp_path, monkeypatch):
    config = SimpleNamespace(hidden_size=8, vocab_size=16, model_type="unsupported")
    target = TargetLMHead(config)
    monkeypatch.setattr(target, "_extract_norm_from_architecture", lambda: None)

    with pytest.raises(RuntimeError, match="Failed to load verifier norm"):
        target._init_and_load_norm(str(tmp_path), "model.norm.weight")

    assert target.norm is None


def test_explicit_norm_load_fails_closed_on_a_missing_checkpoint_key(tmp_path, monkeypatch):
    target = TargetLMHead(_multimodal_config())
    monkeypatch.setattr(target, "_extract_norm_from_architecture", _unbuildable_architecture)
    _write_norm_checkpoint(tmp_path, torch.ones(8))

    with pytest.raises(RuntimeError, match="Failed to load verifier norm"):
        target._init_and_load_norm(str(tmp_path), "model.norm.weight")

    assert target.norm is None


def test_norm_structure_creation_fails_closed_without_a_supported_norm(monkeypatch):
    target = TargetLMHead(SimpleNamespace(hidden_size=8, vocab_size=16, model_type="unsupported"))
    monkeypatch.setattr(target, "_extract_norm_from_architecture", lambda: None)

    with pytest.raises(RuntimeError, match="No final norm structure is available"):
        target._init_norm_structure()


def test_architecture_norm_is_preferred_over_the_config_fallback(monkeypatch):
    target = TargetLMHead(_multimodal_config())
    architecture_norm = torch.nn.RMSNorm(8, eps=1.0e-5, device="meta")
    monkeypatch.setattr(target, "_extract_norm_from_architecture", lambda: architecture_norm)

    target._init_norm_structure()

    assert isinstance(target.norm, torch.nn.RMSNorm)


class _RecordingDist:
    """Stands in for ``torch.distributed``, recording collectives in order.

    ``peer_failed`` simulates another rank reporting a failed initialization
    through the status all-reduce.
    """

    ReduceOp = SimpleNamespace(MIN="min")

    def __init__(self, rank: int, peer_failed: bool = False):
        self._rank = rank
        self._peer_failed = peer_failed
        self.calls: list[str] = []

    def get_rank(self) -> int:
        return self._rank

    def all_reduce(self, tensor, op=None) -> None:
        self.calls.append("all_reduce")
        if self._peer_failed:
            tensor.zero_()

    def broadcast(self, tensor, src: int = 0) -> None:
        self.calls.append("broadcast")


def _patch_dist(monkeypatch, rank: int, peer_failed: bool = False) -> _RecordingDist:
    fake_dist = _RecordingDist(rank=rank, peer_failed=peer_failed)
    monkeypatch.setattr(target_utils, "dist", fake_dist)
    return fake_dist


def _patch_follower_config(monkeypatch) -> None:
    monkeypatch.setattr(
        target_utils,
        "AutoConfig",
        SimpleNamespace(from_pretrained=lambda *args, **kwargs: _multimodal_config()),
    )


def test_rank_zero_load_failure_is_agreed_before_the_parameter_broadcast(monkeypatch):
    fake_dist = _patch_dist(monkeypatch, rank=0)

    def _fail(**kwargs):
        raise RuntimeError("Failed to load verifier norm key 'model.norm.weight'")

    monkeypatch.setattr(TargetLMHead, "from_pretrained", staticmethod(_fail))

    with pytest.raises(RuntimeError, match="Failed to load verifier norm"):
        load_synced_target_lm_head("/nonexistent/target", load_norm=True, device="cpu")

    # The status all-reduce has to happen even on the failing path, or the peer
    # ranks sit in it until the process-group timeout.
    assert fake_dist.calls == ["all_reduce"]


def test_rank_zero_missing_norm_fails_closed_after_the_status_all_reduce(monkeypatch):
    fake_dist = _patch_dist(monkeypatch, rank=0)
    head = TargetLMHead(_multimodal_config())
    monkeypatch.setattr(TargetLMHead, "from_pretrained", staticmethod(lambda **kwargs: head))

    with pytest.raises(RuntimeError, match="requires a loaded verifier norm"):
        load_synced_target_lm_head("/nonexistent/target", load_norm=True, device="cpu")

    assert fake_dist.calls == ["all_reduce"]


def test_follower_rank_reports_a_peer_failure_instead_of_broadcasting(monkeypatch):
    fake_dist = _patch_dist(monkeypatch, rank=1, peer_failed=True)
    _patch_follower_config(monkeypatch)

    with pytest.raises(RuntimeError, match="failed on another rank"):
        load_synced_target_lm_head("/nonexistent/target", load_norm=True, device="cpu")

    assert fake_dist.calls == ["all_reduce"]


def test_follower_rank_broadcasts_every_parameter_including_the_norm(monkeypatch):
    fake_dist = _patch_dist(monkeypatch, rank=1)
    _patch_follower_config(monkeypatch)

    head = load_synced_target_lm_head(
        "/nonexistent/target", load_norm=True, device="cpu", dtype=torch.float32
    )

    assert head.norm is not None
    assert fake_dist.calls == ["all_reduce", "broadcast", "broadcast"]
