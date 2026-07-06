# Copyright (c) 2026 LightSeek Foundation
#
# Licensed under the MIT License (see repository LICENSE / file headers).

"""Correctness tests for the Domino draft model (CPU/float32).

Verifies the new logic - the Domino causal-correction head and the base-anchored
curriculum loss - in isolation from flex attention/CUDA. The reused DFlash
backbone/anchor/mask code is covered by test_dflash.py.
"""

import pytest

torch = pytest.importorskip("torch")

from torchspec.models.domino import DominoModel  # noqa: E402
from torchspec.models.draft.auto import AutoDraftModelConfig  # noqa: E402
from torchspec.models.draft.domino import DominoConfig, DominoDraftModel  # noqa: E402
from torchspec.training.domino_trainer import DominoTrainer  # noqa: E402

DEV = torch.device("cpu")
DT = torch.float32
B, SEQ, BLOCK, NANCH = 2, 24, 4, 4


def _cfg():
    return DominoConfig(
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=1,
        num_attention_heads=4,
        num_key_value_heads=2,
        vocab_size=32,
        num_target_layers=2,
        target_hidden_size=32,
        target_num_hidden_layers=4,
        mask_token_id=31,
        gru_hidden_size=16,
        correction_rank=8,
    )


def _build(cfg, *, loss_objective="decay", loss_decay_gamma=7.0):
    draft = DominoDraftModel(cfg).to(DEV, DT)
    return DominoModel(
        draft_model=draft,
        block_size=BLOCK,
        num_anchors=NANCH,
        loss_objective=loss_objective,
        dpace_alpha=0.5,
        loss_decay_gamma=loss_decay_gamma,
    ).to(DEV, DT)


def _batch(cfg):
    torch.manual_seed(123)
    return (
        torch.randint(0, cfg.vocab_size, (B, SEQ), device=DEV),
        [
            torch.randn(B, SEQ, cfg.target_hidden_size, device=DEV, dtype=DT)
            for _ in range(cfg.num_target_layers)
        ],
        torch.ones(B, SEQ, device=DEV, dtype=DT),
        torch.randn(cfg.vocab_size, cfg.hidden_size, device=DEV, dtype=DT) * 0.02,
    )


def _fwd(model, batch, lam):
    model.curriculum_lambda = lam
    return model(batch[0], batch[1], batch[2], batch[3])


def test_domino_json_resolves_to_domino_config():
    cfg = AutoDraftModelConfig.from_file("torchspec/config/domino_draft_config.json")
    assert isinstance(cfg, DominoConfig)


def test_head_present_and_trainable():
    model = _build(_cfg())
    for name in ("causal_gru", "correction_w1", "correction_w2"):
        module = getattr(model.draft_model, name)
        params = list(module.parameters())
        assert params and all(p.requires_grad for p in params)


def test_forward_output_contract():
    cfg = _cfg()
    model = _build(cfg)
    loss, acc, loss_pp, acc_pp, count_pp, aux_metrics = _fwd(model, _batch(cfg), 0.0)
    assert loss.ndim == 0 and acc.ndim == 0
    assert loss_pp.shape == (BLOCK,) and acc_pp.shape == (BLOCK,)
    assert count_pp.shape == (BLOCK,)
    assert torch.isfinite(loss)
    assert set(aux_metrics) == {
        "base_loss",
        "final_loss",
        "correction_norm",
        "correction_abs_mean",
    }
    assert all(torch.isfinite(v) for v in aux_metrics.values())


def test_trainer_forward_preserves_loss_components_slot(monkeypatch):
    cfg = _cfg()
    model = _build(cfg)
    trainer = object.__new__(DominoTrainer)
    trainer.model = model
    trainer.target_lm_head_weight = torch.randn(
        cfg.vocab_size, cfg.hidden_size, device=DEV, dtype=DT
    )
    trainer.num_target_layers = cfg.num_target_layers

    monkeypatch.setattr("torchspec.training.domino_trainer.torch.device", lambda _: DEV)

    input_ids, hidden_states_list, loss_mask, _ = _batch(cfg)
    batch = {
        "input_ids": input_ids,
        "hidden_states": torch.cat(hidden_states_list, dim=-1),
        "loss_mask": loss_mask,
    }
    output = DominoTrainer._forward(trainer, batch)

    assert len(output) == 6
    assert set(output[-1]) == {
        "base_loss",
        "final_loss",
        "correction_norm",
        "correction_abs_mean",
    }


def test_dpace_objective_changes_domino_loss_weights():
    cfg = _cfg()
    batch = _batch(cfg)
    torch.manual_seed(7)
    decay_model = _build(cfg, loss_objective="decay", loss_decay_gamma=None)
    torch.manual_seed(7)
    dpace_model = _build(cfg, loss_objective="dpace")
    dpace_model.load_state_dict(decay_model.state_dict())

    torch.manual_seed(11)
    decay_loss = _fwd(decay_model, batch, 0.0)[0]
    torch.manual_seed(11)
    dpace_loss = _fwd(dpace_model, batch, 0.0)[0]

    assert torch.isfinite(dpace_loss)
    assert not torch.allclose(decay_loss, dpace_loss)


def test_curriculum_lambda_is_noop_when_correction_zeroed():
    cfg = _cfg()
    model = _build(cfg)
    batch = _batch(cfg)
    with torch.no_grad():
        model.draft_model.correction_w2.weight.zero_()
    torch.manual_seed(7)
    loss_base = _fwd(model, batch, 1.0)[0].item()
    torch.manual_seed(7)
    loss_final = _fwd(model, batch, 0.0)[0].item()
    assert abs(loss_base - loss_final) < 1e-5


def test_curriculum_selects_base_vs_final_when_correction_active():
    cfg = _cfg()
    model = _build(cfg)
    batch = _batch(cfg)
    torch.manual_seed(7)
    base = _fwd(model, batch, 1.0)[0].item()
    torch.manual_seed(7)
    final = _fwd(model, batch, 0.0)[0].item()
    assert abs(base - final) > 1e-4


def test_gradients_flow_to_domino_head():
    cfg = _cfg()
    model = _build(cfg)
    model.zero_grad(set_to_none=True)
    _fwd(model, _batch(cfg), 0.0)[0].backward()
    for name in ("causal_gru", "correction_w1", "correction_w2"):
        module = getattr(model.draft_model, name)
        grad_norm = sum(p.grad.norm().item() for p in module.parameters() if p.grad is not None)
        assert grad_norm > 0, name


def test_model_learns_under_curriculum():
    torch.manual_seed(0)
    cfg = _cfg()
    model = _build(cfg)
    batch = _batch(cfg)
    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=3e-3,
    )
    steps = 200
    first_loss = last_loss = first_acc = last_acc = None
    for step in range(steps):
        lam = max(0.0, 1.0 - step / (steps * 0.5))
        optimizer.zero_grad(set_to_none=True)
        loss, acc, *_ = _fwd(model, batch, lam)
        loss.backward()
        optimizer.step()
        if step == 0:
            first_loss, first_acc = loss.item(), acc.item()
        last_loss, last_acc = loss.item(), acc.item()

    assert last_loss < first_loss * 0.5
    assert last_acc > first_acc + 0.2
