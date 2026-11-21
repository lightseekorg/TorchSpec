# Copyright (c) 2026 LightSeek Foundation
# MIT License

from datetime import timedelta
from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.distributed.tensor import DTensor

from torchspec.training import checkpoint
from torchspec.training.fsdp import apply_fsdp2
from torchspec.training.optimizer import BF16Optimizer
from torchspec.utils.distributed import init_gloo_group


def _make_trainer(root, device, strategy=None, continual=False):
    model = torch.nn.Linear(8, 8, dtype=torch.bfloat16, device=device)
    if device.type == "cpu":
        # A trainable parameter with no gradient must keep its lazy Adam state.
        model.register_parameter(
            "unused", torch.nn.Parameter(torch.ones(8, device=device, dtype=torch.bfloat16))
        )
    args = SimpleNamespace(
        checkpoint_dir=root,
        load_path=root,
        continual_training=continual,
        start_step=0,
        fsdp_strategy=strategy,
    )
    if strategy:
        mesh = dist.device_mesh.init_device_mesh("cuda", (dist.get_world_size(),))
        model = apply_fsdp2(model, mesh=mesh, args=args)
    optimizer = BF16Optimizer(model, lr=0.01 if not continual else 0.025, total_steps=16)
    return SimpleNamespace(
        model=model,
        optimizer=optimizer,
        lr_scheduler=optimizer.lr_scheduler,
        args=args,
        global_step=0,
    )


def _update(trainer, device, batch=None):
    if batch is None:
        batch = torch.rand(4, 8, device=device, dtype=torch.bfloat16)
    trainer.model(batch).float().square().mean().backward()
    trainer.optimizer.step()
    trainer.global_step += 1


def _assert_equal(actual, expected):
    if isinstance(actual, DTensor):
        assert isinstance(expected, DTensor) and actual.placements == expected.placements
        actual, expected = actual.to_local(), expected.to_local()
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)


def _check_weights(trainer, reference):
    for opt_attr in ("model_params", "fp32_params"):
        for actual, expected in zip(
            getattr(trainer.optimizer, opt_attr),
            getattr(reference.optimizer, opt_attr),
            strict=True,
        ):
            _assert_equal(actual, expected)


def _check_resume(trainer, reference):
    _check_weights(trainer, reference)
    actual = trainer.optimizer.state_dict()
    expected = reference.optimizer.state_dict()
    assert len(actual["state"]) == len(expected["state"]) == 2
    assert actual["state"].keys() == expected["state"].keys()
    for param_id, slots in actual["state"].items():
        assert slots.keys() == expected["state"][param_id].keys()
        for key, value in slots.items():
            _assert_equal(value, expected["state"][param_id][key])
    assert actual["param_groups"] == expected["param_groups"]
    assert trainer.lr_scheduler.state_dict() == reference.lr_scheduler.state_dict()
    assert trainer.global_step == reference.global_step


def _round_trip(root, device, strategy=None):
    torch.manual_seed(42)
    reference = _make_trainer(root, device, strategy)
    _update(reference, device)
    _update(reference, device)
    torch.manual_seed(700 + dist.get_rank())
    checkpoint.save(reference, step=1)
    cpu_rng = torch.get_rng_state()
    cuda_rng = torch.cuda.get_rng_state() if device.type == "cuda" else None

    # Continual training takes the checkpoint weights and fp32 masters but
    # retains its own schedule, random stream and empty optimizer state.
    continual = _make_trainer(root, device, strategy, continual=True)
    schedule = continual.lr_scheduler.state_dict()
    groups = continual.optimizer.state_dict()["param_groups"]
    fresh_rng = torch.get_rng_state()
    fresh_cuda = torch.cuda.get_rng_state() if cuda_rng is not None else None
    checkpoint.finalize_load(continual, checkpoint.load(continual))
    assert torch.equal(torch.get_rng_state(), fresh_rng)
    if fresh_cuda is not None:
        assert torch.equal(torch.cuda.get_rng_state(), fresh_cuda)
    assert (
        not continual.optimizer.state and continual.optimizer.state_dict()["param_groups"] == groups
    )
    assert continual.lr_scheduler.state_dict() == schedule
    assert continual.global_step == continual.args.start_step == 0
    _check_weights(continual, reference)

    torch.manual_seed(1234)
    resumed = _make_trainer(root, device, strategy)
    assert not resumed.optimizer.state
    payload = checkpoint.load(resumed)
    checkpoint.finalize_load(resumed, payload)
    _check_resume(resumed, reference)
    assert resumed.args.start_step == 2
    assert torch.equal(torch.get_rng_state(), cpu_rng)
    if cuda_rng is not None:
        assert torch.equal(torch.cuda.get_rng_state(), cuda_rng)
    # Use identical per-rank inputs for the exact update comparison. Fresh and
    # warmed DDP models can round the reductions of unequal bf16 gradients differently.
    batch = torch.rand(4, 8, device=device, dtype=torch.bfloat16)
    dist.broadcast(batch, src=0)
    _update(resumed, device, batch)
    _update(reference, device, batch)
    _check_resume(resumed, reference)


def _gpu_worker(rank, root):
    torch.cuda.set_device(rank)
    dist.init_process_group(
        "nccl",
        rank=rank,
        world_size=2,
        init_method=(root / "rendezvous").as_uri(),
        timeout=timedelta(minutes=2),
    )
    try:
        init_gloo_group()
        for strategy in ("REPLICATE", "FULL_SHARD"):
            _round_trip(root / strategy, torch.device("cuda", rank), strategy)
    finally:
        dist.destroy_process_group()


@pytest.mark.skipif(torch.cuda.device_count() < 2, reason="requires two CUDA devices")
def test_gpu_checkpoint_resume(tmp_path):
    mp.spawn(_gpu_worker, args=(tmp_path,), nprocs=2)


@pytest.fixture
def cpu_checkpoint_dir(tmp_path, monkeypatch):
    from torchspec.utils import distributed

    dist.init_process_group("gloo", rank=0, world_size=1, init_method=(tmp_path / "rdzv").as_uri())
    # Isolate the trainer-owned group's lifetime to this pytest process group.
    monkeypatch.setattr(distributed, "GLOO_GROUP", None)
    try:
        init_gloo_group()
        with (
            patch.object(torch.cuda, "synchronize"),
            patch.object(torch.cuda, "get_rng_state_all", return_value=[]),
        ):
            yield tmp_path
    finally:
        dist.destroy_process_group()


def test_cpu_checkpoint_resume(cpu_checkpoint_dir):
    _round_trip(cpu_checkpoint_dir, torch.device("cpu"))


def test_model_only_checkpoint_requires_continual_training(cpu_checkpoint_dir):
    device = torch.device("cpu")
    trained = _make_trainer(cpu_checkpoint_dir, device)
    checkpoint.save(SimpleNamespace(args=trained.args, model=trained.model, global_step=1), step=0)

    fresh = _make_trainer(cpu_checkpoint_dir, device, continual=True)
    checkpoint.finalize_load(fresh, checkpoint.load(fresh))
    assert not fresh.optimizer.state
    for weight, master in zip(trained.model.parameters(), fresh.optimizer.fp32_params, strict=True):
        _assert_equal(master, weight.float())

    fresh.args.continual_training = False
    with pytest.raises(RuntimeError, match="No optimizer checkpoint"):
        checkpoint.load(fresh)
