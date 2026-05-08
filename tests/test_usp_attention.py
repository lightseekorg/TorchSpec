import importlib.util
import os
import socket
import unittest

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from transformers import LlamaConfig

from torchspec.models.draft.llama3_eagle import LlamaFlexAttention, LlamaUSPFlashAttention
from torchspec.utils.distributed import get_sp_rank, init_usp_groups


def _has_usp_runtime() -> bool:
    return (
        importlib.util.find_spec("flash_attn") is not None
        and importlib.util.find_spec("yunchang") is not None
    )


def _find_free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        sock.listen(1)
        return sock.getsockname()[1]


def _build_config() -> LlamaConfig:
    return LlamaConfig(
        hidden_size=128,
        num_attention_heads=8,
        num_key_value_heads=2,
        max_position_embeddings=4096,
        rms_norm_eps=1e-5,
        vocab_size=32000,
        intermediate_size=688,
        hidden_act="silu",
        num_hidden_layers=1,
        attention_bias=False,
        torch_dtype="bfloat16",
    )


def _make_full_hidden_steps(
    *,
    num_steps: int,
    batch_size: int,
    global_seq_len: int,
    hidden_size: int,
    dtype: torch.dtype,
) -> list[torch.Tensor]:
    generator = torch.Generator(device="cpu").manual_seed(20260423)
    steps = []
    for _ in range(num_steps):
        tensor = torch.randn(
            batch_size,
            global_seq_len,
            hidden_size,
            generator=generator,
            dtype=torch.float32,
        )
        steps.append(tensor.to(dtype))
    return steps


def _broadcast_state_dict(rank: int, module: torch.nn.Module) -> dict[str, torch.Tensor]:
    state = None
    if rank == 0:
        state = {name: tensor.detach().cpu() for name, tensor in module.state_dict().items()}
    obj = [state]
    dist.broadcast_object_list(obj, src=0)
    return obj[0]


def _run_usp_vs_flex_worker(
    rank: int,
    world_size: int,
    port: int,
    sp_ulysses_size: int,
    sp_ring_size: int,
) -> None:
    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = str(port)
    torch.cuda.set_device(rank)
    dist.init_process_group("nccl", rank=rank, world_size=world_size)
    init_usp_groups(sp_ulysses_size=sp_ulysses_size, sp_ring_size=sp_ring_size)

    device = torch.device(f"cuda:{rank}")
    dtype = torch.bfloat16
    num_steps = 2
    batch_size = 1
    global_seq_len = 32
    local_seq_len = global_seq_len // world_size
    config = _build_config()
    hidden_size = config.hidden_size * 2
    flex_position_ids = torch.arange(global_seq_len, device=device, dtype=torch.long).unsqueeze(0)
    if sp_ulysses_size == world_size:
        usp_position_ids = flex_position_ids
    else:
        start = get_sp_rank() * local_seq_len
        usp_position_ids = torch.arange(
            start,
            start + local_seq_len,
            device=device,
            dtype=torch.long,
        ).unsqueeze(0)
    attention_mask = torch.ones(batch_size, global_seq_len, device=device, dtype=torch.bool)

    flex_attention = LlamaFlexAttention(config).to(device).to(dtype) if rank == 0 else None
    usp_attention = LlamaUSPFlashAttention(config).to(device).to(dtype)

    state_dict = _broadcast_state_dict(rank, flex_attention if rank == 0 else usp_attention)
    usp_attention.load_state_dict(state_dict)
    if rank == 0:
        flex_attention.load_state_dict(state_dict)

    full_hidden_steps = _make_full_hidden_steps(
        num_steps=num_steps,
        batch_size=batch_size,
        global_seq_len=global_seq_len,
        hidden_size=hidden_size,
        dtype=dtype,
    )

    local_hidden_steps = []
    for full_hidden in full_hidden_steps:
        start = get_sp_rank() * local_seq_len
        end = start + local_seq_len
        local_hidden = (
            full_hidden[:, start:end, :].to(device=device).clone().detach().requires_grad_(True)
        )
        local_hidden_steps.append(local_hidden)

    if rank == 0:
        flex_hidden_steps = [
            full_hidden.to(device=device).clone().detach().requires_grad_(True)
            for full_hidden in full_hidden_steps
        ]

    usp_cache_keys = None
    usp_cache_values = None
    flex_cache_keys = None
    flex_cache_values = None
    usp_loss = torch.zeros((), device=device, dtype=torch.float32)
    flex_loss = torch.zeros((), device=device, dtype=torch.float32) if rank == 0 else None
    max_output_diff = 0.0
    loss_scale = 1.0 / (num_steps * batch_size * global_seq_len * config.hidden_size)

    for step in range(num_steps):
        usp_out, usp_cache_keys, usp_cache_values = usp_attention(
            hidden_states=local_hidden_steps[step],
            cache_keys=usp_cache_keys,
            cache_values=usp_cache_values,
            attention_mask=None,
            position_ids=usp_position_ids,
            use_cache=True,
        )
        usp_loss = usp_loss + usp_out.float().square().sum() * loss_scale

        gathered_usp_out = [torch.empty_like(usp_out) for _ in range(world_size)]
        dist.all_gather(gathered_usp_out, usp_out.detach())

        if rank == 0:
            flex_out, flex_cache_keys, flex_cache_values = flex_attention(
                hidden_states=flex_hidden_steps[step],
                cache_keys=flex_cache_keys,
                cache_values=flex_cache_values,
                attention_mask=attention_mask,
                position_ids=flex_position_ids,
                use_cache=True,
            )
            flex_loss = flex_loss + flex_out.float().square().sum() * loss_scale
            usp_out_full = torch.cat(gathered_usp_out, dim=1)
            step_output_diff = (usp_out_full.float() - flex_out.float()).abs().max().item()
            max_output_diff = max(max_output_diff, step_output_diff)

    usp_loss.backward()
    if rank == 0:
        flex_loss.backward()

    reduced_usp_loss = usp_loss.detach().clone()
    dist.all_reduce(reduced_usp_loss, op=dist.ReduceOp.SUM)
    if rank == 0:
        torch.testing.assert_close(
            reduced_usp_loss,
            flex_loss.detach(),
            atol=2e-2,
            rtol=2e-2,
            msg=f"USP loss mismatch (max output diff={max_output_diff:.6f})",
        )

    for proj_name in ("q_proj", "k_proj", "v_proj", "o_proj"):
        usp_grad = getattr(usp_attention, proj_name).weight.grad.detach().float().clone()
        dist.all_reduce(usp_grad, op=dist.ReduceOp.SUM)
        if rank == 0:
            flex_grad = getattr(flex_attention, proj_name).weight.grad.detach().float()
            torch.testing.assert_close(
                usp_grad,
                flex_grad,
                atol=3e-2,
                rtol=3e-2,
                msg=(
                    f"USP gradient mismatch for {proj_name} (max output diff={max_output_diff:.6f})"
                ),
            )

    for step in range(num_steps):
        local_grad = local_hidden_steps[step].grad.detach()
        gathered_hidden_grad = [torch.empty_like(local_grad) for _ in range(world_size)]
        dist.all_gather(gathered_hidden_grad, local_grad)
        if rank == 0:
            full_hidden_grad = torch.cat(gathered_hidden_grad, dim=1).float()
            torch.testing.assert_close(
                full_hidden_grad,
                flex_hidden_steps[step].grad.detach().float(),
                atol=3e-2,
                rtol=3e-2,
                msg=(
                    f"USP input gradient mismatch at step {step} "
                    f"(max output diff={max_output_diff:.6f})"
                ),
            )

    dist.barrier()
    dist.destroy_process_group()


class TestUSPAttention(unittest.TestCase):
    @unittest.skipUnless(torch.cuda.is_available(), "CUDA not available")
    @unittest.skipUnless(torch.cuda.device_count() >= 2, "Requires at least 2 CUDA devices")
    @unittest.skipUnless(_has_usp_runtime(), "USP test requires flash_attn and yunchang")
    def test_usp_matches_flex_loss_and_gradients(self):
        port = _find_free_port()
        mp.spawn(
            _run_usp_vs_flex_worker,
            args=(2, port, 2, 1),
            nprocs=2,
            join=True,
        )

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA not available")
    @unittest.skipUnless(torch.cuda.device_count() >= 2, "Requires at least 2 CUDA devices")
    @unittest.skipUnless(_has_usp_runtime(), "USP test requires flash_attn and yunchang")
    def test_usp_ring_matches_flex_loss_and_gradients(self):
        port = _find_free_port()
        mp.spawn(
            _run_usp_vs_flex_worker,
            args=(2, port, 1, 2),
            nprocs=2,
            join=True,
        )


if __name__ == "__main__":
    unittest.main(verbosity=2)
