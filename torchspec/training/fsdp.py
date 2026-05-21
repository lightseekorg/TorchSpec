# Copyright (c) 2026 LightSeek Foundation
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

from contextlib import contextmanager
from typing import List, Optional

import torch
import torch.distributed as dist
import torch.nn as nn

from torchspec.utils.logging import logger


def _allreduce_with_divide_factor_hook(state: dict, bucket: dist.GradBucket):
    tensor = bucket.buffer()
    tensor.div_(state["divide_factor"])
    return (
        dist.all_reduce(tensor, group=state["process_group"], async_op=True)
        .get_future()
        .then(lambda fut: fut.value()[0])
    )


@contextmanager
def _init_on_device(device: torch.device, include_buffers: Optional[bool] = False):
    if include_buffers:
        with device:
            yield
        return

    old_register_parameter = nn.Module.register_parameter
    if include_buffers:
        old_register_buffer = nn.Module.register_buffer

    def register_empty_parameter(module, name, param):
        old_register_parameter(module, name, param)
        if param is not None:
            param_cls = type(module._parameters[name])
            kwargs = module._parameters[name].__dict__
            kwargs["requires_grad"] = param.requires_grad
            module._parameters[name] = param_cls(module._parameters[name].to(device), **kwargs)

    def register_empty_buffer(module, name, buffer, persistent=True):
        old_register_buffer(module, name, buffer, persistent=persistent)
        if buffer is not None:
            module._buffers[name] = module._buffers[name].to(device)

    if include_buffers:
        tensor_constructors_to_patch = {
            torch_function_name: getattr(torch, torch_function_name)
            for torch_function_name in ["empty", "zeros", "ones", "full"]
        }
    else:
        tensor_constructors_to_patch = {}

    def patch_tensor_constructor(fn):
        def wrapper(*args, **kwargs):
            kwargs["device"] = device
            return fn(*args, **kwargs)

        return wrapper

    try:
        nn.Module.register_parameter = register_empty_parameter
        if include_buffers:
            nn.Module.register_buffer = register_empty_buffer
        for torch_function_name in tensor_constructors_to_patch.keys():
            setattr(
                torch,
                torch_function_name,
                patch_tensor_constructor(getattr(torch, torch_function_name)),
            )
        yield
    finally:
        nn.Module.register_parameter = old_register_parameter
        if include_buffers:
            nn.Module.register_buffer = old_register_buffer
        for torch_function_name, old_torch_function in tensor_constructors_to_patch.items():
            setattr(torch, torch_function_name, old_torch_function)


@contextmanager
def init_empty_weights(include_buffers: bool = False):
    """Context manager that initialises models on the meta device (no memory).

    Args:
        include_buffers: Whether to also put all buffers on the meta device.
    """
    with _init_on_device(torch.device("meta"), include_buffers=include_buffers) as f:
        yield f


@contextmanager
def _default_pg_override(group):
    """Temporarily install ``group`` as the process-wide default PG.

    Several PyTorch distributed helpers (notably
    ``set_model_state_dict(broadcast_from_rank0=True)``) issue
    collectives with a hard-coded ``group=None`` and therefore always
    land on the default process group. In colocate mode that default
    PG is the 2N-rank union world, which deadlocks any trainer-only
    collective. Swapping the default PG for the duration of such a
    call redirects those ``group=None`` collectives onto ``group``.
    """
    from torch.distributed import distributed_c10d as c10d

    prev = c10d._world.default_pg
    c10d._world.default_pg = group
    try:
        yield
    finally:
        c10d._world.default_pg = prev


def fsdp2_load_full_state_dict(model, full_state, device_mesh, cpu_offload):
    """Load a full state dict into an FSDP2 model, broadcasting from rank 0.

    Args:
        model: FSDP2-wrapped model.
        full_state: State dict (only rank 0 has real weights, others have empty dict).
        device_mesh: Device mesh for FSDP.
        cpu_offload: If not None, enables StateDictOptions cpu_offload.
    """
    from torch.distributed.checkpoint.state_dict import (
        StateDictOptions,
        set_model_state_dict,
    )

    # In colocate mode the default PG is the 2N-rank union world (N
    # trainers + N engines). The engine never enters this code path,
    # so any broadcast on the default group will hang waiting for
    # engine participation. The FSDP DeviceMesh, by construction,
    # contains only trainer ranks — use its group for any explicit
    # `dist.broadcast`.
    mesh_group = device_mesh.get_group() if device_mesh is not None else None
    src_rank = dist.get_global_rank(mesh_group, 0) if mesh_group is not None else 0
    logger.warning(
        "[TS-COLOCATE-TRACE-T] fsdp2_load_full_state_dict: ENTER mesh_group=%s src_rank=%s",
        mesh_group,
        src_rank,
    )

    if dist.get_rank() == 0:
        model = model.to(device=torch.cuda.current_device(), non_blocking=True)
    else:
        model = model.to_empty(device=torch.cuda.current_device())

    is_cpu_offload = cpu_offload is not None

    # `broadcast_from_rank0=True` makes PyTorch's set_model_state_dict
    # broadcast the rank-0 state dict across the *default* process
    # group (PyTorch's `_broadcast_state_dict` hard-codes `group=None`
    # — there is no public way to scope it). In colocate mode the
    # default PG is the 2N-rank union world; the engine never enters
    # this code path, so that broadcast hangs waiting for engine ranks.
    #
    #   * Single trainer rank (mesh_size == 1): nothing to broadcast —
    #     rank 0 already holds the full state — so disable the
    #     broadcast and let rank 0 load locally.
    #   * Multi-trainer mesh (mesh_size >= 2): keep broadcast_from_rank0
    #     but temporarily swap the process-wide default PG to the
    #     trainer-only FSDP mesh group for the duration of the call, so
    #     PyTorch's internal `group=None` broadcast lands on the
    #     trainer sub-world instead of the 2N-rank union.
    mesh_size = device_mesh.size() if device_mesh is not None else dist.get_world_size()
    single_rank_mesh = mesh_size == 1
    broadcast_from_rank0 = not single_rank_mesh
    options = StateDictOptions(
        full_state_dict=True,
        cpu_offload=is_cpu_offload,
        broadcast_from_rank0=broadcast_from_rank0,
    )

    logger.warning(
        "[TS-COLOCATE-TRACE-T] fsdp2_load_full_state_dict: BEFORE "
        "set_model_state_dict (mesh_size=%s, broadcast_from_rank0=%s)",
        mesh_size,
        broadcast_from_rank0,
    )
    if broadcast_from_rank0 and mesh_group is not None:
        with _default_pg_override(mesh_group):
            set_model_state_dict(model, full_state, options=options)
    else:
        set_model_state_dict(model, full_state, options=options)
    logger.warning("[TS-COLOCATE-TRACE-T] fsdp2_load_full_state_dict: AFTER set_model_state_dict")

    # CRITICAL: pass mesh_group to dist.broadcast so the broadcast
    # only spans the trainer sub-mesh, not the 2N-rank default PG.
    # Without this the trainer blocks forever waiting for engine
    # participation in the buffer broadcast.
    for _name, buf in model.named_buffers():
        dist.broadcast(buf, src=src_rank, group=mesh_group)
    logger.warning("[TS-COLOCATE-TRACE-T] fsdp2_load_full_state_dict: AFTER buffer broadcasts")

    if is_cpu_offload:
        model.to("cpu", non_blocking=True)
        for buf in model.buffers():
            buf.data = buf.data.to(torch.cuda.current_device())

    return model


def apply_fsdp2(
    model,
    mesh=None,
    cpu_offload=False,
    args=None,
    modules_to_shard: Optional[List[nn.Module]] = None,
):
    """Apply FSDP v2 or DDP to a model.

    Args:
        model: The model to wrap with FSDP/DDP.
        mesh: Optional DeviceMesh for FSDP. If None, uses all ranks.
        cpu_offload: If True, offload parameters, gradients, and optimizer states to CPU.
        args: Arguments containing precision settings (fp16/bf16) and fsdp_strategy.
        modules_to_shard: Explicit list of sub-modules to individually shard
            before sharding the root model.  When *None* the root model is
            sharded as a single unit.
    """
    from torch.distributed._composable.replicate import replicate
    from torch.distributed.fsdp import (
        CPUOffloadPolicy,
        MixedPrecisionPolicy,
        fully_shard,
    )

    strategy = getattr(args, "fsdp_strategy", "REPLICATE") if args else "REPLICATE"
    strategy = strategy.upper()

    if strategy == "REPLICATE":
        logger.info("Using REPLICATE strategy (DDP-like, gradient all-reduce only)")
        replicate(model, device_mesh=mesh)
        if args is not None and getattr(args, "attention_backend", None) == "usp":
            sp_size = getattr(args, "sp_ulysses_size", 1) * getattr(args, "sp_ring_size", 1)
            if sp_size > 1:
                process_group = mesh.get_group() if mesh is not None else dist.group.WORLD
                divide_factor = dist.get_world_size(process_group) // sp_size
                if divide_factor <= 0:
                    raise ValueError(
                        f"Invalid USP grad divide factor: world_size="
                        f"{dist.get_world_size(process_group)}, sp_size={sp_size}"
                    )
                model.register_comm_hook(
                    {
                        "process_group": process_group,
                        "divide_factor": divide_factor,
                    },
                    _allreduce_with_divide_factor_hook,
                )
                logger.info(
                    "Registered USP replicate grad hook "
                    f"(all_reduce / {divide_factor}, sp_size={sp_size})"
                )
        return model
    elif strategy != "FULL_SHARD":
        raise ValueError(f"Unknown fsdp_strategy: {strategy}. Use 'FULL_SHARD' or 'REPLICATE'")

    logger.info("Using FULL_SHARD strategy (FSDP, sharded parameters)")

    offload_policy = CPUOffloadPolicy() if cpu_offload else None

    param_dtype = torch.bfloat16
    reduce_dtype_str = getattr(args, "fsdp_reduce_dtype", "float32") if args else "float32"
    reduce_dtype = torch.bfloat16 if reduce_dtype_str == "bfloat16" else torch.float32

    if args is not None and getattr(args, "fp16", False):
        param_dtype = torch.float16

    logger.info(
        f"FSDP MixedPrecision Policy: param_dtype={param_dtype}, reduce_dtype={reduce_dtype}"
    )

    fsdp_kwargs = {
        "mp_policy": MixedPrecisionPolicy(
            param_dtype=param_dtype,
            reduce_dtype=reduce_dtype,
        ),
        "offload_policy": offload_policy,
        "mesh": mesh,
    }

    for module in modules_to_shard or []:
        fully_shard(module, **fsdp_kwargs)

    fully_shard(model, **fsdp_kwargs)

    return model
