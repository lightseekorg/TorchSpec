from __future__ import annotations

from typing import Optional, Tuple

import torch
import torch.nn.functional as F


def split_usp_batch(
    *,
    input_ids: torch.Tensor,
    loss_mask: torch.Tensor,
    hidden_states: torch.Tensor,
    target_hidden_states: torch.Tensor,
    ttt_length: int,
    sp_rank: int,
    sp_size: int,
    ring_rank: int,
    sp_ring_size: int,
    max_len: Optional[int] = None,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    if input_ids.dim() == 1:
        input_ids = input_ids.unsqueeze(0)
    if loss_mask.dim() == 1:
        loss_mask = loss_mask.unsqueeze(0)

    batch_size, input_len = input_ids.shape
    global_len = min(max_len, input_len) if max_len is not None else input_len
    chunk_size = (global_len + sp_size - 1) // sp_size
    sp_ulysses_size = max(1, sp_size // sp_ring_size)
    start = sp_rank * chunk_size
    local_len = chunk_size + ttt_length
    end = min(start + local_len, global_len)

    loss_mask = loss_mask[:, :global_len].clone()

    def _slice_and_pad(tensor: torch.Tensor, axis: int, pad_value: int = 0):
        if tensor.dim() == 1:
            tensor = tensor.unsqueeze(0)
        if axis == 0:
            tensor = tensor[:global_len, :]
            sliced = tensor[start : min(end, tensor.shape[0]), :]
            valid_len = sliced.shape[0]
            if valid_len < local_len:
                sliced = F.pad(sliced, (0, 0, 0, local_len - valid_len), value=pad_value)
        else:
            tensor = tensor[:, :global_len]
            sliced = tensor[:, start : min(end, tensor.shape[1])]
            valid_len = sliced.shape[1]
            if valid_len < local_len:
                pad_len = local_len - valid_len
                if tensor.dim() == 2:
                    sliced = F.pad(sliced, (0, pad_len), value=pad_value)
                else:
                    sliced = F.pad(sliced, (0, 0, 0, pad_len), value=pad_value)
        return sliced.contiguous(), valid_len

    input_ids, valid_len = _slice_and_pad(input_ids, axis=1, pad_value=0)
    loss_mask, _ = _slice_and_pad(loss_mask, axis=1, pad_value=0)
    if hidden_states.dim() == 2:
        hidden_states, _ = _slice_and_pad(hidden_states, axis=0, pad_value=0)
        hidden_states = hidden_states.unsqueeze(0)
    else:
        hidden_states, _ = _slice_and_pad(hidden_states, axis=1, pad_value=0)
    if target_hidden_states.dim() == 2:
        target_hidden_states, _ = _slice_and_pad(target_hidden_states, axis=0, pad_value=0)
        target_hidden_states = target_hidden_states.unsqueeze(0)
    else:
        target_hidden_states, _ = _slice_and_pad(target_hidden_states, axis=1, pad_value=0)

    attention_mask = torch.zeros((batch_size, local_len), dtype=torch.long, device=input_ids.device)
    attention_mask[:, :valid_len] = 1

    usp_chunk_size = max(local_len - ttt_length, 0)
    ring_chunk = usp_chunk_size * sp_ulysses_size
    ring_start = ring_rank * ring_chunk
    position_ids = torch.arange(
        ring_start, ring_start + ring_chunk, device=input_ids.device, dtype=torch.long
    ).unsqueeze(0)
    if batch_size > 1:
        position_ids = position_ids.expand(batch_size, -1)

    return (
        input_ids,
        attention_mask,
        loss_mask,
        hidden_states,
        target_hidden_states,
        position_ids,
    )
