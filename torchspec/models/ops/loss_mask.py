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

from dataclasses import dataclass

import numba
import numpy as np
import torch

DYNAMIC_LOSS_MASK_CACHE_VERSION = "v2"


@dataclass(frozen=True)
class AssistantHeader:
    """A header prefix followed by optional attributes and a structural terminator."""

    prefix: tuple[int, ...]
    terminator_id: int


@numba.njit(cache=True)
def _numba_loss_mask(ids, header, end, out, skip_after, header_terminator_id):
    n = len(ids)
    header_len = len(header)
    end_len = len(end)
    last_body_start = n
    i = 0
    while i <= n - header_len:
        match = True
        for k in range(header_len):
            if ids[i + k] != header[k]:
                match = False
                break
        if not match:
            i += 1
            continue
        j = i + header_len
        if header_terminator_id is not None:
            while j < n and ids[j] != header_terminator_id:
                # An incomplete header must not consume another message.
                if ids[j] == header[0] or ids[j] == end[0]:
                    break
                j += 1
            if j == n or ids[j] != header_terminator_id:
                last_body_start = n
                i = j
                continue
            j += 1
        body_start = j + skip_after
        last_body_start = body_start
        while j <= n - end_len:
            end_match = True
            for k in range(end_len):
                if ids[j + k] != end[k]:
                    end_match = False
                    break
            if end_match:
                break
            j += 1
        else:
            j = n
        out[body_start:j] = 1
        i = j + end_len
    return last_body_start


def _to_contiguous_int64_numpy(t: torch.Tensor) -> np.ndarray:
    if t.is_cuda:
        t = t.cpu()
    if t.dtype != torch.int64:
        t = t.to(torch.int64)
    t = t.contiguous()
    return t.numpy()


def compute_assistant_loss_mask(
    input_ids: torch.Tensor,
    assistant_header_ids: list[int] | AssistantHeader,
    end_token_ids: list[int],
    last_turn_only: bool = False,
    skip_after_header: int = 0,
) -> torch.Tensor:
    """Compute loss mask where 1s mark assistant content tokens only.

    Uses a JIT-compiled single-pass O(N) scan to find assistant header
    sequences and pair them with end token sequences.
    Computes on CPU and returns result on the same device as input_ids.

    Args:
        input_ids: 1-D tensor of token IDs (CPU or CUDA).
        assistant_header_ids: A fixed token ID sequence or a prefix and terminator
            for headers with variable attributes.
        end_token_ids: Token ID sequence marking the end of assistant content.
        last_turn_only: If True, only the last assistant turn is marked.
        skip_after_header: Number of tokens to skip after header match before
            marking content. Use 1 when the header excludes a trailing newline
            that BPE may merge with subsequent content.

    Returns:
        1-D long tensor on the same device as input_ids, with 1s for assistant
        content tokens and 0s elsewhere.
    """
    device = input_ids.device if isinstance(input_ids, torch.Tensor) else torch.device("cpu")
    ids_np = (
        _to_contiguous_int64_numpy(input_ids)
        if isinstance(input_ids, torch.Tensor)
        else np.asarray(input_ids, dtype=np.int64)
    )
    header_terminator_id = None
    if isinstance(assistant_header_ids, AssistantHeader):
        header_terminator_id = assistant_header_ids.terminator_id
        assistant_header_ids = assistant_header_ids.prefix
    header_np = np.array(assistant_header_ids, dtype=np.int64)
    end_np = np.array(end_token_ids, dtype=np.int64)
    out = np.zeros(len(ids_np), dtype=np.int64)
    if header_np.size and end_np.size:
        last_body_start = _numba_loss_mask(
            ids_np, header_np, end_np, out, skip_after_header, header_terminator_id
        )
        if last_turn_only:
            out[:last_body_start] = 0

    result = torch.from_numpy(out)
    return result.to(device) if device.type != "cpu" else result
