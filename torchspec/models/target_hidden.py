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

"""Capture a Hugging Face decoder's hidden states on both sides of its final norm.

Which side of the final norm a hidden state comes from is easy to get wrong:
``output_hidden_states[-1]`` and ``last_hidden_state`` are post-norm in most
Hugging Face decoders, while vLLM layer captures are often pre-norm. Forward
hooks on the final norm and the last decoder block capture the exact tensors on
both sides, and the boundary is checked on every call.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn


@dataclass(frozen=True)
class FinalNormBoundary:
    """Tensors on both sides of the final norm, detached from any graph."""

    pre_norm: torch.Tensor
    post_norm: torch.Tensor


@torch.no_grad()
def capture_final_norm_boundary(decoder: nn.Module, **forward_kwargs) -> FinalNormBoundary:
    """Run ``decoder`` and return the input and output of ``decoder.norm``.

    ``decoder`` must expose ``layers`` and ``norm`` and return an object with
    ``last_hidden_state`` (e.g. a Hugging Face ``*Model`` or ``*TextModel``).
    Raises if the norm input is not bitwise the last block's output, if
    re-applying the norm does not reproduce the captured output, or if the
    decoder's returned state is not the post-norm tensor.
    """

    captured: dict[str, torch.Tensor] = {}

    def norm_hook(module, args, output):
        captured["pre_norm"] = args[0]
        captured["post_norm"] = output

    def block_hook(module, args, output):
        captured["block"] = output[0] if isinstance(output, tuple) else output

    handles = [
        decoder.norm.register_forward_hook(norm_hook),
        decoder.layers[-1].register_forward_hook(block_hook),
    ]
    try:
        output = decoder(**forward_kwargs)
    finally:
        for handle in handles:
            handle.remove()

    pre_norm = captured["pre_norm"]
    post_norm = captured["post_norm"]
    if not torch.equal(pre_norm, captured["block"]):
        raise RuntimeError("final-norm input is not the last decoder block's output")
    if not torch.equal(decoder.norm(pre_norm), post_norm):
        raise RuntimeError("re-applying the final norm does not reproduce its captured output")
    if torch.equal(pre_norm, post_norm):
        raise RuntimeError("final norm is an identity on this input; boundary is not observable")
    if not torch.equal(output.last_hidden_state, post_norm):
        raise RuntimeError("decoder output is not the final norm's output")
    return FinalNormBoundary(pre_norm=pre_norm.detach(), post_norm=post_norm.detach())
