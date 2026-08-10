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

"""
Mapping TorchSpec weight keys to HF / vLLM / SGLang serving names.
"""

from typing import Collection

DRAFT_WEIGHT_KEY_REMAP = [
    ("midlayer.", "layers.0."),
    ("context_proj.", "fc."),
    ("context_norm.", "hidden_norm."),
    ("final_norm.", "norm."),
]


def to_export_keys(tensors: dict) -> dict:
    """Rename internal draft keys to HF/SGLang export names (forward direction)."""
    remapped = {}
    for k, v in tensors.items():
        new_key = k
        for internal_prefix, export_prefix in DRAFT_WEIGHT_KEY_REMAP:
            if k.startswith(internal_prefix):
                new_key = export_prefix + k[len(internal_prefix) :]
                break
        remapped[new_key] = v
    return remapped


def to_internal_keys(tensors: dict, model_keys: Collection[str]) -> dict:
    """Rename exported draft keys back to internal names (reverse direction).

    The forward direction is not injective across draft families -- an Eagle3 draft owns a
    top-level ``norm.*``, which is also DFlash's export name for ``final_norm.*`` -- so a key is
    only renamed when *model_keys* does not already contain it. Unknown keys pass through for the
    caller's own strictness check to report.
    """
    model_keys = set(model_keys)
    remapped = {}
    for k, v in tensors.items():
        new_key = k
        if k not in model_keys:
            for internal_prefix, export_prefix in DRAFT_WEIGHT_KEY_REMAP:
                if k.startswith(export_prefix):
                    candidate = internal_prefix + k[len(export_prefix) :]
                    if candidate in model_keys:
                        new_key = candidate
                        break
        remapped[new_key] = v
    return remapped
