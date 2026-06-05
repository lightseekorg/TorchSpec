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

"""Eagle3 draft model for GLM-4-MoE targets.

GLM-4-MoE uses standard GQA attention (not MLA) plus a DeepSeek-style MoE FFN
(sigmoid + group-limited routing + correction bias — the same routing family as
DeepSeek-V3). So the draft reuses the Llama-style GQA backbone from
``llama3_eagle`` and enables the shared MoE block (``use_moe: true`` in the
config selects ``DeepseekV3MoEBlock`` inside ``LlamaDecoderLayer``).

The Expert-Parallel path (all-to-all dispatch, FSDP-on-experts, EP-aware grad
clip) is shared with the DeepSeek draft and needs no GLM-specific code.

v1 note: GLM's partial rotary (``partial_rotary_factor``) is not yet mirrored —
the draft uses the Llama-eagle RoPE. This is acceptable for a from-scratch draft;
refine to the exact target RoPE if acceptance length needs it.
"""

from transformers.models.glm4_moe.configuration_glm4_moe import Glm4MoeConfig

from torchspec.models.draft.llama3_eagle import LlamaForCausalLMEagle3


class Glm4MoeForCausalLMEagle3(LlamaForCausalLMEagle3):
    """Eagle3 draft for GLM-4-MoE: GQA attention + MoE FFN (set ``use_moe: true``)."""

    config_class = Glm4MoeConfig
