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

from torchspec.models.draft.auto import (
    AutoDFlashDraftModel,
    AutoDraftModelConfig,
    AutoEagle3DraftModel,
)
from torchspec.models.draft.base import Eagle3DraftModel
from torchspec.models.draft.deepseek_eagle import Eagle3DeepseekV2ForCausalLM
from torchspec.models.draft.dflash import DFlashDraftModel
from torchspec.models.draft.llama3_eagle import LlamaForCausalLMEagle3

__all__ = [
    "AutoDFlashDraftModel",
    "AutoDraftModelConfig",
    "AutoEagle3DraftModel",
    "DFlashDraftModel",
    "Eagle3DeepseekV2ForCausalLM",
    "Eagle3DraftModel",
    "LlamaForCausalLMEagle3",
]
