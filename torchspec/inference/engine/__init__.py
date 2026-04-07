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


from torchspec.inference.engine.base import InferenceEngine
from torchspec.inference.engine.hf_engine import HFEngine
from torchspec.inference.engine.hf_runner import HFRunner

__all__ = [
    "InferenceEngine",
    "HFEngine",
    "HFRunner",
]

# Lazy imports: SGLang/vLLM are optional — HF-only training (e.g. single-GPU
# DFlash) should not require these heavy dependencies to be installed.
try:
    from torchspec.inference.engine.sgl_engine import SglEngine

    __all__.append("SglEngine")
except ImportError:
    pass

try:
    from torchspec.inference.engine.vllm_engine import VllmEngine

    __all__.append("VllmEngine")
except ImportError:
    pass
