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

from typing import Any, List

from torchspec.data.renderers.base import DEFAULT_CACHE_VERSION, ConversationRenderer


class RendererRegistry:
    """Registry of renderer classes instantiated with a model tokenizer."""

    def __init__(self):
        self.renderers: dict[str, type[ConversationRenderer]] = {}

    def register(
        self,
        name: str,
        renderer_cls: type[ConversationRenderer],
        override: bool = False,
    ) -> None:
        assert override or name not in self.renderers, (
            f"Renderer for the model type {name} has already been registered"
        )
        self.renderers[name] = renderer_cls

    def get(self, name: str) -> type[ConversationRenderer]:
        return self.renderers[name]

    def create(self, name: str, tokenizer: Any) -> ConversationRenderer:
        return self.get(name)(tokenizer)

    def cache_version(self, name: str) -> str:
        return getattr(self.get(name), "CACHE_VERSION", DEFAULT_CACHE_VERSION)

    def get_all_renderer_names(self) -> List[str]:
        return list(self.renderers.keys())


RENDERER_REGISTRY = RendererRegistry()
