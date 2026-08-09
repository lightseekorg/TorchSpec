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

from typing import Any, ClassVar, Protocol, runtime_checkable

DEFAULT_CACHE_VERSION = "v1"


@runtime_checkable
class ConversationRenderer(Protocol):
    """Turn normalized chat messages into training tokens and supervision.

    A renderer owns both prompt construction and loss-mask derivation for one
    model family, for cases a :class:`~torchspec.data.template.ChatTemplate`
    cannot express — nested message structures, tool-call serialization, or a
    checkpoint whose ``apply_chat_template`` is the only correct source of
    truth. Renderers are instantiated with the target model's tokenizer and are
    selected by name through ``dataset.renderer``.
    """

    # Folded into the dataset tokenization cache key. Bump whenever rendering
    # semantics change in a way that can alter input_ids or loss masks, so
    # stale caches are not silently reused.
    CACHE_VERSION: ClassVar[str] = DEFAULT_CACHE_VERSION

    def __init__(self, tokenizer: Any) -> None: ...

    def get_assistant_token_ids(self) -> tuple[list[int], list[int], int]:
        """Return ``(header_ids, end_ids, skip_after_header)`` for dynamic masks.

        Used when the loss mask has to be recomputed at training time against
        the token IDs the inference engine actually produced. ``header_ids``
        must end at the last structural token before supervision starts, and
        ``end_ids`` must match the token that closes an assistant turn.
        """

    def render(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
        *,
        max_seq_length: int,
        last_turn_only: bool = False,
        generation_config: dict[str, Any] | None = None,
    ) -> tuple[list[int], list[int]]:
        """Return equally sized ``(input_ids, loss_mask)`` lists.

        ``generation_config`` carries opaque per-sample rendering options from
        the dataset row (a serving-time sampling config, a reasoning-effort
        setting, and so on); renderers that do not need it should ignore it.
        Implementations must truncate to ``max_seq_length`` rather than let the
        caller do it, so the mask stays aligned with the token stream.
        """
