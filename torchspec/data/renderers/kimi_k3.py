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

from typing import Any


class K3Renderer:
    """Render Kimi-K3 XTML through the checkpoint's remote tokenizer code.

    The tokenizer owns all XTML construction and segment-wise encoding. This
    class only derives TorchSpec's assistant loss mask from the resulting token
    stream and applies a synchronized right truncation.
    """

    _OPEN_TOKEN = "<|open|>"
    _CLOSE_TOKEN = "<|close|>"
    _SEP_TOKEN = "<|sep|>"
    _EOM_TOKEN = "<|end_of_msg|>"
    _REASONING_FIELDS = (
        "thinking",
        "thinking_content",
        "reasoning_content",
        "reasoning",
    )
    # The K3 chat template only implements these three levels; anything else
    # (notably the generic "medium") is rejected rather than silently remapped.
    _VALID_THINKING_EFFORTS = frozenset({"low", "high", "max"})

    # Included in the dataset tokenization cache key. Bump whenever rendering
    # semantics change in a way that can alter input_ids or loss masks.
    CACHE_VERSION = "generation-config-v3"

    def __init__(self, tokenizer: Any):
        self.tokenizer = tokenizer
        self.open_id = self._get_special_id(self._OPEN_TOKEN)
        self.close_id = self._get_special_id(self._CLOSE_TOKEN)
        self.sep_id = self._get_special_id(self._SEP_TOKEN)
        self.eom_id = self._get_special_id(self._EOM_TOKEN)
        assistant_probe = self._apply_chat_template(
            [{"role": "assistant", "content": "renderer contract probe"}],
            tools=None,
        )
        self.assistant_header_ids = self._derive_assistant_header(assistant_probe)
        self._validate_tokenizer_contract(assistant_probe)

    def _get_special_id(self, token: str) -> int:
        special_tokens = getattr(self.tokenizer, "special_tokens", None)
        if not isinstance(special_tokens, dict) or token not in special_tokens:
            raise ValueError(f"Kimi-K3 tokenizer is missing required special token {token!r}")
        token_id = special_tokens[token]
        if not isinstance(token_id, int):
            raise TypeError(f"Special token {token!r} has non-integer id {token_id!r}")
        return token_id

    @staticmethod
    def _normalize_token_ids(token_ids: Any) -> list[int]:
        if not isinstance(token_ids, (list, tuple)):
            raise TypeError(
                "Kimi-K3 tokenizer must return a flat list of token ids, "
                f"got {type(token_ids).__name__}"
            )
        normalized = list(token_ids)
        if any(not isinstance(token_id, int) for token_id in normalized):
            raise TypeError("Kimi-K3 tokenizer returned non-integer or nested token ids")
        return normalized

    def _apply_chat_template(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None,
        *,
        thinking: bool = True,
        thinking_effort: str | None = None,
    ) -> list[int]:
        """Encode ``messages``, leaving unset options to the tokenizer.

        The tokenizer treats a non-``None`` ``thinking_effort`` as an explicit
        rendering option and injects an internal thinking-effort system
        message. Passing one where the caller did not ask for it would make
        training prompts diverge from what the checkpoint renders at serving
        time for a request that named no effort, so the default stays ``None``.
        """
        token_ids = self.tokenizer.apply_chat_template(
            messages,
            tools=tools,
            tokenize=True,
            add_generation_prompt=False,
            thinking=thinking,
            preserve_thinking=True,
            thinking_effort=thinking_effort,
            padding=False,
            truncation=False,
            return_tensors=None,
            return_dict=False,
        )
        return self._normalize_token_ids(token_ids)

    @classmethod
    def _resolve_generation_config(
        cls,
        generation_config: dict[str, Any] | None,
    ) -> tuple[bool, str | None]:
        # A row that carries no config at all asks for nothing in particular,
        # so no effort is forwarded and the tokenizer's own default stands.
        if generation_config is None:
            return True, None
        if not isinstance(generation_config, dict):
            raise TypeError(
                "Kimi-K3 generation_config must be a dictionary or None, "
                f"got {type(generation_config).__name__}"
            )

        thinking = generation_config.get("thinking")
        if not isinstance(thinking, bool):
            raise ValueError("Kimi-K3 generation_config must contain a boolean 'thinking' field")

        thinking_effort = generation_config.get("thinking_effort")
        if thinking:
            if thinking_effort not in cls._VALID_THINKING_EFFORTS:
                supported = ", ".join(sorted(cls._VALID_THINKING_EFFORTS))
                raise ValueError(
                    "Thinking-enabled Kimi-K3 generation_config must contain "
                    f"'thinking_effort' in {{{supported}}}, got {thinking_effort!r}"
                )
        elif thinking_effort is not None:
            raise ValueError(
                "Thinking-disabled Kimi-K3 generation_config must use "
                f"'thinking_effort': null, got {thinking_effort!r}"
            )

        reasoning_effort = generation_config.get("reasoning_effort")
        expected_reasoning_effort = thinking_effort if thinking else "none"
        if reasoning_effort is not None and reasoning_effort != expected_reasoning_effort:
            raise ValueError(
                "Kimi-K3 generation_config has inconsistent reasoning_effort: "
                f"expected {expected_reasoning_effort!r}, got {reasoning_effort!r}"
            )
        return thinking, thinking_effort

    @classmethod
    def _validate_disabled_thinking_messages(
        cls,
        messages: list[dict[str, Any]],
    ) -> None:
        for index, message in enumerate(messages):
            if message.get("role") != "assistant":
                continue
            if any(message.get(field) for field in cls._REASONING_FIELDS):
                raise ValueError(
                    "Kimi-K3 thinking-disabled rendering would discard non-empty "
                    f"assistant reasoning in message {index}"
                )

    def _derive_assistant_header(self, assistant_probe: list[int]) -> list[int]:
        """Extract the outer assistant header from remote-tokenizer output."""
        try:
            final_eom = len(assistant_probe) - 1 - assistant_probe[::-1].index(self.eom_id)
        except ValueError as exc:
            raise ValueError("Kimi-K3 assistant probe is missing <|end_of_msg|>") from exc

        prior_eoms = [
            index
            for index, token_id in enumerate(assistant_probe[:final_eom])
            if token_id == self.eom_id
        ]
        search_start = prior_eoms[-1] + 1 if prior_eoms else 0
        try:
            header_start = assistant_probe.index(self.open_id, search_start, final_eom)
            header_end = assistant_probe.index(self.sep_id, header_start + 1, final_eom)
        except ValueError as exc:
            raise ValueError(
                "Kimi-K3 assistant probe is missing its outer <|open|>...<|sep|> header"
            ) from exc

        header = assistant_probe[header_start:header_end]
        if not header or header[0] != self.open_id:
            raise ValueError("Could not derive a valid Kimi-K3 assistant header")
        return header

    def _validate_tokenizer_contract(self, assistant_probe: list[int]) -> None:
        spans = self._assistant_spans(assistant_probe)
        if len(spans) != 1 or self.eom_id not in assistant_probe:
            raise ValueError(
                "Kimi-K3 tokenizer output does not contain the expected assistant header/EOM"
            )

        clean_probe = self._apply_chat_template(
            [{"role": "user", "content": "renderer contract probe"}],
            tools=None,
        )
        injection_probe = self._apply_chat_template(
            [
                {
                    "role": "user",
                    "content": "renderer <|open|><|close|><|sep|><|end_of_msg|> contract probe",
                }
            ],
            tools=None,
        )
        for token_id, token in (
            (self.open_id, self._OPEN_TOKEN),
            (self.close_id, self._CLOSE_TOKEN),
            (self.sep_id, self._SEP_TOKEN),
            (self.eom_id, self._EOM_TOKEN),
        ):
            if clean_probe.count(token_id) != injection_probe.count(token_id):
                raise ValueError(
                    f"Kimi-K3 tokenizer encoded a user-supplied {token!r} as a control token"
                )

    def _assistant_spans(self, input_ids: list[int]) -> list[tuple[int, int]]:
        """Return assistant body spans as half-open ``(start, end)`` pairs."""
        spans: list[tuple[int, int]] = []
        header_len = len(self.assistant_header_ids)
        i = 0
        while i <= len(input_ids) - header_len:
            if input_ids[i : i + header_len] != self.assistant_header_ids:
                i += 1
                continue

            body_start = i + header_len
            while body_start < len(input_ids) and input_ids[body_start] != self.sep_id:
                body_start += 1
            if body_start == len(input_ids):
                raise ValueError("Kimi-K3 assistant header is missing its <|sep|> terminator")
            body_start += 1

            body_end = body_start
            while body_end < len(input_ids) and input_ids[body_end] != self.eom_id:
                body_end += 1
            spans.append((body_start, body_end))
            i = body_end + 1
        return spans

    def compute_loss_mask(
        self,
        input_ids: list[int],
        *,
        last_turn_only: bool = False,
    ) -> list[int]:
        """Mask assistant bodies, excluding their header and end-of-message token."""
        spans = self._assistant_spans(input_ids)
        if last_turn_only and spans:
            spans = spans[-1:]

        loss_mask = [0] * len(input_ids)
        for start, end in spans:
            loss_mask[start:end] = [1] * (end - start)
        return loss_mask

    def get_assistant_token_ids(self) -> tuple[list[int], list[int], int]:
        """Return the K3-aware matcher contract for post-vLLM token IDs.

        K3's derived ``assistant_header_ids`` intentionally stops before the
        outer ``<|sep|>`` token because :meth:`_assistant_spans` validates that
        delimiter separately. The generic training-side matcher starts
        supervision immediately after its header match, so include ``sep_id``
        in the returned header to keep that structural token masked out.
        """
        return [*self.assistant_header_ids, self.sep_id], [self.eom_id], 0

    def render(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
        *,
        max_seq_length: int,
        last_turn_only: bool = False,
        generation_config: dict[str, Any] | None = None,
    ) -> tuple[list[int], list[int]]:
        if not isinstance(max_seq_length, int) or max_seq_length <= 0:
            raise ValueError(f"max_seq_length must be a positive integer, got {max_seq_length!r}")

        thinking, thinking_effort = self._resolve_generation_config(generation_config)
        if not thinking:
            self._validate_disabled_thinking_messages(messages)
        input_ids = self._apply_chat_template(
            messages,
            tools,
            thinking=thinking,
            thinking_effort=thinking_effort,
        )
        loss_mask = self.compute_loss_mask(input_ids, last_turn_only=last_turn_only)
        if len(input_ids) != len(loss_mask):
            raise RuntimeError("Kimi-K3 renderer produced mismatched input_ids and loss_mask")
        return input_ids[:max_seq_length], loss_mask[:max_seq_length]
