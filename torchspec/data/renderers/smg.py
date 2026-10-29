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

"""SMG rendering/encoding with TorchSpec-owned supervision.

The first integration accepts complete text conversations with explicit
assistant/end delimiters. Unsupported encodings and ambiguous boundaries are
errors, not a reason to silently fall back to a different tokenizer.
"""

import hashlib
import json
from copy import deepcopy
from pathlib import Path

from packaging.version import Version

from torchspec.data.template import TEMPLATE_REGISTRY

# Default routing is deliberately bounded to the supported text profiles.
# Additional templates can still be requested explicitly with backend="smg".
SMG_DEFAULT_TEMPLATES = frozenset({"llama3", "qwen", "qwen3-thinking", "minimax-m2"})


def effective_renderer(args):
    """Select a renderer for the requested backend and training mode.

    Args:
        args: Namespace with optional rendering_backend, renderer,
            target_model_path, chat_template, defer_tokenization and
            train_with_decode attributes. K3 detection
            also reads model_type from a local checkpoint's config.json.

    Returns:
        "smg" or "smg-kimi-k3" for supported SMG routes, "kimi-k3" for the
        legacy K3 route, an explicit custom renderer name, or None for the
        legacy template parser. Auto routing preserves unsupported modes and
        templates on the legacy path; explicit custom renderers are preserved.

    Raises:
        ValueError: The backend is unknown, forced SMG would override a custom
            renderer, or an SMG route requires an incompatible training mode.
    """
    backend = getattr(args, "rendering_backend", "auto")
    renderer = getattr(args, "renderer", None)
    if backend not in ("auto", "huggingface", "smg"):
        raise ValueError("dataset.rendering_backend must be 'auto', 'huggingface' or 'smg'")
    # Preserve explicit custom renderers, including on K3 model paths.
    if renderer not in (None, "smg", "kimi-k3", "smg-kimi-k3"):
        if backend == "smg":
            raise ValueError(f"SMG cannot override explicit dataset.renderer={renderer!r}")
        return renderer
    model_path = str(getattr(args, "target_model_path", ""))
    is_k3 = renderer in ("kimi-k3", "smg-kimi-k3") or "kimi-k3" in model_path.lower().replace(
        "_", "-"
    )
    config = Path(model_path) / "config.json"
    if config.is_file():
        model_type = json.loads(config.read_text()).get("model_type", "")
        is_k3 |= "kimi-k3" in model_type.lower().replace("_", "-")
    legacy_renderer = "kimi-k3" if is_k3 else None
    # This also allows switching an already-resolved namespace back to legacy.
    if backend == "huggingface":
        return legacy_renderer
    incompatible = getattr(args, "defer_tokenization", False) or getattr(
        args, "train_with_decode", False
    )
    if backend == "auto":
        if incompatible:
            return legacy_renderer
        if (
            not is_k3
            and renderer != "smg"
            and getattr(args, "chat_template", None) not in SMG_DEFAULT_TEMPLATES
        ):
            return legacy_renderer
    if incompatible:
        raise ValueError(
            "SMG rendering requires defer_tokenization=false and train_with_decode=false; "
            "set dataset.rendering_backend=huggingface for the legacy path"
        )
    return "smg-kimi-k3" if is_k3 else "smg"


def prepare_checkpoint(source, revision=None):
    """Resolve once before multiprocessing, and fingerprint actual tokenizer assets."""
    try:
        from torchspec_smg import __version__ as version
        from torchspec_smg import build_features

        if Version(version) < Version("0.1.2"):
            raise ImportError(f"smg-tokenizer {version} is older than the required 0.1.2")
    except ImportError as exc:
        raise ImportError(
            "SMG rendering needs TorchSpec's bundled native binding >=0.1.2; "
            "reinstall TorchSpec with its native extension (see docs/smg-tokenization.md), or set "
            "dataset.rendering_backend=huggingface to use the legacy tokenizer"
        ) from exc

    root = Path(source)
    if not root.is_dir():
        from huggingface_hub import snapshot_download

        root = Path(
            snapshot_download(
                source,
                revision=revision,
                allow_patterns=[
                    "*.json",
                    "*.jinja",
                    "*.model",
                    "*.tiktoken",
                    "*.py",
                    "merges.txt",
                    "vocab.txt",
                ],
            )
        )
    digest = hashlib.sha256(
        f"smg-tokenizer={version};features={build_features};adapter={SmgRenderer.CACHE_VERSION}".encode()
    )
    for path in sorted(root.rglob("*")):
        relative = path.relative_to(root)
        if any(part.startswith(".") for part in relative.parts) or not path.is_file():
            continue
        if path.suffix not in (
            ".json",
            ".jinja",
            ".model",
            ".tiktoken",
            ".py",
        ) and path.name not in ("merges.txt", "vocab.txt"):
            continue
        digest.update(str(relative).encode())
        digest.update(b"\0")
        with path.open("rb") as stream:
            for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(chunk)
    return str(root.resolve()), digest.hexdigest()


def _strings(value):
    if isinstance(value, str):
        yield value
    elif isinstance(value, dict):
        for key, item in value.items():
            yield from _strings(key)
            yield from _strings(item)
    elif isinstance(value, list):
        for item in value:
            yield from _strings(item)


class SmgRenderer:
    """Render complete text conversations with the bundled SMG tokenizer.

    Assistant supervision is derived from template delimiter spans and aligned
    token offsets. Ambiguous delimiters, native segmented encodings and dropped
    supervised reasoning raise errors rather than selecting another backend.
    """

    CACHE_VERSION = "text-spans-v2-eot-marker-only"

    def __init__(
        self,
        source,
        chat_template_name,
        *,
        template_kwargs=None,
        include_end_token=True,
        render_instant=None,
    ):
        from torchspec_smg import Tokenizer

        self.template = TEMPLATE_REGISTRY.get(chat_template_name)
        self.header = self.template.assistant_header
        self.end = self.template.end_of_turn_token
        if not self.header or not self.end:
            raise ValueError(
                "SMG supervision needs a chat_template with assistant_header and end_of_turn_token"
            )
        self.tokenizer = Tokenizer(source)
        self.template_kwargs = dict(template_kwargs or {})
        self.include_end_token = include_end_token
        self.render_instant = render_instant
        root = Path(source)
        templates = [root / "tokenizer_config.json", *root.rglob("*.jinja")]
        if render_instant is None and any(
            p.is_file() and "strftime_now" in p.read_text() for p in templates
        ):
            raise ValueError(
                "Clock-dependent SMG templates require dataset.smg_render_instant (RFC3339)"
            )

    def render(
        self,
        messages,
        tools=None,
        *,
        max_seq_length,
        last_turn_only=False,
        generation_config=None,
        return_formatted_text=False,
    ):
        """Render and tokenize a complete conversation with an assistant loss mask.

        Args:
            messages: Message dictionaries with system, developer, user,
                assistant or tool roles and string or null content. Reasoning
                fields and tool calls are passed to SMG; inputs are copied.
            tools: Optional tool definitions passed to the chat template.
            max_seq_length: Positive integer limiting both returned sequences.
            last_turn_only: Supervise only the final assistant span, selected
                before right truncation, even if truncation removes that span.
            generation_config: Optional dictionary whose chat_template_kwargs
                and explicit thinking/reasoning settings override the renderer's
                template kwargs for this row.
            return_formatted_text: Append the same render's full, untruncated text
                as a third result; it is metadata, never a source for re-encoding.

        Returns:
            An (input_ids, loss_mask) tuple of equally sized lists, right-truncated
            to max_seq_length. The binary mask covers selected assistant spans
            and, when configured, their end markers. Tokens crossing a span
            boundary are excluded when offsets are available.

        Raises:
            ValueError: The length, roles, content, generation settings or tool
                JSON are invalid; payloads forge delimiters; output is segmented;
                assistant spans are unclosed, cross roles or have the wrong count;
                tool calls or supervised reasoning are dropped; or token offsets
                or offset-free encoding boundaries cannot be aligned safely.
        """
        if (
            not isinstance(max_seq_length, int)
            or isinstance(max_seq_length, bool)
            or max_seq_length <= 0
        ):
            raise ValueError("max_seq_length must be a positive integer")
        messages = deepcopy(messages)
        for message in messages:
            if message.get("role") not in ("system", "developer", "user", "assistant", "tool"):
                raise ValueError(f"Unsupported role: {message.get('role')!r}")
            content = message.get("content")
            if content is not None and not isinstance(content, str):
                raise ValueError(
                    "SMG training currently accepts text-only string content; for multimodal training "
                    "set dataset.rendering_backend=huggingface"
                )
            if not self.tokenizer.raw_tool_call_arguments:
                for call in message.get("tool_calls") or []:
                    function = call.get("function", call)
                    arguments = function.get("arguments")
                    if isinstance(arguments, str):
                        try:
                            function["arguments"] = json.loads(arguments)
                        except json.JSONDecodeError as exc:
                            raise ValueError(
                                "Tool arguments must contain valid JSON for this SMG renderer"
                            ) from exc
        # Delimiter searching is only safe when payloads cannot forge those
        # delimiters. Fail instead of training user/tool text as assistant text.
        delimiters = {self.header.rstrip("\n"), self.end.rstrip("\n")}
        if self.template.user_header:
            delimiters.add(self.template.user_header.rstrip("\n"))
        for text in _strings([messages, tools]):
            if any(delimiter in text for delimiter in delimiters):
                raise ValueError(
                    "Literal chat delimiters in message/tool content make SMG supervision ambiguous"
                )
        if self.template.system_prompt and not any(m["role"] == "system" for m in messages):
            messages.insert(0, {"role": "system", "content": self.template.system_prompt})
        kwargs = {"preserve_thinking": True, **self.template_kwargs}
        if self.template.enable_thinking:
            kwargs.setdefault("enable_thinking", True)
        if generation_config is not None:
            if not isinstance(generation_config, dict):
                raise ValueError("generation_config must be a dictionary")
            row_kwargs = generation_config.get("chat_template_kwargs", {})
            if not isinstance(row_kwargs, dict):
                raise ValueError("generation_config.chat_template_kwargs must be a dictionary")
            kwargs.update(row_kwargs)
            for key in (
                "thinking",
                "enable_thinking",
                "thinking_effort",
                "reasoning_effort",
                "preserve_thinking",
            ):
                if key in generation_config:
                    kwargs[key] = generation_config[key]
        rendered = self.tokenizer.render(
            messages, tools=tools, template_kwargs=kwargs, now=self.render_instant
        )
        text, ids = rendered["text"], rendered["input_ids"]
        if rendered["segmentwise"]:
            raise ValueError(
                "SMG native segment encoding needs training span annotations; use the existing renderer for this model"
            )
        expected = sum(message["role"] == "assistant" for message in messages)
        spans = []
        cursor = 0
        while (start := text.find(self.header, cursor)) >= 0:
            body = start + len(self.header)
            end = text.find(self.end, body)
            if end < 0:
                raise ValueError(
                    "SMG output has an unclosed assistant turn; check dataset.chat_template"
                )
            for delimiter in (self.header, self.template.user_header):
                if delimiter and text.find(delimiter, body, end) >= 0:
                    raise ValueError("SMG assistant span crosses a role boundary")
            # Registry delimiters sometimes include a formatting newline
            # (Qwen), sometimes only the marker (MiniMax). End-token loss has
            # one meaning across profiles: supervise the marker itself, not
            # the newline separating it from the next role.
            spans.append(
                (body, end + len(self.end.rstrip("\n")) if self.include_end_token else end)
            )
            cursor = end + len(self.end)
        if len(spans) != expected:
            raise ValueError(
                f"SMG rendered {len(spans)} assistant spans for {expected} assistant messages; check dataset.chat_template"
            )
        assistants = [m for m in messages if m["role"] == "assistant"]
        for message, (a, b) in zip(assistants, spans):
            for call in message.get("tool_calls") or []:
                name = call.get("function", call).get("name")
                if name and name not in text[a:b]:
                    raise ValueError("SMG template dropped a tool call from the conversation")
        selected = list(zip(assistants, spans))
        if last_turn_only:
            selected = selected[-1:]
        for message, (a, b) in selected:
            body = text[a:b]
            for key in ("reasoning_content", "reasoning", "thinking", "thinking_content"):
                reasoning = message.get(key)
                if (
                    isinstance(reasoning, str)
                    and reasoning.strip()
                    and reasoning.strip() not in body
                ):
                    raise ValueError(
                        "SMG template dropped supervised reasoning; use compatible template options or last-turn-only loss"
                    )
        if last_turn_only:
            spans = spans[-1:]
        mask = [0] * len(ids)
        offsets = rendered["offsets"]
        if offsets is not None:
            if len(offsets) != len(ids):
                raise ValueError("SMG offsets do not align with input_ids")
            byte_spans = [(len(text[:a].encode()), len(text[:b].encode())) for a, b in spans]
            # A boundary-straddling token is conservatively unsupervised.
            # Prefix re-encoding would incorrectly assume BPE compositionality.
            region = 0
            for i, (a, b) in enumerate(offsets):
                while region < len(byte_spans) and a >= byte_spans[region][1]:
                    region += 1
                if region < len(byte_spans):
                    lo, hi = byte_spans[region]
                    mask[i] = int(a < b and lo <= a and b <= hi)
        else:
            # A flat tiktoken path has no offsets. Only accept exact prefix
            # boundaries; never assume encode(prefix) is a prefix of encode(all).
            for a, b in spans:
                left, right = (self.tokenizer.encode(text[:p]) for p in (a, b))
                if ids[: len(left)] != left or ids[: len(right)] != right:
                    raise ValueError(
                        "SMG tokenizer has no offsets and BPE crosses a supervision boundary"
                    )
                mask[len(left) : len(right)] = [1] * (len(right) - len(left))
        output = (ids[:max_seq_length], mask[:max_seq_length])
        return (*output, text) if return_formatted_text else output
