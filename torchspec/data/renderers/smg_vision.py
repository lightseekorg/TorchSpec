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

"""Opt-in Qwen3-VL image preparation with final token IDs and supervision."""

import base64
import hashlib
import sys
from pathlib import Path
from urllib.parse import unquote, urlparse
from urllib.request import urlopen

import numpy as np

from torchspec.data.renderers.smg import SmgRenderer, _strings


def _image_bytes(source):
    if not isinstance(source, str):
        raise ValueError("image_url must contain a string URL")
    limit = 64 * 1024 * 1024
    if source.startswith("data:"):
        header, data = source.split(",", 1)
        if not header.startswith("data:image/") or not header.endswith(";base64"):
            raise ValueError("Images require base64 image data URLs")
        if len(data) > (limit * 4 // 3 + 4):
            raise ValueError("Encoded image exceeds 64 MiB")
        result = base64.b64decode(data, validate=True)
    else:
        parsed = urlparse(source)
        if parsed.scheme in ("http", "https"):
            with urlopen(source, timeout=30) as stream:
                result = stream.read(limit + 1)
        elif parsed.scheme in ("", "file") and not parsed.netloc:
            with Path(unquote(parsed.path)).open("rb") as stream:
                result = stream.read(limit + 1)
        else:
            raise ValueError("Images require local files, HTTP(S), or base64 data URLs")
    if len(result) > limit:
        raise ValueError("Encoded image exceeds 64 MiB")
    return result


class SmgVisionRenderer(SmgRenderer):
    """Prepare Qwen3-VL user/tool images through SMG; vLLM consumes the result directly."""

    CACHE_VERSION = "smg-qwen3-vl-v1-" + SmgRenderer.CACHE_VERSION

    def __init__(self, source, **options):
        from torchspec_smg import ImageProcessor

        super().__init__(source, "qwen", **options)
        # The Qwen3-VL checkpoint does not inject the text-Qwen registry's
        # default system message. Reuse only its supervision delimiters.
        self.template = self.template.model_copy(update={"system_prompt": None})
        self.image_processor = ImageProcessor(source)
        self.image_id = self.tokenizer.token_to_id("<|image_pad|>")
        self.vision_start_id = self.tokenizer.token_to_id("<|vision_start|>")
        self.vision_end_id = self.tokenizer.token_to_id("<|vision_end|>")
        if None in (self.image_id, self.vision_start_id, self.vision_end_id):
            raise ValueError("Qwen3-VL checkpoint is missing required vision tokens")

    def _validate_content(self, message):
        content = message.get("content")
        if message.get("role") not in ("system", "user", "assistant", "tool"):
            raise ValueError("Unsupported Qwen3-VL message role")
        if isinstance(content, list):
            for part in content:
                if not isinstance(part, dict):
                    raise ValueError("Multimodal content parts must be dictionaries")
                if part.get("type") == "text" and set(part) == {"type", "text"}:
                    if not isinstance(part["text"], str):
                        raise ValueError("Text parts require string text")
                elif part.get("type") == "image_url" and set(part) == {"type", "image_url"}:
                    if message["role"] not in ("user", "tool"):
                        raise ValueError("Only user/tool images are supported")
                    ref = part["image_url"]
                    if (
                        not isinstance(ref, dict)
                        or set(ref) - {"url", "detail"}
                        or not isinstance(ref.get("url"), str)
                    ):
                        raise ValueError("image_url requires a URL and optional detail=auto")
                    if ref.get("detail", "auto") != "auto":
                        raise ValueError("Only detail=auto is supported")
                else:
                    raise ValueError("Only text and image_url parts are supported; no video/audio")
        else:
            super()._validate_content(message)

    def render(self, *args, **kwargs):
        raise ValueError(
            "Use render_prepared for Qwen3-VL so image tensors and final IDs stay together"
        )

    def render_prepared(
        self, messages, tools=None, *, max_seq_length, last_turn_only=False, generation_config=None
    ):
        """Return final IDs, aligned loss mask, and an optional prepared image payload.

        Rendering uses the checkpoint template via SMG. Image replacements come
        from SMG's model registry. Truncation through an image is rejected; fully
        truncated images are removed from both the sequence and tensor batch.
        """
        if (
            not isinstance(max_seq_length, int)
            or isinstance(max_seq_length, bool)
            or max_seq_length <= 0
        ):
            raise ValueError("max_seq_length must be a positive integer")
        for text in _strings([messages, tools, generation_config]):
            if any(
                marker in text
                for marker in (
                    "<|vision_start|>",
                    "<|vision_end|>",
                    "<|image_pad|>",
                    "<|video_pad|>",
                )
            ):
                raise ValueError("Literal vision delimiters make image ownership ambiguous")
        ids, mask = super().render(
            messages,
            tools,
            max_seq_length=sys.maxsize,
            last_turn_only=last_turn_only,
            generation_config=generation_config,
        )
        sources = [
            part["image_url"]["url"]
            for message in messages
            if isinstance(message.get("content"), list)
            for part in message["content"]
            if part.get("type") == "image_url"
        ]
        positions = [i for i, token in enumerate(ids) if token == self.image_id]
        if len(positions) != len(sources):
            raise ValueError("SMG image placeholders do not match the supplied images")
        if any(
            position == 0
            or position + 1 == len(ids)
            or ids[position - 1] != self.vision_start_id
            or ids[position + 1] != self.vision_end_id
            for position in positions
        ):
            raise ValueError("SMG image placeholder is missing its Qwen3-VL vision wrapper")
        if not sources:
            return ids[:max_seq_length], mask[:max_seq_length], None
        processed = self.image_processor.process([_image_bytes(source) for source in sources])
        replacements = processed.pop("replacements")
        if len(replacements) != len(positions):
            raise ValueError("SMG returned the wrong number of image replacements")
        final_ids, final_mask, ranges = [], [], []
        cursor = 0
        kept = 0
        for position, replacement in zip(positions, replacements):
            if (
                mask[position]
                or not replacement
                or any(token != self.image_id for token in replacement)
            ):
                raise ValueError("Invalid Qwen3-VL image replacement or supervised image")
            final_ids.extend(ids[cursor:position])
            final_mask.extend(mask[cursor:position])
            start, length = len(final_ids), len(replacement)
            if start - 1 < max_seq_length < start + length + 1:
                raise ValueError(
                    "Truncation cuts through an image; increase max_seq_length or reduce image size"
                )
            if start + length + 1 <= max_seq_length:
                ranges.append((start, length))
                kept += 1
            final_ids.extend(replacement)
            final_mask.extend([0] * length)
            cursor = position + 1
        final_ids.extend(ids[cursor:])
        final_mask.extend(mask[cursor:])
        if not kept:
            return final_ids[:max_seq_length], final_mask[:max_seq_length], None
        grid = processed["image_grid_thw"][:kept]
        sizes = grid.prod(axis=1)
        pixels = processed["pixel_values"][: int(sizes.sum())]
        hashes, offset = [], 0
        for item, size in zip(grid, sizes):
            digest = hashlib.sha256(self.CACHE_VERSION.encode() + item.tobytes())
            digest.update(pixels[offset : offset + size].tobytes())
            hashes.append(digest.hexdigest())
            offset += size
        payload = {
            "format": "torchspec-smg-qwen3-vl-v1",
            "pixel_values": np.ascontiguousarray(pixels),
            "image_grid_thw": np.ascontiguousarray(grid),
            "image_ranges": ranges,
            "image_hashes": hashes,
            "image_token_id": self.image_id,
        }
        return final_ids[:max_seq_length], final_mask[:max_seq_length], payload
