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

import json
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import torch
from datasets import IterableDataset, load_dataset
from huggingface_hub import hf_hub_download, list_repo_files

from torchspec.models.ops.loss_mask import compute_assistant_loss_mask

_LOCAL_DATA_EXTS = frozenset({".json", ".jsonl", ".parquet", ".arrow", ".csv", ".tsv", ".txt"})


def is_local_data_path(path: str, base_dir: str | None = None) -> bool:
    """True if *path* looks like a local file/directory rather than a HF Hub dataset ID.

    When *base_dir* is given, relative paths are probed against it instead of
    the process CWD.
    """
    if path.startswith((".", "/", "~")):
        return True
    if os.path.splitext(path)[1].lower() in _LOCAL_DATA_EXTS:
        return True
    probe = os.path.join(base_dir, path) if base_dir is not None else path
    return os.path.exists(probe)


class DataCollatorWithPadding:
    def __init__(self):
        self.sp_degree = 1

    def paddingtensor(self, intensors: torch.Tensor, N: int) -> torch.Tensor:
        B, n, S = intensors.shape
        # Truncate if longer than target (can happen when loss_mask/hidden_states
        # length differs from input_ids after unpacking).
        if n > N:
            return intensors[:, :N, :]
        if n == N:
            return intensors
        padding_tensor = torch.zeros(B, N - n, S, dtype=intensors.dtype, device=intensors.device)
        outtensors = torch.cat((intensors, padding_tensor), dim=1)
        return outtensors

    def paddingtensor2D(self, intensors: torch.Tensor, N: int) -> torch.Tensor:
        B, n = intensors.shape
        # Truncate if longer than target (prevents negative padding dimension
        # when loss_mask length differs from input_ids after collation).
        if n > N:
            return intensors[:, :N]
        if n == N:
            return intensors
        padding_tensor = torch.zeros(B, N - n, dtype=intensors.dtype, device=intensors.device)
        outtensors = torch.cat((intensors, padding_tensor), dim=1)
        return outtensors

    def _get_loss_mask(self, item: Dict[str, Any]) -> torch.Tensor:
        """Read the materialized loss_mask tensor from the item.

        Callers (e.g. MooncakeDataset) are responsible for computing and
        attaching loss_mask before items reach the collator.
        """
        if "loss_mask" in item and isinstance(item["loss_mask"], torch.Tensor):
            return item["loss_mask"]
        raise KeyError(f"loss_mask not found in item: {item}")

    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, Any]:
        max_length = max(item["input_ids"].shape[1] for item in features)
        max_length = ((max_length + self.sp_degree - 1) // self.sp_degree) * self.sp_degree
        # Round up to nearest bucket to reduce unique shapes for torch.compile.
        # Without this, every batch gets a different padded length, causing
        # FlexAttention recompilation (~1s overhead per new shape).
        _BUCKET = 256
        max_length = ((max_length + _BUCKET - 1) // _BUCKET) * _BUCKET

        # All real tokens get attention_mask=1; paddingtensor2D zero-pads the rest.
        attention_masks = [torch.ones_like(item["input_ids"]).long() for item in features]

        batch_input_ids = torch.cat(
            [self.paddingtensor2D(item["input_ids"], max_length) for item in features]
        )
        batch_attention_mask = torch.cat(
            [self.paddingtensor2D(mask, max_length) for mask in attention_masks]
        )
        batch_loss_mask = torch.cat(
            [self.paddingtensor2D(self._get_loss_mask(item), max_length) for item in features]
        )
        batch = {
            "input_ids": batch_input_ids,
            "attention_mask": batch_attention_mask,
            "loss_mask": batch_loss_mask,
            "hidden_states": None,
            "target": None,
            "last_hidden_states": None,
        }
        if all("hidden_states" in item for item in features):
            batch["hidden_states"] = torch.cat(
                [self.paddingtensor(item["hidden_states"], max_length) for item in features]
            )
            has_target = all(item.get("target") is not None for item in features)
            has_last_hs = all(item.get("last_hidden_states") is not None for item in features)
            if not has_target and not has_last_hs:
                raise ValueError(
                    "Either 'target' or 'last_hidden_states' is required when 'hidden_states' is provided"
                )
            if has_target:
                batch["target"] = torch.cat(
                    [self.paddingtensor(item["target"], max_length) for item in features]
                )
            if has_last_hs:
                batch["last_hidden_states"] = torch.cat(
                    [
                        self.paddingtensor(item["last_hidden_states"], max_length)
                        for item in features
                    ]
                )
        return batch


def pack_loss_mask(loss_mask: torch.Tensor) -> List[int]:
    """
    Pack a loss_mask tensor into interleaved segment lengths.

    The returned list alternates between prompt and response lengths,
    always starting with prompt (even if length 0).

    Args:
        loss_mask: 1D tensor of 0s and 1s indicating which tokens contribute to loss.

    Returns:
        List of segment lengths: [prompt_len, response_len, prompt_len, response_len, ...]

    Example:
        loss_mask = [0, 0, 1, 1, 1, 0, 0, 1, 1, 0]
        returns: [2, 3, 2, 2, 1]  # 2 prompt, 3 response, 2 prompt, 2 response, 1 prompt
    """
    if loss_mask.dim() > 1:
        loss_mask = loss_mask.squeeze()

    if len(loss_mask) == 0:
        return []

    lengths = []
    mask_list = loss_mask.tolist()
    current_val = 0
    current_len = 0

    for val in mask_list:
        if val == current_val:
            current_len += 1
        else:
            lengths.append(current_len)
            current_val = val
            current_len = 1

    lengths.append(current_len)
    return lengths


def unpack_loss_mask(packed: Union[List[int], str]) -> torch.Tensor:
    """
    Unpack segment lengths back into a loss_mask tensor.

    Args:
        packed: List of segment lengths [prompt_len, response_len, prompt_len, ...],
                or a serialized packed_loss_mask string (e.g. "2,3,2,2,1").

    Returns:
        1D tensor of 0s and 1s.

    Example:
        packed = [2, 3, 2, 2, 1]
        returns: tensor([0, 0, 1, 1, 1, 0, 0, 1, 1, 0])
    """
    if isinstance(packed, str):
        packed = deserialize_packed_loss_mask(packed)
    if not packed:
        return torch.tensor([], dtype=torch.long)

    total = sum(packed)
    loss_mask = torch.zeros(total, dtype=torch.long)
    pos = 0

    for i, length in enumerate(packed):
        if i % 2 == 1:
            loss_mask[pos : pos + length] = 1
        pos += length

    return loss_mask


def resolve_loss_mask(
    data: Dict[str, Any],
    *,
    dynamic_loss_mask: bool = False,
    assistant_header_ids: Optional[List[int]] = None,
    end_token_ids: Optional[List[int]] = None,
    last_turn_loss_only: bool = False,
    skip_after_header: int = 0,
) -> torch.Tensor | None:
    """
    Two strategies, tried in order:
    1. ``packed_loss_mask`` key present → unpack it.
    2. ``dynamic_loss_mask`` enabled with valid header/end ids → compute from
       ``input_ids`` via :func:`compute_assistant_loss_mask`.
    """
    packed = data.get("packed_loss_mask")
    if packed is not None:
        mask = unpack_loss_mask(packed)
        if not mask.any():
            return None
        data["loss_mask"] = mask
        return mask

    if dynamic_loss_mask and assistant_header_ids and end_token_ids:
        input_ids = data.get("input_ids")
        if input_ids is None:
            return None
        if input_ids.dim() == 2:
            input_ids = input_ids.squeeze(0)
        per_sample = data.get("last_turn_loss_only")
        last_turn_only = per_sample if per_sample is not None else last_turn_loss_only
        mask = compute_assistant_loss_mask(
            input_ids,
            assistant_header_ids,
            end_token_ids,
            last_turn_only=last_turn_only,
            skip_after_header=skip_after_header,
        )
        if not mask.any():
            return None
        data["loss_mask"] = mask
        return mask

    return torch.ones(1)


def serialize_packed_loss_mask(packed: List[int]) -> str:
    """
    Serialize packed loss_mask to a comma-separated string.

    Args:
        packed: List of segment lengths from pack_loss_mask().

    Returns:
        Comma-separated string of integers.

    Example:
        packed = [2, 3, 2, 2, 1]
        returns: "2,3,2,2,1"
    """
    return ",".join(str(x) for x in packed)


def deserialize_packed_loss_mask(s: str) -> List[int]:
    """
    Deserialize a comma-separated string back to packed loss_mask.

    Args:
        s: Comma-separated string of integers.

    Returns:
        List of segment lengths.

    Example:
        s = "2,3,2,2,1"
        returns: [2, 3, 2, 2, 1]
    """
    if not s:
        return []
    return [int(x) for x in s.split(",")]


def extract_media_urls(messages: list) -> dict | None:
    """Extract image/video URLs from structured messages without loading them."""
    images = []
    videos = []

    for msg in messages:
        content = msg.get("content")
        if not isinstance(content, list):
            continue
        for item in content:
            if not isinstance(item, dict):
                continue
            item_type = item.get("type")
            if item_type == "image":
                if "image" in item:
                    images.append(item["image"])
                else:
                    images.append(None)
            elif item_type == "image_url":
                image_url = item.get("image_url")
                if isinstance(image_url, dict) and "url" in image_url:
                    images.append(image_url["url"])
                else:
                    images.append(None)
            elif item_type == "video":
                if "video" in item:
                    videos.append(item["video"])
                else:
                    videos.append(None)

    if not images and not videos:
        return None
    return {"images": images or None, "videos": videos or None}


def flatten_multimodal_content(messages, image_placeholder="<image>"):
    """Convert list-type content parts to plain text strings.

    Transforms the standard HF multimodal format:
      [{"type":"image"}, {"type":"text","text":"Describe"}]
    into a single string:
      "<image>\\nDescribe"

    Messages with string content are left unchanged.
    Must be called AFTER extract_media_urls so structured info is captured first.
    """
    for msg in messages:
        content = msg.get("content")
        if not isinstance(content, (str, list)):
            raise ValueError(
                f"Message content must be a str or list, got {type(content).__name__}: {repr(content)[:100]}"
            )
        if not isinstance(content, list):
            continue
        text_parts = []
        for part in content:
            if not isinstance(part, dict):
                continue
            part_type = part.get("type", "")
            if part_type == "text":
                text_parts.append(part.get("text", ""))
            elif part_type in ("image", "image_url"):
                text_parts.append(image_placeholder)
            elif part_type == "video":
                text_parts.append("<video>")
        msg["content"] = "\n".join(text_parts)
    return messages


def estimate_row_count(data_path):
    if not os.path.isfile(data_path):
        return None
    if data_path.endswith(".jsonl"):
        with open(data_path, "rb") as f:
            return sum(1 for _ in f)
    if data_path.endswith(".parquet"):
        try:
            import pyarrow.parquet as pq

            return pq.ParquetFile(data_path).metadata.num_rows
        except Exception:
            return None
    if data_path.endswith(".json"):
        return None
    return None


def load_local_json(data_path):
    with open(data_path, "r", encoding="utf-8") as f:
        first_char = f.read(1)
        f.seek(0)
        if first_char == "[":
            yield from json.load(f)
        else:
            for line in f:
                line = line.strip()
                if line:
                    yield json.loads(line)


def _load_hub_json_files(data_path):
    """Download JSON/JSONL files from a HF Hub dataset and yield rows.

    Uses raw json module instead of load_dataset to avoid PyArrow schema
    inference failures on datasets with mixed-type columns.
    """

    files = list_repo_files(data_path, repo_type="dataset")
    all_json = sorted(f for f in files if f.endswith((".jsonl", ".json")))
    # Prefer files under data/ to avoid metadata files like stats.json
    data_files = [f for f in all_json if f.startswith("data/")]
    if not data_files:
        data_files = all_json
    if not data_files:
        raise ValueError(
            f"No JSON/JSONL files found in HF Hub dataset '{data_path}'. "
            "Parquet-only Hub datasets are not yet supported via this path."
        )

    for filename in data_files:
        local_path = hf_hub_download(repo_id=data_path, filename=filename, repo_type="dataset")
        yield from load_local_json(local_path)


def load_hf_dataset(data_path: str):
    """Load dataset as a streaming IterableDataset.

    Local paths are loaded directly; everything else goes to HF Hub.
    """
    data_path = os.path.expanduser(data_path)

    if is_local_data_path(data_path):
        if os.path.isfile(data_path):
            if data_path.endswith((".json", ".jsonl")):
                return IterableDataset.from_generator(
                    load_local_json, gen_kwargs={"data_path": data_path}
                )
            ext = os.path.splitext(data_path)[1].lower()
            fmt = {".parquet": "parquet", ".arrow": "arrow"}.get(ext, "json")
            return load_dataset(fmt, data_files=data_path, split="train", streaming=True)

        if os.path.isdir(data_path):
            patterns = {
                "json": ["*.json", "*.jsonl"],
                "parquet": ["*.parquet"],
                "arrow": ["*.arrow"],
            }
            for fmt, globs in patterns.items():
                files = []
                for g in globs:
                    files.extend(str(p) for p in Path(data_path).rglob(g))
                if files:
                    return load_dataset(
                        fmt, data_files=sorted(files), split="train", streaming=True
                    )
            raise ValueError(f"No supported dataset files found in local directory: {data_path}")

        raise FileNotFoundError(f"Local dataset path not found: {data_path}")

    # hub path — try native load_dataset first (handles Arrow, Parquet, etc.),
    # fall back to manual JSON download for repos with mixed-type columns
    _KEEP_COLUMNS = frozenset({"id", "conversations", "text", "messages"})
    try:
        ds = load_dataset(data_path, split="train", streaming=True)
        drop_cols = [c for c in (ds.column_names or []) if c not in _KEEP_COLUMNS]
        if drop_cols:
            ds = ds.remove_columns(drop_cols)
        return ds
    except Exception:
        return IterableDataset.from_generator(_load_hub_json_files, gen_kwargs={"data_path": data_path})
