"""Cross-validation: compute_assistant_loss_mask must match parser.parse for every template.

CI gate — auto-parametrizes over all registered templates so adding a new
template without verifying loss mask correctness will fail the test.

Requires tokenizer downloads (~10MB each).  Templates whose tokenizer is
unavailable are skipped.
"""

import pytest
import torch

from torchspec.data.parse import create_parser
from torchspec.data.template import TEMPLATE_REGISTRY
from torchspec.models.ops.loss_mask import compute_assistant_loss_mask

_END_TOKEN_BPE_XFAIL: set[str] = set()

MESSAGES_MULTI_TURN = [
    {"role": "user", "content": "What is 2+2?"},
    {"role": "assistant", "content": "The answer is 4."},
    {"role": "user", "content": "And 3+3?"},
    {"role": "assistant", "content": "The answer is 6."},
]

MESSAGES_LEADING_NEWLINES = [
    {"role": "user", "content": "What is 2+2?"},
    {"role": "assistant", "content": "\n\nThe answer is 4."},
]

_tokenizer_cache: dict = {}


def _testable_templates():
    """Templates that support dynamic loss mask (need both header and end token)."""
    for name in TEMPLATE_REGISTRY.get_all_template_names():
        template = TEMPLATE_REGISTRY.get(name)
        if template.assistant_header and template.end_of_turn_token:
            yield name


def _get_tokenizer(template_name):
    template = TEMPLATE_REGISTRY.get(template_name)
    model_path = template.reference_model
    if model_path is None:
        pytest.skip(f"No reference_model for template '{template_name}'")

    if model_path in _tokenizer_cache:
        return _tokenizer_cache[model_path]

    from transformers import AutoTokenizer

    try:
        tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    except Exception as e:
        pytest.skip(f"Tokenizer unavailable for {model_path}: {e}")

    _tokenizer_cache[model_path] = tokenizer
    return tokenizer


def _get_header_ids_and_skip(tokenizer, template):
    """Replicate get_assistant_token_ids() logic for the test."""
    full_ids = tokenizer.encode(template.assistant_header, add_special_tokens=False)
    stripped = template.assistant_header.rstrip("\n")
    stripped_ids = tokenizer.encode(stripped, add_special_tokens=False)
    skip_after = len(full_ids) - len(stripped_ids)
    end_ids = tokenizer.encode(template.end_of_turn_token, add_special_tokens=False)
    return stripped_ids, end_ids, skip_after


def _first_diff(a, b):
    for i, (x, y) in enumerate(zip(a, b)):
        if x != y:
            return i
    if len(a) != len(b):
        return min(len(a), len(b))
    return None


def _strip_end_tokens_from_mask(mask_list, ids_list, end_ids):
    """Zero out end-of-turn token positions in a mask.

    The parser regex captures content *including* the end_of_turn_token,
    but compute_assistant_loss_mask marks content *excluding* it.
    This normalizes the parser mask to match the dynamic mask convention.
    """
    result = list(mask_list)
    end_len = len(end_ids)
    i = 0
    while i <= len(ids_list) - end_len:
        if ids_list[i : i + end_len] == end_ids:
            for k in range(end_len):
                result[i + k] = 0
            i += end_len
        else:
            i += 1
    return result


def _cross_validate(template_name, messages, last_turn_only=False):
    """Core cross-validation: parser ground truth vs dynamic mask."""
    template = TEMPLATE_REGISTRY.get(template_name)
    tokenizer = _get_tokenizer(template_name)
    parser = create_parser(tokenizer, template)
    header_ids, end_ids, skip_after = _get_header_ids_and_skip(tokenizer, template)

    formatted = parser.format(messages, add_generation_prompt=False, expand_media_tokens=False)

    gt_ids, gt_mask = parser.parse(
        formatted, max_length=200000, preformatted=True, last_turn_only=last_turn_only
    )
    gt_ids_list = gt_ids.squeeze().tolist()
    gt_mask_list = gt_mask.squeeze().tolist()

    if sum(gt_mask_list) == 0:
        pytest.skip(
            f"Parser produced all-zero mask for '{template_name}' — "
            f"template header may not match tokenizer's chat template"
        )

    engine_ids = tokenizer.encode(formatted, add_special_tokens=False)

    assert gt_ids_list == engine_ids, (
        f"[{template_name}] Tokenization mismatch: "
        f"parser produced {len(gt_ids_list)} tokens, "
        f"tokenizer.encode produced {len(engine_ids)} tokens"
    )

    # Parser includes end_of_turn_token in mask; dynamic mask does not.
    normalized_gt = _strip_end_tokens_from_mask(gt_mask_list, engine_ids, end_ids)

    assert sum(normalized_gt) > 0, (
        f"[{template_name}] After stripping end tokens, parser mask is all zeros — "
        f"this likely means the parser only matched end tokens as content"
    )

    dyn_mask = compute_assistant_loss_mask(
        torch.tensor(engine_ids),
        header_ids,
        end_ids,
        last_turn_only=last_turn_only,
        skip_after_header=skip_after,
    )

    diff_idx = _first_diff(normalized_gt, dyn_mask.tolist())
    assert normalized_gt == dyn_mask.tolist(), (
        f"[{template_name}] Loss mask mismatch "
        f"(last_turn_only={last_turn_only}):\n"
        f"  parser mask 1s (normalized): {sum(normalized_gt)}\n"
        f"  dynamic mask 1s: {dyn_mask.sum().item()}\n"
        f"  first diff at token {diff_idx}"
    )


# ── Parametrized tests ───────────────────────────────────────────────


def _maybe_xfail(template_name):
    if template_name in _END_TOKEN_BPE_XFAIL:
        pytest.xfail(
            f"'{template_name}' has a known end-token BPE merge issue — "
            f"use precomputed masks (defer_tokenization=False) for this template"
        )


@pytest.mark.parametrize("template_name", list(_testable_templates()))
def test_multi_turn(template_name):
    """Dynamic mask matches parser on a standard multi-turn conversation."""
    _maybe_xfail(template_name)
    _cross_validate(template_name, MESSAGES_MULTI_TURN)


@pytest.mark.parametrize("template_name", list(_testable_templates()))
def test_multi_turn_last_turn_only(template_name):
    """Dynamic mask matches parser with last_turn_only=True."""
    _maybe_xfail(template_name)
    _cross_validate(template_name, MESSAGES_MULTI_TURN, last_turn_only=True)


@pytest.mark.parametrize("template_name", list(_testable_templates()))
def test_leading_newlines(template_name):
    """Dynamic mask matches parser when content starts with newlines (BPE merge edge case)."""
    _maybe_xfail(template_name)
    _cross_validate(template_name, MESSAGES_LEADING_NEWLINES)
