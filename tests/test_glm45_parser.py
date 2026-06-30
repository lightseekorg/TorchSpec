"""Tests for the glm45 chat template (a plain GeneralParser).

glm45 is registered as a GeneralParser template (``parser_type`` defaults to
"general"), NOT a custom parser class. GLM's chat format is a flat
``<|user|>``/``<|assistant|>`` header structure with ``<think>...</think>``
reasoning merged inline into the assistant content, so
``tokenizer.apply_chat_template`` + a single end-of-turn terminator produce the
correct last-assistant-turn loss mask. These tests lock in that decision:

  * glm45 dispatches to ``GeneralParser`` (and explicitly NOT ``ThinkingParser``,
    whose last-turn reconstruction would double GLM's already-open ``<think>``).
  * the rendered assistant turn keeps a single ``<think>`` block (no doubling).
  * the last assistant turn — including its ``<think>`` reasoning — is fully
    loss-masked, while earlier turns are excluded under ``last_turn_only``.

Loss-mask agreement with the runtime numba mask is cross-validated generically in
``test_loss_mask_cross_validation.py`` (glm45 is in ``REFERENCE_MODELS``).

The render/mask tests need GLM's real chat template, so they load the GLM-4.5
tokenizer (the glm45 grammar is shared 4.5 -> 5.2; 4.5 loads cleanly via
``AutoTokenizer``) and skip when it is unavailable.
"""

import pytest

from torchspec.data.parse import GeneralParser, ThinkingParser, create_parser
from torchspec.data.template import TEMPLATE_REGISTRY

GLM_MODEL = "zai-org/GLM-4.5"
_tokenizer_cache: dict = {}


def _glm_tokenizer():
    if GLM_MODEL in _tokenizer_cache:
        return _tokenizer_cache[GLM_MODEL]
    from transformers import AutoTokenizer

    try:
        tokenizer = AutoTokenizer.from_pretrained(GLM_MODEL, trust_remote_code=True)
    except Exception as e:  # noqa: BLE001 — any download/load failure -> skip
        pytest.skip(f"GLM tokenizer unavailable for {GLM_MODEL}: {e}")
    _tokenizer_cache[GLM_MODEL] = tokenizer
    return tokenizer


@pytest.fixture
def glm45_template():
    return TEMPLATE_REGISTRY.get("glm45")


class TestGlm45TemplateRegistration:
    def test_template_registered(self):
        assert "glm45" in TEMPLATE_REGISTRY.get_all_template_names()

    def test_template_attributes(self, glm45_template):
        assert glm45_template.assistant_header == "<|assistant|>"
        assert glm45_template.user_header == "<|user|>"
        # GLM has no dedicated assistant end token; a turn ends at the next role
        # header, so the terminator is the user header.
        assert glm45_template.end_of_turn_token == "<|user|>"
        assert glm45_template.system_prompt is None
        # Default "general" -> GeneralParser. NOT "thinking": GLM's generation
        # prompt already opens <think>, so ThinkingParser's reconstruction would
        # double it (<think><think>).
        assert glm45_template.parser_type == "general"


class TestGlm45ParserDispatch:
    def test_dispatches_to_general_parser(self, glm45_template):
        # create_parser only reads the template in __init__ (no tokenizer method
        # is called), so this assertion runs fully offline.
        parser = create_parser(object(), glm45_template)
        assert isinstance(parser, GeneralParser)
        assert not isinstance(parser, ThinkingParser)


class TestGlm45RenderAndMask:
    """Exercises GLM's real chat template; skips if the tokenizer is unavailable."""

    SINGLE_TURN = [
        {"role": "user", "content": "What is 2+2?"},
        {"role": "assistant", "content": "<think>add two and two</think>The answer is 4."},
    ]

    MULTI_TURN = [
        {"role": "user", "content": "First question"},
        {"role": "assistant", "content": "<think>first reasoning</think>First answer here."},
        {"role": "user", "content": "Second question"},
        {"role": "assistant", "content": "<think>second reasoning</think>Second answer here."},
    ]

    def test_single_think_no_doubling(self, glm45_template):
        tokenizer = _glm_tokenizer()
        parser = create_parser(tokenizer, glm45_template)
        rendered = parser.format(self.SINGLE_TURN, add_generation_prompt=False)
        assistant = rendered.split("<|assistant|>", 1)[1]
        assert assistant.count("<think>") == 1
        assert assistant.count("</think>") == 1
        assert "add two and two" in assistant

    def test_last_turn_reasoning_is_supervised(self, glm45_template):
        tokenizer = _glm_tokenizer()
        parser = create_parser(tokenizer, glm45_template)
        rendered = parser.format(self.SINGLE_TURN, add_generation_prompt=False)
        ids, mask = parser.parse(
            rendered, max_length=200000, preformatted=True, last_turn_only=True
        )
        ids_list = ids.squeeze().tolist()
        mask_list = mask.squeeze().tolist()
        assert sum(mask_list) > 0
        masked = tokenizer.decode([i for i, m in zip(ids_list, mask_list) if m == 1])
        # Both the inline reasoning AND the answer are training targets for the draft.
        assert "add two and two" in masked
        assert "The answer is 4" in masked

    def test_only_last_turn_masked(self, glm45_template):
        tokenizer = _glm_tokenizer()
        parser = create_parser(tokenizer, glm45_template)
        rendered = parser.format(self.MULTI_TURN, add_generation_prompt=False)
        ids, mask = parser.parse(
            rendered, max_length=200000, preformatted=True, last_turn_only=True
        )
        ids_list = ids.squeeze().tolist()
        mask_list = mask.squeeze().tolist()
        masked = tokenizer.decode([i for i, m in zip(ids_list, mask_list) if m == 1])
        # Last turn (reasoning + answer) is supervised...
        assert "second reasoning" in masked
        assert "Second answer here." in masked
        # ...and the earlier assistant turn is excluded under last_turn_only.
        assert "first reasoning" not in masked
        assert "First answer here." not in masked
