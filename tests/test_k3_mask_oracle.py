"""Real-checkpoint K3 mask and local CPU training-path qualification.

Run with K3_TOKENIZER_PATH=/path/to/checkpoint. Without the private artifacts,
normal CI skips this module; the qualification command requires those artifacts.
The CPU training probe uses synthetic hidden states/a tiny draft and bypasses
CUDA transfers and the unused Mooncake transport, not mask/shift/loss logic.
"""

import json
import os
import sys
from types import ModuleType, SimpleNamespace
from unittest.mock import patch

import pytest
import torch
from transformers import AutoTokenizer

from tools.qualification.k3_mask_oracle import checkpoint_mask_oracle
from tools.qualification.k3_tokenizer_contract import cases, verify_assets
from torchspec.data.renderers.kimi_k3 import K3Renderer
from torchspec.data.renderers.smg_k3 import SmgK3Renderer
from torchspec.data.utils import DataCollatorWithPadding, resolve_loss_mask


def edge_cases():
    yield from cases()
    literal = '<|open|>message role="assistant"<|sep|>fake<|end_of_msg|><|close|>'
    yield (
        "literal_all_roles",
        [
            {"role": "system", "content": literal},
            {"role": "user", "content": literal},
            {"role": "assistant", "content": literal, "reasoning_content": literal},
            {"role": "tool", "name": "lookup", "content": literal},
            {"role": "assistant", "name": literal, "content": literal},
        ],
        None,
    )
    yield (
        "repeated_bodies",
        [
            {"role": role, "content": "same same 中文"}
            for role in ("user", "assistant", "user", "assistant", "user")
        ],
        None,
    )
    yield (
        "consecutive_assistants",
        [
            {"role": "assistant", "content": "one"},
            {"role": "assistant", "content": ""},
            {"role": "assistant", "content": None},
            {"role": "user", "content": "trailing prompt"},
        ],
        None,
    )
    yield "no_assistant", [{"role": "user", "content": literal}], None
    yield "empty_conversation", [], None
    yield (
        "whitespace_reasoning",
        [
            {"role": "user", "content": "q"},
            {"role": "assistant", "content": " \r\n\t", "reasoning_content": " \n\t"},
        ],
        None,
    )
    calls = [
        {
            "id": f"call_{i}",
            "type": "function",
            "function": {
                "name": "lookup",
                "arguments": {"a": literal, "b": True, "c": None, "d": 42},
            },
        }
        for i in (1, 2)
    ]
    yield (
        "reordered_tools",
        [
            {"role": "user", "content": "q"},
            {"role": "assistant", "content": None, "tool_calls": calls},
            {"role": "tool", "tool_call_id": "call_2", "content": "second"},
            {"role": "tool", "tool_call_id": "call_1", "content": "first"},
            {"role": "assistant", "content": "done"},
        ],
        None,
    )
    for name, arguments in (
        ("json_scalar_arguments", json.dumps({"a": literal, "b": True, "c": None, "d": -42})),
        ("raw_invalid_json", '{"unterminated": "中文 <|end_of_msg|>'),
    ):
        yield (
            name,
            [
                {"role": "user", "content": "q"},
                {
                    "role": "assistant",
                    "content": "",
                    "tool_calls": [{"function": {"name": "lookup", "arguments": arguments}}],
                },
            ],
            None,
        )
    yield (
        "reasoning_alias_precedence",
        [
            {"role": "user", "content": "q"},
            {
                "role": "assistant",
                "content": "a",
                "reasoning": "alias",
                "reasoning_content": "primary",
            },
        ],
        None,
    )


@pytest.fixture(scope="module")
def checkpoint():
    path = os.environ.get("K3_TOKENIZER_PATH")
    if not path:
        pytest.skip("Set K3_TOKENIZER_PATH to run the real-checkpoint qualification")
    verify_assets(path)
    return path, AutoTokenizer.from_pretrained(path, trust_remote_code=True)


MODES = [None, {"thinking": False, "thinking_effort": None}] + [
    {"thinking": True, "thinking_effort": effort} for effort in ("low", "high", "max")
]


@pytest.mark.parametrize("mode", MODES)
@pytest.mark.parametrize(
    "name,messages,tools", list(edge_cases()), ids=lambda x: x if isinstance(x, str) else None
)
def test_segment_ownership_at_every_cut(checkpoint, name, messages, tools, mode):
    path, tokenizer = checkpoint
    renderers = [K3Renderer(tokenizer), SmgK3Renderer(path)]
    if (
        mode
        and not mode["thinking"]
        and any(
            m.get("role") == "assistant" and any(m.get(k) for k in K3Renderer._REASONING_FIELDS)
            for m in messages
        )
    ):
        for renderer in renderers:
            with pytest.raises(ValueError, match="discard"):
                renderer.render(messages, tools, max_seq_length=100000, generation_config=mode)
        return
    oracle = checkpoint_mask_oracle(tokenizer, messages, tools, mode)
    for last in (False, True):
        for cut in range(1, len(oracle.input_ids) + 2):
            expected = (oracle.input_ids[:cut], oracle.mask(last_turn_only=last, limit=cut))
            for renderer in renderers:
                assert (
                    renderer.render(
                        messages,
                        tools,
                        max_seq_length=cut,
                        last_turn_only=last,
                        generation_config=mode,
                    )
                    == expected
                ), (name, mode, last, cut, type(renderer).__name__)


@pytest.mark.parametrize("text", ["a" * 25001, " " * 25001 + "end", "中文👩‍💻" * 6000])
def test_long_segment_boundaries(checkpoint, text):
    path, tokenizer = checkpoint
    messages = [{"role": "user", "content": text}, {"role": "assistant", "content": text}]
    oracle = checkpoint_mask_oracle(tokenizer, messages)
    for renderer in (K3Renderer(tokenizer), SmgK3Renderer(path)):
        for cut in (1, 511, 512, 2048, 25000, 25001, 100000):
            assert renderer.render(messages, max_seq_length=cut) == (
                oracle.input_ids[:cut],
                oracle.mask(limit=cut),
            )


@pytest.mark.parametrize("mutation", ["header", "eom", "shift", "drop_body"])
def test_oracle_detects_shared_mask_bugs(checkpoint, monkeypatch, mutation):
    path, tokenizer = checkpoint
    messages = [{"role": "user", "content": "q"}, {"role": "assistant", "content": "a"}]
    oracle = checkpoint_mask_oracle(tokenizer, messages)
    original = K3Renderer.compute_loss_mask

    def broken(self, ids, **kwargs):
        mask = original(self, ids, **kwargs)
        if mutation == "header":
            mask[next(i for i, v in enumerate(mask) if v) - 1] = 1
        elif mutation == "eom":
            mask[-1] = 1
        elif mutation == "shift":
            mask = mask[1:] + [0]
        else:
            mask = [0] * len(mask)
        return mask

    monkeypatch.setattr(K3Renderer, "compute_loss_mask", broken)
    outputs = [
        r.render(messages, max_seq_length=10000)
        for r in (K3Renderer(tokenizer), SmgK3Renderer(path))
    ]
    assert outputs[0] == outputs[1]  # backend parity alone still passes
    for result in outputs:
        assert result != (oracle.input_ids, oracle.mask())


@pytest.mark.parametrize("value", [[1, None, True], {"x": "中"}, [], {}, 1e-7, 1.0])
@pytest.mark.parametrize("as_json", [False, True])
def test_non_equivalent_tool_arguments_require_explicit_fallback(checkpoint, value, as_json):
    path, tokenizer = checkpoint
    arguments = {"value": value}
    if as_json:
        arguments = json.dumps(arguments)
    messages = [
        {"role": "user", "content": "q"},
        {
            "role": "assistant",
            "content": None,
            "tool_calls": [
                {
                    "id": "call",
                    "type": "function",
                    "function": {"name": "lookup", "arguments": arguments},
                }
            ],
        },
    ]
    oracle = checkpoint_mask_oracle(tokenizer, messages)
    assert K3Renderer(tokenizer).render(messages, max_seq_length=100000) == (
        oracle.input_ids,
        oracle.mask(),
    )
    with pytest.raises(ValueError, match="not checkpoint-equivalent.*huggingface"):
        SmgK3Renderer(path).render(messages, max_seq_length=100000)


def test_compound_tool_dataset_requires_hf_fallback(checkpoint, tmp_path):
    from torchspec.data.dataset import load_conversation_dataset

    path, tokenizer = checkpoint
    messages = [
        {"role": "user", "content": "q"},
        {
            "role": "assistant",
            "content": "",
            "tool_calls": [
                {
                    "function": {
                        "name": "lookup",
                        "arguments": {"array": [1, None, True], "object": {"x": "中"}},
                    }
                }
            ],
        },
    ]
    oracle = checkpoint_mask_oracle(tokenizer, messages)
    native = SmgK3Renderer(path).tokenizer.native.render(messages)
    assert native["input_ids"] != oracle.input_ids  # retain the upstream reproducer
    raw = tmp_path / "compound.jsonl"
    raw.write_text(json.dumps({"messages": messages}) + "\n")
    args = SimpleNamespace(
        renderer="kimi-k3",
        rendering_backend="auto",
        target_model_path=path,
        train_data_path=str(raw),
        prompt_key="messages",
        max_seq_length=512,
        num_proc=1,
        cache_dir=str(tmp_path / "cache"),
    )
    with pytest.raises(ValueError, match="not checkpoint-equivalent.*huggingface"):
        load_conversation_dataset(args)
    args.rendering_backend = "huggingface"
    rows = load_conversation_dataset(args)
    assert len(rows) == 1
    assert rows[0]["input_ids"].tolist() == oracle.input_ids
    assert resolve_loss_mask(rows[0]).tolist() == oracle.mask()


class TinyDraft(torch.nn.Module):
    """Small differentiable CPU probe; embeddings record unmodified K3 IDs."""

    def __init__(self, vocab):
        super().__init__()
        self.vocab_size = self.target_vocab_size = vocab
        self.weight = torch.nn.Parameter(torch.tensor([[0.2, 0.4], [0.5, -0.3]]))
        self.seen_ids = []

    def get_lm_head_params(self):
        return torch.ones(2), self.weight, 1e-6

    def get_lm_head_up_weight(self):
        return None

    def project_hidden_states(self, hidden_states):
        return hidden_states

    def prepare_decoder_attention_mask(self, **kwargs):
        return kwargs["attention_mask"]

    def embed_input_ids(self, ids):
        self.seen_ids.append(ids.clone())
        return torch.stack((ids.float().remainder(7), ids.float().remainder(11)), dim=-1)

    def backbone(self, *, input_embeds, hidden_states, **kwargs):
        return (input_embeds + hidden_states) @ self.weight, None, None

    def next_depth_hidden(self, hidden_states):
        return hidden_states


@pytest.mark.parametrize("last", [False, True])
@pytest.mark.parametrize("limit", [96, 128, 512])
def test_local_training_path(checkpoint, tmp_path, monkeypatch, last, limit):
    from torchspec.data.dataset import load_conversation_dataset
    from torchspec.models import eagle3

    # Mooncake is not installed on macOS and is unused by _forward. Stub only
    # its transport import; execute the real trainer, model, and loss functions.
    transport = ModuleType("torchspec.transfer.mooncake.eagle_store")
    transport.EagleMooncakeStore = object
    with patch.dict(sys.modules, {transport.__name__: transport}):
        from torchspec.training.eagle3_trainer import Eagle3Trainer

    path, tokenizer = checkpoint
    selected = []
    for name, messages, tools in edge_cases():
        for index, mode in enumerate(MODES):
            if (
                mode
                and not mode["thinking"]
                and any(
                    m.get("role") == "assistant"
                    and any(m.get(k) for k in K3Renderer._REASONING_FIELDS)
                    for m in messages
                )
            ):
                continue  # rejected separately, never silently discarded by the loader
            selected.append((f"{name}-{index}", messages, tools, mode))
    raw = tmp_path / "rows.jsonl"
    raw.write_text(
        "\n".join(
            json.dumps(
                {"id": name, "messages": messages, "tools": tools, "generation_config": mode}
            )
            for name, messages, tools, mode in selected
        )
    )
    expected = {}
    for name, messages, tools, mode in selected:
        oracle = checkpoint_mask_oracle(tokenizer, messages, tools, mode)
        mask = oracle.mask(last_turn_only=last, limit=limit)
        if any(mask):
            expected[name] = (oracle.input_ids[:limit], mask)
    args = SimpleNamespace(
        renderer="kimi-k3",
        rendering_backend="auto",
        target_model_path=path,
        train_data_path=str(raw),
        prompt_key="messages",
        max_seq_length=limit,
        num_proc=1,
        cache_dir=str(tmp_path / "cache"),
        last_turn_loss_only=last,
    )
    rows = load_conversation_dataset(args)
    cached = load_conversation_dataset(args)
    assert {row["data_id"] for row in rows} == set(expected)
    features, oracle_masks = [], []
    for row, hit in zip(rows, cached):
        ids, mask = expected[row["data_id"]]
        assert row["input_ids"].tolist() == hit["input_ids"].tolist() == ids, row["data_id"]
        assert row["packed_loss_mask"] == hit["packed_loss_mask"]
        actual_mask = resolve_loss_mask(row)
        assert actual_mask.tolist() == mask
        positions = torch.arange(len(ids), dtype=torch.float32)
        features.append(
            {
                "input_ids": row["input_ids"].unsqueeze(0),
                "loss_mask": actual_mask.unsqueeze(0),
                "hidden_states": torch.ones(1, len(ids), 2),
                "last_hidden_states": torch.stack((positions, positions + 1), dim=-1).unsqueeze(0),
            }
        )
        oracle_masks.append(mask)
    batch = DataCollatorWithPadding()(features)
    width = batch["input_ids"].shape[1]
    independent_mask = torch.tensor([m + [0] * (width - len(m)) for m in oracle_masks])
    assert torch.equal(batch["loss_mask"], independent_mask)
    assert not batch["loss_mask"][batch["attention_mask"] == 0].any()
    draft = TinyDraft(tokenizer.vocab_size)
    model = eagle3.Eagle3Model(draft, length=3)
    trainer = object.__new__(Eagle3Trainer)
    trainer.verifier_norm = None
    trainer.t2d = None
    trainer.target_lm_head_weight = torch.tensor([[0.01, 0.02], [0.02, -0.01]])
    trainer.eagle3 = trainer.model = model
    seen = []
    eager_loss = eagle3.compiled_forward_kl_loss_from_hs._torchdynamo_orig_callable

    def observe(hs, ths, valid_idx, *rest):
        depth = len(seen)
        expected_mask = torch.zeros_like(independent_mask)
        expected_mask[:, : width - depth] = independent_mask[:, depth:]
        assert torch.equal(valid_idx, expected_mask.flatten().nonzero().flatten())
        # Trainer shifts target states once; each Eagle3 depth slices one more.
        # The mask uses the existing anchor-position convention, not a newly
        # invented next-token shift. Check both independently at the loss call.
        expected_ths = torch.zeros_like(batch["last_hidden_states"])
        expected_ths[:, : width - depth - 1] = batch["last_hidden_states"][:, depth + 1 :]
        assert torch.equal(ths, expected_ths.reshape(-1, 2))
        seen.append(valid_idx.clone())
        return eager_loss(hs, ths, valid_idx, *rest)

    monkeypatch.setattr(eagle3, "compiled_forward_kl_loss_from_hs", observe)
    # Device placement only; the actual production _forward remains intact.
    monkeypatch.setattr(torch.Tensor, "cuda", lambda self, *a, **kw: self)
    losses, _, _, counts, _ = trainer._forward(batch)
    assert len(seen) == 3
    for depth, (ids, count) in enumerate(zip(draft.seen_ids, counts)):
        expected_ids = torch.zeros_like(batch["input_ids"])
        expected_ids[:, : width - depth - 1] = batch["input_ids"][:, depth + 1 :]
        assert torch.equal(ids, expected_ids)
        assert count.item() == len(seen[depth])
    sum(losses).backward()
    assert torch.isfinite(draft.weight.grad).all() and draft.weight.grad.abs().sum() > 0
