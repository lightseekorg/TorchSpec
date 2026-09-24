# Copyright (c) 2026 LightSeek Foundation

import torch
import torch.nn.functional as F

from torchspec.models.qwen3_5_mtp import Qwen35MTPConfig, pack_qwen35_qkv
from torchspec.training.mtp import compute_mtp_objective


class _IdentityHead:
    @staticmethod
    def compute_logits(hidden_states):
        return hidden_states


def test_pack_qwen35_qkv_groups_queries_gates_key_value():
    # HF q_proj rows are [q0, gate0, q1, gate1, ...]; two query heads per KV group.
    q = torch.arange(8, dtype=torch.float32).reshape(-1, 1)
    k = torch.tensor([[100.0], [101.0]])
    v = torch.tensor([[200.0], [201.0]])

    packed = pack_qwen35_qkv(q, k, v, num_attention_heads=4, num_key_value_heads=2, head_dim=1)

    assert packed.flatten().tolist() == [0, 2, 1, 3, 100, 200, 4, 6, 5, 7, 101, 201]


def test_qwen35_mtp_config_reads_nested_text_config():
    raw = {
        "text_config": {
            "hidden_size": 16,
            "vocab_size": 32,
            "num_attention_heads": 4,
            "num_key_value_heads": 2,
            "head_dim": 4,
            "num_experts": 8,
            "num_experts_per_tok": 2,
            "moe_intermediate_size": 8,
            "shared_expert_intermediate_size": 8,
            "rms_norm_eps": 1e-6,
            "attn_output_gate": True,
            "rope_parameters": {"rope_theta": 1_000_000, "partial_rotary_factor": 0.25},
        }
    }

    config = Qwen35MTPConfig.from_dict(raw)

    assert config.attention_output_gate is True
    assert (config.rope_theta, config.partial_rotary_factor) == (1_000_000, 0.25)


def test_mtp_objective_scores_row_t_against_token_t_plus_2():
    input_ids = torch.tensor([[0, 1, 2, 3, 4]])
    loss_mask = torch.tensor([[1, 1, 1, 0, 1]])
    hidden = torch.randn(1, 5, 5, requires_grad=True)

    objective = compute_mtp_objective(
        _IdentityHead(), hidden, input_ids=input_ids, loss_mask=loss_mask
    )

    # Targets x[2] and x[4]; x[3] is masked and rows 3-4 have no target.
    expected = F.cross_entropy(hidden[0, [0, 2]], torch.tensor([2, 4]))
    torch.testing.assert_close(objective.loss, expected)
    assert objective.num_targets == 2
    objective.loss.backward()
    assert hidden.grad[0, [1, 3, 4]].abs().sum() == 0


def test_mtp_objective_normalizer_accumulates_to_global_token_mean():
    batches = [
        (
            torch.randn(1, 6, 6),
            torch.tensor([[0, 1, 2, 3, 4, 5]]),
            torch.tensor([[0, 0, 1, 1, 0, 1]]),
        ),
        (torch.randn(1, 4, 6), torch.tensor([[5, 4, 3, 2]]), torch.tensor([[0, 1, 1, 1]])),
    ]
    total = sum(int(mask[:, 2:].sum()) for _, _, mask in batches)

    loss = sum(
        compute_mtp_objective(
            _IdentityHead(), hidden, input_ids=ids, loss_mask=mask, normalizer=total
        ).loss
        for hidden, ids, mask in batches
    )

    rows = torch.cat((batches[0][0][0, [0, 1, 3]], batches[1][0][0, [0, 1]]))
    torch.testing.assert_close(loss, F.cross_entropy(rows, torch.tensor([2, 3, 5, 3, 2])))
