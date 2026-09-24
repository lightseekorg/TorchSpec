# Megatron-Core MTP training

TorchSpec can train a model's own MTP layer with Megatron-Core's native
`MultiTokenPredictionLayer` (including native MoE and expert-parallel token
dispatch). Megatron owns the layer; TorchSpec, or any PyTorch loop, owns the
batches, loss, optimizer and checkpoints.

- `torchspec.megatron` creates the Megatron process groups (TP1/PP1, any EP)
  and reduces/clips gradients with the right ownership for replicated versus
  expert-sharded parameters.
- `torchspec.models.mtp.MegatronMTPModel` wraps the layer with the target's
  frozen embedding and LM head behind a batch-first API.
- `torchspec.models.qwen3_5_mtp` builds the Qwen3.5 MoE MTP layer and imports
  it from a Hugging Face checkpoint.
- `torchspec.training.mtp.compute_mtp_objective` computes hard-label cross
  entropy with PyTorch.

## Usage

Megatron-Core 0.18 requires Python 3.12: `pip install -e '.[megatron]'`.

```python
import torch

from torchspec.megatron import clip_grad_norm_, initialize_model_parallel, synchronize_gradients
from torchspec.models.qwen3_5_mtp import Qwen35MTPConfig, build_qwen35_mtp_model, load_qwen35_mtp_weights
from torchspec.models.target_hidden import capture_final_norm_boundary
from torchspec.training.mtp import compute_mtp_objective

groups = initialize_model_parallel(expert_parallel_size=4)  # after torch.distributed init
config = Qwen35MTPConfig.from_json_file(f"{checkpoint_dir}/config.json")
model = build_qwen35_mtp_model(config, groups)  # fp32 parameters
report = load_qwen35_mtp_weights(model, config, checkpoint_dir)
optimizer = torch.optim.AdamW(p for p in model.parameters() if p.requires_grad)

# target_decoder: the frozen HF target's decoder, e.g. model.model.language_model
hidden = capture_final_norm_boundary(target_decoder, input_ids=input_ids).post_norm
output = model(input_ids=input_ids, last_hidden_states=hidden.float())
objective = compute_mtp_objective(model, output, input_ids=input_ids, loss_mask=loss_mask)
objective.loss.backward()
synchronize_gradients(model, groups)
clip_grad_norm_(model, 1.0, groups)
optimizer.step()
```

`last_hidden_states` is the output of the target's final RMSNorm, the same
state Megatron's GPT+MTP path and vLLM's Qwen3.5 MTP consume.
`capture_final_norm_boundary` hooks both sides of the final norm and checks the
boundary on every call.

MTP output row `t` combines `h_t` with the embedding of `x[t+1]` and is scored
against `x[t+2]`, the next token after the MTP layer's own input token. Only
targets with `loss_mask` set count. When accumulating gradients over
micro-batches, pass `normalizer=<valid targets in the global batch>` so the
step loss is the global token mean. Rows are unpadded; use one sequence per
micro-batch or equal-length rows.

Every rank in an EP group must receive the same batch; shard data by
expert-data-parallel rank. Because each routed expert then sees one copy of
every token per EP rank, `synchronize_gradients` follows Megatron DDP: it sums
replicated gradients over dense DP (which spans EP) and expert gradients over
expert DP, and divides both by the dense DP size. The router has no
auxiliary load-balancing loss, so the objective is cross entropy only.

`load_qwen35_mtp_weights` NaN-fills every parameter before loading, converts
Qwen's zero-centered RMSNorm weights to `1 + w`, and reorders Qwen's per-head
`[query, gate]` rows into Megatron's per-group `[queries, gates, key, value]`
QKV layout. It returns a per-tensor mapping report and raises on any unloaded
parameter or unconsumed checkpoint key.
