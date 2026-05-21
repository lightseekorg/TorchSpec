# Colocate Qwen3-8B Single-Node (MPS + NCCL)

Single-node colocate spec-decoding training: trainer + sglang inference
engine share the **same** physical GPUs via NVIDIA MPS, with hidden
states crossing the engine→trainer boundary over NCCL P2P (no Mooncake).

This is the colocate sibling of
[`examples/qwen3-8b-single-node/`](../qwen3-8b-single-node/). The two
diverge in three places: `colocate_strategy=mps` + `transfer_mode=nccl`
in the config, fractional `train_frac` / `infer_frac` memory budgets,
and `engine_count × tp_size == training_world_size` (so trainer rank
`i` ↔ engine rank `i` on the same GPU).

For background and the full design rationale, see
[`docs/colocate/usage.md`](../../docs/colocate/usage.md).

## Status

⚠️ **The TorchSpec side of this path is complete; an end-to-end
training step also requires an upstream sglang patch** — see
[`docs/colocate/sglang_patch.md`](../../docs/colocate/sglang_patch.md).

Without the patch, init succeeds but the first step hangs on the
trainer's `dist.batch_isend_irecv` (the engine never sends). That hang
is the diagnostic, not a bug.

## Prerequisites

- 1 host with 4 H100 80GB GPUs (smaller GPUs work but you'll need to
  trim `max_seq_length` and the memory fractions).
- NVIDIA driver R535+ with MPS (`nvidia-cuda-mps-control` in `$PATH` —
  ships with the CUDA toolkit).
- HF access to `Qwen/Qwen3-8B`.
- sglang built with the colocate patch (see link above).

## Config

[`configs/colocate_qwen3_8b.yaml`](../../configs/colocate_qwen3_8b.yaml):

- **Strategy:** `colocate_strategy=mps`, `transfer_mode=nccl`.
- **Memory split:** `train_frac=0.45` + `infer_frac=0.45` + `0.10`
  reserved (NCCL workspace + driver + Python).
- **Layout:** 4 trainer ranks (FSDP) + 4 engine ranks (TP=1 each) =
  4 GPUs shared.

## How to run

```bash
./examples/colocate-qwen3-8b-1node/run.sh
```

With a custom config:

```bash
./examples/colocate-qwen3-8b-1node/run.sh configs/colocate_qwen3_8b.yaml
```

Override settings (`train_entry.py`'s flat-args parser):

```bash
./examples/colocate-qwen3-8b-1node/run.sh configs/colocate_qwen3_8b.yaml \
    training.num_train_steps=10 \
    training.train_frac=0.50 \
    training.infer_frac=0.40
```

Pin specific GPUs:

```bash
CUDA_VISIBLE_DEVICES=4,5,6,7 ./examples/colocate-qwen3-8b-1node/run.sh
```

## What to expect

The script:

1. Sets `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` (essential
   under MPS — keeps the long stability run flat).
2. Launches `python -m torchspec.train_entry` with the colocate config
   and the GPU layout pinned to a 1:1 trainer↔engine ratio.
3. The driver:
   - Starts the MPS daemon (idempotent) and propagates
     `CUDA_MPS_PIPE_DIRECTORY` / `CUDA_MPS_LOG_DIRECTORY` into both
     actor groups.
   - Builds a single Ray placement group that both trainer and engine
     actor groups bind to (same bundle ↔ same GPU).
   - Skips Mooncake master and `AsyncInferenceManager`.
4. `TrainerActor.init` runs `init_union_world` on `master_port + 5000`
   so the union NCCL world doesn't collide with FSDP's own port range.
5. Each step: engine forwards on its TP=1 model → P2P-sends the
   hidden-state dict → trainer's `NcclMultiTensorFetcher.recv_step`
   receives it → trainer fwd/bwd. Strictly serialised, no async.

Loss should decrease steadily. Peak GPU memory should plateau by step
~10 and stay flat afterwards (Phase 6 stability gate).

## When to use the disaggregated path instead

See [`docs/colocate/usage.md`](../../docs/colocate/usage.md#when-to-use-colocate-mode)
for the rules. Quick answer: multi-node, multi-replica, async
pipelining, or vLLM ⇒ use
[`examples/qwen3-8b-single-node/`](../qwen3-8b-single-node/) (or one of
the multi-node examples) instead.
