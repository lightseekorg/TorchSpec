# SpecForge DFlash Training Reference

Reference from [SpecForge PR #415](https://github.com/sgl-project/SpecForge/pull/415), [PR #427](https://github.com/sgl-project/SpecForge/pull/427), [PR #472](https://github.com/sgl-project/SpecForge/pull/472), [PR #473](https://github.com/sgl-project/SpecForge/pull/473), [Issue #465](https://github.com/sgl-project/SpecForge/issues/465), and [DFlash paper (arXiv:2602.06036)](https://arxiv.org/abs/2602.06036).

## Architecture

- **Method**: DFlash (Block-wise Parallel Decoding) — dLLM-style speculative decoding
- **Target Model**: Qwen3-8B (36 layers, 4096 hidden_size)
- **Draft Model**: DFlashDraftModel
  - 5 hidden layers
  - block_size = 16
  - 32 attention heads, 8 KV heads (GQA)
  - hidden_size = 4096, intermediate_size = 12288
  - Hybrid Attention: Query from noise stream, K/V from [context + noise]
- **Training Mode**: Online (real-time target model inference via SGLang or HF backend)

## Training Configuration

### SpecForge Default (PR #415 example script)

| Parameter          | Value              |
|--------------------|--------------------|
| Epochs             | 20                 |
| Batch size         | 4                  |
| Learning rate      | 1e-4               |
| Max sequence length| 2048               |
| Block size         | 16                 |
| GPUs               | 1 (default, configurable) |
| Precision          | bfloat16           |
| Warmup ratio       | 0.01               |
| Max grad norm      | 1.0                |
| Optimizer          | BF16Optimizer      |
| Dataset            | ShareGPT (sharegpt_train.jsonl) |
| FSDP               | Yes (FullyShardedDataParallel) |
| Training time      | **Not reported**   |

### z-lab Official (from DFlash paper, arXiv:2602.06036)

| Parameter            | Value                                             |
|----------------------|---------------------------------------------------|
| Epochs               | 6                                                 |
| Optimizer            | AdamW, lr = 6e-4                                  |
| LR schedule          | Cosine with 0.04 warmup ratio                     |
| Gradient clipping    | 1.0                                               |
| Max sequence length  | 3,072 (4,096 for Qwen3-Coder)                     |
| Anchors per sequence | 512 randomly sampled                               |
| Dataset              | ~800K samples (NVIDIA Nemotron Post-Training V2 + CodeAlpaca) |
| Ablation subset      | 100K samples                                       |
| Draft layers         | 5 (8 for Qwen3-Coder)                              |
| Block size           | 16 (10 for LLaMA 3.1)                              |
| Batch size           | **Not disclosed**                                  |
| GPU count/type       | **Not disclosed**                                  |
| Training time        | **Not disclosed**                                  |

## Training Results

- Loss curve shows convergence over training (posted by author in PR #415).
- **No training time reported** in any source — not in SpecForge PRs, z-lab blog, or the DFlash paper.

### Community Results (SpecForge Issue #465)

@jianc99 (likely z-lab affiliated) reported:
> Trained a 6-layer DFlash draft model for gpt-oss-20B on 1.2M PerfectBlend samples (first turn only).
> After **3 epochs**, acceptance length on Math500 reached **6.3**.
> This demonstrates DFlash's strong data efficiency.

@xiaomin-D (SpecForge contributor) shared a demo model:
- **Dataset**: [eigen-ai-labs/perfectblend-qwen3-8b-regen-demo](https://huggingface.co/eigen-ai-labs/perfectblend-qwen3-8b-regen-demo) (0.175M samples, ~15% of full PerfectBlend)
- **Model**: [eigen-ai-labs/qwen3-8b-dflash-demo](https://huggingface.co/eigen-ai-labs/qwen3-8b-dflash-demo)
- **Note**: Dataset must be regenerated with the target model to avoid distribution mismatch.

### Acceptance Length (Inference Evaluation)

Results reported by community member (@wengsnow) in PR #427:

| Training Data             | Acceptance Length |
|---------------------------|-------------------|
| sharegpt_train.jsonl only | 1.02 (poor)       |
| sharegpt + ultrachat      | 2.06              |
| Official z-lab model      | ~3.5+             |

## Official z-lab Model Performance

Source: [z-lab/Qwen3-8B-DFlash-b16](https://huggingface.co/z-lab/Qwen3-8B-DFlash-b16) (1B params, bfloat16) | [Blog](https://z-lab.ai/projects/dflash/) | [Paper](https://arxiv.org/abs/2602.06036)

### Speedup vs EAGLE-3 (Qwen3-8B, temp=0)

| Benchmark     | DFlash Speedup | EAGLE-3 Speedup |
|---------------|----------------|-----------------|
| GSM8K         | 5.20x          | 2.13x           |
| MATH-500      | 6.17x          | 2.18x           |
| AIME24        | 5.91x          | 2.25x           |
| AIME25        | 5.85x          | 2.18x           |
| HumanEval     | 5.20x          | 2.48x           |
| MBPP          | 4.75x          | 2.27x           |
| LiveCodeBench | 5.43x          | 2.24x           |
| SWE-Bench     | 2.92x          | 1.90x           |
| MT-Bench      | 2.79x          | 1.94x           |
| Alpaca        | 2.27x          | 1.88x           |

### Key Performance Highlights

- **Peak speedup**: Up to **6.17x** lossless acceleration (MATH-500)
- **Average**: ~2.5x faster than EAGLE-3
- **Acceptance length** (naive 5-layer model, no target conditioning): 3.38–4.61 tokens (greedy), 3.23–4.12 tokens (sampling, temp=1)
- **Sampling mode** (temp=1): ~4.5x acceleration for reasoning tasks
- **Reasoning/code tasks** benefit most (5-6x), **open-ended generation** (MT-Bench, Alpaca) shows 2-3x

## Known Issues

### 1. Causal Masking Bug (PR #415 → Fixed in PR #427)

- **Problem**: Training code enforced causal attention within draft blocks, but inference uses bidirectional attention. This train/inference mismatch degraded acceptance length.
- **Fix**: PR #427 removed the causal constraint from the noise mask, aligning training with inference behavior.
- **Impact**: Models trained with PR #415 code (before fix) will have poor acceptance length. Must use code after PR #427.

### 2. SGLang Backend Padding Bug + Attention Mask Data Leak (Fixed in PR #472, #473)

- **Problem 1**: SGLang backend had broken padding (shift) logic that prevented training from converging. Went unnoticed because the default script uses the HF backend.
- **Problem 2**: A data leak in the random-anchor attention mask made training loss/accuracy look deceptively good — the model was cheating during training but failed at inference.
- **Symptom**: Training accuracy reaches ~1.0 but acceptance length stays at ~1.02.
- **Fix**: PR #472 and #473 corrected both issues.
- **Impact**: Any model trained before these fixes (especially with SGLang backend) may have poor inference performance despite good training metrics.
- **Credit**: Bugs caught by @TechxGenus and @hukongyi.

### 3. Data Regeneration Requirement

- Training uses teacher forcing on ground-truth tokens, while inference is autoregressive and model-dependent.
- **Dataset must be regenerated with the target model** to avoid distribution mismatch that hurts acceptance length.
- Even if training accuracy reaches 1.0, it doesn't guarantee the same sequence will be reproduced during inference.

## Data Flow

```
Raw Text → Tokenizer → Input IDs
                          ↓
         Target Model (SGLang/HF Backend)
                          ↓
              Context Hidden States
                          ↓
┌─────────────────────────┴─────────────────────────┐
↓                                                   ↓
Context Stream (H_ctx)                    Noise Stream (E_noise)
[Target Hidden States]                    [Block-start: real, others: MASK]
        ↓                                           ↓
        └──────────────── Hybrid Attention ─────────┘
                     Q: Noise | K/V: [Context, Noise]
                          ↓
                  Draft Hidden States
                          ↓
                Target LM Head (Frozen)
                          ↓
                Draft Logits → CE Loss ← Labels
```

## Attention Mask Design

- Tensor layout: Query `[B, L, D]`, Key/Value `[B, 2L, D]` (Context L + Noise L)
- Attention mask: `[L, 2L]`
- Position IDs: `[0..L-1, 0..L-1]` for RoPE alignment

For Query at position `i` in Block `B = i // block_size`:
- **Context region** (cols 0..L-1): See all positions where `block_id < B`
- **Noise region** (cols L..2L-1): See all same-block positions (bidirectional, after PR #427 fix)

## Loss Calculation

- **Excluded**: Block 0 (no preceding context) + block anchor tokens (known inputs)
- **Included**: All other positions predict their corresponding token IDs
- **Method**: Masked Cross-Entropy Loss

## Key Files (SpecForge repo)

```
configs/qwen3-8b-dflash.json              # Draft model config
examples/run_qwen3_8b_dflash_online.sh    # Training launch script
scripts/train_dflash.py                   # Training entry point
specforge/core/dflash.py                  # OnlineDFlashModel wrapper
specforge/modeling/draft/dflash.py        # DFlashDraftModel architecture
specforge/modeling/target/dflash_target_model.py  # SGLang/HF target backend
specforge/modeling/target/target_utils.py  # Weight loader utilities
```

## Eagle3 Training (SpecForge)

No training time information is available for Eagle3 in SpecForge either. PR #398 (Ling-flash-2.0 Eagle3 support) shows benchmark results on H200 hardware but does not report training duration, GPU count, or total compute.

## References

- DFlash paper: https://arxiv.org/abs/2602.06036
- DFlash upstream: https://github.com/z-lab/dflash
- z-lab blog: https://z-lab.ai/projects/dflash/
- Official model: https://huggingface.co/z-lab/Qwen3-8B-DFlash-b16
- SpecForge PR #415 (DFlash implementation): https://github.com/sgl-project/SpecForge/pull/415
- SpecForge PR #427 (causal mask fix): https://github.com/sgl-project/SpecForge/pull/427
- SpecForge PR #472, #473 (SGLang padding + data leak fix): https://github.com/sgl-project/SpecForge/pull/472
- SpecForge Issue #465 (training guide + community results): https://github.com/sgl-project/SpecForge/issues/465
- SpecForge PR #398 (Eagle3 Ling-flash-2.0): https://github.com/sgl-project/SpecForge/pull/398
- Demo dataset: https://huggingface.co/eigen-ai-labs/perfectblend-qwen3-8b-regen-demo
- Demo model: https://huggingface.co/eigen-ai-labs/qwen3-8b-dflash-demo
