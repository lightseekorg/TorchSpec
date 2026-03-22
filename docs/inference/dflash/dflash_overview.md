# DFlash Implementation Overview

**Branch**: `feature/dflash-training`
**Start Date**: 2026-03-19

## Architecture: DFlash vs Eagle3

| | DFlash | Eagle3 |
|---|--------|--------|
| Loss | CE (cross-entropy) | KL (forward KL) |
| Prediction | Block-parallel (block_size=16) | Autoregressive (ttt_length=7) |
| Target layers | 5 layers | 3 layers |
| Mask | Block-causal (FlexAttention) | Causal |
| Context | Dual-source KV (W_proj) | Input fusion (fc layer) |
| Forward passes per step | 1 | 7 (sequential) |

## Files Created

1. **`torchspec/models/draft/dflash.py`** — DFlash draft model
   - `DFlashConfig(PretrainedConfig)` — DFlash-specific configuration
   - `DFlashRMSNorm`, `DFlashRotaryEmbedding` — reused from Llama patterns
   - `DFlashAttention` — Dual-source KV attention: takes `context_hidden` (from target) and `draft_hidden` separately, projects both through shared W_k/W_v, concatenates K/V. Q from draft only. Supports GQA.
   - `DFlashDecoderLayer` — Standard pre-norm transformer layer
   - `DFlashDraftModel(PreTrainedModel)` — Full draft model with `W_proj: Linear(num_target_layers * target_hidden_size, hidden_size)`, shared embedding + LM head from target (frozen)

2. **`torchspec/models/dflash.py`** — DFlash training wrapper
   - `DFlashModel(nn.Module)` — `_sample_anchor_positions()`, `_create_position_ids()`, `_prepare_noise_input()`, `_create_dflash_mask_mod()` (FlexAttention block-causal mask), `forward()` with CE loss + decay weighting

3. **`torchspec/training/dflash_trainer.py`** — DFlash trainer
   - `DFlashTrainer(Trainer)` — `init_model()` (FSDP2, optimizer, frozen LM head), `_train_step()`, `_aggregate_metrics()`, `eval_forward()` / `eval_from_cache()`

4. **`torchspec/config/dflash_draft_config.json`** — Draft model config for Qwen3-8B
5. **`tests/test_dflash.py`** — 67 tests (unit + integration + quality)
6. **`configs/sglang_qwen3_8b_dflash.yaml`** — 4-GPU SGLang training config
7. **`configs/hf_qwen3_8b_1gpu.yaml`** / **`hf_qwen3_8b_dflash_1gpu.yaml`** — Single-GPU colocate configs
8. **`scripts/runpod_dflash_train.sh`** — RunPod training script (auto-detects GPU count)
9. **`scripts/benchmark_dflash_inference.py`** — Inference benchmark (baseline vs speculative)
10. **`scripts/extract_dflash_checkpoint.py`** — FSDP checkpoint extraction

## Files Modified

1. **`torchspec/config/train_config.py`** — Added `dflash_block_size`, `dflash_num_anchors`, `dflash_loss_decay_gamma`, `dflash_num_target_layers`
2. **`torchspec/models/target/eagle3_target_model.py`** — Generalized `set_aux_hidden_states_layers()` for N layers (removed `assert len == 3`)
3. **`torchspec/utils/misc.py`** — Added `get_default_dflash_aux_layer_ids()`
4. **`torchspec/training/trainer_actor.py`** — Config-based trainer dispatch (`DFlashConfig` → `DFlashTrainer`)
5. **`torchspec/train_entry.py`** — Auto-set aux layer IDs when `DFlashConfig` detected
6. **`torchspec/inference/engine/__init__.py`** / **`inference/factory.py`** — Lazy SGLang/vLLM imports (HF-only no longer requires these)
7. **`torchspec/models/ops/flex_attention.py`** — Inductor config (`max_autotune_gemm_backends = "ATEN,TRITON"`)
8. **`torchspec/models/draft/auto.py`**, **`models/draft/__init__.py`**, **`models/__init__.py`**, **`__init__.py`** — DFlash registration + exports

## Key Design Decisions

1. **Not inheriting Eagle3DraftModel**: DFlash has fundamentally different interfaces (dual-source KV vs input fusion, block-parallel vs autoregressive). Clean separation avoids coupling.
2. **FlexAttention for block-causal mask**: Reuses TorchSpec's existing `compile_friendly_flex_attention` and `compile_friendly_create_block_mask` singletons.
3. **CE loss instead of KL**: DFlash uses cross-entropy against ground truth tokens (not KL divergence against target distribution like Eagle3).
4. **Config-based trainer dispatch**: `TrainerActor.init()` resolves the draft config before creating the trainer, using `isinstance(config, DFlashConfig)` to choose.
5. **No separate DFlash target model**: Reuses `Eagle3TargetModel` + `HFTargetModel` with generalized N-layer support. Same hook-based mechanism works for both 3 and 5 layers.
6. **Backward-compatible target model**: `set_aux_hidden_states_layers()` defaults to 3 Eagle3 layers when called with `None`.

## Lessons from SpecForge PR #473

1. `tie_word_embeddings` correctly handled — load from target, freeze
2. Block-causal mask is bidirectional within block (not causal)
3. Loss decay starts at k=1 (first prediction), not k=0 (anchor)
4. Same-position prediction alignment between training and inference
