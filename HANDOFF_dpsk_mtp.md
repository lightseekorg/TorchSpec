# Handoff — DeepSeek/Kimi + GLM MoE MTP draft with Expert Parallel

Branch: `dev/dpsk-mtp`

## Goal

Add **MoE multi-token-prediction (MTP) draft training** to TorchSpec for **Kimi-K2.5
(DeepSeek-V3 arch)** and **GLM-4-MoE** targets, with **Expert Parallel (EP) + FSDP**,
using torch-native `torch._grouped_mm` (no DeepEP, no custom Triton). The structure
is ported from XoRL's EP MoE (`xorl-internal/src/xorl/...`).

Before this work the draft path was **dense-only** (no MoE, no EP — FSDP2/DDP only).

## Status

Everything implementable in a GPU-only env is **done and validated on 8×H100**
(`tests/test_moe_draft.py` — 6 tests pass; `ruff` clean). The only piece not run here
is a **real Kimi-K2.5 end-to-end run**, which needs the Mooncake hidden-state stream +
a real target model + data (not available in the dev env). The full EP training loop
*is* validated end-to-end on **synthetic** data, and a runbook for the real run is below.

| Item | Status |
|------|--------|
| Single-GPU MoE block (router + experts + shared expert) | ✅ validated (grouped_mm == naive ref, rel=0) |
| Expert Parallel all-to-all dispatch | ✅ validated (EP == single-GPU, rel=0; grads flow) |
| FSDP-on-experts (`ep_size < world_size`, ep_fsdp>1) | ✅ validated (4-GPU, sharded forward rel=0) |
| EP-aware grad-norm clip + grad sync + MoE aux loss | ✅ validated (clip rel=0; aux balanced<imbalanced) |
| Checkpoint expert gather (local → full `[E,...]`) | ✅ validated |
| End-to-end EP **training step** (real `Eagle3Model`, synthetic data) | ✅ validated (loss ↓, experts update, global grad-norm consistent) |
| Trainer wiring (2D mesh, `config.ep_group`, EP init/sync/clip) | ✅ code-complete, import-clean, component-validated |
| GLM-4-MoE draft (arch + register + config + chat template) | ✅ validated (build + fwd/bwd) |
| Real Kimi-K2.5 end-to-end run | ⏳ needs infra — see runbook |

**Backward compatibility:** everything is gated. `ep_size=1` (default) and configs
without `use_moe` behave **byte-identical** to the previous dense path.

## How it works

- **MoE block** (`torchspec/models/draft/moe.py`): `DeepseekV3MoEBlock` = a
  `DeepseekV3TopkRouter` (sigmoid scoring + group-limited routing + aux-loss-free
  `e_score_correction_bias` — *not* plain softmax-topk) + `MoEExperts` (stacked
  `(E, in, out)` weights, computed with `torch._grouped_mm`, per-expert token groups
  padded to a multiple of 8) + a shared SwiGLU expert. It is a drop-in replacement for
  the draft layer's dense MLP (`forward(hidden) -> hidden`).
- **Expert Parallel** (`torchspec/models/draft/moe_ep.py`): autograd-aware all-to-all
  (`_AllToAll`, backward swaps the in/out split sizes), token permute/dispatch by
  destination rank, regroup by local expert, local `grouped_mm`, combine back. Each
  rank holds `E/ep_size` experts. `ep_size == world_size` is the primary (tested) case
  (ep_fsdp=1, experts unique per rank).
- **FSDP-on-experts** (`ep_size < world_size`): `ep_utils.shard_experts_fsdp` shards the
  replicated experts across the `ep_fsdp` mesh dim (`Shard(1)` on hidden) for memory.
- **Gradient handling** (`torchspec/training/ep_utils.py`): non-expert grads averaged
  over DP (manual, since EP disables the model-wide DDP); expert grads are unique
  (ep=world) or FSDP-reduced (ep_fsdp>1). `clip_grad_norm_ep` computes the correct
  *global* grad norm (expert sum-of-squares all-reduced over the EP / ep_fsdp groups;
  DTensor-safe). Wired into `BF16Optimizer`.
- **Trainer** (`trainer.py` `_setup_device_mesh`, `eagle3_trainer.py` `init_model`):
  builds a 2D `("ep_fsdp","ep")` mesh, attaches `config.ep_group`, materializes the
  model on CUDA, broadcasts non-expert params from rank 0, FSDP-shards experts if
  `ep_fsdp>1`, and grad-syncs manually (`sync_gradients_ep`) before `optimizer.step`.

## How to use

**Local (offline) draft training, DeepSeek/Kimi MoE draft, EP across N GPUs:**
- Draft config: `configs/draft_models/kimi_k25_eagle3_moe.json` (has `use_moe: true`
  + the MoE fields: `n_routed_experts`, `num_experts_per_tok`, `moe_intermediate_size`,
  `n_shared_experts`, `n_group`, `topk_group`, `routed_scaling_factor`, ...).
- Add `ep_size=<N>` to the training args (in `TrainingConfig`). `ep_size == world_size`
  is the simplest, most-tested setup. `ep_size=1` (default) = no EP.
- Enable a draft from scratch to learn the gate: `train_router: true` (default in the
  MoE block); optional load-balancing aux loss via `router_aux_loss_coef > 0`.

**GLM-4-MoE draft:** `configs/draft_models/glm4_moe_eagle3.json` +
`Glm4MoeForCausalLMEagle3` (GQA attention + shared MoE block). Data: `glm4` chat
template in `torchspec/data/template.py` (⚠️ verify the special tokens against the real
GLM tokenizer before training on GLM-format data).

## Tests

`tests/test_moe_draft.py` (run: `pytest tests/test_moe_draft.py -v`):
- `TestMoEDraftSingleGPU` — grouped_mm numerics vs naive reference, config loading,
  full DeepSeek + GLM model assembly + backbone fwd/bwd.
- `TestMoEDraftExpertParallel` (≥2 GPUs) — EP all-to-all parity vs single-GPU, EP grad
  clip vs reference, ckpt gather; **end-to-end EP training step** (real `Eagle3Model`).
- `TestMoEDraftFSDPExperts` (≥4 GPUs) — FSDP-on-experts forward parity (ep=2, ep_fsdp=2).

Env: `torch 2.9.1+cu128` (has `torch._grouped_mm`), project venv at `mtp-project/.venv`.

## Remaining — real Kimi-K2.5 end-to-end runbook

The trainer EP path is code-complete and validated on synthetic data; to run for real
(needs Mooncake + a real target + data):
1. Install the sglang env (`cp pyproject.sglang.toml pyproject.toml`, see `build_conda.sh`).
2. Stand up the Kimi-K2.5 target in the inference engine + Mooncake master; let the
   inference side stream hidden states (existing TorchSpec server/inference flow).
3. Draft config = `configs/draft_models/kimi_k25_eagle3_moe.json`.
4. Launch training with `ep_size=<num training GPUs>` (ep_size == world_size).
5. Numerical regression / red line: compare `ep_size=1` (experts on one GPU) vs
   `ep_size=N` loss step-by-step — they must match within tolerance.
6. GLM: after Track 1 (DeepSeek) is confirmed, switch to `glm4_moe_eagle3.json` + the
   `glm4` data template (verify tokenizer tokens first).

## File map

**New:**
- `torchspec/models/draft/moe.py` — MoE block, router, experts, grouped_mm, aux loss
- `torchspec/models/draft/moe_ep.py` — Expert-Parallel all-to-all dispatch/combine
- `torchspec/models/draft/glm4_moe_eagle.py` — GLM-4-MoE Eagle3 draft
- `torchspec/training/ep_utils.py` — EP grad sync, grad-norm clip, FSDP-on-experts, ckpt gather
- `configs/draft_models/kimi_k25_eagle3_moe.json`, `configs/draft_models/glm4_moe_eagle3.json`
- `tests/test_moe_draft.py`

**Modified (all gated on `use_moe` / `ep_size>1`):**
- `torchspec/models/draft/deepseek_eagle.py`, `torchspec/models/draft/llama3_eagle.py` — `use_moe` switch
- `torchspec/models/draft/auto.py` — register GLM draft
- `torchspec/config/train_config.py` — `ep_size`
- `torchspec/training/trainer.py` — EP device mesh + manual grad-sync hook
- `torchspec/training/optimizer.py` — EP-aware grad clip (`ep_group`, `ep_extra_group`)
- `torchspec/training/eagle3_trainer.py` — EP init + aux loss + optimizer wiring
- `torchspec/data/template.py` — `glm4` chat template

## Design notes / gotchas

- **`torch._grouped_mm(x, w, offs)`**: `w` is `(E, in, out)` (gate_up fused `[E,H,2I]`,
  chunked after the GEMM); `offs = cumsum(per_expert_counts, int32)`; bf16 needs each
  group's row count to be a multiple of 8 (we pad).
- **DeepSeek-V3 routing ≠ Qwen-MoE**: sigmoid + group routing + correction bias; the
  bias affects *selection* only, the routing *weights* use the un-biased sigmoid scores.
- **`_is_ep` tag** marks expert params; it is dropped when FSDP converts a param to a
  DTensor, so the optimizer also identifies experts by `MoEExperts` module membership.
- **ep_size == world_size** is the primary path; `ep_fsdp>1` is the memory-sharding
  generalization (validated, but the trainer end-to-end was exercised with ep=world).
