# Draft Model Configs

Architecture configurations for speculative draft models. These define the model structure (hidden size, number of layers, attention heads, vocabulary size, etc.) used to initialize the draft model before training.

## Files

| Config | Target model | Notes |
|--------|-------------|-------|
| `qwen3_8b_eagle3.json` | Qwen3-8B | 1-layer LlamaForCausalLMEagle3, hidden_size=4096 |
| `kimi_k25_eagle3.json` | Kimi-K2.5 | 1-layer LlamaForCausalLMEagle3, hidden_size=7168 |
| `kimi_k3_dflash2_mla_swa.json` | Kimi-K3 | 6-layer MLA DFlash2, SWA×5 + Full, block size 8 |
| `kimi_k3_eagle3_mla.json` | Kimi-K3 | 1-layer MLA Eagle3, hidden_size=7168 |

## Usage

Reference a draft model config via the `model.draft_model_config` field in your training config or as a CLI override:

```bash
python -m torchspec.train_entry --config configs/sglang_qwen3_8b.yaml \
    model.draft_model_config=configs/draft_models/qwen3_8b_eagle3.json
```
