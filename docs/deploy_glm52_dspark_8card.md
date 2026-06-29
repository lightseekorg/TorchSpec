# 部署方案:GLM-5.2 DSpark draft 训练(8×B200 单节点 / FP8 / colocate)

> 目标:在**一台 8×B200(sm100)** 机器上,用 **FP8** 的 GLM-5.2 作为 target,训练一个 **DSpark** speculative-decoding draft,加速 GLM-5.2 推理。
> 关键手法:`colocate` —— 推理(SGLang TP=8)与训练(FSDP)**共用同一组 8 卡**,而不是各占一组。

## 0. 拓扑与原理

| 项 | 值 | 依据 |
|---|---|---|
| 节点 | 1 × (8×B200, 192GB/卡) | — |
| target 推理 | SGLang, FP8, **TP=8**(占满 8 卡) | 744GB/8 = 93GB/卡 |
| draft 训练 | FSDP FULL_SHARD, **同 8 卡** | DSpark draft ~4B,分片后每卡 ~6GB |
| 共卡机制 | `training.colocate=true` → GPU 数 = `max(8,8)=8` | `placement_group.py:322-331` |
| 显存预算 | 推理 ~96GB + 训练 ~96GB < 192GB | `mem_fraction_static=0.5` |
| 为什么必须 B200 | H100/H200(80/141GB)装不下「FP8 权重 + 同卡训练」 | — |

config 文件:**`configs/sglang_glm52_dspark_8card_colocate.yaml`**(已就位)。

---

## 1. 环境搭建

```bash
cd <TorchSpec repo>            # 本项目分支 proj/glm52-torchspec-training

# 一键建环境 + 构建 patched sglang(v0.5.14 @ commit 49e384ce)+ 装 torchspec
./tools/build_conda.sh                 # 新建 torchspec env,装 sglang 后端
#   或装进当前环境: ./tools/build_conda.sh current sglang
#   可选 Flash-Attention: pip install -e ".[fa]"

# 激活
micromamba activate torchspec          # 或 conda activate torchspec
```

patched sglang 校验(可选,确认 patch 应用成功):
```bash
python tools/test_sglang_engine_patch.py        # 仓库自带的 patch 自检
```

> patched sglang 的 base commit 锁在 `docker/sglang/v0.5.14/SGLANG_COMMIT`;手动应用 patch 用 `tools/apply_sglang_patch.sh <sglang-checkout>`。也可直接用 `docker/sglang/v0.5.14/Dockerfile` 起容器。

---

## 2. 准备:权重 + 核验门槛值 + 数据

### 2.1 GLM-5.2 FP8 权重
```bash
# HF 下载(或用你已有的镜像/路径)
huggingface-cli download zai-org/GLM-5.2-FP8 --local-dir /models/GLM-5.2-FP8
```
把 config 里的 `model.target_model_path` 指到这个路径。

### 2.2 核验三个 target-相关值(否则静默出错)
```bash
python - <<'PY'
from transformers import AutoTokenizer
import torch, glob, safetensors.torch as st
tok = AutoTokenizer.from_pretrained("/models/GLM-5.2-FP8", trust_remote_code=True)
print("mask_token_id [MASK] =", tok.convert_tokens_to_ids("[MASK]"))   # 填进 draft config
print("chat_template head  =", (tok.chat_template or "")[:200])         # 对照 glm template
# 抽一个分片看 state_dict key 命名是否 = model.embed_tokens.weight / lm_head.weight / model.norm.weight
f = sorted(glob.glob("/models/GLM-5.2-FP8/*.safetensors"))[0]
ks = list(st.load_file(f).keys())
print("keys sample:", [k for k in ks if any(s in k for s in ("embed_tokens","lm_head","model.norm"))][:5])
PY
```
- `mask_token_id`:把 `torchspec/config/glm52_dspark_draft_config.json` 里的占位 **154820** 换成真值。
- `chat_template`:核对 `torchspec/data/template.py` 的 `glm` template(header / end token / 是否 thinking)。
- `embedding_key/lm_head_key/norm_key`:若与默认(`model.embed_tokens.weight` / `lm_head.weight` / `model.norm.weight`)不同,在 config 的 `model:` 段覆盖。

### 2.3 训练数据
- 格式参考 `examples/data/sample_conversations.jsonl`(`prompt_key: conversations`)。
- 自有数据用 `tools/generate_data.py` 生成 / 转换。
- 把 `dataset.train_data_path` 指过去。约束:`min_loss_tokens ≥ 2×dflash_block_size`(7→≥14,config 里已是 32 ✓)。

---

## 3. Step A — 冒烟(门槛,务必先过)

确认 sglang 能在 B200 上 serve `GlmMoeDsaForCausalLM`(FP8/TP=8)且 capture 通路初始化(GLM 继承 `DeepseekV2ForCausalLM` 的 EAGLE3 capture,理论上零改动)。

```bash
# 用 colocate config 但只起推理:临时把 debug.debug_inference_only 设为 true 跑一小步
RAY_ADDRESS=local CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
  python -m torchspec.train_entry --config configs/sglang_glm52_dspark_8card_colocate.yaml \
  --opts debug.debug_inference_only=true
```
**看日志确认**(`sgl_engine.py` 启动时会打):
- `SglEngine ... initialized ... (tp_size=8, aux_layers=[...], hidden_size=6144)` → capture 层解析成功。
- **没有 DSA kernel 崩溃**(sm100 应满足 gating;若崩 → 你的 sglang 太旧或 B200 驱动问题)。
> `--opts key=value` 覆盖单个字段(见 train_entry 的 `_validate_*` 用法约定);若你的版本不支持 `--opts`,直接复制一份 config 改 `debug_inference_only: true`。

---

## 4. Step B — 小样本 dry-run

确认「推理→Mooncake→draft 前向→loss 下降」全链路通 + 显存不 OOM。临时改一份 config:
- `dataset.train_data_path` 指向几十条样本
- `training.max_seq_length: 1024`、`training.num_epochs: 1`
- 其余保持 colocate / FP8 / 8 卡

```bash
RAY_ADDRESS=local CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
  ./examples/qwen3-8b-single-node/run.sh configs/sglang_glm52_dspark_8card_colocate.yaml
```
盯:每卡显存(`nvidia-smi`)、loss 是否下降、有无 NCCL 报错。**OOM 就降 `mem_fraction_static`(0.5→0.45)或 `max_seq_length`。**

---

## 5. Step C — 正式训练

```bash
RAY_ADDRESS=local CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
  ./examples/qwen3-8b-single-node/run.sh configs/sglang_glm52_dspark_8card_colocate.yaml
```
(`run.sh` 会 `ray start --head --num-gpus 8` → `train_entry` → 退出时 `ray stop`。)

监控:`docs/performance_metrics.md` 的指标;checkpoint 落 `output_dir`(config 里 `./outputs/glm52-dspark-8card`),按 `save_interval` / `save_per_epoch`。

---

## 6. Step D — 评估(接受率 vs 自带 MTP)

```bash
# 评估入口:torchspec/controller/eval.py;评估数据 examples/data/eval_conversations.jsonl
python -m torchspec.controller.eval --help          # 先看实际参数
# 也可用 tools/benchmark_eagle3.py 量 draft 接受率 / 加速比
python tools/benchmark_eagle3.py --help
```
目标:DSpark draft 的接受率 / 端到端加速 **优于 GLM-5.2 自带的 MTP(5 token)**,否则没必要替换。

---

## 7. 产物导出(给推理用)

```bash
python tools/convert_to_hf.py --help     # 把训练出的 draft checkpoint 转成 HF/SGLang 可加载格式
```
导出的 draft 即可挂到 SGLang/vLLM/TRT-LLM 做 GLM-5.2 的 speculative decoding。

---

## 8. 故障排查

| 现象 | 处理 |
|---|---|
| 训练 OOM | `inference.sglang.mem_fraction_static` 0.5→0.45→0.4;或降 `training.max_seq_length` |
| 推理 KV 不够 / prefill 失败 | 略升 `mem_fraction_static`,或降 `inference.inference_batch_size` |
| DSA kernel 崩溃 | 确认 B200=sm100 且 sglang ≥ 锁定 commit;**别在 sm89/L20 上跑**(需社区 ada_dsa port) |
| NCCL 端口/超时(colocate 两套通信组冲突) | 设不同 `mooncake` 端口;检查 `ray stop --force` 清干净;`tools/kill_all_torchspec.sh` |
| capture 到的 hidden 维度不对 | 核对 draft config `target_layer_ids`(注意 sglang 内部 +1 偏移)与 `target_hidden_size=6144` |
| 共享机器 Ray 串台 | 用 `RAY_ADDRESS=local` 强制本地实例 |
| `min_loss_tokens` 报错 | 设 ≥ 2×`dflash_block_size` |

---

## 附:本方案涉及的文件

| 文件 | 状态 |
|---|---|
| `configs/sglang_glm52_dspark_8card_colocate.yaml` | 新增(主配置) |
| `torchspec/config/glm52_dspark_draft_config.json` | 新增(draft 维度,mask_token_id 待核验) |
| `torchspec/data/template.py` (`glm`) | 改(chat template,待核验) |
| `configs/sglang_glm52_dspark_2node.yaml` | 新增(备选:2 节点更快) |

> **关键提醒**:`colocate` 是 docs 标注的 "Dev" 模式(`docs/ray.md:40`),实验性。首次务必先过 Step A/B 冒烟,再上正式训练。
