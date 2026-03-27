"""
DFlash inference benchmark on Modal using SGLang server — z-lab methodology.

Launches an SGLang server (tp=2 on 2x H100) with DFlash speculative decoding
inside a Modal container, then benchmarks using HTTP requests. Much faster than
the Transformers backend because SGLang uses:
  - PagedAttention with optimized KV cache management
  - RadixAttention for prefix caching across prompts
  - CUDA graphs for reduced kernel launch overhead
  - FlashInfer/FlashAttention attention kernels
  - Tensor parallelism across 2 GPUs

Uses the same datasets, prompt formatting, and evaluation protocol as
z-lab/dflash (https://github.com/z-lab/dflash).

Setup (one-time — same secrets as modal_dflash_train.py):
    modal token set --token-id <id> --token-secret <secret>
    modal secret create huggingface-secret HF_TOKEN=hf_...

Usage:
    # Default: quick run (30 samples each) on GSM8K + MATH-500
    modal run --env sandbox scripts/modal_dflash_benchmark_sglang.py

    # Full z-lab sample counts (128 each)
    modal run --env sandbox scripts/modal_dflash_benchmark_sglang.py --no-quick

    # All z-lab benchmarks
    modal run --env sandbox scripts/modal_dflash_benchmark_sglang.py --all-datasets

    # Skip baseline (only DFlash, fastest)
    modal run --env sandbox scripts/modal_dflash_benchmark_sglang.py --skip-baseline

    # Custom draft model (HuggingFace repo or Modal volume path)
    modal run --env sandbox scripts/modal_dflash_benchmark_sglang.py \
        --draft-repo Xingh3/dflash-qwen3-8b-1epoch
"""

from __future__ import annotations

import modal

# =============================================================================
# Constants — reuse the same SGLang image as modal_dflash_train.py
# =============================================================================

TORCHSPEC_REPO = "https://github.com/zhubohao911/TorchSpec.git"
TORCHSPEC_BRANCH = "feature/dflash-training"

REPO_DIR = "/workspace/TorchSpec"
HF_CACHE_DIR = "/root/.cache/huggingface"

BENCHMARK_GPU = "H100:1"

ZLAB_BENCHMARKS_FULL = {
    "gsm8k": 128,
    "math500": 128,
    "aime24": 30,
    "aime25": 30,
    "humaneval": 164,
    "mbpp": 128,
    "livecodebench": 128,
    "swe-bench": 128,
    "mt-bench": 80,
    "alpaca": 128,
}

ZLAB_BENCHMARKS_QUICK = {
    "gsm8k": 30,
    "math500": 30,
    "aime24": 30,
    "aime25": 30,
    "humaneval": 30,
    "mbpp": 30,
    "livecodebench": 30,
    "swe-bench": 30,
    "mt-bench": 30,
    "alpaca": 30,
}

# =============================================================================
# Modal app + volumes
# =============================================================================

app = modal.App("torchspec-dflash-benchmark-sglang")

hf_cache_vol = modal.Volume.from_name("hf-cache", create_if_missing=True)
outputs_vol = modal.Volume.from_name("torchspec-outputs", create_if_missing=False)

OUTPUTS_DIR = "/workspace/outputs"

# =============================================================================
# Container image — same SGLang image as training script
# =============================================================================

base_image = (
    modal.Image.from_registry(
        "nvidia/cuda:12.4.0-devel-ubuntu22.04", add_python="3.11"
    )
    .apt_install(
        "git", "vim", "htop",
        "libibverbs-dev", "librdmacm-dev", "libnuma-dev",
        "libcurl4-openssl-dev",
    )
    .pip_install(
        "torch", "torchvision", "torchaudio",
        extra_index_url="https://download.pytorch.org/whl/cu124",
    )
    .run_commands(
        f"git clone {TORCHSPEC_REPO} {REPO_DIR}",
        f"cd {REPO_DIR} && git checkout {TORCHSPEC_BRANCH}",
    )
    .pip_install(
        "huggingface_hub[hf_transfer]",
        "transformers==4.57.1",
        "datasets",
        "tqdm",
        "accelerate",
        "pydantic",
        "omegaconf",
        "psutil",
        "numpy<2.4",
        "pyzmq",
        "numba",
        "cmake",
        "ninja",
        "packaging",
        "setuptools",
        "requests",
    )
    .run_commands(f"cd {REPO_DIR} && pip install -e '.[dev]'")
    .run_commands(
        "mkdir -p /root/.cache && ln -sf /root/.cache/huggingface /root/.cache/huggingface || true",
    )
    .env(
        {
            "HF_HUB_ENABLE_HF_TRANSFER": "1",
            "PYTORCH_ALLOC_CONF": "expandable_segments:True",
            "TORCHINDUCTOR_MAX_AUTOTUNE_GEMM_BACKENDS": "ATEN,TRITON",
            "HF_HOME": HF_CACHE_DIR,
            "SGLANG_ALLOW_OVERWRITE_LONGER_CONTEXT_LEN": "1",
            "SGLANG_ENABLE_SPEC_V2": "1",
            "SGLANG_ENABLE_DFLASH_SPEC_V2": "1",
            "SGLANG_ENABLE_OVERLAP_PLAN_STREAM": "1",
        }
    )
)

SGLANG_BENCH_DIR = "/workspace/sglang_dflash"

sglang_image = (
    base_image
    .run_commands(
        f"git clone https://github.com/sgl-project/sglang.git {SGLANG_BENCH_DIR}",
        f"cd {SGLANG_BENCH_DIR} && git fetch origin refs/pull/16818/head && git checkout FETCH_HEAD",
        f"cd {SGLANG_BENCH_DIR} && pip install -e 'python[all]'",
        # PR #16818 defines set_eagle3_layers_to_capture but not set_dflash_layers_to_capture.
        # The DFlash hidden-state capture uses the same mechanism. Add alias for Qwen2/Qwen3.
        f"cd {SGLANG_BENCH_DIR}/python/sglang/srt/models && "
        "for f in qwen2.py qwen3.py; do "
        "  if grep -q set_eagle3_layers_to_capture $f && ! grep -q set_dflash_layers_to_capture $f; then "
        r"    sed -i '/^    def set_eagle3_layers_to_capture/i\\    def set_dflash_layers_to_capture(self, layer_ids=None):\n        return self.set_eagle3_layers_to_capture(layer_ids)\n' $f; "
        "    echo \"Patched $f\"; "
        "  fi; "
        "done",
    )
)


# =============================================================================
# Dataset loading — mirrors z-lab/dflash model/utils.py exactly
# =============================================================================

def load_benchmark_dataset(data_name: str, max_samples: int | None = None):
    """Load and format datasets using the same logic as z-lab/dflash."""
    from datasets import load_dataset, Features, Sequence, Value

    if data_name == "gsm8k":
        dataset = load_dataset("openai/gsm8k", "main", split="test")
        prompt_fmt = "{question}\nPlease reason step by step, and put your final answer within \\boxed{{}}."
        dataset = dataset.map(lambda x: {"turns": [prompt_fmt.format(**x)]})

    elif data_name == "math500":
        dataset = load_dataset("HuggingFaceH4/MATH-500", split="test")
        prompt_fmt = "{problem}\nPlease reason step by step, and put your final answer within \\boxed{{}}."
        dataset = dataset.map(lambda x: {"turns": [prompt_fmt.format(**x)]})

    elif data_name == "aime24":
        dataset = load_dataset("HuggingFaceH4/aime_2024", split="train")
        prompt_fmt = "{problem}\nPlease reason step by step, and put your final answer within \\boxed{{}}."
        dataset = dataset.map(lambda x: {"turns": [prompt_fmt.format(**x)]})

    elif data_name == "aime25":
        dataset = load_dataset("MathArena/aime_2025", split="train")
        prompt_fmt = "{problem}\nPlease reason step by step, and put your final answer within \\boxed{{}}."
        dataset = dataset.map(lambda x: {"turns": [prompt_fmt.format(**x)]})

    elif data_name == "alpaca":
        dataset = load_dataset("tatsu-lab/alpaca", split="train")
        dataset = dataset.map(lambda x: {
            "formatted_input": (
                f"{x['instruction']}\n\nInput:\n{x['input']}" if x["input"] else x["instruction"]
            )
        })
        dataset = dataset.map(lambda x: {"turns": [x["formatted_input"]]})

    elif data_name == "mt-bench":
        dataset = load_dataset("HuggingFaceH4/mt_bench_prompts", split="train")
        dataset = dataset.map(lambda x: {"turns": x["prompt"]})

    elif data_name == "humaneval":
        dataset = load_dataset("openai/openai_humaneval", split="test")
        prompt_fmt = (
            "Write a solution to the following problem and make sure that it passes the tests:\n"
            "```python\n{prompt}\n```"
        )
        dataset = dataset.map(lambda x: {"turns": [prompt_fmt.format(**x)]})

    elif data_name == "mbpp":
        dataset = load_dataset("google-research-datasets/mbpp", "sanitized", split="test")
        dataset = dataset.map(lambda x: {"turns": [x["prompt"]]})

    elif data_name == "swe-bench":
        dataset = load_dataset("princeton-nlp/SWE-bench_Lite", split="test")
        prompt_fmt = "Problem Statement:\n{problem_statement}\nPlease fix the issue described above."
        dataset = dataset.map(lambda x: {"turns": [prompt_fmt.format(**x)]})

    elif data_name == "livecodebench":
        base = "https://huggingface.co/datasets/livecodebench/code_generation_lite/resolve/main/"
        allowed_files = [f"test{s}.jsonl" for s in ["", "2", "3", "4", "5", "6"]]
        urls = [base + fn for fn in allowed_files]
        dataset = load_dataset("json", data_files={"test": urls})["test"]

        def format_lcb(doc):
            system_prompt = (
                "You are an expert Python programmer. You will be given a question (problem specification) "
                "and will generate a correct Python program that matches the specification and passes all tests. "
                "You will NOT return anything except for the program"
            )
            question_block = f"### Question:\n{doc['question_content']}"
            if doc.get("starter_code"):
                format_message = "### Format: Use the following code structure:"
                code_block = f"```python\n{doc['starter_code']}\n```"
            else:
                format_message = "### Format: Write your code in the following format:"
                code_block = "```python\n# YOUR CODE HERE\n```"
            answer_footer = "### Answer: (use the provided format with backticks)"
            return f"{system_prompt}\n\n{question_block}\n\n{format_message}\n{code_block}\n\n{answer_footer}"

        target_features = Features({"turns": Sequence(Value("large_string"))})
        dataset = dataset.map(
            lambda x: {"turns": [format_lcb(x)]},
            remove_columns=dataset.column_names,
            features=target_features,
        )
    else:
        raise ValueError(f"Unknown dataset: {data_name}")

    if max_samples is not None and len(dataset) > max_samples:
        dataset = dataset.shuffle(seed=0).select(range(max_samples))

    return dataset


# =============================================================================
# Benchmark runner — SGLang server inside container
# =============================================================================

@app.function(
    image=sglang_image,
    gpu=BENCHMARK_GPU,
    volumes={HF_CACHE_DIR: hf_cache_vol, OUTPUTS_DIR: outputs_vol},
    timeout=8 * 3600,
    secrets=[modal.Secret.from_name("huggingface-secret")],
)
def run_benchmark(
    target_model: str,
    draft_repo: str,
    dataset_name: str,
    max_samples: int,
    max_new_tokens: int,
    temperature: float,
    skip_baseline: bool,
    concurrency: int,
):
    import json
    import os
    import signal
    import socket
    import statistics
    import subprocess
    import sys
    import time
    from concurrent.futures import ThreadPoolExecutor, as_completed

    import numpy as np
    import requests
    import torch
    from transformers import AutoTokenizer

    device = "cuda"
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    mem_gb = torch.cuda.get_device_properties(0).total_memory / 1e9
    print(f"VRAM: {mem_gb:.1f} GB")

    # ── Patch draft model config to z-lab format for SGLang compatibility ──
    def patch_draft_model_for_sglang(repo_id: str) -> str:
        """Download draft model and fix config to z-lab format if needed.
        
        SGLang's DFlash support (PR #16818) expects model_type="qwen3" with
        a nested dflash_config. Our TorchSpec conversion produces model_type="dflash".
        Also copies z-lab's custom modeling files (dflash.py, utils.py) needed by auto_map.
        Accepts a local path (e.g. /workspace/outputs/...) or a HuggingFace repo id.
        Returns local path to the patched model.
        """
        import shutil
        from huggingface_hub import snapshot_download, hf_hub_download

        is_local = repo_id.startswith("/") and os.path.isdir(repo_id)
        local_dir = f"/tmp/dflash_model_{repo_id.replace('/', '_')}"

        if is_local:
            print(f"Copying draft model from volume: {repo_id}")
            if os.path.exists(local_dir):
                shutil.rmtree(local_dir)
            shutil.copytree(repo_id, local_dir, symlinks=False)
        else:
            max_retries = 5
            for attempt in range(max_retries):
                try:
                    print(f"Downloading draft model {repo_id} (attempt {attempt + 1}/{max_retries})...")
                    cached_dir = snapshot_download(repo_id=repo_id)
                    if os.path.exists(local_dir):
                        shutil.rmtree(local_dir)
                    shutil.copytree(cached_dir, local_dir, symlinks=False)
                    break
                except Exception as e:
                    if attempt < max_retries - 1:
                        wait = 2 ** attempt * 5
                        print(f"  Download failed: {e.__class__.__name__}: {e}")
                        print(f"  Retrying in {wait}s...")
                        time.sleep(wait)
                    else:
                        raise RuntimeError(
                            f"Failed to download {repo_id} after {max_retries} attempts: {e}"
                        ) from e
        
        config_path = os.path.join(local_dir, "config.json")
        with open(config_path) as f:
            config = json.load(f)
        
        if config.get("model_type") == "qwen3" and "dflash_config" in config:
            print("  Config already in z-lab format, no patching needed.")
            return local_dir
        
        print(f"  Patching config: model_type={config.get('model_type')} -> qwen3 + dflash_config")
        
        target_layer_ids = config.get("target_layer_ids", [1, 9, 17, 25, 33])
        mask_token_id = config.get("mask_token_id", 151669)
        
        new_config = {
            "architectures": ["DFlashDraftModel"],
            "attention_bias": False,
            "attention_dropout": 0.0,
            "auto_map": {"AutoModel": "dflash.DFlashDraftModel"},
            "block_size": 16,
            "bos_token_id": 151643,
            "dflash_config": {
                "mask_token_id": mask_token_id,
                "target_layer_ids": target_layer_ids,
            },
            "dtype": "bfloat16",
            "eos_token_id": 151645,
            "head_dim": 128,
            "hidden_act": "silu",
            "hidden_size": config.get("hidden_size", 4096),
            "initializer_range": 0.02,
            "intermediate_size": config.get("intermediate_size", 12288),
            "layer_types": ["full_attention"] * config.get("num_hidden_layers", 5),
            "max_position_embeddings": config.get("max_position_embeddings", 40960),
            "max_window_layers": config.get("num_hidden_layers", 5),
            "model_type": "qwen3",
            "num_attention_heads": config.get("num_attention_heads", 32),
            "num_hidden_layers": config.get("num_hidden_layers", 5),
            "num_key_value_heads": config.get("num_key_value_heads", 8),
            "num_target_layers": config.get("target_num_hidden_layers", 36),
            "rms_norm_eps": config.get("rms_norm_eps", 1e-06),
            "rope_scaling": None,
            "rope_theta": config.get("rope_theta", 1000000),
            "sliding_window": None,
            "tie_word_embeddings": False,
            "transformers_version": "4.57.1",
            "use_cache": True,
            "use_sliding_window": False,
            "vocab_size": config.get("vocab_size", 151936),
        }
        
        with open(config_path, "w") as f:
            json.dump(new_config, f, indent=2)
        print("  Config patched.")
        
        zlab_ref = "z-lab/Qwen3-8B-DFlash-b16"
        for fname in ["dflash.py", "modeling_dflash.py", "utils.py"]:
            for dl_attempt in range(3):
                try:
                    src = hf_hub_download(repo_id=zlab_ref, filename=fname)
                    break
                except Exception:
                    if dl_attempt < 2:
                        time.sleep(5)
                    else:
                        raise
            shutil.copy2(src, os.path.join(local_dir, fname))
            print(f"  Copied {fname} from {zlab_ref}")
        
        # Remap TorchSpec weight names to z-lab/SGLang convention
        safetensors_path = os.path.join(local_dir, "model.safetensors")
        if os.path.exists(safetensors_path):
            from safetensors.torch import load_file, save_file
            
            WEIGHT_REMAP = {
                "context_proj.weight": "fc.weight",
                "context_norm.weight": "hidden_norm.weight",
                "final_norm.weight": "norm.weight",
            }
            
            print("  Remapping safetensors weight keys...")
            state_dict = load_file(safetensors_path)
            new_state_dict = {}
            skipped = []
            for key, tensor in state_dict.items():
                if key == "embed_tokens.weight":
                    skipped.append(key)
                    continue
                new_key = WEIGHT_REMAP.get(key, key)
                if new_key != key:
                    print(f"    {key} -> {new_key}")
                new_state_dict[new_key] = tensor
            
            if skipped:
                print(f"    Skipped: {skipped}")
            
            save_file(new_state_dict, safetensors_path)
            print(f"  Remapped {len(new_state_dict)} tensors, skipped {len(skipped)}")
        
        return local_dir

    draft_local_path = patch_draft_model_for_sglang(draft_repo)

    # ── Load dataset & tokenizer ──
    print(f"\nLoading dataset: {dataset_name} (max {max_samples} samples)...")
    dataset = load_benchmark_dataset(dataset_name, max_samples)
    print(f"Loaded {len(dataset)} samples")

    tokenizer = AutoTokenizer.from_pretrained(target_model, trust_remote_code=True)

    # Format all prompts with chat template
    all_prompts = []
    for idx in range(len(dataset)):
        instance = dataset[idx]
        turns = instance["turns"]
        messages = [{"role": "user", "content": turns[0]}]
        text = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )
        all_prompts.append(text)

    def find_free_port():
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.bind(("", 0))
            return s.getsockname()[1]

    SERVER_LOG = "/tmp/sglang_server.log"

    def wait_for_server(proc, url, timeout=600):
        """Poll server health endpoint until ready; check for process death."""
        deadline = time.time() + timeout
        while time.time() < deadline:
            ret = proc.poll()
            if ret is not None:
                print(f"\nSGLang server DIED with exit code {ret}. Log tail:")
                _print_server_log(tail=80)
                return False
            try:
                resp = requests.get(f"{url}/health", timeout=5)
                if resp.status_code == 200:
                    return True
            except (requests.ConnectionError, requests.Timeout):
                pass
            elapsed = int(time.time() - (deadline - timeout))
            if elapsed > 0 and elapsed % 30 == 0:
                print(f"  ... still waiting ({elapsed}s elapsed)")
                _print_server_log(tail=5)
            time.sleep(3)
        print(f"\nSGLang server TIMEOUT after {timeout}s. Log tail:")
        _print_server_log(tail=80)
        return False

    def _print_server_log(tail=40):
        try:
            with open(SERVER_LOG) as f:
                lines = f.readlines()
            for line in lines[-tail:]:
                print(f"  [sglang] {line.rstrip()}")
        except FileNotFoundError:
            print("  [sglang] (no log file)")

    def launch_sglang_server(model_path, port, draft_model_path=None):
        """Launch SGLang server as subprocess, return (process, base_url)."""
        cmd = [
            sys.executable, "-m", "sglang.launch_server",
            "--model-path", model_path,
            "--host", "127.0.0.1",
            "--port", str(port),
            "--tp-size", "1",
            "--dtype", "bfloat16",
            "--trust-remote-code",
            "--mem-fraction-static", "0.85",
        ]

        if draft_model_path:
            cmd.extend([
                "--speculative-algorithm", "DFLASH",
                "--speculative-draft-model-path", draft_model_path,
            ])

        env = {**os.environ}
        env["SGLANG_ALLOW_OVERWRITE_LONGER_CONTEXT_LEN"] = "1"
        env["SGLANG_ENABLE_SPEC_V2"] = "1"
        env["SGLANG_ENABLE_DFLASH_SPEC_V2"] = "1"
        env["SGLANG_ENABLE_OVERLAP_PLAN_STREAM"] = "1"

        print(f"Launching SGLang server: {' '.join(cmd)}")
        log_fh = open(SERVER_LOG, "w")
        proc = subprocess.Popen(
            cmd,
            env=env,
            stdout=log_fh,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
        return proc, f"http://127.0.0.1:{port}", log_fh

    def kill_server(proc, log_fh=None):
        """Kill server process tree aggressively (SIGKILL the whole group)."""
        pgid = None
        try:
            pgid = os.getpgid(proc.pid)
        except (ProcessLookupError, OSError):
            pass

        if pgid is not None:
            try:
                os.killpg(pgid, signal.SIGKILL)
            except (ProcessLookupError, OSError):
                pass

        try:
            proc.kill()
            proc.wait(timeout=5)
        except Exception:
            pass

        if log_fh:
            try:
                log_fh.close()
            except Exception:
                pass

    def send_request(base_url, prompt_text, max_new_tokens, temperature, timeout_s=600):
        """Send a single generate request to SGLang server."""
        sampling_params = {
            "temperature": float(temperature),
            "max_new_tokens": int(max_new_tokens),
        }
        if temperature < 1e-5:
            sampling_params["top_k"] = 1

        resp = requests.post(
            f"{base_url}/generate",
            json={"text": prompt_text, "sampling_params": sampling_params},
            timeout=timeout_s,
        )
        resp.raise_for_status()
        return resp.json()

    def flush_cache(base_url):
        try:
            requests.get(f"{base_url}/flush_cache", timeout=30)
        except Exception:
            pass

    def run_requests(base_url, prompts, max_new_tokens, temperature, concurrency, expect_dflash=False):
        """Run all prompts with given concurrency and collect metrics."""
        # Warmup batch (excluded from metrics, warms CUDA graphs)
        warmup_n = min(concurrency, len(prompts))
        for p in prompts[:warmup_n]:
            send_request(base_url, p, max_new_tokens=32, temperature=temperature)
        flush_cache(base_url)

        total_tokens = 0
        spec_accept_lengths = []
        latencies = []

        if concurrency <= 1:
            for i, prompt in enumerate(prompts):
                t0 = time.perf_counter()
                out = send_request(base_url, prompt, max_new_tokens, temperature)
                elapsed = time.perf_counter() - t0

                meta = out.get("meta_info", {}) or {}
                n_tokens = int(meta.get("completion_tokens", 0))
                total_tokens += n_tokens
                latencies.append(elapsed)

                if "spec_accept_length" in meta:
                    try:
                        spec_accept_lengths.append(float(meta["spec_accept_length"]))
                    except (TypeError, ValueError):
                        pass

                if (i + 1) % max(1, len(prompts) // 10) == 0:
                    tau_so_far = statistics.mean(spec_accept_lengths) if spec_accept_lengths else 0
                    tps = n_tokens / elapsed if elapsed > 0 else 0
                    print(f"  [{i+1}/{len(prompts)}] {n_tokens} tok, {elapsed:.2f}s, "
                          f"{tps:.1f} tok/s, τ={tau_so_far:.2f}")
        else:
            with ThreadPoolExecutor(max_workers=concurrency) as pool:
                futures = {
                    pool.submit(
                        send_request, base_url, prompt, max_new_tokens, temperature
                    ): i
                    for i, prompt in enumerate(prompts)
                }
                for fut in as_completed(futures):
                    out = fut.result()
                    meta = out.get("meta_info", {}) or {}
                    total_tokens += int(meta.get("completion_tokens", 0))
                    if "spec_accept_length" in meta:
                        try:
                            spec_accept_lengths.append(float(meta["spec_accept_length"]))
                        except (TypeError, ValueError):
                            pass

        avg_tau = statistics.mean(spec_accept_lengths) if spec_accept_lengths else None
        total_time = sum(latencies) if latencies else None

        return {
            "total_tokens": total_tokens,
            "avg_accept_length": avg_tau,
            "total_latency_s": total_time,
            "num_prompts": len(prompts),
            "spec_accept_lengths": spec_accept_lengths,
        }

    # ──────────────────────────────────────────────────────────────────────
    # Run baseline (target-only, no speculation)
    # ──────────────────────────────────────────────────────────────────────
    baseline_result = None
    if not skip_baseline:
        port_baseline = find_free_port()
        print(f"\n{'='*70}")
        print(f"BASELINE (target-only): {target_model}")
        print(f"{'='*70}")

        proc_baseline, url_baseline, log_fh_baseline = launch_sglang_server(target_model, port_baseline)
        try:
            print("Waiting for baseline server to start...")
            if not wait_for_server(proc_baseline, url_baseline, timeout=600):
                raise RuntimeError("Baseline SGLang server failed to start")
            print("Baseline server ready!")

            baseline_result = run_requests(
                url_baseline, all_prompts, max_new_tokens, temperature, concurrency,
            )
            print(f"\nBaseline: {baseline_result['total_tokens']} tokens generated")
        finally:
            kill_server(proc_baseline, log_fh_baseline)
            time.sleep(5)

    # ──────────────────────────────────────────────────────────────────────
    # Run DFlash (speculative decoding with SGLang KV cache)
    # ──────────────────────────────────────────────────────────────────────
    port_dflash = find_free_port()
    print(f"\n{'='*70}")
    print(f"DFLASH: {target_model} + {draft_repo} (patched: {draft_local_path})")
    print(f"{'='*70}")

    proc_dflash, url_dflash, log_fh_dflash = launch_sglang_server(target_model, port_dflash, draft_model_path=draft_local_path)
    dflash_result = None
    try:
        print("Waiting for DFlash server to start...")
        if not wait_for_server(proc_dflash, url_dflash, timeout=600):
            raise RuntimeError("DFlash SGLang server failed to start — see log above")
        print("DFlash server ready!")

        dflash_result = run_requests(
            url_dflash, all_prompts, max_new_tokens, temperature, concurrency,
            expect_dflash=True,
        )
    finally:
        kill_server(proc_dflash, log_fh_dflash)

    if dflash_result is None:
        raise RuntimeError("DFlash benchmark produced no results")

    # ──────────────────────────────────────────────────────────────────────
    # Results — printed immediately so they survive container shutdown
    # ──────────────────────────────────────────────────────────────────────
    avg_tau = dflash_result["avg_accept_length"] or 0.0

    # τ distribution from SGLang's per-request accept lengths
    if dflash_result["spec_accept_lengths"]:
        print(f"\n{'='*70}")
        print(f"τ DISTRIBUTION ({dataset_name}) — per-request average acceptance length")
        print(f"{'='*70}")
        acc_lens = dflash_result["spec_accept_lengths"]
        bins = [0, 1, 2, 3, 4, 5, 6, 8, 10, 16]
        for i in range(len(bins) - 1):
            count = sum(1 for a in acc_lens if bins[i] <= a < bins[i + 1])
            if count > 0:
                bar = "█" * int(count / len(acc_lens) * 40)
                pct = count / len(acc_lens) * 100
                print(f"  τ∈[{bins[i]},{bins[i+1]}): {count:4d} ({pct:5.1f}%) {bar}")
        count_high = sum(1 for a in acc_lens if a >= bins[-1])
        if count_high > 0:
            bar = "█" * int(count_high / len(acc_lens) * 40)
            pct = count_high / len(acc_lens) * 100
            print(f"  τ≥{bins[-1]:2d}:    {count_high:4d} ({pct:5.1f}%) {bar}")

    result = {
        "dataset": dataset_name,
        "num_samples": len(all_prompts),
        "draft_model": draft_repo,
        "target_model": target_model,
        "gpu": torch.cuda.get_device_name(0),
        "max_new_tokens": max_new_tokens,
        "temperature": temperature,
        "concurrency": concurrency,
        "backend": "sglang",
        "avg_acceptance_length": round(avg_tau, 2) if avg_tau else 0,
    }

    # Compute speedup from per-request latencies (baseline sequential only)
    if baseline_result and baseline_result["total_latency_s"] and dflash_result["total_latency_s"]:
        baseline_tpt = baseline_result["total_tokens"] / baseline_result["total_latency_s"]
        dflash_tpt = dflash_result["total_tokens"] / dflash_result["total_latency_s"]
        speedup = dflash_tpt / baseline_tpt if baseline_tpt > 0 else 0
        result["baseline_tok_per_s"] = round(baseline_tpt, 1)
        result["dflash_tok_per_s"] = round(dflash_tpt, 1)
        result["decoding_speedup"] = round(speedup, 2)

    print(f"\n{'='*70}")
    print(f"RESULTS: {dataset_name} (SGLang backend)")
    print(f"{'='*70}")
    print(f"  Draft model:           {draft_repo}")
    print(f"  Target model:          {target_model}")
    print(f"  GPU:                   {torch.cuda.get_device_name(0)}")
    print(f"  Samples:               {len(all_prompts)}")
    print(f"  Max new tokens:        {max_new_tokens}")
    print(f"  Temperature:           {temperature}")
    print(f"  Concurrency:           {concurrency}")
    print(f"  Backend:               SGLang (KV cache, CUDA graphs, paged attention)")
    print(f"  Acceptance length (τ): {avg_tau:.2f}")

    if "decoding_speedup" in result:
        print(f"  Baseline throughput:   {result['baseline_tok_per_s']} tok/s")
        print(f"  DFlash throughput:     {result['dflash_tok_per_s']} tok/s")
        print(f"  Decoding speedup:      {result['decoding_speedup']:.2f}x")

    # z-lab reference comparison
    zlab_tau_ref = {
        "gsm8k": 3.38, "math500": 4.61, "aime24": 4.12, "aime25": 4.07,
    }
    zlab_speedup_ref = {
        "gsm8k": 5.20, "math500": 6.17, "aime24": 5.91, "aime25": 5.85,
        "humaneval": 5.20, "mbpp": 4.75, "livecodebench": 5.43,
        "swe-bench": 2.92, "mt-bench": 2.79, "alpaca": 2.27,
    }
    if dataset_name in zlab_tau_ref:
        ref_tau = zlab_tau_ref[dataset_name]
        print(f"\n  z-lab reference τ:     {ref_tau:.2f} (ours: {avg_tau:.2f}, "
              f"gap: {ref_tau - avg_tau:+.2f})")
    if dataset_name in zlab_speedup_ref:
        ref_spd = zlab_speedup_ref[dataset_name]
        print(f"  z-lab reference speed: {ref_spd:.2f}x")

    print(f"{'='*70}")
    sys.stdout.flush()

    return result


# =============================================================================
# CLI entry point
# =============================================================================

@app.local_entrypoint()
def main(
    target_model: str = "Qwen/Qwen3-8B",
    draft_repo: str = "/workspace/outputs/dflash-qwen3-8b/hf_model",
    datasets: str = "gsm8k,math500",
    max_new_tokens: int = 2048,
    temperature: float = 0.0,
    skip_baseline: bool = False,
    concurrency: int = 1,
    all_datasets: bool = False,
    quick: bool = True,
):
    benchmarks = ZLAB_BENCHMARKS_QUICK if quick else ZLAB_BENCHMARKS_FULL

    if all_datasets:
        ds_list = list(benchmarks.keys())
    else:
        ds_list = [d.strip() for d in datasets.split(",")]

    mode_label = "quick (30 samples)" if quick else "full (z-lab counts)"

    print("=" * 70)
    print("  DFlash Inference Benchmark on Modal (SGLang tp=2)")
    print("=" * 70)
    print(f"  Target model:   {target_model}")
    print(f"  Draft model:    {draft_repo}")
    print(f"  GPU:            {BENCHMARK_GPU}")
    print(f"  Datasets:       {', '.join(ds_list)}")
    print(f"  Max new tokens: {max_new_tokens}")
    print(f"  Temperature:    {temperature}")
    print(f"  Concurrency:    {concurrency}")
    print(f"  Skip baseline:  {skip_baseline}")
    print(f"  Mode:           {mode_label}")
    print(f"  Backend:        SGLang (tp=1, KV cache, CUDA graphs, paged attn)")
    print("=" * 70)

    all_results = []

    for ds_name in ds_list:
        max_samples = benchmarks.get(ds_name, 30)
        print(f"\n>>> Running: {ds_name} ({max_samples} samples) <<<\n")

        result = run_benchmark.remote(
            target_model=target_model,
            draft_repo=draft_repo,
            dataset_name=ds_name,
            max_samples=max_samples,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            skip_baseline=skip_baseline,
            concurrency=concurrency,
        )
        all_results.append(result)

    # ── Cross-dataset summary ──
    zlab_tau_ref = {
        "gsm8k": 3.38, "math500": 4.61, "aime24": 4.12, "aime25": 4.07,
    }
    zlab_speedup_ref = {
        "gsm8k": 5.20, "math500": 6.17, "aime24": 5.91, "aime25": 5.85,
        "humaneval": 5.20, "mbpp": 4.75, "livecodebench": 5.43,
        "swe-bench": 2.92, "mt-bench": 2.79, "alpaca": 2.27,
    }

    print(f"\n\n{'='*70}")
    print("CROSS-DATASET SUMMARY (SGLang backend)")
    print(f"{'='*70}")
    print(f"{'Dataset':<15} {'τ':>6} {'Speedup':>8} {'z-lab τ':>8} {'z-lab Spd':>10}")
    print("-" * 70)

    for r in all_results:
        ds = r["dataset"]
        tau = r.get("avg_acceptance_length", 0)
        spd = r.get("decoding_speedup", 0)
        ref_tau = zlab_tau_ref.get(ds, "-")
        ref_spd = zlab_speedup_ref.get(ds, "-")
        ref_tau_str = f"{ref_tau:.2f}" if isinstance(ref_tau, float) else ref_tau
        ref_spd_str = f"{ref_spd:.2f}x" if isinstance(ref_spd, float) else ref_spd
        spd_str = f"{spd:.2f}x" if spd else "-"
        print(f"{ds:<15} {tau:>6.2f} {spd_str:>8} {ref_tau_str:>8} {ref_spd_str:>10}")

    print(f"{'='*70}")
