"""
Minimal SGLang server on Modal — for inference benchmarking and learning.

No DoorDash infra dependencies. Just Modal + a HuggingFace model.

SGLang is an alternative inference engine to vLLM with its own scheduling and
memory management strategies. Deploy both side-by-side to compare!

Usage:
    modal serve sglang_server.py --env sandbox    # Dev mode (hot-reload, temporary URL)
    modal deploy sglang_server.py --env sandbox   # Persistent deployment (stable URL)

Test:
    curl <MODAL_URL>/v1/models

    curl <MODAL_URL>/v1/chat/completions \
      -H "Content-Type: application/json" \
      -d '{"model": "default", "messages": [{"role": "user", "content": "Hello!"}], "max_tokens": 64}'

Token Created
Run this command to store the token on your computer.
Refer to the documentation about configuration for how to set credentials via environment variables instead.
modal token set --token-id ak-l6eYjquYvG1smlyMfH0EKl --token-secret as-eJoVAHKgk0J9iRxNgMNPRJ --profile=doordash

This stores the token in a profile named doordash. To activate, run:

modal profile activate doordash

"""

import subprocess

import modal

# =============================================================================
# Configuration — edit these to experiment
# =============================================================================

# Model -----------------------------------------------------------------------
MODEL_ID = "Qwen/Qwen2.5-7B-Instruct"  # Any HuggingFace model ID
MODEL_REVISION = None  # Pin to a commit hash, or None for latest

# Hardware ---------------------------------------------------------------------
GPU_TYPE = "H100"  # Options: "A100", "H100", "H200"
GPU_COUNT = 1  # Tensor parallelism degree

# Memory -----------------------------------------------------------------------
MAX_CONTEXT_LEN = 4096  # Max context window (input + output tokens)
GPU_MEM_FRACTION = 0.90  # Fraction of GPU memory for KV cache (0.0–1.0)

# Scaling ----------------------------------------------------------------------
IDLE_TIMEOUT_SECS = 300  # Scale to zero after N seconds idle
STARTUP_TIMEOUT_SECS = 600  # Max seconds to wait for model load

# SGLang engine flags — toggle these to study their effect on performance ------
CHUNKED_PREFILL_SIZE = None  # Max tokens per prefill chunk (None = auto)
MAX_RUNNING_REQUESTS = None  # Max concurrent decoding requests (None = auto)
ENABLE_METRICS = True  # Expose Prometheus metrics at /metrics

# =============================================================================
# Modal app definition
# =============================================================================

app = modal.App("sglang-benchmark")

sglang_image = (
    modal.Image.from_registry("nvidia/cuda:12.4.0-devel-ubuntu22.04", add_python="3.11")
    .apt_install("libnuma-dev")
    .uv_pip_install(
        "flashinfer-python==0.5.3",
        "sglang==0.5.6",
        "huggingface_hub[hf_transfer]",
        "transformers",
    )
    .env(
        {
            "HF_HUB_ENABLE_HF_TRANSFER": "1",
            "TORCH_CUDA_ARCH_LIST": "9.0 9.0a",
        }
    )
)


@app.function(
    image=sglang_image,
    gpu=f"{GPU_TYPE}:{GPU_COUNT}",
    scaledown_window=IDLE_TIMEOUT_SECS,
    timeout=600,
    min_containers=0,
    max_containers=1,
    volumes={
        "/root/.cache/huggingface": modal.Volume.from_name(
            "hf-cache", create_if_missing=True
        ),
    },
    # For private HuggingFace models, create a Modal secret with your HF_TOKEN:
    #   modal secret create huggingface-secret HF_TOKEN=hf_xxx
    # Then uncomment the line below:
    # secrets=[modal.Secret.from_name("huggingface-secret")],
)
@modal.web_server(port=8000, startup_timeout=STARTUP_TIMEOUT_SECS)
def serve_sglang():
    """Start SGLang's OpenAI-compatible server as a subprocess."""
    cmd = [
        "python3",
        "-m",
        "sglang.launch_server",
        "--model-path",
        MODEL_ID,
        "--host",
        "0.0.0.0",
        "--port",
        "8000",
        "--served-model-name",
        "default",
        "--tp",
        str(GPU_COUNT),
        "--mem-fraction-static",
        str(GPU_MEM_FRACTION),
        "--context-length",
        str(MAX_CONTEXT_LEN),
    ]

    if MODEL_REVISION:
        cmd.extend(["--revision", MODEL_REVISION])
    if CHUNKED_PREFILL_SIZE is not None:
        cmd.extend(["--chunked-prefill-size", str(CHUNKED_PREFILL_SIZE)])
    if MAX_RUNNING_REQUESTS is not None:
        cmd.extend(["--max-running-requests", str(MAX_RUNNING_REQUESTS)])
    if ENABLE_METRICS:
        cmd.append("--enable-metrics")

    print(f"[sglang-benchmark] Starting SGLang: {' '.join(cmd)}")
    subprocess.Popen(cmd)
