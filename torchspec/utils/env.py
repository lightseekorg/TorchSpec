import os

# Environment variables that should be forwarded to all Ray actors.
# NOTE: TORCHINDUCTOR_CACHE_DIR is intentionally excluded — each node should
# use its own node-local default (/tmp/torchinductor_$USER/) to avoid
# cross-node triton kernel cache corruption over NFS.
_TORCHSPEC_ENV_KEYS = [
    "CUDA_LAUNCH_BLOCKING",
    "GLOO_SOCKET_IFNAME",
    "HF_HOME",
    "HF_TOKEN",
    "MC_LOG_LEVEL",
    "MODELOPT_MAX_TOKENS_PER_EXPERT",
    "NCCL_DEBUG",
    "NCCL_SOCKET_IFNAME",
    "SGLANG_ALLOW_OVERWRITE_LONGER_CONTEXT_LEN",
    "SGLANG_DISABLE_CUDNN_CHECK",
    "SGLANG_VLM_CACHE_SIZE_MB",
    "TORCHSPEC_LOG_DIR",
    "TORCHSPEC_LOG_LEVEL",
    "TP_SOCKET_IFNAME",
    "CUTE_DSL_CACHE_DIR",
    "TORCHSPEC_FLASH_ATTN_OPT_LEVEL",
]

# Prevent Ray from overriding VISIBLE_DEVICES so actors manage GPU assignment themselves.
# Reference: https://github.com/ray-project/ray/blob/161849364/python/ray/_private/accelerators/
_RAY_NOSET_VISIBLE_DEVICES_KEYS = [
    "RAY_EXPERIMENTAL_NOSET_CUDA_VISIBLE_DEVICES",
    "RAY_EXPERIMENTAL_NOSET_ROCR_VISIBLE_DEVICES",
    "RAY_EXPERIMENTAL_NOSET_ASCEND_RT_VISIBLE_DEVICES",
    "RAY_EXPERIMENTAL_NOSET_HABANA_VISIBLE_MODULES",
    "RAY_EXPERIMENTAL_NOSET_NEURON_RT_VISIBLE_CORES",
    "RAY_EXPERIMENTAL_NOSET_TPU_VISIBLE_CHIPS",
    "RAY_EXPERIMENTAL_NOSET_ONEAPI_DEVICE_SELECTOR",
]


def get_torchspec_env_vars() -> dict[str, str]:
    """Return common environment variables for all Ray actors.

    Includes:
    - TORCHSPEC_* variables (e.g. log level) from the current process
    - RAY_EXPERIMENTAL_NOSET_*_VISIBLE_DEVICES = "1" to prevent Ray from
      overriding device visibility

    Intended for use with ``ray.remote(runtime_env={"env_vars": ...})``.
    Call-site env vars merged after this dict take higher priority.
    """
    env = {k: "1" for k in _RAY_NOSET_VISIBLE_DEVICES_KEYS}
    env.update({k: os.environ[k] for k in _TORCHSPEC_ENV_KEYS if k in os.environ})
    return env
