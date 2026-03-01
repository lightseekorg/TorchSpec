"""Profile sglang engine generate() with large batches."""

import os
import random
import time

import sglang as sgl

os.environ["MOONCAKE_MASTER_HOST"] = "0.0.0.0"
os.environ["MOONCAKE_MASTER_PORT"] = "50051"
os.environ["MOONCAKE_METADATA_PORT"] = "8090"
os.environ["MOONCAKE_PROTOCOL"] = "rdma"
os.environ["MOONCAKE_DEVICE_NAME"] = "mlx5_4,mlx5_5,mlx5_6,mlx5_7,mlx5_8,mlx5_9,mlx5_10,mlx5_11"

BATCH_SIZE = 4
SEQ_LEN = 32_768
NUM_STEPS = 2
MODEL_PATH = "Qwen/Qwen3-8B"
TP_SIZE = 4
AUX_LAYER_IDS = [2, 4, 6]
HIDDEN_DIM = 4096
NUM_AUX_LAYERS = len(AUX_LAYER_IDS)

# Per-sample bytes: hidden_states(seq×dim×aux×bf16) + input_ids(seq×i64) + last_hidden(seq×dim×bf16)
_per_sample = SEQ_LEN * (HIDDEN_DIM * NUM_AUX_LAYERS * 2 + 8 + HIDDEN_DIM * 2)
_buf_size = int(_per_sample * 1.2)
# global_segment must cover all in-flight samples (batch × steps + warmup headroom)
_segment_size = int(_per_sample * BATCH_SIZE * (NUM_STEPS + 1) * 1.3)
os.environ["MOONCAKE_HOST_BUFFER_SIZE"] = str(_buf_size)
os.environ["MOONCAKE_GLOBAL_SEGMENT_SIZE"] = str(_segment_size)
os.environ["MOONCAKE_LOCAL_BUFFER_SIZE"] = str(_segment_size)
os.environ["MOONCAKE_ENABLE_GPU_DIRECT"] = "0"
os.environ["MOONCAKE_GPU_BUFFER_SIZE"] = str(_buf_size)


def make_batch(batch_size: int, seq_len: int, step: int) -> dict:
    input_ids_list = [
        [random.randint(100, 30000) for _ in range(seq_len)] for _ in range(batch_size)
    ]
    data_ids = [f"step{step}_seq{i}" for i in range(batch_size)]
    return {
        "input_ids": input_ids_list,
        "spec_training_data_id": data_ids,
        "sampling_params": {"max_new_tokens": 0},
        "return_hidden_states": True,
    }


if __name__ == "__main__":
    print(f"Config: batch_size={BATCH_SIZE}, seq_len={SEQ_LEN}, steps={NUM_STEPS}, tp={TP_SIZE}")

    engine = sgl.Engine(
        disable_radix_cache=True,
        model_path=MODEL_PATH,
        disable_cuda_graph=True,
        enable_return_hidden_states=True,
        enable_aux_hidden_states=True,
        aux_hidden_state_layer_ids=AUX_LAYER_IDS,
        enable_spec_training_mooncake=True,
        log_level="info",
        tp_size=TP_SIZE,
    )

    # Warmup (not profiled)
    print("\n=== Warmup ===")
    warmup_batch = make_batch(BATCH_SIZE, SEQ_LEN, step=-1)
    t0 = time.perf_counter()
    engine.generate(**warmup_batch)
    print(f"Warmup done in {time.perf_counter() - t0:.2f}s")

    # Start profiling
    profile_dir = os.environ.get("PROFILE_DIR", "/tmp/sgl_profile")
    os.makedirs(profile_dir, exist_ok=True)
    print(f"\n=== Profiling {NUM_STEPS} steps -> {profile_dir} ===")
    engine.start_profile(
        output_dir=profile_dir,
        activities=["CPU", "GPU"],
        with_stack=True,
        record_shapes=True,
    )

    for step in range(NUM_STEPS):
        batch = make_batch(BATCH_SIZE, SEQ_LEN, step=step)
        t0 = time.perf_counter()
        results = engine.generate(**batch)
        elapsed = time.perf_counter() - t0
        total_tokens = BATCH_SIZE * SEQ_LEN
        print(
            f"  Step {step}: {elapsed:.2f}s | "
            f"{total_tokens} tokens | "
            f"{total_tokens / elapsed:.0f} tok/s | "
            f"mooncake_keys={sum(len(r['meta_info'].get('spec_training_mooncake_store_keys', [])) for r in results)}"
        )

    engine.stop_profile()
    print(f"\n=== Profile traces saved to {profile_dir} ===")
    print("View with: chrome://tracing or tensorboard --logdir", profile_dir)

    engine.shutdown()
