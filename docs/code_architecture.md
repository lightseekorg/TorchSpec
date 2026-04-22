# TorchSpec Code Architecture

## Package Layout

```
torchspec/
├── config/                  # Configuration system (OmegaConf-based)
│   ├── train_config.py      #   Hierarchical dataclass configs (7 sections + Config root)
│   ├── inference_config.py  #   InferenceConfig + SGLangConfig (essential fields + extra_args passthrough)
│   ├── mooncake_config.py   #   Mooncake runtime config (env-var, auto-calculated sizes)
│   └── utils.py             #   Config loading utilities
├── ray/                     # Ray infrastructure (shared across all packages)
│   ├── ray_actor.py         #   RayActor base class (GPU setup, network utils)
│   ├── train_group.py       #   RayTrainGroup (manages training actor group)
│   └── placement_group.py   #   Placement group creation, GPU resource management
├── controller/              # Async pipeline orchestration
│   ├── training_controller.py  # AsyncTrainingController (Ray actor)
│   ├── inference_manager.py    # AsyncInferenceManager (Ray actor)
│   ├── loop.py              #   Main training loop
│   └── setup.py             #   build_mooncake_config, setup_async_training_with_engines, auto_calculate_training_steps
├── inference/               # Inference engine layer
│   ├── factory.py           #   create_inference_engines (placement group → engine actors)
│   └── engine/              #   Engine implementations
│       ├── base.py          #     InferenceEngine (ABC)
│       ├── hf_engine.py     #     HFEngine (Ray actor, inherits RayActor)
│       ├── hf_runner.py     #     HFRunner (core inference logic)
│       ├── sgl_engine.py   #     SglEngine (Ray actor, inherits RayActor)
│       └── vllm_engine.py   #     VllmEngine (Ray actor, uses vLLM extract_hidden_states)
├── models/                  # Model definitions
│   ├── eagle3.py            #   Eagle3Model (core forward/loss)
│   ├── draft/               #   Draft model implementations
│   │   ├── auto.py          #     AutoEagle3DraftModel factory
│   │   ├── base.py          #     Eagle3DraftModel (ABC)
│   │   └── llama3_eagle.py  #     LlamaForCausalLMEagle3
│   ├── target/              #   Target model abstractions
│   │   ├── eagle3_target_model.py # Eagle3TargetModel (ABC), HFTargetModel
│   │   └── target_utils.py  #     Hidden state utilities
│   └── ops/                 #   Custom operations
│       ├── loss.py          #     compiled_forward_kl_loss
│       ├── loss_mask.py     #     Loss mask computation
│       └── flex_attention.py #    FlexAttention utilities
├── training/                # Training actors and utilities
│   ├── trainer_actor.py     #   TrainerActor (Ray actor, wraps Eagle3Trainer)
│   ├── trainer.py           #   Trainer (abstract base: device mesh, data fetcher, loop)
│   ├── eagle3_trainer.py    #   Eagle3Trainer (model init, forward/backward, metrics)
│   ├── fsdp.py              #   FSDP2 helpers (apply_fsdp2, fsdp2_load_full_state_dict)
│   ├── data_fetcher.py      #   MooncakeDataFetcher
│   ├── checkpoint.py        #   Checkpoint save/load
│   ├── optimizer.py         #   Optimizer construction (BF16Optimizer)
│   └── lr_scheduler.py      #   LR scheduling
├── transfer/                # Distributed tensor transfer
│   └── mooncake/            #   Mooncake integration
│       ├── store.py         #     MooncakeHiddenStateStore (base)
│       ├── eagle_store.py   #     EagleMooncakeStore
│       ├── buffers.py       #     HostBufferPool, GPUReceiveBuffer
│       ├── helpers.py       #     Buffer size calculation
│       └── utils.py         #     Mooncake utility helpers
├── data/                    # Data pipeline
│   ├── dataset.py           #   load_conversation_dataset()
│   ├── parse.py             #   Chat format parsers (GeneralParser, etc.)
│   ├── preprocessing.py     #   Tokenization and chat templates
│   ├── template.py          #   Chat template handling
│   └── utils.py             #   Loss mask packing/unpacking
├── utils/                   # Shared utilities
│   ├── distributed.py       #   Device mesh setup, TP/DP primitives (get_tp_group, get_tp_device_mesh)
│   ├── env.py               #   Ray actor env-var forwarding (get_torchspec_env_vars)
│   ├── logging.py           #   Unified logger
│   ├── memory.py            #   Tensor byte estimation
│   ├── profiling.py         #   PyTorch profiler utilities
│   ├── types.py             #   InferenceInput, InferenceOutput
│   ├── wandb.py             #   Weights & Biases integration
│   ├── processing.py        #   Data processing utilities
│   ├── tensor.py            #   Tensor utilities
│   ├── train_dump.py        #   Training debug dump utilities
│   └── misc.py              #   Miscellaneous helpers
└── train_entry.py           # Main entry point
```

## Core Components

### 1. Draft Model (`torchspec/models/draft/`)

A lightweight transformer initialized from the target model's architecture:

- **`auto.py`**: `AutoEagle3DraftModel` - Factory that dispatches by model type
- **`base.py`**: `Eagle3DraftModel` - Abstract base defining the interface (`embed_input_ids`, `backbone`, `compute_logits`)
- **`llama3_eagle.py`**: `LlamaForCausalLMEagle3` - Llama-based draft model with:
  - Shared embedding layer (from target)
  - Reduced number of layers
  - Hidden state projection from target model
  - Token-to-draft vocabulary mapping (`t2d`)

### 2. Target Model (`torchspec/models/target/`)

Abstract interface for running the target model during inference:

- **`eagle3_target_model.py`**: `Eagle3TargetModel` (ABC) with concrete implementation:
  - `HFTargetModel` - HuggingFace-based target model backend
- **`target_utils.py`**: Hidden state layer selection utilities

The target model extracts:
- **Hidden states** from configurable layers (`aux_hidden_states_layers`)
- **Logits** for computing soft labels (KL divergence targets)

### 3. Async Training Controller (`torchspec/controller/training_controller.py`)

Central orchestrator (Ray actor) managing the async pipeline:

```python
@ray.remote
class AsyncTrainingController:
    # Data flow buffers
    prompt_buffer: deque[InferenceInput]   # Samples waiting for inference
    sample_pool: deque[InferenceOutput]    # Completed inferences (mooncake keys)
    train_queues: List[Queue]              # Per-DP-rank Ray queues

    # Key methods
    def add_dataset(dataset)               # Load prompts into buffer
    def get_prompts(n)                     # Inference manager fetches prompts
    def push_inference_results(results)    # Store completed inference keys
    def try_dispatch_batch()               # Send to train queues when pool is full
```

The controller only manages metadata and Mooncake keys, never actual tensor data. It tracks exact bytes in the sample pool for Mooncake backpressure control.

### 4. Async Inference Manager (`torchspec/controller/inference_manager.py`)

Self-regulating inference manager (Ray actor) that dispatches to `HFEngine` / `SglEngine` Ray actors with load balancing.

Includes Mooncake backpressure: pauses generation when `sample_pool` exceeds capacity, resuming when training catches up.

### 5. Inference Engines (`torchspec/inference/engine/`)

- **`base.py`**: `InferenceEngine` - Abstract base class defining the unified engine interface
- **`hf_runner.py`**: `HFRunner` - Core inference logic that runs target model, extracts hidden states, and stores tensors in Mooncake
- **`hf_engine.py`**: `HFEngine` - Ray actor wrapper around `HFRunner` (inherits `RayActor`)
- **`sgl_engine.py`**: `SglEngine` - Ray actor wrapper for SGLang-based inference (inherits `RayActor`)

Factory function in `factory.py`: `create_inference_engines()`

### 6. Training (`torchspec/training/`)

The training side is split across three layers:

- **`trainer_actor.py`**: `TrainerActor` — the Ray actor. Owns the distributed process group (`dist.init_process_group`), holds a `Eagle3Trainer` instance, and exposes the remote API (`init`, `train_from_queue`, `save_model`, `set_vocab_buffers`, etc.)
- **`trainer.py`**: `Trainer` — abstract base class. Sets up device mesh, `MooncakeDataFetcher`, checkpointing, profiling, and the training/eval loop skeleton
- **`eagle3_trainer.py`**: `Eagle3Trainer(Trainer)` — Eagle3-specific logic: initialises `Eagle3Model` with the draft model under FSDP2, runs the forward/backward, and aggregates metrics
- **`fsdp.py`**: FSDP2 helpers (`apply_fsdp2`, `fsdp2_load_full_state_dict`, `init_empty_weights`)

### 7. Mooncake Integration (`torchspec/transfer/mooncake/`)

Distributed tensor transfer for multi-node training:

- **`store.py`**: `MooncakeHiddenStateStore` - Base class with RDMA buffer management
- **`eagle_store.py`**: `EagleMooncakeStore` - Eagle3-specific wrapper with:
  - Zero-copy `batch_put_from` for tensor storage
  - Force deletion via `batch_remove(force=True)`
  - Lazy tensor retrieval interface
- **`buffers.py`**: `HostBufferPool` (pre-allocated host buffers), `GPUReceiveBuffer` (GPU Direct RDMA)
- **`helpers.py`**: Buffer size calculation and Mooncake master process management

## Training Flow

```
1. DATASET LOADING (train_entry.py)
   ├── Parse YAML config with OmegaConf
   ├── Preprocess prompts with chat template
   ├── Tokenize to input_ids + loss_mask
   ├── Auto-generate vocab mapping if needed
   └── Add to controller's prompt_buffer

2. INFERENCE (Inference GPUs, async)
   ├── InferenceManager fetches prompts from controller
   ├── Dispatches to HFEngine / SglEngine Ray actors
   ├── Target model produces hidden_states + logits
   ├── Store tensors in EagleMooncakeStore
   ├── Return mooncake keys to controller
   └── Backpressure: pause if sample_pool exceeds limit

3. TRAINING (Training GPUs, synchronous per step)
   ├── Controller dispatches dispatch_batch_size samples to per-DP-rank queues
   ├── MooncakeDataFetcher retrieves tensors from Mooncake
   ├── Forward pass through Eagle3Model (TTT loop over ttt_length positions)
   ├── Compute loss (forward KL divergence against target distribution)
   ├── Backward pass with gradient accumulation
   ├── Optimizer step, LR scheduling
   └── Periodic checkpointing
```

## Configuration System (`torchspec/config/`)

Hierarchical YAML configs powered by OmegaConf, with 9 typed dataclass sections:

```yaml
dataset:
  chat_template: llama3
  train_data_path: /path/to/data
  max_seq_length: 8192

model:
  target_model_path: Qwen/Qwen3-8B
  target_model_backend: sglang    # or "remote"
  draft_model_config: /path/to/config.json

training:
  num_epochs: 1
  micro_batch_size: 2
  learning_rate: 1e-4
  ttt_length: 7                   # Speculative depth
  train_backend: fsdp
  fsdp_strategy: REPLICATE

inference:
  inference_engine_type: hf       # or "sgl"
  inference_batch_size: 1
  inference_num_gpus: 4
  sglang:                         # nested under inference
    tp_size: 8
    extra_args:                   # power-user passthrough to sgl.Engine
      attention_backend: flashinfer

mooncake:
  master_addr: null
  protocol: rdma                  # or "tcp"

logging:
  report_to: wandb
  wandb_project: torchspec

debug:
  use_pytorch_profiler: false
```

Configs support multi-file merging and CLI overrides:
```bash
python train.py --config base.yaml --config experiment.yaml training.learning_rate=1e-5
```

## Module Reference

### Models

| Module | Purpose |
|--------|---------|
| `torchspec/models/eagle3.py` | `Eagle3Model` - Eagle3 forward pass and loss computation |
| `torchspec/models/ops/loss.py` | `compiled_forward_kl_loss` - Forward KL loss |
| `torchspec/models/ops/loss_mask.py` | Loss mask computation utilities |
| `torchspec/models/ops/flex_attention.py` | FlexAttention utilities |
| `torchspec/models/draft/auto.py` | `AutoEagle3DraftModel` factory |
| `torchspec/models/draft/base.py` | `Eagle3DraftModel` abstract base |
| `torchspec/models/draft/llama3_eagle.py` | `LlamaForCausalLMEagle3` implementation |
| `torchspec/models/target/eagle3_target_model.py` | `Eagle3TargetModel` ABC + `HFTargetModel` implementation |
| `torchspec/models/target/target_utils.py` | Hidden state layer selection utilities |

### Ray Infrastructure

| Module | Purpose |
|--------|---------|
| `torchspec/ray/ray_actor.py` | `RayActor` base class (GPU setup, IP/port utils, master addr negotiation) |
| `torchspec/ray/train_group.py` | `RayTrainGroup` - Manages a group of training actors |
| `torchspec/ray/placement_group.py` | Placement group creation, GPU resource waiting, `create_placement_groups()`, `create_train_group()` |

### Controller

| Module | Purpose |
|--------|---------|
| `torchspec/controller/training_controller.py` | `AsyncTrainingController` - Pipeline orchestration |
| `torchspec/controller/inference_manager.py` | `AsyncInferenceManager` - Inference dispatch and backpressure |
| `torchspec/controller/loop.py` | `run_training_loop()` - Main training loop |
| `torchspec/controller/setup.py` | `build_mooncake_config`, `setup_async_training_with_engines`, `auto_calculate_training_steps` |

### Inference

| Module | Purpose |
|--------|---------|
| `torchspec/inference/factory.py` | `create_inference_engines()` - Engine creation with placement groups |
| `torchspec/inference/engine/base.py` | `InferenceEngine` abstract base class |
| `torchspec/inference/engine/hf_runner.py` | `HFRunner` core inference logic |
| `torchspec/inference/engine/hf_engine.py` | `HFEngine` Ray actor wrapper (inherits `RayActor`) |
| `torchspec/inference/engine/sgl_engine.py` | `SglEngine` Ray actor wrapper (inherits `RayActor`) |

### Training

| Module | Purpose |
|--------|-------|
| `torchspec/training/trainer_actor.py` | `TrainerActor` - Ray actor wrapper; owns distributed process group |
| `torchspec/training/trainer.py` | `Trainer` - Abstract base (device mesh, data fetcher, loop skeleton) |
| `torchspec/training/eagle3_trainer.py` | `Eagle3Trainer` - Eagle3 model init, forward/backward, metric aggregation |
| `torchspec/training/fsdp.py` | `apply_fsdp2`, `fsdp2_load_full_state_dict`, `init_empty_weights` |
| `torchspec/training/data_fetcher.py` | `MooncakeDataFetcher` - Queue-based data retrieval |
| `torchspec/training/checkpoint.py` | Checkpoint save/load |
| `torchspec/training/optimizer.py` | `BF16Optimizer` construction |
| `torchspec/training/lr_scheduler.py` | LR scheduling |

### Data Pipeline

| Module | Purpose |
|--------|---------|
| `torchspec/data/dataset.py` | `load_conversation_dataset()` with format detection |
| `torchspec/data/parse.py` | Chat format parsers (`GeneralParser`, etc.) |
| `torchspec/data/preprocessing.py` | Tokenization, chat templates, loss masks |
| `torchspec/data/template.py` | Chat template handling |
| `torchspec/data/utils.py` | Loss mask packing/unpacking |

### Configuration

| Module | Purpose |
|--------|-------|
| `torchspec/config/train_config.py` | `Config` root + 7 typed dataclass sections (`DatasetConfig`, `DebugConfig`, `InferenceConfig`, `LoggingConfig`, `ModelConfig`, `TrainingConfig`, plus `mooncake: dict`) |
| `torchspec/config/inference_config.py` | `InferenceConfig`, `SGLangConfig` (essential fields + `extra_args` passthrough), `HFInferenceConfig` |
| `torchspec/config/mooncake_config.py` | `MooncakeConfig` with env-var support and `from_flat_args()` |
| `torchspec/config/utils.py` | Config loading helpers, `generate_draft_model_config` |

### Infrastructure

| Module | Purpose |
|--------|-------|
| `torchspec/transfer/mooncake/` | Mooncake tensor transfer (RDMA/TCP, buffer pools, force delete) |
| `torchspec/utils/distributed.py` | Device mesh setup, TP/DP primitives (`get_tp_group`, `get_tp_device_mesh`) |
| `torchspec/utils/env.py` | Ray actor env-var forwarding (`get_torchspec_env_vars`) |
| `torchspec/utils/logging.py` | Unified logger |
| `torchspec/utils/profiling.py` | PyTorch profiler utilities |
| `torchspec/utils/types.py` | `InferenceInput`, `InferenceOutput` |
| `torchspec/utils/memory.py` | Tensor byte estimation |
| `torchspec/utils/wandb.py` | Weights & Biases integration |
| `torchspec/train_entry.py` | Main entry point (config parsing, Ray setup, launch) |
