FROM vllm/vllm-openai:nightly-e9d1398d9edfd90fcc1cf783805240e3effec013@sha256:95bed119f39e2414973cf7224df31c9a3ed213566138722346d7bd360edf842b

ENV VLLM_USE_V2_MODEL_RUNNER=1

ARG TORCHSPEC_SOURCE_REV=unknown
ARG VLLM_SOURCE_REV=e9d1398d9edfd90fcc1cf783805240e3effec013

LABEL org.opencontainers.image.title="TorchSpec vLLM PP hidden-state image" \
      org.opencontainers.image.description="vLLM nightly with TorchSpec pipeline-parallel extract-hidden-states support" \
      org.opencontainers.image.base.name="vllm/vllm-openai:nightly-e9d1398d9edfd90fcc1cf783805240e3effec013" \
      org.opencontainers.image.base.digest="sha256:95bed119f39e2414973cf7224df31c9a3ed213566138722346d7bd360edf842b" \
      org.opencontainers.image.vllm.revision="${VLLM_SOURCE_REV}" \
      org.opencontainers.image.torchspec.revision="${TORCHSPEC_SOURCE_REV}"

WORKDIR /root/

RUN apt-get update && \
    apt-get install -y --no-install-recommends nvtop rsync dnsutils && \
    rm -rf /var/lib/apt/lists/*

COPY patches/vllm/nightly-e9d1398d9edfd90fcc1cf783805240e3effec013/series \
     patches/vllm/nightly-e9d1398d9edfd90fcc1cf783805240e3effec013/*.patch \
     /tmp/patches/
RUN cd /usr/local/lib/python3.12/dist-packages && \
    while read -r p; do \
      case "$p" in ''|'#'*) continue;; esac; \
      echo "applying $p"; \
      patch -p1 --batch --forward --fuzz=0 < "/tmp/patches/$p" || exit 1; \
    done < /tmp/patches/series && \
    python3 -m compileall -q \
      vllm/model_executor/models/extract_hidden_states.py \
      vllm/model_executor/models/qwen2.py \
      vllm/models/kimi_k3/nvidia/model.py \
      vllm/v1/core/sched/async_scheduler.py \
      vllm/v1/core/sched/scheduler.py \
      vllm/v1/engine/input_processor.py \
      vllm/v1/outputs.py \
      vllm/v1/spec_decode/extract_hidden_states.py \
      vllm/v1/worker/gpu/model_runner.py \
      vllm/v1/worker/gpu/spec_decode/extract_hidden_states.py \
      vllm/v1/worker/gpu_model_runner.py && \
    rm -rf /tmp/patches

# Do not install TorchSpec's `fa` extra in this image. flash-attn-4 requires
# apache-tvm-ffi>=0.1.12, while the pinned vLLM, TileLang, and TokenSpeed stack
# requires apache-tvm-ffi==0.1.11. Routing-validation configs use
# flex_attention instead.
COPY . /root/torchspec
RUN cd /root/torchspec && pip install --no-cache-dir -e .

# TorchSpec's generic dependency can install the CUDA 12 Mooncake wheel over
# the base package. Replace it last and verify the binary left in the image.
RUN python3 -m pip uninstall -y \
      mooncake-transfer-engine mooncake-transfer-engine-cuda13 && \
    python3 -m pip install --no-cache-dir --no-deps \
      mooncake-transfer-engine-cuda13==0.3.12.post1 && \
    ldd /usr/local/lib/python3.12/dist-packages/mooncake/mooncake_master \
      | grep -q 'libcudart.so.13'

WORKDIR /root/torchspec
