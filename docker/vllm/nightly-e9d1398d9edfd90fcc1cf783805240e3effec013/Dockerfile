FROM vllm/vllm-openai:nightly-e9d1398d9edfd90fcc1cf783805240e3effec013@sha256:95bed119f39e2414973cf7224df31c9a3ed213566138722346d7bd360edf842b

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
      vllm/model_executor/models/qwen3_next.py \
      vllm/models/kimi_k3/nvidia/model.py \
      vllm/v1/core/sched/async_scheduler.py \
      vllm/v1/core/sched/scheduler.py \
      vllm/v1/engine/input_processor.py \
      vllm/v1/outputs.py \
      vllm/v1/spec_decode/extract_hidden_states.py \
      vllm/v1/worker/gpu_model_runner.py && \
    rm -rf /tmp/patches

# Keep the CUDA-13 Mooncake package shipped by the base image. Installing the
# generic PyPI wheel here can replace it with a CUDA-12 binary.
COPY . /root/torchspec
RUN cd /root/torchspec && pip install --no-cache-dir -e .

WORKDIR /root/torchspec
