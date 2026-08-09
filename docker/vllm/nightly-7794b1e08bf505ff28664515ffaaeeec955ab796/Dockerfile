FROM vllm/vllm-openai:nightly-7794b1e08bf505ff28664515ffaaeeec955ab796

WORKDIR /root/

RUN apt-get update && \
    apt-get install -y --no-install-recommends nvtop rsync dnsutils && \
    rm -rf /var/lib/apt/lists/*

# Applied in the order given by `series` rather than by shell glob order, so the
# stack stays explicit as patches are added or renamed. See that file for what
# each patch is and whether it is an upstream candidate.
COPY patches/vllm/nightly-7794b1e08bf505ff28664515ffaaeeec955ab796/series \
     patches/vllm/nightly-7794b1e08bf505ff28664515ffaaeeec955ab796/*.patch \
     /tmp/patches/
RUN cd /usr/local/lib/python3.12/dist-packages && \
    while read -r p; do \
      case "$p" in ''|'#'*) continue;; esac; \
      echo "applying $p"; \
      patch -p1 --batch --forward < "/tmp/patches/$p" || exit 1; \
    done < /tmp/patches/series && \
    rm -rf /tmp/patches

# NOTE: the base image ships its own CUDA-13-linked mooncake-transfer-engine
# (0.3.10.post2 as of the v0.22.1 image), which satisfies torchspec's
# >=0.3.10.post1 floor. Do NOT upgrade or reinstall mooncake from PyPI here:
# the generic wheel links libcudart.so.12 (broken on this CUDA 13 image) and
# the cuda13 variant's aarch64 wheel needs glibc >= 2.39 (this image is
# Ubuntu 22.04 / 2.35).
COPY . /root/torchspec
# The `fa` extra is deliberately not installed here, unlike in the v0.22.1
# image. Its flash-attn-4 pins nvidia-cutlass-dsl==4.6.0.dev0 against this
# image's 4.6.0 and pulls apache-tvm-ffi past the version vLLM's flashinfer was
# built against, after which every worker aborts while profiling with
# `tvm::ffi::Error: TypeAttr __ffi_repr__ is already registered for type index
# 132`. Pinning the base versions instead does not help: the two are mutually
# exclusive, and the image is not pip-reproducible anyway (its torch pins
# nvidia-nccl-cu13==2.29.7 while the image ships 2.30.7, so any install of a
# torch dependant downgrades nccl). The extra only enables the optional
# flash_attention draft backend, whose import is already guarded and which
# training.attention_backend does not select by default.
RUN cd /root/torchspec && pip install --no-cache-dir -e .

WORKDIR /root/torchspec
