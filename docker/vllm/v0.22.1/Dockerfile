FROM vllm/vllm-openai:v0.22.1@sha256:953d3a06d5e64ab582985cd7401289d3abf2a2c14ef2158e9a84313daeec77d7 AS base

# Compile against the runtime base for matching architecture and libc.
FROM base AS torchspec-builder
COPY . /tmp/torchspec
RUN bash /tmp/torchspec/bindings/smg-tokenizer/build-wheel.sh

FROM base AS runtime

WORKDIR /root/

RUN apt-get update && \
    apt-get install -y --no-install-recommends nvtop rsync dnsutils && \
    rm -rf /var/lib/apt/lists/*

COPY patches/vllm/v0.22.1/*.patch /tmp/patches/
RUN cd /usr/local/lib/python3.12/dist-packages && \
    for p in /tmp/patches/*.patch; do patch -p1 < "$p"; done && \
    rm -rf /tmp/patches

# NOTE: the base image ships its own CUDA-13-linked mooncake-transfer-engine
# (0.3.10.post2), which satisfies torchspec's >=0.3.10.post1 floor. Do NOT
# upgrade or reinstall mooncake from PyPI here: the generic wheel links
# libcudart.so.12 (broken on this CUDA 13 image) and the cuda13 variant's
# aarch64 wheel needs glibc >= 2.39 (this image is Ubuntu 22.04 / 2.35).
# Install this checkout as one wheel, then resolve its Python dependencies.
COPY --from=torchspec-builder /opt/torchspec-wheels/ /tmp/torchspec-wheels/
RUN python3 -m pip install --no-cache-dir --no-deps --force-reinstall /tmp/torchspec-wheels/*.whl && \
    python3 -c "from torchspec_smg import Tokenizer, build_features; assert build_features == 'default'" && \
    rm -rf /tmp/torchspec-wheels

COPY . /root/torchspec
RUN python3 -m pip install --no-cache-dir "torchspec[fa]==$(python3 -c 'from importlib.metadata import version; print(version("torchspec"))')"

WORKDIR /root/torchspec
