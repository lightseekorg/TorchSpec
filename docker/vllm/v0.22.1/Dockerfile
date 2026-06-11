FROM vllm/vllm-openai:v0.22.1

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
COPY . /root/torchspec
RUN cd /root/torchspec && pip install --no-cache-dir -e ".[fa]"

WORKDIR /root/torchspec
