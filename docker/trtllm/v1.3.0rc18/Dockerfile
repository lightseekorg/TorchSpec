ARG TRTLLM_IMAGE=nvcr.io/nvidia/tensorrt-llm/release:1.3.0rc18
FROM ${TRTLLM_IMAGE} AS trtllm

WORKDIR /root/

RUN apt-get update && \
    apt-get install -y --no-install-recommends nvtop rsync dnsutils && \
    rm -rf /var/lib/apt/lists/*

# Patch the installed tensorrt_llm package so SaveHiddenStates mode streams
# captured hidden states to Mooncake instead of writing .pt files to disk.
# tensorrt_llm lives in dist-packages (python 3.12 in this release image, same
# location the vLLM image uses). Do NOT import tensorrt_llm at build time — its
# package init loads libcuda.so.1, which is absent until container runtime.
# Apply with -p1 to strip the diff's a/ prefix.
COPY patches/trtllm/v1.3.0rc18/*.patch /tmp/patches/
RUN cd /usr/local/lib/python3.12/dist-packages && \
    for p in /tmp/patches/*.patch; do patch -p1 < "$p"; done && \
    rm -rf /tmp/patches

COPY . /root/torchspec
RUN cd /root/torchspec && pip install --no-cache-dir -e ".[fa]"

# TorchSpec's dependency pulls the generic CUDA 12 Mooncake wheel. Replace it
# with the CUDA 13 wheel that matches this image family.
RUN pip uninstall -y mooncake-transfer-engine || true && \
    pip install --no-cache-dir --no-deps --force-reinstall \
        mooncake-transfer-engine-cuda13==0.3.11.post1

RUN chmod 755 /usr/local/lib/python3.12/dist-packages/mooncake/mooncake_master || true
RUN if [ -f /usr/local/lib/python3.12/dist-packages/mooncake/cli.py ]; then \
      sed -i 's/os.chmod(bin_path, 0o755)/pass/' /usr/local/lib/python3.12/dist-packages/mooncake/cli.py; \
    fi

WORKDIR /root/torchspec
