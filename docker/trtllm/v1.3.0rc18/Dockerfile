ARG TRTLLM_IMAGE=nvcr.io/nvidia/tensorrt-llm/release:1.3.0rc18
FROM ${TRTLLM_IMAGE} AS base

# Compile against the runtime base for matching architecture and libc.
FROM base AS torchspec-builder
COPY . /tmp/torchspec
RUN bash /tmp/torchspec/bindings/smg-tokenizer/build-wheel.sh

FROM base AS trtllm

WORKDIR /root/

RUN apt-get update && \
    apt-get install -y --no-install-recommends nvtop rsync dnsutils && \
    rm -rf /var/lib/apt/lists/*

# Patch the installed tensorrt_llm (dist-packages) for Mooncake hidden-state
# capture. Don't import tensorrt_llm at build time to locate it — its __init__
# loads libcuda.so.1, absent until container runtime; patch the path directly.
COPY patches/trtllm/v1.3.0rc18/*.patch /tmp/patches/
RUN cd /usr/local/lib/python3.12/dist-packages && \
    for p in /tmp/patches/*.patch; do patch -p1 < "$p"; done && \
    rm -rf /tmp/patches

# Install this checkout as one wheel, then resolve its Python dependencies.
COPY --from=torchspec-builder /opt/torchspec-wheels/ /tmp/torchspec-wheels/
RUN python3 -m pip install --no-cache-dir --no-deps --force-reinstall /tmp/torchspec-wheels/*.whl && \
    python3 -c "from torchspec_smg import Tokenizer, build_features; assert build_features == 'default'" && \
    rm -rf /tmp/torchspec-wheels

COPY . /root/torchspec
RUN python3 -m pip install --no-cache-dir "torchspec[fa]==$(python3 -c 'from importlib.metadata import version; print(version("torchspec"))')"

RUN pip uninstall -y mooncake-transfer-engine || true && \
    pip install --no-cache-dir --no-deps --force-reinstall \
        mooncake-transfer-engine-cuda13==0.3.11.post1

RUN chmod 755 /usr/local/lib/python3.12/dist-packages/mooncake/mooncake_master || true
RUN if [ -f /usr/local/lib/python3.12/dist-packages/mooncake/cli.py ]; then \
      sed -i 's/os.chmod(bin_path, 0o755)/pass/' /usr/local/lib/python3.12/dist-packages/mooncake/cli.py; \
    fi

WORKDIR /root/torchspec
