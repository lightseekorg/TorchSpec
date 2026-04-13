ARG SGLANG_IMAGE_TAG=v0.5.10.post1-cu129-amd64
FROM lmsysorg/sglang:${SGLANG_IMAGE_TAG} AS sglang


WORKDIR /root/

RUN apt-get update && \
    apt-get install -y --no-install-recommends nvtop rsync dnsutils && \
    rm -rf /var/lib/apt/lists/*

ARG SGLANG_COMMIT=94f03a39dbd39edfc2b118b5357bbbadaaa9ad28
COPY patches/sglang/v0.5.10.post1/sglang.patch /sgl-workspace/
RUN rm -rf /sgl-workspace/sglang && \
    git clone --depth=1 --branch main https://github.com/sgl-project/sglang.git /sgl-workspace/sglang && \
    cd /sgl-workspace/sglang && \
    git fetch --depth=1 origin $SGLANG_COMMIT && \
    git checkout $SGLANG_COMMIT && \
    git apply /sgl-workspace/sglang.patch && \
    rm /sgl-workspace/sglang.patch && \
    cd python && pip install -e .

# Explicitly install flashinfer-jit-cache because it is not directly depended by sglang.
RUN pip install flashinfer-jit-cache==0.6.7.post3 --index-url https://flashinfer.ai/whl/cu129

COPY . /root/torchspec
RUN pip uninstall nvidia-cutlass-dsl nvidia-cutlass-dsl-libs-base -y 2>/dev/null; \
    cd /root/torchspec && pip install --no-cache-dir -e ".[fa]"

RUN chmod 755 /usr/local/lib/python3.12/dist-packages/mooncake/mooncake_master
# Avoid mooncake modifying the binary permissions.
RUN sed -i 's/os.chmod(bin_path, 0o755)/pass/' /usr/local/lib/python3.12/dist-packages/mooncake/cli.py
WORKDIR /root/torchspec
