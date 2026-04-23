ARG SGLANG_IMAGE_TAG=v0.5.8-cu129-amd64
FROM lmsysorg/sglang:${SGLANG_IMAGE_TAG} AS sglang


WORKDIR /root/

RUN apt-get update && \
    apt-get install -y --no-install-recommends nvtop rsync dnsutils && \
    rm -rf /var/lib/apt/lists/*

ARG SGLANG_COMMIT=0f2df9370a1de1b4fb11b071d39ab3ce2287a350
COPY patches/sglang/v0.5.8.post1/sglang.patch /sgl-workspace/
RUN rm -rf /sgl-workspace/sglang && \
    git clone --depth=1 --branch main https://github.com/sgl-project/sglang.git /sgl-workspace/sglang && \
    cd /sgl-workspace/sglang && \
    git fetch --depth=1 origin $SGLANG_COMMIT && \
    git checkout $SGLANG_COMMIT && \
    git apply /sgl-workspace/sglang.patch && \
    rm /sgl-workspace/sglang.patch && \
    cd python && pip install -e .

# Explicitly install flashinfer-jit-cache because it is not directly depended by sglang.
RUN pip install flashinfer-jit-cache==0.6.2 --index-url https://flashinfer.ai/whl/cu129

COPY . /root/torchspec
RUN cd /root/torchspec && pip install --no-cache-dir -e ".[fa]"

RUN chmod 755 /usr/local/lib/python3.12/dist-packages/mooncake/mooncake_master
# Avoid mooncake modifying the binary permissions.
RUN sed -i 's/os.chmod(bin_path, 0o755)/pass/' /usr/local/lib/python3.12/dist-packages/mooncake/cli.py
WORKDIR /root/torchspec
