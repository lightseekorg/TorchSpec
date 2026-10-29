ARG SGLANG_IMAGE=lmsysorg/sglang:v0.5.14-cu130
FROM ${SGLANG_IMAGE} AS base

# Compile against the runtime base for matching architecture and libc.
FROM base AS torchspec-builder
COPY . /tmp/torchspec
RUN bash /tmp/torchspec/bindings/smg-tokenizer/build-wheel.sh

FROM base AS sglang

WORKDIR /root/

COPY patches/sglang/v0.5.14/sglang.patch /sgl-workspace/
RUN cd /sgl-workspace/sglang && \
    git apply --3way /sgl-workspace/sglang.patch && \
    rm /sgl-workspace/sglang.patch && \
    cd python && pip install -e .

# Install this checkout as one wheel, then resolve its Python dependencies.
COPY --from=torchspec-builder /opt/torchspec-wheels/ /tmp/torchspec-wheels/
RUN python3 -m pip install --no-cache-dir --no-deps --force-reinstall /tmp/torchspec-wheels/*.whl && \
    python3 -c "from torchspec_smg import Tokenizer, build_features; assert build_features == 'default'" && \
    rm -rf /tmp/torchspec-wheels

COPY . /root/torchspec
RUN python3 -m pip install --no-cache-dir "torchspec[fa]==$(python3 -c 'from importlib.metadata import version; print(version("torchspec"))')"

# TorchSpec's dependency pulls the generic CUDA 12 Mooncake wheel. Replace it
# with the CUDA 13 wheel that matches this image family.
RUN pip uninstall -y mooncake-transfer-engine || true && \
    pip install --no-cache-dir --no-deps --force-reinstall \
        mooncake-transfer-engine-cuda13==0.3.11.post1

RUN chmod 755 /usr/local/lib/python3.12/dist-packages/mooncake/mooncake_master || true
RUN if [ -f /usr/local/lib/python3.12/dist-packages/mooncake/cli.py ]; then \
      sed -i 's/os.chmod(bin_path, 0o755)/pass/' /usr/local/lib/python3.12/dist-packages/mooncake/cli.py; \
    fi

RUN python3 -c "from sglang.srt.server_args import ServerArgs; from sglang.srt.speculative.spec_training_info import SpecTrainingInfo; assert 'enable_spec_training_mooncake' in ServerArgs.__dataclass_fields__; assert 'enable_aux_hidden_states' in ServerArgs.__dataclass_fields__"

WORKDIR /root/torchspec
