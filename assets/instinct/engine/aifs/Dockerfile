# Copyright © Advanced Micro Devices, Inc., or its affiliates.
#
# SPDX-License-Identifier: MIT

# Engine image for ECMWF AIFS Single 2.0 and AIFS ENS 2.0: anemoi-inference with
# ROCm torch 2.7, torch-geometric and flash-attn built for gfx942, the BentoML
# serving code, and ten days of real ECMWF Open Data input. The two checkpoints
# need different anemoi-models/anemoi-graphs versions, kept in /opt/anemoi/<variant>
# and chosen by AIFS_VARIANT. Checkpoints are not in the image; they are read from
# AIM_CACHE_PATH or downloaded at startup.

ARG BASE_IMAGE=docker.io/rocm/pytorch:rocm7.2.4_ubuntu24.04_py3.12_pytorch_release_2.7.1

# Downloads the input fields and the regrid matrix; only the results are kept.
FROM ${BASE_IMAGE} AS inputs

# 40 six-hourly times, about 3.1 GB, plus perturbed ensemble members 1 and 2 at the
# two times each of inits 2026-10-02T00 and 2026-10-03T00 need, about 0.6 GB.
ARG AIFS_INPUT_START=2026-10-01T00
ARG AIFS_INPUT_END=2026-10-10T18
ARG AIFS_MEMBERS=1,2
ARG AIFS_MEMBER_TIMES=2026-10-01T18,2026-10-02T00,2026-10-02T18,2026-10-03T00
ENV PYTHONUNBUFFERED=1

COPY inputs/ /opt/aifs/fetch/
RUN pip install --no-cache-dir earthkit-regrid==0.5.1 && \
    python /opt/aifs/fetch/fetch_regrid_matrix.py /opt/aifs/regrid-db && \
    python /opt/aifs/fetch/fetch_open_data.py \
        --start "${AIFS_INPUT_START}" --end "${AIFS_INPUT_END}" \
        --outdir /opt/aifs/inputs --workers 8 && \
    python /opt/aifs/fetch/fetch_open_data.py \
        --times "${AIFS_MEMBER_TIMES}" --members "${AIFS_MEMBERS}" \
        --outdir /opt/aifs/inputs/members --workers 8

# Builds the flash-attn wheel from ROCm's port with the Composable Kernel backend,
# which supports the sliding-window attention AIFS uses. Tags older than 2.8 do
# not compile with the ROCm 7 compiler.
FROM ${BASE_IMAGE} AS flash-attn

# The commit of tag v2.8.4-cktile.
ARG FLASH_ATTN_COMMIT=8afc617aa7ebb72f70f659b03b9908c30920ad37
ARG GPU_ARCHS=gfx942
ARG MAX_JOBS=64

# `clone --branch` only accepts ref names, so the commit is fetched into an empty repo.
RUN pip install --no-cache-dir wheel==0.45.1 packaging==26.2 psutil==7.2.2 && \
    git init -q /tmp/flash-attention && \
    cd /tmp/flash-attention && \
    git fetch -q --depth 1 https://github.com/ROCm/flash-attention "${FLASH_ATTN_COMMIT}" && \
    git checkout -q FETCH_HEAD && \
    git submodule update --init --recursive && \
    git rev-parse HEAD > /tmp/flash-attention.commit && \
    { FLASH_ATTENTION_FORCE_BUILD=TRUE GPU_ARCHS="${GPU_ARCHS}" MAX_JOBS="${MAX_JOBS}" \
        pip wheel --no-build-isolation --no-deps -v -w /wheels . > /tmp/flash-attn-build.log 2>&1 \
      || { tail -n 200 /tmp/flash-attn-build.log; exit 1; }; } && \
    tail -n 5 /tmp/flash-attn-build.log && \
    ls /wheels/flash_attn-*.whl && \
    cp /tmp/flash-attention.commit /wheels/flash-attention.commit && \
    mkdir -p /wheels/licenses && \
    cp LICENSE /wheels/licenses/flash-attention.LICENSE && \
    cp AUTHORS /wheels/licenses/flash-attention.AUTHORS && \
    cp csrc/composable_kernel/LICENSE /wheels/licenses/composable_kernel.LICENSE

FROM ${BASE_IMAGE}

COPY requirements.txt /tmp/requirements.txt
RUN pip install --no-cache-dir -r /tmp/requirements.txt

# The anemoi-models/anemoi-graphs pair of each checkpoint; their dependencies are
# in requirements.txt.
COPY requirements-single.txt requirements-ens.txt /tmp/
RUN pip install --no-cache-dir --no-deps --target /opt/anemoi/single -r /tmp/requirements-single.txt && \
    pip install --no-cache-dir --no-deps --target /opt/anemoi/ens -r /tmp/requirements-ens.txt

COPY --from=flash-attn /wheels/ /tmp/wheels/
RUN pip install --no-cache-dir --no-deps /tmp/wheels/flash_attn-*.whl && \
    mkdir -p /opt/aifs && cp /tmp/wheels/flash-attention.commit /opt/aifs/flash-attention.commit && \
    cp -r /tmp/wheels/licenses /opt/aifs/licenses && \
    rm -rf /tmp/wheels

# Fail the build if pip swapped any part of the ROCm torch stack for a PyPI CUDA
# wheel, or if the AIFS stack does not import. Runs without a GPU.
RUN python -c "\
import importlib.metadata as md;\
stack = {'torch', 'torchvision', 'torchaudio', 'triton', 'pytorch-triton', 'pytorch-triton-rocm'};\
found = {(d.metadata['Name'] or '').lower(): d.version for d in md.distributions() if (d.metadata['Name'] or '').lower() in stack};\
bad = {k: v for k, v in found.items() if 'rocm' not in v};\
assert not bad, 'ROCm stack clobbered by a non-ROCm wheel: %r (full: %r)' % (bad, found);\
import torch, numpy;\
assert torch.__version__.startswith('2.7.'), torch.__version__;\
assert numpy.__version__ == '1.26.4', numpy.__version__;\
assert float(torch.from_numpy(numpy.arange(4.0)).sum()) == 6.0;\
import torch_geometric, flash_attn, flash_attn_2_cuda;\
from flash_attn.layers.rotary import RotaryEmbedding;\
import importlib.util;\
assert importlib.util.find_spec('anemoi.models') is None, 'anemoi-models must only be in /opt/anemoi/<variant>';\
from anemoi.inference.runners.simple import SimpleRunner;\
import earthkit.data, earthkit.regrid, bentoml;\
print('deps gate OK:', found, 'numpy', numpy.__version__, 'torch_geometric', torch_geometric.__version__, 'flash_attn', flash_attn.__version__, 'anemoi-inference', md.version('anemoi-inference'), 'bentoml', bentoml.__version__)"

# Ten days of step-0 Open Data fields, perturbed members under members/pfNN/, and
# the offline regrid matrix. ECMWF data under CC-BY-4.0; its attribution is in
# /workspace/NOTICE.md.
COPY --from=inputs /opt/aifs/inputs/ /workspace/inputs/
COPY --from=inputs /opt/aifs/regrid-db/ /opt/aifs/regrid-db/
COPY NOTICE.md /workspace/NOTICE.md
ENV AIM_INPUT_PATH=/workspace/inputs
ENV AIFS_REGRID_DB=/opt/aifs/regrid-db

# Serving code. The base image's src/ lands in the same directory, so the
# harness imports config and data_utils from here.
COPY src/ /workspace/model/src/

# bentofile.yaml at the working-dir root: the serve command passes no explicit
# target, so BentoML reads its `service:` import string to find the service class.
COPY bentofile.yaml /workspace/model/bentofile.yaml

WORKDIR /workspace/model
ENV PYTHONPATH=/workspace/model/src
ENV PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# Per variant: the side folder's requirements are met and anemoi imports from it.
RUN for v in single ens; do AIFS_VARIANT=$v python -m anemoi_variant --check || exit 1; done

# Clear the upstream image's interactive CMD; the serve ENTRYPOINT comes from
# the base.
CMD []
