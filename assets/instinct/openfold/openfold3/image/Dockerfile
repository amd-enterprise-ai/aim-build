# Copyright © Advanced Micro Devices, Inc., or its affiliates.
#
# SPDX-License-Identifier: MIT

# Layer 1 (specialized base) in the OpenFold3 image stack:
#   Layer 0: Upstream vendor image (pinned in the FROM below)
#   Layer 1: This image — OF3 Python deps, OpenFold3 source at a pinned ref,
#            and the BentoML serving wrapper (src/service.py + src/runner.py)
#   Layer 2: AIM base (aim-runtime, entrypoint, engine config)
#   Layer 3: Model image (profiles, AIM_ID)
#
# The Layer 1 build (base-image-pipeline.yaml, post-#1087) runs a plain
# `docker build` with no build-args, so the upstream ref is pinned here
# directly rather than injected via ARG.

FROM docker.io/rocm/pytorch:rocm7.2_ubuntu24.04_py3.12_pytorch_release_2.10.0

SHELL ["/bin/bash", "-euo", "pipefail", "-c"]

ARG OPENFOLD3_REPO=https://github.com/aqlaboratory/openfold-3.git
ARG OPENFOLD3_REF=0.4.1

# System packages
RUN apt-get update && apt-get install -y --no-install-recommends \
    wget \
    git \
    libopenmpi-dev \
    libaio-dev \
    build-essential \
    ninja-build \
    && rm -rf /var/lib/apt/lists/*

# Install OF3 Python deps into the base image's Python (which has ROCm torch).
# Must happen BEFORE miniforge; otherwise conda's pip shadows system pip.
# Excludes NVIDIA-specific packages from OF3's pyproject.toml (nvidia-cutlass, cuda-python).
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install \
    pytorch-lightning \
    deepspeed \
    biopython \
    numpy \
    pandas \
    PyYAML \
    requests \
    scipy \
    tqdm \
    typing-extensions \
    wandb \
    modelcif \
    ml-collections \
    biotite==1.2.0 \
    memory_profiler \
    func_timeout \
    boto3 \
    lmdb \
    ijson \
    awscrt \
    pdbeccdutils \
    gemmi \
    kalign-python \
    rdkit

# Verify torch is still the ROCm build (pip didn't replace it). Doubles as a
# canary when the engine-level upstream image tag is bumped.
RUN python3 -c "import torch; print('torch:', torch.__file__, 'hip:', torch.version.hip)"

# Install miniforge (only for bioinformatics CLI tools)
RUN wget -q -P /tmp \
    "https://github.com/conda-forge/miniforge/releases/download/25.11.0-1/Miniforge3-Linux-x86_64.sh" \
    && bash /tmp/Miniforge3-Linux-x86_64.sh -b -p /opt/conda \
    && rm /tmp/Miniforge3-Linux-x86_64.sh

# Install bioinformatics CLI tools via conda (binary tools only; Python packages
# go via pip above so system Python keeps torch importable)
RUN /opt/conda/bin/mamba install -n base -c conda-forge -c bioconda --yes \
    mmseqs2 \
    hmmer \
    hhsuite \
    kalign2 \
    && /opt/conda/bin/conda clean --all --yes

ENV ROCM_PATH=/opt/rocm
ENV PYTORCH_ROCM_ARCH="gfx90a;gfx942"
ENV CONDA_PREFIX=/opt/conda
ENV BENTOML_PORT=8000
ENV AIM_ENGINE=bentoml

# Clone OpenFold3 at pinned ref and install without NVIDIA-specific deps
RUN git clone --depth 1 --branch ${OPENFOLD3_REF} ${OPENFOLD3_REPO} /opt/openfold3 \
 || (git clone ${OPENFOLD3_REPO} /opt/openfold3 \
     && cd /opt/openfold3 && git checkout ${OPENFOLD3_REF})

# Apply ROCm torch.compile (Inductor) source patches against OF3 0.4.1.
# --check runs first so a stale patch fails the build with the conflicting
# hunks named instead of half-patching the tree.
COPY --link patches /workspace/patches
RUN cd /opt/openfold3 \
 && git apply --check /workspace/patches/of3_torch_compile_champion.patch \
 && git apply         /workspace/patches/of3_torch_compile_champion.patch

WORKDIR /opt/openfold3
RUN pip install --no-deps -e .

# BentoML service wrapper. Copy only the package files so unrelated context
# (README.md) does not invalidate the layer cache or leak into the image.
# pyproject.toml uses a src layout; src/ holds service.py + runner.py + harness.py.
WORKDIR /workspace/model
COPY --link pyproject.toml /workspace/model/pyproject.toml
COPY --link src /workspace/model/src

# Benchmark payloads for `aim-runtime benchmark` (21 NVIDIA-reference PDB cases).
# The harness in src/harness.py reads /workspace/model/benchmarks/<pdb>/<pdb>.json.
COPY --link examples/benchmark_sequences /workspace/model/benchmarks

COPY --link bentofile.yaml /workspace/model/bentofile.yaml

RUN --mount=type=cache,target=/root/.cache/pip \
    pip install -e .

RUN pip install --no-cache-dir amdsmi

# Conda at the END of PATH so system python3/pip3 (which have ROCm torch) win.
# Bioinformatics binaries (mmseqs2, hmmer, hhblits, kalign) are still found
# because they only exist under /opt/conda/bin.
ENV PATH=$PATH:/opt/conda/bin
