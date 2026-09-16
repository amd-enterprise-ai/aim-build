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
# Pinned by commit rather than by tag: upstream can move or delete a tag, which
# would silently change the source, and patches/ is context-matched against this
# exact tree. OPENFOLD3_VERSION names the release that OPENFOLD3_SHA points to.
# It names the install (see the pip step) rather than selecting it, so keep
# the two in step.
ARG OPENFOLD3_SHA=c4771653c5d0a3ebb0b3af71b05efd64bc44ee86
ARG OPENFOLD3_VERSION=0.5.0

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
RUN python3 -c "import sys, torch; print('torch:', torch.__file__, 'hip:', torch.version.hip); sys.exit(0 if torch.version.hip is not None else 'ROCm torch not installed')"

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
# Deliberately no BENTOML_PORT: it is a click envvar on `bentoml serve --port`,
# which the CLI applies *after* the service config, so it silently overrides the
# service's own `http={"port": ...}`. Setting it here would discard the port
# aim-runtime forwards as `--arg port=<AIM_PORT>` and pin every container to one
# port, which collides on host-network nodes. src/service.py owns the port.
ENV AIM_ENGINE=bentoml

# Clone OpenFold3 at the pinned commit and install without NVIDIA-specific deps.
# `clone --branch` only accepts ref names, so the commit is fetched into an empty
# repo instead; --depth 1 still keeps this to a single-commit download.
RUN git init -q /opt/openfold3 \
 && cd /opt/openfold3 \
 && git fetch -q --depth 1 ${OPENFOLD3_REPO} ${OPENFOLD3_SHA} \
 && git checkout -q FETCH_HEAD

# Apply source patches against OF3 0.5.0:
#  - of3_torch_compile_champion: ROCm torch.compile (Inductor) recipe.
#  - of3_msa_server_deadline: gives the MSA client a wall clock. Upstream's
#    submit/status/download loops are `while True` and the RATELIMIT path
#    resubmits forever, so a throttled query outlives any serving timeout while
#    holding a worker slot. Also hoists error_count out of the one loop where
#    upstream still reset it each iteration, making the 5-retry cap dead code.
#    Drop once upstream bounds these loops.
#  - of3_template_no_fork: template preprocessing forks an mp.Pool (and an
#    mp.Manager) even for the default n_processes=1. BentoML runs the sync
#    /predict on an AnyIO worker thread, so that forks off a non-main thread
#    and the pool's join() never returns — the request 504s at the serving cap
#    and the worker stays wedged for good. Adds the `if workers > 1` guard the
#    module's batch functions already carry to the three classes that pool
#    unconditionally — TemplatePreprocessor is the served one, the other two
#    are patched for consistency. Drop once upstream stops forking for a
#    single worker.
# --check runs first so a stale patch fails the build with the conflicting
# hunks named instead of half-patching the tree.
COPY --link patches /workspace/patches
RUN cd /opt/openfold3 \
 && git apply --check /workspace/patches/of3_torch_compile_champion.patch \
 && git apply         /workspace/patches/of3_torch_compile_champion.patch \
 && git apply --check /workspace/patches/of3_msa_server_deadline.patch \
 && git apply         /workspace/patches/of3_msa_server_deadline.patch \
 && git apply --check /workspace/patches/of3_template_no_fork.patch \
 && git apply         /workspace/patches/of3_template_no_fork.patch

WORKDIR /opt/openfold3
# openfold3 takes its version from setuptools-scm, i.e. from `git describe`. A
# commit-pinned checkout carries no tag, so the install would otherwise be
# named from the project's fallback_version plus the git suffix — plausible
# but wrong. OF3 gates checkpoints on this version
# (entry_points/validator.py), so it reports the release rather than a guess.
RUN SETUPTOOLS_SCM_PRETEND_VERSION_FOR_OPENFOLD3=${OPENFOLD3_VERSION} pip install --no-deps -e .

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
