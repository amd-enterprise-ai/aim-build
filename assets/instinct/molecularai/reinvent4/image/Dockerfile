# Copyright © Advanced Micro Devices, Inc., or its affiliates.
#
# SPDX-License-Identifier: MIT

# Specialized base image for REINVENT4 (molecular drug design).
# Layer 1 in the specialized image stack:
#   Layer 0: rocm/pytorch upstream vendor image (this FROM)
#   Layer 1: This image — REINVENT4 Python deps and source
#   Layer 2: AIM base (aim-runtime, entrypoint, engine config)
#   Layer 3: Model image (profiles, AIM_ID)
#
# The ROCm PyTorch base provides torch==2.9.1 built for ROCm 7.2 with
# Python 3.12. We must NOT upgrade the base image tag — pin is intentional
# per project conventions for this model.

FROM rocm/pytorch:rocm7.2_ubuntu24.04_py3.12_pytorch_release_2.9.1

SHELL ["/bin/bash", "-o", "pipefail", "-c"]

ARG REINVENT4_REPO=https://github.com/MolecularAI/REINVENT4
# Pinned to the commit of tag v4.8 (80a8d21a...; the tag is annotated, a2fa... is
# the tag object). Requires Python >=3.11 (base image provides 3.12).
ARG REINVENT4_REF=80a8d21aefd9c0d3ec806377522effb30cfca12a

# Verify torch is the ROCm 7.2 build before any pip operations that might
# replace it. Canary: fails the build if hip version does not contain "7.2".
RUN python3 -c "import sys; import torch; v=torch.version.hip; assert v and '7.2' in v, f'Expected ROCm 7.2, got hip={v!r}'; print('ROCm canary OK: hip='+v); sys.exit(0)"

# System packages needed for REINVENT4 and RDKit
RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    && rm -rf /var/lib/apt/lists/*

# Clone REINVENT4 at the pinned ref
RUN git clone ${REINVENT4_REPO} /workspace/reinvent4 \
 && cd /workspace/reinvent4 && git checkout ${REINVENT4_REF}

WORKDIR /workspace/reinvent4

# Remove torch and torchvision entries from pyproject.toml so the
# ROCm wheels already present in the base image are preserved. Installing
# with upstream version pins would replace the ROCm build with a CUDA build.
RUN sed -i '/torch==/d' pyproject.toml && \
    sed -i '/["\x27]torchvision/d' pyproject.toml

# Install REINVENT4 using pip directly (not install.py — that script
# swallows pip errors via sp.run without check=True). Use --no-build-
# isolation so the ROCm torch already in the image is visible during the
# build. The `all` extra (proprietary OpenEye toolkit + iSIM git dependency) is
# deliberately not installed (licence-driven V1); sampling never imports them.
# The trailing `reinvent --help` validates the entry point was created.
# requirements.txt is a constraint file whose pins come from the pip freeze of the image built from this
# branch (the versions left after the whole Dockerfile), not from upstream's uv.lock
# (stale at this commit: pandas 2.3.3 vs pyproject pandas>=3,<4). See the file header.
COPY requirements.txt /tmp/requirements.txt
RUN pip install --no-build-isolation -c /tmp/requirements.txt '.[chemprop2]' \
        --extra-index-url https://download.pytorch.org/whl/cpu && \
    reinvent --help > /dev/null

# Confirm torch is still the ROCm 7.2 build after all pip operations.
RUN python3 -c "import sys; import torch; v=torch.version.hip; assert v and '7.2' in v, f'ROCm torch was replaced: hip={v!r}'; print('ROCm post-install canary OK: hip='+v); sys.exit(0)"

# The prior model weights are NOT baked into the image. The service
# downloads them into $AIM_CACHE_PATH/reinvent4/priors/ on first start.

WORKDIR /workspace/model

# Install the BentoML serving stack.
# The rocm/pytorch base image does not include BentoML, so we install it
# here after all REINVENT4 dependencies so the service can be launched with
# `python -m bentoml serve` (or the installed `bentoml` entry-point which pip
# places on PATH at /usr/local/bin/bentoml).
# bentoml==1.4.39 matches the pin in the dinov2 sibling image (>=1.4.8 is
# needed for `--arg` / bentoml.use_arguments).
RUN pip install --no-cache-dir -c /tmp/requirements.txt "bentoml==1.4.39"

# `bentoml serve --working-dir /workspace/model` reads bentofile.yaml to find
# the service instance (service:REINVENT4Service).
COPY bentofile.yaml /workspace/model/bentofile.yaml
COPY src/ /workspace/model/src/

# Add /workspace/model/src to sys.path so the harness can import sibling modules.
RUN python3 -c "import sysconfig; print(sysconfig.get_path('purelib'))" \
    | xargs -I{} sh -c 'echo /workspace/model/src > {}/workspace-model.pth'

# Deliberately no BENTOML_PORT: it is a click envvar on `bentoml serve --port`,
# applied after the service config, so it would override the service's own
# `http={"port": ...}`. Setting it here would discard the port aim-runtime
# forwards as `--arg port=<AIM_PORT>` and pin every container to one port.
# src/service.py owns the port.

# Select the BentoML engine: aim-runtime defaults AIM_ENGINE to vllm.
ENV AIM_ENGINE=bentoml

# The shipped profiles are throughput profiles (metadata.metric: throughput) and
# aim-runtime defaults AIM_METRIC to latency, which would reject them as a metric
# mismatch. Select throughput so automatic profile selection finds them.
ENV AIM_METRIC=throughput
