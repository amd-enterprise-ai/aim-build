# Copyright © Advanced Micro Devices, Inc., or its affiliates.
#
# SPDX-License-Identifier: MIT

# Specialized base image for DINOv2 ViT-L/14 image embeddings.
# Layer 1 in the specialized image stack:
#   Layer 0: ROCm PyTorch base image
#   Layer 1: This image — DINOv2 deps and BentoML service code
#   Layer 2: AIM base (aim-runtime, entrypoint, engine config)
#   Layer 3: Model image (profiles, AIM_ID)

ARG BASE_IMAGE=docker.io/rocm/pytorch:rocm7.2_ubuntu24.04_py3.12_pytorch_release_2.9.1
FROM ${BASE_IMAGE}

SHELL ["/bin/bash", "-euo", "pipefail", "-c"]

ARG DINOV2_REPO=https://github.com/facebookresearch/dinov2.git
ARG DINOV2_COMMIT=7764ea0f912e53c92e82eb78a2a1631e92725fc8

RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /workspace/model

COPY requirements.txt /workspace/model/requirements.txt
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install --no-cache-dir -r /workspace/model/requirements.txt

# Verify torch is still the ROCm build (pip didn't replace it). Doubles as a
# canary when the upstream base image tag is bumped.
RUN python3 -c "import torch, sys; print('torch:', torch.__version__, 'hip:', torch.version.hip); sys.exit(0 if torch.version.hip is not None else 'ROCm torch not installed')"

RUN git init /opt/dinov2 \
    && git -C /opt/dinov2 remote add origin "${DINOV2_REPO}" \
    && git -C /opt/dinov2 fetch --depth 1 origin "${DINOV2_COMMIT}" \
    && git -C /opt/dinov2 checkout --detach FETCH_HEAD \
    && test "$(git -C /opt/dinov2 rev-parse HEAD)" = "${DINOV2_COMMIT}" \
    && rm -rf /opt/dinov2/.git

# `bentoml serve --working-dir /workspace/model` reads bentofile.yaml to find
# the service instance (service:Dinov2Vitl14Service).
COPY bentofile.yaml /workspace/model/bentofile.yaml
COPY src/ /workspace/model/src/

# Add /workspace/model/src to sys.path so both BentoML and aim-runtime harness
# discovery can import sibling modules.
RUN python3 -c "import sysconfig; print(sysconfig.get_path('purelib'))" \
    | xargs -I{} sh -c 'echo /workspace/model/src > {}/workspace-model.pth'

ENV AIM_ENGINE=bentoml
ENV DINOV2_DISABLE_XFORMERS=true
ENV XFORMERS_DISABLED=1
ENV TORCH_HOME=/workspace/model-cache/torch

CMD []
