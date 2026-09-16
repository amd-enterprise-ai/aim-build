# Copyright © Advanced Micro Devices, Inc., or its affiliates.
#
# SPDX-License-Identifier: MIT

# Layer 1 (specialized base) in the Swin UNETR image stack:
#   Layer 0: Upstream vendor image (pinned in the FROM below)
#   Layer 1: This image — MONAI + bundle Python deps, the MONAI Swin UNETR
#            bundle (configs/scripts), and the BentoML serving wrapper
#            (src/service.py + src/harness.py + src/monai_harness.py)
#   Layer 2: AIM base (aim-runtime, entrypoint, engine config)
#   Layer 3: Model image (profiles, AIM_ID)
#
# The Layer 1 build runs a plain `docker build` with no build-args, so the
# upstream ref is pinned here directly.

FROM docker.io/rocm/pytorch:rocm7.2_ubuntu24.04_py3.12_pytorch_release_2.9.1

SHELL ["/bin/bash", "-euo", "pipefail", "-c"]

WORKDIR /workspace/model

# MONAI bundle Python deps. monai itself is installed with --no-deps so it does
# not pull a CPU torch over the base image's ROCm torch.
COPY --link requirements.txt /workspace/model/requirements.txt
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install -r requirements.txt \
 && pip install --no-deps monai==1.5.2

# Canary: ensure the base image's ROCm torch survived the dependency install
# (doubles as a tripwire if the upstream FROM tag is bumped).
RUN python3 -c "import torch, sys; print('torch:', torch.__version__, 'hip:', torch.version.hip); sys.exit(0 if torch.version.hip is not None else 'ROCm torch not installed')"

ENV BENTOML_PORT=8000
ENV AIM_ENGINE=bentoml

# Fetch the MONAI bundle at build time (configs + weights) rather than vendoring
# third-party (Apache-2.0) content into this repo — mirrors how the openfold3
# image clones its upstream in the Dockerfile. The service reads the bundle from
# /workspace/model/bundles/swin_unetr_btcv_segmentation (SWINUNETR_BUNDLE_DIR).
RUN --mount=type=cache,target=/root/.cache/pip \
    python3 -m monai.bundle download "swin_unetr_btcv_segmentation" \
        --bundle_dir /workspace/model/bundles

# BentoML serving wrapper (src layout).
COPY --link pyproject.toml /workspace/model/pyproject.toml
COPY --link src /workspace/model/src
COPY --link bentofile.yaml /workspace/model/bentofile.yaml

RUN --mount=type=cache,target=/root/.cache/pip \
    pip install -e .
