# Copyright © Advanced Micro Devices, Inc., or its affiliates.
#
# SPDX-License-Identifier: MIT

# Layer 1 (specialized base): MONAI deps + bundle + BentoML serving wrapper.
# Layer 2 (aim-runtime/entrypoint) and Layer 3 (profiles, AIM_ID) are added by CI.

FROM docker.io/rocm/pytorch:rocm7.2_ubuntu24.04_py3.12_pytorch_release_2.9.1

SHELL ["/bin/bash", "-euo", "pipefail", "-c"]

WORKDIR /workspace/model

# monai is installed with --no-deps so it doesn't pull a CPU torch over the
# base image's ROCm torch; MONAI's own deps are pinned in requirements.txt.
COPY --link requirements.txt /workspace/model/requirements.txt
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install -r requirements.txt \
 && pip install --no-deps monai==1.5.2

RUN python3 -c "import torch, sys; print('torch:', torch.__version__, 'hip:', torch.version.hip); sys.exit(0 if torch.version.hip is not None else 'ROCm torch not installed')"

ENV BENTOML_PORT=8000
ENV AIM_ENGINE=bentoml

# Fetch the MONAI bundle at build time (configs + weights) rather than vendoring
# third-party content. The service reads it from
# /workspace/model/bundles/wholeBrainSeg_Large_UNEST_segmentation (BRAINSEG_BUNDLE_DIR).
RUN --mount=type=cache,target=/root/.cache/pip \
    python3 -m monai.bundle download "wholeBrainSeg_Large_UNEST_segmentation" \
        --bundle_dir /workspace/model/bundles

COPY --link pyproject.toml /workspace/model/pyproject.toml
COPY --link src /workspace/model/src
COPY --link bentofile.yaml /workspace/model/bentofile.yaml

RUN --mount=type=cache,target=/root/.cache/pip \
    pip install -e .
