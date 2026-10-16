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

# Canary: ensure the base image's ROCm torch survived the dependency install
# (build fails if hip is None — a CPU torch was silently installed over the base).
RUN python3 -c "import torch, sys; print('torch:', torch.__version__, 'hip:', torch.version.hip); sys.exit(0 if torch.version.hip is not None else 'ROCm torch not installed')"

ENV BENTOML_PORT=8000
ENV AIM_ENGINE=bentoml

# Fetch the MONAI bundle configs at build time so the service does not need
# network access for non-weight assets. The service reads the bundle from
# /workspace/model/bundles/spleen_deepedit_annotation (SPLEEN_DEEPEDIT_BUNDLE_DIR).
#
# Model weights are NOT baked into this image. monai.bundle download pulls
# configs/ and models/ together, so the checkpoint is removed in the same layer
# and is never committed. It is fetched at service startup by
# ensure_checkpoint() (see pipeline.py) from MONAI/spleen_deepedit_annotation.
# The glob is deliberate: name the files and the list drifts -- the pancreas and
# spleen bundles also ship a model.ts that no service reads.
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install fire==0.7.1 \
 && python3 -m monai.bundle download "spleen_deepedit_annotation" \
        --bundle_dir /workspace/model/bundles \
 && rm -rf /workspace/model/bundles/spleen_deepedit_annotation/models/* \
 && pip uninstall -y fire termcolor

COPY --link pyproject.toml /workspace/model/pyproject.toml
COPY --link src /workspace/model/src
COPY --link bentofile.yaml /workspace/model/bentofile.yaml

RUN --mount=type=cache,target=/root/.cache/pip \
    pip install -e .
