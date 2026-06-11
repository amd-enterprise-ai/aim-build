# Copyright © Advanced Micro Devices, Inc., or its affiliates.
#
# SPDX-License-Identifier: MIT

# Specialized base image for Boltz-2 (protein structure prediction).
# Layer 1 in the specialized image stack:
#   Layer 0: Upstream vendor image (this FROM)
#   Layer 1: This image — model-specific Python deps, BentoML service code
#   Layer 2: AIM base (aim-runtime, entrypoint, engine config)
#   Layer 3: Model image (profiles, AIM_ID)
#
# TODO: Replace this placeholder with the real Boltz-2 Dockerfile.

FROM python:3.12-slim

RUN echo "Specialized base image placeholder for Boltz-2"

WORKDIR /workspace/model

COPY src/ /workspace/model/src/

# Add /workspace/model/src to sys.path so the harness can import sibling modules.
RUN python -c "import sysconfig; print(sysconfig.get_path('purelib'))" \
    | xargs -I{} sh -c 'echo /workspace/model/src > {}/workspace-model.pth'
