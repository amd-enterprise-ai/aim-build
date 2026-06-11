# Copyright © Advanced Micro Devices, Inc., or its affiliates.
#
# SPDX-License-Identifier: MIT

# Specialized base image for the play echo-model.
# Layer 1 in the specialized image stack:
#   Layer 0: python:3.12-slim (this FROM)
#   Layer 1: This image — BentoML echo service source + deps
#   Layer 2: AIM base (aim-runtime, entrypoint, engine config)
#   Layer 3: Model image (profiles, AIM_ID)
#
# Hosts its endpoints with BentoML (see src/service.py).

ARG BASE_IMAGE=python:3.12-slim
FROM ${BASE_IMAGE}

WORKDIR /workspace/model

# Install the BentoML serving stack.
COPY requirements.txt /workspace/model/requirements.txt
RUN pip install --no-cache-dir -r /workspace/model/requirements.txt

# `bentoml serve --working-dir /workspace/model` reads bentofile.yaml to find
# the service instance (service:EchoService).
COPY bentofile.yaml /workspace/model/bentofile.yaml
COPY src/ /workspace/model/src/

# Add /workspace/model/src to sys.path so both BentoML (resolving the
# "service:EchoService" import string) and the harness can import sibling modules.
RUN python3 -c "import sysconfig; print(sysconfig.get_path('purelib'))" \
    | xargs -I{} sh -c 'echo /workspace/model/src > {}/workspace-model.pth'
