# Copyright © Advanced Micro Devices, Inc., or its affiliates.
#
# SPDX-License-Identifier: MIT

# Layer 1 (specialized base) in the Whisper image stack:
#   Layer 0: Upstream vendor image (pinned below)
#   Layer 1: This image — Whisper benchmark deps, harness, and benchmark config
#   Layer 2: AIM base (aim-runtime, entrypoint, engine config)
#   Layer 3: Model image (profiles, AIM_ID)
#
# The Layer 1 build in CI runs a plain `docker build` with no build args, so
# the upstream ref must be pinned directly here.
FROM docker.io/rocm/vllm:rocm10.0.0_ubuntu24.04_py3.14_pytorch_2.12.0_vllm_0.27.0

SHELL ["/bin/bash", "-euo", "pipefail", "-c"]

COPY requirements.txt /tmp/requirements.txt
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install --no-cache-dir -r /tmp/requirements.txt

# Install the model-specific harness and benchmark config used by the AIM runtime.
COPY src/whisper_harness.py /workspace/model/src/harness.py
COPY benchmark-config.yaml /workspace/model/benchmark-config.yaml
