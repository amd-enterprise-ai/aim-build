# Copyright © Advanced Micro Devices, Inc., or its affiliates.
#
# SPDX-License-Identifier: MIT

# Engine image for the Aurora weather models: Python dependencies, the BentoML
# serving code and the bentofile. The base image at assets/instinct/base/aurora/
# adds the AIM runtime and the harness; the model image adds profiles and config.

ARG BASE_IMAGE=docker.io/rocm/pytorch:rocm7.2_ubuntu24.04_py3.12_pytorch_release_2.9.1
FROM ${BASE_IMAGE}

# Application dependencies.
COPY requirements.txt /tmp/requirements.txt
RUN pip install --no-cache-dir -r /tmp/requirements.txt

# Fail the build if the pip install above swapped any part of the ROCm torch
# stack for a PyPI CUDA wheel. The +rocm local build tag is checkable without a
# GPU. Distribution names are discovered rather than hardcoded, since they
# differ between the ROCm base (`triton`) and PyPI's CUDA wheels
# (`pytorch-triton`).
RUN python -c "\
import importlib.metadata as md;\
stack = {'torch', 'torchvision', 'torchaudio', 'triton', 'pytorch-triton', 'pytorch-triton-rocm'};\
found = {(d.metadata['Name'] or '').lower(): d.version for d in md.distributions() if (d.metadata['Name'] or '').lower() in stack};\
bad = {k: v for k, v in found.items() if 'rocm' not in v};\
assert not bad, 'ROCm stack clobbered by a non-ROCm wheel: %r (full: %r)' % (bad, found);\
assert 'torch' in found and 'torchvision' in found, 'torch/torchvision missing: %r' % (found,);\
import torch, torchvision;\
import aurora;\
from aurora import AuroraPretrained, Batch, Metadata, rollout;\
assert AuroraPretrained.default_checkpoint_name == 'aurora-0.25-pretrained.ckpt', AuroraPretrained.default_checkpoint_name;\
print('deps gate OK:', found, 'aurora', md.version('microsoft-aurora'))"

# Serving code. The base image's src/ lands in the same directory, so the
# harness imports config and data_utils from here.
COPY src/ /workspace/model/src/

# bentofile.yaml at the working-dir root: the serve command passes no explicit
# target, so BentoML reads its `service:` import string to find the service class.
COPY bentofile.yaml /workspace/model/bentofile.yaml

WORKDIR /workspace/model
ENV PYTHONPATH=/workspace/model/src

# Clear the upstream image's interactive CMD; the serve ENTRYPOINT comes from
# the base.
CMD []
