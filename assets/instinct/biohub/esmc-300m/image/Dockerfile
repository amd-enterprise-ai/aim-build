# Copyright © Advanced Micro Devices, Inc., or its affiliates.
#
# SPDX-License-Identifier: MIT

# Specialized base image for ESMC 300M (protein language model embedding).
# Layer 1 in the specialized image stack:
#   Layer 0: Upstream vendor image (this FROM)
#   Layer 1: This image — HuggingFace Transformers + BentoML service code
#   Layer 2: AIM base (aim-runtime, entrypoint, engine config)
#   Layer 3: Model image (profiles, AIM_ID)
#
# The Layer 1 specialized base is content-addressed on the image/ directory only, so a
# moving FROM tag would not invalidate the cache. FROM is therefore pinned by digest, so the cached
# Layer 1 cannot silently drift to a new upstream. Update the tag and the digest together.
FROM docker.io/rocm/pytorch:rocm7.2_ubuntu24.04_py3.12_pytorch_release_2.10.0@sha256:bcb2efa1d43f66cc59fa45a1975d621a50b2b06d21a2abdfa2ba821ab32763e4

SHELL ["/bin/bash", "-euo", "pipefail", "-c"]

WORKDIR /workspace/model

COPY --link requirements.txt /workspace/model/requirements.txt
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install -r requirements.txt

# Canary: ensure the base image's ROCm torch survived the dependency install
# (doubles as a tripwire if the upstream FROM tag is bumped).
RUN python3 -c "import torch, sys; print('torch:', torch.__version__, 'hip:', torch.version.hip); sys.exit(0 if torch.version.hip is not None else 'ROCm torch not installed')"

ENV AIM_ENGINE=bentoml

# The shipped profiles are throughput profiles (metadata.metric: throughput) and
# aim-runtime defaults AIM_METRIC to latency, which would reject them as a metric
# mismatch. Select throughput so automatic profile selection finds them.
ENV AIM_METRIC=throughput

COPY src/ /workspace/model/src/

# `bentoml serve --working-dir /workspace/model` reads bentofile.yaml to find the service.
COPY bentofile.yaml /workspace/model/bentofile.yaml

# Add /workspace/model/src to sys.path so the harness can import sibling modules.
RUN python3 -c "import sysconfig; print(sysconfig.get_path('purelib'))" \
    | xargs -I{} sh -c 'echo /workspace/model/src > {}/workspace-model.pth'
