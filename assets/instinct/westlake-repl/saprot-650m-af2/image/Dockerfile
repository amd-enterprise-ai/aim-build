# Copyright © Advanced Micro Devices, Inc., or its affiliates.
#
# SPDX-License-Identifier: MIT

# Specialized base image for SaProt 650M AF2 (structure-aware protein language model embedding).
# Layer 1 in the specialized image stack:
#   Layer 0: Upstream vendor image (this FROM)
#   Layer 1: This image — HuggingFace Transformers + BentoML service code
#   Layer 2: AIM base (aim-runtime, entrypoint, engine config)
#   Layer 3: Model image (profiles, AIM_ID)
#
# The Layer 1 specialized base is content-addressed on the image/ directory only, so the
# parent is pinned by digest: changing the parent changes this line and therefore the
# Layer 1 cache key.
# Tag: docker.io/rocm/pytorch:rocm7.2_ubuntu24.04_py3.12_pytorch_release_2.9.1
# Digest resolved from Docker Hub on 2026-10-25 (single-platform image manifest; the tag
# is not a manifest list).
FROM docker.io/rocm/pytorch@sha256:a3b65813621095e3389269417e963725b59310184588c9d2490d44e6e83fa01c

SHELL ["/bin/bash", "-euo", "pipefail", "-c"]

WORKDIR /workspace/model

COPY --link requirements.txt /workspace/model/requirements.txt
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install -r requirements.txt

# Canary: ensure the base image's ROCm torch survived the dependency install
# (doubles as a tripwire if the upstream FROM tag is bumped).
RUN python3 -c "import torch, sys; print('torch:', torch.__version__, 'hip:', torch.version.hip); sys.exit(0 if torch.version.hip is not None else 'ROCm torch not installed')"

# Deliberately no BENTOML_PORT: it is a click envvar on `bentoml serve --port`,
# which the CLI applies *after* the service config, so it silently overrides the
# service's own `http={"port": ...}`. Setting it here would discard the port
# aim-runtime forwards as `--arg port=<AIM_PORT>` and pin every container to one
# port, which collides on host-network nodes. src/service.py owns the port.
ENV AIM_ENGINE=bentoml

# The shipped profiles are throughput profiles (metadata.metric: throughput) and
# aim-runtime defaults AIM_METRIC to latency, which would reject them as a metric
# mismatch. Select throughput so automatic profile selection finds them.
ENV AIM_METRIC=throughput

# `bentoml serve --working-dir /workspace/model` reads bentofile.yaml to find
# the service instance (service:SaProtEmbeddingService).
COPY bentofile.yaml /workspace/model/bentofile.yaml
COPY src/ /workspace/model/src/

# Add /workspace/model/src to sys.path so the harness can import sibling modules.
RUN python3 -c "import sysconfig; print(sysconfig.get_path('purelib'))" \
    | xargs -I{} sh -c 'echo /workspace/model/src > {}/workspace-model.pth'
