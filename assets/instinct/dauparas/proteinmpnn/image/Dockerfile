# Copyright © Advanced Micro Devices, Inc., or its affiliates.
#
# SPDX-License-Identifier: MIT

# Specialized base image for ProteinMPNN (protein sequence design).
# Layer 1 in the specialized image stack:
#   Layer 0: Upstream vendor image (this FROM)
#   Layer 1: This image — ProteinMPNN source, Python deps, BentoML service
#   Layer 2: AIM base (aim-runtime, entrypoint, engine config)
#   Layer 3: Model image (profiles, AIM_ID)
#
# The Layer 1 build runs a plain `docker build` with no build-args, so the
# upstream ref is pinned here directly rather than injected via ARG.

FROM docker.io/rocm/pytorch:rocm7.2_ubuntu24.04_py3.12_pytorch_release_2.9.1

SHELL ["/bin/bash", "-euo", "pipefail", "-c"]

ARG PROTEINMPNN_REPO=https://github.com/dauparas/ProteinMPNN.git
# Commit of tag v1.0.1 (resolved with `git ls-remote`); a tag can be moved, a SHA cannot.
ARG PROTEINMPNN_REF=905b0081f3aa3404d7ac22174bc25710f9995ff8

# System packages needed for git clone
RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    && rm -rf /var/lib/apt/lists/*

# Verify torch is the ROCm build: the build fails with exit code 1 when HIP is absent.
RUN python3 -c "import torch, sys; ok = torch.version.hip is not None; print('ROCm HIP found' if ok else 'ROCm HIP not found'); sys.exit(0 if ok else 1)"

# Install BentoML and service dependencies (numpy ships with ROCm torch; pin
# bentoml for reproducibility). httpx, pydantic and starlette are imported directly by
# service.py; their versions are what `pip freeze` reports in the validation image, built
# from PR head ff2fc868a on a Crusoe MI355X node (not a CI image; Layer 2 built locally).
# They match the aim-runtime layer's repo-wide pins.
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install \
    "bentoml==1.4.39" \
    "httpx==0.28.1" \
    "pydantic==2.13.5" \
    "starlette==1.10.0"

# Clone, check out the pinned commit (a SHA cannot be cloned by branch name) and prune in ONE
# RUN, so the weights, .git packs and training/example assets never enter an image layer.
#  - Weights are not shipped: the service fetches the four vanilla files into
#    $AIM_CACHE_PATH/proteinmpnn/vanilla_model_weights at startup. .git is removed too, since its
#    pack would still hold the weight blobs.
#  - Licence-driven prune: the service only runs protein_mpnn_run.py, which imports
#    protein_mpnn_utils.py. Drop training/, examples/, inputs/, helper_scripts/,
#    colab_notebooks/ and images (training-data provenance unverified); keep LICENSE.
RUN git clone ${PROTEINMPNN_REPO} /opt/proteinmpnn \
 && git -C /opt/proteinmpnn checkout ${PROTEINMPNN_REF} \
 && find /opt/proteinmpnn -mindepth 1 -maxdepth 1 \
        ! -name protein_mpnn_run.py ! -name protein_mpnn_utils.py ! -name LICENSE \
        -exec rm -rf {} + \
 && ! find /opt/proteinmpnn -name '*.pt' | grep -q . \
 && ls /opt/proteinmpnn | sort | tr '\n' ' ' | grep -qx 'LICENSE protein_mpnn_run.py protein_mpnn_utils.py ' \
 && python3 /opt/proteinmpnn/protein_mpnn_run.py --help >/dev/null

# git is only needed for the clone; purge it (the layer still holds it from the
# earlier RUN, so this removes it from the final filesystem, not from image size).
RUN apt-get purge -y --auto-remove git && rm -rf /var/lib/apt/lists/*

ENV PROTEINMPNN_PATH=/opt/proteinmpnn
# Deliberately no BENTOML_PORT: it is a click envvar on `bentoml serve --port`,
# applied after the service config, so it would override http={"port": ...}.
# aim-runtime forwards AIM_PORT as `--arg port=...`, consumed in service.py.
ENV AIM_ENGINE=bentoml

# The shipped profiles are throughput profiles (metadata.metric: throughput) and
# aim-runtime defaults AIM_METRIC to latency, which would reject them as a metric
# mismatch. Select throughput so automatic profile selection finds them.
ENV AIM_METRIC=throughput

# BentoML service wrapper.
WORKDIR /workspace/model
COPY --link pyproject.toml /workspace/model/pyproject.toml
COPY --link src /workspace/model/src
COPY --link bentofile.yaml /workspace/model/bentofile.yaml

# Install the workspace package (exposes src/ modules to bentoml serve).
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install --no-deps -e .

# Add /workspace/model/src to sys.path so the harness can import sibling modules.
RUN python3 -c "import sysconfig; print(sysconfig.get_path('purelib'))" \
    | xargs -I{} sh -c 'echo /workspace/model/src > {}/workspace-model.pth'
