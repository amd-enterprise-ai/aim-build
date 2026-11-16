# Copyright © Advanced Micro Devices, Inc., or its affiliates.
#
# SPDX-License-Identifier: MIT

# Specialized base image for SemlaFlow (small-molecule generation via flow matching).
# Layer 1 in the specialized image stack:
#   Layer 0: Upstream vendor image (this FROM)
#   Layer 1: This image — model-specific Python deps, SemlaFlow source at a pinned ref
#   Layer 2: AIM base (aim-runtime, entrypoint, engine config)
#   Layer 3: Model image (profiles, AIM_ID)
#
# The Layer 1 build runs a plain `docker build` with no build-args, so the
# upstream ref is pinned here directly rather than injected via ARG.

FROM rocm/pytorch:rocm7.2_ubuntu24.04_py3.12_pytorch_release_2.9.1

SHELL ["/bin/bash", "-euo", "pipefail", "-c"]

# System packages
RUN apt-get update && apt-get install -y --no-install-recommends \
    ca-certificates \
    git \
    && rm -rf /var/lib/apt/lists/*

# Verify torch is the ROCm 7.2 build. Doubles as a canary when the upstream image
# tag is bumped. sys.exit(1) makes the RUN layer fail loudly if ROCm torch is missing.
RUN python3 -c "import sys, torch; print('torch:', torch.__file__, 'version:', torch.__version__, 'hip:', torch.version.hip); sys.exit(0 if (torch.version.hip is not None and torch.version.hip.startswith('7.2')) else 'ROCm 7.2 torch not found')"

# Install SemlaFlow Python dependencies.
# bentoml is included here (not in a separate Layer 2) because the service
# can be started from Layer 1 directly, without a separate Layer 2 image.
# Layer 2 normally supplies bentoml; we pin it explicitly so the image is
# self-contained.
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install \
    numpy==2.5.2 \
    pandas==3.0.5 \
    scipy==1.18.0 \
    rdkit==2026.3.5 \
    lightning==2.6.5 \
    torchmetrics==1.9.0 \
    typing_extensions==4.16.0 \
    "bentoml==1.4.39"

# Fetch SemlaFlow source at a pinned commit (rssrwn/semla-flow).
# --depth 1 on git-fetch keeps this to a single-commit download.
# The SHA pins the exact upstream revision; bump it deliberately when updating.
ARG SEMLAFLOW_SHA=3f43103d3af138b86dbe9f29fe8085e83f9a6283
RUN git init -q /opt/semlaflow \
 && cd /opt/semlaflow \
 && git fetch -q --depth 1 https://github.com/rssrwn/semla-flow.git ${SEMLAFLOW_SHA} \
 && git checkout -q FETCH_HEAD

# Upstream imports the GPL-2.0 openbabel wheel at module level (scriptutil.py, util/rdkit.py) for one
# log-silencing call and a bond-inference helper the served predict path never reaches. Patch both
# away (exact-match, fails on a SHA bump that changes them), then assert nothing references it and
# that the modules the service runs import with the package blocked.
COPY patches/drop_openbabel.py /tmp/drop_openbabel.py
RUN python3 /tmp/drop_openbabel.py /opt/semlaflow \
 && rm /tmp/drop_openbabel.py \
 && ! grep -rnE '(from|import) +openbabel|pybel' /opt/semlaflow --include='*.py' \
 && cd /opt/semlaflow \
 && python3 -c "import sys; sys.modules['openbabel'] = None; import semlaflow.scriptutil, semlaflow.util.rdkit, semlaflow.predict; print('semlaflow imports without openbabel: ok')"

ENV ROCM_PATH=/opt/rocm
ENV PYTORCH_ROCM_ARCH="gfx942"
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

# BentoML service wrapper.
WORKDIR /workspace/model
COPY src/ /workspace/model/src/
COPY bentofile.yaml /workspace/model/bentofile.yaml

# Add /workspace/model/src and /opt/semlaflow to sys.path.
# /opt/semlaflow makes `import semlaflow` resolve to /opt/semlaflow/semlaflow/.
RUN python3 -c "import sysconfig; print(sysconfig.get_path('purelib'))" \
    | xargs -I{} sh -c 'printf "/workspace/model/src\n/opt/semlaflow\n" > {}/workspace-model.pth'
