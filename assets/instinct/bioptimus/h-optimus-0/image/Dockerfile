# Copyright © Advanced Micro Devices, Inc., or its affiliates.
#
# SPDX-License-Identifier: MIT

# Specialized base image for H-optimus-0 (Bioptimus pathology ViT embedding model).
# Layer 1 in the specialized image stack:
#   Layer 0: Upstream vendor image (this FROM)
#   Layer 1: This image — model-specific Python deps, BentoML service code
#   Layer 2: AIM base (aim-runtime, entrypoint, engine config)
#   Layer 3: Model image (profiles, AIM_ID)

FROM docker.io/rocm/pytorch:rocm7.2_ubuntu24.04_py3.12_pytorch_release_2.9.1

SHELL ["/bin/bash", "-euo", "pipefail", "-c"]

WORKDIR /workspace/model

COPY --link requirements.txt /workspace/model/requirements.txt
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install -r requirements.txt

# ROCm canary — verify torch and ROCm stack are importable.
# NOTE: torch.cuda.is_available() is intentionally NOT checked here — GPUs are
# not accessible during docker build (only at runtime). We verify importability
# and the HIP version string only; GPU visibility is confirmed by the smoke test.
RUN python3 -c "import torch, sys; print('ROCm canary ok: torch importable, hip=', torch.version.hip); sys.exit(0 if torch.version.hip is not None else 'ROCm torch not installed')"

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
# the service instance (service:HOptimus0Service).
COPY --link bentofile.yaml /workspace/model/bentofile.yaml
COPY --link src /workspace/model/src

# Add /workspace/model/src to sys.path so aim-runtime can discover harness.py
# and the harness can use sibling imports (from service import ...).
RUN python3 -c "import sysconfig; print(sysconfig.get_path('purelib'))" \
    | xargs -I{} sh -c 'echo /workspace/model/src > {}/workspace-model.pth'
