# Copyright © Advanced Micro Devices, Inc., or its affiliates.
#
# SPDX-License-Identifier: MIT

# Layer 1 (specialized base) in the Boltz-2 image stack:
#   Layer 0: Upstream vendor image (pinned in the FROM below)
#   Layer 1: This image — Boltz-2 Python deps at a pinned release and the
#            BentoML serving wrapper (src/service.py + src/runner.py)
#   Layer 2: AIM base (aim-runtime, entrypoint, engine config)
#   Layer 3: Model image (profiles, AIM_ID)
#
# The Layer 1 build (base-image-pipeline.yaml) runs a plain `docker build` with
# no build-args, so the upstream ref is pinned here directly rather than
# injected via ARG. The digest is the one the validated image was built from.

FROM docker.io/rocm/pytorch:rocm7.2_ubuntu24.04_py3.12_pytorch_release_2.10.0@sha256:bcb2efa1d43f66cc59fa45a1975d621a50b2b06d21a2abdfa2ba821ab32763e4

SHELL ["/bin/bash", "-euo", "pipefail", "-c"]

# PyPI's newest release is still 2.2.1. This pins past it, to upstream main at
# the commit verified HEAD on 2026-10-02 — six commits ahead of 2.2.1, carrying
# an active-device-type autocast fix and a direct-to-device tensor allocation
# fix (both non-CUDA-specific; neither is in any PyPI release). pyproject.toml's
# own version string at this commit is unchanged at "2.2.1".
ARG BOLTZ_COMMIT=b1ebfc46ecf57f5414e0d1a6f9027bbb122c53bc

# System packages. Boltz pulls no compiled extensions of its own, but rdkit,
# gemmi and numba resolve faster with build tooling present, and `git` is
# needed by pip to fetch the commit-pinned checkout below.
RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Version constraints for every package pip installs below. The file leaves out
# the packages the base image owns (torch, torchvision, torchaudio, triton), so pip
# never replaces the ROCm build.
COPY --link constraints.txt /opt/constraints.txt

# Boltz-2 from a pinned upstream commit (see BOLTZ_COMMIT above), into the base
# image's Python (which carries the ROCm torch build). `torch>=2.2` is already
# satisfied by the base, so pip leaves torch alone; everything else resolves
# from PyPI within /opt/constraints.txt.
#
# Deliberately NOT `boltz[cuda]`: that extra pulls cuequivariance_ops_cu12 /
# cuequivariance_torch, which are CUDA-only. Without them the model must run
# with use_kernels=False — cuEquivariance stays off (src/service.py).
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install -c /opt/constraints.txt "git+https://github.com/jwohlwend/boltz.git@${BOLTZ_COMMIT}"

# Canary: prove the ROCm torch survived pip's dependency resolution, that the
# numpy downgrade boltz forces (it pins numpy<2.0, the base ships 2.x) did not
# break torch's numpy interop, and that boltz itself imports. Any of these
# failing is a base-image/dependency regression that must fail the BUILD, not
# surface later as a broken smoke.
RUN python3 -c "\
import sys, numpy, torch; \
assert torch.version.hip, f'torch is not a ROCm build: {torch.__version__}'; \
t = torch.from_numpy(numpy.zeros((2, 2), dtype='float32')); \
assert t.shape == (2, 2) and t.numpy().shape == (2, 2), 'torch<->numpy interop broken'; \
import boltz, boltz.main; \
print('torch', torch.__version__, 'hip', torch.version.hip, 'numpy', numpy.__version__); sys.exit(0)"

ENV ROCM_PATH=/opt/rocm
ENV PYTORCH_ROCM_ARCH="gfx90a;gfx942;gfx950"
# Weight/CCD cache of a bare `boltz` CLI invocation inside the image. The
# service does not read it: it takes its cache from AIM_CACHE_PATH (see
# src/service.py).
ENV BOLTZ_CACHE=/workspace/model-cache/boltz
# Deliberately no BENTOML_PORT: it is a click envvar on `bentoml serve --port`,
# which the CLI applies *after* the service config, so it would silently
# override the service's own `http={"port": ...}` and discard the port
# aim-runtime forwards as `--arg port=<AIM_PORT>`. src/service.py owns the port.
ENV AIM_ENGINE=bentoml
ENV WANDB_MODE=disabled
ENV WANDB_SILENT=true
ENV HF_HUB_DISABLE_TELEMETRY=1

# BentoML serving wrapper. Copy only the package files so unrelated context
# (README.md) does not invalidate the layer cache or leak into the image.
# pyproject.toml uses a src layout; src/ holds service.py + runner.py + harness.py.
WORKDIR /workspace/model
COPY --link pyproject.toml /workspace/model/pyproject.toml
COPY --link src /workspace/model/src

# Benchmark + evaluation payloads for `aim-runtime benchmark` / `evaluate`.
# The harness reads /workspace/model/benchmarks/<case>/<case>.json.
#
# ALL EIGHTEEN CASES ARE BAKED, but an evaluate run uses only the ones in the
# set named by BOLTZ2_BENCHMARK_SET, which defaults to the six of "fast". The
# other twelve complete the eighteen-target timing set and are reached with
# BOLTZ2_BENCHMARK_SET=timing-18. Baking a case is not selecting it; the
# selector is metadata.sets, asserted by tests/unit/test_benchmark_sets.py.
COPY --link examples/benchmark_sequences /workspace/model/benchmarks

# Accuracy-gate case set — a SEPARATE directory, not another name in
# metadata.sets, because it differs from the set above by RECIPE
# (harness.ACCURACY_GATE_RECIPE: the served defaults plus use_msa_server=true)
# rather than by coverage. Same six RCSB targets as "fast"
# (7qsj/8are/8c4d/8cpk/8eil/8gpp), same staged reference structures — this
# directory exists so harness._check_structure_accuracy has at least one case
# set it can grade instead of abstaining. Reached only when BOTH
# BOLTZ2_BENCHMARK_DIR=/workspace/model/benchmarks-accuracy AND
# BOLTZ2_BENCHMARK_SET=accuracy-gate are exported; neither is set by any
# profile, so the default evaluate run is unaffected. See
# tests/unit/test_accuracy_gate_fixtures.py.
COPY --link examples/benchmark_sequences_accuracy /workspace/model/benchmarks-accuracy

COPY --link bentofile.yaml /workspace/model/bentofile.yaml

RUN --mount=type=cache,target=/root/.cache/pip \
    pip install -c /opt/constraints.txt -e .

# bentoml's own chain (starlette/httpx/httpcore, none of which pin anyio's
# floor above 4.0) lets pip resolve anyio up to 4.15.1, which then requires
# typing_extensions>=4.16.0 - a conflict with the repo-wide
# typing-extensions==4.15.0 pin in requirements/{requirements,instinct-requirements}.txt
# (neither of which pins anyio, so Layer 2's reinstall cannot re-upgrade it).
# Pin anyio down instead of typing_extensions up, matching existing precedent
# in requirements/{radeon,epyc,evaluation,dev}-requirements.txt. anyio==4.13.0
# requires only typing_extensions>=4.5, satisfied by the 4.15.0 pin.
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install "anyio==4.13.0"

RUN pip install --no-cache-dir "amdsmi==7.0.2"

# Neutral serve argv. The base image declares CMD ["/bin/bash"]; a Layer-1
# image must not hand off an application command. Layer 2 sets the real
# ENTRYPOINT (which also resets CMD), so this is a no-op for the AIM stack and
# keeps the standalone Layer-1 image's effective argv empty.
CMD []
