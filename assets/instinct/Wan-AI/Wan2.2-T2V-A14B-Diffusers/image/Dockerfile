# Copyright © Advanced Micro Devices, Inc., or its affiliates.
#
# SPDX-License-Identifier: MIT

# Specialized base image for Wan-AI/Wan2.2-T2V-A14B-Diffusers.
#
# Layer 1 in the specialized image stack (PR #959 pattern):
#   Layer 0: Upstream vendor image (vllm/vllm-omni-rocm)
#   Layer 1: This image — drops the VllmOmniDiffusionHarness sources into
#            /workspace/model/src/ so aim_runtime.harness.discovery picks
#            them up at runtime.
#   Layer 2: AIM omni base — adds aim-runtime + the omni-flavor engines.yaml
#            and general-fallback profiles. FROMs Layer 1 via build-args.
#   Layer 3: Model image (built from docker/Dockerfile.aim) — adds the
#            wan2.2-specific profiles and AIM_ID env. FROMs Layer 2.
#
# All Python runtime deps (vllm-omni, aiohttp, requests, tqdm, the diffusion
# benchmark script at /app/vllm-omni/benchmarks/diffusion/) are already
# present in the vllm/vllm-omni-rocm vendor image and survive Layers 2-3.

ARG BASE_IMAGE=docker.io/vllm/vllm-omni-rocm:v0.24.0
FROM ${BASE_IMAGE}

WORKDIR /workspace/model

COPY src/ /workspace/model/src/

# Add /workspace/model/src to sys.path so the harness can import sibling modules
# (e.g. ``from recipes import get_recipe``). This .pth file is read by Python on
# interpreter start; aim_runtime.harness.discovery.discover_harness() also adds
# the directory to sys.path defensively at runtime in case this layer is absent.
RUN python3 -c "import sysconfig; print(sysconfig.get_path('purelib'))" \
    | xargs -I{} sh -c 'echo /workspace/model/src > {}/workspace-model.pth'
