# Copyright © Advanced Micro Devices, Inc., or its affiliates.
#
# SPDX-License-Identifier: MIT

# ==============================================================================
# makefile-zen5.mk
# ==============================================================================
# EPYC/Zen5-specific overrides for AIM container builds.
# Included by the main Makefile when BUILD_ZEN5=true.
#
# Only sets values that differ from makefile-defaults.mk (Instinct).
# Common values (registry, date version, etc.) are inherited from defaults.
#
# Example:
#   make build BUILD_ZEN5=true
#   make build-model BUILD_ZEN5=true ORG=meta-llama MODEL=Llama-3.1-70B-Instruct
# ==============================================================================

# Accelerator type (must be set before Makefile conditionals)
ACCELERATOR_TYPE = epyc

# Upstream base image (ZenDNN + ZenTorch)
BASE_REGISTRY_NAMESPACE = amdih
BASE_REPOSITORY         = zendnn_zentorch
BASE_TAG                = vllm_v0.13.0_zentorch_v5.2.0_ubuntu22.04_2026_ww02

# Default model for EPYC builds
ORG   ?= meta-llama
MODEL ?= Llama-3.2-1B-Instruct
