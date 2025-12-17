# Copyright © Advanced Micro Devices, Inc., or its affiliates.
#
# SPDX-License-Identifier: MIT

# ==============================================================================
# makefile-defaults.mk
# ==============================================================================
# Default configuration values for AIM container builds
# All defaults are centralized here and can be overridden from the command line
# or by setting environment variables
#
# Example overrides:
#   make build-model ORG=meta-llama MODEL=Llama-3.1-70B-Instruct
#   make build-base AIM_BASE_IMAGE_TAG=0.4
# ==============================================================================

# ==============================================================================
# Container Registry Configuration
# ==============================================================================
# Target registry/namespace for pushing built images
AIM_REGISTRY_HOSTNAME  = ghcr.io
AIM_REGISTRY_NAMESPACE = silogen

# ==============================================================================
# Version Configuration
# ==============================================================================
# Date-based version suffix for model-specific containers
# Used for: aim:0.3.0-meta-llama-llama-3.1-8b-instruct-v20251001
DATE_VERSION = v$(shell date +%Y%m%d)

# ==============================================================================
# Base Image Source
# ==============================================================================
# Upstream base image to build from (e.g., ROCm runtime image)
BASE_REGISTRY_NAMESPACE = rocm
BASE_REPOSITORY         = vllm
BASE_TAG                = rocm7.0.0_vllm_0.11.1_20251103

# ==============================================================================
# Container Image Names
# ==============================================================================
# Base container name (contains AIM runtime, no model profiles)
AIM_BASE_REPOSITORY = aim-base

# Model-specific container name (includes model-specific profiles)
AIM_REPOSITORY = aim

# ==============================================================================
# Default Model Configuration
# ==============================================================================
# Default organization and model for model-specific builds
# These values are used when ORG and MODEL are not specified
ORG   = meta-llama
MODEL = Llama-3.1-8B-Instruct
