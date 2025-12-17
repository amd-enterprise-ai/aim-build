<!--
Copyright © Advanced Micro Devices, Inc., or its affiliates.

SPDX-License-Identifier: MIT
-->

# Docker deployment

This guide provides step-by-step instructions for deploying AMD Inference Microservice (AIM) container that supports
different variants of Llama-3.1-8B-Instruct model. Follow these instructions to quickly get started with running an AI
model on AMD GPUs. For more detailed information, please refer to [development documentation](https://github.com/amd-enterprise-ai/aim-build/blob/main/README.md).

## Prerequisites

* AMD GPU with ROCm support (e.g., MI300X, MI325X)
* Docker installed and configured with GPU support
* Access to model repositories (Hugging Face account with appropriate permissions for gated models)

## 1. Docker deployment

### 1.1 Running the container

```bash
docker run \
  -e HF_TOKEN=<YOUR_HUGGINGFACE_TOKEN> \
  --device=/dev/kfd --device=/dev/dri \
  -p 8000:8000 \
  amdenterpriseai/aim-meta-llama-llama-3-1-8b-instruct:0.8.5
```

Where <YOUR_HUGGINGFACE_TOKEN> is your Hugging Face access token (required for gated models)

### 1.2 Customizing deployment with environment variables

Customize your deployment with optional environment variables:

```bash
docker run \
  -e AIM_PRECISION=fp16 \
  -e AIM_GPU_COUNT=1 \
  -e AIM_METRIC=throughput \
  -e AIM_PORT=8080 \
  --device=/dev/kfd --device=/dev/dri \
  -p 8080:8080 \
  amdenterpriseai/aim-meta-llama-llama-3-1-8b-instruct:0.8.5
```

## 2. Model caching for production

For production environments, pre-download models to a persistent cache:

### 2.1 Download model to cache

```bash
# Create persistent cache directory
mkdir -p /path/to/model-cache

# Download model using the download-to-cache command
docker run --rm \
  -e HF_TOKEN=<YOUR_HUGGINGFACE_TOKEN> \
  -v /path/to/model-cache:/workspace/model-cache \
  amdenterpriseai/aim-meta-llama-llama-3-1-8b-instruct:0.8.5 \
  download-to-cache --model-id meta-llama/Llama-3.1-8B-Instruct
```

### 2.2 Run with pre-cached model

```bash
docker run \
  -e HF_TOKEN=<YOUR_HUGGINGFACE_TOKEN> \
  -v /path/to/model-cache:/workspace/model-cache \
  --device=/dev/kfd --device=/dev/dri \
  -p 8000:8000 \
  amdenterpriseai/aim-meta-llama-llama-3-1-8b-instruct:0.8.5
```

## 3. Monitoring and troubleshooting

### 3.1 Getting help on the commands

A general help command is available as follows:

```bash
docker run \
  amdenterpriseai/aim-meta-llama-llama-3-1-8b-instruct:0.8.5 \
  --help
```

A help command for specific subcommands is also available:

```bash
docker run \
  amdenterpriseai/aim-meta-llama-llama-3-1-8b-instruct:0.8.5 \
  <subcommand> --help
```

### 3.2 Enabling detailed logging

```bash
docker run \
  -e AIM_LOG_LEVEL=DEBUG \
  -e HF_TOKEN=<YOUR_HUGGINGFACE_TOKEN> \
  --device=/dev/kfd --device=/dev/dri \
  -p 8000:8000 \
  amdenterpriseai/aim-meta-llama-llama-3-1-8b-instruct:0.8.5
```

### 3.3 Checking profile selection results

It is possible to check which profile AIM selects based on the provided environment variables.

```bash
docker run \
  -e AIM_GPU_COUNT=1 \
  -e AIM_PRECISION=fp16 \
  -e AIM_GPU_MODEL=MI300X \
  -e HF_TOKEN=<YOUR_HUGGINGFACE_TOKEN> \
  amdenterpriseai/aim-meta-llama-llama-3-1-8b-instruct:0.8.5 \
  dry-run
```

### 3.4 List available profiles

```bash
docker run \
  amdenterpriseai/aim-meta-llama-llama-3-1-8b-instruct:0.8.5 \
  list-profiles
```
