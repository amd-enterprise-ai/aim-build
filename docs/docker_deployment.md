<!--
Copyright © Advanced Micro Devices, Inc., or its affiliates.

SPDX-License-Identifier: MIT
-->

# Docker deployment

This guide provides step-by-step instructions for deploying AMD Inference Microservice (AIM) container that supports
different variants of Llama-3.1-8B-Instruct model. Follow these instructions to quickly get started with running an AI
model on AMD accelerators. For more detailed information, please refer to the [main README](https://github.com/amd-enterprise-ai/aim-build/blob/main/README.md).

## Prerequisites

* AMD GPU with ROCm support (e.g., MI300X, MI325X for Instinct™; W7900, R9700 for Radeon™ Pro)
* AMD EPYC™ CPU for CPU-only deployment
* Docker installed and configured
* Access to model repositories (Hugging Face account with appropriate permissions for gated models)

## 1. Docker deployment

### 1.1 Running the container

```bash
docker run \
  -e HF_TOKEN=<YOUR_HUGGINGFACE_TOKEN> \
  --device=/dev/kfd --device=/dev/dri \
  -p 8000:8000 \
  amdenterpriseai/aim-meta-llama-llama-3-1-8b-instruct:0.11.1
```

Where <YOUR_HUGGINGFACE_TOKEN> is your Hugging Face access token (required for gated models)

### 1.2 Customizing deployment with environment variables

Customize your deployment with optional environment variables. In the example below `AIM_PORT` is set to `8080` instead
of `8000`. `AIM_METRIC` is set to `throughput` instead of `latency`. `AIM_ACCELERATOR_COUNT` is set to `1` instead of `auto`,
`AIM_PRECISION` is set to `fp16` instead of `auto`.

```bash
docker run \
  -e HF_TOKEN=<YOUR_HUGGINGFACE_TOKEN> \
  -e AIM_PRECISION=fp16 \
  -e AIM_ACCELERATOR_COUNT=1 \
  -e AIM_METRIC=throughput \
  -e AIM_PORT=8080 \
  --device=/dev/kfd --device=/dev/dri \
  -p 8080:8080 \
  amdenterpriseai/aim-meta-llama-llama-3-1-8b-instruct:0.11.1
```

Override automatic profile selection by specifying a profile directly. In the example below, `AIM_PROFILE_ID` is set to
`vllm-mi300x-fp8-tp1-latency`. All other environment variables' values are set implicitly according to the specified
profile.

```bash
docker run \
  -e HF_TOKEN=<YOUR_HUGGINGFACE_TOKEN> \
  -e AIM_PROFILE_ID=vllm-mi300x-fp8-tp1-latency \
  --device=/dev/kfd --device=/dev/dri \
  -p 8000:8000 \
  amdenterpriseai/aim-meta-llama-llama-3-1-8b-instruct:0.11.1
```

### 1.3 Running larger models in multi-GPU environments

```bash
docker run \
  -e HF_TOKEN=<YOUR_HUGGINGFACE_TOKEN> \
  --device=/dev/kfd --device=/dev/dri \
  --shm-size=32g \
  -e AIM_GPU_COUNT=2 \
  -p 8000:8000 \
  amdenterpriseai/aim-coherelabs-command-a-reasoning-08-2025:0.11.1
```

**Note:** `--shm-size` sets the size of `/dev/shm` inside the container. Docker's default (64 MB) is too small
for multi-GPU inference — tensor-parallel workers rely on shared memory for inter-process communication and will
fail with cryptic errors without sufficient space. Use `--shm-size=4g` as a safe minimum; increase it for larger
models or higher tensor-parallelism values. Alternatively, `--ipc=host` shares the host's IPC namespace with the
container, which also resolves the issue but reduces isolation and is not recommended for production.

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
  amdenterpriseai/aim-meta-llama-llama-3-1-8b-instruct:0.11.1 \
  download-to-cache --model-id meta-llama/Llama-3.1-8B-Instruct
```

**Note:** `/workspace/model-cache` is the `HF_HOME` inside the container. To reuse an existing host HF cache, mount it
to that path:

```bash
docker run --rm \
  -e HF_TOKEN=<YOUR_HUGGINGFACE_TOKEN> \
  -v $HF_HOME:/workspace/model-cache \
  amdenterpriseai/aim-meta-llama-llama-3-1-8b-instruct:0.11.1 \
  download-to-cache --model-id meta-llama/Llama-3.1-8B-Instruct
```

### 2.2 Run with pre-cached model

```bash
docker run \
  -e HF_TOKEN=<YOUR_HUGGINGFACE_TOKEN> \
  -v /path/to/model-cache:/workspace/model-cache \
  --device=/dev/kfd --device=/dev/dri \
  -p 8000:8000 \
  amdenterpriseai/aim-meta-llama-llama-3-1-8b-instruct:0.11.1
```

## 3. Monitoring and troubleshooting

### 3.1 Getting help on the commands

A general help command is available as follows:

```bash
docker run \
  amdenterpriseai/aim-meta-llama-llama-3-1-8b-instruct:0.11.1 \
  --help
```

A help command for specific subcommands is also available:

```bash
docker run \
  amdenterpriseai/aim-meta-llama-llama-3-1-8b-instruct:0.11.1 \
  <subcommand> --help
```

### 3.2 Enabling detailed logging

```bash
docker run \
  -e AIM_LOG_LEVEL=DEBUG \
  -e HF_TOKEN=<YOUR_HUGGINGFACE_TOKEN> \
  --device=/dev/kfd --device=/dev/dri \
  -p 8000:8000 \
  amdenterpriseai/aim-meta-llama-llama-3-1-8b-instruct:0.11.1
```

### 3.3 Checking profile selection results

It is possible to check which profile AIM selects based on the provided environment variables.

```bash
docker run \
  -e AIM_ACCELERATOR_COUNT=1 \
  -e AIM_PRECISION=fp16 \
  -e AIM_ACCELERATOR_MODEL=MI300X \
  -e HF_TOKEN=<YOUR_HUGGINGFACE_TOKEN> \
  amdenterpriseai/aim-meta-llama-llama-3-1-8b-instruct:0.11.1 \
  dry-run
```

### 3.4 List available profiles

```bash
docker run \
  amdenterpriseai/aim-meta-llama-llama-3-1-8b-instruct:0.11.1 \
  list-profiles
```
