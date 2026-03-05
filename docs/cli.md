<!--
Copyright © Advanced Micro Devices, Inc., or its affiliates.

SPDX-License-Identifier: MIT
-->

# AIM Runtime CLI

The AIM runtime provides a command-line interface with the following subcommands:

## Serve (`serve`)

Performs profile selection and starts the inference server. This is the default behavior when no subcommand is specified.

```bash
# Default behavior - starts the server
docker run -e AIM_MODEL_ID=meta-llama/Llama-3.1-8B-Instruct \
  --device=/dev/kfd --device=/dev/dri \
  -p 8000:8000 \
  aim-base:0.9

# Or explicitly specify serve
docker run -e AIM_MODEL_ID=meta-llama/Llama-3.1-8B-Instruct \
  --device=/dev/kfd --device=/dev/dri \
  -p 8000:8000 \
  aim-base:0.9 \
  serve
```

## Dry run (`dry-run`)

Performs profile selection and displays the selected profile without starting the server. Supports two output formats:

**Options:**
- `--format yaml` (default): Display the complete profile as YAML text
- `--format json`: Display the selected profile as structured JSON

This is useful for:
- Verifying which profile will be selected for your configuration
- Debugging profile selection issues
- Understanding the full configuration before execution
- Programmatic integration with CI/CD pipelines (JSON format for easy parsing)

### YAML Format (default)

```bash
docker run -e AIM_MODEL_ID=meta-llama/Llama-3.1-8B-Instruct \
  -e AIM_GPU_COUNT=1 \
  -e AIM_PRECISION=fp16 \
  -e AIM_ENGINE=vllm \
  aim-base:0.9 \
  dry-run

# Or explicitly specify --format yaml
docker run -e AIM_MODEL_ID=meta-llama/Llama-3.1-8B-Instruct \
  aim-base:0.9 \
  dry-run --format yaml
```

**Example output:**
```yaml
# Selected profile: /workspace/aim-runtime/profiles/meta-llama/Llama-3.1-8B-Instruct/vllm-mi300x-fp16-tp1-latency.yaml

- filename: vllm-mi300x-fp16-tp1-latency.yaml
  path: /workspace/aim-runtime/profiles/general/vllm-mi300x-fp16-tp1-latency.yaml
  profile:
    metadata:
      engine: vllm
      gpu: MI300X
      gpu_count: 1
      manual_selection_only: false
      metric: latency
      precision: fp16
      type: general
    engine_args:
      dtype: float16
      gpu-memory-utilization: 0.95
      no-enable-chunked-prefill: null
      tensor-parallel-size: 1
    env_vars:
      NCCL_MIN_NCHANNELS: '112'
      TORCH_BLAS_PREFER_HIPBLASLT: '1'
      VLLM_DO_NOT_TRACK: '1'
  models:
    - name: meta-llama/Llama-3.1-8B-Instruct
      source: 'hf://meta-llama/Llama-3.1-8B-Instruct'
      size_gb: 29.93
  script: '#!/bin/bash
    .
    .
    .

    '


```

### JSON Format

Performs profile selection and displays the selected profile without starting the server. Supports two output formats:
- CI/CD pipeline integration and programmatic parsing
- Automated testing and validation
- Extracting specific configuration values

```bash
docker run -e AIM_MODEL_ID=meta-llama/Llama-3.1-8B-Instruct \
  aim-base:0.9 \
  dry-run --format json
```

**Example output:**
```json
[
  {
    "filename": "vllm-mi300x-fp16-tp1-latency.yaml",
    "path": "/workspace/aim-runtime/profiles/general/vllm-mi300x-fp16-tp1-latency.yaml",
    "profile": {
      "metadata": {
        "engine": "vllm",
        "gpu": "MI300X",
        "gpu_count": 1,
        "manual_selection_only": false,
        "metric": "latency",
        "precision": "fp16",
        "type": "general"
      },
      "engine_args": {
        "dtype": "float16",
        "gpu-memory-utilization": 0.95,
        "no-enable-chunked-prefill": null,
        "tensor-parallel-size": 1
      },
      "env_vars": {
        "NCCL_MIN_NCHANNELS": "112",
        "TORCH_BLAS_PREFER_HIPBLASLT": "1",
        "VLLM_DO_NOT_TRACK": "1"
      }
    },
    "models": [
      {
        "name": "meta-llama/Llama-3.1-8B-Instruct",
        "source": "hf://meta-llama/Llama-3.1-8B-Instruct",
        "size_gb": 29.93
      }
    ],
    "script":"#!/bin/bash ..."
  }
]
```

## List profiles (`list-profiles`)

Lists and categorizes all available profiles by their compatibility with the current configuration. This helps you understand which profiles are available and why certain profiles may or may not be selected.

**Options:**
- `--state <state>`: Filter profiles by compatibility state
  - `all` (default): Show all profiles
  - `compatible`: Show only profiles that can run with current configuration
  - `gpu_mismatch`: Show profiles that don't match detected GPU
  - `precision_mismatch`: Show profiles with different precision
  - `model_mismatch`: Show profiles for different models
  - `engine_mismatch`: Show profiles using different engines
  - `metric_mismatch`: Show profiles optimized for different metrics
  - `unknown`: Show profiles with unknown compatibility
- `--format <format>`: Choose output format
  - `text` (default): Human-readable grouped output by state
  - `table`: Colored table with all profiles and their states
- `--verbose` / `-v`: Enable verbose logging for debugging

This is useful for:
- Understanding which profiles are available for your model
- Debugging why a specific profile wasn't selected
- Discovering available precision/GPU/metric combinations
- Verifying profile compatibility before running

### Text Format (default)

```bash
docker run -e AIM_MODEL_ID=meta-llama/Llama-3.1-8B-Instruct \
  aim-base:0.9 \
  list-profiles
```

**Example output:**
```
AIM Profile Compatibility Report
==================================================
Model ID: meta-llama/Llama-3.1-8B-Instruct
Precision: auto
Engine: vllm
Metric: latency
GPU Count: auto

Total profiles analyzed: 12

COMPATIBLE (2 profiles):
----------------------------------------
  • vllm-mi300x-fp16-tp1-latency
    GPU: MI300X
    Precision: fp16
    Engine: vllm
    Priority: 1
  • vllm-mi300x-fp8-tp1-latency
    GPU: MI300X
    Precision: fp8
    Engine: vllm
    Priority: 1

METRIC_MISMATCH (4 profiles):
----------------------------------------
  • vllm-mi300x-fp16-tp1-throughput
    GPU: MI300X
    Precision: fp16
    Engine: vllm
    Priority: 1
```

### Table Format

```bash
docker run -e AIM_MODEL_ID=meta-llama/Llama-3.1-8B-Instruct \
  aim-base:0.9 \
  list-profiles --format table
```

Displays all profiles in a colored table showing their compatibility state at a glance.

### Filter by State

```bash
# Show only compatible profiles
docker run -e AIM_MODEL_ID=meta-llama/Llama-3.1-8B-Instruct \
  aim-base:0.9 \
  list-profiles --state compatible

# Show profiles with GPU mismatch
docker run -e AIM_MODEL_ID=meta-llama/Llama-3.1-8B-Instruct \
  -e AIM_GPU_MODEL=MI325X \
  aim-base:0.9 \
  list-profiles --state gpu_mismatch --format table
```

## Download to cache (`download-to-cache`)

Pre-downloads models to a local cache directory before running inference. This is useful for:
- Pre-warming containers during build time
- Offline deployment scenarios
- Bandwidth optimization by scheduling downloads during off-peak hours
- Verifying model availability before serving

**Key Features:**
- **Local-Dir Mode (default)**: Downloads directly to organized `{cache_dir}/org/model/` directories
- **Hugging Face Cache Mode**: Optional `--use-hf-cache` flag for HF's standard cache structure

### Basic Usage

```bash
# Download using profile selection (default: local-dir mode)
docker run --rm \
  -e AIM_MODEL_ID=meta-llama/Llama-3.1-8B-Instruct \
  -v /host/model-cache:/workspace/model-cache \
  aim-base:0.9 \
  download-to-cache

# Result: /workspace/model-cache/meta-llama/Llama-3.1-8B-Instruct/
```

### Explicit Model ID (with protocol)

Override profile selection by specifying the model ID directly with protocol:

```bash
# Download a Hugging Face model directly (local-dir mode)
docker run --rm \
  -v /host/model-cache:/workspace/model-cache \
  aim-base:0.9 \
  download-to-cache --model-id hf://mistralai/Mistral-7B-v0.1

# Result: /workspace/model-cache/mistralai/Mistral-7B-v0.1/
```

### Hugging Face Cache Mode

Use `--use-hf-cache` flag to download using Hugging Face's standard cache structure:

```bash
docker run --rm \
  -e AIM_MODEL_ID=meta-llama/Llama-3.1-8B-Instruct \
  -v /host/model-cache:/workspace/model-cache \
  aim-base:0.9 \
  download-to-cache --use-hf-cache

# Result: /workspace/model-cache/models--meta-llama--Llama-3.1-8B-Instruct/
```

### Two-Step Workflow (Download then Serve)

```bash
# Step 1: Download model
docker run --rm \
  -e AIM_MODEL_ID=meta-llama/Llama-3.1-8B-Instruct \
  -e AIM_GPU_MODEL=MI300X \
  -e HF_TOKEN=your_token \
  -v /host/model-cache:/workspace/model-cache \
  aim-base:0.9 \
  download-to-cache

# Step 2: Serve with pre-cached model
docker run \
  -e AIM_MODEL_ID=meta-llama/Llama-3.1-8B-Instruct \
  -v /host/model-cache:/workspace/model-cache \
  --device=/dev/kfd --device=/dev/dri \
  -p 8000:8000 \
  aim-base:0.9
```

**Note:** The actual model downloaded may differ from `AIM_MODEL_ID` if the selected profile specifies a quantized variant.

**See [Model Caching Documentation](model_caching.md) for comprehensive details on downloads, cache formats, and runtime behavior.**
