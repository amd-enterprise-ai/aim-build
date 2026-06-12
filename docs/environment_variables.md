<!--
Copyright © Advanced Micro Devices, Inc., or its affiliates.

SPDX-License-Identifier: MIT
-->

# Environment Variables

Along with profiles, environment variables are a key mechanism to configure AIM containers at runtime. They allow
you to specify model IDs, authentication tokens, engine parameters, and more. The examples below use `aim-base` (Instinct); for Radeon Pro use `aim-radeon-base`, and for EPYC use `aim-epyc-base`.

## Gated models support (`HF_TOKEN`)

This variable is needed for running gated models hosted on Hugging Face Hub. It should be set to your Hugging Face access
token.

### Running with `HF_TOKEN`

For the base container:

```bash
docker run -e HF_TOKEN=your_token_here \
  -e AIM_MODEL_ID=meta-llama/Llama-3.1-8B-Instruct \
  --device=/dev/kfd --device=/dev/dri \
  aim-base:0.9
```

For a model-specific container:

```bash
docker run -e HF_TOKEN=your_token_here \
  --device=/dev/kfd --device=/dev/dri \
  aim-meta-llama-llama-3-1-8b-instruct:0.11.1
```

### Security Note

* Never include `HF_TOKEN` in your Dockerfiles or commit it to version control
* Use environment variables or secrets management systems to provide the token at runtime
* The token is only used during model download and inference startup

## Engine Arguments Override (`AIM_ENGINE_ARGS`)

You can customize inference engine behavior at runtime by passing additional arguments via the `AIM_ENGINE_ARGS` environment variable. This allows you to override profile defaults or add new arguments without modifying profiles.

**Example:**
```bash
docker run \
  -e AIM_MODEL_ID=meta-llama/Llama-3.1-8B-Instruct \
  -e AIM_ENGINE_ARGS='{"max-model-len": 8192, "gpu-memory-utilization": 0.85}' \
  --device=/dev/kfd --device=/dev/dri \
  -p 8000:8000 \
  aim-base:0.9
```

Arguments are validated against the engine's argument definitions and merged with values provided by a profile.

**See [Engine Arguments Override Documentation](./engine-args-override.md)** for complete usage guide, validation details, and examples.

## GPU Model Override (`AIM_GPU_MODEL`)

The `AIM_GPU_MODEL` environment variable allows you to override automatic GPU detection. This is useful in scenarios where:
- Running in containers without GPU access during the selection phase
- Testing profiles for different GPU models
- Working in environments where GPU detection is not available

**Example usage:**
```bash
# Override GPU detection to use MI325X profiles
docker run \
  -e AIM_MODEL_ID=meta-llama/Llama-3.1-8B-Instruct \
  -e AIM_GPU_MODEL=MI325X \
  -e AIM_GPU_COUNT=2 \
  --device=/dev/kfd --device=/dev/dri \
  -p 8000:8000 \
  aim-base:0.9
```

**Note:** When using `AIM_GPU_MODEL`, it's recommended to also explicitly set `AIM_GPU_COUNT` to ensure the correct profile is selected.

## Model Caching (`AIM_CACHE_PATH`)

AIM supports two model cache formats within `AIM_CACHE_PATH`:

1. **Hugging Face Hub Cache** (default): Models are cached in `AIM_CACHE_PATH/hub/` using Hugging Face's standard cache format
2. **Local Directory Format**: Models stored directly as `AIM_CACHE_PATH/org/model/` (e.g., `/workspace/model-cache/meta-llama/Llama-3.1-8B-Instruct`)

**Cache Resolution Order:**
1. **Local directory first**: If a model exists at `AIM_CACHE_PATH/org/model/`, it's loaded directly and `--served-model-name` is set to `org/model`
2. **Hugging Face fallback**: Otherwise, the model_id is used and Hugging Face handles cache lookup or download transparently

**Example: Using Local Directory Format**
```bash
# Pre-populate local directory format
mkdir -p /workspace/model-cache/meta-llama/Llama-3.1-8B-Instruct
# Copy model files to /workspace/model-cache/meta-llama/Llama-3.1-8B-Instruct/

# Run container - will use local directory model
docker run -e AIM_MODEL_ID=meta-llama/Llama-3.1-8B-Instruct \
  -v /host/model-cache:/workspace/model-cache \
  --device=/dev/kfd --device=/dev/dri \
  -p 8000:8000 \
  aim-base:0.9
```

For more details on model caching, see [Model Caching Documentation](./model_caching.md).
