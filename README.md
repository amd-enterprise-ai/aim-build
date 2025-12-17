<!--
Copyright © Advanced Micro Devices, Inc., or its affiliates.

SPDX-License-Identifier: MIT
-->

# AIM Build: AMD Inference Microservice Containers

This repository contains the tools and profiles to build AMD Inference Microservice (AIM) containers. AIM provides a standardized, production-ready framework for serving AI models on AMD Instinct™ GPUs.
High-level overview can be found [here](docs/overview.md).

## What It Does

* **Standardized Containers**: Builds portable inference microservices for AMD GPUs.
* **Validated Profiles**: Uses YAML profiles to configure models for specific hardware, ensuring optimal performance for different precision formats (FP16, BF16, FP8) and tensor parallel layouts.
* **Intelligent Configuration**: Automatically detects hardware and selects the best profile for the given GPU count and precision.
* **Multiple Engines**: Supports multiple inference engines, starting with vLLM.
* **Model Caching**: Integrates with external caches to accelerate model loading and reduce network usage.
* **Observability**: Provides logging and metrics for monitoring and diagnostics.

## How It Works

The core of AIM is a **profile-driven system**.

1.  **Profile System**: A collection of YAML files defines optimized configurations for various models and hardware. These profiles specify e.g. the inference engine, data types, and command-line arguments.
2.  **Runtime Logic**: An entrypoint script inside the container detects the host environment (like GPU count) and selects the most appropriate profile.
3.  **Command Generation**: The selected profile is used to generate the final command and environment variables needed to launch the inference server.

Profiles are chosen automatically based on the provided parameters such as:
* Model ID (e.g., `meta-llama/Llama-3.1-8B-Instruct`)
* Precision (e.g., `fp16`, `bf16`, `fp8`)
* Engine (e.g., `vllm`)
* Metric (e.g., `latency`, `throughput`)
* GPU count (e.g., `1`, `2`, `4`, `8`)
* GPU architecture (e.g., `MI300X`)

It is possible to bypass automatic selection and specify a particular profile directly using an environment variable.

## Container Build Patterns

AIM uses a two-tiered approach to container images:

1.  **Generic Base Container (`aim-base`)**: A single, universal image that can run any supported model. The model is downloaded at runtime.
2.  **Model-Specific Container (`aim`)**: An extension of the base image that includes optimized profiles for a particular model. This ensures the best possible performance.

## Quick Start

### Prerequisites

*   AMD GPU with ROCm support (e.g., MI300X).
*   Docker installed and running.

### Build the Base Container

```bash
make build-base
```

### Build a Model-Specific Container

To build a container for a specific model, such as `meta-llama/Llama-3.1-8B-Instruct` , use the following command:

```bash
make build-model ORG=meta-llama MODEL=Llama-3.1-8B-Instruct
```

### Other deployment scenarios

For more options on deploying AIMs please refer to the [Deployment Overview](docs/deployment_overview.md) and
[Kubernetes Deployment](docs/kubernetes_deployment.md) documentation.

## Environment Variables

AIM containers support the following environment variables:

### Required

* `AIM_MODEL_ID`: **Required for base container (`aim-base`) only.** The Hugging Face model identifier to deploy (e.g., `meta-llama/Llama-3.1-8B-Instruct`).

### Optional

* `HF_TOKEN`: Hugging Face access token for gated models
* `AIM_PRECISION`: Precision format (`auto`, `fp4`, `fp8`, `fp16`, `fp32`, `bf16`, `int4`, `int8`, default: `auto`)
* `AIM_GPU_COUNT`: Number of GPUs to use (`auto` or specific number from `0` to `8`)
* `AIM_GPU_MODEL`: Override detected GPU model (e.g., `MI300X`, `MI325X`, `MI350X`, `MI355X`, ...). Use this when automatic GPU detection is not available or when you want to force a specific GPU profile.
* `AIM_ENGINE`: Inference engine (`vllm`)
* `AIM_METRIC`: Optimization metric (`latency`,   `throughput`)
* `AIM_PROFILE_ID`: Specific profile to use (overrides automatic selection)
* `AIM_ALLOW_GENERAL_PROFILE_FALLBACK`: Allow automatic selection of general profiles (`true`/`false`, default: `true` for base containers, `false` for model-specific containers). When `false`, general profiles are still loaded but marked as manual-selection-only.
* `AIM_CACHE_PATH`: Directory for model caching (default: `/workspace/model-cache`)
* `AIM_LOG_LEVEL_ROOT`: Log level for root logger controlling third-party packages (`DEBUG`, `INFO`, `WARNING`, `ERROR`, `CRITICAL`, default: `WARNING`)
* `AIM_LOG_LEVEL`: Log level for AIM runtime packages (`DEBUG`, `INFO`, `WARNING`, `ERROR`, `CRITICAL`, default: `INFO`)
* `AIM_PORT`: Port for accessing inference API (numeric value up to `65535`, default: `8000`)
* `AIM_ENGINE_ARGS`: Override or add engine-specific arguments as JSON (see [Engine Arguments Override](docs/engine-args-override.md) for details)

### S3 Storage (for S3-hosted models)

When using models stored in S3 or S3-compatible storage (e.g., MinIO, Ceph), the following AWS environment variables are required for boto3:

* `AWS_ACCESS_KEY_ID`: AWS access key (optional if using IAM role)
* `AWS_SECRET_ACCESS_KEY`: AWS secret key (optional if using IAM role)
* `AWS_DEFAULT_REGION`: AWS region (default: `us-east-1`)
* `AWS_ENDPOINT_URL`: Custom S3 endpoint for S3-compatible storage like MinIO or Ceph (optional, only needed for non-AWS S3)

**Example with S3:**
```bash
docker run \
  -e AIM_MODEL_ID=s3://my-bucket/models/meta-llama/Llama-3.1-8B-Instruct \
  -e AWS_ACCESS_KEY_ID=your_access_key \
  -e AWS_SECRET_ACCESS_KEY=your_secret_key \
  -e AWS_DEFAULT_REGION=us-west-2 \
  --device=/dev/kfd --device=/dev/dri \
  -p 8000:8000 \
  aim-base:0.9
```

**Example with MinIO:**
```bash
docker run \
  -e AIM_MODEL_ID=s3://models/meta-llama/Llama-3.1-8B-Instruct \
  -e AWS_ACCESS_KEY_ID=minioadmin \
  -e AWS_SECRET_ACCESS_KEY=minioadmin \
  -e AWS_ENDPOINT_URL=https://minio.example.com:9000 \
  --device=/dev/kfd --device=/dev/dri \
  -p 8000:8000 \
  aim-base:0.9
```

### Engine Arguments Override

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

Arguments are validated against the engine's schema and merged with profile defaults. User-provided arguments override profile defaults, while system arguments (like `port`) always take precedence.

**See [Engine Arguments Override Documentation](docs/engine-args-override.md)** for complete usage guide, validation details, and examples.

### Model Caching

AIM supports two model cache formats within `AIM_CACHE_PATH`:

1. **HuggingFace Hub Cache** (default): Models are cached in `AIM_CACHE_PATH/hub/` using HuggingFace's standard cache format
2. **Local Directory Format**: Models stored directly as `AIM_CACHE_PATH/org/model/` (e.g., `/workspace/model-cache/meta-llama/Llama-3.1-8B-Instruct`)

**Cache Resolution Order:**
1. **Local directory first**: If a model exists at `AIM_CACHE_PATH/org/model/`, it's loaded directly and `--served-model-name` is set to `org/model`
2. **HuggingFace fallback**: Otherwise, the model_id is used and HuggingFace handles cache lookup or download transparently

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

### Logging Configuration

AIM provides fine-grained logging control with separate log levels for AIM runtime and third-party packages:

- **`AIM_LOG_LEVEL_ROOT`** (default: `WARNING`): Controls the root logger level, affecting third-party packages and external libraries. This helps reduce noise from dependency logging.

- **`AIM_LOG_LEVEL`** (default: `INFO`): Controls the `aim_runtime` package logger level. This allows you to see AIM operational messages while keeping third-party logs quiet.

Both variables accept standard Python logging levels: `DEBUG`, `INFO`, `WARNING`, `ERROR`, `CRITICAL`.

**Usage Examples:**

```bash
# Production (default): Show AIM info but suppress third-party warnings
docker run \
  -e AIM_LOG_LEVEL_ROOT=WARNING \
  -e AIM_LOG_LEVEL=INFO \
  -e AIM_MODEL_ID=meta-llama/Llama-3.1-8B-Instruct \
  aim-base:0.9

# Debugging: Show all AIM details but only errors from dependencies
docker run \
  -e AIM_LOG_LEVEL_ROOT=ERROR \
  -e AIM_LOG_LEVEL=DEBUG \
  -e AIM_MODEL_ID=meta-llama/Llama-3.1-8B-Instruct \
  aim-base:0.9

# Maximum verbosity: Show everything
docker run \
  -e AIM_LOG_LEVEL_ROOT=DEBUG \
  -e AIM_LOG_LEVEL=DEBUG \
  -e AIM_MODEL_ID=meta-llama/Llama-3.1-8B-Instruct \
  aim-base:0.9

# Minimal logging: Only critical errors
docker run \
  -e AIM_LOG_LEVEL_ROOT=CRITICAL \
  -e AIM_LOG_LEVEL=WARNING \
  -e AIM_MODEL_ID=meta-llama/Llama-3.1-8B-Instruct \
  aim-base:0.9
```

**Detailed logging includes:**
- Complete configuration dump with all environment variables (DEBUG level)
- Detailed profile selection criteria and filtering steps (DEBUG level)
- Generated command script contents for verification (DEBUG level)
- GPU detection and hardware configuration details (DEBUG level)
- Operational status messages (INFO level)
- Warnings and errors (WARNING/ERROR levels)

### GPU Model Override

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

## Running Containers with Gated Models

Some models (like those from Meta's Llama family) are gated and require a Hugging Face token for access. AIM containers support this through the `HF_TOKEN` environment variable.

### Getting a Hugging Face Token

1. Create an account at [Hugging Face](https://huggingface.co/)
2. Go to your [Settings > Access Tokens](https://huggingface.co/settings/tokens)
3. Create a new token with read permissions
4. Request access to the specific gated model (e.g., `meta-llama/Llama-3.1-8B-Instruct`)

### Running with HF_TOKEN

For the base container:

```bash
docker run -e HF_TOKEN=your_token_here -e AIM_MODEL_ID=meta-llama/Llama-3.1-8B-Instruct aim-base:0.9
```

For model-specific containers:

```bash
docker run -e HF_TOKEN=your_token_here aim:0.3.0-meta-llama-llama-3.1-8b-instruct-v20250930
```

### Running with GPUs shared with container

Sharing GPU with container may be needed to run the models. Also, the port mapping is needed to access the inference service.

```bash
docker run -e HF_TOKEN=your_token_here -p 8000:8000 --device=/dev/kfd --device=/dev/dri -t aim:0.3.0-meta-llama-llama-3.1-8b-instruct-v20250930
```

### Security Note

* Never include HF_TOKEN in your Dockerfiles or commit it to version control
* Use environment variables or secrets management systems to provide the token at runtime
* The token is only used during model download and inference startup

## Custom Profiles

AIM supports custom profiles through a convention-based directory structure. Custom profiles allow you to override built-in configurations or add support for new models without modifying the container image.

### Custom Profile Location

Custom profiles are automatically discovered in the `custom/` subdirectory within the profile directory:

```
/workspace/aim-runtime/profiles/custom/
```

### Profile Search Order

Profiles are searched in the following order of precedence:

1. **Custom profiles** (`/workspace/aim-runtime/profiles/custom/`)
2. **Model-specific profiles** (`/workspace/aim-runtime/profiles/{org}/{model}/`)
3. **General profiles** (`/workspace/aim-runtime/profiles/general/`)

Custom profiles take the highest precedence, allowing you to override any built-in configuration.

### Using Custom Profiles

To use custom profiles, mount a directory containing your profile YAML files to the custom profile location:

```bash
docker run \
  -e AIM_MODEL_ID=meta-llama/Llama-3.1-8B-Instruct \
  -v /host/path/to/custom-profiles:/workspace/aim-runtime/profiles/custom \
  --device=/dev/kfd --device=/dev/dri \
  -p 8000:8000 \
  aim-base:0.9
```

### Custom Profile Example

Create a custom profile at `/host/path/to/custom-profiles/vllm-mi300x-fp8-tp1-latency.yaml`:

```yaml
metadata:
  engine: vllm
  gpu: MI300X
  precision: fp8
  gpu_count: 1
  metric: latency
  manual_selection_only: false
  type: optimized

engine_args:
  gpu-memory-utilization: 0.98
  dtype: float8
  tensor-parallel-size: 1
  max-model-len: 16384

env_vars:
  VLLM_FP8_PADDING: "1"
```

This custom profile will be selected over built-in profiles when matching criteria are met.

## AIM Runtime CLI

The AIM runtime provides a command-line interface with the following subcommands:

### `serve` (default)

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

### `dry-run`

Performs profile selection and displays the selected profile without starting the server. Supports two output formats:

**Options:**
- `--format yaml` (default): Display the complete profile as YAML text
- `--format json`: Display the selected profile as structured JSON

This is useful for:
- Verifying which profile will be selected for your configuration
- Debugging profile selection issues
- Understanding the full configuration before execution
- Programmatic integration with CI/CD pipelines (JSON format for easy parsing)

#### YAML Format (default)

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

model: meta-llama/Llama-3.1-8B-Instruct
precision: fp16
gpu_count: 1
metadata:
  engine: vllm
  gpu: MI300X
  precision: fp16
  gpu_count: 1
  metric: latency
  manual_selection_only: false
  type: unoptimized
env_vars:
  VLLM_WORKER_MULTIPROC_METHOD: spawn
engine_args:
  max-model-len: 4096
  tensor-parallel-size: 1
```

#### JSON Format

Returns the selected profile as a structured JSON object with the profile filename as key and parsed profile data as value. Useful for:
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
{
  "vllm-mi300x-fp16-tp1-latency.yaml": {
    "model": "meta-llama/Llama-3.1-8B-Instruct",
    "precision": "fp16",
    "gpu_count": 1,
    "metadata": {
      "engine": "vllm",
      "gpu": "MI300X",
      "precision": "fp16",
      "gpu_count": 1,
      "metric": "latency",
      "manual_selection_only": false,
      "type": "unoptimized"
    },
    "env_vars": {
      "VLLM_WORKER_MULTIPROC_METHOD": "spawn"
    },
    "engine_args": {
      "max-model-len": 4096,
      "tensor-parallel-size": 1
    }
  }
}
```

### `list-profiles`

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

#### Text Format (default)

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

#### Table Format

```bash
docker run -e AIM_MODEL_ID=meta-llama/Llama-3.1-8B-Instruct \
  aim-base:0.9 \
  list-profiles --format table
```

Displays all profiles in a colored table showing their compatibility state at a glance.

#### Filter by State

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

### `download-to-cache`

Pre-downloads models to a local cache directory before running inference. This is useful for:
- Pre-warming containers during build time
- Offline deployment scenarios
- Bandwidth optimization by scheduling downloads during off-peak hours
- Verifying model availability before serving

**Key Features:**
- **Local-Dir Mode (default)**: Downloads directly to organized `{cache_dir}/org/model/` directories
- **HuggingFace Cache Mode**: Optional `--use-hf-cache` flag for HF's standard cache structure
- **S3 Support**: Download models from S3 or S3-compatible storage
- **Custom Naming**: Use `--model-name` to specify custom directory names for S3 models

#### Basic Usage

```bash
# Download using profile selection (default: local-dir mode)
docker run --rm \
  -e AIM_MODEL_ID=meta-llama/Llama-3.1-8B-Instruct \
  -v /host/model-cache:/workspace/model-cache \
  aim-base:0.9 \
  download-to-cache

# Result: /workspace/model-cache/meta-llama/Llama-3.1-8B-Instruct/
```

#### Explicit Model ID (with protocol)

Override profile selection by specifying the model ID directly with protocol:

```bash
# Download a HuggingFace model directly (local-dir mode)
docker run --rm \
  -v /host/model-cache:/workspace/model-cache \
  aim-base:0.9 \
  download-to-cache --model-id hf://mistralai/Mistral-7B-v0.1

# Result: /workspace/model-cache/mistralai/Mistral-7B-v0.1/

# Download from S3 with automatic naming
docker run --rm \
  -v /host/model-cache:/workspace/model-cache \
  aim-base:0.9 \
  download-to-cache --model-id s3://bucket/models/org/llama-3.1-8b

# Result: /workspace/model-cache/org/llama-3.1-8b/

# Download from S3 with custom name
docker run --rm \
  -v /host/model-cache:/workspace/model-cache \
  aim-base:0.9 \
  download-to-cache --model-id s3://bucket/path/to/model --model-name custom-org/my-model

# Result: /workspace/model-cache/custom-org/my-model/
```

#### HuggingFace Cache Mode

Use `--use-hf-cache` flag to download using HuggingFace's standard cache structure:

```bash
docker run --rm \
  -e AIM_MODEL_ID=meta-llama/Llama-3.1-8B-Instruct \
  -v /host/model-cache:/workspace/model-cache \
  aim-base:0.9 \
  download-to-cache --use-hf-cache

# Result: /workspace/model-cache/models--meta-llama--Llama-3.1-8B-Instruct/
```

#### Two-Step Workflow (Download then Serve)

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

**See [Model Caching Documentation](docs/model_caching.md) for comprehensive details on downloads, cache formats, and runtime behavior.**

## Development

### Setup Development Environment

```bash
# Create and activate virtual environment
python -m venv .venv
source .venv/bin/activate

# Install development dependencies
make dev-setup
```

### Running Tests

```bash
# Run unit tests only (default)
make test

# Run integration tests (requires GPU/ROCm environment)
make test-integration

# Run all tests (unit + integration)
make test-all

# Run tests with coverage report
make test-cov

# Run tests with coverage and open HTML report
make test-cov-open

# Run pre-commit hooks (linting, formatting, etc.)
make lint
```

#### Test Types

* **Unit Tests**: Fast tests that run without hardware dependencies (default)
* **Integration Tests**: Tests that require AMD GPU hardware and ROCm drivers
  + Marked with `@pytest.mark.integration`
  + Automatically skipped unless running in GPU environment
  + Use `make test-integration` or `pytest -m integration` to run explicitly


### Test Configuration

The project uses pytest with the following features:
* **Coverage reporting**: Generates HTML and XML coverage reports
* **Test markers**: Use `@pytest.mark.slow` for slow tests,  `@pytest.mark.integration` for integration tests
* **Pre-commit integration**: Tests run automatically on git commits
* **Configuration**: See `pyproject.toml` for pytest settings

### Available Make Targets

#### Container Build & Management
- `make build` - Build both base and model-specific containers
- `make build-base` - Build the base AIM container
- `make build-model` - Build model-specific container (requires ORG and MODEL variables)
- `make tag` - Tag containers for registry push
- `make push` - Push containers to registry
- `make clean` - Clean up Docker images and containers

#### Testing & Quality
- `make test` - Run unit tests only (fast, no GPU required)
- `make test-integration` - Run integration tests (requires GPU/ROCm environment)
- `make test-all` - Run all tests (unit + integration)
- `make test-cov` - Run unit tests with coverage reporting
- `make test-cov-open` - Run tests and open coverage report in browser
- `make lint` - Run pre-commit hooks (formatting, linting, etc.)

#### Development Setup
- `make dev-setup` - Install development dependencies and pre-commit hooks

### Versioning

Versioning for containers and for the codebase is tied together. Versioning for model-specific containers follows full
semantic versioning. Base images use MAJOR.MINOR (e.g., `0.4`), while model-specific images use MAJOR.MINOR.PATCH (e.g., `0.4.2`).
The versioning is based on the `pyproject.toml` file which contains only MAJOR.MINOR. Patch versions for model-specific images
are automatically determined from the registry. Version suffixes indicate the release stage:

1. `x.y-rcN` or `x.y.z-rcN` - Release Candidate (N auto-increments with each push)
2. `x.y-preview` or `x.y.z-preview` - Preview Release (default for PRs to main)
3. `x.y` or `x.y.z` - Official Release (requires source branch starting with `release/`)

**Version bumping guidelines:**

Update the version in `pyproject.toml` (MAJOR.MINOR only) when making changes to:
- Base image (vLLM version updates)
- Runtime code (src/aim_runtime/)
- General profiles (profiles/general/)
- Breaking API or architectural changes

Do NOT bump the version for model-specific profile changes - patch versions auto-increment per model.

**CI Workflow:**
- Feature branches: Each push creates a new `-rcN` version (N auto-increments)
- PR to main: Creates a `-preview` version by default
- PR from `release/` branch to main: Creates official version (no suffix)

If the target version already exists in the registry, the build will fail and you must bump the version in `pyproject.toml`.

## Documentation

For a detailed explanation of the architecture, see the [AIM Container Technical Architecture](docs/aim_architecture.md) document.
