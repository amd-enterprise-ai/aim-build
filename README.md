<!--
Copyright © Advanced Micro Devices, Inc., or its affiliates.

SPDX-License-Identifier: MIT
-->

# AIM Build: AMD Inference Microservice Containers

This repository contains the implementation of AMD Inference Microservice (AIM), profiles, and build tools. AIM provides
a standardized, production-ready framework for serving AI models on AMD Instinct™ GPUs. High-level AIM's overview can be
found [here](docs/overview.md).

## What It Does

* **Standardized Containers**: Builds portable inference microservices for AMD GPUs.
* **Validated Profiles**: Uses YAML profiles to configure models for specific hardware, ensuring optimal performance for different precision formats (FP16, BF16, FP8, etc.) and tensor parallel layouts.
* **Intelligent Configuration**: Automatically detects hardware and selects the best profile for the given GPU count and precision.
* **Multiple Engines**: Supports multiple inference engines, starting with vLLM.
* **Model Caching**: Integrates with external caches to accelerate model loading and reduce network usage.
* **Observability**: Provides logging and metrics for monitoring and diagnostics.

## How It Works

The core of AIM is a **profile-driven system**.

1.  **Profile Registry**: A collection of YAML files each of which defines optimized configurations for various models and hardware. These profiles specify e.g. the inference engine and its parameters, recommended deployments and environment variables.
2.  **Runtime Logic**: An entrypoint script inside the container detects characteristics of the host environment (e.g. GPU model and count) and selects the most appropriate profile.
3.  **Command Generation**: The selected profile is used to generate the final command and set environment variables needed to launch the inference server.

Profiles are chosen automatically based on the provided parameters such as:
* Model ID (e.g., `meta-llama/Llama-3.1-8B-Instruct`)
* Engine (e.g., `vllm`)
* GPU model (e.g., `MI300X`)
* GPU count (e.g., `1`, `2`, `4`, `8`)
* Metric (e.g., `latency`, `throughput`)
* Precision (e.g., `fp16`, `bf16`, `fp8`)

It is possible to bypass automatic selection and specify a particular profile directly using `AIM_PROFILE_ID`
environment variable.

## Container Build Patterns

AIM uses a two-tiered approach to building images:

1.  **Generic Base Container (`aim-base`)**: A single, universal image that can run any supported model. The model is chosen at runtime. This provides maximum flexibility for deploying different models.
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

### Run the Container

The following command runs the model-specific container with GPU support and port mapping. It does set any environment
variables and relies on their default values:

```bash
docker run \
  -e HF_TOKEN=<YOUR_HUGGINGFACE_TOKEN> \
  --device=/dev/kfd --device=/dev/dri \
  -p 8000:8000 \
  amdenterpriseai/aim-meta-llama-llama-3-1-8b-instruct:0.8.5
```

Sharing GPU with container is needed to run the models on GPU. Also, the port mapping is needed to access the inference
service. Sharing GPU is achieved by adding the following flags to `docker run` command: `--device=/dev/kfd --device=/dev/dri`.
Alternatively, it can be done by adding `--runtime=amd –gpus 1` but it will require very recent [Docker version and AMD Conatiner Toolkit installed](https://instinct.docs.amd.com/projects/container-toolkit/en/latest/container-runtime/quick-start-guide.html#using-gpus-flag-with-docker-28-x).

For more examples on how to run AIMs on Docker see the [Docker Deployment](docs/docker_deployment.md) documentation.

### Other deployment scenarios

For more options on deploying AIMs please refer to the [Deployment Overview](docs/deployment_overview.md),
[Kubernetes Deployment](docs/kubernetes_deployment.md), and [KServe Deployment](docs/kserve_deployment.md) documentation.

## Environment Variables

AIM containers support the following environment variables:

### Required

* `AIM_MODEL_ID`: **Required for base container (`aim-base`) only.** The Hugging Face model identifier to deploy (e.g., `meta-llama/Llama-3.1-8B-Instruct`).

### Optional

* `HF_TOKEN`: Hugging Face access token for gated models
* `AIM_PRECISION`: Precision format (`auto`, `fp4`, `fp8`, `fp16`, `fp32`, `bf16`, `int4`, `int8`, default: `auto`)
* `AIM_GPU_COUNT`: Number of GPUs to use (`auto` or specific number from `0` to `8`)
* `AIM_GPU_MODEL`: Override detected GPU model (e.g., `MI300X`, `MI325X`, `MI350X`, `MI355X`, ...) or device id (e.g. `0x74a1`, `0x74a5`, `0x75a0`, `0x75a3`, ...). Use this when automatic GPU detection is not available or when you want to force a specific GPU profile.
* `AIM_ENGINE`: Inference engine (`vllm`)
* `AIM_METRIC`: Optimization metric (`latency`, `throughput`)
* `AIM_PROFILE_ID`: Specific profile to use (overrides automatic selection)
* `AIM_ALLOW_GENERAL_PROFILE_FALLBACK`: Allow automatic selection of general profiles (`true`/`false`, default: `true` for base containers, `false` for model-specific containers). When `false`, general profiles are still loaded but marked as manual-selection-only.
* `AIM_ALLOW_UNOPTIMIZED`: Allow automatic fallback to unoptimized profiles when no optimized or preview profiles are auto-selectable for the current hardware (`true`/`false`, default: `false`).
* `AIM_CACHE_PATH`: Directory for model caching (default: `/workspace/model-cache`)
* `AIM_LOG_LEVEL_ROOT`: Log level for root logger controlling third-party packages (`DEBUG`, `INFO`, `WARNING`, `ERROR`, `CRITICAL`, default: `WARNING`)
* `AIM_LOG_LEVEL`: Log level for AIM runtime packages (`DEBUG`, `INFO`, `WARNING`, `ERROR`, `CRITICAL`, default: `INFO`). See [Logging Configuration](docs/logging_configuration.md) for more details
* `AIM_PORT`: Port for accessing inference API (numeric value up to `65535`, default: `8000`)
* `AIM_ENGINE_ARGS`: Override or add engine-specific arguments as JSON (see [Engine Arguments Override](docs/engine-args-override.md) for details)

Please refer to [Environment Variables](docs/environment_variables.md) document for detailed examples of the usage of
the environment variables. The document covers the following topics:
* Gated models support
* Engine Arguments Override
* GPU Model Override
* Model Caching

## Running Containers with Gated Models

Some models (like those from Meta's Llama family) are gated and require a Hugging Face token for access. AIM containers support this through the `HF_TOKEN` environment variable.

### Getting a Hugging Face Token

1. Create an account at [Hugging Face](https://huggingface.co/)
2. Go to your [Settings > Access Tokens](https://huggingface.co/settings/tokens)
3. Create a new token with read permissions
4. Request access to the specific gated model (e.g., `meta-llama/Llama-3.1-8B-Instruct`)

Obtained token can then be passed to the container using the `HF_TOKEN` environment variable.

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

Custom profiles take the highest precedence, allowing you to override any built-in configuration. Profile selection
logic can be bypassed by setting `AIM_PROFILE_ID` to the desired profile filename. See section 3.6 in
[AIM Container Technical Architecture](docs/aim_architecture.md#36-controlling-profile-selection) for more details.

## AIM Runtime CLI

The AIM runtime provides a command-line interface with the following subcommands:

| Command             | Purpose                                                                                                                                                                                                         |
|---------------------|-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| `serve` (default)   | Performs profile selection and starts the inference server. This is the default behavior when no subcommand is specified.                                                                                       |
| `dry-run`           | Performs profile selection and displays the selected profile without starting the server. Supports two output formats: YAML and JSON.                                                                           |
| `list-profiles`     | Lists and categorizes all available profiles by their compatibility with the current configuration. This helps you understand which profiles are available and why certain profiles may or may not be selected. |
| `download-to-cache` | Pre-downloads models to a local cache directory before running inference.                                                                                                                                       |

See more details in the [AIM Runtime CLI Documentation](docs/cli.md).

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

Versioning for containers and for the codebase is tied together. Versioning for model-specific containers follows partial
semantic versioning. It is partial because `rc` and `preview` versions do not follow alphabetical order and this should
be considered when comparing versions. Base images use MAJOR.MINOR (e.g., `0.4`), while model-specific images use
MAJOR.MINOR.PATCH (e.g., `0.4.2`). The versioning is based on the `pyproject.toml` file which contains only MAJOR.MINOR.
Patch versions for model-specific images are automatically determined from the registry. Version suffixes indicate the
release stage:

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
