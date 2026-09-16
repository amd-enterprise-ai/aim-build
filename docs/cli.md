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
  aim-base:0.11

# Or explicitly specify serve
docker run -e AIM_MODEL_ID=meta-llama/Llama-3.1-8B-Instruct \
  --device=/dev/kfd --device=/dev/dri \
  -p 8000:8000 \
  aim-base:0.11 \
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
  aim-base:0.11 \
  dry-run

# Or explicitly specify --format yaml
docker run -e AIM_MODEL_ID=meta-llama/Llama-3.1-8B-Instruct \
  aim-base:0.11 \
  dry-run --format yaml
```

**Example output:**
```yaml
# Selected profile: /workspace/aim-runtime/profiles/meta-llama/Llama-3.1-8B-Instruct/vllm-mi300x-fp16-tp1-latency.yaml

- filename: vllm-mi300x-fp16-tp1-latency.yaml
  path: /workspace/aim-runtime/profiles/general/vllm-mi300x-fp16-tp1-latency.yaml
  profile:
    metadata:
      accelerator_count: 1
      accelerator_model: MI300X
      accelerator_type: gpu
      engine: vllm
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
  aim-base:0.11 \
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
        "accelerator_count": 1,
        "accelerator_model": "MI300X",
        "accelerator_type": "gpu",
        "engine": "vllm",
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
  aim-base:0.11 \
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
  aim-base:0.11 \
  list-profiles --format table
```

Displays all profiles in a colored table showing their compatibility state at a glance.

### Filter by State

```bash
# Show only compatible profiles
docker run -e AIM_MODEL_ID=meta-llama/Llama-3.1-8B-Instruct \
  aim-base:0.11 \
  list-profiles --state compatible

# Show profiles with GPU mismatch
docker run -e AIM_MODEL_ID=meta-llama/Llama-3.1-8B-Instruct \
  -e AIM_GPU_MODEL=MI325X \
  aim-base:0.11 \
  list-profiles --state gpu_mismatch --format table
```

## Validate (`validate`)

Runs the image's check suite and reports each check as part of a `HarnessResult`. Exit code is `0` when every check that ran passed, `1` otherwise.

Checks belong to one of two scopes. `runtime` checks read the profile only, need no service and take milliseconds; `offline` checks need something to talk to. Both run by default. With `--service-url` the offline checks go to that service; without it, and only when the `offline` scope is selected, the CLI starts the model server for the profile being validated and shuts it down again afterwards.

**Options:**
- `--profile <name>`: Profile name. Defaults to the auto-detected profile.
- `--service-url <url>`: URL of a running service. Omit to have the CLI start and stop the server itself.
- `--scope <runtime|offline>` (default: both): Repeat to select several, e.g. `--scope runtime --scope offline`.
- `--config <path>`: Config YAML with per-run overrides, including warmup's own `max_warmup_time` budget.
- `--timeout <seconds>` (default: `300`): Budget for the service to become ready, and the per-request timeout of each check.
- `--output-format <json|table|ci>` (default: `json`).

**Checks** (vLLM images; `list-checks` prints the catalog for the image at hand):

| Check | Scope | Verifies |
|-------|-------|----------|
| `profile_schema` | runtime | The profile validates against the Pydantic schema |
| `engine_validation` | runtime | `engine_args` are accepted by vLLM's own argument parser |
| `api_health` | offline | The service is up and serving a model |
| `warmup` | offline | The first inference succeeds |
| `completions_endpoint` | offline | `/v1/completions` is OpenAI-compatible |
| `chat_completions_endpoint` | offline | `/v1/chat/completions` is OpenAI-compatible |
| `tool_invocation` | offline | Tool/function calling works |
| `tool_avoidance` | offline | Ordinary chat does not leak tool calls |
| `structured_output` | offline | Constrained decoding, flat schema |
| `structured_output_nested` | offline | Constrained decoding, nested schema |
| `structured_output_choice` | offline | Constrained decoding, choice from a fixed set |
| `reasoning` | offline | Reasoning prompts produce output |

The last six run only when the profile declares the matching capability under `metadata.capabilities` (`tool_calling`, `structured_outputs`, `reasoning`), and are reported as skipped otherwise: `validate` returns a single exit code, so it must not fail a model for a capability it never claimed.

The offline checks run in dependency order and stop early rather than pile up timeouts: nothing is attempted before the service reports a model, the behavioural checks are skipped if warmup fails, and the capability checks are skipped if either endpoint check fails.

```bash
# Profile-only validation. No service, no GPU, milliseconds.
docker run --rm aim-base:0.11 validate --scope runtime

# Full suite against a running service
python3 /workspace/entrypoint.py validate --service-url http://localhost:8000 --output-format table

# Full suite with the CLI starting and stopping the server itself
docker run --rm \
  -e AIM_MODEL_ID=meta-llama/Llama-3.1-8B-Instruct \
  --device=/dev/kfd --device=/dev/dri \
  aim-base:0.11 \
  validate
```

## List checks (`list-checks`)

Prints the checks this image can run, with each one's result type and scope. Reads the profile only — no service, no accelerator — so CI can ask an image what it supports before deciding what to run. Exit code is `0` unless the profile cannot be resolved.

Only checks the image actually implements are listed; planned ones are deliberately absent, so the output is a contract rather than a roadmap.

**Options:**
- `--profile <name>`: Profile name. Defaults to the auto-detected profile.
- `--output-format <table|json>` (default: `table`).

```bash
docker run --rm aim-base:0.11 list-checks
docker run --rm aim-base:0.11 list-checks --output-format json
```

```
Name                       Type           Scope      Description
-------------------------------------------------------------------------------
profile_schema             pass_fail      runtime    Pydantic schema validation
engine_validation          pass_fail      runtime    vLLM-specific argument checks
api_health                 pass_fail      offline    Service serves a model
...
```

## Benchmark (`benchmark`)

Runs a benchmark suite against an AIM inference service using `vllm bench serve`. If `--service-url` is omitted, the server is started automatically, benchmarked, and shut down on exit. Results are exported as JSON and CSV. Exit code is `0` on success, `1` on failure.

**Options:**
- `--service-url <url>`: URL of a running AIM service (e.g. `http://localhost:8000`). If omitted, the server is started automatically.
- `--timeout-seconds <seconds>` (default: `30`): Timeout for individual service requests.
- `--config <path>`: Path to benchmark config YAML. Defaults to the built-in config, which selects a suite based on accelerator count.
- `--output-dir <path>` (default: `.`): Directory for result files.
- `--startup-timeout <seconds>` (default: `120`): How long to wait for auto-started server readiness.

**Output files** (all written to `--output-dir`):

| File | Contents |
|------|----------|
| `benchmark_results.json` | The suite's own results (`overall_success`, `benchmark_configs`). This is what `ci/benchmarking/parse_benchmark_results.py` reads. |
| `benchmark_results.csv` | One row per benchmark configuration. The only source the async feedback workflow reads metrics from. |
| `harness_benchmark_results.json` | The `HarnessResult` envelope (`success`, `summary`, `checks`, `metrics`, `artifacts`) that every harness returns. |

```bash
# Automatic mode — starts server, benchmarks, then shuts down
docker run --rm \
  -e AIM_MODEL_ID=meta-llama/Llama-3.1-8B-Instruct \
  --device=/dev/kfd --device=/dev/dri \
  aim-base:0.11 \
  benchmark --output-dir /workspace/results

# External server mode — benchmark an already-running service
docker run --rm \
  aim-base:0.11 \
  benchmark --service-url http://host.docker.internal:8000
```

### Benchmark Configuration

The config YAML defines suites as lists of `[ISL, OSL, concurrency, num_prompts]` tuples. Suite selection priority: `ACTIVE_SUITE` env var > `accelerator_count_suite_map` match > `active_config` fallback.

```yaml
active_config: "my_suite"
accelerator_count_suite_map:
  1: "tp1_suite"
  2: "tp2_suite"
config_suites:
  my_suite: [[256,256,8,80], [1024,1024,128,256]]
settings:
  timeout_seconds_per_config: 14400
  ignore_eos: true
  num_warmups: 5
  trust_remote_code: null
  percentile_metrics: "ttft,tpot,itl,e2el"
  metric_percentiles: "75,90,99"
  dataset_name: "random"
```

`num_warmups` sends that many warmup requests per configuration before measuring; `0` omits the flag, and anything that is not a non-negative integer is rejected. `trust_remote_code` left unset follows the profile's `trust-remote-code` engine arg so the benchmark client loads the tokenizer the way the served engine did; `true` or `false` overrides the profile, and must be written as an unquoted boolean — a string such as `"true"` is rejected rather than guessed at. Both are checked when the runner is constructed, so a malformed value fails before the service is probed rather than part-way through a sweep. The tokenizer mode is not a setting — it comes from the profile's `tokenizer-mode` engine arg, so a benchmark cannot tokenize differently from the engine it measures.

`VLLM_BENCH_EXTRA_ARGS` is appended verbatim after every flag derived from these settings, so whatever it contains is the last word — including `--trust-remote-code`.

**Environment variables:**

| Variable | Description |
|----------|-------------|
| `ACTIVE_SUITE` | Override automatic suite selection |
| `VLLM_BENCH_EXTRA_ARGS` | Extra arguments passed to `vllm bench serve` |
| `BENCHMARK_JSON_FILE` | Override JSON output filename |
| `BENCHMARK_CSV_FILE` | Override CSV output filename |

## Evaluate (`evaluate`)

Runs accuracy evaluation against an already-running AIM service using [lm-evaluation-harness](https://github.com/EleutherAI/lm-evaluation-harness), and reports the score as a `HarnessResult`. Unlike `benchmark`, it never starts the server itself — it waits for the service serving the image. Exit code is `0` on success, `1` on failure.

**Options:**
- `--profile <name>`: Profile name. Defaults to the auto-detected profile.
- `--service-url <url>`: URL of the running service. Defaults to `http://localhost:{profile.port}`.
- `--config <path>`: Evaluation config YAML with per-run overrides (tasks, `num_fewshot`, `limit`, `data_dir`). Defaults to the config shipped in the image.
- `--timeout <seconds>` (default: `1800`): Timeout for the evaluation run.
- `--output-dir <path>`: Directory for result files. Omit to report to stdout only.
- `--startup-timeout <seconds>` (default: `120`): How long to wait for service readiness before evaluating.
- `--output-format <json|table|ci>` (default: `json`).

**Output files** (written to `--output-dir`):

| File | Contents |
|------|----------|
| `evaluate_results.json` | The `HarnessResult` envelope (`success`, `summary`, `checks`, `metrics`, `artifacts`). |
| `accuracy_evaluation_results.json` | The results envelope CI records: score, task, metric, sample counts, backend version. |
| `accuracy_evaluation_results.csv` | One row per task and metric. |
| `evaluation_backend_results.json` | The backend's own raw output, kept for reproducing a score. |

```bash
# Against the service inside a running AIM container. The image ships the CLI as
# its entrypoint script and does not install the `aim-runtime` console script, so
# call the script — `aim-runtime` is not on PATH there.
docker exec <container> \
  python3 /workspace/entrypoint.py evaluate --output-dir /workspace/results

# Same thing in Kubernetes
kubectl exec <pod> -- \
  python3 /workspace/entrypoint.py evaluate --service-url http://localhost:8000

# From an environment that pip-installed this package (dev box, CI runner),
# where the console script does exist
aim-runtime evaluate --service-url http://localhost:8000 --output-dir ./results
```

### The evaluation backend in AIM images

lm-eval runs as a **subprocess**, not an import, and images install it into its own virtualenv at `/workspace/tools/eval-venv`. Both facts follow from one constraint: lm-eval pins its own `transformers`, `numpy` and `datasets` versions, and installing those in the serving environment would rebind the interpreter that serves the model — against `vllm/vllm-openai-rocm:v0.25.1` the pins are a *downgrade* of the `transformers` vLLM was built against (5.12.0 over 5.13.1).

The virtualenv costs about **860 MB uncompressed** (measured on that image: 857 MB, roughly 2.5% of a 33 GB base). It carries no torch — lm-eval needs torch only for its `hf` extra, and AIM uses the API backend — so there is no second copy of the framework.

Images point the runtime at that virtualenv with `AIM_EVAL_BACKEND_COMMAND`. Set it yourself to relocate the backend — for example to a dev environment that installed the `evaluation` extra with `pip install -e ".[evaluation]"`, where plain `lm_eval` is already on `PATH`:

```bash
AIM_EVAL_BACKEND_COMMAND=lm_eval aim-runtime evaluate --service-url http://localhost:8000
```

Shipping the backend is **opt in per base target**, via `install_evaluation_deps` at the top level of `assets/<accelerator>/base/config.yaml`, beside the `base_image:` block. The key has no effect inside `base_image:`, and a config that puts it there gets a warning. Images built without the opt-in serve normally and report one failed check naming the missing command, so a size-sensitive build can leave it out. `instinct` ships it today; see [`.github/README-CI.md`](../.github/README-CI.md) for the build-arg wiring.

## Detect hardware (`detect-hardware`)

Detects hardware accelerators (GPU and/or CPU) and reports identifiers. Runs each detector independently and returns the accumulated results as a list of dicts.

**Options:**
- `--type <type>`: Which accelerator types to detect
  - `all` (default): Run both GPU and CPU detectors
  - `gpu`: Run GPU detection only
  - `cpu`: Run CPU detection only
- `--format <format>`: Output format
  - `json` (default): JSON output
  - `yaml`: YAML output
- `--verbose` / `-v`: Show full detection details (GPUInfo, CPUInfo)

This is useful for:
- Verifying what hardware aim-runtime detects before running inference
- Node labelling in Kubernetes (the output format matches the node-labelling interface)
- Debugging hardware detection issues

```bash
# Detect all hardware (GPU + CPU)
docker run --rm \
  --device=/dev/kfd --device=/dev/dri \
  aim-base:0.11 \
  detect-hardware

# GPU only, YAML output
docker run --rm \
  --device=/dev/kfd --device=/dev/dri \
  aim-base:0.11 \
  detect-hardware --type gpu --format yaml

# Full details including VRAM, utilization, core counts
docker run --rm \
  --device=/dev/kfd --device=/dev/dri \
  aim-base:0.11 \
  detect-hardware --verbose
```

**Example output (default):**
```json
[
  {"accelerator_type": "GPU", "accelerator_model": "MI300X", "accelerator_count": 8},
  {"accelerator_type": "CPU", "accelerator_model": "EPYC_9965", "accelerator_count": 192}
]
```

**Example output (verbose, GPU only):**
```json
[
  {
    "accelerator_type": "gpu",
    "accelerator_model": "MI300X",
    "accelerator_count": 8,
    "gpu_info": [
      {
        "device_id": "0x74a1",
        "model": "MI300X",
        "vram_total": 65536,
        "vram_used": 0,
        "vram_free": 65536,
        "gfx_utilization": 0,
        "mem_utilization": 0,
        "is_idle": true
      }
    ]
  }
]
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
  aim-base:0.11 \
  download-to-cache

# Result: /workspace/model-cache/meta-llama/Llama-3.1-8B-Instruct/
```

### Explicit Model ID (with protocol)

Override profile selection by specifying the model ID directly with protocol:

```bash
# Download a Hugging Face model directly (local-dir mode)
docker run --rm \
  -v /host/model-cache:/workspace/model-cache \
  aim-base:0.11 \
  download-to-cache --model-id hf://mistralai/Mistral-7B-v0.1

# Result: /workspace/model-cache/mistralai/Mistral-7B-v0.1/
```

### Hugging Face Cache Mode

Use `--use-hf-cache` flag to download using Hugging Face's standard cache structure:

```bash
docker run --rm \
  -e AIM_MODEL_ID=meta-llama/Llama-3.1-8B-Instruct \
  -v /host/model-cache:/workspace/model-cache \
  aim-base:0.11 \
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
  aim-base:0.11 \
  download-to-cache

# Step 2: Serve with pre-cached model
docker run \
  -e AIM_MODEL_ID=meta-llama/Llama-3.1-8B-Instruct \
  -v /host/model-cache:/workspace/model-cache \
  --device=/dev/kfd --device=/dev/dri \
  -p 8000:8000 \
  aim-base:0.11
```

**Note:** The actual model downloaded may differ from `AIM_MODEL_ID` if the selected profile specifies a quantized variant.

**See [Model Caching Documentation](model_caching.md) for comprehensive details on downloads, cache formats, and runtime behavior.**
