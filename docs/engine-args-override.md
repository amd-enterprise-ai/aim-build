<!--
Copyright © Advanced Micro Devices, Inc., or its affiliates.

SPDX-License-Identifier: MIT
-->

# Engine Arguments Override

## Overview

Pass additional engine arguments at runtime using the `AIM_ENGINE_ARGS` environment variable to customize inference behavior without modifying profiles.

**Validation**: Arguments are validated against the engine's JSON schema to ensure correct types and formats. Direct execution via `os.execv()` makes command injection architecturally impossible.

## Usage

### Basic Example

```bash
docker run \
  -e AIM_MODEL_ID=meta-llama/Llama-3.1-8B-Instruct \
  -e AIM_ENGINE_ARGS='{"max-model-len": 4096, "gpu-memory-utilization": 0.85}' \
  aim:latest
```

### Complex Example with Nested Structures

```bash
docker run \
  -e AIM_MODEL_ID=meta-llama/Llama-3.1-8B-Instruct \
  -e AIM_ENGINE_ARGS='{
    "max-model-len": 8192,
    "enable-chunked-prefill": true,
    "rope-scaling": {
      "type": "linear",
      "factor": 2.0
    }
  }' \
  aim:latest
```

## Merge Precedence

Arguments are merged with increasing priority:

1. **Profile Defaults** - From selected YAML profile
2. **User Overrides** - From `AIM_ENGINE_ARGS` (overrides profile)
3. **System Overrides** - From runtime (e.g., port, always wins)

Example:
```bash
# Profile has: gpu-memory-utilization: 0.95, dtype: float16
AIM_ENGINE_ARGS='{"gpu-memory-utilization": 0.85, "max-model-len": 4096}'
# Result: 0.85 (user), float16 (profile), 4096 (user addition)
```

## Validation

Arguments are validated against the engine's JSON schema (`schemas/vllm_engine_schema.json`):
- **Type checking**: integer, float, string, boolean, null
- **Enum validation**: e.g., `dtype` must be "auto", "bfloat16", "float16", etc.
- **Structure validation**: nested objects must match expected format

**Why no security checks?** Arguments are passed directly to the engine process via `os.execv()` as a list, with no shell interpretation. This makes command injection architecturally impossible.

**Example validation errors**:
```bash
AIM_ENGINE_ARGS='{"max-model-len": "not-a-number"}'
# Error: 'not-a-number' is not of type 'integer'

AIM_ENGINE_ARGS='{"dtype": "invalid"}'
# Error: 'invalid' is not one of ['auto', 'bfloat16', 'float16', ...]
```

## Supported Types

All standard JSON types are supported:
```json
{
  "max-model-len": 4096,                    // integer
  "gpu-memory-utilization": 0.85,           // float
  "enforce-eager": true,                    // boolean
  "seed": null,                              // null/flag
  "dtype": "float16",                       // string
  "rope-scaling": {                         // nested object
    "type": "linear",
    "factor": 2.0
  }
}
```

## Common Use Cases

```bash
# Adjust context length
AIM_ENGINE_ARGS='{"max-model-len": 8192}'

# Reduce memory usage
AIM_ENGINE_ARGS='{"gpu-memory-utilization": 0.75}'

# Enable chunked prefill
AIM_ENGINE_ARGS='{"enable-chunked-prefill": true}'

# Change KV cache precision
AIM_ENGINE_ARGS='{"kv-cache-dtype": "fp8"}'

# Set random seed
AIM_ENGINE_ARGS='{"seed": 42}'

# Configure RoPE scaling
AIM_ENGINE_ARGS='{"rope-scaling": {"type": "linear", "factor": 2.0}}'
```

## Kubernetes Example

```yaml
apiVersion: v1
kind: Pod
metadata:
  name: aim-inference
spec:
  containers:
  - name: aim
    image: aim:latest
    env:
    - name: AIM_MODEL_ID
      value: "meta-llama/Llama-3.1-8B-Instruct"
    - name: AIM_ENGINE_ARGS
      value: '{"max-model-len": 4096, "gpu-memory-utilization": 0.85}'
```

## Troubleshooting

**Invalid JSON**:
```bash
AIM_ENGINE_ARGS='{invalid}'  # Warning: Failed to parse as JSON
```
→ Use valid JSON syntax. Test with `python -m json.tool`

**Type Mismatch**:
```bash
AIM_ENGINE_ARGS='{"max-model-len": "4096"}'  # String instead of int
# Error: '4096' is not of type 'integer'
```
→ Check [vLLM docs](https://docs.vllm.ai/en/latest/serving/openai_compatible_server.html) for correct types

## Debugging & Dry Run

**Enable debug logging**:
```bash
AIM_DEBUG=true aim-runtime serve
# Shows: parsed args, merge process, validation, final command
```

**Preview without execution**:
```bash
aim-runtime dry-run --format=json
# Displays: selected profile + final merged arguments
```

## Common vLLM Arguments

| Argument | Type | Description |
|----------|------|-------------|
| `max-model-len` | int | Maximum context length |
| `gpu-memory-utilization` | float | GPU memory fraction (0.0-1.0) |
| `tensor-parallel-size` | int | Number of GPUs for tensor parallelism |
| `dtype` | string | Model data type (auto, float16, bfloat16, etc.) |
| `kv-cache-dtype` | string | KV cache precision (auto, fp8, etc.) |
| `enable-chunked-prefill` | bool | Enable chunked prefill |

**Full reference**: [vLLM OpenAI-Compatible Server Docs](https://docs.vllm.ai/en/latest/serving/openai_compatible_server.html)
