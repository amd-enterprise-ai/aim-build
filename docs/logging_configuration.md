<!--
Copyright © Advanced Micro Devices, Inc., or its affiliates.

SPDX-License-Identifier: MIT
-->

# Logging Configuration

AIM provides fine-grained logging control with separate log levels for AIM runtime and third-party packages with help of environment variables:

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
