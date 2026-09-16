# Copyright © Advanced Micro Devices, Inc., or its affiliates.
#
# SPDX-License-Identifier: MIT

"""
Configuration, logging, and optimization flags for the MONAI AIM service.

All environment variables are read once at import time.
"""

import logging
import os
import sys

import torch._dynamo.config as dynamo_config

# ---------------------------------------------------------------------------
# torch.compile global settings
# ---------------------------------------------------------------------------
os.environ.setdefault("TORCH_DYNAMO_CACHE_SIZE_LIMIT", "128")
os.environ.setdefault("TORCHINDUCTOR_CUDA_GRAPHS", "0")
os.environ.setdefault("TORCHINDUCTOR_TRITON_CUDAGRAPHS", "0")

dynamo_config.cache_size_limit = 128
dynamo_config.assume_static_by_default = False
dynamo_config.automatic_dynamic_shapes = True

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
ENV_PREFIX = "SWINUNETR"

LOG_LEVEL = os.environ.get(f"{ENV_PREFIX}_LOG_LEVEL", "INFO").upper()
SERVICE_TIMEOUT_SECONDS = int(os.environ.get(f"{ENV_PREFIX}_TIMEOUT_SECONDS", "1800"))

logging.basicConfig(
    level=getattr(logging, LOG_LEVEL, logging.INFO),
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    stream=sys.stderr,
    force=True,
)
LOGGER = logging.getLogger("swinunetr.service")

# ---------------------------------------------------------------------------
# Optimization flags
# ---------------------------------------------------------------------------
OPT_COMPILE = os.environ.get(f"{ENV_PREFIX}_COMPILE", "false").lower() == "true"
OPT_COMPILE_MODE = os.environ.get(f"{ENV_PREFIX}_COMPILE_MODE", "max-autotune")
OPT_DYNAMIC = os.environ.get(f"{ENV_PREFIX}_DYNAMIC", "false").lower() == "true"
OPT_AUTOCAST = os.environ.get(f"{ENV_PREFIX}_AUTOCAST", "false").lower() == "true"
OPT_GPU_TRANSFORMS = os.environ.get(f"{ENV_PREFIX}_GPU_TRANSFORMS", "false").lower() == "true"

LOGGER.info(
    "Optimization flags  COMPILE=%s (mode=%s dynamic=%s)  AUTOCAST=%s  " "GPU_TRANSFORMS=%s",
    OPT_COMPILE,
    OPT_COMPILE_MODE,
    OPT_DYNAMIC,
    OPT_AUTOCAST,
    OPT_GPU_TRANSFORMS,
)

# ---------------------------------------------------------------------------
# Model constants
# ---------------------------------------------------------------------------
# Fixed 14-channel output (background + 13 abdominal organs) from the
# bundle's metadata.json `network_data_format.outputs.pred.channel_def`.
CHANNEL_DEF: dict[int, str] = {
    0: "background",
    1: "spleen",
    2: "Right Kidney",
    3: "Left Kidney",
    4: "Gallbladder",
    5: "Esophagus",
    6: "Liver",
    7: "Stomach",
    8: "Aorta",
    9: "IVC",
    10: "Portal and Splenic Veins",
    11: "Pancreas",
    12: "Right adrenal gland",
    13: "Left adrenal gland",
}

OPT_SUMMARY = {
    "compile": OPT_COMPILE,
    "compile_mode": OPT_COMPILE_MODE if OPT_COMPILE else None,
    "dynamic": OPT_DYNAMIC if OPT_COMPILE else None,
    "autocast": OPT_AUTOCAST,
    "gpu_transforms": OPT_GPU_TRANSFORMS,
}
