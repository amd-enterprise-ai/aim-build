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
from constants import ENV_PREFIX  # noqa: F401  (re-exported)

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

LOG_LEVEL = os.environ.get(f"{ENV_PREFIX}_LOG_LEVEL", "INFO").upper()
SERVICE_TIMEOUT_SECONDS = int(os.environ.get(f"{ENV_PREFIX}_TIMEOUT_SECONDS", "1800"))

logging.basicConfig(
    level=getattr(logging, LOG_LEVEL, logging.INFO),
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    stream=sys.stderr,
    force=True,
)
LOGGER = logging.getLogger("pancreas_dints.service")

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
# DiNTS Pancreas has a fixed 3-channel output (background + pancreas + tumor).
# Mirrors metadata.json -> network_data_format -> outputs -> pred -> channel_def.
# ---------------------------------------------------------------------------
CHANNEL_DEF = {
    "0": "background",
    "1": "pancreas",
    "2": "pancreatic tumor",
}

# Foreground class indices used by the per-class Dice override in
# PancreasDintsHarness._compute_metric. Mirrors metadata.json
# label_classes (1=pancreas, 2=pancreatic tumor; 0=everything else).
# The bundle's authoritative mean_dice=0.62 is the per-class mean of Dice
# over these two indices; the harness override reproduces that math so
# Tuner-side evaluate matches the published number instead of the binary
# voxel>0 fallback (which collapses pancreas+tumor and inflates Dice to
# ~0.83 on this dataset).
FOREGROUND_CLASS_INDICES = (1, 2)

OPT_SUMMARY = {
    "compile": OPT_COMPILE,
    "compile_mode": OPT_COMPILE_MODE if OPT_COMPILE else None,
    "dynamic": OPT_DYNAMIC if OPT_COMPILE else None,
    "autocast": OPT_AUTOCAST,
    "gpu_transforms": OPT_GPU_TRANSFORMS,
}
