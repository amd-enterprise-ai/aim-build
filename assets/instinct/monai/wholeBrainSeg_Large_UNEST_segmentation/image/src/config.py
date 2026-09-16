# Copyright © Advanced Micro Devices, Inc., or its affiliates.
#
# SPDX-License-Identifier: MIT

"""
Configuration, logging, and optimization flags for the MONAI AIM service.

All environment variables are read once at import time.
"""

import json
import logging
import os
import sys
from pathlib import Path

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
ENV_PREFIX = "BRAINSEG"

LOG_LEVEL = os.environ.get(f"{ENV_PREFIX}_LOG_LEVEL", "INFO").upper()
SERVICE_TIMEOUT_SECONDS = int(os.environ.get(f"{ENV_PREFIX}_TIMEOUT_SECONDS", "1800"))

logging.basicConfig(
    level=getattr(logging, LOG_LEVEL, logging.INFO),
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    stream=sys.stderr,
    force=True,
)
LOGGER = logging.getLogger("brainseg.service")

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
# wholeBrainSeg_Large_UNEST_segmentation has a fixed 133-channel output
# (background + 132 anatomical brain regions). The full label map is
# loaded from the bundle's metadata.json at import time so we do not
# duplicate the (large) dictionary in source. CHANNEL_DEF is then
# exposed as a normal dict for downstream consumers (service /v1/models,
# release-notes generators, the harness's docstring) without forcing
# them to crack open metadata.json themselves.
# ---------------------------------------------------------------------------

NUM_CLASSES = 133


def _load_channel_def() -> dict[str, str]:
    """Load the 133-class channel definition from bundle metadata.json.

    Probes the three layouts this AIM may be imported from:

      * Host layout: ``app/src/config.py`` → bundles at
        ``app/bundles/<bundle>/...`` (here.parent / bundles).
      * Container layout: Dockerfile flattens ``src/`` to
        ``/workspace/model/`` and bundles sit at
        ``/workspace/model/bundles/<bundle>/...`` (here / bundles).
      * Older src/-relative layout (defensive): ``here.parent.parent``.

    Falls back to a minimal {"0": "background"} dict if the file is
    unreachable — this keeps the process bootable in tests that do not
    need the full label map. ``test_channel_def_has_133_classes`` is
    the regression guard that catches the fallback in production.
    """
    here = Path(__file__).resolve().parent
    candidates = [
        here / "bundles" / "wholeBrainSeg_Large_UNEST_segmentation" / "configs" / "metadata.json",
        here.parent / "bundles" / "wholeBrainSeg_Large_UNEST_segmentation" / "configs" / "metadata.json",
        here.parent.parent / "bundles" / "wholeBrainSeg_Large_UNEST_segmentation" / "configs" / "metadata.json",
    ]
    for p in candidates:
        if p.is_file():
            try:
                meta = json.loads(p.read_text())
                cd = meta.get("network_data_format", {}).get("outputs", {}).get("pred", {}).get("channel_def", {})
                if isinstance(cd, dict) and cd:
                    return cd
            except (json.JSONDecodeError, OSError) as exc:
                LOGGER.warning("Failed to load CHANNEL_DEF from %s: %s", p, exc)
    LOGGER.warning("CHANNEL_DEF metadata.json not found; falling back to background-only")
    return {"0": "background"}


CHANNEL_DEF = _load_channel_def()

# Class indices that contribute to the per-class mean Dice computed by
# WholebrainsegHarness._compute_metric. Background (0) is excluded —
# the bundle's published eval_metrics.mean_dice = 0.71 is the
# per-class mean over the 132 foreground anatomical regions.
FOREGROUND_CLASS_INDICES = tuple(range(1, NUM_CLASSES))

OPT_SUMMARY = {
    "compile": OPT_COMPILE,
    "compile_mode": OPT_COMPILE_MODE if OPT_COMPILE else None,
    "dynamic": OPT_DYNAMIC if OPT_COMPILE else None,
    "autocast": OPT_AUTOCAST,
    "gpu_transforms": OPT_GPU_TRANSFORMS,
}
