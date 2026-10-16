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
LOGGER = logging.getLogger("spleen_deepedit.service")

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
# spleen_deepedit_annotation is a 2-channel-output model (background + spleen)
# trained as part of MONAI's DeepEdit interactive-segmentation family. The
# pred channel_def is loaded from the bundle's metadata.json at import time
# so the served `/v1/models` payload and the harness's CHANNEL_DEF stay in
# sync with the bundle's own declaration of class semantics.
# ---------------------------------------------------------------------------

NUM_CLASSES = 2

# DeepEdit-specific: the network consumes (image, fg-click-map, bg-click-map).
# `number_intensity_ch` is the image-only channel count; the two extra
# channels are appended by AddGuidanceSignalDeepEditd at preprocess time.
# Mirrors @number_intensity_ch in configs/inference.json.
NUMBER_INTENSITY_CH = 1
LABEL_NAMES: dict[str, int] = {
    "spleen": 1,
    "background": 0,
}
SPATIAL_SIZE = (128, 128, 128)


def _load_channel_def() -> dict[str, str]:
    """Load the channel definition for the prediction head from bundle metadata.

    Probes the three layouts this module may be imported from:

      * Container layout (flat): ``/workspace/model/config.py`` →
        bundles at ``/workspace/model/bundles/<bundle>/...``.
      * Host layout: ``app/src/config.py`` → bundles at
        ``app/bundles/<bundle>/...`` (here.parent / bundles).
      * Defensive older src/-relative layout: ``here.parent.parent``.

    Returns ``{"0": "background", "1": "spleen"}`` as a fallback when the
    metadata is unreachable — boot-survivable, but caught by the
    ``test_channel_def_present`` regression in production.

    KB applies: ``config-channel-def-probe-must-include-here-slash-bundles``.
    """
    here = Path(__file__).resolve().parent
    candidates = [
        here / "bundles" / "spleen_deepedit_annotation" / "configs" / "metadata.json",
        here.parent / "bundles" / "spleen_deepedit_annotation" / "configs" / "metadata.json",
        here.parent.parent / "bundles" / "spleen_deepedit_annotation" / "configs" / "metadata.json",
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
    LOGGER.warning("CHANNEL_DEF metadata.json not found; falling back to default")
    return {"0": "background", "1": "spleen"}


CHANNEL_DEF = _load_channel_def()

OPT_SUMMARY = {
    "compile": OPT_COMPILE,
    "compile_mode": OPT_COMPILE_MODE if OPT_COMPILE else None,
    "dynamic": OPT_DYNAMIC if OPT_COMPILE else None,
    "autocast": OPT_AUTOCAST,
    "gpu_transforms": OPT_GPU_TRANSFORMS,
}
