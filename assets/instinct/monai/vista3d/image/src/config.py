# Copyright © Advanced Micro Devices, Inc., or its affiliates.
#
# SPDX-License-Identifier: MIT

"""
Configuration, logging, and optimization flags for the MONAI VISTA-3D AIM service.

All environment variables are read once at import time.
"""

import json
import logging
import os
import sys
from pathlib import Path

import torch._dynamo.config as dynamo_config
from constants import BUNDLE_NAME, ENV_PREFIX  # noqa: F401  (re-exported)

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

# Hugging Face token required to download model weights at service startup.
# Weights are governed by the NVIDIA OneWay Noncommercial License; the account
# linked to HF_TOKEN must have accepted the license at
# https://huggingface.co/MONAI/vista3d before the service can start.
HF_TOKEN: str | None = os.environ.get("HF_TOKEN")

logging.basicConfig(
    level=getattr(logging, LOG_LEVEL, logging.INFO),
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    stream=sys.stderr,
    force=True,
)
LOGGER = logging.getLogger(f"{BUNDLE_NAME}.service")

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
#
# VISTA-3D ships its 132-class label dictionary inside the bundle's
# ``configs/metadata.json`` under
# ``network_data_format.outputs.pred.channel_def``. We expose that as the
# ``CHANNEL_DEF`` constant for the service's ``/v1/models`` endpoint and for
# the harness test that asserts the served pod surfaces all 133 entries
# (background + 132 anatomical labels).
#
# The probe candidate order is important — see KB
# ``config-channel-def-probe-must-include-here-slash-bundles``. The flat
# container layout puts ``config.py`` at ``/workspace/model/config.py`` and
# the bundle at ``/workspace/model/bundles/vista3d/...``, so the FIRST probe
# must be ``here / 'bundles' / 'vista3d' / 'configs' / 'metadata.json'``.
# Host layout (``app/src/config.py`` + ``app/bundles/...``) is covered
# by the ``here.parent`` fallback.
# ---------------------------------------------------------------------------

# VISTA-3D inference contract constants (mirror the bundle's
# ``configs/inference.json``). Kept here so tests, harness, and benchmark
# script all read the same source of truth.
ROI_SIZE = [128, 128, 128]
SW_OVERLAP = 0.3
SW_BATCH_SIZE = 1
USE_POINT_WINDOW = True
NETWORK_IN_CHANNELS = 1
# ``vista3d132`` outputs 132 anatomical classes (background pseudo-class is
# index 0 in the channel_def map). The network's raw output is multi-channel
# logits gated by ``label_prompt``; the bundle's VistaPostTransformd reduces
# it to a single-channel class-ID map at SaveImaged time.
NETWORK_OUT_CHANNELS = 132


def _load_channel_def() -> dict:
    """Read ``network_data_format.outputs.pred.channel_def`` from the bundle.

    Returns ``{"0": "background"}`` as a silent fallback only if every
    candidate path is missing — that path means the served pod will surface
    a degenerate /v1/models, which is the symptom KB
    ``config-channel-def-probe-must-include-here-slash-bundles`` calls out.
    """
    here = Path(__file__).resolve().parent
    candidates = [
        here / "bundles" / BUNDLE_NAME / "configs" / "metadata.json",
        here.parent / "bundles" / BUNDLE_NAME / "configs" / "metadata.json",
        here.parent.parent / "bundles" / BUNDLE_NAME / "configs" / "metadata.json",
    ]
    for candidate in candidates:
        if candidate.is_file():
            try:
                metadata = json.loads(candidate.read_text())
                cd = metadata.get("network_data_format", {}).get("outputs", {}).get("pred", {}).get("channel_def")
                if cd:
                    return cd
            except (json.JSONDecodeError, OSError) as exc:
                LOGGER.warning(
                    "Failed to read channel_def from %s: %s",
                    candidate,
                    exc,
                )
                continue
    LOGGER.warning(
        "channel_def not found in any of %s — falling back to {'0':'background'}",
        [str(c) for c in candidates],
    )
    return {"0": "background"}


CHANNEL_DEF: dict = _load_channel_def()


OPT_SUMMARY = {
    "compile": OPT_COMPILE,
    "compile_mode": OPT_COMPILE_MODE if OPT_COMPILE else None,
    "dynamic": OPT_DYNAMIC if OPT_COMPILE else None,
    "autocast": OPT_AUTOCAST,
    "gpu_transforms": OPT_GPU_TRANSFORMS,
}
