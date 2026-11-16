# Copyright © Advanced Micro Devices, Inc., or its affiliates.
#
# SPDX-License-Identifier: MIT

"""Settings for the Aurora 0.25 degree pretrained AIM.

Reads ``AURORALG_*`` variables, plus ``AIM_INPUT_PATH``, ``AIM_OUTPUT_PATH``,
``AIM_CACHE_PATH`` and ``AIM_PORT``. Standard library only, so it imports on a
host without torch or the aurora package.
"""

import os

import canonical_api

ENV_PREFIX = "AURORALG"

MODEL_ID = canonical_api.resolve_model_id("microsoft/aurora-v1-0p25deg-pretrained-large")


def _flag(name, default="0"):
    return os.environ.get("%s_%s" % (ENV_PREFIX, name), default).strip().lower() in ("1", "true", "yes", "on")


def _val(name, default):
    return os.environ.get("%s_%s" % (ENV_PREFIX, name), default).strip()


# The two tunable knobs, both off by default.
AUTOCAST = _flag("AUTOCAST")
COMPILE = _flag("COMPILE")

OUTPUT_FORMAT = _val("OUTPUT_FORMAT", "netcdf")

# Where forecasts are written. The /tmp default lasts only as long as the
# container; mount a directory and set AIM_OUTPUT_PATH to keep them.
DEFAULT_OUTPUT_DIR = os.environ.get("AIM_OUTPUT_PATH", "").strip() or "/tmp/weather/output"

# Directory an input_netcdf_path request resolves against. Empty refuses the
# field, so a request cannot reach a file outside a mounted directory.
INPUT_DIR = os.environ.get("AIM_INPUT_PATH", "").strip()

# Port the service binds by default; AIM_PORT overrides it.
PORT = int(_val("PORT", os.environ.get("AIM_PORT", "8000")))

# The full-size pretrained checkpoint, not the small or fine-tuned variants.
CHECKPOINT = _val("CHECKPOINT", "aurora-0.25-pretrained.ckpt")

# Directory holding an already-downloaded checkpoint. Empty means download it
# from HuggingFace at startup, which is the default.
WEIGHTS_DIR = _val("WEIGHTS_DIR", "")

# Where the checkpoint is cached. Set before anything imports huggingface_hub,
# which reads HF_HOME once at import.
CACHE_DIR = os.environ.get("AIM_CACHE_PATH", "").strip()
if CACHE_DIR:
    os.environ["HF_HOME"] = CACHE_DIR
    os.environ.setdefault("HF_DATASETS_CACHE", CACHE_DIR)

# Load the model while the service starts rather than on the first request.
WARMUP_ON_STARTUP = _flag("WARMUP_ON_STARTUP", "1")


def weights_candidates(checkpoint=None):
    """Configured checkpoint paths, or an empty list meaning download it."""
    name = checkpoint or CHECKPOINT
    if not WEIGHTS_DIR:
        return []
    return [os.path.join(WEIGHTS_DIR, name)]


# Reported in the /v1/inference response and alongside benchmark metrics.
OPT_SUMMARY = {
    "autocast": AUTOCAST,
    "compile": COMPILE,
}
