# Copyright © Advanced Micro Devices, Inc., or its affiliates.
#
# SPDX-License-Identifier: MIT

"""Settings for the AIFS AIM (AIFS Single 2.0 or AIFS ENS 2.0).

Reads ``AIFS_*`` variables, plus ``AIM_INPUT_PATH``, ``AIM_OUTPUT_PATH``,
``AIM_CACHE_PATH`` and ``AIM_PORT``. Standard library only, so it imports on a
host without torch or anemoi. Import it before anything that imports anemoi.
"""

import os

import anemoi_variant
import canonical_api

ENV_PREFIX = "AIFS"

# AIFS_VARIANT=single (default) serves ecmwf/aifs-single-2.0, AIFS_VARIANT=ens serves
# ecmwf/aifs-ens-2.0; each uses its own anemoi-models/anemoi-graphs folder.
VARIANT = anemoi_variant.activate()
ENSEMBLE = VARIANT == "ens"

# Repo, checkpoint, model class and the Hugging Face commit the checkpoint was validated at.
_DEFAULTS = {
    "single": (
        "ecmwf/aifs-single-2.0",
        "aifs-single-mse-2.0.ckpt",
        "AnemoiModelEncProcDec",
        "08286fc247419bc40666226c816ad4ac686d2a8b",
    ),
    "ens": (
        "ecmwf/aifs-ens-2.0",
        "aifs-ens-crps-2.0.ckpt",
        "AnemoiEnsModelEncProcDec",
        "29a1f8c4ab76cd0e5b64cfd415f4ff9e08edf7cf",
    ),
}


def _val(name, default):
    return os.environ.get("%s_%s" % (ENV_PREFIX, name), default).strip()


HF_REPO = _val("HF_REPO", _DEFAULTS[VARIANT][0])
MODEL_ID = canonical_api.resolve_model_id(HF_REPO)
# Commit the checkpoint and lsm.grib are read at. The pinned default applies to the
# default repo only; another repo is read at its main branch unless this is set.
HF_REVISION = _val("HF_REVISION", _DEFAULTS[VARIANT][3] if HF_REPO == _DEFAULTS[VARIANT][0] else "") or None
# Model class the checkpoint must hold; a mismatch means the wrong checkpoint for the variant.
MODEL_CLASS = _DEFAULTS[VARIANT][2]


# Autocast precision passed to anemoi-inference; empty uses the checkpoint's
# own setting (16-mixed).
PRECISION = _val("PRECISION", "")

# Number of chunks anemoi-models splits the mapper into to save memory; empty
# keeps the anemoi default.
NUM_CHUNKS = _val("NUM_CHUNKS", "")
if NUM_CHUNKS:
    os.environ["ANEMOI_INFERENCE_NUM_CHUNKS"] = NUM_CHUNKS

OUTPUT_FORMAT = _val("OUTPUT_FORMAT", "netcdf")

# Where forecasts are written. The /tmp default lasts only as long as the
# container; mount a directory and set AIM_OUTPUT_PATH to keep them.
DEFAULT_OUTPUT_DIR = os.environ.get("AIM_OUTPUT_PATH", "").strip() or "/tmp/weather/output"

# Directory of input GRIB2 files, one per 6-hourly time; the image bakes one.
INPUT_DIR = os.environ.get("AIM_INPUT_PATH", "").strip()

# Local earthkit-regrid matrix DB; empty downloads matrices from ECMWF.
REGRID_DB = _val("REGRID_DB", "")

PORT = int(_val("PORT", os.environ.get("AIM_PORT", "8000")))

CHECKPOINT = _val("CHECKPOINT", _DEFAULTS[VARIANT][1])
LSM_FILE = _val("LSM_FILE", "lsm.grib")

# Directory holding an already-downloaded checkpoint and lsm.grib. Empty means
# resolve them through the HuggingFace cache, downloading if missing.
WEIGHTS_DIR = _val("WEIGHTS_DIR", "")

# Where the checkpoint is cached. Set before anything imports huggingface_hub,
# which reads HF_HOME once at import.
CACHE_DIR = os.environ.get("AIM_CACHE_PATH", "").strip()
if CACHE_DIR:
    os.environ["HF_HOME"] = CACHE_DIR
    os.environ.setdefault("HF_DATASETS_CACHE", CACHE_DIR)

WARMUP_ON_STARTUP = _val("WARMUP_ON_STARTUP", "1").lower() in ("1", "true", "yes", "on")

# Forecast length cap, in 6-hour steps (10 days).
MAX_FORECAST_STEPS = int(_val("MAX_FORECAST_STEPS", "40"))

# Ensemble members per request (AIFS_VARIANT=ens).
MAX_MEMBERS = int(_val("MAX_MEMBERS", "16"))

# Cap on the forecast held in memory and written per request, in GB of float32.
MAX_OUTPUT_GB = float(_val("MAX_OUTPUT_GB", "32"))

# Subdirectory of the input directory holding perturbed members as pf01, pf02, ...
MEMBERS_SUBDIR = "members"


def weights_candidates(name):
    """Configured local paths for ``name``, or an empty list meaning use the HF cache."""
    if not WEIGHTS_DIR:
        return []
    return [os.path.join(WEIGHTS_DIR, name)]


# Reported in the /v1/inference response and alongside benchmark metrics.
OPT_SUMMARY = {
    "precision": PRECISION or "checkpoint",
    "num_chunks": NUM_CHUNKS or "default",
}
