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
LOGGER = logging.getLogger("wholebody_ct.service")

# ---------------------------------------------------------------------------
# Optimization flags
# ---------------------------------------------------------------------------
OPT_COMPILE = os.environ.get(f"{ENV_PREFIX}_COMPILE", "false").lower() == "true"
OPT_COMPILE_MODE = os.environ.get(f"{ENV_PREFIX}_COMPILE_MODE", "max-autotune")
OPT_DYNAMIC = os.environ.get(f"{ENV_PREFIX}_DYNAMIC", "false").lower() == "true"
OPT_AUTOCAST = os.environ.get(f"{ENV_PREFIX}_AUTOCAST", "false").lower() == "true"
OPT_GPU_TRANSFORMS = os.environ.get(f"{ENV_PREFIX}_GPU_TRANSFORMS", "false").lower() == "true"

# ---------------------------------------------------------------------------
# Variant flag — bundle ships a 1.5 mm "highres" model.pt and a 3.0 mm
# "lowres" model_lowres.pt. The bundle's inference.json controls this via
# displayable_configs#highres. The AIM exposes the same toggle via env var.
# Default: highres (matches bundle default; expects ~28-30 GB GPU at full
# volume, comfortable on MI300X). Set WHOLEBODY_CT_HIGHRES=false to use
# the lowres variant when GPU budget is tight.
# ---------------------------------------------------------------------------
HIGHRES = os.environ.get(f"{ENV_PREFIX}_HIGHRES", "true").lower() == "true"

LOGGER.info(
    "Optimization flags  COMPILE=%s (mode=%s dynamic=%s)  AUTOCAST=%s  GPU_TRANSFORMS=%s  HIGHRES=%s",
    OPT_COMPILE,
    OPT_COMPILE_MODE,
    OPT_DYNAMIC,
    OPT_AUTOCAST,
    OPT_GPU_TRANSFORMS,
    HIGHRES,
)

# ---------------------------------------------------------------------------
# Model constants
# wholeBody_ct_segmentation has a fixed 105-channel output: index 0 is
# background and indices 1..104 are TotalSegmentator anatomical structures.
# Mirrors metadata.json -> network_data_format -> outputs -> pred -> channel_def.
# ---------------------------------------------------------------------------
CHANNEL_DEF = {
    "0": "background",
    "1": "spleen",
    "2": "kidney_right",
    "3": "kidney_left",
    "4": "gallbladder",
    "5": "liver",
    "6": "stomach",
    "7": "aorta",
    "8": "inferior_vena_cava",
    "9": "portal_vein_and_splenic_vein",
    "10": "pancreas",
    "11": "adrenal_gland_right",
    "12": "adrenal_gland_left",
    "13": "lung_upper_lobe_left",
    "14": "lung_lower_lobe_left",
    "15": "lung_upper_lobe_right",
    "16": "lung_middle_lobe_right",
    "17": "lung_lower_lobe_right",
    "18": "vertebrae_L5",
    "19": "vertebrae_L4",
    "20": "vertebrae_L3",
    "21": "vertebrae_L2",
    "22": "vertebrae_L1",
    "23": "vertebrae_T12",
    "24": "vertebrae_T11",
    "25": "vertebrae_T10",
    "26": "vertebrae_T9",
    "27": "vertebrae_T8",
    "28": "vertebrae_T7",
    "29": "vertebrae_T6",
    "30": "vertebrae_T5",
    "31": "vertebrae_T4",
    "32": "vertebrae_T3",
    "33": "vertebrae_T2",
    "34": "vertebrae_T1",
    "35": "vertebrae_C7",
    "36": "vertebrae_C6",
    "37": "vertebrae_C5",
    "38": "vertebrae_C4",
    "39": "vertebrae_C3",
    "40": "vertebrae_C2",
    "41": "vertebrae_C1",
    "42": "esophagus",
    "43": "trachea",
    "44": "heart_myocardium",
    "45": "heart_atrium_left",
    "46": "heart_ventricle_left",
    "47": "heart_atrium_right",
    "48": "heart_ventricle_right",
    "49": "pulmonary_artery",
    "50": "brain",
    "51": "iliac_artery_left",
    "52": "iliac_artery_right",
    "53": "iliac_vena_left",
    "54": "iliac_vena_right",
    "55": "small_bowel",
    "56": "duodenum",
    "57": "colon",
    "58": "rib_left_1",
    "59": "rib_left_2",
    "60": "rib_left_3",
    "61": "rib_left_4",
    "62": "rib_left_5",
    "63": "rib_left_6",
    "64": "rib_left_7",
    "65": "rib_left_8",
    "66": "rib_left_9",
    "67": "rib_left_10",
    "68": "rib_left_11",
    "69": "rib_left_12",
    "70": "rib_right_1",
    "71": "rib_right_2",
    "72": "rib_right_3",
    "73": "rib_right_4",
    "74": "rib_right_5",
    "75": "rib_right_6",
    "76": "rib_right_7",
    "77": "rib_right_8",
    "78": "rib_right_9",
    "79": "rib_right_10",
    "80": "rib_right_11",
    "81": "rib_right_12",
    "82": "humerus_left",
    "83": "humerus_right",
    "84": "scapula_left",
    "85": "scapula_right",
    "86": "clavicula_left",
    "87": "clavicula_right",
    "88": "femur_left",
    "89": "femur_right",
    "90": "hip_left",
    "91": "hip_right",
    "92": "sacrum",
    "93": "face",
    "94": "gluteus_maximus_left",
    "95": "gluteus_maximus_right",
    "96": "gluteus_medius_left",
    "97": "gluteus_medius_right",
    "98": "gluteus_minimus_left",
    "99": "gluteus_minimus_right",
    "100": "autochthon_left",
    "101": "autochthon_right",
    "102": "iliopsoas_left",
    "103": "iliopsoas_right",
    "104": "urinary_bladder",
}

# Total output channels (background + 104 foreground structures). Mirrors
# metadata.json -> network_data_format -> outputs -> pred -> num_channels and
# the SegResNet out_channels in pipeline.build_network.
NUM_CLASSES = 105

# Foreground class indices (1..104) contributing to the bundle's published
# per-class mean Dice. Kept in sync with harness._FOREGROUND_CLASS_INDICES;
# duplicated there so the harness imports cleanly on a Builder host without
# pulling torch through this module.
FOREGROUND_CLASS_INDICES = tuple(range(1, NUM_CLASSES))

OPT_SUMMARY = {
    "compile": OPT_COMPILE,
    "compile_mode": OPT_COMPILE_MODE if OPT_COMPILE else None,
    "dynamic": OPT_DYNAMIC if OPT_COMPILE else None,
    "autocast": OPT_AUTOCAST,
    "gpu_transforms": OPT_GPU_TRANSFORMS,
    "highres": HIGHRES,
}
