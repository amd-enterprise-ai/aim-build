# Copyright © Advanced Micro Devices, Inc., or its affiliates.
#
# SPDX-License-Identifier: MIT

"""Identity constants for the MONAI wholeBrainSeg_Large_UNEST segmentation AIM.

Free of imports and side effects so the harness can read them without
pulling in torch, logging setup, or the bundle metadata probe from config.py.

The bundle name is also hardcoded in the Dockerfile's ``monai.bundle
download`` step — the two must change together.
"""

BUNDLE_NAME = "wholeBrainSeg_Large_UNEST_segmentation"
ENV_PREFIX = "BRAINSEG"
