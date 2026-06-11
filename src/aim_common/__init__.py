# Copyright © Advanced Micro Devices, Inc., or its affiliates.
#
# SPDX-License-Identifier: MIT

"""
Shared common module for AIM project.

This module contains shared code used by both:
- CI team (ci/ directory)
- AIM runtime team (src/aim_runtime/ directory)

WARNING: This is SHARED CODE. Changes here may affect both teams.
Please coordinate with both teams before making breaking changes.
"""

from aim_common.engine_args_models import (
    BentomlEngineArgsModel,
    EngineArgsModel,
    VllmEngineArgsModel,
    VllmOmniEngineArgsModel,
)
from aim_common.object_model import (
    AcceleratorFamily,
    AcceleratorModel,
    AcceleratorType,
    CPUModel,
    Engine,
    EnumerationType,
    GPUModel,
    Metric,
    ModelProfileData,
    Precision,
    ProfileCapabilities,
    ProfileData,
    ProfileMetadata,
    ProfileType,
)

__all__ = [
    "AcceleratorFamily",
    "AcceleratorModel",
    "AcceleratorType",
    "BentomlEngineArgsModel",
    "CPUModel",
    "Engine",
    "EngineArgsModel",
    "EnumerationType",
    "GPUModel",
    "Metric",
    "ModelProfileData",
    "Precision",
    "ProfileCapabilities",
    "ProfileData",
    "ProfileMetadata",
    "ProfileType",
    "VllmEngineArgsModel",
    "VllmOmniEngineArgsModel",
]
