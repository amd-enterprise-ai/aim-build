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

from aim_common.object_model import (
    Engine,
    EnumerationType,
    GPUModel,
    Metric,
    Precision,
    ProfileMetadata,
    ProfileType,
)

__all__ = [
    "Engine",
    "EnumerationType",
    "GPUModel",
    "Metric",
    "Precision",
    "ProfileMetadata",
    "ProfileType",
]
