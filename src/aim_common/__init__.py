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

The object model is re-exported lazily: ``from aim_common import ProfileMetadata``
works exactly as before, but importing a submodule that needs nothing but the
standard library — ``aim_common.engine_args``, say — no longer drags pydantic in
with it. Some CI runner pools run the scripts in ``ci/`` straight from a checkout,
with no installed package and nothing beyond PyYAML, and they can only share code
from here if importing this package costs nothing.
"""

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from aim_common.object_model import (
        AcceleratorFamily,
        AcceleratorModel,
        AcceleratorType,
        AdapterToken,
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
    "AdapterToken",
    "CPUModel",
    "Engine",
    "EnumerationType",
    "GPUModel",
    "Metric",
    "ModelProfileData",
    "Precision",
    "ProfileCapabilities",
    "ProfileData",
    "ProfileMetadata",
    "ProfileType",
]


def __getattr__(name: str) -> Any:
    """Resolve a re-exported object-model name on first use."""
    if name in __all__:
        from aim_common import object_model

        return getattr(object_model, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
