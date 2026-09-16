# Copyright © Advanced Micro Devices, Inc., or its affiliates.
#
# SPDX-License-Identifier: MIT

"""
Profile structure validation using Pydantic models.

Validates profile structure (metadata, env_vars, engine_args, aim_id, model_id).
Engine-specific arg validation is handled by EngineConfig validators, not here.
"""

from __future__ import annotations

import logging
from typing import Any, Optional

from aim_common import ModelProfileData, ProfileData

logger = logging.getLogger(__name__)


class ProfileValidator:
    """Validates AIM profile YAML files using Pydantic models."""

    def validate(
        self, profile_data: dict[str, Any], is_general_profile: bool = False, source: Optional[str] = None
    ) -> ProfileData:
        """
        Validate profile data structure.

        Args:
            profile_data: Already loaded YAML profile data.
            is_general_profile: If True, validate against the general profile schema.
                Otherwise, validate against the model profile schema.
            source: Profile path or name to include in metadata warnings.

        Returns:
            The validated profile data.

        Raises:
            pydantic.ValidationError: If the profile structure or metadata is invalid.
        """
        model_class = ProfileData if is_general_profile else ModelProfileData
        return model_class.model_validate(profile_data, context={"source": source})
