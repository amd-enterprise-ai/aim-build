# Copyright © Advanced Micro Devices, Inc., or its affiliates.
#
# SPDX-License-Identifier: MIT

import sys
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Any, Dict, Optional, TypeVar

# Import shared enums and ProfileMetadata from aim_common
from aim_common import Engine, GPUModel, Metric, Precision, ProfileMetadata

# TODO: Remove this compatibility workaround once the ROCm base image is updated to Python 3.12
# Python 3.10 compatibility: define StrEnum if not available
if sys.version_info >= (3, 11):
    from enum import StrEnum
else:

    class StrEnum(str, Enum):
        """Minimal StrEnum for Python <3.11."""


EnumerationType = TypeVar("EnumerationType", bound=StrEnum)


class EngineModule(StrEnum):
    VLLM = "vllm.entrypoints.openai.api_server"


@dataclass(frozen=True)
class ProfileHandling:
    path: str
    filename: str
    priority: int  # Lower number = higher priority (1 = highest)

    @property
    def profile_name(self) -> str:
        """Get the profile name (filename without extension)."""
        return Path(self.filename).stem

    @property
    def is_general(self) -> bool:
        normalized_path = self.path.replace("\\", "/")
        return "/general/" in normalized_path

    @property
    def is_custom(self) -> bool:
        normalized_path = self.path.replace("\\", "/")
        return "/custom/" in normalized_path


@dataclass(frozen=True)
class Profile:
    """Represents a loaded and validated AIM profile."""

    profile_handling: ProfileHandling
    metadata: ProfileMetadata
    aim_id: str  # The AIM profile identifier (e.g., 'meta-llama/Llama-3.1-8B-Instruct')
    model_id: str  # The model identifier to use for inference
    engine_args: Optional[Dict[str, Any]] = None
    env_vars: Optional[Dict[str, str]] = None

    @property
    def profile_id(self) -> str:
        """
        Generate an identifier for this profile. The identifier is not globally unique.

        For custom general profiles: "custom/general/<profile_name>"
        For custom model-specific profiles: "custom/<org>/<model>/<profile_name>"
        For general profiles: "general/<profile_name>"
        For model-specific profiles: "<profile_name>"

        Returns:
            str: The unique profile identifier in the context of a model
        """
        profile_name = self.profile_handling.profile_name

        mapping = {
            (True, True): f"custom/general/{profile_name}",
            (True, False): f"custom/{self.aim_id}/{profile_name}",
            (False, True): f"general/{profile_name}",
        }

        return mapping.get((self.profile_handling.is_custom, self.profile_handling.is_general), profile_name)

    def matches_gpu(self, gpu: GPUModel) -> bool:
        """Check if this profile matches the given GPU."""
        return self.metadata.gpu == gpu

    def matches_gpu_count(self, gpu_count: int) -> bool:
        """Check if this profile supports the given GPU count."""
        return self.metadata.gpu_count == gpu_count

    def matches_engine(self, engine: Engine) -> bool:
        """Check if this profile matches the given engine."""
        return self.metadata.engine == engine

    def matches_metric(self, metric: Metric) -> bool:
        """Check if this profile matches the given metric."""
        return self.metadata.metric == metric

    def matches_precision(self, precision: Precision) -> bool:
        """Check if this profile matches the given precision."""
        return self.metadata.precision == precision

    def matches_aim_id(self, aim_id: Optional[str]) -> bool:
        """Check if this profile matches the given AIM id.

        Args:
            aim_model_id: The AIM id to match against (e.g., 'meta-llama/Llama-3.1-8B-Instruct')

        Returns:
            True if the profile matches the given AIM id

        Note:
            This matches against aim_id (the AIM identifier), not model_id (the actual model to load).
            For example, a profile with aim_id='meta-llama/Llama-3.1-8B-Instruct' will match even if
            model_id='amd/Llama-3.1-8B-Instruct-FP8-KV' (quantized version).
        """
        if self.profile_handling.is_general:
            return True  # General profiles always match
        return self.aim_id == aim_id
