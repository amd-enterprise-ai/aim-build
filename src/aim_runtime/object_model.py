# Copyright © Advanced Micro Devices, Inc., or its affiliates.
#
# SPDX-License-Identifier: MIT

import sys
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Any, Dict, Optional, TypeVar

# TODO: Remove this compatibility workaround once the ROCm base image is updated to Python 3.12
# Python 3.10 compatibility: define StrEnum if not available
if sys.version_info >= (3, 11):
    from enum import StrEnum
else:

    class StrEnum(str, Enum):
        """Minimal StrEnum for Python <3.11."""


EnumerationType = TypeVar("EnumerationType", bound=StrEnum)


class Precision(StrEnum):
    """Supported precision types."""

    AUTO = "auto"
    FP4 = "fp4"
    FP8 = "fp8"
    FP16 = "fp16"
    FP32 = "fp32"
    BF16 = "bf16"
    INT4 = "int4"
    INT8 = "int8"


class Engine(StrEnum):
    """Supported engine types."""

    AUTO = "auto"
    VLLM = "vllm"


class EngineModule(StrEnum):
    VLLM = "vllm.entrypoints.openai.api_server"


class Metric(StrEnum):
    """Supported metric types."""

    AUTO = "auto"
    LATENCY = "latency"
    THROUGHPUT = "throughput"
    # Add more metrics here if needed


class ProfileType(StrEnum):
    """Profile type categories."""

    OPTIMIZED = "optimized"
    UNOPTIMIZED = "unoptimized"
    GENERAL = "general"


class GPUModel(StrEnum):
    """GPU model identifiers with corresponding device IDs.

    Device ID mappings from gpu_detector.py GPU_MODEL_MAPPING.
    Reference: https://github.com/ROCm/gpu-operator/blob/main/helm-charts-k8s/templates/gpu-nfd-default-rule.yaml
    """

    # AMD Instinct series
    MI100 = "MI100"  # 0x738c, 0x738e
    MI250X = "MI250X"  # 0x7408, 0x740c (MI250/MI250X)
    MI210 = "MI210"  # 0x740f, 0x7410 (MI210 VF)
    MI300A = "MI300A"  # 0x74a0
    MI300X = "MI300X"  # 0x74a1, 0x74a9 (MI300X HF), 0x74b5 (MI300X VF), 0x74bd (MI300X HF)
    MI308X = "MI308X"  # 0x74a2, 0x74a8 (MI308X HF), 0x74b6
    MI325X = "MI325X"  # 0x74a5, 0x74b9 (MI325X VF)
    MI350X = "MI350X"  # 0x75a0, 0x75b0 (MI350X VF)
    MI355X = "MI355X"  # 0x75a3, 0x75b3 (MI355X VF)
    # AMD Radeon Pro series
    V710 = "V710"  # 0x7460, 0x7461 (Radeon Pro V710 MxGPU)
    W7900 = "W7900"  # 0x7448, 0x744a (W7900 Dual Slot)
    W7800 = "W7800"  # 0x7449 (W7800 48GB), 0x745e
    W6900X = "W6900X"  # 0x73a2
    W6800 = "W6800"  # 0x73a3 (W6800 GL-XL)
    W6800X = "W6800X"  # 0x73ab (W6800X / W6800X Duo)
    V620 = "V620"  # 0x73a1, 0x73ae (Radeon Pro V620 MxGPU)
    # AMD Radeon series
    RX9070 = "RX9070"  # 0x7550 (RX 9070 / 9070 XT)
    RX7900 = "RX7900"  # 0x744c (RX 7900 XT / 7900 XTX / 7900 GRE / 7900M)
    RX6900 = "RX6900"  # 0x73af
    RX6800 = "RX6800"  # 0x73bf (RX 6800 / 6800 XT / 6900 XT)
    # other
    UNKNOWN = "UNKNOWN"
    NONE = "NONE"


@dataclass(frozen=True)
class ProfileMetadata:
    """Metadata information from a profile."""

    engine: Engine
    gpu: GPUModel
    precision: Precision
    gpu_count: int
    metric: Metric
    manual_selection_only: bool
    type: ProfileType


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
