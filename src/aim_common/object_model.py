# Copyright © Advanced Micro Devices, Inc., or its affiliates.
#
# SPDX-License-Identifier: MIT

import sys
from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict, TypeVar

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
    PREVIEW = "preview"


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

    def __str__(self) -> str:
        """Generate the profile ID string."""
        return self.profile_id

    @property
    def profile_id(self) -> str:
        return f"{self.engine.value.lower()}-{self.gpu.value.lower()}-{self.precision.value.lower()}-tp{self.gpu_count}-{self.metric.value.lower()}"

    def to_dict(self) -> Dict[str, Any]:
        """Convert the profile metadata to a dictionary for serialization."""
        return {
            "engine": self.engine.value,
            "gpu": self.gpu.value,
            "precision": self.precision.value,
            "gpu_count": self.gpu_count,
            "metric": self.metric.value,
            "manual_selection_only": self.manual_selection_only,
            "type": self.type.value,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ProfileMetadata":
        """Create a ProfileMetadata instance from a dictionary."""

        # Helper function for case-insensitive enum conversion
        def to_enum(enum_class, value):
            if isinstance(value, str):
                try:
                    return enum_class(value)
                except ValueError:
                    # Try case-insensitive match
                    for member in enum_class:
                        if member.value.lower() == value.lower():
                            return member
                    raise

            return value

        return cls(
            engine=to_enum(Engine, data["engine"]),
            gpu=to_enum(GPUModel, data["gpu"]),
            precision=to_enum(Precision, data["precision"]),
            gpu_count=data.get("gpu_count", 1),
            metric=to_enum(Metric, data["metric"]),
            manual_selection_only=bool(data.get("manual_selection_only", False)),
            type=to_enum(ProfileType, data.get("type", "general")),
        )
