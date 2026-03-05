# Copyright © Advanced Micro Devices, Inc., or its affiliates.
#
# SPDX-License-Identifier: MIT
import re
import sys
from dataclasses import dataclass
from enum import Enum
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

    @classmethod
    def _all_device_ids(cls) -> Dict[str, "GPUModel"]:
        """
        Returns a mapping from GPU device IDs to their corresponding GPUModel. The keys are device IDs as lowercase
        hexadecimal strings with a "0x" prefix (for example, "0x740c"), and the values are the associated "GPUModel"
        enum members. This provides a canonical lookup table for known GPU device IDs. The mapping is cached on the class
        to avoid recomputation on subsequent calls.
        """

        cache_attribute_name = "_CACHED_DEVICE_IDS"
        if not hasattr(cls, cache_attribute_name):
            # GPU device ID to model name mapping
            # reference: https://github.com/ROCm/gpu-operator/blob/main/helm-charts-k8s/templates/gpu-nfd-default-rule.yaml
            mapping = {
                # AMD Instinct
                "0x738c": GPUModel.MI100,
                "0x738e": GPUModel.MI100,
                "0x7408": GPUModel.MI250X,
                "0x740c": GPUModel.MI250X,  # MI250/MI250X
                "0x740f": GPUModel.MI210,
                "0x7410": GPUModel.MI210,  # MI210 VF
                "0x74a0": GPUModel.MI300A,
                "0x74a1": GPUModel.MI300X,
                "0x74a2": GPUModel.MI308X,
                "0x74a5": GPUModel.MI325X,
                "0x74a8": GPUModel.MI308X,  # MI308X HF
                "0x74a9": GPUModel.MI300X,  # MI300X HF
                "0x74b5": GPUModel.MI300X,  # MI300X VF
                "0x74b6": GPUModel.MI308X,
                "0x74b9": GPUModel.MI325X,  # MI325X VF
                "0x74bd": GPUModel.MI300X,  # MI300X HF
                "0x75a0": GPUModel.MI350X,
                "0x75a3": GPUModel.MI355X,
                "0x75b0": GPUModel.MI350X,  # MI350X VF
                "0x75b3": GPUModel.MI355X,  # MI355X VF
                # AMD Radeon Pro
                "0x7460": GPUModel.V710,
                "0x7461": GPUModel.V710,  # Radeon Pro V710 MxGPU
                "0x7448": GPUModel.W7900,
                "0x744a": GPUModel.W7900,  # W7900 Dual Slot
                "0x7449": GPUModel.W7800,  # W7800 48GB
                "0x745e": GPUModel.W7800,
                "0x73a2": GPUModel.W6900X,
                "0x73a3": GPUModel.W6800,  # W6800 GL-XL
                "0x73ab": GPUModel.W6800X,  # W6800X / W6800X Duo
                "0x73a1": GPUModel.V620,
                "0x73ae": GPUModel.V620,  # Radeon Pro V620 MxGPU
                # AMD Radeon
                "0x7550": GPUModel.RX9070,  # RX 9070 / 9070 XT
                "0x744c": GPUModel.RX7900,  # RX 7900 XT / 7900 XTX / 7900 GRE / 7900M
                "0x73af": GPUModel.RX6900,
                "0x73bf": GPUModel.RX6800,  # RX 6800 / 6800 XT / 6900 XT
            }

            setattr(cls, cache_attribute_name, mapping)

        return getattr(cls, cache_attribute_name)

    @classmethod
    def from_string_with_default(cls, value: Optional[str], default: Optional["GPUModel"] = None) -> "GPUModel":
        """Parse a GPU model from a string, returning a default instead of raising. This is a convenience wrapper around
        "from_string". It attempts to resolve "value" using the same rules as "from_string", but will return a default
        value instead of propagating "ValueError". The "value" parameter may be provided in one of the following formats:
        * A GPU model name (case-insensitive string), e.g. "MI300X" or "rx7900".
        * A GPU PCI device ID string with a "0x" prefix, e.g. "0x73bf".
        Args:
            value: GPU model identifier to parse, either a model name or a device ID string with a "0x" prefix.
            default: Value to return if parsing fails. If "None" (the default), "GPUModel.UNKNOWN" is returned when the
            value cannot be resolved.
        Returns:
            A "GPUModel" instance corresponding to "value", or the provided "default" or "GPUModel.UNKNOWN" when parsing
            fails.
        """
        try:
            return cls.from_string(value)
        except ValueError:
            if default is not None:
                return default

            return cls.UNKNOWN

    @classmethod
    def from_string(cls, value: Optional[str]) -> "GPUModel":
        """
        Create a GPUModel from a string. This method accepts two kinds of inputs:
        - GPU model names (case-insensitive), e.g. "MI300X" or "mi300x". These are matched against the enum member values.
        - GPU device IDs as strings with a "0x" prefix (case-insensitive), e.g. "0x740c". These are looked up via
        _all_device_ids method.
        Notes
        -----
        - If value is None, "GPUModel.NONE" is returned.
        - Device IDs must include the "0x" prefix. Passing a bare hex value such as "740c" will not be recognized as a
        device ID and will result in a "ValueError" unless it also matches a GPU model name.
        - If value does not correspond to any known GPU model name or device ID, a "ValueError" is raised.
        """
        gpu_model_mapping = cls._all_device_ids()

        if value is None:
            return cls.NONE

        name_value = value.upper()
        try:
            return cls(name_value)
        except ValueError:
            device_value = value.lower()
            # Try to map from device ID
            if device_value in gpu_model_mapping:
                return gpu_model_mapping[device_value]

            raise


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
            gpu=GPUModel.from_string(data["gpu"]),
            precision=to_enum(Precision, data["precision"]),
            gpu_count=data.get("gpu_count", 1),
            metric=to_enum(Metric, data["metric"]),
            manual_selection_only=bool(data.get("manual_selection_only", False)),
            type=to_enum(ProfileType, data.get("type", "general")),
        )


@dataclass(frozen=True)
class CanonicalName:

    org: str
    model_name: str

    @classmethod
    def _sanitize(cls, name: str) -> str:
        sanitized = name.lower()
        sanitized = re.sub(r"[^a-z0-9._-]", "-", sanitized)
        sanitized = re.sub(r"[-_.]+", "-", sanitized)
        sanitized = sanitized.strip("-_.")
        return sanitized

    @property
    def sanitize(self):
        return self._sanitize(self.canonical)

    @property
    def canonical(self) -> str:
        return f"{self.org}/{self.model_name}"

    @property
    def org_sanitized(self) -> str:
        return self._sanitize(self.org)

    @property
    def model_name_sanitized(self) -> str:
        return self._sanitize(self.model_name)

    @classmethod
    def from_string(cls, canonical_name: Optional[str]) -> Optional["CanonicalName"]:
        if not canonical_name:
            return None
        org, model_name = canonical_name.split("/", 1)
        return cls(org, model_name)

    @property
    def publisher(self):
        org_publisher_mapping = {
            "meta-llama": "Meta",
            "mistralai": "Mistral AI",
            "Qwen": "Qwen",
            "CohereLabs": "Cohere Labs",
        }

        return org_publisher_mapping.get(self.org, self.org)

    @property
    def title(self):
        return self.model_name.replace("-", " ").title()
