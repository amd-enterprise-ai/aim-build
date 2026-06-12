# Copyright © Advanced Micro Devices, Inc., or its affiliates.
#
# SPDX-License-Identifier: MIT
import logging
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional, TypeVar

from pydantic import AliasChoices, BaseModel, ConfigDict, Field, field_validator

from aim_common.compat import StrEnum
from aim_common.enums import ParseableStrEnum

logger = logging.getLogger(__name__)

EnumerationType = TypeVar("EnumerationType", bound=StrEnum)


class Precision(StrEnum):
    """Supported precision types."""

    FP4 = "fp4"
    FP8 = "fp8"
    FP16 = "fp16"
    FP32 = "fp32"
    BF16 = "bf16"
    INT4 = "int4"
    INT8 = "int8"


class Engine(StrEnum):
    """Supported engine types."""

    BENTOML = "bentoml"
    VLLM = "vllm"
    VLLM_OMNI = "vllm_omni"


class Metric(StrEnum):
    """Supported metric types."""

    LATENCY = "latency"
    THROUGHPUT = "throughput"
    # Add more metrics here if needed


class ProfileType(StrEnum):
    """Profile type categories."""

    OPTIMIZED = "optimized"
    UNOPTIMIZED = "unoptimized"
    GENERAL = "general"
    PREVIEW = "preview"


class AcceleratorType(StrEnum):
    """Accelerator type — distinguishes CPU from GPU.

    Used by AIM Engine to build pod spec resource requests
    (e.g. resources.requests.cpu vs amd.com/gpu).
    """

    CPU = "cpu"
    GPU = "gpu"


class AcceleratorFamily(ParseableStrEnum):
    EPYC = "epyc"
    INSTINCT = "instinct"
    RADEON = "radeon"
    CPU = "cpu"


class AcceleratorModel(StrEnum):
    """Unified accelerator model identifiers covering GPUs and CPUs.

    GPU device ID mappings sourced from the AMD GPU operator NFD rules.
    CPU chip IDs sourced from /proc/cpuinfo model name extraction.
    No collision between GPU PCI IDs (0x-prefixed hex) and CPU chip IDs (decimal).

    Reference: https://github.com/ROCm/gpu-operator/blob/main/helm-charts-k8s/templates/gpu-nfd-default-rule.yaml
    """

    # AMD Instinct series (GPUs)
    MI100 = "MI100"  # 0x738c, 0x738e
    MI250X = "MI250X"  # 0x7408, 0x740c (MI250/MI250X)
    MI210 = "MI210"  # 0x740f, 0x7410 (MI210 VF)
    MI300A = "MI300A"  # 0x74a0
    MI300X = "MI300X"  # 0x74a1, 0x74a9 (MI300X HF), 0x74b5 (MI300X VF), 0x74bd (MI300X HF)
    MI308X = "MI308X"  # 0x74a2, 0x74a8 (MI308X HF), 0x74b6
    MI325X = "MI325X"  # 0x74a5, 0x74b9 (MI325X VF)
    MI350X = "MI350X"  # 0x75a0, 0x75b0 (MI350X VF)
    MI355X = "MI355X"  # 0x75a3, 0x75b3 (MI355X VF)
    MI350P = "MI350P"  # 0x75a8
    # AMD Radeon Pro series (GPUs)
    R9700 = "R9700"  # 0x7551 (AI PRO R9700 / R9700S / R9600D)
    V710 = "V710"  # 0x7460, 0x7461 (Radeon Pro V710 MxGPU)
    W7900 = "W7900"  # 0x7448, 0x744a (W7900 Dual Slot), 0x744b (W7900D)
    W7800 = "W7800"  # 0x7449 (W7800 48GB), 0x745e
    W6900X = "W6900X"  # 0x73a2
    W6800 = "W6800"  # 0x73a3 (W6800 GL-XL)
    W6800X = "W6800X"  # 0x73ab (W6800X / W6800X Duo)
    V620 = "V620"  # 0x73a1, 0x73ae (Radeon Pro V620 MxGPU)
    # AMD Radeon series (GPUs)
    RX9070 = "RX9070"  # 0x7550 (RX 9070 / 9070 XT)
    RX7900 = "RX7900"  # 0x744c (RX 7900 XT / 7900 XTX / 7900 GRE / 7900M)
    RX6900 = "RX6900"  # 0x73af
    RX6800 = "RX6800"  # 0x73bf (RX 6800 / 6800 XT / 6900 XT)

    # AMD EPYC CPUs
    EPYC_9965 = "EPYC_9965"  # Zen5, 192 cores
    EPYC_ZEN5 = "EPYC_ZEN5"  # Fallback for other Zen5 chips
    EPYC_ZEN4 = "EPYC_ZEN4"  # Fallback for Zen4 chips

    # Sentinel for unknown/undetected CPU
    CPU = "CPU"

    @classmethod
    def _all_accelerator_ids(cls) -> Dict[str, "AcceleratorModel"]:
        """Returns a mapping from accelerator detection IDs to AcceleratorModel.

        GPU PCI device IDs (0x-prefixed hex) and CPU chip IDs (decimal strings)
        share the same lookup — no collision is possible.
        The mapping is cached on the class to avoid recomputation.
        """
        cache_attribute_name = "_CACHED_ACCELERATOR_IDS"
        if not hasattr(cls, cache_attribute_name):
            mapping = {
                # GPU PCI device IDs
                # Reference: https://github.com/ROCm/gpu-operator/blob/main/helm-charts-k8s/templates/gpu-nfd-default-rule.yaml
                # AMD Instinct
                "0x738c": cls.MI100,
                "0x738e": cls.MI100,
                "0x7408": cls.MI250X,
                "0x740c": cls.MI250X,  # MI250/MI250X
                "0x740f": cls.MI210,
                "0x7410": cls.MI210,  # MI210 VF
                "0x74a0": cls.MI300A,
                "0x74a1": cls.MI300X,
                "0x74a2": cls.MI308X,
                "0x74a5": cls.MI325X,
                "0x74a8": cls.MI308X,  # MI308X HF
                "0x74a9": cls.MI300X,  # MI300X HF
                "0x74b5": cls.MI300X,  # MI300X VF
                "0x74b6": cls.MI308X,
                "0x74b9": cls.MI325X,  # MI325X VF
                "0x74bd": cls.MI300X,  # MI300X HF
                "0x75a0": cls.MI350X,
                "0x75a3": cls.MI355X,
                "0x75b0": cls.MI350X,  # MI350X VF
                "0x75b3": cls.MI355X,  # MI355X VF
                "0x75a8": cls.MI350P,
                # AMD Radeon Pro
                "0x7551": cls.R9700,  # AI PRO R9700 / R9700S / R9600D
                "0x7460": cls.V710,
                "0x7461": cls.V710,  # Radeon Pro V710 MxGPU
                "0x7448": cls.W7900,
                "0x744a": cls.W7900,  # W7900 Dual Slot
                "0x744b": cls.W7900,  # W7900D
                "0x7449": cls.W7800,  # W7800 48GB
                "0x745e": cls.W7800,
                "0x73a2": cls.W6900X,
                "0x73a3": cls.W6800,  # W6800 GL-XL
                "0x73ab": cls.W6800X,  # W6800X / W6800X Duo
                "0x73a1": cls.V620,
                "0x73ae": cls.V620,  # Radeon Pro V620 MxGPU
                # AMD Radeon
                "0x7550": cls.RX9070,  # RX 9070 / 9070 XT
                "0x744c": cls.RX7900,  # RX 7900 XT / 7900 XTX / 7900 GRE / 7900M
                "0x73af": cls.RX6900,
                "0x73bf": cls.RX6800,  # RX 6800 / 6800 XT / 6900 XT
                # CPU chip IDs (from /proc/cpuinfo)
                # See https://www.amd.com/en/products/processors/server/epyc/9005-series.html#zen5
                # AMD EPYC 9005 series (Zen5)
                "9965": cls.EPYC_9965,
                "9845": cls.EPYC_ZEN5,
                "9825": cls.EPYC_ZEN5,
                "9755": cls.EPYC_ZEN5,
                "9745": cls.EPYC_ZEN5,
                "9655P": cls.EPYC_ZEN5,
                "9655": cls.EPYC_ZEN5,
                "9645": cls.EPYC_ZEN5,
                "9575F": cls.EPYC_ZEN5,
                "9565": cls.EPYC_ZEN5,
                "9555P": cls.EPYC_ZEN5,
                "9555": cls.EPYC_ZEN5,
                "9375F": cls.EPYC_ZEN5,
                "9365": cls.EPYC_ZEN5,
                "9355P": cls.EPYC_ZEN5,
                "9355": cls.EPYC_ZEN5,
                "9335": cls.EPYC_ZEN5,
                "9275F": cls.EPYC_ZEN5,
                "9255": cls.EPYC_ZEN5,
                "9175F": cls.EPYC_ZEN5,
                "9135": cls.EPYC_ZEN5,
                "9115": cls.EPYC_ZEN5,
                "9015": cls.EPYC_ZEN5,
                # AMD EPYC 9004 series (Zen4)
                "9754": cls.EPYC_ZEN4,
                "9754S": cls.EPYC_ZEN4,
                "9734": cls.EPYC_ZEN4,
                "9654": cls.EPYC_ZEN4,
                "9654P": cls.EPYC_ZEN4,
                "9634": cls.EPYC_ZEN4,
                "9554": cls.EPYC_ZEN4,
                "9554P": cls.EPYC_ZEN4,
                "9534": cls.EPYC_ZEN4,
                "9454": cls.EPYC_ZEN4,
                "9454P": cls.EPYC_ZEN4,
                "9354": cls.EPYC_ZEN4,
                "9354P": cls.EPYC_ZEN4,
                "9334": cls.EPYC_ZEN4,
                "9254": cls.EPYC_ZEN4,
                "9224": cls.EPYC_ZEN4,
                "9124": cls.EPYC_ZEN4,
                # AMD EPYC 9004 with 3D V-Cache
                "9684X": cls.EPYC_ZEN4,
                "9384X": cls.EPYC_ZEN4,
                "9184X": cls.EPYC_ZEN4,
                # High-frequency AMD EPYC 9004
                "9474F": cls.EPYC_ZEN4,
                "9374F": cls.EPYC_ZEN4,
                "9274F": cls.EPYC_ZEN4,
                "9174F": cls.EPYC_ZEN4,
                "9J14": cls.EPYC_ZEN4,
            }
            setattr(cls, cache_attribute_name, mapping)

        return getattr(cls, cache_attribute_name)

    @classmethod
    def from_string_with_default(
        cls, value: Optional[str], default: Optional["AcceleratorModel"] = None
    ) -> Optional["AcceleratorModel"]:
        """Parse an accelerator model from a string, returning a default instead of raising.

        Args:
            value: Identifier to parse (model name, 0x-prefixed GPU device ID, or CPU chip ID).
            default: Value to return if parsing fails. Defaults to None.
        """
        try:
            return cls.from_string(value)
        except ValueError:
            return default

    @classmethod
    def from_string(cls, value: Optional[str]) -> Optional["AcceleratorModel"]:
        """Create an AcceleratorModel from a string.

        Accepts model names (e.g. 'MI300X', 'EPYC_9965'), GPU device IDs
        (e.g. '0x74a1'), or CPU chip IDs (e.g. '9965').

        Returns None if value is None. Raises ValueError if value is unrecognized.
        """
        if value is None:
            return None

        accelerator_mapping = cls._all_accelerator_ids()

        # Try as enum member name (case-insensitive)
        name_value = value.upper()
        try:
            return cls(name_value)
        except ValueError:
            pass

        # Try as accelerator detection ID (GPU device IDs are lowercase hex, CPU chip IDs are as-is)
        device_value = value.lower()
        if device_value in accelerator_mapping:
            return accelerator_mapping[device_value]

        # Try original case and uppercase for CPU chip IDs with suffixes (e.g. "9575F", "9654P")
        if value in accelerator_mapping:
            return accelerator_mapping[value]
        if name_value in accelerator_mapping:
            return accelerator_mapping[name_value]

        raise ValueError(f"Unknown AcceleratorModel: {value!r}")


# Backwards-compatible aliases — kept so existing imports of GPUModel/CPUModel
# continue to work.  Remove once all callers have migrated to AcceleratorModel.
GPUModel = AcceleratorModel
CPUModel = AcceleratorModel


class ProfileCapabilities(BaseModel):
    """Per-profile capability flags used to gate validation behavior.

    All capabilities default to ``False`` so omitting the block (or individual
    fields) in YAML keeps the corresponding validations skipped/strict.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    tool_calling: bool = False
    structured_outputs: bool = False
    reasoning: bool = False


class ProfileMetadata(BaseModel):
    """Metadata information from a profile."""

    model_config = ConfigDict(frozen=True, use_enum_values=False, populate_by_name=True)

    engine: Engine
    accelerator_type: Optional[AcceleratorType] = None
    accelerator_model: Optional[AcceleratorModel] = Field(
        default=None,
        validation_alias=AliasChoices("accelerator_model", "gpu"),
    )
    precision: Precision
    accelerator_count: int = Field(
        ge=0,
        validation_alias=AliasChoices("accelerator_count", "gpu_count"),
    )
    metric: Metric
    manual_selection_only: bool
    type: ProfileType
    capabilities: ProfileCapabilities = Field(default_factory=ProfileCapabilities)
    primary: Optional[bool] = None

    @field_validator("engine", "precision", "metric", "type", "accelerator_type", mode="before")
    @classmethod
    def _case_insensitive_enum(cls, v, info):
        """Parse enum values case-insensitively."""
        if isinstance(v, str):
            enum_map = {
                "engine": Engine,
                "precision": Precision,
                "metric": Metric,
                "type": ProfileType,
                "accelerator_type": AcceleratorType,
            }
            enum_class = enum_map[info.field_name]
            try:
                return enum_class(v)
            except ValueError:
                for member in enum_class:
                    if member.value.lower() == v.lower():
                        return member
                raise
        return v

    @field_validator("accelerator_model", mode="before")
    @classmethod
    def _parse_accelerator(cls, v):
        """Parse accelerator values using AcceleratorModel.from_string for device ID support.

        Handles legacy sentinel value 'NONE' by converting to Python None.
        """
        if v is None:
            return None
        if isinstance(v, str):
            # Handle legacy YAML sentinel value
            if v.upper() == "NONE":
                return None
            return AcceleratorModel.from_string(v)
        return v

    def __str__(self) -> str:
        """Human-readable label for display purposes. Generally, matches profile file name. Currently, there is one case
        where it is different, namely for gpt-oss models' precision (mxfp4 in file name and fp4 in the metadata)"""
        return self.accelerator_label

    def __hash__(self) -> int:
        return hash(
            (
                self.engine,
                self.accelerator_type,
                self.accelerator_model,
                self.precision,
                self.accelerator_count,
                self.metric,
                self.manual_selection_only,
                self.type,
                self.capabilities.tool_calling,
                self.capabilities.structured_outputs,
                self.capabilities.reasoning,
            )
        )

    @property
    def accelerator_label(self) -> str:
        """Human-readable label mirroring the profile id format (e.g. 'vllm-mi300x-fp16-tp1-latency').

        Matches the profile filename stem convention
        ``{engine}-{accelerator}-{precision}-tp{accelerator_count}-{metric}`` so the label
        can be used interchangeably with profile ids in logs and UI surfaces.
        """
        acc_segment = self.accelerator_model.value.lower() if self.accelerator_model else "none"
        return (
            f"{self.engine.value.lower()}-{acc_segment}-{self.precision.value.lower()}"
            f"-tp{self.accelerator_count}-{self.metric.value.lower()}"
        )

    def to_dict(self) -> Dict[str, Any]:
        """Convert the profile metadata to a dictionary for serialization."""
        output = self.model_dump(mode="json", exclude_none=True)
        # Omit the capabilities block when no capability is enabled to keep YAML
        # round-trips minimal for legacy profiles.
        capabilities = output.get("capabilities")
        if isinstance(capabilities, dict) and not any(capabilities.values()):
            output.pop("capabilities", None)
        return output

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ProfileMetadata":
        """Create a ProfileMetadata instance from a dictionary."""
        return cls.model_validate(data)


class ProfileData(BaseModel):
    """Pydantic model for general AIM profile YAML files."""

    metadata: ProfileMetadata
    env_vars: Dict[str, str]
    engine_args: Dict[str, Any]


class ModelProfileData(ProfileData):
    """Pydantic model for model-specific AIM profile YAML files."""

    aim_id: str
    model_id: str


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

    @classmethod
    def from_profile_yaml_path(cls, profile_yaml_path: Path) -> Optional["CanonicalName"]:
        if profile_yaml_path is None:
            return None

        parts = profile_yaml_path.parts
        if len(parts) < 4:
            return None

        org, model_name = profile_yaml_path.parts[-4], profile_yaml_path.parts[-3]
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
