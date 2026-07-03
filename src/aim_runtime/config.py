# Copyright © Advanced Micro Devices, Inc., or its affiliates.
#
# SPDX-License-Identifier: MIT

"""
AIM Runtime Configuration Management

This module handles configuration from environment variables and provides
default values for AIM runtime parameters.
"""

import json
import logging
import os
from dataclasses import dataclass
from typing import Any, Dict, Optional, Type, Union

from aim_common import EnumerationType
from aim_common.compat import StrEnum

from .object_model import AcceleratorFamily, AcceleratorModel, AcceleratorType, Engine, Metric, Precision

logger = logging.getLogger(__name__)

# Default paths for AIM runtime
DEFAULT_PROFILE_BASE_PATH = "/workspace/aim-runtime/profiles"
DEFAULT_CONFIG_PATH = "/workspace/aim-runtime/config"
DEFAULT_CACHE_PATH = "/workspace/model-cache"

# LoRA adapter contract (ADR-0004) defaults.
DEFAULT_ADAPTER_SOURCE = "/workspace/cache/adapters/"
DEFAULT_ADAPTER_REFRESH_INTERVAL = 30  # seconds
MIN_ADAPTER_REFRESH_INTERVAL = 5
DEFAULT_ADAPTER_MAX_COUNT = 8
DEFAULT_ADAPTER_MAX_CPU_COUNT = 16
DEFAULT_ADAPTER_MAX_RANK = 32
# Allowed LoRA ranks — mirrors vLLM's MaxLoRARanks (vllm/config/lora.py,
# identical at v0.16.0 and v0.21.0). Intentionally wider than ADR-0004's stated
# {8,16,32,64} so the image never rejects a rank the engine would accept.
ADAPTER_ALLOWED_RANKS = (1, 8, 16, 32, 64, 128, 256, 320, 512)


class AdapterMode(StrEnum):
    """How adapters are managed at runtime (ADR-0004)."""

    STATIC = "static"
    DYNAMIC = "dynamic"


@dataclass
class AIMConfig:
    """Configuration class for AIM runtime parameters."""

    aim_id: Optional[str] = None  # The AIM container identifier (from AIM_ID, for model-specific containers)
    model_id: Optional[str] = None  # The model to deploy (from AIM_MODEL_ID, for base container)
    precision: Optional[Precision] = None  # None = auto-detect best precision
    accelerator_type: AcceleratorType = AcceleratorType.GPU
    accelerator_family: AcceleratorFamily = AcceleratorFamily.INSTINCT
    accelerator_count: Union[int, str] = "auto"
    accelerator_model: Optional[AcceleratorModel] = None
    engine: Engine = Engine.VLLM  # type: ignore[attr-defined]
    metric: Metric = Metric.LATENCY
    profile_id: Optional[str] = None
    profile_base_path: str = DEFAULT_PROFILE_BASE_PATH
    config_path: str = DEFAULT_CONFIG_PATH
    cache_path: str = DEFAULT_CACHE_PATH
    port: int = 8000
    engine_args_override: Optional[Dict[str, Any]] = None
    # LoRA adapter contract (ADR-0004)
    adapter_source: str = DEFAULT_ADAPTER_SOURCE
    adapter_mode: AdapterMode = AdapterMode.STATIC
    adapter_refresh_interval: int = DEFAULT_ADAPTER_REFRESH_INTERVAL
    adapter_max_count: int = DEFAULT_ADAPTER_MAX_COUNT
    adapter_max_cpu_count: int = DEFAULT_ADAPTER_MAX_CPU_COUNT
    adapter_max_rank: int = DEFAULT_ADAPTER_MAX_RANK
    log_level_root: str = "WARNING"
    log_level: str = "INFO"
    allow_general_profile_fallback: bool = (
        True  # Whether to allow fallback to general profiles (from AIM_ALLOW_GENERAL_PROFILE_FALLBACK)
    )
    allow_unoptimized: bool = False  # Whether to allow fallback to unoptimized profiles (from AIM_ALLOW_UNOPTIMIZED)

    def __post_init__(self):
        """Validate configuration after initialization."""
        # Validate that at least one identifier is provided
        if not self.aim_id and not self.model_id:
            raise ValueError("Either AIM_MODEL_ID or AIM_ID must be provided")

        # Validate that only one of aim_id or model_id is set, not both
        # This maintains clear separation: model-specific containers use AIM_ID,
        # base containers use AIM_MODEL_ID
        if self.aim_id and self.model_id:
            logger.error(f"AIM_ID {self.aim_id} and AIM_MODEL_ID {self.model_id} are mutually exclusive.")
            raise ValueError("Cannot set both AIM_ID and AIM_MODEL_ID. Only one should be set.")

    @classmethod
    def _read_accelerator_count(cls):
        """Read accelerator count from AIM_ACCELERATOR_COUNT (or deprecated AIM_GPU_COUNT)."""
        value = os.environ.get("AIM_ACCELERATOR_COUNT")
        if value is None:
            value = os.environ.get("AIM_GPU_COUNT")
            if value is not None:
                logger.warning("AIM_GPU_COUNT is deprecated, use AIM_ACCELERATOR_COUNT instead.")
        if value is None:
            return "auto"

        if value == "auto":
            return "auto"

        try:
            return int(value)
        except ValueError:
            logger.warning(f"AIM_ACCELERATOR_COUNT must be 'auto' or an integer. Was {value}. Defaulting to 'auto'.")
            return "auto"

    @classmethod
    def _read_enum(cls, name: str, default: str, enum: Type[EnumerationType]) -> EnumerationType:
        value = os.environ.get(name, default)
        try:
            return enum(value.lower())
        except ValueError:
            logger.warning(f"{name} must be one of {[e.value for e in enum]}. Was {value}. Defaulting to {default}.")
            return enum(default.lower())

    @classmethod
    def _read_accelerator_model(cls) -> Optional[AcceleratorModel]:
        """Read accelerator model from AIM_ACCELERATOR_MODEL (or deprecated AIM_GPU_MODEL)."""
        value = os.environ.get("AIM_ACCELERATOR_MODEL")
        if not value:
            value = os.environ.get("AIM_GPU_MODEL")
            if value:
                logger.warning("AIM_GPU_MODEL is deprecated, use AIM_ACCELERATOR_MODEL instead.")
        if not value:
            return None
        result = AcceleratorModel.from_string_with_default(value)
        if result is None:
            logger.warning(
                f"AIM_ACCELERATOR_MODEL must be one of {[e.value for e in AcceleratorModel]}. "
                f"Was {value}. Ignoring."
            )
        return result

    @classmethod
    def _read_precision(cls) -> Optional[Precision]:
        """Read AIM_PRECISION env var. Returns None (auto-detect) if unset."""
        value = os.environ.get("AIM_PRECISION")
        if not value or value.lower() == "auto":
            return None
        try:
            return Precision(value.lower())
        except ValueError:
            logger.warning(f"AIM_PRECISION must be one of {[e.value for e in Precision]}. Was {value}. Ignoring.")
            return None

    @classmethod
    def _read_engine_args_override(cls) -> Optional[Dict[str, Any]]:
        """Parse AIM_ENGINE_ARGS environment variable as JSON."""
        engine_args_str = os.environ.get("AIM_ENGINE_ARGS", "")

        if not engine_args_str:
            return None

        try:
            engine_args = json.loads(engine_args_str)

            if not isinstance(engine_args, dict):
                logger.warning(
                    f"AIM_ENGINE_ARGS must be a JSON object/dict. Got {type(engine_args).__name__}. Ignoring."
                )
                return None

            logger.info(f"Loaded {len(engine_args)} engine argument overrides from AIM_ENGINE_ARGS")
            return engine_args

        except json.JSONDecodeError as e:
            logger.warning(f"Failed to parse AIM_ENGINE_ARGS as JSON: {e}. Ignoring engine args override.")
            return None

    @classmethod
    def _read_bool(cls, name: str, default: bool) -> bool:
        """Read a boolean environment variable.

        Accepts: true/false, yes/no, 1/0 (case-insensitive).
        Invalid values log a warning and return the default.
        """
        value = os.environ.get(name, "").lower()

        if not value:
            return default

        if value in ("true", "yes", "1"):
            return True
        elif value in ("false", "no", "0"):
            return False
        else:
            logger.warning(
                f"{name} must be a boolean value (true/false, yes/no, 1/0). Was '{value}'. Defaulting to {default}."
            )
            return default

    @classmethod
    def _read_int_min(cls, name: str, default: int, minimum: int) -> int:
        """Read an integer env var, clamping to a minimum; warn + default on bad input."""
        raw = os.environ.get(name)
        if raw is None:
            return default
        try:
            value = int(raw)
        except ValueError:
            logger.warning(f"{name} must be an integer. Was '{raw}'. Defaulting to {default}.")
            return default
        if value < minimum:
            logger.warning(f"{name}={value} is below the minimum {minimum}. Using {minimum}.")
            return minimum
        return value

    @classmethod
    def _read_adapter_mode(cls) -> AdapterMode:
        """Read AIM_ADAPTER_MODE; default static on missing/invalid."""
        raw = os.environ.get("AIM_ADAPTER_MODE")
        if not raw:
            return AdapterMode.STATIC
        try:
            return AdapterMode(raw.lower())
        except ValueError:
            logger.warning(
                f"AIM_ADAPTER_MODE must be one of {[m.value for m in AdapterMode]}. "
                f"Was '{raw}'. Defaulting to static."
            )
            return AdapterMode.STATIC

    @classmethod
    def _read_adapter_max_rank(cls) -> int:
        """Read AIM_ADAPTER_MAX_RANK; must be one of vLLM's allowed ranks."""
        raw = os.environ.get("AIM_ADAPTER_MAX_RANK")
        if raw is None:
            return DEFAULT_ADAPTER_MAX_RANK
        try:
            value = int(raw)
        except ValueError:
            logger.warning(
                f"AIM_ADAPTER_MAX_RANK must be an integer. Was '{raw}'. Defaulting to {DEFAULT_ADAPTER_MAX_RANK}."
            )
            return DEFAULT_ADAPTER_MAX_RANK
        if value not in ADAPTER_ALLOWED_RANKS:
            logger.warning(
                f"AIM_ADAPTER_MAX_RANK={value} is not one of the allowed ranks {ADAPTER_ALLOWED_RANKS}. "
                f"Defaulting to {DEFAULT_ADAPTER_MAX_RANK}."
            )
            return DEFAULT_ADAPTER_MAX_RANK
        return value

    @classmethod
    def from_environment(cls, model_id_param: Optional[str] = None) -> "AIMConfig":
        """Create configuration from environment variables.

        Args:
            model_id_param: Model ID provided as a parameter (overrides AIM_MODEL_ID if AIM_ID is not set)

        AIM_ID: Identifies the model-specific AIM container
        AIM_MODEL_ID: Specifies the model to deploy (base containers only)

        These serve different purposes:
        - AIM_ID:
          * Used for profile filtering (determines which org/model profiles to search)
          * Example: "meta-llama/Llama-3.1-8B-Instruct" -> searches /workspace/aim-runtime/profiles/meta-llama/Llama-3.1-8B-Instruct/
          * Also serves as the model identifier when using general profiles in model-specific containers

        - AIM_MODEL_ID:
          * Actual model to deploy at runtime (for base containers)
          * Example: "meta-llama/Llama-3.1-8B-Instruct" or "amd/Llama-3.1-8B-Instruct-FP8-KV"

        Container types:
        - Model-specific containers: AIM_ID is set (uses model-specific profiles, falls back to general)
        - Base containers: AIM_MODEL_ID is set (uses general profiles only)

        Model resolution order (in command_generator.py):
        1. Profile model_id (from model-specific profile YAML)
        2. Config model_id (from AIM_MODEL_ID environment variable)
        3. Config aim_id (fallback for general profiles in model-specific containers)

        IMPORTANT: Only one of AIM_ID or AIM_MODEL_ID can be set, not both.
        """
        aim_id = os.environ.get("AIM_ID")
        model_id = os.environ.get("AIM_MODEL_ID")
        accelerator_type_str = os.environ.get("AIM_ACCELERATOR_TYPE", "gpu").lower()
        accelerator_family_str = os.environ.get("AIM_ACCELERATOR_FAMILY", "instinct").lower()

        logger.debug(f"Read AIM_ID: {aim_id}")
        logger.debug(f"Read AIM_MODEL_ID: {model_id}")
        logger.debug(f"Read AIM_ACCELERATOR_TYPE: {accelerator_type_str}")
        logger.debug(f"Read AIM_ACCELERATOR_FAMILY: {accelerator_family_str}")

        try:
            accelerator_type = AcceleratorType(accelerator_type_str)
        except ValueError:
            allowed_values = ", ".join(list(map(str, AcceleratorType)))
            error_message = f"AIM_ACCELERATOR_TYPE must be one of [{allowed_values}]. Was '{accelerator_type_str}'."
            raise ValueError(error_message)

        try:
            accelerator_family = AcceleratorFamily(accelerator_family_str)
        except ValueError:
            allowed_values = ", ".join(list(map(str, AcceleratorFamily)))
            error_message = f"AIM_ACCELERATOR_FAMILY must be one of [{allowed_values}]. Was '{accelerator_family_str}'."
            raise ValueError(error_message)

        # Validate that only one is set, not both
        if aim_id and model_id:
            raise ValueError("Cannot set both AIM_ID and AIM_MODEL_ID. Only one should be set.")

        # At least one must be provided
        if not aim_id and not model_id:
            if model_id_param:
                model_id = model_id_param
            else:
                raise ValueError("Either AIM_MODEL_ID or AIM_ID environment variable is required")

        # Adapter caps: CPU cap must be >= GPU cap (vLLM/SGLang requirement).
        adapter_max_count = cls._read_int_min("AIM_ADAPTER_MAX_COUNT", DEFAULT_ADAPTER_MAX_COUNT, minimum=1)
        adapter_max_cpu_count = cls._read_int_min("AIM_ADAPTER_MAX_CPU_COUNT", DEFAULT_ADAPTER_MAX_CPU_COUNT, minimum=1)
        if adapter_max_cpu_count < adapter_max_count:
            logger.warning(
                f"AIM_ADAPTER_MAX_CPU_COUNT={adapter_max_cpu_count} is below "
                f"AIM_ADAPTER_MAX_COUNT={adapter_max_count}; raising it to match."
            )
            adapter_max_cpu_count = adapter_max_count

        return cls(
            aim_id=aim_id,
            model_id=model_id,
            precision=cls._read_precision(),
            accelerator_type=accelerator_type,
            accelerator_family=accelerator_family,
            accelerator_count=cls._read_accelerator_count(),
            accelerator_model=cls._read_accelerator_model(),
            engine=cls._read_enum("AIM_ENGINE", "vllm", Engine),
            metric=cls._read_enum("AIM_METRIC", "latency", Metric),
            profile_id=os.environ.get("AIM_PROFILE_ID"),
            profile_base_path=DEFAULT_PROFILE_BASE_PATH,
            config_path=DEFAULT_CONFIG_PATH,
            cache_path=os.environ.get("AIM_CACHE_PATH", DEFAULT_CACHE_PATH),
            port=int(os.environ.get("AIM_PORT", 8000)),
            engine_args_override=cls._read_engine_args_override(),
            adapter_source=os.environ.get("AIM_ADAPTER_SOURCE", DEFAULT_ADAPTER_SOURCE),
            adapter_mode=cls._read_adapter_mode(),
            adapter_refresh_interval=cls._read_int_min(
                "AIM_ADAPTER_REFRESH_INTERVAL", DEFAULT_ADAPTER_REFRESH_INTERVAL, minimum=MIN_ADAPTER_REFRESH_INTERVAL
            ),
            adapter_max_count=adapter_max_count,
            adapter_max_cpu_count=adapter_max_cpu_count,
            adapter_max_rank=cls._read_adapter_max_rank(),
            log_level_root=os.environ.get("AIM_LOG_LEVEL_ROOT", "WARNING"),
            log_level=os.environ.get("AIM_LOG_LEVEL", "INFO"),
            allow_general_profile_fallback=cls._read_bool("AIM_ALLOW_GENERAL_PROFILE_FALLBACK", True),
            allow_unoptimized=cls._read_bool("AIM_ALLOW_UNOPTIMIZED", False),
        )

    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary."""
        return {
            "aim_id": self.aim_id,
            "model_id": self.model_id,
            "precision": self.precision,
            "accelerator_type": self.accelerator_type,
            "accelerator_count": self.accelerator_count,
            "accelerator_model": self.accelerator_model,
            "engine": self.engine,
            "metric": self.metric,
            "profile_id": self.profile_id,
            "profile_base_path": self.profile_base_path,
            "config_path": self.config_path,
            "cache_path": self.cache_path,
            "port": self.port,
            "engine_args_override": self.engine_args_override,
            "adapter_source": self.adapter_source,
            "adapter_mode": self.adapter_mode,
            "adapter_refresh_interval": self.adapter_refresh_interval,
            "adapter_max_count": self.adapter_max_count,
            "adapter_max_cpu_count": self.adapter_max_cpu_count,
            "adapter_max_rank": self.adapter_max_rank,
            "log_level_root": self.log_level_root,
            "log_level": self.log_level,
            "allow_general_profile_fallback": self.allow_general_profile_fallback,
            "allow_unoptimized": self.allow_unoptimized,
        }

    def log_debug_info(self) -> None:
        """Log configuration information at debug level."""
        logger.debug("--- AIM Runtime Configuration ---")
        for key, value in self.to_dict().items():
            logger.debug(f"{key.upper()}: {value}")
        logger.debug("--------------------------------")
