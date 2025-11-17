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

from .object_model import Engine, EnumerationType, GPUModel, Metric, Precision

logger = logging.getLogger(__name__)

# Default paths for AIM runtime
DEFAULT_PROFILE_BASE_PATH = "/workspace/aim-runtime/profiles"
DEFAULT_SCHEMA_SEARCH_PATH = "/workspace/aim-runtime/schemas"
DEFAULT_CACHE_PATH = "/workspace/model-cache"


@dataclass
class AIMConfig:
    """Configuration class for AIM runtime parameters."""

    aim_id: Optional[str] = None  # The AIM container identifier (from AIM_ID, for model-specific containers)
    model_id: Optional[str] = None  # The model to deploy (from AIM_MODEL_ID, for base container)
    precision: Precision = Precision.AUTO
    gpu_count: Union[int, str] = "auto"
    gpu_model: Optional[GPUModel] = None
    engine: Engine = Engine.VLLM
    metric: Metric = Metric.LATENCY
    profile_id: Optional[str] = None
    profile_base_path: str = DEFAULT_PROFILE_BASE_PATH
    schema_search_path: str = DEFAULT_SCHEMA_SEARCH_PATH
    cache_path: str = DEFAULT_CACHE_PATH
    port: int = 8000
    engine_args_override: Optional[Dict[str, Any]] = None
    log_level_root: str = "WARNING"
    log_level: str = "INFO"
    custom_model_name: Optional[str] = None  # Custom directory name for downloaded models (from AIM_CUSTOM_MODEL_NAME)
    allow_general_profile_fallback: bool = (
        True  # Whether to allow fallback to general profiles (from AIM_ALLOW_GENERAL_PROFILE_FALLBACK)
    )

    def __post_init__(self):
        """Validate configuration after initialization."""
        # Validate that at least one identifier is provided
        if not self.aim_id and not self.model_id:
            raise ValueError("Either AIM_MODEL_ID or AIM_ID must be provided")

        # Validate that only one of aim_id or model_id is set, not both
        # This maintains clear separation: model-specific containers use AIM_ID,
        # base containers use AIM_MODEL_ID
        if self.aim_id and self.model_id:
            raise ValueError("Cannot set both AIM_ID and AIM_MODEL_ID. Only one should be set.")

    @classmethod
    def _read_gpu_count(cls):
        gpu_count = os.environ.get("AIM_GPU_COUNT", "auto")

        if gpu_count == "auto":
            return "auto"

        try:
            return int(gpu_count)
        except ValueError:
            logger.warning(f"AIM_GPU_COUNT must be 'auto' or an integer. Was {gpu_count}. Defaulting to 'auto'.")
            return "auto"

    @classmethod
    def _read_enum(cls, name: str, default: str, enum: Type[EnumerationType]) -> EnumerationType:
        value = os.environ.get(name, default)
        # GPUModel enum values are uppercase, so convert appropriately
        if enum == GPUModel:
            value = value.upper()
        else:
            value = value.lower()
        try:
            return enum(value)
        except ValueError:
            logger.warning(f"{name} must be one of {[e.value for e in enum]}. Was {value}. Defaulting to {default}.")
            return enum(default)

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
    def from_environment(cls) -> "AIMConfig":
        """Create configuration from environment variables.

        AIM_ID: Identifies the model-specific AIM container
        AIM_MODEL_ID: Specifies the model to deploy (base containers only)

        These serve different purposes:
        - AIM_ID:
          * Used for profile filtering (determines which org/model profiles to search)
          * Example: "meta-llama/Llama-3.1-8B-Instruct" -> searches profiles/meta-llama/Llama-3.1-8B-Instruct/
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

        # Validate that only one is set, not both
        if aim_id and model_id:
            raise ValueError("Cannot set both AIM_ID and AIM_MODEL_ID. Only one should be set.")

        # At least one must be provided
        if not aim_id and not model_id:
            raise ValueError("Either AIM_MODEL_ID or AIM_ID environment variable is required")

        return cls(
            aim_id=aim_id,
            model_id=model_id,
            precision=cls._read_enum("AIM_PRECISION", "auto", Precision),
            gpu_count=cls._read_gpu_count(),
            gpu_model=cls._read_enum("AIM_GPU_MODEL", "NONE", GPUModel),
            engine=cls._read_enum("AIM_ENGINE", "vllm", Engine),
            metric=cls._read_enum("AIM_METRIC", "latency", Metric),
            profile_id=os.environ.get("AIM_PROFILE_ID"),
            profile_base_path=DEFAULT_PROFILE_BASE_PATH,
            schema_search_path=DEFAULT_SCHEMA_SEARCH_PATH,
            cache_path=os.environ.get("AIM_CACHE_PATH", DEFAULT_CACHE_PATH),
            port=int(os.environ.get("AIM_PORT", 8000)),
            engine_args_override=cls._read_engine_args_override(),
            log_level_root=os.environ.get("AIM_LOG_LEVEL_ROOT", "WARNING"),
            log_level=os.environ.get("AIM_LOG_LEVEL", "INFO"),
            custom_model_name=os.environ.get("AIM_CUSTOM_MODEL_NAME"),
            allow_general_profile_fallback=cls._read_bool("AIM_ALLOW_GENERAL_PROFILE_FALLBACK", True),
        )

    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary."""
        return {
            "aim_id": self.aim_id,
            "model_id": self.model_id,
            "precision": self.precision,
            "gpu_count": self.gpu_count,
            "gpu_model": self.gpu_model,
            "engine": self.engine,
            "metric": self.metric,
            "profile_id": self.profile_id,
            "profile_base_path": self.profile_base_path,
            "schema_search_path": self.schema_search_path,
            "cache_path": self.cache_path,
            "custom_model_name": self.custom_model_name,
            "port": self.port,
            "engine_args_override": self.engine_args_override,
            "log_level_root": self.log_level_root,
            "log_level": self.log_level,
            "allow_general_profile_fallback": self.allow_general_profile_fallback,
        }

    def log_debug_info(self) -> None:
        """Log configuration information at debug level."""
        logger.debug("--- AIM Runtime Configuration ---")
        for key, value in self.to_dict().items():
            logger.debug(f"{key.upper()}: {value}")
        logger.debug("--------------------------------")
