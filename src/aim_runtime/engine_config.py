# Copyright © Advanced Micro Devices, Inc., or its affiliates.
#
# SPDX-License-Identifier: MIT

"""
Engine configuration — YAML-driven engine definitions and validator registry.

Engine-specific details (launch command, model argument, validation) are defined
in config/engines.yaml rather than in code. Adding a new engine requires only a
YAML entry in engines.yaml.
"""

import logging
from pathlib import Path
from typing import ClassVar

from pydantic import BaseModel, ConfigDict, model_validator

from aim_common import Engine
from aim_common.engine_args_models import EngineArgsFormat
from aim_runtime.utils import read_yaml

logger = logging.getLogger(__name__)


class EngineConfig(BaseModel):
    """Engine launch and validation configuration, loaded from engines.yaml."""

    model_config = ConfigDict(frozen=True)

    # Implicit args_format per engine; engines not listed default to STANDARD.
    ENGINE_ARGS_FORMATS: ClassVar[dict[str, EngineArgsFormat]] = {
        "bentoml": EngineArgsFormat.FORWARDED,
    }

    engine: Engine | None = None
    launch: str
    model_arg: str = ""
    validator: str = ""
    args_format: EngineArgsFormat = EngineArgsFormat.STANDARD

    @model_validator(mode="after")
    def _infer_args_format(self) -> "EngineConfig":
        """Infer args_format from engine name when not explicitly set.

        object.__setattr__ is the standard pattern for mutating frozen Pydantic
        models inside a mode="after" validator — it bypasses __setattr__ without
        disabling the frozen constraint for external callers.
        """
        if "args_format" not in self.model_fields_set and self.engine in self.ENGINE_ARGS_FORMATS:
            object.__setattr__(self, "args_format", self.ENGINE_ARGS_FORMATS[self.engine])
        return self


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------


def load_engine_config(engine: Engine, config_dir: str) -> EngineConfig:
    """Load engine configuration from engines.yaml.

    Args:
        engine: The engine to load config for.
        config_dir: Path to directory containing engines.yaml.

    Returns:
        EngineConfig for the requested engine.

    Raises:
        ValueError: If the engine is not defined in engines.yaml or engines.yaml does not exist.
    """
    engines_path = Path(config_dir) / "engines.yaml"
    engines = read_yaml(engines_path)

    if not isinstance(engines, dict):
        raise ValueError(f"Invalid or empty engines.yaml: {engines_path}")

    if engine.value not in engines:
        available = list(engines.keys())
        raise ValueError(
            f"No configuration for engine '{engine.value}' in {engines_path}. " f"Available engines: {available}"
        )

    raw = engines[engine.value]
    return EngineConfig(**raw, engine=engine)
