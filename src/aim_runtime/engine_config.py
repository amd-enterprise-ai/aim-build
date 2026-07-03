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

from pydantic import BaseModel, ConfigDict

from aim_common import Engine
from aim_runtime.utils import read_yaml

logger = logging.getLogger(__name__)


class EngineConfig(BaseModel):
    """Engine launch configuration, loaded from engines.yaml.

    Holds only the per-deployment launch data. Engine-arg validation and CLI
    serialization format are owned by the engine class (see
    ``aim_runtime.engines``), selected from ``engine`` via ``build_engine``.
    Unknown keys in engines.yaml (e.g. a legacy ``validator``) are ignored.
    """

    model_config = ConfigDict(frozen=True, extra="ignore")

    engine: Engine | None = None
    launch: str
    model_arg: str = ""


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
