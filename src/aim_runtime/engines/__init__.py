# Copyright © Advanced Micro Devices, Inc., or its affiliates.
#
# SPDX-License-Identifier: MIT

"""Engine abstraction package.

Houses the per-engine ``BaseEngine`` hierarchy plus its factory, and
re-exports each engine's argument-validation model for a stable import
surface. The engine-argument machinery is runtime-only (it moved here from
``aim_common`` because the CI side never imported it). Each concrete model now
lives next to its engine: ``VllmEngineArgsModel`` in ``vllm.py``,
``BentomlEngineArgsModel`` in ``bentoml.py``, ``VllmOmniEngineArgsModel`` in
``vllm_omni.py``; only the shared base, format enum, and serializer remain in
``engine_args_models``.

Engine selection keys off the :class:`~aim_common.object_model.Engine` enum:
``engine_class_for`` returns the class (usable at profile-load time, no
instance needed); ``build_engine`` instantiates it.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from aim_common.object_model import Engine
from aim_runtime.engines.base import BaseEngine
from aim_runtime.engines.bentoml import BentomlEngine, BentomlEngineArgsModel
from aim_runtime.engines.engine_args_models import (
    EngineArgsFormat,
    EngineArgsModel,
    engine_args_to_cli_list,
)
from aim_runtime.engines.vllm import VllmEngine, VllmEngineArgsModel, validate_vllm_env_vars
from aim_runtime.engines.vllm_omni import VllmOmniEngine, VllmOmniEngineArgsModel

if TYPE_CHECKING:
    from aim_runtime.config import AIMConfig
    from aim_runtime.engine_config import EngineConfig

# Per-engine class map — the single dispatch point from the Engine enum to a
# concrete engine class. Replaces the former ENGINE_ARGS_MODELS registry.
# Keep in sync with the ``Engine`` enum: every enum member must have an entry
# here, or ``build_engine``/``engine_class_for`` raises for that engine.
ENGINE_CLASSES: dict[Engine, type[BaseEngine]] = {
    Engine.VLLM: VllmEngine,
    Engine.VLLM_OMNI: VllmOmniEngine,
    Engine.BENTOML: BentomlEngine,
}


def engine_class_for(engine: Engine) -> type[BaseEngine]:
    """Return the engine class for an ``Engine`` enum value.

    Usable without an instance (e.g. profile-load-time validation).

    Raises:
        ValueError: if the engine has no registered class.
    """
    try:
        return ENGINE_CLASSES[engine]
    except KeyError as exc:
        available = ", ".join(sorted(e.value for e in ENGINE_CLASSES))
        raise ValueError(f"No engine class registered for '{engine}'. Available: {available}") from exc


def build_engine(config: "AIMConfig", engine_config: "EngineConfig") -> BaseEngine:
    """Instantiate the concrete engine for ``config.engine``.

    ``config.engine`` is the single source of truth for class selection. When
    ``engine_config.engine`` is also set (``load_engine_config`` stamps it from
    the requested engine) the two must agree; a mismatch means the launch
    config was loaded for a different engine than the AIM declares, so we fail
    loudly rather than build an engine class against a mismatched launch config.

    Raises:
        ValueError: if ``engine_config.engine`` is set and differs from ``config.engine``.
    """
    if engine_config.engine is not None and engine_config.engine != config.engine:
        raise ValueError(
            f"Engine mismatch: AIM config declares '{config.engine}' but engine_config "
            f"is for '{engine_config.engine}'."
        )
    return engine_class_for(config.engine)(config, engine_config)


__all__ = [
    "BaseEngine",
    "BentomlEngine",
    "BentomlEngineArgsModel",
    "ENGINE_CLASSES",
    "EngineArgsFormat",
    "EngineArgsModel",
    "VllmEngine",
    "VllmEngineArgsModel",
    "VllmOmniEngine",
    "VllmOmniEngineArgsModel",
    "build_engine",
    "engine_args_to_cli_list",
    "engine_class_for",
    "validate_vllm_env_vars",
]
