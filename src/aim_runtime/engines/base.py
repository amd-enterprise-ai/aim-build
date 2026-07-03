# Copyright © Advanced Micro Devices, Inc., or its affiliates.
#
# SPDX-License-Identifier: MIT

"""Engine abstraction base class.

``BaseEngine`` encapsulates every engine-specific step the runtime needs:
building the launch prefix, positioning the model argument, applying
engine-specific argument defaults, validating engine args / env vars, and
serializing args to a CLI list. Concrete engines subclass this and declare
their argument model and serialization format as class attributes, replacing
the former ``ENGINE_ARGS_MODELS`` registry and ``EngineConfig`` format
inference with a single source of truth on the class.

``validate_engine_args`` and ``validate_env_vars`` are classmethods so they
can be called at profile-load time (dispatched per profile engine) without an
engine instance.
"""

from __future__ import annotations

import logging
import shlex
import shutil
from abc import ABC
from typing import TYPE_CHECKING, Any, ClassVar, Optional

from aim_runtime.engines.engine_args_models import (
    EngineArgsFormat,
    EngineArgsModel,
    engine_args_to_cli_list,
)

if TYPE_CHECKING:
    from aim_runtime.config import AIMConfig
    from aim_runtime.engine_config import EngineConfig
    from aim_runtime.object_model import Profile

logger = logging.getLogger(__name__)


class BaseEngine(ABC):
    """Abstract base for serving engines.

    Subclasses declare:
        ARGS_MODEL: the ``EngineArgsModel`` subclass used to validate engine
            args, or ``None`` to skip native validation.
        ARGS_FORMAT: how engine args serialize on the CLI.
        requires_aiter_kernels: whether the runtime should install pre-built
            AITER kernels for this engine before launch.
    """

    ARGS_MODEL: ClassVar[Optional[type[EngineArgsModel]]] = None
    ARGS_FORMAT: ClassVar[EngineArgsFormat] = EngineArgsFormat.STANDARD
    requires_aiter_kernels: ClassVar[bool] = False

    def __init__(self, config: "AIMConfig", engine_config: "EngineConfig") -> None:
        self.config = config
        self.engine_config = engine_config

    def launch_prefix(self) -> list[str]:
        """Return the launch command prefix as an argv list.

        Splits ``engine_config.launch`` and resolves a leading ``python`` to
        ``python``/``python3`` depending on what is on PATH.
        """
        launch = shlex.split(self.engine_config.launch)
        if launch and launch[0] == "python":
            launch[0] = "python" if shutil.which("python") else "python3"
        return launch

    @property
    def model_arg(self) -> str:
        """CLI flag used to pass the model path (empty when the model is embedded in launch)."""
        return self.engine_config.model_arg

    def apply_engine_defaults(self, engine_args: dict[str, Any], served_model_names: list[str]) -> None:
        """Apply engine-specific argument defaults in place. Base: no-op."""
        return None

    @classmethod
    def validate_engine_args(cls, engine_args: dict[str, Any]) -> None:
        """Validate engine args via ``cls.ARGS_MODEL`` (no-op when unset)."""
        if cls.ARGS_MODEL is not None:
            cls.ARGS_MODEL.model_validate(engine_args)

    @classmethod
    def validate_env_vars(cls, env_vars: dict[str, str], source: str = "") -> None:
        """Validate engine-specific env vars. Base: no-op."""
        return None

    def serialize_engine_args(self, engine_args: dict[str, Any]) -> list[str]:
        """Serialize engine args to a CLI list using this engine's format."""
        return engine_args_to_cli_list(engine_args, self.ARGS_FORMAT)

    def needs_runtime_supervisor(self, profile: "Profile") -> bool:
        """Whether serve() must supervise the engine (vs. os.execv replace).

        Base engines run via os.execv. Engines that need an in-pod sidecar
        process (e.g. the dynamic-mode LoRA watcher) override this.
        """
        return False
