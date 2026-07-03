# Copyright © Advanced Micro Devices, Inc., or its affiliates.
#
# SPDX-License-Identifier: MIT

"""vLLM-Omni engine.

Subclasses :class:`VllmEngine`; uses the vLLM-Omni argument model. By
inheriting :meth:`VllmEngine.apply_engine_defaults`, Omni profiles now also
receive ``served-model-name`` — a deliberate change from the previous
exact-``Engine.VLLM`` check, which excluded Omni. See EAI-5778 PR notes.

The vLLM-Omni argument-validation model (:class:`VllmOmniEngineArgsModel`)
lives here too, next to the engine it describes; it is re-exported from
``aim_runtime.engines``.
"""

from __future__ import annotations

import functools
from typing import Any, ClassVar, Optional

from aim_runtime.engines.engine_args_models import EngineArgsModel
from aim_runtime.engines.vllm import VllmEngine, VllmEngineArgsModel


class VllmOmniEngineArgsModel(VllmEngineArgsModel):
    """Pydantic model mirroring vLLM-Omni ``vllm serve --omni`` CLI arguments.

    When vLLM-Omni is installed, validation delegates to the same argparse tree
    built by ``OmniServeCommand.subparser_init`` (engine + frontend + Omni
    flags). When it is not installed, Pydantic field validation is used as a
    fallback, identical in spirit to :class:`VllmEngineArgsModel`.
    """

    @staticmethod
    @functools.lru_cache(maxsize=None)
    def _vllm_parser() -> Any:
        """Return the vLLM-Omni ``serve`` subcommand parser, or None if unavailable."""
        try:
            from vllm.utils.argparse_utils import FlexibleArgumentParser
            from vllm_omni.entrypoints.cli.serve import OmniServeCommand

            def _raise_on_error(message: str) -> None:
                raise ValueError(message)

            parser = FlexibleArgumentParser()
            subparsers = parser.add_subparsers(dest="vllm_subcommand")
            OmniServeCommand().subparser_init(subparsers)
            parser.error = _raise_on_error  # type: ignore[assignment]
            return parser
        except Exception:
            return None

    # Mirrors CommandGenerator order: serve --omni --model <path> <engine_args>
    _vllm_cli_argv_prefix: ClassVar[tuple[str, ...]] = (
        "serve",
        "--omni",
        "--model",
        "aim-engine-args-validation-placeholder",
    )
    _vllm_validation_label: ClassVar[str] = "vLLM-Omni"

    usp: int | None = None
    vae_patch_parallel_size: int | None = None
    vae_use_tiling: bool | None = None


class VllmOmniEngine(VllmEngine):
    """vLLM-Omni LLM engine."""

    ARGS_MODEL: ClassVar[Optional[type[EngineArgsModel]]] = VllmOmniEngineArgsModel
