# Copyright © Advanced Micro Devices, Inc., or its affiliates.
#
# SPDX-License-Identifier: MIT

"""
Engine configuration — YAML-driven engine definitions and validator registry.

Engine-specific details (launch command, model argument, validation) are defined
in config/engines.yaml rather than in code. Adding a new engine requires only a
YAML entry in engines.yaml.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Callable

import yaml
from pydantic import BaseModel, ConfigDict

from aim_common import Engine

logger = logging.getLogger(__name__)


class EngineConfig(BaseModel):
    """Engine launch and validation configuration, loaded from engines.yaml."""

    model_config = ConfigDict(frozen=True)

    launch: str
    model_arg: str
    validator: str = ""


# ---------------------------------------------------------------------------
# Validator registry
# ---------------------------------------------------------------------------

VALIDATORS: dict[str, Callable[[dict[str, Any]], None]] = {}

try:
    from vllm.entrypoints.openai.cli_args import make_arg_parser
    from vllm.utils.argparse_utils import FlexibleArgumentParser

    _VLLM_AVAILABLE: bool = True
except Exception:
    _VLLM_AVAILABLE = False


def _make_vllm_cli_parser() -> FlexibleArgumentParser:
    """Build a fresh vLLM CLI arg parser (engine + frontend args)."""
    return make_arg_parser(FlexibleArgumentParser())


def _engine_args_to_cli_list(engine_args: dict[str, Any]) -> list[str]:
    """Convert an engine_args dict (kebab-case keys) to a CLI argument list.

    Mirrors the conversion in CommandGenerator._build_engine_args() so
    that the validator sees exactly the same flags the runtime would pass.
    """
    cli_args: list[str] = []
    for key, value in engine_args.items():
        if value is None:
            cli_args.append(f"--{key}")
        elif isinstance(value, bool):
            if value:
                cli_args.append(f"--{key}")
        elif isinstance(value, (list, tuple)):
            cli_args.append(f"--{key}")
            for item in value:
                cli_args.append(str(item))
        elif isinstance(value, dict):
            cli_args.extend([f"--{key}", json.dumps(value)])
        else:
            cli_args.extend([f"--{key}", str(value)])
    return cli_args


def _validate_vllm(engine_args: dict[str, Any]) -> None:
    """Validate engine_args using vLLM's full CLI arg parser.

    Constructs the same argparse parser that ``vllm.entrypoints.openai.api_server``
    uses (via ``make_arg_parser``), which registers *both* engine args
    (``EngineArgs``) and server/frontend args (``FrontendArgs``).  This lets us
    validate the complete CLI surface in one pass — typos in either category are
    caught, and legitimate frontend args like ``--disable-uvicorn-access-log``
    are accepted without a manual allow-list.
    """
    parser = _make_vllm_cli_parser()
    cli_args = _engine_args_to_cli_list(engine_args)

    # Override parser.error() to raise instead of calling sys.exit()
    def _raise_on_error(message: str) -> None:
        raise ValueError(message)

    parser.error = _raise_on_error  # type: ignore[assignment]
    try:
        parser.parse_args(cli_args)
    except SystemExit as e:
        raise ValueError(f"vLLM engine_args validation failed (parser exit code {e.code})") from e
    except ValueError as e:
        raise ValueError(f"vLLM engine_args validation failed: {e}") from e


if _VLLM_AVAILABLE:
    VALIDATORS["vllm"] = _validate_vllm
else:
    # Fallback: use Pydantic VllmEngineArgsModel when native vLLM is not installed
    try:
        from aim_common.vllm_engine_args_model import VllmEngineArgsModel

        def _validate_vllm_pydantic(engine_args: dict[str, Any]) -> None:
            """Validate engine_args against VllmEngineArgsModel (kebab-to-snake conversion)."""
            converted = {k.replace("-", "_"): v for k, v in engine_args.items()}
            VllmEngineArgsModel(**converted)

        VALIDATORS["vllm"] = _validate_vllm_pydantic
    except ImportError:
        pass


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
        ValueError: If the engine is not defined in engines.yaml.
        FileNotFoundError: If engines.yaml does not exist.
    """
    engines_path = Path(config_dir) / "engines.yaml"
    with open(engines_path) as f:
        engines = yaml.safe_load(f)

    if not isinstance(engines, dict):
        raise ValueError(f"Invalid or empty engines.yaml: {engines_path}")

    if engine.value not in engines:
        available = list(engines.keys())
        raise ValueError(
            f"No configuration for engine '{engine.value}' in {engines_path}. " f"Available engines: {available}"
        )

    raw = engines[engine.value]
    return EngineConfig(
        launch=raw["launch"],
        model_arg=raw["model_arg"],
        validator=raw.get("validator", ""),
    )
