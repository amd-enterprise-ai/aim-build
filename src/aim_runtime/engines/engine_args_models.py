# Copyright © Advanced Micro Devices, Inc., or its affiliates.
#
# SPDX-License-Identifier: MIT

"""
Shared building blocks for engine-argument validation models.

This module holds only the engine-agnostic pieces: the CLI serialization
format enum, the ``EngineArgsModel`` base class, and the
``engine_args_to_cli_list`` serializer. Each engine's concrete argument model
lives in that engine's module (e.g. ``VllmEngineArgsModel`` in ``vllm.py``,
``BentomlEngineArgsModel`` in ``bentoml.py``) so the validation rules sit next
to the engine they describe. They are re-exported from ``aim_runtime.engines``
for a stable import surface.

Each engine model mirrors the CLI arguments of its respective serving engine.
Validation is performed by calling ``Model.model_validate(engine_args)``,
where *engine_args* is a raw ``dict[str, Any]`` whose keys may be either
kebab-case (as they appear in YAML / ``AIM_ENGINE_ARGS``) or snake_case.
Both forms are accepted because the base model is configured with
``alias_generator`` (underscore → hyphen) and ``populate_by_name=True``.

Adding support for a new engine
--------------------------------
1. Create a subclass of ``EngineArgsModel`` in the engine's module.
2. Declare fields with standard Python snake_case names; the kebab-case alias
   is generated automatically.
3. If the engine ships its own native argument class, add a
   ``@model_validator(mode="wrap")`` that delegates to it when available,
   mirroring the pattern e.g. in ``VllmEngineArgsModel``.
4. Reference it as the ``ARGS_MODEL`` class attribute on the engine class and
   register that engine class in ``ENGINE_CLASSES`` (in ``__init__``).
"""

import json
from typing import Any

from pydantic import BaseModel, ConfigDict

from aim_common.compat import StrEnum


class EngineArgsFormat(StrEnum):
    """How engine_args are serialized on the command line.

    STANDARD:   --key value  (default, used by vLLM, sglang, llama.cpp, …)
    FORWARDED:  --arg key=value  (used by engines that accept forwarded KV pairs,
                e.g. BentoML ``serve --arg …``)
    """

    STANDARD = "standard"
    FORWARDED = "forwarded"


class EngineArgsModel(BaseModel):
    """Base class for all engine argument Pydantic models.

    Subclasses inherit ``model_config`` so they accept both the kebab-case
    CLI alias (auto-generated from the snake_case field name) and the Python
    snake_case field name, and silently pass through any extra keys that are
    not declared as fields (``extra="allow"``).
    """

    model_config = ConfigDict(
        extra="allow",
        populate_by_name=True,
        alias_generator=lambda field_name: field_name.replace("_", "-"),
    )


def engine_args_to_cli_list(
    engine_args: dict[str, Any],
    args_format: EngineArgsFormat = EngineArgsFormat.STANDARD,
) -> list[str]:
    """Convert an engine_args dict to a CLI argument list.

    Args:
        engine_args: Engine arguments dict whose keys may be snake_case or kebab-case.
        args_format: Serialization format. ``STANDARD`` (default) produces ``--key value``
            flags. ``FORWARDED`` produces ``--arg key=value`` pairs for engines that
            accept forwarded KV pairs (e.g. BentoML ``serve --arg …``).

    Keys are normalized (underscore → hyphen) for STANDARD format so both
    snake_case and kebab-case inputs produce the same ``--kebab-case`` flags.
    """
    cli_args: list[str] = []
    for key, value in engine_args.items():
        if args_format == EngineArgsFormat.FORWARDED:
            cli_args.extend(["--arg", f"{key}={True if value is None else value}"])
            continue
        flag = f"--{key.replace('_', '-')}"
        if value is None:
            cli_args.append(flag)
        elif isinstance(value, bool):
            if value:
                cli_args.append(flag)
        elif isinstance(value, (list, tuple)):
            cli_args.append(flag)
            for item in value:
                cli_args.append(str(item))
        elif isinstance(value, dict):
            cli_args.extend([flag, json.dumps(value)])
        else:
            cli_args.extend([flag, str(value)])
    return cli_args
