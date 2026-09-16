# Copyright © Advanced Micro Devices, Inc., or its affiliates.
#
# SPDX-License-Identifier: MIT

"""
How a profile's ``engine_args`` are spelled, and when a flag in them is on.

Profiles write ``engine_args`` in the serving engine's own CLI vocabulary so the
runtime can hand them to the engine unchanged. Keys are the engine's flags, and a
boolean flag is written the way a command line takes it: valueless
(``trust-remote-code:``) rather than ``true``, with the engine's paired negative
flag (``no-trust-remote-code:``) as the off switch. YAML has no syntax for a bare
key, so ``key: null`` is how a valueless flag is transcribed.
``aim_runtime.engines.engine_args_to_cli_list`` serializes that spelling back to
argv, and :func:`engine_flag_is_set` is the rule it applies.

Anything that needs to *read* a flag rather than pass it on asks here — the
harness from the profile it already holds, CI from a profile file it checked out —
so the engine's spelling is interpreted in one place and a change in what the
engine accepts lands in one place too.

This module sits in ``aim_common`` because that is the only package both the
in-image runtime and the CI scripts depend on, and it stays free of anything
heavier than PyYAML so the lightweight CI validators can import it.
"""

from __future__ import annotations

import logging
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import yaml

logger = logging.getLogger(__name__)

#: Engine argument a profile sets to let the tokenizer execute code that ships
#: with the model repository, spelled as the engine's CLI takes it.
ENGINE_ARG_TRUST_REMOTE_CODE = "trust-remote-code"

#: Engine argument a profile sets to pick which tokenizer implementation loads a
#: model, spelled as the engine's CLI takes it.
ENGINE_ARG_TOKENIZER_MODE = "tokenizer-mode"

#: Tokenizer modes vLLM's CLI accepts. Anything else reads as absent, so a
#: malformed profile cannot build a command the engine would reject.
VALID_TOKENIZER_MODES = ("auto", "custom", "mistral", "slow")

#: Key under which a profile document holds the engine's CLI arguments.
PROFILE_ENGINE_ARGS_KEY = "engine_args"

#: Absence of a key, distinct from a key present with the value None — which for a
#: valueless engine flag is how the profile spells "on".
_MISSING = object()


def normalize_engine_arg_key(key: Any) -> str:
    """Spell an engine-arg key the way the engine's CLI does.

    Authors write either separator in profiles and in ``AIM_ENGINE_ARGS``, which
    is also why ``EngineArgsModel`` accepts both.
    """
    return str(key).replace("_", "-")


def normalize_engine_args(engine_args: Any) -> dict[str, Any]:
    """Spell every key of *engine_args* the way the engine's CLI does.

    A document can spell one flag twice, for example ``trust-remote-code`` and
    ``trust_remote_code``. The engine's CLI takes one flag, so a reader and the
    serializer must see one entry as well. The last spelling in the document
    wins, which is what a command line does with a repeated flag. A duplicate is
    a mistake in the profile, so it also gets a warning.

    The case of a key is kept, because the engine's CLI is what receives it. Two
    spellings that differ only in case still count as a duplicate.

    Engine args that are not a mapping give an empty mapping.
    """
    if not isinstance(engine_args, Mapping):
        return {}

    normalized: dict[str, Any] = {}
    kept: dict[str, tuple[str, Any]] = {}
    for key, value in engine_args.items():
        canonical = normalize_engine_arg_key(key)
        previous = kept.get(canonical.lower())
        if previous is not None:
            logger.warning(
                f"engine_args spell one flag twice: '{previous[0]}' and '{key}' both mean "
                f"--{canonical}. The last value wins, so the flag is used as "
                f"{key}={value!r} and not as {previous[0]}={previous[1]!r}."
            )
            normalized.pop(normalize_engine_arg_key(previous[0]), None)
        kept[canonical.lower()] = (str(key), value)
        normalized[canonical] = value
    return normalized


def engine_flag_is_set(value: Any) -> bool:
    """Whether a boolean engine arg's value means the flag reaches the engine.

    Valueless (``None``) and ``true`` both mean it does, because the flag takes no
    value on the command line; only an explicit ``false`` withholds it. The
    serializer that builds argv applies this same rule, so what a reader concludes
    and what the engine was launched with cannot drift apart.
    """
    return value is not False


def engine_arg_value(engine_args: Any, arg: str, default: Any = None) -> Any:
    """The value *engine_args* carry for *arg*, or *default* when they do not.

    Keys are compared in the engine's spelling, case-insensitively, so a profile
    that writes the argument either way reads the same.

    Engine args that are not a mapping cost the argument rather than the caller.
    """
    normalized = {key.lower(): value for key, value in normalize_engine_args(engine_args).items()}
    return normalized.get(normalize_engine_arg_key(arg).lower(), default)


def engine_flag_enabled(engine_args: Any, flag: str) -> bool:
    """Whether *flag* is enabled in *engine_args*.

    Keys are compared in the engine's spelling, case-insensitively, so a profile
    that writes the flag either way reads the same. Note that an engine's negative
    flag is a *different* key (``no-trust-remote-code``), so a profile disabling a
    flag that way simply does not carry the flag asked about here.

    Engine args that are not a mapping cost the flag rather than the caller.
    """
    value = engine_arg_value(engine_args, flag, _MISSING)
    if value is _MISSING:
        return False
    return engine_flag_is_set(value)


def tokenizer_mode_from_engine_args(engine_args: Any) -> str | None:
    """The tokenizer mode *engine_args* declare, or None when they declare none.

    Only the modes in :data:`VALID_TOKENIZER_MODES` are returned. A tool that has
    to reproduce the engine's tokenization — a benchmark client loading the same
    tokenizer, say — reads the mode here so it cannot tokenize differently from
    the engine it measures, and a mode the engine would reject reads as absent
    rather than becoming a command that fails to start.
    """
    value = engine_arg_value(engine_args, ENGINE_ARG_TOKENIZER_MODE)
    if not isinstance(value, str):
        return None
    mode = value.strip().lower()
    if mode not in VALID_TOKENIZER_MODES:
        logger.warning(
            f"engine_args declare --{ENGINE_ARG_TOKENIZER_MODE} {value!r}, which is not one of "
            f"{', '.join(VALID_TOKENIZER_MODES)}. Reading it as unset."
        )
        return None
    return mode


def profile_engine_args(profile_path: str | Path | None) -> Mapping[str, Any] | None:
    """Read a profile file's ``engine_args``, or None when they cannot be read.

    The file is the only source of engine args in CI: what the pipeline passes
    between jobs carries a profile's metadata, not the arguments the engine is
    launched with. A missing, unreadable or malformed profile yields None with a
    warning rather than raising, because every caller reads engine args to decide
    whether to add something and can proceed without them.
    """
    if profile_path is None or not str(profile_path).strip():
        return None

    path = Path(profile_path)
    if not path.is_file():
        logger.warning(f"Profile file not found, cannot read engine args: {path}")
        return None

    try:
        document = yaml.safe_load(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, yaml.YAMLError) as error:
        logger.warning(f"Could not read profile at {path}: {error}")
        return None

    if not isinstance(document, Mapping):
        return None
    engine_args = document.get(PROFILE_ENGINE_ARGS_KEY)
    return engine_args if isinstance(engine_args, Mapping) else None


def profile_flag_enabled(profile_path: str | Path | None, flag: str) -> bool:
    """Whether the profile on disk enables *flag* in its ``engine_args``."""
    enabled = engine_flag_enabled(profile_engine_args(profile_path), flag)
    if enabled:
        logger.info(f"Profile at {profile_path} enables --{normalize_engine_arg_key(flag)}")
    return enabled
