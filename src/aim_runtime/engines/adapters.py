# Copyright © Advanced Micro Devices, Inc., or its affiliates.
#
# SPDX-License-Identifier: MIT

"""Adapter directory enumeration for the LoRA contract (ADR-0004).

Pure helpers (no engine deps) so they are easy to unit test. The mount layout
at ``AIM_ADAPTER_SOURCE`` is::

    ${AIM_ADAPTER_SOURCE}/
        <adapter-name>/
            adapter_config.json     # HF PEFT config (contains "r" = rank)
            adapter_model.safetensors   # or adapter_model.bin

Layout rules enforced here: subdirectory name = adapter name (unique), flat
file layout, adapter rank must not exceed the configured max.
"""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass
from pathlib import Path

from pydantic import BaseModel, ConfigDict, ValidationError

logger = logging.getLogger(__name__)

ADAPTER_CONFIG_FILENAME = "adapter_config.json"
# Routable adapter name (= directory basename). Conservative: it becomes both a
# `name=path` token in vLLM's --lora-modules and a `lora_name` / OpenAI model id
# over the runtime API, so it must not contain '=', whitespace, path separators,
# or leading dots. Letters, digits, and -._ only; must start alphanumeric.
ADAPTER_NAME_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*$")
# A usable adapter directory must contain the config plus one of these weight
# files; an adapter with config but no weights is guaranteed unusable (static
# startup failure or repeated dynamic-watcher load failures), so it is skipped.
ADAPTER_WEIGHT_FILENAMES = ("adapter_model.safetensors", "adapter_model.bin")


class AdapterConfig(BaseModel):
    """The subset of an HF PEFT ``adapter_config.json`` that AIM validates.

    Extra keys are ignored so the full PEFT config round-trips without us
    tracking every field. ``r`` (LoRA rank) is optional because some configs
    omit it; when present it must be a positive int.
    """

    model_config = ConfigDict(extra="ignore")

    r: int | None = None

    @classmethod
    def from_path(cls, config_path: Path) -> "AdapterConfig | None":
        """Parse + validate an ``adapter_config.json``, or None if unusable."""
        try:
            with open(config_path, "r", encoding="utf-8") as f:
                data = json.load(f)
        except (OSError, json.JSONDecodeError) as exc:
            logger.warning(f"Could not read adapter config {config_path}: {exc}")
            return None
        try:
            return cls.model_validate(data)
        except ValidationError as exc:
            logger.warning(f"Invalid adapter config {config_path}: {exc}")
            return None


@dataclass(frozen=True)
class Adapter:
    """A discovered adapter: its routable name and on-disk directory."""

    name: str
    path: str


def _has_weights(adapter_dir: Path) -> bool:
    """True when the adapter directory contains a recognized weights file."""
    return any((adapter_dir / fname).is_file() for fname in ADAPTER_WEIGHT_FILENAMES)


def enumerate_adapters(source: str, max_rank: int) -> list[Adapter]:
    """Enumerate adapters under ``source`` that satisfy the layout + rank rules.

    Globs ``source/*/adapter_config.json``. Skips (with a warning) any adapter
    that has no weights file, an unreadable/invalid config, or a rank exceeding
    ``max_rank``. Raises on duplicate adapter names, since duplicates would
    produce ambiguous routing in ``/v1/models``.

    Returns adapters sorted by name for deterministic CLI output.
    """
    source_dir = Path(source)
    if not source_dir.is_dir():
        logger.info(f"Adapter source {source} does not exist or is not a directory; no adapters loaded.")
        return []

    adapters: dict[str, Adapter] = {}
    for config_path in sorted(source_dir.glob(f"*/{ADAPTER_CONFIG_FILENAME}")):
        adapter_dir = config_path.parent
        name = adapter_dir.name

        if not ADAPTER_NAME_RE.match(name):
            logger.warning(
                f"Adapter directory name '{name}' is not a valid adapter name "
                f"(allowed: letters, digits, '-', '_', '.'; must start alphanumeric); skipping. "
                f"Such names break --lora-modules tokens and OpenAI model ids."
            )
            continue

        if name in adapters:
            raise ValueError(f"Duplicate adapter name '{name}' under {source}; adapter names must be unique.")

        if not _has_weights(adapter_dir):
            logger.warning(
                f"Adapter '{name}' has {ADAPTER_CONFIG_FILENAME} but no weights file "
                f"({' or '.join(ADAPTER_WEIGHT_FILENAMES)}); skipping."
            )
            continue

        config = AdapterConfig.from_path(config_path)
        if config is None:
            continue  # already logged

        if config.r is not None and config.r > max_rank:
            logger.warning(
                f"Adapter '{name}' has rank {config.r} > max rank {max_rank}; skipping. "
                f"Raise AIM_ADAPTER_MAX_RANK to load it."
            )
            continue

        adapters[name] = Adapter(name=name, path=str(adapter_dir))

    return list(adapters.values())
