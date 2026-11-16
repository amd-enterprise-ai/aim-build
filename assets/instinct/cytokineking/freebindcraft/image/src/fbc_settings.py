# Copyright © Advanced Micro Devices, Inc., or its affiliates.
#
# SPDX-License-Identifier: MIT

"""Run settings for the FreeBindCraft job.

Settings come from ``AIM_FBC_*`` environment variables and/or an optional
``settings.json`` in ``AIM_INPUT_PATH``. The environment wins over the file,
the file wins over the defaults. A setting this job does not know is an error:
an unknown key in ``settings.json`` and an unknown ``AIM_*`` environment
variable are both rejected, so a typo never turns into a silently ignored knob.

Pure standard library; no import of the heavy upstream stack.
"""

from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Mapping, Optional

ENV_PREFIX = "AIM_FBC_"

# Settings of the job itself: settings.json key -> environment variable.
SETTING_KEYS: dict[str, str] = {
    "target_chains": ENV_PREFIX + "TARGET_CHAINS",
    "hotspots": ENV_PREFIX + "HOTSPOTS",
    "binder_name": ENV_PREFIX + "BINDER_NAME",
    "min_length": ENV_PREFIX + "MIN_LENGTH",
    "max_length": ENV_PREFIX + "MAX_LENGTH",
    "num_designs": ENV_PREFIX + "NUM_DESIGNS",
    "max_trajectories": ENV_PREFIX + "MAX_TRAJECTORIES",
    "time_limit": ENV_PREFIX + "TIME_LIMIT",
    "max_restarts": ENV_PREFIX + "MAX_RESTARTS",
    "preflight_only": ENV_PREFIX + "PREFLIGHT_ONLY",
}

# Platform setting that carries a profile's engine arguments as a JSON object. This job has no engine arguments (its
# profiles declare engine_args: {}), so it is accepted, reported with a warning and not applied.
ENGINE_ARGS_VAR = "AIM_ENGINE_ARGS"

# Platform-level AIM_* variables the entrypoint reads.
PLATFORM_READ = frozenset(
    {
        "AIM_INPUT_PATH",
        "AIM_OUTPUT_PATH",
        "AIM_CACHE_PATH",
        "AIM_ACCELERATOR_COUNT",
        "AIM_ACCELERATOR_MODEL",
    }
)

# AIM_* variables an AIM platform may inject into any container (profile
# identity and generic runtime knobs). They are accepted and not interpreted,
# so a deployment that sets them is not rejected as having a typo.
PLATFORM_PASSTHROUGH = frozenset(
    {
        "AIM_ID",
        "AIM_MODEL_ID",
        "AIM_PROFILE_ID",
        "AIM_ENGINE",
        "AIM_PRECISION",
        "AIM_METRIC",
        "AIM_ACCELERATOR_TYPE",
        "AIM_ACCELERATOR_FAMILY",
        ENGINE_ARGS_VAR,
        "AIM_LOG_LEVEL",
        "AIM_LOG_LEVEL_ROOT",
    }
)

# Deprecated spellings; named in the error so the operator knows what to use.
DEPRECATED = {
    "AIM_GPU_COUNT": "AIM_ACCELERATOR_COUNT",
    "AIM_GPU_MODEL": "AIM_ACCELERATOR_MODEL",
}

KNOWN_AIM_ENV = frozenset(PLATFORM_READ | PLATFORM_PASSTHROUGH | set(SETTING_KEYS.values()))

MIN_BINDER_LENGTH = 31  # upstream miniprotein design requires >= 31 residues
MAX_BINDER_LENGTH = 300

_CHAINS_RE = re.compile(r"^[A-Za-z0-9](,[A-Za-z0-9])*$")
_HOTSPOT_ITEM = r"([A-Za-z]|[A-Za-z]?[0-9]+(-[0-9]+)?)"  # a chain letter alone targets the whole chain
_HOTSPOTS_RE = re.compile(rf"^({_HOTSPOT_ITEM})(,{_HOTSPOT_ITEM})*$")
_NAME_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,63}$")
_DURATION_RE = re.compile(r"^\s*(\d+(?:\.\d+)?)\s*([smhd]?)\s*$", re.IGNORECASE)
_UNITS = {"": 1, "s": 1, "m": 60, "h": 3600, "d": 86400}


class SettingsError(ValueError):
    """Invalid or unknown setting (exit status 2)."""


@dataclass(frozen=True)
class Settings:
    target_chains: str = "A"
    hotspots: str = ""
    binder_name: str = "binder"
    min_length: int = 80
    max_length: int = 150
    num_designs: int = 200
    max_trajectories: Optional[int] = None  # None: upstream default (no cap)
    time_limit_s: Optional[float] = None  # None: no wall-clock cap
    max_restarts: int = 20  # restarts of a crashed upstream process; 0 disables
    preflight_only: bool = False  # check settings, target and GPU, write log and summary, exit 0; design nothing

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


def parse_duration(value: Any) -> Optional[float]:
    """Seconds from ``90``, ``"90s"``, ``"30m"``, ``"2h"``, ``"1.5h"``; ``0``/``none``/empty: no limit."""
    if value is None:
        return None
    if isinstance(value, bool):
        raise SettingsError(f"time_limit must be a duration, got {value!r}")
    if isinstance(value, (int, float)):
        seconds = float(value)
    else:
        text = str(value).strip().lower()
        if text in ("", "0", "none", "off", "unlimited"):
            return None
        match = _DURATION_RE.match(text)
        if not match:
            raise SettingsError(f"time_limit {value!r} is not a duration (examples: 3600, 90m, 2h)")
        seconds = float(match.group(1)) * _UNITS[match.group(2).lower()]
    if seconds < 0:
        raise SettingsError("time_limit must not be negative")
    return None if seconds == 0 else seconds


_TRUE = frozenset({"1", "true", "yes", "on"})
_FALSE = frozenset({"0", "false", "no", "off", ""})


def _as_bool(name: str, value: Any) -> bool:
    if isinstance(value, bool):
        return value
    text = str(value).strip().lower()
    if text in _TRUE:
        return True
    if text in _FALSE:
        return False
    raise SettingsError(f"{name} must be 1 or 0 (true or false), got {value!r}")


def _as_int(name: str, value: Any, minimum: int, maximum: Optional[int] = None) -> int:
    if isinstance(value, bool):
        raise SettingsError(f"{name} must be an integer, got {value!r}")
    try:
        number = int(str(value).strip()) if not isinstance(value, int) else value
    except ValueError:
        raise SettingsError(f"{name} must be an integer, got {value!r}") from None
    if number < minimum or (maximum is not None and number > maximum):
        bound = f">= {minimum}" if maximum is None else f"between {minimum} and {maximum}"
        raise SettingsError(f"{name} must be {bound}, got {number}")
    return number


def check_environment(env: Mapping[str, str]) -> None:
    """Reject unknown or deprecated ``AIM_*`` variables."""
    problems = []
    for name in sorted(env):
        if not name.startswith("AIM_") or name in KNOWN_AIM_ENV:
            continue
        if name in DEPRECATED:
            problems.append(f"{name} is deprecated; use {DEPRECATED[name]}")
        else:
            problems.append(f"unknown setting {name}")
    if problems:
        known = ", ".join(sorted(SETTING_KEYS.values()))
        raise SettingsError("; ".join(problems) + f". Job settings: {known}")


def load_settings_file(input_dir: Path) -> dict[str, Any]:
    path = input_dir / "settings.json"
    if not path.exists():
        return {}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise SettingsError(f"cannot read {path}: {exc}") from None
    if not isinstance(data, dict):
        raise SettingsError(f"{path} must contain a JSON object")
    unknown = sorted(set(data) - set(SETTING_KEYS))
    if unknown:
        raise SettingsError(
            f"unknown key(s) in settings.json: {', '.join(unknown)}. Known: {', '.join(sorted(SETTING_KEYS))}"
        )
    return data


def resolve_settings(env: Mapping[str, str], input_dir: Path) -> Settings:
    """Merge defaults < settings.json < environment and validate the result."""
    check_environment(env)
    merged: dict[str, Any] = dict(load_settings_file(input_dir))
    for key, var in SETTING_KEYS.items():
        if env.get(var, "") != "":
            merged[key] = env[var]

    defaults = Settings()
    chains = str(merged.get("target_chains", defaults.target_chains)).replace(" ", "")
    if not _CHAINS_RE.match(chains):
        raise SettingsError(f"target_chains {chains!r} must look like 'A' or 'A,B'")
    hotspots = str(merged.get("hotspots", defaults.hotspots)).replace(" ", "")
    if hotspots and not _HOTSPOTS_RE.match(hotspots):
        raise SettingsError(
            f"hotspots {hotspots!r} must look like '56', 'A56', 'A1-10,B20-25' or 'A' (residues, ranges, or a chain letter for a whole chain)"
        )
    name = str(merged.get("binder_name", defaults.binder_name))
    if not _NAME_RE.match(name):
        raise SettingsError(f"binder_name {name!r} must match {_NAME_RE.pattern}")
    min_len = _as_int("min_length", merged.get("min_length", defaults.min_length), MIN_BINDER_LENGTH, MAX_BINDER_LENGTH)
    max_len = _as_int("max_length", merged.get("max_length", defaults.max_length), MIN_BINDER_LENGTH, MAX_BINDER_LENGTH)
    if min_len > max_len:
        raise SettingsError(f"min_length ({min_len}) must not exceed max_length ({max_len})")
    num_designs = _as_int("num_designs", merged.get("num_designs", defaults.num_designs), 1)
    max_traj_raw = merged.get("max_trajectories")
    max_traj = None if max_traj_raw in (None, "", 0, "0") else _as_int("max_trajectories", max_traj_raw, 1)
    return Settings(
        target_chains=chains,
        hotspots=hotspots,
        binder_name=name,
        min_length=min_len,
        max_length=max_len,
        num_designs=num_designs,
        max_trajectories=max_traj,
        time_limit_s=parse_duration(merged.get("time_limit")),
        max_restarts=_as_int("max_restarts", merged.get("max_restarts", defaults.max_restarts), 0),
        preflight_only=_as_bool("preflight_only", merged.get("preflight_only", defaults.preflight_only)),
    )
