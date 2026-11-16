# Copyright © Advanced Micro Devices, Inc., or its affiliates.
#
# SPDX-License-Identifier: MIT

"""Settings parsing: defaults, precedence, validation, unknown settings rejected."""

from __future__ import annotations

import json
from pathlib import Path

import fbc_settings as settings
import pytest


def _resolve(tmp_path: Path, env: dict[str, str] | None = None, file: dict | None = None):
    if file is not None:
        (tmp_path / "settings.json").write_text(json.dumps(file))
    return settings.resolve_settings(env or {}, tmp_path)


def test_defaults(tmp_path: Path) -> None:
    s = _resolve(tmp_path)
    assert (s.target_chains, s.hotspots, s.binder_name) == ("A", "", "binder")
    assert (s.min_length, s.max_length, s.num_designs) == (80, 150, 200)
    assert s.max_trajectories is None and s.time_limit_s is None


def test_env_overrides_file_overrides_default(tmp_path: Path) -> None:
    s = _resolve(
        tmp_path,
        env={"AIM_FBC_MIN_LENGTH": "31", "AIM_FBC_TIME_LIMIT": "2h"},
        file={"min_length": 40, "max_length": 45, "num_designs": 3, "max_trajectories": 2},
    )
    assert (s.min_length, s.max_length, s.num_designs, s.max_trajectories) == (31, 45, 3, 2)
    assert s.time_limit_s == 7200


def test_unknown_aim_env_is_rejected(tmp_path: Path) -> None:
    with pytest.raises(settings.SettingsError, match="AIM_FBC_TYPO"):
        _resolve(tmp_path, env={"AIM_FBC_TYPO": "1"})


def test_deprecated_gpu_names_are_rejected_with_the_replacement(tmp_path: Path) -> None:
    with pytest.raises(settings.SettingsError, match="AIM_ACCELERATOR_COUNT"):
        _resolve(tmp_path, env={"AIM_GPU_COUNT": "1"})


def test_known_platform_variables_are_accepted(tmp_path: Path) -> None:
    env = {
        "AIM_INPUT_PATH": "/i",
        "AIM_OUTPUT_PATH": "/o",
        "AIM_CACHE_PATH": "/c",
        "AIM_ACCELERATOR_COUNT": "1",
        "AIM_ID": "x",
        "PATH": "/bin",
    }
    assert _resolve(tmp_path, env=env).num_designs == 200


def test_unknown_settings_json_key_is_rejected(tmp_path: Path) -> None:
    with pytest.raises(settings.SettingsError, match="nope"):
        _resolve(tmp_path, file={"nope": 1})


def test_settings_json_must_be_an_object(tmp_path: Path) -> None:
    (tmp_path / "settings.json").write_text("[1]")
    with pytest.raises(settings.SettingsError, match="JSON object"):
        settings.resolve_settings({}, tmp_path)


def test_invalid_json_is_a_settings_error(tmp_path: Path) -> None:
    (tmp_path / "settings.json").write_text("{not json")
    with pytest.raises(settings.SettingsError, match="cannot read"):
        settings.resolve_settings({}, tmp_path)


@pytest.mark.parametrize(
    "env",
    [
        {"AIM_FBC_MIN_LENGTH": "30"},  # upstream miniprotein minimum is 31
        {"AIM_FBC_MIN_LENGTH": "50", "AIM_FBC_MAX_LENGTH": "40"},
        {"AIM_FBC_MIN_LENGTH": "abc"},
        {"AIM_FBC_NUM_DESIGNS": "0"},
        {"AIM_FBC_MAX_TRAJECTORIES": "-1"},
        {"AIM_FBC_TARGET_CHAINS": "AB"},
        {"AIM_FBC_HOTSPOTS": "56;rm"},
        {"AIM_FBC_BINDER_NAME": "../x"},
        {"AIM_FBC_TIME_LIMIT": "soon"},
    ],
)
def test_invalid_values_are_rejected(tmp_path: Path, env: dict[str, str]) -> None:
    with pytest.raises(settings.SettingsError):
        _resolve(tmp_path, env=env)


@pytest.mark.parametrize("hotspots", ["56", "A56", "A1,B20-25", "A1-5", "A", "A,B", "A,B20-25"])
def test_valid_hotspots(tmp_path: Path, hotspots: str) -> None:
    assert _resolve(tmp_path, env={"AIM_FBC_HOTSPOTS": hotspots}).hotspots == hotspots


@pytest.mark.parametrize(
    ("text", "seconds"),
    [
        ("90", 90),
        (90, 90),
        ("30m", 1800),
        ("2h", 7200),
        ("1.5h", 5400),
        ("1d", 86400),
        ("0", None),
        ("none", None),
        ("", None),
        (None, None),
    ],
)
def test_parse_duration(text, seconds) -> None:
    assert settings.parse_duration(text) == seconds


def test_zero_max_trajectories_means_no_cap(tmp_path: Path) -> None:
    assert _resolve(tmp_path, env={"AIM_FBC_MAX_TRAJECTORIES": "0"}).max_trajectories is None


def test_binder_length_cap_is_300(tmp_path: Path) -> None:
    s = _resolve(tmp_path, file={"min_length": 31, "max_length": 300})
    assert s.max_length == 300
    with pytest.raises(settings.SettingsError, match="between 31 and 300"):
        _resolve(tmp_path, file={"max_length": 301})


@pytest.mark.parametrize("hotspots", ["AB", "A-5", "A,,B", "56,", "A1-"])
def test_invalid_hotspots_rejected(tmp_path: Path, hotspots: str) -> None:
    with pytest.raises(settings.SettingsError, match="hotspots"):
        _resolve(tmp_path, env={"AIM_FBC_HOTSPOTS": hotspots})


def test_preflight_only_is_off_by_default_and_parsed_as_a_flag(tmp_path: Path) -> None:
    assert _resolve(tmp_path).preflight_only is False
    assert _resolve(tmp_path, env={"AIM_FBC_PREFLIGHT_ONLY": "1"}).preflight_only is True
    assert _resolve(tmp_path, env={"AIM_FBC_PREFLIGHT_ONLY": "true"}).preflight_only is True
    assert _resolve(tmp_path, env={"AIM_FBC_PREFLIGHT_ONLY": "0"}).preflight_only is False
    assert _resolve(tmp_path, file={"preflight_only": True}).preflight_only is True
    assert (
        _resolve(tmp_path, env={"AIM_FBC_PREFLIGHT_ONLY": "0"}, file={"preflight_only": True}).preflight_only is False
    )
    with pytest.raises(settings.SettingsError, match="preflight_only"):
        _resolve(tmp_path, env={"AIM_FBC_PREFLIGHT_ONLY": "maybe"})


def test_engine_args_variable_is_accepted_not_rejected(tmp_path: Path) -> None:
    # CI and operators may pass the profile's engine_args as AIM_ENGINE_ARGS; this job has none, so it must not fail.
    assert _resolve(tmp_path, env={"AIM_ENGINE_ARGS": '{"nsteps": 10}'}).num_designs == 200
