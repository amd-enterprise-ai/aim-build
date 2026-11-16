# Copyright © Advanced Micro Devices, Inc., or its affiliates.
#
# SPDX-License-Identifier: MIT

"""The image sets AIM_METRIC=throughput so that automatic selection finds the shipped throughput profiles.

aim-runtime defaults AIM_METRIC to latency and rejects a profile whose metric differs (METRIC_MISMATCH). The tests run
the real ``ProfileSelector`` over the asset's real profiles directory, CPU-only, with the accelerator given by
override instead of detection.
"""

from __future__ import annotations

import re
import shutil
from pathlib import Path

import pytest
import yaml

from aim_runtime.config import AIMConfig
from aim_runtime.profile_selector import ProfileCompatibilityState, ProfileNotFound, ProfileSelector

IMAGE_DIR = Path(__file__).resolve().parents[1]
ASSET = IMAGE_DIR.parent
DOCKERFILE = (IMAGE_DIR / "Dockerfile").read_text(encoding="utf-8")
PROFILES = sorted((ASSET / "profiles").glob("*.yaml"))
DATA = [yaml.safe_load(p.read_text(encoding="utf-8")) for p in PROFILES]
AIM_ID = DATA[0]["aim_id"]
PROFILE_TYPES = {d["metadata"]["type"] for d in DATA}


def _dockerfile_env(name: str) -> str:
    m = re.search(rf"^ENV {name}=(\S+)\s*$", DOCKERFILE, re.M)
    assert m, f"Dockerfile has no ENV {name}=..."
    return m.group(1)


def _selector(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, accelerator: str, metric: str | None) -> ProfileSelector:
    """A selector over the asset's profiles, with the environment the image sets (engine, optional metric)."""
    base = tmp_path / "profiles"
    shutil.copytree(ASSET / "profiles", base / AIM_ID)
    for name in ("AIM_PROFILE_ID", "AIM_METRIC", "AIM_PRECISION", "AIM_ALLOW_UNOPTIMIZED", "AIM_MODEL_ID"):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("AIM_ID", AIM_ID)
    monkeypatch.setenv("AIM_ENGINE", _dockerfile_env("AIM_ENGINE"))
    monkeypatch.setenv("AIM_ACCELERATOR_MODEL", accelerator)
    monkeypatch.setenv("AIM_ACCELERATOR_COUNT", "1")
    if metric is not None:
        monkeypatch.setenv("AIM_METRIC", metric)
    if "unoptimized" in PROFILE_TYPES:
        monkeypatch.setenv("AIM_ALLOW_UNOPTIMIZED", "true")
    config = AIMConfig.from_environment()
    config.profile_base_path = str(base)
    return ProfileSelector(config)


def test_dockerfile_sets_throughput_metric() -> None:
    assert _dockerfile_env("AIM_METRIC") == "throughput"


def test_the_shipped_profiles_are_throughput_profiles() -> None:
    assert DATA and {d["metadata"]["metric"] for d in DATA} == {"throughput"}


@pytest.mark.parametrize("accelerator", ["MI300X", "MI355X"])
def test_default_latency_metric_does_not_select_the_profiles(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, accelerator: str
) -> None:
    selector = _selector(tmp_path, monkeypatch, accelerator, metric=None)
    results = selector.assess_all_profiles()
    assert results[ProfileCompatibilityState.COMPATIBLE] == []
    assert len(results[ProfileCompatibilityState.METRIC_MISMATCH]) == len(DATA)
    with pytest.raises(ProfileNotFound):
        selector.find_profile()


@pytest.mark.parametrize("accelerator", ["MI300X", "MI355X"])
def test_image_metric_selects_the_profile_automatically(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, accelerator: str
) -> None:
    selector = _selector(tmp_path, monkeypatch, accelerator, metric=_dockerfile_env("AIM_METRIC"))
    profile = selector.find_profile()
    assert profile.metadata.accelerator_model.value == accelerator
    assert profile.metadata.metric.value == "throughput"
