# Copyright © Advanced Micro Devices, Inc., or its affiliates.
#
# SPDX-License-Identifier: MIT

"""The GPU smoke test validates FBC_AF2_CACHE_PATH before it starts Docker (checked here without a GPU)."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

SMOKE = Path(__file__).resolve().parents[1] / "smoke" / "test_job_gpu_freebindcraft.py"


@pytest.fixture(scope="module")
def smoke():
    spec = importlib.util.spec_from_file_location("fbc_gpu_smoke", SMOKE)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_valid_directory_is_accepted(smoke, tmp_path: Path) -> None:
    (tmp_path / "params").mkdir()
    (tmp_path / "params/params_model_1_multimer_v3.npz").write_text("")
    assert smoke.validated_af2_cache(str(tmp_path)) == tmp_path


@pytest.mark.parametrize("raw", [None, "", "   "])
def test_unset_is_rejected(smoke, raw) -> None:
    with pytest.raises(ValueError, match="not set"):
        smoke.validated_af2_cache(raw)


def test_relative_missing_and_empty_directories_are_rejected(smoke, tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="absolute"):
        smoke.validated_af2_cache("relative/af2")
    with pytest.raises(ValueError, match="not a directory"):
        smoke.validated_af2_cache(str(tmp_path / "nope"))
    with pytest.raises(ValueError, match="holds no"):
        smoke.validated_af2_cache(str(tmp_path))


def test_docker_run_mounts_the_cache_read_only_with_aim_cache_path(smoke, tmp_path: Path, monkeypatch) -> None:
    seen = {}

    def fake_run(cmd, **_kw):
        seen["cmd"] = cmd

    monkeypatch.setattr(smoke.subprocess, "run", fake_run)
    smoke._docker_run("img", tmp_path / "af2", tmp_path / "in", tmp_path / "out")
    cmd = seen["cmd"]
    assert f"{tmp_path / 'af2'}:/af2:ro" in cmd
    assert "AIM_CACHE_PATH=/af2" in cmd
