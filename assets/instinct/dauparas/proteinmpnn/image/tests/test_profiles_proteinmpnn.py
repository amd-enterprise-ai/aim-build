# Copyright © Advanced Micro Devices, Inc., or its affiliates.
#
# SPDX-License-Identifier: MIT

"""Profile guards: declared env vars are ones the service applies, job_type is explicit."""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

ASSET_DIR = Path(__file__).resolve().parents[2]
PROFILES = sorted((ASSET_DIR / "profiles").glob("*.yaml"))


def test_profiles_found() -> None:
    assert PROFILES


@pytest.mark.parametrize("path", PROFILES, ids=lambda p: p.name)
def test_job_type_is_service(path: Path) -> None:
    assert yaml.safe_load(path.read_text())["metadata"]["job_type"] == "service"


@pytest.mark.parametrize("path", PROFILES, ids=lambda p: p.name)
def test_no_nccl_variables(path: Path) -> None:
    # Single-GPU service: nothing in it reads NCCL settings.
    env = yaml.safe_load(path.read_text()).get("env_vars", {})
    assert not [k for k in env if k.startswith("NCCL_")]


def test_api_doc_lists_only_applied_variables() -> None:
    assert "NCCL_MIN_NCHANNELS" not in (ASSET_DIR / "templates/API.md").read_text()


def test_naming_contract_matches_asset_path() -> None:
    """canonicalName, variants, aim_id and model_id equal the lowercase asset path."""
    asset_path = "/".join(ASSET_DIR.parts[-2:])
    assert asset_path == asset_path.lower()
    meta = yaml.safe_load((ASSET_DIR / "metadata.yaml").read_text())["com"]["amd"]["aim"]["model"]
    assert meta["canonicalName"] == asset_path
    assert meta["variants"] == [asset_path]
    for path in PROFILES:
        prof = yaml.safe_load(path.read_text())
        assert prof["aim_id"] == asset_path
        assert prof["model_id"] == asset_path
