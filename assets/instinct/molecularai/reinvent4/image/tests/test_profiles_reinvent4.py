# Copyright © Advanced Micro Devices, Inc., or its affiliates.
#
# SPDX-License-Identifier: MIT

"""Profile keys must be ones the runtime reads; the only env var a profile sets is the sampler batch size."""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

ASSET = Path(__file__).resolve().parents[2]
PROFILES = sorted((ASSET / "profiles").glob("*.yaml"))
ALLOWED_TOP = {"aim_id", "model_id", "metadata", "engine_args", "env_vars"}


@pytest.mark.parametrize("path", PROFILES, ids=lambda p: p.name)
def test_profile_keys(path: Path) -> None:
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    assert set(data) <= ALLOWED_TOP
    assert data["env_vars"] == {"REINVENT4_SAMPLE_BATCH_SIZE": "1024"}
    assert data["metadata"]["job_type"] == "service"


def test_api_doc_does_not_mention_removed_env_var() -> None:
    assert "PYTORCH_ROCM_ARCH" not in (ASSET / "templates" / "API.md").read_text(encoding="utf-8")
