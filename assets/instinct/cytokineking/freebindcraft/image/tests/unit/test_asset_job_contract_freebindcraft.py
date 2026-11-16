# Copyright © Advanced Micro Devices, Inc., or its affiliates.
#
# SPDX-License-Identifier: MIT

"""The shipped profiles declare themselves jobs where the schema reads it (metadata.job_type), and its
profiles carry no keys that a job image would silently ignore (env_vars is required by the schema but empty)."""

from __future__ import annotations

import re
from pathlib import Path

import pytest
import yaml

ASSET = Path(__file__).resolve().parents[3]
PROFILES = sorted((ASSET / "profiles").glob("job-*.yaml"))


def test_profiles_declare_long_running_job() -> None:
    """The job_type lives in the profile YAML (metadata.job_type), not in metadata.yaml."""
    assert PROFILES
    meta = yaml.safe_load((ASSET / "metadata.yaml").read_text(encoding="utf-8"))
    assert "job_type" not in meta["com"]["amd"]["aim"]
    for path in PROFILES:
        data = yaml.safe_load(path.read_text(encoding="utf-8"))
        assert data["metadata"]["job_type"] == "long_running", path.name


def test_profiles_have_no_dropped_or_unapplied_keys() -> None:
    assert PROFILES
    for path in PROFILES:
        data = yaml.safe_load(path.read_text(encoding="utf-8"))
        assert "job_type" not in data, path.name  # a top-level key is dropped; it belongs under metadata
        assert not data["env_vars"], path.name  # required key, must stay empty
        allowed = {"aim_id", "model_id", "metadata", "engine_args", "env_vars"}
        assert set(data) <= allowed, (path.name, sorted(set(data) - allowed))


def test_readme_does_not_claim_profiles_set_rocr() -> None:
    text = (ASSET / "image" / "README.md").read_text(encoding="utf-8")
    assert "profiles set" not in text


def test_profiles_declare_throughput_metric_and_matching_names() -> None:
    for path in PROFILES:
        data = yaml.safe_load(path.read_text(encoding="utf-8"))
        assert data["metadata"]["metric"] == "throughput", path.name
        assert path.stem.endswith("-tp1-throughput"), path.name


def test_docs_state_headline_metric_definition() -> None:
    for text in (
        (ASSET / "image" / "README.md").read_text(encoding="utf-8"),
        (ASSET / "templates" / "API.md").read_text(encoding="utf-8"),
    ):
        assert "accepted designs per GPU-hour" in text
        assert "AIM_FBC_TIME_LIMIT" in text
        assert "time to first trajectory" in text
        assert "startup cost" in text
        assert "mean" in text


@pytest.mark.parametrize("doc", ["templates/API.md", "image/README.md"])
def test_docker_run_examples_mount_the_af2_parameters_read_only(doc: str) -> None:
    text = (Path(__file__).resolve().parents[3] / doc).read_text(encoding="utf-8")
    blocks = [b for b in re.findall(r"```bash\n(.*?)```", text, re.S) if b.lstrip().startswith("docker run")]
    assert blocks, f"{doc}: no docker run example found"
    for block in blocks:
        assert "-e AIM_CACHE_PATH=/af2" in block and "-v /host/af2:/af2:ro" in block, f"{doc}: {block}"
