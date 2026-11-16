# Copyright © Advanced Micro Devices, Inc., or its affiliates.
#
# SPDX-License-Identifier: MIT

"""Static guards for the SaProt image: bentofile, dependency pins, pooling."""

from __future__ import annotations

import re
from pathlib import Path

import yaml

IMAGE_DIR = Path(__file__).resolve().parents[1]
REPO_ROOT = IMAGE_DIR.parents[4]


def test_bentofile_names_service_class_and_is_copied() -> None:
    cfg = yaml.safe_load((IMAGE_DIR / "bentofile.yaml").read_text(encoding="utf-8"))
    assert cfg["service"] == "service:SaProtEmbeddingService"
    assert "class SaProtEmbeddingService" in (IMAGE_DIR / "src/service.py").read_text(encoding="utf-8")
    dockerfile = (IMAGE_DIR / "Dockerfile").read_text(encoding="utf-8")
    assert "COPY bentofile.yaml /workspace/model/bentofile.yaml" in dockerfile


def _pin(text: str, name: str) -> str:
    match = re.search(rf"^{re.escape(name)}==(\S+)", text, re.MULTILINE | re.IGNORECASE)
    assert match, f"{name} not pinned"
    return match.group(1)


def _version(text: str) -> tuple[int, ...]:
    return tuple(int(part) for part in text.split(".") if part.isdigit())


def test_hub_pin_matches_aim_runtime_layer() -> None:
    reqs = (IMAGE_DIR / "requirements.txt").read_text(encoding="utf-8")
    layer = (REPO_ROOT / "requirements/instinct-requirements.txt").read_text(encoding="utf-8")
    # The layer installs its hub version first; this image must not pin an older one.
    assert _version(_pin(reqs, "huggingface_hub")) >= _version(_pin(layer, "huggingface-hub"))
    # transformers 4.x declares huggingface-hub<1.0, incompatible with the layer pin.
    assert int(_pin(reqs, "transformers").split(".")[0]) >= 5


def test_pooling_excludes_bos_eos() -> None:
    src = (IMAGE_DIR / "src/service.py").read_text(encoding="utf-8")
    assert "hidden[0, 1:-1]" in src
    assert "attention_mask" not in src.split("def inference")[1]


def _service_text() -> str:
    return (IMAGE_DIR / "src/service.py").read_text(encoding="utf-8")


def test_profiles_declare_service_and_every_engine_arg_is_read() -> None:
    """Every ``engine_args`` key a profile sets must be read by the service.

    The service reads ``port`` and ``accelerator_count`` from ``bentoml.use_arguments``;
    aim-runtime supplies both (the count from ``metadata.accelerator_count``), so the
    profiles need no ``engine_args`` for them. A key nothing consumes would be silently
    ignored by BentoML, so it is rejected. ``env_vars`` stay empty.
    """
    fields = set(re.findall(r"^    (\w+): .*$", _service_text().split("class BentoArgs")[1].split("args =")[0], re.M))
    assert {"port", "accelerator_count"} <= fields
    profiles = sorted((IMAGE_DIR.parent / "profiles").glob("*.yaml"))
    assert profiles
    for path in profiles:
        profile = yaml.safe_load(path.read_text(encoding="utf-8"))
        assert profile["metadata"]["job_type"] == "service", path.name
        unread = set(profile["engine_args"]) - fields
        assert not unread, f"{path.name}: engine_args keys the service does not read: {sorted(unread)}"
        assert profile["env_vars"] == {}, path.name
        count = profile["engine_args"].get("accelerator_count")
        if count is not None:
            assert count == profile["metadata"]["accelerator_count"], path.name
