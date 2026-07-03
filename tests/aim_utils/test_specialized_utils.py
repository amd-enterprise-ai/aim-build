# Copyright © Advanced Micro Devices, Inc., or its affiliates.
#
# SPDX-License-Identifier: MIT

import pytest

from aim_utils.specialized_utils import enumerate_specialized_base_targets, resolve_base_assets_dir

ACC = "cpu"
ORG = "example"
MODEL = "echo-model"


def _write(path, content=""):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content)


@pytest.fixture
def assets_root(tmp_path):
    return tmp_path / "assets"


def _model_dir(assets_root):
    return assets_root / ACC / ORG / MODEL


# --------------------------------------------------------------------------- #
# enumerate_specialized_base_targets (CI discovery)
# --------------------------------------------------------------------------- #


def test_enumerate_engine_level_target(assets_root):
    engine_image = assets_root / ACC / "engines" / "monai" / "image"
    _write(engine_image / "Dockerfile", "FROM python:3.12-slim\n")

    targets = enumerate_specialized_base_targets(ACC, assets_root=str(assets_root))

    assert len(targets) == 1
    target = targets[0]
    assert target.scope == "engine"
    assert target.target_id == "monai"
    assert target.context_path == engine_image.as_posix()
    assert target.dockerfile == (engine_image / "Dockerfile").as_posix()
    # Layer 1 (specialized) and Layer 2 (consumable base) names.
    assert target.layer1_repository == "aim-cpu-specialized-monai"
    assert target.base_repository == "aim-cpu-monai-base"


def test_enumerate_model_level_target_uses_model_key(assets_root):
    image_dir = _model_dir(assets_root) / "image"
    _write(image_dir / "Dockerfile", "FROM python:3.12-slim\n")

    targets = enumerate_specialized_base_targets(ACC, assets_root=str(assets_root))

    assert len(targets) == 1
    target = targets[0]
    assert target.scope == "model"
    # target_id == sanitized <org>-<model>; Layer 3 is target-qualified and repeats the key.
    assert target.target_id == "example-echo-model"
    assert target.base_repository == "aim-cpu-example-echo-model-base"
    assert target.layer1_repository == "aim-cpu-specialized-example-echo-model"


def test_enumerate_returns_both_levels_sorted(assets_root):
    _write(assets_root / ACC / "engines" / "monai" / "image" / "Dockerfile", "FROM x\n")
    _write(_model_dir(assets_root) / "image" / "Dockerfile", "FROM x\n")

    targets = enumerate_specialized_base_targets(ACC, assets_root=str(assets_root))

    target_ids = [t.target_id for t in targets]
    assert target_ids == sorted(target_ids)
    assert set(target_ids) == {"monai", "example-echo-model"}


def test_enumerate_skips_base_and_engines_as_orgs(assets_root):
    # A base/ dir and an engines/ dir without image/ must not be enumerated as models.
    _write(assets_root / ACC / "base" / "config.yaml", "base_image: {}\n")
    _write(assets_root / ACC / "engines" / "vllm" / "README.md", "no image dir\n")

    assert enumerate_specialized_base_targets(ACC, assets_root=str(assets_root)) == []


def test_enumerate_no_assets_returns_empty(assets_root):
    assert enumerate_specialized_base_targets(ACC, assets_root=str(assets_root)) == []


# --------------------------------------------------------------------------- #
# resolve_base_assets_dir (engine config + general-profile source)
# --------------------------------------------------------------------------- #


def _write_named_base(assets_root, engine_dir_name, engine_key):
    """Create a named base dir with config.yaml and config/engines.yaml."""
    base = assets_root / ACC / "base" / engine_dir_name
    _write(base / "config.yaml", "base_image: {}\n")
    _write(base / "config" / "engines.yaml", f"{engine_key}:\n  launch: run\n")
    _write(base / "profiles" / "general" / ".gitkeep")
    return base


def _write_model_with_profile(assets_root, profile_stem):
    """Create a model-dedicated specialized base (image/Dockerfile) with one profile."""
    model = _model_dir(assets_root)
    _write(model / "image" / "Dockerfile", "FROM python:3.12-slim\n")
    _write(model / "profiles" / f"{profile_stem}.yaml", "metadata: {}\n")
    return model


def test_resolve_named_base_target_uses_its_own_dir(assets_root):
    base = _write_named_base(assets_root, "bentoml", "bentoml")

    result = resolve_base_assets_dir(ACC, "bentoml", assets_root=str(assets_root))

    assert result == base.as_posix()


def test_resolve_legacy_vllm_uses_base_root(assets_root):
    _write(assets_root / ACC / "base" / "config.yaml", "base_image: {}\n")

    result = resolve_base_assets_dir(ACC, "legacy_vllm", assets_root=str(assets_root))

    assert result == (assets_root / ACC / "base").as_posix()


def test_resolve_model_base_infers_engine_from_profiles(assets_root):
    # Model-dedicated base whose profiles use the bentoml engine should ship the
    # bentoml named base dir, not the generic accelerator base root.
    bentoml_base = _write_named_base(assets_root, "bentoml", "bentoml")
    _write_model_with_profile(assets_root, "bentoml-mi300x-fp16-tp1-latency")

    result = resolve_base_assets_dir(ACC, "example-echo-model", assets_root=str(assets_root))

    assert result == bentoml_base.as_posix()


def test_resolve_model_base_matches_engine_key_not_dir_name(assets_root):
    # Engine key (vllm_omni) differs from the dir name (vllm-omni); matching is on
    # the engines.yaml key so the right dir is still found.
    omni_base = _write_named_base(assets_root, "vllm-omni", "vllm_omni")
    _write_model_with_profile(assets_root, "vllm_omni-mi300x-fp16-tp1-latency")

    result = resolve_base_assets_dir(ACC, "example-echo-model", assets_root=str(assets_root))

    assert result == omni_base.as_posix()


def test_resolve_model_base_falls_back_when_engine_base_missing(assets_root):
    # No named base dir declares the model's engine -> fall back to base root.
    _write(assets_root / ACC / "base" / "config.yaml", "base_image: {}\n")
    _write_model_with_profile(assets_root, "bentoml-mi300x-fp16-tp1-latency")

    result = resolve_base_assets_dir(ACC, "example-echo-model", assets_root=str(assets_root))

    assert result == (assets_root / ACC / "base").as_posix()


def test_resolve_model_base_falls_back_when_profiles_disagree(assets_root):
    # Profiles spanning multiple engines are ambiguous -> safe fallback to base root.
    _write(assets_root / ACC / "base" / "config.yaml", "base_image: {}\n")
    _write_named_base(assets_root, "bentoml", "bentoml")
    model = _model_dir(assets_root)
    _write(model / "image" / "Dockerfile", "FROM python:3.12-slim\n")
    _write(model / "profiles" / "bentoml-mi300x-fp16-tp1-latency.yaml", "metadata: {}\n")
    _write(model / "profiles" / "vllm-mi300x-fp16-tp1-latency.yaml", "metadata: {}\n")

    result = resolve_base_assets_dir(ACC, "example-echo-model", assets_root=str(assets_root))

    assert result == (assets_root / ACC / "base").as_posix()
