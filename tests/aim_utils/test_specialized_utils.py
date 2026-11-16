# Copyright © Advanced Micro Devices, Inc., or its affiliates.
#
# SPDX-License-Identifier: MIT

import pytest
from click.testing import CliRunner

from aim_common.object_model import AcceleratorFamily
from aim_utils.config_utils import cli
from aim_utils.specialized_utils import (
    base_leaf_declares_dockerfile,
    enumerate_specialized_base_targets,
    is_job_aim,
    model_leaf_dockerfile_config,
    model_target_id,
    parse_specialized_target_from_path,
    resolve_asset_test_dir,
    resolve_base_assets_dir,
)

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


def test_adr0007_engine_leaf_is_not_a_specialized_target(assets_root, monkeypatch):
    """An engine/<engine>/image/ Dockerfile builds an engine image, not a model-level specialized base."""
    engine_image = assets_root / ACC / "engine" / "monai-bentoml" / "image"
    _write(engine_image / "Dockerfile", "FROM python:3.12-slim\n")
    monkeypatch.chdir(assets_root.parent)

    assert enumerate_specialized_base_targets(ACC, assets_root=str(assets_root)) == []
    assert parse_specialized_target_from_path(f"assets/{ACC}/engine/monai-bentoml/image/Dockerfile") is None


def test_adr0007_model_directory_is_not_an_org(assets_root, monkeypatch):
    """A stray image/ under a grouped model leaf must not become a bogus specialized base."""
    _write(assets_root / ACC / "model" / ORG / "image" / "Dockerfile", "FROM python:3.12-slim\n")
    monkeypatch.chdir(assets_root.parent)

    assert enumerate_specialized_base_targets(ACC, assets_root=str(assets_root)) == []
    assert parse_specialized_target_from_path(f"assets/{ACC}/model/{ORG}/image/Dockerfile") is None


@pytest.mark.parametrize(
    ("config", "expected"),
    [
        ("base_image: {}\ndockerfile: assets/instinct/base/aifs/Dockerfile\n", True),
        ("base_image: {}\n", False),
        (None, False),
        ("base_image: [unclosed\ndockerfile: x\n", False),
        ("- not\n- a mapping\n", False),
    ],
    ids=["declares", "no-dockerfile", "no-config", "malformed-yaml", "not-a-mapping"],
)
def test_base_leaf_declares_dockerfile(tmp_path, config, expected):
    leaf = tmp_path / "assets" / "instinct" / "base" / "aifs"
    leaf.mkdir(parents=True)
    if config is not None:
        (leaf / "config.yaml").write_text(config)

    assert base_leaf_declares_dockerfile(leaf) is expected


def test_an_unreadable_base_config_is_reported(tmp_path, caplog):
    leaf = tmp_path / "assets" / "instinct" / "base" / "aifs"
    leaf.mkdir(parents=True)
    (leaf / "config.yaml").write_text("base_image: [unclosed\n")

    assert base_leaf_declares_dockerfile(leaf) is False
    assert f"Cannot read {leaf / 'config.yaml'}" in caplog.text


@pytest.mark.parametrize("check", [base_leaf_declares_dockerfile, is_job_aim], ids=lambda f: f.__name__)
def test_an_unexpected_error_is_not_swallowed(tmp_path, monkeypatch, check):
    leaf = tmp_path / "leaf"
    _write(leaf / "config.yaml", "dockerfile: x\n")
    _write(leaf / "profiles" / "job-a.yaml", "metadata:\n  engine: job\n")

    def broken_reader(*_):
        raise RuntimeError("bug in the reader")

    monkeypatch.setattr("aim_utils.specialized_utils.read_yaml", broken_reader)
    with pytest.raises(RuntimeError, match="bug in the reader"):
        check(leaf)


@pytest.mark.parametrize(
    ("path", "expected"),
    [
        (
            "assets/instinct/model/ecmwf/aifs-single-2.0/Dockerfile",
            "assets/instinct/model/ecmwf/aifs-single-2.0/config.yaml",
        ),
        ("assets/instinct/model/ecmwf/aifs-single-2.0/templates/API.md", None),
        ("assets/instinct/meta-llama/Llama-3.1-8B-Instruct/Dockerfile", None),
        ("assets/instinct/base/aifs/Dockerfile", None),
        ("assets/instinct/engine/aifs/Dockerfile", None),
    ],
)
def test_model_leaf_dockerfile_config(path, expected):
    assert model_leaf_dockerfile_config(path) == expected


def test_enumerate_model_level_target_uses_model_key(assets_root):
    image_dir = _model_dir(assets_root) / "image"
    _write(image_dir / "Dockerfile", "FROM python:3.12-slim\n")

    targets = enumerate_specialized_base_targets(ACC, assets_root=str(assets_root))

    assert len(targets) == 1
    target = targets[0]
    assert target.scope == "model"
    # target_id == sanitized <org>-<model>; Layer 3 drops the duplicate (aim-<acc>-model-<org>-<model>).
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


# --------------------------------------------------------------------------- #
# resolve_asset_test_dir (engine tests of the asset under test)
# --------------------------------------------------------------------------- #


def _write_model_image(assets_root, org, model, with_tests=True):
    image = assets_root / ACC / org / model / "image"
    _write(image / "Dockerfile", "FROM python:3.12-slim\n")
    if with_tests:
        _write(image / "tests" / "test_engine.py")
    return image


@pytest.mark.parametrize(
    ("org", "model", "target_id"),
    [
        ("monai", "wholeBody_ct_segmentation", "monai-wholebody-ct-segmentation"),
        ("mit", "boltz2", "mit-boltz2"),
    ],
)
def test_resolve_asset_test_dir_matches_sanitized_target_id(assets_root, org, model, target_id):
    image = _write_model_image(assets_root, org, model)

    found = resolve_asset_test_dir(ACC, target_id, assets_root=str(assets_root))

    assert found == (image / "tests").as_posix()


def test_resolve_asset_test_dir_ignores_raw_directory_names(assets_root):
    _write_model_image(assets_root, "monai", "wholeBody_ct_segmentation")

    assert resolve_asset_test_dir(ACC, "monai-wholeBody_ct_segmentation", assets_root=str(assets_root)) is None


def test_resolve_asset_test_dir_no_matching_asset(assets_root):
    _write_model_image(assets_root, "mit", "boltz2")

    assert resolve_asset_test_dir(ACC, "legacy_vllm", assets_root=str(assets_root)) is None
    assert resolve_asset_test_dir(ACC, "bentoml", assets_root=str(assets_root)) is None
    assert resolve_asset_test_dir(ACC, "mit-other", assets_root=str(assets_root)) is None


def test_resolve_asset_test_dir_asset_without_tests(assets_root):
    _write_model_image(assets_root, "mit", "boltz2", with_tests=False)

    assert resolve_asset_test_dir(ACC, "mit-boltz2", assets_root=str(assets_root)) is None


def test_resolve_asset_test_dir_cli_prints_path_or_nothing(assets_root, monkeypatch):
    _write_model_image(assets_root, "monai", "wholeBody_ct_segmentation")
    monkeypatch.chdir(assets_root.parent)
    runner = CliRunner()

    def run(target_id):
        return runner.invoke(
            cli, ["resolve-asset-test-dir", "--accelerator-family", ACC, "--base-target-id", target_id]
        )

    match = run("monai-wholebody-ct-segmentation")
    assert match.exit_code == 0
    assert match.output.strip() == "assets/cpu/monai/wholeBody_ct_segmentation/image/tests"

    no_match = run("legacy_vllm")
    assert no_match.exit_code == 0
    assert no_match.output.strip() == ""


# --------------------------------------------------------------------------- #
# Job AIMs: image/ is the complete image, not a specialized base
# --------------------------------------------------------------------------- #

JOB_PROFILE = "metadata:\n  engine: job\n  job_type: long_running\n"
SERVE_PROFILE = "metadata:\n  engine: bentoml\n"


def _write_job_model(assets_root, profiles):
    model = _model_dir(assets_root)
    _write(model / "image" / "Dockerfile", "FROM rocm/dev-ubuntu-24.04:7.2.4\n")
    for stem, content in profiles.items():
        _write(model / "profiles" / f"{stem}.yaml", content)
    return model


@pytest.mark.parametrize(
    "profiles, expected",
    [
        ({"job-mi300x-fp32-tp1-throughput": JOB_PROFILE}, True),
        ({"job-a": JOB_PROFILE, "job-b": JOB_PROFILE}, True),
        ({"job-a": "metadata:\n  engine: job\n"}, True),
        ({"x-a": "metadata:\n  engine: bentoml\n  job_type: long_running\n"}, True),
        ({"job-a": JOB_PROFILE, "bentoml-a": SERVE_PROFILE}, False),
        ({"bentoml-a": SERVE_PROFILE}, False),
        ({"bentoml-a": "metadata:\n  engine: bentoml\n  job_type: service\n"}, False),
        ({"broken": "metadata: [not, a, mapping]\n"}, False),
        ({"job-a": "metadata: [unclosed\n"}, False),
        ({"job-a": "- not\n- a mapping\n"}, False),
        ({}, False),
    ],
)
def test_is_job_aim(assets_root, profiles, expected):
    model = _write_job_model(assets_root, profiles)
    assert is_job_aim(model) is expected


def test_is_job_aim_without_profiles_dir(assets_root):
    assert is_job_aim(_model_dir(assets_root)) is False


def test_enumerate_skips_job_aim(assets_root):
    _write_job_model(assets_root, {"job-mi300x-fp32-tp1-throughput": JOB_PROFILE})
    assert enumerate_specialized_base_targets(ACC, assets_root=str(assets_root)) == []


def test_enumerate_keeps_served_model_next_to_job_aim(assets_root):
    _write_job_model(assets_root, {"job-a": JOB_PROFILE})
    served = assets_root / ACC / ORG / "served-model"
    _write(served / "image" / "Dockerfile", "FROM python:3.12-slim\n")
    _write(served / "profiles" / "bentoml-a.yaml", SERVE_PROFILE)

    targets = enumerate_specialized_base_targets(ACC, assets_root=str(assets_root))
    assert [t.target_id for t in targets] == [f"{ORG}-served-model"]


def test_parse_specialized_target_ignores_job_aim(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    _write_job_model(tmp_path / "assets", {"job-a": JOB_PROFILE})
    path = f"assets/{ACC}/{ORG}/{MODEL}/image/src/entrypoint.py"
    assert parse_specialized_target_from_path(path) is None


def test_parse_specialized_target_keeps_served_model(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    _write_job_model(tmp_path / "assets", {"bentoml-a": SERVE_PROFILE})
    path = f"assets/{ACC}/{ORG}/{MODEL}/image/Dockerfile"
    assert parse_specialized_target_from_path(path) == (AcceleratorFamily(ACC), f"{ORG}-{MODEL}")


def test_model_target_id_sanitizes():
    assert model_target_id("Cytokine_King", "FreeBindCraft") == "cytokine-king-freebindcraft"
