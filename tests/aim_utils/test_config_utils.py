# Copyright © Advanced Micro Devices, Inc., or its affiliates.
#
# SPDX-License-Identifier: MIT

import json
import os
from unittest.mock import MagicMock, patch

import pytest
from click.testing import CliRunner
from pydantic import ValidationError

from aim_utils.asset_utils import AssetDescriptor
from aim_utils.config_utils import (
    LEGACY_VLLM_BASE_TARGET_ID,
    BaseImageConfig,
    BaseImageTargetConfig,
    CiBaseImageConfig,
    CiBaseImageTarget,
    ConfigInitializer,
    Initializer,
    cli,
    get_canonical_name,
    normalize_base_image_targets,
    resolve_ci_base_image_targets,
)
from aim_utils.image_naming import ImageName

REGISTRY_HOST = "docker.io"
REGISTRY_NAMESPACE = "namespace"


@pytest.fixture
def runner():
    return CliRunner()


@pytest.fixture
def temp_assets_dir(tmp_path):
    assets_dir = tmp_path / "assets"
    assets_dir.mkdir()
    return assets_dir


@pytest.fixture
def sample_config_file(temp_assets_dir):
    model_dir = temp_assets_dir / "org" / "model"
    model_dir.mkdir(parents=True)
    config_file = model_dir / "config.yaml"
    config_file.write_text(
        """
base_image:
  registry_host: ghcr.io
  base_registry_namespace: test-namespace
  base_repository: test-repo
  base_tag: 1.0.0
"""
    )
    return config_file


ACCELERATOR_CASES = [
    (
        "instinct",
        {
            "registry_host": "docker.io",
            "base_registry_namespace": "vllm",
            "base_repository": "vllm-openai-rocm",
            "base_tag": "v0.16.0",
            "expected_repository": "aim-instinct-base",  # Private push repository
            "expected_canonical_repository": "aim-instinct-base",
            "expected_public_repository": "aim-base",  # Public backward-compatible name
            "expected_dockerfile": "docker/Dockerfile.aim-instinct-base",
            "expected_run_validation": True,
            "expected_upstream_image_ref": "docker.io/vllm/vllm-openai-rocm:v0.16.0",
        },
    ),
    (
        "epyc",
        {
            "registry_host": "docker.io",
            "base_registry_namespace": "amdih",
            "base_repository": "zendnn_zentorch",
            "base_tag": "vllm_v0.15.1_zentorch_v5.2.0_ubuntu22.04_r5.2_rc3",
            "expected_repository": "aim-epyc-base",
            "expected_canonical_repository": "aim-epyc-base",
            "expected_public_repository": "aim-epyc-base",  # Same as canonical
            "expected_dockerfile": "docker/Dockerfile.aim-epyc-base",
            "expected_run_validation": False,
            "expected_upstream_image_ref": "docker.io/amdih/zendnn_zentorch:vllm_v0.15.1_zentorch_v5.2.0_ubuntu22.04_r5.2_rc3",
        },
    ),
    (
        "radeon",
        {
            "registry_host": "docker.io",
            "base_registry_namespace": "hyoon11",
            "base_repository": "vllm-dev",
            "base_tag": "20260317_81_py3.12_torch2.9_triton3.5_navi_upstream_89a77b1_ubuntu24.04",
            "expected_repository": "aim-radeon-base",
            "expected_canonical_repository": "aim-radeon-base",
            "expected_public_repository": "aim-radeon-base",  # Same as canonical
            "expected_dockerfile": "docker/Dockerfile.aim-radeon-base",
            "expected_run_validation": False,
            "expected_upstream_image_ref": "docker.io/hyoon11/vllm-dev:20260317_81_py3.12_torch2.9_triton3.5_navi_upstream_89a77b1_ubuntu24.04",
        },
    ),
    (
        "cpu",
        {
            "registry_host": "docker.io",
            "base_registry_namespace": "vllm",
            "base_repository": "vllm-openai-cpu",
            "base_tag": "v0.16.0",
            "expected_repository": "aim-cpu-base",
            "expected_canonical_repository": "aim-cpu-base",
            "expected_public_repository": "aim-cpu-base",  # Same as canonical
            "expected_dockerfile": "docker/Dockerfile.aim-cpu-base",
            "expected_run_validation": False,
            "expected_upstream_image_ref": "docker.io/vllm/vllm-openai-cpu:v0.16.0",
        },
    ),
]


def write_base_config(tmp_path, accelerator, case):
    config_dir = tmp_path / "assets" / accelerator / "base"
    config_dir.mkdir(parents=True)
    config_file = config_dir / "config.yaml"
    config_file.write_text(
        f"""
base_image:
  registry_host: {case["registry_host"]}
  base_registry_namespace: {case["base_registry_namespace"]}
  base_repository: {case["base_repository"]}
  base_tag: {case["base_tag"]}
"""
    )
    return config_file


def write_named_base_target_config(
    tmp_path,
    accelerator,
    target_id,
    *,
    registry_host,
    base_registry_namespace,
    base_repository,
    base_tag,
):
    config_dir = tmp_path / "assets" / accelerator / "base" / target_id
    config_dir.mkdir(parents=True)
    config_file = config_dir / "config.yaml"

    config_file.write_text(
        f"""
base_image:
  registry_host: {registry_host}
  base_registry_namespace: {base_registry_namespace}
  base_repository: {base_repository}
  base_tag: {base_tag}
"""
    )
    return config_file


def get_accelerator_case(accelerator):
    for case_accelerator, case in ACCELERATOR_CASES:
        if case_accelerator == accelerator:
            return case
    raise KeyError(f"Unknown accelerator case: {accelerator}")


@pytest.fixture
def mock_file_readers():
    with patch("aim_utils.config_utils.TomlFileReader") as toml_reader:
        toml_instance = MagicMock()
        toml_instance.read_value.return_value = "2.0.0"
        toml_reader.return_value = toml_instance

        yield toml_reader


class TestGetCanonicalName:
    def test_get_canonical_name_from_option(self):
        result = get_canonical_name("org/model")
        assert result == "org/model"

    def test_get_canonical_name_from_env(self):
        with patch.dict(os.environ, {"CANONICAL_NAME": "env-org/env-model"}):
            result = get_canonical_name(None)
            assert result == "env-org/env-model"

    def test_get_canonical_name_missing_raises_error(self):
        with patch.dict(os.environ, {}, clear=True):
            os.environ.pop("CANONICAL_NAME", None)
            with pytest.raises(ValueError, match="Canonical name must be provided"):
                get_canonical_name(None)

    def test_get_canonical_name_option_takes_precedence(self):
        with patch.dict(os.environ, {"CANONICAL_NAME": "env-value"}):
            result = get_canonical_name("option-value")
            assert result == "option-value"


class TestCliInit:
    def test_init_command(self, runner, temp_assets_dir, monkeypatch):
        monkeypatch.setenv("AIM_REGISTRY_HOSTNAME", REGISTRY_HOST)
        monkeypatch.setenv("AIM_REGISTRY_NAMESPACE", REGISTRY_NAMESPACE)
        with patch("aim_utils.config_utils.ConfigInitializer") as mock_initializer:
            mock_instance = MagicMock()
            mock_initializer.return_value = mock_instance

            result = runner.invoke(cli, ["init", "--assets_path", str(temp_assets_dir)])

            assert result.exit_code == 0
            mock_initializer.assert_called_once_with(
                assets_path=str(temp_assets_dir),
                recreate=False,
                registry_host=REGISTRY_HOST,
                registry_namespace=REGISTRY_NAMESPACE,
            )
            mock_instance.initialize_all.assert_called_once()

    def test_init_command_with_options(self, runner, temp_assets_dir, monkeypatch):
        monkeypatch.setenv("AIM_REGISTRY_HOSTNAME", REGISTRY_HOST)
        monkeypatch.setenv("AIM_REGISTRY_NAMESPACE", REGISTRY_NAMESPACE)
        with patch("aim_utils.config_utils.ConfigInitializer") as mock_initializer:
            mock_instance = MagicMock()
            mock_initializer.return_value = mock_instance

            result = runner.invoke(cli, ["init", "--assets_path", str(temp_assets_dir), "--recreate"])

            assert result.exit_code == 0
            mock_initializer.assert_called_once_with(
                assets_path=str(temp_assets_dir),
                recreate=True,
                registry_host=REGISTRY_HOST,
                registry_namespace=REGISTRY_NAMESPACE,
            )


class TestCliGet:
    def test_get_command(self, runner, temp_assets_dir, sample_config_file):
        result = runner.invoke(
            cli, ["get", "base_image.base_tag", "--canonical_name", "org/model", "--assets_path", str(temp_assets_dir)]
        )

        assert result.exit_code == 0
        assert "1.0.0" == result.output.strip()

    def test_get_command_nested_key(self, runner, temp_assets_dir, sample_config_file):
        result = runner.invoke(
            cli,
            ["get", "base_image.registry_host", "--canonical_name", "org/model", "--assets_path", str(temp_assets_dir)],
        )

        assert result.exit_code == 0
        assert "ghcr.io" == result.output.strip()

    def test_get_command_missing_key(self, runner, temp_assets_dir, sample_config_file):
        result = runner.invoke(
            cli, ["get", "nonexistent.key", "--canonical_name", "org/model", "--assets_path", str(temp_assets_dir)]
        )

        assert result.exit_code == 0
        assert "None" == result.output.strip()


class TestCliGetBaseImageRef:
    def test_get_base_image_ref_with_registry(self, runner, temp_assets_dir, sample_config_file):
        result = runner.invoke(
            cli, ["get-base-image-ref", "--canonical_name", "org/model", "--assets_path", str(temp_assets_dir)]
        )

        assert result.exit_code == 0

        expected_result = "base_image_ref=ghcr.io/test-namespace/test-repo:1.0.0"
        github_output = os.getenv("GITHUB_OUTPUT")

        if github_output:
            assert "" == result.output.strip()
            with open(github_output, "r") as f:
                output_content = f.read().strip()
                assert expected_result in output_content
        else:
            assert "base_image_ref=ghcr.io/test-namespace/test-repo:1.0.0" == result.output.strip()

    def test_get_base_image_ref_incomplete_config(self, runner, temp_assets_dir):
        model_dir = temp_assets_dir / "org3" / "model3"
        model_dir.mkdir(parents=True)
        config_file = model_dir / "config.yaml"
        config_file.write_text(
            """
base_image:
  base_registry_namespace: namespace
"""
        )

        result = runner.invoke(
            cli, ["get-base-image-ref", "--canonical_name", "org3/model3", "--assets_path", str(temp_assets_dir)]
        )

        assert result.exit_code != 0
        assert isinstance(result.exception, ValueError)


class TestCliInvokeWithoutCommand:
    def test_cli_without_command(self, runner):
        result = runner.invoke(cli)
        assert result.exit_code == 0


class TestConfigInitializer:
    def test_config_initializer_creation(self, temp_assets_dir):
        initializer = ConfigInitializer(
            assets_path=str(temp_assets_dir),
            registry_host=REGISTRY_HOST,
            registry_namespace=REGISTRY_NAMESPACE,
        )
        assert isinstance(initializer, Initializer)

    def test_initialize_base_config(self, temp_assets_dir):
        base_dir = temp_assets_dir / "base"
        base_dir.mkdir()

        with patch("aim_utils.config_utils.save_yaml") as mock_save:
            initializer = ConfigInitializer(
                assets_path=str(temp_assets_dir),
                registry_host=REGISTRY_HOST,
                registry_namespace=REGISTRY_NAMESPACE,
            )
            descriptor = AssetDescriptor(
                is_base=True,
                is_custom=False,
                directory=base_dir,
                org=None,
                model_name=None,
            )
            initializer.initialize(descriptor)

            mock_save.assert_called_once()
            call_args = mock_save.call_args
            config = call_args[0][0]
            assert "base_image" in config
            assert config["base_image"]["registry_host"] == "docker.io"
            assert config["base_image"]["base_registry_namespace"] == "vllm"
            assert config["base_image"]["base_repository"] == "vllm-openai-rocm"
            assert config["base_image"]["base_tag"] == "v0.20.0"

    @pytest.mark.parametrize(("accelerator", "case"), ACCELERATOR_CASES, ids=["instinct", "epyc", "radeon", "cpu"])
    def test_initialize_model_config(self, tmp_path, mock_file_readers, accelerator, case):
        model_dir = tmp_path / "assets" / accelerator / "org" / "model"
        model_dir.mkdir(parents=True)

        with patch("aim_utils.config_utils.save_yaml") as mock_save:
            initializer = ConfigInitializer(
                assets_path=str(tmp_path / "assets" / accelerator),
                registry_host=REGISTRY_HOST,
                registry_namespace=REGISTRY_NAMESPACE,
            )
            descriptor = AssetDescriptor(
                is_base=False,
                is_custom=False,
                directory=model_dir,
                org="org",
                model_name="model",
            )
            initializer.initialize(descriptor)

            mock_save.assert_called_once()
            call_args = mock_save.call_args
            config = call_args[0][0]
            assert "base_image" in config
            assert config["base_image"]["registry_host"] == REGISTRY_HOST
            assert config["base_image"]["base_registry_namespace"] == REGISTRY_NAMESPACE
            assert config["base_image"]["base_repository"] == case["expected_canonical_repository"]
            assert config["base_image"]["base_tag"] == "2.0.0"

    def test_initialize_skips_existing_file(self, temp_assets_dir, mock_file_readers):
        model_dir = temp_assets_dir / "org" / "model"
        model_dir.mkdir(parents=True)
        config_file = model_dir / "config.yaml"
        config_file.write_text("existing: config")

        with patch("aim_utils.config_utils.save_yaml") as mock_save:
            initializer = ConfigInitializer(
                assets_path=str(temp_assets_dir),
                registry_host=REGISTRY_HOST,
                registry_namespace=REGISTRY_NAMESPACE,
            )
            descriptor = AssetDescriptor(
                is_base=False,
                is_custom=False,
                directory=model_dir,
                org="org",
                model_name="model",
            )
            initializer.initialize(descriptor)

            mock_save.assert_not_called()

    def test_initialize_recreates_existing_file(self, temp_assets_dir, mock_file_readers):
        # Create a directory structure with accelerator family
        instinct_dir = temp_assets_dir / "instinct"
        instinct_dir.mkdir()
        model_dir = instinct_dir / "org" / "model"
        model_dir.mkdir(parents=True)
        config_file = model_dir / "config.yaml"
        config_file.write_text("existing: config")

        with patch("aim_utils.config_utils.save_yaml") as mock_save:
            initializer = ConfigInitializer(
                assets_path=str(instinct_dir),
                recreate=True,
                registry_host=REGISTRY_HOST,
                registry_namespace=REGISTRY_NAMESPACE,
            )
            descriptor = AssetDescriptor(
                is_base=False,
                is_custom=False,
                directory=model_dir,
                org="org",
                model_name="model",
            )
            initializer.initialize(descriptor)

            mock_save.assert_called_once()

    def test_initialize_all_calls_initialize(self, temp_assets_dir):
        leaf_dir1 = temp_assets_dir / "org1" / "model1"
        leaf_dir2 = temp_assets_dir / "org2" / "model2"
        leaf_dir1.mkdir(parents=True)
        leaf_dir2.mkdir(parents=True)

        descriptor1 = AssetDescriptor(
            is_base=False,
            is_custom=False,
            directory=leaf_dir1,
            org="org1",
            model_name="model1",
        )
        descriptor2 = AssetDescriptor(
            is_base=False,
            is_custom=False,
            directory=leaf_dir2,
            org="org2",
            model_name="model2",
        )

        with (
            patch.object(ConfigInitializer, "get_reference_descriptors") as mock_get_descriptors,
            patch.object(ConfigInitializer, "initialize") as mock_init,
        ):
            mock_get_descriptors.return_value = [descriptor1, descriptor2]

            initializer = ConfigInitializer(
                assets_path=str(temp_assets_dir),
                registry_host=REGISTRY_HOST,
                registry_namespace=REGISTRY_NAMESPACE,
            )
            initializer.initialize_all()

            assert mock_init.call_count == 2
            mock_init.assert_any_call(descriptor1)
            mock_init.assert_any_call(descriptor2)


class TestCiBaseImageConfig:
    @pytest.mark.parametrize(("accelerator", "case"), ACCELERATOR_CASES, ids=["instinct", "epyc", "radeon", "cpu"])
    def test_legacy_target_from_resolve_ci_base_image_targets(self, tmp_path, monkeypatch, accelerator, case):
        """Test CiBaseImageConfig creation via resolve_ci_base_image_targets for legacy target."""
        monkeypatch.chdir(tmp_path)
        write_base_config(tmp_path, accelerator, case)

        ci_targets = resolve_ci_base_image_targets(accelerator)
        legacy_target = next((t for t in ci_targets if t.target_id == LEGACY_VLLM_BASE_TARGET_ID), None)
        assert legacy_target is not None

        ci_config = CiBaseImageConfig(
            base_target_id=legacy_target.target_id,
            image_name=legacy_target.image_name,
            dockerfile=legacy_target.dockerfile,
            run_validation=legacy_target.run_validation,
            upstream_image_ref=legacy_target.upstream_image_ref,
        )

        assert ci_config.repository == case["expected_repository"]
        assert ci_config.public_repository == case["expected_public_repository"]
        assert ci_config.has_alias == (case["expected_canonical_repository"] != case["expected_public_repository"])
        assert ci_config.image_name == ImageName(
            canonical=case["expected_canonical_repository"], public=case["expected_public_repository"]
        )
        assert ci_config.dockerfile == case["expected_dockerfile"]
        assert ci_config.run_validation is case["expected_run_validation"]
        assert ci_config.upstream_image_ref == case["expected_upstream_image_ref"]

    def test_model_dump_json(self):
        """Test JSON serialization of CiBaseImageConfig."""
        ci_config = CiBaseImageConfig(
            base_target_id=LEGACY_VLLM_BASE_TARGET_ID,
            image_name=ImageName(canonical="aim-instinct-base", public="aim-base"),
            dockerfile="docker/Dockerfile.aim-instinct-base",
            run_validation=True,
            upstream_image_ref="docker.io/rocm/vllm:test",
        )

        json_output = ci_config.model_dump_json()
        parsed = json.loads(json_output)

        assert parsed["base_target_id"] == LEGACY_VLLM_BASE_TARGET_ID
        assert parsed["repository"] == "aim-instinct-base"
        assert parsed["public_repository"] == "aim-base"
        assert parsed["has_alias"] is True
        assert parsed["dockerfile"] == "docker/Dockerfile.aim-instinct-base"
        assert parsed["run_validation"] is True  # Boolean, not string
        assert parsed["upstream_image_ref"] == "docker.io/rocm/vllm:test"


class TestBaseImageTargetNormalization:
    def test_normalize_legacy_base_image_to_legacy_vllm_target(self):
        normalized = normalize_base_image_targets(
            {
                "base_image": {
                    "registry_host": "docker.io",
                    "base_registry_namespace": "vllm",
                    "base_repository": "vllm-openai-rocm",
                    "base_tag": "v0.16.0",
                }
            }
        )

        assert list(normalized.keys()) == [LEGACY_VLLM_BASE_TARGET_ID]
        assert normalized[LEGACY_VLLM_BASE_TARGET_ID] == BaseImageTargetConfig(
            target_id=LEGACY_VLLM_BASE_TARGET_ID,
            registry_host="docker.io",
            base_registry_namespace="vllm",
            base_repository="vllm-openai-rocm",
            base_tag="v0.16.0",
        )

    def test_normalize_empty_config_returns_empty_mapping(self):
        normalized = normalize_base_image_targets({})
        assert normalized == {}

    def test_normalize_rejects_inline_base_images_mapping(self):
        with pytest.raises(ValueError, match="inline 'base_images' mapping is not supported"):
            normalize_base_image_targets(
                {
                    "base_images": {
                        "bentoml": {
                            "registry_host": "docker.io",
                            "base_registry_namespace": "rocm",
                            "base_repository": "pytorch",
                            "base_tag": "rocm7.0",
                        }
                    }
                }
            )

    def test_normalize_rejects_base_image_plus_inline_base_images(self):
        with pytest.raises(ValueError, match="inline 'base_images' mapping is not supported"):
            normalize_base_image_targets(
                {
                    "base_image": {
                        "registry_host": "docker.io",
                        "base_registry_namespace": "vllm",
                        "base_repository": "vllm-openai-rocm",
                        "base_tag": "v0.16.0",
                    },
                    "base_images": {
                        "bentoml": {
                            "registry_host": "docker.io",
                            "base_registry_namespace": "rocm",
                            "base_repository": "pytorch",
                            "base_tag": "rocm7.0",
                        }
                    },
                }
            )

    def test_normalize_requires_base_image_required_fields(self):
        with pytest.raises(ValidationError, match="base_tag"):
            normalize_base_image_targets(
                {
                    "base_image": {
                        "registry_host": "docker.io",
                        "base_registry_namespace": "vllm",
                        "base_repository": "vllm-openai-rocm",
                    }
                }
            )

    def test_normalize_allows_extra_top_level_fields(self):
        normalized = normalize_base_image_targets(
            {
                "metadata": {"owner": "team-ai"},
                "base_image": {
                    "registry_host": "docker.io",
                    "base_registry_namespace": "vllm",
                    "base_repository": "vllm-openai-rocm",
                    "base_tag": "v0.16.0",
                },
            }
        )

        assert list(normalized.keys()) == [LEGACY_VLLM_BASE_TARGET_ID]
        assert normalized[LEGACY_VLLM_BASE_TARGET_ID].base_repository == "vllm-openai-rocm"


class TestCiBaseImageTarget:
    def test_model_dump_json(self):
        target = CiBaseImageTarget(
            target_id=LEGACY_VLLM_BASE_TARGET_ID,
            image_name=ImageName(canonical="aim-instinct-base", public="aim-base"),
            dockerfile="docker/Dockerfile.aim-instinct-base",
            run_validation=True,
            upstream_image_ref="docker.io/rocm/vllm:test",
        )

        parsed = json.loads(target.model_dump_json())

        assert parsed["target_id"] == LEGACY_VLLM_BASE_TARGET_ID
        assert parsed["repository"] == "aim-instinct-base"
        assert parsed["public_repository"] == "aim-base"
        assert parsed["has_alias"] is True


class TestResolveBuildConfigCommand:
    @pytest.mark.parametrize(("accelerator", "case"), ACCELERATOR_CASES, ids=["instinct", "epyc", "radeon", "cpu"])
    def test_resolve_build_config(self, runner, tmp_path, monkeypatch, accelerator, case):
        """Test resolve-build-config command for supported accelerators."""
        monkeypatch.chdir(tmp_path)
        monkeypatch.delenv("GITHUB_OUTPUT", raising=False)
        write_base_config(tmp_path, accelerator, case)

        result = runner.invoke(cli, ["resolve-build-config", "--accelerator-family", accelerator])

        assert result.exit_code == 0
        output = json.loads(result.output.strip())
        assert output["base_target_id"] == LEGACY_VLLM_BASE_TARGET_ID
        assert output["repository"] == case["expected_repository"]
        assert output["dockerfile"] == case["expected_dockerfile"]
        assert output["run_validation"] is case["expected_run_validation"]
        assert output["upstream_image_ref"] == case["expected_upstream_image_ref"]

    @pytest.mark.parametrize(("accelerator", "case"), ACCELERATOR_CASES, ids=["instinct", "epyc", "radeon", "cpu"])
    def test_resolve_build_config_with_github_output(self, runner, tmp_path, monkeypatch, accelerator, case):
        """Test resolve-build-config writes to GITHUB_OUTPUT when set."""
        monkeypatch.chdir(tmp_path)
        write_base_config(tmp_path, accelerator, case)

        github_output_file = tmp_path / "github_output"
        github_output_file.touch()

        with patch.dict(os.environ, {"GITHUB_OUTPUT": str(github_output_file)}):
            result = runner.invoke(cli, ["resolve-build-config", "--accelerator-family", accelerator])

            assert result.exit_code == 0

            # Verify GITHUB_OUTPUT file content
            content = github_output_file.read_text()
            assert content.startswith("config=")
            json_part = content.replace("config=", "").strip()
            parsed = json.loads(json_part)
            assert parsed["base_target_id"] == LEGACY_VLLM_BASE_TARGET_ID
            assert parsed["repository"] == case["expected_repository"]
            assert parsed["dockerfile"] == case["expected_dockerfile"]
            assert parsed["run_validation"] is case["expected_run_validation"]
            assert parsed["upstream_image_ref"] == case["expected_upstream_image_ref"]

    def test_resolve_build_config_missing_accelerator_family(self, runner):
        """Test resolve-build-config fails without accelerator type."""
        result = runner.invoke(cli, ["resolve-build-config"])
        assert result.exit_code != 0
        assert "Missing option" in result.output or "required" in result.output.lower()

    def test_resolve_build_config_invalid_accelerator_family(self, runner):
        """Test resolve-build-config rejects unsupported accelerator values at CLI parsing."""
        result = runner.invoke(cli, ["resolve-build-config", "--accelerator-family", "invalid"])
        assert result.exit_code != 0
        assert "Invalid value for '--accelerator-family'" in result.output

    def test_resolve_build_config_uses_legacy_target_when_named_targets_exist(self, runner, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        monkeypatch.delenv("GITHUB_OUTPUT", raising=False)
        case = get_accelerator_case("instinct")
        write_base_config(tmp_path, "instinct", case)
        write_named_base_target_config(
            tmp_path,
            "instinct",
            "bentoml",
            registry_host="docker.io",
            base_registry_namespace="rocm",
            base_repository="pytorch",
            base_tag="rocm7.0_ubuntu24.04_py3.12_pytorch_release_2.8.0",
        )

        result = runner.invoke(cli, ["resolve-build-config", "--accelerator-family", "instinct"])

        assert result.exit_code == 0
        output = json.loads(result.output.strip())
        assert output["base_target_id"] == LEGACY_VLLM_BASE_TARGET_ID
        assert output["upstream_image_ref"] == case["expected_upstream_image_ref"]

    def test_resolve_build_config_with_explicit_target_id(self, runner, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        monkeypatch.delenv("GITHUB_OUTPUT", raising=False)
        case = get_accelerator_case("instinct")
        write_base_config(tmp_path, "instinct", case)
        write_named_base_target_config(
            tmp_path,
            "instinct",
            "bentoml",
            registry_host="docker.io",
            base_registry_namespace="rocm",
            base_repository="pytorch",
            base_tag="rocm7.0_ubuntu24.04_py3.12_pytorch_release_2.8.0",
        )

        result = runner.invoke(
            cli, ["resolve-build-config", "--accelerator-family", "instinct", "--base-target-id", "bentoml"]
        )

        assert result.exit_code == 0
        output = json.loads(result.output.strip())
        assert output["base_target_id"] == "bentoml"
        assert output["upstream_image_ref"] == "docker.io/rocm/pytorch:rocm7.0_ubuntu24.04_py3.12_pytorch_release_2.8.0"

    def test_resolve_build_config_with_unknown_target_id(self, runner, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        monkeypatch.delenv("GITHUB_OUTPUT", raising=False)
        case = get_accelerator_case("instinct")
        write_base_config(tmp_path, "instinct", case)

        result = runner.invoke(
            cli, ["resolve-build-config", "--accelerator-family", "instinct", "--base-target-id", "nonexistent"]
        )

        assert result.exit_code != 0
        assert "nonexistent" in result.output
        assert LEGACY_VLLM_BASE_TARGET_ID in result.output

    def test_resolve_build_config_fails_without_legacy_target(self, runner, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        monkeypatch.delenv("GITHUB_OUTPUT", raising=False)
        write_named_base_target_config(
            tmp_path,
            "instinct",
            "bentoml",
            registry_host="docker.io",
            base_registry_namespace="rocm",
            base_repository="pytorch",
            base_tag="rocm7.0_ubuntu24.04_py3.12_pytorch_release_2.8.0",
        )

        result = runner.invoke(cli, ["resolve-build-config", "--accelerator-family", "instinct"])

        assert result.exit_code != 0
        assert LEGACY_VLLM_BASE_TARGET_ID in result.output


class TestResolveBuildTargetsCommand:
    def test_resolve_build_targets_legacy_only(self, runner, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        monkeypatch.delenv("GITHUB_OUTPUT", raising=False)
        case = get_accelerator_case("instinct")
        write_base_config(tmp_path, "instinct", case)

        result = runner.invoke(cli, ["resolve-build-targets", "--accelerator-family", "instinct"])

        assert result.exit_code == 0
        targets = json.loads(result.output.strip())
        assert len(targets) == 1
        assert targets[0]["target_id"] == LEGACY_VLLM_BASE_TARGET_ID
        assert targets[0]["upstream_image_ref"] == case["expected_upstream_image_ref"]

    def test_resolve_build_targets_discovers_named_target_folder(self, runner, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        monkeypatch.delenv("GITHUB_OUTPUT", raising=False)
        case = get_accelerator_case("instinct")
        write_base_config(tmp_path, "instinct", case)
        write_named_base_target_config(
            tmp_path,
            "instinct",
            "bentoml",
            registry_host="docker.io",
            base_registry_namespace="rocm",
            base_repository="pytorch",
            base_tag="rocm7.0_ubuntu24.04_py3.12_pytorch_release_2.8.0",
        )

        result = runner.invoke(cli, ["resolve-build-targets", "--accelerator-family", "instinct"])

        assert result.exit_code == 0
        targets = json.loads(result.output.strip())
        assert [target["target_id"] for target in targets] == [LEGACY_VLLM_BASE_TARGET_ID, "bentoml"]

        bentoml_target = targets[1]

        assert bentoml_target["upstream_image_ref"] == (
            "docker.io/rocm/pytorch:rocm7.0_ubuntu24.04_py3.12_pytorch_release_2.8.0"
        )

    def test_resolve_build_targets_with_github_output(self, runner, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        case = get_accelerator_case("instinct")
        write_base_config(tmp_path, "instinct", case)

        github_output_file = tmp_path / "github_output"
        github_output_file.touch()

        with patch.dict(os.environ, {"GITHUB_OUTPUT": str(github_output_file)}):
            result = runner.invoke(cli, ["resolve-build-targets", "--accelerator-family", "instinct"])

            assert result.exit_code == 0

            content = github_output_file.read_text()
            assert content.startswith("targets=")
            json_part = content.replace("targets=", "").strip()
            targets = json.loads(json_part)
            assert len(targets) == 1
            assert targets[0]["target_id"] == LEGACY_VLLM_BASE_TARGET_ID

    def test_legacy_target_gets_legacy_vllm_image_name(self, runner, tmp_path, monkeypatch):
        """legacy_vllm target uses the backward-compatible per-accelerator name (no target discriminator)."""
        monkeypatch.chdir(tmp_path)
        monkeypatch.delenv("GITHUB_OUTPUT", raising=False)
        case = get_accelerator_case("instinct")
        write_base_config(tmp_path, "instinct", case)

        result = runner.invoke(cli, ["resolve-build-targets", "--accelerator-family", "instinct"])

        assert result.exit_code == 0
        targets = json.loads(result.output.strip())
        legacy = targets[0]
        assert legacy["target_id"] == LEGACY_VLLM_BASE_TARGET_ID
        assert legacy["repository"] == "aim-instinct-base"
        assert legacy["public_repository"] == "aim-base"
        assert legacy["has_alias"] is True

    def test_named_target_gets_discriminated_image_name(self, runner, tmp_path, monkeypatch):
        """Named targets include target_id in repository name with no public alias."""
        monkeypatch.chdir(tmp_path)
        monkeypatch.delenv("GITHUB_OUTPUT", raising=False)
        case = get_accelerator_case("instinct")
        write_base_config(tmp_path, "instinct", case)
        write_named_base_target_config(
            tmp_path,
            "instinct",
            "bentoml",
            registry_host="docker.io",
            base_registry_namespace="rocm",
            base_repository="pytorch",
            base_tag="rocm7.0_ubuntu24.04_py3.12_pytorch_release_2.8.0",
        )

        result = runner.invoke(cli, ["resolve-build-targets", "--accelerator-family", "instinct"])

        assert result.exit_code == 0
        targets = json.loads(result.output.strip())
        bentoml = next(t for t in targets if t["target_id"] == "bentoml")
        assert bentoml["repository"] == "aim-instinct-bentoml-base"
        assert bentoml["public_repository"] == "aim-instinct-bentoml-base"
        assert bentoml["has_alias"] is False

    def test_each_target_gets_independent_image_name(self, runner, tmp_path, monkeypatch):
        """Different targets in the same accelerator family get different repository names."""
        monkeypatch.chdir(tmp_path)
        monkeypatch.delenv("GITHUB_OUTPUT", raising=False)
        case = get_accelerator_case("instinct")
        write_base_config(tmp_path, "instinct", case)
        write_named_base_target_config(
            tmp_path,
            "instinct",
            "sglang",
            registry_host="docker.io",
            base_registry_namespace="rocm",
            base_repository="pytorch",
            base_tag="rocm7.0_ubuntu24.04_py3.12_pytorch_release_2.8.0",
        )

        result = runner.invoke(cli, ["resolve-build-targets", "--accelerator-family", "instinct"])

        assert result.exit_code == 0
        targets = json.loads(result.output.strip())
        repos = {t["target_id"]: t["repository"] for t in targets}
        assert repos[LEGACY_VLLM_BASE_TARGET_ID] == "aim-instinct-base"
        assert repos["sglang"] == "aim-instinct-sglang-base"


# ---------------------------------------------------------------------------
# Tests for BaseImageConfig field validators
# ---------------------------------------------------------------------------


class TestBaseImageConfigRegistryValidator:
    """Validate the registry_host field_validator on BaseImageConfig."""

    @pytest.mark.parametrize("host", sorted(BaseImageConfig.ALLOWED_REGISTRY_HOSTS))
    def test_allowed_registries_accepted(self, host):
        config = BaseImageConfig(
            registry_host=host,
            base_registry_namespace="ns",
            base_repository="repo",
            base_tag="some-upstream-tag",
        )
        assert config.registry_host == host

    @pytest.mark.parametrize("host", ["evil.io", "", "quay.io", "localhost:5000"])
    def test_disallowed_registries_rejected(self, host):
        with pytest.raises(Exception, match="Invalid registry host"):
            BaseImageConfig(
                registry_host=host,
                base_registry_namespace="ns",
                base_repository="repo",
                base_tag="some-tag",
            )


class TestBaseImageConfigTagValidator:
    """Validate the base_tag field_validator on BaseImageConfig."""

    @pytest.mark.parametrize("base_repository", ["aim-base", "aim-radeon-base", "aim-epyc-base", "aim-cpu-base"])
    def test_aim_repo_valid_base_version(self, base_repository):
        """When base_repository starts with 'aim', base_tag must be a valid AIM base version."""
        config = BaseImageConfig(
            registry_host="ghcr.io",
            base_registry_namespace="silogen",
            base_repository=base_repository,
            base_tag="0.11",
        )
        assert config.base_tag == "0.11"

    @pytest.mark.parametrize("tag", ["0.11-rc1", "0.11-preview", "1.0"])
    @pytest.mark.parametrize("base_repository", ["aim-base", "aim-radeon-base", "aim-epyc-base", "aim-cpu-base"])
    def test_aim_repo_valid_base_version_with_suffix(self, tag, base_repository):
        config = BaseImageConfig(
            registry_host="ghcr.io",
            base_registry_namespace="silogen",
            base_repository=base_repository,
            base_tag=tag,
        )
        assert config.base_tag == tag

    @pytest.mark.parametrize("tag", ["BROKEN", "0.11.0", "0.11-rc0", "abc"])
    @pytest.mark.parametrize("base_repository", ["aim-base", "aim-radeon-base", "aim-epyc-base", "aim-cpu-base"])
    def test_aim_repo_invalid_base_version_rejected(self, tag, base_repository):
        with pytest.raises(Exception, match="Invalid version format"):
            BaseImageConfig(
                registry_host="ghcr.io",
                base_registry_namespace="silogen",
                base_repository=base_repository,
                base_tag=tag,
            )

    def test_non_aim_repo_allows_freeform_tag(self):
        """Upstream repos (e.g. docker.io/rocm/vllm) can have arbitrary tags."""
        config = BaseImageConfig(
            registry_host="docker.io",
            base_registry_namespace="rocm",
            base_repository="vllm",
            base_tag="rocm7.0.0_vllm_0.11.2_20251210",
        )
        assert config.base_tag == "rocm7.0.0_vllm_0.11.2_20251210"


class TestValidateConfigCommand:
    """Tests for the 'validate' CLI command."""

    def test_valid_config_file(self, runner, tmp_path):
        config_file = tmp_path / "config.yaml"
        config_file.write_text(
            "base_image:\n"
            "  registry_host: ghcr.io\n"
            "  base_registry_namespace: silogen\n"
            "  base_repository: aim-base\n"
            '  base_tag: "0.11"\n'
        )
        result = runner.invoke(cli, ["validate", str(config_file)])
        assert result.exit_code == 0

    def test_invalid_registry_reports_error(self, runner, tmp_path):
        config_file = tmp_path / "config.yaml"
        config_file.write_text(
            "base_image:\n"
            "  registry_host: evil.io\n"
            "  base_registry_namespace: ns\n"
            "  base_repository: aim-base\n"
            '  base_tag: "0.11"\n'
        )
        result = runner.invoke(cli, ["validate", str(config_file)])
        assert result.exit_code != 0

    def test_invalid_tag_reports_error(self, runner, tmp_path):
        config_file = tmp_path / "config.yaml"
        config_file.write_text(
            "base_image:\n"
            "  registry_host: ghcr.io\n"
            "  base_registry_namespace: silogen\n"
            "  base_repository: aim-base\n"
            "  base_tag: BROKEN\n"
        )
        result = runner.invoke(cli, ["validate", str(config_file)])
        assert result.exit_code != 0
