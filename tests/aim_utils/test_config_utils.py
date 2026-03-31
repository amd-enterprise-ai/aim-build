# Copyright © Advanced Micro Devices, Inc., or its affiliates.
#
# SPDX-License-Identifier: MIT

import json
import os
from unittest.mock import MagicMock, patch

import pytest
from click.testing import CliRunner

from aim_utils.asset_utils import AssetDescriptor
from aim_utils.config_utils import (
    BaseImageConfig,
    CiBaseImageConfig,
    ConfigInitializer,
    Initializer,
    cli,
    get_canonical_name,
)


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


@pytest.fixture
def mock_file_readers():
    with (
        patch("aim_utils.config_utils.KeyValueFileReader") as kv_reader,
        patch("aim_utils.config_utils.TomlFileReader") as toml_reader,
    ):
        kv_instance = MagicMock()
        kv_instance.read_value.side_effect = lambda key: {
            "BASE_REGISTRY_NAMESPACE": "test-namespace",
            "BASE_REPOSITORY": "test-repo",
            "BASE_TAG": "1.0.0",
        }.get(key)
        kv_reader.return_value = kv_instance

        toml_instance = MagicMock()
        toml_instance.read_value.return_value = "2.0.0"
        toml_reader.return_value = toml_instance

        yield kv_reader, toml_reader


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
    def test_init_command(self, runner, temp_assets_dir):
        with patch("aim_utils.config_utils.ConfigInitializer") as mock_initializer:
            mock_instance = MagicMock()
            mock_initializer.return_value = mock_instance

            result = runner.invoke(cli, ["init", "--assets_path", str(temp_assets_dir)])

            assert result.exit_code == 0
            mock_initializer.assert_called_once_with(assets_path=str(temp_assets_dir), recreate=False)
            mock_instance.initialize_all.assert_called_once()

    def test_init_command_with_options(self, runner, temp_assets_dir):
        with patch("aim_utils.config_utils.ConfigInitializer") as mock_initializer:
            mock_instance = MagicMock()
            mock_initializer.return_value = mock_instance

            result = runner.invoke(cli, ["init", "--assets_path", str(temp_assets_dir), "--recreate"])

            assert result.exit_code == 0
            mock_initializer.assert_called_once_with(assets_path=str(temp_assets_dir), recreate=True)


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
        initializer = ConfigInitializer(assets_path=str(temp_assets_dir))
        assert isinstance(initializer, Initializer)

    def test_initialize_base_config(self, temp_assets_dir, mock_file_readers):
        base_dir = temp_assets_dir / "base"
        base_dir.mkdir()

        with patch("aim_utils.config_utils.save_yaml") as mock_save:
            initializer = ConfigInitializer(assets_path=str(temp_assets_dir))
            descriptor = AssetDescriptor(is_base=True, is_custom=False, directory=base_dir, org=None, model_name=None)
            initializer.initialize(descriptor)

            mock_save.assert_called_once()
            call_args = mock_save.call_args
            config = call_args[0][0]
            assert "base_image" in config
            assert config["base_image"]["base_registry_namespace"] == "test-namespace"

    def test_initialize_model_config(self, temp_assets_dir, mock_file_readers):
        model_dir = temp_assets_dir / "org" / "model"
        model_dir.mkdir(parents=True)

        with patch("aim_utils.config_utils.save_yaml") as mock_save:
            initializer = ConfigInitializer(assets_path=str(temp_assets_dir))
            descriptor = AssetDescriptor(
                is_base=False, is_custom=False, directory=model_dir, org="org", model_name="model"
            )
            initializer.initialize(descriptor)

            mock_save.assert_called_once()
            call_args = mock_save.call_args
            config = call_args[0][0]
            assert "base_image" in config
            assert config["base_image"]["registry_host"] == "ghcr.io"

    def test_initialize_skips_existing_file(self, temp_assets_dir, mock_file_readers):
        model_dir = temp_assets_dir / "org" / "model"
        model_dir.mkdir(parents=True)
        config_file = model_dir / "config.yaml"
        config_file.write_text("existing: config")

        with patch("aim_utils.config_utils.save_yaml") as mock_save:
            initializer = ConfigInitializer(assets_path=str(temp_assets_dir))
            descriptor = AssetDescriptor(
                is_base=False, is_custom=False, directory=model_dir, org="org", model_name="model"
            )
            initializer.initialize(descriptor)

            mock_save.assert_not_called()

    def test_initialize_recreates_existing_file(self, temp_assets_dir, mock_file_readers):
        model_dir = temp_assets_dir / "org" / "model"
        model_dir.mkdir(parents=True)
        config_file = model_dir / "config.yaml"
        config_file.write_text("existing: config")

        with patch("aim_utils.config_utils.save_yaml") as mock_save:
            initializer = ConfigInitializer(assets_path=str(temp_assets_dir), recreate=True)
            descriptor = AssetDescriptor(
                is_base=False, is_custom=False, directory=model_dir, org="org", model_name="model"
            )
            initializer.initialize(descriptor)

            mock_save.assert_called_once()

    def test_initialize_all_calls_initialize(self, temp_assets_dir):
        leaf_dir1 = temp_assets_dir / "org1" / "model1"
        leaf_dir2 = temp_assets_dir / "org2" / "model2"
        leaf_dir1.mkdir(parents=True)
        leaf_dir2.mkdir(parents=True)

        descriptor1 = AssetDescriptor(
            is_base=False, is_custom=False, directory=leaf_dir1, org="org1", model_name="model1"
        )
        descriptor2 = AssetDescriptor(
            is_base=False, is_custom=False, directory=leaf_dir2, org="org2", model_name="model2"
        )

        with (
            patch.object(ConfigInitializer, "get_reference_descriptors") as mock_get_descriptors,
            patch.object(ConfigInitializer, "initialize") as mock_init,
        ):
            mock_get_descriptors.return_value = [descriptor1, descriptor2]

            initializer = ConfigInitializer(assets_path=str(temp_assets_dir))
            initializer.initialize_all()

            assert mock_init.call_count == 2
            mock_init.assert_any_call(descriptor1)
            mock_init.assert_any_call(descriptor2)


@pytest.fixture
def instinct_base_config_file(tmp_path):
    """Create instinct base config file for testing."""
    config_dir = tmp_path / "assets" / "instinct" / "base"
    config_dir.mkdir(parents=True)
    config_file = config_dir / "config.yaml"
    config_file.write_text(
        """
base_image:
  registry_host: docker.io
  base_registry_namespace: rocm
  base_repository: vllm
  base_tag: rocm7.0.0_vllm_0.11.2_20251210
"""
    )
    return config_file


@pytest.fixture
def epyc_base_config_file(tmp_path):
    """Create epyc base config file for testing."""
    config_dir = tmp_path / "assets" / "epyc" / "base"
    config_dir.mkdir(parents=True)
    config_file = config_dir / "config.yaml"
    config_file.write_text(
        """
base_image:
  registry_host: docker.io
  base_registry_namespace: amdih
  base_repository: zendnn_zentorch
  base_tag: vllm_v0.15.1_zentorch_v5.2.0_ubuntu22.04_r5.2_rc3
"""
    )
    return config_file


class TestCiBaseImageConfig:
    def test_from_accelerator_type_instinct(self, tmp_path, instinct_base_config_file):
        """Test CiBaseImageConfig creation for instinct accelerator."""
        with patch("aim_utils.config_utils.BaseImageConfig.from_yaml_file") as mock_from_yaml:
            from aim_utils.config_utils import BaseImageConfig

            mock_config = BaseImageConfig(
                registry_host="docker.io",
                base_registry_namespace="rocm",
                base_repository="vllm",
                base_tag="rocm7.0.0_vllm_0.11.2_20251210",
            )
            mock_from_yaml.return_value = mock_config

            ci_config = CiBaseImageConfig.from_accelerator_type("instinct")

            assert ci_config.repository == "aim-base"
            assert ci_config.dockerfile == "docker/Dockerfile.aim-base"
            assert ci_config.run_validation is True
            assert ci_config.base_image_ref == "docker.io/rocm/vllm:rocm7.0.0_vllm_0.11.2_20251210"

    def test_from_accelerator_type_epyc(self, tmp_path, epyc_base_config_file):
        """Test CiBaseImageConfig creation for epyc accelerator."""
        with patch("aim_utils.config_utils.BaseImageConfig.from_yaml_file") as mock_from_yaml:
            from aim_utils.config_utils import BaseImageConfig

            mock_config = BaseImageConfig(
                registry_host="docker.io",
                base_registry_namespace="amdih",
                base_repository="zendnn_zentorch",
                base_tag="vllm_v0.15.1_zentorch_v5.2.0_ubuntu22.04_r5.2_rc3",
            )
            mock_from_yaml.return_value = mock_config

            ci_config = CiBaseImageConfig.from_accelerator_type("epyc")

            assert ci_config.repository == "aim-epyc-base"
            assert ci_config.dockerfile == "docker/Dockerfile.aim-epyc-base"
            assert ci_config.run_validation is False
            assert (
                ci_config.base_image_ref
                == "docker.io/amdih/zendnn_zentorch:vllm_v0.15.1_zentorch_v5.2.0_ubuntu22.04_r5.2_rc3"
            )

    def test_model_dump_json(self):
        """Test JSON serialization of CiBaseImageConfig."""
        ci_config = CiBaseImageConfig(
            repository="aim-base",
            dockerfile="docker/Dockerfile.aim-base",
            run_validation=True,
            base_image_ref="docker.io/rocm/vllm:test",
        )

        json_output = ci_config.model_dump_json()
        parsed = json.loads(json_output)

        assert parsed["repository"] == "aim-base"
        assert parsed["dockerfile"] == "docker/Dockerfile.aim-base"
        assert parsed["run_validation"] is True  # Boolean, not string
        assert parsed["base_image_ref"] == "docker.io/rocm/vllm:test"


class TestResolveBuildConfigCommand:
    def test_resolve_build_config_instinct(self, runner, tmp_path, instinct_base_config_file, monkeypatch):
        """Test resolve-build-config command for instinct accelerator."""
        # Change to tmp_path so the config files can be found
        monkeypatch.chdir(tmp_path)
        # Clear GITHUB_OUTPUT to force output to stdout instead of file
        monkeypatch.delenv("GITHUB_OUTPUT", raising=False)

        with patch("aim_utils.config_utils.CiBaseImageConfig.from_accelerator_type") as mock_from_type:
            mock_ci_config = CiBaseImageConfig(
                repository="aim-base",
                dockerfile="docker/Dockerfile.aim-base",
                run_validation=True,
                base_image_ref="docker.io/rocm/vllm:rocm7.0.0_vllm_0.11.2_20251210",
            )
            mock_from_type.return_value = mock_ci_config

            result = runner.invoke(cli, ["resolve-build-config", "--accelerator-type", "instinct"])

            assert result.exit_code == 0
            output = json.loads(result.output.strip())
            assert output["repository"] == "aim-base"
            assert output["dockerfile"] == "docker/Dockerfile.aim-base"
            assert output["run_validation"] is True
            assert output["base_image_ref"] == "docker.io/rocm/vllm:rocm7.0.0_vllm_0.11.2_20251210"

    def test_resolve_build_config_epyc(self, runner, tmp_path, epyc_base_config_file, monkeypatch):
        """Test resolve-build-config command for epyc accelerator."""
        # Change to tmp_path so the config files can be found
        monkeypatch.chdir(tmp_path)
        # Clear GITHUB_OUTPUT to force output to stdout instead of file
        monkeypatch.delenv("GITHUB_OUTPUT", raising=False)

        with patch("aim_utils.config_utils.CiBaseImageConfig.from_accelerator_type") as mock_from_type:
            mock_ci_config = CiBaseImageConfig(
                repository="aim-epyc-base",
                dockerfile="docker/Dockerfile.aim-epyc-base",
                run_validation=False,
                base_image_ref="docker.io/amdih/zendnn_zentorch:vllm_v0.15.1_zentorch_v5.2.0_ubuntu22.04_r5.2_rc3",
            )
            mock_from_type.return_value = mock_ci_config

            result = runner.invoke(cli, ["resolve-build-config", "--accelerator-type", "epyc"])

            assert result.exit_code == 0
            output = json.loads(result.output.strip())
            assert output["repository"] == "aim-epyc-base"
            assert output["dockerfile"] == "docker/Dockerfile.aim-epyc-base"
            assert output["run_validation"] is False
            assert (
                output["base_image_ref"]
                == "docker.io/amdih/zendnn_zentorch:vllm_v0.15.1_zentorch_v5.2.0_ubuntu22.04_r5.2_rc3"
            )

    def test_resolve_build_config_with_github_output(self, runner, tmp_path, instinct_base_config_file, monkeypatch):
        """Test resolve-build-config writes to GITHUB_OUTPUT when set."""
        # Change to tmp_path so the config files can be found
        monkeypatch.chdir(tmp_path)

        github_output_file = tmp_path / "github_output"
        github_output_file.touch()

        with patch("aim_utils.config_utils.CiBaseImageConfig.from_accelerator_type") as mock_from_type:
            mock_ci_config = CiBaseImageConfig(
                repository="aim-base",
                dockerfile="docker/Dockerfile.aim-base",
                run_validation=True,
                base_image_ref="docker.io/rocm/vllm:test",
            )
            mock_from_type.return_value = mock_ci_config

            with patch.dict(os.environ, {"GITHUB_OUTPUT": str(github_output_file)}):
                result = runner.invoke(cli, ["resolve-build-config", "--accelerator-type", "instinct"])

                assert result.exit_code == 0

                # Verify GITHUB_OUTPUT file content
                content = github_output_file.read_text()
                assert content.startswith("config=")
                json_part = content.replace("config=", "").strip()
                parsed = json.loads(json_part)
                assert parsed["repository"] == "aim-base"
                assert parsed["run_validation"] is True

    def test_resolve_build_config_missing_accelerator_type(self, runner):
        """Test resolve-build-config fails without accelerator type."""
        result = runner.invoke(cli, ["resolve-build-config"])
        assert result.exit_code != 0
        assert "Missing option" in result.output or "required" in result.output.lower()


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

    def test_aim_repo_valid_base_version(self):
        """When base_repository starts with 'aim', base_tag must be a valid AIM base version."""
        config = BaseImageConfig(
            registry_host="ghcr.io",
            base_registry_namespace="silogen",
            base_repository="aim-base",
            base_tag="0.11",
        )
        assert config.base_tag == "0.11"

    @pytest.mark.parametrize("tag", ["0.11-rc1", "0.11-preview", "1.0"])
    def test_aim_repo_valid_base_version_with_suffix(self, tag):
        config = BaseImageConfig(
            registry_host="ghcr.io",
            base_registry_namespace="silogen",
            base_repository="aim-base",
            base_tag=tag,
        )
        assert config.base_tag == tag

    @pytest.mark.parametrize("tag", ["BROKEN", "0.11.0", "0.11-rc0", "abc"])
    def test_aim_repo_invalid_base_version_rejected(self, tag):
        with pytest.raises(Exception, match="Invalid version format"):
            BaseImageConfig(
                registry_host="ghcr.io",
                base_registry_namespace="silogen",
                base_repository="aim-base",
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
