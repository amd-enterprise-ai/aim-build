# Copyright © Advanced Micro Devices, Inc., or its affiliates.
#
# SPDX-License-Identifier: MIT

import os
from unittest.mock import MagicMock, patch

import pytest
from click.testing import CliRunner

from aim_utils.config_utils import ConfigInitializer, Initializer, cli, get_canonical_name


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

            result = runner.invoke(cli, ["init", "--assets_path", str(temp_assets_dir), "--recreate", "True"])

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

        with (
            patch("aim_utils.config_utils.extract_model_info") as mock_extract,
            patch("aim_utils.config_utils.save_yaml") as mock_save,
        ):
            mock_extract.return_value = (None, None, True)

            initializer = ConfigInitializer(assets_path=str(temp_assets_dir))
            initializer.initialize(base_dir)

            mock_save.assert_called_once()
            call_args = mock_save.call_args
            config = call_args[0][0]
            assert "base_image" in config
            assert config["base_image"]["base_registry_namespace"] == "test-namespace"

    def test_initialize_model_config(self, temp_assets_dir, mock_file_readers):
        model_dir = temp_assets_dir / "org" / "model"
        model_dir.mkdir(parents=True)

        with (
            patch("aim_utils.config_utils.extract_model_info") as mock_extract,
            patch("aim_utils.config_utils.save_yaml") as mock_save,
        ):
            mock_extract.return_value = ("org", "model", False)

            initializer = ConfigInitializer(assets_path=str(temp_assets_dir))
            initializer.initialize(model_dir)

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

        with (
            patch("aim_utils.config_utils.extract_model_info") as mock_extract,
            patch("aim_utils.config_utils.save_yaml") as mock_save,
        ):
            mock_extract.return_value = ("org", "model", False)

            initializer = ConfigInitializer(assets_path=str(temp_assets_dir))
            initializer.initialize(model_dir)

            mock_save.assert_not_called()

    def test_initialize_recreates_existing_file(self, temp_assets_dir, mock_file_readers):
        model_dir = temp_assets_dir / "org" / "model"
        model_dir.mkdir(parents=True)
        config_file = model_dir / "config.yaml"
        config_file.write_text("existing: config")

        with (
            patch("aim_utils.config_utils.extract_model_info") as mock_extract,
            patch("aim_utils.config_utils.save_yaml") as mock_save,
        ):
            mock_extract.return_value = ("org", "model", False)

            initializer = ConfigInitializer(assets_path=str(temp_assets_dir), recreate=True)
            initializer.initialize(model_dir)

            mock_save.assert_called_once()

    def test_initialize_invalid_model_info(self, temp_assets_dir):
        model_dir = temp_assets_dir / "invalid"
        model_dir.mkdir()

        with (
            patch("aim_utils.config_utils.extract_model_info") as mock_extract,
            patch("aim_utils.config_utils.logger") as mock_logger,
        ):
            mock_extract.return_value = (None, None, False)

            initializer = ConfigInitializer(assets_path=str(temp_assets_dir))
            initializer.initialize(model_dir)

            mock_logger.warning.assert_called_once()

    def test_initialize_all_calls_initialize(self, temp_assets_dir):
        leaf_dir1 = temp_assets_dir / "org1" / "model1"
        leaf_dir2 = temp_assets_dir / "org2" / "model2"
        leaf_dir1.mkdir(parents=True)
        leaf_dir2.mkdir(parents=True)

        with (
            patch("aim_utils.file_utils.get_leaf_dirs") as mock_get_leaf,
            patch.object(ConfigInitializer, "initialize") as mock_init,
        ):
            mock_get_leaf.return_value = [leaf_dir1, leaf_dir2]

            initializer = ConfigInitializer(assets_path=str(temp_assets_dir))
            initializer.initialize_all()

            assert mock_init.call_count == 2
