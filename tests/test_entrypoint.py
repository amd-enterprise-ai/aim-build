# Copyright © Advanced Micro Devices, Inc., or its affiliates.
#
# SPDX-License-Identifier: MIT

"""
Tests for entrypoint CLI functionality.
"""

from unittest.mock import Mock, patch

import pytest
from click.testing import CliRunner

from aim_runtime.config import AIMConfig
from aim_runtime.object_model import Engine, Precision
from entrypoint import cli


@pytest.fixture
def mock_config():
    """Create a mock AIMConfig for testing."""
    return AIMConfig(
        aim_id="test-org/test-model",
        precision=Precision.FP16,
        gpu_count=1,
        engine=Engine.VLLM,
        port=8000,
        log_level="INFO",
        custom_model_name=None,
    )


@pytest.fixture
def runner():
    """Create a Click CLI test runner."""
    return CliRunner()


class TestEntrypointCLI:
    """Test suite for entrypoint CLI functionality."""

    def test_cli_defaults_to_serve_when_no_command(self, mock_config, runner):
        """Test that CLI defaults to serve command when no subcommand is specified."""
        with patch("entrypoint.AIMConfig.from_environment", return_value=mock_config):
            with patch("entrypoint.configure_logging"):
                with patch("entrypoint.AIMRuntime") as mock_runtime_class:
                    mock_runtime = Mock()
                    mock_runtime_class.return_value = mock_runtime

                    runner.invoke(cli, [])
                    mock_runtime_class.assert_called_once_with(mock_config)
                    mock_runtime.serve.assert_called_once()

    def test_cli_executes_serve_command(self, mock_config, runner):
        """Test that CLI executes serve command when explicitly specified."""
        with patch("entrypoint.AIMConfig.from_environment", return_value=mock_config):
            with patch("entrypoint.configure_logging"):
                with patch("entrypoint.AIMRuntime") as mock_runtime_class:
                    mock_runtime = Mock()
                    mock_runtime_class.return_value = mock_runtime

                    runner.invoke(cli, ["serve"])
                    mock_runtime_class.assert_called_once_with(mock_config)
                    mock_runtime.serve.assert_called_once()

    def test_cli_executes_dry_run_command_yaml(self, mock_config, runner):
        """Test that CLI executes dry-run command in YAML format (default)."""
        with patch("entrypoint.AIMConfig.from_environment", return_value=mock_config):
            with patch("entrypoint.configure_logging"):
                with patch("entrypoint.AIMRuntime") as mock_runtime_class:
                    mock_runtime = Mock()
                    mock_runtime_class.return_value = mock_runtime

                    runner.invoke(cli, ["dry-run"])
                    mock_runtime_class.assert_called_once_with(mock_config)
                    mock_runtime.dry_run.assert_called_once()
                    mock_runtime.dry_run_json.assert_not_called()

    def test_cli_executes_dry_run_command_json(self, mock_config, runner):
        """Test that CLI executes dry-run command in JSON format."""
        with patch("entrypoint.AIMConfig.from_environment", return_value=mock_config):
            with patch("entrypoint.configure_logging"):
                with patch("entrypoint.AIMRuntime") as mock_runtime_class:
                    mock_runtime = Mock()
                    mock_runtime.dry_run_json.return_value = [{"filename": "test.yaml", "profile": {"test": "data"}}]
                    mock_runtime_class.return_value = mock_runtime

                    runner.invoke(cli, ["dry-run", "--format", "json"])
                    mock_runtime_class.assert_called_once_with(mock_config)
                    mock_runtime.dry_run_json.assert_called_once()
                    mock_runtime.dry_run.assert_not_called()

    def test_cli_handles_configuration_error(self, runner):
        """Test that CLI handles configuration errors gracefully."""
        with patch("entrypoint.AIMConfig.from_environment", side_effect=ValueError("Missing required config")):
            with patch("entrypoint.configure_logging"):
                result = runner.invoke(cli, ["serve"])
                assert result.exit_code == 1

    def test_cli_handles_file_not_found_error(self, mock_config, runner):
        """Test that CLI handles FileNotFoundError gracefully."""
        with patch("entrypoint.AIMConfig.from_environment", return_value=mock_config):
            with patch("entrypoint.configure_logging"):
                with patch("entrypoint.AIMRuntime", side_effect=FileNotFoundError("Profile not found")):
                    result = runner.invoke(cli, ["serve"])
                    assert result.exit_code == 1

    def test_cli_handles_unexpected_error(self, mock_config, runner):
        """Test that CLI handles unexpected errors gracefully."""
        with patch("entrypoint.AIMConfig.from_environment", return_value=mock_config):
            with patch("entrypoint.configure_logging"):
                with patch("entrypoint.AIMRuntime", side_effect=RuntimeError("Unexpected error")):
                    result = runner.invoke(cli, ["serve"])
                    assert result.exit_code == 1


class TestServeCommand:
    """Test suite for serve command."""

    def test_serve_command_integration(self, mock_config, runner):
        """Test serve command creates AIMRuntime with correct config and calls serve."""
        with patch("entrypoint.AIMConfig.from_environment", return_value=mock_config):
            with patch("entrypoint.configure_logging") as mock_logging:
                with patch("entrypoint.AIMRuntime") as mock_runtime_class:
                    mock_runtime = Mock()
                    mock_runtime_class.return_value = mock_runtime

                    result = runner.invoke(cli, ["serve"])

                    # Verify the workflow
                    mock_runtime_class.assert_called_once_with(mock_config)
                    mock_logging.assert_called_once_with(
                        root_log_level=mock_config.log_level_root, aim_log_level=mock_config.log_level
                    )
                    mock_runtime.serve.assert_called_once()
                    assert result.exit_code == 0

    def test_serve_command_handles_value_error(self, runner):
        """Test serve command handles ValueError from config loading."""
        with patch("entrypoint.AIMConfig.from_environment", side_effect=ValueError("Invalid config")):
            with patch("entrypoint.configure_logging"):
                result = runner.invoke(cli, ["serve"])
                assert result.exit_code == 1


class TestDryRunCommand:
    """Test suite for dry-run command."""

    def test_dry_run_yaml_creates_runtime_and_calls_dry_run(self, mock_config, runner):
        """Test that dry-run command with YAML format creates AIMRuntime and calls dry_run method."""
        with patch("entrypoint.AIMConfig.from_environment", return_value=mock_config):
            with patch("entrypoint.configure_logging"):
                with patch("entrypoint.AIMRuntime") as mock_runtime_class:
                    mock_runtime = Mock()
                    mock_runtime_class.return_value = mock_runtime

                    runner.invoke(cli, ["dry-run"])
                    mock_runtime_class.assert_called_once_with(mock_config)
                    mock_runtime.dry_run.assert_called_once()

    def test_dry_run_json_creates_runtime_and_calls_dry_run_json(self, mock_config, runner):
        """Test that dry-run command with JSON format creates AIMRuntime and calls dry_run_json method."""
        with patch("entrypoint.AIMConfig.from_environment", return_value=mock_config):
            with patch("entrypoint.configure_logging"):
                with patch("entrypoint.AIMRuntime") as mock_runtime_class:
                    mock_runtime = Mock()
                    mock_runtime.dry_run_json.return_value = [{"filename": "test.yaml", "profile": {"test": "data"}}]
                    mock_runtime_class.return_value = mock_runtime

                    runner.invoke(cli, ["dry-run", "--format", "json"])
                    mock_runtime_class.assert_called_once_with(mock_config)
                    mock_runtime.dry_run_json.assert_called_once()

    def test_dry_run_does_not_execute_script(self, mock_config, runner):
        """Test that dry-run command does not execute any script."""
        with patch("entrypoint.AIMConfig.from_environment", return_value=mock_config):
            with patch("entrypoint.configure_logging"):
                with patch("entrypoint.AIMRuntime") as mock_runtime_class:
                    mock_runtime = Mock()
                    mock_runtime_class.return_value = mock_runtime

                    with patch("os.execv") as mock_execv:
                        runner.invoke(cli, ["dry-run"])
                        mock_execv.assert_not_called()


class TestDownloadToCacheCommand:
    def test_download_to_cache_with_explicit_model_id(self, mock_config, runner, tmp_path):
        """Test download-to-cache command with explicit --model-id argument (protocol override)."""
        # No profile file needed, but create a dummy for completeness
        profile_path = tmp_path / "test_profile.yaml"
        profile_path.write_text("model: meta-llama/Llama-3.1-8B-Instruct\n")

        with patch("entrypoint.AIMConfig.from_environment", return_value=mock_config):
            with patch("entrypoint.configure_logging"):
                with patch("entrypoint.AIMRuntime") as mock_runtime_class:
                    mock_runtime = Mock()
                    mock_runtime.download_to_cache.return_value = "/workspace/model-cache/custom-model"
                    mock_runtime_class.return_value = mock_runtime

                    # Test HuggingFace protocol
                    result = runner.invoke(cli, ["download-to-cache", "--model-id", "hf://org/model"])
                    assert result.exit_code == 0
                    mock_runtime.download_to_cache.assert_called_with(
                        model_id="hf://org/model", use_hf_cache=False, custom_model_name=None
                    )

                    # Test S3 protocol
                    mock_runtime.download_to_cache.reset_mock()
                    result = runner.invoke(cli, ["download-to-cache", "--model-id", "s3://bucket/path/to/model"])
                    assert result.exit_code == 0
                    mock_runtime.download_to_cache.assert_called_with(
                        model_id="s3://bucket/path/to/model", use_hf_cache=False, custom_model_name=None
                    )

    """Test suite for download-to-cache command."""

    def test_download_to_cache_with_default_cache_path(self, mock_config, runner, tmp_path):
        """Test download-to-cache command with default cache path."""
        with patch("entrypoint.AIMConfig.from_environment", return_value=mock_config):
            with patch("entrypoint.configure_logging"):
                with patch("entrypoint.AIMRuntime") as mock_runtime_class:
                    mock_runtime = Mock()
                    mock_runtime.download_to_cache.return_value = (
                        "/workspace/model-cache/models--meta-llama--Llama-3.1-8B-Instruct"
                    )
                    mock_runtime_class.return_value = mock_runtime

                    result = runner.invoke(cli, ["download-to-cache"])
                    assert result.exit_code == 0
                    mock_runtime.download_to_cache.assert_called_once_with(
                        model_id=None, use_hf_cache=False, custom_model_name=None
                    )

    def test_download_to_cache_with_custom_cache_dir(self, runner, tmp_path):
        """Test download-to-cache command with custom cache directory via AIM_CACHE_PATH env var."""
        custom_cache = "/custom/cache/path"

        # Create a custom config with the custom cache directory
        custom_config = AIMConfig(
            aim_id="test-model",
            precision=Precision.FP16,
            gpu_count=1,
            cache_path=custom_cache,  # Custom cache path
        )

        with patch("entrypoint.AIMConfig.from_environment", return_value=custom_config):
            with patch("entrypoint.configure_logging"):
                with patch("entrypoint.AIMRuntime") as mock_runtime_class:
                    mock_runtime = Mock()
                    mock_runtime.download_to_cache.return_value = (
                        f"{custom_cache}/models--meta-llama--Llama-3.1-8B-Instruct"
                    )
                    mock_runtime_class.return_value = mock_runtime

                    result = runner.invoke(cli, ["download-to-cache"])
                    assert result.exit_code == 0
                    mock_runtime.download_to_cache.assert_called_once_with(
                        model_id=None, use_hf_cache=False, custom_model_name=None
                    )

    def test_download_to_cache_with_quantized_model(self, mock_config, runner, tmp_path):
        """
        Test download-to-cache command for quantized models.
        This validates that the CLI calls download_to_cache correctly.
        """
        with patch("entrypoint.AIMConfig.from_environment", return_value=mock_config):
            with patch("entrypoint.configure_logging"):
                with patch("entrypoint.AIMRuntime") as mock_runtime_class:
                    mock_runtime = Mock()
                    mock_runtime.download_to_cache.return_value = (
                        "/workspace/model-cache/models--meta-llama--Llama-3.1-8B-Instruct-FP8-KV"
                    )
                    mock_runtime_class.return_value = mock_runtime

                    result = runner.invoke(cli, ["download-to-cache"])
                    assert result.exit_code == 0
                    mock_runtime.download_to_cache.assert_called_once_with(
                        model_id=None, use_hf_cache=False, custom_model_name=None
                    )

    def test_download_to_cache_no_model_in_profile(self, mock_config, runner, tmp_path):
        """Test download-to-cache command with profile missing model field."""
        with patch("entrypoint.AIMConfig.from_environment", return_value=mock_config):
            with patch("entrypoint.configure_logging"):
                with patch("entrypoint.AIMRuntime") as mock_runtime_class:
                    mock_runtime = Mock()
                    # Simulate error when download_to_cache is called without model
                    mock_runtime.download_to_cache.side_effect = ValueError(
                        "No model_id specified and profile missing model field"
                    )
                    mock_runtime_class.return_value = mock_runtime

                    result = runner.invoke(cli, ["download-to-cache"])
                    assert result.exit_code == 1
                    mock_runtime.download_to_cache.assert_called_once_with(
                        model_id=None, use_hf_cache=False, custom_model_name=None
                    )

    def test_download_to_cache_handles_download_error(self, mock_config, runner, tmp_path):
        """Test download-to-cache command handles download errors gracefully."""
        with patch("entrypoint.AIMConfig.from_environment", return_value=mock_config):
            with patch("entrypoint.configure_logging"):
                with patch("entrypoint.AIMRuntime") as mock_runtime_class:
                    mock_runtime = Mock()
                    mock_runtime.download_to_cache.side_effect = RuntimeError("Network error")
                    mock_runtime_class.return_value = mock_runtime

                    result = runner.invoke(cli, ["download-to-cache"])
                    assert result.exit_code == 1
                    mock_runtime.download_to_cache.assert_called_once_with(
                        model_id=None, use_hf_cache=False, custom_model_name=None
                    )

    def test_download_to_cache_with_use_hf_cache_flag(self, mock_config, runner, tmp_path):
        """Test download-to-cache command with --use-hf-cache flag."""
        with patch("entrypoint.AIMConfig.from_environment", return_value=mock_config):
            with patch("entrypoint.configure_logging"):
                with patch("entrypoint.AIMRuntime") as mock_runtime_class:
                    mock_runtime = Mock()
                    mock_runtime.download_to_cache.return_value = (
                        "/workspace/model-cache/models--meta-llama--Llama-3.1-8B-Instruct"
                    )
                    mock_runtime_class.return_value = mock_runtime

                    # Test with --use-hf-cache flag
                    result = runner.invoke(cli, ["download-to-cache", "--use-hf-cache"])
                    assert result.exit_code == 0
                    # Verify the download was called with use_hf_cache=True
                    mock_runtime.download_to_cache.assert_called_once_with(
                        model_id=None, use_hf_cache=True, custom_model_name=None
                    )

    def test_download_to_cache_with_explicit_custom_model_name(self, mock_config, runner, tmp_path):
        """Test download-to-cache command with explicit --custom-model-name flag for S3 URIs."""
        with patch("entrypoint.AIMConfig.from_environment", return_value=mock_config):
            with patch("entrypoint.configure_logging"):
                with patch("entrypoint.AIMRuntime") as mock_runtime_class:
                    mock_runtime = Mock()
                    mock_runtime.download_to_cache.return_value = "/workspace/model-cache/custom-model-name"
                    mock_runtime_class.return_value = mock_runtime

                    # Test with --custom-model-name flag for S3 URI
                    result = runner.invoke(cli, ["download-to-cache", "--custom-model-name", "custom-model-name"])
                    assert result.exit_code == 0
                    # Verify the download was called with custom_model_name parameter
                    mock_runtime.download_to_cache.assert_called_once_with(
                        model_id=None, use_hf_cache=False, custom_model_name="custom-model-name"
                    )
