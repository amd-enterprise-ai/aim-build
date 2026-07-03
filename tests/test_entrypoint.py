# Copyright © Advanced Micro Devices, Inc., or its affiliates.
#
# SPDX-License-Identifier: MIT

"""
Tests for entrypoint CLI functionality.
"""

import json
from unittest.mock import Mock, patch

import pytest
from click.testing import CliRunner

from aim_common import Engine, Precision
from aim_runtime.accelerator_detector import AcceleratorDetectionResult
from aim_runtime.config import AIMConfig
from aim_runtime.object_model import AcceleratorFamily, AcceleratorModel, AcceleratorType
from aim_utils.yaml_utils import load_yaml_string
from entrypoint import cli


@pytest.fixture
def mock_config():
    """Create a mock AIMConfig for testing."""
    return AIMConfig(
        aim_id="test-org/test-model",
        precision=Precision.FP16,
        accelerator_count=1,
        engine=Engine.VLLM,
        port=8000,
        log_level="INFO",
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

    def test_cli_executes_dry_run_command_json(self, mock_config, runner):
        """Test that CLI executes dry-run command in JSON format."""
        with patch("entrypoint.AIMConfig.from_environment", return_value=mock_config):
            with patch("entrypoint.configure_logging"):
                with patch("entrypoint.AIMRuntime") as mock_runtime_class:
                    mock_runtime = Mock()
                    mock_runtime.dry_run.return_value = [{"filename": "test.yaml", "profile": {"test": "data"}}]
                    mock_runtime_class.return_value = mock_runtime

                    runner.invoke(cli, ["dry-run", "--format", "json"])
                    mock_runtime_class.assert_called_once_with(mock_config)
                    mock_runtime.dry_run.assert_called_once_with()

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
                    mock_runtime.dry_run.return_value = [{"filename": "test.yaml", "profile": {"test": "data"}}]
                    mock_runtime_class.return_value = mock_runtime

                    runner.invoke(cli, ["dry-run", "--format", "json"])
                    mock_runtime_class.assert_called_once_with(mock_config)
                    mock_runtime.dry_run.assert_called_once_with()

    def test_dry_run_format_yaml_writes_only_yaml(self, mock_config, runner):
        """Test that dry-run with --format yaml prints only YAML to stdout."""
        with patch("entrypoint.AIMConfig.from_environment", return_value=mock_config):
            with patch("entrypoint.configure_logging"):
                with patch("entrypoint.AIMRuntime") as mock_runtime_class:
                    mock_runtime = Mock()
                    mock_runtime.dry_run.return_value = [{"aim_id": "test-model", "precision": "fp16"}]
                    mock_runtime_class.return_value = mock_runtime

                    result = runner.invoke(cli, ["dry-run", "--format", "yaml"])
                    assert result.exit_code == 0
                    assert load_yaml_string(result.output) == [{"aim_id": "test-model", "precision": "fp16"}]
                    mock_runtime.dry_run.assert_called_once()

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
                    mock_runtime.download_to_cache.assert_called_with(model_id="hf://org/model", use_hf_cache=False)

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
                    mock_runtime.download_to_cache.assert_called_once_with(model_id=None, use_hf_cache=False)

    def test_download_to_cache_with_custom_cache_dir(self, runner, tmp_path):
        """Test download-to-cache command with custom cache directory via AIM_CACHE_PATH env var."""
        custom_cache = "/custom/cache/path"

        # Create a custom config with the custom cache directory
        custom_config = AIMConfig(
            aim_id="test-model",
            precision=Precision.FP16,
            accelerator_count=1,
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
                    mock_runtime.download_to_cache.assert_called_once_with(model_id=None, use_hf_cache=False)

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
                    mock_runtime.download_to_cache.assert_called_once_with(model_id=None, use_hf_cache=False)

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
                    mock_runtime.download_to_cache.assert_called_once_with(model_id=None, use_hf_cache=False)

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
                    mock_runtime.download_to_cache.assert_called_once_with(model_id=None, use_hf_cache=False)

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
                    mock_runtime.download_to_cache.assert_called_once_with(model_id=None, use_hf_cache=True)


class TestBenchmarkCommand:
    """Test suite for benchmark command (legacy AIMBenchmark path)."""

    @pytest.fixture(autouse=True)
    def _no_custom_harness(self):
        """Force the legacy AIMBenchmark path for this suite.

        The benchmark command delegates to a custom ModelHarness when one
        exists at /workspace/model/src/harness.py. These tests target the
        legacy path, so pin discovery to "no custom harness" regardless of
        what is actually on disk (e.g. when running inside a specialized image).
        """
        with patch("aim_runtime.harness.discovery.has_custom_harness", return_value=False):
            yield

    def _make_benchmark_mock(self, overall_success=True):
        mock_runner = Mock()
        mock_runner.run_benchmark_suite.return_value = {"overall_success": overall_success}
        return mock_runner

    def test_benchmark_with_service_url(self, runner, tmp_path):
        """When --service-url is provided, no server is spawned."""
        mock_runner = self._make_benchmark_mock()

        with (
            patch("entrypoint.configure_logging"),
            patch("entrypoint._start_server_in_background") as mock_start,
            patch("entrypoint._wait_for_service") as mock_wait,
            patch("aim_runtime.benchmarking.AIMBenchmark", return_value=mock_runner) as mock_cls,
        ):

            result = runner.invoke(
                cli,
                [
                    "benchmark",
                    "--service-url",
                    "http://localhost:8000",
                    "--output-dir",
                    str(tmp_path),
                ],
            )

            assert result.exit_code == 0
            mock_start.assert_not_called()
            mock_wait.assert_not_called()
            mock_cls.assert_called_once_with(
                service_url="http://localhost:8000",
                timeout_seconds=30,
                config_file=None,
            )
            mock_runner.run_benchmark_suite.assert_called_once()
            mock_runner.export_results.assert_called_once()

    def test_benchmark_without_service_url_spawns_server(self, mock_config, runner, tmp_path):
        """When no --service-url is given, a server is started and cleaned up."""
        mock_runner = self._make_benchmark_mock()
        mock_process = Mock()
        mock_process.poll.return_value = None

        with (
            patch("entrypoint.configure_logging"),
            patch("entrypoint.AIMConfig.from_environment", return_value=mock_config),
            patch("entrypoint._start_server_in_background", return_value=mock_process) as mock_start,
            patch("entrypoint._wait_for_service") as mock_wait,
            patch("aim_runtime.benchmarking.AIMBenchmark", return_value=mock_runner) as mock_cls,
        ):

            result = runner.invoke(
                cli,
                [
                    "benchmark",
                    "--output-dir",
                    str(tmp_path),
                ],
            )

            assert result.exit_code == 0
            mock_start.assert_called_once_with(mock_config)
            mock_wait.assert_called_once_with(
                f"http://localhost:{mock_config.port}",
                120,
            )
            mock_cls.assert_called_once_with(
                service_url=f"http://localhost:{mock_config.port}",
                timeout_seconds=30,
                config_file=None,
            )
            mock_process.send_signal.assert_called_once()
            mock_process.wait.assert_called_once()

    def test_benchmark_failed_results_exit_code_1(self, runner, tmp_path):
        """Exit code is 1 when benchmarks report overall_success=False."""
        mock_runner = self._make_benchmark_mock(overall_success=False)

        with (
            patch("entrypoint.configure_logging"),
            patch("aim_runtime.benchmarking.AIMBenchmark", return_value=mock_runner),
        ):

            result = runner.invoke(
                cli,
                [
                    "benchmark",
                    "--service-url",
                    "http://localhost:8000",
                    "--output-dir",
                    str(tmp_path),
                ],
            )

            assert result.exit_code == 1

    def test_benchmark_exception_exits_1_and_cleans_up(self, mock_config, runner, tmp_path):
        """An exception during benchmarking still cleans up the server process."""
        mock_process = Mock()
        mock_process.poll.return_value = None

        with (
            patch("entrypoint.configure_logging"),
            patch("entrypoint.AIMConfig.from_environment", return_value=mock_config),
            patch("entrypoint._start_server_in_background", return_value=mock_process),
            patch("entrypoint._wait_for_service"),
            patch("aim_runtime.benchmarking.AIMBenchmark", side_effect=RuntimeError("boom")),
        ):

            result = runner.invoke(
                cli,
                [
                    "benchmark",
                    "--output-dir",
                    str(tmp_path),
                ],
            )

            assert result.exit_code == 1
            mock_process.send_signal.assert_called_once()

    def test_benchmark_custom_options_forwarded(self, runner, tmp_path):
        """CLI options are forwarded to AIMBenchmark correctly."""
        mock_runner = self._make_benchmark_mock()
        config_path = str(tmp_path / "custom.yaml")

        with (
            patch("entrypoint.configure_logging"),
            patch("aim_runtime.benchmarking.AIMBenchmark", return_value=mock_runner) as mock_cls,
        ):

            result = runner.invoke(
                cli,
                [
                    "benchmark",
                    "--service-url",
                    "http://host:9090",
                    "--timeout-seconds",
                    "60",
                    "--config",
                    config_path,
                    "--output-dir",
                    str(tmp_path),
                ],
            )

            assert result.exit_code == 0
            mock_cls.assert_called_once_with(
                service_url="http://host:9090",
                timeout_seconds=60,
                config_file=config_path,
            )


class TestListProfilesCommand:
    """Test suite for list-profiles command with JSON/YAML output."""

    def _make_mock_selector(self):
        """Create a mock ProfileSelector with serialize methods."""
        mock_selector = Mock()
        sample_data = [
            {
                "profile_id": "vllm-mi300x-fp16-tp1-latency",
                "compatibility": "compatible",
                "profile": {
                    "aim_id": "test-org/test-model",
                    "model_id": "test-org/test-model",
                    "metadata": {"engine": "vllm", "gpu": "MI300X", "precision": "fp16"},
                    "engine_args": {"dtype": "float16"},
                    "env_vars": {},
                },
            }
        ]
        mock_selector.serialize_profiles.return_value = sample_data
        mock_selector.serialize_all_profiles.return_value = sample_data
        mock_selector.get_categorized_profiles.return_value = {}
        mock_selector.format_table_report.return_value = "table output"
        mock_selector.format_all_profiles_report.return_value = "all profiles output"
        return mock_selector

    def test_list_profiles_json_output(self, mock_config, runner):
        """Test that list-profiles --format json returns parseable JSON with correct keys."""
        with patch("entrypoint.AIMConfig.from_environment", return_value=mock_config):
            with patch("entrypoint.configure_logging"):
                with patch("entrypoint.ProfileSelector") as mock_selector_class:
                    mock_selector_class.return_value = self._make_mock_selector()

                    result = runner.invoke(cli, ["list-profiles", "--format", "json"])
                    assert result.exit_code == 0

                    parsed = json.loads(result.output)
                    assert isinstance(parsed, list)
                    assert len(parsed) == 1
                    assert set(parsed[0].keys()) == {"profile_id", "compatibility", "profile"}

    def test_list_profiles_yaml_output(self, mock_config, runner):
        """Test that list-profiles --format yaml returns parseable YAML with correct keys."""
        with patch("entrypoint.AIMConfig.from_environment", return_value=mock_config):
            with patch("entrypoint.configure_logging"):
                with patch("entrypoint.ProfileSelector") as mock_selector_class:
                    mock_selector_class.return_value = self._make_mock_selector()

                    result = runner.invoke(cli, ["list-profiles", "--format", "yaml"])
                    assert result.exit_code == 0

                    parsed = load_yaml_string(result.output)
                    assert isinstance(parsed, list)
                    assert len(parsed) == 1
                    assert set(parsed[0].keys()) == {"profile_id", "compatibility", "profile"}

    def test_list_profiles_table_output_unchanged(self, mock_config, runner):
        """Test that list-profiles --format table still uses the existing table formatter."""
        with patch("entrypoint.AIMConfig.from_environment", return_value=mock_config):
            with patch("entrypoint.configure_logging"):
                with patch("entrypoint.ProfileSelector") as mock_selector_class:
                    mock_selector = self._make_mock_selector()
                    mock_selector_class.return_value = mock_selector

                    result = runner.invoke(cli, ["list-profiles", "--format", "table"])
                    assert result.exit_code == 0
                    mock_selector.format_table_report.assert_called_once()

    def test_list_profiles_skip_compatibility_json(self, mock_config, runner):
        """Test that --skip-compatibility-check with --format json uses serialize_all_profiles."""
        with patch("entrypoint.AIMConfig.from_environment", return_value=mock_config):
            with patch("entrypoint.configure_logging"):
                with patch("entrypoint.ProfileSelector") as mock_selector_class:
                    mock_selector = self._make_mock_selector()
                    mock_selector_class.return_value = mock_selector

                    result = runner.invoke(cli, ["list-profiles", "--skip-compatibility-check", "--format", "json"])
                    assert result.exit_code == 0
                    mock_selector.serialize_all_profiles.assert_called_once()

                    parsed = json.loads(result.output)
                    assert isinstance(parsed, list)


class TestDetectHardwareCommand:
    """Test suite for detect-hardware command."""

    def _make_gpu_result(self):
        return AcceleratorDetectionResult(
            accelerator_type=AcceleratorType.GPU,
            accelerator_model=AcceleratorModel.MI300X,
            accelerator_count=8,
        )

    def _make_cpu_result(self):
        return AcceleratorDetectionResult(
            accelerator_type=AcceleratorType.CPU,
            accelerator_model=AcceleratorModel.EPYC_9965,
            accelerator_count=192,
        )

    def test_detect_hardware_json_default(self, runner):
        """Test detect-hardware with default --type all returns combined JSON list."""
        gpu_result = self._make_gpu_result()
        cpu_result = self._make_cpu_result()

        with (
            patch("entrypoint.configure_logging"),
            patch("entrypoint.AcceleratorDetector") as mock_cls,
        ):
            mock_detector = Mock()
            mock_detector.detect.side_effect = [gpu_result, cpu_result]
            mock_cls.return_value = mock_detector

            result = runner.invoke(cli, ["detect-hardware"])

            assert result.exit_code == 0
            parsed = json.loads(result.output)
            assert isinstance(parsed, list)
            assert len(parsed) == 2
            assert parsed[0] == {"accelerator_type": "GPU", "accelerator_model": "MI300X", "accelerator_count": 8}
            assert parsed[1] == {
                "accelerator_type": "CPU",
                "accelerator_model": "EPYC_9965",
                "accelerator_count": 192,
            }

    def test_detect_hardware_gpu_only(self, runner):
        """Test detect-hardware --type gpu runs only GPU detection."""
        gpu_result = self._make_gpu_result()

        with (
            patch("entrypoint.configure_logging"),
            patch("entrypoint.AcceleratorDetector") as mock_cls,
        ):
            mock_detector = Mock()
            mock_detector.detect.return_value = gpu_result
            mock_cls.return_value = mock_detector

            result = runner.invoke(cli, ["detect-hardware", "--type", "gpu"])

            assert result.exit_code == 0
            mock_detector.detect.assert_called_once_with(accelerator_type=AcceleratorType.GPU)
            parsed = json.loads(result.output)
            assert len(parsed) == 1
            assert parsed[0]["accelerator_model"] == "MI300X"

    def test_detect_hardware_cpu_only(self, runner):
        """Test detect-hardware --type cpu runs only CPU detection."""
        cpu_result = self._make_cpu_result()

        with (
            patch("entrypoint.configure_logging"),
            patch("entrypoint.AcceleratorDetector") as mock_cls,
        ):
            mock_detector = Mock()
            mock_detector.detect.return_value = cpu_result
            mock_cls.return_value = mock_detector

            result = runner.invoke(cli, ["detect-hardware", "--type", "cpu"])

            assert result.exit_code == 0
            mock_detector.detect.assert_called_once_with(
                accelerator_type=AcceleratorType.CPU,
                accelerator_family=AcceleratorFamily.EPYC,
            )
            parsed = json.loads(result.output)
            assert len(parsed) == 1
            assert parsed[0]["accelerator_model"] == "EPYC_9965"

    def test_detect_hardware_cpu_falls_back_to_generic_family(self, runner):
        """CPU detect-hardware falls back to generic CPU family when EPYC is generic."""
        epyc_generic_result = AcceleratorDetectionResult(
            accelerator_type=AcceleratorType.CPU,
            accelerator_model=AcceleratorModel.CPU,
            accelerator_count=1,
        )
        cpu_result = AcceleratorDetectionResult(
            accelerator_type=AcceleratorType.CPU,
            accelerator_model=AcceleratorModel.CPU,
            accelerator_count=192,
        )

        with (
            patch("entrypoint.configure_logging"),
            patch("entrypoint.AcceleratorDetector") as mock_cls,
        ):
            mock_detector = Mock()
            mock_detector.detect.side_effect = [epyc_generic_result, cpu_result]
            mock_cls.return_value = mock_detector

            result = runner.invoke(cli, ["detect-hardware", "--type", "cpu"])

            assert result.exit_code == 0
            assert mock_detector.detect.call_count == 2
            mock_detector.detect.assert_any_call(
                accelerator_type=AcceleratorType.CPU, accelerator_family=AcceleratorFamily.EPYC
            )
            mock_detector.detect.assert_any_call(
                accelerator_type=AcceleratorType.CPU, accelerator_family=AcceleratorFamily.CPU
            )

    def test_detect_hardware_yaml_output(self, runner):
        """Test detect-hardware --format yaml."""
        gpu_result = self._make_gpu_result()

        with (
            patch("entrypoint.configure_logging"),
            patch("entrypoint.AcceleratorDetector") as mock_cls,
        ):
            mock_detector = Mock()
            mock_detector.detect.return_value = gpu_result
            mock_cls.return_value = mock_detector

            result = runner.invoke(cli, ["detect-hardware", "--type", "gpu", "--format", "yaml"])

            assert result.exit_code == 0
            parsed = load_yaml_string(result.output)
            assert isinstance(parsed, list)
            assert parsed[0]["accelerator_model"] == "MI300X"

    def test_detect_hardware_verbose(self, runner):
        """Test detect-hardware --verbose returns detail dicts."""
        gpu_result = self._make_gpu_result()

        with (
            patch("entrypoint.configure_logging"),
            patch("entrypoint.AcceleratorDetector") as mock_cls,
        ):
            mock_detector = Mock()
            mock_detector.detect.return_value = gpu_result
            mock_cls.return_value = mock_detector

            result = runner.invoke(cli, ["detect-hardware", "--type", "gpu", "--verbose"])

            assert result.exit_code == 0
            parsed = json.loads(result.output)
            assert isinstance(parsed, list)
            assert len(parsed) == 1
            # Verbose mode uses to_detail_dict which keeps lowercase accelerator_type
            assert parsed[0]["accelerator_type"] == "gpu"
            assert parsed[0]["accelerator_model"] == "MI300X"

    def test_detect_hardware_handles_error(self, runner):
        """Test detect-hardware exits with code 1 on error."""
        with (
            patch("entrypoint.configure_logging"),
            patch("entrypoint.AcceleratorDetector", side_effect=RuntimeError("detection failed")),
        ):
            result = runner.invoke(cli, ["detect-hardware"])
            assert result.exit_code == 1

    def test_detect_hardware_no_model_detected(self, runner):
        """Test detect-hardware when no hardware is detected returns empty list."""
        empty_result = AcceleratorDetectionResult(
            accelerator_type=AcceleratorType.GPU,
            accelerator_model=None,
            accelerator_count=0,
        )

        with (
            patch("entrypoint.configure_logging"),
            patch("entrypoint.AcceleratorDetector") as mock_cls,
        ):
            mock_detector = Mock()
            mock_detector.detect.return_value = empty_result
            mock_cls.return_value = mock_detector

            result = runner.invoke(cli, ["detect-hardware", "--type", "gpu"])

            assert result.exit_code == 0
            parsed = json.loads(result.output)
            assert parsed == []
