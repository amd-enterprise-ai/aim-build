# Copyright © Advanced Micro Devices, Inc., or its affiliates.
#
# SPDX-License-Identifier: MIT

"""
Tests for AIMRuntime class including dry-run functionality.
"""

from unittest.mock import Mock, patch

import pytest

from aim_runtime.aim_runtime import AIMRuntime
from aim_runtime.config import AIMConfig
from aim_runtime.object_model import Engine, Precision, Profile, ProfileHandling, ProfileMetadata
from aim_runtime.profile_registry import ProfileRegistry


class TestNormalizeModelSource:
    """Test suite for normalize_model_source static method."""

    def test_normalize_huggingface_model_without_protocol(self):
        """Test that HuggingFace model IDs get hf:// prefix added."""
        assert (
            AIMRuntime.normalize_model_source("meta-llama/Llama-3.1-8B-Instruct")
            == "hf://meta-llama/Llama-3.1-8B-Instruct"
        )
        assert AIMRuntime.normalize_model_source("mistralai/Mistral-7B-v0.1") == "hf://mistralai/Mistral-7B-v0.1"

    def test_normalize_s3_uri_unchanged(self):
        """Test that S3 URIs are returned unchanged."""
        s3_uri = "s3://my-bucket/path/to/model"
        assert AIMRuntime.normalize_model_source(s3_uri) == s3_uri

    def test_normalize_hf_uri_unchanged(self):
        """Test that hf:// URIs are returned unchanged."""
        hf_uri = "hf://org/model"
        assert AIMRuntime.normalize_model_source(hf_uri) == hf_uri

    def test_normalize_complex_s3_paths(self):
        """Test that complex S3 paths are preserved."""
        complex_uri = "s3://bucket/models/org/llama-3.1-8b/checkpoint-1000"
        assert AIMRuntime.normalize_model_source(complex_uri) == complex_uri


@pytest.fixture
def mock_config():
    """Create a mock AIMConfig for testing."""
    return AIMConfig(
        aim_id="test/aim",
        precision=Precision.FP16,
        gpu_count=1,
        engine=Engine.VLLM,
        port=8000,
        log_level="DEBUG",
    )


@pytest.fixture
def mock_profile():
    """Create a mock Profile for testing."""
    profile = Mock(spec=Profile)
    profile.profile_handling = Mock(spec=ProfileHandling)
    profile.profile_handling.filename = "test_profile.yaml"
    profile.profile_handling.path = "/path/to/test_profile.yaml"
    profile.aim_id = "test-aim-id"
    profile.model_id = None  # Model ID from profile (optional)
    profile.precision = Precision.FP16
    profile.gpu_count = 1
    profile.metadata = Mock(spec=ProfileMetadata)
    profile.metadata.engine = Engine.VLLM
    profile.env_vars = {}
    profile.engine_args = {}
    return profile


@pytest.fixture
def test_script_content():
    """Simple test script content for bash scripts."""
    return "#!/bin/bash\necho 'test'"


@pytest.fixture
def script_file_factory(tmp_path, test_script_content):
    """Factory fixture to create script files."""

    def _create_script_file(content=None, filename="test_script.sh"):
        script_file = tmp_path / filename
        script_file.write_text(content or test_script_content)
        return str(script_file)

    return _create_script_file


class TestAIMRuntimeDryRun:
    """Test suite for AIMRuntime dry-run functionality."""

    def test_dry_run_displays_yaml_content(self, mock_config, model_profile, script_file_factory):
        """Test that dry_run returns the profile YAML content."""
        script_path = script_file_factory(content="#!/bin/bash\necho 'test script'")

        with patch("aim_runtime.aim_runtime.ProfileSelector") as mock_ps:
            with patch("aim_runtime.aim_runtime.CommandGenerator") as mock_cg:
                runtime = AIMRuntime(mock_config)
                runtime.profile_selector = mock_ps.return_value
                runtime.command_generator = mock_cg.return_value
                runtime.profile_selector.find_profile.return_value = model_profile
                runtime.command_generator.generate_command_script.return_value = script_path

                result = runtime.dry_run()

        # Check for profile path (format-agnostic)
        assert model_profile.profile_handling.path in result
        # Check for YAML content (format-agnostic)
        assert "aim_id:" in result
        assert "meta-llama/Llama-3.1-8B-Instruct" in result
        assert "precision: fp16" in result or 'precision: "fp16"' in result
        assert "gpu_count: 1" in result
        assert "engine: vllm" in result or 'engine: "vllm"' in result
        # Check for generated script
        assert "#!/bin/bash" in result
        assert "echo 'test script'" in result

    def test_dry_run_with_complex_yaml(self, mock_config, complex_profile, script_file_factory):
        """Test that dry_run returns complex YAML content correctly."""
        script_path = script_file_factory(content="#!/bin/bash\necho 'complex test'")

        with patch("aim_runtime.aim_runtime.ProfileSelector") as mock_ps:
            with patch("aim_runtime.aim_runtime.CommandGenerator") as mock_cg:
                runtime = AIMRuntime(mock_config)
                runtime.profile_selector = mock_ps.return_value
                runtime.command_generator = mock_cg.return_value
                runtime.profile_selector.find_profile.return_value = complex_profile
                runtime.command_generator.generate_command_script.return_value = script_path

                result = runtime.dry_run()

        # Check for profile path (format-agnostic)
        assert complex_profile.profile_handling.path in result
        # Complex profile has comprehensive test data
        assert "aim_id:" in result or "model_id:" in result
        assert "metadata:" in result
        assert "engine: vllm" in result or 'engine: "vllm"' in result
        # Check for generated script
        assert "#!/bin/bash" in result

    def test_dry_run_includes_profile_path(self, mock_config, model_profile, script_file_factory):
        """Test that dry_run includes profile path."""
        script_path = script_file_factory(content="#!/bin/bash\necho 'path test'")

        with patch("aim_runtime.aim_runtime.ProfileSelector") as mock_ps:
            with patch("aim_runtime.aim_runtime.CommandGenerator") as mock_cg:
                runtime = AIMRuntime(mock_config)
                runtime.profile_selector = mock_ps.return_value
                runtime.command_generator = mock_cg.return_value
                runtime.profile_selector.find_profile.return_value = model_profile
                runtime.command_generator.generate_command_script.return_value = script_path

                result = runtime.dry_run()

        # Check for profile path (format-agnostic - could be in header or comment)
        assert model_profile.profile_handling.path in result

    def test_dry_run_includes_generated_script_section(self, mock_config, model_profile, script_file_factory):
        """Test that dry_run includes the generated script section."""
        script_content = "#!/bin/bash\nset -e\nexport TEST_VAR=value\nexec python -m vllm.entrypoints.openai.api_server"
        script_path = script_file_factory(content=script_content)

        with patch("aim_runtime.aim_runtime.ProfileSelector") as mock_ps:
            with patch("aim_runtime.aim_runtime.CommandGenerator") as mock_cg:
                runtime = AIMRuntime(mock_config)
                runtime.profile_selector = mock_ps.return_value
                runtime.command_generator = mock_cg.return_value
                runtime.profile_selector.find_profile.return_value = model_profile
                runtime.command_generator.generate_command_script.return_value = script_path

                result = runtime.dry_run()

        # Verify sections are present
        assert "SELECTED PROFILE" in result or "Selected profile" in result
        assert "GENERATED SCRIPT" in result or "Generated script" in result
        # Verify script content is included
        assert "#!/bin/bash" in result
        assert "export TEST_VAR=value" in result
        assert "exec python -m vllm.entrypoints.openai.api_server" in result


class TestAIMRuntimeServe:
    """Test suite for AIMRuntime serve functionality."""

    def test_serve_executes_command_successfully(self, mock_config, mock_profile):
        """Test that serve executes the inference server successfully."""
        with patch("aim_runtime.aim_runtime.ProfileSelector") as mock_ps:
            with patch("aim_runtime.aim_runtime.CommandGenerator") as mock_cg:
                with patch("aim_runtime.aim_runtime.shutil.which") as mock_which:
                    with patch("aim_runtime.aim_runtime.os.execv") as mock_execv:
                        runtime = AIMRuntime(mock_config)
                        runtime.profile_selector = mock_ps.return_value
                        runtime.command_generator = mock_cg.return_value
                        runtime.profile_selector.find_profile.return_value = mock_profile
                        runtime.command_generator.generate_execution_params.return_value = (
                            ["python", "-m", "vllm.entrypoints.openai.api_server"],
                            {"TEST_VAR": "value"},
                        )
                        mock_which.return_value = "/usr/bin/python"

                        runtime.serve()

                        mock_execv.assert_called_once_with(
                            "/usr/bin/python", ["python", "-m", "vllm.entrypoints.openai.api_server"]
                        )

    def test_serve_logs_profile_selection(self, mock_config, mock_profile):
        """Test that serve logs profile selection information."""
        with patch("aim_runtime.aim_runtime.ProfileSelector") as mock_ps:
            with patch("aim_runtime.aim_runtime.CommandGenerator") as mock_cg:
                with patch("aim_runtime.aim_runtime.logger") as mock_logger:
                    with patch("aim_runtime.aim_runtime.shutil.which") as mock_which:
                        with patch("aim_runtime.aim_runtime.os.execv"):
                            runtime = AIMRuntime(mock_config)
                            runtime.profile_selector = mock_ps.return_value
                            runtime.command_generator = mock_cg.return_value
                            runtime.profile_selector.find_profile.return_value = mock_profile
                            runtime.command_generator.generate_execution_params.return_value = (
                                ["python", "-m", "vllm.entrypoints.openai.api_server"],
                                {},
                            )
                            mock_which.return_value = "/usr/bin/python"

                            runtime.serve()

                            mock_logger.info.assert_any_call("Selecting profile...")
                            mock_logger.info.assert_any_call(f"Selected profile: {mock_profile.profile_handling.path}")


class TestAIMRuntimeDryRunJson:
    """Test suite for AIMRuntime dry_run_json functionality."""

    def test_dry_run_json_returns_profile_dict(self, mock_config, model_profile):
        """Test that dry_run_json returns list with filename and parsed YAML content."""
        with patch("aim_runtime.aim_runtime.ProfileSelector") as mock_ps:
            with patch("aim_runtime.aim_runtime.CommandGenerator"):
                runtime = AIMRuntime(mock_config)
                runtime.profile_selector = mock_ps.return_value
                runtime.profile_selector.find_profile.return_value = model_profile

                result = runtime.dry_run_json()

                assert isinstance(result, list)
                assert len(result) == 1
                profile_entry = result[0]
                assert profile_entry["filename"] == model_profile.profile_handling.filename
                profile_data = profile_entry["profile"]
                assert isinstance(profile_data, dict)
                assert profile_data["aim_id"] == "meta-llama/Llama-3.1-8B-Instruct"
                assert profile_data["model_id"] == "meta-llama/Llama-3.1-8B-Instruct"
                assert profile_data["metadata"]["precision"] == "fp16"
                assert profile_data["metadata"]["gpu_count"] == 1
                assert profile_data["metadata"]["engine"] == "vllm"
                # Check models field
                assert "models" in profile_entry
                assert isinstance(profile_entry["models"], list)
                assert len(profile_entry["models"]) >= 1
                assert profile_entry["models"][0]["name"] == "meta-llama/Llama-3.1-8B-Instruct"
                assert profile_entry["models"][0]["source"] == "hf://meta-llama/Llama-3.1-8B-Instruct"
                # Check size_gb field is present
                assert "size_gb" in profile_entry["models"][0]

    def test_dry_run_json_with_base_container_and_model_id(self, general_aim_config, general_profiles_path):
        """Test that dry_run_json includes model info from AIM_MODEL_ID for base containers with general profiles."""
        from aim_runtime.profile_validator import ProfileValidator

        with patch("aim_runtime.aim_runtime.ProfileSelector") as mock_ps:
            with patch("aim_runtime.aim_runtime.CommandGenerator"):
                runtime = AIMRuntime(general_aim_config)
                runtime.profile_selector = mock_ps.return_value

                # Load a general profile (no model_id in YAML)
                validator = ProfileValidator(general_aim_config.schema_search_path)
                registry = ProfileRegistry.discover_and_validate(
                    search_paths=[general_profiles_path], validator=validator
                )
                general_profile = registry.find_by_id("general/minimal_profile_no_model")
                runtime.profile_selector.find_profile.return_value = general_profile

                result = runtime.dry_run_json()

                assert isinstance(result, list)
                assert len(result) == 1
                profile_entry = result[0]

                # Check that models field includes the model from config.model_id
                assert "models" in profile_entry
                assert isinstance(profile_entry["models"], list)
                assert len(profile_entry["models"]) == 1
                assert profile_entry["models"][0]["name"] == "meta-llama/Llama-3.1-8B-Instruct"
                assert profile_entry["models"][0]["source"] == "hf://meta-llama/Llama-3.1-8B-Instruct"
                # Check size_gb field is present with storage estimate
                assert "size_gb" in profile_entry["models"][0]

    def test_dry_run_json_returns_empty_dict_on_profile_not_found(self, mock_config):
        """Test that dry_run_json returns empty list when no profile is found."""
        from aim_runtime.profile_selector import ProfileNotFound

        with patch("aim_runtime.aim_runtime.ProfileSelector") as mock_ps:
            with patch("aim_runtime.aim_runtime.CommandGenerator"):
                runtime = AIMRuntime(mock_config)
                runtime.profile_selector = mock_ps.return_value
                runtime.profile_selector.find_profile.side_effect = ProfileNotFound("No profile found")

                result = runtime.dry_run_json()

                assert result == []

    def test_dry_run_json_returns_empty_dict_on_file_not_found(self, mock_config, mock_profile):
        """Test that dry_run_json returns empty list when profile file cannot be read."""
        mock_profile.profile_handling.path = "/nonexistent/path/to/profile.yaml"
        mock_profile.profile_handling.filename = "profile.yaml"

        with patch("aim_runtime.aim_runtime.ProfileSelector") as mock_ps:
            with patch("aim_runtime.aim_runtime.CommandGenerator"):
                runtime = AIMRuntime(mock_config)
                runtime.profile_selector = mock_ps.return_value
                runtime.profile_selector.find_profile.return_value = mock_profile

                result = runtime.dry_run_json()

                assert result == []

    def test_dry_run_json_with_complex_yaml(self, mock_config, complex_profile):
        """Test that dry_run_json handles complex YAML content correctly."""
        with patch("aim_runtime.aim_runtime.ProfileSelector") as mock_ps:
            with patch("aim_runtime.aim_runtime.CommandGenerator"):
                runtime = AIMRuntime(mock_config)
                runtime.profile_selector = mock_ps.return_value
                runtime.profile_selector.find_profile.return_value = complex_profile

                result = runtime.dry_run_json()

                assert isinstance(result, list)
                assert len(result) == 1
                profile_entry = result[0]
                assert profile_entry["filename"] == complex_profile.profile_handling.filename
                profile_data = profile_entry["profile"]
                assert isinstance(profile_data, dict)
                # Complex profile has comprehensive test data
                assert profile_data["aim_id"] == "test/model"
                assert profile_data["model_id"] == "test/model"
                assert profile_data["metadata"]["engine"] == "vllm"
                assert profile_data["metadata"]["precision"] == "fp16"
                # Check engine_args with various types
                assert "engine_args" in profile_data
                assert profile_data["engine_args"]["string-arg"] == "string_value"
                assert profile_data["engine_args"]["int-arg"] == 42
                assert profile_data["engine_args"]["float-arg"] == 3.14159
                assert profile_data["engine_args"]["bool-true-arg"] is True
                # Check env_vars
                assert "env_vars" in profile_data
                assert profile_data["env_vars"]["SIMPLE_VAR"] == "simple"
                # Check models field
                assert "models" in profile_entry
                assert isinstance(profile_entry["models"], list)
                assert len(profile_entry["models"]) >= 1
                assert profile_entry["models"][0]["name"] == "test/model"
                # Check size_gb field is present
                assert "size_gb" in profile_entry["models"][0]


class TestExtractModelsFromProfile:
    """Test suite for _extract_models_from_profile helper function."""

    def test_extract_model_id_from_profile_data(self):
        """Test that model_id is extracted from profile data."""
        from aim_runtime.aim_runtime import _extract_models_from_profile

        profile_data = {
            "aim_id": "meta-llama/Llama-3.1-8B-Instruct",
            "model_id": "amd/Llama-3.1-8B-Instruct-FP8-KV",
        }
        result = _extract_models_from_profile(profile_data)

        assert len(result) == 1
        assert result[0]["name"] == "amd/Llama-3.1-8B-Instruct-FP8-KV"
        assert result[0]["source"] == "hf://amd/Llama-3.1-8B-Instruct-FP8-KV"

    def test_extract_models_returns_empty_list_when_no_model(self):
        """Test that empty list is returned for general profiles without model_id."""
        from aim_runtime.aim_runtime import _extract_models_from_profile

        profile_data = {"metadata": {"engine": "vllm"}}
        result = _extract_models_from_profile(profile_data)

        assert result == []

    def test_extract_models_handles_empty_model_field(self):
        """Test that empty list is returned when model_id field is empty."""
        from aim_runtime.aim_runtime import _extract_models_from_profile

        profile_data = {"aim_id": "test/model", "model_id": ""}
        result = _extract_models_from_profile(profile_data)

        assert result == []


class TestAddStorageEstimates:
    """Test suite for _add_storage_estimates helper function."""

    def test_add_storage_estimates_to_models(self):
        """Test that storage estimates are added to model dictionaries."""
        from aim_runtime.aim_runtime import _add_storage_estimates
        from aim_runtime.model_storage import StorageBackendRegistry

        models = [
            {"name": "meta-llama/Llama-3.1-8B-Instruct", "source": "hf://meta-llama/Llama-3.1-8B-Instruct"},
            {"name": "amd/Llama-3.1-8B-Instruct-FP8-KV", "source": "hf://amd/Llama-3.1-8B-Instruct-FP8-KV"},
        ]

        storage_registry = StorageBackendRegistry()
        _add_storage_estimates(models, storage_registry)

        # Check that size_gb was added to all models
        for model in models:
            assert "size_gb" in model
            # size_gb should be either a float/int or None
            assert model["size_gb"] is None or isinstance(model["size_gb"], (int, float))

    def test_add_storage_estimates_handles_empty_list(self):
        """Test that empty model list is handled gracefully."""
        from aim_runtime.aim_runtime import _add_storage_estimates
        from aim_runtime.model_storage import StorageBackendRegistry

        models = []
        storage_registry = StorageBackendRegistry()

        # Should not raise any errors
        _add_storage_estimates(models, storage_registry)
        assert models == []
