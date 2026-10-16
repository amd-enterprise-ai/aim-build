# Copyright © Advanced Micro Devices, Inc., or its affiliates.
#
# SPDX-License-Identifier: MIT

"""
Tests for AIMRuntime class including dry-run functionality.
"""

import sys
from pathlib import Path
from unittest.mock import MagicMock, Mock, patch

import pytest

from aim_common import Engine, Precision, ProfileMetadata
from aim_runtime.aim_runtime import AIMRuntime
from aim_runtime.config import AIMConfig
from aim_runtime.object_model import Profile, ProfileHandling
from aim_runtime.profile_registry import ProfileRegistry
from aim_utils.yaml_utils import dump_yaml


class TestNormalizeModelSource:
    """Test suite for normalize_model_source static method."""

    def test_normalize_huggingface_model_without_protocol(self):
        """Test that HuggingFace model IDs get hf:// prefix added."""
        assert (
            AIMRuntime.normalize_model_source("meta-llama/Llama-3.1-8B-Instruct")
            == "hf://meta-llama/Llama-3.1-8B-Instruct"
        )
        assert AIMRuntime.normalize_model_source("mistralai/Mistral-7B-v0.1") == "hf://mistralai/Mistral-7B-v0.1"

    def test_normalize_hf_uri_unchanged(self):
        """Test that hf:// URIs are returned unchanged."""
        hf_uri = "hf://org/model"
        assert AIMRuntime.normalize_model_source(hf_uri) == hf_uri


@pytest.fixture
def mock_config():
    """Create a mock AIMConfig for testing."""
    return AIMConfig(
        aim_id="test/aim",
        precision=Precision.FP16,
        accelerator_count=1,
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
    profile.metadata.accelerator_model = None
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

        with patch("aim_runtime.aim_runtime.load_engine_config") as mock_lec:
            mock_lec.return_value.engine = None
            with patch("aim_runtime.aim_runtime.ProfileSelector") as mock_ps:
                with patch("aim_runtime.aim_runtime.CommandGenerator") as mock_cg:
                    runtime = AIMRuntime(mock_config)
                    runtime.profile_selector = mock_ps.return_value
                    runtime.command_generator = mock_cg.return_value
                    runtime.profile_selector.find_profile.return_value = model_profile
                    runtime.command_generator.generate_command_script.return_value = script_path

                    result = dump_yaml(runtime.dry_run())

        # Check for profile path (format-agnostic)
        assert model_profile.profile_handling.path in result
        # Check for YAML content (format-agnostic)
        assert "aim_id:" in result
        assert "meta-llama/Llama-3.1-8B-Instruct" in result
        assert "precision: fp16" in result or 'precision: "fp16"' in result
        assert "accelerator_count: 1" in result  # dry_run reads the raw YAML
        assert "engine: vllm" in result or 'engine: "vllm"' in result
        # Check for generated script
        assert "#!/bin/bash" in result
        assert "echo ''test script''" in result

    def test_dry_run_with_complex_yaml(self, mock_config, complex_profile, script_file_factory):
        """Test that dry_run returns complex YAML content correctly."""
        script_path = script_file_factory(content="#!/bin/bash\necho 'complex test'")

        with patch("aim_runtime.aim_runtime.load_engine_config") as mock_lec:
            mock_lec.return_value.engine = None
            with patch("aim_runtime.aim_runtime.ProfileSelector") as mock_ps:
                with patch("aim_runtime.aim_runtime.CommandGenerator") as mock_cg:
                    runtime = AIMRuntime(mock_config)
                    runtime.profile_selector = mock_ps.return_value
                    runtime.command_generator = mock_cg.return_value
                    runtime.profile_selector.find_profile.return_value = complex_profile
                    runtime.command_generator.generate_command_script.return_value = script_path

                    result = dump_yaml(runtime.dry_run())

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

        with patch("aim_runtime.aim_runtime.load_engine_config") as mock_lec:
            mock_lec.return_value.engine = None
            with patch("aim_runtime.aim_runtime.ProfileSelector") as mock_ps:
                with patch("aim_runtime.aim_runtime.CommandGenerator") as mock_cg:
                    runtime = AIMRuntime(mock_config)
                    runtime.profile_selector = mock_ps.return_value
                    runtime.command_generator = mock_cg.return_value
                    runtime.profile_selector.find_profile.return_value = model_profile
                    runtime.command_generator.generate_command_script.return_value = script_path

                    result = dump_yaml(runtime.dry_run())

        # Check for profile path (format-agnostic - could be in header or comment)
        assert model_profile.profile_handling.path in result

    def test_dry_run_includes_generated_script_section(self, mock_config, model_profile, script_file_factory):
        """Test that dry_run includes the generated script section."""
        script_content = "#!/bin/bash\nset -e\nexport TEST_VAR=value\nexec python -m vllm.entrypoints.openai.api_server"
        script_path = script_file_factory(content=script_content)

        with patch("aim_runtime.aim_runtime.load_engine_config") as mock_lec:
            mock_lec.return_value.engine = None
            with patch("aim_runtime.aim_runtime.ProfileSelector") as mock_ps:
                with patch("aim_runtime.aim_runtime.CommandGenerator") as mock_cg:
                    runtime = AIMRuntime(mock_config)
                    runtime.profile_selector = mock_ps.return_value
                    runtime.command_generator = mock_cg.return_value
                    runtime.profile_selector.find_profile.return_value = model_profile
                    runtime.command_generator.generate_command_script.return_value = script_path

                    result = dump_yaml(runtime.dry_run())

        # Verify script content is included
        assert "#!/bin/bash" in result
        assert "export TEST_VAR=value" in result
        assert "exec python -m vllm.entrypoints.openai.api_server" in result


class TestAIMRuntimeServe:
    """Test suite for AIMRuntime serve functionality."""

    def test_serve_executes_command_successfully(self, mock_config, mock_profile):
        """Test that serve executes the inference server successfully."""
        with patch("aim_runtime.aim_runtime.load_engine_config") as mock_lec:
            mock_lec.return_value.engine = None
            with patch("aim_runtime.aim_runtime.ProfileSelector") as mock_ps:
                with patch("aim_runtime.aim_runtime.CommandGenerator") as mock_cg:
                    with patch("aim_runtime.aim_runtime.shutil.which") as mock_which:
                        with patch.object(AIMRuntime, "_serve_supervised") as mock_supervised:
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

                            mock_supervised.assert_called_once_with(
                                ["python", "-m", "vllm.entrypoints.openai.api_server"], watcher=None
                            )

    def test_serve_uses_execv_when_supervisor_is_disabled(self, mock_config, mock_profile):
        mock_config.serve_supervisor = False
        command_list = ["python", "-m", "vllm.entrypoints.openai.api_server"]

        with patch("aim_runtime.aim_runtime.load_engine_config") as mock_lec:
            mock_lec.return_value.engine = None
            with patch("aim_runtime.aim_runtime.ProfileSelector") as mock_ps:
                with patch("aim_runtime.aim_runtime.CommandGenerator") as mock_cg:
                    with patch("aim_runtime.aim_runtime.shutil.which", return_value="/usr/bin/python"):
                        with (
                            patch("aim_runtime.aim_runtime.os.execv") as mock_execv,
                            patch.object(AIMRuntime, "_serve_supervised") as mock_supervised,
                        ):
                            runtime = AIMRuntime(mock_config)
                            runtime.profile_selector = mock_ps.return_value
                            runtime.command_generator = mock_cg.return_value
                            runtime.profile_selector.find_profile.return_value = mock_profile
                            runtime.command_generator.generate_execution_params.return_value = (command_list, {})

                            runtime.serve()

        mock_execv.assert_called_once_with("/usr/bin/python", command_list)
        mock_supervised.assert_not_called()

    def test_serve_supervises_dynamic_adapters_when_supervisor_is_disabled(self, mock_config, mock_profile):
        mock_config.serve_supervisor = False
        command_list = ["python", "-m", "vllm.entrypoints.openai.api_server"]

        with patch("aim_runtime.aim_runtime.load_engine_config") as mock_lec:
            mock_lec.return_value.engine = None
            with patch("aim_runtime.aim_runtime.ProfileSelector") as mock_ps:
                with patch("aim_runtime.aim_runtime.CommandGenerator") as mock_cg:
                    with patch("aim_runtime.aim_runtime.shutil.which", return_value="/usr/bin/python"):
                        with (
                            patch("aim_runtime.engines.adapter_watcher.AdapterWatcher") as mock_watcher,
                            patch("aim_runtime.aim_runtime.os.execv") as mock_execv,
                            patch.object(AIMRuntime, "_serve_supervised") as mock_supervised,
                        ):
                            runtime = AIMRuntime(mock_config)
                            runtime.engine = Mock(
                                requires_aiter_kernels=False,
                                needs_adapter_watcher=Mock(return_value=True),
                            )
                            runtime.profile_selector = mock_ps.return_value
                            runtime.command_generator = mock_cg.return_value
                            runtime.profile_selector.find_profile.return_value = mock_profile
                            runtime.command_generator.generate_execution_params.return_value = (command_list, {})

                            runtime.serve()

        mock_supervised.assert_called_once_with(command_list, watcher=mock_watcher.return_value)
        mock_execv.assert_not_called()

    def test_serve_logs_profile_selection(self, mock_config, mock_profile):
        """Test that serve logs profile selection information."""
        with patch("aim_runtime.aim_runtime.load_engine_config") as mock_lec:
            mock_lec.return_value.engine = None
            with patch("aim_runtime.aim_runtime.ProfileSelector") as mock_ps:
                with patch("aim_runtime.aim_runtime.CommandGenerator") as mock_cg:
                    with patch("aim_runtime.aim_runtime.logger") as mock_logger:
                        with patch("aim_runtime.aim_runtime.shutil.which") as mock_which:
                            with patch.object(AIMRuntime, "_serve_supervised"):
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
                                mock_logger.info.assert_any_call(
                                    f"Selected profile: {mock_profile.profile_handling.path}"
                                )


@pytest.fixture
def dry_run_json_mocks():
    """Patch ProfileSelector, CommandGenerator, and load_engine_config for dry_run tests."""
    with patch("aim_runtime.aim_runtime.load_engine_config") as mock_lec:
        mock_lec.return_value.engine = None
        with patch("aim_runtime.aim_runtime.ProfileSelector") as mock_ps:
            with patch("aim_runtime.aim_runtime.CommandGenerator") as mock_cg:
                mock_cg_return = Mock()
                mock_cg_return.generate_command_script.return_value = "tests/assets/test_script.sh"
                mock_cg.return_value = mock_cg_return

                def create_runtime(config, profile=None, find_profile_side_effect=None):
                    runtime = AIMRuntime(config)
                    runtime.profile_selector = mock_ps.return_value
                    if find_profile_side_effect is not None:
                        runtime.profile_selector.find_profile.side_effect = find_profile_side_effect
                    elif profile is not None:
                        runtime.profile_selector.find_profile.return_value = profile
                    return runtime

                yield create_runtime


class TestAIMRuntimeDryRunJson:
    """Test suite for AIMRuntime dry_run_json functionality."""

    def test_dry_run_json_returns_profile_dict(self, mock_config, model_profile, dry_run_json_mocks):
        """Test that dry_run_json returns list with filename and parsed YAML content."""
        create_runtime = dry_run_json_mocks
        runtime = create_runtime(mock_config, profile=model_profile)
        result = runtime.dry_run()

        assert isinstance(result, list)
        profile_entry = result[0]

        assert profile_entry["filename"] == model_profile.profile_handling.filename
        profile_data = profile_entry["profile"]
        assert isinstance(profile_data, dict)
        assert profile_data["aim_id"] == "meta-llama/Llama-3.1-8B-Instruct"
        assert profile_data["model_id"] == "meta-llama/Llama-3.1-8B-Instruct"
        assert profile_data["metadata"]["precision"] == "fp16"
        assert profile_data["metadata"]["accelerator_count"] == 1  # dry_run reads the raw YAML
        assert profile_data["metadata"]["engine"] == "vllm"
        # Check models field
        assert "models" in profile_entry
        assert isinstance(profile_entry["models"], list)
        assert len(profile_entry["models"]) >= 1
        assert profile_entry["models"][0]["name"] == "meta-llama/Llama-3.1-8B-Instruct"
        assert profile_entry["models"][0]["source"] == "hf://meta-llama/Llama-3.1-8B-Instruct"
        # Check size_gb field is present
        assert "size_gb" in profile_entry["models"][0]

    def test_dry_run_json_with_base_container_and_model_id(
        self, general_aim_config, general_profiles_path, dry_run_json_mocks
    ):
        """Test that dry_run_json includes model info from AIM_MODEL_ID for base containers with general profiles."""
        from aim_runtime.profile_validator import ProfileValidator

        validator = ProfileValidator()
        registry = ProfileRegistry.discover_and_validate(search_paths=[general_profiles_path], validator=validator)
        general_profile = registry.find_by_id("general/minimal_profile_no_model")

        create_runtime = dry_run_json_mocks
        runtime = create_runtime(general_aim_config, profile=general_profile)
        result = runtime.dry_run()

        assert isinstance(result, list)
        profile_entry = result[0]

        # Check that models field includes the model from config.model_id
        assert "models" in profile_entry
        assert isinstance(profile_entry["models"], list)
        assert len(profile_entry["models"]) == 1
        assert profile_entry["models"][0]["name"] == "meta-llama/Llama-3.1-8B-Instruct"
        assert profile_entry["models"][0]["source"] == "hf://meta-llama/Llama-3.1-8B-Instruct"
        # Check size_gb field is present with storage estimate
        assert "size_gb" in profile_entry["models"][0]

    def test_dry_run_json_returns_empty_dict_on_profile_not_found(self, mock_config, dry_run_json_mocks):
        """Test that dry_run_json returns empty list when no profile is found."""
        from aim_runtime.profile_selector import ProfileNotFound

        create_runtime = dry_run_json_mocks
        runtime = create_runtime(mock_config, find_profile_side_effect=ProfileNotFound("No profile found"))
        result = runtime.dry_run()

        assert result == []

    def test_dry_run_json_returns_empty_dict_on_file_not_found(self, mock_config, mock_profile, dry_run_json_mocks):
        """Test that dry_run_json returns empty list when profile file cannot be read."""
        mock_profile.profile_handling.path = "/nonexistent/path/to/profile.yaml"
        mock_profile.profile_handling.filename = "profile.yaml"

        create_runtime = dry_run_json_mocks
        runtime = create_runtime(mock_config, profile=mock_profile)
        result = runtime.dry_run()

        assert result == []

    def test_dry_run_json_with_complex_yaml(self, mock_config, complex_profile, dry_run_json_mocks):
        """Test that dry_run_json handles complex YAML content correctly."""
        create_runtime = dry_run_json_mocks
        runtime = create_runtime(mock_config, profile=complex_profile)
        result = runtime.dry_run()

        assert isinstance(result, list)

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


class TestInstallAiterPrebuiltKernels:
    """Test suite for _install_aiter_prebuilt_kernels function."""

    @staticmethod
    def _inject_fake_aiter(monkeypatch, aiter_init_path):
        """Pre-register a fake `aiter` module in sys.modules so the function-local
        `import aiter` in _install_aiter_prebuilt_kernels picks up our mock.

        Necessary because some upstream base images (e.g. vllm-openai-rocm v0.21+)
        execute rocminfo at `aiter` import time, which fails on no-GPU CI runners.
        Patching `aim_runtime.aim_runtime.aiter` does not help — the name is
        imported inside the function body, not at module load time.
        """
        fake_aiter = MagicMock()
        fake_aiter.__file__ = str(aiter_init_path)
        monkeypatch.setitem(sys.modules, "aiter", fake_aiter)

    @staticmethod
    def _redirect_prebuilt_path(monkeypatch, target_dir):
        """Patch `Path` inside aim_runtime.aim_runtime so the hard-coded
        /workspace/aiter-jit-prebuilt/<arch> path resolves to a tmp dir,
        while all other Path() calls (including Path(aiter.__file__)) pass
        through to the real pathlib.Path unchanged.
        """
        real_path = Path

        def fake_path(arg):
            if isinstance(arg, str) and arg.startswith("/workspace/aiter-jit-prebuilt/"):
                return target_dir
            return real_path(arg)

        monkeypatch.setattr("aim_runtime.aim_runtime.Path", fake_path)

    def test_installs_kernels_for_valid_gpu(self, tmp_path, monkeypatch):
        """All pre-built .so files should be copied into AITER's jit dir."""
        from aim_runtime.aim_runtime import _install_aiter_prebuilt_kernels

        aiter_jit_dir = tmp_path / "aiter" / "jit"
        aiter_jit_dir.mkdir(parents=True)
        aiter_init = tmp_path / "aiter" / "__init__.py"
        aiter_init.touch()

        prebuilt_dir = tmp_path / "aiter-jit-prebuilt" / "gfx942"
        prebuilt_dir.mkdir(parents=True)
        (prebuilt_dir / "kernel1.so").write_text("kernel-1-content")
        (prebuilt_dir / "kernel2.so").write_text("kernel-2-content")

        self._inject_fake_aiter(monkeypatch, aiter_init)
        self._redirect_prebuilt_path(monkeypatch, prebuilt_dir)
        monkeypatch.setattr("aim_runtime.aim_runtime.get_gfx_arch", lambda _: "gfx942")

        _install_aiter_prebuilt_kernels("MI300X")

        assert (aiter_jit_dir / "kernel1.so").read_text() == "kernel-1-content"
        assert (aiter_jit_dir / "kernel2.so").read_text() == "kernel-2-content"

    def test_returns_early_for_unknown_gpu(self):
        """Test that function returns early when GPU arch cannot be resolved."""
        from aim_runtime.aim_runtime import _install_aiter_prebuilt_kernels

        with patch("aim_runtime.aim_runtime.get_gfx_arch", return_value=None):
            # Should return early without attempting any imports or file operations
            _install_aiter_prebuilt_kernels("UNKNOWN_GPU")
            # If it doesn't raise, test passes

    def test_returns_early_when_prebuilt_dir_missing(self):
        """Test that function returns early when prebuilt directory doesn't exist."""
        from aim_runtime.aim_runtime import _install_aiter_prebuilt_kernels

        with patch("aim_runtime.aim_runtime.get_gfx_arch", return_value="gfx942"):
            with patch("aim_runtime.aim_runtime.Path") as mock_path:
                mock_prebuilt_dir = Mock()
                mock_prebuilt_dir.is_dir.return_value = False
                mock_path.return_value = mock_prebuilt_dir

                _install_aiter_prebuilt_kernels("MI300X")
                # Should return without importing aiter

    def test_handles_aiter_import_error(self, tmp_path):
        """Test that function handles ImportError gracefully when aiter is not installed."""
        from aim_runtime.aim_runtime import _install_aiter_prebuilt_kernels

        prebuilt_dir = tmp_path / "aiter-jit-prebuilt" / "gfx942"
        prebuilt_dir.mkdir(parents=True)

        with patch("aim_runtime.aim_runtime.get_gfx_arch", return_value="gfx942"):
            with patch("aim_runtime.aim_runtime.Path") as mock_path:
                mock_prebuilt = Mock()
                mock_prebuilt.is_dir.return_value = True
                mock_path.return_value = mock_prebuilt

                # Mock the import to raise ImportError
                with patch.dict("sys.modules", {"aiter": None}):
                    with patch("builtins.__import__", side_effect=ImportError("aiter not found")):
                        _install_aiter_prebuilt_kernels("MI300X")
                        # Should log debug message and return gracefully

    def test_skips_existing_kernels(self, tmp_path, monkeypatch):
        """Existing .so files in AITER's jit dir must NOT be overwritten,
        but new .so files should still be installed alongside them."""
        from aim_runtime.aim_runtime import _install_aiter_prebuilt_kernels

        aiter_jit_dir = tmp_path / "aiter" / "jit"
        aiter_jit_dir.mkdir(parents=True)
        aiter_init = tmp_path / "aiter" / "__init__.py"
        aiter_init.touch()
        # Pre-existing kernel that must be preserved
        (aiter_jit_dir / "kernel1.so").write_text("preserved-content")

        prebuilt_dir = tmp_path / "aiter-jit-prebuilt" / "gfx942"
        prebuilt_dir.mkdir(parents=True)
        (prebuilt_dir / "kernel1.so").write_text("would-overwrite-but-skipped")
        (prebuilt_dir / "kernel2.so").write_text("freshly-installed")

        self._inject_fake_aiter(monkeypatch, aiter_init)
        self._redirect_prebuilt_path(monkeypatch, prebuilt_dir)
        monkeypatch.setattr("aim_runtime.aim_runtime.get_gfx_arch", lambda _: "gfx942")

        _install_aiter_prebuilt_kernels("MI300X")

        assert (aiter_jit_dir / "kernel1.so").read_text() == "preserved-content"
        assert (aiter_jit_dir / "kernel2.so").read_text() == "freshly-installed"


class TestAdapterWatcherReadiness:
    def _runtime(self):
        runtime = AIMRuntime.__new__(AIMRuntime)
        runtime.config = Mock(port=8000)
        return runtime

    def test_ready_uses_vllm_readiness_endpoint(self):
        proc = Mock()
        proc.poll.return_value = None
        response = Mock(status_code=200)
        with patch("requests.get", return_value=response) as get:
            self._runtime()._wait_for_engine_ready(proc, 10)
        get.assert_called_once_with("http://localhost:8000/v1/models", timeout=5)

    def test_engine_exit_before_ready_returns(self):
        proc = Mock()
        proc.poll.return_value = 1
        with patch("requests.get") as get:
            self._runtime()._wait_for_engine_ready(proc, 10)
        get.assert_not_called()


# --- real-process supervision ---------------------------------------------- #

#: Launches the real `_serve_supervised` against a fake engine, in its own
#: process so signal handlers and process groups are the genuine article.
_SUPERVISOR_SCRIPT = Path(__file__).parent / "supervisor_probe" / "run_supervisor.py"


def _free_port() -> int:
    import socket

    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


def _port_is_free(port: int) -> bool:
    import socket

    with socket.socket() as sock:
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        try:
            sock.bind(("127.0.0.1", port))
            return True
        except OSError:
            return False


class TestServeSupervisedRealProcess:
    """Supervision driven with real processes, signals, and process groups.

    The mocked tests above patch out ``signal.signal``, ``subprocess.Popen``,
    ``os.getpgid`` and ``os.killpg``, so `process_group=0` and the signal
    forwarding closure never actually execute there. Dropping the process group
    would make ``killpg`` take down the supervisor along with the engine and
    still pass every one of those tests, so exercise it for real here.
    """

    @pytest.fixture(autouse=True)
    def _cleanup_processes(self):
        import os
        import signal
        import subprocess

        self._processes = []
        yield
        for proc in self._processes:
            try:
                os.killpg(proc.pid, signal.SIGKILL)
            except ProcessLookupError:
                # The supervisor may have already cleaned up its own group.
                pass
            try:
                proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                proc.kill()
                proc.wait()

    def _spawn(self, *, status: int, crash_exit_code: int | None = None):
        import subprocess

        port = _free_port()
        command = [sys.executable, str(_SUPERVISOR_SCRIPT), str(status), str(port)]
        if crash_exit_code is not None:
            command += ["--crash-exit-code", str(crash_exit_code)]
        proc = subprocess.Popen(
            command,
            # Own group, so a stray killpg from the supervisor can't reach pytest.
            process_group=0,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        self._processes.append(proc)
        return proc, port

    def _wait_until_serving(self, proc, port, timeout=20.0):
        import socket
        import time

        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            assert proc.poll() is None, "supervisor exited before the engine came up"
            with socket.socket() as sock:
                if sock.connect_ex(("127.0.0.1", port)) == 0:
                    return
            time.sleep(0.1)
        pytest.fail(f"engine never listened on {port}")

    def test_sigterm_is_forwarded_to_the_engine(self):
        """Real handler registration + real forwarding: SIGTERM to the
        supervisor must stop the engine and surface as 143."""
        import signal

        proc, port = self._spawn(status=200)
        self._wait_until_serving(proc, port)

        proc.send_signal(signal.SIGTERM)

        assert proc.wait(timeout=30) == 143
        assert _port_is_free(port), "engine outlived the supervisor"

    def test_engine_exit_one_makes_supervisor_exit_one(self):
        """The real engine failure case must preserve exit code 1."""
        proc, port = self._spawn(status=200, crash_exit_code=1)

        assert proc.wait(timeout=30) == 1
        assert _port_is_free(port), "engine should never have bound its port"
