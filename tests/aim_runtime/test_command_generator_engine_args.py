# Copyright © Advanced Micro Devices, Inc., or its affiliates.
#
# SPDX-License-Identifier: MIT

"""
Tests for CommandGenerator engine arguments override and validation
"""

from unittest.mock import patch

import pytest
from jsonschema import ValidationError

from aim_common import Engine, GPUModel, Metric, Precision, ProfileMetadata, ProfileType
from aim_runtime.command_generator import CommandGenerator
from aim_runtime.config import DEFAULT_CACHE_PATH, DEFAULT_PROFILE_BASE_PATH, DEFAULT_SCHEMA_SEARCH_PATH, AIMConfig
from aim_runtime.object_model import Profile, ProfileHandling


@pytest.fixture
def mock_config():
    """Create a mock AIMConfig for testing."""
    return AIMConfig(
        aim_id="meta-llama/Llama-3.1-8B-Instruct",
        precision=Precision.FP16,
        gpu_count=1,
        gpu_model=GPUModel.MI300X,
        engine=Engine.VLLM,
        metric=Metric.LATENCY,
        profile_id=None,
        profile_base_path=DEFAULT_PROFILE_BASE_PATH,
        schema_search_path=DEFAULT_SCHEMA_SEARCH_PATH,
        cache_path=DEFAULT_CACHE_PATH,
        port=8000,
        engine_args_override=None,
    )


@pytest.fixture
def mock_profile():
    """Create a mock profile for testing."""
    return Profile(
        profile_handling=ProfileHandling(path="/workspace/profiles/test.yaml", filename="test.yaml", priority=1),
        metadata=ProfileMetadata(
            engine=Engine.VLLM,
            gpu=GPUModel.MI300X,
            precision=Precision.FP16,
            gpu_count=1,
            metric=Metric.LATENCY,
            manual_selection_only=False,
            type=ProfileType.UNOPTIMIZED,
        ),
        aim_id="meta-llama/Llama-3.1-8B-Instruct",
        model_id="meta-llama/Llama-3.1-8B-Instruct",
        engine_args={"gpu-memory-utilization": 0.95, "dtype": "float16", "tensor-parallel-size": 1},
        env_vars={},
    )


class TestCommandGeneratorEngineArgsOverride:
    """Test engine arguments override in CommandGenerator."""

    def test_no_override(self, mock_config, mock_profile, schemas_path):
        """Test command generation without user overrides."""
        mock_config.schema_search_path = str(schemas_path)
        generator = CommandGenerator(mock_config)

        merged_args = generator._merge_and_validate_engine_args(mock_profile, Engine.VLLM)

        # Should have only profile args
        assert merged_args["gpu-memory-utilization"] == 0.95
        assert merged_args["dtype"] == "float16"
        assert merged_args["tensor-parallel-size"] == 1

    def test_with_override(self, mock_config, mock_profile, schemas_path):
        """Test command generation with user overrides."""
        mock_config.schema_search_path = str(schemas_path)
        mock_config.engine_args_override = {
            "gpu-memory-utilization": 0.85,  # Override existing
            "max-model-len": 4096,  # Add new
        }
        generator = CommandGenerator(mock_config)

        merged_args = generator._merge_and_validate_engine_args(mock_profile, Engine.VLLM)

        # User override should win
        assert merged_args["gpu-memory-utilization"] == 0.85
        # Profile default preserved
        assert merged_args["dtype"] == "float16"
        # New arg added
        assert merged_args["max-model-len"] == 4096

    def test_override_validation_success(self, mock_config, mock_profile, schemas_path):
        """Test that valid overrides pass validation."""
        mock_config.schema_search_path = str(schemas_path)
        mock_config.engine_args_override = {"max-model-len": 8192, "enforce-eager": True, "kv-cache-dtype": "fp8"}
        generator = CommandGenerator(mock_config)

        # Should not raise
        merged_args = generator._merge_and_validate_engine_args(mock_profile, Engine.VLLM)

        assert merged_args["max-model-len"] == 8192
        assert merged_args["enforce-eager"] is True

    def test_override_validation_failure_wrong_type(self, mock_config, mock_profile, schemas_path):
        """Test that invalid types in overrides are caught."""
        mock_config.schema_search_path = str(schemas_path)
        mock_config.engine_args_override = {"max-model-len": "not-a-number"}  # Should be integer
        generator = CommandGenerator(mock_config)

        with pytest.raises(ValidationError):
            generator._merge_and_validate_engine_args(mock_profile, Engine.VLLM)

    def test_override_validation_failure_invalid_enum(self, mock_config, mock_profile, schemas_path):
        """Test that invalid enum values are caught."""
        mock_config.schema_search_path = str(schemas_path)
        mock_config.engine_args_override = {"dtype": "invalid-dtype"}  # Not in enum
        generator = CommandGenerator(mock_config)

        with pytest.raises(ValidationError):
            generator._merge_and_validate_engine_args(mock_profile, Engine.VLLM)

    def test_empty_profile_args(self, mock_config, schemas_path):
        """Test with profile that has no engine args."""
        mock_config.schema_search_path = str(schemas_path)
        mock_config.engine_args_override = {"max-model-len": 4096}

        empty_profile = Profile(
            profile_handling=ProfileHandling(path="/workspace/profiles/test.yaml", filename="test.yaml", priority=1),
            metadata=ProfileMetadata(
                engine=Engine.VLLM,
                gpu=GPUModel.MI300X,
                precision=Precision.FP16,
                gpu_count=1,
                metric=Metric.LATENCY,
                manual_selection_only=False,
                type=ProfileType.UNOPTIMIZED,
            ),
            aim_id="meta-llama/Llama-3.1-8B-Instruct",
            model_id="meta-llama/Llama-3.1-8B-Instruct",
            engine_args=None,  # No args in profile
            env_vars={},
        )

        generator = CommandGenerator(mock_config)
        merged_args = generator._merge_and_validate_engine_args(empty_profile, Engine.VLLM)

        # Should only have user override
        assert merged_args["max-model-len"] == 4096

    def test_complex_nested_override(self, mock_config, mock_profile, schemas_path):
        """Test override with complex nested structures."""
        mock_config.schema_search_path = str(schemas_path)
        mock_config.engine_args_override = {"rope-scaling": {"type": "linear", "factor": 2.0}}
        generator = CommandGenerator(mock_config)

        merged_args = generator._merge_and_validate_engine_args(mock_profile, Engine.VLLM)

        assert "rope-scaling" in merged_args
        assert merged_args["rope-scaling"]["type"] == "linear"

    def test_logging_override_info(self, mock_config, mock_profile, schemas_path, caplog):
        """Test that overrides are logged appropriately."""
        import logging

        # Clear any previous log records and handlers to ensure test isolation
        caplog.clear()

        # Get the logger and clear any handlers
        logger = logging.getLogger("aim_runtime.command_generator")
        logger.handlers.clear()
        logger.setLevel(logging.DEBUG)  # Ensure we capture all logs

        caplog.set_level(logging.INFO)

        mock_config.schema_search_path = str(schemas_path)
        mock_config.engine_args_override = {"gpu-memory-utilization": 0.80, "max-model-len": 4096}
        generator = CommandGenerator(mock_config)

        # Clear caplog before the operation we want to test
        caplog.clear()
        generator._merge_and_validate_engine_args(mock_profile, Engine.VLLM)

        # Check logging
        assert "user-provided engine argument overrides" in caplog.text.lower()

        # Also check debug-level logs if present
        caplog.set_level(logging.DEBUG)
        caplog.clear()  # Clear before second call
        generator._merge_and_validate_engine_args(mock_profile, Engine.VLLM)
        assert any("gpu-memory-utilization" in record.message for record in caplog.records)

    def test_integration_with_build_command_list(self, mock_config, mock_profile, schemas_path):
        """Test full integration with _build_command_list."""
        mock_config.schema_search_path = str(schemas_path)
        mock_config.engine_args_override = {"max-model-len": 4096}

        generator = CommandGenerator(mock_config)

        # Mock the cache resolver
        with patch.object(generator.cache_resolver, "resolve_model_path", return_value=None):
            command_list = generator._build_command_list(mock_profile)

        # Convert to command string for easier checking
        command_str = " ".join(command_list)

        # Should include overridden arg
        assert "--max-model-len" in command_str
        assert "4096" in command_str

        # Should include profile defaults
        assert "--dtype" in command_str
        assert "float16" in command_str

        # Should include system override
        assert "--port" in command_str
        assert "8000" in command_str
