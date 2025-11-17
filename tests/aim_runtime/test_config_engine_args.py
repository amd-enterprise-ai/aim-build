# Copyright © Advanced Micro Devices, Inc., or its affiliates.
#
# SPDX-License-Identifier: MIT

"""
Tests for AIMConfig engine_args_override functionality
"""

import json

from aim_runtime.config import AIMConfig


class TestEngineArgsOverride:
    """Test engine arguments override functionality in AIMConfig."""

    def test_no_engine_args_env_var(self, monkeypatch):
        """Test config without AIM_ENGINE_ARGS environment variable."""
        monkeypatch.setenv("AIM_MODEL_ID", "meta-llama/Llama-3.1-8B-Instruct")
        monkeypatch.delenv("AIM_ENGINE_ARGS", raising=False)

        config = AIMConfig.from_environment()

        assert config.engine_args_override is None

    def test_empty_engine_args_env_var(self, monkeypatch):
        """Test config with empty AIM_ENGINE_ARGS."""
        monkeypatch.setenv("AIM_MODEL_ID", "meta-llama/Llama-3.1-8B-Instruct")
        monkeypatch.setenv("AIM_ENGINE_ARGS", "")

        config = AIMConfig.from_environment()

        assert config.engine_args_override is None

    def test_valid_json_engine_args(self, monkeypatch):
        """Test parsing valid JSON engine arguments."""
        monkeypatch.setenv("AIM_MODEL_ID", "meta-llama/Llama-3.1-8B-Instruct")
        engine_args = {"max-model-len": 4096, "gpu-memory-utilization": 0.85, "enable-chunked-prefill": True}
        monkeypatch.setenv("AIM_ENGINE_ARGS", json.dumps(engine_args))

        config = AIMConfig.from_environment()

        assert config.engine_args_override == engine_args
        assert config.engine_args_override["max-model-len"] == 4096
        assert config.engine_args_override["gpu-memory-utilization"] == 0.85
        assert config.engine_args_override["enable-chunked-prefill"] is True

    def test_invalid_json_engine_args(self, monkeypatch, caplog):
        """Test handling of invalid JSON."""
        monkeypatch.setenv("AIM_MODEL_ID", "meta-llama/Llama-3.1-8B-Instruct")
        monkeypatch.setenv("AIM_ENGINE_ARGS", "{invalid json}")

        config = AIMConfig.from_environment()

        # Should fallback to None and log warning
        assert config.engine_args_override is None
        assert "Failed to parse AIM_ENGINE_ARGS" in caplog.text

    def test_non_dict_json_engine_args(self, monkeypatch, caplog):
        """Test handling of non-dict JSON (e.g., array or string)."""
        monkeypatch.setenv("AIM_MODEL_ID", "meta-llama/Llama-3.1-8B-Instruct")
        monkeypatch.setenv("AIM_ENGINE_ARGS", '["not", "a", "dict"]')

        config = AIMConfig.from_environment()

        # Should fallback to None and log warning
        assert config.engine_args_override is None
        assert "must be a JSON object/dict" in caplog.text

    def test_engine_args_in_to_dict(self, monkeypatch):
        """Test that engine_args_override appears in to_dict output."""
        monkeypatch.setenv("AIM_MODEL_ID", "meta-llama/Llama-3.1-8B-Instruct")
        engine_args = {"max-model-len": 8192}
        monkeypatch.setenv("AIM_ENGINE_ARGS", json.dumps(engine_args))

        config = AIMConfig.from_environment()
        config_dict = config.to_dict()

        assert "engine_args_override" in config_dict
        assert config_dict["engine_args_override"] == engine_args

    def test_complex_nested_engine_args(self, monkeypatch):
        """Test parsing complex nested JSON structures."""
        monkeypatch.setenv("AIM_MODEL_ID", "meta-llama/Llama-3.1-8B-Instruct")
        engine_args = {
            "rope-scaling": {"type": "linear", "factor": 2.0},
            "hf-overrides": {"max_position_embeddings": 8192},
        }
        monkeypatch.setenv("AIM_ENGINE_ARGS", json.dumps(engine_args))

        config = AIMConfig.from_environment()

        assert config.engine_args_override == engine_args
        assert config.engine_args_override["rope-scaling"]["type"] == "linear"
        assert config.engine_args_override["rope-scaling"]["factor"] == 2.0

    def test_null_values_in_engine_args(self, monkeypatch):
        """Test that null values are preserved."""
        monkeypatch.setenv("AIM_MODEL_ID", "meta-llama/Llama-3.1-8B-Instruct")
        engine_args = {"seed": None, "trust-remote-code": None}
        monkeypatch.setenv("AIM_ENGINE_ARGS", json.dumps(engine_args))

        config = AIMConfig.from_environment()

        assert config.engine_args_override == engine_args
        assert config.engine_args_override["seed"] is None

    def test_boolean_values_in_engine_args(self, monkeypatch):
        """Test that boolean values are parsed correctly."""
        monkeypatch.setenv("AIM_MODEL_ID", "meta-llama/Llama-3.1-8B-Instruct")
        engine_args = {"enforce-eager": True, "disable-sliding-window": False}
        monkeypatch.setenv("AIM_ENGINE_ARGS", json.dumps(engine_args))

        config = AIMConfig.from_environment()

        assert config.engine_args_override["enforce-eager"] is True
        assert config.engine_args_override["disable-sliding-window"] is False

    def test_numeric_values_in_engine_args(self, monkeypatch):
        """Test various numeric types."""
        monkeypatch.setenv("AIM_MODEL_ID", "meta-llama/Llama-3.1-8B-Instruct")
        engine_args = {
            "max-model-len": 4096,  # int
            "gpu-memory-utilization": 0.85,  # float
            "tensor-parallel-size": 2,  # int
        }
        monkeypatch.setenv("AIM_ENGINE_ARGS", json.dumps(engine_args))

        config = AIMConfig.from_environment()

        assert config.engine_args_override["max-model-len"] == 4096
        assert config.engine_args_override["gpu-memory-utilization"] == 0.85
        assert config.engine_args_override["tensor-parallel-size"] == 2

    def test_string_values_in_engine_args(self, monkeypatch):
        """Test string values."""
        monkeypatch.setenv("AIM_MODEL_ID", "meta-llama/Llama-3.1-8B-Instruct")
        engine_args = {"dtype": "float16", "tokenizer-mode": "auto"}
        monkeypatch.setenv("AIM_ENGINE_ARGS", json.dumps(engine_args))

        config = AIMConfig.from_environment()

        assert config.engine_args_override["dtype"] == "float16"
        assert config.engine_args_override["tokenizer-mode"] == "auto"
