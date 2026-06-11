# Copyright © Advanced Micro Devices, Inc., or its affiliates.
#
# SPDX-License-Identifier: MIT

"""Tests for AIM configuration module."""

import os
from unittest.mock import patch

import pytest

from aim_common import AcceleratorModel
from aim_runtime.config import AIMConfig


class TestAcceleratorModelConfig:
    """Tests for AIM_ACCELERATOR_MODEL environment variable handling."""

    def test_valid_accelerator_model(self):
        """Test that valid accelerator model is accepted."""
        env_vars = {
            "AIM_MODEL_ID": "test/model",
            "AIM_ACCELERATOR_MODEL": "MI300X",
        }
        with patch.dict(os.environ, env_vars, clear=False):
            config = AIMConfig.from_environment()
            assert config.accelerator_model == AcceleratorModel.MI300X

    def test_accelerator_model_case_insensitive(self):
        """Test that accelerator model is case-insensitive."""
        env_vars = {
            "AIM_MODEL_ID": "test/model",
            "AIM_ACCELERATOR_MODEL": "mi300x",
        }
        with patch.dict(os.environ, env_vars, clear=False):
            config = AIMConfig.from_environment()
            assert config.accelerator_model == AcceleratorModel.MI300X

    def test_invalid_accelerator_model(self):
        """Test that invalid accelerator model logs warning and defaults to None."""
        env_vars = {
            "AIM_MODEL_ID": "test/model",
            "AIM_ACCELERATOR_MODEL": "INVALID_GPU",
        }
        with patch.dict(os.environ, env_vars, clear=False):
            config = AIMConfig.from_environment()
            assert config.accelerator_model is None

    def test_no_accelerator_model(self):
        """Test that missing accelerator model defaults to None."""
        env_vars = {
            "AIM_MODEL_ID": "test/model",
        }
        clean_env = {k: v for k, v in os.environ.items() if k not in ("AIM_ACCELERATOR_MODEL", "AIM_GPU_MODEL")}
        with patch.dict(os.environ, clean_env, clear=True):
            with patch.dict(os.environ, env_vars, clear=False):
                config = AIMConfig.from_environment()
                assert config.accelerator_model is None

    def test_all_accelerator_models_valid(self):
        """Test that all AcceleratorModel enum values are valid."""
        for model in AcceleratorModel:
            env_vars = {
                "AIM_MODEL_ID": "test/model",
                "AIM_ACCELERATOR_MODEL": model.value,
            }
            with patch.dict(os.environ, env_vars, clear=False):
                config = AIMConfig.from_environment()
                assert config.accelerator_model == model

    def test_accelerator_model_in_to_dict(self):
        """Test that accelerator_model appears in to_dict output."""
        config = AIMConfig(
            aim_id="test/aim",
            accelerator_model=AcceleratorModel.MI300X,
        )
        config_dict = config.to_dict()
        assert "accelerator_model" in config_dict
        assert config_dict["accelerator_model"] == AcceleratorModel.MI300X

    def test_deprecated_gpu_model_env_var(self):
        """Test that deprecated AIM_GPU_MODEL env var still works as fallback."""
        env_vars = {
            "AIM_MODEL_ID": "test/model",
            "AIM_GPU_MODEL": "MI300X",
        }
        clean_env = {k: v for k, v in os.environ.items() if k != "AIM_ACCELERATOR_MODEL"}
        with patch.dict(os.environ, clean_env, clear=True):
            with patch.dict(os.environ, env_vars, clear=False):
                config = AIMConfig.from_environment()
                assert config.accelerator_model == AcceleratorModel.MI300X

    def test_new_env_var_takes_precedence_over_deprecated(self):
        """Test that AIM_ACCELERATOR_MODEL takes precedence over AIM_GPU_MODEL."""
        env_vars = {
            "AIM_MODEL_ID": "test/model",
            "AIM_ACCELERATOR_MODEL": "MI325X",
            "AIM_GPU_MODEL": "MI300X",
        }
        with patch.dict(os.environ, env_vars, clear=False):
            config = AIMConfig.from_environment()
            assert config.accelerator_model == AcceleratorModel.MI325X


class TestLogLevelConfig:
    """Tests for AIM_LOG_LEVEL_ROOT and AIM_LOG_LEVEL environment variables."""

    def test_default_log_levels(self):
        """Test that log levels default to WARNING for root and INFO for aim_runtime."""
        env_vars = {
            "AIM_MODEL_ID": "test/model",
        }
        # Clear log level env vars if they exist
        clean_env = {k: v for k, v in os.environ.items() if k not in ["AIM_LOG_LEVEL_ROOT", "AIM_LOG_LEVEL"]}
        with patch.dict(os.environ, clean_env, clear=True):
            with patch.dict(os.environ, env_vars, clear=False):
                config = AIMConfig.from_environment()
                assert config.log_level_root == "WARNING"
                assert config.log_level == "INFO"

    def test_custom_log_levels(self):
        """Test that custom log levels are read from environment."""
        env_vars = {
            "AIM_MODEL_ID": "test/model",
            "AIM_LOG_LEVEL_ROOT": "ERROR",
            "AIM_LOG_LEVEL": "DEBUG",
        }
        with patch.dict(os.environ, env_vars, clear=False):
            config = AIMConfig.from_environment()
            assert config.log_level_root == "ERROR"
            assert config.log_level == "DEBUG"

    def test_all_log_levels_valid(self):
        """Test that all standard Python log levels are accepted."""
        valid_levels = ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]

        for level in valid_levels:
            env_vars = {
                "AIM_MODEL_ID": "test/model",
                "AIM_LOG_LEVEL_ROOT": level,
                "AIM_LOG_LEVEL": level,
            }
            with patch.dict(os.environ, env_vars, clear=False):
                config = AIMConfig.from_environment()
                assert config.log_level_root == level
                assert config.log_level == level

    def test_log_levels_case_sensitivity(self):
        """Test that log levels are accepted as-is (case preserved)."""
        env_vars = {
            "AIM_MODEL_ID": "test/model",
            "AIM_LOG_LEVEL_ROOT": "error",
            "AIM_LOG_LEVEL": "debug",
        }
        with patch.dict(os.environ, env_vars, clear=False):
            config = AIMConfig.from_environment()
            # Values are stored as-is; configure_logging handles case conversion
            assert config.log_level_root == "error"
            assert config.log_level == "debug"

    def test_log_levels_in_to_dict(self):
        """Test that log levels appear in to_dict output."""
        config = AIMConfig(
            aim_id="test/aim",
            log_level_root="ERROR",
            log_level="DEBUG",
        )
        config_dict = config.to_dict()
        assert "log_level_root" in config_dict
        assert config_dict["log_level_root"] == "ERROR"
        assert "log_level" in config_dict
        assert config_dict["log_level"] == "DEBUG"


class TestAIMIDValidation:
    """Tests for AIM_ID and AIM_MODEL_ID validation."""

    def test_aim_model_id_only(self):
        """Test that AIM_MODEL_ID alone is valid."""
        env_vars = {
            "AIM_MODEL_ID": "test/model",
        }
        # Clear AIM_ID if it exists
        clean_env = {k: v for k, v in os.environ.items() if k != "AIM_ID"}
        with patch.dict(os.environ, clean_env, clear=True):
            with patch.dict(os.environ, env_vars, clear=False):
                config = AIMConfig.from_environment()
                assert config.model_id == "test/model"
                assert config.aim_id is None

    def test_aim_id_only(self):
        """Test that AIM_ID alone is valid."""
        env_vars = {
            "AIM_ID": "test/aim-container",
        }
        # Clear AIM_MODEL_ID if it exists
        clean_env = {k: v for k, v in os.environ.items() if k != "AIM_MODEL_ID"}
        with patch.dict(os.environ, clean_env, clear=True):
            with patch.dict(os.environ, env_vars, clear=False):
                config = AIMConfig.from_environment()
                assert config.aim_id == "test/aim-container"
                assert config.model_id is None

    def test_model_id_param_only(self):
        """Test that model_id_param alone is valid."""
        env_vars = {}
        model_id_param = "test/aim-container"
        # Clear AIM_MODEL_ID and AIM_ID if they exist
        clean_env = {k: v for k, v in os.environ.items() if k not in ("AIM_MODEL_ID", "AIM_ID")}
        with patch.dict(os.environ, clean_env, clear=True):
            with patch.dict(os.environ, env_vars, clear=False):
                config = AIMConfig.from_environment(model_id_param)
                assert config.aim_id is None
                assert config.model_id == "test/aim-container"

    def test_both_aim_id_and_model_id_raises_error(self):
        """Test that setting both AIM_ID and AIM_MODEL_ID raises ValueError."""
        env_vars = {
            "AIM_ID": "test/aim-container",
            "AIM_MODEL_ID": "test/model",
        }
        with patch.dict(os.environ, env_vars, clear=False):
            with pytest.raises(ValueError, match="Cannot set both AIM_ID and AIM_MODEL_ID"):
                AIMConfig.from_environment()

    def test_neither_aim_id_nor_model_id_raises_error(self):
        """Test that missing both AIM_ID and AIM_MODEL_ID raises ValueError."""
        # Clear both env vars if they exist
        clean_env = {k: v for k, v in os.environ.items() if k not in ["AIM_ID", "AIM_MODEL_ID"]}
        with patch.dict(os.environ, clean_env, clear=True):
            with pytest.raises(ValueError, match="Either AIM_MODEL_ID or AIM_ID environment variable is required"):
                AIMConfig.from_environment()


class TestAllowGeneralProfileFallbackConfig:
    """Tests for AIM_ALLOW_GENERAL_PROFILE_FALLBACK environment variable handling."""

    def test_default_allow_general_profile_fallback(self):
        """Test that allow_general_profile_fallback defaults to True."""
        env_vars = {
            "AIM_MODEL_ID": "test/model",
        }
        # Clear AIM_ALLOW_GENERAL_PROFILE_FALLBACK if it exists
        clean_env = {k: v for k, v in os.environ.items() if k != "AIM_ALLOW_GENERAL_PROFILE_FALLBACK"}
        with patch.dict(os.environ, clean_env, clear=True):
            with patch.dict(os.environ, env_vars, clear=False):
                config = AIMConfig.from_environment()
                assert config.allow_general_profile_fallback is True

    def test_allow_general_profile_fallback_true(self):
        """Test that AIM_ALLOW_GENERAL_PROFILE_FALLBACK=true is accepted."""
        env_vars = {
            "AIM_MODEL_ID": "test/model",
            "AIM_ALLOW_GENERAL_PROFILE_FALLBACK": "true",
        }
        with patch.dict(os.environ, env_vars, clear=False):
            config = AIMConfig.from_environment()
            assert config.allow_general_profile_fallback is True

    def test_allow_general_profile_fallback_false(self):
        """Test that AIM_ALLOW_GENERAL_PROFILE_FALLBACK=false is accepted."""
        env_vars = {
            "AIM_MODEL_ID": "test/model",
            "AIM_ALLOW_GENERAL_PROFILE_FALLBACK": "false",
        }
        with patch.dict(os.environ, env_vars, clear=False):
            config = AIMConfig.from_environment()
            assert config.allow_general_profile_fallback is False

    def test_allow_general_profile_fallback_case_insensitive(self):
        """Test that AIM_ALLOW_GENERAL_PROFILE_FALLBACK is case-insensitive."""
        for value in ["TRUE", "True", "FALSE", "False"]:
            env_vars = {
                "AIM_MODEL_ID": "test/model",
                "AIM_ALLOW_GENERAL_PROFILE_FALLBACK": value,
            }
            with patch.dict(os.environ, env_vars, clear=False):
                config = AIMConfig.from_environment()
                assert config.allow_general_profile_fallback == (value.lower() == "true")

    def test_allow_general_profile_fallback_yes_no(self):
        """Test that AIM_ALLOW_GENERAL_PROFILE_FALLBACK accepts yes/no."""
        env_vars_yes = {
            "AIM_MODEL_ID": "test/model",
            "AIM_ALLOW_GENERAL_PROFILE_FALLBACK": "yes",
        }
        with patch.dict(os.environ, env_vars_yes, clear=False):
            config = AIMConfig.from_environment()
            assert config.allow_general_profile_fallback is True

        env_vars_no = {
            "AIM_MODEL_ID": "test/model",
            "AIM_ALLOW_GENERAL_PROFILE_FALLBACK": "no",
        }
        with patch.dict(os.environ, env_vars_no, clear=False):
            config = AIMConfig.from_environment()
            assert config.allow_general_profile_fallback is False

    def test_allow_general_profile_fallback_numeric(self):
        """Test that AIM_ALLOW_GENERAL_PROFILE_FALLBACK accepts 1/0."""
        env_vars_one = {
            "AIM_MODEL_ID": "test/model",
            "AIM_ALLOW_GENERAL_PROFILE_FALLBACK": "1",
        }
        with patch.dict(os.environ, env_vars_one, clear=False):
            config = AIMConfig.from_environment()
            assert config.allow_general_profile_fallback is True

        env_vars_zero = {
            "AIM_MODEL_ID": "test/model",
            "AIM_ALLOW_GENERAL_PROFILE_FALLBACK": "0",
        }
        with patch.dict(os.environ, env_vars_zero, clear=False):
            config = AIMConfig.from_environment()
            assert config.allow_general_profile_fallback is False

    def test_allow_general_profile_fallback_invalid_value(self):
        """Test that invalid values default to True with a warning."""
        env_vars = {
            "AIM_MODEL_ID": "test/model",
            "AIM_ALLOW_GENERAL_PROFILE_FALLBACK": "invalid",
        }
        with patch.dict(os.environ, env_vars, clear=False):
            config = AIMConfig.from_environment()
            # Should default to True due to invalid value
            assert config.allow_general_profile_fallback is True

    def test_allow_general_profile_fallback_in_to_dict(self):
        """Test that allow_general_profile_fallback appears in to_dict output."""
        config = AIMConfig(
            aim_id="test/aim",
            allow_general_profile_fallback=False,
        )
        config_dict = config.to_dict()
        assert "allow_general_profile_fallback" in config_dict
        assert config_dict["allow_general_profile_fallback"] is False


class TestAllowUnoptimizedConfig:
    """Tests for AIM_ALLOW_UNOPTIMIZED environment variable handling."""

    def test_default_allow_unoptimized(self):
        """Test that allow_unoptimized defaults to False."""
        env_vars = {
            "AIM_MODEL_ID": "test/model",
        }
        clean_env = {k: v for k, v in os.environ.items() if k != "AIM_ALLOW_UNOPTIMIZED"}
        with patch.dict(os.environ, clean_env, clear=True):
            with patch.dict(os.environ, env_vars, clear=False):
                config = AIMConfig.from_environment()
                assert config.allow_unoptimized is False

    def test_allow_unoptimized_true(self):
        """Test that AIM_ALLOW_UNOPTIMIZED=true is accepted."""
        env_vars = {
            "AIM_MODEL_ID": "test/model",
            "AIM_ALLOW_UNOPTIMIZED": "true",
        }
        with patch.dict(os.environ, env_vars, clear=False):
            config = AIMConfig.from_environment()
            assert config.allow_unoptimized is True

    def test_allow_unoptimized_false(self):
        """Test that AIM_ALLOW_UNOPTIMIZED=false is accepted."""
        env_vars = {
            "AIM_MODEL_ID": "test/model",
            "AIM_ALLOW_UNOPTIMIZED": "false",
        }
        with patch.dict(os.environ, env_vars, clear=False):
            config = AIMConfig.from_environment()
            assert config.allow_unoptimized is False

    def test_allow_unoptimized_case_insensitive(self):
        """Test that AIM_ALLOW_UNOPTIMIZED is case-insensitive."""
        for value in ["TRUE", "True", "FALSE", "False"]:
            env_vars = {
                "AIM_MODEL_ID": "test/model",
                "AIM_ALLOW_UNOPTIMIZED": value,
            }
            with patch.dict(os.environ, env_vars, clear=False):
                config = AIMConfig.from_environment()
                assert config.allow_unoptimized == (value.lower() == "true")

    def test_allow_unoptimized_yes_no(self):
        """Test that AIM_ALLOW_UNOPTIMIZED accepts yes/no."""
        env_vars_yes = {
            "AIM_MODEL_ID": "test/model",
            "AIM_ALLOW_UNOPTIMIZED": "yes",
        }
        with patch.dict(os.environ, env_vars_yes, clear=False):
            config = AIMConfig.from_environment()
            assert config.allow_unoptimized is True

        env_vars_no = {
            "AIM_MODEL_ID": "test/model",
            "AIM_ALLOW_UNOPTIMIZED": "no",
        }
        with patch.dict(os.environ, env_vars_no, clear=False):
            config = AIMConfig.from_environment()
            assert config.allow_unoptimized is False

    def test_allow_unoptimized_numeric(self):
        """Test that AIM_ALLOW_UNOPTIMIZED accepts 1/0."""
        env_vars_one = {
            "AIM_MODEL_ID": "test/model",
            "AIM_ALLOW_UNOPTIMIZED": "1",
        }
        with patch.dict(os.environ, env_vars_one, clear=False):
            config = AIMConfig.from_environment()
            assert config.allow_unoptimized is True

        env_vars_zero = {
            "AIM_MODEL_ID": "test/model",
            "AIM_ALLOW_UNOPTIMIZED": "0",
        }
        with patch.dict(os.environ, env_vars_zero, clear=False):
            config = AIMConfig.from_environment()
            assert config.allow_unoptimized is False

    def test_allow_unoptimized_invalid_value(self):
        """Test that invalid values default to False with a warning."""
        env_vars = {
            "AIM_MODEL_ID": "test/model",
            "AIM_ALLOW_UNOPTIMIZED": "invalid",
        }
        with patch.dict(os.environ, env_vars, clear=False):
            config = AIMConfig.from_environment()
            # Should default to False due to invalid value
            assert config.allow_unoptimized is False

    def test_allow_unoptimized_in_to_dict(self):
        """Test that allow_unoptimized appears in to_dict output."""
        config = AIMConfig(
            aim_id="test/aim",
            allow_unoptimized=True,
        )
        config_dict = config.to_dict()
        assert "allow_unoptimized" in config_dict
        assert config_dict["allow_unoptimized"] is True
