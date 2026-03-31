# Copyright © Advanced Micro Devices, Inc., or its affiliates.
#
# SPDX-License-Identifier: MIT

"""
Tests for engine_config.py — YAML-driven engine configuration and validator registry.
"""

import pytest
import yaml
from pydantic import ValidationError

from aim_common import Engine
from aim_runtime.engine_config import (
    _VLLM_AVAILABLE,
    VALIDATORS,
    EngineConfig,
    _engine_args_to_cli_list,
    load_engine_config,
)


class TestEngineConfig:
    """Test EngineConfig dataclass."""

    def test_create_with_all_fields(self):
        config = EngineConfig(
            launch="python -m vllm.entrypoints.openai.api_server",
            model_arg="--model",
            validator="vllm",
        )
        assert config.launch == "python -m vllm.entrypoints.openai.api_server"
        assert config.model_arg == "--model"
        assert config.validator == "vllm"

    def test_create_with_defaults(self):
        config = EngineConfig(launch="llama-server", model_arg="-m")
        assert config.validator == ""

    def test_frozen(self):
        config = EngineConfig(launch="test", model_arg="--model")
        with pytest.raises(ValidationError):
            config.launch = "changed"


class TestLoadEngineConfig:
    """Test load_engine_config from engines.yaml."""

    def test_load_vllm(self, tmp_path):
        engines_yaml = tmp_path / "engines.yaml"
        engines_yaml.write_text(
            yaml.dump(
                {
                    "vllm": {
                        "launch": "python -m vllm.entrypoints.openai.api_server",
                        "model_arg": "--model",
                        "validator": "vllm",
                    }
                }
            )
        )

        config = load_engine_config(Engine.VLLM, str(tmp_path))

        assert config.launch == "python -m vllm.entrypoints.openai.api_server"
        assert config.model_arg == "--model"
        assert config.validator == "vllm"

    def test_load_minimal_engine(self, tmp_path):
        engines_yaml = tmp_path / "engines.yaml"
        engines_yaml.write_text(
            yaml.dump(
                {
                    "vllm": {
                        "launch": "python -m vllm.entrypoints.openai.api_server",
                        "model_arg": "--model",
                    }
                }
            )
        )

        config = load_engine_config(Engine.VLLM, str(tmp_path))

        assert config.validator == ""

    def test_load_missing_engine_raises(self, tmp_path):
        engines_yaml = tmp_path / "engines.yaml"
        engines_yaml.write_text(
            yaml.dump(
                {
                    "sglang": {
                        "launch": "python -m sglang.launch_server",
                        "model_arg": "--model-path",
                    }
                }
            )
        )

        with pytest.raises(ValueError, match="No configuration for engine 'vllm'"):
            load_engine_config(Engine.VLLM, str(tmp_path))

    def test_load_missing_file_raises(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            load_engine_config(Engine.VLLM, str(tmp_path))

    def test_load_from_test_config(self, project_root):
        """Test loading from the test engines.yaml fixture."""
        config_dir = str(project_root / "tests" / "assets" / "instinct" / "base" / "config")
        config = load_engine_config(Engine.VLLM, config_dir)

        assert "vllm" in config.launch
        assert config.model_arg == "--model"
        assert config.validator == "vllm"


class TestValidatorRegistry:
    """Test VALIDATORS registry."""

    def test_vllm_validator_registered(self):
        """vLLM validator is always registered (native or Pydantic fallback)."""
        assert "vllm" in VALIDATORS
        assert callable(VALIDATORS["vllm"])

    def test_vllm_validator_rejects_invalid_dtype(self):
        """Registered vLLM validator rejects invalid engine args."""
        validator = VALIDATORS["vllm"]
        with pytest.raises((ValueError, ValidationError)):
            validator({"dtype": "invalid-dtype"})

    def test_vllm_validator_accepts_frontend_args(self):
        """Frontend args like disable-uvicorn-access-log should not cause validation errors."""
        validator = VALIDATORS["vllm"]
        # These are FrontendArgs, not EngineArgs — should be accepted
        validator(
            {
                "dtype": "auto",
                "disable-uvicorn-access-log": None,
                "enable-auto-tool-choice": None,
                "tool-call-parser": "hermes",
            }
        )


class TestEngineArgsToCli:
    """Test _engine_args_to_cli_list conversion."""

    def test_none_value_becomes_flag(self):
        assert _engine_args_to_cli_list({"disable-uvicorn-access-log": None}) == ["--disable-uvicorn-access-log"]

    def test_string_value(self):
        assert _engine_args_to_cli_list({"dtype": "auto"}) == ["--dtype", "auto"]

    def test_numeric_value(self):
        assert _engine_args_to_cli_list({"gpu-memory-utilization": 0.9}) == [
            "--gpu-memory-utilization",
            "0.9",
        ]

    def test_bool_true_becomes_flag(self):
        assert _engine_args_to_cli_list({"enforce-eager": True}) == ["--enforce-eager"]

    def test_bool_false_is_skipped(self):
        assert _engine_args_to_cli_list({"enforce-eager": False}) == []

    def test_list_value(self):
        result = _engine_args_to_cli_list({"served-model-name": ["a", "b"]})
        assert result == ["--served-model-name", "a", "b"]

    def test_dict_value_becomes_json(self):
        result = _engine_args_to_cli_list({"rope-scaling": {"type": "linear"}})
        assert result == ["--rope-scaling", '{"type": "linear"}']

    def test_mixed_args(self):
        result = _engine_args_to_cli_list(
            {
                "dtype": "auto",
                "disable-uvicorn-access-log": None,
                "gpu-memory-utilization": 0.9,
            }
        )
        assert result == [
            "--dtype",
            "auto",
            "--disable-uvicorn-access-log",
            "--gpu-memory-utilization",
            "0.9",
        ]


# ---------------------------------------------------------------------------
# Integration tests — vLLM parser contract
# ---------------------------------------------------------------------------

_SKIP_VLLM = pytest.mark.skipif(
    not _VLLM_AVAILABLE,
    reason="vLLM not installed",
)


@_SKIP_VLLM
class TestVllmParserIntegration:
    """Integration tests verifying vLLM's CLI parser API surface.

    These tests validate that the vLLM imports and parser behaviour we depend on
    for validation still work as expected.  They act as a contract test to catch
    breaking changes in vLLM across upgrades.
    """

    def test_make_arg_parser_importable(self):
        """make_arg_parser and FlexibleArgumentParser must be importable."""
        from vllm.entrypoints.openai.cli_args import make_arg_parser  # noqa: F811
        from vllm.utils.argparse_utils import FlexibleArgumentParser  # noqa: F811

        parser = make_arg_parser(FlexibleArgumentParser())
        assert parser is not None

    def test_frontend_args_has_expected_fields(self):
        """FrontendArgs must contain the fields that appear in AIM profiles."""
        import dataclasses

        from vllm.entrypoints.openai.cli_args import FrontendArgs

        fields = {f.name for f in dataclasses.fields(FrontendArgs)}
        # Fields commonly used in AIM profiles
        assert "disable_uvicorn_access_log" in fields
        assert "host" in fields
        assert "port" in fields

    def test_parser_accepts_engine_args(self):
        """Parser must accept standard engine args."""
        from aim_runtime.engine_config import _validate_vllm

        _validate_vllm(
            {
                "dtype": "auto",
                "max-model-len": 4096,
                "tensor-parallel-size": 1,
                "gpu-memory-utilization": 0.9,
            }
        )

    def test_parser_accepts_frontend_args(self):
        """Parser must accept server/frontend args from FrontendArgs."""
        from aim_runtime.engine_config import _validate_vllm

        _validate_vllm(
            {
                "disable-uvicorn-access-log": None,
                "enable-auto-tool-choice": None,
                "tool-call-parser": "hermes",
            }
        )

    def test_parser_accepts_no_prefixed_bool_flags(self):
        """Parser must accept --no-<flag> negation via BooleanOptionalAction."""
        from aim_runtime.engine_config import _validate_vllm

        _validate_vllm(
            {
                "no-async-scheduling": None,
                "no-enable-prefix-caching": None,
                "no-trust-remote-code": None,
            }
        )

    def test_parser_rejects_unknown_args(self):
        """Parser must reject completely unknown arguments."""
        from aim_runtime.engine_config import _validate_vllm

        with pytest.raises(ValueError, match="unrecognized arguments: --totally-bogus-flag"):
            _validate_vllm({"totally-bogus-flag": None})

    def test_parser_rejects_typos(self):
        """Parser must reject misspelled engine args."""
        from aim_runtime.engine_config import _validate_vllm

        with pytest.raises(ValueError, match="unrecognized arguments: --dtyep"):
            _validate_vllm({"dtyep": "auto"})

    def test_typical_profile_validates(self):
        """A representative set of args from an actual AIM profile must validate."""
        from aim_runtime.engine_config import _validate_vllm

        _validate_vllm(
            {
                "attention-backend": "ROCM_AITER_FA",
                "disable-uvicorn-access-log": None,
                "dtype": "auto",
                "gpu-memory-utilization": 0.9,
                "max-model-len": 32768,
                "max-num-batched-tokens": 2048,
                "max-num-seqs": 512,
                "no-async-scheduling": None,
                "no-enable-log-requests": None,
                "no-enable-prefix-caching": None,
                "no-trust-remote-code": None,
                "swap-space": 64,
                "tensor-parallel-size": 1,
            }
        )

    def test_engine_args_with_overrides(self):
        """Profile args merged with AIM_ENGINE_ARGS overrides must validate."""
        from aim_runtime.engine_config import _validate_vllm

        # Simulate: profile has dtype=auto, user overrides to float16
        _validate_vllm(
            {
                "disable-uvicorn-access-log": None,
                "dtype": "float16",
                "max-model-len": 4096,
                "gpu-memory-utilization": 0.85,
            }
        )
