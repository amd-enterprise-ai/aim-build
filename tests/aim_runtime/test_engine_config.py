# Copyright © Advanced Micro Devices, Inc., or its affiliates.
#
# SPDX-License-Identifier: MIT

"""
Tests for engine_config.py — YAML-driven engine configuration and validator registry.
"""

import pytest
from pydantic import ValidationError

from aim_common import Engine
from aim_common.engine_args_models import (
    ENGINE_ARGS_MODELS,
    EngineArgsFormat,
    VllmEngineArgsModel,
    VllmOmniEngineArgsModel,
    engine_args_to_cli_list,
)
from aim_runtime.engine_config import EngineConfig, load_engine_config

_VLLM_AVAILABLE = bool(VllmEngineArgsModel._vllm_parser())
from aim_utils.yaml_utils import dump_yaml  # noqa: E402


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

    @pytest.mark.parametrize(
        "engine,expected_format",
        [
            ("bentoml", EngineArgsFormat.FORWARDED),
            ("vllm", EngineArgsFormat.STANDARD),
        ],
    )
    def test_args_format_inferred_from_engine(self, engine, expected_format):
        """args_format is derived from engine name when not explicitly set."""
        config = EngineConfig(engine=engine, launch="dummy-launch-cmd")
        assert config.args_format == expected_format

    def test_args_format_explicit_overrides_inference(self):
        """Explicit args_format takes precedence over the implicit mapping."""
        config = EngineConfig(engine="bentoml", launch="python -m bentoml serve", args_format=EngineArgsFormat.STANDARD)
        assert config.args_format == EngineArgsFormat.STANDARD

    def test_frozen(self):
        config = EngineConfig(launch="test", model_arg="--model")
        with pytest.raises(ValidationError):
            config.launch = "changed"


class TestLoadEngineConfig:
    """Test load_engine_config from engines.yaml."""

    def test_load_vllm(self, tmp_path):
        engines_yaml = tmp_path / "engines.yaml"
        engines_yaml.write_text(
            dump_yaml(
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
            dump_yaml(
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
            dump_yaml(
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
        with pytest.raises(ValueError, match="YAML file does not exist"):
            load_engine_config(Engine.VLLM, str(tmp_path))

    @pytest.mark.parametrize(
        "accelerator,engine,config_subpath,expected_model_arg,expected_format",
        [
            ("instinct", Engine.VLLM, "base/config", "--model", EngineArgsFormat.STANDARD),
            ("radeon", Engine.VLLM, "base/config", "--model", EngineArgsFormat.STANDARD),
            ("instinct", Engine.BENTOML, "base/bentoml/config", "", EngineArgsFormat.FORWARDED),
        ],
    )
    def test_load_from_test_config(
        self, assets_path, accelerator, engine, config_subpath, expected_model_arg, expected_format
    ):
        """Test loading engine configs from the test engines.yaml fixture."""
        config_dir = str(assets_path / accelerator / config_subpath)
        config = load_engine_config(engine, config_dir)

        assert engine.value in config.launch
        assert config.model_arg == expected_model_arg
        assert config.args_format == expected_format

    def test_load_vllm_omni_from_yaml(self, tmp_path):
        """Load vllm_omni engine config from YAML."""
        engines_yaml = tmp_path / "engines.yaml"
        engines_yaml.write_text(
            dump_yaml(
                {
                    "vllm_omni": {
                        "launch": "vllm serve --omni",
                        "model_arg": "--model",
                        "validator": "vllm_omni",
                    }
                }
            )
        )
        config = load_engine_config(Engine.VLLM_OMNI, str(tmp_path))
        assert "serve" in config.launch and "--omni" in config.launch
        assert config.model_arg == "--model"
        assert config.validator == "vllm_omni"


class TestEngineModelRegistry:
    """Test ENGINE_ARGS_MODELS registry."""

    def test_vllm_model_registered(self):
        """vLLM model is always registered."""
        assert "vllm" in ENGINE_ARGS_MODELS
        assert issubclass(ENGINE_ARGS_MODELS["vllm"], VllmEngineArgsModel)

    def test_vllm_omni_model_registered(self):
        """vLLM-Omni model is registered for engines.yaml validator vllm_omni."""
        assert "vllm_omni" in ENGINE_ARGS_MODELS
        assert ENGINE_ARGS_MODELS["vllm_omni"] is VllmOmniEngineArgsModel

    def test_vllm_model_rejects_invalid_dtype(self):
        """Registered vLLM model rejects invalid engine args."""
        with pytest.raises((ValueError, ValidationError)):
            ENGINE_ARGS_MODELS["vllm"].model_validate({"dtype": "invalid-dtype"})

    def test_vllm_model_accepts_frontend_args(self):
        """Frontend args like disable-uvicorn-access-log should not cause validation errors."""
        # These are FrontendArgs, not EngineArgs — should be accepted
        ENGINE_ARGS_MODELS["vllm"].model_validate(
            {
                "dtype": "auto",
                "disable-uvicorn-access-log": None,
                "enable-auto-tool-choice": None,
                "tool-call-parser": "hermes",
            }
        )


class TestEngineArgsToCli:
    """Test engine_args_to_cli_list conversion."""

    def test_none_value_becomes_flag(self):
        assert engine_args_to_cli_list({"disable-uvicorn-access-log": None}) == ["--disable-uvicorn-access-log"]

    def test_string_value(self):
        assert engine_args_to_cli_list({"dtype": "auto"}) == ["--dtype", "auto"]

    def test_numeric_value(self):
        assert engine_args_to_cli_list({"gpu-memory-utilization": 0.9}) == [
            "--gpu-memory-utilization",
            "0.9",
        ]

    def test_bool_true_becomes_flag(self):
        assert engine_args_to_cli_list({"enforce-eager": True}) == ["--enforce-eager"]

    def test_bool_false_is_skipped(self):
        assert engine_args_to_cli_list({"enforce-eager": False}) == []

    def test_list_value(self):
        result = engine_args_to_cli_list({"served-model-name": ["a", "b"]})
        assert result == ["--served-model-name", "a", "b"]

    def test_dict_value_becomes_json(self):
        result = engine_args_to_cli_list({"rope-scaling": {"type": "linear"}})
        assert result == ["--rope-scaling", '{"type": "linear"}']

    def test_mixed_args(self):
        result = engine_args_to_cli_list(
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

    def test_snake_case_keys_become_kebab_case_flags(self):
        """Snake_case keys must be normalized to --kebab-case flags.

        Profiles use snake_case (e.g. distributed_executor_backend); the engine
        CLI expects --distributed-executor-backend.  Without normalization the
        flag would be passed verbatim and silently ignored or rejected.
        """
        result = engine_args_to_cli_list(
            {
                "distributed_executor_backend": "mp",
                "gpu_memory_utilization": 0.9,
                "max_model_len": 4096,
            }
        )
        assert result == [
            "--distributed-executor-backend",
            "mp",
            "--gpu-memory-utilization",
            "0.9",
            "--max-model-len",
            "4096",
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
        VllmEngineArgsModel.model_validate(
            {
                "dtype": "auto",
                "max-model-len": 4096,
                "tensor-parallel-size": 1,
                "gpu-memory-utilization": 0.9,
            }
        )

    def test_parser_accepts_frontend_args(self):
        """Parser must accept server/frontend args from FrontendArgs."""
        VllmEngineArgsModel.model_validate(
            {
                "disable-uvicorn-access-log": None,
                "enable-auto-tool-choice": None,
                "tool-call-parser": "hermes",
            }
        )

    def test_parser_accepts_no_prefixed_bool_flags(self):
        """Parser must accept --no-<flag> negation via BooleanOptionalAction."""
        VllmEngineArgsModel.model_validate(
            {
                "no-async-scheduling": None,
                "no-enable-prefix-caching": None,
                "no-trust-remote-code": None,
            }
        )

    def test_parser_rejects_unknown_args(self):
        """Parser must reject completely unknown arguments."""
        with pytest.raises(ValueError, match="unrecognized arguments: --totally-bogus-flag"):
            VllmEngineArgsModel.model_validate({"totally-bogus-flag": None})

    def test_parser_rejects_typos(self):
        """Parser must reject misspelled engine args."""
        with pytest.raises(ValueError, match="unrecognized arguments: --dtyep"):
            VllmEngineArgsModel.model_validate({"dtyep": "auto"})

    def test_typical_profile_validates(self):
        """A representative set of args from an actual AIM profile must validate."""
        VllmEngineArgsModel.model_validate(
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
        # Simulate: profile has dtype=auto, user overrides to float16
        VllmEngineArgsModel.model_validate(
            {
                "disable-uvicorn-access-log": None,
                "dtype": "float16",
                "max-model-len": 4096,
                "gpu-memory-utilization": 0.85,
            }
        )

    def test_snake_case_keys_validate(self):
        """Snake_case keys must be accepted and treated identically to kebab-case.

        Real YAML profiles use snake_case for some args (e.g. distributed_executor_backend).
        The validator must normalize underscore→hyphen before invoking vLLM's argparse.
        """
        VllmEngineArgsModel.model_validate(
            {
                "dtype": "auto",
                "max_model_len": 4096,
                "tensor_parallel_size": 1,
                "gpu_memory_utilization": 0.9,
                "distributed_executor_backend": "mp",
            }
        )
