# Copyright © Advanced Micro Devices, Inc., or its affiliates.
#
# SPDX-License-Identifier: MIT

"""Tests for the engine abstraction classes and the build_engine factory.

These exercise the engine layer directly (class attributes, the factory,
launch-prefix and serialization behavior). The CommandGenerator / AIMRuntime
wiring that consumes these engines is covered in their own test modules.
"""

from __future__ import annotations

import pytest

from aim_common import Engine
from aim_runtime.config import AIMConfig
from aim_runtime.engine_config import EngineConfig
from aim_runtime.engines import (
    ENGINE_CLASSES,
    BaseEngine,
    BentomlEngine,
    BentomlEngineArgsModel,
    EngineArgsFormat,
    VllmEngine,
    VllmEngineArgsModel,
    VllmOmniEngine,
    VllmOmniEngineArgsModel,
    build_engine,
    engine_class_for,
)


def _config(engine: Engine) -> AIMConfig:
    return AIMConfig(aim_id="meta-llama/Llama-3.1-8B-Instruct", engine=engine)


def _engine_config(engine: Engine, launch: str = "python -m x", model_arg: str = "--model") -> EngineConfig:
    return EngineConfig(engine=engine, launch=launch, model_arg=model_arg)


class TestFactory:
    @pytest.mark.parametrize(
        "engine,expected",
        [
            (Engine.VLLM, VllmEngine),
            (Engine.VLLM_OMNI, VllmOmniEngine),
            (Engine.BENTOML, BentomlEngine),
        ],
    )
    def test_engine_class_for(self, engine, expected):
        assert engine_class_for(engine) is expected

    @pytest.mark.parametrize("engine", list(Engine))
    def test_every_engine_enum_is_mapped(self, engine):
        assert engine in ENGINE_CLASSES

    @pytest.mark.parametrize(
        "engine,expected",
        [
            (Engine.VLLM, VllmEngine),
            (Engine.VLLM_OMNI, VllmOmniEngine),
            (Engine.BENTOML, BentomlEngine),
        ],
    )
    def test_build_engine_instantiates(self, engine, expected):
        eng = build_engine(_config(engine), _engine_config(engine))
        assert isinstance(eng, expected)
        assert isinstance(eng, BaseEngine)

    def test_build_engine_rejects_engine_mismatch(self):
        # engine_config stamped for a different engine than the AIM declares.
        with pytest.raises(ValueError, match="Engine mismatch"):
            build_engine(_config(Engine.VLLM), _engine_config(Engine.BENTOML))

    def test_build_engine_allows_unset_engine_config_engine(self):
        # engine_config without an engine stamp dispatches on config.engine alone.
        eng = build_engine(_config(Engine.VLLM), EngineConfig(launch="python -m x", model_arg="--model"))
        assert isinstance(eng, VllmEngine)


class TestClassAttributes:
    def test_vllm_args_model_and_format(self):
        assert VllmEngine.ARGS_MODEL is VllmEngineArgsModel
        assert VllmEngine.ARGS_FORMAT is EngineArgsFormat.STANDARD
        assert VllmEngine.requires_aiter_kernels is True

    def test_vllm_omni_args_model(self):
        assert VllmOmniEngine.ARGS_MODEL is VllmOmniEngineArgsModel
        # Omni is an LLM/vLLM engine; inherits vLLM run behavior.
        assert issubclass(VllmOmniEngine, VllmEngine)

    def test_bentoml_args_model_and_format(self):
        assert BentomlEngine.ARGS_MODEL is BentomlEngineArgsModel
        assert BentomlEngine.ARGS_FORMAT is EngineArgsFormat.FORWARDED
        assert BentomlEngine.requires_aiter_kernels is False

    def test_engine_hierarchy(self):
        assert issubclass(VllmEngine, BaseEngine)
        assert issubclass(VllmOmniEngine, VllmEngine)
        assert issubclass(BentomlEngine, BaseEngine)


class TestLaunchPrefix:
    def test_python_resolution(self):
        eng = build_engine(_config(Engine.VLLM), _engine_config(Engine.VLLM, launch="python -m vllm.x"))
        prefix = eng.launch_prefix()
        assert prefix[0] in ("python", "python3")
        assert prefix[1:] == ["-m", "vllm.x"]

    def test_non_python_launch_untouched(self):
        eng = build_engine(_config(Engine.BENTOML), _engine_config(Engine.BENTOML, launch="llama-server"))
        assert eng.launch_prefix() == ["llama-server"]

    def test_model_arg(self):
        eng = build_engine(_config(Engine.VLLM), _engine_config(Engine.VLLM, model_arg="--model"))
        assert eng.model_arg == "--model"


class TestEngineDefaults:
    def test_vllm_sets_served_model_name(self):
        eng = build_engine(_config(Engine.VLLM), _engine_config(Engine.VLLM))
        args: dict = {}
        eng.apply_engine_defaults(args, ["model-a", "aim-a"])
        assert args["served-model-name"] == ["model-a", "aim-a"]

    def test_vllm_omni_inherits_served_model_name(self):
        # Intentional behavior change (EAI-5778): Omni now also sets served-model-name.
        eng = build_engine(_config(Engine.VLLM_OMNI), _engine_config(Engine.VLLM_OMNI))
        args: dict = {}
        eng.apply_engine_defaults(args, ["model-a"])
        assert args["served-model-name"] == ["model-a"]

    def test_bentoml_no_served_model_name(self):
        eng = build_engine(_config(Engine.BENTOML), _engine_config(Engine.BENTOML))
        args: dict = {}
        eng.apply_engine_defaults(args, ["model-a"])
        assert "served-model-name" not in args


class TestSerialize:
    def test_vllm_standard_format(self):
        eng = build_engine(_config(Engine.VLLM), _engine_config(Engine.VLLM))
        assert eng.serialize_engine_args({"max-loras": 8}) == ["--max-loras", "8"]

    def test_bentoml_forwarded_format(self):
        eng = build_engine(_config(Engine.BENTOML), _engine_config(Engine.BENTOML))
        assert eng.serialize_engine_args({"port": 8000}) == ["--arg", "port=8000"]


class TestEnvValidation:
    def test_base_env_validation_is_noop(self):
        # BentomlEngine inherits the base no-op; unknown vars are tolerated.
        BentomlEngine.validate_env_vars({"VLLM_NONSENSE": "1"}, source="x")

    def test_classmethods_callable_without_instance(self):
        # Load-time validation calls these on the class directly.
        BentomlEngine.validate_engine_args({})
