# Copyright © Advanced Micro Devices, Inc., or its affiliates.
#
# SPDX-License-Identifier: MIT

"""Tests for adapter enumeration + vLLM static-mode synthesis (ADR-0004)."""

from __future__ import annotations

import json

from aim_common import Engine, Metric, Precision, ProfileMetadata, ProfileType
from aim_runtime.command_generator import CommandGenerator
from aim_runtime.config import AIMConfig
from aim_runtime.engine_config import EngineConfig
from aim_runtime.engines import build_engine
from aim_runtime.engines.adapters import Adapter, enumerate_adapters
from aim_runtime.object_model import Profile, ProfileHandling


def _make_adapter(source, name: str, rank: int = 16) -> None:
    d = source / name
    d.mkdir(parents=True)
    (d / "adapter_config.json").write_text(json.dumps({"r": rank, "lora_alpha": rank * 2}))
    (d / "adapter_model.safetensors").write_text("weights")


class TestEnumerateAdapters:
    def test_empty_or_missing_source(self, tmp_path):
        assert enumerate_adapters(str(tmp_path / "nope"), max_rank=32) == []
        assert enumerate_adapters(str(tmp_path), max_rank=32) == []

    def test_discovers_sorted(self, tmp_path):
        _make_adapter(tmp_path, "b-adapter")
        _make_adapter(tmp_path, "a-adapter")
        result = enumerate_adapters(str(tmp_path), max_rank=32)
        assert result == [
            Adapter(name="a-adapter", path=str(tmp_path / "a-adapter")),
            Adapter(name="b-adapter", path=str(tmp_path / "b-adapter")),
        ]

    def test_over_rank_skipped(self, tmp_path):
        _make_adapter(tmp_path, "ok", rank=16)
        _make_adapter(tmp_path, "toobig", rank=64)
        names = [a.name for a in enumerate_adapters(str(tmp_path), max_rank=32)]
        assert names == ["ok"]

    def test_missing_rank_is_kept(self, tmp_path):
        d = tmp_path / "noranked"
        d.mkdir()
        (d / "adapter_config.json").write_text(json.dumps({"lora_alpha": 32}))
        (d / "adapter_model.safetensors").write_text("weights")
        assert [a.name for a in enumerate_adapters(str(tmp_path), max_rank=32)] == ["noranked"]

    def test_missing_weights_skipped(self, tmp_path):
        # Config present but no weights file -> guaranteed unusable, skip it.
        d = tmp_path / "noweights"
        d.mkdir()
        (d / "adapter_config.json").write_text(json.dumps({"r": 16}))
        assert enumerate_adapters(str(tmp_path), max_rank=32) == []

    def test_bin_weights_accepted(self, tmp_path):
        d = tmp_path / "binweights"
        d.mkdir()
        (d / "adapter_config.json").write_text(json.dumps({"r": 16}))
        (d / "adapter_model.bin").write_text("weights")
        assert [a.name for a in enumerate_adapters(str(tmp_path), max_rank=32)] == ["binweights"]

    def test_invalid_config_skipped(self, tmp_path):
        # Malformed JSON / wrong types -> skip rather than crash.
        d = tmp_path / "broken"
        d.mkdir()
        (d / "adapter_config.json").write_text("{ not valid json")
        (d / "adapter_model.safetensors").write_text("weights")
        assert enumerate_adapters(str(tmp_path), max_rank=32) == []

    def test_invalid_names_skipped(self, tmp_path):
        # Names that would break --lora-modules tokens / OpenAI model ids.
        _make_adapter(tmp_path, "good")
        _make_adapter(tmp_path, "bad=name")
        _make_adapter(tmp_path, "bad name")
        names = [a.name for a in enumerate_adapters(str(tmp_path), max_rank=32)]
        assert names == ["good"]


def _vllm_engine(adapter_source: str, **adapter_kwargs):
    config = AIMConfig(
        aim_id="meta-llama/Llama-3.1-8B-Instruct",
        engine=Engine.VLLM,
        adapter_source=adapter_source,
        **adapter_kwargs,
    )
    return build_engine(config, EngineConfig(engine=Engine.VLLM, launch="python -m x", model_arg="--model"))


class TestVllmStaticSynthesis:
    def test_caps_set_when_adapters_present(self, tmp_path):
        # Static mode enables LoRA + caps only when adapters are actually on disk.
        _make_adapter(tmp_path, "a1")
        eng = _vllm_engine(str(tmp_path), adapter_max_count=8, adapter_max_cpu_count=16, adapter_max_rank=32)
        engine_args, env = eng.build_adapter_runtime(profile=None)
        assert engine_args["enable-lora"] is True
        assert engine_args["max-loras"] == 8
        assert engine_args["max-cpu-loras"] == 16
        assert engine_args["max-lora-rank"] == 32
        assert env == {}  # static mode: no resolver env

    def test_no_enable_lora_when_empty(self, tmp_path):
        # Static mode with no adapters mounted leaves LoRA off entirely so the
        # base model keeps full performance (e.g. AITER fused-MoE on ROCm).
        eng = _vllm_engine(str(tmp_path))
        engine_args, env = eng.build_adapter_runtime(profile=None)
        # No LoRA flags at all: neither the caps (enable-lora/max-loras/
        # max-cpu-loras/max-lora-rank) nor --lora-modules, and no resolver env.
        assert engine_args == {}
        assert env == {}

    def test_lora_modules_from_disk(self, tmp_path):
        _make_adapter(tmp_path, "cs-tone-v3")
        _make_adapter(tmp_path, "cs-billing-v1")
        eng = _vllm_engine(str(tmp_path))
        engine_args, _ = eng.build_adapter_runtime(profile=None)
        assert engine_args["lora-modules"] == [
            f"cs-billing-v1={tmp_path / 'cs-billing-v1'}",
            f"cs-tone-v3={tmp_path / 'cs-tone-v3'}",
        ]

    def test_no_lora_modules_when_empty(self, tmp_path):
        eng = _vllm_engine(str(tmp_path))
        engine_args, _ = eng.build_adapter_runtime(profile=None)
        assert "lora-modules" not in engine_args

    def test_serialized_cli_shape(self, tmp_path):
        # --lora-modules must serialize as one flag followed by each name=path.
        _make_adapter(tmp_path, "a1")
        _make_adapter(tmp_path, "a2")
        eng = _vllm_engine(str(tmp_path))
        engine_args, _ = eng.build_adapter_runtime(profile=None)
        cli = eng.serialize_engine_args({"lora-modules": engine_args["lora-modules"]})
        assert cli == [
            "--lora-modules",
            f"a1={tmp_path / 'a1'}",
            f"a2={tmp_path / 'a2'}",
        ]


def _adapter_profile(features):
    return Profile(
        profile_handling=ProfileHandling(path="/x/test.yaml", filename="test.yaml", priority=1),
        metadata=ProfileMetadata(
            engine=Engine.VLLM,
            precision=Precision.BF16,
            accelerator_count=1,
            metric=Metric.LATENCY,
            manual_selection_only=False,
            type=ProfileType.OPTIMIZED,
            features=features,
        ),
        aim_id="meta-llama/Llama-3.1-8B-Instruct",
        model_id="meta-llama/Llama-3.1-8B-Instruct",
        engine_args={"dtype": "bfloat16"},
        env_vars={},
    )


class TestVllmDynamicSynthesis:
    def test_dynamic_env_triplet(self, tmp_path):
        eng = _vllm_engine(str(tmp_path), adapter_mode="dynamic")
        engine_args, env = eng.build_adapter_runtime(profile=None)
        assert engine_args["enable-lora"] is True
        assert env == {
            "VLLM_ALLOW_RUNTIME_LORA_UPDATING": "True",
            "VLLM_LORA_RESOLVER_CACHE_DIR": str(tmp_path),
            "VLLM_PLUGINS": "lora_filesystem_resolver",
        }

    def test_dynamic_does_not_emit_lora_modules(self, tmp_path):
        _make_adapter(tmp_path, "a1")
        eng = _vllm_engine(str(tmp_path), adapter_mode="dynamic")
        engine_args, _ = eng.build_adapter_runtime(profile=None)
        assert "lora-modules" not in engine_args

    def test_static_refusal_no_triplet(self, tmp_path):
        # Static mode must leave the resolver off so vLLM rejects runtime POSTs.
        eng = _vllm_engine(str(tmp_path), adapter_mode="static")
        _, env = eng.build_adapter_runtime(profile=None)
        assert env == {}


class TestSupervisorGating:
    def test_dynamic_adapter_profile_needs_supervisor(self, tmp_path):
        eng = _vllm_engine(str(tmp_path), adapter_mode="dynamic")
        assert eng.needs_runtime_supervisor(_adapter_profile(["adapters"])) is True

    def test_static_adapter_profile_no_supervisor(self, tmp_path):
        eng = _vllm_engine(str(tmp_path), adapter_mode="static")
        assert eng.needs_runtime_supervisor(_adapter_profile(["adapters"])) is False

    def test_non_adapter_profile_no_supervisor(self, tmp_path):
        eng = _vllm_engine(str(tmp_path), adapter_mode="dynamic")
        assert eng.needs_runtime_supervisor(_adapter_profile([])) is False

    def test_scale_only_profile_no_supervisor_in_dynamic_mode(self, tmp_path):
        # adapters-scale-only has no runtime add/remove -> static path, no watcher.
        eng = _vllm_engine(str(tmp_path), adapter_mode="dynamic")
        assert eng.needs_runtime_supervisor(_adapter_profile(["adapters-scale-only"])) is False

    def test_scale_only_profile_gets_static_not_resolver_env(self, tmp_path):
        # Even in dynamic mode, a scale-only profile must not get the resolver
        # triplet; it stays on the static --lora-modules path.
        _make_adapter(tmp_path, "a1")
        eng = _vllm_engine(str(tmp_path), adapter_mode="dynamic")
        engine_args, env = eng.build_adapter_runtime(_adapter_profile(["adapters-scale-only"]))
        assert env == {}  # no runtime-update env
        assert engine_args["lora-modules"] == [f"a1={tmp_path / 'a1'}"]

    def test_full_adapter_profile_gets_resolver_env_in_dynamic_mode(self, tmp_path):
        _make_adapter(tmp_path, "a1")
        eng = _vllm_engine(str(tmp_path), adapter_mode="dynamic")
        engine_args, env = eng.build_adapter_runtime(_adapter_profile(["adapters"]))
        assert env["VLLM_ALLOW_RUNTIME_LORA_UPDATING"] == "True"
        assert "lora-modules" not in engine_args


class TestCommandGeneratorInjection:
    def test_adapter_profile_injects_lora_cli(self, tmp_path):
        _make_adapter(tmp_path, "cs-tone-v3")
        config = AIMConfig(
            aim_id="meta-llama/Llama-3.1-8B-Instruct",
            engine=Engine.VLLM,
            adapter_source=str(tmp_path),
        )
        engine = build_engine(config, EngineConfig(engine=Engine.VLLM, launch="python -m x", model_arg="--model"))
        gen = CommandGenerator(config, engine)

        command_list = gen._build_command_list(_adapter_profile(["adapters"]))

        assert "--enable-lora" in command_list
        assert "--max-loras" in command_list
        assert "--lora-modules" in command_list
        idx = command_list.index("--lora-modules")
        assert command_list[idx + 1] == f"cs-tone-v3={tmp_path / 'cs-tone-v3'}"

    def test_non_adapter_profile_untouched(self, tmp_path):
        config = AIMConfig(
            aim_id="meta-llama/Llama-3.1-8B-Instruct",
            engine=Engine.VLLM,
            adapter_source=str(tmp_path),
        )
        engine = build_engine(config, EngineConfig(engine=Engine.VLLM, launch="python -m x", model_arg="--model"))
        gen = CommandGenerator(config, engine)

        command_list = gen._build_command_list(_adapter_profile([]))

        assert "--enable-lora" not in command_list
        assert "--lora-modules" not in command_list
