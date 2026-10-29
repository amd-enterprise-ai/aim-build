# Copyright © Advanced Micro Devices, Inc., or its affiliates.
#
# SPDX-License-Identifier: MIT

"""Tests for the llama.cpp engine: arguments, defaults, model selection, conversion and KV cache sizing."""

from __future__ import annotations

import fnmatch
import json
import subprocess
from types import SimpleNamespace

import pytest
from pydantic import ValidationError

from aim_common import Engine, Precision
from aim_runtime.config import AIMConfig
from aim_runtime.engine_config import EngineConfig
from aim_runtime.engines import LlamaCppEngine, LlamaCppEngineArgsModel, build_engine
from aim_runtime.engines.llamacpp import CONVERTER_PATH
from aim_runtime.model_cache_resolver import ResolvedModelPath

MODEL_ID = "org/model-GGUF"

LLAMA_8B_CONFIG = {
    "hidden_size": 4096,
    "num_attention_heads": 32,
    "num_key_value_heads": 8,
    "num_hidden_layers": 32,
}


def _engine(cache_path=None, engine_args_override=None) -> LlamaCppEngine:
    config = AIMConfig(model_id=MODEL_ID, engine=Engine.LLAMACPP, engine_args_override=engine_args_override)
    if cache_path is not None:
        config.cache_path = str(cache_path)
    engine_config = EngineConfig(engine=Engine.LLAMACPP, launch="llama-server", model_arg="-m")
    return build_engine(config, engine_config)


def _cached(path) -> ResolvedModelPath:
    return ResolvedModelPath(path=str(path), is_local_dir=True, model_id=MODEL_ID)


def _profile(precision: Precision = Precision.BF16, engine_args=None):
    return SimpleNamespace(metadata=SimpleNamespace(precision=precision), engine_args=engine_args or {})


def _hf_checkpoint(path):
    (path / "config.json").write_text("{}")
    (path / "model-00001-of-00001.safetensors").touch()
    return path


class TestLlamaCppEngineArgsModel:
    def test_profile_like_kebab_keys_accepted(self):
        LlamaCppEngineArgsModel.model_validate(
            {"host": "0.0.0.0", "ctx-size": 8192, "threads": 96, "load-mode": "none", "parallel": 4}
        )

    def test_snake_case_keys_accepted(self):
        LlamaCppEngineArgsModel.model_validate({"ctx_size": 8192, "load_mode": "mmap"})

    def test_invalid_type_rejected(self):
        with pytest.raises(ValidationError):
            LlamaCppEngineArgsModel.model_validate({"threads": "many"})

    @pytest.mark.parametrize(
        "field,value", [("load-mode", "no-mmap"), ("cache-type-k", "bogus"), ("flash-attn", "yes")]
    )
    def test_invalid_choice_rejected(self, field, value):
        with pytest.raises(ValidationError):
            LlamaCppEngineArgsModel.model_validate({field: value})

    def test_unknown_flag_passes_through(self):
        LlamaCppEngineArgsModel.model_validate({"some-new-flag": 1})


class TestResolveModelPath:
    def test_not_cached_raises(self):
        resolved = ResolvedModelPath(path=MODEL_ID, is_local_dir=False, model_id=MODEL_ID)
        with pytest.raises(ValueError, match="not in the AIM model cache"):
            _engine().resolve_model_path(resolved, {}, _profile())

    def test_local_directory_outside_cache(self, tmp_path):
        (tmp_path / "model-q8_0.gguf").touch()
        resolved = ResolvedModelPath(path=str(tmp_path), is_local_dir=False, model_id=str(tmp_path))
        assert _engine().resolve_model_path(resolved, {}, _profile()) == str(tmp_path / "model-q8_0.gguf")

    def test_gguf_file_as_model_id(self, tmp_path):
        gguf = tmp_path / "model-q8_0.gguf"
        gguf.touch()
        resolved = ResolvedModelPath(path=str(gguf), is_local_dir=False, model_id=str(gguf))
        assert _engine().resolve_model_path(resolved, {}, _profile()) == str(gguf)

    def test_single_gguf_selected(self, tmp_path):
        (tmp_path / "model-q8_0.gguf").touch()
        assert _engine().resolve_model_path(_cached(tmp_path), {}, _profile()) == str(tmp_path / "model-q8_0.gguf")

    def test_split_gguf_selects_first_part(self, tmp_path):
        for part in (1, 2, 3):
            (tmp_path / f"model-q8_0-{part:05d}-of-00003.gguf").touch()
        path = _engine().resolve_model_path(_cached(tmp_path), {}, _profile())
        assert path == str(tmp_path / "model-q8_0-00001-of-00003.gguf")

    def test_no_gguf_raises(self, tmp_path):
        with pytest.raises(ValueError, match="No .gguf file"):
            _engine().resolve_model_path(_cached(tmp_path), {}, _profile())

    @pytest.mark.parametrize(
        "precision,expected", [(Precision.BF16, "model-bf16.gguf"), (Precision.INT8, "model-Q8_0.gguf")]
    )
    def test_multiple_gguf_picks_precision(self, tmp_path, precision, expected):
        for name in ("model-bf16.gguf", "model-Q8_0.gguf", "model-f16.gguf", "mmproj-model-f16.gguf"):
            (tmp_path / name).touch()
        path = _engine().resolve_model_path(_cached(tmp_path), {}, _profile(precision))
        assert path == str(tmp_path / expected)

    def test_ambiguous_gguf_raises(self, tmp_path):
        (tmp_path / "model-Q8_0.gguf").touch()
        (tmp_path / "model-Q8_0-imatrix.gguf").touch()
        with pytest.raises(ValueError, match="Multiple .gguf files"):
            _engine().resolve_model_path(_cached(tmp_path), {}, _profile(Precision.INT8))

    @pytest.mark.parametrize("key", ["gguf_filename", "gguf-filename"])
    def test_gguf_filename_picks_file_and_is_stripped(self, tmp_path, key):
        (tmp_path / "model-bf16.gguf").touch()
        (tmp_path / "model-q8_0.gguf").touch()
        engine_args = {key: "model-q8_0.gguf", "threads": 8}
        assert _engine().resolve_model_path(_cached(tmp_path), engine_args, _profile()) == str(
            tmp_path / "model-q8_0.gguf"
        )
        assert engine_args == {"threads": 8}

    def test_gguf_filename_missing_raises(self, tmp_path):
        (tmp_path / "model-bf16.gguf").touch()
        with pytest.raises(ValueError, match="not found"):
            _engine().resolve_model_path(_cached(tmp_path), {"gguf_filename": "other.gguf"}, _profile())


class TestApplyEngineDefaults:
    @pytest.fixture
    def cores(self, monkeypatch):
        def _set(available: int, cpuset: str | None):
            monkeypatch.setattr("aim_runtime.engines.llamacpp.CpuDetector.available_cores", available)
            monkeypatch.setattr("aim_runtime.engines.llamacpp.CpuDetector.cpuset_bind", cpuset)

        return _set

    def test_threads_and_pinning_follow_cpuset(self, cores):
        cores(4, "0-1,4-5")
        engine_args: dict = {}
        _engine().apply_engine_defaults(engine_args, [MODEL_ID])
        assert engine_args == {"alias": MODEL_ID, "threads": 4, "cpu-mask": "33", "cpu-strict": 1}

    def test_no_pinning_without_known_cpuset(self, cores):
        cores(8, None)
        engine_args: dict = {}
        _engine().apply_engine_defaults(engine_args, [MODEL_ID])
        assert engine_args == {"alias": MODEL_ID, "threads": 8}

    def test_explicit_values_kept(self, cores):
        cores(96, "0-95")
        engine_args = {"alias": "custom", "threads": 48, "cpu-range": "0-47"}
        _engine().apply_engine_defaults(engine_args, [MODEL_ID])
        assert engine_args == {"alias": "custom", "threads": 48, "cpu-range": "0-47"}

    def test_alias_lists_all_served_model_names(self, cores):
        cores(8, None)
        engine_args: dict = {}
        _engine().apply_engine_defaults(engine_args, [MODEL_ID, "org/aim-id"])
        assert engine_args["alias"] == f"{MODEL_ID},org/aim-id"


def _fake_converter(calls: list, fail: bool = False):
    def run(command, check):
        calls.append(command)
        open(command[command.index("--outfile") + 1], "w").close()
        if fail:
            raise subprocess.CalledProcessError(1, command)

    return run


class TestHfCheckpointConversion:
    def test_resolves_to_converted_file_named_by_precision(self, tmp_path):
        _hf_checkpoint(tmp_path)
        path = _engine().resolve_model_path(_cached(tmp_path), {}, _profile(Precision.INT8))
        assert path == str(tmp_path / f"{tmp_path.name}-q8_0.gguf")

    def test_existing_converted_file_is_reused(self, tmp_path, monkeypatch):
        _hf_checkpoint(tmp_path)
        (tmp_path / f"{tmp_path.name}-bf16.gguf").touch()
        monkeypatch.setattr("aim_runtime.engines.llamacpp.subprocess.run", pytest.fail)
        _engine().prepare_model(str(tmp_path), _profile())

    def test_prepare_model_runs_converter_once(self, tmp_path, monkeypatch):
        _hf_checkpoint(tmp_path)
        calls: list = []
        monkeypatch.setattr("aim_runtime.engines.llamacpp.subprocess.run", _fake_converter(calls))
        engine = _engine()
        engine.prepare_model(str(tmp_path), _profile())
        engine.prepare_model(str(tmp_path), _profile())

        assert len(calls) == 1
        assert calls[0][1:5] == [CONVERTER_PATH, str(tmp_path), "--outtype", "bf16"]
        assert (tmp_path / f"{tmp_path.name}-bf16.gguf").is_file()
        assert not list(tmp_path.glob("*.partial"))

    def test_failed_conversion_leaves_no_file(self, tmp_path, monkeypatch):
        _hf_checkpoint(tmp_path)
        monkeypatch.setattr("aim_runtime.engines.llamacpp.subprocess.run", _fake_converter([], fail=True))
        with pytest.raises(RuntimeError, match="conversion"):
            _engine().prepare_model(str(tmp_path), _profile())
        assert not list(tmp_path.glob("*.gguf*"))

    def test_pytorch_bin_checkpoint_is_converted(self, tmp_path):
        (tmp_path / "config.json").write_text("{}")
        (tmp_path / "pytorch_model.bin").touch()
        path = _engine().resolve_model_path(_cached(tmp_path), {}, _profile())
        assert path == str(tmp_path / f"{tmp_path.name}-bf16.gguf")

    def test_unsupported_precision_raises(self, tmp_path):
        _hf_checkpoint(tmp_path)
        with pytest.raises(ValueError, match="precision"):
            _engine().resolve_model_path(_cached(tmp_path), {}, _profile(Precision.FP8))

    def test_gguf_filename_wins_over_checkpoint(self, tmp_path):
        _hf_checkpoint(tmp_path)
        (tmp_path / "custom.gguf").touch()
        path = _engine().resolve_model_path(_cached(tmp_path), {"gguf_filename": "custom.gguf"}, _profile())
        assert path == str(tmp_path / "custom.gguf")


class TestDownload:
    """prepare_model downloads a model that is not in the AIM cache, like download-to-cache."""

    @pytest.fixture
    def hub(self, monkeypatch, tmp_path):
        """Fake Hugging Face repo listing and download; records the download filters."""
        state = {"files": [], "downloads": []}

        class FakeApi:
            def __init__(self, token=None):
                pass

            def list_repo_files(self, repo_id):
                return state["files"]

        class FakeRegistry:
            def download(self, model_source, cache_dir, allow_patterns=None, ignore_patterns=None):
                state["downloads"].append((model_source, allow_patterns, ignore_patterns))
                model_dir = tmp_path / model_source.removeprefix("hf://")
                model_dir.mkdir(parents=True, exist_ok=True)
                for name in state["files"]:
                    if allow_patterns and not any(fnmatch.fnmatch(name, p) for p in allow_patterns):
                        continue
                    if ignore_patterns and any(fnmatch.fnmatch(name, p) for p in ignore_patterns):
                        continue
                    (model_dir / name).parent.mkdir(parents=True, exist_ok=True)
                    (model_dir / name).write_text(json.dumps(LLAMA_8B_CONFIG) if name == "config.json" else "")
                return str(model_dir)

        monkeypatch.setattr("huggingface_hub.HfApi", FakeApi)
        monkeypatch.setattr("aim_runtime.model_storage.StorageBackendRegistry", FakeRegistry)
        return state

    def test_gguf_repo_downloads_only_the_precision_file(self, hub, tmp_path):
        hub["files"] = ["README.md", "Model-bf16.gguf", "Model-Q8_0.gguf", "Model-Q4_K_M.gguf"]
        _engine(cache_path=tmp_path).prepare_model("org/Model-GGUF", _profile(Precision.INT8))
        assert hub["downloads"] == [("hf://org/Model-GGUF", ["Model-Q8_0.gguf"], None)]
        assert sorted(p.name for p in (tmp_path / "org/Model-GGUF").iterdir()) == ["Model-Q8_0.gguf"]

    def test_split_gguf_downloads_every_part(self, hub, tmp_path):
        hub["files"] = [f"m-q8_0-0000{i}-of-00002.gguf" for i in (1, 2)] + ["m-bf16.gguf"]
        _engine(cache_path=tmp_path).prepare_model("org/m-GGUF", _profile(Precision.INT8))
        assert hub["downloads"][0][1] == ["m-q8_0-*-of-00002.gguf"]

    def test_gguf_filename_from_engine_args_override(self, hub, tmp_path):
        hub["files"] = ["Model-Q8_0.gguf", "Model-Q6_K.gguf"]
        engine = _engine(cache_path=tmp_path, engine_args_override={"gguf_filename": "Model-Q6_K.gguf"})
        engine.prepare_model("org/Model-GGUF", _profile(Precision.INT8))
        assert hub["downloads"][0][1] == ["Model-Q6_K.gguf"]

    def test_checkpoint_downloaded_without_original_then_converted(self, hub, tmp_path, monkeypatch):
        hub["files"] = ["config.json", "model-00001-of-00001.safetensors", "original/consolidated.00.pth"]
        calls: list = []
        monkeypatch.setattr("aim_runtime.engines.llamacpp.subprocess.run", _fake_converter(calls))
        _engine(cache_path=tmp_path).prepare_model("org/Model", _profile())
        assert hub["downloads"] == [("hf://org/Model", None, ["original/*"])]
        assert not (tmp_path / "org/Model/original").exists()
        assert len(calls) == 1 and (tmp_path / "org/Model/Model-bf16.gguf").is_file()

    def test_cached_model_is_not_downloaded(self, hub, tmp_path):
        (tmp_path / "org/Model-GGUF").mkdir(parents=True)
        (tmp_path / "org/Model-GGUF/Model-Q8_0.gguf").touch()
        _engine(cache_path=tmp_path).prepare_model("org/Model-GGUF", _profile(Precision.INT8))
        assert hub["downloads"] == []

    def test_repo_without_gguf_or_checkpoint_raises(self, hub, tmp_path):
        hub["files"] = ["README.md"]
        with pytest.raises(ValueError, match="neither .gguf files nor a checkpoint"):
            _engine(cache_path=tmp_path).prepare_model("org/Model", _profile())

    def test_missing_local_path_is_not_downloaded(self, hub, tmp_path):
        with pytest.raises(ValueError, match="not a Hugging Face model id"):
            _engine(cache_path=tmp_path).prepare_model(str(tmp_path / "missing"), _profile())
        assert hub["downloads"] == []


class TestKvCacheSpace:
    def _checkpoint(self, path, config=LLAMA_8B_CONFIG):
        _hf_checkpoint(path)
        (path / "config.json").write_text(json.dumps(config))
        return path

    def test_ctx_size_from_hf_config(self, tmp_path):
        self._checkpoint(tmp_path)
        engine_args = {"kv-cache-space": 80, "parallel": 128}
        _engine().resolve_model_path(_cached(tmp_path), engine_args, _profile())
        # 32 layers * 8 KV heads * (128 + 128) * 2 bytes = 128 KiB per token.
        assert engine_args == {"ctx-size": 655360, "parallel": 128}

    def test_quantized_kv_cache_fits_more_tokens(self, tmp_path):
        self._checkpoint(tmp_path)
        engine_args = {"kv_cache_space": 80, "cache-type-k": "q8_0", "cache-type-v": "q8_0"}
        _engine().resolve_model_path(_cached(tmp_path), engine_args, _profile())
        assert engine_args["ctx-size"] == 1233408

    def test_explicit_ctx_size_wins(self, tmp_path):
        self._checkpoint(tmp_path)
        engine_args = {"kv-cache-space": 80, "ctx-size": 8192}
        _engine().resolve_model_path(_cached(tmp_path), engine_args, _profile())
        assert engine_args == {"ctx-size": 8192}

    def test_ctx_size_from_gguf_metadata(self, tmp_path):
        gguf = pytest.importorskip("gguf")
        writer = gguf.GGUFWriter(str(tmp_path / "model.gguf"), "llama")
        writer.add_block_count(32)
        writer.add_embedding_length(4096)
        writer.add_head_count(32)
        writer.add_head_count_kv(8)
        writer.write_header_to_file()
        writer.write_kv_data_to_file()
        writer.write_tensors_to_file()
        writer.close()

        engine_args = {"kv-cache-space": 1}
        _engine().resolve_model_path(_cached(tmp_path), engine_args, _profile())
        assert engine_args == {"ctx-size": 8192}

    def test_unknown_attention_layout_raises_clearly(self, tmp_path):
        self._checkpoint(tmp_path, {"hidden_size": 4096, "num_hidden_layers": 32})
        with pytest.raises(ValueError, match="set ctx-size instead"):
            _engine().resolve_model_path(_cached(tmp_path), {"kv-cache-space": 80}, _profile())

    def test_budget_too_small_raises(self, tmp_path):
        self._checkpoint(tmp_path)
        with pytest.raises(ValueError, match="too small"):
            _engine().resolve_model_path(_cached(tmp_path), {"kv-cache-space": 0.001}, _profile())


class TestShortFlagNames:
    def test_short_names_become_long(self, tmp_path):
        (tmp_path / "model.gguf").touch()
        engine_args = {"ctk": "q8_0", "np": 4, "c": 8192}
        _engine().resolve_model_path(_cached(tmp_path), engine_args, _profile())
        assert engine_args == {"cache-type-k": "q8_0", "parallel": 4, "ctx-size": 8192}

    @pytest.mark.parametrize(
        "engine_args,expected",
        [
            ({"threads": 16, "t": 8}, {"threads": 8}),
            ({"t": 8, "threads": 16}, {"threads": 16}),
            ({"threads_batch": 4, "tb": 2}, {"threads-batch": 2}),
        ],
    )
    def test_later_name_wins_as_in_merge(self, tmp_path, engine_args, expected):
        # The merge puts AIM_ENGINE_ARGS after the profile's keys.
        (tmp_path / "model.gguf").touch()
        _engine().resolve_model_path(_cached(tmp_path), engine_args, _profile())
        assert engine_args == expected

    def test_short_name_values_are_validated(self, tmp_path):
        (tmp_path / "model.gguf").touch()
        with pytest.raises(ValidationError):
            _engine().resolve_model_path(_cached(tmp_path), {"ctk": "bogus"}, _profile())

    def test_short_kv_cache_type_used_for_sizing(self, tmp_path):
        _hf_checkpoint(tmp_path)
        (tmp_path / "config.json").write_text(json.dumps(LLAMA_8B_CONFIG))
        engine_args = {"kv-cache-space": 80, "ctk": "q8_0", "ctv": "q8_0"}
        _engine().resolve_model_path(_cached(tmp_path), engine_args, _profile())
        assert engine_args["ctx-size"] == 1233408
