# Copyright © Advanced Micro Devices, Inc., or its affiliates.
#
# SPDX-License-Identifier: MIT

"""
Unit tests for the wan2.2 / vLLM-Omni diffusion ModelHarness.

These tests load the harness file directly from
``assets/instinct/Wan-AI/Wan2.2-T2V-A14B-Diffusers/image/src/harness.py`` —
the same way ``aim_runtime.harness.discovery.discover_harness()`` loads it
inside a built model image — so they exercise the real on-disk file rather
than a fixture copy.

No live HTTP, no live subprocess: we monkey-patch ``subprocess.run`` to
write a canned ``diffusion_benchmark_serving.py``-shaped JSON to the
requested ``--output-file`` path.  That keeps these tests fast (<1s) and
runnable without a server.
"""

from __future__ import annotations

import importlib.util
import inspect
import json
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from aim_runtime.harness import (
    CheckResultType,
    CheckScope,
    HarnessConfig,
    HarnessResult,
    ModelHarness,
)

WAN22_HARNESS_PATH = (
    Path(__file__).parent.parent.parent / "assets/instinct/Wan-AI/Wan2.2-T2V-A14B-Diffusers/image/src/harness.py"
)
WAN22_SRC_DIR = str(WAN22_HARNESS_PATH.parent)


def _load_wan22_harness_module():
    """Load the wan2.2 harness module the way discovery.py would.

    Inserts the harness's src directory into ``sys.path`` first so the
    sibling ``recipes`` import resolves the same way it does inside the
    container at ``/workspace/model/src``.
    """
    if WAN22_SRC_DIR not in sys.path:
        sys.path.insert(0, WAN22_SRC_DIR)
    spec = importlib.util.spec_from_file_location("_wan22_harness", str(WAN22_HARNESS_PATH))
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def harness_module():
    return _load_wan22_harness_module()


@pytest.fixture
def harness(harness_module) -> ModelHarness:
    for _name, obj in inspect.getmembers(harness_module, inspect.isclass):
        if issubclass(obj, ModelHarness) and obj is not ModelHarness:
            return obj()
    raise RuntimeError("VllmOmniDiffusionHarness not found in harness.py")


@pytest.fixture
def harness_config() -> HarnessConfig:
    return HarnessConfig(
        profile={
            "aim_id": "Wan-AI/Wan2.2-T2V-A14B-Diffusers",
            "model_id": "Wan-AI/Wan2.2-T2V-A14B-Diffusers",
            "engine": "vllm_omni",
        },
        service_url="http://localhost:8000",
        timeout_seconds=900,
    )


# ---------------------------------------------------------------------------
# Discovery / importability
# ---------------------------------------------------------------------------


class TestDiscovery:
    def test_harness_file_exists(self):
        assert WAN22_HARNESS_PATH.is_file(), f"wan2.2 harness file is missing at {WAN22_HARNESS_PATH}"

    def test_module_loads_and_exposes_harness_class(self, harness_module):
        candidates = [
            obj
            for _name, obj in inspect.getmembers(harness_module, inspect.isclass)
            if issubclass(obj, ModelHarness) and obj is not ModelHarness
        ]
        assert len(candidates) == 1, (
            "Expected exactly one ModelHarness subclass in harness.py " f"but found: {[c.__name__ for c in candidates]}"
        )
        assert candidates[0].__name__ == "VllmOmniDiffusionHarness"

    def test_engine_marker(self, harness):
        assert getattr(harness, "ENGINE", None) == "vllm_omni"


# ---------------------------------------------------------------------------
# list_checks
# ---------------------------------------------------------------------------


class TestListChecks:
    EXPECTED_BY_NAME: dict[str, tuple[CheckResultType, CheckScope]] = {
        "health": (CheckResultType.PASS_FAIL, CheckScope.RUNTIME),
        "smoke_video": (CheckResultType.PASS_FAIL, CheckScope.RUNTIME),
        "throughput_qps": (CheckResultType.SCORE, CheckScope.OFFLINE),
        "benchmark_completion": (CheckResultType.PASS_FAIL, CheckScope.OFFLINE),
        "latency_p99_under_slo": (CheckResultType.PASS_FAIL, CheckScope.OFFLINE),
    }

    def test_list_checks_returns_expected_metadata(self, harness):
        checks = harness.list_checks()
        by_name = {c.name: c for c in checks}
        assert set(by_name) == set(
            self.EXPECTED_BY_NAME
        ), f"Unexpected check names: got {sorted(by_name)} expected {sorted(self.EXPECTED_BY_NAME)}"
        for name, (rtype, scope) in self.EXPECTED_BY_NAME.items():
            assert by_name[name].result_type is rtype, f"{name} result_type mismatch"
            assert by_name[name].scope is scope, f"{name} scope mismatch"

    def test_list_checks_returns_fresh_list(self, harness):
        """Returned list should be a copy — caller must not be able to mutate
        the class-level CHECKS list."""
        first = harness.list_checks()
        first.append("not a CheckInfo")
        second = harness.list_checks()
        assert "not a CheckInfo" not in second


# ---------------------------------------------------------------------------
# benchmark() — argv shape + JSON parsing via monkey-patched subprocess.run
# ---------------------------------------------------------------------------


def _make_bench_json(
    *,
    completed: int = 10,
    total: int = 10,
    throughput: float = 0.05,
    latency_mean: float = 22.5,
    latency_p99: float = 24.9,
) -> dict[str, Any]:
    """Return a dict matching diffusion_benchmark_serving.py's output shape."""
    return {
        "duration": completed * latency_mean / max(1, total),
        "completed_requests": completed,
        "throughput_qps": throughput,
        "latency_mean": latency_mean,
        "latency_median": latency_mean,
        "latency_p95": latency_p99 - 0.1,
        "latency_p99": latency_p99,
        "peak_memory_mb_max": 0,
        "peak_memory_mb_mean": 0,
        "peak_memory_mb_median": 0,
        "stage_durations_mean": {},
        "backend": "v1/videos",
        "model": "Wan-AI/Wan2.2-T2V-A14B-Diffusers",
        "dataset": "random",
        "task": "t2v",
    }


class _FakeRunner:
    """Stand-in for ``subprocess.run`` that records its argv and writes a
    canned JSON to the ``--output-file`` requested by the caller."""

    def __init__(self, bench_payload: dict[str, Any], returncode: int = 0):
        self.bench_payload = bench_payload
        self.returncode = returncode
        self.calls: list[list[str]] = []

    def __call__(self, argv, **kwargs):  # noqa: D401 - mimics subprocess.run
        self.calls.append(list(argv))
        if "--output-file" in argv:
            out = argv[argv.index("--output-file") + 1]
            Path(out).write_text(json.dumps(self.bench_payload))
        return SimpleNamespace(
            returncode=self.returncode,
            stdout="(captured stdout)",
            stderr="(captured stderr)",
        )


class TestBenchmarkArgv:
    """Pure-Python tests against ``_build_benchmark_argv``."""

    def test_argv_contains_all_required_flags(self, harness):
        argv = harness._build_benchmark_argv(
            service_url="http://localhost:8000",
            model_id="Wan-AI/Wan2.2-T2V-A14B-Diffusers",
            task="t2v",
            recipe={
                "num_prompts": 7,
                "max_concurrency": 3,
                "random_request_config": [{"width": 854, "height": 480, "num_inference_steps": 18, "weight": 1}],
            },
            output_file="/tmp/out.json",
        )
        # spot-check the important pairs
        pairs = {argv[i]: argv[i + 1] for i in range(0, len(argv) - 1) if argv[i].startswith("--")}
        assert pairs["--backend"] == "v1/videos"
        assert pairs["--base-url"] == "http://localhost:8000"
        assert pairs["--model"] == "Wan-AI/Wan2.2-T2V-A14B-Diffusers"
        assert pairs["--task"] == "t2v"
        assert pairs["--dataset"] == "random"
        assert pairs["--num-prompts"] == "7"
        assert pairs["--max-concurrency"] == "3"
        assert pairs["--output-file"] == "/tmp/out.json"
        # flag-only switches:
        assert "--enable-negative-prompt" in argv
        # script invocation:
        assert argv[1].endswith("diffusion_benchmark_serving.py")

    def test_random_request_config_is_json_encoded(self, harness):
        recipe = {
            "num_prompts": 1,
            "max_concurrency": 1,
            "random_request_config": [{"width": 1280, "height": 720, "weight": 1}],
        }
        argv = harness._build_benchmark_argv(
            service_url="http://localhost:8000",
            model_id="Wan-AI/Wan2.2-T2V-A14B-Diffusers",
            task="t2v",
            recipe=recipe,
            output_file="/tmp/out.json",
        )
        idx = argv.index("--random-request-config")
        decoded = json.loads(argv[idx + 1])
        assert decoded == recipe["random_request_config"]


class TestBenchmarkRun:
    """End-to-end benchmark() with monkey-patched health + subprocess."""

    @pytest.fixture(autouse=True)
    def patch_health(self, harness_module, monkeypatch):
        """Bypass /v1/models health check so tests don't need a server."""
        for _name, cls in inspect.getmembers(harness_module, inspect.isclass):
            if issubclass(cls, ModelHarness) and cls is not ModelHarness:
                monkeypatch.setattr(cls, "health_check", lambda self, *a, **kw: True)

    def test_benchmark_returns_metrics_from_bench_json(
        self, harness_module, harness, harness_config, monkeypatch, tmp_path
    ):
        payload = _make_bench_json(completed=2, total=2, throughput=0.0408, latency_mean=22.5, latency_p99=24.9)
        runner = _FakeRunner(payload)
        monkeypatch.setattr(harness_module.subprocess, "run", runner)

        out_file = tmp_path / "bench.json"
        harness_config.extra["output_file"] = str(out_file)
        harness_config.extra["recipe"] = "smoke"

        result: HarnessResult = harness.benchmark(harness_config)

        assert result.success is True
        assert "throughput_qps" in result.metrics
        assert result.metrics["throughput_qps"] == pytest.approx(0.0408)
        assert result.metrics["latency_p99"] == pytest.approx(24.9)
        assert result.metrics["recipe"] == "smoke"
        assert result.metrics["completed_requests"] == 2

        # exactly one subprocess.run call, hitting the right script & args
        assert len(runner.calls) == 1
        argv = runner.calls[0]
        assert "--backend" in argv
        assert argv[argv.index("--backend") + 1] == "v1/videos"
        assert argv[argv.index("--output-file") + 1] == str(out_file)

        # output file artifact recorded
        assert str(out_file) in result.artifacts

    def test_benchmark_marks_failure_when_subprocess_nonzero(
        self, harness_module, harness, harness_config, monkeypatch
    ):
        runner = _FakeRunner({}, returncode=2)
        monkeypatch.setattr(harness_module.subprocess, "run", runner)

        result = harness.benchmark(harness_config)
        assert result.success is False
        assert "exited with 2" in result.summary

    def test_benchmark_fails_gracefully_when_bench_script_missing(
        self, harness_module, harness, harness_config, monkeypatch
    ):
        """A missing benchmark dir/script (OSError) must not crash aim-runtime."""

        def _raise(*_a, **_kw):
            raise FileNotFoundError(2, "No such file or directory")

        monkeypatch.setattr(harness_module.subprocess, "run", _raise)

        result = harness.benchmark(harness_config)
        assert result.success is False
        assert harness_module.DIFFUSION_BENCH_SCRIPT in result.summary

    def test_benchmark_marks_failure_when_recipe_unknown(self, harness_module, harness, harness_config, monkeypatch):
        runner = _FakeRunner({})
        monkeypatch.setattr(harness_module.subprocess, "run", runner)

        harness_config.extra["recipe"] = "does_not_exist"
        result = harness.benchmark(harness_config)

        assert result.success is False
        assert "Unknown recipe" in result.summary
        assert runner.calls == [], "subprocess.run should not be called on bad recipe"

    def test_benchmark_slo_pass_when_p99_under_slo(
        self, harness_module, harness, harness_config, monkeypatch, tmp_path
    ):
        payload = _make_bench_json(latency_p99=10.0)
        runner = _FakeRunner(payload)
        monkeypatch.setattr(harness_module.subprocess, "run", runner)

        harness_config.extra["output_file"] = str(tmp_path / "bench.json")
        harness_config.extra["slo_seconds"] = 30.0
        result = harness.benchmark(harness_config)

        slo_check = next(c for c in result.checks if c.name == "latency_p99_under_slo")
        assert slo_check.success is True
        assert result.metrics["slo_seconds"] == pytest.approx(30.0)

    def test_benchmark_slo_fails_when_p99_over_slo(
        self, harness_module, harness, harness_config, monkeypatch, tmp_path
    ):
        payload = _make_bench_json(latency_p99=120.0)
        runner = _FakeRunner(payload)
        monkeypatch.setattr(harness_module.subprocess, "run", runner)

        harness_config.extra["output_file"] = str(tmp_path / "bench.json")
        harness_config.extra["slo_seconds"] = 30.0
        result = harness.benchmark(harness_config)

        slo_check = next(c for c in result.checks if c.name == "latency_p99_under_slo")
        assert slo_check.success is False
        assert result.success is False


# ---------------------------------------------------------------------------
# evaluate — stub returns success
# ---------------------------------------------------------------------------


class TestEvaluateStub:
    def test_evaluate_is_a_pass_stub(self, harness, harness_config):
        result = harness.evaluate(harness_config)
        assert result.success is True
        assert "not implemented" in result.summary
        assert len(result.checks) == 1
        assert result.checks[0].name == "evaluate_stub"


# ---------------------------------------------------------------------------
# Recipes — sanity-check the bundled presets
# ---------------------------------------------------------------------------


class TestRecipes:
    @pytest.fixture
    def recipes(self, harness_module):
        return sys.modules["recipes"] if "recipes" in sys.modules else __import__("recipes")

    def test_default_recipe_is_resolvable(self, recipes):
        recipe = recipes.get_recipe(None)
        assert "num_prompts" in recipe
        assert "max_concurrency" in recipe
        assert isinstance(recipe["random_request_config"], list)
        assert recipe["random_request_config"], "default recipe must have at least one entry"

    @pytest.mark.parametrize("name", ["smoke", "dataset_a_480p", "dataset_b_720p", "dataset_c_mix"])
    def test_named_recipes_are_resolvable(self, recipes, name):
        recipe = recipes.get_recipe(name)
        assert recipe["num_prompts"] >= 1
        assert recipe["max_concurrency"] >= 1
        assert sum(item.get("weight", 0) for item in recipe["random_request_config"]) > 0

    def test_unknown_recipe_raises(self, recipes):
        with pytest.raises(KeyError):
            recipes.get_recipe("does_not_exist")
