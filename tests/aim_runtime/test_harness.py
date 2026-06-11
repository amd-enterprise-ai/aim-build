# Copyright © Advanced Micro Devices, Inc., or its affiliates.
#
# SPDX-License-Identifier: MIT

"""
Tests for the ModelHarness ABC, discovery, VLLMHarness, and the echo-model example harness.

These tests verify:
  - The ABC enforces its contract (can't instantiate without all methods)
  - Dataclasses serialize correctly
  - Discovery finds custom harness files and raises RuntimeError when no custom harness exists
  - VLLMHarness is a valid ModelHarness
  - The echo-model example harness works end-to-end against a live service
"""

import importlib.util
import inspect
import json
import subprocess
import sys
import textwrap
import time
from pathlib import Path
from unittest.mock import MagicMock
from urllib.error import URLError
from urllib.request import Request, urlopen

import pytest

from aim_runtime.harness import (
    CheckInfo,
    CheckResult,
    CheckResultType,
    CheckScope,
    HarnessConfig,
    HarnessResult,
    ModelHarness,
)
from aim_runtime.harness.discovery import discover_harness, has_custom_harness

# from aim_runtime.harness.vllm_harness import VLLMHarness


VLLMHarness = MagicMock()

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

ECHO_SERVICE_PATH = Path(__file__).parent.parent.parent / "assets/instinct/example/echo-model/image/src/service.py"
ECHO_HARNESS_PATH = Path(__file__).parent.parent.parent / "assets/instinct/example/echo-model/image/src/harness.py"


ECHO_SRC_DIR = str(ECHO_HARNESS_PATH.parent)

_HAS_BENTOML = importlib.util.find_spec("bentoml") is not None

_requires_bentoml = pytest.mark.skipif(not _HAS_BENTOML, reason="bentoml is required to host the echo service")


def _load_echo_harness() -> ModelHarness:
    """Dynamically load EchoHarness the same way discovery.py would.

    Adds the echo-model's src/ directory to sys.path so the harness can
    do sibling imports (e.g. ``from test_cases import KNOWN_INPUTS``).
    """
    inserted_echo_src_dir = False
    if ECHO_SRC_DIR not in sys.path:
        sys.path.insert(0, ECHO_SRC_DIR)
        inserted_echo_src_dir = True
    try:
        spec = importlib.util.spec_from_file_location("_echo_harness", str(ECHO_HARNESS_PATH))
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        for _name, obj in inspect.getmembers(module, inspect.isclass):
            if issubclass(obj, ModelHarness) and obj is not ModelHarness:
                return obj()
        raise RuntimeError("EchoHarness not found")
    finally:
        if inserted_echo_src_dir:
            sys.path.remove(ECHO_SRC_DIR)


# ---------------------------------------------------------------------------
# ABC contract
# ---------------------------------------------------------------------------


class TestModelHarnessABC:
    """Verify the ABC enforces its interface."""

    def test_cannot_instantiate_abstract(self):
        with pytest.raises(TypeError, match="abstract method"):
            ModelHarness()

    def test_minimal_subclass_must_implement_all(self):
        class IncompleteHarness(ModelHarness):
            def validate(self, config): ...

        with pytest.raises(TypeError, match="abstract method"):
            IncompleteHarness()

    def test_complete_subclass_can_instantiate(self):
        class MinimalHarness(ModelHarness):
            def validate(self, config):
                return HarnessResult(success=True, summary="ok")

            def benchmark(self, config):
                return HarnessResult(success=True, summary="ok")

            def evaluate(self, config):
                return HarnessResult(success=True, summary="ok")

            def list_checks(self):
                return []

        h = MinimalHarness()
        assert isinstance(h, ModelHarness)

    def test_abstract_methods_are_correct_set(self):
        assert ModelHarness.__abstractmethods__ == frozenset({"validate", "benchmark", "evaluate", "list_checks"})


# ---------------------------------------------------------------------------
# Dataclass serialization
# ---------------------------------------------------------------------------


class TestDataclasses:
    """Verify harness dataclasses behave correctly."""

    def test_check_result_type_values(self):
        assert CheckResultType.PASS_FAIL.value == "pass_fail"
        assert CheckResultType.SCORE.value == "score"
        assert CheckResultType.COMPARISON.value == "comparison"

    def test_check_info_frozen(self):
        ci = CheckInfo("test", CheckResultType.PASS_FAIL, CheckScope.RUNTIME, "A test")
        with pytest.raises(AttributeError):
            ci.name = "changed"

    def test_harness_config_defaults(self):
        cfg = HarnessConfig(profile={"key": "val"})
        assert cfg.timeout_seconds == 300
        assert cfg.output_format == "json"
        assert cfg.check_scopes == {CheckScope.RUNTIME, CheckScope.OFFLINE}

    def test_harness_result_to_dict(self):
        result = HarnessResult(
            success=True,
            summary="All good",
            checks=[
                CheckResult("check_a", CheckResultType.PASS_FAIL, True, True, "passed"),
                CheckResult("check_b", CheckResultType.SCORE, True, 42.5, "42.5 tok/s"),
            ],
            metrics={"latency": 1.23},
            artifacts=["/tmp/out.json"],
        )
        d = result.to_dict()

        assert d["success"] is True
        assert d["summary"] == "All good"
        assert len(d["checks"]) == 2
        assert d["checks"][0]["name"] == "check_a"
        assert d["checks"][0]["result_type"] == "pass_fail"
        assert d["checks"][1]["value"] == 42.5
        assert d["metrics"]["latency"] == 1.23
        assert d["artifacts"] == ["/tmp/out.json"]

    def test_harness_result_to_dict_is_json_serializable(self):
        result = HarnessResult(success=False, summary="fail")
        json.dumps(result.to_dict())

    def test_harness_result_defaults(self):
        result = HarnessResult(success=True, summary="ok")
        assert result.checks == []
        assert result.metrics == {}
        assert result.artifacts == []


# ---------------------------------------------------------------------------
# Discovery
# ---------------------------------------------------------------------------


class TestDiscovery:
    """Verify harness discovery logic."""

    def test_has_custom_harness_false_by_default(self):
        """No harness at /workspace/model/src/harness.py in a dev/test env."""
        assert has_custom_harness() is False

    def test_has_custom_harness_true_when_file_exists(self, tmp_path):
        from aim_runtime.harness import discovery as disc_mod

        harness_file = tmp_path / "harness.py"
        harness_file.write_text("# stub")
        original = disc_mod.HARNESS_PATH
        try:
            disc_mod.HARNESS_PATH = harness_file
            assert has_custom_harness() is True
        finally:
            disc_mod.HARNESS_PATH = original

    def test_discover_raises_when_no_harness_file(self):
        """When no custom harness exists, discover_harness raises RuntimeError."""
        with pytest.raises(RuntimeError, match="No custom harness found"):
            discover_harness()

    def test_discover_from_file(self, tmp_path):
        """Discovery loads a ModelHarness subclass from a .py file."""
        from aim_runtime.harness import discovery as disc_mod

        harness_file = tmp_path / "harness.py"
        harness_file.write_text(
            textwrap.dedent(
                """\
            from aim_runtime.harness import ModelHarness, HarnessConfig, HarnessResult, CheckInfo, CheckResultType, CheckScope

            class TestHarness(ModelHarness):
                def validate(self, config):
                    return HarnessResult(success=True, summary="test-validate")
                def benchmark(self, config):
                    return HarnessResult(success=True, summary="test-benchmark")
                def evaluate(self, config):
                    return HarnessResult(success=True, summary="test-evaluate")
                def list_checks(self):
                    return [CheckInfo("dummy", CheckResultType.PASS_FAIL, CheckScope.RUNTIME, "dummy check")]
        """
            )
        )

        original_path = disc_mod.HARNESS_PATH
        try:
            disc_mod.HARNESS_PATH = harness_file
            harness = discover_harness()
            assert type(harness).__name__ == "TestHarness"
            result = harness.validate(HarnessConfig(profile={}))
            assert result.summary == "test-validate"
        finally:
            disc_mod.HARNESS_PATH = original_path

    def test_discover_selects_by_engine(self, tmp_path):
        """When profile specifies an engine, discovery picks the matching harness."""
        from aim_runtime.harness import discovery as disc_mod

        harness_file = tmp_path / "harness.py"
        harness_file.write_text(
            textwrap.dedent(
                """\
            from aim_runtime.harness import ModelHarness, HarnessConfig, HarnessResult, CheckInfo

            class AlphaHarness(ModelHarness):
                ENGINE = "alpha"
                def validate(self, config):
                    return HarnessResult(success=True, summary="alpha-validate")
                def benchmark(self, config):
                    return HarnessResult(success=True, summary="alpha-benchmark")
                def evaluate(self, config):
                    return HarnessResult(success=True, summary="alpha-evaluate")
                def list_checks(self):
                    return []

            class BetaHarness(ModelHarness):
                ENGINE = "beta"
                def validate(self, config):
                    return HarnessResult(success=True, summary="beta-validate")
                def benchmark(self, config):
                    return HarnessResult(success=True, summary="beta-benchmark")
                def evaluate(self, config):
                    return HarnessResult(success=True, summary="beta-evaluate")
                def list_checks(self):
                    return []
        """
            )
        )

        original_path = disc_mod.HARNESS_PATH
        try:
            disc_mod.HARNESS_PATH = harness_file

            harness_a = discover_harness(profile={"engine": "alpha"})
            assert type(harness_a).__name__ == "AlphaHarness"

            harness_b = discover_harness(profile={"engine": "beta"})
            assert type(harness_b).__name__ == "BetaHarness"

            # No profile → picks first candidate
            harness_default = discover_harness()
            assert isinstance(harness_default, ModelHarness)
        finally:
            disc_mod.HARNESS_PATH = original_path

    def test_discover_adds_model_dir_to_sys_path(self, tmp_path):
        """Discovery adds the harness parent directory to sys.path so
        sibling imports (``from helper import func``) work."""
        from aim_runtime.harness import discovery as disc_mod

        # Create a helper module alongside the harness
        helper = tmp_path / "helper.py"
        helper.write_text("MAGIC = 42\n")

        harness_file = tmp_path / "harness.py"
        harness_file.write_text(
            textwrap.dedent(
                """\
            from aim_runtime.harness import ModelHarness, HarnessConfig, HarnessResult, CheckInfo
            from helper import MAGIC

            class SiblingHarness(ModelHarness):
                def validate(self, config):
                    return HarnessResult(success=True, summary=f"magic={MAGIC}")
                def benchmark(self, config):
                    return HarnessResult(success=True, summary="ok")
                def evaluate(self, config):
                    return HarnessResult(success=True, summary="ok")
                def list_checks(self):
                    return []
        """
            )
        )

        original_path = disc_mod.HARNESS_PATH
        original_model_dir = disc_mod.MODEL_DIR
        try:
            disc_mod.HARNESS_PATH = harness_file
            disc_mod.MODEL_DIR = str(tmp_path)
            harness = discover_harness()
            result = harness.validate(HarnessConfig(profile={}))
            assert result.summary == "magic=42"
            assert str(tmp_path) in sys.path
        finally:
            disc_mod.HARNESS_PATH = original_path
            disc_mod.MODEL_DIR = original_model_dir
            if str(tmp_path) in sys.path:
                sys.path.remove(str(tmp_path))

    def test_discover_raises_on_no_subclass(self, tmp_path):
        """Discovery raises RuntimeError if harness.py has no ModelHarness subclass."""
        from aim_runtime.harness import discovery as disc_mod

        harness_file = tmp_path / "harness.py"
        harness_file.write_text("class NotAHarness:\n    pass\n")

        original_path = disc_mod.HARNESS_PATH
        try:
            disc_mod.HARNESS_PATH = harness_file
            with pytest.raises(RuntimeError, match="No ModelHarness subclass"):
                discover_harness()
        finally:
            disc_mod.HARNESS_PATH = original_path


# ---------------------------------------------------------------------------
# VLLMHarness
# ---------------------------------------------------------------------------


@pytest.mark.skip(reason="VLLMHarness is not implemented")
class TestVLLMHarness:
    """Verify VLLMHarness is properly wired."""

    def test_is_model_harness(self):
        assert issubclass(VLLMHarness, ModelHarness)

    def test_list_checks_returns_check_info_instances(self):
        harness = VLLMHarness()
        checks = harness.list_checks()
        assert len(checks) > 0
        assert all(isinstance(c, CheckInfo) for c in checks)

    def test_list_checks_has_runtime_and_offline(self):
        harness = VLLMHarness()
        scopes = {c.scope for c in harness.list_checks()}
        assert CheckScope.RUNTIME in scopes
        assert CheckScope.OFFLINE in scopes

    def test_validate_runtime_scope_without_service(self):
        """Runtime validation with an empty profile should still return a result."""
        harness = VLLMHarness()
        config = HarnessConfig(profile={}, check_scopes={CheckScope.RUNTIME}, timeout_seconds=1)
        result = harness.validate(config)
        assert isinstance(result, HarnessResult)
        check_names = [c.name for c in result.checks]
        assert "profile_schema" in check_names
        assert "engine_validation" in check_names

    def test_benchmark_unreachable_service(self):
        """Benchmark against a non-existent service returns failure."""
        harness = VLLMHarness()
        config = HarnessConfig(profile={"port": 19999}, timeout_seconds=1)
        result = harness.benchmark(config)
        assert result.success is False
        assert "not reachable" in result.summary.lower()


# ---------------------------------------------------------------------------
# Echo-model harness (integration, requires service.py)
# ---------------------------------------------------------------------------


class _EchoServer:
    """Context manager that starts the echo service in a subprocess.

    Launches the service the same way AIM's BentoML engine does — via
    ``bentoml serve`` pointed at the ``service:EchoService`` instance — but
    overrides the bound port with ``--port`` for test isolation.
    """

    def __init__(self, port: int):
        self.port = port
        self.proc = None

    def __enter__(self):
        self.proc = subprocess.Popen(
            [
                sys.executable,
                "-m",
                "bentoml",
                "serve",
                "service:EchoService",
                "--working-dir",
                str(ECHO_SERVICE_PATH.parent),
                "--port",
                str(self.port),
            ],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        # BentoML's worker startup is slower than a stdlib server, so allow
        # a generous window for the /healthz probe to come up.
        deadline = time.time() + 60
        while time.time() < deadline:
            if self.proc.poll() is not None:
                break
            try:
                with urlopen(Request(f"http://localhost:{self.port}/healthz"), timeout=1) as r:
                    if r.status == 200:
                        return self
            except (URLError, OSError):
                pass
            time.sleep(0.25)

        # Startup failed: __exit__ won't be called because __enter__ is raising,
        # so we must clean up the subprocess ourselves to avoid leaking a stray
        # process in CI.
        self._stop_proc()
        raise RuntimeError(f"Echo service did not start on port {self.port}")

    def __exit__(self, *exc):
        self._stop_proc()

    def _stop_proc(self):
        if self.proc is None:
            return
        try:
            if self.proc.poll() is None:
                self.proc.terminate()
                try:
                    self.proc.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    self.proc.kill()
                    self.proc.wait()
        finally:
            self.proc = None


@pytest.fixture(scope="module")
def echo_server():
    """Start the echo service on a high port for the test module."""
    with _EchoServer(port=19876) as server:
        yield server


@pytest.fixture
def echo_harness():
    return _load_echo_harness()


@pytest.fixture
def echo_config(echo_server):
    return HarnessConfig(
        profile={"port": echo_server.port, "benchmark_n_requests": 20},
        timeout_seconds=5,
        check_scopes={CheckScope.RUNTIME, CheckScope.OFFLINE},
    )


@_requires_bentoml
class TestEchoHarness:
    """Integration tests for the play echo-model harness."""

    def test_is_model_harness(self, echo_harness):
        assert isinstance(echo_harness, ModelHarness)

    def test_list_checks(self, echo_harness):
        checks = echo_harness.list_checks()
        names = {c.name for c in checks}
        assert "health" in names
        assert "single_predict" in names
        assert "correctness" in names
        assert "throughput_rps" in names
        assert "batch_predict" in names

    def test_health_check(self, echo_harness, echo_server):
        assert echo_harness.health_check(f"http://localhost:{echo_server.port}", timeout_seconds=3)

    def test_health_check_unreachable(self, echo_harness):
        assert not echo_harness.health_check("http://localhost:19999", timeout_seconds=1)

    def test_validate_full(self, echo_harness, echo_config):
        result = echo_harness.validate(echo_config)
        assert result.success is True
        check_names = [c.name for c in result.checks]
        assert "health" in check_names
        assert "single_predict" in check_names
        assert "batch_predict" in check_names

    def test_validate_runtime_only(self, echo_harness, echo_server):
        config = HarnessConfig(
            profile={"port": echo_server.port},
            timeout_seconds=5,
            check_scopes={CheckScope.RUNTIME},
        )
        result = echo_harness.validate(config)
        assert result.success is True
        check_names = [c.name for c in result.checks]
        assert "health" in check_names
        assert "single_predict" in check_names
        assert "batch_predict" not in check_names

    def test_benchmark(self, echo_harness, echo_config):
        result = echo_harness.benchmark(echo_config)
        assert result.success is True
        assert result.metrics["succeeded"] == result.metrics["total"]
        assert result.metrics["rps"] > 0

    def test_evaluate(self, echo_harness, echo_config):
        result = echo_harness.evaluate(echo_config)
        assert result.success is True
        assert result.metrics["accuracy"] == 1.0
        assert result.metrics["correct"] == result.metrics["total"]

    def test_result_json_serializable(self, echo_harness, echo_config):
        for method in (echo_harness.validate, echo_harness.benchmark, echo_harness.evaluate):
            result = method(echo_config)
            serialized = json.dumps(result.to_dict())
            parsed = json.loads(serialized)
            assert isinstance(parsed["success"], bool)
            assert isinstance(parsed["summary"], str)
