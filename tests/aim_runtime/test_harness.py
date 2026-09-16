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

import csv
import importlib.util
import inspect
import json
import os
import subprocess
import sys
import textwrap
import time
from pathlib import Path
from unittest.mock import patch

import pytest
import requests

from aim_runtime.benchmarking import AIMBenchmark
from aim_runtime.evaluation.results import EvaluationResults, TaskScore
from aim_runtime.evaluation.settings import EvaluationSettings
from aim_runtime.harness import (
    CheckInfo,
    CheckResult,
    CheckResultType,
    CheckScope,
    HarnessConfig,
    HarnessResult,
    ModelHarness,
    failed,
    passed,
)
from aim_runtime.harness.discovery import discover_harness, has_custom_harness
from aim_runtime.harness.vllm_harness import VLLMHarness

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

ECHO_SERVICE_PATH = Path(__file__).parent.parent.parent / "assets/cpu/example/echo-model/image/src/service.py"
ECHO_HARNESS_PATH = Path(__file__).parent.parent.parent / "assets/cpu/example/echo-model/image/src/harness.py"


ECHO_SRC_DIR = str(ECHO_HARNESS_PATH.parent)

_HAS_BENTOML = importlib.util.find_spec("bentoml") is not None

_requires_bentoml = pytest.mark.skipif(not _HAS_BENTOML, reason="bentoml is required to host the echo service")

# The echo integration tests spin up a live bentoml subprocess, which is flaky
# inside the in-container CI unit-test job. Skip only those on GitHub Actions;
# the discovery/unit tests remain hermetic and run everywhere.
_skip_on_ci = pytest.mark.skipif(
    os.environ.get("GITHUB_ACTIONS") == "true",
    reason="Echo integration tests are skipped on GitHub Actions",
)


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
        assert d["checks"][0]["gating"] is True
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

    def test_has_custom_harness_false_when_absent(self, tmp_path):
        """has_custom_harness() is False when no file exists at HARNESS_PATH.

        Pinned to a non-existent path so the result does not depend on whether
        a real harness happens to be installed at /workspace/model/src/harness.py
        (e.g. when running inside a specialized image).
        """
        from aim_runtime.harness import discovery as disc_mod

        original = disc_mod.HARNESS_PATH
        try:
            disc_mod.HARNESS_PATH = tmp_path / "does-not-exist" / "harness.py"
            assert has_custom_harness() is False
        finally:
            disc_mod.HARNESS_PATH = original

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

    def test_discover_falls_back_to_vllm_harness(self, tmp_path):
        """When no custom harness exists, discover_harness returns VLLMHarness.

        Pinned to a non-existent path so it does not load a real harness that
        may be present at /workspace/model/src/harness.py inside an image.
        """
        from aim_runtime.harness import discovery as disc_mod

        original = disc_mod.HARNESS_PATH
        try:
            disc_mod.HARNESS_PATH = tmp_path / "does-not-exist" / "harness.py"
            assert isinstance(discover_harness(), VLLMHarness)
            assert isinstance(discover_harness(profile={"engine": "vllm"}), VLLMHarness)
            assert isinstance(discover_harness(profile={"engine": "vllm_omni"}), VLLMHarness)
        finally:
            disc_mod.HARNESS_PATH = original

    def test_discover_raises_for_unsupported_engine_without_harness(self, tmp_path):
        """A missing harness on a non-vLLM image is an error, not a silent fallback."""
        from aim_runtime.harness import discovery as disc_mod

        original = disc_mod.HARNESS_PATH
        try:
            disc_mod.HARNESS_PATH = tmp_path / "does-not-exist" / "harness.py"
            with pytest.raises(RuntimeError, match="VLLMHarness cannot drive"):
                discover_harness(profile={"engine": "bentoml"})
        finally:
            disc_mod.HARNESS_PATH = original

    def test_discover_falls_back_when_engine_unknown(self, tmp_path):
        """Profile resolution can fail and yield no engine; falling back is still best-effort."""
        from aim_runtime.harness import discovery as disc_mod

        original = disc_mod.HARNESS_PATH
        try:
            disc_mod.HARNESS_PATH = tmp_path / "does-not-exist" / "harness.py"
            assert isinstance(discover_harness(profile={}), VLLMHarness)
            assert isinstance(discover_harness(profile={"profile_id": "x"}), VLLMHarness)
        finally:
            disc_mod.HARNESS_PATH = original

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

    def test_discover_raises_when_no_candidate_declares_engine(self, tmp_path):
        """A multi-harness image with no match for the engine is a wiring bug."""
        from aim_runtime.harness import discovery as disc_mod

        harness_file = tmp_path / "harness.py"
        harness_file.write_text(
            textwrap.dedent(
                """\
            from aim_runtime.harness import ModelHarness, HarnessResult

            class AlphaHarness(ModelHarness):
                ENGINE = "alpha"
                def validate(self, config):
                    return HarnessResult(success=True, summary="alpha")
                def benchmark(self, config):
                    return HarnessResult(success=True, summary="alpha")
                def evaluate(self, config):
                    return HarnessResult(success=True, summary="alpha")
                def list_checks(self):
                    return []

            class BetaHarness(ModelHarness):
                ENGINE = "beta"
                def validate(self, config):
                    return HarnessResult(success=True, summary="beta")
                def benchmark(self, config):
                    return HarnessResult(success=True, summary="beta")
                def evaluate(self, config):
                    return HarnessResult(success=True, summary="beta")
                def list_checks(self):
                    return []
        """
            )
        )

        original_path = disc_mod.HARNESS_PATH
        try:
            disc_mod.HARNESS_PATH = harness_file
            with pytest.raises(RuntimeError, match="none declares engine 'gamma'"):
                discover_harness(profile={"engine": "gamma"})
        finally:
            disc_mod.HARNESS_PATH = original_path

    def test_shipped_harness_classes_are_not_candidates(self, tmp_path):
        """A harness may subclass VLLMHarness without the import outranking it.

        ``inspect.getmembers`` sorts alphabetically, so a name after
        'VLLMHarness' would lose to the imported base class.
        """
        from aim_runtime.harness import discovery as disc_mod

        harness_file = tmp_path / "harness.py"
        harness_file.write_text(
            textwrap.dedent(
                """\
            from aim_runtime.harness.vllm_harness import VLLMHarness

            class ZebraHarness(VLLMHarness):
                pass
        """
            )
        )

        original_path = disc_mod.HARNESS_PATH
        try:
            disc_mod.HARNESS_PATH = harness_file
            assert type(discover_harness()).__name__ == "ZebraHarness"
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

    def test_discover_prefers_leaf_over_imported_base(self, tmp_path):
        """A base class imported from a sibling must not shadow the leaf.

        The base sorts alphabetically before the leaf, so an implementation
        that merely took the first candidate would pick the shared base.
        """
        from aim_runtime.harness import discovery as disc_mod

        base = tmp_path / "base_harness.py"
        base.write_text(
            textwrap.dedent(
                """\
            from aim_runtime.harness import ModelHarness, HarnessResult

            class AbcSharedHarness(ModelHarness):
                def validate(self, config):
                    return HarnessResult(success=True, summary="base")
                def benchmark(self, config):
                    return HarnessResult(success=True, summary="base")
                def evaluate(self, config):
                    return HarnessResult(success=True, summary="base")
                def list_checks(self):
                    return []
        """
            )
        )

        harness_file = tmp_path / "harness.py"
        harness_file.write_text(
            textwrap.dedent(
                """\
            from base_harness import AbcSharedHarness
            from aim_runtime.harness import HarnessResult

            class ZLeafHarness(AbcSharedHarness):
                def validate(self, config):
                    return HarnessResult(success=True, summary="leaf")
        """
            )
        )

        original_path = disc_mod.HARNESS_PATH
        original_model_dir = disc_mod.MODEL_DIR
        try:
            disc_mod.HARNESS_PATH = harness_file
            disc_mod.MODEL_DIR = str(tmp_path)
            harness = discover_harness()
            assert type(harness).__name__ == "ZLeafHarness"
            assert harness.validate(HarnessConfig(profile={})).summary == "leaf"
        finally:
            disc_mod.HARNESS_PATH = original_path
            disc_mod.MODEL_DIR = original_model_dir
            if str(tmp_path) in sys.path:
                sys.path.remove(str(tmp_path))

    def test_discover_keeps_engine_declaring_base_selectable(self, tmp_path):
        """A base that declares ENGINE stays selectable in a multi-engine image.

        Dropping it as scaffolding would leave one candidate, skip engine
        dispatch, and silently return the subclass for every engine.
        """
        from aim_runtime.harness import discovery as disc_mod

        harness_file = tmp_path / "harness.py"
        harness_file.write_text(
            textwrap.dedent(
                """\
            from aim_runtime.harness import ModelHarness, HarnessResult

            class VllmModelHarness(ModelHarness):
                ENGINE = "vllm"
                def validate(self, config):
                    return HarnessResult(success=True, summary="vllm")
                def benchmark(self, config):
                    return HarnessResult(success=True, summary="vllm")
                def evaluate(self, config):
                    return HarnessResult(success=True, summary="vllm")
                def list_checks(self):
                    return []

            class BentoHarness(VllmModelHarness):
                ENGINE = "bentoml"
                def validate(self, config):
                    return HarnessResult(success=True, summary="bentoml")
        """
            )
        )

        original_path = disc_mod.HARNESS_PATH
        try:
            disc_mod.HARNESS_PATH = harness_file

            base = discover_harness(profile={"engine": "vllm"})
            assert type(base).__name__ == "VllmModelHarness"

            leaf = discover_harness(profile={"engine": "bentoml"})
            assert type(leaf).__name__ == "BentoHarness"
        finally:
            disc_mod.HARNESS_PATH = original_path

    def test_discover_selects_by_supported_engines(self, tmp_path):
        """A VLLMHarness subclass is selectable via inherited SUPPORTED_ENGINES.

        VLLMHarness declares SUPPORTED_ENGINES and no ENGINE, so matching on
        ENGINE alone left it unselectable for 'vllm' in a multi-harness file.
        """
        from aim_runtime.harness import discovery as disc_mod

        harness_file = tmp_path / "harness.py"
        harness_file.write_text(
            textwrap.dedent(
                """\
            from aim_runtime.harness import ModelHarness, HarnessResult
            from aim_runtime.harness.vllm_harness import VLLMHarness

            class ZVllmChildHarness(VLLMHarness):
                def validate(self, config):
                    return HarnessResult(success=True, summary="vllm-child")

            class BentoHarness(ModelHarness):
                ENGINE = "bentoml"
                def validate(self, config):
                    return HarnessResult(success=True, summary="bentoml")
                def benchmark(self, config):
                    return HarnessResult(success=True, summary="bentoml")
                def evaluate(self, config):
                    return HarnessResult(success=True, summary="bentoml")
                def list_checks(self):
                    return []
        """
            )
        )

        original_path = disc_mod.HARNESS_PATH
        try:
            disc_mod.HARNESS_PATH = harness_file

            vllm = discover_harness(profile={"engine": "vllm"})
            assert type(vllm).__name__ == "ZVllmChildHarness"

            bento = discover_harness(profile={"engine": "bentoml"})
            assert type(bento).__name__ == "BentoHarness"
        finally:
            disc_mod.HARNESS_PATH = original_path

    def test_discover_own_engine_declaration_narrows_inherited(self, tmp_path):
        """Declaring ENGINE replaces the engines a class inherits.

        The subclass inherits SUPPORTED_ENGINES={vllm, vllm-omni} from its base.
        Treating that as additive would match it for every engine and, being the
        most-derived candidate, silently hand it every profile.
        """
        from aim_runtime.harness import discovery as disc_mod

        harness_file = tmp_path / "harness.py"
        harness_file.write_text(
            textwrap.dedent(
                """\
            from typing import ClassVar

            from aim_runtime.harness import ModelHarness, HarnessResult

            class VllmStyleHarness(ModelHarness):
                SUPPORTED_ENGINES: ClassVar[frozenset[str]] = frozenset({"vllm", "vllm-omni"})
                def validate(self, config):
                    return HarnessResult(success=True, summary="vllm")
                def benchmark(self, config):
                    return HarnessResult(success=True, summary="vllm")
                def evaluate(self, config):
                    return HarnessResult(success=True, summary="vllm")
                def list_checks(self):
                    return []

            class BentoHarness(VllmStyleHarness):
                ENGINE = "bentoml"
                def validate(self, config):
                    return HarnessResult(success=True, summary="bentoml")
        """
            )
        )

        original_path = disc_mod.HARNESS_PATH
        try:
            disc_mod.HARNESS_PATH = harness_file

            for engine in ("vllm", "vllm-omni"):
                harness = discover_harness(profile={"engine": engine})
                assert type(harness).__name__ == "VllmStyleHarness", engine

            bento = discover_harness(profile={"engine": "bentoml"})
            assert type(bento).__name__ == "BentoHarness"
        finally:
            disc_mod.HARNESS_PATH = original_path

    def test_discover_engine_match_prefers_subclass(self, tmp_path):
        """When a subclass inherits its base's engine, the subclass wins."""
        from aim_runtime.harness import discovery as disc_mod

        harness_file = tmp_path / "harness.py"
        harness_file.write_text(
            textwrap.dedent(
                """\
            from aim_runtime.harness import ModelHarness, HarnessResult

            class AbcBaseHarness(ModelHarness):
                ENGINE = "bentoml"
                def validate(self, config):
                    return HarnessResult(success=True, summary="base")
                def benchmark(self, config):
                    return HarnessResult(success=True, summary="base")
                def evaluate(self, config):
                    return HarnessResult(success=True, summary="base")
                def list_checks(self):
                    return []

            class ZChildHarness(AbcBaseHarness):
                def validate(self, config):
                    return HarnessResult(success=True, summary="child")
        """
            )
        )

        original_path = disc_mod.HARNESS_PATH
        try:
            disc_mod.HARNESS_PATH = harness_file
            harness = discover_harness(profile={"engine": "bentoml"})
            assert type(harness).__name__ == "ZChildHarness"
            assert harness.validate(HarnessConfig(profile={})).summary == "child"
        finally:
            disc_mod.HARNESS_PATH = original_path

    def test_discover_accepts_harness_imported_from_sibling(self, tmp_path):
        """A harness defined in a sibling and re-exported by harness.py works."""
        from aim_runtime.harness import discovery as disc_mod

        sibling = tmp_path / "impl.py"
        sibling.write_text(
            textwrap.dedent(
                """\
            from aim_runtime.harness import ModelHarness, HarnessResult

            class SiblingDefinedHarness(ModelHarness):
                def validate(self, config):
                    return HarnessResult(success=True, summary="from-sibling")
                def benchmark(self, config):
                    return HarnessResult(success=True, summary="ok")
                def evaluate(self, config):
                    return HarnessResult(success=True, summary="ok")
                def list_checks(self):
                    return []
        """
            )
        )

        harness_file = tmp_path / "harness.py"
        harness_file.write_text("from impl import SiblingDefinedHarness\n")

        original_path = disc_mod.HARNESS_PATH
        original_model_dir = disc_mod.MODEL_DIR
        try:
            disc_mod.HARNESS_PATH = harness_file
            disc_mod.MODEL_DIR = str(tmp_path)
            harness = discover_harness()
            assert type(harness).__name__ == "SiblingDefinedHarness"
            assert harness.validate(HarnessConfig(profile={})).summary == "from-sibling"
        finally:
            disc_mod.HARNESS_PATH = original_path
            disc_mod.MODEL_DIR = original_model_dir
            if str(tmp_path) in sys.path:
                sys.path.remove(str(tmp_path))

    def test_discover_ignores_imported_vllm_harness(self, tmp_path):
        """An imported VLLMHarness never competes with the image's own harness."""
        from aim_runtime.harness import discovery as disc_mod

        harness_file = tmp_path / "harness.py"
        harness_file.write_text(
            textwrap.dedent(
                """\
            from aim_runtime.harness import ModelHarness, HarnessResult
            from aim_runtime.harness.vllm_harness import VLLMHarness

            class AaaOwnHarness(ModelHarness):
                def validate(self, config):
                    return HarnessResult(success=True, summary="own")
                def benchmark(self, config):
                    return HarnessResult(success=True, summary="own")
                def evaluate(self, config):
                    return HarnessResult(success=True, summary="own")
                def list_checks(self):
                    return []
        """
            )
        )

        original_path = disc_mod.HARNESS_PATH
        try:
            disc_mod.HARNESS_PATH = harness_file
            harness = discover_harness()
            assert type(harness).__name__ == "AaaOwnHarness"
        finally:
            disc_mod.HARNESS_PATH = original_path

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

    def test_benchmark_exports_suite_artifacts(self, tmp_path):
        """The suite's own JSON and CSV are written, and reported as artifacts.

        The async feedback workflow reads its metrics from the CSV alone, so a
        run that produces no CSV records nothing.
        """
        raw_results = {
            "overall_success": True,
            "model_name": "test-model",
            "benchmark_configs": [{"config_name": "c1", "success": True, "output_tok_throughput": 42.0}],
        }
        harness = VLLMHarness()
        config = HarnessConfig(profile={}, service_url="http://svc:8000", extra={"output_dir": str(tmp_path)})

        with (
            patch.object(AIMBenchmark, "run_benchmark_suite", return_value=raw_results),
            patch.object(VLLMHarness, "health_check", return_value=True),
        ):
            result = harness.benchmark(config)

        assert (tmp_path / "benchmark_results.json").exists()
        csv_rows = list(csv.DictReader((tmp_path / "benchmark_results.csv").read_text().splitlines()))
        assert [row["config_name"] for row in csv_rows] == ["c1"]
        assert sorted(Path(a).name for a in result.artifacts) == [
            "benchmark_results.csv",
            "benchmark_results.json",
        ]

    def test_benchmark_honors_results_filename_overrides(self, tmp_path, monkeypatch):
        """BENCHMARK_{JSON,CSV}_FILE still name the suite's output files."""
        monkeypatch.setenv("BENCHMARK_JSON_FILE", "custom_results.json")
        monkeypatch.setenv("BENCHMARK_CSV_FILE", "custom_results.csv")

        harness = VLLMHarness()
        config = HarnessConfig(profile={}, service_url="http://svc:8000", extra={"output_dir": str(tmp_path)})

        with (
            patch.object(AIMBenchmark, "run_benchmark_suite", return_value={"overall_success": True}),
            patch.object(VLLMHarness, "health_check", return_value=True),
        ):
            harness.benchmark(config)

        assert (tmp_path / "custom_results.json").exists()
        assert (tmp_path / "custom_results.csv").exists()

    def test_benchmark_metrics_carry_per_config_latency(self):
        """Latency percentiles survive into metrics; checks only carry throughput."""
        bench_config = {
            "config_name": "c1",
            "success": True,
            "output_tok_throughput": 42.0,
            "mean_ttft": 120.5,
            "p99_e2el": 3400.0,
        }
        harness = VLLMHarness()
        config = HarnessConfig(profile={}, service_url="http://svc:8000")

        with (
            patch.object(
                AIMBenchmark,
                "run_benchmark_suite",
                return_value={"overall_success": True, "benchmark_configs": [bench_config]},
            ),
            patch.object(VLLMHarness, "health_check", return_value=True),
        ):
            result = harness.benchmark(config)

        assert result.metrics["benchmark_configs"] == [bench_config]
        assert result.metrics["configs_run"] == 1

    def test_benchmark_forwards_profile_engine_args(self):
        """The suite gets the profile's engine args, so it can match the served engine.

        Without them AIMBenchmark would re-select the profile itself, which the
        harness has already done.
        """
        engine_args = {"tokenizer-mode": "mistral", "trust-remote-code": None}
        harness = VLLMHarness()
        config = HarnessConfig(profile={"engine_args": engine_args}, service_url="http://svc:8000")

        with (
            patch("aim_runtime.benchmarking.AIMBenchmark") as mock_cls,
            patch.object(VLLMHarness, "health_check", return_value=True),
        ):
            mock_cls.return_value.run_benchmark_suite.return_value = {"overall_success": True}
            harness.benchmark(config)

        assert mock_cls.call_args.kwargs["engine_args"] == engine_args

    def test_benchmark_without_output_dir_skips_export(self):
        """No output directory means no artifacts, not a crash."""
        harness = VLLMHarness()
        config = HarnessConfig(profile={}, service_url="http://svc:8000")

        with (
            patch.object(AIMBenchmark, "run_benchmark_suite", return_value={"overall_success": True}),
            patch.object(AIMBenchmark, "export_results") as mock_export,
            patch.object(VLLMHarness, "health_check", return_value=True),
        ):
            result = harness.benchmark(config)

        mock_export.assert_not_called()
        assert result.artifacts == []


class TestVLLMHarnessServiceValidation:
    """Verify how validate() sequences and gates the live-service checks.

    The checks themselves are covered in test_service_checks.py; these tests
    stub them out and assert on orchestration — ordering, short-circuiting,
    capability gating, and the metrics that CI reads.
    """

    @pytest.fixture
    def stub_checks(self, mocker):
        """Replace every service check with a passing stub."""
        from aim_runtime.harness import service_checks
        from aim_runtime.harness import vllm_harness as vh

        stubs = {
            "probe_api_health": mocker.patch.object(
                vh,
                "probe_api_health",
                return_value=service_checks.HealthProbe(
                    check=passed("api_health", "ready"),
                    model_id="stub-model",
                    ready_time_seconds=12.5,
                ),
            ),
            "run_warmup": mocker.patch.object(
                vh,
                "run_warmup",
                return_value=service_checks.WarmupOutcome(
                    check=passed("warmup", "warm", gating=False),
                    elapsed_seconds=31.4,
                    attempts=2,
                    jit_suspected=True,
                ),
            ),
        }
        for name in (
            "check_completions_endpoint",
            "check_chat_completions_endpoint",
            "check_tool_invocation",
            "check_tool_avoidance",
            "check_structured_output",
            "check_structured_output_nested",
            "check_structured_output_choice",
            "check_reasoning",
        ):
            check_name = name.removeprefix("check_")
            stubs[name] = mocker.patch.object(vh, name, return_value=passed(check_name, "ok"))
        return stubs

    @staticmethod
    def _config(capabilities=None, **extra):
        metadata = {
            "engine": "vllm",
            "precision": "fp8",
            "accelerator_type": "gpu",
            "accelerator_model": "mi300x",
            "accelerator_count": 1,
            "metric": "latency",
            "type": "optimized",
        }
        if capabilities:
            metadata["capabilities"] = capabilities
        profile = {"engine": "vllm", "metadata": metadata, "env_vars": {}, "engine_args": {}}
        return HarnessConfig(
            profile=profile,
            service_url="http://stub:8000",
            check_scopes={CheckScope.RUNTIME, CheckScope.OFFLINE},
            extra=extra,
        )

    @staticmethod
    def _by_name(result):
        return {c.name: c for c in result.checks}

    def test_runs_all_checks_when_every_capability_is_declared(self, stub_checks):
        capabilities = {"tool_calling": True, "structured_outputs": True, "reasoning": True}

        result = VLLMHarness().validate(self._config(capabilities))

        assert result.success
        assert {c.name for c in result.checks} == {info.name for info in VLLMHarness.CHECKS}
        assert not any(c.skipped for c in result.checks)

    def test_undeclared_capabilities_are_skipped_not_failed(self, stub_checks):
        result = VLLMHarness().validate(self._config())

        checks = self._by_name(result)
        assert result.success
        for name in (
            "tool_invocation",
            "tool_avoidance",
            "structured_output",
            "structured_output_nested",
            "structured_output_choice",
            "reasoning",
        ):
            assert checks[name].skipped, f"{name} should be skipped"
        stub_checks["check_tool_invocation"].assert_not_called()
        stub_checks["check_reasoning"].assert_not_called()

    def test_offline_scope_skips_the_runtime_checks(self, stub_checks):
        config = self._config()
        config.check_scopes = {CheckScope.OFFLINE}

        names = {c.name for c in VLLMHarness().validate(config).checks}

        assert "profile_schema" not in names and "engine_validation" not in names
        assert "api_health" in names

    def test_capabilities_are_gated_independently(self, stub_checks):
        result = VLLMHarness().validate(self._config({"reasoning": True}))

        checks = self._by_name(result)
        assert not checks["reasoning"].skipped
        assert checks["tool_invocation"].skipped
        assert checks["structured_output"].skipped

    def test_config_override_enables_a_capability(self, stub_checks):
        """CI overrides the profile flag the same way (REASONING_ENABLED)."""
        result = VLLMHarness().validate(self._config(reasoning=True))

        assert not self._by_name(result)["reasoning"].skipped
        stub_checks["check_reasoning"].assert_called_once()

    def test_chat_completions_check_receives_declared_reasoning_capability(self, stub_checks):
        """A null-content chat response is only forgiven when reasoning is declared."""
        VLLMHarness().validate(self._config({"reasoning": True}))

        stub_checks["check_chat_completions_endpoint"].assert_called_once_with(
            "http://stub:8000", "stub-model", timeout=300.0, reasoning_enabled=True
        )

    def test_chat_completions_check_defaults_reasoning_to_false_when_undeclared(self, stub_checks):
        VLLMHarness().validate(self._config())

        stub_checks["check_chat_completions_endpoint"].assert_called_once_with(
            "http://stub:8000", "stub-model", timeout=300.0, reasoning_enabled=False
        )

    def test_timeout_bounds_every_service_request(self, stub_checks):
        """--timeout governs the behavioural checks, not just readiness."""
        config = self._config({"tool_calling": True, "structured_outputs": True, "reasoning": True})
        config.timeout_seconds = 45

        VLLMHarness().validate(config)

        stub_checks["probe_api_health"].assert_called_once_with("http://stub:8000", timeout_seconds=45)
        for name, stub in stub_checks.items():
            if name == "check_chat_completions_endpoint":
                stub.assert_called_once_with("http://stub:8000", "stub-model", timeout=45.0, reasoning_enabled=True)
            elif name.startswith("check_"):
                stub.assert_called_once_with("http://stub:8000", "stub-model", timeout=45.0)

    def test_startup_metrics_are_reported(self, stub_checks):
        result = VLLMHarness().validate(self._config())

        assert result.metrics["ready_time_seconds"] == 12.5
        assert result.metrics["warmup_time_seconds"] == 31.4
        assert result.metrics["warmup_attempts"] == 2
        assert result.metrics["warmup_succeeded"] is True
        assert result.metrics["jit_suspected"] is True

    def test_unready_service_skips_everything_downstream(self, stub_checks, mocker):
        from aim_runtime.harness import service_checks
        from aim_runtime.harness import vllm_harness as vh

        mocker.patch.object(
            vh,
            "probe_api_health",
            return_value=service_checks.HealthProbe(check=failed("api_health", "no model")),
        )

        result = VLLMHarness().validate(self._config())

        assert not result.success
        assert "api_health" in result.summary
        stub_checks["run_warmup"].assert_not_called()
        stub_checks["check_completions_endpoint"].assert_not_called()

    def test_failed_warmup_skips_behavioural_checks(self, stub_checks, mocker):
        from aim_runtime.harness import service_checks
        from aim_runtime.harness import vllm_harness as vh

        mocker.patch.object(
            vh,
            "run_warmup",
            return_value=service_checks.WarmupOutcome(
                check=failed("warmup", "no inference"),
                elapsed_seconds=300.0,
                attempts=60,
                jit_suspected=False,
            ),
        )

        result = VLLMHarness().validate(self._config({"tool_calling": True}))

        assert not result.success
        stub_checks["check_completions_endpoint"].assert_not_called()
        stub_checks["check_tool_invocation"].assert_not_called()
        assert result.metrics["warmup_succeeded"] is False

    def test_broken_endpoint_skips_capability_checks(self, stub_checks, mocker):
        from aim_runtime.harness import vllm_harness as vh

        mocker.patch.object(
            vh,
            "check_chat_completions_endpoint",
            return_value=failed("chat_completions_endpoint", "missing content"),
        )

        result = VLLMHarness().validate(self._config({"tool_calling": True}))

        assert not result.success
        stub_checks["check_tool_invocation"].assert_not_called()

    def test_warnings_are_surfaced_without_failing(self, stub_checks, mocker):
        from aim_runtime.harness import vllm_harness as vh

        mocker.patch.object(
            vh,
            "check_tool_invocation",
            return_value=passed("tool_invocation", "ok", warnings=["Model did not call a tool"]),
        )

        result = VLLMHarness().validate(self._config({"tool_calling": True}))

        assert result.success
        assert "warning" in result.summary
        assert self._by_name(result)["tool_invocation"].warnings == ["Model did not call a tool"]

    def test_skipped_checks_serialize_for_ci(self, stub_checks):
        result = VLLMHarness().validate(self._config())

        reasoning = next(c for c in result.to_dict()["checks"] if c["name"] == "reasoning")
        assert reasoning["skipped"] is True
        assert reasoning["gating"] is True
        assert reasoning["warnings"] == []

    def test_non_gating_checks_serialize_for_ci(self, stub_checks):
        result = VLLMHarness().validate(self._config())

        by_name = {c["name"]: c for c in result.to_dict()["checks"]}
        assert by_name["warmup"]["gating"] is False
        assert by_name["profile_schema"]["gating"] is False
        assert by_name["engine_validation"]["gating"] is False

    def test_empty_check_list_reports_that_nothing_ran(self):
        """An empty run fails, and says so rather than "0 of 0 ... failed"."""
        success, summary = VLLMHarness._summarize([])

        assert not success
        assert summary == "No validation checks ran"

    def test_raising_check_fails_only_itself(self, stub_checks, mocker):
        """A check that blows up must not take the rest of the run with it."""
        from aim_runtime.harness import vllm_harness as vh

        mocker.patch.object(
            vh,
            "check_structured_output",
            side_effect=AttributeError("'str' object has no attribute 'get'"),
        )

        capabilities = {"tool_calling": True, "structured_outputs": True, "reasoning": True}
        result = VLLMHarness().validate(self._config(capabilities))

        checks = self._by_name(result)
        assert not result.success
        assert "AttributeError" in checks["structured_output"].detail
        # Everything ordered after the raising check still ran and reported.
        assert checks["structured_output_nested"].success
        assert checks["reasoning"].success
        assert {info.name for info in VLLMHarness.CHECKS} == set(checks)

    def test_raising_startup_phase_keeps_earlier_results(self, stub_checks, mocker):
        """An exception outside a leaf check still returns what already ran."""
        from aim_runtime.harness import vllm_harness as vh

        mocker.patch.object(vh, "run_warmup", side_effect=KeyError("choices"))

        result = VLLMHarness().validate(self._config())

        checks = self._by_name(result)
        assert not result.success
        assert checks["api_health"].success
        assert result.metrics["ready_time_seconds"] == 12.5
        assert "KeyError" in checks["service_checks"].detail


def _profile(
    *,
    aim_id="org/model",
    model_id="org/model",
    accelerator_type="gpu",
    accelerator_count=1,
    engine_args=None,
):
    """A resolved profile dict, shaped as the entrypoint hands one to the harness."""
    return {
        "aim_id": aim_id,
        "model_id": model_id,
        "engine": "vllm",
        "engine_args": {} if engine_args is None else engine_args,
        "env_vars": {},
        "metadata": {
            "engine": "vllm",
            "accelerator_type": accelerator_type,
            "accelerator_model": "mi300x",
            "accelerator_count": accelerator_count,
            "precision": "fp16",
            "metric": "latency",
            "type": "optimized",
        },
        "port": 8000,
    }


def _score(**overrides):
    """A full gsm8k record, the case the evaluation path is tuned for."""
    fields = {
        "task": "gsm8k",
        "dataset": "openai/gsm8k/main",
        "metric": "exact_match",
        "metric_variant": "flexible-extract",
        "value": 0.8317,
        "stderr": 0.0103,
        "num_samples": 1319,
        "num_samples_available": 1319,
        "num_fewshot": 5,
        "higher_is_better": True,
    }
    return TaskScore(**{**fields, **overrides})


def _results(*scores, failure_reason=None, document=None):
    return EvaluationResults(
        settings=EvaluationSettings(model="org/model", service_url="http://svc:8000"),
        tasks=scores,
        backend_version="0.4.12",
        elapsed_seconds=12.5,
        failure_reason=failure_reason,
        document=document,
    )


class TestVLLMHarnessEvaluate:
    """Verify how evaluate() drives AIMEvaluation and maps its results back.

    The evaluation itself is covered under tests/aim_runtime/evaluation/; these
    tests stub the runner out and assert on the two translations evaluate() owns:
    profile and ``--config`` into settings, and results into a HarnessResult.
    """

    @pytest.fixture
    def stub_runner(self, mocker):
        """Replace AIMEvaluation with a recording stub: no backend, no service."""
        from aim_runtime.evaluation import runner as runner_module

        stub = mocker.patch.object(runner_module, "AIMEvaluation")
        stub.return_value.run_evaluation_suite.return_value = _results(_score())
        stub.return_value.export_results.return_value = []
        return stub

    @staticmethod
    def _config(profile=None, service_url="http://svc:8000", **extra):
        return HarnessConfig(
            profile=_profile() if profile is None else profile,
            service_url=service_url,
            extra=extra,
        )

    @staticmethod
    def _settings_of(stub):
        """The settings the harness built, as passed to the runner."""
        return stub.call_args.args[0]

    @staticmethod
    def _by_name(result):
        return {c.name: c for c in result.checks}

    # -- results mapping ---------------------------------------------------- #

    def test_headline_score_becomes_the_accuracy_check_and_the_metrics(self, stub_runner):
        result = VLLMHarness().evaluate(self._config())

        accuracy = self._by_name(result)["accuracy"]
        assert result.success
        assert accuracy.result_type is CheckResultType.SCORE
        assert accuracy.value == 0.8317
        assert accuracy.detail == "gsm8k exact_match (flexible-extract): 0.8317 over 1319/1319 samples, 5-shot"
        assert result.metrics["accuracy"] == 0.8317
        assert result.metrics["primary_task"] == "gsm8k"
        assert result.metrics["evaluation"]["tasks"][0]["dataset"] == "openai/gsm8k/main"

    def test_single_task_reports_only_the_headline_check(self, stub_runner):
        """A second check would restate the same number under another name."""
        result = VLLMHarness().evaluate(self._config())

        assert [c.name for c in result.checks] == ["accuracy"]

    def test_multi_task_run_reports_a_check_per_task(self, stub_runner):
        stub_runner.return_value.run_evaluation_suite.return_value = _results(
            _score(),
            _score(task="mmlu", metric="acc", metric_variant=None, value=0.71, num_samples=14042),
        )

        result = VLLMHarness().evaluate(self._config())

        checks = self._by_name(result)
        assert [c.name for c in result.checks] == ["accuracy", "eval_gsm8k", "eval_mmlu"]
        assert checks["eval_mmlu"].value == 0.71
        assert checks["accuracy"].value == checks["eval_gsm8k"].value

    def test_no_score_check_gates(self, stub_runner):
        """A score needs a baseline to judge; the verdict is not this run's to make."""
        stub_runner.return_value.run_evaluation_suite.return_value = _results(
            _score(),
            _score(task="mmlu", metric="acc", metric_variant=None, value=0.71, num_samples=14042),
        )

        result = VLLMHarness().evaluate(self._config())

        assert not any(check.gating for check in result.checks)

    def test_failed_run_reports_its_reason_and_no_number(self, stub_runner):
        stub_runner.return_value.run_evaluation_suite.return_value = _results(failure_reason="lm_eval exited 1")

        result = VLLMHarness().evaluate(self._config())

        accuracy = self._by_name(result)["accuracy"]
        assert not result.success
        assert accuracy.value is None
        assert accuracy.detail == "lm_eval exited 1"
        assert "lm_eval exited 1" in result.summary

    def test_completed_run_without_a_score_is_not_a_success(self, stub_runner):
        """The harness has nothing to compare, even though CI records a NULL row."""
        stub_runner.return_value.run_evaluation_suite.return_value = _results()

        result = VLLMHarness().evaluate(self._config())

        accuracy = self._by_name(result)["accuracy"]
        assert not result.success
        assert accuracy.value is None
        assert accuracy.detail == "No task produced a score"
        assert result.metrics["accuracy"] is None

    def test_absent_protocol_detail_is_left_out_of_the_check(self, stub_runner):
        """A record without a variant, sample cap or few-shot count says so by omission."""
        stub_runner.return_value.run_evaluation_suite.return_value = _results(
            _score(
                task="spleen",
                metric="dice",
                metric_variant=None,
                value=0.94,
                num_samples=20,
                num_samples_available=None,
                num_fewshot=None,
            )
        )

        result = VLLMHarness().evaluate(self._config())

        assert self._by_name(result)["accuracy"].detail == "spleen dice: 0.94 over 20 samples"

    # -- settings derivation ------------------------------------------------ #

    def test_model_prefers_the_profile_aim_id(self, stub_runner):
        """CI scores against org/name, and vLLM serves that name too."""
        profile = _profile(aim_id="meta-llama/Llama-3.1-8B-Instruct", model_id="amd/Llama-3.1-8B-Instruct-FP8-KV")

        VLLMHarness().evaluate(self._config(profile))

        assert self._settings_of(stub_runner).model == "meta-llama/Llama-3.1-8B-Instruct"

    def test_model_falls_back_to_the_model_id(self, stub_runner):
        VLLMHarness().evaluate(self._config(_profile(aim_id="", model_id="org/model")))

        assert self._settings_of(stub_runner).model == "org/model"

    def test_model_is_asked_of_the_service_when_the_profile_is_silent(self, stub_runner, mocker):
        """A base image's general profile names neither, so the endpoint answers."""
        from aim_runtime.harness import service_checks
        from aim_runtime.harness import vllm_harness as vh

        probe = mocker.patch.object(
            vh,
            "probe_api_health",
            return_value=service_checks.HealthProbe(
                check=passed("api_health", "ready"),
                model_id="served/model",
                ready_time_seconds=1.0,
            ),
        )

        VLLMHarness().evaluate(self._config(_profile(aim_id="", model_id="")))

        probe.assert_called_once()
        assert self._settings_of(stub_runner).model == "served/model"

    def test_unnameable_model_fails_before_evaluating(self, stub_runner, mocker):
        from aim_runtime.harness import service_checks
        from aim_runtime.harness import vllm_harness as vh

        mocker.patch.object(
            vh,
            "probe_api_health",
            return_value=service_checks.HealthProbe(
                check=failed("api_health", "no model"),
                model_id=None,
                ready_time_seconds=1.0,
            ),
        )

        result = VLLMHarness().evaluate(self._config(_profile(aim_id="", model_id="")))

        assert not result.success
        assert "determine which model" in result.summary
        stub_runner.assert_not_called()

    def test_cpu_profile_is_sized_as_one_slow_endpoint(self, stub_runner):
        """Its accelerator_count is a core count, so it must not raise concurrency."""
        profile = _profile(accelerator_type="cpu", accelerator_count=188)

        VLLMHarness().evaluate(self._config(profile))

        settings = self._settings_of(stub_runner)
        assert settings.concurrency == 32
        assert settings.request_timeout_seconds == 1800

    def test_wide_gpu_profile_takes_more_concurrency(self, stub_runner):
        VLLMHarness().evaluate(self._config(_profile(accelerator_count=4)))

        settings = self._settings_of(stub_runner)
        assert settings.concurrency == 64
        assert settings.request_timeout_seconds == 600

    def test_trust_remote_code_is_read_from_engine_args(self, stub_runner):
        """A valueless key is how a profile writes a bare vLLM flag, i.e. enabled."""
        profile = _profile(engine_args={"trust-remote-code": None})

        VLLMHarness().evaluate(self._config(profile))

        assert self._settings_of(stub_runner).backend_args["trust_remote_code"] is True

    def test_trust_remote_code_off_when_the_profile_disables_it(self, stub_runner):
        profile = _profile(engine_args={"trust_remote_code": False})

        VLLMHarness().evaluate(self._config(profile))

        assert self._settings_of(stub_runner).backend_args["trust_remote_code"] is False

    def test_malformed_engine_args_still_evaluate(self, stub_runner):
        """A profile whose engine_args are not a mapping loses the flag, not the run."""
        profile = _profile()
        profile["engine_args"] = ["--trust-remote-code"]

        result = VLLMHarness().evaluate(self._config(profile))

        assert result.success
        assert self._settings_of(stub_runner).backend_args["trust_remote_code"] is False

    def test_shipped_backend_args_survive_the_profile_flag(self, stub_runner):
        """Contributing trust_remote_code must not drop apply_chat_template.

        Losing it would prompt without the chat template and score every
        instruction-tuned model lower than the number CI recorded.
        """
        VLLMHarness().evaluate(self._config())

        assert self._settings_of(stub_runner).backend_args["apply_chat_template"] is True

    def test_config_entries_override_settings(self, stub_runner):
        VLLMHarness().evaluate(self._config(limit=5, tasks="mmlu,gsm8k"))

        settings = self._settings_of(stub_runner)
        assert settings.limit == 5
        assert settings.tasks == ("mmlu", "gsm8k")

    def test_config_backend_args_win_over_the_profile(self, stub_runner):
        profile = _profile(engine_args={"trust-remote-code": None})

        VLLMHarness().evaluate(self._config(profile, backend_args={"trust_remote_code": False}))

        assert self._settings_of(stub_runner).backend_args["trust_remote_code"] is False

    def test_unusable_settings_fail_without_evaluating(self, stub_runner):
        result = VLLMHarness().evaluate(self._config(limit=0))

        assert not result.success
        assert "Invalid evaluation settings" in result.summary
        stub_runner.assert_not_called()

    def test_warmup_budget_comes_from_the_config(self, stub_runner):
        VLLMHarness().evaluate(self._config(max_warmup_time=42))

        assert stub_runner.call_args.kwargs["max_warmup_time"] == 42

    # -- artifacts ---------------------------------------------------------- #

    def test_backend_writes_into_scratch_when_no_output_dir_is_asked_for(self, stub_runner):
        """The backend still needs somewhere to write, but not the caller's CWD."""
        VLLMHarness().evaluate(self._config())

        scratch = Path(self._settings_of(stub_runner).output_dir)
        assert scratch != Path(".")
        assert not scratch.exists()

    def test_a_requested_output_dir_creates_no_scratch_directory(self, stub_runner, tmp_path, mocker):
        """The scratch directory is the fallback, not a step on every run."""
        from aim_runtime.harness import vllm_harness as vh

        scratch = mocker.spy(vh, "TemporaryDirectory")

        VLLMHarness().evaluate(self._config(output_dir=str(tmp_path)))

        scratch.assert_not_called()
        assert Path(self._settings_of(stub_runner).output_dir) == tmp_path

    def test_no_output_dir_means_no_artifacts(self, stub_runner):
        result = VLLMHarness().evaluate(self._config())

        stub_runner.return_value.export_results.assert_not_called()
        assert result.artifacts == []

    def test_output_dir_gets_the_envelope_the_csv_and_the_document(self, tmp_path, mocker):
        """The real exporter runs, so the files CI uploads are the ones written."""
        from aim_runtime.evaluation.runner import AIMEvaluation

        results = _results(_score(), document={"results": {"gsm8k": {}}})
        mocker.patch.object(AIMEvaluation, "run_evaluation_suite", return_value=results)

        result = VLLMHarness().evaluate(self._config(output_dir=str(tmp_path)))

        assert sorted(Path(a).name for a in result.artifacts) == [
            "accuracy_evaluation_results.csv",
            "accuracy_evaluation_results.json",
            "evaluation_backend_results.json",
        ]
        envelope = json.loads((tmp_path / "accuracy_evaluation_results.json").read_text())
        assert envelope["accuracy"] == 0.8317
        csv_rows = list(csv.DictReader((tmp_path / "accuracy_evaluation_results.csv").read_text().splitlines()))
        assert [row["task"] for row in csv_rows] == ["gsm8k"]

    def test_unwritable_output_dir_does_not_lose_the_score(self, stub_runner):
        """An export that cannot write is reported, not raised: the score still stands."""
        stub_runner.return_value.export_results.side_effect = OSError("read-only file system")

        result = VLLMHarness().evaluate(self._config(output_dir="/nonexistent/output"))

        assert result.success
        assert result.artifacts == []


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
                if requests.get(f"http://localhost:{self.port}/healthz", timeout=1).status_code == 200:
                    return self
            except requests.RequestException:
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
@_skip_on_ci
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
