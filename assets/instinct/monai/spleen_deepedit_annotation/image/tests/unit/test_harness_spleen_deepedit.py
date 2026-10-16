# Copyright © Advanced Micro Devices, Inc., or its affiliates.
#
# SPDX-License-Identifier: MIT

"""CPU unit tests for SpleenDeepeditHarness and MonaiHarness shared base.

No GPU, no live BentoML service, no real NIfTI files required.
Tests cover:
  - CHECKS list completeness (every emitted CheckResult name is declared)
  - bentoml_health scope is RUNTIME
  - _pair_samples default filename-match pairing
  - compute_stats statistics math
"""

import importlib.util
import pathlib
import sys

import pytest

_SRC = pathlib.Path(__file__).parents[2] / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))


# ---------------------------------------------------------------------------
# Load harness without importing aim_runtime (not installed in unit-test env)
# ---------------------------------------------------------------------------


def _load_harness_module():
    """Load harness.py and monai_harness.py from the AIM src/ directory."""
    # Stub aim_runtime.harness so imports work without the package installed
    import types

    # The aim_runtime stand-ins intentionally stay in sys.modules: tests below
    # do `from aim_runtime.harness import CheckScope` at call time, and no real
    # package of that name is installed in the unit-test environment.
    if "aim_runtime" not in sys.modules:
        aim_runtime_pkg = types.ModuleType("aim_runtime")
        aim_runtime_harness = types.ModuleType("aim_runtime.harness")

        class CheckScope:
            OFFLINE = "OFFLINE"
            RUNTIME = "RUNTIME"

        class CheckResultType:
            PASS_FAIL = "PASS_FAIL"
            SCORE = "SCORE"

        class CheckInfo:
            def __init__(self, name, result_type, scope, description=""):
                self.name = name
                self.result_type = result_type
                self.scope = scope
                self.description = description

        class CheckResult:
            def __init__(self, name, result_type, success, value, detail=""):
                self.name = name
                self.result_type = result_type
                self.success = success
                self.value = value
                self.detail = detail

        class HarnessConfig:
            def __init__(self, profile=None, timeout_seconds=300, check_scopes=None):
                self.profile = profile or {}
                self.timeout_seconds = timeout_seconds
                self.check_scopes = check_scopes or [CheckScope.OFFLINE, CheckScope.RUNTIME]

            def resolve_service_url(self):
                explicit = self.profile.get("service_url")
                if explicit:
                    return str(explicit).rstrip("/")
                port = int(self.profile.get("port", 8000))
                return f"http://localhost:{port}"

        class HarnessResult:
            def __init__(self, success, summary="", checks=None, metrics=None):
                self.success = success
                self.summary = summary
                self.checks = checks or []
                self.metrics = metrics or {}

        class ModelHarness:
            pass

        aim_runtime_harness.CheckScope = CheckScope
        aim_runtime_harness.CheckResultType = CheckResultType
        aim_runtime_harness.CheckInfo = CheckInfo
        aim_runtime_harness.CheckResult = CheckResult
        aim_runtime_harness.HarnessConfig = HarnessConfig
        aim_runtime_harness.HarnessResult = HarnessResult
        aim_runtime_harness.ModelHarness = ModelHarness

        aim_runtime_pkg.harness = aim_runtime_harness
        sys.modules["aim_runtime"] = aim_runtime_pkg
        sys.modules["aim_runtime.harness"] = aim_runtime_harness

    # Pin `constants` to THIS AIM before harness.py imports it by bare name.
    # Every MONAI AIM ships a constants.py, and when several AIMs' tests are
    # collected into one pytest process each conftest prepends its own src/ to
    # sys.path — so the last one loaded would otherwise decide whose
    # BUNDLE_NAME every harness sees.
    #
    # The pin only has to hold while harness.py and monai_harness.py execute;
    # both bind what they need into their own globals. Putting sys.modules back
    # afterwards keeps this AIM's modules from deciding that for the next one.
    displaced = {name: sys.modules.get(name) for name in ("constants", "monai_harness")}
    try:
        spec0 = importlib.util.spec_from_file_location("constants", str(_SRC / "constants.py"))
        const_mod = importlib.util.module_from_spec(spec0)
        sys.modules["constants"] = const_mod
        spec0.loader.exec_module(const_mod)

        spec = importlib.util.spec_from_file_location("monai_harness", str(_SRC / "monai_harness.py"))
        monai_mod = importlib.util.module_from_spec(spec)
        sys.modules["monai_harness"] = monai_mod
        spec.loader.exec_module(monai_mod)

        spec2 = importlib.util.spec_from_file_location("harness", str(_SRC / "harness.py"))
        harness_mod = importlib.util.module_from_spec(spec2)
        spec2.loader.exec_module(harness_mod)
        return harness_mod, monai_mod
    finally:
        for name, original in displaced.items():
            if original is None:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = original


_harness_mod, _monai_mod = _load_harness_module()
SpleenDeepeditHarness = _harness_mod.SpleenDeepeditHarness
MonaiHarness = _monai_mod.MonaiHarness
compute_stats = _monai_mod.compute_stats


# ---------------------------------------------------------------------------
# CHECKS completeness — every name emitted by evaluate() must be declared
# ---------------------------------------------------------------------------


class TestChecksCompleteness:
    """Verify all CheckResult names emitted by the harness are in CHECKS."""

    EMITTED_NAMES = {
        "bentoml_health",
        "predict_smoke",
        "p50_latency_ms",
        "throughput_img_per_sec",
        "success_rate",
        "mean_dice",
    }

    def test_declared_check_names_cover_emitted(self):
        declared = {c.name for c in SpleenDeepeditHarness.CHECKS}
        missing = self.EMITTED_NAMES - declared
        assert not missing, f"CheckResult names emitted by harness but not declared in CHECKS: {missing}"

    def test_success_rate_is_declared(self):
        declared = {c.name for c in SpleenDeepeditHarness.CHECKS}
        assert "success_rate" in declared

    def test_bentoml_health_scope_is_offline(self):
        """bentoml_health is an OFFLINE check — it needs a live service.

        OFFLINE means the check needs a running service (health probes,
        endpoints). RUNTIME means profile-only, no service needed.
        See docs/aim_conventions.md#harness-contract and
        aim_runtime.harness.CheckScope for the canonical definitions.
        """
        from aim_runtime.harness import CheckScope

        health_check = next(c for c in SpleenDeepeditHarness.CHECKS if c.name == "bentoml_health")
        assert health_check.scope == CheckScope.OFFLINE


# ---------------------------------------------------------------------------
# _pair_samples — default filename-match pairing
# ---------------------------------------------------------------------------


class TestPairSamples:
    def test_pairs_matching_filenames(self, tmp_path):
        data_dir = tmp_path / "images"
        labels_dir = tmp_path / "labels"
        data_dir.mkdir()
        labels_dir.mkdir()
        (data_dir / "spleen_1.nii.gz").write_bytes(b"")
        (data_dir / "spleen_2.nii.gz").write_bytes(b"")
        (labels_dir / "spleen_1.nii.gz").write_bytes(b"")
        # spleen_2 has no label — should be skipped

        pairs = SpleenDeepeditHarness._pair_samples(str(data_dir), str(labels_dir))
        assert len(pairs) == 1
        assert pairs[0]["name"] == "spleen_1.nii.gz"

    def test_returns_empty_when_data_dir_missing(self, tmp_path):
        labels_dir = tmp_path / "labels"
        labels_dir.mkdir()
        pairs = SpleenDeepeditHarness._pair_samples("/nonexistent/path", str(labels_dir))
        assert pairs == []

    def test_returns_empty_when_labels_dir_missing(self, tmp_path):
        data_dir = tmp_path / "images"
        data_dir.mkdir()
        (data_dir / "spleen_1.nii.gz").write_bytes(b"")
        pairs = SpleenDeepeditHarness._pair_samples(str(data_dir), "/nonexistent/labels")
        assert pairs == []

    def test_limit_caps_results(self, tmp_path):
        data_dir = tmp_path / "images"
        labels_dir = tmp_path / "labels"
        data_dir.mkdir()
        labels_dir.mkdir()
        for i in range(5):
            (data_dir / f"spleen_{i}.nii.gz").write_bytes(b"")
            (labels_dir / f"spleen_{i}.nii.gz").write_bytes(b"")

        pairs = SpleenDeepeditHarness._pair_samples(str(data_dir), str(labels_dir), limit=3)
        assert len(pairs) == 3

    def test_ignores_dot_files(self, tmp_path):
        data_dir = tmp_path / "images"
        labels_dir = tmp_path / "labels"
        data_dir.mkdir()
        labels_dir.mkdir()
        (data_dir / ".hidden.nii.gz").write_bytes(b"")
        (labels_dir / ".hidden.nii.gz").write_bytes(b"")

        pairs = SpleenDeepeditHarness._pair_samples(str(data_dir), str(labels_dir))
        assert pairs == []


# ---------------------------------------------------------------------------
# compute_stats — statistics math
# ---------------------------------------------------------------------------


class TestComputeStats:
    def test_empty_returns_empty_dict(self):
        assert compute_stats([]) == {}

    def test_single_value(self):
        stats = compute_stats([100.0])
        assert stats["p50_ms"] == pytest.approx(100.0)
        assert stats["mean_ms"] == pytest.approx(100.0)
        assert stats["stdev_ms"] == pytest.approx(0.0)
        assert stats["min_ms"] == pytest.approx(100.0)
        assert stats["max_ms"] == pytest.approx(100.0)

    def test_throughput_from_p50(self):
        # p50 of [500, 500] = 500ms → throughput_img_per_sec_from_p50 = 2.0
        stats = compute_stats([500.0, 500.0])
        assert stats["throughput_img_per_sec_from_p50"] == pytest.approx(2.0)

    def test_throughput_measured(self):
        # 2 requests at 1000ms each → total 2s → 1.0 img/s
        stats = compute_stats([1000.0, 1000.0])
        assert stats["throughput_img_per_sec"] == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# MODEL_ID check
# ---------------------------------------------------------------------------


def test_model_id():
    assert SpleenDeepeditHarness.MODEL_ID == "monai/spleen_deepedit_annotation"
