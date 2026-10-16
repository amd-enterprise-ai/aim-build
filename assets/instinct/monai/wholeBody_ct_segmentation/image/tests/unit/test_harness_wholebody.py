# Copyright © Advanced Micro Devices, Inc., or its affiliates.
#
# SPDX-License-Identifier: MIT

"""CPU unit tests for WholeBodyCtHarness and MonaiHarness shared base.

No GPU, no live BentoML service, no real NIfTI files required.
Tests cover:
  - CHECKS list completeness (every emitted CheckResult name is declared)
  - bentoml_health scope is OFFLINE
  - _pair_samples default filename-match pairing
  - _per_class_mean_dice arithmetic (via synthetic numpy arrays)
  - compute_stats statistics math
  - MODEL_ID is set
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
WholeBodyCtHarness = _harness_mod.WholeBodyCtHarness
MonaiHarness = _monai_mod.MonaiHarness
compute_stats = _monai_mod.compute_stats


# ---------------------------------------------------------------------------
# CHECKS completeness — every name emitted by harness methods must be declared
# ---------------------------------------------------------------------------


class TestChecksCompleteness:
    """Verify all CheckResult names emitted by the harness are in CHECKS."""

    # Names emitted by validate(), benchmark(), and evaluate() respectively.
    EMITTED_NAMES = {
        "bentoml_health",
        "predict_smoke",
        "p50_latency_ms",
        "throughput_img_per_sec",
        "success_rate",
        "mean_dice",
    }

    def test_declared_check_names_cover_emitted(self):
        declared = {c.name for c in WholeBodyCtHarness.CHECKS}
        missing = self.EMITTED_NAMES - declared
        assert not missing, f"CheckResult names emitted by harness but not declared in CHECKS: {missing}"

    def test_success_rate_is_declared(self):
        declared = {c.name for c in WholeBodyCtHarness.CHECKS}
        assert "success_rate" in declared

    def test_mean_dice_is_declared(self):
        declared = {c.name for c in WholeBodyCtHarness.CHECKS}
        assert "mean_dice" in declared

    def test_bentoml_health_scope_is_offline(self):
        """bentoml_health is an OFFLINE check — it needs a live service.

        OFFLINE means the check needs a running service (health probes,
        endpoints). RUNTIME means profile-only, no service needed.
        See docs/aim_conventions.md#harness-contract and
        aim_runtime.harness.CheckScope for the canonical definitions.
        """
        from aim_runtime.harness import CheckScope

        health_check = next(c for c in WholeBodyCtHarness.CHECKS if c.name == "bentoml_health")
        assert health_check.scope == CheckScope.OFFLINE


# ---------------------------------------------------------------------------
# MODEL_ID check
# ---------------------------------------------------------------------------


def test_model_id():
    assert WholeBodyCtHarness.MODEL_ID == "monai/wholeBody_ct_segmentation"


# ---------------------------------------------------------------------------
# to_dict() round-trip — only if the method exists on the harness
# ---------------------------------------------------------------------------


def test_to_dict_round_trip_if_present():
    """If to_dict() is defined, verify it round-trips the key fields."""
    harness = WholeBodyCtHarness()
    if not hasattr(harness, "to_dict"):
        pytest.skip("to_dict() not defined on this harness")
    d = harness.to_dict()
    assert d.get("model_id") == WholeBodyCtHarness.MODEL_ID


# ---------------------------------------------------------------------------
# _pair_samples — default filename-match pairing (inherited from MonaiHarness)
# ---------------------------------------------------------------------------


class TestPairSamples:
    def test_pairs_matching_filenames(self, tmp_path):
        data_dir = tmp_path / "imagesTs"
        labels_dir = tmp_path / "labelsTs"
        data_dir.mkdir()
        labels_dir.mkdir()
        (data_dir / "case_001.nii.gz").write_bytes(b"")
        (data_dir / "case_002.nii.gz").write_bytes(b"")
        (labels_dir / "case_001.nii.gz").write_bytes(b"")
        # case_002 has no label — should be skipped

        pairs = WholeBodyCtHarness._pair_samples(str(data_dir), str(labels_dir))
        assert len(pairs) == 1
        assert pairs[0]["name"] == "case_001.nii.gz"

    def test_returns_empty_when_data_dir_missing(self, tmp_path):
        labels_dir = tmp_path / "labelsTs"
        labels_dir.mkdir()
        pairs = WholeBodyCtHarness._pair_samples("/nonexistent/path", str(labels_dir))
        assert pairs == []

    def test_returns_empty_when_labels_dir_missing(self, tmp_path):
        data_dir = tmp_path / "imagesTs"
        data_dir.mkdir()
        (data_dir / "case_001.nii.gz").write_bytes(b"")
        pairs = WholeBodyCtHarness._pair_samples(str(data_dir), "/nonexistent/labels")
        assert pairs == []

    def test_limit_caps_results(self, tmp_path):
        data_dir = tmp_path / "imagesTs"
        labels_dir = tmp_path / "labelsTs"
        data_dir.mkdir()
        labels_dir.mkdir()
        for i in range(5):
            (data_dir / f"case_{i:03d}.nii.gz").write_bytes(b"")
            (labels_dir / f"case_{i:03d}.nii.gz").write_bytes(b"")

        pairs = WholeBodyCtHarness._pair_samples(str(data_dir), str(labels_dir), limit=3)
        assert len(pairs) == 3

    def test_ignores_dot_files(self, tmp_path):
        data_dir = tmp_path / "imagesTs"
        labels_dir = tmp_path / "labelsTs"
        data_dir.mkdir()
        labels_dir.mkdir()
        (data_dir / ".hidden.nii.gz").write_bytes(b"")
        (labels_dir / ".hidden.nii.gz").write_bytes(b"")

        pairs = WholeBodyCtHarness._pair_samples(str(data_dir), str(labels_dir))
        assert pairs == []


# ---------------------------------------------------------------------------
# _per_class_mean_dice arithmetic — uses synthetic NIfTI-like numpy arrays
# ---------------------------------------------------------------------------


class TestPerClassMeanDice:
    """Exercise _per_class_mean_dice without real NIfTI files by writing
    synthetic arrays as NIfTI files using nibabel (which is available in the
    base MONAI image). Tests are skipped if nibabel is not installed."""

    @pytest.fixture(autouse=True)
    def require_nibabel(self):
        pytest.importorskip("nibabel")
        pytest.importorskip("numpy")

    def _write_nifti(self, tmp_path, name, array):
        import nibabel as nib
        import numpy as np

        img = nib.Nifti1Image(array.astype(np.float32), affine=np.eye(4))
        path = str(tmp_path / name)
        nib.save(img, path)
        return path

    def test_perfect_match_returns_one(self, tmp_path):
        import numpy as np

        data = np.zeros((4, 4, 4), dtype=np.float32)
        data[1, 1, 1] = 1  # class 1 voxel
        data[2, 2, 2] = 2  # class 2 voxel
        pred = self._write_nifti(tmp_path, "pred.nii.gz", data)
        label = self._write_nifti(tmp_path, "label.nii.gz", data)

        score = WholeBodyCtHarness._per_class_mean_dice(pred, label, class_indices=(1, 2))
        assert score == pytest.approx(1.0)

    def test_no_overlap_returns_zero(self, tmp_path):
        import numpy as np

        pred_data = np.zeros((4, 4, 4), dtype=np.float32)
        pred_data[0, 0, 0] = 1
        gt_data = np.zeros((4, 4, 4), dtype=np.float32)
        gt_data[3, 3, 3] = 1

        pred = self._write_nifti(tmp_path, "pred.nii.gz", pred_data)
        label = self._write_nifti(tmp_path, "label.nii.gz", gt_data)

        score = WholeBodyCtHarness._per_class_mean_dice(pred, label, class_indices=(1,))
        assert score == pytest.approx(0.0)

    def test_empty_empty_per_class_contributes_one(self, tmp_path):
        """Empty pred and GT for a class contribute 1.0 (not punish absent classes)."""
        import numpy as np

        zeros = np.zeros((4, 4, 4), dtype=np.float32)
        pred = self._write_nifti(tmp_path, "pred.nii.gz", zeros)
        label = self._write_nifti(tmp_path, "label.nii.gz", zeros)

        score = WholeBodyCtHarness._per_class_mean_dice(pred, label, class_indices=(1, 2, 3))
        assert score == pytest.approx(1.0)

    def test_shape_mismatch_raises(self, tmp_path):
        import numpy as np

        pred_arr = np.zeros((4, 4, 4), dtype=np.float32)
        gt_arr = np.zeros((5, 5, 5), dtype=np.float32)
        pred = self._write_nifti(tmp_path, "pred.nii.gz", pred_arr)
        label = self._write_nifti(tmp_path, "label.nii.gz", gt_arr)

        with pytest.raises(ValueError, match="Shape mismatch"):
            WholeBodyCtHarness._per_class_mean_dice(pred, label, class_indices=(1,))


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

    def test_p95_greater_than_or_equal_p50(self):
        stats = compute_stats([100.0, 200.0, 300.0, 400.0, 500.0])
        assert stats["p95_ms"] >= stats["p50_ms"]
