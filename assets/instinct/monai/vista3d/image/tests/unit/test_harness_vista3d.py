# Copyright © Advanced Micro Devices, Inc., or its affiliates.
#
# SPDX-License-Identifier: MIT

"""CPU unit tests for Vista3dHarness and MonaiHarness shared base.

No GPU, no live BentoML service, no real NIfTI files required.
Tests cover:
  - CHECKS list completeness (every emitted CheckResult name is declared)
  - _pair_samples default filename-match pairing
  - compute_stats statistics math
  - MODEL_ID is set
  - to_dict() round-trip (via CHECKS list)
"""

import importlib
import importlib.util
import pathlib
import sys
import types as _types

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
Vista3dHarness = _harness_mod.Vista3dHarness
MonaiHarness = _monai_mod.MonaiHarness
compute_stats = _monai_mod.compute_stats


# ---------------------------------------------------------------------------
# CHECKS completeness — every name emitted by evaluate() must be declared
# ---------------------------------------------------------------------------


class TestChecksCompleteness:
    """Verify all CheckResult names emitted by the harness are in CHECKS."""

    # Names emitted by validate(), benchmark(), and evaluate() in monai_harness.py
    EMITTED_NAMES = {
        "bentoml_health",
        "predict_smoke",
        "p50_latency_ms",
        "throughput_img_per_sec",
        "success_rate",
        "mean_dice",
    }

    def test_declared_check_names_cover_emitted(self):
        declared = {c.name for c in Vista3dHarness.CHECKS}
        missing = self.EMITTED_NAMES - declared
        assert not missing, f"CheckResult names emitted by harness but not declared in CHECKS: {missing}"

    def test_success_rate_is_declared(self):
        declared = {c.name for c in Vista3dHarness.CHECKS}
        assert "success_rate" in declared

    def test_bentoml_health_scope_is_offline(self):
        """bentoml_health is an OFFLINE check — it needs a live service.

        OFFLINE means the check needs a running service (health probes,
        endpoints). RUNTIME means profile-only, no service needed.
        See docs/aim_conventions.md#harness-contract and
        aim_runtime.harness.CheckScope for the canonical definitions.
        """
        from aim_runtime.harness import CheckScope

        health_check = next(c for c in Vista3dHarness.CHECKS if c.name == "bentoml_health")
        assert health_check.scope == CheckScope.OFFLINE

    def test_all_checks_have_non_empty_description(self):
        for check in Vista3dHarness.CHECKS:
            assert check.description, f"CheckInfo '{check.name}' has empty description"


# ---------------------------------------------------------------------------
# MODEL_ID check
# ---------------------------------------------------------------------------


def test_model_id():
    assert Vista3dHarness.MODEL_ID == "monai/vista3d"


# ---------------------------------------------------------------------------
# _pair_samples — default filename-match pairing
# ---------------------------------------------------------------------------


class TestPairSamples:
    def test_pairs_matching_filenames(self, tmp_path):
        data_dir = tmp_path / "images"
        labels_dir = tmp_path / "labels"
        data_dir.mkdir()
        labels_dir.mkdir()
        (data_dir / "case_001.nii.gz").write_bytes(b"")
        (data_dir / "case_002.nii.gz").write_bytes(b"")
        (labels_dir / "case_001.nii.gz").write_bytes(b"")
        # case_002 has no label — should be skipped

        pairs = Vista3dHarness._pair_samples(str(data_dir), str(labels_dir))
        assert len(pairs) == 1
        assert pairs[0]["name"] == "case_001.nii.gz"

    def test_returns_empty_when_data_dir_missing(self, tmp_path):
        labels_dir = tmp_path / "labels"
        labels_dir.mkdir()
        pairs = Vista3dHarness._pair_samples("/nonexistent/path", str(labels_dir))
        assert pairs == []

    def test_returns_empty_when_labels_dir_missing(self, tmp_path):
        data_dir = tmp_path / "images"
        data_dir.mkdir()
        (data_dir / "case_001.nii.gz").write_bytes(b"")
        pairs = Vista3dHarness._pair_samples(str(data_dir), "/nonexistent/labels")
        assert pairs == []

    def test_limit_caps_results(self, tmp_path):
        data_dir = tmp_path / "images"
        labels_dir = tmp_path / "labels"
        data_dir.mkdir()
        labels_dir.mkdir()
        for i in range(5):
            (data_dir / f"case_{i:03d}.nii.gz").write_bytes(b"")
            (labels_dir / f"case_{i:03d}.nii.gz").write_bytes(b"")

        pairs = Vista3dHarness._pair_samples(str(data_dir), str(labels_dir), limit=3)
        assert len(pairs) == 3

    def test_ignores_dot_files(self, tmp_path):
        data_dir = tmp_path / "images"
        labels_dir = tmp_path / "labels"
        data_dir.mkdir()
        labels_dir.mkdir()
        (data_dir / ".hidden.nii.gz").write_bytes(b"")
        (labels_dir / ".hidden.nii.gz").write_bytes(b"")

        pairs = Vista3dHarness._pair_samples(str(data_dir), str(labels_dir))
        assert pairs == []

    def test_pair_has_required_keys(self, tmp_path):
        data_dir = tmp_path / "images"
        labels_dir = tmp_path / "labels"
        data_dir.mkdir()
        labels_dir.mkdir()
        (data_dir / "case_001.nii.gz").write_bytes(b"")
        (labels_dir / "case_001.nii.gz").write_bytes(b"")

        pairs = Vista3dHarness._pair_samples(str(data_dir), str(labels_dir))
        assert len(pairs) == 1
        assert set(pairs[0].keys()) == {"name", "image", "label"}


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

    def test_stats_keys_present(self):
        stats = compute_stats([100.0, 200.0, 300.0])
        expected_keys = {
            "mean_ms",
            "stdev_ms",
            "p50_ms",
            "p95_ms",
            "p99_ms",
            "min_ms",
            "max_ms",
            "throughput_img_per_sec",
            "throughput_img_per_sec_from_p50",
        }
        assert expected_keys.issubset(stats.keys())


# ---------------------------------------------------------------------------
# INPUT_GLOB — Vista3D uses *.nii.gz (same as base)
# ---------------------------------------------------------------------------


def test_input_glob():
    assert Vista3dHarness.INPUT_GLOB == "*.nii.gz"


# ---------------------------------------------------------------------------
# METRIC_NAME — Vista3D uses dice
# ---------------------------------------------------------------------------


def test_metric_name():
    assert Vista3dHarness.METRIC_NAME == "dice"


# ---------------------------------------------------------------------------
# ensure_checkpoint — HF_TOKEN requirement
#
# Tests exercise the real pipeline.ensure_checkpoint() by stubbing heavy deps
# (torch, monai, huggingface_hub, config) in sys.modules before import.
# Patching pipeline.HF_TOKEN directly (monkeypatch.setattr) is sufficient
# because `from config import HF_TOKEN` binds the name into pipeline's
# own namespace at import time.
# ---------------------------------------------------------------------------


def _load_pipeline_module():
    """Load pipeline.py with all heavy deps stubbed out.

    The stand-ins are removed again before returning. pipeline.py binds every
    name it needs into its own globals while it executes, and the tests below
    reach those through `_pipeline` (monkeypatching attributes on the module
    object), so none of them need the sys.modules entries to outlive the load.

    Leaving them registered would hand the next AIM's tests a fake `torch`,
    `huggingface_hub` or `monai`: every assets/**/image/tests directory is
    collected into one pytest session, and CI installs the real packages, so a
    later `import huggingface_hub` would silently get the stand-in instead.
    """
    displaced = {}

    def _stub(name, **attrs):
        # setdefault, not assignment: some names below are stubbed twice, and
        # only the first call sees the genuine pre-existing module.
        displaced.setdefault(name, sys.modules.get(name))
        m = _types.ModuleType(name)
        for k, v in attrs.items():
            setattr(m, k, v)
        sys.modules[name] = m
        return m

    try:
        # torch — must expose every name used at module/class-definition level
        torch_mod = _stub("torch")
        torch_mod.device = type("device", (), {"__new__": lambda cls, *a, **kw: object.__new__(cls)})
        torch_mod.float32 = "float32"
        torch_mod.compile = lambda net, **kw: net
        torch_mod.nan_to_num = lambda x, **kw: x
        torch_mod.load = lambda *a, **kw: {}
        torch_mod.cuda = _stub("torch.cuda", is_available=lambda: False)
        torch_mod.amp = _stub("torch.amp", autocast=lambda *a, **kw: None)
        _nn = _stub("torch.nn")
        _nn.Module = type("Module", (), {})
        torch_mod.nn = _nn

        # monai sub-modules
        for name in [
            "monai",
            "monai.bundle",
            "monai.networks",
            "monai.networks.nets",
            "monai.transforms",
            "monai.data",
            "monai.inferers",
            "monai.apps",
            "monai.apps.vista3d",
            "monai.apps.vista3d.inferer",
            "monai.utils",
        ]:
            _stub(name)
        # Each stub must expose every name that pipeline.py imports at module level.
        _stub(
            "monai.apps.vista3d.transforms",
            VistaPostTransformd=type("VistaPostTransformd", (), {}),
            VistaPreTransformd=type("VistaPreTransformd", (), {}),
        )
        _stub("monai.inferers", Inferer=type("Inferer", (), {}))
        _stub("monai.networks.nets", vista3d132=lambda *a, **kw: None)
        # monai.transforms — provide every name imported in pipeline.py
        _transform_names = [
            "CastToTyped",
            "Compose",
            "CropForegroundd",
            "EnsureChannelFirstd",
            "EnsureTyped",
            "Invertd",
            "Lambdad",
            "LoadImaged",
            "Orientationd",
            "SaveImaged",
            "ScaleIntensityRanged",
            "Spacingd",
        ]
        _stub("monai.transforms", **{n: type(n, (), {}) for n in _transform_names})

        # huggingface_hub — stub hf_hub_download; tests replace it via monkeypatch
        _stub("huggingface_hub", hf_hub_download=lambda **kw: None)

        # constants / config
        _stub("constants", BUNDLE_NAME="vista3d", ENV_PREFIX="VISTA3D")
        _stub(
            "config",
            BUNDLE_NAME="vista3d",
            ENV_PREFIX="VISTA3D",
            HF_TOKEN=None,
            LOGGER=__import__("logging").getLogger("test.pipeline"),
            NETWORK_IN_CHANNELS=1,
            OPT_GPU_TRANSFORMS=False,
            ROI_SIZE=[128, 128, 128],
            SW_BATCH_SIZE=1,
            SW_OVERLAP=0.3,
            USE_POINT_WINDOW=True,
        )

        spec = importlib.util.spec_from_file_location("pipeline", str(_SRC / "pipeline.py"))
        assert spec is not None and spec.loader is not None
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)  # type: ignore[union-attr]
        return mod
    finally:
        for name, original in displaced.items():
            if original is None:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = original


_pipeline = _load_pipeline_module()


def _must_not_be_called(**kwargs):
    raise AssertionError(f"hf_hub_download should not be called; got kwargs={kwargs}")


class TestEnsureCheckpoint:
    """Unit tests for the real pipeline.ensure_checkpoint() HF_TOKEN requirement."""

    def test_raises_without_token(self, tmp_path, monkeypatch):
        monkeypatch.setattr(_pipeline, "HF_TOKEN", None)
        monkeypatch.setattr(_pipeline, "hf_hub_download", _must_not_be_called)
        with pytest.raises(RuntimeError, match="NVIDIA OneWay Noncommercial"):
            _pipeline.ensure_checkpoint(tmp_path, tmp_path / "model.pt")

    def test_raises_with_empty_token(self, tmp_path, monkeypatch):
        monkeypatch.setattr(_pipeline, "HF_TOKEN", "")
        monkeypatch.setattr(_pipeline, "hf_hub_download", _must_not_be_called)
        with pytest.raises(RuntimeError, match="HF_TOKEN"):
            _pipeline.ensure_checkpoint(tmp_path, tmp_path / "model.pt")

    def test_error_includes_license_url(self, tmp_path, monkeypatch):
        monkeypatch.setattr(_pipeline, "HF_TOKEN", None)
        with pytest.raises(RuntimeError, match="huggingface.co/MONAI/vista3d"):
            _pipeline.ensure_checkpoint(tmp_path, tmp_path / "model.pt")

    def test_no_download_when_token_absent(self, tmp_path, monkeypatch):
        monkeypatch.setattr(_pipeline, "HF_TOKEN", None)
        monkeypatch.setattr(_pipeline, "hf_hub_download", _must_not_be_called)
        with pytest.raises(RuntimeError):
            _pipeline.ensure_checkpoint(tmp_path, tmp_path / "model.pt")

    def test_skips_download_when_checkpoint_present(self, tmp_path, monkeypatch):
        monkeypatch.setattr(_pipeline, "HF_TOKEN", None)
        monkeypatch.setattr(_pipeline, "hf_hub_download", _must_not_be_called)
        ckpt = tmp_path / "model.pt"
        ckpt.write_bytes(b"fake weights")
        _pipeline.ensure_checkpoint(tmp_path, ckpt)  # must not raise

    def test_passes_token_to_hf_hub_download(self, tmp_path, monkeypatch):
        captured = {}

        def fake_download(**kwargs):
            captured.update(kwargs)
            dest = tmp_path / "models" / "model.pt"
            dest.parent.mkdir(parents=True, exist_ok=True)
            dest.write_bytes(b"fake")
            return str(dest)

        monkeypatch.setattr(_pipeline, "HF_TOKEN", "hf-test-token-xyz")
        monkeypatch.setattr(_pipeline, "hf_hub_download", fake_download)
        _pipeline.ensure_checkpoint(tmp_path, tmp_path / "models" / "model.pt")
        assert captured.get("token") == "hf-test-token-xyz"
