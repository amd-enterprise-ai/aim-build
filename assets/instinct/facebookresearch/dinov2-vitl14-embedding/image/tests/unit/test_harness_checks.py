# Copyright © Advanced Micro Devices, Inc., or its affiliates.
#
# SPDX-License-Identifier: MIT

import sys
from pathlib import Path

import dinov2_harness as harness_module
import dinov2_inference as inference_module
import pytest
from dinov2_harness import Dinov2Vitl14Harness

from aim_runtime.harness import CheckScope, ModelHarness
from aim_runtime.harness import discovery as harness_discovery

HARNESS_PATH = Path(__file__).resolve().parents[2] / "src" / "harness.py"


class _Config(dict):
    timeout_seconds = 300
    check_scopes = CheckScope.all()

    @staticmethod
    def resolve_service_url():
        return "http://localhost:8000"


def test_discover_harness_finds_dinov2(monkeypatch):
    monkeypatch.setattr(harness_discovery, "HARNESS_PATH", HARNESS_PATH)
    monkeypatch.setattr(harness_discovery, "MODEL_DIR", str(HARNESS_PATH.parent))
    # Discovery prepends MODEL_DIR to sys.path and harness.py imports `inference`;
    # undo both so other AIMs' same-named modules still resolve to their own files.
    monkeypatch.setattr(sys, "path", list(sys.path))
    monkeypatch.setitem(sys.modules, "inference", inference_module)

    harness = harness_discovery.discover_harness(profile={"engine": "bentoml"})

    assert isinstance(harness, ModelHarness)
    assert type(harness).__name__ == "Dinov2Vitl14Harness"
    assert getattr(type(harness), "ENGINE", None) == "bentoml"


def test_checks_catalog():
    checks = Dinov2Vitl14Harness().list_checks()

    assert [c.name for c in checks] == [
        "bentoml_health",
        "latency_p50_ms",
        "gpu_inference_p50_ms",
        "preprocessing_p50_ms",
        "benchmark_integrity",
        "golden_embedding_cosine",
        "golden_embedding_regression",
    ]
    assert all(c.scope is CheckScope.OFFLINE for c in checks)


@pytest.mark.parametrize("healthy", [True, False])
def test_validate_reports_only_bentoml_health(monkeypatch, healthy):
    def post_predict(*args, **kwargs):
        raise AssertionError("validate must not call /predict")

    harness = Dinov2Vitl14Harness()
    monkeypatch.setattr(harness, "health_check", lambda *args: healthy)
    monkeypatch.setattr(harness, "_post_predict", post_predict)

    result = harness.validate(_Config())

    assert result.success is healthy
    assert [(check.name, check.success) for check in result.checks] == [("bentoml_health", healthy)]


def test_runtime_only_validate_makes_no_http_requests(monkeypatch):
    def no_network(*args, **kwargs):
        raise AssertionError("runtime-only validate must not touch the network")

    harness = Dinov2Vitl14Harness()
    monkeypatch.setattr(harness_module, "urlopen", no_network)
    monkeypatch.setattr(harness, "health_check", no_network)
    config = _Config()
    config.check_scopes = {CheckScope.RUNTIME}

    result = harness.validate(config)

    assert result.success is True
    assert result.checks == []


def test_health_check_uses_monotonic_deadline(monkeypatch):
    clock = iter((10.0, 10.0, 12.0))

    def unavailable(*args, **kwargs):
        raise harness_module.URLError("service unavailable")

    def wall_clock():
        raise AssertionError("wall clock must not be used")

    monkeypatch.setattr(harness_module.time, "monotonic", lambda: next(clock))
    monkeypatch.setattr(harness_module.time, "time", wall_clock)
    monkeypatch.setattr(harness_module.time, "sleep", lambda seconds: None)
    monkeypatch.setattr(harness_module, "urlopen", unavailable)

    assert Dinov2Vitl14Harness().health_check("http://localhost:8000", timeout_seconds=1) is False


def test_benchmark_warms_up_and_reports_separate_timings(monkeypatch):
    calls = []

    def post_predict(service_url, payload, timeout_seconds):
        calls.append((service_url, payload, timeout_seconds))
        call_number = len(calls)
        return {
            "embedding": [1.0, 0.0],
            "timings": {
                "preprocessing_ms": float(call_number),
                "gpu_inference_ms": 10.0,
            },
        }

    harness = Dinov2Vitl14Harness()
    harness.embed_dim = 2
    monkeypatch.setattr(harness, "health_check", lambda *args: True)
    monkeypatch.setattr(harness, "_golden_fixture_payloads", lambda: {"first": "aaa", "second": "bbb"})
    monkeypatch.setattr(harness, "_post_predict", post_predict)

    result = harness.benchmark(_Config(num_requests=3, num_warmup=2, batch_size=1))

    assert result.success is True
    assert len(calls) == 5
    assert [call[1]["image_base64"] for call in calls] == ["aaa", "bbb", "aaa", "bbb", "aaa"]
    assert all(call[1]["include_timings"] is True for call in calls)
    assert result.metrics["warmup_requests_per_batch_size"] == 2
    assert result.metrics["requests_succeeded"] == 3
    assert result.metrics["batch_size_1_preprocessing_mean_ms"] == 4.0
    assert result.metrics["batch_size_1_gpu_inference_p50_ms"] == 10.0
    assert result.metrics["batch_size_1_http_e2e_p50_ms"] >= 0.0
    assert [check.name for check in result.checks] == [
        "bentoml_health",
        "latency_p50_ms",
        "gpu_inference_p50_ms",
        "preprocessing_p50_ms",
        "benchmark_integrity",
    ]


def test_benchmark_defaults_to_batch_size_1(monkeypatch):
    payloads = []

    def post_predict(service_url, payload, timeout_seconds):
        payloads.append(payload)
        return {
            "embedding": [1.0, 0.0],
            "timings": {"preprocessing_ms": 1.0, "gpu_inference_ms": 2.0},
        }

    harness = Dinov2Vitl14Harness()
    harness.embed_dim = 2
    monkeypatch.setattr(harness, "health_check", lambda *args: True)
    monkeypatch.setattr(harness, "_golden_fixture_payloads", lambda: {"first": "aaa", "second": "bbb"})
    monkeypatch.setattr(harness, "_post_predict", post_predict)

    result = harness.benchmark(_Config(num_requests=1, num_warmup=1))

    assert result.success is True
    assert len(payloads) == 2
    assert all("image_base64" in payload for payload in payloads)
    assert result.metrics["batch_sizes"] == "1"
    assert result.metrics["batch_size_1_gpu_inference_p50_ms"] == 2.0
    assert "throughput" in result.summary


def test_benchmark_rejects_batch_size_64(monkeypatch):
    harness = Dinov2Vitl14Harness()
    monkeypatch.setattr(harness, "health_check", lambda *args: True)

    result = harness.benchmark(_Config(batch_sizes=[64]))

    assert result.success is False
    assert result.summary == "DINOv2 benchmark configuration is invalid"
    assert result.checks[-1].name == "benchmark_integrity"
    assert "only batch size 1" in result.checks[-1].detail


def test_benchmark_fails_if_a_measured_response_has_no_timings(monkeypatch):
    calls = 0

    def post_predict(service_url, payload, timeout_seconds):
        nonlocal calls
        calls += 1
        response = {"embedding": [1.0, 0.0]}
        if calls != 2:
            response["timings"] = {"preprocessing_ms": 1.0, "gpu_inference_ms": 2.0}
        return response

    harness = Dinov2Vitl14Harness()
    harness.embed_dim = 2
    monkeypatch.setattr(harness, "health_check", lambda *args: True)
    monkeypatch.setattr(harness, "_golden_fixture_payloads", lambda: {"fixture": "aaa"})
    monkeypatch.setattr(harness, "_post_predict", post_predict)

    result = harness.benchmark(_Config(num_requests=2, num_warmup=1, batch_size=1))

    assert result.success is False
    assert result.metrics["requests_succeeded"] == 1
    assert result.metrics["failure_count"] == 1
    assert result.checks[-1].name == "benchmark_integrity"
    assert result.checks[-1].success is False


def test_requires_rocm_device(monkeypatch):
    class FakeCuda:
        @staticmethod
        def is_available():
            return False

    class FakeTorch:
        cuda = FakeCuda()

    harness = Dinov2Vitl14Harness()
    monkeypatch.setattr(harness, "_torch", lambda: FakeTorch())

    with pytest.raises(RuntimeError, match="requires a ROCm-compatible GPU"):
        harness._get_device()


def test_loads_only_staged_pinned_source(tmp_path, monkeypatch):
    source_dir = tmp_path / "dinov2"
    source_dir.mkdir()
    (source_dir / "hubconf.py").write_text("")
    calls = {}

    class FakeModel:
        def eval(self):
            return self

        def to(self, device):
            calls["device"] = device
            return self

    class FakeHub:
        @staticmethod
        def load(repository, model_name, **kwargs):
            calls["load"] = (repository, model_name, kwargs)
            return FakeModel()

    class FakeTorch:
        hub = FakeHub()

    monkeypatch.setattr(inference_module, "DINOV2_SOURCE_DIR", source_dir)
    monkeypatch.setenv("AIM_DATA_DIR", str(tmp_path / "data"))
    harness = Dinov2Vitl14Harness()
    monkeypatch.setattr(harness, "_torch", lambda: FakeTorch())
    monkeypatch.setattr(harness, "_get_device", lambda: "cuda:0")

    harness._get_model()

    assert calls["load"] == (
        str(source_dir),
        "dinov2_vitl14",
        {"source": "local", "pretrained": True},
    )
    assert calls["device"] == "cuda:0"


def test_rejects_missing_pinned_source(tmp_path, monkeypatch):
    monkeypatch.setattr(inference_module, "DINOV2_SOURCE_DIR", tmp_path / "missing")

    with pytest.raises(RuntimeError, match="pinned DINOv2 source is missing"):
        Dinov2Vitl14Harness()._dinov2_source_dir()


def test_evaluate_compares_five_golden_embeddings(monkeypatch):
    fixture_names = ("gradient", "checkerboard", "rings", "quadrants", "diagonal_stripes")
    payloads = {name: f"payload-{name}" for name in fixture_names}
    golden_embeddings = {
        "gradient": [1.0, 0.0],
        "checkerboard": [0.0, 1.0],
        "rings": [1.0, 1.0],
        "quadrants": [-1.0, 1.0],
        "diagonal_stripes": [-1.0, 0.0],
    }
    embeddings_by_payload = {payloads[name]: embedding for name, embedding in golden_embeddings.items()}

    harness = Dinov2Vitl14Harness()
    harness.embed_dim = 2
    monkeypatch.setattr(harness, "health_check", lambda *args: True)
    monkeypatch.setattr(harness, "_golden_fixture_payloads", lambda: payloads)
    monkeypatch.setattr(harness, "_load_golden_embeddings", lambda path: (0.999, golden_embeddings))
    monkeypatch.setattr(
        harness,
        "_post_predict",
        lambda service_url, payload, timeout_seconds: {"embedding": embeddings_by_payload[payload["image_base64"]]},
    )

    result = harness.evaluate(_Config())

    assert result.success is True
    assert result.metrics["fixtures_passed"] == 5
    assert result.metrics["min_cosine_similarity"] == 1.0
    assert [check.name for check in result.checks] == [
        "bentoml_health",
        "golden_embedding_cosine",
        "golden_embedding_regression",
    ]


def test_evaluate_fails_on_embedding_regression(monkeypatch):
    payloads = {f"fixture_{index}": f"payload_{index}" for index in range(5)}
    golden_embeddings = {name: [1.0, 0.0] for name in payloads}

    harness = Dinov2Vitl14Harness()
    harness.embed_dim = 2
    monkeypatch.setattr(harness, "health_check", lambda *args: True)
    monkeypatch.setattr(harness, "_golden_fixture_payloads", lambda: payloads)
    monkeypatch.setattr(harness, "_load_golden_embeddings", lambda path: (0.999, golden_embeddings))
    monkeypatch.setattr(
        harness,
        "_post_predict",
        lambda service_url, payload, timeout_seconds: {
            "embedding": [-1.0, 0.0] if payload["image_base64"] == "payload_0" else [1.0, 0.0]
        },
    )

    result = harness.evaluate(_Config())

    assert result.success is False
    assert result.metrics["fixtures_passed"] == 4
    assert result.metrics["min_cosine_similarity"] == -1.0


@pytest.mark.parametrize(
    ("timeout_seconds", "overrides", "expected"),
    [(300, {}, 300), (17, {}, 17), (300, {"request_timeout_s": 42}, 42)],
)
def test_evaluate_forwards_request_timeout(monkeypatch, timeout_seconds, overrides, expected):
    payloads = {f"fixture_{index}": f"payload_{index}" for index in range(5)}
    golden_embeddings = {name: [1.0, 0.0] for name in payloads}
    timeouts = []

    def post_predict(service_url, payload, timeout_seconds):
        timeouts.append(timeout_seconds)
        return {"embedding": [1.0, 0.0]}

    harness = Dinov2Vitl14Harness()
    harness.embed_dim = 2
    monkeypatch.setattr(harness, "health_check", lambda *args: True)
    monkeypatch.setattr(harness, "_golden_fixture_payloads", lambda: payloads)
    monkeypatch.setattr(harness, "_load_golden_embeddings", lambda path: (0.999, golden_embeddings))
    monkeypatch.setattr(harness, "_post_predict", post_predict)
    config = _Config(overrides)
    config.timeout_seconds = timeout_seconds

    result = harness.evaluate(config)

    assert result.success is True
    assert timeouts == [expected] * 5
