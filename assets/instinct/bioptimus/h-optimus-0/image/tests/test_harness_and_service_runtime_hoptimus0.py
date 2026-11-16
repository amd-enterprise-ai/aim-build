# Copyright © Advanced Micro Devices, Inc., or its affiliates.
#
# SPDX-License-Identifier: MIT

"""Behavioural tests: the real harness against a stub HTTP server, and the real
service module with its GPU-only imports stubbed (see ``conftest.py``)."""

from __future__ import annotations

import time
from types import SimpleNamespace

import pytest

OK_EMBEDDING = {"embedding": [0.1] * 1536, "embedding_dim": 1536, "model": "bioptimus/H-optimus-0"}


@pytest.fixture
def healthy(stub_server):
    stub_server.on("GET", "/health", lambda _b: (200, {"status": "ok"}))
    stub_server.on("POST", "/v1/inference", lambda _b: (200, OK_EMBEDDING))
    return stub_server


@pytest.fixture
def harness(harness_module):
    return harness_module.HOptimus0Harness()


def _scopes(harness_module, *names):
    return {getattr(harness_module.CheckScope, n) for n in names}


# --- health_check / _send_inference ----------------------------------------


def test_health_check_true_when_200(harness, healthy):
    assert harness.health_check(healthy.url, timeout_seconds=2) is True


def test_health_check_false_on_non_200_until_deadline(harness, stub_server):
    stub_server.on("GET", "/health", lambda _b: (503, {"status": "unavailable"}))
    assert harness.health_check(stub_server.url, timeout_seconds=1) is False
    assert stub_server.count("GET", "/health") >= 1
    assert stub_server.count("GET", "/healthz") == 0  # /health only


def test_health_deadline_is_monotonic(harness, harness_module, stub_server, monkeypatch):
    """A wall-clock jump must not end (or extend) the readiness wait."""
    stub_server.on("GET", "/health", lambda _b: (503, {}))
    jumped = SimpleNamespace(
        monotonic=time.monotonic,
        sleep=lambda _s: time.sleep(0.01),
        time=lambda: pytest.fail("harness must not use wall-clock time for deadlines"),
    )
    monkeypatch.setattr(harness_module, "time", jumped)
    assert harness.health_check(stub_server.url, timeout_seconds=0.3) is False


def test_send_inference_posts_body_and_returns_json(harness_module, healthy):
    got = []
    healthy.on("POST", "/v1/inference", lambda b: (got.append(b) or (200, OK_EMBEDDING)))
    out = harness_module.HOptimus0Harness._send_inference(healthy.url, image_base64="QUJD")
    assert out["embedding_dim"] == 1536
    assert got == [{"image_base64": "QUJD"}]


def test_send_inference_requires_an_image(harness_module, healthy):
    with pytest.raises(ValueError):
        harness_module.HOptimus0Harness._send_inference(healthy.url)


# --- validate: scope gating --------------------------------------------------


def test_validate_offline_scope_runs_smoke_and_shape(harness, harness_module, healthy, make_config):
    result = harness.validate(make_config(healthy.url, check_scopes=_scopes(harness_module, "OFFLINE")))
    assert result.success
    assert [c.name for c in result.checks] == ["bentoml_health", "smoke_inference", "embedding_shape"]


def test_validate_without_offline_scope_skips_and_never_calls_service(harness, harness_module, healthy, make_config):
    result = harness.validate(make_config(healthy.url, check_scopes=_scopes(harness_module, "RUNTIME")))
    assert result.success
    assert result.checks and all(c.skipped for c in result.checks)
    assert healthy.calls == []


def test_all_declared_checks_need_a_live_service(harness, harness_module):
    assert {c.scope for c in harness.list_checks()} == {harness_module.CheckScope.OFFLINE}


def test_validate_fails_when_service_down(harness, make_config, stub_server):
    stub_server.on("GET", "/health", lambda _b: (503, {}))
    result = harness.validate(make_config(stub_server.url, timeout_seconds=1))
    assert not result.success
    assert [c.name for c in result.checks] == ["bentoml_health"]


def test_validate_fails_on_wrong_embedding_dim(harness, healthy, make_config):
    healthy.on("POST", "/v1/inference", lambda _b: (200, {"embedding": [0.0] * 8}))
    result = harness.validate(make_config(healthy.url))
    assert not result.success
    assert {c.name: c.success for c in result.checks}["embedding_shape"] is False


def test_validate_fails_on_inference_error(harness, healthy, make_config):
    healthy.on("POST", "/v1/inference", lambda _b: (500, {"detail": "boom"}))
    result = harness.validate(make_config(healthy.url))
    assert not result.success
    assert {c.name: c.success for c in result.checks}["smoke_inference"] is False


# --- benchmark ---------------------------------------------------------------


def test_benchmark_happy_path(harness, healthy, make_config):
    cfg = make_config(healthy.url, profile={"benchmark_n_requests": 3, "benchmark_n_warmup": 1})
    result = harness.benchmark(cfg)
    assert result.success
    assert result.metrics["requests_succeeded"] == 3
    assert result.metrics["requests_failed"] == 0
    assert healthy.count("POST", "/v1/inference") == 4  # 1 warmup + 3 timed


def test_benchmark_partial_failure_is_failure(harness, healthy, make_config):
    n = {"i": 0}

    def flaky(_b):
        n["i"] += 1
        return (500, {}) if n["i"] == 3 else (200, OK_EMBEDDING)

    healthy.on("POST", "/v1/inference", flaky)
    result = harness.benchmark(make_config(healthy.url, extra={"benchmark_n_requests": 4, "benchmark_n_warmup": 0}))
    assert result.success is False
    assert result.metrics["requests_failed"] == 1
    assert result.metrics["requests_succeeded"] == 3
    assert result.checks[0].success is False


def test_benchmark_all_failed(harness, healthy, make_config):
    healthy.on("POST", "/v1/inference", lambda _b: (500, {}))
    result = harness.benchmark(make_config(healthy.url, extra={"benchmark_n_requests": 2, "benchmark_n_warmup": 0}))
    assert result.success is False


def test_benchmark_unreachable_service(harness, stub_server, make_config):
    stub_server.on("GET", "/health", lambda _b: (503, {}))
    assert harness.benchmark(make_config(stub_server.url, timeout_seconds=1)).success is False


def test_benchmark_counts_come_from_config_get(harness, healthy, make_config):
    """``config.extra`` overrides the profile; defaults apply when neither sets them."""
    cfg = make_config(healthy.url, profile={"benchmark_n_requests": 9}, extra={"benchmark_n_requests": 2})
    cfg.extra["benchmark_n_warmup"] = 0
    assert harness.benchmark(cfg).metrics["requests_total"] == 2


# --- evaluate ----------------------------------------------------------------


def test_evaluate_shape_ok_and_bad(harness, healthy, make_config):
    assert harness.evaluate(make_config(healthy.url)).success
    healthy.on("POST", "/v1/inference", lambda _b: (200, {"embedding": [float("nan")] * 1536}))
    assert not harness.evaluate(make_config(healthy.url)).success


# --- service module (GPU stack stubbed) ---------------------------------------


def test_service_health_forwards_upstream_status(load_service, monkeypatch):
    from starlette.testclient import TestClient

    service = load_service(port=4321)
    seen = []

    class FakeClient:
        def __init__(self, **_kw):
            pass

        async def __aenter__(self):
            return self

        async def __aexit__(self, *_a):
            return False

        async def get(self, url):
            seen.append(url)
            return SimpleNamespace(is_success=True, status_code=200)

    monkeypatch.setattr(service.httpx, "AsyncClient", FakeClient)
    resp = TestClient(service.canonical_app).get("/health")
    assert (resp.status_code, resp.json()) == (200, {"status": "ok"})
    assert seen == ["http://127.0.0.1:4321/healthz"]


def test_service_health_503_when_upstream_unreachable(load_service, monkeypatch):
    from starlette.testclient import TestClient

    service = load_service()

    class DownClient:
        def __init__(self, **_kw):
            pass

        async def __aenter__(self):
            return self

        async def __aexit__(self, *_a):
            return False

        async def get(self, _url):
            raise service.httpx.ConnectError("down")

    monkeypatch.setattr(service.httpx, "AsyncClient", DownClient)
    resp = TestClient(service.canonical_app).get("/health")
    assert (resp.status_code, resp.json()) == (503, {"status": "unavailable"})


def test_service_models_shape(load_service):
    from starlette.testclient import TestClient

    service = load_service(cuda_available=True)
    body = TestClient(service.canonical_app).get("/v1/models").json()
    assert body["object"] == "list"
    (entry,) = body["data"]
    assert entry["id"] == "bioptimus/H-optimus-0"
    assert entry["object"] == "model"
    assert entry["metadata"] == {"dtype": "fp16", "accelerator": "Stub GPU"}


def test_service_models_accelerator_none_without_gpu(load_service):
    from starlette.testclient import TestClient

    service = load_service(cuda_available=False)
    entry = TestClient(service.canonical_app).get("/v1/models").json()["data"][0]
    assert entry["metadata"]["accelerator"] is None


def test_single_gpu_by_default(load_service):
    service = load_service(cuda_available=True)
    assert service.args.accelerator_count == 1
    (kwargs,) = service.RECORDED_SERVICE_KWARGS
    assert kwargs["resources"] == {"gpu": 1}
    assert kwargs["workers"] == 1


def test_two_accelerators_request_two_gpus_and_two_workers(load_service):
    service = load_service(cuda_available=True, accelerator_count=2, device_count=2)
    (kwargs,) = service.RECORDED_SERVICE_KWARGS
    assert kwargs["resources"] == {"gpu": 2}
    assert kwargs["workers"] == 2


def test_each_worker_takes_its_own_gpu(load_service):
    # Both GPUs visible: worker k uses cuda:k-1 and makes it the current device.
    for w in (1, 2):
        service = load_service(cuda_available=True, worker_index=w, accelerator_count=2, device_count=2)
        assert service._get_device() == f"cuda:{w - 1}"
        assert service.torch.set_device_calls == [w - 1]


def test_single_gpu_still_uses_cuda0(load_service):
    service = load_service(cuda_available=True, worker_index=1, device_count=1)
    assert service._get_device() == "cuda:0"
    assert service.torch.set_device_calls == [0]


def test_fewer_gpus_than_workers_fails_at_startup(load_service):
    service = load_service(cuda_available=True, worker_index=2, accelerator_count=2, device_count=1)
    with pytest.raises(
        RuntimeError, match=r"AIM_ACCELERATOR_COUNT=2 requires 2 accelerator\(s\), but this container has 1"
    ):
        service._get_device()
    assert service.torch.set_device_calls == []


def test_accelerator_count_must_be_positive(load_service):
    with pytest.raises(ValueError):
        load_service(accelerator_count=0)


def test_startup_without_gpu_stops(load_service):
    service = load_service(cuda_available=False)
    with pytest.raises(RuntimeError, match="ROCm GPU"):
        service.HOptimus0Service()


def test_startup_without_weights_or_token_stops_naming_path_and_token(load_service, tmp_path, monkeypatch):
    service = load_service(cuda_available=True)
    monkeypatch.setenv("AIM_CACHE_PATH", str(tmp_path))
    monkeypatch.delenv("HF_TOKEN", raising=False)
    monkeypatch.delenv("HUGGING_FACE_HUB_TOKEN", raising=False)
    with pytest.raises(RuntimeError) as err:
        service.HOptimus0Service()
    assert str(tmp_path) in str(err.value)
    assert "HF_TOKEN" in str(err.value)


def test_decode_image_rejects_bad_input(load_service, tmp_path):
    service = load_service(cuda_available=True)
    svc = object.__new__(service.HOptimus0Service)
    svc._input_root = tmp_path
    with pytest.raises(service.InvalidArgument):
        svc._decode_image(None, "not-base64!!")
    with pytest.raises(service.InvalidArgument):
        svc._decode_image("/etc/passwd", None)
    with pytest.raises(service.InvalidArgument):
        svc._decode_image(None, None)


def _png_b64(width, height):
    import base64
    import io

    from PIL import Image

    buf = io.BytesIO()
    Image.new("RGB", (width, height), (200, 100, 150)).save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode()


@pytest.mark.parametrize("size", [(256, 256), (224, 300), (300, 224), (1000, 600)])
def test_decode_image_rejects_non_224_tile(load_service, tmp_path, size):
    service = load_service(cuda_available=True)
    svc = object.__new__(service.HOptimus0Service)
    svc._input_root = tmp_path
    with pytest.raises(service.InvalidArgument, match="224x224"):
        svc._decode_image(None, _png_b64(*size))


def test_decode_image_accepts_224_tile(load_service, tmp_path):
    service = load_service(cuda_available=True)
    svc = object.__new__(service.HOptimus0Service)
    svc._input_root = tmp_path
    assert svc._decode_image(None, _png_b64(224, 224)).size == (224, 224)


def test_benchmark_metric_key_is_images_per_sec(harness, healthy, make_config):
    cfg = make_config(healthy.url, extra={"benchmark_n_requests": 2, "benchmark_n_warmup": 0})
    result = harness.benchmark(cfg)
    assert [c.name for c in result.checks] == ["images_per_sec"]
    assert "images_per_sec" in result.metrics
    assert "throughput_rps" not in {c.name for c in harness.list_checks()}


@pytest.mark.parametrize("method", ["benchmark", "evaluate"])
def test_benchmark_and_evaluate_skip_without_offline_scope(harness, harness_module, healthy, make_config, method):
    result = getattr(harness, method)(make_config(healthy.url, check_scopes=_scopes(harness_module, "RUNTIME")))
    assert result.success
    assert result.checks and all(c.skipped for c in result.checks)
    assert healthy.calls == []
