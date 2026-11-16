# Copyright © Advanced Micro Devices, Inc., or its affiliates.
#
# SPDX-License-Identifier: MIT

"""Behavioural tests for the SaProt harness, driven against a stub HTTP server.

The harness is imported from its file path (it is copied into the image, not
installed as a package) and exercises the real ``aim_runtime.harness`` types.
No GPU, torch or bentoml is needed: the stub server plays the service.
"""

from __future__ import annotations

import importlib.util
import json
import threading
import time
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
from typing import Iterator

import pytest

from aim_runtime.harness import CheckScope, HarnessConfig

HARNESS_PATH = Path(__file__).resolve().parents[1] / "src/harness.py"


def _load_harness():
    spec = importlib.util.spec_from_file_location("saprot_harness_under_test", HARNESS_PATH)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


harness = _load_harness()


class _State:
    health_status = 200
    healthz_hits = 0
    embedding_dim = harness.EXPECTED_EMBEDDING_DIM
    fail_inference = False
    fail_every_other = False
    malformed: str | None = None
    delay = 0.0
    calls = 0
    paths: list[str] = []


@pytest.fixture
def server() -> Iterator[str]:
    _State.health_status = 200
    _State.healthz_hits = 0
    _State.embedding_dim = harness.EXPECTED_EMBEDDING_DIM
    _State.fail_inference = False
    _State.fail_every_other = False
    _State.malformed = None
    _State.delay = 0.0
    _State.calls = 0
    _State.paths = []

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *args) -> None:
            pass

        def _send(self, status: int, body: dict) -> None:
            data = json.dumps(body).encode()
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def do_GET(self) -> None:
            _State.paths.append(self.path)
            if self.path == "/health":
                self._send(_State.health_status, {"status": "ok"})
            elif self.path == "/v1/models":
                self._send(200, {"data": [{"metadata": {"dtype": "fp16"}}]})
            else:
                self._send(404, {})

        def do_POST(self) -> None:
            self.rfile.read(int(self.headers.get("Content-Length", 0)))
            _State.calls += 1
            time.sleep(_State.delay)
            if _State.fail_inference or (_State.fail_every_other and _State.calls % 2 == 0):
                self._send(500, {"error": "boom"})
                return
            if _State.malformed == "empty":
                self._send(200, {})
                return
            if _State.malformed == "short":
                self._send(200, {"embedding": [0.0] * 8, "embedding_dim": 8, "model": "m"})
                return
            if _State.malformed == "dim_mismatch":
                self._send(200, {"embedding": [0.0] * harness.EXPECTED_EMBEDDING_DIM, "embedding_dim": 8, "model": "m"})
                return
            if _State.malformed == "non_numeric":
                self._send(
                    200,
                    {
                        "embedding": ["x"] * harness.EXPECTED_EMBEDDING_DIM,
                        "embedding_dim": harness.EXPECTED_EMBEDDING_DIM,
                        "model": "m",
                    },
                )
                return
            if _State.malformed == "no_model":
                self._send(
                    200,
                    {
                        "embedding": [0.0] * harness.EXPECTED_EMBEDDING_DIM,
                        "embedding_dim": harness.EXPECTED_EMBEDDING_DIM,
                    },
                )
                return
            if _State.malformed == "not_object":
                self._send(200, [1, 2, 3])
                return
            if _State.malformed == "status_201":
                self._send(
                    201,
                    {"embedding": [0.0] * _State.embedding_dim, "embedding_dim": _State.embedding_dim, "model": "m"},
                )
                return
            self._send(
                200,
                {
                    "embedding": [0.0] * _State.embedding_dim,
                    "embedding_dim": _State.embedding_dim,
                    "model": "westlake-repl/SaProt_650M_AF2",
                },
            )

    httpd = HTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=httpd.serve_forever, daemon=True)
    thread.start()
    try:
        yield f"http://127.0.0.1:{httpd.server_address[1]}"
    finally:
        httpd.shutdown()
        httpd.server_close()


def _profile(precision: str = "fp16", accelerator_count: int = 1) -> dict:
    return {"metadata": {"precision": precision, "accelerator_count": accelerator_count}}


def _config(url: str, scopes: set[CheckScope] | None = None, profile: dict | None = None) -> HarnessConfig:
    kwargs = {"check_scopes": scopes} if scopes is not None else {}
    return HarnessConfig(
        profile=_profile() if profile is None else profile, service_url=url, timeout_seconds=3, **kwargs
    )


def test_health_check_polls_health_only(server: str) -> None:
    assert harness.SaProtHarness().health_check(server, timeout_seconds=3) is True
    assert set(_State.paths) == {"/health"}


def test_health_check_times_out_when_unhealthy(server: str) -> None:
    _State.health_status = 503
    assert harness.SaProtHarness().health_check(server, timeout_seconds=1) is False


def test_validate_passes_and_gates_smoke_on_offline_scope(server: str) -> None:
    full = harness.SaProtHarness().validate(_config(server))
    assert full.success
    assert [c.name for c in full.checks] == ["bentoml_health", "embed_smoke"]

    only_runtime = harness.SaProtHarness().validate(_config(server, {CheckScope.RUNTIME}))
    assert only_runtime.checks == []


def test_validate_runtime_scope_does_not_probe_health(server: str) -> None:
    _State.paths = []
    harness.SaProtHarness().validate(_config(server, {CheckScope.RUNTIME}))
    assert _State.paths == []


def test_validate_fails_on_wrong_dimension(server: str) -> None:
    _State.embedding_dim = 8
    result = harness.SaProtHarness().validate(_config(server))
    assert not result.success


def test_benchmark_succeeds_and_reports_metrics(server: str) -> None:
    result = harness.SaProtHarness().benchmark(_config(server))
    assert result.success
    assert result.metrics["failed_requests"] == 0
    assert result.checks[0].success is True
    assert result.metrics["model_dtype"] == "float16"
    assert result.metrics["expected_dtype"] == "float16"
    assert result.metrics["accelerator_count"] == 1
    assert result.metrics["n_timed_total"] == 10 * len(harness.BENCHMARK_LENGTHS)


def test_benchmark_reports_profile_accelerator_count(server: str) -> None:
    result = harness.SaProtHarness().benchmark(_config(server, profile=_profile(accelerator_count=2)))
    assert result.success
    assert result.metrics["accelerator_count"] == 2


def test_benchmark_fails_on_dtype_mismatch_with_profile(server: str) -> None:
    result = harness.SaProtHarness().benchmark(_config(server, profile=_profile(precision="bf16")))
    assert result.success is False
    assert "bfloat16" in result.summary and "float16" in result.summary
    assert result.metrics["model_dtype"] == "float16"
    assert result.metrics["expected_dtype"] == "bfloat16"
    assert _State.calls == 0  # fails before any timed request


def test_benchmark_without_profile_precision_does_not_compare(server: str) -> None:
    result = harness.SaProtHarness().benchmark(_config(server, profile={}))
    assert result.success
    assert result.metrics["expected_dtype"] == "unknown"


def test_benchmark_partial_failure_is_not_success(server: str) -> None:
    _State.fail_every_other = True
    result = harness.SaProtHarness().benchmark(_config(server))
    assert result.metrics["failed_requests"] > 0
    assert result.success is False
    # No throughput figure unless every timed request succeeded.
    assert "sequences_per_sec" not in result.metrics
    assert not any(c.name == "sequences_per_sec" for c in result.checks)


@pytest.mark.parametrize(
    "mode", ["empty", "short", "dim_mismatch", "non_numeric", "no_model", "not_object", "status_201"]
)
def test_benchmark_fails_on_malformed_200_response(server: str, mode: str) -> None:
    _State.malformed = mode
    result = harness.SaProtHarness().benchmark(_config(server))
    assert result.success is False
    assert result.metrics["failed_requests"] == 10 * len(harness.BENCHMARK_LENGTHS)
    assert "sequences_per_sec" not in result.metrics
    assert result.checks == []


def test_benchmark_and_evaluate_skip_without_offline_scope(server: str) -> None:
    _State.paths = []
    for result in (
        harness.SaProtHarness().benchmark(_config(server, {CheckScope.RUNTIME})),
        harness.SaProtHarness().evaluate(_config(server, {CheckScope.RUNTIME})),
    ):
        assert result.success
        assert result.checks and all(c.skipped for c in result.checks)
    assert _State.paths == []


def test_benchmark_all_failures(server: str) -> None:
    _State.fail_inference = True
    result = harness.SaProtHarness().benchmark(_config(server))
    assert result.success is False


def test_evaluate_checks_embedding_shape(server: str) -> None:
    assert harness.SaProtHarness().evaluate(_config(server)).success
    _State.embedding_dim = 4
    assert not harness.SaProtHarness().evaluate(_config(server)).success


def test_benchmark_sequence_lengths_match_buckets() -> None:
    for length, encoded in harness.BENCHMARK_SEQUENCES.items():
        assert len(encoded) == 2 * length


def test_longest_benchmark_bucket_within_service_limit() -> None:
    assert max(harness.BENCHMARK_LENGTHS) <= 1022
    assert harness.BENCHMARK_LENGTHS[-1] == 1022 == harness.MAX_RESIDUES
    assert set(harness.BENCHMARK_SEQUENCES) == set(harness.BENCHMARK_LENGTHS)


def test_send_embed_default_timeout_is_unchanged() -> None:
    assert harness.DEFAULT_REQUEST_TIMEOUT_SECONDS == 60
    assert harness.SaProtHarness._send_embed.__defaults__ == (harness.DEFAULT_REQUEST_TIMEOUT_SECONDS,)


def test_send_embed_honours_explicit_timeout(server: str) -> None:
    _State.delay = 1.5
    assert harness.SaProtHarness._send_embed(server, harness.SMOKE_SEQUENCE)["embedding_dim"]
    with pytest.raises(OSError):
        harness.SaProtHarness._send_embed(server, harness.SMOKE_SEQUENCE, timeout=0.3)


def test_config_timeout_is_forwarded_to_embedding_requests(server: str, monkeypatch: pytest.MonkeyPatch) -> None:
    seen: list[float] = []
    real_urlopen = harness.urlopen

    def spy(req, timeout=None):
        if req.full_url.endswith(harness.INFERENCE_ENDPOINT):
            seen.append(timeout)
        return real_urlopen(req, timeout=timeout)

    monkeypatch.setattr(harness, "urlopen", spy)
    config = HarnessConfig(profile={}, service_url=server, timeout_seconds=7)
    h = harness.SaProtHarness()
    assert h.validate(config).success
    assert h.evaluate(config).success
    assert h.benchmark(config).success
    assert seen and set(seen) == {7}
