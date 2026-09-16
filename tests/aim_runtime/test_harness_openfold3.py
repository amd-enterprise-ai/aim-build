# Copyright © Advanced Micro Devices, Inc., or its affiliates.
#
# SPDX-License-Identifier: MIT

"""Unit tests for the OpenFold3 model harness.

These tests do **not** require a running OpenFold3 BentoML service or a GPU —
they exercise discovery, the check catalog, and the BentoML ``/healthz``
polling logic against a stub HTTP server.
"""

from __future__ import annotations

import importlib.util
import socket
import threading
import time
from http.server import BaseHTTPRequestHandler, HTTPServer, ThreadingHTTPServer
from pathlib import Path
from types import SimpleNamespace

import pytest

from aim_runtime.harness import CheckScope, ModelHarness
from aim_runtime.harness import discovery as harness_discovery

OF3_HARNESS_PATH = Path(__file__).resolve().parents[2] / "assets/instinct/openfold/openfold3/image/src/harness.py"


@pytest.fixture
def of3_harness_module():
    """Load assets/.../image/src/harness.py directly without monkey-patching the global path."""
    spec = importlib.util.spec_from_file_location("_of3_harness_under_test", OF3_HARNESS_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


# ---------------------------------------------------------------------------
# Discovery
# ---------------------------------------------------------------------------


def test_of3_harness_file_exists():
    assert OF3_HARNESS_PATH.is_file(), f"OF3 harness expected at {OF3_HARNESS_PATH} — Dockerfile COPY relies on it"


def test_discover_harness_finds_of3(monkeypatch):
    """``discover_harness`` returns OpenFold3Harness when HARNESS_PATH points at our file."""
    monkeypatch.setattr(harness_discovery, "HARNESS_PATH", OF3_HARNESS_PATH)
    monkeypatch.setattr(harness_discovery, "MODEL_DIR", str(OF3_HARNESS_PATH.parent))

    harness = harness_discovery.discover_harness(profile={"engine": "bentoml"})

    assert isinstance(harness, ModelHarness)
    assert type(harness).__name__ == "OpenFold3Harness"
    assert getattr(type(harness), "ENGINE", None) == "bentoml"


# ---------------------------------------------------------------------------
# Check catalog
# ---------------------------------------------------------------------------


def test_of3_checks_catalog(of3_harness_module):
    harness = of3_harness_module.OpenFold3Harness()
    checks = harness.list_checks()

    assert [c.name for c in checks] == [
        "bentoml_health",
        "predict_smoke",
        "predict_inline_msa",
        "benchmark_21_pdb",
        "structure_sanity",
        "overload_sheds_requests",
    ]
    # Exactly one RUNTIME check (the health probe); the rest are OFFLINE.
    runtime_checks = [c for c in checks if c.scope is CheckScope.RUNTIME]
    assert len(runtime_checks) == 1
    assert runtime_checks[0].name == "bentoml_health"


# ---------------------------------------------------------------------------
# Inline-MSA smoke payload
# ---------------------------------------------------------------------------


def test_build_smoke_payload_inline_msa(of3_harness_module):
    payload = of3_harness_module._build_smoke_payload(inline_msa=True)
    chain = payload["data"]["queries"][of3_harness_module.SMOKE_QUERY_NAME]["chains"][0]
    assert isinstance(chain.get("main_msa"), str)
    assert chain["main_msa"].startswith(">")
    assert payload["data"]["use_msa_server"] is False


def test_build_smoke_payload_default_omits_inline_msa(of3_harness_module):
    payload = of3_harness_module._build_smoke_payload()
    chain = payload["data"]["queries"][of3_harness_module.SMOKE_QUERY_NAME]["chains"][0]
    assert "main_msa" not in chain


# ---------------------------------------------------------------------------
# health_check polls /healthz
# ---------------------------------------------------------------------------


class _HealthzHandler(BaseHTTPRequestHandler):
    def do_GET(self):  # noqa: N802
        if self.path == "/healthz":
            self.send_response(200)
            self.end_headers()
        else:
            self.send_response(404)
            self.end_headers()

    def log_message(self, *_args):  # silence stderr noise during tests
        return


def _free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def test_health_check_passes_against_local_healthz(of3_harness_module):
    port = _free_port()
    server = HTTPServer(("127.0.0.1", port), _HealthzHandler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        harness = of3_harness_module.OpenFold3Harness()
        assert harness.health_check(f"http://127.0.0.1:{port}", timeout_seconds=5) is True
    finally:
        server.shutdown()
        server.server_close()


def test_health_check_fails_on_dead_port(of3_harness_module):
    harness = of3_harness_module.OpenFold3Harness()
    # Short timeout — we are intentionally pointing at a port nothing listens on.
    assert harness.health_check(f"http://127.0.0.1:{_free_port()}", timeout_seconds=2) is False


# ---------------------------------------------------------------------------
# _post_predict error-body handling
# ---------------------------------------------------------------------------


def _predict_handler(status: int, body: bytes, extra_headers: dict[str, str] | None = None):
    class Handler(BaseHTTPRequestHandler):
        def do_POST(self):  # noqa: N802
            length = int(self.headers.get("Content-Length", 0))
            self.rfile.read(length)
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            for name, value in (extra_headers or {}).items():
                self.send_header(name, value)
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, *_args):  # silence stderr noise during tests
            return

    return Handler


def _run_predict_server(handler_cls) -> tuple[HTTPServer, threading.Thread, str]:
    port = _free_port()
    server = HTTPServer(("127.0.0.1", port), handler_cls)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    return server, thread, f"http://127.0.0.1:{port}"


def test_post_predict_503_surfaces_server_message(of3_harness_module):
    """A 503 with the OF3 error body puts the server's ``message`` in the raised error."""
    body = b'{"error": true, "message": "MSA server budget exceeded"}'
    server, thread, url = _run_predict_server(_predict_handler(503, body))
    try:
        with pytest.raises(RuntimeError, match="MSA server budget exceeded"):
            of3_harness_module._post_predict(url, {"data": {}}, timeout_seconds=5)
    finally:
        server.shutdown()
        server.server_close()


def test_post_predict_429_surfaces_actionable_message(of3_harness_module):
    """A 429 from ``traffic.max_concurrency`` overload surfaces something actionable, not just the status line."""
    body = b'{"error": "Too many requests"}'
    server, thread, url = _run_predict_server(_predict_handler(429, body, {"Retry-After": "5"}))
    try:
        with pytest.raises(RuntimeError, match="Too many requests"):
            of3_harness_module._post_predict(url, {"data": {}}, timeout_seconds=5)
    finally:
        server.shutdown()
        server.server_close()


def test_post_predict_non_json_error_body_does_not_raise_json_error(of3_harness_module):
    """A non-JSON / empty error body still produces a usable RuntimeError, not a JSON decode error."""
    server, thread, url = _run_predict_server(_predict_handler(503, b""))
    try:
        with pytest.raises(RuntimeError, match="HTTP 503"):
            of3_harness_module._post_predict(url, {"data": {}}, timeout_seconds=5)
    finally:
        server.shutdown()
        server.server_close()


def test_post_predict_success_path_unaffected(of3_harness_module):
    body = b'{"structures": [{"content": "ATOM 1"}]}'
    server, thread, url = _run_predict_server(_predict_handler(200, body))
    try:
        data, elapsed = of3_harness_module._post_predict(url, {"data": {}}, timeout_seconds=5)
        assert data["structures"][0]["content"] == "ATOM 1"
        assert elapsed >= 0
    finally:
        server.shutdown()
        server.server_close()


# ---------------------------------------------------------------------------
# overload_sheds_requests — the load-shedding contract
# ---------------------------------------------------------------------------


def test_expected_in_flight_prefers_the_explicit_override(of3_harness_module, monkeypatch):
    monkeypatch.setenv("OPENFOLD3_MAX_CONCURRENCY", "5")
    monkeypatch.setenv("AIM_ACCELERATOR_COUNT", "8")

    assert of3_harness_module._expected_in_flight() == 5


def test_expected_in_flight_scales_with_accelerators(of3_harness_module, monkeypatch):
    """Mirrors the service: three in flight per worker, one worker per accelerator."""
    monkeypatch.delenv("OPENFOLD3_MAX_CONCURRENCY", raising=False)
    monkeypatch.setenv("AIM_ACCELERATOR_COUNT", "4")

    assert of3_harness_module._expected_in_flight() == 12


@pytest.mark.parametrize("bogus", ["", "auto", "0", "-1"])
def test_expected_in_flight_falls_back_to_single_accelerator(of3_harness_module, monkeypatch, bogus):
    """An unusable count must not yield 0, which would expect every request shed."""
    monkeypatch.setenv("OPENFOLD3_MAX_CONCURRENCY", bogus)
    monkeypatch.setenv("AIM_ACCELERATOR_COUNT", bogus)

    assert of3_harness_module._expected_in_flight() == 3


class _SheddingHandler(BaseHTTPRequestHandler):
    """Stands in for the service's max-concurrency middleware.

    Admits up to ``in_flight`` overlapping requests and refuses the rest with
    429, holding each admitted one long enough that the burst genuinely
    overlaps — the same shape as the real semaphore.
    """

    in_flight = 3
    # Long enough that the whole burst arrives while the first admissions are
    # still held: too short and a second wave slips in, and the stub stops
    # modelling a saturated limit.
    hold_s = 2.0
    _lock = threading.Lock()
    _active = 0

    def do_POST(self):  # noqa: N802
        self.rfile.read(int(self.headers.get("Content-Length", 0)))
        with self._lock:
            admitted = type(self)._active < type(self).in_flight
            if admitted:
                type(self)._active += 1
        if not admitted:
            self.send_response(429)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(b'{"error": "Too many requests"}')
            return
        try:
            time.sleep(type(self).hold_s)
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(b'{"error": false, "structures": []}')
        finally:
            with self._lock:
                type(self)._active -= 1

    def log_message(self, *_args):
        return


def _run_threaded_predict_server(handler_cls) -> tuple[ThreadingHTTPServer, str]:
    port = _free_port()
    server = ThreadingHTTPServer(("127.0.0.1", port), handler_cls)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    return server, f"http://127.0.0.1:{port}"


def _shed_check(module, url, **cfg):
    harness = module.OpenFold3Harness()
    config = SimpleNamespace(
        resolve_service_url=lambda: url,
        timeout_seconds=10,
        get=lambda key, default=None: cfg.get(key, default),
    )
    return harness._overload_shed_check(url, config)


def test_overload_check_passes_when_the_excess_is_shed(of3_harness_module, monkeypatch):
    monkeypatch.delenv("OPENFOLD3_MAX_CONCURRENCY", raising=False)
    monkeypatch.setenv("AIM_ACCELERATOR_COUNT", "1")

    class Handler(_SheddingHandler):
        in_flight = 3
        hold_s = 2.0
        _lock = threading.Lock()
        _active = 0

    server, url = _run_threaded_predict_server(Handler)
    try:
        result = _shed_check(of3_harness_module, url, overload_requests=10)
    finally:
        server.shutdown()
        server.server_close()

    assert result.success is True
    assert "3 accepted" in result.detail
    assert "7 shed" in result.detail


def test_overload_check_fails_when_nothing_is_shed(of3_harness_module, monkeypatch):
    """The regression this guards: every request accepted and queued instead of refused."""
    monkeypatch.delenv("OPENFOLD3_MAX_CONCURRENCY", raising=False)
    monkeypatch.setenv("AIM_ACCELERATOR_COUNT", "1")

    class Handler(_SheddingHandler):
        in_flight = 100
        hold_s = 0.05
        _lock = threading.Lock()
        _active = 0

    server, url = _run_threaded_predict_server(Handler)
    try:
        result = _shed_check(of3_harness_module, url, overload_requests=10)
    finally:
        server.shutdown()
        server.server_close()

    assert result.success is False
    assert "10 accepted" in result.detail
    assert "0 shed" in result.detail


def test_overload_check_reports_unexpected_status_codes(of3_harness_module, monkeypatch):
    """A 500 must not be silently counted as shedding just because it is not a 200."""
    monkeypatch.delenv("OPENFOLD3_MAX_CONCURRENCY", raising=False)
    monkeypatch.setenv("AIM_ACCELERATOR_COUNT", "1")

    server, _thread, url = _run_predict_server(_predict_handler(500, b'{"error": true}'))
    try:
        result = _shed_check(of3_harness_module, url, overload_requests=4)
    finally:
        server.shutdown()
        server.server_close()

    assert result.success is False
    assert "unexpected outcomes" in result.detail


def test_overload_check_refuses_a_burst_that_cannot_exceed_the_limit(of3_harness_module, monkeypatch):
    """Fewer requests than the limit can never shed, so the check would pass vacuously."""
    monkeypatch.setenv("OPENFOLD3_MAX_CONCURRENCY", "10")
    monkeypatch.setenv("AIM_ACCELERATOR_COUNT", "1")

    result = _shed_check(of3_harness_module, "http://127.0.0.1:1", overload_requests=4)

    assert result.success is False
    assert "misconfigured check" in result.detail
