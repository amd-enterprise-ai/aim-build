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
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path

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
