# Copyright © Advanced Micro Devices, Inc., or its affiliates.
#
# SPDX-License-Identifier: MIT

"""Import the real REINVENT4 harness and drive it against a stub HTTP service.

Only ``aim_runtime`` is needed (it is part of this repository); no torch,
bentoml or reinvent. The stub answers ``/health`` and ``POST /v1/inference``.
"""

from __future__ import annotations

import importlib.util
import json
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
from typing import Any, Iterator

import pytest

pytest.importorskip("aim_runtime.harness")
from aim_runtime.harness import CheckScope, HarnessConfig  # noqa: E402

HARNESS_PATH = Path(__file__).resolve().parents[1] / "src" / "harness.py"


@pytest.fixture(scope="module")
def harness_mod() -> Any:
    spec = importlib.util.spec_from_file_location("reinvent4_harness_under_test", HARNESS_PATH)
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


class _Stub:
    """Mutable behaviour for the stub server."""

    def __init__(self) -> None:
        self.health_status = 200
        self.inference_status = 200
        self.smiles = ["CCO", "c1ccccc1"]
        self.requests: list[dict] = []


@pytest.fixture
def stub() -> Iterator[tuple[_Stub, str]]:
    state = _Stub()

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *_args: Any) -> None:  # silence
            pass

        def _send(self, status: int, body: dict) -> None:
            raw = json.dumps(body).encode()
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(raw)))
            self.end_headers()
            self.wfile.write(raw)

        def do_GET(self) -> None:  # noqa: N802
            if self.path == "/health":
                self._send(state.health_status, {"status": "ok"})
            else:
                self._send(404, {})

        def do_POST(self) -> None:  # noqa: N802
            body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            state.requests.append(body)
            if state.inference_status != 200:
                self._send(state.inference_status, {"error": "boom"})
                return
            self._send(200, {"smiles": state.smiles, "num_generated": len(state.smiles)})

    server = HTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield state, f"http://127.0.0.1:{server.server_port}"
    finally:
        server.shutdown()
        server.server_close()


def _config(url: str, **kw: Any) -> HarnessConfig:
    return HarnessConfig(profile={}, service_url=url, timeout_seconds=3, **kw)


def test_health_check_true_and_false(harness_mod: Any, stub: tuple[_Stub, str]) -> None:
    state, url = stub
    h = harness_mod.REINVENT4Harness()
    assert h.health_check(url, timeout_seconds=3) is True
    state.health_status = 503
    assert h.health_check(url, timeout_seconds=1) is False


def test_post_inference_wraps_body_in_data(harness_mod: Any, stub: tuple[_Stub, str]) -> None:
    state, url = stub
    resp = harness_mod._post_inference(url, num_smiles=7)
    assert resp["num_generated"] == 2
    assert state.requests == [{"data": {"num_smiles": 7}}]


def test_sampling_smoke_happy_and_failing(harness_mod: Any, stub: tuple[_Stub, str]) -> None:
    state, url = stub
    check, n = harness_mod.REINVENT4Harness._run_sampling_smoke(url)
    assert check.success and n == 2
    state.inference_status = 500
    check, n = harness_mod.REINVENT4Harness._run_sampling_smoke(url)
    assert not check.success and n == 0
    state.inference_status = 200
    state.smiles = []
    check, _ = harness_mod.REINVENT4Harness._run_sampling_smoke(url)
    assert not check.success


def test_benchmark_success_and_partial_failure(harness_mod: Any, stub: tuple[_Stub, str]) -> None:
    state, url = stub
    h = harness_mod.REINVENT4Harness()
    ok = h.benchmark(_config(url))
    assert ok.success and ok.metrics["smiles_generated"] == 2 * harness_mod._BENCH_TIMED_ITERS
    state.inference_status = 500
    bad = h.benchmark(_config(url))
    assert not bad.success


def test_benchmark_unreachable_service_fails(harness_mod: Any, stub: tuple[_Stub, str]) -> None:
    state, url = stub
    state.health_status = 503
    cfg = HarnessConfig(profile={}, service_url=url, timeout_seconds=1)
    assert not harness_mod.REINVENT4Harness().benchmark(cfg).success


def test_validate_offline_scope_runs_sampling(harness_mod: Any, stub: tuple[_Stub, str]) -> None:
    state, url = stub
    h = harness_mod.REINVENT4Harness()
    # RUNTIME checks spawn subprocesses (reinvent, torch); replace them with fixed results.
    from aim_runtime.harness import CheckResult, CheckResultType

    ok = CheckResult(name="x", result_type=CheckResultType.PASS_FAIL, success=True, value=True, detail="")
    for name in ("_check_importable", "_check_cli_executable", "_check_gpu_visible"):
        setattr(harness_mod.REINVENT4Harness, name, staticmethod(lambda: ok))
    result = h.validate(_config(url, check_scopes={CheckScope.OFFLINE}))
    assert result.success
    assert any(c.name == "sampling_smoke" for c in result.checks)
    assert len(state.requests) == 1
    runtime_only = h.validate(_config(url, check_scopes={CheckScope.RUNTIME}))
    assert not any(c.name == "sampling_smoke" for c in runtime_only.checks)


def test_validate_offline_checks_health_first(harness_mod: Any, stub: tuple[_Stub, str]) -> None:
    state, url = stub
    state.health_status = 503
    cfg = HarnessConfig(profile={}, service_url=url, timeout_seconds=1, check_scopes={CheckScope.OFFLINE})
    result = harness_mod.REINVENT4Harness().validate(cfg)
    assert not result.success
    assert any(c.name == "service_health" and not c.success for c in result.checks)
    assert state.requests == []


def test_benchmark_and_evaluate_skip_without_offline_scope(harness_mod: Any, stub: tuple[_Stub, str]) -> None:
    state, url = stub
    h = harness_mod.REINVENT4Harness()
    cfg = _config(url, check_scopes={CheckScope.RUNTIME})
    for fn in (h.benchmark, h.evaluate):
        result = fn(cfg)
        assert result.success is True
        assert result.checks and all(c.skipped for c in result.checks)
    assert state.requests == []


def _fake_rdkit(monkeypatch: pytest.MonkeyPatch, invalid: set[str]) -> None:
    import sys
    import types

    chem = types.SimpleNamespace(MolFromSmiles=lambda s: None if s in invalid else object())
    mod = types.ModuleType("rdkit")
    mod.Chem = chem  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "rdkit", mod)


def test_smiles_validity_checks_every_smiles(harness_mod: Any, monkeypatch: pytest.MonkeyPatch) -> None:
    _fake_rdkit(monkeypatch, {"bad"})
    fn = harness_mod.REINVENT4Harness._check_smiles_validity_from_list
    assert fn(["CCO", "c1ccccc1"]).success is True
    # First SMILES valid, a later one invalid: must fail and report the fraction.
    check = fn(["CCO", "bad", "CC"])
    assert check.success is False
    assert "2/3" in check.detail and "'bad'" in check.detail


def test_smiles_validity_fails_without_rdkit(harness_mod: Any, monkeypatch: pytest.MonkeyPatch) -> None:
    import builtins

    real_import = builtins.__import__

    def fake_import(name: str, *a: Any, **k: Any) -> Any:
        if name == "rdkit" or name.startswith("rdkit."):
            raise ImportError(name)
        return real_import(name, *a, **k)

    monkeypatch.setattr(builtins, "__import__", fake_import)
    check = harness_mod.REINVENT4Harness._check_smiles_validity_from_list(["CCO"])
    assert check.success is False
    assert "RDKit" in check.detail


def test_offline_only_validate_skips_local_runtime_checks(
    harness_mod: Any, stub: tuple[_Stub, str], monkeypatch: pytest.MonkeyPatch
) -> None:
    """Validating a remote service must not depend on REINVENT4 or a GPU in the caller's environment."""
    _, url = stub

    def boom() -> None:
        raise AssertionError("local RUNTIME check ran for an OFFLINE-only validation")

    for name in ("_check_importable", "_check_cli_executable", "_check_gpu_visible"):
        monkeypatch.setattr(harness_mod.REINVENT4Harness, name, staticmethod(boom))
    result = harness_mod.REINVENT4Harness().validate(_config(url, check_scopes={CheckScope.OFFLINE}))
    assert result.success
    assert [(c.name, c.success) for c in result.checks] == [("service_health", True), ("sampling_smoke", True)]


def test_configured_timeout_reaches_every_inference_call(
    harness_mod: Any, stub: tuple[_Stub, str], monkeypatch: pytest.MonkeyPatch
) -> None:
    _, url = stub
    seen: list[float] = []

    def fake_post(service_url: str, num_smiles: int, timeout: float = 540) -> dict:
        seen.append(timeout)
        return {"smiles": ["CCO"], "num_generated": 1}

    monkeypatch.setattr(harness_mod, "_post_inference", fake_post)
    cfg = HarnessConfig(profile={}, service_url=url, timeout_seconds=7, check_scopes={CheckScope.OFFLINE})
    h = harness_mod.REINVENT4Harness()
    h.benchmark(cfg)
    h.evaluate(cfg)
    h._run_sampling_smoke(url, timeout=cfg.timeout_seconds)
    assert seen and set(seen) == {7}
    assert len(seen) == harness_mod._BENCH_WARMUP_ITERS + harness_mod._BENCH_TIMED_ITERS + 1 + 1
