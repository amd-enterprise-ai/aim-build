# Copyright © Advanced Micro Devices, Inc., or its affiliates.
#
# SPDX-License-Identifier: MIT

"""Behavioural tests of SemlaFlowHarness against a stub HTTP server.

Imports the real harness (aim_runtime is a pure-Python package) and drives it
over HTTP; no GPU, torch or bentoml is needed.
"""

from __future__ import annotations

import importlib.util
import json
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
from types import SimpleNamespace

import pytest

from aim_runtime.harness import CheckScope, HarnessConfig

HARNESS_PATH = Path(__file__).resolve().parents[1] / "src" / "harness.py"
_spec = importlib.util.spec_from_file_location("semlaflow_harness", HARNESS_PATH)
assert _spec is not None and _spec.loader is not None
harness = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(harness)


class _Stub:
    """Mutable behaviour of the stub service."""

    def __init__(self) -> None:
        self.health_status = 200
        self.smiles: object = ["CCO", "c1ccccc1"]
        self.n_molecules: int | None = None  # None = len(smiles)
        self.inference_status = 200
        self.paths: list[str] = []


@pytest.fixture
def stub():
    state = _Stub()

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *args):  # silence
            pass

        def _send(self, code: int, body: dict) -> None:
            data = json.dumps(body).encode()
            self.send_response(code)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def do_GET(self):
            state.paths.append(self.path)
            self._send(state.health_status, {"status": "ok"})

        def do_POST(self):
            state.paths.append(self.path)
            self.rfile.read(int(self.headers.get("Content-Length", 0)))
            if state.inference_status != 200:
                self._send(state.inference_status, {"error": "boom"})
                return
            n = state.n_molecules if state.n_molecules is not None else len(state.smiles)  # type: ignore[arg-type]
            self._send(200, {"smiles": state.smiles, "n_molecules": n, "elapsed_s": 0.1})

    server = HTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    state.url = f"http://127.0.0.1:{server.server_port}"  # type: ignore[attr-defined]
    yield state
    server.shutdown()
    server.server_close()


@pytest.fixture
def data_dir(tmp_path, monkeypatch):
    (tmp_path / "test.smol").write_bytes(b"x")
    monkeypatch.setenv("AIM_INPUT_PATH", str(tmp_path))
    return tmp_path


def _config(stub, scopes=None) -> HarnessConfig:
    return HarnessConfig(
        profile={"benchmark_n_timed": 2, "benchmark_n_warmup": 1},
        service_url=stub.url,
        timeout_seconds=5,
        check_scopes=scopes if scopes is not None else CheckScope.all(),
    )


def test_health_check_polls_health_only(stub):
    assert harness.SemlaFlowHarness().health_check(stub.url, timeout_seconds=3) is True
    assert set(stub.paths) == {"/health"}


def test_health_check_times_out_when_unhealthy(stub):
    stub.health_status = 503
    assert harness.SemlaFlowHarness().health_check(stub.url, timeout_seconds=1) is False
    assert "/healthz" not in stub.paths


def test_validate_happy_path(stub, data_dir):
    result = harness.SemlaFlowHarness().validate(_config(stub))
    assert result.success
    assert {c.name for c in result.checks} == {"health", "input_data_check", "smoke_inference"}


def test_validate_runtime_scope_skips_inference(stub, data_dir):
    result = harness.SemlaFlowHarness().validate(_config(stub, {CheckScope.RUNTIME}))
    assert result.success
    assert "/v1/inference" not in stub.paths
    assert {c.name for c in result.checks} == {"health", "input_data_check", "smoke_inference"}
    assert stub.paths == []
    by = {c.name: c for c in result.checks}
    assert by["health"].skipped and by["smoke_inference"].skipped and not by["input_data_check"].skipped


def test_validate_fails_on_missing_data(stub, tmp_path, monkeypatch):
    monkeypatch.setenv("AIM_INPUT_PATH", str(tmp_path))
    result = harness.SemlaFlowHarness().validate(_config(stub))
    assert not result.success
    assert not next(c for c in result.checks if c.name == "input_data_check").success


def test_validate_fails_on_non_smiles_payload(stub, data_dir):
    stub.smiles = ["not a smiles!!"]
    result = harness.SemlaFlowHarness().validate(_config(stub))
    assert not next(c for c in result.checks if c.name == "smoke_inference").success


def test_benchmark_reports_end_to_end_throughput(stub, data_dir):
    result = harness.SemlaFlowHarness().benchmark(_config(stub))
    assert result.success
    assert result.metrics["mol_per_min"] > 0
    assert "end-to-end" in result.checks[0].detail


def test_benchmark_fails_when_inference_errors(stub, data_dir):
    stub.inference_status = 500
    result = harness.SemlaFlowHarness().benchmark(_config(stub))
    assert not result.success


def test_benchmark_fails_on_zero_molecules(stub, data_dir):
    stub.smiles = []
    result = harness.SemlaFlowHarness().benchmark(_config(stub))
    assert not result.success


def test_benchmark_fails_without_input_path(stub, monkeypatch):
    monkeypatch.delenv("AIM_INPUT_PATH", raising=False)
    result = harness.SemlaFlowHarness().benchmark(_config(stub))
    assert not result.success


def test_evaluate_reports_validity_and_uniqueness(stub, data_dir):
    stub.smiles = ["CCO", "CCO", "c1ccccc1", "CC"]  # 4 returned of 8 requested, 3 distinct
    result = harness.SemlaFlowHarness().evaluate(
        HarnessConfig(
            profile={"evaluate_n_molecules": 8},
            service_url=stub.url,
            timeout_seconds=5,
            check_scopes=CheckScope.all(),
        )
    )
    assert result.success
    by = {c.name: c for c in result.checks}
    assert by["validity"].value == 0.5 and by["uniqueness"].value == 0.75
    assert result.metrics["molecules_requested"] == 8 and result.metrics["molecules_returned"] == 4
    assert "/v1/inference" in stub.paths


def test_evaluate_applies_optional_minimums(stub, data_dir):
    stub.smiles = ["CCO", "CCO"]
    cfg = HarnessConfig(
        profile={"evaluate_n_molecules": 2, "evaluate_min_uniqueness": 0.9},
        service_url=stub.url,
        timeout_seconds=5,
        check_scopes=CheckScope.all(),
    )
    result = harness.SemlaFlowHarness().evaluate(cfg)
    assert not result.success
    by = {c.name: c for c in result.checks}
    assert by["validity"].success and not by["uniqueness"].success


def test_evaluate_fails_on_no_molecules_or_garbage(stub, data_dir):
    stub.smiles = []
    assert not harness.SemlaFlowHarness().evaluate(_config(stub)).success
    stub.smiles = ["not a smiles!!"]
    assert not harness.SemlaFlowHarness().evaluate(_config(stub)).success


def test_evaluate_fails_without_data_and_skips_without_offline_scope(stub, tmp_path, monkeypatch):
    monkeypatch.setenv("AIM_INPUT_PATH", str(tmp_path))
    assert not harness.SemlaFlowHarness().evaluate(_config(stub)).success
    stub.paths.clear()
    result = harness.SemlaFlowHarness().evaluate(_config(stub, {CheckScope.RUNTIME}))
    assert result.success and all(c.skipped for c in result.checks)
    assert stub.paths == []


def test_scopes_of_checks():
    scopes = {c.name: c.scope for c in harness.SemlaFlowHarness().list_checks()}
    assert scopes["health"] == CheckScope.OFFLINE
    assert scopes["smoke_inference"] == CheckScope.OFFLINE
    assert scopes["throughput_mol_per_min"] == CheckScope.OFFLINE
    assert scopes["validity"] == scopes["uniqueness"] == CheckScope.OFFLINE
    assert scopes["input_data_check"] == CheckScope.RUNTIME


def test_benchmark_without_offline_scope_makes_no_call(stub, data_dir):
    result = harness.SemlaFlowHarness().benchmark(_config(stub, {CheckScope.RUNTIME}))
    assert result.success and result.checks[0].skipped
    assert stub.paths == []


def test_benchmark_throughput_uses_returned_count(stub, data_dir, monkeypatch):
    stub.smiles = ["CCO", "c1ccccc1"]  # 2 returned; 100 requested by default
    stub.n_molecules = None

    # Deterministic clock: every timed call (two monotonic() reads) takes exactly 2.0 s,
    # so the test does not depend on how fast the stub answers.
    ticks = {"n": 0}

    def fake_monotonic() -> float:
        n = ticks["n"]
        ticks["n"] += 1
        return (n // 2) * 100.0 + (n % 2) * 2.0

    monkeypatch.setattr(harness, "time", SimpleNamespace(monotonic=fake_monotonic, sleep=lambda _s: None))
    result = harness.SemlaFlowHarness().benchmark(_config(stub))
    assert result.success
    assert result.metrics["molecules_requested"] == 100
    assert result.metrics["molecules_returned"] == 2
    assert result.metrics["mean_latency_s"] == 2.0
    # 2 returned molecules in 2.0 s -> 60 mol/min (a requested-count figure would be 3000)
    assert result.metrics["mol_per_min"] == 60.0


def test_validate_offline_scope_skips_input_check(stub, data_dir):
    result = harness.SemlaFlowHarness().validate(_config(stub, {CheckScope.OFFLINE}))
    by_name = {c.name: c for c in result.checks}
    assert by_name["input_data_check"].skipped
    assert by_name["smoke_inference"].success


def test_validate_offline_scope_reports_missing_input_in_smoke(stub, monkeypatch, tmp_path):
    monkeypatch.setenv("AIM_INPUT_PATH", str(tmp_path))  # no test.smol here
    result = harness.SemlaFlowHarness().validate(_config(stub, {CheckScope.OFFLINE}))
    by_name = {c.name: c for c in result.checks}
    assert by_name["input_data_check"].skipped
    assert not by_name["smoke_inference"].success and not result.success
    assert "input file unavailable" in by_name["smoke_inference"].detail


def test_dockerfile_arch_matches_profile():
    import re

    root = Path(__file__).resolve().parents[2]
    docker = (root / "image" / "Dockerfile").read_text(encoding="utf-8")
    image_arch = re.search(r'^ENV PYTORCH_ROCM_ARCH="([^"]+)"', docker, re.M).group(1)
    profile = (root / "profiles" / "bentoml-mi300x-fp32-tp1-throughput.yaml").read_text(encoding="utf-8")
    profile_arch = re.search(r"^\s*PYTORCH_ROCM_ARCH:\s*(\S+)", profile, re.M).group(1)
    assert image_arch == profile_arch == "gfx942"


def test_mi355x_profile_targets_gfx950_and_keeps_the_mi300x_settings():
    import re

    root = Path(__file__).resolve().parents[2]
    mi300x = (root / "profiles" / "bentoml-mi300x-fp32-tp1-throughput.yaml").read_text(encoding="utf-8")
    mi355x = (root / "profiles" / "bentoml-mi355x-fp32-tp1-throughput.yaml").read_text(encoding="utf-8")
    assert re.search(r"^\s*accelerator_model:\s*MI355X\s*$", mi355x, re.M)
    assert re.search(r"^\s*PYTORCH_ROCM_ARCH:\s*gfx950\s*$", mi355x, re.M)
    # The MI355X profile is the MI300X profile with the model and the architecture changed, nothing else.
    expected = mi300x.replace("accelerator_model: MI300X", "accelerator_model: MI355X").replace(
        "PYTORCH_ROCM_ARCH: gfx942", "PYTORCH_ROCM_ARCH: gfx950"
    )
    assert mi355x == expected


def test_health_loop_uses_monotonic_clock():
    src = HARNESS_PATH.read_text(encoding="utf-8")
    assert "time.time()" not in src
    assert '"/healthz"' not in src
