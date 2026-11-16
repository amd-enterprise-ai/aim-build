# Copyright © Advanced Micro Devices, Inc., or its affiliates.
#
# SPDX-License-Identifier: MIT

"""Real-harness tests: import harness.py and drive it against a stub HTTP service."""

from __future__ import annotations

import importlib.util
import json
import math
import re
import sys
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path

import pytest

pytest.importorskip("aim_runtime")

from aim_runtime.harness import CheckScope, HarnessConfig  # noqa: E402

HARNESS_PATH = Path(__file__).resolve().parents[1] / "src/harness.py"


@pytest.fixture(scope="module")
def harness():
    spec = importlib.util.spec_from_file_location("proteinmpnn_harness_behaviour", HARNESS_PATH)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


class _Stub:
    """Threaded stub of the service: /health and /v1/inference with switchable behaviour."""

    def __init__(self) -> None:
        self.health_status = 200
        self.inference_status = 200
        self.sequences = ["ACDE", "GHIK", "LMNP", "QRST"]
        self.scores = [0.5, 0.6, 0.7, 0.8]
        self.inference_calls = 0
        stub = self

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *args):  # noqa: D102
                pass

            def _send(self, status: int, body: dict) -> None:
                raw = json.dumps(body).encode()
                self.send_response(status)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(raw)))
                self.end_headers()
                self.wfile.write(raw)

            def do_GET(self):  # noqa: N802
                self._send(stub.health_status, {})

            def do_POST(self):  # noqa: N802
                self.rfile.read(int(self.headers.get("Content-Length", 0)))
                stub.inference_calls += 1
                self._send(stub.inference_status, {"sequences": stub.sequences, "scores": stub.scores})

        self.server = HTTPServer(("127.0.0.1", 0), Handler)
        self.url = f"http://127.0.0.1:{self.server.server_port}"
        threading.Thread(target=self.server.serve_forever, daemon=True).start()

    def close(self) -> None:
        self.server.shutdown()
        self.server.server_close()


@pytest.fixture
def stub(tmp_path, harness, monkeypatch):
    monkeypatch.setattr(harness, "FIXTURE_DIR", tmp_path)
    monkeypatch.setattr(harness, "SMOKE_PDB_PATH", str(tmp_path / "smoke.pdb"))
    s = _Stub()
    yield s
    s.close()


def _config(stub: _Stub, **kwargs) -> HarnessConfig:
    return HarnessConfig(profile={}, service_url=stub.url, timeout_seconds=2, **kwargs)


def test_health_check_ok_and_failing(harness, stub) -> None:
    assert harness.ProteinMPNNHarness().health_check(stub.url, timeout_seconds=2)
    stub.health_status = 503
    assert not harness.ProteinMPNNHarness().health_check(stub.url, timeout_seconds=1)


def test_post_inference_happy_and_failing(harness, stub) -> None:
    data, elapsed = harness._post_inference(stub.url, "/x.pdb", timeout_seconds=5)
    assert data["sequences"] and elapsed >= 0
    stub.inference_status = 500
    with pytest.raises(Exception):
        harness._post_inference(stub.url, "/x.pdb", timeout_seconds=5)


def test_validate_offline_passes(harness, stub) -> None:
    result = harness.ProteinMPNNHarness().validate(_config(stub))
    assert result.success
    assert {c.name for c in result.checks} >= {"bentoml_health", "inference_smoke", "sequence_count", "score_finite"}


def test_validate_failing_inference_fails(harness, stub) -> None:
    stub.inference_status = 500
    assert not harness.ProteinMPNNHarness().validate(_config(stub)).success


def test_validate_runtime_scope_does_not_probe_and_is_explicit(harness, stub) -> None:
    stub.health_status = 503  # would fail if probed
    result = harness.ProteinMPNNHarness().validate(_config(stub, check_scopes={CheckScope.RUNTIME}))
    assert stub.inference_calls == 0
    assert len(result.checks) == 1 and result.checks[0].skipped


def test_benchmark_success_and_failure(harness, stub, monkeypatch) -> None:
    monkeypatch.setattr(harness.time, "sleep", lambda _s: None)
    ok = harness.ProteinMPNNHarness().benchmark(_config(stub, extra={"benchmark_n_requests": 1}))
    assert ok.success and ok.metrics["aa_per_second"] > 0
    assert ok.metrics["aa_calls_total"] >= ok.metrics["aa_calls_succeeded"] > 0
    assert not any(k.startswith("nim_") for k in ok.metrics)
    stub.inference_status = 500
    bad = harness.ProteinMPNNHarness().benchmark(_config(stub, extra={"benchmark_n_requests": 1}))
    assert not bad.success


def test_benchmark_partial_failure_is_not_success(harness, stub, monkeypatch) -> None:
    real = harness._post_inference
    calls = {"n": 0}

    def flaky(*args, **kwargs):
        calls["n"] += 1
        if calls["n"] == 2:
            raise RuntimeError("boom")
        return real(*args, **kwargs)

    monkeypatch.setattr(harness, "_post_inference", flaky)
    result = harness.ProteinMPNNHarness().benchmark(
        HarnessConfig(profile={}, service_url=stub.url, timeout_seconds=2, extra={"benchmark_n_requests": 3})
    )
    assert not result.success
    assert result.metrics["requests_succeeded"] == 2


def test_benchmark_unreachable_service_fails(harness) -> None:
    cfg = HarnessConfig(profile={}, service_url="http://127.0.0.1:9", timeout_seconds=1)
    assert not harness.ProteinMPNNHarness().benchmark(cfg).success


def test_fixtures_are_labelled_synthetic(harness) -> None:
    names = [n for n, _, _ in harness.BENCHMARK_PDBS]
    assert all(n.startswith("synthetic_") for n in names)
    text = HARNESS_PATH.read_text()
    assert not re.search(r"\bnims?\b", text, re.IGNORECASE)
    assert not any(code in text for code in ("1L2Y", "1YRF", "1PGB", "2GI9"))


@pytest.mark.parametrize(
    ("seq", "valid"),
    [
        ("ACDEFGHIKLMNPQRSTVWY", True),
        ("ACDXE", True),
        ("ACDE/FGHI", True),
        ("acde", True),
        ("", False),
        ("ACDE/", False),
        ("AC1E", False),
        ("ACDB", False),
    ],
)
def test_design_alphabet(harness, seq, valid) -> None:
    assert harness._is_valid_design(seq) is valid


def test_evaluate_accepts_gapped_multichain_output(harness, stub) -> None:
    stub.sequences = ["ACDX/EFGH", "AAAA/CCCC"]
    stub.scores = [0.1, 0.2]
    assert harness.ProteinMPNNHarness().evaluate(_config(stub)).success


def test_evaluate_rejects_invalid_or_nonfinite(harness, stub) -> None:
    stub.sequences = ["AC1E"]
    assert not harness.ProteinMPNNHarness().evaluate(_config(stub)).success
    stub.sequences = ["ACDE"]
    stub.scores = [math.nan]
    assert not harness.ProteinMPNNHarness().evaluate(_config(stub)).success


def test_validate_requires_one_score_per_sequence(harness, stub) -> None:
    stub.scores = [0.5]  # four sequences, one finite score
    result = harness.ProteinMPNNHarness().validate(_config(stub))
    assert not result.success
    assert not next(c for c in result.checks if c.name == "score_finite").success


@pytest.mark.parametrize(
    ("sequences", "scores"),
    [(["ACDE", "GHIK"], [0.5]), (["ACDE"], [None]), ([None], [0.5]), (["ACDE"], "0.5")],
)
def test_evaluate_malformed_response_fails_cleanly(harness, stub, sequences, scores) -> None:
    stub.sequences, stub.scores = sequences, scores
    result = harness.ProteinMPNNHarness().evaluate(_config(stub))  # must not raise
    assert not result.success


def test_inference_timeout_follows_config(harness, stub, monkeypatch) -> None:
    seen: list[int] = []
    real = harness._post_inference

    def spy(*args, **kwargs):
        seen.append(kwargs["timeout_seconds"])
        return real(*args, **kwargs)

    monkeypatch.setattr(harness, "_post_inference", spy)
    cfg = _config(stub)
    cfg.timeout_seconds = 77
    harness.ProteinMPNNHarness().validate(cfg)
    cfg.extra["inference_timeout_seconds"] = 5
    harness.ProteinMPNNHarness().validate(cfg)
    assert seen == [77, 5]


def test_dtype_probe_removed(harness) -> None:
    assert not hasattr(harness, "_query_model_dtype")
    assert not hasattr(harness, "EXPECTED_DTYPE")


def test_benchmark_and_evaluate_skip_without_offline_scope(harness, stub) -> None:
    cfg = _config(stub, check_scopes={CheckScope.RUNTIME})
    for fn in (harness.ProteinMPNNHarness().benchmark, harness.ProteinMPNNHarness().evaluate):
        result = fn(cfg)
        assert result.success is True
        assert result.checks and all(c.skipped for c in result.checks)
    assert stub.inference_calls == 0


def test_aa_per_second_is_ratio_of_sums(harness, stub, monkeypatch) -> None:
    """aa_per_second = total residues / total time, not the mean of per-call ratios."""

    class Clock:
        now = 0.0

        def monotonic(self) -> float:
            self.now += 1.0  # every read costs 1 s, so a call lasts extra + 1 s
            return self.now

        def sleep(self, _s: float) -> None:
            pass

    clock = Clock()
    timed: list[tuple[int, float]] = []
    seen: dict[str, int] = {}

    def fake_post(_url, pdb_path, num_seq_per_target=4, sampling_temp=0.1, timeout_seconds=300):
        residues = harness.count_residues(Path(pdb_path).read_text())
        clock.now += residues / 10.0  # longer structures take longer, unevenly
        if num_seq_per_target == 1:  # aa/s calls (smoke requests ask for 4)
            n = seen[pdb_path] = seen.get(pdb_path, 0) + 1
            if n > 2:  # first 2 per fixture are warm-ups
                timed.append((residues, residues / 10.0 + 1.0))
        return {"sequences": ["A"], "scores": [0.1]}, 0.5

    monkeypatch.setattr(harness, "time", clock)
    monkeypatch.setattr(harness, "_post_inference", fake_post)
    result = harness.ProteinMPNNHarness().benchmark(_config(stub, extra={"benchmark_n_requests": 1}))
    assert result.success
    expected = sum(r for r, _ in timed) / sum(t for _, t in timed)
    mean_of_ratios = sum(r / t for r, t in timed) / len(timed)
    assert expected != pytest.approx(mean_of_ratios, rel=1e-3)  # the test discriminates
    assert result.metrics["aa_per_second"] == pytest.approx(expected, abs=0.01)
    assert result.metrics["aa_total_residues"] == sum(r for r, _ in timed)
