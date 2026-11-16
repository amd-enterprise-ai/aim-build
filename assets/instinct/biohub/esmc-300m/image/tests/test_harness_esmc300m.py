# Copyright © Advanced Micro Devices, Inc., or its affiliates.
#
# SPDX-License-Identifier: MIT

"""Harness tests: the real ``harness.py`` against a local stub HTTP server."""

from __future__ import annotations

import time


def _scope(harness_module):
    return harness_module.CheckScope


def _cfg(make_config, harness_module, url, *scopes):
    return make_config(url, check_scopes=set(scopes) or set(_scope(harness_module)))


def _embed_ok(_body):
    return 200, {"embedding": [0.1] * 960, "embedding_dim": 960, "model": "biohub/ESMC-300M"}


def test_checks_and_scopes(harness_module):
    checks = {c.name: c for c in harness_module.ESMCHarness().list_checks()}
    assert set(checks) == {
        "bentoml_health",
        "embed_smoke",
        "sequences_per_sec",
        "embedding_shape",
        "masked_lm_perplexity",
    }
    assert checks["bentoml_health"].scope == _scope(harness_module).OFFLINE


def test_benchmark_sequences_have_exact_length_and_valid_alphabet(harness_module):
    for length in harness_module.BENCHMARK_LENGTHS:
        seq = harness_module.benchmark_sequence(length)
        assert len(seq) == length
        assert set(seq) <= set("ACDEFGHIKLMNPQRSTVWY")
    assert max(harness_module.BENCHMARK_LENGTHS) == 2046


def test_reference_sequences_valid(harness_module):
    for seq in [harness_module.SMOKE_SEQUENCE, *harness_module.MLM_EVAL_SEQUENCES]:
        assert set(seq) <= set("ACDEFGHIKLMNPQRSTVWY")
    assert len(harness_module.UBIQUITIN) == 76
    assert len(harness_module.GFP) == 238


def test_health_check_polls_health_only(harness_module, stub_server):
    stub_server.on("GET", "/health", lambda _b: (200, {"status": "ok"}))
    assert harness_module.ESMCHarness().health_check(stub_server.url, timeout_seconds=5) is True
    assert stub_server.calls == [("GET", "/health")]


def test_health_check_times_out_on_monotonic_deadline(harness_module, stub_server, monkeypatch):
    stub_server.on("GET", "/health", lambda _b: (503, {"status": "unavailable"}))
    real_sleep = time.sleep
    monkeypatch.setattr(harness_module.time, "sleep", lambda _s: real_sleep(0.01))
    start = time.monotonic()
    assert harness_module.ESMCHarness().health_check(stub_server.url, timeout_seconds=1) is False
    assert time.monotonic() - start < 5
    assert stub_server.count("GET", "/health") >= 1


def test_validate_runtime_scope_does_not_probe_health(harness_module, stub_server, make_config):
    cfg = _cfg(make_config, harness_module, stub_server.url, _scope(harness_module).RUNTIME)
    result = harness_module.ESMCHarness().validate(cfg)
    assert result.success is True and result.checks == []
    assert stub_server.calls == []


def test_validate_offline_scope_passes_with_healthy_service(harness_module, stub_server, make_config):
    stub_server.on("GET", "/health", lambda _b: (200, {"status": "ok"}))
    stub_server.on("POST", "/v1/inference", _embed_ok)
    cfg = _cfg(make_config, harness_module, stub_server.url, _scope(harness_module).OFFLINE)
    result = harness_module.ESMCHarness().validate(cfg)
    assert result.success is True
    assert {c.name for c in result.checks} == {"bentoml_health", "embed_smoke"}


def test_validate_offline_scope_fails_on_wrong_dimension(harness_module, stub_server, make_config):
    stub_server.on("GET", "/health", lambda _b: (200, {"status": "ok"}))
    stub_server.on("POST", "/v1/inference", lambda _b: (200, {"embedding": [0.1] * 8, "embedding_dim": 8}))
    cfg = _cfg(make_config, harness_module, stub_server.url, _scope(harness_module).OFFLINE)
    assert harness_module.ESMCHarness().validate(cfg).success is False


def _bench(harness_module, stub_server, make_config, handler):
    stub_server.on("GET", "/health", lambda _b: (200, {"status": "ok"}))
    stub_server.on("POST", "/v1/inference", handler)
    cfg = _cfg(make_config, harness_module, stub_server.url, _scope(harness_module).OFFLINE)
    return harness_module.ESMCHarness().benchmark(cfg)


def test_benchmark_all_requests_succeed(harness_module, stub_server, make_config):
    result = _bench(harness_module, stub_server, make_config, _embed_ok)
    assert result.success is True
    assert result.metrics["n_timed_failed"] == 0
    assert result.metrics["n_timed_total"] == len(harness_module.BENCHMARK_LENGTHS) * 10


def test_benchmark_one_timed_failure_fails_the_run(harness_module, stub_server, make_config):
    count = {"n": 0}

    def handler(body):
        count["n"] += 1
        return (500, {"detail": "boom"}) if count["n"] == 5 else _embed_ok(body)  # 2 warm-ups, so a timed one

    result = _bench(harness_module, stub_server, make_config, handler)
    assert result.success is False
    assert result.checks[0].success is False
    assert result.metrics["n_timed_failed"] == 1


def test_benchmark_all_failures_fails_the_run(harness_module, stub_server, make_config):
    result = _bench(harness_module, stub_server, make_config, lambda _b: (500, {"detail": "boom"}))
    assert result.success is False


def test_benchmark_unreachable_service_fails(harness_module, make_config, monkeypatch):
    h = harness_module.ESMCHarness()
    monkeypatch.setattr(h, "health_check", lambda *a, **k: False)
    cfg = _cfg(make_config, harness_module, "http://127.0.0.1:9", _scope(harness_module).OFFLINE)
    assert h.benchmark(cfg).success is False


def test_evaluate_perplexity_gate(harness_module, stub_server, make_config):
    stub_server.on("GET", "/health", lambda _b: (200, {"status": "ok"}))

    def inference(loss: float):
        # one route, two modes: embedding (default) and masked_lm
        return lambda b: (200, {"masked_positions": 10, "loss": loss}) if b.get("mode") == "masked_lm" else _embed_ok(b)

    stub_server.on("POST", "/v1/inference", inference(1.0))
    cfg = _cfg(make_config, harness_module, stub_server.url, _scope(harness_module).OFFLINE)
    assert harness_module.ESMCHarness().evaluate(cfg).success is True
    stub_server.on("POST", "/v1/inference", inference(3.5))  # exp(3.5)=33 > 8
    assert harness_module.ESMCHarness().evaluate(cfg).success is False


def test_post_http_error_is_a_failed_check(harness_module, stub_server):
    stub_server.on("POST", "/v1/inference", lambda _b: (400, {"detail": "bad"}))
    check = harness_module.ESMCHarness._embed_smoke_check(stub_server.url)
    assert check.success is False


def test_benchmark_and_evaluate_skip_without_offline_scope(harness_module, stub_server, make_config):
    cfg = _cfg(make_config, harness_module, stub_server.url, _scope(harness_module).RUNTIME)
    for fn in (harness_module.ESMCHarness().benchmark, harness_module.ESMCHarness().evaluate):
        result = fn(cfg)
        assert result.success is True
        assert result.checks and all(c.skipped for c in result.checks)
    assert stub_server.calls == []
