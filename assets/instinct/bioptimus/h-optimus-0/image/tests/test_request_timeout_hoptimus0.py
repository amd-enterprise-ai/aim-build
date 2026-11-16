# Copyright © Advanced Micro Devices, Inc., or its affiliates.
#
# SPDX-License-Identifier: MIT

"""Inference requests honour ``HarnessConfig.timeout_seconds`` (default stays 120 s)."""

from __future__ import annotations

import pytest

OK_EMBEDDING = {"embedding": [0.1] * 1536}


@pytest.fixture
def seen_timeouts(harness_module, monkeypatch, stub_server):
    stub_server.on("GET", "/health", lambda _b: (200, {"status": "ok"}))
    stub_server.on("POST", "/v1/inference", lambda _b: (200, OK_EMBEDDING))
    real = harness_module.urlopen
    seen: list[float] = []

    def spy(req, timeout=None):
        if req.full_url.endswith("/v1/inference"):
            seen.append(timeout)
        return real(req, timeout=timeout)

    monkeypatch.setattr(harness_module, "urlopen", spy)
    return seen


def test_send_inference_default_timeout_is_120(harness_module, seen_timeouts, stub_server):
    harness_module.HOptimus0Harness._send_inference(stub_server.url, image_base64="QUJD")
    assert seen_timeouts == [120]


def test_send_inference_explicit_timeout(harness_module, seen_timeouts, stub_server):
    harness_module.HOptimus0Harness._send_inference(stub_server.url, image_base64="QUJD", timeout_seconds=7)
    assert seen_timeouts == [7]


def test_validate_uses_config_timeout(harness_module, make_config, seen_timeouts, stub_server):
    cfg = make_config(stub_server.url, timeout_seconds=5)
    harness_module.HOptimus0Harness().validate(cfg)
    assert seen_timeouts and set(seen_timeouts) == {5}


def test_evaluate_uses_config_timeout(harness_module, make_config, seen_timeouts, stub_server):
    cfg = make_config(stub_server.url, timeout_seconds=6)
    harness_module.HOptimus0Harness().evaluate(cfg)
    assert seen_timeouts and set(seen_timeouts) == {6}


def test_benchmark_uses_config_timeout(harness_module, make_config, seen_timeouts, stub_server):
    cfg = make_config(stub_server.url, timeout_seconds=9, extra={"benchmark_n_requests": 2, "benchmark_n_warmup": 1})
    harness_module.HOptimus0Harness().benchmark(cfg)
    assert len(seen_timeouts) == 3 and set(seen_timeouts) == {9}
