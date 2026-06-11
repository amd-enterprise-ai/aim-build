#!/usr/bin/env python3

# Copyright © Advanced Micro Devices, Inc., or its affiliates.
#
# SPDX-License-Identifier: MIT

"""
EchoHarness — example harness for the play "echo-model".

Demonstrates how a model author implements ModelHarness for a specialized
(non-vLLM) model. This file gets copied to /workspace/model/src/harness.py
by the specialized Dockerfile and is auto-discovered by aim-runtime.

The echo-model is a trivial BentoML service:
  GET  /healthz        -> 200 (BentoML built-in readiness probe)
  POST /predict        -> {"reversed": "...", "length": N}
  POST /predict_batch  -> [{"reversed": "...", "length": N}, ...]

This harness validates, benchmarks, and evaluates it — showing the full
lifecycle with real HTTP calls against the service.
"""

from __future__ import annotations

import json
import logging
import time
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

# Sibling import — test_cases.py lives next to this file in /workspace/model/src/.
# This works because discovery.py adds /workspace/model/src to sys.path before
# loading this module, and the Dockerfile.aim-*-base images install a .pth
# file that does the same at the Python level.
from test_cases import KNOWN_INPUTS, expected_output

from aim_runtime.harness import (
    CheckInfo,
    CheckResult,
    CheckResultType,
    CheckScope,
    HarnessConfig,
    HarnessResult,
    ModelHarness,
)

logger = logging.getLogger(__name__)

DEFAULT_PORT = 3000


class EchoHarness(ModelHarness):
    """Harness for the play echo-model.

    Shows the minimal implementation a model author needs:
      1. list_checks()  — declare what checks exist
      2. health_check() — override if your engine isn't OpenAI-compatible
      3. validate()     — runtime + offline checks
      4. benchmark()    — measure throughput
      5. evaluate()     — measure accuracy
    """

    # ------------------------------------------------------------------ #
    # 1. Declare available checks
    # ------------------------------------------------------------------ #

    CHECKS: list[CheckInfo] = [
        CheckInfo("health", CheckResultType.PASS_FAIL, CheckScope.RUNTIME, "Service /healthz returns 200"),
        CheckInfo(
            "single_predict", CheckResultType.PASS_FAIL, CheckScope.RUNTIME, "Single prediction returns expected shape"
        ),
        CheckInfo(
            "correctness", CheckResultType.SCORE, CheckScope.OFFLINE, "Fraction of test cases producing correct output"
        ),
        CheckInfo(
            "throughput_rps", CheckResultType.SCORE, CheckScope.OFFLINE, "Requests per second on single-item predict"
        ),
        CheckInfo(
            "batch_predict", CheckResultType.PASS_FAIL, CheckScope.OFFLINE, "Batch prediction returns correct count"
        ),
    ]

    def list_checks(self) -> list[CheckInfo]:
        return list(self.CHECKS)

    # ------------------------------------------------------------------ #
    # 2. Override health_check — echo-model uses /healthz, not /v1/models
    # ------------------------------------------------------------------ #

    def health_check(self, service_url: str, timeout_seconds: int = 60) -> bool:
        deadline = time.time() + timeout_seconds
        while time.time() < deadline:
            try:
                with urlopen(Request(f"{service_url}/healthz"), timeout=5) as resp:
                    if resp.status == 200:
                        return True
            except (HTTPError, URLError, OSError):
                pass
            time.sleep(1.0)
        return False

    # ------------------------------------------------------------------ #
    # 3. validate — runtime + offline checks
    # ------------------------------------------------------------------ #

    def validate(self, config: HarnessConfig) -> HarnessResult:
        url = self._url(config)
        checks: list[CheckResult] = []

        # Runtime check: health
        healthy = self.health_check(url, timeout_seconds=config.timeout_seconds)
        checks.append(
            CheckResult(
                "health",
                CheckResultType.PASS_FAIL,
                healthy,
                healthy,
                "" if healthy else f"Service unreachable at {url}",
            )
        )
        if not healthy:
            return HarnessResult(success=False, summary="Health check failed", checks=checks)

        # Runtime check: single predict shape
        checks.append(self._check_single_predict(url))

        # Offline checks (only when offline scope is requested)
        if CheckScope.OFFLINE in config.check_scopes:
            checks.append(self._check_batch_predict(url))

        success = all(c.success for c in checks)
        return HarnessResult(
            success=success,
            summary=f"Echo validation: {sum(c.success for c in checks)}/{len(checks)} checks passed",
            checks=checks,
        )

    # ------------------------------------------------------------------ #
    # 4. benchmark — measure throughput
    # ------------------------------------------------------------------ #

    def benchmark(self, config: HarnessConfig) -> HarnessResult:
        url = self._url(config)

        if not self.health_check(url, timeout_seconds=config.timeout_seconds):
            return HarnessResult(success=False, summary="Cannot benchmark — service unreachable")

        n_requests = config.get("benchmark_n_requests", 200)
        payload = json.dumps({"text": "benchmark payload"}).encode()

        successes = 0
        t0 = time.time()
        for _ in range(n_requests):
            try:
                req = Request(f"{url}/predict", data=payload, headers={"Content-Type": "application/json"})
                with urlopen(req, timeout=10):
                    successes += 1
            except Exception:
                pass
        elapsed = time.time() - t0

        rps = successes / elapsed if elapsed > 0 else 0.0

        checks = [
            CheckResult(
                name="throughput_rps",
                result_type=CheckResultType.SCORE,
                success=successes > 0,
                value=round(rps, 1),
                detail=f"{successes}/{n_requests} ok in {elapsed:.2f}s = {rps:.1f} req/s",
            )
        ]

        return HarnessResult(
            success=successes == n_requests,
            summary=f"Echo benchmark: {rps:.1f} req/s ({successes}/{n_requests} succeeded)",
            checks=checks,
            metrics={"rps": round(rps, 1), "elapsed_s": round(elapsed, 2), "succeeded": successes, "total": n_requests},
        )

    # ------------------------------------------------------------------ #
    # 5. evaluate — measure correctness
    # ------------------------------------------------------------------ #

    def evaluate(self, config: HarnessConfig) -> HarnessResult:
        url = self._url(config)

        if not self.health_check(url, timeout_seconds=config.timeout_seconds):
            return HarnessResult(success=False, summary="Cannot evaluate — service unreachable")

        correct = 0
        details: list[str] = []

        for case in KNOWN_INPUTS:
            try:
                text = str(case["text"])
                result = self._predict(url, text)
                want = expected_output(text)
                if result.get("reversed") == want["reversed"] and result.get("length") == want["length"]:
                    correct += 1
                else:
                    details.append(
                        f"FAIL '{text}': got reversed={result.get('reversed')!r}, " f"length={result.get('length')}"
                    )
            except Exception as exc:
                details.append(f"ERROR '{case['text']}': {exc}")

        accuracy = correct / len(KNOWN_INPUTS) if KNOWN_INPUTS else 0.0
        all_correct = correct == len(KNOWN_INPUTS)

        checks = [
            CheckResult(
                name="correctness",
                result_type=CheckResultType.SCORE,
                success=all_correct,
                value=round(accuracy, 4),
                detail=f"{correct}/{len(KNOWN_INPUTS)} correct" + (f" — {'; '.join(details)}" if details else ""),
            )
        ]

        return HarnessResult(
            success=all_correct,
            summary=f"Echo evaluation: {accuracy:.0%} accuracy ({correct}/{len(KNOWN_INPUTS)})",
            checks=checks,
            metrics={"accuracy": round(accuracy, 4), "correct": correct, "total": len(KNOWN_INPUTS)},
        )

    # ------------------------------------------------------------------ #
    # Helpers
    # ------------------------------------------------------------------ #

    @staticmethod
    def _url(config: HarnessConfig) -> str:
        return config.resolve_service_url()

    @staticmethod
    def _predict(service_url: str, text: str) -> dict[str, Any]:
        payload = json.dumps({"text": text}).encode()
        req = Request(f"{service_url}/predict", data=payload, headers={"Content-Type": "application/json"})
        with urlopen(req, timeout=10) as resp:
            return json.loads(resp.read().decode())

    @classmethod
    def _check_single_predict(cls, service_url: str) -> CheckResult:
        """Verify /predict returns the expected keys."""
        try:
            result = cls._predict(service_url, "test")
            if "reversed" in result and "length" in result:
                return CheckResult(
                    "single_predict",
                    CheckResultType.PASS_FAIL,
                    True,
                    True,
                    f"reversed={result['reversed']!r}, length={result['length']}",
                )
            return CheckResult("single_predict", CheckResultType.PASS_FAIL, False, False, f"Missing keys: {result}")
        except Exception as exc:
            return CheckResult("single_predict", CheckResultType.PASS_FAIL, False, False, str(exc))

    @staticmethod
    def _check_batch_predict(service_url: str) -> CheckResult:
        """Verify /predict_batch returns a list of correct length."""
        try:
            batch = ["alpha", "beta", "gamma"]
            payload = json.dumps({"texts": batch}).encode()
            req = Request(f"{service_url}/predict_batch", data=payload, headers={"Content-Type": "application/json"})
            with urlopen(req, timeout=10) as resp:
                result = json.loads(resp.read().decode())

            if isinstance(result, list) and len(result) == len(batch):
                return CheckResult(
                    "batch_predict",
                    CheckResultType.PASS_FAIL,
                    True,
                    True,
                    f"Batch of {len(batch)} returned {len(result)} results",
                )
            return CheckResult(
                "batch_predict",
                CheckResultType.PASS_FAIL,
                False,
                False,
                f"Expected list of {len(batch)}, got {type(result).__name__}({len(result) if isinstance(result, list) else '?'})",
            )
        except Exception as exc:
            return CheckResult("batch_predict", CheckResultType.PASS_FAIL, False, False, str(exc))
