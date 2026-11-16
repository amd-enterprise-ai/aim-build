# Copyright © Advanced Micro Devices, Inc., or its affiliates.
#
# SPDX-License-Identifier: MIT

"""
HOptimus0Harness — model-specific harness for the H-optimus-0 pathology
foundation model served via BentoML.

This file lives at ``assets/instinct/bioptimus/h-optimus-0/image/src/harness.py``
and is copied to ``/workspace/model/src/harness.py`` by the specialized
Dockerfile. aim-runtime discovers it automatically via the well-known path.

Because ``/workspace/model/src/`` is on ``sys.path`` (via a ``.pth`` file
installed by the specialized Dockerfile), this module can import sibling
modules in the same directory as well as aim-runtime's public API.

H-optimus-0 produces 1536-dimensional CLS-token embeddings from 224x224
histopathology tiles. The /v1/inference endpoint accepts one image at a time
(either a file path under AIM_INPUT_PATH or a base64-encoded image) and returns:
  {"embedding": [...], "embedding_dim": 1536, "model": "bioptimus/H-optimus-0"}

Benchmark metric ``images_per_sec``: one request = one 224x224 image, sent
sequentially (concurrency 1); value = ok / sum(latencies) over the timed
requests, excluding warm-up. Any failed timed request makes the benchmark
unsuccessful.
"""

from __future__ import annotations

import base64
import json
import logging
import struct
import time
import zlib
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from aim_runtime.harness import (
    CheckInfo,
    CheckResult,
    CheckResultType,
    CheckScope,
    HarnessConfig,
    HarnessResult,
    ModelHarness,
    skipped,
)

logger = logging.getLogger(__name__)

HEALTH_ENDPOINT = "/health"
INFERENCE_ENDPOINT = "/v1/inference"

EMBEDDING_DIM = 1536
MODEL_ID = "bioptimus/H-optimus-0"

# Default per-request timeout (seconds) for ``_send_inference`` when the caller
# does not pass one; harness entry points pass ``config.timeout_seconds``.
DEFAULT_REQUEST_TIMEOUT_SECONDS = 120


def _make_png(width: int, height: int, rgb: tuple[int, int, int]) -> bytes:
    """Build a solid-colour 8-bit RGB PNG with stdlib only (valid CRCs by construction)."""

    def chunk(tag: bytes, data: bytes) -> bytes:
        body = tag + data
        return struct.pack(">I", len(data)) + body + struct.pack(">I", zlib.crc32(body) & 0xFFFFFFFF)

    row = b"\x00" + bytes(rgb) * width  # filter type 0 + pixels
    return (
        b"\x89PNG\r\n\x1a\n"
        + chunk(b"IHDR", struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0))
        + chunk(b"IDAT", zlib.compress(row * height))
        + chunk(b"IEND", b"")
    )


# 224x224 solid white PNG — kept for smoke/validate only (lightweight, no GPU
# path needed). Generated at import time so the CRCs are correct by construction.
_SMOKE_PNG_B64 = base64.b64encode(_make_png(224, 224, (255, 255, 255))).decode("ascii")

# 224x224 synthetic H&E-profile tile used for the benchmark.
# Procedurally generated (PIL/numpy, fixed seed): an eosin-pink background
# (~RGB 230,180,210) modulated by smooth low-frequency tissue-like variation,
# with ~20 soft-edged hematoxylin-purple nucleus blobs (~RGB 120,50,110,
# randomized size/placement/color jitter) scattered across the tile, then
# quantized to a 256-color palette and PNG-optimized to keep the embedded
# base64 literal small.
#
# The committed PNG has 256 palette colours at 224x224 (decoded with PIL).
_BENCH_PNG_B64 = (
    "iVBORw0KGgoAAAANSUhEUgAAAOAAAADgCAMAAAAt85rTAAADAFBMVEXwu9rvutnuutnuutjuudjt"
    "udjtudftuNfsuNfsuNbruNbrt9bqt9XrttXqttXqtdXpttXpttTptdTottTotdTotdPntdPptNTn"
    "tNPntNLmtNLns9Lls9HlstHks9Hks9DkstDjstDjss/msdHjsc/ksNDisc/isc7isM7hsc7hsM7h"
    "r87gsM3gr83fr83fr8zer8zjrs7grc3erszersvdrsvdrcvdrcrcrcrhq83eq8vfqsveqMrbqsnd"
    "qMndp8nbpsjcpcfZpcbZo8bYosXXosTXocTVosPVoMLRoMHWn8PVnsLUn8HTnsDRnsDTnMHSncDT"
    "m8DQnL7Pmr7QmL3OmbzPl7zMmLvMlrvNlbvMlbrLlbrLk7nJlbnJk7fJkrjJkLfIj7fGj7XGjrXF"
    "j7TDj7TGjbXEjLPFi7PBjLLDirLCibHBia/CiLG/ibC/h66/hq+/ha++ha27ha2+g629g628gq28"
    "gay7g6u6gau5gKm6f6q5f6m4fqm2fqe3e6a1e6a0e6W1eaS0eKWxeaOyd6OzdqSxdKOwdaGwc6Kt"
    "c6CtcZ6ucJ+sb52sbp6pb52obZuqa52qbJmobJmnapinaJqlZ5ekZ5alZpaiZ5ahZpWkZJWhZJSf"
    "ZJOhYpOgYJOeYZKeX5GeXpKfXo2cXY6ZXo2cW4+bWo2aW42aWYyXWoqXWIqXV4qWVYuUVomVVIqV"
    "U4aSVImTUYaQUoWSUIaSToGQToKOT4KQTYGQS4GNTIKNSoGMSIGKTH+ITH6KSn+KSYGKSH6GSH2L"
    "Rn+JRn6HRn2JQ3uHQ3yFRHuEQnuBQ3mFQHuDP3qDPHiBPXZ+P3d/PHd9O3R/OnR9OHF+NnN7OHR4"
    "N3B5Nm56NXJ6NW15NW12NWxzNW16NHJ6NGx4NHN5NG91NGh9M256M3J6M2x6M2t5M3R5M295M2t1"
    "M2x6MnB6Mm14Mmt2Mm91MmxxMmt8MWp6MW94MW53MWx2MW12MGx2MGp2L252LW92LW51MXBzMWx1"
    "MG10LmtzLmpyLWqOZctwAAAiy0lEQVR42t3dCZRVxZk48CvQ/baeN/3Sy3Q/n68XX2Bg2BNNlCju"
    "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"
)

_NEEDS_SERVICE = "needs a live service (scope OFFLINE not selected)"

#: Warmup requests discarded before timing begins.
_BENCH_WARMUP_N = 2


class HOptimus0Harness(ModelHarness):
    """Harness for H-optimus-0 pathology ViT embedding model (BentoML)."""

    CHECKS: list[CheckInfo] = [
        CheckInfo(
            "bentoml_health",
            CheckResultType.PASS_FAIL,
            CheckScope.OFFLINE,
            "GET /health returns 200",
        ),
        CheckInfo(
            "smoke_inference",
            CheckResultType.PASS_FAIL,
            CheckScope.OFFLINE,
            "Single /v1/inference call returns a 1536-dim embedding",
        ),
        CheckInfo(
            "images_per_sec",
            CheckResultType.SCORE,
            CheckScope.OFFLINE,
            "Images per second: one request = one 224x224 image, sequential, ok / sum(latencies)",
        ),
        CheckInfo(
            "embedding_shape",
            CheckResultType.PASS_FAIL,
            CheckScope.OFFLINE,
            "Embedding vector has correct dimension (1536) and finite values",
        ),
    ]

    def list_checks(self) -> list[CheckInfo]:
        return list(self.CHECKS)

    def health_check(self, service_url: str, timeout_seconds: int = 60) -> bool:
        deadline = time.monotonic() + timeout_seconds
        while time.monotonic() < deadline:
            try:
                req = Request(f"{service_url}{HEALTH_ENDPOINT}")
                with urlopen(req, timeout=5) as resp:
                    if resp.status == 200:
                        return True
            except (HTTPError, URLError, OSError) as exc:
                # Service not up yet; keep polling until the deadline.
                logger.debug("Health probe %s failed, retrying: %s", HEALTH_ENDPOINT, exc)
            time.sleep(2.0)
        return False

    def validate(self, config: HarnessConfig) -> HarnessResult:
        # Every check here talks to a live service, i.e. scope OFFLINE (see
        # ``CheckScope``). Without it there is nothing to run: report skipped
        # checks rather than a vacuous pass.
        if CheckScope.OFFLINE not in config.check_scopes:
            checks = [skipped(info.name, _NEEDS_SERVICE) for info in self.CHECKS]
            return HarnessResult(success=True, summary="H-optimus-0 validation: no checks selected", checks=checks)

        service_url = self._service_url(config)
        checks: list[CheckResult] = []

        health_ok = self.health_check(service_url, timeout_seconds=config.timeout_seconds)
        checks.append(
            CheckResult(
                name="bentoml_health",
                result_type=CheckResultType.PASS_FAIL,
                success=health_ok,
                value=health_ok,
                detail="" if health_ok else f"BentoML not reachable at {service_url}",
            )
        )
        if not health_ok:
            return HarnessResult(success=False, summary="Health check failed", checks=checks)

        checks.append(self._check_smoke_inference(service_url, config.timeout_seconds))
        checks.append(self._check_embedding_shape(service_url, config.timeout_seconds))

        success = all(c.success for c in checks)
        return HarnessResult(
            success=success,
            summary=(f"H-optimus-0 validation: {sum(c.success for c in checks)}/{len(checks)} checks passed"),
            checks=checks,
        )

    def benchmark(self, config: HarnessConfig) -> HarnessResult:
        if CheckScope.OFFLINE not in config.check_scopes:
            checks = [skipped("images_per_sec", _NEEDS_SERVICE)]
            return HarnessResult(success=True, summary="H-optimus-0 benchmark: no checks selected", checks=checks)
        service_url = self._service_url(config)

        if not self.health_check(service_url, timeout_seconds=config.timeout_seconds):
            return HarnessResult(
                success=False,
                summary="Cannot benchmark — BentoML service unreachable",
            )

        n_requests = config.get("benchmark_n_requests", 20)
        n_warmup = config.get("benchmark_n_warmup", _BENCH_WARMUP_N)
        latencies: list[float] = []

        for i in range(n_warmup):
            try:
                self._send_inference(service_url, image_base64=_BENCH_PNG_B64, timeout_seconds=config.timeout_seconds)
            except Exception as exc:  # noqa: BLE001
                logger.warning("Warmup request %d/%d failed: %s", i + 1, n_warmup, exc)

        for i in range(n_requests):
            t0 = time.monotonic()
            try:
                self._send_inference(service_url, image_base64=_BENCH_PNG_B64, timeout_seconds=config.timeout_seconds)
                latencies.append(time.monotonic() - t0)
            except Exception as exc:  # noqa: BLE001 — per-request boundary
                logger.warning("Benchmark request %d/%d failed: %s", i + 1, n_requests, exc)

        if not latencies:
            return HarnessResult(success=False, summary="All benchmark requests failed")

        ok = len(latencies)
        failed = n_requests - ok
        elapsed = sum(latencies)
        avg_latency = elapsed / ok
        rps = ok / elapsed if elapsed > 0 else 0.0

        checks = [
            CheckResult(
                name="images_per_sec",
                result_type=CheckResultType.SCORE,
                success=failed == 0,
                value=round(rps, 2),
                detail=(
                    f"{ok}/{n_requests} requests succeeded, {failed} failed "
                    f"({n_warmup} warmup discarded), "
                    f"input=224x224 synthetic H&E tile, "
                    f"avg latency={avg_latency:.3f}s, "
                    f"images_per_sec={rps:.2f}"
                ),
            )
        ]

        return HarnessResult(
            # A timed request that failed makes the throughput figure unreliable:
            # fail the benchmark (the entrypoint turns success=False into exit 1).
            success=failed == 0,
            summary=f"H-optimus-0 benchmark: {rps:.2f} images/s ({ok}/{n_requests} succeeded)",
            checks=checks,
            metrics={
                "avg_latency_s": round(avg_latency, 3),
                "images_per_sec": round(rps, 2),
                "requests_succeeded": ok,
                "requests_failed": failed,
                "requests_total": n_requests,
            },
        )

    def evaluate(self, config: HarnessConfig) -> HarnessResult:
        if CheckScope.OFFLINE not in config.check_scopes:
            checks = [skipped("embedding_shape", _NEEDS_SERVICE)]
            return HarnessResult(success=True, summary="H-optimus-0 evaluation: no checks selected", checks=checks)
        service_url = self._service_url(config)

        if not self.health_check(service_url, timeout_seconds=config.timeout_seconds):
            return HarnessResult(
                success=False,
                summary="Cannot evaluate — BentoML service unreachable",
            )

        checks: list[CheckResult] = []
        checks.append(self._check_embedding_shape(service_url, config.timeout_seconds))

        success = all(c.success for c in checks)
        return HarnessResult(
            success=success,
            summary=f"H-optimus-0 evaluation {'passed' if success else 'failed'}",
            checks=checks,
            metrics={
                "expected_embedding_dim": EMBEDDING_DIM,
                "model": MODEL_ID,
            },
        )

    @staticmethod
    def _service_url(config: HarnessConfig) -> str:
        # The port comes from the runtime-resolved config (``--config`` extra,
        # then profile); 8000 is only the HarnessConfig fallback, not hardcoded here.
        return config.resolve_service_url()

    @staticmethod
    def _send_inference(
        service_url: str,
        *,
        image_path: str | None = None,
        image_base64: str | None = None,
        timeout_seconds: float = DEFAULT_REQUEST_TIMEOUT_SECONDS,
    ) -> dict[str, Any]:
        """POST one image to /v1/inference and return the parsed JSON response."""
        payload: dict[str, Any] = {}
        if image_path is not None:
            payload["image_path"] = image_path
        elif image_base64 is not None:
            payload["image_base64"] = image_base64
        else:
            raise ValueError("Provide image_path or image_base64.")

        body = json.dumps(payload).encode()
        req = Request(
            f"{service_url}{INFERENCE_ENDPOINT}",
            data=body,
            headers={"Content-Type": "application/json"},
        )
        with urlopen(req, timeout=timeout_seconds) as resp:
            return json.loads(resp.read().decode())

    @classmethod
    def _check_smoke_inference(
        cls, service_url: str, timeout_seconds: float = DEFAULT_REQUEST_TIMEOUT_SECONDS
    ) -> CheckResult:
        """Send the smoke PNG and verify the response contains an 'embedding' key."""
        try:
            response = cls._send_inference(service_url, image_base64=_SMOKE_PNG_B64, timeout_seconds=timeout_seconds)
            if "embedding" not in response:
                return CheckResult(
                    name="smoke_inference",
                    result_type=CheckResultType.PASS_FAIL,
                    success=False,
                    value=False,
                    detail=f"Response missing 'embedding' key: {list(response.keys())}",
                )
            embedding = response["embedding"]
            if not isinstance(embedding, list) or len(embedding) == 0:
                return CheckResult(
                    name="smoke_inference",
                    result_type=CheckResultType.PASS_FAIL,
                    success=False,
                    value=False,
                    detail=f"'embedding' is not a non-empty list: {type(embedding).__name__}",
                )
            return CheckResult(
                name="smoke_inference",
                result_type=CheckResultType.PASS_FAIL,
                success=True,
                value=True,
                detail=f"Returned embedding with {len(embedding)} elements",
            )
        except Exception as exc:  # noqa: BLE001 — per-check boundary
            return CheckResult(
                name="smoke_inference",
                result_type=CheckResultType.PASS_FAIL,
                success=False,
                value=False,
                detail=str(exc),
            )

    @classmethod
    def _check_embedding_shape(
        cls, service_url: str, timeout_seconds: float = DEFAULT_REQUEST_TIMEOUT_SECONDS
    ) -> CheckResult:
        """Verify embedding dimension is exactly 1536 and all values are finite."""
        import math

        try:
            response = cls._send_inference(service_url, image_base64=_SMOKE_PNG_B64, timeout_seconds=timeout_seconds)
            embedding = response.get("embedding", [])
            got_dim = len(embedding)

            if got_dim != EMBEDDING_DIM:
                return CheckResult(
                    name="embedding_shape",
                    result_type=CheckResultType.PASS_FAIL,
                    success=False,
                    value=False,
                    detail=f"Expected {EMBEDDING_DIM}-dim embedding, got {got_dim}",
                )

            non_finite = [i for i, v in enumerate(embedding) if not math.isfinite(v)]
            if non_finite:
                return CheckResult(
                    name="embedding_shape",
                    result_type=CheckResultType.PASS_FAIL,
                    success=False,
                    value=False,
                    detail=(
                        f"Embedding contains {len(non_finite)} non-finite value(s) " f"at indices {non_finite[:5]}"
                    ),
                )

            return CheckResult(
                name="embedding_shape",
                result_type=CheckResultType.PASS_FAIL,
                success=True,
                value=True,
                detail=f"{got_dim}-dim embedding, all values finite",
            )
        except Exception as exc:  # noqa: BLE001 — per-check boundary
            return CheckResult(
                name="embedding_shape",
                result_type=CheckResultType.PASS_FAIL,
                success=False,
                value=False,
                detail=str(exc),
            )
