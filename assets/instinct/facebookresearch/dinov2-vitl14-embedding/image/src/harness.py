# Copyright © Advanced Micro Devices, Inc., or its affiliates.
#
# SPDX-License-Identifier: MIT

"""DINOv2 ViT-L/14 AIM harness."""

from __future__ import annotations

import json
import math
import os
import statistics
import time
from pathlib import Path
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from inference import Dinov2Predictor

from aim_runtime.harness import (
    CheckInfo,
    CheckResult,
    CheckResultType,
    CheckScope,
    HarnessConfig,
    HarnessResult,
    ModelHarness,
)

DEFAULT_BENCHMARK_WARMUP = 5
DEFAULT_BENCHMARK_REQUESTS = 100
GOLDEN_EMBEDDINGS_PATH = Path(__file__).with_name("dinov2_golden_embeddings.json")


def _cosine_similarity(left: list[float], right: list[float]) -> float:
    dot = sum(a * b for a, b in zip(left, right))
    left_norm = math.sqrt(sum(value * value for value in left))
    right_norm = math.sqrt(sum(value * value for value in right))
    if left_norm == 0.0 or right_norm == 0.0:
        return 0.0
    return dot / (left_norm * right_norm)


def _percentile(values: list[float], percentile: float) -> float:
    ordered = sorted(values)
    index = round((percentile / 100.0) * (len(ordered) - 1))
    return ordered[index]


def _timing_metrics(prefix: str, values: list[float]) -> dict[str, float]:
    return {
        f"{prefix}_mean_ms": round(statistics.fmean(values), 3),
        f"{prefix}_stdev_ms": round(statistics.stdev(values), 3) if len(values) > 1 else 0.0,
        f"{prefix}_p50_ms": round(_percentile(values, 50), 3),
        f"{prefix}_p95_ms": round(_percentile(values, 95), 3),
        f"{prefix}_p99_ms": round(_percentile(values, 99), 3),
        f"{prefix}_min_ms": round(min(values), 3),
        f"{prefix}_max_ms": round(max(values), 3),
    }


class Dinov2Vitl14Harness(Dinov2Predictor, ModelHarness):
    """Harness for DINOv2 ViT-L/14 image embedding generation."""

    ENGINE = "bentoml"

    CHECKS = [
        CheckInfo("bentoml_health", CheckResultType.PASS_FAIL, CheckScope.OFFLINE, "BentoML /healthz returns 200."),
        CheckInfo(
            "latency_p50_ms",
            CheckResultType.SCORE,
            CheckScope.OFFLINE,
            "p50 HTTP end-to-end latency for batch size 1.",
        ),
        CheckInfo(
            "gpu_inference_p50_ms",
            CheckResultType.SCORE,
            CheckScope.OFFLINE,
            "p50 model GPU inference time for batch size 1.",
        ),
        CheckInfo(
            "preprocessing_p50_ms",
            CheckResultType.SCORE,
            CheckScope.OFFLINE,
            "p50 image preprocessing time for batch size 1.",
        ),
        CheckInfo(
            "benchmark_integrity",
            CheckResultType.PASS_FAIL,
            CheckScope.OFFLINE,
            "All measured benchmark requests returned valid embeddings and timings.",
        ),
        CheckInfo(
            "golden_embedding_cosine",
            CheckResultType.SCORE,
            CheckScope.OFFLINE,
            "Mean cosine similarity to five reviewed golden embeddings.",
        ),
        CheckInfo(
            "golden_embedding_regression",
            CheckResultType.PASS_FAIL,
            CheckScope.OFFLINE,
            "All five embeddings meet the reviewed cosine-similarity threshold.",
        ),
    ]

    def __init__(self) -> None:
        Dinov2Predictor.__init__(self)

    def list_checks(self) -> list[CheckInfo]:
        return list(self.CHECKS)

    def health_check(self, service_url: str, timeout_seconds: int = 60) -> bool:
        deadline = time.monotonic() + timeout_seconds
        while time.monotonic() < deadline:
            try:
                with urlopen(Request(f"{service_url.rstrip('/')}/healthz"), timeout=5) as resp:
                    if resp.status == 200:
                        return True
            except (HTTPError, URLError, OSError):
                pass
            time.sleep(2.0)
        return False

    def validate(self, config: HarnessConfig) -> HarnessResult:
        if CheckScope.OFFLINE not in config.check_scopes:
            return HarnessResult(True, "DINOv2 validate: no runtime-scope checks", [])
        service_url = config.resolve_service_url().rstrip("/")
        healthy = self.health_check(service_url, min(config.timeout_seconds, 120))
        checks = [
            CheckResult("bentoml_health", CheckResultType.PASS_FAIL, healthy, healthy, f"GET {service_url}/healthz")
        ]
        return HarnessResult(healthy, "DINOv2 validate passed" if healthy else "DINOv2 validate failed", checks)

    def benchmark(self, config: HarnessConfig) -> HarnessResult:
        service_url = config.resolve_service_url().rstrip("/")
        checks: list[CheckResult] = []
        healthy = self.health_check(service_url, min(config.timeout_seconds, 120))
        checks.append(
            CheckResult(
                "bentoml_health",
                CheckResultType.PASS_FAIL,
                healthy,
                healthy,
                f"GET {service_url}/healthz",
            )
        )
        if not healthy:
            return HarnessResult(False, "DINOv2 benchmark: service did not become healthy", checks)

        requests = int(
            config.get("num_requests", os.environ.get("AIM_HARNESS_NUM_REQUESTS", DEFAULT_BENCHMARK_REQUESTS))
        )
        warmup_requests = int(
            config.get("num_warmup", os.environ.get("AIM_HARNESS_NUM_WARMUP", DEFAULT_BENCHMARK_WARMUP))
        )
        request_timeout = int(config.get("request_timeout_s", config.timeout_seconds))
        configured_batch_sizes = config.get("batch_sizes")
        if configured_batch_sizes is None:
            legacy_batch_size = config.get("batch_size", os.environ.get("DINOV2_BENCHMARK_BATCH_SIZE"))
            batch_sizes = [int(legacy_batch_size)] if legacy_batch_size is not None else [1]
        elif isinstance(configured_batch_sizes, str):
            batch_sizes = [int(value.strip()) for value in configured_batch_sizes.split(",") if value.strip()]
        elif isinstance(configured_batch_sizes, int):
            batch_sizes = [configured_batch_sizes]
        else:
            batch_sizes = [int(value) for value in configured_batch_sizes]
        if (
            requests < 1
            or warmup_requests < 0
            or request_timeout < 1
            or not batch_sizes
            or len(set(batch_sizes)) != len(batch_sizes)
            or any(batch_size != 1 for batch_size in batch_sizes)
        ):
            checks.append(
                CheckResult(
                    "benchmark_integrity",
                    CheckResultType.PASS_FAIL,
                    False,
                    False,
                    "num_requests and request_timeout_s must be positive, num_warmup must be non-negative, "
                    "and batch_sizes must contain only batch size 1",
                )
            )
            return HarnessResult(False, "DINOv2 benchmark configuration is invalid", checks)

        fixtures = list(self._golden_fixture_payloads().items())
        metrics: dict[str, Any] = {
            "batch_sizes": ",".join(str(batch_size) for batch_size in batch_sizes),
            "requests_per_batch_size": requests,
            "warmup_requests_per_batch_size": warmup_requests,
            "fixture_count": len(fixtures),
        }
        failures: list[str] = []
        summaries: list[str] = []
        total_succeeded = 0

        for batch_size in batch_sizes:
            for index in range(warmup_requests):
                payload = self._benchmark_payload(fixtures, index)
                try:
                    response = self._post_predict(
                        service_url,
                        payload,
                        timeout_seconds=request_timeout,
                    )
                    self._benchmark_response_timings(response)
                except Exception as exc:
                    checks.append(
                        CheckResult(
                            "benchmark_integrity",
                            CheckResultType.PASS_FAIL,
                            False,
                            False,
                            f"batch_size={batch_size} warmup request {index + 1} failed: {exc}",
                        )
                    )
                    return HarnessResult(False, "DINOv2 benchmark warmup failed", checks, metrics)

            http_latencies_ms: list[float] = []
            preprocessing_latencies_ms: list[float] = []
            gpu_inference_latencies_ms: list[float] = []
            batch_failures: list[str] = []
            benchmark_start = time.perf_counter()
            for index in range(requests):
                payload = self._benchmark_payload(fixtures, index)
                request_start = time.perf_counter()
                try:
                    response = self._post_predict(
                        service_url,
                        payload,
                        timeout_seconds=request_timeout,
                    )
                    http_latency_ms = (time.perf_counter() - request_start) * 1000.0
                    preprocessing_ms, gpu_inference_ms = self._benchmark_response_timings(response)
                    http_latencies_ms.append(http_latency_ms)
                    preprocessing_latencies_ms.append(preprocessing_ms)
                    gpu_inference_latencies_ms.append(gpu_inference_ms)
                except Exception as exc:
                    batch_failures.append(f"batch_size={batch_size} request {index + 1}: {exc}")
            benchmark_duration_s = time.perf_counter() - benchmark_start

            succeeded = len(http_latencies_ms)
            total_succeeded += succeeded
            failures.extend(batch_failures)
            if not succeeded:
                continue

            http_p50 = _percentile(http_latencies_ms, 50)
            preprocessing_p50 = _percentile(preprocessing_latencies_ms, 50)
            gpu_inference_p50 = _percentile(gpu_inference_latencies_ms, 50)
            latency_check, gpu_check, preprocessing_check = self._benchmark_check_names(batch_size)
            checks.extend(
                [
                    CheckResult(
                        latency_check,
                        CheckResultType.SCORE,
                        True,
                        round(http_p50, 3),
                        f"batch_size={batch_size}, {succeeded}/{requests} requests, HTTP p50={http_p50:.3f} ms",
                    ),
                    CheckResult(
                        gpu_check,
                        CheckResultType.SCORE,
                        True,
                        round(gpu_inference_p50, 3),
                        f"batch_size={batch_size}, {succeeded}/{requests} requests, "
                        f"GPU inference p50={gpu_inference_p50:.3f} ms",
                    ),
                    CheckResult(
                        preprocessing_check,
                        CheckResultType.SCORE,
                        True,
                        round(preprocessing_p50, 3),
                        f"batch_size={batch_size}, {succeeded}/{requests} requests, "
                        f"preprocessing p50={preprocessing_p50:.3f} ms",
                    ),
                ]
            )
            prefix = f"batch_size_{batch_size}"
            throughput = (succeeded * batch_size / benchmark_duration_s) if benchmark_duration_s > 0.0 else 0.0
            metrics.update(
                {
                    **_timing_metrics(f"{prefix}_http_e2e", http_latencies_ms),
                    **_timing_metrics(f"{prefix}_preprocessing", preprocessing_latencies_ms),
                    **_timing_metrics(f"{prefix}_gpu_inference", gpu_inference_latencies_ms),
                    f"{prefix}_throughput_img_per_sec": round(throughput, 3),
                    f"{prefix}_benchmark_duration_s": round(benchmark_duration_s, 3),
                    f"{prefix}_requests_succeeded": succeeded,
                    f"{prefix}_requests_total": requests,
                    f"{prefix}_failure_count": len(batch_failures),
                }
            )
            summaries.append(
                f"batch_size={batch_size}: HTTP p50={http_p50:.3f} ms, "
                f"GPU p50={gpu_inference_p50:.3f} ms, preprocessing p50={preprocessing_p50:.3f} ms, "
                f"throughput={throughput:.3f} img/s"
            )

        requests_total = requests * len(batch_sizes)
        integrity_ok = total_succeeded == requests_total and not failures
        checks.append(
            CheckResult(
                "benchmark_integrity",
                CheckResultType.PASS_FAIL,
                integrity_ok,
                integrity_ok,
                f"{total_succeeded}/{requests_total} measured requests succeeded"
                + (f"; {'; '.join(failures[:3])}" if failures else ""),
            )
        )
        metrics.update(
            {
                "requests_succeeded": total_succeeded,
                "requests_total": requests_total,
                "failure_count": len(failures),
            }
        )
        if not total_succeeded:
            return HarnessResult(False, "DINOv2 benchmark: all measured requests failed", checks, metrics)
        return HarnessResult(
            integrity_ok,
            "DINOv2 benchmark: " + "; ".join(summaries),
            checks,
            metrics,
        )

    def evaluate(self, config: HarnessConfig) -> HarnessResult:
        service_url = config.resolve_service_url().rstrip("/")
        checks: list[CheckResult] = []
        healthy = self.health_check(service_url, min(config.timeout_seconds, 120))
        checks.append(
            CheckResult(
                "bentoml_health",
                CheckResultType.PASS_FAIL,
                healthy,
                healthy,
                f"GET {service_url}/healthz",
            )
        )
        if not healthy:
            return HarnessResult(False, "DINOv2 evaluation: service did not become healthy", checks)

        try:
            payloads = self._golden_fixture_payloads()
            manifest_path = Path(config.get("golden_embeddings_path", str(GOLDEN_EMBEDDINGS_PATH)))
            threshold, golden_embeddings = self._load_golden_embeddings(manifest_path)
            if len(payloads) != 5 or set(payloads) != set(golden_embeddings):
                raise ValueError(
                    "golden manifest must contain exactly the five fixtures: " f"{', '.join(sorted(payloads))}"
                )

            request_timeout = int(config.get("request_timeout_s", config.timeout_seconds))
            similarities = {}
            for name, image_base64 in payloads.items():
                response = self._post_predict(
                    service_url,
                    {"image_base64": image_base64},
                    timeout_seconds=request_timeout,
                )
                embedding = response.get("embedding", [])
                if len(embedding) != self.embed_dim or not all(math.isfinite(float(value)) for value in embedding):
                    raise ValueError(f"fixture {name!r} returned an invalid embedding")
                similarities[name] = _cosine_similarity(
                    [float(value) for value in embedding],
                    golden_embeddings[name],
                )
        except Exception as exc:
            checks.append(
                CheckResult(
                    "golden_embedding_regression",
                    CheckResultType.PASS_FAIL,
                    False,
                    False,
                    str(exc),
                )
            )
            return HarnessResult(False, "DINOv2 golden embedding evaluation failed", checks)

        mean_cosine = statistics.fmean(similarities.values())
        worst_fixture = min(similarities, key=lambda name: similarities[name])
        min_cosine = similarities[worst_fixture]
        passed = sum(similarity >= threshold for similarity in similarities.values())
        regression_ok = passed == len(similarities)
        checks.append(
            CheckResult(
                "golden_embedding_cosine",
                CheckResultType.SCORE,
                True,
                mean_cosine,
                f"mean={mean_cosine:.6f} min={min_cosine:.6f} fixture={worst_fixture}",
            )
        )
        checks.append(
            CheckResult(
                "golden_embedding_regression",
                CheckResultType.PASS_FAIL,
                regression_ok,
                regression_ok,
                f"{passed}/{len(similarities)} fixtures met cosine threshold {threshold:.6f}",
            )
        )
        return HarnessResult(
            regression_ok,
            f"DINOv2 golden embeddings: {passed}/{len(similarities)} passed " f"(minimum cosine={min_cosine:.6f})",
            checks,
            {
                "mean_cosine_similarity": round(mean_cosine, 6),
                "min_cosine_similarity": round(min_cosine, 6),
                "cosine_threshold": threshold,
                "fixtures_passed": passed,
                "fixtures_total": len(similarities),
            },
        )

    @staticmethod
    def _golden_fixture_payloads() -> dict[str, str]:
        from golden_fixtures import fixture_payloads

        return fixture_payloads()

    def _load_golden_embeddings(self, path: Path) -> tuple[float, dict[str, list[float]]]:
        if not path.is_file():
            raise FileNotFoundError(
                f"golden embedding manifest not found at {path}; run "
                "capture_golden_embeddings.py against an approved MI300X AIM"
            )
        manifest = json.loads(path.read_text())
        if manifest.get("model") != self.model_name:
            raise ValueError(f"golden manifest model is {manifest.get('model')!r}, expected {self.model_name!r}")
        if manifest.get("precision") != self.precision:
            raise ValueError(f"golden manifest precision is {manifest.get('precision')!r}, expected {self.precision!r}")
        threshold = float(manifest["cosine_threshold"])
        if not 0.0 < threshold <= 1.0:
            raise ValueError("golden cosine threshold must be in the interval (0, 1]")
        fixtures = manifest.get("fixtures")
        if not isinstance(fixtures, dict):
            raise ValueError("golden manifest fixtures must be an object")

        embeddings = {}
        for name, embedding in fixtures.items():
            if not isinstance(embedding, list) or len(embedding) != self.embed_dim:
                raise ValueError(f"golden fixture {name!r} must contain {self.embed_dim} values")
            values = [float(value) for value in embedding]
            if not all(math.isfinite(value) for value in values):
                raise ValueError(f"golden fixture {name!r} contains non-finite values")
            embeddings[str(name)] = values
        return threshold, embeddings

    @staticmethod
    def _benchmark_payload(fixtures: list[tuple[str, str]], request_index: int) -> dict[str, Any]:
        return {
            "image_base64": fixtures[request_index % len(fixtures)][1],
            "include_timings": True,
        }

    @staticmethod
    def _benchmark_check_names(batch_size: int) -> tuple[str, str, str]:
        if batch_size != 1:
            raise ValueError(f"unsupported benchmark batch size: {batch_size}")
        return "latency_p50_ms", "gpu_inference_p50_ms", "preprocessing_p50_ms"

    def _benchmark_response_timings(self, response: dict[str, Any]) -> tuple[float, float]:
        embedding = response.get("embedding")
        if not isinstance(embedding, list) or len(embedding) != self.embed_dim:
            raise ValueError(f"response does not contain an embedding of length {self.embed_dim}")
        if any(not math.isfinite(float(value)) for value in embedding):
            raise ValueError("the embedding contains non-finite values")
        timings = response.get("timings")
        if not isinstance(timings, dict):
            raise ValueError("response does not contain benchmark timings")
        preprocessing_ms = float(timings["preprocessing_ms"])
        gpu_inference_ms = float(timings["gpu_inference_ms"])
        if (
            not math.isfinite(preprocessing_ms)
            or preprocessing_ms < 0.0
            or not math.isfinite(gpu_inference_ms)
            or gpu_inference_ms < 0.0
        ):
            raise ValueError("response contains invalid benchmark timings")
        return preprocessing_ms, gpu_inference_ms

    def _post_predict(
        self,
        service_url: str,
        payload: dict[str, Any],
        timeout_seconds: int,
    ) -> dict[str, Any]:
        body = json.dumps({"request": payload}).encode("utf-8")
        req = Request(f"{service_url}/predict", data=body, headers={"Content-Type": "application/json"}, method="POST")
        with urlopen(req, timeout=timeout_seconds) as resp:
            return json.loads(resp.read().decode("utf-8"))
