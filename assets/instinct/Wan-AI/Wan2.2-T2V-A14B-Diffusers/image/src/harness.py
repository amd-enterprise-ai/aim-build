#!/usr/bin/env python3

# Copyright © Advanced Micro Devices, Inc., or its affiliates.
#
# SPDX-License-Identifier: MIT

"""
VllmOmniDiffusionHarness — ModelHarness implementation for vLLM-Omni
diffusion (text-to-video / text-to-image) models, written generically so
the same file can ship in any vllm_omni AIM image.

It currently lives under ``assets/instinct/Wan-AI/Wan2.2-T2V-A14B-Diffusers/
image/src/harness.py`` because wan2.2 is the first vllm_omni model on this
branch.  Because it has no wan2.2-specific paths (only the *default* recipes
in ``recipes.py`` do), it can later be lifted into a shared
``assets/instinct/base/vllm-omni/image/src/`` without changes.  The omni base
image (``aim-instinct-vllm-omni-base``) is built from the shared
``docker/Dockerfile.aim-instinct-base`` with the vllm-omni base assets injected
via build args (see ``.github/actions/build-image``).

Discovery: ``docker/Dockerfile.aim`` copies the contents of this directory
into ``/workspace/model/src/`` at build time, and the PR-959 entrypoint's
``aim_runtime.harness.discovery.discover_harness()`` picks it up at runtime.

The benchmark itself is delegated to the upstream
``benchmarks/diffusion/diffusion_benchmark_serving.py`` script that ships
inside ``vllm/vllm-omni-rocm`` at ``/app/vllm-omni/`` — we invoke it via
``subprocess.run`` and parse its ``--output-file`` JSON.
"""

from __future__ import annotations

import json
import logging
import os
import subprocess
import sys
import time
from pathlib import Path
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
)

try:
    from recipes import DEFAULT_RECIPE, get_recipe
except ImportError:  # pragma: no cover - keeps the harness importable from tests
    from .recipes import DEFAULT_RECIPE, get_recipe  # type: ignore[no-redef]

logger = logging.getLogger(__name__)

# Location of the upstream diffusion benchmark script inside the
# vllm/vllm-omni-rocm container layer.  Pinned by
# vllm-omni/docker/Dockerfile.rocm (``COPY . /app/vllm-omni``).
DIFFUSION_BENCH_DIR = Path("/app/vllm-omni/benchmarks/diffusion")
DIFFUSION_BENCH_SCRIPT = "diffusion_benchmark_serving.py"

# Default output path for the benchmark's JSON metrics file.  Caller may
# override via ``--config`` (``output_file: /tmp/my_run.json``).
DEFAULT_BENCH_OUTPUT_FILE = "/tmp/diffusion_bench.json"

# Timeout (seconds) for the POST /v1/videos submit call. Some servers generate
# the clip synchronously on submit rather than returning a job id immediately,
# so this must be generous enough for a short video generation, not just an
# enqueue round-trip. Overridable via config (``smoke_submit_timeout``).
VIDEO_SUBMIT_TIMEOUT_SECONDS = 300

# Smoke /v1/videos parameters — kept deliberately small so the request goes
# through the full inference pipeline without paying for a full-quality
# generation.  ``num_inference_steps`` matters most for wall-clock here.
SMOKE_VIDEO_PARAMS: dict[str, Any] = {
    "prompt": "a smoke-test video for the AIM harness",
    "size": "512x288",
    "seconds": 1,
    "fps": 8,
    "num_inference_steps": 3,
    "seed": 0,
}


class VllmOmniDiffusionHarness(ModelHarness):
    """Harness for vLLM-Omni diffusion models exposing ``/v1/videos``.

    Designed to be engine-generic.  All model-specific knobs (model id,
    task type, port, recipe name) come from the resolved profile dict or
    from ``HarnessConfig.extra`` (populated by ``aim-runtime --config``).
    """

    ENGINE = "vllm_omni"

    # ------------------------------------------------------------------ #
    # 1. Declare available checks
    # ------------------------------------------------------------------ #

    CHECKS: list[CheckInfo] = [
        CheckInfo(
            "health",
            CheckResultType.PASS_FAIL,
            CheckScope.RUNTIME,
            "vLLM-Omni /v1/models returns 200",
        ),
        CheckInfo(
            "smoke_video",
            CheckResultType.PASS_FAIL,
            CheckScope.RUNTIME,
            "POST /v1/videos with a minimal payload completes and yields non-empty content",
        ),
        CheckInfo(
            "throughput_qps",
            CheckResultType.SCORE,
            CheckScope.OFFLINE,
            "Requests per second from diffusion_benchmark_serving.py",
        ),
        CheckInfo(
            "benchmark_completion",
            CheckResultType.PASS_FAIL,
            CheckScope.OFFLINE,
            "All benchmark requests completed successfully",
        ),
        CheckInfo(
            "latency_p99_under_slo",
            CheckResultType.PASS_FAIL,
            CheckScope.OFFLINE,
            "p99 latency stays under config.slo_seconds (only checked when slo_seconds is set)",
        ),
    ]

    def list_checks(self) -> list[CheckInfo]:
        return list(self.CHECKS)

    # ------------------------------------------------------------------ #
    # 2. validate — "is it alive?"
    # ------------------------------------------------------------------ #

    def validate(self, config: HarnessConfig) -> HarnessResult:
        service_url = config.resolve_service_url()
        checks: list[CheckResult] = []

        healthy = self.health_check(service_url, timeout_seconds=config.timeout_seconds)
        checks.append(
            CheckResult(
                name="health",
                result_type=CheckResultType.PASS_FAIL,
                success=healthy,
                value=healthy,
                detail="" if healthy else f"vLLM-Omni not reachable at {service_url}",
            )
        )
        if not healthy:
            return HarnessResult(success=False, summary="vLLM-Omni health check failed", checks=checks)

        if CheckScope.RUNTIME in config.check_scopes:
            checks.append(self._check_smoke_video(service_url, config))

        success = all(c.success for c in checks)
        return HarnessResult(
            success=success,
            summary=("vLLM-Omni validation: " f"{sum(c.success for c in checks)}/{len(checks)} checks passed"),
            checks=checks,
        )

    # ------------------------------------------------------------------ #
    # 3. benchmark — "how fast is it?"
    # ------------------------------------------------------------------ #

    def benchmark(self, config: HarnessConfig) -> HarnessResult:
        service_url = config.resolve_service_url()

        if not self.health_check(service_url, timeout_seconds=config.timeout_seconds):
            return HarnessResult(
                success=False,
                summary=f"vLLM-Omni service not reachable at {service_url}",
            )

        recipe_name = config.get("recipe", DEFAULT_RECIPE)
        try:
            recipe = get_recipe(recipe_name)
        except KeyError as exc:
            return HarnessResult(success=False, summary=str(exc))

        task = config.get("task", "t2v")
        model_id = config.profile.get("model_id") or config.get("model", "default")
        output_file = config.get("output_file", DEFAULT_BENCH_OUTPUT_FILE)

        argv = self._build_benchmark_argv(
            service_url=service_url,
            model_id=model_id,
            task=task,
            recipe=recipe,
            output_file=output_file,
        )

        logger.info("Running diffusion benchmark: recipe=%s task=%s", recipe_name, task)
        logger.debug("argv=%s cwd=%s", argv, DIFFUSION_BENCH_DIR)

        try:
            completed = subprocess.run(
                argv,
                cwd=str(DIFFUSION_BENCH_DIR),
                check=False,
                timeout=config.timeout_seconds,
                capture_output=True,
                text=True,
            )
        except subprocess.TimeoutExpired:
            return HarnessResult(
                success=False,
                summary=(
                    f"diffusion_benchmark_serving.py timed out after "
                    f"{config.timeout_seconds}s (recipe={recipe_name})"
                ),
                metrics={"recipe": recipe_name, "timeout_seconds": config.timeout_seconds},
                artifacts=[output_file] if Path(output_file).is_file() else [],
            )
        except OSError as exc:
            # Raised when the benchmark dir/script is absent (e.g. the vllm-omni
            # image layout changed, or this harness runs on a non-omni base):
            # subprocess.run raises FileNotFoundError/NotADirectoryError. Fail
            # gracefully with a clear summary instead of crashing aim-runtime.
            return HarnessResult(
                success=False,
                summary=(
                    f"failed to launch {DIFFUSION_BENCH_SCRIPT} from {DIFFUSION_BENCH_DIR} "
                    f"(recipe={recipe_name}): {exc}"
                ),
                metrics={"recipe": recipe_name},
            )

        if completed.returncode != 0:
            tail = (completed.stderr or completed.stdout or "").strip()[-400:]
            summary = f"diffusion_benchmark_serving.py exited with " f"{completed.returncode} (recipe={recipe_name})"
            if tail:
                summary = f"{summary}: {tail}"
            return HarnessResult(
                success=False,
                summary=summary,
                metrics={"recipe": recipe_name, "returncode": completed.returncode},
                artifacts=[output_file] if Path(output_file).is_file() else [],
            )

        bench = self._load_benchmark_json(output_file)
        if bench is None:
            return HarnessResult(
                success=False,
                summary=f"diffusion_benchmark_serving.py wrote no metrics file at {output_file}",
                metrics={"recipe": recipe_name},
            )

        return self._build_benchmark_result(
            recipe_name=recipe_name,
            recipe=recipe,
            task=task,
            output_file=output_file,
            bench=bench,
            slo_seconds=config.get("slo_seconds"),
        )

    # ------------------------------------------------------------------ #
    # 4. evaluate — stub, not implemented for the diffusion harness
    # ------------------------------------------------------------------ #

    def evaluate(self, config: HarnessConfig) -> HarnessResult:
        # Quality evaluation for diffusion video models requires an external
        # reference dataset (e.g. VBench).  Deferred — return a single
        # informational PASS so CI doesn't block on this.
        return HarnessResult(
            success=True,
            summary="evaluation not implemented for vllm_omni diffusion harness",
            checks=[
                CheckResult(
                    name="evaluate_stub",
                    result_type=CheckResultType.PASS_FAIL,
                    success=True,
                    value=True,
                    detail="VBench/accuracy evaluation is out of scope for this harness",
                )
            ],
        )

    # ------------------------------------------------------------------ #
    # Helpers — benchmark
    # ------------------------------------------------------------------ #

    def _build_benchmark_argv(
        self,
        *,
        service_url: str,
        model_id: str,
        task: str,
        recipe: dict[str, Any],
        output_file: str,
    ) -> list[str]:
        """Construct the diffusion_benchmark_serving.py argv list."""
        return [
            sys.executable,
            DIFFUSION_BENCH_SCRIPT,
            "--backend",
            "v1/videos",
            "--base-url",
            service_url,
            "--model",
            model_id,
            "--task",
            task,
            "--dataset",
            "random",
            "--num-prompts",
            str(recipe["num_prompts"]),
            "--max-concurrency",
            str(recipe["max_concurrency"]),
            "--enable-negative-prompt",
            "--random-request-config",
            json.dumps(recipe["random_request_config"]),
            "--output-file",
            output_file,
        ]

    @staticmethod
    def _load_benchmark_json(path: str) -> dict[str, Any] | None:
        try:
            with open(path) as fh:
                return json.load(fh)
        except (FileNotFoundError, json.JSONDecodeError):
            return None

    def _build_benchmark_result(
        self,
        *,
        recipe_name: str,
        recipe: dict[str, Any],
        task: str,
        output_file: str,
        bench: dict[str, Any],
        slo_seconds: float | None,
    ) -> HarnessResult:
        # Keys that diffusion_benchmark_serving.py emits via calculate_metrics().
        # See vllm-omni/benchmarks/diffusion/diffusion_benchmark_serving.py:919-925.
        metric_keys = (
            "duration",
            "completed_requests",
            "throughput_qps",
            "latency_mean",
            "latency_median",
            "latency_p99",
            "latency_p95",
        )
        metrics = {key: bench[key] for key in metric_keys if key in bench}
        metrics["recipe"] = recipe_name
        metrics["task"] = task
        metrics["num_prompts"] = recipe["num_prompts"]
        metrics["max_concurrency"] = recipe["max_concurrency"]
        metrics["output_file"] = output_file

        total_requests = recipe["num_prompts"]
        completed = int(bench.get("completed_requests", 0))
        all_completed = completed == total_requests

        checks: list[CheckResult] = [
            CheckResult(
                name="throughput_qps",
                result_type=CheckResultType.SCORE,
                success=completed > 0,
                value=round(float(bench.get("throughput_qps", 0.0)), 4),
                detail=(
                    f"{completed}/{total_requests} requests, "
                    f"mean={bench.get('latency_mean', 0.0):.2f}s, "
                    f"p99={bench.get('latency_p99', 0.0):.2f}s"
                ),
            ),
            CheckResult(
                name="benchmark_completion",
                result_type=CheckResultType.PASS_FAIL,
                success=all_completed,
                value=all_completed,
                detail=f"{completed}/{total_requests} requests completed",
            ),
        ]

        if slo_seconds is not None:
            p99 = float(bench.get("latency_p99", float("inf")))
            slo_ok = p99 <= float(slo_seconds)
            checks.append(
                CheckResult(
                    name="latency_p99_under_slo",
                    result_type=CheckResultType.PASS_FAIL,
                    success=slo_ok,
                    value=slo_ok,
                    detail=f"p99={p99:.2f}s vs slo={float(slo_seconds):.2f}s",
                )
            )
            metrics["slo_seconds"] = float(slo_seconds)

        success = all(c.success for c in checks)
        return HarnessResult(
            success=success,
            summary=(
                f"vLLM-Omni diffusion benchmark (recipe={recipe_name}): "
                f"{bench.get('throughput_qps', 0.0):.3f} req/s, "
                f"mean={bench.get('latency_mean', 0.0):.2f}s, "
                f"p99={bench.get('latency_p99', 0.0):.2f}s"
            ),
            checks=checks,
            metrics=metrics,
            artifacts=[output_file] if Path(output_file).is_file() else [],
        )

    # ------------------------------------------------------------------ #
    # Helpers — validate / smoke
    # ------------------------------------------------------------------ #

    @classmethod
    def _check_smoke_video(cls, service_url: str, config: HarnessConfig) -> CheckResult:
        """Submit a minimal /v1/videos job and wait for completion.

        Uses urllib + a hand-rolled multipart body so the harness has no
        third-party HTTP dependency (matches the boltz2 harness's pattern).
        """
        params = dict(SMOKE_VIDEO_PARAMS)
        if "smoke_prompt" in config.extra:
            params["prompt"] = str(config.extra["smoke_prompt"])
        poll_timeout = float(config.get("smoke_poll_timeout", config.timeout_seconds))
        submit_timeout = float(config.get("smoke_submit_timeout", VIDEO_SUBMIT_TIMEOUT_SECONDS))

        try:
            job_id = cls._post_video_multipart(service_url, params, timeout=submit_timeout)
            if not job_id:
                return CheckResult(
                    "smoke_video",
                    CheckResultType.PASS_FAIL,
                    False,
                    False,
                    "POST /v1/videos returned no job id",
                )
            status, detail = cls._poll_video_status(service_url, job_id, poll_timeout)
            if status != "completed":
                return CheckResult(
                    "smoke_video",
                    CheckResultType.PASS_FAIL,
                    False,
                    False,
                    f"job {job_id} ended with status={status} ({detail})",
                )

            size = cls._head_video_content_size(service_url, job_id)
            return CheckResult(
                "smoke_video",
                CheckResultType.PASS_FAIL,
                size > 0,
                size > 0,
                f"job {job_id} completed, content size={size} bytes",
            )
        except Exception as exc:  # noqa: BLE001 - want a descriptive failure
            return CheckResult(
                "smoke_video",
                CheckResultType.PASS_FAIL,
                False,
                False,
                f"{type(exc).__name__}: {exc}",
            )

    @staticmethod
    def _post_video_multipart(
        service_url: str, fields: dict[str, Any], timeout: float = VIDEO_SUBMIT_TIMEOUT_SECONDS
    ) -> str | None:
        """POST /v1/videos with multipart/form-data; return the job id."""
        boundary = "----aim-harness-" + os.urandom(8).hex()
        body_parts: list[bytes] = []
        for key, value in fields.items():
            body_parts.append(f"--{boundary}\r\n".encode())
            body_parts.append(f'Content-Disposition: form-data; name="{key}"\r\n\r\n'.encode())
            body_parts.append(f"{value}\r\n".encode())
        body_parts.append(f"--{boundary}--\r\n".encode())
        body = b"".join(body_parts)

        req = Request(
            f"{service_url}/v1/videos",
            data=body,
            headers={
                "Content-Type": f"multipart/form-data; boundary={boundary}",
                "Accept": "application/json",
            },
            method="POST",
        )
        with urlopen(req, timeout=timeout) as resp:
            payload = json.loads(resp.read().decode())
        return payload.get("id")

    @staticmethod
    def _poll_video_status(service_url: str, job_id: str, timeout_seconds: float) -> tuple[str, str]:
        deadline = time.time() + timeout_seconds
        last_status = "unknown"
        last_detail = ""
        while time.time() < deadline:
            try:
                with urlopen(f"{service_url}/v1/videos/{job_id}", timeout=10) as resp:
                    payload = json.loads(resp.read().decode())
            except (HTTPError, URLError, OSError) as exc:
                last_detail = str(exc)
                time.sleep(2.0)
                continue
            last_status = str(payload.get("status", "unknown"))
            last_detail = json.dumps({k: payload.get(k) for k in ("status", "error")})
            if last_status in {"completed", "failed"}:
                return last_status, last_detail
            time.sleep(2.0)
        return last_status, f"poll timed out after {timeout_seconds:.0f}s; last={last_detail}"

    @staticmethod
    def _head_video_content_size(service_url: str, job_id: str) -> int:
        """Fetch the generated content and return its size in bytes.

        We download (rather than HEAD) because vLLM-Omni streams the file
        and may not return Content-Length on a HEAD.
        """
        with urlopen(f"{service_url}/v1/videos/{job_id}/content", timeout=60) as resp:
            data = resp.read()
        return len(data)
