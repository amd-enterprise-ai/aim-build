# Copyright © Advanced Micro Devices, Inc., or its affiliates.
#
# SPDX-License-Identifier: MIT

"""
REINVENT4Harness — model-specific harness for REINVENT4, AstraZeneca's
molecular drug design and de novo generation toolkit.

This file lives at
``assets/instinct/molecularai/reinvent4/image/src/harness.py`` and is
copied to ``/workspace/model/src/harness.py`` by the specialized
Dockerfile.  aim-runtime discovers it automatically via the well-known path.

REINVENT4 is served by a BentoML service (service.py) that wraps the
reinvent CLI's sampling run mode and exposes POST /v1/inference.  The harness
talks to that HTTP service — it never calls the CLI directly.

POST /v1/inference request schema (the whole body is nested under a top-level
``data`` object; all fields inside it are optional):
    {"data": {
      "num_smiles":       int   = 100,
      "unique_molecules": bool  = true
    }}

POST /v1/inference response schema:
    {
      "smiles":        list[str],
      "num_generated": int
    }

The service answers ``GET /health`` (forwarding to BentoML's ``/healthz``); ``health_check`` polls
``/health`` then ``/healthz``.
``validate`` at RUNTIME scope verifies the CLI is importable and executable
(local subprocess checks that do not use the service; they run only when RUNTIME is
requested); at
OFFLINE scope it first waits for the service health check (failing clearly
with ``service_health`` if it is not reachable, without any POST) and then
calls POST /v1/inference with num_smiles=1 to exercise the GPU path end-to-end.
``benchmark`` and ``evaluate`` only call the service when OFFLINE is among the
requested scopes; otherwise they return skipped results without any network
call.

Metric definitions
------------------
``throughput_smiles_per_min`` = total SMILES x 60 / total wall time, a ratio of
sums over the 5 timed requests of ``benchmark_n_smiles`` (default 512) each,
after 2 discarded warm-up requests. The wall time includes the HTTP round trip,
the sampling on the resident model and the RDKit validation of the sampled
SMILES, so it is end-to-end service throughput, not raw sampler speed. Any failed timed request, or zero
SMILES, makes the benchmark ``success=False``.

``smiles_validity`` parses EVERY returned SMILES with RDKit and requires ALL of
them to be valid (no fraction threshold); the detail reports the valid fraction
and the first invalid SMILES.
"""

from __future__ import annotations

import json
import logging
import subprocess
import sys
import time
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

# ---------------------------------------------------------------------------
# Benchmark constants
# ---------------------------------------------------------------------------

#: Number of timed iterations for the throughput measurement.
_BENCH_TIMED_ITERS = 5
#: Warmup iterations that are discarded before timing begins.
_BENCH_WARMUP_ITERS = 2
#: Default SMILES count per benchmark request. Upstream RL runs
#: typically sample 128 SMILES per step; 512 per request covers more chemical space for a throughput measurement.
_BENCH_DEFAULT_N_SMILES = 512


def _run_cmd(cmd: list[str], *, cwd: str | None = None, timeout: int = 120) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        cwd=cwd,
        timeout=timeout,
    )


def _post_inference(service_url: str, num_smiles: int, timeout: float = 540) -> dict:
    """POST to {service_url}/v1/inference and return the parsed JSON response.

    The body is sent as ``{"data": {"num_smiles": n}}``: the service's
    ``inference(self, data)`` parameter makes BentoML require the ``data`` wrapper
    (a flat body is rejected with HTTP 400 ``data: Field required``).

    Parameters
    ----------
    service_url:
        Base URL of the running REINVENT4 BentoML service, e.g.
        ``http://localhost:3000``.
    num_smiles:
        How many SMILES to request from the service.
    timeout:
        HTTP socket timeout in seconds.

    Returns
    -------
    dict
        Parsed response with keys ``smiles`` (list[str]) and
        ``num_generated`` (int).

    Raises
    ------
    urllib.error.URLError / HTTPError
        On network or HTTP-level failures.
    ValueError
        If the response body cannot be decoded as JSON.
    """
    payload = json.dumps({"data": {"num_smiles": num_smiles}}).encode()
    req = Request(
        f"{service_url}/v1/inference",
        data=payload,
        headers={"Content-Type": "application/json"},
    )
    with urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read().decode())


class REINVENT4Harness(ModelHarness):
    """Harness for REINVENT4 molecular drug design CLI tool."""

    CHECKS: list[CheckInfo] = [
        CheckInfo(
            "cli_importable",
            CheckResultType.PASS_FAIL,
            CheckScope.RUNTIME,
            "REINVENT4 Python package is importable",
        ),
        CheckInfo(
            "cli_executable",
            CheckResultType.PASS_FAIL,
            CheckScope.RUNTIME,
            "reinvent CLI entry point is callable (--help exits 0)",
        ),
        CheckInfo(
            "gpu_visible",
            CheckResultType.PASS_FAIL,
            CheckScope.RUNTIME,
            "torch.cuda.is_available() returns True inside the container",
        ),
        CheckInfo(
            "service_health",
            CheckResultType.PASS_FAIL,
            CheckScope.OFFLINE,
            "Service answers /health before OFFLINE checks run",
        ),
        CheckInfo(
            "sampling_smoke",
            CheckResultType.PASS_FAIL,
            CheckScope.OFFLINE,
            "Sampling run mode generates at least 1 valid SMILES",
        ),
        CheckInfo(
            "smiles_validity",
            CheckResultType.PASS_FAIL,
            CheckScope.OFFLINE,
            "Every generated SMILES parses as a valid molecule via RDKit (all must be valid)",
        ),
        CheckInfo(
            "throughput_smiles_per_min",
            CheckResultType.SCORE,
            CheckScope.OFFLINE,
            "Total SMILES x 60 / total wall time (ratio of sums; end-to-end: HTTP round trip, sampling on the resident model, RDKit validation)",
        ),
    ]

    def list_checks(self) -> list[CheckInfo]:
        return list(self.CHECKS)

    def health_check(self, service_url: str, timeout_seconds: int = 60) -> bool:
        """Return True when the service answers 200 on /health (or BentoML's /healthz)."""
        if not service_url:
            # Fallback for callers that pass an empty string (legacy validate path).
            result = _run_cmd([sys.executable, "-c", "import reinvent; print('ok')"])
            return result.returncode == 0
        deadline = time.monotonic() + timeout_seconds
        while time.monotonic() < deadline:
            for endpoint in ("/health", "/healthz"):
                try:
                    with urlopen(Request(f"{service_url}{endpoint}"), timeout=5) as resp:
                        if resp.status == 200:
                            return True
                except (HTTPError, URLError, OSError) as exc:
                    logger.debug("%s not ready yet: %s", endpoint, exc)
            time.sleep(1.0)
        return False

    def validate(self, config: HarnessConfig) -> HarnessResult:
        checks: list[CheckResult] = []

        # Local subprocess checks: they need REINVENT4 and a GPU in *this* environment, so they
        # run only for the RUNTIME scope (an OFFLINE-only validation of a remote service must not
        # fail because the caller has neither).
        if CheckScope.RUNTIME in config.check_scopes:
            checks.append(self._check_importable())
            checks.append(self._check_cli_executable())
            checks.append(self._check_gpu_visible())

        if CheckScope.OFFLINE in config.check_scopes:
            service_url = config.resolve_service_url()
            if self.health_check(service_url, timeout_seconds=config.timeout_seconds):
                checks.append(
                    CheckResult(
                        name="service_health",
                        result_type=CheckResultType.PASS_FAIL,
                        success=True,
                        value=True,
                        detail=f"REINVENT4 service answers /health at {service_url}",
                    )
                )
                result, _ = self._run_sampling_smoke(service_url, timeout=config.timeout_seconds)
                checks.append(result)
            else:
                checks.append(
                    CheckResult(
                        name="service_health",
                        result_type=CheckResultType.PASS_FAIL,
                        success=False,
                        value=False,
                        detail=f"REINVENT4 service not reachable at {service_url}; OFFLINE checks not run",
                    )
                )

        success = all(c.success for c in checks)
        return HarnessResult(
            success=success,
            summary=f"REINVENT4 validation {'passed' if success else 'failed'}",
            checks=checks,
        )

    def benchmark(self, config: HarnessConfig) -> HarnessResult:
        """Benchmark REINVENT4 throughput via HTTP POST to the running service.

        Runs ``_BENCH_WARMUP_ITERS`` warmup iterations (discarded) followed by
        ``_BENCH_TIMED_ITERS`` timed iterations.  Each iteration POSTs to
        ``{service_url}/v1/inference`` with the configured ``benchmark_n_smiles``
        value and counts the SMILES returned.  Throughput is computed as the
        total SMILES generated across all timed iterations divided by the total
        elapsed wall-clock time, converted to SMILES per minute.

        Returns a single SCORE CheckResult: ``throughput_smiles_per_min``.
        """
        if CheckScope.OFFLINE not in config.check_scopes:
            return HarnessResult(
                True,
                "REINVENT4 benchmark skipped: OFFLINE scope not requested",
                [skipped("throughput_smiles_per_min", "OFFLINE scope not requested; no service call made")],
            )
        service_url = config.resolve_service_url()
        n_smiles = config.get("benchmark_n_smiles", _BENCH_DEFAULT_N_SMILES)

        if not self.health_check(service_url, timeout_seconds=config.timeout_seconds):
            return HarnessResult(
                success=False,
                summary=f"REINVENT4 service unreachable at {service_url} — cannot benchmark",
            )

        # ── Warmup ────────────────────────────────────────────────────────────
        logger.info("benchmark: running %d warmup iteration(s) …", _BENCH_WARMUP_ITERS)
        for i in range(_BENCH_WARMUP_ITERS):
            try:
                _post_inference(service_url, num_smiles=n_smiles, timeout=config.timeout_seconds)
                logger.debug("warmup %d/%d OK", i + 1, _BENCH_WARMUP_ITERS)
            except Exception as exc:
                logger.warning("warmup %d/%d failed: %s", i + 1, _BENCH_WARMUP_ITERS, exc)

        # ── Timed iterations ──────────────────────────────────────────────────
        logger.info("benchmark: running %d timed iteration(s) …", _BENCH_TIMED_ITERS)
        total_smiles = 0
        errors: list[str] = []

        t_start = time.monotonic()
        for i in range(_BENCH_TIMED_ITERS):
            try:
                resp = _post_inference(service_url, num_smiles=n_smiles, timeout=config.timeout_seconds)
                count = resp.get("num_generated", len(resp.get("smiles", [])))
                total_smiles += count
                logger.debug("timed iter %d/%d: %d SMILES", i + 1, _BENCH_TIMED_ITERS, count)
            except Exception as exc:
                errors.append(f"iter {i + 1}: {exc}")
                logger.warning("timed iter %d/%d failed: %s", i + 1, _BENCH_TIMED_ITERS, exc)
        elapsed_s = time.monotonic() - t_start

        # ── Compute metric ─────────────────────────────────────────────────────
        throughput = (total_smiles * 60.0) / elapsed_s if elapsed_s > 0 else 0.0
        success = len(errors) == 0 and total_smiles > 0

        detail = (
            f"{total_smiles} SMILES in {elapsed_s:.2f}s over {_BENCH_TIMED_ITERS} iter(s) "
            f"= {throughput:.1f} SMILES/min"
        )
        if errors:
            detail += f"; errors: {'; '.join(errors)}"

        throughput_check = CheckResult(
            name="throughput_smiles_per_min",
            result_type=CheckResultType.SCORE,
            success=success,
            value=round(throughput, 2),
            detail=detail,
        )

        return HarnessResult(
            success=success,
            summary=f"REINVENT4 benchmark: {throughput:.1f} SMILES/min ({total_smiles} SMILES, {_BENCH_TIMED_ITERS} iters)",
            checks=[throughput_check],
            metrics={
                "throughput_smiles_per_min": round(throughput, 2),
                "smiles_generated": total_smiles,
                "elapsed_s": round(elapsed_s, 3),
                "timed_iters": _BENCH_TIMED_ITERS,
                "warmup_iters": _BENCH_WARMUP_ITERS,
            },
        )

    def evaluate(self, config: HarnessConfig) -> HarnessResult:
        if CheckScope.OFFLINE not in config.check_scopes:
            return HarnessResult(
                True,
                "REINVENT4 evaluation skipped: OFFLINE scope not requested",
                [
                    skipped("sampling_smoke", "OFFLINE scope not requested; no service call made"),
                    skipped("smiles_validity", "OFFLINE scope not requested; no service call made"),
                ],
            )
        service_url = config.resolve_service_url()

        if not self.health_check(service_url, timeout_seconds=config.timeout_seconds):
            return HarnessResult(
                success=False,
                summary=f"REINVENT4 service unreachable at {service_url} — cannot evaluate",
            )

        check, smiles_list = self._run_sampling_via_http(service_url, n_smiles=10, timeout=config.timeout_seconds)
        validity_check = self._check_smiles_validity_from_list(smiles_list)

        success = check.success and validity_check.success
        return HarnessResult(
            success=success,
            summary=f"REINVENT4 evaluation {'passed' if success else 'failed'}: {len(smiles_list)} SMILES generated",
            checks=[check, validity_check],
            metrics={
                "smiles_generated": len(smiles_list),
            },
        )

    @staticmethod
    def _check_importable() -> CheckResult:
        try:
            result = _run_cmd([sys.executable, "-c", "import reinvent; print(reinvent.__version__)"])
            if result.returncode == 0:
                version = result.stdout.strip()
                return CheckResult(
                    name="cli_importable",
                    result_type=CheckResultType.PASS_FAIL,
                    success=True,
                    value=True,
                    detail=f"reinvent package importable, version={version!r}",
                )
            return CheckResult(
                name="cli_importable",
                result_type=CheckResultType.PASS_FAIL,
                success=False,
                value=False,
                detail=f"import reinvent failed: {result.stderr.strip()!r}",
            )
        except Exception as exc:
            return CheckResult(
                name="cli_importable",
                result_type=CheckResultType.PASS_FAIL,
                success=False,
                value=False,
                detail=str(exc),
            )

    @staticmethod
    def _check_cli_executable() -> CheckResult:
        try:
            result = _run_cmd(["reinvent", "--help"], timeout=30)
            if result.returncode == 0:
                return CheckResult(
                    name="cli_executable",
                    result_type=CheckResultType.PASS_FAIL,
                    success=True,
                    value=True,
                    detail="reinvent --help returned 0",
                )
            return CheckResult(
                name="cli_executable",
                result_type=CheckResultType.PASS_FAIL,
                success=False,
                value=False,
                detail=f"reinvent --help returned {result.returncode}: {result.stderr.strip()!r}",
            )
        except FileNotFoundError:
            return CheckResult(
                name="cli_executable",
                result_type=CheckResultType.PASS_FAIL,
                success=False,
                value=False,
                detail="reinvent binary not found on PATH",
            )
        except Exception as exc:
            return CheckResult(
                name="cli_executable",
                result_type=CheckResultType.PASS_FAIL,
                success=False,
                value=False,
                detail=str(exc),
            )

    @staticmethod
    def _check_gpu_visible() -> CheckResult:
        try:
            script = (
                "import torch; "
                "avail = torch.cuda.is_available(); "
                "n = torch.cuda.device_count() if avail else 0; "
                "name = torch.cuda.get_device_name(0) if avail else 'none'; "
                "print(avail, n, name)"
            )
            result = _run_cmd([sys.executable, "-c", script], timeout=30)
            if result.returncode == 0:
                parts = result.stdout.strip().split(None, 2)
                avail = parts[0] == "True"
                detail = result.stdout.strip()
                return CheckResult(
                    name="gpu_visible",
                    result_type=CheckResultType.PASS_FAIL,
                    success=avail,
                    value=avail,
                    detail=detail if avail else f"torch.cuda.is_available()=False; {detail}",
                )
            return CheckResult(
                name="gpu_visible",
                result_type=CheckResultType.PASS_FAIL,
                success=False,
                value=False,
                detail=f"GPU visibility check failed (rc={result.returncode}): {result.stderr.strip()!r}",
            )
        except Exception as exc:
            return CheckResult(
                name="gpu_visible",
                result_type=CheckResultType.PASS_FAIL,
                success=False,
                value=False,
                detail=str(exc),
            )

    @staticmethod
    def _run_sampling_smoke(service_url: str, timeout: float = 300) -> tuple[CheckResult, int]:
        """Verify the GPU pipeline end-to-end by requesting 1 SMILES via HTTP."""
        try:
            resp = _post_inference(service_url, num_smiles=1, timeout=timeout)
            smiles_list: list[str] = resp.get("smiles", [])
            n = resp.get("num_generated", len(smiles_list))

            if not smiles_list:
                return (
                    CheckResult(
                        name="sampling_smoke",
                        result_type=CheckResultType.PASS_FAIL,
                        success=False,
                        value=False,
                        detail="POST /v1/inference returned 0 SMILES",
                    ),
                    0,
                )

            return (
                CheckResult(
                    name="sampling_smoke",
                    result_type=CheckResultType.PASS_FAIL,
                    success=True,
                    value=True,
                    detail=f"Sampling produced {n} SMILES; first={smiles_list[0]!r}",
                ),
                n,
            )

        except Exception as exc:
            return (
                CheckResult(
                    name="sampling_smoke",
                    result_type=CheckResultType.PASS_FAIL,
                    success=False,
                    value=False,
                    detail=str(exc),
                ),
                0,
            )

    @staticmethod
    def _run_sampling_via_http(
        service_url: str, n_smiles: int = 10, timeout: float = 600
    ) -> tuple[CheckResult, list[str]]:
        """POST to the service and return a PASS_FAIL check + the SMILES list."""
        try:
            resp = _post_inference(service_url, num_smiles=n_smiles, timeout=timeout)
            smiles_list: list[str] = resp.get("smiles", [])

            return (
                CheckResult(
                    name="sampling_smoke",
                    result_type=CheckResultType.PASS_FAIL,
                    success=len(smiles_list) > 0,
                    value=len(smiles_list) > 0,
                    detail=f"Generated {len(smiles_list)}/{n_smiles} SMILES",
                ),
                smiles_list,
            )

        except Exception as exc:
            return (
                CheckResult(
                    name="sampling_smoke",
                    result_type=CheckResultType.PASS_FAIL,
                    success=False,
                    value=False,
                    detail=str(exc),
                ),
                [],
            )

    @staticmethod
    def _check_smiles_validity_from_list(smiles_list: list[str]) -> CheckResult:
        """Parse every generated SMILES with RDKit; ALL must be valid (no fraction threshold)."""
        if not smiles_list:
            return CheckResult(
                name="smiles_validity",
                result_type=CheckResultType.PASS_FAIL,
                success=False,
                value=False,
                detail="Skipped — no SMILES to validate",
            )

        try:
            from rdkit import Chem  # type: ignore[import-untyped]

            invalid = [
                smi for smi in smiles_list if not isinstance(smi, str) or not smi or Chem.MolFromSmiles(smi) is None
            ]
            n_valid = len(smiles_list) - len(invalid)
            valid = not invalid
            detail = f"RDKit parse of all SMILES: {n_valid}/{len(smiles_list)} valid (fraction {n_valid / len(smiles_list):.3f}); all must be valid"
            if invalid:
                detail += f"; first invalid: {invalid[0]!r}"
            return CheckResult(
                name="smiles_validity",
                result_type=CheckResultType.PASS_FAIL,
                success=valid,
                value=valid,
                detail=detail,
            )

        except ImportError:
            logger.error("rdkit not importable; the image is missing a REINVENT4 dependency")
            return CheckResult(
                name="smiles_validity",
                result_type=CheckResultType.PASS_FAIL,
                success=False,
                value=False,
                detail="RDKit is not importable, so the SMILES cannot be validated",
            )
