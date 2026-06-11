# Copyright © Advanced Micro Devices, Inc., or its affiliates.
#
# SPDX-License-Identifier: MIT

"""
Boltz2Harness — model-specific harness for the Boltz-2 protein structure
prediction model served via BentoML.

This file lives at ``assets/instinct/mit/boltz2/image/src/harness.py`` and
is copied to ``/workspace/model/src/harness.py`` by the specialized
Dockerfile. aim-runtime discovers it automatically via the well-known path.

Because ``/workspace/model/src/`` is on ``sys.path`` (via a ``.pth`` file
installed by the specialized Dockerfile), this module can import sibling modules
in the same directory (e.g. ``from service import BoltzService``) as well
as aim-runtime's public API (``from aim_runtime.harness import ...``).
"""

from __future__ import annotations

import json
import logging
import time
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

logger = logging.getLogger(__name__)

BENTOML_HEALTH_ENDPOINT = "/healthz"
BENTOML_PREDICT_ENDPOINT = "/predict"
BENTOML_PORT = 3000

# Trp-cage miniprotein (20 residues) — folds in microseconds, ideal for smoke tests.
SMOKE_SEQUENCE = "NLYIQWLKDGGPSSGRPPPS"

# Hemoglobin alpha subunit (50-residue fragment) — a well-characterized structure
# useful for evaluating whether predicted output is chemically reasonable.
EVAL_SEQUENCE = "MVLSPADKTNVKAAWGKVGAHAGEYGAEALERMFLSFPTTKTYFPHFDLSH"

# Minimal diffusion parameters for fast smoke/benchmark runs.
FAST_PREDICT_PARAMS: dict[str, Any] = {
    "sampling_steps": 10,
    "diffusion_samples": 1,
    "recycling_steps": 1,
}


def _build_predict_payload(
    sequence: str,
    *,
    chain_id: str = "A",
    name: str = "harness_run",
    output_format: str = "pdb",
    use_msa_server: bool = False,
    **overrides: Any,
) -> dict[str, Any]:
    """Build a Boltz-2 /predict request body."""
    params = {**FAST_PREDICT_PARAMS, **overrides}
    return {
        "data": {
            "version": 1,
            "sequences": [
                {"protein": {"id": chain_id, "sequence": sequence}},
            ],
            "name": name,
            "use_msa_server": use_msa_server,
            "output_format": output_format,
            **params,
        },
    }


class Boltz2Harness(ModelHarness):
    """Harness for Boltz-2 protein structure prediction model (BentoML)."""

    # ------------------------------------------------------------------ #
    # Check registry
    # ------------------------------------------------------------------ #

    CHECKS: list[CheckInfo] = [
        CheckInfo("bentoml_health", CheckResultType.PASS_FAIL, CheckScope.RUNTIME, "BentoML /healthz returns 200"),
        CheckInfo(
            "predict_smoke",
            CheckResultType.PASS_FAIL,
            CheckScope.OFFLINE,
            "Single-sequence prediction returns PDB output",
        ),
        CheckInfo("throughput", CheckResultType.SCORE, CheckScope.OFFLINE, "Structures predicted per minute"),
        CheckInfo(
            "structure_sanity",
            CheckResultType.PASS_FAIL,
            CheckScope.OFFLINE,
            "Predicted PDB contains ATOM records and plausible coordinates",
        ),
    ]

    def list_checks(self) -> list[CheckInfo]:
        return list(self.CHECKS)

    # ------------------------------------------------------------------ #
    # health_check override — BentoML uses /healthz, not /v1/models
    # ------------------------------------------------------------------ #

    def health_check(self, service_url: str, timeout_seconds: int = 60) -> bool:
        deadline = time.time() + timeout_seconds
        while time.time() < deadline:
            try:
                req = Request(f"{service_url}{BENTOML_HEALTH_ENDPOINT}")
                with urlopen(req, timeout=5) as resp:
                    if resp.status == 200:
                        return True
            except (HTTPError, URLError, OSError):
                pass
            time.sleep(2.0)
        return False

    # ------------------------------------------------------------------ #
    # validate — "does it work?"
    # ------------------------------------------------------------------ #

    def validate(self, config: HarnessConfig) -> HarnessResult:
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

        if health_ok and CheckScope.OFFLINE in config.check_scopes:
            checks.append(self._predict_smoke_test(service_url))

        success = all(c.success for c in checks)
        return HarnessResult(
            success=success,
            summary=f"Boltz-2 validation {'passed' if success else 'failed'}",
            checks=checks,
        )

    # ------------------------------------------------------------------ #
    # benchmark — "how fast is it?"
    # ------------------------------------------------------------------ #

    def benchmark(self, config: HarnessConfig) -> HarnessResult:
        service_url = self._service_url(config)

        if not self.health_check(service_url, timeout_seconds=config.timeout_seconds):
            return HarnessResult(success=False, summary="BentoML service not reachable for benchmarking")

        n_requests = config.profile.get("benchmark_n_requests", 10)
        latencies: list[float] = []

        for i in range(n_requests):
            t0 = time.time()
            try:
                self._send_predict(service_url, SMOKE_SEQUENCE, name=f"bench_{i}")
                latencies.append(time.time() - t0)
            except Exception as exc:
                logger.warning("Benchmark request %d/%d failed: %s", i + 1, n_requests, exc)

        if not latencies:
            return HarnessResult(success=False, summary="All benchmark requests failed")

        avg_latency = sum(latencies) / len(latencies)
        throughput = 60.0 / avg_latency if avg_latency > 0 else 0.0

        checks = [
            CheckResult(
                name="throughput",
                result_type=CheckResultType.SCORE,
                success=True,
                value=round(throughput, 2),
                detail=f"{len(latencies)}/{n_requests} requests succeeded, "
                f"avg latency={avg_latency:.2f}s, "
                f"throughput={throughput:.1f} structures/min",
            )
        ]

        return HarnessResult(
            success=True,
            summary=f"Boltz-2 benchmark: {throughput:.1f} structures/min",
            checks=checks,
            metrics={
                "avg_latency_s": round(avg_latency, 3),
                "throughput_per_min": round(throughput, 2),
                "requests_succeeded": len(latencies),
                "requests_total": n_requests,
            },
        )

    # ------------------------------------------------------------------ #
    # evaluate — "does it produce correct structures?"
    # ------------------------------------------------------------------ #

    def evaluate(self, config: HarnessConfig) -> HarnessResult:
        service_url = self._service_url(config)

        if not self.health_check(service_url, timeout_seconds=config.timeout_seconds):
            return HarnessResult(success=False, summary="BentoML service not reachable for evaluation")

        checks: list[CheckResult] = []
        checks.append(self._check_structure_sanity(service_url))

        success = all(c.success for c in checks)
        return HarnessResult(
            success=success,
            summary=f"Boltz-2 evaluation {'passed' if success else 'failed'}",
            checks=checks,
            metrics={
                "sequence": EVAL_SEQUENCE,
                "sequence_length": len(EVAL_SEQUENCE),
            },
        )

    # ------------------------------------------------------------------ #
    # Internal helpers
    # ------------------------------------------------------------------ #

    @staticmethod
    def _service_url(config: HarnessConfig) -> str:
        return config.resolve_service_url()

    @staticmethod
    def _send_predict(service_url: str, sequence: str, **kwargs: Any) -> str:
        """Send a prediction request and return the raw response body.

        Returns the response as a string — for PDB output this is the PDB
        text, for JSON output it would be a JSON string.
        """
        payload = json.dumps(_build_predict_payload(sequence, **kwargs)).encode()
        req = Request(
            f"{service_url}{BENTOML_PREDICT_ENDPOINT}",
            data=payload,
            headers={"Content-Type": "application/json"},
        )
        with urlopen(req, timeout=300) as resp:
            return resp.read().decode()

    @classmethod
    def _predict_smoke_test(cls, service_url: str) -> CheckResult:
        """Submit a tiny protein and verify the service returns non-empty output.

        This is a validate-level check: it only verifies the prediction
        pipeline runs end-to-end and returns *something*. It doesn't assess
        the quality of the predicted structure (that's evaluate's job).
        """
        try:
            body = cls._send_predict(service_url, SMOKE_SEQUENCE, name="smoke_test")

            if body and len(body) > 0:
                return CheckResult(
                    name="predict_smoke",
                    result_type=CheckResultType.PASS_FAIL,
                    success=True,
                    value=True,
                    detail=f"Prediction returned {len(body)} bytes",
                )

            return CheckResult(
                name="predict_smoke",
                result_type=CheckResultType.PASS_FAIL,
                success=False,
                value=False,
                detail="Prediction returned empty response",
            )

        except Exception as exc:
            return CheckResult(
                name="predict_smoke",
                result_type=CheckResultType.PASS_FAIL,
                success=False,
                value=False,
                detail=str(exc),
            )

    @classmethod
    def _check_structure_sanity(cls, service_url: str) -> CheckResult:
        """Predict a known protein and verify the output looks like a valid structure.

        Checks that the PDB output contains ATOM records with reasonable
        coordinate values — a basic sanity gate that catches total prediction
        failures without requiring a reference structure.
        """
        try:
            body = cls._send_predict(
                service_url,
                EVAL_SEQUENCE,
                name="eval_hemoglobin",
                use_msa_server=True,
            )

            atom_lines = [line for line in body.splitlines() if line.startswith("ATOM")]
            if not atom_lines:
                return CheckResult(
                    name="structure_sanity",
                    result_type=CheckResultType.PASS_FAIL,
                    success=False,
                    value=False,
                    detail=f"No ATOM records in PDB output ({len(body)} bytes total)",
                )

            # Verify coordinates are finite and in a plausible range (< 1000 Å)
            coords_ok = True
            for line in atom_lines[:10]:
                try:
                    x = float(line[30:38])
                    y = float(line[38:46])
                    z = float(line[46:54])
                    if any(abs(c) > 1000 for c in (x, y, z)):
                        coords_ok = False
                        break
                except (ValueError, IndexError):
                    coords_ok = False
                    break

            return CheckResult(
                name="structure_sanity",
                result_type=CheckResultType.PASS_FAIL,
                success=coords_ok,
                value=coords_ok,
                detail=f"{len(atom_lines)} ATOM records, "
                f"coordinates {'plausible' if coords_ok else 'out of range'}",
            )

        except Exception as exc:
            return CheckResult(
                name="structure_sanity",
                result_type=CheckResultType.PASS_FAIL,
                success=False,
                value=False,
                detail=str(exc),
            )
