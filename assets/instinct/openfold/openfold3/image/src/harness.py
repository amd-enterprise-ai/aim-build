# Copyright © Advanced Micro Devices, Inc., or its affiliates.
#
# SPDX-License-Identifier: MIT

"""
OpenFold3Harness — ModelHarness implementation for the OpenFold3 BentoML
service.

The harness talks to the local BentoML service over HTTP (``/healthz`` and
``/predict``); it does not import the OpenFold3 model directly.
"""

from __future__ import annotations

import json
import logging
import math
import statistics
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

logger = logging.getLogger(__name__)

BENTOML_HEALTH_ENDPOINT = "/healthz"
BENTOML_PREDICT_ENDPOINT = "/predict"
BENTOML_PORT = 8000  # matches ENV BENTOML_PORT in the OF3 Dockerfile

# Per-PDB payloads (output of build_of3_inputs.py) baked into the image by
# the Dockerfile at /workspace/model/benchmarks/<pdb>/<pdb>.json.
BENCHMARK_DIR = Path("/workspace/model/benchmarks")

# NVIDIA-published reference latencies for the standard 21-PDB set —
# H100 80GB TRT + cuEquivariance, single diffusion sample, no templates.
# Source: https://docs.nvidia.com/nim/bionemo/openfold3/latest/performance.html
NVIDIA_REFERENCE: dict[str, dict[str, float]] = {
    "8eil": {"length": 186, "nvidia_trt_s": 12.20},
    "7r6r": {"length": 203, "nvidia_trt_s": 12.72},
    "1a3n": {"length": 287, "nvidia_trt_s": 22.54},
    "8c4d": {"length": 331, "nvidia_trt_s": 13.80},
    "7qsj": {"length": 375, "nvidia_trt_s": 15.24},
    "8cpk": {"length": 384, "nvidia_trt_s": 18.33},
    "8are": {"length": 530, "nvidia_trt_s": 20.60},
    "8owf": {"length": 575, "nvidia_trt_s": 21.57},
    "8aw3": {"length": 590, "nvidia_trt_s": 34.50},
    "7tpu": {"length": 616, "nvidia_trt_s": 19.78},
    "7ylz": {"length": 623, "nvidia_trt_s": 26.78},
    "8gpp": {"length": 628, "nvidia_trt_s": 25.15},
    "8clz": {"length": 684, "nvidia_trt_s": 25.64},
    "8k7x": {"length": 858, "nvidia_trt_s": 30.69},
    "8ibx": {"length": 1286, "nvidia_trt_s": 39.58},
    "8gi1": {"length": 1464, "nvidia_trt_s": 56.50},
    "8sm6": {"length": 1496, "nvidia_trt_s": 63.45},
    "8pso": {"length": 1499, "nvidia_trt_s": 54.01},
    "8jue": {"length": 1657, "nvidia_trt_s": 73.99},
    "8bsh": {"length": 1762, "nvidia_trt_s": 85.16},
    "5xgo": {"length": 1869, "nvidia_trt_s": 97.41},
}

# E. coli thioredoxin 1 / TrxA (UniProt P00274, 109 AA) — small, fast-folding,
# well-characterised. Used for /validate and /evaluate smoke runs
SMOKE_SEQUENCE = (
    "MSDKIIHLTDDSFDTDVLKADGAILVDFWAEWCGPCKMIAPILDEIADEYQGKLTVAKLNI" "DQNPGTAPKYGIRGIPTLLLFKNGEVAATKVGALSKGQLKEFLDANLA"
)
SMOKE_QUERY_NAME = "thioredoxin_109aa_smoke"


def _build_smoke_payload(output_format: str = "pdb", inline_msa: bool = False) -> dict[str, Any]:
    """Build a single-protein /predict payload for the smoke sequence.

    BentoML wraps the OpenFold3Request fields under a top-level ``data`` key;
    fields match :class:`service.OpenFold3Request`. When ``inline_msa`` is set,
    the chain carries a single-sequence a3m via ``main_msa``. ``use_msa_server``
    is always ``False`` regardless of ``inline_msa``.
    """
    chain: dict[str, Any] = {
        "molecule_type": "protein",
        "chain_ids": ["A"],
        "sequence": SMOKE_SEQUENCE,
    }
    if inline_msa:
        chain["main_msa"] = f">query\n{SMOKE_SEQUENCE}\n"
    return {
        "data": {
            "queries": {
                SMOKE_QUERY_NAME: {
                    "chains": [chain],
                },
            },
            "num_diffusion_samples": 1,
            "num_model_seeds": 1,
            "use_msa_server": False,
            "use_templates": False,
            "output_format": output_format,
        },
    }


def _post_predict(service_url: str, payload: dict[str, Any], *, timeout_seconds: int) -> tuple[dict[str, Any], float]:
    """POST a payload to ``/predict``; return (response_json, elapsed_seconds).

    Raises on transport error, non-200 status, server-reported
    ``error`` flag, or empty ``structures`` list.
    """
    body = json.dumps(payload).encode("utf-8")
    req = Request(
        f"{service_url}{BENTOML_PREDICT_ENDPOINT}",
        data=body,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    t0 = time.monotonic()
    with urlopen(req, timeout=timeout_seconds) as resp:
        elapsed = time.monotonic() - t0
        if resp.status != 200:
            raise RuntimeError(f"/predict returned HTTP {resp.status}")
        data = json.loads(resp.read().decode("utf-8"))
    if data.get("error"):
        raise RuntimeError(f"server error: {data.get('message', 'unknown')}")
    if not data.get("structures"):
        raise RuntimeError("server returned no structures")
    return data, elapsed


def _count_atom_records(structure_content: str) -> int:
    """Count ATOM lines in PDB or mmCIF output (works for both formats)."""
    return sum(1 for line in structure_content.splitlines() if line.startswith("ATOM"))


def _structure_has_finite_coords(structure_content: str) -> bool:
    """Return True if at least one ATOM line carries three finite-float coords.

    PDB fixed-column layout (cols 31-38, 39-46, 47-54). mmCIF whitespace-split
    layout: protein atoms come out with the three x/y/z floats embedded in the
    line; we scan tokens and look for the first three parse-able floats.
    """
    for line in structure_content.splitlines():
        if not line.startswith("ATOM"):
            continue
        # Try PDB fixed-column slice first.
        try:
            x, y, z = float(line[30:38]), float(line[38:46]), float(line[46:54])
        except (ValueError, IndexError):
            # Fall back to whitespace tokens (mmCIF).
            floats: list[float] = []
            for tok in line.split():
                try:
                    floats.append(float(tok))
                except ValueError:
                    continue
                if len(floats) == 3:
                    break
            if len(floats) < 3:
                continue
            x, y, z = floats[0], floats[1], floats[2]
        if all(math.isfinite(v) for v in (x, y, z)):
            return True
    return False


def _percentile(values: list[float], pct: float) -> float:
    if not values:
        return 0.0
    sorted_v = sorted(values)
    if len(sorted_v) == 1:
        return sorted_v[0]
    idx = (pct / 100.0) * (len(sorted_v) - 1)
    lo, hi = int(idx), min(int(idx) + 1, len(sorted_v) - 1)
    return sorted_v[lo] + (idx - lo) * (sorted_v[hi] - sorted_v[lo])


class OpenFold3Harness(ModelHarness):
    """Harness for the OpenFold3 protein/nucleic-acid/ligand structure model."""

    ENGINE = "bentoml"

    CHECKS: list[CheckInfo] = [
        CheckInfo(
            "bentoml_health",
            CheckResultType.PASS_FAIL,
            CheckScope.RUNTIME,
            "BentoML /healthz returns 200",
        ),
        CheckInfo(
            "predict_smoke",
            CheckResultType.PASS_FAIL,
            CheckScope.OFFLINE,
            "Single-sequence prediction returns a structure with ATOM records",
        ),
        CheckInfo(
            "predict_inline_msa",
            CheckResultType.PASS_FAIL,
            CheckScope.OFFLINE,
            "Prediction with an inline precomputed MSA returns a structure with ATOM records",
        ),
        CheckInfo(
            "benchmark_21_pdb",
            CheckResultType.SCORE,
            CheckScope.OFFLINE,
            "Geomean latency ratio across the 21 NVIDIA-reference PDB cases (vs H100 TRT)",
        ),
        CheckInfo(
            "structure_sanity",
            CheckResultType.PASS_FAIL,
            CheckScope.OFFLINE,
            "Smoke prediction's structure parses with finite atomic coordinates",
        ),
    ]

    def list_checks(self) -> list[CheckInfo]:
        return list(self.CHECKS)

    def health_check(self, service_url: str, timeout_seconds: int = 60) -> bool:
        deadline = time.monotonic() + timeout_seconds
        while time.monotonic() < deadline:
            try:
                with urlopen(
                    Request(f"{service_url}{BENTOML_HEALTH_ENDPOINT}"),
                    timeout=5,
                ) as resp:
                    if resp.status == 200:
                        return True
            except (HTTPError, URLError, OSError):
                pass
            time.sleep(2.0)
        return False

    def validate(self, config: HarnessConfig) -> HarnessResult:
        """Check service health and that it produces structures for the smoke and inline-MSA paths."""
        service_url = config.resolve_service_url()
        health_ok = self.health_check(service_url, timeout_seconds=config.timeout_seconds)

        checks: list[CheckResult] = [
            CheckResult(
                name="bentoml_health",
                result_type=CheckResultType.PASS_FAIL,
                success=health_ok,
                value=health_ok,
                detail="" if health_ok else f"BentoML not reachable at {service_url}",
            )
        ]

        if health_ok and CheckScope.OFFLINE in config.check_scopes:
            checks.append(self._predict_smoke(service_url, config))
            checks.append(
                self._predict_check(
                    service_url,
                    config,
                    payload=_build_smoke_payload(output_format="pdb", inline_msa=True),
                    name="predict_inline_msa",
                )
            )

        success = all(c.success for c in checks)
        return HarnessResult(
            success=success,
            summary=f"OpenFold3 validation {'passed' if success else 'failed'}",
            checks=checks,
        )

    def benchmark(self, config: HarnessConfig) -> HarnessResult:
        """Run the 21-PDB benchmark suite and compare latency to NVIDIA H100 TRT reference."""
        service_url = config.resolve_service_url()

        if not self.health_check(service_url, timeout_seconds=config.timeout_seconds):
            return HarnessResult(
                success=False,
                summary=f"BentoML service not reachable at {service_url}",
            )

        runs_per_pdb = int(config.get("benchmark_runs", 3))
        request_timeout = int(config.get("predict_timeout_seconds", 600))

        per_pdb_metrics: dict[str, dict[str, Any]] = {}
        ratios: list[float] = []
        any_failed = False

        for pdb_id, ref in NVIDIA_REFERENCE.items():
            payload = self._load_pdb_payload(pdb_id)
            if payload is None:
                logger.warning("Benchmark payload missing for %s; skipping", pdb_id)
                any_failed = True
                continue

            latencies: list[float] = []
            for run_idx in range(runs_per_pdb):
                try:
                    _, elapsed = _post_predict(service_url, payload, timeout_seconds=request_timeout)
                except Exception as exc:
                    logger.warning("Benchmark %s run %d/%d failed: %s", pdb_id, run_idx + 1, runs_per_pdb, exc)
                    any_failed = True
                    continue
                latencies.append(elapsed)

            if not latencies:
                continue

            mean_s = statistics.mean(latencies)
            ratio = mean_s / ref["nvidia_trt_s"]
            ratios.append(ratio)
            per_pdb_metrics[pdb_id] = {
                "length": ref["length"],
                "runs": len(latencies),
                "mean_s": round(mean_s, 3),
                "median_s": round(statistics.median(latencies), 3),
                "p95_s": round(_percentile(latencies, 95), 3),
                "min_s": round(min(latencies), 3),
                "max_s": round(max(latencies), 3),
                "nvidia_trt_s": ref["nvidia_trt_s"],
                "vs_nvidia_ratio": round(ratio, 3),
            }

        if not ratios:
            return HarnessResult(
                success=False,
                summary="All 21 benchmark cases failed",
            )

        # Geomean is the right aggregator for a ratio across multiple
        # workloads — equal weight to small-protein and large-protein latency
        # differences. Lower is better (faster than NVIDIA H100 TRT).
        geomean_ratio = math.exp(sum(math.log(r) for r in ratios) / len(ratios))

        checks = [
            CheckResult(
                name="benchmark_21_pdb",
                result_type=CheckResultType.SCORE,
                success=not any_failed,
                value=round(geomean_ratio, 3),
                detail=(
                    f"{len(ratios)}/{len(NVIDIA_REFERENCE)} PDB cases benchmarked "
                    f"(runs/case={runs_per_pdb}); geomean MI300X/H100 TRT ratio = "
                    f"{geomean_ratio:.3f} (<1 is faster than NVIDIA reference)"
                ),
            )
        ]

        return HarnessResult(
            success=not any_failed,
            summary=f"OpenFold3 benchmark: geomean vs NVIDIA H100 TRT = {geomean_ratio:.3f}",
            checks=checks,
            metrics={
                "geomean_vs_nvidia_ratio": round(geomean_ratio, 4),
                "per_pdb": per_pdb_metrics,
                "runs_per_pdb": runs_per_pdb,
            },
        )

    def evaluate(self, config: HarnessConfig) -> HarnessResult:
        """Shallow domain check on a single smoke prediction.

        OF3 has no structural-accuracy code today (no RMSD/lDDT/TM-score
        vs ground truth). evaluate is intentionally a shallow correctness
        check — does the model produce a parseable structure with finite
        coordinates?
        """
        service_url = config.resolve_service_url()

        if not self.health_check(service_url, timeout_seconds=config.timeout_seconds):
            return HarnessResult(
                success=False,
                summary=f"BentoML service not reachable at {service_url}",
            )

        request_timeout = int(config.get("predict_timeout_seconds", 600))
        payload = _build_smoke_payload(output_format="pdb")

        try:
            response, _ = _post_predict(service_url, payload, timeout_seconds=request_timeout)
        except Exception as exc:
            return HarnessResult(
                success=False,
                summary=f"OpenFold3 evaluate: smoke prediction failed: {exc}",
                checks=[
                    CheckResult(
                        name="structure_sanity",
                        result_type=CheckResultType.PASS_FAIL,
                        success=False,
                        value=False,
                        detail=str(exc),
                    )
                ],
            )

        structures = response.get("structures", [])
        content = structures[0].get("content", "") if structures else ""
        atom_count = _count_atom_records(content)
        coords_finite = _structure_has_finite_coords(content)

        success = atom_count > 0 and coords_finite

        checks = [
            CheckResult(
                name="structure_sanity",
                result_type=CheckResultType.PASS_FAIL,
                success=success,
                value=success,
                detail=(f"structures={len(structures)}, ATOM records={atom_count}, " f"finite_coords={coords_finite}"),
            )
        ]

        return HarnessResult(
            success=success,
            summary=f"OpenFold3 evaluate {'passed' if success else 'failed'}",
            checks=checks,
            metrics={
                "atom_records": atom_count,
                "structures_returned": len(structures),
            },
        )

    def _predict_smoke(self, service_url: str, config: HarnessConfig) -> CheckResult:
        return self._predict_check(
            service_url,
            config,
            payload=_build_smoke_payload(output_format="pdb"),
            name="predict_smoke",
        )

    def _predict_check(
        self,
        service_url: str,
        config: HarnessConfig,
        *,
        payload: dict[str, Any],
        name: str,
    ) -> CheckResult:
        request_timeout = int(config.get("predict_timeout_seconds", 600))
        try:
            response, elapsed = _post_predict(service_url, payload, timeout_seconds=request_timeout)
        except Exception as exc:
            return CheckResult(
                name=name,
                result_type=CheckResultType.PASS_FAIL,
                success=False,
                value=False,
                detail=f"prediction failed: {exc}",
            )

        structures = response.get("structures", [])
        content = structures[0].get("content", "") if structures else ""
        atom_count = _count_atom_records(content)
        success = atom_count > 0

        return CheckResult(
            name=name,
            result_type=CheckResultType.PASS_FAIL,
            success=success,
            value=success,
            detail=(f"elapsed={elapsed:.1f}s, structures={len(structures)}, " f"ATOM records={atom_count}"),
        )

    def _load_pdb_payload(self, pdb_id: str) -> dict[str, Any] | None:
        """Read ``/workspace/model/benchmarks/<pdb>/<pdb>.json``.

        ``build_of3_inputs.py`` emits each payload as ``{"metadata": ..., "data": {...}}``;
        the wire format the OF3 service expects is the same ``{"data": {...}}`` wrapper.
        We pass through the file contents and let the server unwrap.
        """
        payload_path = BENCHMARK_DIR / pdb_id / f"{pdb_id}.json"
        if not payload_path.exists():
            return None
        try:
            doc = json.loads(payload_path.read_text())
        except json.JSONDecodeError as exc:
            logger.warning("Failed to parse %s: %s", payload_path, exc)
            return None
        if "data" not in doc:
            return None
        return {"data": doc["data"]}
