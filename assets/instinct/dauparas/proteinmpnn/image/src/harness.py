# Copyright © Advanced Micro Devices, Inc., or its affiliates.
#
# SPDX-License-Identifier: MIT

"""
ProteinMPNNHarness — ModelHarness implementation for the ProteinMPNN BentoML
service.

The harness talks to the local BentoML service over HTTP (``/health``, falling back to
``/healthz``, and ``/v1/inference``); it does not invoke ProteinMPNN directly.

This file lives at ``assets/instinct/dauparas/proteinmpnn/image/src/harness.py``
and is copied to ``/workspace/model/src/harness.py`` by the specialized
Dockerfile. aim-runtime discovers it automatically via the well-known path.

Scope: every check calls the service and is therefore OFFLINE (ProteinMPNN has no
RUNTIME checks). ``validate``, ``benchmark`` and ``evaluate`` make no network
call and return skipped results when OFFLINE is not among the requested scopes.

Metric definitions
------------------
``throughput`` = 60 / mean latency of ``benchmark_n_requests`` (default 5)
serial requests on the smoke PDB with ``benchmark_num_seq_per_target`` (default 4)
sequences each, in structures per minute. There is no warm-up and no concurrency.
The service keeps one resident, warmed-up model (it warms up before it reports ready), so a
latency is the in-process design time plus the HTTP round trip.

``aa_per_second`` = total residues designed / total wall time, a RATIO OF SUMS
over the timed calls of four synthetic backbone fixtures (1 sequence per call,
2 discarded warm-up and 5 timed calls per fixture). The time includes HTTP and
the in-process design time (with the earlier one-process-per-request service it also
included the per-call process start and weight load). (Earlier revisions reported the mean
of per-call ratios; the ratio of sums weights long calls proportionally and gives
a lower, more representative number whenever call times differ.) Any failed
timed request makes the benchmark ``success=False``.
"""

from __future__ import annotations

import json
import logging
import math
import os
import statistics
import tempfile
import time
from pathlib import Path
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

# Sibling module: /workspace/model/src is on sys.path in the image (see the Dockerfile .pth).
from benchmark_fixtures import (
    SYNTHETIC_A_PDB,
    SYNTHETIC_B_PDB,
    SYNTHETIC_C_PDB,
    SYNTHETIC_D_PDB,
    count_residues,
)

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
# BentoML built-in; /health forwards to it, so either answer means ready.
BENTOML_HEALTHZ_ENDPOINT = "/healthz"
INFERENCE_ENDPOINT = "/v1/inference"


PROTEIN_ALPHABET = frozenset("ACDEFGHIKLMNPQRSTVWYX")


def _is_valid_design(sequence: str) -> bool:
    """True when *sequence* is a non-empty design over the protein alphabet (plus X).

    Multi-chain designs join chains with "/"; every chain must be non-empty.
    """
    if not sequence:
        return False
    chains = sequence.upper().split("/")
    return all(chain and all(c in PROTEIN_ALPHABET for c in chain) for chain in chains)


# List of (fixture_name, residue_count, pdb_string); residue_count is computed.
BENCHMARK_PDBS: list[tuple[str, int, str]] = [
    (name, count_residues(pdb), pdb)
    for name, pdb in (
        ("synthetic_a", SYNTHETIC_A_PDB),
        ("synthetic_b", SYNTHETIC_B_PDB),
        ("synthetic_c", SYNTHETIC_C_PDB),
        ("synthetic_d", SYNTHETIC_D_PDB),
    )
]

# Minimal smoke PDB: poly-alanine alpha-helix of 5 residues baked into the
# image under /workspace/model/fixtures/smoke.pdb.  The harness writes this
# file at startup if the path does not yet exist so the service can be
# validated without any external data mount.
# The service only reads PDB files under AIM_INPUT_PATH or this fixture directory.
FIXTURE_DIR = Path(os.environ.get("PROTEINMPNN_FIXTURE_DIR", "/workspace/model/fixtures"))
SMOKE_PDB_PATH = str(FIXTURE_DIR / "smoke.pdb")

# A minimal 5-residue poly-alanine PDB (generated coordinates, HETATM-free).
# Used for smoke and benchmark runs — small enough for smoke and benchmark runs.
_SMOKE_PDB_CONTENT = """\
ATOM      1  N   ALA A   1       1.000   0.000   0.000  1.00  0.00           N
ATOM      2  CA  ALA A   1       2.457   0.000   0.000  1.00  0.00           C
ATOM      3  C   ALA A   1       2.970   1.418   0.000  1.00  0.00           C
ATOM      4  O   ALA A   1       2.198   2.341   0.000  1.00  0.00           O
ATOM      5  CB  ALA A   1       2.953  -0.776   1.209  1.00  0.00           C
ATOM      6  N   ALA A   2       4.293   1.582   0.000  1.00  0.00           N
ATOM      7  CA  ALA A   2       4.905   2.893   0.000  1.00  0.00           C
ATOM      8  C   ALA A   2       6.420   2.820   0.000  1.00  0.00           C
ATOM      9  O   ALA A   2       7.073   3.875   0.000  1.00  0.00           O
ATOM     10  CB  ALA A   2       4.475   3.648   1.250  1.00  0.00           C
ATOM     11  N   ALA A   3       7.028   1.640   0.000  1.00  0.00           N
ATOM     12  CA  ALA A   3       8.470   1.487   0.000  1.00  0.00           C
ATOM     13  C   ALA A   3       9.066   2.841   0.000  1.00  0.00           C
ATOM     14  O   ALA A   3       8.313   3.826   0.000  1.00  0.00           O
ATOM     15  CB  ALA A   3       8.824   0.745  -1.280  1.00  0.00           C
ATOM     16  N   ALA A   4      10.387   2.883   0.000  1.00  0.00           N
ATOM     17  CA  ALA A   4      11.081   4.136   0.000  1.00  0.00           C
ATOM     18  C   ALA A   4      12.568   3.938   0.000  1.00  0.00           C
ATOM     19  O   ALA A   4      13.208   4.992   0.000  1.00  0.00           O
ATOM     20  CB  ALA A   4      10.741   4.923  -1.260  1.00  0.00           C
ATOM     21  N   ALA A   5      13.205   2.776   0.000  1.00  0.00           N
ATOM     22  CA  ALA A   5      14.618   2.491   0.000  1.00  0.00           C
ATOM     23  C   ALA A   5      15.263   3.831   0.000  1.00  0.00           C
ATOM     24  O   ALA A   5      14.468   4.763   0.000  1.00  0.00           O
ATOM     25  CB  ALA A   5      14.918   1.727  -1.278  1.00  0.00           C
END
"""


def _ensure_smoke_pdb() -> str:
    """Write the smoke PDB file if not already present; return its path."""
    path = Path(SMOKE_PDB_PATH)
    path.parent.mkdir(parents=True, exist_ok=True)
    if not path.exists():
        path.write_text(_SMOKE_PDB_CONTENT)
    return str(path)


def _post_inference(
    service_url: str,
    pdb_path: str,
    num_seq_per_target: int = 4,
    sampling_temp: float = 0.1,
    *,
    timeout_seconds: int = 300,
) -> tuple[dict[str, Any], float]:
    """POST a /v1/inference request; return (response_json, elapsed_seconds)."""
    # BentoML nests the single pydantic parameter ``request`` under its own name.
    payload = json.dumps(
        {
            "request": {
                "pdb_path": pdb_path,
                "num_seq_per_target": num_seq_per_target,
                "sampling_temp": sampling_temp,
            }
        }
    ).encode("utf-8")
    req = Request(
        f"{service_url}{INFERENCE_ENDPOINT}",
        data=payload,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    t0 = time.monotonic()
    with urlopen(req, timeout=timeout_seconds) as resp:
        elapsed = time.monotonic() - t0
        if resp.status != 200:
            raise RuntimeError(f"/v1/inference returned HTTP {resp.status}")
        data = json.loads(resp.read().decode("utf-8"))
    return data, elapsed


class ProteinMPNNHarness(ModelHarness):
    """Harness for the ProteinMPNN protein sequence design model (BentoML)."""

    ENGINE = "bentoml"

    CHECKS: list[CheckInfo] = [
        CheckInfo(
            "bentoml_health",
            CheckResultType.PASS_FAIL,
            CheckScope.OFFLINE,  # needs a live service; RUNTIME-scope checks must not
            "GET /health (forwards to BentoML /healthz) returns 200",
        ),
        CheckInfo(
            "inference_smoke",
            CheckResultType.PASS_FAIL,
            CheckScope.OFFLINE,
            "Single-structure prediction returns non-empty sequences",
        ),
        CheckInfo(
            "sequence_count",
            CheckResultType.PASS_FAIL,
            CheckScope.OFFLINE,
            "Prediction returns the requested number of sequences",
        ),
        CheckInfo(
            "throughput",
            CheckResultType.SCORE,
            CheckScope.OFFLINE,
            "Structures designed per minute = 60 / mean latency (serial, smoke PDB, 4 sequences each; resident warmed-up model, so it covers in-process design time plus the HTTP round trip)",
        ),
        CheckInfo(
            "aa_per_second",
            CheckResultType.SCORE,
            CheckScope.OFFLINE,
            "Amino acids designed per second = total residues / total time (ratio of sums) over four synthetic backbone fixtures (not real PDB entries)",
        ),
        CheckInfo(
            "score_finite",
            CheckResultType.PASS_FAIL,
            CheckScope.OFFLINE,
            "All returned scores are finite floats, one per returned sequence (evaluate also requires every sequence to be a non-empty design over the protein alphabet, X for an unknown residue and / between chains)",
        ),
    ]

    def list_checks(self) -> list[CheckInfo]:
        return list(self.CHECKS)

    def health_check(self, service_url: str, timeout_seconds: int = 60) -> bool:
        deadline = time.monotonic() + timeout_seconds
        while time.monotonic() < deadline:
            for endpoint in (HEALTH_ENDPOINT, BENTOML_HEALTHZ_ENDPOINT):
                try:
                    with urlopen(Request(f"{service_url}{endpoint}"), timeout=5) as resp:
                        if resp.status == 200:
                            return True
                except (HTTPError, URLError, OSError) as exc:
                    # Not ready yet (connection refused, 503 while starting): retry until the deadline.
                    logger.debug("Health probe %s not ready: %s", endpoint, exc)
            time.sleep(2.0)
        return False

    def validate(self, config: HarnessConfig) -> HarnessResult:
        service_url = config.resolve_service_url()
        checks: list[CheckResult] = []
        health_ok = False

        # Every check here needs the service; a RUNTIME-only run must not probe it.
        if CheckScope.OFFLINE in config.check_scopes:
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

        if health_ok:
            pdb_path = _ensure_smoke_pdb()
            request_timeout = int(config.get("inference_timeout_seconds", config.timeout_seconds))
            num_seq = 4

            try:
                response, elapsed = _post_inference(
                    service_url,
                    pdb_path,
                    num_seq_per_target=num_seq,
                    timeout_seconds=request_timeout,
                )
                sequences = response.get("sequences", [])
                scores = response.get("scores", [])

                smoke_ok = len(sequences) > 0
                checks.append(
                    CheckResult(
                        name="inference_smoke",
                        result_type=CheckResultType.PASS_FAIL,
                        success=smoke_ok,
                        value=smoke_ok,
                        detail=(
                            f"elapsed={elapsed:.1f}s, sequences={len(sequences)}, " f"scores={len(scores)}"
                            if smoke_ok
                            else f"No sequences returned after {elapsed:.1f}s"
                        ),
                    )
                )

                count_ok = len(sequences) >= num_seq
                checks.append(
                    CheckResult(
                        name="sequence_count",
                        result_type=CheckResultType.PASS_FAIL,
                        success=count_ok,
                        value=count_ok,
                        detail=f"requested={num_seq}, returned={len(sequences)}",
                    )
                )

                scores_finite = bool(scores) and len(scores) == len(sequences) and all(math.isfinite(s) for s in scores)
                checks.append(
                    CheckResult(
                        name="score_finite",
                        result_type=CheckResultType.PASS_FAIL,
                        success=scores_finite,
                        value=scores_finite,
                        detail=f"{len(scores)} scores returned; finite={scores_finite}",
                    )
                )

            except Exception as exc:
                for name in ("inference_smoke", "sequence_count", "score_finite"):
                    checks.append(
                        CheckResult(
                            name=name,
                            result_type=CheckResultType.PASS_FAIL,
                            success=False,
                            value=False,
                            detail=str(exc),
                        )
                    )

        if not checks:
            # Every check needs the live service (OFFLINE scope). With only RUNTIME
            # requested nothing ran: say so explicitly instead of passing vacuously.
            checks.append(
                skipped(
                    "bentoml_health",
                    "no RUNTIME-scope checks exist for this model; the service checks need the OFFLINE scope",
                )
            )

        success = all(c.success for c in checks)
        return HarnessResult(
            success=success,
            summary=f"ProteinMPNN validation {'passed' if success else 'failed'}",
            checks=checks,
        )

    def benchmark(self, config: HarnessConfig) -> HarnessResult:
        if CheckScope.OFFLINE not in config.check_scopes:
            return HarnessResult(
                True,
                "ProteinMPNN benchmark skipped: OFFLINE scope not requested",
                [
                    skipped("throughput", "OFFLINE scope not requested; no service call made"),
                    skipped("aa_per_second", "OFFLINE scope not requested; no service call made"),
                ],
            )
        service_url = config.resolve_service_url()

        if not self.health_check(service_url, timeout_seconds=config.timeout_seconds):
            return HarnessResult(
                success=False,
                summary="BentoML service not reachable for benchmarking",
            )

        # -------------------------------------------------------------------
        # Smoke-PDB throughput
        # -------------------------------------------------------------------
        n_smoke_requests = int(config.get("benchmark_n_requests", 5))
        num_seq = int(config.get("benchmark_num_seq_per_target", 4))
        request_timeout = int(config.get("inference_timeout_seconds", config.timeout_seconds))
        smoke_pdb_path = _ensure_smoke_pdb()

        smoke_latencies: list[float] = []
        any_smoke_failed = False

        for i in range(n_smoke_requests):
            try:
                _, elapsed = _post_inference(
                    service_url,
                    smoke_pdb_path,
                    num_seq_per_target=num_seq,
                    timeout_seconds=request_timeout,
                )
                smoke_latencies.append(elapsed)
            except Exception as exc:
                logger.warning("Smoke benchmark request %d/%d failed: %s", i + 1, n_smoke_requests, exc)
                any_smoke_failed = True

        if not smoke_latencies:
            return HarnessResult(success=False, summary="All smoke benchmark requests failed")

        avg_smoke_latency = statistics.mean(smoke_latencies)
        throughput = 60.0 / avg_smoke_latency if avg_smoke_latency > 0 else 0.0

        # -------------------------------------------------------------------
        # aa/s benchmark over the synthetic backbone fixtures.
        #
        # Protocol:
        #   - 2 warmup calls per PDB (results discarded)
        #   - 5 timed calls per PDB
        #   - aa_per_second = total residues designed / total elapsed time over
        #     all successful timed calls x all PDBs (ratio of sums)
        # -------------------------------------------------------------------
        N_WARMUP = 2
        N_TIMED = 5
        aa_total_residues = 0
        aa_total_time_s = 0.0
        aa_calls_succeeded = 0
        any_aa_failed = False

        for pdb_name, residue_count, pdb_string in BENCHMARK_PDBS:
            FIXTURE_DIR.mkdir(parents=True, exist_ok=True)
            with tempfile.NamedTemporaryFile(
                mode="w",
                suffix=".pdb",
                prefix=f"aim_bench_{pdb_name}_",
                dir=FIXTURE_DIR,
                delete=False,
            ) as tmp_f:
                tmp_f.write(pdb_string)
                tmp_pdb_path = tmp_f.name

            try:
                logger.info(
                    "aa/s benchmark: %s (%d AA) — %d warmup + %d timed calls",
                    pdb_name,
                    residue_count,
                    N_WARMUP,
                    N_TIMED,
                )

                # Warmup
                for w in range(N_WARMUP):
                    try:
                        _post_inference(
                            service_url,
                            tmp_pdb_path,
                            num_seq_per_target=1,
                            sampling_temp=0.1,
                            timeout_seconds=request_timeout,
                        )
                    except Exception as exc:
                        logger.warning(
                            "aa/s warmup %d/%d for %s failed: %s",
                            w + 1,
                            N_WARMUP,
                            pdb_name,
                            exc,
                        )

                # Timed calls
                for t in range(N_TIMED):
                    try:
                        t0 = time.monotonic()
                        _post_inference(
                            service_url,
                            tmp_pdb_path,
                            num_seq_per_target=1,
                            sampling_temp=0.1,
                            timeout_seconds=request_timeout,
                        )
                        elapsed_s = time.monotonic() - t0
                        aa_total_residues += residue_count
                        aa_total_time_s += elapsed_s
                        aa_calls_succeeded += 1
                        aa_per_sec = residue_count / elapsed_s if elapsed_s > 0 else 0.0
                        logger.debug(
                            "aa/s timed call %d/%d for %s: %.3fs → %.1f aa/s",
                            t + 1,
                            N_TIMED,
                            pdb_name,
                            elapsed_s,
                            aa_per_sec,
                        )
                    except Exception as exc:
                        logger.warning(
                            "aa/s timed call %d/%d for %s failed: %s",
                            t + 1,
                            N_TIMED,
                            pdb_name,
                            exc,
                        )
                        any_aa_failed = True
            finally:
                Path(tmp_pdb_path).unlink(missing_ok=True)

        mean_aa_per_sec = aa_total_residues / aa_total_time_s if aa_total_time_s > 0 else 0.0
        aa_calls_total = len(BENCHMARK_PDBS) * N_TIMED

        any_failed = any_smoke_failed or any_aa_failed

        checks = [
            CheckResult(
                name="throughput",
                result_type=CheckResultType.SCORE,
                success=not any_smoke_failed,
                value=round(throughput, 2),
                detail=(
                    f"{len(smoke_latencies)}/{n_smoke_requests} smoke requests succeeded, "
                    f"avg latency={avg_smoke_latency:.2f}s, "
                    f"throughput={throughput:.1f} structures/min"
                ),
            ),
            CheckResult(
                name="aa_per_second",
                result_type=CheckResultType.SCORE,
                success=not any_aa_failed and aa_calls_succeeded > 0,
                value=round(mean_aa_per_sec, 2),
                detail=(
                    f"{aa_calls_succeeded}/{aa_calls_total} aa/s timed calls succeeded "
                    f"across {len(BENCHMARK_PDBS)} PDBs "
                    f"({', '.join(f'{n} {c}AA' for n, c, _ in BENCHMARK_PDBS)}); "
                    f"aa/s={mean_aa_per_sec:.2f} ({aa_total_residues} residues / {aa_total_time_s:.2f}s)"
                ),
            ),
        ]

        return HarnessResult(
            success=not any_failed,
            summary=(f"ProteinMPNN benchmark: {throughput:.1f} structures/min " f"| {mean_aa_per_sec:.1f} aa/s"),
            checks=checks,
            metrics={
                "avg_latency_s": round(avg_smoke_latency, 3),
                "throughput_per_min": round(throughput, 2),
                "requests_succeeded": len(smoke_latencies),
                "requests_total": n_smoke_requests,
                "aa_per_second": round(mean_aa_per_sec, 2),
                "aa_calls_succeeded": aa_calls_succeeded,
                "aa_calls_total": aa_calls_total,
                "aa_total_residues": aa_total_residues,
                "aa_total_time_s": round(aa_total_time_s, 3),
            },
        )

    def evaluate(self, config: HarnessConfig) -> HarnessResult:
        """Run a smoke prediction and verify the output is chemically reasonable.

        ProteinMPNN has no ground-truth reference sequence available without
        a real complex PDB.  evaluate checks that the returned sequences are
        non-empty, protein-alphabet-only strings with finite associated scores.
        """
        if CheckScope.OFFLINE not in config.check_scopes:
            return HarnessResult(
                True,
                "ProteinMPNN evaluate skipped: OFFLINE scope not requested",
                [skipped("score_finite", "OFFLINE scope not requested; no service call made")],
            )
        service_url = config.resolve_service_url()

        if not self.health_check(service_url, timeout_seconds=config.timeout_seconds):
            return HarnessResult(
                success=False,
                summary="BentoML service not reachable for evaluation",
            )

        pdb_path = _ensure_smoke_pdb()
        request_timeout = int(config.get("inference_timeout_seconds", config.timeout_seconds))
        num_seq = 4

        try:
            response, elapsed = _post_inference(
                service_url,
                pdb_path,
                num_seq_per_target=num_seq,
                timeout_seconds=request_timeout,
            )
        except Exception as exc:
            return HarnessResult(
                success=False,
                summary=f"ProteinMPNN evaluate: prediction failed: {exc}",
                checks=[
                    CheckResult(
                        name="score_finite",
                        result_type=CheckResultType.PASS_FAIL,
                        success=False,
                        value=False,
                        detail=str(exc),
                    )
                ],
            )

        sequences = response.get("sequences", [])
        scores = response.get("scores", [])
        # A malformed response (null score, non-string sequence, non-list) is a failed check,
        # not an exception that would skip evaluate_results.json.
        well_typed = (
            isinstance(sequences, list)
            and isinstance(scores, list)
            and all(isinstance(s, str) for s in sequences)
            and all(isinstance(x, (int, float)) and not isinstance(x, bool) for x in scores)
        )
        if not well_typed:
            sequences, scores = [], []

        # X marks an unknown residue; "/" separates chains in multi-chain designs.
        valid_seqs = [s for s in sequences if _is_valid_design(s)]
        all_valid = len(valid_seqs) == len(sequences) and len(sequences) > 0
        scores_finite = len(scores) == len(sequences) and bool(scores) and all(math.isfinite(s) for s in scores)
        success = all_valid and scores_finite

        checks = [
            CheckResult(
                name="score_finite",
                result_type=CheckResultType.PASS_FAIL,
                success=success,
                value=success,
                detail=(
                    f"sequences={len(sequences)}, valid_protein_alphabet={len(valid_seqs)}, "
                    f"scores_finite={scores_finite}, elapsed={elapsed:.1f}s"
                ),
            )
        ]

        return HarnessResult(
            success=success,
            summary=f"ProteinMPNN evaluate {'passed' if success else 'failed'}",
            checks=checks,
            metrics={
                "sequences_returned": len(sequences),
                "valid_sequences": len(valid_seqs),
                "scores_finite": scores_finite,
            },
        )
