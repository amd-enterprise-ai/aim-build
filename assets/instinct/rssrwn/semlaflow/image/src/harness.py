# Copyright © Advanced Micro Devices, Inc., or its affiliates.
#
# SPDX-License-Identifier: MIT

"""
SemlaFlowHarness — model-specific harness for the SemlaFlow small-molecule
generation model (the service returns SMILES and 3D mol blocks).

The service loads the checkpoint from AIM_CACHE_PATH once at startup and keeps the
model resident (it is warmed up before the port opens), so the harness measures
generation on the loaded model, not a per-request model load.
Benchmark and evaluation data (GEOM-Drugs .smol files) are read from
AIM_INPUT_PATH. The harness sends no dataset_split, so the service default
("test") applies and the service reads test.smol from that directory.

This file lives at ``assets/instinct/rssrwn/semlaflow/image/src/harness.py``
and is copied to ``/workspace/model/src/harness.py`` by the specialized
Dockerfile. aim-runtime discovers it automatically via the well-known path.
"""

from __future__ import annotations

import json
import logging
import os
import pathlib
import re
import statistics
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

# Default number of timed inference calls during benchmarking.
_DEFAULT_N_TIMED = 5
_DEFAULT_N_WARMUP = 1

# Default number of molecules to generate per /v1/inference call. Large values need
# more memory, so the default is kept modest.
_DEFAULT_N_MOLECULES = 100

# ODE integration steps for the benchmark /v1/inference call.
_DEFAULT_INTEGRATION_STEPS = 100

# Molecules per evaluate() request (one call; the service samples the conditioning molecules with replacement).
_DEFAULT_EVAL_N_MOLECULES = 500

# Small molecule count for the smoke /v1/inference during validate().
_SMOKE_N_MOLECULES = 5
_SMOKE_INTEGRATION_STEPS = 20

# /health forwards to BentoML's own /healthz, so it is the only endpoint polled.
HEALTH_ENDPOINT = "/health"
INFERENCE_ENDPOINT = "/v1/inference"


# Characters that can occur in a SMILES string. The service returns canonical SMILES; this
# catches non-SMILES payloads (binary, tracebacks) without needing RDKit in the harness env.
_SMILES_RE = re.compile(r"[A-Za-z0-9@+\-\[\]()=#$:/\\.%*]+")


def _count_smiles_like(smiles: object) -> int:
    """Number of entries of ``smiles`` that are non-empty strings made only of SMILES characters."""
    if not isinstance(smiles, list):
        return 0
    return sum(1 for s in smiles if isinstance(s, str) and _SMILES_RE.fullmatch(s) is not None)


def _smiles_problem(result: dict) -> str | None:
    """Return a description if the response does not hold n_molecules valid-looking SMILES, else None."""
    smiles = result.get("smiles")
    n = result.get("n_molecules", 0)
    if not isinstance(smiles, list) or n != len(smiles):
        return f"n_molecules={n!r} does not match the smiles list"
    good = _count_smiles_like(smiles)
    if good != len(smiles):
        return f"{len(smiles) - good} of {len(smiles)} entries are not SMILES strings"
    return None


def _probe_data_file() -> tuple[bool, str]:
    """Check the input file the service reads; return (present, detail)."""
    try:
        data_path = _resolve_data_path()
    except EnvironmentError as exc:
        return False, str(exc)
    if data_path.is_file():
        return True, f"Data file found: {data_path}"
    return False, f"Data file not found: {data_path}"


def _resolve_data_path() -> pathlib.Path:
    """Return the path of the file the harness probes in AIM_INPUT_PATH.

    The harness probes $AIM_INPUT_PATH/test.smol, the file the service reads for
    the default dataset_split ("test"); the harness sends no dataset_split. The
    operator mounts the dataset directory at AIM_INPUT_PATH; that directory is
    sent as data_path.
    """
    input_path = os.environ.get("AIM_INPUT_PATH", "")
    if not input_path:
        raise EnvironmentError("AIM_INPUT_PATH is not set. Mount your benchmark data directory and set AIM_INPUT_PATH.")
    return pathlib.Path(input_path) / "test.smol"


def _post_inference(service_url: str, data_dir: str, n_molecules: int, integration_steps: int, timeout: int) -> dict:
    """POST to the BentoML /v1/inference endpoint and return the parsed response."""
    payload = json.dumps(
        {
            "data_path": data_dir,
            "n_molecules": n_molecules,
            "integration_steps": integration_steps,
        }
    ).encode()
    req = Request(
        f"{service_url}{INFERENCE_ENDPOINT}",
        data=payload,
        headers={"Content-Type": "application/json"},
    )
    with urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read().decode())


class SemlaFlowHarness(ModelHarness):
    """Harness for SemlaFlow small-molecule generation.

    Benchmark strategy
    ------------------
    Calls the BentoML /v1/inference endpoint with data from AIM_INPUT_PATH
    The service loads the checkpoint from AIM_CACHE_PATH once, at startup, and
    keeps the model resident — the harness does not interact with the weight path.

    Reports mol/min = mean returned molecules per call * 60 / mean latency over
    benchmark_n_timed timed calls after benchmark_n_warmup warmup calls. The
    numerator is the number of molecules the service RETURNED (valid SMILES),
    not the requested count, because the service drops molecules that fail
    sanitisation. The latency is that of the whole HTTP call on the resident
    model (no per-request model load or process start; see API.md 4.3).

    evaluate() requests one batch and reports ``validity`` (returned / requested:
    the share of generated molecules that passed RDKit sanitisation) and
    ``uniqueness`` (distinct canonical SMILES / returned), the two metrics of
    upstream's evaluation that need no training set or 3D analysis.
    """

    CHECKS: list[CheckInfo] = [
        CheckInfo(
            "health",
            CheckResultType.PASS_FAIL,
            CheckScope.OFFLINE,
            "GET /health (the service forwards it to BentoML /healthz) returns 200",
        ),
        CheckInfo(
            "input_data_check",
            CheckResultType.PASS_FAIL,
            CheckScope.RUNTIME,
            "AIM_INPUT_PATH/test.smol (the file the service reads for the default split) is a readable file",
        ),
        CheckInfo(
            "smoke_inference",
            CheckResultType.PASS_FAIL,
            CheckScope.OFFLINE,
            "Small /v1/inference call succeeds and returns n_molecules SMILES strings",
        ),
        CheckInfo(
            "throughput_mol_per_min",
            CheckResultType.SCORE,
            CheckScope.OFFLINE,
            "End-to-end generation throughput (mol/min) of /v1/inference calls on the resident model",
        ),
        CheckInfo(
            "validity",
            CheckResultType.SCORE,
            CheckScope.OFFLINE,
            "evaluate(): fraction of requested molecules returned (passed RDKit sanitisation)",
        ),
        CheckInfo(
            "uniqueness",
            CheckResultType.SCORE,
            CheckScope.OFFLINE,
            "evaluate(): fraction of returned molecules with a distinct canonical SMILES",
        ),
    ]

    def list_checks(self) -> list[CheckInfo]:
        return list(self.CHECKS)

    # ------------------------------------------------------------------ #
    # health_check — probe GET /health
    # ------------------------------------------------------------------ #

    def health_check(self, service_url: str, timeout_seconds: int = 60) -> bool:
        deadline = time.monotonic() + timeout_seconds
        while time.monotonic() < deadline:
            try:
                with urlopen(Request(f"{service_url}{HEALTH_ENDPOINT}"), timeout=5) as resp:
                    if resp.status == 200:
                        return True
            except (HTTPError, URLError, OSError) as exc:
                # Service not ready yet; retry until the deadline.
                logger.debug("%s not ready: %s", HEALTH_ENDPOINT, exc)
            time.sleep(1.0)
        return False

    # ------------------------------------------------------------------ #
    # validate
    # ------------------------------------------------------------------ #

    def validate(self, config: HarnessConfig) -> HarnessResult:
        url = config.resolve_service_url()
        checks: list[CheckResult] = []
        offline = CheckScope.OFFLINE in config.check_scopes
        runtime = CheckScope.RUNTIME in config.check_scopes

        # 1. Service health (offline scope: calls the service)
        healthy = False
        if offline:
            healthy = self.health_check(url, timeout_seconds=config.timeout_seconds)
            checks.append(
                CheckResult(
                    name="health",
                    result_type=CheckResultType.PASS_FAIL,
                    success=healthy,
                    value=healthy,
                    detail="GET /health returned 200" if healthy else f"Service unreachable at {url}/health",
                )
            )
        else:
            checks.append(skipped("health", "OFFLINE scope not requested; the service was not contacted"))

        # 2. Input data mount (runtime scope; local file check, no network)
        data_ok = False
        if runtime:
            data_ok, data_detail = _probe_data_file()
            checks.append(
                CheckResult(
                    name="input_data_check",
                    result_type=CheckResultType.PASS_FAIL,
                    success=data_ok,
                    value=data_ok,
                    detail=data_detail,
                )
            )
        else:
            checks.append(skipped("input_data_check", "RUNTIME scope not requested; the input mount was not checked"))

        # 3. Smoke /v1/inference (offline scope only: requires a live service). It needs the
        # input file, so it checks it itself and reports a missing one in its own result.
        if not offline:
            checks.append(skipped("smoke_inference", "OFFLINE scope not requested; the service was not contacted"))
        elif not healthy:
            checks.append(skipped("smoke_inference", "not run because the health check failed"))
        else:
            smoke_ok = False
            smoke_detail = ""
            present, probe_detail = (data_ok, "") if runtime else _probe_data_file()
            if not present:
                smoke_detail = (
                    f"Smoke /v1/inference not run: input file unavailable ({probe_detail or 'see input_data_check'})"
                )
            else:
                data_path = _resolve_data_path()
                try:
                    result = _post_inference(
                        service_url=url,
                        data_dir=str(data_path.parent),
                        n_molecules=_SMOKE_N_MOLECULES,
                        integration_steps=_SMOKE_INTEGRATION_STEPS,
                        timeout=config.timeout_seconds,
                    )
                    n = result.get("n_molecules", 0)
                    problem = _smiles_problem(result)
                    smoke_ok = n > 0 and problem is None
                    if smoke_ok:
                        smoke_detail = f"Smoke inference returned {n} SMILES strings in {result.get('elapsed_s', '?')}s"
                    else:
                        smoke_detail = f"Unexpected response: {problem or 'no molecules'} (n_molecules={n})"
                except Exception as exc:
                    smoke_detail = f"Smoke /v1/inference failed: {exc}"

            checks.append(
                CheckResult(
                    name="smoke_inference",
                    result_type=CheckResultType.PASS_FAIL,
                    success=smoke_ok,
                    value=smoke_ok,
                    detail=smoke_detail,
                )
            )

        success = all(c.success for c in checks)
        return HarnessResult(
            success=success,
            summary=f"SemlaFlow validation: {sum(c.success for c in checks)}/{len(checks)} checks passed",
            checks=checks,
        )

    # ------------------------------------------------------------------ #
    # benchmark
    # ------------------------------------------------------------------ #

    def benchmark(self, config: HarnessConfig) -> HarnessResult:
        """Measure molecule generation throughput via the BentoML /v1/inference endpoint.

        Data directory is AIM_INPUT_PATH, the directory of the AIM_INPUT_PATH/test.smol
        file the harness probes; no dataset_split is sent, so the service default
        (test) reads test.smol from it.
        The service checks at startup that the checkpoint exists in AIM_CACHE_PATH.

        Reports end-to-end mol/min = n_molecules * 60 / mean latency of the whole
        HTTP call on the resident model.
        """
        if CheckScope.OFFLINE not in config.check_scopes:
            return HarnessResult(
                success=True,
                summary="SemlaFlow benchmark skipped: OFFLINE scope not requested",
                checks=[
                    skipped("throughput_mol_per_min", "OFFLINE scope not requested; the service was not contacted")
                ],
            )
        url = config.resolve_service_url()

        if not self.health_check(url, timeout_seconds=config.timeout_seconds):
            return HarnessResult(
                success=False,
                summary=f"SemlaFlow benchmark cannot run — service unreachable at {url}",
                checks=[
                    CheckResult(
                        name="throughput_mol_per_min",
                        result_type=CheckResultType.SCORE,
                        success=False,
                        value=0.0,
                        detail=f"Service unreachable at {url}{HEALTH_ENDPOINT}",
                    )
                ],
            )

        n_timed = config.get("benchmark_n_timed", _DEFAULT_N_TIMED)
        n_warmup = config.get("benchmark_n_warmup", _DEFAULT_N_WARMUP)
        n_molecules = config.get("benchmark_n_molecules", _DEFAULT_N_MOLECULES)
        integration_steps = config.get("benchmark_n_steps", _DEFAULT_INTEGRATION_STEPS)

        try:
            data_path = _resolve_data_path()
        except EnvironmentError as exc:
            return HarnessResult(
                success=False,
                summary=f"SemlaFlow benchmark cannot run: {exc}",
                checks=[
                    CheckResult(
                        name="throughput_mol_per_min",
                        result_type=CheckResultType.SCORE,
                        success=False,
                        value=0.0,
                        detail=str(exc),
                    )
                ],
            )

        if not data_path.is_file():
            return HarnessResult(
                success=False,
                summary=f"SemlaFlow benchmark cannot run: data file not found at {data_path}",
                checks=[
                    CheckResult(
                        name="throughput_mol_per_min",
                        result_type=CheckResultType.SCORE,
                        success=False,
                        value=0.0,
                        detail=f"Expected data file at {data_path}. Mount your benchmark data directory to AIM_INPUT_PATH.",
                    )
                ],
            )

        latencies: list[float] = []
        returned: list[int] = []
        service_elapsed: list[float] = []
        for i in range(n_warmup + n_timed):
            try:
                t0 = time.monotonic()
                result = _post_inference(
                    service_url=url,
                    data_dir=str(data_path.parent),
                    n_molecules=n_molecules,
                    integration_steps=integration_steps,
                    timeout=config.timeout_seconds,
                )
                elapsed = time.monotonic() - t0
            except Exception as exc:
                return HarnessResult(
                    success=False,
                    summary=f"SemlaFlow /v1/inference failed on call {i + 1}: {exc}",
                    checks=[
                        CheckResult(
                            name="throughput_mol_per_min",
                            result_type=CheckResultType.SCORE,
                            success=False,
                            value=0.0,
                            detail=str(exc),
                        )
                    ],
                )

            if result.get("n_molecules", 0) == 0:
                return HarnessResult(
                    success=False,
                    summary=f"SemlaFlow /v1/inference returned 0 molecules on call {i + 1}",
                    checks=[
                        CheckResult(
                            name="throughput_mol_per_min",
                            result_type=CheckResultType.SCORE,
                            success=False,
                            value=0.0,
                            detail=f"Unexpected response: {result}",
                        )
                    ],
                )

            problem = _smiles_problem(result)
            if problem is not None:
                return HarnessResult(
                    success=False,
                    summary=f"SemlaFlow /v1/inference returned invalid SMILES on call {i + 1}",
                    checks=[
                        CheckResult(
                            name="throughput_mol_per_min",
                            result_type=CheckResultType.SCORE,
                            success=False,
                            value=0.0,
                            detail=problem,
                        )
                    ],
                )

            if i >= n_warmup:
                latencies.append(elapsed)
                returned.append(len(result["smiles"]))
                service_elapsed.append(float(result.get("elapsed_s", 0.0) or 0.0))

        mean_lat = statistics.mean(latencies)
        mean_returned = statistics.mean(returned)
        mol_per_min = mean_returned * 60 / mean_lat if mean_lat > 0 else 0.0

        return HarnessResult(
            success=True,
            summary=f"SemlaFlow benchmark: {mol_per_min:.2f} mol/min ({mean_returned:g} of {n_molecules} molecules returned/call, {integration_steps} steps)",
            checks=[
                CheckResult(
                    name="throughput_mol_per_min",
                    result_type=CheckResultType.SCORE,
                    success=True,
                    value=round(mol_per_min, 2),
                    detail=(
                        f"mean latency {mean_lat:.1f}s over {n_timed} timed calls "
                        f"({mean_returned:g} of {n_molecules} requested molecules returned per call, {integration_steps} steps, data={data_path}) [mol/min, end-to-end HTTP call on the resident model]"
                    ),
                )
            ],
            metrics={
                "mol_per_min": round(mol_per_min, 2),
                "mean_latency_s": round(mean_lat, 3),
                "mean_service_elapsed_s": round(statistics.mean(service_elapsed), 3),
                "molecules_requested": n_molecules,
                "molecules_returned": round(mean_returned, 2),
                "integration_steps": integration_steps,
                "n_timed_calls": n_timed,
            },
        )

    # ------------------------------------------------------------------ #
    # evaluate
    # ------------------------------------------------------------------ #

    def evaluate(self, config: HarnessConfig) -> HarnessResult:
        """Generate one batch and report ``validity`` and ``uniqueness`` (both SCORE, 0..1).

        * validity = returned / requested. The service returns only molecules that passed RDKit
          sanitisation (upstream's ``Validity`` metric counts the same molecules).
        * uniqueness = distinct SMILES / returned. The service returns canonical SMILES, so
          string equality is molecule equality (upstream's ``Uniqueness`` metric).

        The run succeeds when the call returns well-formed SMILES and both scores are above the
        optional ``evaluate_min_validity`` / ``evaluate_min_uniqueness`` profile values (default 0,
        i.e. report only, and at least one molecule must come back).
        Novelty, energy and stability metrics of the upstream evaluation are not computed here.
        """
        names = ("validity", "uniqueness")

        def failed(summary: str, detail: str) -> HarnessResult:
            return HarnessResult(
                success=False,
                summary=summary,
                checks=[
                    CheckResult(name=n, result_type=CheckResultType.SCORE, success=False, value=0.0, detail=detail)
                    for n in names
                ],
            )

        if CheckScope.OFFLINE not in config.check_scopes:
            return HarnessResult(
                success=True,
                summary="SemlaFlow evaluate skipped: OFFLINE scope not requested",
                checks=[skipped(n, "OFFLINE scope not requested; the service was not contacted") for n in names],
            )
        url = config.resolve_service_url()
        if not self.health_check(url, timeout_seconds=config.timeout_seconds):
            return failed(
                f"SemlaFlow evaluate cannot run — service unreachable at {url}",
                f"Service unreachable at {url}{HEALTH_ENDPOINT}",
            )
        try:
            data_path = _resolve_data_path()
        except EnvironmentError as exc:
            return failed(f"SemlaFlow evaluate cannot run: {exc}", str(exc))
        if not data_path.is_file():
            return failed(
                f"SemlaFlow evaluate cannot run: data file not found at {data_path}",
                f"Expected data file at {data_path}. Mount your data directory to AIM_INPUT_PATH.",
            )

        n_requested = int(config.get("evaluate_n_molecules", _DEFAULT_EVAL_N_MOLECULES))
        integration_steps = int(config.get("evaluate_n_steps", _DEFAULT_INTEGRATION_STEPS))
        min_validity = float(config.get("evaluate_min_validity", 0.0))
        min_uniqueness = float(config.get("evaluate_min_uniqueness", 0.0))
        try:
            result = _post_inference(
                service_url=url,
                data_dir=str(data_path.parent),
                n_molecules=n_requested,
                integration_steps=integration_steps,
                timeout=config.timeout_seconds,
            )
        except Exception as exc:
            return failed(f"SemlaFlow /v1/inference failed during evaluate: {exc}", str(exc))
        problem = _smiles_problem(result)
        smiles = result.get("smiles")
        if problem is not None or not smiles:
            return failed(
                "SemlaFlow evaluate: unusable /v1/inference response",
                problem or "the service returned no molecules",
            )

        validity = len(smiles) / n_requested
        uniqueness = len(set(smiles)) / len(smiles)
        ok_validity = validity > 0 and validity >= min_validity
        ok_uniqueness = uniqueness > 0 and uniqueness >= min_uniqueness
        basis = f"{len(smiles)} of {n_requested} requested molecules returned, {integration_steps} steps"
        return HarnessResult(
            success=ok_validity and ok_uniqueness,
            summary=f"SemlaFlow evaluate: validity {validity:.4f}, uniqueness {uniqueness:.4f} ({basis})",
            checks=[
                CheckResult(
                    name="validity",
                    result_type=CheckResultType.SCORE,
                    success=ok_validity,
                    value=round(validity, 4),
                    detail=f"{basis}; minimum {min_validity:g}",
                ),
                CheckResult(
                    name="uniqueness",
                    result_type=CheckResultType.SCORE,
                    success=ok_uniqueness,
                    value=round(uniqueness, 4),
                    detail=f"{len(set(smiles))} distinct of {len(smiles)} returned; minimum {min_uniqueness:g}",
                ),
            ],
            metrics={
                "validity": round(validity, 4),
                "uniqueness": round(uniqueness, 4),
                "molecules_requested": n_requested,
                "molecules_returned": len(smiles),
                "integration_steps": integration_steps,
            },
        )
