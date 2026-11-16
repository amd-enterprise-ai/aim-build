# Copyright © Advanced Micro Devices, Inc., or its affiliates.
#
# SPDX-License-Identifier: MIT

"""
SaProtHarness — model-specific harness for the SaProt 650M AF2 structure-aware
protein language model served via BentoML.

This file lives at
``assets/instinct/westlake-repl/saprot-650m-af2/image/src/harness.py``
and is copied to ``/workspace/model/src/harness.py`` by the specialized
Dockerfile. aim-runtime discovers it automatically via the well-known path.

Because ``/workspace/model/src/`` is on ``sys.path`` (via a ``.pth`` file
installed by the specialized Dockerfile), this module can import sibling
modules in the same directory (e.g. ``from service import SaProtEmbeddingService``)
as well as aim-runtime's public API (``from aim_runtime.harness import ...``).

SaProt sequences interleave amino acid (AA) tokens and 3Di structural tokens
(one per residue, derived from Foldseek or AlphaFold2).  The '#' character
is the unknown-structure placeholder used when no 3D prediction is available.

Benchmark metric ``sequences_per_sec`` = 1 / mean latency of sequential
requests (concurrency 1), measured per length bucket and over all timed
requests. Latency includes the HTTP round trip and tokenization and excludes
warm-up requests. Each check's ``success`` reflects the failures it measured
(any failed timed request makes it unsuccessful); the run succeeds only when
``failed_requests == 0``. A timed request counts as successful only if the HTTP
status is 200 and the body matches the documented embedding response (1280
floats, ``embedding_dim`` 1280, a ``model`` string); otherwise the benchmark
fails and reports no throughput figure.
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
    skipped,
)

logger = logging.getLogger(__name__)

HEALTH_ENDPOINT = "/health"
INFERENCE_ENDPOINT = "/v1/inference"
MODELS_ENDPOINT = "/v1/models"

# Per-request timeout for embedding calls when the caller does not pass one;
# harness entry points forward ``HarnessConfig.timeout_seconds`` instead.
DEFAULT_REQUEST_TIMEOUT_SECONDS = 60

# Residue counts for the length-sweep benchmark.
# The longest bucket is the service limit of 1022 residues (1024 tokens minus BOS/EOS);
# longer inputs are rejected with HTTP 400.
MAX_RESIDUES = 1022
BENCHMARK_LENGTHS = [100, 250, 500, 1022]

# The expected dtype and accelerator count come from the active profile
# (``config.profile["metadata"]``: ``precision`` and ``accelerator_count``), not from
# constants here. The service reports its dtype on GET /v1/models (``fp16``); both
# sides are normalised to torch-style names before they are compared.
_DTYPE_ALIASES = {"fp16": "float16", "half": "float16", "bf16": "bfloat16", "fp32": "float32"}

# 10-residue peptide in SaProt format (AA interleaved with '#' unknown-structure tokens).
# Used as the fast smoke-test sequence.
SMOKE_SEQUENCE = "M#A#K#T#A#Y#I#A#K#Q#"

# Ubiquitin (76 residues) in SaProt format with unknown-structure placeholders.
# A well-characterised, stable protein used for embedding quality checks.
EVAL_SEQUENCE = (
    "M#Q#I#F#V#K#T#L#T#G#K#T#I#T#L#E#V#E#P#S#D#T#I#E#N#V#K#A#K#I#Q#D#K#E#G#I#P#P#D#Q#Q#R#"
    "L#I#F#A#G#K#Q#L#E#D#G#R#T#L#S#D#Y#N#I#Q#K#E#S#T#L#H#L#V#L#R#L#R#G#G#"
)

# Real benchmark sequences in SaProt M#X# format, keyed by target residue count.
# Each AA is followed by '#' (unknown-structure token), the same convention as
# EVAL_SEQUENCE.  Using real amino-acid distributions avoids tokenizer artefacts
# from degenerate "M#M#M#…" tiles.
#
# Sources:
#   100 AA: human ubiquitin (76 AA) + E. coli TrxA N-terminal tail
#   250 AA: E. coli thioredoxin A (TrxA, 109 AA) + human DHFR N-terminal part
#   500 AA: human DHFR (159 AA) + E. coli AK (175 AA) + HSA fragment (first 166 AA)
#   1022 AA: HSA fragment (549 AA) + ubiquitin/TrxA/DHFR/AK filler, cut at 1022


def _to_saprot(seq: str) -> str:
    """Encode a plain AA sequence into SaProt M#A# format using '#' placeholders."""
    return "".join(f"{aa}#" for aa in seq)


# Plain AA sequences for each benchmark length bucket:
_UBIQUITIN = "MQIFVKTLTGKTITLEVEPSDTIENVKAKIQDKEGIPPDQQRLIFAGKQLEDGRTLSDYNIQKESTLHLVLRLRGG"  # 76 AA
_TRXA = "MSDKIIHLTDDSFDTDVLKADGAILVDFWAEWCGPCKMIAPILDEIADEYQGKLTVAKLNIDQNPGTAPKYGIRGIPTLLLFKNGEVAATKVGALSKGQLKEFLDANLA"  # 109 AA
_DHFR = "MVGSLNCIVAVSQNMGIGKNGDLPWPPLRNEFRYFQRMTTTSSVEGKQNLVIMGKKTWFSIPEKNRPLKGRINLVLSRELKEPPQGAHFLSRSLDDALKLTEQPELANKAVDMVWIVGGSSVYKEAMNHPGHLKLFVTRIMQDFESDTFFPEIDLGKYK"  # 159 AA
_AK = "MRIILLGAPGAGKGTQAKFIEEKYGIPQISTGDMLRAAVKSGSELGKQAKDIMDAGKLVTDELVIALVKERIAQEDCRNGFLLDGFPRTIPQADAMKEAGINVDYVLEFDVPDELIVDRIVGRRVHAPSGRVYHVKFNPPKVEGKDDVTGEELTTRKDDQEETVRKRLVEYHQMT"  # 175 AA
_HSA_DOMAIN = "MKWVTFISLLFLFSSAYSRGVFRRDAHKSEVAHRFKDLGEENFKALVLIAFAQYLQQCPFEDHVKLVNEVTEFAKTCVADESAENCDKSLHTLFGDKLCTVATLRETYGEMADCCAKQEPERNECFLQHKDDNPNLPRLVRPEVDVMCTAFHDNEETFLKKYLYEIARRHPYFYAPELLFFAKRYKAAFTECCQAADKDACLLPKLDELRDEGKASSAKQRLKCASLQKFGERAFKAWAVARLSQRFPKAEFAEVSKLVTDLTKVHTECCHGDLLECADDRADLAKYICDNQDTISSKLKECCDKPLLEKSHCIAEVENDEMPADLPSLAADFVESKDVCKNYAEAKDVFLGMFLYEYARRHPDYSVVLLLRLAKTYETTLEKCCAAADPHECYAKVFDEFKPLVEEPQNLIKQNCELFEQLGEYKFQNALLVRYTKKVPQVSTPTLVEVSRNLGKVGSKCCKHPEAKRMPCAEDYLSVVLNQLCVLHEKTPVSDRVTKCCTESLVNRRPCFSALEVDETYVPKEFNAETFTFHADICTLSEKERCIAS"  # 549 AA

# Build 100 AA: ubiquitin (76) + TrxA N-term tail (24)
_SEQ_100 = (_UBIQUITIN + _TRXA)[:100]
# Build 250 AA: TrxA (109) + DHFR (first 141 of 159)
_SEQ_250 = (_TRXA + _DHFR)[:250]
# Build 500 AA: DHFR (159) + AK (175) + HSA fragment (first 166)
_SEQ_500 = (_DHFR + _AK + _HSA_DOMAIN)[:500]
# Build 1022 AA: HSA fragment (549) + filler, cut at 1022 (the service limit)
_SEQ_1022 = (_HSA_DOMAIN + _UBIQUITIN + _TRXA + _DHFR + _AK + _UBIQUITIN + _TRXA)[:MAX_RESIDUES]

BENCHMARK_SEQUENCES: dict[int, str] = {
    100: _to_saprot(_SEQ_100),
    250: _to_saprot(_SEQ_250),
    500: _to_saprot(_SEQ_500),
    1022: _to_saprot(_SEQ_1022),
}

# Expected embedding dimension for westlake-repl/SaProt_650M_AF2.
EXPECTED_EMBEDDING_DIM = 1280


class SaProtHarness(ModelHarness):
    """Harness for SaProt 650M AF2 structure-aware protein embedding model (BentoML)."""

    ENGINE = "bentoml"

    CHECKS: list[CheckInfo] = [
        CheckInfo(
            "bentoml_health",
            CheckResultType.PASS_FAIL,
            CheckScope.OFFLINE,
            "GET /health returns 200",
        ),
        CheckInfo(
            "embed_smoke",
            CheckResultType.PASS_FAIL,
            CheckScope.OFFLINE,
            "Single-sequence embedding returns vector of correct dimensionality",
        ),
        CheckInfo(
            "sequences_per_sec",
            CheckResultType.SCORE,
            CheckScope.OFFLINE,
            "Sequences embedded per second, averaged across benchmark lengths",
        ),
        CheckInfo(
            "embedding_shape",
            CheckResultType.PASS_FAIL,
            CheckScope.OFFLINE,
            "Eval-sequence embedding has the expected 1280-dimensional shape",
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
            except (HTTPError, URLError, OSError):
                # Service not up yet; retry until the deadline.
                logger.debug("health check %s not ready at %s", HEALTH_ENDPOINT, service_url)
            time.sleep(2.0)
        return False

    def validate(self, config: HarnessConfig) -> HarnessResult:
        service_url = self._service_url(config)
        checks: list[CheckResult] = []

        # Scope gate first: a runtime-only scope must not block on the health poll.
        if CheckScope.OFFLINE not in config.check_scopes:
            return HarnessResult(
                success=True,
                summary="SaProt 650M AF2 validation skipped: no OFFLINE-scope checks requested",
                checks=checks,
            )

        health_ok = self.health_check(service_url, timeout_seconds=config.timeout_seconds)
        checks.append(
            CheckResult(
                name="bentoml_health",
                result_type=CheckResultType.PASS_FAIL,
                success=health_ok,
                value=health_ok,
                detail="" if health_ok else f"BentoML not reachable at {service_url}{HEALTH_ENDPOINT}",
            )
        )

        if health_ok:
            checks.append(self._embed_smoke_check(service_url, config.timeout_seconds))

        success = all(c.success for c in checks)
        return HarnessResult(
            success=success,
            summary=f"SaProt 650M AF2 validation {'passed' if success else 'failed'}",
            checks=checks,
        )

    def benchmark(self, config: HarnessConfig) -> HarnessResult:
        if CheckScope.OFFLINE not in config.check_scopes:
            return HarnessResult(
                success=True,
                summary="SaProt benchmark skipped: no OFFLINE-scope checks requested",
                checks=[skipped("sequences_per_sec", "needs a live service (scope OFFLINE not selected)")],
            )
        service_url = self._service_url(config)

        if not self.health_check(service_url, timeout_seconds=config.timeout_seconds):
            return HarnessResult(
                success=False,
                summary="BentoML service not reachable for benchmarking",
            )

        expected_dtype, accelerator_count = self._profile_expectations(config)
        reported_dtype = self._query_model_dtype(service_url)
        if expected_dtype and reported_dtype and reported_dtype != expected_dtype:
            return HarnessResult(
                success=False,
                summary=(
                    f"SaProt benchmark failed: the profile declares {expected_dtype} but the service "
                    f"reports {reported_dtype} on {MODELS_ENDPOINT}"
                ),
                metrics={
                    "model_dtype": reported_dtype,
                    "expected_dtype": expected_dtype,
                    "accelerator_count": accelerator_count,
                },
            )

        n_warmup, n_timed = 2, 10
        all_latencies: list[float] = []
        per_length: dict[int, dict] = {}
        failed_requests = 0

        for length in BENCHMARK_LENGTHS:
            # Use real SaProt-encoded protein sequences instead of degenerate M# tiles.
            sequence = BENCHMARK_SEQUENCES[length]
            length_latencies: list[float] = []
            for i in range(n_warmup + n_timed):
                t0 = time.monotonic()
                try:
                    self._check_embedding_response(self._send_embed(service_url, sequence, config.timeout_seconds))
                    elapsed = time.monotonic() - t0
                    if i >= n_warmup:
                        length_latencies.append(elapsed)
                        all_latencies.append(elapsed)
                except Exception as exc:
                    logger.warning("Benchmark length=%d iter=%d failed: %s", length, i, exc)
                    if i >= n_warmup:
                        failed_requests += 1

            if length_latencies:
                mean_l = sum(length_latencies) / len(length_latencies)
                per_length[length] = {
                    "mean_latency_s": round(mean_l, 4),
                    "sequences_per_sec": round(1.0 / mean_l, 4),
                    "n_timed": len(length_latencies),
                }

        if failed_requests:
            # No throughput figure unless every timed request succeeded.
            total_timed = len(all_latencies) + failed_requests
            return HarnessResult(
                success=False,
                summary=(
                    f"SaProt benchmark failed: {failed_requests} of {total_timed} timed request(s) "
                    "failed or returned a malformed embedding response"
                ),
                metrics={
                    "failed_requests": failed_requests,
                    "n_timed_total": total_timed,
                    "accelerator_count": accelerator_count,
                },
            )

        mean_latency = sum(all_latencies) / len(all_latencies)
        sequences_per_sec = 1.0 / mean_latency if mean_latency > 0 else 0.0

        checks = [
            CheckResult(
                name="sequences_per_sec",
                result_type=CheckResultType.SCORE,
                success=True,
                value=round(sequences_per_sec, 4),
                detail=(
                    f"lengths={BENCHMARK_LENGTHS}, "
                    f"mean_latency={mean_latency:.4f}s, "
                    f"sequences_per_sec={sequences_per_sec:.4f}, "
                    "failed_timed_requests=0"
                ),
            )
        ]

        metrics: dict = {
            "mean_latency_s": round(mean_latency, 4),
            "sequences_per_sec": round(sequences_per_sec, 4),
            "n_timed_total": len(all_latencies),
            "model_dtype": reported_dtype or "unknown",
            "expected_dtype": expected_dtype or "unknown",
            "accelerator_count": accelerator_count,
        }
        for length, stats in per_length.items():
            metrics[f"length_{length}_mean_latency_s"] = stats["mean_latency_s"]
            metrics[f"length_{length}_sequences_per_sec"] = stats["sequences_per_sec"]

        metrics["failed_requests"] = 0
        return HarnessResult(
            success=True,
            summary=f"SaProt benchmark: {sequences_per_sec:.4f} seqs/s (lengths={BENCHMARK_LENGTHS})",
            checks=checks,
            metrics=metrics,
        )

    def evaluate(self, config: HarnessConfig) -> HarnessResult:
        if CheckScope.OFFLINE not in config.check_scopes:
            return HarnessResult(
                success=True,
                summary="SaProt evaluation skipped: no OFFLINE-scope checks requested",
                checks=[skipped("embedding_shape", "needs a live service (scope OFFLINE not selected)")],
            )
        service_url = self._service_url(config)

        if not self.health_check(service_url, timeout_seconds=config.timeout_seconds):
            return HarnessResult(
                success=False,
                summary="BentoML service not reachable for evaluation",
            )

        checks: list[CheckResult] = []
        checks.append(self._embedding_shape_check(service_url, config.timeout_seconds))

        success = all(c.success for c in checks)
        return HarnessResult(
            success=success,
            summary=f"SaProt 650M AF2 evaluation {'passed' if success else 'failed'}",
            checks=checks,
            metrics={
                "eval_sequence": EVAL_SEQUENCE,
                "eval_sequence_length": len(EVAL_SEQUENCE),
                "expected_embedding_dim": EXPECTED_EMBEDDING_DIM,
            },
        )

    @staticmethod
    def _profile_expectations(config: HarnessConfig) -> tuple[str | None, int | None]:
        """Read ``precision`` and ``accelerator_count`` from ``config.profile["metadata"]``.

        Returns the precision normalised to a torch-style dtype name (``fp16`` ->
        ``float16``) and the GPU count; either is None when the profile does not declare it.
        """
        metadata = config.profile.get("metadata") or {}
        precision = metadata.get("precision")
        dtype = _DTYPE_ALIASES.get(str(precision), str(precision)) if precision else None
        count = metadata.get("accelerator_count")
        return dtype, int(count) if count is not None else None

    @staticmethod
    def _service_url(config: HarnessConfig) -> str:
        return config.resolve_service_url()

    @staticmethod
    def _query_model_dtype(service_url: str, timeout: int = 10) -> str | None:
        """Read ``data[0].metadata.dtype`` from GET /v1/models.

        Returns the dtype normalised to a torch-style name (``fp16`` -> ``float16``),
        or None if the endpoint is absent, unreachable or lacks the field.
        """
        try:
            req = Request(f"{service_url}{MODELS_ENDPOINT}")
            with urlopen(req, timeout=timeout) as resp:
                if resp.status != 200:
                    return None
                data = json.loads(resp.read().decode())
            dtype = data["data"][0]["metadata"]["dtype"]
        except (HTTPError, URLError, OSError, ValueError, KeyError, IndexError, TypeError) as exc:
            # Absent, unreachable, non-JSON or unexpected shape means "unknown".
            logger.debug("dtype query %s failed: %s", MODELS_ENDPOINT, exc)
            return None
        if not dtype:
            return None
        return _DTYPE_ALIASES.get(str(dtype), str(dtype))

    @staticmethod
    def _send_embed(
        service_url: str,
        sequence: str,
        timeout: float = DEFAULT_REQUEST_TIMEOUT_SECONDS,
    ) -> dict[str, Any]:
        """Send an embedding request and return the parsed JSON response body.

        ``timeout`` is the per-request timeout in seconds; callers forward
        ``HarnessConfig.timeout_seconds``.

        Raises ``ValueError`` unless the HTTP status is 200 and the body is a JSON object.
        """
        # The endpoint parameter is named `request`, so BentoML expects the body nested under it;
        # a flat {"sequence": ...} is rejected with a 400.
        payload = json.dumps({"request": {"sequence": sequence}}).encode()
        req = Request(
            f"{service_url}{INFERENCE_ENDPOINT}",
            data=payload,
            headers={"Content-Type": "application/json"},
        )
        with urlopen(req, timeout=timeout) as resp:
            if resp.status != 200:
                raise ValueError(f"unexpected HTTP status {resp.status}")
            body = json.loads(resp.read().decode())
        if not isinstance(body, dict):
            raise ValueError(f"response body is not a JSON object: {type(body).__name__}")
        return body

    @staticmethod
    def _check_embedding_response(body: dict[str, Any]) -> None:
        """Raise ``ValueError`` unless ``body`` matches the documented response schema.

        The schema is the one in templates/API.md section 4.6: ``embedding`` is a list of
        1280 floats, ``embedding_dim`` is the integer 1280 and ``model`` is a non-empty string.
        """
        embedding = body.get("embedding")
        if not isinstance(embedding, list):
            raise ValueError(f"'embedding' is not a list: {type(embedding).__name__}")
        if len(embedding) != EXPECTED_EMBEDDING_DIM:
            raise ValueError(f"len(embedding)={len(embedding)}, expected {EXPECTED_EMBEDDING_DIM}")
        if not all(isinstance(v, (int, float)) and not isinstance(v, bool) for v in embedding):
            raise ValueError("'embedding' contains non-numeric values")
        dim = body.get("embedding_dim")
        if isinstance(dim, bool) or dim != EXPECTED_EMBEDDING_DIM:
            raise ValueError(f"embedding_dim={dim!r}, expected {EXPECTED_EMBEDDING_DIM}")
        model = body.get("model")
        if not isinstance(model, str) or not model:
            raise ValueError(f"'model' is not a non-empty string: {model!r}")

    @classmethod
    def _embed_smoke_check(cls, service_url: str, timeout: float = DEFAULT_REQUEST_TIMEOUT_SECONDS) -> CheckResult:
        """Embed a short sequence and verify the response has the correct shape."""
        try:
            body = cls._send_embed(service_url, SMOKE_SEQUENCE, timeout)
            embedding = body.get("embedding")
            embedding_dim = body.get("embedding_dim")

            if not isinstance(embedding, list) or len(embedding) == 0:
                return CheckResult(
                    name="embed_smoke",
                    result_type=CheckResultType.PASS_FAIL,
                    success=False,
                    value=False,
                    detail=f"Response missing 'embedding' list: {body!r}",
                )

            dim_match = embedding_dim == EXPECTED_EMBEDDING_DIM and len(embedding) == EXPECTED_EMBEDDING_DIM
            return CheckResult(
                name="embed_smoke",
                result_type=CheckResultType.PASS_FAIL,
                success=dim_match,
                value=dim_match,
                detail=(
                    f"embedding_dim={embedding_dim}, len(embedding)={len(embedding)}, "
                    f"expected={EXPECTED_EMBEDDING_DIM}"
                ),
            )

        except Exception as exc:
            return CheckResult(
                name="embed_smoke",
                result_type=CheckResultType.PASS_FAIL,
                success=False,
                value=False,
                detail=str(exc),
            )

    @classmethod
    def _embedding_shape_check(cls, service_url: str, timeout: float = DEFAULT_REQUEST_TIMEOUT_SECONDS) -> CheckResult:
        """Embed a known protein and verify the embedding dimension is correct."""
        try:
            body = cls._send_embed(service_url, EVAL_SEQUENCE, timeout)
            embedding = body.get("embedding")
            embedding_dim = body.get("embedding_dim")

            if not isinstance(embedding, list):
                return CheckResult(
                    name="embedding_shape",
                    result_type=CheckResultType.PASS_FAIL,
                    success=False,
                    value=False,
                    detail=f"Response 'embedding' is not a list: {type(embedding).__name__}",
                )

            dim_ok = len(embedding) == EXPECTED_EMBEDDING_DIM
            return CheckResult(
                name="embedding_shape",
                result_type=CheckResultType.PASS_FAIL,
                success=dim_ok,
                value=dim_ok,
                detail=(
                    f"embedding_dim={embedding_dim}, len(embedding)={len(embedding)}, "
                    f"expected={EXPECTED_EMBEDDING_DIM}, "
                    f"sequence_length={len(EVAL_SEQUENCE)}"
                ),
            )

        except Exception as exc:
            return CheckResult(
                name="embedding_shape",
                result_type=CheckResultType.PASS_FAIL,
                success=False,
                value=False,
                detail=str(exc),
            )
