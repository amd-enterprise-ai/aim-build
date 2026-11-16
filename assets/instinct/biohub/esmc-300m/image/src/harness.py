# Copyright © Advanced Micro Devices, Inc., or its affiliates.
#
# SPDX-License-Identifier: MIT

"""
ESMCHarness — model-specific harness for the Biohub ESMC 300M protein language
model served via BentoML.

This file lives at ``assets/instinct/biohub/esmc-300m/image/src/harness.py``
and is copied to ``/workspace/model/src/harness.py`` by the specialized
Dockerfile. aim-runtime discovers it automatically via the well-known path.

All checks go through the running HTTP service (never the model in-process):
``/health`` and ``/v1/inference`` (embedding mode and the model-specific ``masked_lm`` mode).

Reference data
--------------
No dataset was supplied for this AIM.  Smoke / benchmark / evaluation inputs
are a handful of well-characterised natural proteins (UniProt accessions in
the comments below), inlined so the harness works inside the weightless image.

* ``embed_smoke`` / ``embedding_shape``: the embedding returned for a known
  protein has the 960-dimensional shape of ESMC-300M (``hidden_size``).
* ``masked_lm_perplexity`` (evaluate): masked-token pseudo-perplexity on the
  reference proteins, 15 % of residues masked with a fixed seed.  A functioning
  language-model head scores far below the ~20 of a uniform guess over the 20
  amino acids; the threshold is a sanity bound, not a quality benchmark.
* ``benchmark``: per-request latency over a length sweep up to the 2046-residue
  ceiling (2048-token context minus ``<cls>``/``<eos>``).

Metric definitions
------------------
``sequences_per_sec`` = 1 / mean latency, measured per length bucket
(``length_<N>_sequences_per_sec`` in the metrics) with strictly sequential
requests (concurrency 1). Latency is the wall time of one HTTP POST to
``/v1/inference`` (it includes HTTP and JSON handling) and excludes the
2 warm-up requests per bucket; 10 timed requests per bucket. The headline
``sequences_per_sec`` is 1 / mean latency over all timed requests of all
buckets pooled. Any failed timed request makes the benchmark ``success=False``.

Scope: every check calls the service and is therefore OFFLINE. ``validate``,
``benchmark`` and ``evaluate`` return skipped results, without any network
call, when OFFLINE is not among the requested scopes.
"""

from __future__ import annotations

import json
import logging
import math
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

# Canonical shared paths (GET /health forwards to BentoML /healthz).
HEALTH_ENDPOINT = "/health"
INFERENCE_ENDPOINT = "/v1/inference"

# Sequence lengths (residues) for the length-sweep benchmark; 2046 is the
# ESMC ceiling (max_position_embeddings 2048 minus <cls>/<eos>).
BENCHMARK_LENGTHS = [64, 256, 512, 1022, 2046]

# Hidden size of biohub/ESMC-300M (config.json) = mean-pooled embedding dim.
EXPECTED_EMBEDDING_DIM = 960

# Trp-cage miniprotein (20 residues, PDB 1L2Y) — fast smoke test.
SMOKE_SEQUENCE = "NLYIQWLKDGGPSSGRPPPS"

# Human ubiquitin, UniProt P0CG48 (UBC_HUMAN) residues 1-76.
UBIQUITIN = "MQIFVKTLTGKTITLEVEPSDTIENVKAKIQDKEGIPPDQQRLIFAGKQLEDGRTLSDYNIQKESTLHLVLRLRGG"
# E. coli thioredoxin 1, UniProt P0AA25 (109 AA).
THIOREDOXIN = (
    "MSDKIIHLTDDSFDTDVLKADGAILVDFWAEWCGPCKMIAPILDEIADEYQGKLTVAKLNIDQNPGTAPKYGIRGIPTLLLFKNGEVAATKVGALSKGQLKEFLDANLA"
)
# Human dihydrofolate reductase, UniProt P00374 (187 AA).
DHFR = (
    "MVGSLNCIVAVSQNMGIGKNGDLPWPPLRNEFRYFQRMTTTSSVEGKQNLVIMGKKTWFSIPEKNRPLKG"
    "RINLVLSRELKEPPQGAHFLSRSLDDALKLTEQPELANKVDMVWIVGGSSVYKEAMNHPGHLKLFVTRIM"
    "QDFESDTFFPEIDLEKYKLLPEYPGVLSDVQEEKGIKYKFEVYEKND"
)
# E. coli adenylate kinase, UniProt P69441 (214 AA).
ADENYLATE_KINASE = (
    "MRIILLGAPGAGKGTQAQFIMEKYGIPQISTGDMLRAAVKSGSELGKQAKDIMDAGKLVTDELVIALVKE"
    "RIAQEDCRNGFLLDGFPRTIPQADAMKEAGINVDYVLEFDVPDELIVDRIVGRRVHAPSGRVYHVKFNPP"
    "KVEGKDDVTGEELTTRKDDQEETVRKRLVEYHQMTAPLIGYYSKEAEAGNTKYAKVDGTKPVAEVRADLE"
    "KILG"
)
# Aequorea victoria GFP, UniProt P42212 (238 AA).
GFP = (
    "MSKGEELFTGVVPILVELDGDVNGHKFSVSGEGEGDATYGKLTLKFICTTGKLPVPWPTLVTTFSYGVQC"
    "FSRYPDHMKQHDFFKSAMPEGYVQERTIFFKDDGNYKTRAEVKFEGDTLVNRIELKGIDFKEDGNILGHK"
    "LEYNYNSHNVYIMADKQKNGIKVNFKIRHNIEDGSVQLADHYQQNTPIGDGPVLLPDNHYLSTQSALSKD"
    "PNEKRDHMVLLEFVTAAGITHGMDELYK"
)
# Human serum albumin precursor, UniProt P02768 (609 AA).
ALBUMIN = (
    "MKWVTFISLLFLFSSAYSRGVFRRDAHKSEVAHRFKDLGEENFKALVLIAFAQYLQQCPFEDHVKLVNEV"
    "TEFAKTCVADESAENCDKSLHTLFGDKLCTVATLRETYGEMADCCAKQEPERNECFLQHKDDNPNLPRLV"
    "RPEVDVMCTAFHDNEETFLKKYLYEIARRHPYFYAPELLFFAKRYKAAFTECCQAADKAACLLPKLDELR"
    "DEGKASSAKQRLKCASLQKFGERAFKAWAVARLSQRFPKAEFAEVSKLVTDLTKVHTECCHGDLLECADD"
    "RADLAKYICENQDSISSKLKECCEKPLLEKSHCIAEVENDEMPADLPSLAADFVESKDVCKNYAEAKDVF"
    "LGMFLYEYARRHPDYSVVLLLRLAKTYETTLEKCCAAADPHECYAKVFDEFKPLVEEPQNLIKQNCELFE"
    "QLGEYKFQNALLVRYTKKVPQVSTPTLVEVSRNLGKVGSKCCKHPEAKRMPCAEDYLSVVLNQLCVLHEK"
    "TPVSDRVTKCCTESLVNRRPCFSALEVDETYVPKEFNAETFTFHADICTLSEKERQIKKQTALVELVKHK"
    "PKATKEQLKAVMDDFAAFVEKCCKADDKETCFAEEGKKLVAASQAALGL"
)

# Evaluation / embedding-shape input.
EVAL_SEQUENCE = UBIQUITIN

# Proteins scored by the masked-LM perplexity check.
MLM_EVAL_SEQUENCES = [UBIQUITIN, THIOREDOXIN, DHFR, ADENYLATE_KINASE, GFP]
MLM_SEEDS = [0, 1, 2, 3]
MLM_MASK_FRACTION = 0.15
# A uniform guess over the 20 standard amino acids scores 20.0. The bound of 8.0 sits
# well below that, so a broken LM head or mis-mapped weights fails while bf16/GPU
# numerical noise passes. These are well-known proteins likely seen in training, so
# this is a wiring/sanity check, not a generalisation benchmark.
MLM_PERPLEXITY_MAX = 8.0

_BENCHMARK_POOL = ALBUMIN + GFP + ADENYLATE_KINASE + DHFR + THIOREDOXIN + UBIQUITIN


def benchmark_sequence(length: int) -> str:
    """Natural-protein concatenation of exactly ``length`` residues.

    Benchmarks characterise scale, not a specific fold, so the pool of real
    proteins is cycled to reach the requested length.
    """
    reps = -(-length // len(_BENCHMARK_POOL))
    return (_BENCHMARK_POOL * reps)[:length]


class ESMCHarness(ModelHarness):
    """Harness for ESMC 300M protein embedding model (BentoML)."""

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
            "Sequences per second = 1 / mean latency (sequential, incl. HTTP, excl. warm-up); per length bucket and pooled",
        ),
        CheckInfo(
            "embedding_shape",
            CheckResultType.PASS_FAIL,
            CheckScope.OFFLINE,
            "Eval-sequence embedding has the expected 960-dimensional shape",
        ),
        CheckInfo(
            "masked_lm_perplexity",
            CheckResultType.PASS_FAIL,
            CheckScope.OFFLINE,
            "Masked-token perplexity on reference proteins is below the sanity bound",
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
                # Expected until the service has finished loading the model.
                logger.debug("%s%s not ready yet", service_url, HEALTH_ENDPOINT, exc_info=True)
            time.sleep(2.0)
        return False

    def validate(self, config: HarnessConfig) -> HarnessResult:
        # All checks need a running service, which only OFFLINE-scope runs start;
        # a runtime-scope validate starts none and must not wait on /health.
        if CheckScope.OFFLINE not in config.check_scopes:
            return HarnessResult(True, "ESMC 300M validate: no runtime-scope checks", [])
        service_url = self._service_url(config)
        checks: list[CheckResult] = []

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
            checks.append(self._embed_smoke_check(service_url))

        success = all(c.success for c in checks)
        return HarnessResult(
            success=success,
            summary=f"ESMC 300M validation {'passed' if success else 'failed'}",
            checks=checks,
        )

    def benchmark(self, config: HarnessConfig) -> HarnessResult:
        if CheckScope.OFFLINE not in config.check_scopes:
            return HarnessResult(
                True,
                "ESMC benchmark skipped: OFFLINE scope not requested",
                [skipped("sequences_per_sec", "OFFLINE scope not requested; no service call made")],
            )
        service_url = self._service_url(config)

        if not self.health_check(service_url, timeout_seconds=config.timeout_seconds):
            return HarnessResult(
                success=False,
                summary="BentoML service not reachable for benchmarking",
            )

        n_warmup, n_timed = 2, 10
        all_latencies: list[float] = []
        n_failed = 0
        per_length: dict[int, dict] = {}

        for length in BENCHMARK_LENGTHS:
            sequence = benchmark_sequence(length)
            length_latencies: list[float] = []
            embedding_dim: int | None = None
            for i in range(n_warmup + n_timed):
                t0 = time.monotonic()
                try:
                    body = self._post(service_url, INFERENCE_ENDPOINT, {"sequence": sequence})
                    elapsed = time.monotonic() - t0
                    if i >= n_warmup:
                        length_latencies.append(elapsed)
                        all_latencies.append(elapsed)
                        if embedding_dim is None:
                            embedding_dim = body.get("embedding_dim")
                except Exception as exc:
                    logger.warning("Benchmark length=%d iter=%d failed: %s", length, i, exc)
                    if i >= n_warmup:
                        n_failed += 1

            if length_latencies:
                mean_l = sum(length_latencies) / len(length_latencies)
                per_length[length] = {
                    "mean_latency_s": round(mean_l, 4),
                    "sequences_per_sec": round(1.0 / mean_l, 4),
                    "embedding_dim": embedding_dim,
                    "n_timed": len(length_latencies),
                }

        if not all_latencies:
            return HarnessResult(
                success=False,
                summary="All benchmark requests failed",
            )

        mean_latency = sum(all_latencies) / len(all_latencies)
        sequences_per_sec = 1.0 / mean_latency if mean_latency > 0 else 0.0

        n_expected = len(BENCHMARK_LENGTHS) * n_timed
        all_ok = n_failed == 0 and len(all_latencies) == n_expected
        checks = [
            CheckResult(
                name="sequences_per_sec",
                result_type=CheckResultType.SCORE,
                success=all_ok,
                value=round(sequences_per_sec, 4),
                detail=(
                    f"lengths={BENCHMARK_LENGTHS}, "
                    f"mean_latency={mean_latency:.4f}s, "
                    f"sequences_per_sec={sequences_per_sec:.4f}, "
                    f"timed_ok={len(all_latencies)}/{n_expected}, timed_failed={n_failed}"
                ),
            )
        ]

        metrics: dict = {
            "mean_latency_s": round(mean_latency, 4),
            "sequences_per_sec": round(sequences_per_sec, 4),
            "n_timed_total": len(all_latencies),
            "n_timed_failed": n_failed,
        }
        for length, stats in per_length.items():
            metrics[f"length_{length}_mean_latency_s"] = stats["mean_latency_s"]
            metrics[f"length_{length}_sequences_per_sec"] = stats["sequences_per_sec"]
            metrics[f"length_{length}_embedding_dim"] = stats["embedding_dim"]

        return HarnessResult(
            success=all_ok and len(per_length) == len(BENCHMARK_LENGTHS),
            summary=(
                f"ESMC benchmark: {sequences_per_sec:.4f} seqs/s (lengths={BENCHMARK_LENGTHS})"
                if all_ok
                else f"ESMC benchmark FAILED: {n_failed} of {n_expected} timed requests failed"
            ),
            checks=checks,
            metrics=metrics,
        )

    def evaluate(self, config: HarnessConfig) -> HarnessResult:
        if CheckScope.OFFLINE not in config.check_scopes:
            return HarnessResult(
                True,
                "ESMC evaluation skipped: OFFLINE scope not requested",
                [
                    skipped("embedding_shape", "OFFLINE scope not requested; no service call made"),
                    skipped("masked_lm_perplexity", "OFFLINE scope not requested; no service call made"),
                ],
            )
        service_url = self._service_url(config)

        if not self.health_check(service_url, timeout_seconds=config.timeout_seconds):
            return HarnessResult(
                success=False,
                summary="BentoML service not reachable for evaluation",
            )

        shape_check = self._embedding_shape_check(service_url)
        ppl_check, ppl_metrics = self._masked_lm_perplexity_check(service_url)
        checks = [shape_check, ppl_check]

        success = all(c.success for c in checks)
        return HarnessResult(
            success=success,
            summary=f"ESMC evaluation {'passed' if success else 'failed'}",
            checks=checks,
            metrics={
                "eval_sequence_length": len(EVAL_SEQUENCE),
                "expected_embedding_dim": EXPECTED_EMBEDDING_DIM,
                "mlm_perplexity_max": MLM_PERPLEXITY_MAX,
                **ppl_metrics,
            },
        )

    @staticmethod
    def _service_url(config: HarnessConfig) -> str:
        return config.resolve_service_url()

    @staticmethod
    def _post(service_url: str, endpoint: str, payload: dict[str, Any], timeout: int = 60) -> dict[str, Any]:
        """POST JSON and return the parsed JSON response body."""
        req = Request(
            f"{service_url}{endpoint}",
            data=json.dumps(payload).encode(),
            headers={"Content-Type": "application/json"},
        )
        with urlopen(req, timeout=timeout) as resp:
            return json.loads(resp.read().decode())

    @classmethod
    def _embed_smoke_check(cls, service_url: str) -> CheckResult:
        """Embed a tiny sequence and verify the response has the correct shape."""
        try:
            body = cls._post(service_url, INFERENCE_ENDPOINT, {"sequence": SMOKE_SEQUENCE})
            embedding = body.get("embedding")
            embedding_dim = body.get("embedding_dim")

            if not isinstance(embedding, list) or len(embedding) == 0:
                return CheckResult(
                    name="embed_smoke",
                    result_type=CheckResultType.PASS_FAIL,
                    success=False,
                    value=False,
                    detail=f"Response missing 'embedding' list: {str(body)[:200]}",
                )

            finite = all(math.isfinite(x) for x in embedding)
            dim_match = embedding_dim == EXPECTED_EMBEDDING_DIM and len(embedding) == EXPECTED_EMBEDDING_DIM
            ok = dim_match and finite
            return CheckResult(
                name="embed_smoke",
                result_type=CheckResultType.PASS_FAIL,
                success=ok,
                value=ok,
                detail=(
                    f"embedding_dim={embedding_dim}, len(embedding)={len(embedding)}, "
                    f"expected={EXPECTED_EMBEDDING_DIM}, all_finite={finite}"
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
    def _embedding_shape_check(cls, service_url: str) -> CheckResult:
        """Embed a known protein and verify the embedding dimension is correct."""
        try:
            body = cls._post(service_url, INFERENCE_ENDPOINT, {"sequence": EVAL_SEQUENCE})
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

            finite = all(math.isfinite(x) for x in embedding)
            dim_ok = len(embedding) == EXPECTED_EMBEDDING_DIM and finite
            return CheckResult(
                name="embedding_shape",
                result_type=CheckResultType.PASS_FAIL,
                success=dim_ok,
                value=dim_ok,
                detail=(
                    f"embedding_dim={embedding_dim}, len(embedding)={len(embedding)}, "
                    f"expected={EXPECTED_EMBEDDING_DIM}, all_finite={finite}, "
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

    @classmethod
    def _masked_lm_perplexity_check(cls, service_url: str) -> tuple[CheckResult, dict[str, Any]]:
        """Masked-token perplexity over the reference proteins (token-weighted)."""
        total_loss = 0.0
        total_masked = 0
        try:
            for seq in MLM_EVAL_SEQUENCES:
                for seed in MLM_SEEDS:
                    body = cls._post(
                        service_url,
                        INFERENCE_ENDPOINT,  # masked-LM is the "masked_lm" mode of /v1/inference
                        {"sequence": seq, "mode": "masked_lm", "mask_fraction": MLM_MASK_FRACTION, "seed": seed},
                    )
                    n = int(body["masked_positions"])
                    total_loss += float(body["loss"]) * n
                    total_masked += n
        except Exception as exc:
            return (
                CheckResult(
                    name="masked_lm_perplexity",
                    result_type=CheckResultType.PASS_FAIL,
                    success=False,
                    value=False,
                    detail=str(exc),
                ),
                {},
            )

        mean_loss = total_loss / total_masked
        perplexity = math.exp(mean_loss)
        ok = math.isfinite(perplexity) and perplexity < MLM_PERPLEXITY_MAX
        return (
            CheckResult(
                name="masked_lm_perplexity",
                result_type=CheckResultType.PASS_FAIL,
                success=ok,
                value=ok,
                detail=(
                    f"perplexity={perplexity:.3f} (mean_loss={mean_loss:.4f} nats over {total_masked} masked "
                    f"residues, {len(MLM_EVAL_SEQUENCES)} proteins x {len(MLM_SEEDS)} seeds); "
                    f"threshold < {MLM_PERPLEXITY_MAX}"
                ),
            ),
            {
                "mlm_perplexity": round(perplexity, 4),
                "mlm_mean_loss_nats": round(mean_loss, 4),
                "mlm_masked_residues": total_masked,
            },
        )
