# Copyright © Advanced Micro Devices, Inc., or its affiliates.
#
# SPDX-License-Identifier: MIT

"""
Boltz2Harness — model-specific harness for the Boltz-2 protein structure
prediction model served via BentoML.

This file lives at ``assets/instinct/mit/boltz2/image/src/harness.py`` and
is copied to ``/workspace/model/src/harness.py`` by the specialized
Dockerfile. aim-runtime discovers it automatically via the well-known path.

It imports only the standard library and aim-runtime's public API
(``from aim_runtime.harness import ...``), so it loads on any node without
BentoML, torch or Boltz.
"""

from __future__ import annotations

import contextlib
import ctypes
import json
import logging
import math
import os
import statistics
import sys
import tempfile
import time
from collections.abc import Iterator
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

BENTOML_HEALTH_ENDPOINT = "/health"
# The one prediction route. It answers JSON with the structure and
# Boltz-2's confidence head (pLDDT / pTM).
BENTOML_INFERENCE_ENDPOINT = "/v1/inference"

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

# Affinity smoke complex: Trp-cage with aspirin as the binder.
AFFINITY_SMOKE_LIGAND = "CC(=O)OC1=CC=CC=C1C(=O)O"
AFFINITY_SMOKE_PARAMS: dict[str, Any] = {
    **FAST_PREDICT_PARAMS,
    "sampling_steps_affinity": 10,
    "diffusion_samples_affinity": 1,
}

# Evaluation cases baked into the image by the Dockerfile. The reference
# structures are not in the image: the user mounts them under DATA_DIR.
BENCHMARK_DIR = Path(os.environ.get("BOLTZ2_BENCHMARK_DIR", "/workspace/model/benchmarks"))

# One directory holds every case; a case belongs to one or more NAMED SETS, which
# it declares in metadata.sets. The set is the selector, not the directory.
#
# A fixture file is a (target, recipe) pair, so a set that differs only in which
# targets it covers shares the files. A set that differs by RECIPE needs its own
# directory, because the same target cannot hold two request bodies in one file;
# BOLTZ2_BENCHMARK_DIR selects that directory.
#
# THE DEFAULT IS LOAD-BEARING. Every timing, accuracy and gate-abstention figure
# this asset has published is a sum over the six cases of DEFAULT_BENCHMARK_SET.
# Widening it re-baselines those figures with no code change and no signal, which
# is why tests/unit/test_benchmark_sets.py pins the default membership by id
# rather than by count.
DEFAULT_BENCHMARK_SET = "fast"
BENCHMARK_SET = os.environ.get("BOLTZ2_BENCHMARK_SET", DEFAULT_BENCHMARK_SET)
DATA_DIR = Path(os.environ.get("AIM_DATA_DIR", "/workspace/data"))
REFERENCE_SUBDIR = "references"

# Cα-RMSD threshold separating "the model still folds these targets" from
# "something in this build is broken". Deliberately loose: four of the six
# targets were released BEFORE Boltz-2's stated 2023-06-01 PDB training cutoff,
# so they were plausibly in training and the check is a regression detector, not
# a capability measurement. AlphaFold-class models land well under 2 Å on
# targets they know; 8 Å is far enough above that to tolerate diffusion-sampling
# variance while still catching a wrong precision, a disabled kernel path, or a
# mis-wired featurizer.
RMSD_REGRESSION_THRESHOLD_A = 8.0

# complex_plddt floors — the second half of the accuracy gate, and the metric
# tuning.yaml hands the Tuner. RMSD needs a staged reference structure;
# complex_plddt is self-reported by the model, so it also covers inputs that
# have no reference and it degrades for different reasons (a broken confidence
# head, a wrong precision, a truncated recycling schedule).
#
# What the number is: boltz/model/modules/confidencev2.py computes complex_plddt
# as a pad-mask-weighted mean over tokens of the per-token pLDDT, where that
# pLDDT is compute_aggregated_metric(logits, end=1.0) — an expected value over
# 50 bins spanning [0, 1]. It is the AlphaFold pLDDT quantity on a 0–1 scale, so
# AlphaFold's published bands transfer directly after dividing by 100:
# very high > 90 | confident 90–70 | low 70–50 | very low < 50 (EMBL-EBI
# AlphaFold course, "pLDDT: Understanding local confidence").
#
# MEAN floor 0.90 is the "very high"/"confident" band boundary: the gate is
# "stay in the band the build ships in".
#
# PER-CASE floor 0.70 is the "confident"/"low" boundary. It catches one target
# collapsing while the mean stays afloat, which the mean cannot see over six
# targets.
PLDDT_MEAN_FLOOR = 0.90
PLDDT_PER_CASE_FLOOR = 0.70

# The request defaults the SERVICE applies when a case omits a field
# (service.Boltz2Request). The gate has to resolve an absent field against these
# rather than treat "absent" as "matching", or a case that says nothing about the
# MSA would be graded as though it had one. Duplicated here instead of imported:
# harness.py is loaded by aim-runtime on any node, and importing service.py would
# drag BentoML (and its required `port` bento argument) in with it. A unit test
# in tests/unit/test_service_contract.py pins these to Boltz2Request's real
# defaults, so the duplication cannot drift silently.
SERVED_REQUEST_DEFAULTS: dict[str, Any] = {
    "use_msa_server": False,
    "sampling_steps": 200,
    "recycling_steps": 3,
    "diffusion_samples": 1,
    "step_scale": None,
}

# The request configuration the floors above describe, and the ONLY one this gate
# will return a verdict on.
#
# The floors are properties of the model run a particular way: six targets
# predicted with an MSA at boltz's default sampling schedule. The image also
# ships a timing set that is deliberately NOT that configuration (single-sequence,
# sampling_steps=50; see the metadata.recipe block in each
# examples/benchmark_sequences/<id>/<id>.json). Grading those cases against these
# floors would detect the configuration difference, not a regression.
#
# So the gate asks first whether it is looking at the configuration it was
# calibrated for. It grades cases that match; for cases that do not it reports
# the measured numbers and ABSTAINS, naming the mismatching fields. It never
# grades an unmatched case, and it never silently passes one either.
#
# The recipe is the SERVING defaults with exactly one departure — MSA on. An MSA
# is a first-order input to structure prediction, and every number these floors
# rest on was measured with one, so a gate that accepted single-sequence input
# would be applying an MSA-derived floor to an MSA-free prediction.
ACCURACY_GATE_RECIPE: dict[str, Any] = {**SERVED_REQUEST_DEFAULTS, "use_msa_server": True}


def accuracy_recipe_mismatch(case_data: dict[str, Any]) -> dict[str, dict[str, Any]]:
    """Fields where ``case_data`` departs from :data:`ACCURACY_GATE_RECIPE`.

    Pure, so the CPU unit tier owns every branch. Returns ``{}`` when the case is
    gradeable; otherwise ``{field: {"gate": <required>, "case": <effective>}}``.
    An absent field resolves to :data:`SERVED_REQUEST_DEFAULTS`, because that is
    what the service would apply to it.
    """
    mismatch: dict[str, dict[str, Any]] = {}
    for field, required in ACCURACY_GATE_RECIPE.items():
        effective = case_data.get(field, SERVED_REQUEST_DEFAULTS[field])
        if effective != required:
            mismatch[field] = {"gate": required, "case": effective}
    return mismatch


def describe_recipe_mismatch(mismatch: dict[str, dict[str, Any]]) -> str:
    """One-line rendering of :func:`accuracy_recipe_mismatch`'s output."""
    return ", ".join(f"{field}={m['case']!r} (gate needs {m['gate']!r})" for field, m in sorted(mismatch.items()))


def _summarize(metrics: dict[str, Any], records: list[dict[str, Any]], prefix: str) -> None:
    """Write ``<prefix>mean_ca_rmsd_a`` / ``<prefix>mean_complex_plddt`` into ``metrics``.

    Each mean covers the records that carry its value; a record with no
    reference structure adds to the pLDDT mean only. Writes nothing for a
    quantity no record carries, so an absent key means "no number of this kind
    was produced" rather than "zero".
    """
    rmsds = [r["rmsd_a"] for r in records if r.get("rmsd_a") is not None]
    if rmsds:
        metrics[f"{prefix}mean_ca_rmsd_a"] = round(sum(rmsds) / len(rmsds), 3)
    plddts = [r["complex_plddt"] for r in records if r.get("complex_plddt") is not None]
    if plddts:
        metrics[f"{prefix}mean_complex_plddt"] = round(sum(plddts) / len(plddts), 4)


def grade_confidence(per_case: list[dict[str, Any]]) -> tuple[bool, float | None, str]:
    """Apply the complex_plddt floors to a list of per-case evaluation records.

    Pure, so the CPU unit tier can exercise every branch without a GPU, a
    service or weights. Returns ``(passed, mean_plddt, detail)``.

    ``per_case`` records that carry no ``complex_plddt`` are ignored rather than
    failed — a case can be skipped for reasons this gate has no opinion about
    (missing reference, transport error), and those are already surfaced by the
    accuracy check. If NOT ONE case reports a confidence value the gate cannot
    judge anything and says so, rather than passing vacuously.

    See ``PLDDT_MEAN_FLOOR`` / ``PLDDT_PER_CASE_FLOOR`` for how the two floors
    were derived.
    """
    values = [(r.get("label", "?"), float(r["complex_plddt"])) for r in per_case if r.get("complex_plddt") is not None]
    if not values:
        return (False, None, f"0/{len(per_case)} cases reported complex_plddt; the confidence gate cannot be applied")

    mean = sum(v for _, v in values) / len(values)
    worst_label, worst = min(values, key=lambda item: item[1])

    mean_ok = mean >= PLDDT_MEAN_FLOOR
    per_case_ok = worst >= PLDDT_PER_CASE_FLOOR
    passed = mean_ok and per_case_ok

    detail = (
        f"{len(values)}/{len(per_case)} cases reported complex_plddt; "
        f"mean={mean:.4f} vs floor {PLDDT_MEAN_FLOOR:.2f} => {'PASS' if mean_ok else 'FAIL'}; "
        f"worst={worst_label} at {worst:.4f} vs floor {PLDDT_PER_CASE_FLOOR:.2f} "
        f"=> {'PASS' if per_case_ok else 'FAIL'}"
    )
    return (passed, mean, detail)


# The timing the benchmark runs with no options, and the explicit choice of the
# 20-residue smoke timing.
DEFAULT_TIMING_SET = "timing-18"
SMOKE_SET = "smoke"

# Request fields the "served" benchmark recipe leaves to the service's defaults.
SERVED_RECIPE_FIELDS = ("sampling_steps", "step_scale", "use_msa_server")
BENCHMARK_RECIPES = ("fixture", "served")
DEFAULT_BENCHMARK_RECIPE = "fixture"
BENCHMARK_RECORDS_FILE = "benchmark_records.json"
EVALUATE_RECORDS_FILE = "evaluate_records.json"


def _elapsed_s(started: float) -> float:
    """Seconds since ``started`` (a ``time.perf_counter()`` reading), to the millisecond."""
    return round(time.perf_counter() - started, 3)


def _label_list(value: Any) -> list[str]:
    """A label filter from a YAML list or a comma-separated string."""
    if not value:
        return []
    items = value.split(",") if isinstance(value, str) else value
    return [str(item).strip() for item in items if str(item).strip()]


def _progress(line: str) -> None:
    """One progress line on stderr; stdout stays the machine-readable result."""
    print(line, file=sys.stderr, flush=True)


# What gemmi's superposition solver prints from C++ when it hits its iteration limit.
QCP_LIMIT_MESSAGE = "More than 50 iterations needed"


@contextlib.contextmanager
def _native_output_captured() -> Iterator[list[str]]:
    """Capture what native code writes to fd 1 and 2; the lines are in the yielded list on exit.

    gemmi prints its solver-limit message to stderr from C++, past
    ``sys.stderr``, so only a file-descriptor redirect lets the harness count it
    and keep it out of its own progress lines. Both descriptors are redirected so
    that no native output can reach the stream that carries the JSON result.
    """
    lines: list[str] = []
    sys.stdout.flush()
    sys.stderr.flush()
    saved = (os.dup(1), os.dup(2))
    with tempfile.TemporaryFile("w+", errors="replace") as sink:
        os.dup2(sink.fileno(), 1)
        os.dup2(sink.fileno(), 2)
        try:
            yield lines
        finally:
            sys.stdout.flush()
            sys.stderr.flush()
            with contextlib.suppress(OSError, AttributeError):
                ctypes.CDLL(None).fflush(None)
            os.dup2(saved[0], 1)
            os.dup2(saved[1], 2)
            os.close(saved[0])
            os.close(saved[1])
            sink.seek(0)
            lines.extend(sink.read().splitlines())


def concurrent_benchmark_processes(proc_root: str = "/proc", own_pid: int | None = None) -> list[str]:
    """Other ``entrypoint.py benchmark`` processes in this container, as ``"<pid>: <command>"``.

    A second benchmark shares the one GPU and the service, so both timings are
    wrong. Best effort: it reads ``/proc``, skips this process and its ancestors
    (a shell that launched it names the same command), and returns nothing where
    ``/proc`` is not readable.
    """
    own = os.getpid() if own_pid is None else own_pid
    root = Path(proc_root)
    ancestors = {own}
    pid = own
    while pid > 1:
        try:
            stat = (root / str(pid) / "stat").read_text()
            pid = int(stat.rsplit(")", 1)[1].split()[1])
        except (OSError, ValueError, IndexError):
            break
        ancestors.add(pid)
    found: list[str] = []
    try:
        entries = sorted(root.iterdir(), key=lambda e: e.name)
    except OSError:
        return found
    for entry in entries:
        if not entry.name.isdigit() or int(entry.name) in ancestors:
            continue
        try:
            argv = (entry / "cmdline").read_bytes().decode(errors="replace").split("\0")
        except OSError:
            continue
        for i, arg in enumerate(argv):
            launched_by_python = i == 0 or Path(argv[0]).name.startswith("python")
            if arg.endswith("entrypoint.py") and launched_by_python and argv[i + 1 : i + 2] == ["benchmark"]:
                found.append(f"{entry.name}: {' '.join(a for a in argv if a)}")
                break
    return found


def _benchmark_line(rec: dict[str, Any], blocks: int, set_name: str, served: bool = False) -> str:
    """The progress line of one timed request."""
    parts = [f"[benchmark] block {rec['block']}/{blocks} {rec['label']}"]
    if rec.get("tokens") is not None:
        parts.append(f"{rec['tokens']} tok")
    parts.append(f"wall {rec['wall_s']:.2f} s")
    reference = rec.get("reference")
    if reference and not rec.get("error") and rec["wall_s"] > 0:
        ref_s = reference["first_s"] if rec["block"] == 1 else reference["warm_s"]
        parts.append(
            f"ratio {ref_s / rec['wall_s']:.3f} vs reference {ref_s:.2f} s"
            + (" (served recipe: different workload)" if served else "")
        )
    compiled = rec.get("compiled")
    parts.append("compiled=" + ("unknown" if compiled is None else "yes" if compiled else "no"))
    if rec.get("error"):
        parts.append(f"ERROR: {rec['error']}")
    return "  ".join(parts)


def _evaluate_line(record: dict[str, Any], index: int, total: int) -> str:
    """The progress line of one evaluated case."""
    rmsd = record.get("rmsd_a")
    parts = [f"[evaluate] {index}/{total} {record['label']}"]
    parts.append(f"RMSD {rmsd:.3f} A" if rmsd is not None else "RMSD n/a")
    plddt = record.get("complex_plddt")
    parts.append(f"pLDDT {plddt:.4f}" if plddt is not None else "pLDDT n/a")
    if record.get("wall_s") is not None:
        parts.append(f"wall {record['wall_s']:.2f} s")
    if record.get("error"):
        parts.append(f"ERROR: {record['error']}")
    return "  ".join(parts)


def geometric_mean(values: list[float]) -> float:
    return math.exp(sum(math.log(v) for v in values) / len(values))


def load_reference_times(path: str | Path) -> dict[str, dict[str, float]]:
    """Read a reference-times file: ``{"targets": {label: {"first_s", "warm_s", "alt_warm_s"?}}}``.

    Times are seconds. ``first_s`` is compared with the first request of a
    target; ``warm_s`` and the optional ``alt_warm_s`` with the median of its
    warm requests. Raises ``ValueError`` for a file that is not in this format.
    """
    try:
        document = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"cannot read benchmark_reference_file {str(path)!r}: {exc}") from exc
    targets = document.get("targets") if isinstance(document, dict) else None
    if not isinstance(targets, dict) or not targets:
        raise ValueError(f"benchmark_reference_file {str(path)!r} has no 'targets' mapping")
    out: dict[str, dict[str, float]] = {}
    for label, entry in targets.items():
        times: dict[str, float] = {}
        for key in ("first_s", "warm_s", "alt_warm_s"):
            value = entry.get(key) if isinstance(entry, dict) else None
            if value is None and key == "alt_warm_s":
                continue
            if not isinstance(value, (int, float)) or isinstance(value, bool) or value <= 0:
                raise ValueError(f"benchmark_reference_file target {label!r}: {key} must be a positive number")
            times[key] = float(value)
        out[label] = times
    return out


def reference_ratios(records: list[dict[str, Any]]) -> dict[str, dict[str, float]]:
    """Per-target times and speed ratios against the reference times of one benchmark run.

    A ratio is reference time / our time, so above 1 means we are faster. "First"
    is the block-1 request of a target; "warm" is the median over blocks 2..N.
    Requests that errored are left out. A target with no reference time has
    times but no ratio. Returns ``{"first", "warm", "alt_warm", "first_s",
    "warm_s"}``, each keyed by label. Pure, so the CPU unit tier owns every branch.
    """
    ok = [r for r in records if not r.get("error") and (r.get("wall_s") or 0) > 0]
    out: dict[str, dict[str, float]] = {k: {} for k in ("first", "warm", "alt_warm", "first_s", "warm_s")}
    for rec in ok:
        if rec["block"] == 1:
            out["first_s"][rec["label"]] = rec["wall_s"]
            if rec.get("reference"):
                out["first"][rec["label"]] = rec["reference"]["first_s"] / rec["wall_s"]
    for label in {r["label"] for r in ok if r["block"] > 1}:
        walls = [r["wall_s"] for r in ok if r["label"] == label and r["block"] > 1]
        reference = next((r["reference"] for r in ok if r["label"] == label and r.get("reference")), None)
        median = statistics.median(walls)
        out["warm_s"][label] = median
        if reference:
            out["warm"][label] = reference["warm_s"] / median
            if "alt_warm_s" in reference:
                out["alt_warm"][label] = reference["alt_warm_s"] / median
    return out


def _build_predict_payload(
    sequence: str,
    *,
    chain_id: str = "A",
    name: str = "harness_run",
    output_format: str = "pdb",
    use_msa_server: bool = False,
    **overrides: Any,
) -> dict[str, Any]:
    """Build a Boltz-2 /v1/inference request body."""
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

    # The AIM's serving engine. aim-runtime's discovery uses this to pick the
    # right harness when a profile names an engine and the module exports more
    # than one ModelHarness subclass.
    ENGINE = "bentoml"

    # ------------------------------------------------------------------ #
    # Check registry
    # ------------------------------------------------------------------ #

    CHECKS: list[CheckInfo] = [
        CheckInfo("bentoml_health", CheckResultType.PASS_FAIL, CheckScope.RUNTIME, "GET /health returns 200"),
        CheckInfo(
            "predict_smoke",
            CheckResultType.PASS_FAIL,
            CheckScope.OFFLINE,
            "Single-sequence prediction returns PDB output",
        ),
        CheckInfo(
            "predict_affinity_smoke",
            CheckResultType.PASS_FAIL,
            CheckScope.OFFLINE,
            "In BOLTZ2_MODE=affinity: a protein with one ligand returns the affinity values; not run in structure mode",
        ),
        CheckInfo(
            "throughput",
            CheckResultType.SCORE,
            CheckScope.OFFLINE,
            "Structures predicted per minute (smoke: all requests; fixture set: all successful timed requests, "
            "cold block 1 included)",
        ),
        CheckInfo(
            "p50_ms",
            CheckResultType.SCORE,
            CheckScope.OFFLINE,
            "Median end-to-end HTTP predict latency (ms) (smoke: all requests; fixture set: warm requests of "
            "blocks 2..N)",
        ),
        CheckInfo(
            "reference_ratio_warm",
            CheckResultType.SCORE,
            CheckScope.OFFLINE,
            "Geometric mean of reference time / our warm time over the targets of the reference file (above 1 = "
            "faster); needs benchmark_reference_file",
        ),
        CheckInfo(
            "reference_ratio_first",
            CheckResultType.SCORE,
            CheckScope.OFFLINE,
            "Geometric mean of reference time / our first-request time over the targets of the reference file; "
            "needs benchmark_reference_file",
        ),
        CheckInfo(
            "reference_alt_ratio_warm",
            CheckResultType.SCORE,
            CheckScope.OFFLINE,
            "Geometric mean of the second reference time / our warm time; needs benchmark_reference_file with "
            "alt_warm_s entries",
        ),
        CheckInfo(
            "requests_ok",
            CheckResultType.PASS_FAIL,
            CheckScope.OFFLINE,
            "Every timed request of a fixture set returned a structure",
        ),
        CheckInfo(
            "compiles_after_ready",
            CheckResultType.PASS_FAIL,
            CheckScope.OFFLINE,
            "No timed request triggered a compile after the start-up warm-up (skipped when optimizations are off or the service reports no compile count)",
        ),
        CheckInfo(
            "structure_sanity",
            CheckResultType.PASS_FAIL,
            CheckScope.OFFLINE,
            "Predicted PDB contains ATOM records and plausible coordinates",
        ),
        CheckInfo(
            "structure_accuracy",
            CheckResultType.SCORE,
            CheckScope.OFFLINE,
            "Mean C-alpha RMSD (Angstrom) against the experimental RCSB structures",
        ),
        CheckInfo(
            "confidence_regression",
            CheckResultType.SCORE,
            CheckScope.OFFLINE,
            "Mean Boltz-2 complex_plddt against the derived floors (see PLDDT_MEAN_FLOOR)",
        ),
    ]

    def list_checks(self) -> list[CheckInfo]:
        return list(self.CHECKS)

    # ------------------------------------------------------------------ #
    # health_check override — probes GET /health
    # ------------------------------------------------------------------ #

    def health_check(self, service_url: str, timeout_seconds: int = 60) -> bool:
        deadline = time.monotonic() + timeout_seconds
        while time.monotonic() < deadline:
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
            smoke, mode = self._run_predict_smoke(service_url)
            checks.append(smoke)
            if smoke.success and mode == "affinity":
                checks.append(self._affinity_smoke_test(service_url))

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

        set_name = str(config.get("benchmark_set") or DEFAULT_TIMING_SET)
        if set_name != SMOKE_SET:
            return self._benchmark_fixtures(service_url, config, set_name)

        n_requests = config.get("benchmark_n_requests", 10)
        latencies: list[float] = []

        for i in range(n_requests):
            t0 = time.perf_counter()
            try:
                self._send_predict(service_url, SMOKE_SEQUENCE, name=f"bench_{i}")
                latencies.append(_elapsed_s(t0))
            except Exception as exc:
                logger.warning("Benchmark request %d/%d failed: %s", i + 1, n_requests, exc)

        if not latencies:
            return HarnessResult(success=False, summary="All benchmark requests failed")

        avg_latency = sum(latencies) / len(latencies)
        throughput = 60.0 / avg_latency if avg_latency > 0 else 0.0

        ordered = sorted(latencies)
        p50 = ordered[len(ordered) // 2]
        p95 = ordered[min(len(ordered) - 1, int(round(0.95 * (len(ordered) - 1))))]

        checks = [
            CheckResult(
                name="throughput",
                result_type=CheckResultType.SCORE,
                success=True,
                value=round(throughput, 2),
                detail=f"{len(latencies)}/{n_requests} requests succeeded, "
                f"avg latency={avg_latency:.2f}s, "
                f"throughput={throughput:.1f} structures/min",
            ),
            # Named p50_ms to match aim_runtime.benchmarking.AIMBenchmark's field,
            # so the Tuner's sweeps and CI can compare against this run directly.
            CheckResult(
                name="p50_ms",
                result_type=CheckResultType.SCORE,
                success=True,
                value=round(p50 * 1000.0, 1),
                detail=f"p50={p50 * 1000.0:.1f} ms, p95={p95 * 1000.0:.1f} ms over "
                f"{len(latencies)} successful requests",
            ),
            *(
                self._not_applicable(
                    name, f"benchmark_set is {SMOKE_SET!r}; the smoke timing does not time a fixture set"
                )
                for name in (
                    "reference_ratio_warm",
                    "reference_ratio_first",
                    "reference_alt_ratio_warm",
                    "requests_ok",
                    "compiles_after_ready",
                )
            ),
        ]

        return HarnessResult(
            success=True,
            summary=f"Boltz-2 benchmark: {throughput:.1f} structures/min, p50={p50 * 1000.0:.0f} ms",
            checks=checks,
            metrics={
                "avg_latency_s": round(avg_latency, 3),
                "p50_ms": round(p50 * 1000.0, 1),
                "p95_ms": round(p95 * 1000.0, 1),
                "throughput_per_min": round(throughput, 2),
                "throughput_img_per_sec": round(1.0 / avg_latency, 4) if avg_latency > 0 else 0.0,
                "requests_succeeded": len(latencies),
                "requests_total": n_requests,
            },
        )

    # ------------------------------------------------------------------ #
    # benchmark of a fixture set (timing, optionally against reference times)
    # ------------------------------------------------------------------ #

    @staticmethod
    def _not_applicable(name: str, detail: str) -> CheckResult:
        info = next(c for c in Boltz2Harness.CHECKS if c.name == name)
        return CheckResult(name, info.result_type, True, None, detail, skipped=True)

    @staticmethod
    def _request_data(case_data: dict[str, Any], recipe: str) -> dict[str, Any]:
        """The /v1/inference body for one case; ``served`` drops the recipe fields."""
        data = {**case_data, "output_format": "mmcif"}
        if recipe == "served":
            for field in SERVED_RECIPE_FIELDS:
                data.pop(field, None)
        return data

    @classmethod
    def _time_request(cls, service_url: str, data: dict[str, Any], timeout: int) -> dict[str, Any]:
        """POST one /v1/inference request; return client time and the service's own readings."""
        started = time.perf_counter()
        rec: dict[str, Any] = {}
        try:
            body = cls._post_json(f"{service_url}{BENTOML_INFERENCE_ENDPOINT}", {"data": data}, timeout=timeout)
            rec["wall_s"] = _elapsed_s(started)
            payload = json.loads(body)
            runtime = payload.get("runtime") or {}
            rec["runtime_s"] = (payload.get("timing") or {}).get("runtime_s")
            rec["cuda_peak_gb"] = runtime.get("cuda_peak_gb")
            rec["optimizations"] = runtime.get("optimizations")
            warmup = runtime.get("warmup") or {}
            rec["warmup_state"] = warmup.get("state")
            since = warmup.get("since_warmup")
            compiles = since.get("compiles") if isinstance(since, dict) else None
            if isinstance(compiles, dict):
                rec["compiles_since_warmup"] = {k: v.get("successful_compiles", 0) for k, v in compiles.items()}
                rec["failed_compiles_since_warmup"] = {k: v.get("failed_compiles", 0) for k, v in compiles.items()}
            if payload.get("error"):
                rec["error"] = payload.get("message", "prediction failed")
        except Exception as exc:  # noqa: BLE001 — recorded per request, not raised
            rec["wall_s"] = _elapsed_s(started)
            rec["error"] = f"{type(exc).__name__}: {exc}"
        return rec

    @staticmethod
    def _write_records(config: HarnessConfig, filename: str, document: dict[str, Any]) -> str | None:
        output_dir = config.get("output_dir")
        if not output_dir:
            return None
        try:
            path = Path(output_dir) / filename
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps(document, indent=1))
        except OSError as exc:
            logger.warning("Could not write %s: %s", filename, exc)
            return None
        return str(path)

    def _benchmark_fixtures(self, service_url: str, config: HarnessConfig, set_name: str) -> HarnessResult:
        """Time every case of a fixture set, in blocks, optionally against reference times.

        Options (``--config``): ``benchmark_reference_file`` (JSON with reference times per
        target, see :func:`load_reference_times`; the ratios need the fixture recipe; without the file the run reports our times only),
        ``benchmark_set`` (default
        :data:`DEFAULT_TIMING_SET`; ``smoke`` selects the smoke timing), ``benchmark_dir``,
        ``benchmark_blocks`` (default 3), ``benchmark_labels``,
        ``benchmark_recipe`` (``fixture`` | ``served``),
        ``benchmark_timeout_seconds``, ``benchmark_compiles_baseline``.
        """
        recipe = config.get("benchmark_recipe", DEFAULT_BENCHMARK_RECIPE)
        blocks = int(config.get("benchmark_blocks", 3))
        if recipe not in BENCHMARK_RECIPES:
            return HarnessResult(False, f"benchmark_recipe must be one of {BENCHMARK_RECIPES}, got {recipe!r}")
        if blocks < 1:
            return HarnessResult(False, f"benchmark_blocks must be at least 1, got {blocks}")

        cases = self._load_eval_cases(set_name, config.get("benchmark_dir"))
        wanted = _label_list(config.get("benchmark_labels"))
        if wanted:
            missing = sorted(set(wanted) - {c["metadata"].get("label") for c in cases})
            if missing:
                return HarnessResult(False, f"benchmark_labels not in set {set_name!r}: {', '.join(missing)}")
            cases = [c for c in cases if c["metadata"].get("label") in wanted]
        if not cases:
            return HarnessResult(False, f"no benchmark cases in set {set_name!r}")
        references: dict[str, dict[str, float]] = {}
        reference_file = config.get("benchmark_reference_file")
        if reference_file:
            try:
                references = load_reference_times(reference_file)
            except ValueError as exc:
                return HarnessResult(False, str(exc))
        cases.sort(key=lambda c: (c["metadata"].get("residues", 0), c["metadata"]["label"]))

        timeout = int(config.get("benchmark_timeout_seconds", 3600))
        prev_total = int(config.get("benchmark_compiles_baseline", 0))
        _progress(
            f"[benchmark] set {set_name}, {recipe} recipe: {len(cases)} targets x {blocks} blocks "
            f"= {len(cases) * blocks} requests"
        )
        concurrent = concurrent_benchmark_processes()
        for other in concurrent:
            _progress(f"[benchmark] WARNING: another benchmark is running in this container ({other})")
        baseline = prev_total
        prev_by_label: dict[str, int] = {}
        first_known = True
        records: list[dict[str, Any]] = []
        for block in range(1, blocks + 1):
            for case in cases:
                meta = case["metadata"]
                reference = references.get(meta["label"])
                data = self._request_data(case["data"], recipe)
                rec: dict[str, Any] = {
                    "block": block,
                    "label": meta["label"],
                    "tokens": meta.get("residues"),
                    "reference": reference,
                    "recipe": {f: data.get(f, default) for f, default in SERVED_REQUEST_DEFAULTS.items()},
                }
                rec.update(self._time_request(service_url, data, timeout))
                rec["compiled"] = None
                if "compiles_since_warmup" in rec:
                    by_label = {
                        k: rec["compiles_since_warmup"][k] + rec["failed_compiles_since_warmup"].get(k, 0)
                        for k in rec["compiles_since_warmup"]
                    }
                    total = sum(by_label.values())
                    rec["compiled"] = total > prev_total
                    if first_known and baseline:
                        # The per-selector counts of the baseline are not known.
                        rec["new_compiles"] = {"above_baseline": total - baseline} if total > baseline else {}
                    else:
                        rec["new_compiles"] = {
                            k: v - prev_by_label.get(k, 0) for k, v in by_label.items() if v > prev_by_label.get(k, 0)
                        }
                    first_known = False
                    rec["compiles_since_warmup_total"] = total
                    prev_total, prev_by_label = total, by_label
                records.append(rec)
                _progress(_benchmark_line(rec, blocks, set_name, served=recipe == "served"))

        for other in concurrent_benchmark_processes():
            if other not in concurrent:
                concurrent.append(other)
                _progress(f"[benchmark] WARNING: another benchmark started during this run ({other})")
        concurrent_warnings = [
            f"another benchmark ran in this container ({other}); timings share the GPU and are not clean"
            for other in concurrent
        ]
        ratios = reference_ratios(records)
        n_reference = sum(1 for c in cases if c["metadata"]["label"] in references)
        served = recipe == "served"
        recipe_note = (
            "SERVED RECIPE (service defaults) is a different workload from the fixture timing recipe; "
            "the ratio is not a speed claim. "
            if served
            else ""
        )

        failed = [r for r in records if r.get("error")]

        def ratio_check(name: str, per_target: dict[str, float], what: str, warm: bool) -> CheckResult:
            if reference_file and recipe != "fixture":
                return self._not_applicable(
                    name,
                    f"the reference times are for the fixture recipe; benchmark_recipe is {recipe!r}, "
                    "so the reference ratios are not reported",
                )
            if n_reference == 0:
                return self._not_applicable(
                    name, "no benchmark_reference_file given, or it has no target of this run; timed only"
                )
            if warm and blocks < 2:
                return self._not_applicable(name, "warm timing needs benchmark_blocks >= 2")
            if not per_target:
                errors = sorted({str(r["error"]) for r in records if r.get("error")})
                return CheckResult(
                    name, CheckResultType.SCORE, False, None, f"no target produced a ratio; errors: {'; '.join(errors)}"
                )
            lo = min(per_target, key=lambda k: per_target[k])
            hi = max(per_target, key=lambda k: per_target[k])
            absent = sorted(c["metadata"]["label"] for c in cases if c["metadata"]["label"] in references)
            absent = [label for label in absent if label not in per_target]
            ok_warm = [
                sum(1 for r in records if r["label"] == label and r["block"] > 1 and not r.get("error"))
                for label in per_target
            ]
            sample = (
                f"median of blocks 2..{blocks} (n={min(ok_warm)}"
                + (f"-{max(ok_warm)}" if max(ok_warm) != min(ok_warm) else "")
                + " per target)"
                if warm
                else "block 1 request"
            )
            detail = (
                f"{recipe_note}geometric mean over {len(per_target)}/{n_reference} targets of {what}, "
                f"our time = {sample}; above 1 = faster than the reference. "
                f"per-target min {per_target[lo]:.3f} ({lo}), max {per_target[hi]:.3f} ({hi})"
            )
            if absent:
                detail += f"; no ratio for {', '.join(absent)}"
            if failed:
                detail += f"; {len(failed)} request(s) failed: " + ", ".join(
                    f"{r['label']} block {r['block']}: {r['error']}" for r in failed
                )
            return CheckResult(
                name,
                CheckResultType.SCORE,
                not absent and not failed,
                round(geometric_mean(list(per_target.values())), 3),
                detail,
            )

        modes = {r["optimizations"] for r in records if r.get("optimizations") is not None}
        states = {r["warmup_state"] for r in records if r.get("warmup_state") is not None}
        optimizations = (next(iter(modes)) if len(modes) == 1 else "mixed") if modes else None
        warmup_state = (next(iter(states)) if len(states) == 1 else "mixed") if states else None
        compile_check_skipped = optimizations == "off" or warmup_state == "disabled"

        flagged = [r for r in records if r.get("compiled")]
        unknown = [r for r in records if r.get("compiled") is None]
        if compile_check_skipped:
            compile_check = self._not_applicable(
                "compiles_after_ready",
                f"optimizations={optimizations}, warmup state={warmup_state}: there is no start-up warm-up to compile after",
            )
        elif len(unknown) == len(records):
            compile_check_skipped = True
            compile_check = self._not_applicable(
                "compiles_after_ready", "no request reported runtime.warmup.since_warmup.compiles"
            )
        else:
            compile_detail = (
                f"{len(flagged)}/{len(records)} timed requests compiled after the start-up warm-up"
                + (
                    ": " + "; ".join(f"block {r['block']} {r['label']} {r['new_compiles']}" for r in flagged)
                    if flagged
                    else ""
                )
                + f". Cumulative compiles since warm-up at the end: {prev_total}"
            )
            compile_check = CheckResult(
                "compiles_after_ready",
                CheckResultType.PASS_FAIL,
                not flagged,
                not flagged,
                compile_detail,
                warnings=(
                    [f"{len(unknown)} request(s) have no compile count; their compile state is unknown"]
                    if unknown
                    else []
                ),
            )

        ok_records = [r for r in records if not r.get("error") and r.get("wall_s") is not None]
        warm_walls = [r["wall_s"] for r in ok_records if r["block"] > 1]
        if ok_records:
            span = sum(r["wall_s"] for r in ok_records)
            per_min = 60.0 * len(ok_records) / span if span > 0 else 0.0
            throughput_check = CheckResult(
                "throughput",
                CheckResultType.SCORE,
                True,
                round(per_min, 2),
                f"{len(ok_records)}/{len(records)} successful timed requests of all blocks (mixed target sizes) in "
                f"{span:.1f} s of client time = {per_min:.2f} structures/min",
            )
        else:
            throughput_check = CheckResult(
                "throughput", CheckResultType.SCORE, False, None, "no timed request succeeded"
            )
        if blocks < 2:
            p50_check = self._not_applicable("p50_ms", "warm requests need benchmark_blocks >= 2")
        elif warm_walls:
            ordered = sorted(warm_walls)
            p95 = ordered[min(len(ordered) - 1, int(round(0.95 * (len(ordered) - 1))))]
            p50 = statistics.median(warm_walls)
            p50_check = CheckResult(
                "p50_ms",
                CheckResultType.SCORE,
                True,
                round(p50 * 1000.0, 1),
                f"median client time of {len(warm_walls)} successful warm requests (blocks 2..{blocks}, mixed target "
                f"sizes) = {p50 * 1000.0:.1f} ms, p95 = {p95 * 1000.0:.1f} ms",
            )
        else:
            p50_check = CheckResult("p50_ms", CheckResultType.SCORE, False, None, "no warm request succeeded")

        requests_ok = CheckResult(
            "requests_ok",
            CheckResultType.PASS_FAIL,
            not failed,
            not failed,
            f"{len(records) - len(failed)}/{len(records)} timed requests succeeded"
            + (": " + "; ".join(f"{r['label']} block {r['block']}: {r['error']}" for r in failed) if failed else ""),
            warnings=concurrent_warnings,
        )
        checks = [
            throughput_check,
            p50_check,
            ratio_check("reference_ratio_warm", ratios["warm"], "reference time / our time", True),
            ratio_check("reference_ratio_first", ratios["first"], "reference time / our time", False),
            ratio_check("reference_alt_ratio_warm", ratios["alt_warm"], "second reference time / our time", True),
            requests_ok,
            compile_check,
        ]
        compile_check.warnings.extend(concurrent_warnings)
        success = all(c.success for c in checks if not c.skipped)

        succeeded = len(records) - len(failed)
        peaks = [r["cuda_peak_gb"] for r in records if r.get("cuda_peak_gb") is not None]
        metrics: dict[str, Any] = {
            "benchmark_set": set_name,
            "recipe_name": recipe,
            "recipe": records[0]["recipe"],
            "recipe_note": recipe_note.strip() or "fixture recipe",
            "optimizations": optimizations,
            "warmup_state": warmup_state,
            "blocks": blocks,
            "targets_total": len(cases),
            "requests_total": len(records),
            "requests_succeeded": succeeded,
            "requests_failed": len(failed),
            "compiles_after_ready_requests": None if compile_check_skipped else len(flagged),
            "compiles_since_warmup_final": None if compile_check_skipped else prev_total,
            "cuda_peak_gb_max": max(peaks) if peaks else None,
        }
        if ratios["first_s"]:
            metrics["first_s_sum"] = round(sum(ratios["first_s"].values()), 2)
            metrics["targets_in_first_sum"] = len(ratios["first_s"])
        if ratios["warm_s"]:
            metrics["warm_median_s_sum"] = round(sum(ratios["warm_s"].values()), 2)
            metrics["targets_in_warm_sum"] = len(ratios["warm_s"])
        records_path = self._write_records(
            config,
            BENCHMARK_RECORDS_FILE,
            {
                "set": set_name,
                "recipe_name": recipe,
                "recipe": metrics["recipe"],
                "optimizations": optimizations,
                "warmup_state": warmup_state,
                "blocks": blocks,
                "records": records,
            },
        )
        artifacts = [records_path] if records_path else []

        warm = next(c for c in checks if c.name == "reference_ratio_warm")
        first = next(c for c in checks if c.name == "reference_ratio_first")
        issues = []
        if failed:
            issues.append(f"{len(failed)} REQUEST(S) FAILED")
        if flagged:
            issues.append(f"NOT CLEAN: {len(flagged)} request(s) compiled")
        if concurrent:
            issues.append(f"{len(concurrent)} CONCURRENT BENCHMARK PROCESS(ES)")
        if compile_check_skipped:
            issues.append("compile check skipped")
        elif len(unknown) > len(failed):
            issues.append(f"compile state unknown for {len(unknown) - len(failed)} request(s)")
        summary = (
            f"Boltz-2 {set_name} benchmark ({recipe} recipe"
            f"{', DIFFERENT WORKLOAD, not a speed claim' if served else ''}, optimizations={optimizations}): "
            f"reference ratio warm {warm.value if warm.value is not None else 'n/a'}x, "
            f"first {first.value if first.value is not None else 'n/a'}x, "
            f"{succeeded}/{len(records)} requests ok, {'; '.join(issues) or 'clean'}"
        )
        return HarnessResult(success=success, summary=summary, checks=checks, metrics=metrics, artifacts=artifacts)

    # ------------------------------------------------------------------ #
    # evaluate — "does it produce correct structures?"
    # ------------------------------------------------------------------ #

    def evaluate(self, config: HarnessConfig) -> HarnessResult:
        service_url = self._service_url(config)

        if not self.health_check(service_url, timeout_seconds=config.timeout_seconds):
            return HarnessResult(success=False, summary="BentoML service not reachable for evaluation")

        checks: list[CheckResult] = []
        checks.append(self._check_structure_sanity(service_url))

        accuracy_check, per_case = self._check_structure_accuracy(service_url, config)
        for c in per_case:
            for note in c.get("rmsd_notes", []):
                accuracy_check.warnings.append(f"{c['label']}: {note}")
        lost = [c["label"] for c in per_case if c.get("reference_missing")]
        if lost:
            accuracy_check.warnings.append(
                f"reference structure not staged for {', '.join(lost)}; predicted, RMSD not available"
            )
        checks.append(accuracy_check)

        # Only gate confidence when there was something the gate may judge. Two
        # separate "nothing to judge" cases, and neither may redden the build:
        # no eval cases staged at all (per_case empty), and cases staged that run
        # a configuration these floors were not derived for (gradeable empty).
        gradeable = [c for c in per_case if c.get("accuracy_gradeable")]
        if gradeable:
            plddt_ok, mean_plddt, plddt_detail = grade_confidence(gradeable)
            checks.append(
                CheckResult(
                    name="confidence_regression",
                    result_type=CheckResultType.SCORE,
                    success=plddt_ok,
                    value=round(mean_plddt, 4) if mean_plddt is not None else None,
                    detail=plddt_detail,
                )
            )
        elif per_case:
            checks.append(
                CheckResult(
                    name="confidence_regression",
                    result_type=CheckResultType.SCORE,
                    success=True,
                    value=None,
                    detail=self._gate_abstention_detail(per_case),
                    skipped=True,
                )
            )

        success = all(c.success for c in checks if not c.skipped)
        scored = [c for c in per_case if c.get("rmsd_a") is not None]
        graded_scored = [c for c in scored if c.get("accuracy_gradeable")]
        metrics: dict[str, Any] = {
            "sequence": EVAL_SEQUENCE,
            "sequence_length": len(EVAL_SEQUENCE),
            "cases": per_case,
            "cases_scored": len(scored),
            "cases_total": len(per_case),
            "accuracy_gate": {
                "applied": bool(graded_scored),
                "recipe": dict(ACCURACY_GATE_RECIPE),
                "cases_graded": len(graded_scored),
                "cases_ungraded": len(scored) - len(graded_scored),
                "detail": self._gate_abstention_detail(per_case) if not graded_scored else "",
            },
        }

        # mean_ca_rmsd_a / mean_complex_plddt are the keys the Tuner reads and
        # ranks on, so they carry ONLY gate-valid numbers. When the gate did not
        # apply they are absent rather than filled from ungraded cases: a caller
        # comparing an ungraded value against the 0.90 floor would read a
        # configuration change as a regression, which is exactly the failure this
        # split exists to stop. The measurements are still reported, under keys
        # that say what they are.
        answered = [
            c
            for c in per_case
            if not c.get("error") and (c.get("rmsd_a") is not None or c.get("complex_plddt") is not None)
        ]
        _summarize(metrics, [c for c in answered if c.get("accuracy_gradeable")], "")
        _summarize(metrics, [c for c in answered if not c.get("accuracy_gradeable")], "ungraded_")
        metrics["cases_predicted"] = len(answered)

        records_path = self._write_records(
            config,
            EVALUATE_RECORDS_FILE,
            {"set": config.get("eval_set") or BENCHMARK_SET, "cases": per_case},
        )

        return HarnessResult(
            success=success,
            summary=f"Boltz-2 evaluation {'passed' if success else 'failed'}: "
            f"{len(answered)}/{len(per_case)} cases predicted, {len(scored)} with RMSD",
            checks=checks,
            metrics=metrics,
            artifacts=[records_path] if records_path else [],
        )

    @staticmethod
    def _gate_abstention_detail(per_case: list[dict[str, Any]]) -> str:
        """Why the accuracy gate declined to judge this set of cases."""
        fields: dict[str, set[str]] = {}
        for record in per_case:
            for field, seen in (record.get("accuracy_recipe_mismatch") or {}).items():
                fields.setdefault(field, set()).add(f"{seen['case']!r} (gate needs {seen['gate']!r})")
        if not fields:
            return "no evaluation cases carried a request configuration to compare against the accuracy gate recipe"
        rendered = "; ".join(f"{field}: {', '.join(sorted(values))}" for field, values in sorted(fields.items()))
        return (
            f"GATE NOT APPLIED to {len(per_case)} case(s): they do not run the configuration these floors were "
            f"derived under. Mismatching fields — {rendered}. The floors (mean {PLDDT_MEAN_FLOOR:.2f}, per-case "
            f"{PLDDT_PER_CASE_FLOOR:.2f}, C-alpha RMSD {RMSD_REGRESSION_THRESHOLD_A:.1f} A) were derived with an MSA "
            "at boltz's default sampling schedule; applying them here would report a request-configuration "
            "difference as a model regression. Measured values are reported under the ungraded_* metrics."
        )

    # ------------------------------------------------------------------ #
    # Internal helpers
    # ------------------------------------------------------------------ #

    @staticmethod
    def _service_url(config: HarnessConfig) -> str:
        return config.resolve_service_url()

    @staticmethod
    def _post_json(url: str, payload: dict[str, Any], timeout: int = 3600) -> str:
        """POST ``payload`` as JSON and return the raw response body."""
        req = Request(
            url,
            data=json.dumps(payload).encode(),
            headers={"Content-Type": "application/json"},
        )
        with urlopen(req, timeout=timeout) as resp:
            return resp.read().decode()

    @staticmethod
    def _post_predict(service_url: str, payload: dict[str, Any]) -> dict[str, Any]:
        """POST a prediction request and return the parsed answer.

        A non-2xx answer raises ``HTTPError`` from ``urlopen``; a 2xx answer
        without a structure raises ``RuntimeError``, so no caller counts it as a structure.
        """
        req = Request(
            f"{service_url}{BENTOML_INFERENCE_ENDPOINT}",
            data=json.dumps(payload).encode(),
            headers={"Content-Type": "application/json"},
        )
        with urlopen(req, timeout=300) as resp:
            result = json.loads(resp.read().decode())
        if not result.get("structures"):
            raise RuntimeError(f"service answered the prediction without a structure: {str(result)[:300]!r}")
        return dict(result)

    @classmethod
    def _send_predict(cls, service_url: str, sequence: str, **kwargs: Any) -> str:
        """Send a prediction request and return the structure document.

        ``output_format`` defaults to ``pdb``.
        """
        result = cls._post_predict(service_url, _build_predict_payload(sequence, **kwargs))
        return str(result["structures"][0]["content"])

    @classmethod
    def _run_predict_smoke(cls, service_url: str) -> tuple[CheckResult, str | None]:
        """The ``predict_smoke`` check, and the ``BOLTZ2_MODE`` the service reports in ``runtime.mode``.

        This is a validate-level check: it only verifies the prediction
        pipeline runs end-to-end and returns atom records. It doesn't assess
        the quality of the predicted structure (that's evaluate's job).
        """
        mode: str | None = None
        try:
            result = cls._post_predict(service_url, _build_predict_payload(SMOKE_SEQUENCE, name="smoke_test"))
            mode = (result.get("runtime") or {}).get("mode")
            body = str(result["structures"][0]["content"])
        except Exception as exc:
            return cls._smoke_result(False, str(exc)), mode

        atoms = sum(1 for line in body.splitlines() if line.startswith(("ATOM", "HETATM")))
        if atoms:
            return cls._smoke_result(True, f"Prediction returned {len(body)} bytes with {atoms} atom records"), mode
        detail = (
            "Prediction returned empty response"
            if not body
            else f"Prediction returned {len(body)} bytes with no ATOM/HETATM record: {body[:200]!r}"
        )
        return cls._smoke_result(False, detail), mode

    @staticmethod
    def _smoke_result(success: bool, detail: str, name: str = "predict_smoke") -> CheckResult:
        return CheckResult(
            name=name, result_type=CheckResultType.PASS_FAIL, success=success, value=success, detail=detail
        )

    @classmethod
    def _affinity_smoke_test(cls, service_url: str) -> CheckResult:
        """Predict a protein with one ligand and verify the answer has finite affinity values."""
        name = "predict_affinity_smoke"
        payload = {
            "data": {
                "version": 1,
                "name": "smoke_affinity",
                "sequences": [
                    {"protein": {"id": "A", "sequence": SMOKE_SEQUENCE}},
                    {"ligand": {"id": "B", "smiles": AFFINITY_SMOKE_LIGAND}},
                ],
                "properties": [{"affinity": {"binder": "B"}}],
                "output_format": "pdb",
                **AFFINITY_SMOKE_PARAMS,
            }
        }
        try:
            affinity = cls._post_predict(service_url, payload).get("affinity") or {}
        except Exception as exc:
            return cls._smoke_result(False, str(exc), name)
        values = {k: affinity.get(k) for k in ("affinity_pred_value", "affinity_probability_binary")}
        bad = [
            k
            for k, v in values.items()
            if isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v)
        ]
        if bad:
            return cls._smoke_result(False, f"the answer has no finite value for {', '.join(bad)}: {affinity!r}", name)
        return cls._smoke_result(True, f"affinity {values}", name)

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

    # ------------------------------------------------------------------ #
    # Accuracy anchor — C-alpha RMSD against experimental structures
    # ------------------------------------------------------------------ #

    @classmethod
    def _load_eval_cases(cls, set_name: str | None = None, directory: str | Path | None = None) -> list[dict[str, Any]]:
        """Load the baked evaluation cases (``benchmarks/<id>/<id>.json``).

        Each case carries the Boltz-2 request under ``data`` and, under
        ``metadata``, the PDB id whose experimental structure it is scored
        against. A case whose reference has not been staged is still predicted: it
        reports confidence and time, and no RMSD.

        Only cases whose ``metadata.sets`` contains the set are returned. The
        set and directory default to :data:`BENCHMARK_SET` and
        :data:`BENCHMARK_DIR` (the environment); a caller may name others. A case that declares no ``sets`` joins NO set and is skipped
        loudly: defaulting it into the current set is the one behaviour that
        could widen the published baseline by accident, which is exactly what
        naming the sets was meant to prevent.
        """
        set_name = set_name or BENCHMARK_SET
        root = Path(directory) if directory else BENCHMARK_DIR
        if not root.is_dir():
            return []
        cases: list[dict[str, Any]] = []
        for case_dir in sorted(root.iterdir()):
            case_file = case_dir / f"{case_dir.name}.json"
            if not case_file.is_file():
                continue
            try:
                case = json.loads(case_file.read_text())
            except (OSError, json.JSONDecodeError) as exc:
                logger.warning("Skipping unreadable evaluation case %s: %s", case_file, exc)
                continue
            sets = case.get("metadata", {}).get("sets")
            if not sets:
                logger.warning(
                    "Evaluation case %s declares no metadata.sets, so it belongs to no set "
                    "and is skipped. Add %r to include it in the default set.",
                    case_file,
                    DEFAULT_BENCHMARK_SET,
                )
                continue
            if set_name in sets:
                cases.append(case)
        return cases

    @staticmethod
    def _ca_rmsd(predicted_mmcif: str, reference_path: Path, notes: list[str] | None = None) -> tuple[float, int]:
        """Return (C-alpha RMSD in Angstrom, number of aligned pairs).

        Superposes each predicted polymer chain onto every reference polymer
        chain with gemmi and keeps, per predicted chain, the pairing that aligns
        the most residues — gemmi aligns by sequence, so this resolves chain
        naming and ordering differences without assuming the two files label
        their chains identically. The returned RMSD is the aligned-pair-weighted
        mean across predicted chains.

        gemmi ships with Boltz-2 (it is a declared dependency), so this needs no
        extra package in the image. Imported lazily so the CPU unit tier can
        import this module without it.
        """
        import gemmi

        def _polymers(structure: Any) -> list[Any]:
            if len(structure) == 0:
                return []
            structure.setup_entities()
            structure.remove_ligands_and_waters()
            structure.remove_hydrogens()
            # The chain objects must outlive this call: gemmi polymers are views
            # into the parent Structure, so the caller keeps `structure` alive by
            # holding the returned list only for the duration of the comparison.
            return [p for p in (chain.get_polymer() for chain in structure[0]) if len(p) > 0]

        # cif.read_string + make_structure_from_block is the parse path that
        # exists in BOTH the gemmi 0.6.5 the image pins (via Boltz-2) and the
        # newer releases a host venv is likely to install. gemmi's one-shot
        # string parser is spelled differently across those versions
        # (`read_structure_string` only appeared later), so going through the
        # CIF document keeps this working either way.
        pred_st = gemmi.make_structure_from_block(gemmi.cif.read_string(predicted_mmcif).sole_block())
        ref_st = gemmi.read_structure(str(reference_path))
        predicted = _polymers(pred_st)
        reference = _polymers(ref_st)
        if not predicted or not reference:
            raise ValueError("no polymer chains to superpose")

        total_sq, total_n = 0.0, 0
        hit_limit = 0
        for pred_chain in predicted:
            candidates = []
            for ref_chain in reference:
                with _native_output_captured() as native:
                    try:
                        sup = gemmi.calculate_superposition(
                            pred_chain, ref_chain, pred_chain.check_polymer_type(), gemmi.SupSelect.CaP
                        )
                    except Exception:  # noqa: BLE001 — an unpairable chain is not an error
                        sup = None
                hit_limit += sum(QCP_LIMIT_MESSAGE in line for line in native)
                for line in native:
                    if QCP_LIMIT_MESSAGE not in line and line.strip():
                        _progress(line)
                if sup is None:
                    continue
                # A degenerate chain (collinear or near-collinear C-alphas)
                # makes gemmi's superposition solver fail to converge and return
                # NaN. Dropping those here keeps one bad chain from turning the
                # whole evaluation's mean RMSD into NaN.
                if sup.count > 0 and sup.rmsd == sup.rmsd and sup.rmsd != float("inf"):
                    candidates.append(sup)
            if not candidates:
                continue
            # Pick by lowest RMSD among the pairings that align a comparable
            # number of residues. Ranking by residue count alone mis-pairs
            # homomultimers: in a homotetramer every copy aligns to every other
            # copy, so the longest alignment can be a *different* copy than the
            # matching one, and a structure scored against itself then reports a
            # non-zero RMSD.
            widest = max(s.count for s in candidates)
            viable = [s for s in candidates if s.count >= 0.5 * widest]
            best = min(viable, key=lambda s: s.rmsd)
            total_sq += (best.rmsd**2) * best.count
            total_n += best.count

        if hit_limit and notes is not None:
            notes.append(
                f"gemmi's superposition solver reached its 50-iteration limit for {hit_limit} chain pair(s); "
                "a pair with a non-finite RMSD is dropped, the others keep the solver's last estimate"
            )
        if total_n == 0:
            raise ValueError("no residues could be aligned between prediction and reference")
        return (total_sq / total_n) ** 0.5, total_n

    @classmethod
    def _check_structure_accuracy(
        cls, service_url: str, config: HarnessConfig
    ) -> tuple[CheckResult, list[dict[str, Any]]]:
        """Predict every staged evaluation case and score it against its reference.

        Every case is predicted; every case with a staged reference is scored. The VERDICT, though, is
        rendered only over cases whose request matches
        :data:`ACCURACY_GATE_RECIPE` — see that constant for why. When no case
        matches, the check reports the measured mean and is marked ``skipped``
        with the mismatching fields named, rather than failing a build for
        running the configuration it was asked to run.

        Read ``RMSD_REGRESSION_THRESHOLD_A``'s comment before treating the number
        as an accuracy claim.
        """
        reference_dir = DATA_DIR / REFERENCE_SUBDIR
        set_name = config.get("eval_set") or BENCHMARK_SET
        directory = config.get("eval_dir") or BENCHMARK_DIR
        cases = cls._load_eval_cases(set_name, directory)
        per_case: list[dict[str, Any]] = []

        if not cases:
            return (
                CheckResult(
                    name="structure_accuracy",
                    result_type=CheckResultType.SCORE,
                    success=True,
                    value=None,
                    detail=(f"no evaluation cases in set {set_name!r} under {directory}"),
                    skipped=True,
                ),
                per_case,
            )

        timeout = int(config.get("evaluate_timeout_seconds", 3600))
        for index, case in enumerate(cases, 1):
            meta = case.get("metadata", {})
            label = meta.get("label", "?")
            record: dict[str, Any] = {"label": label, "rmsd_a": None, "aligned_residues": None}

            mismatch = accuracy_recipe_mismatch(case.get("data", {}))
            record["accuracy_gradeable"] = not mismatch
            if mismatch:
                record["accuracy_recipe_mismatch"] = mismatch

            reference_path = reference_dir / f"{label}.cif"
            has_reference = reference_path.is_file()
            if not has_reference:
                record["reference"] = f"not staged at {reference_path}; RMSD not available"
                # A target with no PDB id has no reference by design; one with
                # an id and no staged file is a deployment fault.
                record["reference_missing"] = bool(meta.get("reference_pdb_id"))

            started = time.perf_counter()
            try:
                body = cls._post_json(
                    f"{service_url}{BENTOML_INFERENCE_ENDPOINT}",
                    {"data": {**case["data"], "output_format": "mmcif"}},
                    timeout=timeout,
                )
                record["wall_s"] = _elapsed_s(started)
                payload = json.loads(body)
                if payload.get("error") or not payload.get("structures"):
                    record["error"] = payload.get("message", "prediction returned no structure")
                    per_case.append(record)
                    _progress(_evaluate_line(record, index, len(cases)))
                    continue

                structure = payload["structures"][0]
                record["residues"] = meta.get("residues")
                peak_gb = (payload.get("runtime") or {}).get("cuda_peak_gb")
                if peak_gb is not None:
                    record["cuda_peak_gb"] = peak_gb
                confidence = payload.get("confidence", {}).get(structure["record_id"], {})
                if "complex_plddt" in confidence:
                    record["complex_plddt"] = round(confidence["complex_plddt"], 4)
                if "ptm" in confidence:
                    record["ptm"] = round(confidence["ptm"], 4)
                runtime = payload.get("timing", {}).get("runtime_s")
                if runtime is not None:
                    record["runtime_s"] = runtime
                if has_reference:
                    notes: list[str] = []
                    try:
                        rmsd, aligned = cls._ca_rmsd(structure["content"], reference_path, notes=notes)
                    finally:
                        if notes:
                            record["rmsd_notes"] = notes
                    record["rmsd_a"] = round(rmsd, 3)
                    record["aligned_residues"] = aligned
            except Exception as exc:  # noqa: BLE001 — recorded per case, not raised
                record["error"] = f"{type(exc).__name__}: {exc}"
                for key in ("complex_plddt", "ptm"):
                    record.pop(key, None)
            per_case.append(record)
            _progress(_evaluate_line(record, index, len(cases)))

        scored = [r for r in per_case if r["rmsd_a"] is not None]
        if not scored and not any(r.get("error") or r.get("reference_missing") for r in per_case):
            return (
                CheckResult(
                    name="structure_accuracy",
                    result_type=CheckResultType.SCORE,
                    success=True,
                    value=None,
                    detail=(
                        f"0/{len(per_case)} cases have a staged reference under {reference_dir}; "
                        "all were predicted, confidence only"
                    ),
                    skipped=True,
                ),
                per_case,
            )
        if not scored:
            reasons = "; ".join(sorted({r.get("error") or r.get("reference", "unknown") for r in per_case}))
            return (
                CheckResult(
                    name="structure_accuracy",
                    result_type=CheckResultType.SCORE,
                    success=False,
                    value=None,
                    detail=f"0/{len(per_case)} evaluation cases scored: {reasons}",
                ),
                per_case,
            )

        graded = [r for r in scored if r.get("accuracy_gradeable")]
        if not graded:
            ungraded_mean = sum(r["rmsd_a"] for r in scored) / len(scored)
            return (
                CheckResult(
                    name="structure_accuracy",
                    result_type=CheckResultType.SCORE,
                    success=True,
                    value=None,
                    detail=(
                        f"{len(scored)}/{len(per_case)} cases scored, 0 gradeable. "
                        f"Measured mean C-alpha RMSD={ungraded_mean:.2f} A, NOT compared against the "
                        f"{RMSD_REGRESSION_THRESHOLD_A:.1f} A threshold. " + cls._gate_abstention_detail(per_case)
                    ),
                    skipped=True,
                ),
                per_case,
            )

        mean_rmsd = sum(r["rmsd_a"] for r in graded) / len(graded)
        worst = max(graded, key=lambda r: r["rmsd_a"])
        passed = mean_rmsd <= RMSD_REGRESSION_THRESHOLD_A
        detail = (
            f"{len(graded)}/{len(scored)} scored cases gradeable ({len(scored)}/{len(per_case)} scored); "
            f"mean C-alpha RMSD={mean_rmsd:.2f} A, threshold={RMSD_REGRESSION_THRESHOLD_A:.1f} A "
            f"=> {'PASS' if passed else 'FAIL'}; "
            f"worst={worst['label']} at {worst['rmsd_a']:.2f} A over {worst['aligned_residues']} residues"
        )
        return (
            CheckResult(
                name="structure_accuracy",
                result_type=CheckResultType.SCORE,
                success=passed,
                value=round(mean_rmsd, 3),
                detail=detail,
            ),
            per_case,
        )
