# Copyright © Advanced Micro Devices, Inc., or its affiliates.
#
# SPDX-License-Identifier: MIT

"""
VLLMHarness — batteries-included harness for standard vLLM-based AIM images.

Wraps the existing benchmarking (``AIMBenchmark`` / ``vllm bench serve``),
validation (OpenAI API health, endpoint, tool calling, structured output),
and accuracy evaluation (``AIMEvaluation`` / ``lm-eval``) tooling, each of
which the CI pipeline drives through the same code.

Validation scope (per profile):
  Runtime checks (no service required, safe on mounted profiles):
    - profile_schema:       Pydantic schema validation
    - engine_validation:    vLLM-specific argument validation (native engine checks)

  Offline checks (require a live service; mirror the ``/validate`` workflow):
    - api_health:                 Service serves a model, measures readiness time
    - warmup:                     First inference succeeds, surfaces JIT cold-start
    - completions_endpoint:       ``/v1/completions`` is OpenAI-compatible
    - chat_completions_endpoint:  ``/v1/chat/completions`` is OpenAI-compatible
    - tool_invocation:            Tool calling returns a tool call
    - tool_avoidance:             Regular chat does not leak tool calls
    - structured_output:          Constrained decoding, flat schema
    - structured_output_nested:   Constrained decoding, nested schema
    - structured_output_choice:   Constrained decoding, choice from a fixed set
    - reasoning:                  Reasoning prompts produce output

The three capability checks (tool calling, structured outputs, reasoning) only
run when the profile declares the matching flag under ``metadata.capabilities``;
otherwise they are skipped. CI can afford to run them unconditionally because it
reports a separate flag per family, whereas this harness returns one exit code
and must not fail a model for a capability it never claimed.

Planned but not yet implemented. These are deliberately kept out of ``CHECKS``
so ``aim-runtime list-checks`` only advertises checks CI can actually run:
    - functional:           Profile starts and doesn't crash
    - eval_baseline:        Evaluation score doesn't deteriorate vs baseline profile
"""

from __future__ import annotations

import functools
import logging
from contextlib import ExitStack
from tempfile import TemporaryDirectory
from typing import TYPE_CHECKING, Any, ClassVar, Protocol

from aim_common.object_model import AcceleratorType, Engine
from aim_runtime.harness import (
    STARTUP_READY_TIME_SECONDS_KEY,
    CheckInfo,
    CheckResult,
    CheckResultType,
    CheckScope,
    HarnessConfig,
    HarnessResult,
    ModelHarness,
    failed,
    passed,
    skipped,
)
from aim_runtime.harness.service_checks import (
    JIT_LATENCY_THRESHOLD_S_DEFAULT,
    MAX_WARMUP_TIME_DEFAULT,
    check_chat_completions_endpoint,
    check_completions_endpoint,
    check_reasoning,
    check_structured_output,
    check_structured_output_choice,
    check_structured_output_nested,
    check_tool_avoidance,
    check_tool_invocation,
    probe_api_health,
    run_warmup,
)

if TYPE_CHECKING:
    from aim_runtime.evaluation.results import EvaluationResults, TaskScore
    from aim_runtime.evaluation.runner import AIMEvaluation
    from aim_runtime.evaluation.settings import EvaluationSettings

logger = logging.getLogger(__name__)


class ServiceCheck(Protocol):
    """Signature every capability check in ``service_checks`` shares.

    ``timeout`` is keyword-only so a check is free to take extra positional
    parameters of its own (``check_reasoning`` takes the case list) without
    changing the call shape :func:`_guarded` relies on.
    """

    def __call__(self, service_url: str, model_id: str, *, timeout: float) -> CheckResult: ...


def _guarded(name: str, check: ServiceCheck, service_url: str, model_id: str, timeout: float) -> CheckResult:
    """Run a service check, recording an unexpected exception as its failure.

    Keeps one malformed response from aborting the checks that follow it.
    """
    try:
        return check(service_url, model_id, timeout=timeout)
    except Exception as exc:
        logger.exception("Check %r raised", name)
        return failed(name, f"Check raised {type(exc).__name__}: {exc}")


class VLLMHarness(ModelHarness):
    """Default harness for vLLM-based AIM images.

    Each method delegates to the existing tooling already installed in the
    image (``aim_runtime.benchmarking.AIMBenchmark``, OpenAI API helpers).
    """

    #: Engines this harness can drive. Both speak the OpenAI-compatible API
    #: these checks assume; discovery refuses to fall back to this harness for
    #: anything else.
    SUPPORTED_ENGINES: ClassVar[frozenset[str]] = frozenset({Engine.VLLM, Engine.VLLM_OMNI})

    CHECKS: ClassVar[list[CheckInfo]] = [
        CheckInfo("profile_schema", CheckResultType.PASS_FAIL, CheckScope.RUNTIME, "Pydantic schema validation"),
        CheckInfo("engine_validation", CheckResultType.PASS_FAIL, CheckScope.RUNTIME, "vLLM-specific argument checks"),
        CheckInfo("api_health", CheckResultType.PASS_FAIL, CheckScope.OFFLINE, "Service serves a model"),
        CheckInfo("warmup", CheckResultType.PASS_FAIL, CheckScope.OFFLINE, "First inference succeeds"),
        CheckInfo(
            "completions_endpoint",
            CheckResultType.PASS_FAIL,
            CheckScope.OFFLINE,
            "/v1/completions is OpenAI-compatible",
        ),
        CheckInfo(
            "chat_completions_endpoint",
            CheckResultType.PASS_FAIL,
            CheckScope.OFFLINE,
            "/v1/chat/completions is OpenAI-compatible",
        ),
        CheckInfo("tool_invocation", CheckResultType.PASS_FAIL, CheckScope.OFFLINE, "Tool/function calling works"),
        CheckInfo(
            "tool_avoidance", CheckResultType.PASS_FAIL, CheckScope.OFFLINE, "Regular chat does not leak tool calls"
        ),
        CheckInfo(
            "structured_output", CheckResultType.PASS_FAIL, CheckScope.OFFLINE, "Constrained decoding, flat schema"
        ),
        CheckInfo(
            "structured_output_nested",
            CheckResultType.PASS_FAIL,
            CheckScope.OFFLINE,
            "Constrained decoding, nested schema",
        ),
        CheckInfo(
            "structured_output_choice",
            CheckResultType.PASS_FAIL,
            CheckScope.OFFLINE,
            "Constrained decoding, choice from a fixed set",
        ),
        CheckInfo("reasoning", CheckResultType.PASS_FAIL, CheckScope.OFFLINE, "Reasoning prompts produce output"),
    ]

    def list_checks(self) -> list[CheckInfo]:
        return list(self.CHECKS)

    def validate(self, config: HarnessConfig) -> HarnessResult:
        checks: list[CheckResult] = []
        metrics: dict[str, Any] = {}

        if CheckScope.RUNTIME in config.check_scopes:
            checks += [
                self._check_profile_schema(config),
                self._check_engine_validation(config),
            ]

        if CheckScope.OFFLINE in config.check_scopes:
            offline_checks, metrics = self._run_service_checks(config)
            checks.extend(offline_checks)

        success, summary = self._summarize(checks)
        return HarnessResult(success=success, summary=summary, checks=checks, metrics=metrics)

    @staticmethod
    def _summarize(checks: list[CheckResult]) -> tuple[bool, str]:
        """Reduce the collected checks to a verdict and a one-line summary.

        A failed precondition is recorded as a failed check rather than silently
        dropped, so the checks that never ran can't vanish from the verdict and
        leave the run reporting a vacuous "passed".
        """
        if not checks:
            return False, "No validation checks ran"

        failures = [c for c in checks if not c.success and not c.skipped]
        skips = [c for c in checks if c.skipped]
        if failures:
            return False, f"{len(failures)} of {len(checks)} validation check(s) failed: " + ", ".join(
                c.name for c in failures
            )

        parts = [f"{len(checks) - len(skips)} validation check(s) passed"]
        if skips:
            parts.append(f"{len(skips)} skipped")
        if warning_count := sum(len(c.warnings) for c in checks):
            parts.append(f"{warning_count} warning(s)")
        return True, ", ".join(parts)

    def _run_service_checks(self, config: HarnessConfig) -> tuple[list[CheckResult], dict[str, Any]]:
        """Run the live-service checks, keeping partial results if one blows up.

        The checks handle :class:`ServiceError` themselves, but not every way a
        service can misbehave: a 200 whose body isn't shaped the way the OpenAI
        schema promises raises out of the parsing code. Without a boundary here
        that one response would discard every result collected before it, so
        ``checks`` and ``metrics`` accumulate as phases complete and an
        unexpected failure is recorded as its own failed check.
        """
        checks: list[CheckResult] = []
        metrics: dict[str, Any] = {}
        try:
            self._run_service_check_phases(config, checks, metrics)
        except Exception as exc:
            logger.exception("Service validation aborted after %d check(s)", len(checks))
            checks.append(failed("service_checks", f"Validation aborted by {type(exc).__name__}: {exc}"))
        return checks, metrics

    def _run_service_check_phases(
        self,
        config: HarnessConfig,
        checks: list[CheckResult],
        metrics: dict[str, Any],
    ) -> None:
        """Run the live-service checks in dependency order, filling ``checks`` and ``metrics`` as it goes.

        Each phase gates the next: nothing can be validated before a model is
        served, and behavioural checks would only time out against a service
        that cannot answer a trivial request.

        ``config.timeout_seconds`` is both the readiness budget and the
        per-request timeout of the checks that follow, so ``--timeout`` bounds
        every request the harness makes. Warmup is the exception: kernel JIT
        makes the first inference an order of magnitude slower than the rest, so
        it keeps its own ``max_warmup_time`` budget.
        """
        service_url = config.resolve_service_url()
        timeout = float(config.timeout_seconds)

        health = probe_api_health(service_url, timeout_seconds=config.timeout_seconds)
        checks.append(health.check)
        pre_measured_ready_time = config.get(STARTUP_READY_TIME_SECONDS_KEY)
        metrics["ready_time_seconds"] = (
            pre_measured_ready_time if pre_measured_ready_time is not None else health.ready_time_seconds
        )
        if health.model_id is None:
            return

        warmup = run_warmup(
            service_url,
            health.model_id,
            max_warmup_time=config.get("max_warmup_time", MAX_WARMUP_TIME_DEFAULT),
            jit_threshold_seconds=config.get("jit_latency_threshold_s", JIT_LATENCY_THRESHOLD_S_DEFAULT),
        )
        checks.append(warmup.check)
        metrics.update(
            warmup_time_seconds=warmup.elapsed_seconds,
            warmup_attempts=warmup.attempts,
            warmup_succeeded=warmup.succeeded,
            jit_suspected=warmup.jit_suspected,
        )
        if not warmup.succeeded:
            return

        chat_completions_check = functools.partial(
            check_chat_completions_endpoint, reasoning_enabled=config.has_capability("reasoning")
        )
        endpoints = [
            _guarded("completions_endpoint", check_completions_endpoint, service_url, health.model_id, timeout),
            _guarded("chat_completions_endpoint", chat_completions_check, service_url, health.model_id, timeout),
        ]
        checks.extend(endpoints)
        if any(not check.success for check in endpoints):
            logger.error("API endpoint validation failed — skipping capability checks")
            return

        checks.extend(self._capability_checks(config, service_url, health.model_id, timeout))

    @staticmethod
    def _capability_checks(config: HarnessConfig, service_url: str, model_id: str, timeout: float) -> list[CheckResult]:
        """Run the checks whose capability the profile declares, skip the rest."""

        def gated(capability: str, *runners: tuple[str, ServiceCheck]) -> list[CheckResult]:
            declared = config.has_capability(capability)
            reason = f"Profile does not declare the {capability} capability"
            return [
                _guarded(name, runner, service_url, model_id, timeout) if declared else skipped(name, reason)
                for name, runner in runners
            ]

        return [
            *gated(
                "tool_calling",
                ("tool_invocation", check_tool_invocation),
                ("tool_avoidance", check_tool_avoidance),
            ),
            *gated(
                "structured_outputs",
                ("structured_output", check_structured_output),
                ("structured_output_nested", check_structured_output_nested),
                ("structured_output_choice", check_structured_output_choice),
            ),
            *gated("reasoning", ("reasoning", check_reasoning)),
        ]

    def benchmark(self, config: HarnessConfig) -> HarnessResult:
        """Delegate to :class:`aim_runtime.benchmarking.AIMBenchmark`.

        A ``config_file`` override may be supplied via the harness config
        (``--config`` on the CLI).

        ``AIMBenchmark`` — not the CLI — owns the ``benchmark_results.{json,csv}``
        artifacts. The CSV is the only source the async feedback workflow reads
        per-config metrics from, and both files honour the
        ``BENCHMARK_{JSON,CSV}_FILE`` overrides the self-hosted action sets, so
        the suite has to write them itself for either consumer to see a result.
        """
        from aim_runtime.benchmarking import AIMBenchmark

        service_url = config.resolve_service_url()

        if not self.health_check(service_url, timeout_seconds=config.timeout_seconds):
            return HarnessResult(success=False, summary="Service not reachable for benchmarking")

        benchmark_runner = AIMBenchmark(
            service_url=service_url,
            timeout_seconds=config.timeout_seconds,
            config_file=config.get("config_file"),
            engine_args=config.profile.get("engine_args", {}),
        )
        raw_results = benchmark_runner.run_benchmark_suite()

        artifacts: list[str] = []
        if output_dir := config.get("output_dir"):
            try:
                artifacts = [str(path) for path in benchmark_runner.export_results(raw_results, output_dir=output_dir)]
            except OSError:
                logger.exception("Failed to export benchmark results to %s", output_dir)

        checks = []
        for bench_config in raw_results.get("benchmark_configs", []):
            name = bench_config.get("config_name", "unknown")
            throughput = bench_config.get("output_tok_throughput")
            checks.append(
                CheckResult(
                    name=f"bench_{name}",
                    result_type=CheckResultType.SCORE,
                    success=bench_config.get("success", False),
                    value=float(throughput) if throughput is not None else 0.0,
                    detail=(
                        f"output_tok_throughput={throughput}"
                        if throughput is not None
                        else bench_config.get("error", "")
                    ),
                )
            )

        # Per-check success comes from each config; the overall verdict is
        # AIMBenchmark's own, which also accounts for suite-level failures.
        overall = raw_results.get("overall_success", False)
        return HarnessResult(
            success=overall,
            summary=f"Benchmark suite {'passed' if overall else 'failed'}",
            checks=checks,
            metrics={
                "model_name": raw_results.get("model_name"),
                "configs_run": len(raw_results.get("benchmark_configs", [])),
                "benchmark_configs": raw_results.get("benchmark_configs", []),
            },
            artifacts=artifacts,
        )

    def evaluate(self, config: HarnessConfig) -> HarnessResult:
        """Delegate to :class:`aim_runtime.evaluation.runner.AIMEvaluation`.

        The runner owns the evaluation itself — which backend, what is fatal, what
        it writes — and this method owns the translation into harness terms: the
        published envelope as ``metrics``, one ``SCORE`` check per number, and the
        exported files as ``artifacts``. The CI delegator consumes the same runner,
        so a score obtained here and one recorded by the pipeline come from the
        same code.

        This method starts no readiness probe of its own. The warmup loop in the
        runner is that wait, and the CLI also waits before it calls this method.
        One case is different. When the profile names no model,
        :meth:`_evaluation_model` polls ``/v1/models`` for the served name, and
        that poll waits for readiness as well.

        ``evaluate`` succeeds only when a score came back. A completed run that
        scored nothing gives the caller nothing to compare, so this method exits
        non-zero even though CI records that run as a NULL score and exits zero.
        """
        # Imported here, as everywhere below that the evaluation package is
        # touched, for import weight alone: nothing in it imports this module, so
        # there is no cycle to break — but `validate` and `list-checks` would
        # otherwise pay for pydantic settings and the backend they never use.
        from aim_runtime.evaluation.runner import AIMEvaluation

        service_url = config.resolve_service_url()
        model = self._evaluation_model(config, service_url)
        if not model:
            return HarnessResult(success=False, summary="Could not determine which model to evaluate")

        requested_dir = config.get("output_dir")
        with ExitStack() as stack:
            # The backend writes its own results under the run's output directory,
            # which a caller that asked for no artifacts still needs somewhere for.
            output_dir = requested_dir or stack.enter_context(TemporaryDirectory(prefix="aim-evaluate-"))
            try:
                settings = self._evaluation_settings(config, model, service_url, output_dir)
            except ValueError as exc:  # includes pydantic's ValidationError
                return HarnessResult(success=False, summary=f"Invalid evaluation settings: {exc}")

            runner = AIMEvaluation(
                settings,
                max_warmup_time=config.get("max_warmup_time", MAX_WARMUP_TIME_DEFAULT),
            )
            results = runner.run_evaluation_suite()
            artifacts = _export_evaluation(runner, results, requested_dir)

        return HarnessResult(
            success=results.success and results.accuracy is not None,
            summary=_evaluation_summary(results),
            checks=_evaluation_checks(results),
            metrics=results.to_dict(),
            artifacts=artifacts,
        )

    # --------------------------------------------------------------------- #
    # Internal helpers
    # --------------------------------------------------------------------- #

    @staticmethod
    def _evaluation_model(config: HarnessConfig, service_url: str) -> str:
        """The name to evaluate the served model under.

        ``aim_id`` first: it is the name CI evaluates and records its scores
        against, and vLLM answers to it because ``--served-model-name`` carries
        both it and ``model_id``. Using it also keeps the tokenizer lm-eval loads
        the same one CI scored with, which a quantized ``model_id`` would change.

        A base image's general profile names neither. This method then asks the
        endpoint what it serves. That call polls until a model answers, so it
        waits for readiness as well. It runs only when the profile cannot answer.
        """
        model = config.get("model") or config.profile.get("aim_id") or config.profile.get("model_id")
        if model:
            return str(model)

        logger.info("Profile names no model; asking %s what it serves", service_url)
        return probe_api_health(service_url, timeout_seconds=config.timeout_seconds).model_id or ""

    @staticmethod
    def _evaluation_settings(
        config: HarnessConfig,
        model: str,
        service_url: str,
        output_dir: str,
    ) -> "EvaluationSettings":
        """Translate profile and ``--config`` into settings the runner can take.

        Everything from ``--config`` is offered as an override, and the settings
        model keeps the keys it recognises. The profile contributes what only it
        knows: how wide the endpoint is, whether it is a CPU one, and whether the
        tokenizer may run code shipped with the model.
        """
        from aim_common.engine_args import ENGINE_ARG_TRUST_REMOTE_CODE, engine_flag_enabled
        from aim_runtime.evaluation.config import BACKEND_ARG_TRUST_REMOTE_CODE
        from aim_runtime.evaluation.settings import EvaluationSettings

        metadata = config.profile.get("metadata") or {}
        cpu_endpoint = str(metadata.get("accelerator_type", "")).lower() == AcceleratorType.CPU

        backend_args = dict(config.get("backend_args") or {})
        # An explicit setting wins: the profile flag is a default, not a mandate.
        backend_args.setdefault(
            BACKEND_ARG_TRUST_REMOTE_CODE,
            engine_flag_enabled(config.profile.get("engine_args"), ENGINE_ARG_TRUST_REMOTE_CODE),
        )

        return EvaluationSettings.resolve(
            model=model,
            service_url=service_url,
            # CPU profiles record a recommended core count in ``accelerator_count``,
            # so they are sized as the single endpoint they are.
            tensor_parallel_size=1 if cpu_endpoint else int(metadata.get("accelerator_count") or 1),
            cpu_endpoint=cpu_endpoint,
            config_file=config.get("config_file"),
            overrides={**config.extra, "backend_args": backend_args, "output_dir": output_dir},
        )

    @staticmethod
    def _check_profile_schema(config: HarnessConfig) -> CheckResult:
        """Validate profile YAML against the Pydantic schema."""
        try:
            from aim_common.object_model import ProfileData

            ProfileData.model_validate(config.profile)
            return passed("profile_schema", "Schema valid", gating=False)
        except Exception as exc:
            return failed("profile_schema", str(exc), gating=False)

    @staticmethod
    def _check_engine_validation(config: HarnessConfig) -> CheckResult:
        """Run engine-native argument validation against the profile's engine_args."""
        try:
            from aim_runtime.engines import engine_class_for

            engine = config.profile.get("engine", "vllm")
            engine_args = config.profile.get("engine_args", {})

            engine_class = engine_class_for(Engine(engine))
            if engine_class.ARGS_MODEL is None:
                return skipped("engine_validation", f"No native validator for engine '{engine}'", gating=False)

            engine_class.validate_engine_args(engine_args)
            return passed("engine_validation", f"Engine args valid for '{engine}'", gating=False)
        except Exception as exc:
            return failed("engine_validation", str(exc), gating=False)


# --------------------------------------------------------------------------- #
# Evaluation: mapping a run's results onto harness terms
# --------------------------------------------------------------------------- #


def _export_evaluation(runner: "AIMEvaluation", results: "EvaluationResults", output_dir: str | None) -> list[str]:
    """Write the results files, if a destination was asked for."""
    if not output_dir:
        return []
    try:
        return [str(path) for path in runner.export_results(results, output_dir=output_dir)]
    except OSError:
        logger.exception("Failed to export evaluation results to %s", output_dir)
        return []


def _evaluation_checks(results: "EvaluationResults") -> list[CheckResult]:
    """One ``SCORE`` check for the headline number, plus one per task when several ran.

    The headline check is always present and always a ``SCORE``, carrying ``None``
    when the run produced no number — a scoreless run must not be reported as
    having scored ``0.0``, which is itself a legitimate result. Per-task checks are
    added only for a multi-task run, where the headline alone would hide the rest.

    None of them gate: a score is only meaningful against a baseline this harness
    does not have, so the pass/fail call belongs to a human reading the number (or
    to CI's threshold comparison), not to the run that produced it. ``evaluate``
    still exits non-zero when no score came back at all — that is a failed run,
    not a bad score.
    """
    from aim_runtime.evaluation.config import PRIMARY_CHECK_NAME, TASK_CHECK_PREFIX

    primary = results.primary
    checks = [
        CheckResult(
            name=PRIMARY_CHECK_NAME,
            result_type=CheckResultType.SCORE,
            success=results.success and primary is not None,
            value=results.accuracy,
            detail=_score_detail(primary) if primary else (results.failure_reason or "No task produced a score"),
            gating=False,
        )
    ]
    if len(results.tasks) > 1:
        checks += [
            CheckResult(
                name=f"{TASK_CHECK_PREFIX}{score.task}",
                result_type=CheckResultType.SCORE,
                success=True,
                value=score.value,
                detail=_score_detail(score),
                gating=False,
            )
            for score in results.tasks
        ]
    return checks


def _evaluation_summary(results: "EvaluationResults") -> str:
    if results.failure_reason:
        return f"Accuracy evaluation failed: {results.failure_reason}"
    if results.primary is None:
        return "Accuracy evaluation completed without a score"
    return f"Accuracy evaluation scored {_score_detail(results.primary)}"


def _score_detail(score: "TaskScore") -> str:
    """The score and everything needed to interpret it, in one line.

    Reproducibility detail rather than decoration: the same task scores
    differently under another metric variant or a different number of in-context
    examples, and a capped run is not the full split.
    """
    metric = f"{score.metric} ({score.metric_variant})" if score.metric_variant else score.metric
    samples = f"{score.num_samples}"
    if score.num_samples_available:
        samples = f"{score.num_samples}/{score.num_samples_available}"
    detail = f"{score.task} {metric}: {score.value} over {samples} samples"
    if score.num_fewshot is not None:
        detail += f", {score.num_fewshot}-shot"
    return detail
