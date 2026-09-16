# Copyright © Advanced Micro Devices, Inc., or its affiliates.
#
# SPDX-License-Identifier: MIT

"""
Runs one evaluation end to end, and writes the results out.

:class:`AIMEvaluation` is what both consumers call — ``VLLMHarness.evaluate()``
in the image and the CI delegator in the pipeline — so everything either of them
would otherwise repeat lives here: phase order, what is fatal, what is advisory,
what environment the backend inherits, and which files an evaluation leaves
behind. Its surface deliberately mirrors
:class:`~aim_runtime.benchmarking.AIMBenchmark` (construct, run a suite, export
results), so a reader who knows ``benchmark`` can read ``evaluate``.

Four phases, and the interesting part is which of them can end the run:

1. **Availability — fatal.** The backend is probed before anything else, so an
   image without the evaluation dependencies produces one clear reason rather
   than a subprocess error minutes in.
2. **Warmup — advisory.** A first inference absorbs kernel JIT compilation. CI
   logs a warning and proceeds when it does not complete, so this does too:
   failing here would fail runs that would have scored fine, and the retry loop
   doubles as the readiness wait.
3. **Run — fatal.** Whatever the backend reports as a failure ends the run.
4. **Parse — fatal.** Results that cannot be read are a failure, not a run that
   scored nothing.

There is deliberately **no health phase**: warmup already retries until the
service answers, so probing ``/v1/models`` first would only add a second
readiness wait that CI never had.

A run that completes and yields no score is *not* a failure. CI records a NULL
score and exits zero for that case, so the distinction survives here — see
:attr:`~aim_runtime.evaluation.results.EvaluationResults.success`.
"""

from __future__ import annotations

import csv
import json
import logging
import os
import time
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from aim_runtime.evaluation.backends.base import EvaluationBackend, RawRun
from aim_runtime.evaluation.backends.lm_eval import LmEvalBackend
from aim_runtime.evaluation.config import (
    BACKEND_DOCUMENT_FILENAME,
    DATASET_ENV_VARS,
    ELAPSED_ROUND_DECIMALS,
    RESULTS_CSV_FILE_ENV,
    RESULTS_CSV_FILENAME,
    RESULTS_JSON_FILE_ENV,
    RESULTS_JSON_FILENAME,
)
from aim_runtime.evaluation.results import CSV_COLUMNS, EvaluationResults, TaskScore
from aim_runtime.evaluation.settings import EvaluationSettings
from aim_runtime.harness.service_checks import MAX_WARMUP_TIME_DEFAULT, run_warmup

logger = logging.getLogger(__name__)


class AIMEvaluation:
    """Evaluates a served model and reports what it scored."""

    def __init__(
        self,
        settings: EvaluationSettings,
        backend: EvaluationBackend | None = None,
        max_warmup_time: float = MAX_WARMUP_TIME_DEFAULT,
    ) -> None:
        """
        Args:
            settings: What to evaluate, where, and how hard to push it.
            backend: The tool that produces the score. Defaults to the lm-eval
                adapter; a name-to-class registry resolved from
                ``settings.backend`` replaces this default once a second backend
                exists, which is why nothing else reads that field yet.
            max_warmup_time: Ceiling for the first-inference warmup. ``0``
                skips warmup, which is the only way to opt out of a phase.
        """
        self.settings = settings
        self.backend = backend or LmEvalBackend()
        self.max_warmup_time = max_warmup_time

    def run_evaluation_suite(self) -> EvaluationResults:
        """Run every phase and return the results, failure included.

        Never raises: a broken backend is a reported failure, because both
        consumers turn this into a result rather than a traceback — CI still has
        to write a NULL-score row, and the harness still has to return a
        ``HarnessResult``.
        """
        logger.info(
            "Starting AIM evaluation: %s on %s via %s",
            self.settings.tasks_argument,
            self.settings.model,
            self.settings.service_url,
        )

        info = self.backend.probe()
        if not info.available:
            return self._failed(info.detail or f"evaluation backend {info.name!r} is not available")

        self._warmup()

        started = time.monotonic()
        raw = self._run()
        elapsed = round(time.monotonic() - started, ELAPSED_ROUND_DECIMALS)

        tasks, parse_failure = self._parse(raw)
        results = EvaluationResults(
            settings=self.settings,
            tasks=tasks,
            backend_version=raw.version or info.version,
            elapsed_seconds=elapsed,
            failure_reason=raw.failure_reason or parse_failure,
            document=raw.document,
        )
        self._report(results)
        return results

    def export_results(self, results: EvaluationResults, output_dir: str | Path | None = None) -> list[Path]:
        """Write the results as files and return what was written.

        JSON and CSV carry the filenames the accuracy action already uploads,
        each honouring its environment override, which may be a bare name or an
        absolute path. The backend's own document is written beside them when
        there is one, so a suspicious score can be traced back to what the tool
        reported.

        ``output_dir`` defaults to the settings', so a caller that already
        configured one need not repeat it.
        """
        directory = Path(output_dir) if output_dir is not None else Path(results.settings.output_dir)
        directory.mkdir(parents=True, exist_ok=True)

        json_path = _output_path(directory, RESULTS_JSON_FILE_ENV, RESULTS_JSON_FILENAME)
        _write_json(json_path, results.to_dict())

        csv_path = _output_path(directory, RESULTS_CSV_FILE_ENV, RESULTS_CSV_FILENAME)
        _write_csv(csv_path, results.to_csv_rows())

        written = [json_path, csv_path]
        if results.document is not None:
            document_path = directory / BACKEND_DOCUMENT_FILENAME
            _write_json(document_path, results.document)
            written.append(document_path)

        for path in written:
            logger.info("Wrote %s", path)
        return written

    def _warmup(self) -> None:
        """Send a first inference so the evaluation does not pay for JIT."""
        if not self.max_warmup_time:
            logger.info("Skipping warmup (max_warmup_time=%s)", self.max_warmup_time)
            return

        try:
            outcome = run_warmup(
                self.settings.service_url,
                self.settings.model,
                max_warmup_time=self.max_warmup_time,
            )
        except Exception:
            # Advisory means advisory: the evaluation is still worth attempting,
            # and the backend's own retries are the real defence.
            logger.warning("Model warmup raised; continuing to the evaluation", exc_info=True)
            return

        if not outcome.succeeded:
            logger.warning("Model warmup incomplete - evaluation may experience timeouts: %s", outcome.check.detail)

    def _run(self) -> RawRun:
        """Hand the run to the backend, turning an unexpected raise into a reason."""
        try:
            return self.backend.run(self.settings, env=self._env(), output_dir=Path(self.settings.output_dir))
        except Exception as error:
            logger.exception("Evaluation backend %r raised", self.backend.name)
            return RawRun(failure_reason=f"{self.backend.name} raised {type(error).__name__}: {error}")

    def _parse(self, raw: RawRun) -> tuple[tuple[TaskScore, ...], str | None]:
        """Interpret the run's output. Returns the scores and a failure, if any."""
        if raw.document is None:
            return (), None
        try:
            return tuple(self.backend.parse(raw)), None
        except Exception as error:
            logger.exception("Could not read %r results", self.backend.name)
            return (), f"could not read {self.backend.name} results: {type(error).__name__}: {error}"

    def _env(self) -> Mapping[str, str]:
        """The environment the backend runs with.

        The whole environment is passed through rather than filtered: the
        backend may live in another virtualenv and needs its own ``PATH`` and
        ``PYTHON*`` variables. The dataset variables are reported because a
        missing one shows up as a download failure minutes into a run.
        """
        env = dict(os.environ)
        for name in DATASET_ENV_VARS:
            logger.debug("%s is %s", name, "set" if env.get(name) else "not set")
        return env

    def _failed(self, reason: str) -> EvaluationResults:
        logger.error(reason)
        return EvaluationResults(settings=self.settings, failure_reason=reason)

    def _report(self, results: EvaluationResults) -> None:
        if not results.success:
            logger.error("Evaluation failed: %s", results.failure_reason)
            return
        if results.accuracy is None:
            logger.warning("Evaluation completed but no task produced a score")
            return
        logger.info(
            "Evaluation completed: %s=%s on %s (%ss)",
            results.primary.metric if results.primary else "accuracy",
            results.accuracy,
            results.primary.task if results.primary else self.settings.tasks_argument,
            results.elapsed_seconds,
        )


def _output_path(directory: Path, env_var: str, default_name: str) -> Path:
    """Where a results file goes, honouring its environment override.

    An absolute override wins over ``directory`` — joining an absolute path onto
    another discards the left side — which is how the accuracy action's absolute
    paths take effect. Mirrors ``AIMBenchmark.export_results``.
    """
    return directory / (os.environ.get(env_var) or default_name)


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    """Write the per-task rows, header first.

    The header is written even with no rows, so a run that scored nothing is
    distinguishable from one that never wrote a file.
    """
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=CSV_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)
