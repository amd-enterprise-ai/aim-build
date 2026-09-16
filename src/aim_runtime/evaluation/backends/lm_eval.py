# Copyright © Advanced Micro Devices, Inc., or its affiliates.
#
# SPDX-License-Identifier: MIT

"""
The lm-evaluation-harness backend.

lm-eval is invoked as a **subprocess**, and nothing here imports it. Two reasons,
both from packaging: the evaluation dependencies may be installed in their own
virtualenv, in which case they are not importable from the interpreter running
``aim-runtime`` at all; and lm-eval pins its own ``transformers``, which an
in-process import would bind in place of the image's. The command is therefore
configurable — a bare name, or an absolute path into another virtualenv.

The work is split so each piece can be tested without the others: building argv
is pure, running is the only part that needs a process, discovery is the only part
that needs a filesystem, and :func:`parse_document` is a pure function over a
results document — which is also what lets the CI metrics parser delegate to it.
"""

from __future__ import annotations

import json
import logging
import os
import shlex
import subprocess
import threading
from collections import deque
from collections.abc import Iterator, Mapping
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from aim_runtime.evaluation.backends.base import BackendInfo, EvaluationBackend, RawRun
from aim_runtime.evaluation.config import (
    BACKEND_ARG_APPLY_CHAT_TEMPLATE,
    BACKEND_ARG_BATCH_SIZE,
    BACKEND_ARG_MAX_GEN_TOKS,
    BACKEND_ARG_MAX_MODEL_LENGTH,
    BACKEND_ARG_MODEL_TYPE,
    BACKEND_ARG_TOKENIZER,
    BACKEND_ARG_TRUST_REMOTE_CODE,
    BACKEND_COMMAND,
    BACKEND_COMMAND_ENV,
    BACKEND_LM_EVAL,
    BACKEND_OUTPUT_SUBDIR,
    COMPLETIONS_PATH,
    FAILURE_TAIL_LINES,
    LM_EVAL_BACKEND_ARG_DEFAULTS,
    LM_EVAL_BACKEND_ARGS,
    LM_EVAL_SCORE_METRIC_NAMES,
    PROBE_TIMEOUT_SECONDS,
    lm_eval_preferred_variant,
)
from aim_runtime.evaluation.results import MODALITY_TEXT, TaskScore
from aim_runtime.evaluation.settings import EvaluationSettings

logger = logging.getLogger(__name__)

#: Results files lm-eval writes, as
#: ``<output_path>/<sanitized model name>/results_<timestamp>.json``. Matched by
#: glob rather than by rebuilding lm-eval's model-name sanitization, which is its
#: implementation detail and has changed between releases.
RESULTS_GLOB = "results_*.json"

#: Suffix lm-eval appends to a metric name to report its standard error, inside
#: the key rather than around it: ``exact_match_stderr,flexible-extract``.
STDERR_SUFFIX = "_stderr"

#: Blocks of the results document, each keyed by task name.
RESULTS_BLOCK = "results"
CONFIGS_BLOCK = "configs"
SAMPLES_BLOCK = "n-samples"
FEWSHOT_BLOCK = "n-shot"
HIGHER_IS_BETTER_BLOCK = "higher_is_better"
VERSION_KEY = "lm_eval_version"


@dataclass(frozen=True)
class _Metric:
    """One metric key of a task's results entry, already split and coerced."""

    key: str
    name: str
    variant: str | None
    value: float


class LmEvalBackend(EvaluationBackend):
    """Scores a served model by running the lm-eval CLI against its endpoint."""

    name = BACKEND_LM_EVAL
    #: lm-eval evaluates text prompts against text completions, and its metric
    #: vocabulary only makes sense there.
    modality = MODALITY_TEXT

    def __init__(self, command: str | None = None) -> None:
        """
        Args:
            command: How to invoke lm-eval. Defaults to the
                :data:`~aim_runtime.evaluation.config.BACKEND_COMMAND_ENV`
                environment variable, then to
                :data:`~aim_runtime.evaluation.config.BACKEND_COMMAND`. Parsed as
                a shell word list, so ``"python -m lm_eval"`` works as well as a
                path.
        """
        self.command = command or os.environ.get(BACKEND_COMMAND_ENV) or BACKEND_COMMAND
        self._argv = shlex.split(self.command)

    def probe(self) -> BackendInfo:
        """Ask the command to describe itself, which is enough to know it runs."""
        argv = [*self._argv, "--help"]
        try:
            completed = subprocess.run(argv, capture_output=True, text=True, timeout=PROBE_TIMEOUT_SECONDS)
        except FileNotFoundError:
            return self._unavailable(
                f"{self.command!r} was not found; the evaluation dependencies may not be installed"
            )
        except (OSError, subprocess.SubprocessError) as error:
            return self._unavailable(f"{self.command!r} could not be run: {error}")

        if completed.returncode != 0:
            detail = _tail(completed.stderr or completed.stdout)
            return self._unavailable(f"{self.command!r} exited {completed.returncode}: {detail}")

        return BackendInfo(name=self.name, available=True, command=self.command)

    def build_argv(self, settings: EvaluationSettings, output_path: Path) -> list[str]:
        """The command line for this run. Pure, so it can be asserted on directly.

        Argument order matches what CI has always invoked, so a parity diff of the
        two is a diff of values rather than of shape. The flat form (no ``run``
        subcommand) is deliberate: lm-eval 0.4.12 inserts ``run`` itself when a
        bare argument list is given, and the flat form also works with the older
        releases a relocated command might point at.
        """
        backend_args = self._backend_args(settings)
        model_args = self._model_args(settings, backend_args)

        argv = [
            *self._argv,
            "--model",
            str(_arg(backend_args, BACKEND_ARG_MODEL_TYPE)),
            "--model_args",
            json.dumps(model_args),
            "--tasks",
            settings.tasks_argument,
            "--batch_size",
            str(_arg(backend_args, BACKEND_ARG_BATCH_SIZE)),
        ]
        if settings.num_fewshot is not None:
            argv += ["--num_fewshot", str(settings.num_fewshot)]
        argv += ["--output_path", str(output_path)]
        if settings.limit is not None:
            argv += ["--limit", str(settings.limit)]
        if backend_args.get(BACKEND_ARG_APPLY_CHAT_TEMPLATE):
            argv.append("--apply_chat_template")
            _warn_if_not_instruction_tuned(settings.model)

        return argv

    def run(self, settings: EvaluationSettings, *, env: Mapping[str, str], output_dir: Path) -> RawRun:
        """Evaluate the endpoint, then load whatever lm-eval left behind."""
        output_path = Path(output_dir) / BACKEND_OUTPUT_SUBDIR
        output_path.mkdir(parents=True, exist_ok=True)

        argv = self.build_argv(settings, output_path)
        logger.info("[LM_EVAL] Running lm_eval with command: %s", shlex.join(argv))

        failure = self._execute(argv, env=env, budget=settings.wall_clock_timeout_seconds)
        if failure:
            return RawRun(failure_reason=failure)

        results_file = self._discover(output_path)
        if results_file is None:
            # The evaluation itself succeeded, so this is not a failure — it is a
            # run with no score, which the caller reports as such.
            return RawRun()

        try:
            document = json.loads(results_file.read_text(encoding="utf-8"))
        except (OSError, ValueError) as error:
            return RawRun(
                failure_reason=f"could not read lm-eval results at {results_file}: {error}",
                artifacts=(results_file,),
            )
        if not isinstance(document, dict):
            return RawRun(
                failure_reason=f"lm-eval results at {results_file} are not a JSON object",
                artifacts=(results_file,),
            )

        version = document.get(VERSION_KEY)
        return RawRun(
            document=document,
            artifacts=(results_file,),
            version=str(version) if version else None,
        )

    def parse(self, raw: RawRun) -> list[TaskScore]:
        return parse_document(raw.document, modality=self.modality) if raw.document else []

    def _unavailable(self, detail: str) -> BackendInfo:
        """Why the backend cannot run. The caller logs it, so this does not."""
        return BackendInfo(name=self.name, available=False, command=self.command, detail=detail)

    def _backend_args(self, settings: EvaluationSettings) -> Mapping[str, Any]:
        """The backend arguments, having reported any this backend cannot use."""
        unknown = sorted(set(settings.backend_args) - LM_EVAL_BACKEND_ARGS)
        if unknown:
            logger.warning("Ignoring backend_args lm-eval does not understand: %s", ", ".join(unknown))
        return settings.backend_args

    def _model_args(self, settings: EvaluationSettings, backend_args: Mapping[str, Any]) -> dict[str, Any]:
        """lm-eval's ``--model_args``, which it accepts as a JSON object.

        The generic settings supply how hard to push the endpoint; ``backend_args``
        only overrides lm-eval's own vocabulary.
        """
        model_args: dict[str, Any] = {
            "model": settings.model,
            "base_url": settings.service_url + COMPLETIONS_PATH,
            "num_concurrent": settings.concurrency,
            "max_retries": settings.max_retries,
            "timeout": settings.request_timeout_seconds,
            "max_gen_toks": _arg(backend_args, BACKEND_ARG_MAX_GEN_TOKS),
            "max_length": _arg(backend_args, BACKEND_ARG_MAX_MODEL_LENGTH),
            # Not from the defaults table: the fallback is known only per run.
            "tokenizer": backend_args.get(BACKEND_ARG_TOKENIZER) or settings.model,
        }
        if backend_args.get(BACKEND_ARG_TRUST_REMOTE_CODE):
            # Models that ship custom tokenizer code cannot be loaded without it.
            model_args[BACKEND_ARG_TRUST_REMOTE_CODE] = True
        return model_args

    def _execute(self, argv: list[str], *, env: Mapping[str, str], budget: int | None) -> str | None:
        """Run the command, streaming its output. Returns a failure reason or None.

        ``budget`` bounds the whole run — see :func:`_kill_after` for why it acts
        on the process rather than on its output.
        """
        try:
            process = subprocess.Popen(
                argv,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
                env=dict(env),
            )
        except OSError as error:
            return f"lm_eval could not be started: {error}"

        tail: deque[str] = deque(maxlen=FAILURE_TAIL_LINES)
        with _kill_after(process, budget) as expired:
            _log_output(process, tail)
            process.wait()

        if expired.is_set():
            return f"lm_eval exceeded its {budget}s wall-clock budget"
        if process.returncode != 0:
            return f"lm_eval failed with code {process.returncode}: {' '.join(tail) or 'Unknown error'}"

        logger.info("[LM_EVAL] Accuracy evaluation completed successfully")
        return None

    def _discover(self, output_path: Path) -> Path | None:
        """The results file lm-eval just wrote, or None with the directory logged."""
        candidates = sorted(output_path.rglob(RESULTS_GLOB), key=lambda path: path.stat().st_mtime)
        if not candidates:
            logger.error("No lm-eval results file matching %s under %s", RESULTS_GLOB, output_path)
            for item in sorted(output_path.rglob("*")):
                logger.error("  %s", item)
            return None

        newest = candidates[-1]
        logger.info("Found lm-eval results file: %s", newest)
        return newest


def _arg(backend_args: Mapping[str, Any], key: str) -> str | int:
    """One overridable ``backend_args`` value, or lm-eval's default for it.

    A falsy override counts as unset, which is what makes an absent key, an
    explicit ``null`` in the config file and an empty string behave alike.
    """
    return backend_args.get(key) or LM_EVAL_BACKEND_ARG_DEFAULTS[key]


@contextmanager
def _kill_after(process: subprocess.Popen[str], budget: int | None) -> Iterator[threading.Event]:
    """Kill ``process`` if it outlives ``budget`` seconds, and say whether it did.

    The deadline acts on the process rather than on its output stream because an
    evaluation that stops making progress without exiting is silent: there is no
    line to time out waiting for, and the run would otherwise hold the harness
    open indefinitely. A falsy budget means no deadline, and the event it yields
    then simply never sets.

    The process is always dead on the way out. Reading its output can raise, and
    the cancelled deadline is then the only thing that would have stopped it, so
    a process still running here is killed and reaped rather than left to outlive
    the run it belongs to.

    Yields:
        An event that is set only if the deadline fired, which the caller reads
        after the process has exited to tell a kill from an ordinary failure —
        both of which surface as a non-zero return code.
    """
    expired = threading.Event()

    def _expire() -> None:
        expired.set()
        process.kill()

    deadline = threading.Timer(budget, _expire) if budget else None
    if deadline is not None:
        deadline.start()
    try:
        yield expired
    finally:
        if deadline is not None:
            deadline.cancel()
        if process.poll() is None:
            process.kill()
            process.wait()


def _log_output(process: subprocess.Popen[str], tail: deque[str]) -> None:
    """Log the process's output as it arrives, keeping the last lines in ``tail``.

    Streaming rather than collecting: an evaluation runs for tens of minutes, and
    its progress is the only sign it is alive. ``tail`` is what a failure is
    reported with, which is why the lines are kept as well as logged.
    """
    if process.stdout is None:
        # Only reachable if the pipe was not requested, but reading None is fatal
        # and losing the output is not.
        logger.warning("[LM_EVAL] Process output is not available; running unmonitored")
        return

    for line in process.stdout:
        logger.info("[LM_EVAL] %s", line.rstrip())
        tail.append(line.strip())


def parse_document(document: Mapping[str, Any], *, modality: str = MODALITY_TEXT) -> list[TaskScore]:
    """Map an lm-eval results document onto one score per task.

    ``modality`` is stamped onto every record, so what a score measures is stated
    by the backend that produced it rather than inferred from a default.

    Tolerant by design: this reads documents from any lm-eval release the image
    might carry, and a missing metadata block should cost the detail it describes
    rather than the score itself. A task whose metrics are all unrecognised is
    skipped, since promoting an arbitrary number to "accuracy" is worse than
    reporting none.
    """
    preferred = lm_eval_preferred_variant()
    scores = []
    for task, entry in _block(document, RESULTS_BLOCK).items():
        if not isinstance(entry, Mapping):
            continue
        score = _task_score(str(task), entry, document, preferred, modality)
        if score is None:
            logger.debug("No recognised accuracy metric for task %r", task)
            continue
        scores.append(score)
    return scores


def _task_score(
    task: str,
    entry: Mapping[str, Any],
    document: Mapping[str, Any],
    preferred: str,
    modality: str,
) -> TaskScore | None:
    metrics = _score_metrics(entry)
    if not metrics:
        return None

    chosen = next((metric for metric in metrics if metric.variant == preferred), metrics[0])
    evaluated, available = _sample_counts(document, task, entry)

    return TaskScore(
        task=task,
        dataset=_dataset(document, task),
        metric=chosen.name,
        metric_variant=chosen.variant,
        value=chosen.value,
        stderr=_stderr(entry, chosen),
        num_samples=evaluated,
        num_samples_available=available,
        num_fewshot=_as_int(_block(document, FEWSHOT_BLOCK).get(task)),
        higher_is_better=_higher_is_better(document, task, chosen.name),
        modality=modality,
        extra_metrics={metric.key: metric.value for metric in metrics if metric.key != chosen.key},
    )


def _score_metrics(entry: Mapping[str, Any]) -> list[_Metric]:
    """The entry's accuracy metrics, in document order.

    Keys are ``"<metric>,<variant>"``, or a bare metric name in documents from
    releases that predate filters. Standard errors are excluded by the same check
    that excludes perplexity: ``exact_match_stderr`` is not a score name.
    """
    metrics = []
    for key, value in entry.items():
        name, _, variant = str(key).partition(",")
        if name not in LM_EVAL_SCORE_METRIC_NAMES:
            continue
        number = _as_float(value)
        if number is None:
            continue
        metrics.append(_Metric(key=str(key), name=name, variant=variant or None, value=number))
    return metrics


def _stderr(entry: Mapping[str, Any], metric: _Metric) -> float | None:
    """The chosen metric's standard error, which lm-eval may report as "N/A"."""
    key = f"{metric.name}{STDERR_SUFFIX}"
    if metric.variant:
        key = f"{key},{metric.variant}"
    return _as_float(entry.get(key))


def _dataset(document: Mapping[str, Any], task: str) -> str:
    """The data the task evaluated, as ``<path>/<name>`` where both are reported."""
    config = _block(_block(document, CONFIGS_BLOCK), task)
    path = config.get("dataset_path")
    name = config.get("dataset_name")
    if not path:
        return task
    return f"{path}/{name}" if name else str(path)


def _sample_counts(document: Mapping[str, Any], task: str, entry: Mapping[str, Any]) -> tuple[int, int | None]:
    """How many documents were evaluated, and how many the split holds.

    ``n-samples`` is the authoritative pair. The task entry's own count is the
    fallback for documents that omit the block — ``sample_len`` in current
    releases, ``samples`` in older ones.
    """
    counts = _block(_block(document, SAMPLES_BLOCK), task)
    available = _as_int(counts.get("original"))
    evaluated = _first_int(counts.get("effective"), entry.get("sample_len"), entry.get("samples"), available)
    return evaluated or 0, available


def _higher_is_better(document: Mapping[str, Any], task: str, metric: str) -> bool | None:
    value = _block(_block(document, HIGHER_IS_BETTER_BLOCK), task).get(metric)
    return value if isinstance(value, bool) else None


def _block(source: Mapping[str, Any], key: str) -> Mapping[str, Any]:
    """A nested mapping, or an empty one when it is absent or malformed."""
    value = source.get(key) if isinstance(source, Mapping) else None
    return value if isinstance(value, Mapping) else {}


def _first_int(*values: Any) -> int | None:
    for value in values:
        number = _as_int(value)
        if number is not None:
            return number
    return None


def _as_int(value: Any) -> int | None:
    number = _as_float(value)
    return int(number) if number is not None else None


def _as_float(value: Any) -> float | None:
    """``float(value)``, or None for anything that is not a number."""
    if isinstance(value, bool) or not isinstance(value, (int, float, str)):
        return None
    try:
        return float(value)
    except ValueError:
        return None


def _tail(output: str | None) -> str:
    lines = [line.strip() for line in (output or "").splitlines() if line.strip()]
    return " ".join(lines[-FAILURE_TAIL_LINES:]) or "no output"


def _warn_if_not_instruction_tuned(model: str) -> None:
    """Applying a chat template to a base model measures something else.

    Only a warning, which is what CI does: the served model's name is the one
    signal available here, and it is a convention rather than a guarantee.
    """
    lowered = model.lower()
    if "instruct" not in lowered and "chat" not in lowered:
        logger.warning(
            "Applying a chat template to %r, whose name suggests it may not be "
            "instruction-tuned; confirm the model expects one",
            model,
        )
