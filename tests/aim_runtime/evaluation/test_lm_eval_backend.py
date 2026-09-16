# Copyright © Advanced Micro Devices, Inc., or its affiliates.
#
# SPDX-License-Identifier: MIT

"""
Tests for aim_runtime.evaluation.backends.lm_eval.

Four independent concerns, matching the adapter's own split: the argv it builds
(pure), the process it runs (a fake ``Popen``, never the real lm-eval), the
results file it finds, and the document it parses. lm-eval is neither installed
nor served for any of this.
"""

import json
import logging
import os
import subprocess
from collections import deque
from pathlib import Path

import pytest

from aim_runtime.evaluation.backends.base import RawRun
from aim_runtime.evaluation.backends.lm_eval import LmEvalBackend, _kill_after, _log_output, parse_document
from aim_runtime.evaluation.config import (
    BACKEND_ARG_APPLY_CHAT_TEMPLATE,
    BACKEND_ARG_MODEL_TYPE,
    BACKEND_ARG_TOKENIZER,
    BACKEND_ARG_TRUST_REMOTE_CODE,
    BACKEND_COMMAND_ENV,
    BACKEND_LM_EVAL,
    BACKEND_OUTPUT_SUBDIR,
    FAILURE_TAIL_LINES,
    LM_EVAL_PREFERRED_VARIANT_ENV,
)
from aim_runtime.evaluation.results import MODALITY_TEXT
from aim_runtime.evaluation.settings import EvaluationSettings

MODULE = "aim_runtime.evaluation.backends.lm_eval"
POPEN = f"{MODULE}.subprocess.Popen"
RUN = f"{MODULE}.subprocess.run"

MODEL = "openai/gpt-oss-20b"
SERVICE_URL = "http://localhost:8000"
#: Directory lm-eval writes MODEL's document into, sanitized as it sanitizes it.
MODEL_SUBDIR = "openai__gpt-oss-20b"
RESULTS_NAME = "results_2026-08-12T09-00-00.json"


class FakeProcess:
    """A stand-in for ``subprocess.Popen``: streams canned lines, then exits."""

    def __init__(self, exit_code: int = 0, lines: tuple[str, ...] = ()):
        self._exit_code = exit_code
        self._lines = lines
        self.killed = False
        self.returncode: int | None = None
        self.stdout = (f"{line}\n" for line in lines)
        #: The patched ``Popen``, set by the ``fake_process`` fixture, for the
        #: tests that assert on how the process was started rather than on what
        #: it did.
        self.popen = None

    def wait(self):
        self.returncode = -9 if self.killed else self._exit_code
        return self.returncode

    def poll(self):
        """``None`` until the process has been waited for, as ``Popen.poll`` is."""
        return self.returncode

    def kill(self):
        self.killed = True


class ImmediateTimer:
    """A `threading.Timer` that fires on `start()` instead of after a wait.

    The deadline's own scheduling is the standard library's business; what this
    module owns is what the callback does and how the outcome is reported, and
    firing synchronously tests exactly that without a sleeping test.
    """

    scheduled: list[float] = []

    def __init__(self, interval, function):
        ImmediateTimer.scheduled.append(interval)
        self._function = function

    def start(self):
        self._function()

    def cancel(self):
        pass


@pytest.fixture
def backend() -> LmEvalBackend:
    return LmEvalBackend()


@pytest.fixture
def settings() -> EvaluationSettings:
    return EvaluationSettings(model=MODEL, service_url=SERVICE_URL)


@pytest.fixture
def fake_process(mocker):
    """Install a canned process in place of ``Popen`` and return it.

    Every run test needs one, so the patch target lives here rather than in each
    body: a test then states what the process did — exited non-zero, printed a
    traceback — instead of how the fake was wired in.
    """

    def install(exit_code: int = 0, lines: tuple[str, ...] = ()) -> FakeProcess:
        process = FakeProcess(exit_code=exit_code, lines=lines)
        process.popen = mocker.patch(POPEN, return_value=process)
        return process

    return install


@pytest.fixture
def deadline_timer(mocker):
    """The stand-in ``threading.Timer``, for asserting how the deadline was scheduled."""
    return mocker.patch(f"{MODULE}.threading.Timer")


@pytest.fixture
def immediate_deadline(mocker) -> type[ImmediateTimer]:
    """Make the deadline fire on ``start()``, and forget any earlier schedule."""
    mocker.patch(f"{MODULE}.threading.Timer", ImmediateTimer)
    ImmediateTimer.scheduled.clear()
    return ImmediateTimer


@pytest.fixture
def probe_answer(mocker):
    """Answer the availability probe with a canned ``CompletedProcess``."""

    def install(returncode: int = 0, stdout: str = "usage: lm-eval", stderr: str = ""):
        return mocker.patch(RUN, return_value=mocker.Mock(returncode=returncode, stdout=stdout, stderr=stderr))

    return install


def write_document(output_dir: Path, document, name: str = RESULTS_NAME) -> Path:
    """Put a results document where lm-eval would have written it."""
    results_file = output_dir / BACKEND_OUTPUT_SUBDIR / MODEL_SUBDIR / name
    results_file.parent.mkdir(parents=True, exist_ok=True)
    results_file.write_text(json.dumps(document), encoding="utf-8")
    return results_file


def model_args_of(argv: list[str]) -> dict:
    return json.loads(argv[argv.index("--model_args") + 1])


class TestCommand:
    """Test suite for how the backend is invoked."""

    def test_the_default_command_is_the_bare_name(self, backend):
        assert backend.command == "lm_eval"
        assert backend.name == BACKEND_LM_EVAL

    def test_the_environment_can_relocate_the_command(self, monkeypatch):
        """An image may install the evaluation dependencies in their own virtualenv."""
        monkeypatch.setenv(BACKEND_COMMAND_ENV, "/workspace/tools/eval-venv/bin/lm_eval")

        assert LmEvalBackend().command == "/workspace/tools/eval-venv/bin/lm_eval"

    def test_an_explicit_command_wins_over_the_environment(self, monkeypatch):
        monkeypatch.setenv(BACKEND_COMMAND_ENV, "/from/env")

        assert LmEvalBackend("/explicit").command == "/explicit"

    def test_a_multi_word_command_is_split_into_argv(self, settings, tmp_path):
        argv = LmEvalBackend("python -m lm_eval").build_argv(settings, tmp_path)

        assert argv[:3] == ["python", "-m", "lm_eval"]


class TestProbe:
    """Test suite for the availability check."""

    def test_a_command_that_describes_itself_is_available(self, backend, probe_answer):
        run = probe_answer()

        info = backend.probe()

        assert info.available is True
        assert info.command == "lm_eval"
        assert run.call_args.args[0] == ["lm_eval", "--help"]

    def test_a_missing_command_is_reported_as_unavailable(self, backend, mocker):
        """The expected case in an image without the evaluation extra installed."""
        mocker.patch(RUN, side_effect=FileNotFoundError("no lm_eval"))

        info = backend.probe()

        assert info.available is False
        assert "not found" in info.detail
        assert "lm_eval" in info.detail

    def test_a_command_that_fails_quotes_what_it_said(self, backend, probe_answer):
        probe_answer(returncode=2, stdout="", stderr="ImportError: torch")

        info = backend.probe()

        assert info.available is False
        assert "exited 2" in info.detail
        assert "ImportError: torch" in info.detail

    def test_a_probe_that_never_returns_is_not_available(self, backend, mocker):
        mocker.patch(RUN, side_effect=subprocess.TimeoutExpired(cmd="lm_eval", timeout=120))

        assert backend.probe().available is False

    def test_an_unavailable_backend_is_returned_and_not_logged(self, backend, mocker, caplog):
        """The caller decides whether this ends a run, so the caller logs it."""
        mocker.patch(RUN, side_effect=FileNotFoundError("no lm_eval"))

        assert backend.probe().detail

        assert [record.getMessage() for record in caplog.records if record.levelno >= logging.WARNING] == []


class TestArgv:
    """Test suite for the command line, which is pure and so asserted literally."""

    def test_the_argv_matches_what_ci_has_always_run(self, tmp_path):
        """Parity baseline: the same arguments, in the same order, as CI's command."""
        settings = EvaluationSettings(
            model=MODEL,
            service_url=SERVICE_URL,
            backend_args={BACKEND_ARG_APPLY_CHAT_TEMPLATE: True},
        )
        output_path = tmp_path / BACKEND_OUTPUT_SUBDIR

        assert LmEvalBackend().build_argv(settings, output_path) == [
            "lm_eval",
            "--model",
            "local-completions",
            "--model_args",
            json.dumps(
                {
                    "model": MODEL,
                    "base_url": "http://localhost:8000/v1/completions",
                    "num_concurrent": 32,
                    "max_retries": 10,
                    "timeout": 600,
                    "max_gen_toks": 8192,
                    "max_length": 16384,
                    "tokenizer": MODEL,
                }
            ),
            "--tasks",
            "gsm8k",
            "--batch_size",
            "auto",
            "--num_fewshot",
            "5",
            "--output_path",
            str(output_path),
            "--apply_chat_template",
        ]

    def test_the_chat_template_flag_is_off_unless_asked_for(self, backend, settings, tmp_path):
        assert "--apply_chat_template" not in backend.build_argv(settings, tmp_path)

    def test_a_model_that_may_not_be_instruction_tuned_only_warns(self, backend, tmp_path, caplog):
        """CI warns rather than refusing, and the name is the only signal available."""
        settings = EvaluationSettings(
            model="meta-llama/Llama-3.1-8B",
            service_url=SERVICE_URL,
            backend_args={BACKEND_ARG_APPLY_CHAT_TEMPLATE: True},
        )

        argv = backend.build_argv(settings, tmp_path)

        assert "--apply_chat_template" in argv
        assert "instruction-tuned" in caplog.text

    def test_an_instruction_tuned_name_does_not_warn(self, backend, tmp_path, caplog):
        settings = EvaluationSettings(
            model="meta-llama/Llama-3.1-8B-Instruct",
            service_url=SERVICE_URL,
            backend_args={BACKEND_ARG_APPLY_CHAT_TEMPLATE: True},
        )

        backend.build_argv(settings, tmp_path)

        assert "instruction-tuned" not in caplog.text

    def test_the_endpoint_settings_size_the_request_load(self, backend, tmp_path):
        """Concurrency, timeout and retries are generic settings, not lm-eval vocabulary."""
        settings = EvaluationSettings(
            model=MODEL,
            service_url=SERVICE_URL,
            concurrency=64,
            request_timeout_seconds=1800,
            max_retries=3,
        )

        model_args = model_args_of(backend.build_argv(settings, tmp_path))

        assert model_args["num_concurrent"] == 64
        assert model_args["timeout"] == 1800
        assert model_args["max_retries"] == 3

    def test_trust_remote_code_travels_in_the_model_args(self, backend, tmp_path):
        """Where CI puts it, so the tokenizer of a custom-code model can load."""
        settings = EvaluationSettings(
            model=MODEL,
            service_url=SERVICE_URL,
            backend_args={BACKEND_ARG_TRUST_REMOTE_CODE: True},
        )

        assert model_args_of(backend.build_argv(settings, tmp_path))["trust_remote_code"] is True

    def test_trust_remote_code_is_absent_rather_than_false(self, backend, settings, tmp_path):
        assert "trust_remote_code" not in model_args_of(backend.build_argv(settings, tmp_path))

    def test_the_tokenizer_defaults_to_the_served_model(self, backend, settings, tmp_path):
        assert model_args_of(backend.build_argv(settings, tmp_path))["tokenizer"] == MODEL

    def test_the_tokenizer_can_be_overridden(self, backend, tmp_path):
        settings = EvaluationSettings(
            model=MODEL,
            service_url=SERVICE_URL,
            backend_args={BACKEND_ARG_TOKENIZER: "openai/gpt-oss-120b"},
        )

        assert model_args_of(backend.build_argv(settings, tmp_path))["tokenizer"] == "openai/gpt-oss-120b"

    def test_the_model_type_can_be_overridden(self, backend, tmp_path):
        settings = EvaluationSettings(
            model=MODEL,
            service_url=SERVICE_URL,
            backend_args={BACKEND_ARG_MODEL_TYPE: "local-chat-completions"},
        )

        argv = backend.build_argv(settings, tmp_path)

        assert argv[argv.index("--model") + 1] == "local-chat-completions"

    def test_a_sample_cap_is_passed_through(self, backend, tmp_path):
        settings = EvaluationSettings(model=MODEL, service_url=SERVICE_URL, limit=5)

        argv = backend.build_argv(settings, tmp_path)

        assert argv[argv.index("--limit") + 1] == "5"

    def test_no_cap_means_no_limit_flag(self, backend, settings, tmp_path):
        """CI evaluates whole splits, and a stray cap would change every score."""
        assert "--limit" not in backend.build_argv(settings, tmp_path)

    def test_task_defaults_are_left_alone_when_fewshot_is_unset(self, backend, tmp_path):
        settings = EvaluationSettings(model=MODEL, service_url=SERVICE_URL, num_fewshot=None)

        assert "--num_fewshot" not in backend.build_argv(settings, tmp_path)

    def test_several_tasks_are_passed_as_one_comma_separated_value(self, backend, tmp_path):
        settings = EvaluationSettings(model=MODEL, service_url=SERVICE_URL, tasks=["gsm8k", "mmlu"])

        argv = backend.build_argv(settings, tmp_path)

        assert argv[argv.index("--tasks") + 1] == "gsm8k,mmlu"

    def test_backend_args_this_backend_cannot_use_are_reported_and_ignored(self, backend, tmp_path, caplog):
        """Silently dropping an argument is how a typo becomes a wrong measurement."""
        settings = EvaluationSettings(
            model=MODEL,
            service_url=SERVICE_URL,
            backend_args={"apply_chat_temlpate": True, "eval_type": "openai_api"},
        )

        argv = backend.build_argv(settings, tmp_path)

        assert "--apply_chat_template" not in argv
        assert "apply_chat_temlpate" in caplog.text
        assert "eval_type" in caplog.text


class TestRun:
    """Test suite for running the command and loading what it wrote."""

    def test_a_successful_run_returns_the_document_and_its_file(
        self, backend, settings, tmp_path, gsm8k_document, fake_process
    ):
        results_file = write_document(tmp_path, gsm8k_document)
        fake_process()

        raw = backend.run(settings, env={}, output_dir=tmp_path)

        assert raw.succeeded is True
        assert raw.document == gsm8k_document
        assert raw.artifacts == (results_file,)

    def test_the_version_comes_from_the_document(self, backend, settings, tmp_path, lm_eval_document, fake_process):
        """One less packaging-sensitive call than asking the command for it."""
        write_document(tmp_path, lm_eval_document(lm_eval_version="0.4.12"))
        fake_process()

        assert backend.run(settings, env={}, output_dir=tmp_path).version == "0.4.12"

    def test_the_backend_runs_with_exactly_the_environment_it_is_given(
        self, backend, settings, tmp_path, gsm8k_document, fake_process
    ):
        """Cache and token variables reach the process because the caller passes them."""
        write_document(tmp_path, gsm8k_document)
        process = fake_process()

        backend.run(settings, env={"HF_HOME": "/workspace/model-cache"}, output_dir=tmp_path)

        assert process.popen.call_args.kwargs["env"] == {"HF_HOME": "/workspace/model-cache"}

    def test_the_backend_output_directory_is_created_under_the_run_directory(
        self, backend, settings, tmp_path, fake_process
    ):
        fake_process()

        backend.run(settings, env={}, output_dir=tmp_path / "run")

        assert (tmp_path / "run" / BACKEND_OUTPUT_SUBDIR).is_dir()

    def test_a_nonzero_exit_fails_the_run_and_quotes_the_last_output(
        self, backend, settings, tmp_path, gsm8k_document, fake_process
    ):
        write_document(tmp_path, gsm8k_document)
        fake_process(exit_code=1, lines=("loading", "Traceback", "ValueError: bad"))

        raw = backend.run(settings, env={}, output_dir=tmp_path)

        assert raw.succeeded is False
        assert "failed with code 1" in raw.failure_reason
        assert "ValueError: bad" in raw.failure_reason
        assert raw.document is None, "a failed run must not report a document it did not produce"

    def test_a_failure_with_no_output_still_explains_itself(self, backend, settings, tmp_path, fake_process):
        fake_process(exit_code=2)

        assert "Unknown error" in backend.run(settings, env={}, output_dir=tmp_path).failure_reason

    def test_a_command_that_cannot_be_started_fails_the_run(self, backend, settings, tmp_path, mocker):
        mocker.patch(POPEN, side_effect=FileNotFoundError("No such file or directory: 'lm_eval'"))

        raw = backend.run(settings, env={}, output_dir=tmp_path)

        assert raw.succeeded is False
        assert "could not be started" in raw.failure_reason

    def test_a_run_that_outlives_its_budget_is_killed_and_reported(
        self, backend, tmp_path, fake_process, immediate_deadline
    ):
        """A stalled evaluation emits no output, so the deadline acts on the process."""
        settings = EvaluationSettings(model=MODEL, service_url=SERVICE_URL, wall_clock_timeout_seconds=900)
        process = fake_process(lines=("starting",))

        raw = backend.run(settings, env={}, output_dir=tmp_path)

        assert immediate_deadline.scheduled == [900], "the deadline is the configured budget"
        assert process.killed is True
        assert raw.succeeded is False
        assert "900s wall-clock budget" in raw.failure_reason
        assert raw.document is None

    def test_a_run_with_no_budget_is_left_alone(
        self, backend, settings, tmp_path, gsm8k_document, fake_process, deadline_timer
    ):
        """CI has no wall-clock budget today; a default one could kill a healthy run."""
        write_document(tmp_path, gsm8k_document)
        fake_process()

        assert backend.run(settings, env={}, output_dir=tmp_path).succeeded is True
        deadline_timer.assert_not_called()

    def test_output_that_cannot_be_read_still_kills_the_process(
        self, backend, settings, tmp_path, fake_process, mocker
    ):
        """The guard has to be reached from here, not only be correct on its own.

        These settings carry no budget, which is what CI runs with, so no deadline
        exists to kill the process later. ``AIMEvaluation._run`` turns the raise
        into a reported failure, so what this asserts is the process, not the
        message.
        """
        process = fake_process()
        mocker.patch(f"{MODULE}._log_output", side_effect=OSError("stream closed"))

        with pytest.raises(OSError):
            backend.run(settings, env={}, output_dir=tmp_path)

        assert process.killed is True
        assert process.returncode == -9, "and it is reaped, rather than left as a zombie"

    def test_a_run_that_wrote_no_results_succeeded_without_scoring(
        self, backend, settings, tmp_path, fake_process, caplog
    ):
        """CI records a NULL score and exits zero here, so this is not a failure."""
        fake_process()

        raw = backend.run(settings, env={}, output_dir=tmp_path)

        assert raw.succeeded is True
        assert raw.document is None
        assert "No lm-eval results file" in caplog.text

    def test_what_the_backend_did_write_is_logged_when_no_results_are_found(
        self, backend, settings, tmp_path, fake_process, caplog
    ):
        """The directory listing is the only clue when lm-eval wrote something else."""
        stray = tmp_path / BACKEND_OUTPUT_SUBDIR / MODEL_SUBDIR / "samples_gsm8k_2026-08-12.jsonl"
        stray.parent.mkdir(parents=True)
        stray.touch()
        fake_process()

        raw = backend.run(settings, env={}, output_dir=tmp_path)

        assert raw.document is None
        assert str(stray) in caplog.text

    def test_an_unreadable_document_fails_the_run(self, backend, settings, tmp_path, fake_process):
        results_file = tmp_path / BACKEND_OUTPUT_SUBDIR / MODEL_SUBDIR / RESULTS_NAME
        results_file.parent.mkdir(parents=True)
        results_file.write_text("{not json", encoding="utf-8")
        fake_process()

        raw = backend.run(settings, env={}, output_dir=tmp_path)

        assert raw.succeeded is False
        assert "could not read lm-eval results" in raw.failure_reason
        assert raw.artifacts == (results_file,), "the unreadable file is still worth publishing"

    def test_a_document_that_is_not_an_object_fails_the_run(self, backend, settings, tmp_path, fake_process):
        write_document(tmp_path, ["not", "a", "document"])
        fake_process()

        assert "not a JSON object" in backend.run(settings, env={}, output_dir=tmp_path).failure_reason


class TestDeadline:
    """Test suite for the wall-clock deadline, which guards the run on its own."""

    def test_a_process_that_outlives_the_budget_is_killed_and_the_event_says_so(self, immediate_deadline):
        process = FakeProcess()

        with _kill_after(process, 900) as expired:
            pass

        assert immediate_deadline.scheduled == [900], "the deadline is the budget it was given"
        assert process.killed is True
        assert expired.is_set() is True

    def test_a_process_that_finishes_in_time_is_left_alone(self, deadline_timer):
        process = FakeProcess()

        with _kill_after(process, 900) as expired:
            process.wait()

        assert process.killed is False
        assert expired.is_set() is False
        # A timer left running would fire long after this run, killing whatever
        # holds the pid by then.
        deadline_timer.return_value.cancel.assert_called_once()

    def test_no_budget_schedules_no_deadline(self, deadline_timer):
        """CI has no wall-clock budget today, so this is the path it takes."""
        with _kill_after(FakeProcess(), None) as expired:
            pass

        deadline_timer.assert_not_called()
        assert expired.is_set() is False

    def test_the_deadline_is_cancelled_even_when_the_run_raises(self, deadline_timer, mocker):
        """Reading the output can fail; a timer left running would kill a later process."""
        read_the_output = mocker.Mock(side_effect=OSError("stream closed"))
        process = FakeProcess()

        with pytest.raises(OSError):
            with _kill_after(process, 900):
                read_the_output()

        deadline_timer.return_value.cancel.assert_called_once()
        # Cancelling the deadline removes the last thing that would have stopped
        # this process, so the guard has to stop it here.
        assert process.killed is True
        assert process.returncode == -9, "a killed process is reaped, not left as a zombie"

    def test_a_run_that_raises_without_a_budget_still_kills_the_process(self, mocker):
        """The path CI takes: no wall-clock budget, so no deadline would ever fire."""
        read_the_output = mocker.Mock(side_effect=OSError("stream closed"))
        process = FakeProcess()

        with pytest.raises(OSError):
            with _kill_after(process, None):
                read_the_output()

        assert process.killed is True

    def test_a_process_that_exited_on_its_own_is_not_killed_afterwards(self):
        """`poll` is what tells an exited process from one the guard must stop."""
        process = FakeProcess(exit_code=1)

        with _kill_after(process, 900) as expired:
            process.wait()

        assert process.killed is False
        assert process.returncode == 1, "the exit code the run reports is the real one"
        assert expired.is_set() is False


class TestOutput:
    """Test suite for streaming what the process says."""

    def test_every_line_is_logged_as_it_arrives(self, caplog):
        tail: deque[str] = deque(maxlen=FAILURE_TAIL_LINES)

        with caplog.at_level(logging.INFO):
            _log_output(FakeProcess(lines=("loading gsm8k", "Requests: 100%")), tail)

        assert "loading gsm8k" in caplog.text
        assert list(tail) == ["loading gsm8k", "Requests: 100%"]

    def test_only_the_last_lines_are_kept_for_a_failure(self, caplog):
        """The tail is what a failure is reported with; the log keeps the rest."""
        lines = tuple(f"line {number}" for number in range(FAILURE_TAIL_LINES + 5))
        tail: deque[str] = deque(maxlen=FAILURE_TAIL_LINES)

        with caplog.at_level(logging.INFO):
            _log_output(FakeProcess(lines=lines), tail)

        assert list(tail) == list(lines[-FAILURE_TAIL_LINES:])
        assert "line 0" in caplog.text, "everything is still logged as it arrives"

    def test_output_that_was_never_piped_is_survivable(self, caplog):
        """Reading a stdout of None would raise, and losing the output should not."""
        process = FakeProcess()
        process.stdout = None
        tail: deque[str] = deque(maxlen=FAILURE_TAIL_LINES)

        _log_output(process, tail)

        assert not tail
        assert "not available" in caplog.text


class TestDiscovery:
    """Test suite for finding the file lm-eval wrote."""

    def test_the_file_is_found_without_rebuilding_the_name_sanitization(
        self, backend, settings, tmp_path, gsm8k_document, fake_process
    ):
        """How lm-eval mangles a model name is its own business, and has changed."""
        results_file = tmp_path / BACKEND_OUTPUT_SUBDIR / "some__other__mangling" / "results_x.json"
        results_file.parent.mkdir(parents=True)
        results_file.write_text(json.dumps(gsm8k_document), encoding="utf-8")
        fake_process()

        assert backend.run(settings, env={}, output_dir=tmp_path).artifacts == (results_file,)

    def test_the_newest_results_file_wins(self, backend, settings, tmp_path, gsm8k_document, fake_process):
        """A re-run into the same directory leaves the earlier document beside it."""
        stale = write_document(tmp_path, gsm8k_document, name="results_2026-08-11T09-00-00.json")
        fresh = write_document(tmp_path, gsm8k_document, name=RESULTS_NAME)
        os.utime(stale, (1_000_000, 1_000_000))
        os.utime(fresh, (2_000_000, 2_000_000))
        fake_process()

        assert backend.run(settings, env={}, output_dir=tmp_path).artifacts == (fresh,)


class TestParse:
    """Test suite for reading scores out of a results document."""

    def test_a_gsm8k_run_parses_into_one_score(self, gsm8k_document):
        (score,) = parse_document(gsm8k_document)

        assert score.task == "gsm8k"
        assert score.dataset == "openai/gsm8k/main"
        assert score.metric == "exact_match"
        assert score.metric_variant == "flexible-extract"
        assert score.value == 0.8317
        assert score.stderr == 0.0103
        assert score.num_samples == 1319
        assert score.num_samples_available == 1319
        assert score.num_fewshot == 5
        assert score.higher_is_better is True

    def test_the_variant_that_was_not_chosen_is_kept_as_detail(self, gsm8k_document):
        """Two extractors over identical generations disagree; neither is discarded."""
        (score,) = parse_document(gsm8k_document)

        assert score.extra_metrics == {"exact_match,strict-match": 0.7892}

    def test_the_preferred_variant_can_be_overridden(self, gsm8k_document, monkeypatch):
        """CI's existing PREFERRED_METRIC_FILTER override keeps working."""
        monkeypatch.setenv(LM_EVAL_PREFERRED_VARIANT_ENV, "strict-match")

        (score,) = parse_document(gsm8k_document)

        assert score.metric_variant == "strict-match"
        assert score.value == 0.7892

    def test_a_task_with_no_filters_reports_the_variant_lm_eval_gave_it(self, lm_eval_document):
        """mmlu and friends report "acc,none" rather than a bare "acc"."""
        document = lm_eval_document(
            {
                "task": "mmlu_anatomy",
                "metric": "acc",
                "variants": {"none": (0.6296, 0.0417)},
                "dataset_path": "hails/mmlu_no_train",
                "dataset_name": None,
                "samples_original": 135,
            }
        )

        (score,) = parse_document(document)

        assert score.metric == "acc"
        assert score.metric_variant == "none"
        assert score.dataset == "hails/mmlu_no_train"

    def test_a_stderr_lm_eval_could_not_compute_is_not_a_number(self, lm_eval_document):
        document = lm_eval_document({"variants": {"flexible-extract": (0.8317, "N/A")}})

        (score,) = parse_document(document)

        assert score.value == 0.8317
        assert score.stderr is None

    def test_a_capped_run_reports_both_counts(self, lm_eval_document):
        """What separates a smoke run from a score comparable with the database."""
        document = lm_eval_document({"samples_effective": 100}, limit=100)

        (score,) = parse_document(document)

        assert score.num_samples == 100
        assert score.num_samples_available == 1319

    def test_sample_counts_fall_back_to_the_task_entry(self, lm_eval_document):
        """Releases that omit the n-samples block still report the count per task."""
        document = lm_eval_document()
        del document["n-samples"]

        (score,) = parse_document(document)

        assert score.num_samples == 1319
        assert score.num_samples_available is None

    def test_every_task_of_a_multi_task_run_is_reported_in_document_order(self, lm_eval_document):
        document = lm_eval_document(
            {"task": "gsm8k"},
            {"task": "mmlu", "metric": "acc", "variants": {"none": (0.42, 0.01)}, "samples_original": 14042},
        )

        assert [score.task for score in parse_document(document)] == ["gsm8k", "mmlu"]

    def test_a_task_reporting_no_accuracy_metric_is_skipped(self, lm_eval_document):
        """Promoting an arbitrary number to "accuracy" is worse than reporting none."""
        document = lm_eval_document()
        document["results"] = {"wikitext": {"word_perplexity,none": 12.4, "name": "wikitext"}}

        assert parse_document(document) == []

    def test_a_minimal_document_still_yields_its_score(self):
        """The shape older CI tests use: a metric, and nothing to interpret it with."""
        (score,) = parse_document({"results": {"gsm8k": {"acc": 0.5}}})

        assert score.value == 0.5
        assert score.metric_variant is None
        assert score.dataset == "gsm8k", "with no configs block, the task names itself"
        assert score.num_samples == 0
        assert score.num_fewshot is None
        assert score.higher_is_better is None
        assert score.stderr is None

    @pytest.mark.parametrize(
        "document",
        [{}, {"results": None}, {"results": "gsm8k"}, {"results": {"gsm8k": "0.5"}}],
        ids=["empty", "null-results", "results-not-a-mapping", "entry-not-a-mapping"],
    )
    def test_a_malformed_document_yields_no_scores_rather_than_raising(self, document):
        """A backend that wrote nonsense should report no score, not crash the harness."""
        assert parse_document(document) == []

    def test_a_metric_value_that_is_not_a_number_is_ignored(self):
        assert parse_document({"results": {"gsm8k": {"acc": "unavailable"}}}) == []

    def test_the_backend_parses_the_document_it_ran(self, backend, gsm8k_document):
        scores = backend.parse(RawRun(document=gsm8k_document))

        assert [score.task for score in scores] == ["gsm8k"]

    def test_the_backend_stamps_its_own_modality_onto_every_score(self, backend, gsm8k_document):
        """What a metric means is the backend's knowledge, not a shared default."""
        (score,) = backend.parse(RawRun(document=gsm8k_document))

        assert backend.modality == MODALITY_TEXT
        assert score.modality == MODALITY_TEXT

    def test_a_parser_can_be_told_what_modality_it_is_reading(self, gsm8k_document):
        """The seam a non-text backend reuses: same records, its own vocabulary."""
        (score,) = parse_document(gsm8k_document, modality="image_3d")

        assert score.modality == "image_3d"

    def test_a_run_with_no_document_parses_to_no_scores(self, backend):
        assert backend.parse(RawRun()) == []
