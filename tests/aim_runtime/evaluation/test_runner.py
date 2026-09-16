# Copyright © Advanced Micro Devices, Inc., or its affiliates.
#
# SPDX-License-Identifier: MIT

"""
Tests for aim_runtime.evaluation.runner.

The runner's job is phase order and severity, so these tests are about which
phase can end a run and what a consumer is left holding when one does. A fake
backend stands in for the real one: no subprocess, no service, no filesystem
beyond ``tmp_path``.
"""

import csv
import json
import logging
from pathlib import Path

import pytest

from aim_runtime.evaluation.backends.base import BackendInfo, EvaluationBackend, RawRun
from aim_runtime.evaluation.results import EvaluationResults, TaskScore
from aim_runtime.evaluation.runner import AIMEvaluation
from aim_runtime.evaluation.settings import EvaluationSettings
from aim_runtime.harness import failed, passed
from aim_runtime.harness.service_checks import WarmupOutcome


class FakeBackend(EvaluationBackend):
    """A backend that answers with whatever the test configured, and records calls."""

    name = "fake"

    def __init__(self, *, info=None, raw=None, scores=(), run_error=None, parse_error=None):
        self.info = info or BackendInfo(name=self.name, available=True, command="fake")
        self.raw = raw if raw is not None else RawRun()
        self.scores = list(scores)
        self.run_error = run_error
        self.parse_error = parse_error
        self.phases: list[str] = []
        self.run_env: dict[str, str] = {}
        self.run_output_dir: Path | None = None

    def probe(self):
        self.phases.append("probe")
        return self.info

    def run(self, settings, *, env, output_dir):
        self.phases.append("run")
        self.run_env = dict(env)
        self.run_output_dir = Path(output_dir)
        if self.run_error:
            raise self.run_error
        return self.raw

    def parse(self, raw):
        self.phases.append("parse")
        if self.parse_error:
            raise self.parse_error
        return list(self.scores)


@pytest.fixture
def settings(tmp_path):
    return EvaluationSettings(
        model="meta-llama/Llama-3.1-8B-Instruct",
        service_url="http://localhost:8000",
        output_dir=tmp_path,
    )


@pytest.fixture
def score():
    return TaskScore(
        task="gsm8k",
        dataset="gsm8k",
        metric="exact_match",
        metric_variant="flexible-extract",
        value=0.82,
        num_samples=1319,
    )


@pytest.fixture(autouse=True)
def warmup(mocker):
    """Warmup succeeds unless a test says otherwise, and never touches a service."""
    return mocker.patch(
        "aim_runtime.evaluation.runner.run_warmup",
        return_value=WarmupOutcome(passed("warmup", "First inference completed in 2s"), 2.0, 1, False),
    )


class TestRunEvaluationSuite:
    """Test suite for phase order and what ends a run."""

    def test_an_unavailable_backend_ends_the_run_before_anything_else(self, settings):
        """One clear reason, and no evaluation attempted."""
        backend = FakeBackend(
            info=BackendInfo(
                name="fake",
                available=False,
                command="lm_eval",
                detail="'lm_eval' was not found; the evaluation dependencies may not be installed",
            )
        )

        results = AIMEvaluation(settings, backend=backend).run_evaluation_suite()

        assert results.success is False
        assert results.failure_reason == "'lm_eval' was not found; the evaluation dependencies may not be installed"
        assert results.tasks == ()
        assert backend.phases == ["probe"]

    def test_an_unavailable_backend_with_no_detail_still_names_itself(self, settings):
        backend = FakeBackend(info=BackendInfo(name="fake", available=False))

        results = AIMEvaluation(settings, backend=backend).run_evaluation_suite()

        assert "fake" in results.failure_reason

    def test_an_unavailable_backend_is_reported_once(self, settings, caplog):
        """This layer logs the reason. The backend only returns it, so it appears once."""
        detail = "'lm_eval' was not found; the evaluation dependencies may not be installed"
        backend = FakeBackend(info=BackendInfo(name="fake", available=False, command="lm_eval", detail=detail))

        AIMEvaluation(settings, backend=backend).run_evaluation_suite()

        logged = [record for record in caplog.records if detail in record.getMessage()]
        assert len(logged) == 1, "the operator needs the reason, and needs it once"
        assert logged[0].levelno == logging.ERROR

    def test_a_scored_run_reports_its_tasks(self, settings, score):
        backend = FakeBackend(raw=RawRun(document={"results": {}}, version="0.4.12"), scores=[score])

        results = AIMEvaluation(settings, backend=backend).run_evaluation_suite()

        assert results.success is True
        assert results.tasks == (score,)
        assert results.accuracy == 0.82
        assert results.backend_version == "0.4.12"
        assert results.elapsed_seconds is not None
        assert backend.phases == ["probe", "run", "parse"]

    def test_the_document_travels_with_the_results(self, settings, score):
        """The runner is the only place holding both, and export needs it."""
        document = {"results": {"gsm8k": {}}}
        backend = FakeBackend(raw=RawRun(document=document), scores=[score])

        results = AIMEvaluation(settings, backend=backend).run_evaluation_suite()

        assert results.document == document
        assert "document" not in results.to_dict()["evaluation"]

    def test_the_backend_version_falls_back_to_the_probe(self, settings, score):
        """A backend that cannot report its version in its results is probed for it."""
        backend = FakeBackend(
            info=BackendInfo(name="fake", available=True, version="1.2.3"),
            raw=RawRun(document={"results": {}}),
            scores=[score],
        )

        results = AIMEvaluation(settings, backend=backend).run_evaluation_suite()

        assert results.backend_version == "1.2.3"

    def test_a_completed_run_that_scored_nothing_is_not_a_failure(self, settings):
        """CI records a NULL score and exits zero for this; the distinction survives."""
        backend = FakeBackend(raw=RawRun())

        results = AIMEvaluation(settings, backend=backend).run_evaluation_suite()

        assert results.success is True
        assert results.accuracy is None
        assert results.tasks == ()
        assert backend.phases == ["probe", "run"]

    def test_a_backend_failure_becomes_the_runs_failure(self, settings):
        backend = FakeBackend(raw=RawRun(failure_reason="lm_eval failed with code 1: boom"))

        results = AIMEvaluation(settings, backend=backend).run_evaluation_suite()

        assert results.success is False
        assert results.failure_reason == "lm_eval failed with code 1: boom"
        assert results.tasks == ()

    def test_a_backend_that_raises_is_reported_rather_than_propagated(self, settings):
        """Both consumers need a result: a NULL-score row, or a HarnessResult."""
        backend = FakeBackend(run_error=OSError("no such file"))

        results = AIMEvaluation(settings, backend=backend).run_evaluation_suite()

        assert results.success is False
        assert "OSError" in results.failure_reason
        assert "no such file" in results.failure_reason

    def test_results_that_cannot_be_read_are_a_failure(self, settings):
        """Not a run that scored nothing — we know a score was there to read."""
        backend = FakeBackend(raw=RawRun(document={"results": {}}), parse_error=KeyError("results"))

        results = AIMEvaluation(settings, backend=backend).run_evaluation_suite()

        assert results.success is False
        assert "could not read" in results.failure_reason

    def test_the_backend_is_given_the_whole_environment(self, settings, monkeypatch):
        """The backend may live in another virtualenv, so it needs PATH as well as HF_TOKEN."""
        monkeypatch.setenv("HF_TOKEN", "secret")
        backend = FakeBackend()

        AIMEvaluation(settings, backend=backend).run_evaluation_suite()

        assert backend.run_env["HF_TOKEN"] == "secret"
        assert "PATH" in backend.run_env

    def test_the_backend_writes_under_the_configured_output_dir(self, settings, tmp_path):
        backend = FakeBackend()

        AIMEvaluation(settings, backend=backend).run_evaluation_suite()

        assert backend.run_output_dir == tmp_path

    def test_the_default_backend_is_the_lm_eval_adapter(self, settings):
        """Until a second backend exists, settings.backend selects nothing."""
        from aim_runtime.evaluation.backends.lm_eval import LmEvalBackend

        assert isinstance(AIMEvaluation(settings).backend, LmEvalBackend)


class TestWarmup:
    """Test suite for the advisory phase."""

    def test_warmup_failure_does_not_stop_the_evaluation(self, settings, warmup):
        """CI logs a warning and proceeds; failing here would fail healthy runs."""
        warmup.return_value = WarmupOutcome(failed("warmup", "No successful inference within 300s"), 300.0, 5, False)
        backend = FakeBackend(raw=RawRun())

        results = AIMEvaluation(settings, backend=backend).run_evaluation_suite()

        assert results.success is True
        assert "run" in backend.phases

    def test_warmup_raising_does_not_stop_the_evaluation(self, settings, warmup):
        warmup.side_effect = RuntimeError("connection reset")
        backend = FakeBackend(raw=RawRun())

        results = AIMEvaluation(settings, backend=backend).run_evaluation_suite()

        assert results.success is True
        assert "run" in backend.phases

    def test_warmup_warms_the_configured_model_within_its_budget(self, settings, warmup):
        backend = FakeBackend()

        AIMEvaluation(settings, backend=backend, max_warmup_time=42).run_evaluation_suite()

        warmup.assert_called_once_with(
            "http://localhost:8000",
            "meta-llama/Llama-3.1-8B-Instruct",
            max_warmup_time=42,
        )

    def test_a_zero_budget_skips_warmup(self, settings, warmup):
        backend = FakeBackend()

        AIMEvaluation(settings, backend=backend, max_warmup_time=0).run_evaluation_suite()

        warmup.assert_not_called()
        assert "run" in backend.phases


class TestExportResults:
    """Test suite for the files an evaluation leaves behind."""

    @pytest.fixture
    def results(self, settings, score):
        return EvaluationResults(
            settings=settings,
            tasks=(score,),
            backend_version="0.4.12",
            elapsed_seconds=12.5,
            document={"results": {"gsm8k": {"exact_match,flexible-extract": 0.82}}},
        )

    def test_it_writes_the_envelope_the_rows_and_the_document(self, settings, results, tmp_path):
        written = AIMEvaluation(settings).export_results(results, output_dir=tmp_path)

        assert [path.name for path in written] == [
            "accuracy_evaluation_results.json",
            "accuracy_evaluation_results.csv",
            "evaluation_backend_results.json",
        ]
        assert all(path.exists() for path in written)

    def test_the_envelope_is_what_a_consumer_publishes(self, settings, results, tmp_path):
        json_path, _, document_path = AIMEvaluation(settings).export_results(results, output_dir=tmp_path)

        published = json.loads(json_path.read_text(encoding="utf-8"))
        assert published["accuracy"] == 0.82
        assert published["primary_task"] == "gsm8k"
        assert published["evaluation"]["backend_version"] == "0.4.12"

        document = json.loads(document_path.read_text(encoding="utf-8"))
        assert document == {"results": {"gsm8k": {"exact_match,flexible-extract": 0.82}}}

    def test_the_csv_carries_one_row_per_task(self, settings, results, tmp_path):
        _, csv_path, _ = AIMEvaluation(settings).export_results(results, output_dir=tmp_path)

        rows = list(csv.DictReader(csv_path.read_text(encoding="utf-8").splitlines()))
        assert len(rows) == 1
        assert rows[0]["task"] == "gsm8k"
        assert rows[0]["value"] == "0.82"
        assert rows[0]["metric_variant"] == "flexible-extract"

    def test_the_csv_has_its_header_even_when_nothing_scored(self, settings, tmp_path):
        """So a scoreless run is distinguishable from a missing file."""
        results = EvaluationResults(settings=settings, failure_reason="lm_eval failed with code 1")

        _, csv_path = AIMEvaluation(settings).export_results(results, output_dir=tmp_path)

        lines = csv_path.read_text(encoding="utf-8").splitlines()
        assert lines[0].startswith("model,backend,backend_version,task")
        assert len(lines) == 1

    def test_a_run_without_a_document_writes_only_the_two(self, settings, score, tmp_path):
        results = EvaluationResults(settings=settings, tasks=(score,))

        written = AIMEvaluation(settings).export_results(results, output_dir=tmp_path)

        assert [path.name for path in written] == [
            "accuracy_evaluation_results.json",
            "accuracy_evaluation_results.csv",
        ]

    def test_an_absolute_environment_override_wins(self, settings, results, tmp_path, monkeypatch):
        """The accuracy action sets absolute paths; AIMBenchmark honours them the same way."""
        elsewhere = tmp_path / "action" / "results.json"
        elsewhere.parent.mkdir()
        monkeypatch.setenv("ACCURACY_JSON_FILE", str(elsewhere))

        json_path, _, _ = AIMEvaluation(settings).export_results(results, output_dir=tmp_path)

        assert json_path == elsewhere
        assert elsewhere.exists()

    def test_a_bare_environment_override_renames_within_the_directory(self, settings, results, tmp_path, monkeypatch):
        monkeypatch.setenv("ACCURACY_CSV_FILE", "scores.csv")

        _, csv_path, _ = AIMEvaluation(settings).export_results(results, output_dir=tmp_path)

        assert csv_path == tmp_path / "scores.csv"

    def test_it_falls_back_to_the_settings_output_dir(self, settings, results, tmp_path):
        written = AIMEvaluation(settings).export_results(results)

        assert all(path.parent == tmp_path for path in written)

    def test_it_creates_a_directory_that_does_not_exist_yet(self, settings, results, tmp_path):
        target = tmp_path / "nested" / "run"

        written = AIMEvaluation(settings).export_results(results, output_dir=target)

        assert target.is_dir()
        assert all(path.exists() for path in written)
