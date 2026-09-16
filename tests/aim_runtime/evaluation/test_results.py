# Copyright © Advanced Micro Devices, Inc., or its affiliates.
#
# SPDX-License-Identifier: MIT

"""
Tests for aim_runtime.evaluation.results.

This is the published output contract, so the assertions here spell keys out as
literals instead of sharing them with the module under test: a rename should fail
a test rather than quietly travel through it.
"""

import dataclasses
import json

import pytest

from aim_runtime.evaluation.config import BACKEND_LM_EVAL, SCHEMA_VERSION
from aim_runtime.evaluation.results import CSV_COLUMNS, MODALITY_TEXT, EvaluationResults, TaskScore
from aim_runtime.evaluation.settings import EvaluationSettings

MODEL = "openai/gpt-oss-20b"
SERVICE_URL = "http://localhost:8000"

#: A complete gsm8k record, in the shape the lm-eval adapter will produce.
GSM8K = {
    "task": "gsm8k",
    "dataset": "openai/gsm8k/main",
    "metric": "exact_match",
    "metric_variant": "flexible-extract",
    "value": 0.8317,
    "stderr": 0.0103,
    "num_samples": 1319,
    "num_samples_available": 1319,
    "num_fewshot": 5,
    "higher_is_better": True,
}


@pytest.fixture
def settings() -> EvaluationSettings:
    return EvaluationSettings(model=MODEL, service_url=SERVICE_URL)


@pytest.fixture
def gsm8k_score() -> TaskScore:
    return TaskScore(**GSM8K)


@pytest.fixture
def results(settings, gsm8k_score) -> EvaluationResults:
    return EvaluationResults(
        settings=settings,
        tasks=(gsm8k_score,),
        backend_version="0.4.12",
        elapsed_seconds=612.5,
    )


class TestTaskScore:
    """Test suite for a single task's score."""

    @pytest.mark.parametrize("omitted", ["task", "dataset", "metric", "value", "num_samples"])
    def test_a_score_cannot_be_built_without_the_fields_that_interpret_it(self, omitted):
        """A value with no metric, or a metric with no data behind it, is not a result."""
        fields = {key: value for key, value in GSM8K.items() if key != omitted}

        with pytest.raises(TypeError, match=omitted):
            TaskScore(**fields)

    def test_fields_have_to_be_named(self):
        """Twelve fields, several of them numbers: positional construction misreads easily."""
        with pytest.raises(TypeError):
            TaskScore("gsm8k", "openai/gsm8k/main", "exact_match", 0.8317, 1319)

    def test_a_score_is_immutable(self, gsm8k_score):
        with pytest.raises(dataclasses.FrozenInstanceError):
            gsm8k_score.value = 0.9

    def test_detail_that_was_not_reported_defaults_to_null(self):
        score = TaskScore(task="mmlu", dataset="hails/mmlu_no_train", metric="acc", value=0.42, num_samples=100)

        assert score.metric_variant is None
        assert score.stderr is None
        assert score.num_samples_available is None
        assert score.num_fewshot is None
        assert score.higher_is_better is None
        assert score.extra_metrics == {}
        assert score.modality == MODALITY_TEXT

    def test_to_dict_reports_the_whole_record(self, gsm8k_score):
        assert gsm8k_score.to_dict() == {
            "task": "gsm8k",
            "dataset": "openai/gsm8k/main",
            "modality": "text",
            "metric": "exact_match",
            "metric_variant": "flexible-extract",
            "value": 0.8317,
            "stderr": 0.0103,
            "num_samples": 1319,
            "num_samples_available": 1319,
            "num_fewshot": 5,
            "higher_is_better": True,
            "extra_metrics": {},
        }

    def test_to_dict_publishes_every_field_of_the_record(self):
        """Guards the hand-written mapping: a field added later must be published too.

        ``to_dict`` restates the field list, so nothing fails if a new field is
        only added to the dataclass — it would simply be missing from every
        envelope, including CI's.
        """
        assert set(TaskScore(**GSM8K).to_dict()) == {field.name for field in dataclasses.fields(TaskScore)}

    def test_unknown_detail_is_reported_as_null_rather_than_dropped(self):
        """Every record has the same keys, so a consumer need not test for absence."""
        score = TaskScore(task="mmlu", dataset="hails/mmlu_no_train", metric="acc", value=0.42, num_samples=100)

        assert score.to_dict()["num_fewshot"] is None
        assert score.to_dict()["stderr"] is None

    def test_other_metrics_from_the_same_task_travel_as_detail(self):
        """The headline value stays unambiguous while nothing measured is thrown away."""
        score = TaskScore(**GSM8K, extra_metrics={"exact_match,strict-match": 0.8218})

        assert score.to_dict()["extra_metrics"] == {"exact_match,strict-match": 0.8218}
        assert score.to_dict()["value"] == 0.8317


class TestEvaluationResults:
    """Test suite for the published run envelope."""

    def test_the_headline_number_is_published_flat(self, results):
        """A consumer that wants one float needs no knowledge of the envelope."""
        published = results.to_dict()

        assert published["accuracy"] == 0.8317
        assert published["primary_task"] == "gsm8k"
        assert 0.0 <= published["accuracy"] <= 1.0

    def test_the_envelope_describes_the_run(self, results):
        envelope = results.to_dict()["evaluation"]
        run = {key: value for key, value in envelope.items() if key not in {"settings", "tasks"}}

        assert run == {
            "schema_version": SCHEMA_VERSION,
            "backend": BACKEND_LM_EVAL,
            "backend_version": "0.4.12",
            "model": MODEL,
            "service_url": SERVICE_URL,
            "elapsed_seconds": 612.5,
            "failure_reason": None,
        }

    def test_the_scores_are_nested_under_the_envelope(self, results, gsm8k_score):
        assert results.to_dict()["evaluation"]["tasks"] == [gsm8k_score.to_dict()]

    def test_the_settings_that_produced_the_score_travel_with_it(self, results):
        reported = results.to_dict()["evaluation"]["settings"]

        assert reported["tasks"] == ["gsm8k"]
        assert reported["num_fewshot"] == 5
        assert reported["limit"] is None
        assert reported["concurrency"] > 0
        assert reported["backend_args"] == {}

    @pytest.mark.parametrize("reported_elsewhere", ["model", "service_url", "backend", "output_dir"])
    def test_the_settings_block_does_not_repeat_the_envelope(self, results, reported_elsewhere):
        """Except output_dir, which is local to the runner and says nothing about the score."""
        assert reported_elsewhere not in results.to_dict()["evaluation"]["settings"]

    def test_the_published_mapping_survives_a_json_round_trip(self, results):
        published = results.to_dict()

        assert json.loads(json.dumps(published)) == published

    def test_the_first_task_reported_is_the_primary_one(self, settings, gsm8k_score):
        mmlu = TaskScore(task="mmlu", dataset="hails/mmlu_no_train", metric="acc", value=0.42, num_samples=100)
        results = EvaluationResults(settings=settings, tasks=(mmlu, gsm8k_score))

        assert results.primary is mmlu
        assert results.to_dict()["primary_task"] == "mmlu"
        assert results.to_dict()["accuracy"] == 0.42

    def test_a_run_that_scored_nothing_is_still_a_completed_run(self, settings):
        """CI records a NULL score for this and exits zero; parity needs the distinction."""
        results = EvaluationResults(settings=settings)

        assert results.success is True
        assert results.accuracy is None
        assert results.to_dict()["accuracy"] is None
        assert results.to_dict()["primary_task"] is None
        assert results.to_dict()["evaluation"]["tasks"] == []

    def test_a_run_that_broke_says_why(self, settings):
        results = EvaluationResults(settings=settings, failure_reason="lm_eval is not installed")

        assert results.success is False
        assert results.accuracy is None
        assert results.to_dict()["evaluation"]["failure_reason"] == "lm_eval is not installed"

    def test_results_are_immutable(self, results):
        with pytest.raises(dataclasses.FrozenInstanceError):
            results.failure_reason = "too late"


class TestCsvRows:
    """Test suite for the flat per-task export."""

    def test_every_row_has_exactly_the_declared_columns(self, results):
        assert [tuple(row) for row in results.to_csv_rows()] == [CSV_COLUMNS]

    def test_the_columns_track_the_record_they_flatten(self):
        """Rows are built with ``.get``, so a field renamed here empties a column silently."""
        run_context = {"model", "backend", "backend_version"}
        flattened = {field.name for field in dataclasses.fields(TaskScore)} - {"extra_metrics"}

        assert set(CSV_COLUMNS) - run_context == flattened

    def test_a_row_carries_its_score_and_enough_run_context_to_stand_alone(self, results):
        (row,) = results.to_csv_rows()

        assert row["model"] == MODEL
        assert row["backend"] == BACKEND_LM_EVAL
        assert row["backend_version"] == "0.4.12"
        assert row["task"] == "gsm8k"
        assert row["dataset"] == "openai/gsm8k/main"
        assert row["metric"] == "exact_match"
        assert row["metric_variant"] == "flexible-extract"
        assert row["value"] == 0.8317
        assert row["num_samples"] == 1319

    def test_there_is_one_row_per_task(self, settings, gsm8k_score):
        mmlu = TaskScore(task="mmlu", dataset="hails/mmlu_no_train", metric="acc", value=0.42, num_samples=100)
        results = EvaluationResults(settings=settings, tasks=(gsm8k_score, mmlu))

        assert [row["task"] for row in results.to_csv_rows()] == ["gsm8k", "mmlu"]

    def test_mappings_do_not_become_columns(self, settings):
        """extra_metrics has no flat form; the JSON is where it is readable."""
        score = TaskScore(**GSM8K, extra_metrics={"exact_match,strict-match": 0.8218})
        results = EvaluationResults(settings=settings, tasks=(score,))

        assert "extra_metrics" not in results.to_csv_rows()[0]

    def test_a_run_without_scores_produces_no_rows(self, settings):
        """The export still writes a header, so an empty file is distinguishable from a missing one."""
        assert EvaluationResults(settings=settings).to_csv_rows() == []
