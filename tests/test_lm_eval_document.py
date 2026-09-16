# Copyright © Advanced Micro Devices, Inc., or its affiliates.
#
# SPDX-License-Identifier: MIT

"""
Tests for the lm-eval results-document builders in conftest.

These pin the wire format the accuracy parsers are written against - the key
spellings ("<metric>,<variant>", "n-samples", "sample_len") come from lm-eval,
not from us, so they are asserted literally here rather than shared with the
builder.
"""

import json

import pytest


class TestBuildLmEvalDocument:
    """Test suite for build_lm_eval_document and its fixtures."""

    def test_default_document_is_a_full_gsm8k_run(self, gsm8k_document):
        """The no-argument document describes one uncapped 5-shot gsm8k task."""
        assert list(gsm8k_document["results"]) == ["gsm8k"]
        assert gsm8k_document["n-shot"] == {"gsm8k": 5}
        assert gsm8k_document["n-samples"] == {"gsm8k": {"original": 1319, "effective": 1319}}
        assert gsm8k_document["config"]["limit"] is None
        assert gsm8k_document["configs"]["gsm8k"]["dataset_path"] == "openai/gsm8k"
        assert gsm8k_document["configs"]["gsm8k"]["dataset_name"] == "main"

    def test_both_answer_extractors_are_reported_with_their_stderrs(self, gsm8k_document):
        """Each variant contributes a "<metric>,<variant>" and a stderr key."""
        task = gsm8k_document["results"]["gsm8k"]

        assert task["exact_match,strict-match"] == 0.7892
        assert task["exact_match_stderr,strict-match"] == 0.0112
        assert task["exact_match,flexible-extract"] == 0.8317
        assert task["exact_match_stderr,flexible-extract"] == 0.0103

    def test_task_entry_carries_the_fixed_keys(self, gsm8k_document):
        """`name`, `alias` and `sample_len` sit beside the metric keys."""
        task = gsm8k_document["results"]["gsm8k"]

        assert task["name"] == "gsm8k"
        assert task["alias"] == "gsm8k"
        assert task["sample_len"] == 1319

    def test_alias_can_differ_from_the_task_name(self, lm_eval_document):
        """A task_alias in the task config surfaces as `alias`."""
        document = lm_eval_document({"task": "gsm8k", "alias": "gsm8k (5-shot)"})

        assert document["results"]["gsm8k"]["name"] == "gsm8k"
        assert document["results"]["gsm8k"]["alias"] == "gsm8k (5-shot)"

    def test_capped_run_reports_fewer_effective_samples(self, lm_eval_document):
        """A --limit run keeps `original` and drops `effective` and `sample_len`."""
        document = lm_eval_document({"samples_effective": 50}, limit=50)

        assert document["n-samples"]["gsm8k"] == {"original": 1319, "effective": 50}
        assert document["results"]["gsm8k"]["sample_len"] == 50
        assert document["config"]["limit"] == 50

    def test_task_without_filters_reports_a_none_variant(self, lm_eval_document):
        """Tasks with no filter_list still name their variant "none"."""
        document = lm_eval_document(
            {
                "task": "mmlu_anatomy",
                "metric": "acc",
                "variants": {"none": (0.6074, 0.0422)},
                "dataset_path": "hails/mmlu_no_train",
                "dataset_name": "anatomy",
                "output_type": "multiple_choice",
                "samples_original": 135,
            }
        )
        task = document["results"]["mmlu_anatomy"]

        assert task["acc,none"] == 0.6074
        assert task["acc_stderr,none"] == 0.0422
        assert "filter_list" not in document["configs"]["mmlu_anatomy"]

    def test_every_block_is_keyed_by_every_task(self, lm_eval_document):
        """A multi-task run keys the per-task blocks by each task."""
        document = lm_eval_document(
            {"task": "gsm8k"},
            {"task": "mmlu_anatomy", "metric": "acc", "variants": {"none": (0.6074, 0.0422)}, "num_fewshot": 0},
        )
        tasks = {"gsm8k", "mmlu_anatomy"}

        for block in ("results", "configs", "versions", "n-shot", "higher_is_better", "n-samples", "group_subtasks"):
            assert set(document[block]) == tasks, block
        assert document["n-shot"] == {"gsm8k": 5, "mmlu_anatomy": 0}
        assert document["higher_is_better"]["mmlu_anatomy"] == {"acc": True}

    def test_uncomputable_stderr_is_the_na_string(self, lm_eval_document):
        """lm-eval writes "N/A", not a float, when it cannot bootstrap a stderr."""
        document = lm_eval_document({"variants": {"strict-match": (1.0, "N/A")}, "samples_effective": 1})

        assert document["results"]["gsm8k"]["exact_match_stderr,strict-match"] == "N/A"

    def test_overrides_reach_the_top_level(self, lm_eval_document):
        """Overrides add or replace top-level keys, for malformed-input tests."""
        document = lm_eval_document(results={}, groups={"mmlu": {"acc,none": 0.71}})

        assert document["results"] == {}
        assert document["groups"] == {"mmlu": {"acc,none": 0.71}}

    def test_run_metadata_identifies_the_backend_and_model(self, lm_eval_document):
        """The document reports its own lm-eval version and the served model."""
        document = lm_eval_document(model_name="meta-llama/Llama-3.1-8B-Instruct")

        assert document["lm_eval_version"] == "0.4.12"
        assert document["model_name"] == "meta-llama/Llama-3.1-8B-Instruct"
        assert document["model_name_sanitized"] == "meta-llama__Llama-3.1-8B-Instruct"
        assert json.loads(document["config"]["model_args"])["model"] == "meta-llama/Llama-3.1-8B-Instruct"

    def test_document_survives_a_json_round_trip(self, gsm8k_document):
        """Parsers receive the document via json.load, so it must serialise."""
        assert json.loads(json.dumps(gsm8k_document)) == gsm8k_document

    def test_unknown_task_field_is_rejected(self, lm_eval_document):
        """Mapping specs are validated, so a typo fails loudly."""
        with pytest.raises(TypeError):
            lm_eval_document({"tasks": "gsm8k"})
