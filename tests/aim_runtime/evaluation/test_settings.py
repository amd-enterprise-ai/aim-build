# Copyright © Advanced Micro Devices, Inc., or its affiliates.
#
# SPDX-License-Identifier: MIT

"""
Tests for aim_runtime.evaluation.settings.

Cover the resolution order settings are built through, the normalisations both
consumers rely on, and the CI wire format AccuracyEvalConfig has to keep.
"""

import json
from pathlib import Path

import pytest
from pydantic import ValidationError

from aim_runtime.evaluation import settings as evaluation_settings
from aim_runtime.evaluation.config import (
    BACKEND_LM_EVAL,
    CONCURRENCY_HIGH_TP,
    CONCURRENCY_LOW_TP,
    DEFAULT_CONFIG_FILENAME,
    MAX_RETRIES,
    NUM_FEWSHOT,
    REQUEST_TIMEOUT_SECONDS_ACCELERATED,
    REQUEST_TIMEOUT_SECONDS_CPU,
)
from aim_runtime.evaluation.settings import (
    AccuracyEvalConfig,
    EvaluationSettings,
    concurrency_for,
    load_evaluation_config,
    request_timeout_for,
)

MODEL = "openai/gpt-oss-20b"
SERVICE_URL = "http://localhost:8000"


@pytest.fixture
def settings() -> EvaluationSettings:
    """Settings with only the two values no config file can supply."""
    return EvaluationSettings(model=MODEL, service_url=SERVICE_URL)


class TestEvaluationSettings:
    """Test suite for the EvaluationSettings model."""

    def test_defaults_come_from_the_constants(self, settings):
        assert settings.tasks == ("gsm8k",)
        assert settings.num_fewshot == NUM_FEWSHOT
        assert settings.limit is None
        assert settings.concurrency == CONCURRENCY_LOW_TP
        assert settings.request_timeout_seconds == REQUEST_TIMEOUT_SECONDS_ACCELERATED
        assert settings.max_retries == MAX_RETRIES
        assert settings.wall_clock_timeout_seconds is None
        assert settings.backend == BACKEND_LM_EVAL
        assert settings.backend_args == {}

    @pytest.mark.parametrize(
        "value",
        ["gsm8k,mmlu", " gsm8k , mmlu ", ["gsm8k", "mmlu"], ("gsm8k", " mmlu")],
        ids=["string", "padded-string", "list", "tuple"],
    )
    def test_task_lists_normalize_to_a_tuple_of_names(self, value):
        """CI carries one comma-separated string; a YAML config is a list."""
        assert EvaluationSettings(model=MODEL, service_url=SERVICE_URL, tasks=value).tasks == ("gsm8k", "mmlu")

    def test_tasks_argument_is_the_command_line_spelling(self):
        settings = EvaluationSettings(model=MODEL, service_url=SERVICE_URL, tasks=["gsm8k", "mmlu"])

        assert settings.tasks_argument == "gsm8k,mmlu"

    @pytest.mark.parametrize("value", ["", ",", [], [" "]], ids=["empty", "separator", "empty-list", "blank"])
    def test_an_empty_task_list_is_rejected(self, value):
        """Evaluating nothing would report a vacuous pass."""
        with pytest.raises(ValidationError, match="at least one task"):
            EvaluationSettings(model=MODEL, service_url=SERVICE_URL, tasks=value)

    def test_a_task_list_that_is_not_a_list_of_names_is_rejected(self):
        """Normalization passes an unusable value on for the type check to report."""
        with pytest.raises(ValidationError):
            EvaluationSettings(model=MODEL, service_url=SERVICE_URL, tasks=5)

    def test_trailing_slash_is_dropped_from_the_service_url(self):
        """Endpoint paths are appended, so a slash would double up."""
        settings = EvaluationSettings(model=MODEL, service_url="http://localhost:8000/")

        assert settings.service_url == SERVICE_URL

    @pytest.mark.parametrize("value", ["localhost:8000", "/v1/completions", ""])
    def test_a_non_absolute_service_url_is_rejected(self, value):
        with pytest.raises(ValidationError, match="must be absolute"):
            EvaluationSettings(model=MODEL, service_url=value)

    def test_an_unknown_backend_is_rejected(self):
        with pytest.raises(ValidationError, match="unknown backend"):
            EvaluationSettings(model=MODEL, service_url=SERVICE_URL, backend="evalscope")

    def test_backend_vocabulary_cannot_leak_into_the_core(self):
        """lm-eval's own arguments belong in backend_args, not beside the core."""
        with pytest.raises(ValidationError):
            EvaluationSettings(model=MODEL, service_url=SERVICE_URL, apply_chat_template=True)

    def test_backend_args_pass_through_untouched(self):
        settings = EvaluationSettings(
            model=MODEL,
            service_url=SERVICE_URL,
            backend_args={"apply_chat_template": True, "trust_remote_code": True},
        )

        assert settings.backend_args == {"apply_chat_template": True, "trust_remote_code": True}

    def test_settings_are_frozen(self, settings):
        with pytest.raises(ValidationError):
            settings.limit = 10

    @pytest.mark.parametrize("field, value", [("limit", 0), ("concurrency", 0), ("request_timeout_seconds", -1)])
    def test_non_positive_budgets_are_rejected(self, field, value):
        with pytest.raises(ValidationError):
            EvaluationSettings(model=MODEL, service_url=SERVICE_URL, **{field: value})


class TestEndpointDerivedTiers:
    """Test suite for the concurrency and timeout tier helpers."""

    @pytest.mark.parametrize(
        "tensor_parallel_size, expected",
        [(1, CONCURRENCY_LOW_TP), (2, CONCURRENCY_LOW_TP), (4, CONCURRENCY_HIGH_TP), (8, CONCURRENCY_HIGH_TP)],
    )
    def test_concurrency_scales_with_endpoint_width(self, tensor_parallel_size, expected):
        assert concurrency_for(tensor_parallel_size) == expected

    def test_a_cpu_endpoint_gets_the_generous_request_timeout(self):
        assert request_timeout_for(cpu_endpoint=True) == REQUEST_TIMEOUT_SECONDS_CPU
        assert request_timeout_for(cpu_endpoint=False) == REQUEST_TIMEOUT_SECONDS_ACCELERATED


class TestResolve:
    """Test suite for EvaluationSettings.resolve and its precedence order."""

    def test_shipped_defaults_produce_a_usable_run(self):
        """No config file: the file shipped beside the module is the default."""
        settings = EvaluationSettings.resolve(model=MODEL, service_url=SERVICE_URL)

        assert settings.model == MODEL
        assert settings.service_url == SERVICE_URL
        assert settings.tasks == ("gsm8k",)
        assert settings.backend == BACKEND_LM_EVAL
        assert settings.backend_args["apply_chat_template"] is True

    def test_endpoint_shape_drives_concurrency_and_timeout(self):
        settings = EvaluationSettings.resolve(
            model=MODEL, service_url=SERVICE_URL, tensor_parallel_size=8, cpu_endpoint=True
        )

        assert settings.concurrency == CONCURRENCY_HIGH_TP
        assert settings.request_timeout_seconds == REQUEST_TIMEOUT_SECONDS_CPU

    def test_a_settings_file_beats_the_derived_values(self, tmp_path):
        """An operator who wrote a number down meant it."""
        config_file = tmp_path / "evaluation-config.yaml"
        config_file.write_text("concurrency: 4\ntasks: mmlu\n")

        settings = EvaluationSettings.resolve(
            model=MODEL, service_url=SERVICE_URL, tensor_parallel_size=8, config_file=config_file
        )

        assert settings.concurrency == 4
        assert settings.tasks == ("mmlu",)

    def test_overrides_beat_the_settings_file(self, tmp_path):
        config_file = tmp_path / "evaluation-config.yaml"
        config_file.write_text("limit: 100\n")

        settings = EvaluationSettings.resolve(
            model=MODEL, service_url=SERVICE_URL, config_file=config_file, overrides={"limit": 5}
        )

        assert settings.limit == 5

    def test_backend_args_merge_instead_of_replacing_the_file(self, tmp_path):
        """A caller contributing one backend argument must not drop the others.

        The shipped file's ``apply_chat_template`` decides how prompts are built,
        so losing it to an override that only names ``trust_remote_code`` would
        silently change every score.
        """
        config_file = tmp_path / "evaluation-config.yaml"
        config_file.write_text("backend_args:\n  apply_chat_template: true\n  max_gen_toks: 2048\n")

        settings = EvaluationSettings.resolve(
            model=MODEL,
            service_url=SERVICE_URL,
            config_file=config_file,
            overrides={"backend_args": {"trust_remote_code": True, "max_gen_toks": 4096}},
        )

        assert settings.backend_args == {
            "apply_chat_template": True,
            "trust_remote_code": True,
            "max_gen_toks": 4096,
        }

    def test_overrides_for_other_subsystems_are_ignored(self):
        """A harness --config is one bag for the whole run, not just evaluation."""
        settings = EvaluationSettings.resolve(
            model=MODEL,
            service_url=SERVICE_URL,
            overrides={"limit": 5, "max_warmup_time": 900, "jit_latency_threshold_s": 30},
        )

        assert settings.limit == 5

    def test_model_and_service_url_cannot_be_overridden(self, tmp_path):
        """No configuration file knows which endpoint this run is aimed at."""
        config_file = tmp_path / "evaluation-config.yaml"
        config_file.write_text("model: someone/else\n")

        settings = EvaluationSettings.resolve(
            model=MODEL,
            service_url=SERVICE_URL,
            config_file=config_file,
            overrides={"service_url": "http://elsewhere:9000"},
        )

        assert settings.model == MODEL
        assert settings.service_url == SERVICE_URL


class TestLoadEvaluationConfig:
    """Test suite for reading the settings file."""

    def test_the_shipped_file_is_present_and_valid(self):
        values = load_evaluation_config()

        assert values["backend"] == BACKEND_LM_EVAL
        assert values["backend_args"] == {"apply_chat_template": True}

    def test_the_shipped_file_sits_beside_the_module(self):
        """It has to be package data, or an installed image would not have it."""
        shipped = Path(evaluation_settings.__file__).parent / DEFAULT_CONFIG_FILENAME

        assert shipped.is_file()

    def test_an_unknown_setting_is_reported(self, tmp_path):
        """A typo in a file that exists only for these settings is a mistake."""
        config_file = tmp_path / "evaluation-config.yaml"
        config_file.write_text("tasks: gsm8k\nnum_few_shot: 5\n")

        with pytest.raises(ValueError, match="num_few_shot"):
            load_evaluation_config(config_file)

    def test_a_missing_file_is_reported(self, tmp_path):
        with pytest.raises(ValueError, match="does not exist"):
            load_evaluation_config(tmp_path / "absent.yaml")


class TestAccuracyEvalConfig:
    """Test suite for the CI wire contract, which moved here unchanged.

    ``tests/ci/test_parse_config.py`` exercises it through the re-export; these
    guard the fields and JSON shape the workflow passes between jobs.
    """

    def test_defaults_match_the_ci_contract(self):
        config = AccuracyEvalConfig()

        assert config.lm_eval_model_type == "local-completions"
        assert config.tasks == "gsm8k"
        assert config.apply_chat_template is False

    def test_json_shape_is_unchanged(self):
        payload = json.loads(AccuracyEvalConfig(tasks="mmlu", apply_chat_template=True).to_json())

        assert payload == {"lm_eval_model_type": "local-completions", "tasks": "mmlu", "apply_chat_template": True}

    def test_json_round_trip(self):
        original = AccuracyEvalConfig(lm_eval_model_type="local-chat-completions", tasks="hellaswag")

        assert AccuracyEvalConfig.from_json(original.to_json()) == original

    def test_loads_from_yaml(self, tmp_path):
        config_file = tmp_path / "accuracy-config.yaml"
        config_file.write_text("tasks: winogrande\napply_chat_template: true\n")

        config = AccuracyEvalConfig.from_yaml_file(config_file)

        assert config.tasks == "winogrande"
        assert config.apply_chat_template is True
