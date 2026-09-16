# Copyright © Advanced Micro Devices, Inc., or its affiliates.
#
# SPDX-License-Identifier: MIT

import json
from collections.abc import Callable, Mapping
from copy import deepcopy
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pytest

from aim_common import Engine, Metric, Precision
from aim_runtime import AIMConfig
from aim_runtime.object_model import Profile
from aim_runtime.profile_registry import ProfileRegistry
from aim_runtime.profile_validator import ProfileValidator


# Common fixtures
@pytest.fixture
def project_root() -> Path:
    """Get the project root directory."""
    return Path(__file__).parent.parent


@pytest.fixture
def test_root() -> Path:
    """Get the test directory root."""
    return Path(__file__).parent


# aim_runtime fixtures
@pytest.fixture
def profile_base_path(test_root: Path) -> Path:
    return test_root / "workspace" / "profiles"


@pytest.fixture
def general_profiles_path(test_root: Path) -> str:
    """Get the test profiles directory path."""
    return str(test_root / "workspace" / "profiles" / "general")


@pytest.fixture
def custom_profiles_path(profile_base_path: Path) -> str:
    return str(profile_base_path / "custom")


@pytest.fixture
def aim_config(profile_base_path: Path) -> AIMConfig:
    """Create a test configuration with known valid parameters."""
    return AIMConfig(
        aim_id="meta-llama/Llama-3.1-8B-Instruct",
        profile_base_path=str(profile_base_path),
        precision=Precision.FP16,
        engine=Engine.VLLM,
        metric=Metric.LATENCY,
        accelerator_count="1",
        accelerator_model=None,
    )


@pytest.fixture
def general_aim_config(aim_config: AIMConfig, profile_base_path: Path) -> AIMConfig:
    """Create a test configuration for general profiles."""
    config = deepcopy(aim_config)
    config.aim_id = ""  # Clear aim_id
    config.model_id = "meta-llama/Llama-3.1-8B-Instruct"  # Set model_id instead
    config.profile_base_path = str(profile_base_path)
    return config


@pytest.fixture
def faulty_aim_config_with_no_model(aim_config: AIMConfig) -> AIMConfig:
    """Create a test configuration with no model specified."""
    config = deepcopy(aim_config)
    config.aim_id = ""
    config.model_id = ""
    return config


@pytest.fixture
def profile_validator() -> ProfileValidator:
    """Create a profile validator for testing."""
    return ProfileValidator()


@pytest.fixture
def no_op_profile_validator() -> ProfileValidator:
    """Create a no-op profile validator for testing (skips validation)."""

    class NoOpProfileValidator(ProfileValidator):
        def validate(
            self,
            profile_data: dict[str, Any],
            is_general_profile: bool = False,
            source: str | None = None,
        ) -> None:
            return

    return NoOpProfileValidator()


@pytest.fixture
def model_profile(model_profiles_path: str, profile_validator: ProfileValidator) -> Profile:
    """Create a sample model profile for testing."""
    registry = ProfileRegistry.discover_and_validate(search_paths=[model_profiles_path], validator=profile_validator)
    return registry.find_by_id("test_profile_correct")


@pytest.fixture
def general_profile(general_profiles_path: str, profile_validator: ProfileValidator) -> Profile:
    """Get a valid general test profile."""
    registry = ProfileRegistry.discover_and_validate(search_paths=[general_profiles_path], validator=profile_validator)
    return registry.find_by_id("general/test_profile_correct")


@pytest.fixture
def complex_profile(assets_instinct_path: Path, no_op_profile_validator: ProfileValidator) -> Profile:
    """Get a complex test profile with comprehensive test data."""
    profiles_path = assets_instinct_path / "test" / "model" / "profiles"
    registry = ProfileRegistry.discover_and_validate(
        search_paths=[str(profiles_path)], validator=no_op_profile_validator
    )
    return registry.find_by_id("complex_profile")


# aim_utils fixtures
@pytest.fixture
def model_profiles_path(assets_instinct_path: Path) -> str:
    return str(assets_instinct_path / "meta-llama" / "Llama-3.1-8B-Instruct" / "profiles")


@pytest.fixture
def assets_instinct_path(assets_path: Path) -> Path:
    return assets_path / "instinct"


@pytest.fixture
def assets_radeon_path(assets_path: Path) -> Path:
    return assets_path / "radeon"


@pytest.fixture
def assets_accelerator_path(request, assets_path: Path) -> Path:
    accelerator = getattr(request, "param", "instinct")
    return assets_path / accelerator


@pytest.fixture
def assets_path(test_root: Path) -> Path:
    return test_root / "assets"


# lm-eval fixtures
#
# Accuracy evaluation reads the results document lm-eval writes as
# `results_<timestamp>.json`. The builders below stand in for that document so
# parser tests need neither lm-eval nor a served model.
#
# Field names are those of the pinned `lm-eval[api]==0.4.12`
# (`requirements/evaluation-requirements.txt`), which declares the document as
# the `EvalResults` TypedDict in `lm_eval/result_schema.py`:
#   - `results[<task>]` carries `name`, `alias` and `sample_len` alongside the
#     dynamic `"<metric>,<variant>"` and `"<metric>_stderr,<variant>"` keys built
#     in `lm_eval/evaluator_utils.py`.
#   - `configs`, `versions`, `n-shot`, `higher_is_better` and `n-samples` are
#     keyed by task name; `config` and the model/chat-template fields are merged
#     in afterwards by `EvaluationTracker.save_results_aggregated`.
# lm-eval 0.4.11 keys the per-task entry differently (`samples`, and no `name`),
# so these names must not be re-derived from whatever version happens to be
# importable on a dev machine.

#: Variant name lm-eval gives metrics of a task that declares no `filter_list`.
LM_EVAL_NO_VARIANT = "none"

#: What lm-eval stores instead of a stderr float when it cannot compute one
#: (`--bootstrap_iters 0`, or a single sample).
LM_EVAL_STDERR_UNAVAILABLE = "N/A"

#: gsm8k's two answer extractors, and plausible scores for them. Kept as the
#: default because it is the task CI evaluates today.
GSM8K_VARIANTS = {"strict-match": (0.7892, 0.0112), "flexible-extract": (0.8317, 0.0103)}


@dataclass(frozen=True)
class LmEvalTask:
    """One task's contribution to an lm-eval results document.

    Defaults describe a full gsm8k run: 5-shot, both answer extractors, no
    sample cap. Override `variants` with a single `LM_EVAL_NO_VARIANT` entry for
    a task that declares no filters (mmlu and friends), and `samples_effective`
    for a `--limit` run.
    """

    task: str = "gsm8k"
    metric: str = "exact_match"
    #: Metric variant (lm-eval calls it a filter) -> (value, stderr).
    variants: Mapping[str, tuple[float, float | str]] = field(default_factory=lambda: dict(GSM8K_VARIANTS))
    dataset_path: str = "openai/gsm8k"
    dataset_name: str | None = "main"
    output_type: str = "generate_until"
    num_fewshot: int = 5
    #: Documents in the evaluation split.
    samples_original: int = 1319
    #: Documents actually evaluated; defaults to the whole split.
    samples_effective: int | None = None
    version: float | str | None = 3.0
    higher_is_better: bool = True
    alias: str | None = None

    @property
    def evaluated(self) -> int:
        return self.samples_original if self.samples_effective is None else self.samples_effective

    def metrics_entry(self) -> dict[str, Any]:
        """The task's `results` entry: fixed keys plus one pair per variant."""
        entry: dict[str, Any] = {
            "name": self.task,
            "alias": self.alias or self.task,
            "sample_len": self.evaluated,
        }
        for variant, (value, stderr) in self.variants.items():
            entry[f"{self.metric},{variant}"] = value
            entry[f"{self.metric}_stderr,{variant}"] = stderr
        return entry

    def config_entry(self) -> dict[str, Any]:
        """The task's YAML config, as lm-eval dumps it into `configs`."""
        config: dict[str, Any] = {
            "task": self.task,
            "dataset_path": self.dataset_path,
            "output_type": self.output_type,
            "num_fewshot": self.num_fewshot,
            "metric_list": [{"metric": self.metric, "aggregation": "mean", "higher_is_better": self.higher_is_better}],
            "metadata": {"version": self.version},
        }
        if self.dataset_name is not None:
            config["dataset_name"] = self.dataset_name
        if set(self.variants) != {LM_EVAL_NO_VARIANT}:
            config["filter_list"] = [
                {"name": variant, "filter": [{"function": "regex"}, {"function": "take_first"}]}
                for variant in self.variants
            ]
        return config


def build_lm_eval_document(
    *tasks: LmEvalTask | Mapping[str, Any],
    model_type: str = "local-completions",
    model_name: str = "openai/gpt-oss-20b",
    service_url: str = "http://localhost:8000",
    limit: int | float | None = None,
    apply_chat_template: bool = True,
    lm_eval_version: str = "0.4.12",
    **overrides: Any,
) -> dict[str, Any]:
    """Build an lm-eval results document.

    Args:
        tasks: `LmEvalTask` instances, or mappings of its field names. Defaults
            to a single full gsm8k run.
        model_type: lm-eval's `--model`, i.e. the served endpoint flavour.
        model_name: The model behind that endpoint, as CI passes it in
            `--model_args`.
        service_url: Endpoint base URL, `/v1/completions` appended as CI does.
        limit: The `--limit` the run was invoked with, echoed under `config`.
            Cap the counts a task reports with its `samples_effective`.
        apply_chat_template: Whether the run applied a chat template.
        lm_eval_version: Version to report; the document carries the version
            that produced it, so a parser can read it instead of probing.
        overrides: Merged into the document last, to add or replace any
            top-level key (a missing block, a `groups` section, junk values).

    Returns:
        A JSON-serialisable document of the shape lm-eval writes on disk.
    """
    specs = [task if isinstance(task, LmEvalTask) else LmEvalTask(**task) for task in tasks] or [LmEvalTask()]

    model_args = {
        "model": model_name,
        "base_url": f"{service_url}/v1/completions",
        "num_concurrent": 32,
        "max_retries": 10,
        "timeout": 600,
        "max_gen_toks": 8192,
    }
    chat_template = "{% for message in messages %}{{ message['content'] }}{% endfor %}" if apply_chat_template else None

    document: dict[str, Any] = {
        "results": {spec.task: spec.metrics_entry() for spec in specs},
        "group_subtasks": {spec.task: [] for spec in specs},
        "configs": {spec.task: spec.config_entry() for spec in specs},
        "versions": {spec.task: spec.version for spec in specs},
        "n-shot": {spec.task: spec.num_fewshot for spec in specs},
        "higher_is_better": {spec.task: {spec.metric: spec.higher_is_better} for spec in specs},
        "n-samples": {spec.task: {"original": spec.samples_original, "effective": spec.evaluated} for spec in specs},
        "config": {
            "model": model_type,
            # CI passes --model_args as a JSON string, and lm-eval echoes it verbatim.
            "model_args": json.dumps(model_args),
            "batch_size": "1",
            "batch_sizes": [],
            "device": None,
            "use_cache": None,
            "limit": limit,
            "bootstrap_iters": 100000,
            "gen_kwargs": None,
            "random_seed": 0,
            "numpy_seed": 1234,
            "torch_seed": 1234,
            "fewshot_seed": 1234,
        },
        "git_hash": "9f1c2ab",
        "date": 1765108800.0,
        "pretty_env_info": "PyTorch version: 2.9.0+rocm7.0",
        "transformers_version": "5.12.0",
        "lm_eval_version": lm_eval_version,
        "upper_git_hash": None,
        "tokenizer_pad_token": ["<|endoftext|>", "199999"],
        "tokenizer_eos_token": ["<|return|>", "200002"],
        "tokenizer_bos_token": ["<|startoftext|>", "199998"],
        "eot_token_id": 200002,
        "max_length": 131072,
        "model_source": model_type,
        "model_name": model_name,
        "model_name_sanitized": model_name.replace("/", "__"),
        "system_instruction": None,
        "system_instruction_sha": None,
        "fewshot_as_multiturn": False,
        "chat_template": chat_template,
        "chat_template_sha": "e1b4a7c9" if apply_chat_template else None,
        "task_hashes": {},
        "total_evaluation_time_seconds": "812.4",
    }
    document.update(overrides)
    return document


@pytest.fixture
def lm_eval_document() -> Callable[..., dict[str, Any]]:
    """Factory for lm-eval results documents; see `build_lm_eval_document`."""
    return build_lm_eval_document


@pytest.fixture
def gsm8k_document() -> dict[str, Any]:
    """A full 5-shot gsm8k run reporting both answer extractors."""
    return build_lm_eval_document()
