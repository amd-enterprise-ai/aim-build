# Copyright © Advanced Micro Devices, Inc., or its affiliates.
#
# SPDX-License-Identifier: MIT

"""
What an evaluation run is asked to do, as validated models.

* :class:`EvaluationSettings` — what a run needs, expressed without reference to
  whichever tool performs it, plus an opaque ``backend_args`` for the vocabulary
  only that tool understands. It is the single construction point both consumers
  build, through :meth:`EvaluationSettings.resolve`.
* :class:`AccuracyEvalConfig` — the CI wire contract, moved here from
  ``ci/accuracy_evaluation/parse_config.py`` (which now re-exports it) so the
  shared package can read it without importing from ``ci``.

The values these models default to live in :mod:`aim_runtime.evaluation.config`,
which holds every tunable of the evaluation path and nothing behavioural. The two
functions here that read those values — :func:`concurrency_for` and
:func:`request_timeout_for` — are the endpoint-shape rules ``resolve`` applies,
which is why they sit beside it rather than beside the constants they pick from.
"""

from __future__ import annotations

import logging
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

from pydantic import BaseModel, ConfigDict, Field, field_validator

from aim_runtime.evaluation.config import (
    BACKEND_LM_EVAL,
    CONCURRENCY_HIGH_TP,
    CONCURRENCY_LOW_TP,
    CONCURRENCY_TP_THRESHOLD,
    DEFAULT_CONFIG_FILENAME,
    DEFAULT_TASKS,
    LM_EVAL_MODEL_TYPE,
    MAX_RETRIES,
    NUM_FEWSHOT,
    REQUEST_TIMEOUT_SECONDS_ACCELERATED,
    REQUEST_TIMEOUT_SECONDS_CPU,
    SUPPORTED_BACKENDS,
)
from aim_runtime.utils import read_yaml

logger = logging.getLogger(__name__)


def concurrency_for(tensor_parallel_size: int) -> int:
    """Concurrent requests to aim at an endpoint of this width.

    ``tensor_parallel_size`` is the nominal tensor-parallel degree, not a raw
    device or core count: a CPU profile records its recommended core count in
    ``accelerator_count`` (188, say) and passing that here would put a single CPU
    endpoint in the high tier.
    """
    return CONCURRENCY_LOW_TP if tensor_parallel_size <= CONCURRENCY_TP_THRESHOLD else CONCURRENCY_HIGH_TP


def request_timeout_for(cpu_endpoint: bool) -> int:
    """Per-request timeout, scoped by endpoint class rather than applied globally."""
    return REQUEST_TIMEOUT_SECONDS_CPU if cpu_endpoint else REQUEST_TIMEOUT_SECONDS_ACCELERATED


class EvaluationSettings(BaseModel):
    """What to evaluate, where, and how hard to push it.

    Deliberately backend-agnostic: every field here is something any accuracy
    evaluation needs. Anything only one tool understands goes in ``backend_args``,
    which this model never interprets.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    #: Served model, as the endpoint names it.
    model: str
    #: Base URL of the OpenAI-compatible endpoint, without a trailing slash.
    service_url: str

    #: Tasks to score, one name each. A comma-separated string is accepted and
    #: split, which is the spelling :class:`AccuracyEvalConfig` carries because CI
    #: passes it through the workflow as one value.
    tasks: tuple[str, ...] = DEFAULT_TASKS
    #: Documents per task; ``None`` evaluates the whole split, which is what CI
    #: records and therefore what a comparable score requires.
    limit: int | None = Field(default=None, gt=0)
    num_fewshot: int | None = Field(default=NUM_FEWSHOT, ge=0)

    concurrency: int = Field(default=CONCURRENCY_LOW_TP, gt=0)
    request_timeout_seconds: int = Field(default=REQUEST_TIMEOUT_SECONDS_ACCELERATED, gt=0)
    max_retries: int = Field(default=MAX_RETRIES, ge=0)
    #: Budget for the whole run. ``None`` is unbounded, which is how CI runs
    #: today; a value here would kill a long-but-healthy evaluation.
    wall_clock_timeout_seconds: int | None = Field(default=None, gt=0)

    #: Where results and side artifacts are written.
    output_dir: Path = Path(".")

    backend: str = BACKEND_LM_EVAL
    #: Backend-private arguments, passed through untouched.
    backend_args: Mapping[str, Any] = Field(default_factory=dict)

    @field_validator("tasks", mode="before")
    @classmethod
    def _normalize_tasks(cls, value: Any) -> Any:
        """Accept a comma-separated string as well as a sequence of names.

        Both spellings reach us: CI carries one string through the workflow,
        while a YAML config is naturally a list.
        """
        names = value.split(",") if isinstance(value, str) else value
        if not isinstance(names, Sequence):
            return value
        stripped = tuple(str(name).strip() for name in names)
        return tuple(name for name in stripped if name)

    @field_validator("tasks", mode="after")
    @classmethod
    def _require_a_task(cls, value: tuple[str, ...]) -> tuple[str, ...]:
        if not value:
            raise ValueError("at least one task must be configured")
        return value

    @field_validator("service_url", mode="after")
    @classmethod
    def _absolute_url(cls, value: str) -> str:
        """Require an absolute URL and drop trailing slashes.

        Endpoint paths are appended to this, so a trailing slash would produce
        ``http://host:8000//v1/completions``.
        """
        parsed = urlparse(value)
        if not (parsed.scheme and parsed.netloc):
            raise ValueError(f"service_url must be absolute, e.g. http://localhost:8000 (got {value!r})")
        return value.rstrip("/")

    @field_validator("backend", mode="after")
    @classmethod
    def _known_backend(cls, value: str) -> str:
        if value not in SUPPORTED_BACKENDS:
            raise ValueError(f"unknown backend {value!r}; supported: {', '.join(sorted(SUPPORTED_BACKENDS))}")
        return value

    @property
    def tasks_argument(self) -> str:
        """The task list as a backend command line takes it."""
        return ",".join(self.tasks)

    @classmethod
    def resolve(
        cls,
        *,
        model: str,
        service_url: str,
        tensor_parallel_size: int = 1,
        cpu_endpoint: bool = False,
        config_file: str | Path | None = None,
        overrides: Mapping[str, Any] | None = None,
    ) -> "EvaluationSettings":
        """Build settings from every source, lowest precedence first.

        1. Field defaults, i.e. the constants in ``config.py``.
        2. Concurrency and per-request timeout derived from the endpoint's shape.
        3. The settings file — shipped defaults, or ``config_file``.
        4. ``overrides``: explicit per-run values, such as a harness ``--config``.

        ``model`` and ``service_url`` are arguments rather than overrides because
        no configuration file can know them.

        ``backend_args`` is the one field merged key by key rather than replaced,
        since it is a bag of independent arguments: a caller contributing one of
        them (a profile's ``trust_remote_code``, say) must not drop the shipped
        file's others, such as ``apply_chat_template``, whose absence would change
        every score.

        Keys in ``overrides`` that do not name a field are ignored: a harness
        ``--config`` is one bag shared by the whole run, so it legitimately
        carries values for other subsystems. The settings file is checked
        strictly instead, since it exists only for these settings.
        """
        derived: dict[str, Any] = {
            "concurrency": concurrency_for(tensor_parallel_size),
            "request_timeout_seconds": request_timeout_for(cpu_endpoint),
        }
        from_file = load_evaluation_config(config_file)
        selected, ignored = _split_known_fields(overrides or {})
        if ignored:
            logger.debug("Ignoring non-evaluation override(s): %s", ", ".join(ignored))

        resolved = {**derived, **from_file, **selected, "model": model, "service_url": service_url}
        backend_args = {**(from_file.get("backend_args") or {}), **(selected.get("backend_args") or {})}
        if backend_args:
            resolved["backend_args"] = backend_args

        return cls(**resolved)


def load_evaluation_config(config_file: str | Path | None = None) -> dict[str, Any]:
    """Read evaluation settings from YAML, defaulting to the shipped file.

    Raises:
        ValueError: The file is missing, is not a mapping, or names a setting
            that does not exist — a typo in a settings file is a mistake worth
            reporting rather than ignoring.
    """
    path = Path(config_file) if config_file else Path(__file__).parent / DEFAULT_CONFIG_FILENAME
    values = read_yaml(path)
    _, unknown = _split_known_fields(values)
    if unknown:
        raise ValueError(f"Unknown evaluation setting(s) in {path}: {', '.join(unknown)}")
    logger.debug("Loaded evaluation settings from %s", path)
    return values


def _split_known_fields(values: Mapping[str, Any]) -> tuple[dict[str, Any], list[str]]:
    """Split a mapping into settings fields and everything else."""
    known = {key: value for key, value in values.items() if key in EvaluationSettings.model_fields}
    unknown = sorted(set(values) - set(known))
    return known, unknown


class AccuracyEvalConfig(BaseModel):
    """Configuration for accuracy evaluation using the lm-eval framework.

    The CI wire contract: ``ci/accuracy_evaluation/parse_config.py`` reads it from
    ``accuracy-config.yaml`` and the workflow passes it between jobs as JSON, so
    the field names and JSON shape are fixed. It lives here, and is re-exported
    there, because the shared package must not import from ``ci``.

    Note ``apply_chat_template`` defaults to ``False`` — a base model must not be
    evaluated with a chat template — while the shipped ``evaluation-config.yaml``
    turns it on, matching what CI configures for the instruction-tuned models it
    evaluates.
    """

    lm_eval_model_type: str = Field(default=LM_EVAL_MODEL_TYPE, description="LM Eval model type")
    tasks: str = Field(
        default=",".join(DEFAULT_TASKS),
        description=(
            "Comma-separated evaluation tasks (e.g. 'gsm8k', 'mmlu,gsm8k'). A string rather than a list "
            "because the workflow passes this config between jobs as one value; EvaluationSettings.tasks "
            "is the same list as a tuple, and accepts this spelling directly."
        ),
    )
    apply_chat_template: bool = Field(default=False, description="Whether to apply chat template during evaluation")

    model_config = {"frozen": True}

    @classmethod
    def from_yaml_file(cls, config_path: Path) -> "AccuracyEvalConfig":
        """Load AccuracyEvalConfig from a YAML config file."""
        config_dict = read_yaml(config_path)
        return cls(**config_dict)

    @classmethod
    def from_json(cls, json_str: str) -> "AccuracyEvalConfig":
        """Deserialize from a JSON string."""
        return cls.model_validate_json(json_str)

    def to_json(self) -> str:
        """Serialize to a JSON string."""
        return self.model_dump_json()
