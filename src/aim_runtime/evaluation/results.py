# Copyright © Advanced Micro Devices, Inc., or its affiliates.
#
# SPDX-License-Identifier: MIT

"""
The results of an evaluation, and the shape they are published in.

This module is the externally visible output contract. Both consumers publish
:meth:`EvaluationResults.to_dict`: the harness carries it in
``HarnessResult.metrics`` (which the CLI writes to ``evaluate_results.json``),
and the CI delegator exports the same mapping as its JSON artifact.

Two levels, mirroring how a score is read:

* :class:`TaskScore` — one task, in the terms any evaluation can express. Which
  data, how much of it, which metric, what value. A backend adapter is the only
  place that maps its own vocabulary onto these names.
* :class:`EvaluationResults` — the run: its settings, its scores, and whether it
  finished. Publishes the headline number flat, so a consumer that wants one
  float needs no knowledge of the envelope.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any

from aim_runtime.evaluation.config import SCHEMA_VERSION
from aim_runtime.evaluation.settings import EvaluationSettings

#: Modality of a text-in/text-out evaluation, which is all we run today. Other
#: modalities fill the same four fields with their own vocabulary — a
#: segmentation run reports ``dataset="Task09_Spleen/labelsTs"``,
#: ``modality="image_3d"``, ``metric="dice"``. A backend declares which one it
#: measures (``EvaluationBackend.modality``) and stamps it onto its records; this
#: default only covers a record built without one.
MODALITY_TEXT = "text"

#: Settings the envelope reports at its top level, so the ``settings`` block does
#: not repeat them, plus the output directory, which is local to the machine that
#: ran the evaluation and says nothing about the score.
SETTINGS_REPORTED_ELSEWHERE = frozenset({"backend", "model", "service_url", "output_dir"})

#: Columns of the flat per-task CSV. It lives here rather than in ``config.py``
#: because it is the row form of :class:`TaskScore`, not an operator knob: the two
#: have to change together. ``extra_metrics`` is deliberately absent — a mapping
#: does not flatten into a column, and the JSON carries it.
CSV_COLUMNS = (
    "model",
    "backend",
    "backend_version",
    "task",
    "dataset",
    "modality",
    "metric",
    "metric_variant",
    "value",
    "stderr",
    "num_samples",
    "num_samples_available",
    "num_fewshot",
    "higher_is_better",
)


@dataclass(frozen=True, kw_only=True)
class TaskScore:
    """One task's score.

    The first five fields have no defaults because a record missing any of them
    cannot be interpreted: a value without its metric, or a metric without the
    data and sample count behind it, is not a result. Everything else is
    additional detail.

    Optional fields are reported as ``null`` rather than dropped, so every record
    has the same keys and consumers need not distinguish "absent" from "unknown".
    """

    #: Task as the backend names it; also what the harness builds check names from.
    task: str
    #: Data evaluated, in the backend's own identifier form.
    dataset: str
    metric: str
    value: float
    #: Documents actually evaluated, which a sample cap reduces.
    num_samples: int

    #: Which answer extractor produced the value, where a task offers several.
    #: Two variants over identical generations give different numbers, so a value
    #: is only comparable against the same variant.
    metric_variant: str | None = None
    stderr: float | None = None
    #: Documents in the full split, so a capped smoke run is distinguishable from
    #: a complete one.
    num_samples_available: int | None = None
    #: In-context examples per request. ``None`` where few-shot does not apply,
    #: never ``0``, which would claim a zero-shot measurement.
    num_fewshot: int | None = None
    higher_is_better: bool | None = None
    modality: str = MODALITY_TEXT
    #: Other metrics the same task reported, kept as detail so the headline value
    #: stays unambiguous.
    extra_metrics: Mapping[str, float] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "task": self.task,
            "dataset": self.dataset,
            "modality": self.modality,
            "metric": self.metric,
            "metric_variant": self.metric_variant,
            "value": self.value,
            "stderr": self.stderr,
            "num_samples": self.num_samples,
            "num_samples_available": self.num_samples_available,
            "num_fewshot": self.num_fewshot,
            "higher_is_better": self.higher_is_better,
            "extra_metrics": dict(self.extra_metrics),
        }


@dataclass(frozen=True)
class EvaluationResults:
    """An evaluation run: what it was asked to do, and what came back."""

    #: The settings that produced these scores, so a number is reproducible.
    settings: EvaluationSettings
    #: Scores in the order the backend reported them.
    tasks: tuple[TaskScore, ...] = ()
    #: Version of the backend that produced the scores.
    backend_version: str | None = None
    elapsed_seconds: float | None = None
    #: Why the run did not complete; ``None`` means it did.
    failure_reason: str | None = None

    #: The backend's own results document, carried so the runner can write it out
    #: beside the envelope. Not published by :meth:`to_dict`: the envelope reports
    #: the interpretation, and a backend document is large, backend-shaped, and
    #: already exported as its own file. Typed as a plain mapping so this module
    #: stays independent of the backends package.
    document: Mapping[str, Any] | None = None

    @property
    def success(self) -> bool:
        """Whether the evaluation ran to completion.

        A completed run that yielded no usable score is still a success: CI
        records a NULL score for it and exits zero, and parity requires the same
        distinction between "the evaluation broke" and "it ran but scored
        nothing".
        """
        return self.failure_reason is None

    @property
    def primary(self) -> TaskScore | None:
        """The score the run is summarised by: the first task reported."""
        return self.tasks[0] if self.tasks else None

    @property
    def accuracy(self) -> float | None:
        """The headline number, or ``None`` when no task produced one."""
        return self.primary.value if self.primary else None

    def to_dict(self) -> dict[str, Any]:
        """The published mapping: the headline number flat, the detail nested."""
        primary = self.primary
        return {
            "accuracy": self.accuracy,
            "primary_task": primary.task if primary else None,
            "evaluation": {
                "schema_version": SCHEMA_VERSION,
                "backend": self.settings.backend,
                "backend_version": self.backend_version,
                "model": self.settings.model,
                "service_url": self.settings.service_url,
                "elapsed_seconds": self.elapsed_seconds,
                "failure_reason": self.failure_reason,
                "settings": self.settings_summary(),
                "tasks": [task.to_dict() for task in self.tasks],
            },
        }

    def settings_summary(self) -> dict[str, Any]:
        """The settings as the envelope reports them, minus what it states elsewhere."""
        return self.settings.model_dump(mode="json", exclude=set(SETTINGS_REPORTED_ELSEWHERE))

    def to_csv_rows(self) -> list[dict[str, Any]]:
        """One row per task, each carrying enough run context to stand alone."""
        run = {
            "model": self.settings.model,
            "backend": self.settings.backend,
            "backend_version": self.backend_version,
        }
        return [{column: (run | task.to_dict()).get(column) for column in CSV_COLUMNS} for task in self.tasks]
