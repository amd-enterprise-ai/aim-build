# Copyright © Advanced Micro Devices, Inc., or its affiliates.
#
# SPDX-License-Identifier: MIT

"""
The seam between "run an evaluation" and "run *this* evaluation tool".

Three methods, in the order the runner calls them:

* :meth:`EvaluationBackend.probe` — can this backend run here at all? Answered
  separately from running so an absent backend becomes one clear failed check
  rather than a subprocess error mid-evaluation.
* :meth:`EvaluationBackend.run` — perform the evaluation and hand back a
  :class:`RawRun`.
* :meth:`EvaluationBackend.parse` — turn that into
  :class:`~aim_runtime.evaluation.results.TaskScore` records.

:class:`RawRun` is deliberately a handle rather than a process result: a backend
may be a subprocess, an in-process call or an HTTP service, and only the first of
those has an exit code. Subprocess machinery is a detail of the adapters that
need it, not of this interface.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, ClassVar

from aim_runtime.evaluation.results import MODALITY_TEXT, TaskScore
from aim_runtime.evaluation.settings import EvaluationSettings


@dataclass(frozen=True)
class BackendInfo:
    """Whether a backend can run, and what it is."""

    name: str
    available: bool
    #: How the backend is invoked, for a failure message that can be acted on.
    command: str | None = None
    #: Version, where the probe can learn it cheaply. A backend that reports its
    #: version in its own results leaves this ``None`` and fills
    #: :attr:`RawRun.version` instead.
    version: str | None = None
    #: Why it cannot run, or what the probe saw.
    detail: str | None = None


@dataclass(frozen=True)
class RawRun:
    """What a backend produced, before anything interprets it.

    ``failure_reason is None`` with ``document is None`` is a legitimate outcome:
    the evaluation ran but left nothing readable behind. The two are separate
    facts for the same reason they are separate on
    :class:`~aim_runtime.evaluation.results.EvaluationResults` — CI exits zero and
    records a NULL score for that case.
    """

    #: The backend's own results document, loaded but not interpreted.
    document: Mapping[str, Any] | None = None
    #: Files worth keeping: the backend's raw output, for the harness to publish.
    artifacts: tuple[Path, ...] = ()
    #: Backend version, when the document reports it.
    version: str | None = None
    failure_reason: str | None = None
    #: Backend-specific detail an adapter wants to carry to its own parser.
    extra: Mapping[str, Any] = field(default_factory=dict)

    @property
    def succeeded(self) -> bool:
        return self.failure_reason is None


class EvaluationBackend(ABC):
    """A tool that can score a served model."""

    #: Backend slug, matching ``EvaluationSettings.backend`` and reported in the
    #: results envelope.
    name: ClassVar[str]

    #: What this backend measures. Text is the default because it is what AIM
    #: evaluates today, but it is declared here so a backend for another modality
    #: states its own and stamps it onto the records it parses — what a metric
    #: means is the backend's knowledge, not something the shared layer infers.
    modality: ClassVar[str] = MODALITY_TEXT

    @abstractmethod
    def probe(self) -> BackendInfo:
        """Report whether this backend can run, without evaluating anything."""

    @abstractmethod
    def run(self, settings: EvaluationSettings, *, env: Mapping[str, str], output_dir: Path) -> RawRun:
        """Evaluate ``settings``, writing backend output under ``output_dir``.

        ``env`` is the complete environment the backend runs with, passed
        explicitly rather than inherited: the evaluation dependencies may live in
        their own virtualenv, and dataset access needs cache and token variables
        that the caller — not the backend — decides on.
        """

    @abstractmethod
    def parse(self, raw: RawRun) -> list[TaskScore]:
        """Map a run's output onto task scores. Pure: no I/O, no subprocess.

        The backend owns its metric vocabulary: which of the numbers it reports is
        the headline score, which direction is better, and what :attr:`modality`
        the score belongs to.
        """
