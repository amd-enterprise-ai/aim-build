# Copyright © Advanced Micro Devices, Inc., or its affiliates.
#
# SPDX-License-Identifier: MIT

"""
ModelHarness — abstract base class for model validation, benchmarking, and evaluation.

Each AIM image ships a concrete harness implementation. Standard vLLM/VLM models
use VLLMHarness (shipped with aim-runtime). Specialized models (BentoML, custom
engines) subclass ModelHarness in ``image/src/harness.py`` within their asset
directory.

The entrypoint CLI discovers the active harness via
:func:`aim_runtime.harness.discovery.discover_harness` and dispatches
``validate``, ``benchmark``, and ``evaluate`` commands through it.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any

# ---------------------------------------------------------------------------
# Result types
# ---------------------------------------------------------------------------


class CheckResultType(StrEnum):
    """How CI should interpret a check's result value."""

    PASS_FAIL = "pass_fail"
    SCORE = "score"
    COMPARISON = "comparison"


# ---------------------------------------------------------------------------
# Check metadata & results
# ---------------------------------------------------------------------------


class CheckScope(StrEnum):
    """When a check is eligible to run.

    RUNTIME — needs no service. Reads the profile only (schema, engine args), so
              it is safe anywhere and takes milliseconds.
    OFFLINE — needs a live service to talk to (health, endpoints, capabilities,
              accuracy, throughput). May take minutes.

    ``HarnessConfig.check_scopes`` is a *set* of scopes.  A check runs when
    its scope is in the set — e.g. ``{RUNTIME}`` for profile-only validation,
    ``{RUNTIME, OFFLINE}`` for the full suite. The CLI also reads this to decide
    whether it has to start a server: ``OFFLINE`` in the set means yes.
    """

    RUNTIME = "runtime"
    OFFLINE = "offline"

    @classmethod
    def all(cls) -> set["CheckScope"]:
        return set(cls)


@dataclass(frozen=True)
class CheckInfo:
    """Metadata about a single available check — returned by :meth:`ModelHarness.list_checks`.

    .. todo:: Scope filtering in the framework

       Currently each harness manually checks ``config.check_scopes`` to decide
       which checks to run.  The plan is to move scope filtering into the
       framework so that:

       1. The entrypoint (or a base-class helper) filters the check list
          *before* calling the harness method, passing only the applicable
          checks.
       2. ``validate`` receives a pre-filtered list and runs everything in it —
          no manual scope conditionals.
       3. ``benchmark`` and ``evaluate`` are unaffected (they always run all
          their checks regardless of scope).
    """

    name: str
    result_type: CheckResultType
    scope: CheckScope
    description: str


@dataclass
class CheckResult:
    """Outcome of a single check within a validation / benchmark run.

    ``warnings`` carries observations that are worth surfacing but must not fail
    the run — e.g. a model that declines to call a tool it was offered. ``skipped``
    marks a check that never ran (an undeclared capability, a missing validator);
    skipped checks are excluded from the pass/fail verdict.
    """

    name: str
    result_type: CheckResultType
    success: bool
    value: Any  # bool for PASS_FAIL, float for SCORE/COMPARISON, None when skipped
    detail: str = ""
    warnings: list[str] = field(default_factory=list)
    skipped: bool = False
    gating: bool = True


# Constructors for the three PASS_FAIL outcomes every harness reports. Plain
# functions rather than CheckResult classmethods because a ``skipped``
# classmethod would shadow the field of the same name.


def passed(
    name: str,
    detail: str = "",
    warnings: list[str] | None = None,
    *,
    gating: bool = True,
) -> CheckResult:
    return CheckResult(name, CheckResultType.PASS_FAIL, True, True, detail, warnings=warnings or [], gating=gating)


def failed(name: str, detail: str, *, gating: bool = True) -> CheckResult:
    return CheckResult(name, CheckResultType.PASS_FAIL, False, False, detail, gating=gating)


def skipped(name: str, detail: str, *, gating: bool = True) -> CheckResult:
    """A check that never ran; excluded from the pass/fail verdict."""
    return CheckResult(name, CheckResultType.PASS_FAIL, True, None, detail, skipped=True, gating=gating)


# ---------------------------------------------------------------------------
# Harness config & aggregate result
# ---------------------------------------------------------------------------

#: ``HarnessConfig.extra`` key the CLI uses to hand a harness the readiness time
#: it already measured while starting the service, so ``validate()`` doesn't
#: re-probe an already-live service and report a near-zero ``ready_time_seconds``.
#: Shared between ``entrypoint.py`` (writer) and each harness (reader) so the
#: key can't drift between the two call sites.
STARTUP_READY_TIME_SECONDS_KEY = "startup_ready_time_seconds"


@dataclass
class HarnessConfig:
    """Configuration passed to harness methods from the entrypoint CLI.

    .. todo:: Add ``run_check(name)`` dispatch so individual checks can be run
       in isolation for CI debugging (e.g. ``aim-runtime validate --check tool_calling``).
    """

    profile: dict[str, Any]
    service_url: str | None = None
    timeout_seconds: int = 300
    output_format: str = "json"  # "json" | "table" | "ci"
    check_scopes: set[CheckScope] = field(default_factory=CheckScope.all)
    extra: dict[str, Any] = field(default_factory=dict)

    def resolve_service_url(self) -> str:
        """Return the service URL, falling back to localhost:{port}.

        Port is resolved via :meth:`get` so ``--config`` overrides are
        honored (extra → profile → 8000).
        """
        if self.service_url:
            return self.service_url
        port = self.get("port", 8000)
        return f"http://localhost:{port}"

    def get(self, key: str, default: Any = None) -> Any:
        """Look up a value in extra first, then profile, then default.

        This lets harnesses read per-run overrides (from ``--config``) that
        shadow static profile fields with a single call::

            num_requests = config.get("num_requests", 100)
        """
        if key in self.extra:
            return self.extra[key]
        if key in self.profile:
            return self.profile[key]
        return default

    def has_capability(self, name: str) -> bool:
        """Return whether the profile declares support for a capability.

        Reads ``metadata.capabilities.<name>`` (all flags default to ``False``),
        so checks for a capability the profile never claimed can be skipped
        rather than failed. A ``--config`` entry of the same name wins, which is
        how CI overrides the profile flag (e.g. ``REASONING_ENABLED``).
        """
        if name in self.extra:
            return bool(self.extra[name])
        metadata = self.profile.get("metadata") or {}
        capabilities = metadata.get("capabilities") or {}
        return bool(capabilities.get(name, False))


@dataclass
class HarnessResult:
    """Structured result from any harness operation."""

    success: bool
    summary: str
    checks: list[CheckResult] = field(default_factory=list)
    metrics: dict[str, Any] = field(default_factory=dict)
    artifacts: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "success": self.success,
            "summary": self.summary,
            "checks": [
                {
                    "name": c.name,
                    "result_type": c.result_type.value,
                    "success": c.success,
                    "value": c.value,
                    "detail": c.detail,
                    "warnings": list(c.warnings),
                    "skipped": c.skipped,
                    "gating": c.gating,
                }
                for c in self.checks
            ],
            "metrics": self.metrics,
            "artifacts": self.artifacts,
        }


# ---------------------------------------------------------------------------
# Abstract base class
# ---------------------------------------------------------------------------


class ModelHarness(ABC):
    """Abstract base class for model validation, benchmarking, and evaluation.

    Harness selection is per-engine: the entrypoint resolves the active profile,
    determines its engine, and dispatches to the corresponding harness.

    Standard vLLM/VLM engines use :class:`VLLMHarness` (shipped with aim-runtime).
    Specialized engines subclass this in ``image/src/harness.py``.
    """

    @abstractmethod
    def validate(self, config: HarnessConfig) -> HarnessResult:
        """Validate that the model service is healthy and producing correct output."""
        ...

    @abstractmethod
    def benchmark(self, config: HarnessConfig) -> HarnessResult:
        """Run performance benchmarks and report throughput / latency metrics."""
        ...

    @abstractmethod
    def evaluate(self, config: HarnessConfig) -> HarnessResult:
        """Run accuracy / quality evaluation."""
        ...

    @abstractmethod
    def list_checks(self) -> list[CheckInfo]:
        """Return metadata about all checks this harness can run."""
        ...

    def health_check(self, service_url: str, timeout_seconds: int = 60) -> bool:
        """Check if the model server is ready.

        Default implementation polls ``/v1/models`` until it serves a model, the
        same readiness signal the OpenAI-compatible checks use. Specialized
        harnesses should override for non-OpenAI endpoints.
        """
        from aim_runtime.harness.service_checks import probe_api_health

        return probe_api_health(service_url, timeout_seconds=timeout_seconds).check.success
