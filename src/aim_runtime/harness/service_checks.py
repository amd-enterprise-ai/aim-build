# Copyright © Advanced Micro Devices, Inc., or its affiliates.
#
# SPDX-License-Identifier: MIT

"""
Checks against a live OpenAI-compatible model service.

These mirror the validations the ``/validate`` workflow runs
(``ci/model_service_validation/``), reimplemented here so a harness can run them
from inside any AIM image without importing the ``ci`` package.

Any harness that speaks the OpenAI API can reuse these; ``VLLMHarness``
orchestrates them.

Severity follows CI: behavioural quirks are warnings that still pass (a model
declining to call an offered tool, or calling one when it shouldn't), while
transport errors, malformed responses and schema violations fail.

Two deliberate deviations from CI, which reports a flag per validation family
and lets the workflow decide what gates a build:

* A harness returns a single verdict, so a failed warmup fails the run here
  instead of only warning.
* ``/v1/models`` is the authoritative readiness signal rather than ``/health``,
  since every OpenAI-compatible engine serves it; a missing ``/health`` route is
  reported as a warning.
"""

from __future__ import annotations

import json
import logging
import time
from collections.abc import Iterator
from dataclasses import dataclass, field
from typing import Any, Literal

import requests
from pydantic import BaseModel, ValidationError

from aim_runtime.harness import CheckResult, failed, passed

logger = logging.getLogger(__name__)

#: Ceiling (seconds) for the first-inference warmup loop.
MAX_WARMUP_TIME_DEFAULT = 300

#: Per-request timeout (seconds) for the behavioural checks below. Only applies
#: when a caller doesn't pass one; ``VLLMHarness`` always forwards the run's
#: ``--timeout`` so a single flag governs readiness, warmup, and these requests.
SERVICE_CHECK_TIMEOUT_DEFAULT = 120.0

#: First-request latency (seconds) above which AITER kernel JIT compilation is
#: suspected. Purely informational — it never gates the result.
JIT_LATENCY_THRESHOLD_S_DEFAULT = 30.0

#: A single slow attempt is not enough signal; JIT typically forces a retry.
JIT_MIN_WARMUP_ATTEMPTS = 2

LATENCY_ROUND_DECIMALS = 1
RETRY_INTERVAL_SECONDS = 5.0
DEFAULT_REASONING_MAX_TOKENS = 1024

#: Chat-template syntax that must never leak into assistant content.
TOOL_LEAK_TOKENS = ("[TOOL_CALLS]", "<tool_call>", "</tool_call>", "functools[", "<｜tool▁call▁begin｜>")

#: Fields every OpenAI-compatible completion response carries.
_OPENAI_COMPLETION_FIELDS = ("id", "object", "created", "model", "choices")

_MAX_DETAIL_CHARS = 300


# --------------------------------------------------------------------------- #
# HTTP plumbing
# --------------------------------------------------------------------------- #


class ServiceError(Exception):
    """A request to the model service failed.

    Carries the server's response body when there was one — a 400 from vLLM
    explains *why* it rejected the request (e.g. tool calling not enabled), and
    losing that turns every failure into an unactionable "HTTP Error 400".
    """

    def __init__(self, message: str, status: int | None = None, body: str = "") -> None:
        super().__init__(message)
        self.status = status
        self.body = body


def _truncate(text: str, limit: int = _MAX_DETAIL_CHARS) -> str:
    text = " ".join(text.split())
    return text if len(text) <= limit else f"{text[:limit]}…"


def _send(url: str, payload: dict[str, Any] | None, timeout: float) -> requests.Response:
    """GET (no payload) or POST JSON, raising ServiceError for anything non-2xx."""
    try:
        if payload is None:
            response = requests.get(url, timeout=timeout)
        else:
            response = requests.post(url, json=payload, timeout=timeout)
    except requests.RequestException as exc:
        raise ServiceError(f"{url} unreachable: {exc}") from exc

    if not response.ok:
        raise ServiceError(
            f"HTTP {response.status_code} from {url}: {_truncate(response.text)}",
            status=response.status_code,
            body=response.text,
        )
    return response


def request_json(url: str, payload: dict[str, Any] | None = None, timeout: float = 60.0) -> dict[str, Any]:
    """GET (no payload) or POST JSON, returning the decoded response body."""
    response = _send(url, payload, timeout)

    try:
        decoded = response.json()
    except ValueError as exc:
        raise ServiceError(
            f"{url} returned invalid JSON: {exc}", status=response.status_code, body=response.text
        ) from exc

    if not isinstance(decoded, dict):
        raise ServiceError(
            f"{url} returned {type(decoded).__name__}, expected a JSON object",
            status=response.status_code,
            body=response.text,
        )
    return decoded


def request_ok(url: str, timeout: float = 5.0) -> None:
    """GET a URL whose body is irrelevant (``/health`` returns an empty body)."""
    _send(url, None, timeout)


def _chat(
    service_url: str,
    model_id: str,
    messages: list[dict[str, Any]],
    timeout: float,
    **options: Any,
) -> dict[str, Any]:
    """POST a chat completion, returning the decoded body.

    Most checks differ only in their messages and a couple of request options,
    so they share this instead of restating the endpoint and model each time.
    """
    return request_json(
        f"{service_url}/v1/chat/completions",
        {"model": model_id, "messages": messages, **options},
        timeout=timeout,
    )


@dataclass
class _RetryClock:
    """Attempt counter and deadline shared by the polling phases.

    Iterating yields the attempt number until the timeout expires, sleeping
    ``retry_interval`` between attempts without ever overshooting the deadline.
    """

    timeout_seconds: float
    retry_interval: float
    attempts: int = 0
    _start: float = field(default_factory=time.monotonic)

    @property
    def elapsed(self) -> float:
        return time.monotonic() - self._start

    @property
    def elapsed_rounded(self) -> float:
        return round(self.elapsed, LATENCY_ROUND_DECIMALS)

    def __iter__(self) -> Iterator[int]:
        while self.elapsed < self.timeout_seconds:
            self.attempts += 1
            yield self.attempts
            remaining = self.timeout_seconds - self.elapsed
            if remaining <= 0:
                break
            time.sleep(min(self.retry_interval, remaining))


def _message(body: dict[str, Any]) -> dict[str, Any]:
    """Return the first choice's message object, or an empty dict."""
    choices = body.get("choices") or [{}]
    return choices[0].get("message") or {}


def _text(value: Any) -> str:
    """Return stripped string content, or "" for anything non-string."""
    return value.strip() if isinstance(value, str) else ""


# --------------------------------------------------------------------------- #
# Readiness and warmup
# --------------------------------------------------------------------------- #


@dataclass
class HealthProbe:
    """Outcome of waiting for the service to serve a model."""

    check: CheckResult
    model_id: str | None = None
    ready_time_seconds: float | None = None


@dataclass
class WarmupOutcome:
    """Timing data from the first-inference warmup phase.

    ``elapsed_seconds`` is wall-clock from the first POST attempt until either a
    successful response or the timeout.
    """

    check: CheckResult
    elapsed_seconds: float
    attempts: int
    jit_suspected: bool

    @property
    def succeeded(self) -> bool:
        return self.check.success


def probe_api_health(
    service_url: str,
    timeout_seconds: float = 300.0,
    retry_interval: float = RETRY_INTERVAL_SECONDS,
) -> HealthProbe:
    """Poll ``/v1/models`` until the service reports a served model.

    The served model id is what subsequent requests must use — it is not always
    the profile's ``model_id`` (quantized variants and local paths differ).
    """
    logger.info("Waiting for a served model at %s ...", service_url)
    clock = _RetryClock(timeout_seconds, retry_interval)
    last_error = "no attempt completed"

    for attempt in clock:
        try:
            body = request_json(f"{service_url}/v1/models", timeout=10)
            models = body.get("data") or []
            model_id = models[0].get("id") if models else None
            if model_id:
                ready = clock.elapsed_rounded
                logger.info("Model '%s' ready after %ss", model_id, ready)
                return HealthProbe(
                    check=passed(
                        "api_health",
                        f"Model '{model_id}' ready after {ready}s ({attempt} attempt(s))",
                        warnings=_health_route_warnings(service_url),
                    ),
                    model_id=model_id,
                    ready_time_seconds=ready,
                )
            last_error = "/v1/models served no models"
        except ServiceError as exc:
            last_error = str(exc)
        logger.debug("Service not ready after %.0fs: %s", clock.elapsed, last_error)

    detail = f"No model served within {clock.elapsed_rounded}s ({clock.attempts} attempt(s)): {last_error}"
    return HealthProbe(check=failed("api_health", detail))


def _health_route_warnings(service_url: str) -> list[str]:
    """Report a missing ``/health`` route without failing an otherwise ready service."""
    try:
        request_ok(f"{service_url}/health")
    except ServiceError as exc:
        # Only the status matters here; error pages are long and say nothing useful.
        reason = f"HTTP {exc.status}" if exc.status else str(exc)
        return [f"Serving models but /health is not available ({reason})"]
    return []


def run_warmup(
    service_url: str,
    model_id: str,
    max_warmup_time: float = MAX_WARMUP_TIME_DEFAULT,
    jit_threshold_seconds: float = JIT_LATENCY_THRESHOLD_S_DEFAULT,
    retry_interval: float = RETRY_INTERVAL_SECONDS,
) -> WarmupOutcome:
    """Send minimal inference requests until one succeeds.

    A served model is not necessarily an inferring model: the first request
    triggers kernel JIT compilation, which can take minutes. Absorbing that here
    keeps the timings out of the behavioural checks that follow.
    """
    logger.info("Warming up '%s' (max %ss) ...", model_id, max_warmup_time)
    payload = {"model": model_id, "prompt": "Hi", "max_tokens": 1, "temperature": 0.0}
    clock = _RetryClock(max_warmup_time, retry_interval)
    last_error = "no attempt completed"

    for attempt in clock:
        try:
            request_json(
                f"{service_url}/v1/completions",
                payload,
                timeout=max(1.0, min(60.0, max_warmup_time - clock.elapsed)),
            )
            return _warmup_outcome(clock, jit_threshold_seconds, last_error=None)
        except ServiceError as exc:
            last_error = str(exc)
            logger.debug("Warmup attempt %d failed: %s", attempt, last_error)

    return _warmup_outcome(clock, jit_threshold_seconds, last_error=last_error)


def _warmup_outcome(clock: _RetryClock, jit_threshold_seconds: float, last_error: str | None) -> WarmupOutcome:
    """Build the outcome; ``last_error`` is None when warmup succeeded."""
    elapsed = clock.elapsed_rounded
    attempts = clock.attempts
    jit_suspected = attempts >= JIT_MIN_WARMUP_ATTEMPTS and clock.elapsed > jit_threshold_seconds

    if last_error is not None:
        detail = f"No successful inference within {elapsed}s ({attempts} attempt(s)): {last_error}"
        return WarmupOutcome(failed("warmup", detail, gating=False), elapsed, attempts, jit_suspected)

    warnings = []
    if jit_suspected:
        warnings.append(
            f"AITER JIT compilation suspected: first-request latency {elapsed}s over "
            f"{attempts} attempt(s) (threshold {jit_threshold_seconds}s)"
        )
        logger.warning(warnings[-1])

    detail = f"First inference completed in {elapsed}s ({attempts} attempt(s))"
    logger.info(detail)
    return WarmupOutcome(passed("warmup", detail, warnings, gating=False), elapsed, attempts, jit_suspected)


# --------------------------------------------------------------------------- #
# OpenAI API compatibility
# --------------------------------------------------------------------------- #


def check_completions_endpoint(
    service_url: str, model_id: str, timeout: float = SERVICE_CHECK_TIMEOUT_DEFAULT
) -> CheckResult:
    """Verify ``/v1/completions`` returns an OpenAI-shaped response."""
    try:
        body = request_json(
            f"{service_url}/v1/completions",
            {"model": model_id, "prompt": "Hello", "max_tokens": 10, "temperature": 0.7},
            timeout=timeout,
        )
    except ServiceError as exc:
        return failed("completions_endpoint", str(exc))

    if missing := [f for f in _OPENAI_COMPLETION_FIELDS if f not in body]:
        return failed("completions_endpoint", f"Response missing OpenAI field(s): {', '.join(missing)}")

    choices = body.get("choices") or []
    if not choices or "text" not in choices[0]:
        return failed("completions_endpoint", "Response missing choices[0].text")

    return passed("completions_endpoint", f"OpenAI-compatible, generated {_text(choices[0]['text'])[:80]!r}")


def check_chat_completions_endpoint(
    service_url: str,
    model_id: str,
    timeout: float = SERVICE_CHECK_TIMEOUT_DEFAULT,
    *,
    reasoning_enabled: bool = False,
) -> CheckResult:
    """Verify ``/v1/chat/completions`` returns an OpenAI-shaped response.

    A reasoning-enabled model can legitimately return ``content: null`` when the
    completion is cut short mid-"thinking" (the low ``max_tokens`` used here makes
    that common); ``reasoning_content`` standing in for ``content`` is only
    accepted when the profile declares the reasoning capability, so a broken
    chat template on a non-reasoning model still fails this check.
    """
    try:
        body = _chat(
            service_url,
            model_id,
            [{"role": "user", "content": "Hello"}],
            timeout,
            max_tokens=5,
            temperature=0.7,
        )
    except ServiceError as exc:
        return failed("chat_completions_endpoint", str(exc))

    if not (body.get("choices") or []):
        return failed("chat_completions_endpoint", "Response missing choices")

    message = _message(body)
    if "content" not in message:
        return failed("chat_completions_endpoint", "Response missing choices[0].message.content")

    content = message["content"]
    if isinstance(content, str):
        return passed("chat_completions_endpoint", f"OpenAI-compatible, generated {_text(content)[:80]!r}")

    reasoning_content = message.get("reasoning") or message.get("reasoning_content")
    if reasoning_enabled and content is None and isinstance(reasoning_content, str):
        return passed(
            "chat_completions_endpoint",
            f"OpenAI-compatible, generated reasoning content {_text(reasoning_content)[:80]!r}",
        )

    if content is None:
        detail = (
            "Response content is null without reasoning content"
            if reasoning_enabled
            else "Response content is null and the profile does not declare reasoning support"
        )
    else:
        detail = f"Response content is {type(content).__name__}, expected str"
    return failed("chat_completions_endpoint", detail)


# --------------------------------------------------------------------------- #
# Tool calling
# --------------------------------------------------------------------------- #

_WEATHER_TOOL = {
    "type": "function",
    "function": {
        "name": "get_weather",
        "description": "Get current weather",
        "parameters": {
            "type": "object",
            "properties": {
                "location": {"type": "string", "description": "City and country, e.g., 'Helsinki, Finland'"},
                "unit": {"type": "string", "enum": ["celsius", "fahrenheit"]},
            },
            "required": ["location", "unit"],
        },
    },
}

_WEATHER_QUESTION = "What's the weather like in Helsinki, Finland in Celsius?"


def check_tool_invocation(
    service_url: str, model_id: str, timeout: float = SERVICE_CHECK_TIMEOUT_DEFAULT
) -> CheckResult:
    """Verify the service accepts a tool-calling request and returns a tool call.

    Whether the model *chooses* to call the tool is model behaviour, not a
    serving defect, so a missing call is a warning. A rejected request (tool
    parsing not enabled on the server) is a failure.
    """
    try:
        body = _chat(
            service_url,
            model_id,
            [{"role": "user", "content": _WEATHER_QUESTION}],
            timeout,
            tools=[_WEATHER_TOOL],
            tool_choice="auto",
            temperature=0.0,
        )
    except ServiceError as exc:
        return failed("tool_invocation", str(exc))

    message = _message(body)
    tool_calls = message.get("tool_calls")
    if not tool_calls:
        warning = f"Model did not call a tool when offered one (question: {_WEATHER_QUESTION!r})"
        logger.warning("%s. Full message: %s", warning, message)
        return passed("tool_invocation", "Request accepted but no tool_calls returned", warnings=[warning])

    warnings = []
    if _text(message.get("content")):
        warnings.append("Response has non-empty content alongside tool_calls")

    function = (tool_calls[0].get("function") or {}) if isinstance(tool_calls[0], dict) else {}
    called = function.get("name", "unknown")
    return passed(
        "tool_invocation", f"Called {called}() with {_truncate(str(function.get('arguments', '')), 120)}", warnings
    )


def check_tool_avoidance(
    service_url: str, model_id: str, timeout: float = SERVICE_CHECK_TIMEOUT_DEFAULT
) -> CheckResult:
    """Verify regular chat turns come back as prose, not tool calls.

    Covers both a request with no tools at all and one where tools are offered
    but shouldn't be needed. Spurious tool calls and leaked chat-template tokens
    are warnings; empty or malformed content is a failure, since that means the
    template is broken.
    """
    scenarios: tuple[tuple[list[dict[str, Any]], dict[str, Any]], ...] = (
        ([{"role": "user", "content": _WEATHER_QUESTION}], {}),
        (
            [{"role": "user", "content": "Hello! Briefly introduce yourself."}],
            {"tools": [_WEATHER_TOOL], "tool_choice": "auto"},
        ),
    )

    warnings: list[str] = []
    failures: list[str] = []
    for index, (messages, options) in enumerate(scenarios):
        try:
            body = _chat(service_url, model_id, messages, timeout, temperature=0.7, **options)
        except ServiceError as exc:
            failures.append(f"Request #{index} failed: {exc}")
            continue

        message = _message(body)
        if message.get("tool_calls"):
            warnings.append(f"Request #{index} unexpectedly returned tool calls in regular chat")
            logger.warning("%s. Full message: %s", warnings[-1], message)
            continue

        content = _text(message.get("content"))
        if not content:
            failures.append(f"Request #{index} returned empty or non-string content")
            continue

        for token in TOOL_LEAK_TOKENS:
            if token in content:
                warnings.append(f"Request #{index} content contains tool-call token {token!r}")
                logger.warning("%s: %s", warnings[-1], _truncate(content, 100))
                break

    if failures:
        return failed("tool_avoidance", "; ".join(failures))

    return passed("tool_avoidance", f"{len(scenarios)} regular chat request(s) returned well-formed content", warnings)


# --------------------------------------------------------------------------- #
# Structured outputs
# --------------------------------------------------------------------------- #


class _CalendarEvent(BaseModel):
    name: str
    date: str
    participants: list[str]


class _Customer(BaseModel):
    name: str
    email: str


class _LineItem(BaseModel):
    description: str
    quantity: int
    unit_price: float


class _Invoice(BaseModel):
    """Nested schema with a constrained literal and an array of objects.

    A flat schema is too forgiving — most backends satisfy it even when nested
    object and enum guidance is broken, so regressions only surface here.
    """

    invoice_id: str
    status: Literal["draft", "sent", "paid"]
    customer: _Customer
    line_items: list[_LineItem]


def check_structured_output(
    service_url: str, model_id: str, timeout: float = SERVICE_CHECK_TIMEOUT_DEFAULT
) -> CheckResult:
    """Verify constrained decoding against a flat schema."""
    return _check_schema_conformance(
        name="structured_output",
        service_url=service_url,
        model_id=model_id,
        schema_name="calendar-event",
        model=_CalendarEvent,
        messages=[
            {"role": "system", "content": "Extract the event information."},
            {"role": "user", "content": "Alice and Bob are going to a science fair on Friday."},
        ],
        timeout=timeout,
    )


def check_structured_output_nested(
    service_url: str, model_id: str, timeout: float = SERVICE_CHECK_TIMEOUT_DEFAULT
) -> CheckResult:
    """Verify constrained decoding against a nested schema."""
    return _check_schema_conformance(
        name="structured_output_nested",
        service_url=service_url,
        model_id=model_id,
        schema_name="invoice",
        model=_Invoice,
        messages=[
            {"role": "system", "content": "Extract the invoice information into the requested schema."},
            {
                "role": "user",
                "content": (
                    "Invoice INV-001, status sent, for customer Alice (alice@example.com). "
                    "Two line items: 3 widgets at 9.99 each, and 1 sprocket at 4.50."
                ),
            },
        ],
        timeout=timeout,
    )


def _check_schema_conformance(
    name: str,
    service_url: str,
    model_id: str,
    schema_name: str,
    model: type[BaseModel],
    messages: list[dict[str, str]],
    timeout: float,
) -> CheckResult:
    try:
        body = _chat(
            service_url,
            model_id,
            messages,
            timeout,
            temperature=0.0,
            response_format={
                "type": "json_schema",
                "json_schema": {"name": schema_name, "schema": model.model_json_schema()},
            },
        )
    except ServiceError as exc:
        return failed(name, str(exc))

    content = _message(body).get("content") or ""
    try:
        model.model_validate(json.loads(content))
    except json.JSONDecodeError as exc:
        return failed(name, f"Response is not valid JSON: {exc}; content={_truncate(content, 200)!r}")
    except ValidationError as exc:
        return failed(
            name, f"Response violates schema: {_truncate(str(exc), 200)}; content={_truncate(content, 200)!r}"
        )

    return passed(name, f"Response conforms to {schema_name} schema")


#: Labels offered to — and required back from — the constrained-choice check.
_CHOICE_LABELS = ("positive", "negative")


def check_structured_output_choice(
    service_url: str, model_id: str, timeout: float = SERVICE_CHECK_TIMEOUT_DEFAULT
) -> CheckResult:
    """Verify a constrained-choice request answers with one of the offered labels.

    ``structured_outputs`` goes on the request body directly. CI nests it under
    ``extra_body``, which is an OpenAI *client* convention the server never sees,
    so there the constraint silently does nothing and the check only passes when
    the model happens to reply with a bare label.
    """
    try:
        body = _chat(
            service_url,
            model_id,
            [{"role": "user", "content": "Classify this sentiment: the product is great!"}],
            timeout,
            temperature=0.0,
            structured_outputs={"choice": list(_CHOICE_LABELS)},
        )
    except ServiceError as exc:
        return failed("structured_output_choice", str(exc))

    content = _text(_message(body).get("content"))
    if content.lower() not in _CHOICE_LABELS:
        return failed(
            "structured_output_choice",
            f"Response is not one of {', '.join(_CHOICE_LABELS)}: {_truncate(content, 200)!r}",
        )

    return passed("structured_output_choice", f"Constrained choice returned {content!r}")


# --------------------------------------------------------------------------- #
# Reasoning
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class ReasoningCase:
    domain: str
    question: str
    max_tokens: int = DEFAULT_REASONING_MAX_TOKENS


#: Mirrors ``ci/model_service_validation/validation_data/reasoning_test_cases.yaml``.
#: Inlined rather than read from YAML so the checks need no packaged data files.
REASONING_CASES: tuple[ReasoningCase, ...] = (
    ReasoningCase("arithmetic", "What is 25 multiplied by 17? Think step by step."),
    ReasoningCase(
        "word problem",
        "A farmer has 15 chickens and 8 cows. How many total legs do all the animals have? Explain your reasoning.",
    ),
    ReasoningCase(
        "logical reasoning",
        "If all roses are flowers and some flowers fade quickly, "
        "can we conclude that some roses fade quickly? Why or why not?",
    ),
    ReasoningCase("science", "Why does ice float on water instead of sinking? Explain the science.", 2048),
    ReasoningCase(
        "practical math",
        "A store sells apples for $1.50 each and oranges for $2.00 each. "
        "If I buy 4 apples and 3 oranges, how much do I spend in total? Show your work.",
    ),
)


def check_reasoning(
    service_url: str,
    model_id: str,
    cases: tuple[ReasoningCase, ...] = REASONING_CASES,
    timeout: float = SERVICE_CHECK_TIMEOUT_DEFAULT,
) -> CheckResult:
    """Verify reasoning prompts produce content or a parsed ``reasoning_content``.

    A model with a reasoning parser configured splits its chain of thought into
    ``reasoning_content``; either field being non-empty means the parser did not
    swallow the whole response.
    """
    failures: list[str] = []
    reasoning_fields = 0

    for case in cases:
        try:
            body = _chat(
                service_url,
                model_id,
                [{"role": "user", "content": case.question}],
                timeout,
                temperature=0.0,
                max_tokens=case.max_tokens,
            )
        except ServiceError as exc:
            failures.append(f"{case.domain}: {exc}")
            continue

        choices = body.get("choices") or []
        if not choices:
            failures.append(f"{case.domain}: response has no choices")
            continue

        message = _message(body)
        if _text(message.get("reasoning_content")):
            reasoning_fields += 1
        elif not _text(message.get("content")):
            usage = body.get("usage") or {}
            failures.append(
                f"{case.domain}: empty response "
                f"(finish_reason={choices[0].get('finish_reason', 'unknown')}, "
                f"completion_tokens={usage.get('completion_tokens', '?')})"
            )

    if failures:
        return failed("reasoning", f"{len(failures)}/{len(cases)} case(s) failed: {'; '.join(failures)}")

    warnings = []
    if not reasoning_fields:
        warnings.append("No response carried reasoning_content — a reasoning parser may not be configured")
    return passed(
        "reasoning", f"{len(cases)} case(s) produced output ({reasoning_fields} with reasoning_content)", warnings
    )
