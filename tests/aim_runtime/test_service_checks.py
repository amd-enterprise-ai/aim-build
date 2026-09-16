# Copyright © Advanced Micro Devices, Inc., or its affiliates.
#
# SPDX-License-Identifier: MIT

"""
Tests for aim_runtime.harness.service_checks.

Each check runs against an in-process OpenAI-compatible stub rather than mocked
transport, so real HTTP requests are exercised — including a 400 response whose
body has to reach the CheckResult detail.
"""

import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest

from aim_runtime.harness.service_checks import (
    ServiceError,
    check_chat_completions_endpoint,
    check_completions_endpoint,
    check_reasoning,
    check_structured_output,
    check_structured_output_choice,
    check_structured_output_nested,
    check_tool_avoidance,
    check_tool_invocation,
    probe_api_health,
    request_json,
    run_warmup,
)

# ---------------------------------------------------------------------------
# Stub service
# ---------------------------------------------------------------------------


class _Server(ThreadingHTTPServer):
    #: Keep-alive handler threads outlive the request, so don't join them on
    #: close — waiting adds half a second of teardown to every test.
    daemon_threads = True
    block_on_close = False


class StubService:
    """In-process HTTP stub whose routes tests configure per case.

    A route is either a ``(status, body)`` pair or a callable taking the request
    payload and returning one, which is how the retry/multi-request checks vary
    their answer per call.
    """

    def __init__(self):
        self.routes = {}
        self.requests = []
        self._server = _Server(("127.0.0.1", 0), self._handler())
        # A short poll interval keeps shutdown() from costing serve_forever's
        # 0.5s default poll on every test teardown.
        self._thread = threading.Thread(target=self._server.serve_forever, kwargs={"poll_interval": 0.01}, daemon=True)
        self._thread.start()

    @property
    def url(self) -> str:
        host, port = self._server.server_address[:2]
        return f"http://{host}:{port}"

    def route(self, path, response):
        self.routes[path] = response

    def paths(self) -> list[str]:
        return [path for path, _ in self.requests]

    def shutdown(self):
        self._server.shutdown()
        self._server.server_close()
        self._thread.join(timeout=5)

    def _handler(self):
        service = self

        class Handler(BaseHTTPRequestHandler):
            protocol_version = "HTTP/1.1"

            def log_message(self, *args):
                pass

            def do_GET(self):
                service.requests.append((self.path, None))
                self._respond(None)

            def do_POST(self):
                length = int(self.headers.get("Content-Length") or 0)
                payload = json.loads(self.rfile.read(length) or b"{}")
                service.requests.append((self.path, payload))
                self._respond(payload)

            def _respond(self, payload):
                route = service.routes.get(self.path)
                if route is None:
                    self.send_error(404, "no route")
                    return

                status, body = route(payload) if callable(route) else route
                raw = b"" if body is None else (body if isinstance(body, bytes) else json.dumps(body).encode())
                self.send_response(status)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(raw)))
                self.end_headers()
                if raw:
                    self.wfile.write(raw)

        return Handler


@pytest.fixture
def service():
    stub = StubService()
    yield stub
    stub.shutdown()


# ---------------------------------------------------------------------------
# Response builders
# ---------------------------------------------------------------------------


def models_body(*ids):
    return {"object": "list", "data": [{"id": model_id, "object": "model"} for model_id in ids]}


def completion_body(text="Hi there"):
    return {
        "id": "cmpl-1",
        "object": "text_completion",
        "created": 1,
        "model": "stub",
        "choices": [{"index": 0, "text": text, "finish_reason": "length"}],
        "usage": {"completion_tokens": 2},
    }


def chat_body(content="Hi there", **message_extras):
    message = {"role": "assistant", "content": content}
    message.update(message_extras)
    return {
        "id": "chatcmpl-1",
        "object": "chat.completion",
        "created": 1,
        "model": "stub",
        "choices": [{"index": 0, "message": message, "finish_reason": "stop"}],
        "usage": {"completion_tokens": 2},
    }


def tool_call_body(name="get_weather", arguments='{"location": "Helsinki, Finland", "unit": "celsius"}', content=None):
    body = chat_body(content=content)
    body["choices"][0]["message"]["tool_calls"] = [
        {"id": "call-1", "type": "function", "function": {"name": name, "arguments": arguments}}
    ]
    return body


def serve_chat(body):
    return (200, body)


# ---------------------------------------------------------------------------
# HTTP plumbing
# ---------------------------------------------------------------------------


class TestRequestJson:
    def test_surfaces_error_body_on_http_error(self, service):
        service.route(
            "/v1/chat/completions",
            (400, {"error": {"message": '"auto" tool choice requires --enable-auto-tool-choice'}}),
        )

        with pytest.raises(ServiceError) as excinfo:
            request_json(f"{service.url}/v1/chat/completions", {"model": "stub"})

        assert excinfo.value.status == 400
        assert "enable-auto-tool-choice" in str(excinfo.value)
        assert "enable-auto-tool-choice" in excinfo.value.body

    def test_unreachable_service_raises(self):
        with pytest.raises(ServiceError, match="unreachable"):
            request_json("http://127.0.0.1:1/v1/models", timeout=1)

    def test_invalid_json_raises(self, service):
        service.route("/v1/models", (200, b"not json"))

        with pytest.raises(ServiceError, match="invalid JSON"):
            request_json(f"{service.url}/v1/models")


# ---------------------------------------------------------------------------
# Readiness and warmup
# ---------------------------------------------------------------------------


class TestProbeApiHealth:
    def test_returns_served_model_id(self, service):
        service.route("/v1/models", (200, models_body("amd/Llama-3.1-8B-FP8")))
        service.route("/health", (200, None))

        probe = probe_api_health(service.url, timeout_seconds=5)

        assert probe.check.success
        assert probe.model_id == "amd/Llama-3.1-8B-FP8"
        assert probe.ready_time_seconds is not None
        assert not probe.check.warnings

    def test_served_model_id_wins_over_profile_model_id(self, service):
        service.route("/v1/models", (200, models_body("/workspace/model-cache/quantized")))
        service.route("/health", (200, None))

        assert probe_api_health(service.url, timeout_seconds=5).model_id == "/workspace/model-cache/quantized"

    def test_missing_health_route_is_a_warning_not_a_failure(self, service):
        service.route("/v1/models", (200, models_body("stub")))

        probe = probe_api_health(service.url, timeout_seconds=5)

        assert probe.check.success
        assert probe.model_id == "stub"
        assert any("/health" in warning for warning in probe.check.warnings)

    def test_fails_when_no_model_is_served(self, service):
        service.route("/v1/models", (200, models_body()))

        probe = probe_api_health(service.url, timeout_seconds=1, retry_interval=0.01)

        assert not probe.check.success
        assert probe.model_id is None
        assert "served no models" in probe.check.detail

    def test_fails_when_service_unreachable(self):
        probe = probe_api_health("http://127.0.0.1:1", timeout_seconds=1, retry_interval=0.01)

        assert not probe.check.success
        assert probe.model_id is None

    def test_retries_until_model_appears(self, service):
        attempts = {"n": 0}

        def models(_payload):
            attempts["n"] += 1
            return (200, models_body("stub")) if attempts["n"] > 2 else (503, {"error": "loading"})

        service.route("/v1/models", models)
        service.route("/health", (200, None))

        probe = probe_api_health(service.url, timeout_seconds=5, retry_interval=0.01)

        assert probe.check.success
        assert attempts["n"] == 3


class TestRunWarmup:
    def test_succeeds_on_first_inference(self, service):
        service.route("/v1/completions", (200, completion_body()))

        outcome = run_warmup(service.url, "stub", max_warmup_time=5)

        assert outcome.succeeded
        assert outcome.check.success
        assert outcome.attempts == 1
        assert not outcome.jit_suspected

    def test_sends_a_minimal_prompt(self, service):
        service.route("/v1/completions", (200, completion_body()))

        run_warmup(service.url, "stub", max_warmup_time=5)

        _, payload = service.requests[0]
        assert payload["max_tokens"] == 1
        assert payload["model"] == "stub"

    def test_retries_until_inference_succeeds(self, service):
        attempts = {"n": 0}

        def completions(_payload):
            attempts["n"] += 1
            return (200, completion_body()) if attempts["n"] > 1 else (500, {"error": "not ready"})

        service.route("/v1/completions", completions)

        outcome = run_warmup(service.url, "stub", max_warmup_time=5, retry_interval=0.01)

        assert outcome.succeeded
        assert outcome.attempts == 2

    def test_flags_suspected_jit_without_failing(self, service):
        attempts = {"n": 0}

        def completions(_payload):
            attempts["n"] += 1
            return (200, completion_body()) if attempts["n"] > 1 else (500, {"error": "compiling"})

        service.route("/v1/completions", completions)

        outcome = run_warmup(service.url, "stub", max_warmup_time=5, jit_threshold_seconds=0.0, retry_interval=0.01)

        assert outcome.succeeded
        assert outcome.check.success
        assert outcome.jit_suspected
        assert any("JIT" in warning for warning in outcome.check.warnings)

    def test_fails_when_inference_never_succeeds(self, service):
        service.route("/v1/completions", (500, {"error": "broken"}))

        outcome = run_warmup(service.url, "stub", max_warmup_time=1, retry_interval=0.01)

        assert not outcome.succeeded
        assert not outcome.check.success
        assert "broken" in outcome.check.detail


# ---------------------------------------------------------------------------
# OpenAI API compatibility
# ---------------------------------------------------------------------------


class TestEndpointChecks:
    def test_completions_passes_on_openai_shape(self, service):
        service.route("/v1/completions", (200, completion_body("Hello!")))

        assert check_completions_endpoint(service.url, "stub").success

    def test_completions_fails_on_missing_openai_fields(self, service):
        body = completion_body()
        del body["created"]
        del body["model"]
        service.route("/v1/completions", (200, body))

        result = check_completions_endpoint(service.url, "stub")

        assert not result.success
        assert "created" in result.detail and "model" in result.detail

    def test_completions_fails_on_missing_text(self, service):
        body = completion_body()
        del body["choices"][0]["text"]
        service.route("/v1/completions", (200, body))

        result = check_completions_endpoint(service.url, "stub")

        assert not result.success
        assert "choices[0].text" in result.detail

    def test_chat_completions_passes_on_openai_shape(self, service):
        service.route("/v1/chat/completions", (200, chat_body()))

        assert check_chat_completions_endpoint(service.url, "stub").success

    def test_chat_completions_fails_on_missing_content(self, service):
        body = chat_body()
        del body["choices"][0]["message"]["content"]
        service.route("/v1/chat/completions", (200, body))

        result = check_chat_completions_endpoint(service.url, "stub")

        assert not result.success
        assert "message.content" in result.detail

    def test_chat_completions_reports_error_body(self, service):
        service.route("/v1/chat/completions", (400, {"error": "max_model_len exceeded"}))

        result = check_chat_completions_endpoint(service.url, "stub")

        assert not result.success
        assert "max_model_len exceeded" in result.detail

    def test_chat_completions_fails_on_null_content_when_reasoning_not_declared(self, service):
        """A broken chat template that always returns null content must not pass silently."""
        service.route("/v1/chat/completions", (200, chat_body(content=None)))

        result = check_chat_completions_endpoint(service.url, "stub", reasoning_enabled=False)

        assert not result.success
        assert "reasoning support" in result.detail

    def test_chat_completions_fails_on_null_content_with_reasoning_declared_but_no_reasoning_content(self, service):
        service.route("/v1/chat/completions", (200, chat_body(content=None)))

        result = check_chat_completions_endpoint(service.url, "stub", reasoning_enabled=True)

        assert not result.success
        assert "without reasoning content" in result.detail

    def test_chat_completions_passes_on_null_content_with_reasoning_content_when_declared(self, service):
        service.route(
            "/v1/chat/completions",
            (200, chat_body(content=None, reasoning_content="step by step...")),
        )

        result = check_chat_completions_endpoint(service.url, "stub", reasoning_enabled=True)

        assert result.success
        assert "reasoning content" in result.detail


# ---------------------------------------------------------------------------
# Tool calling
# ---------------------------------------------------------------------------


class TestToolInvocation:
    def test_passes_when_tool_is_called(self, service):
        service.route("/v1/chat/completions", (200, tool_call_body()))

        result = check_tool_invocation(service.url, "stub")

        assert result.success
        assert "get_weather()" in result.detail
        assert not result.warnings

    def test_offers_tools_with_auto_choice(self, service):
        service.route("/v1/chat/completions", (200, tool_call_body()))

        check_tool_invocation(service.url, "stub")

        _, payload = service.requests[0]
        assert payload["tool_choice"] == "auto"
        assert payload["tools"][0]["function"]["name"] == "get_weather"

    def test_declining_to_call_a_tool_warns_but_passes(self, service):
        service.route("/v1/chat/completions", (200, chat_body("It is cold in Helsinki.")))

        result = check_tool_invocation(service.url, "stub")

        assert result.success
        assert any("did not call a tool" in warning for warning in result.warnings)

    def test_content_alongside_tool_calls_warns(self, service):
        service.route("/v1/chat/completions", (200, tool_call_body(content="Let me check that")))

        result = check_tool_invocation(service.url, "stub")

        assert result.success
        assert any("non-empty content" in warning for warning in result.warnings)

    def test_rejected_request_fails_with_server_reason(self, service):
        service.route("/v1/chat/completions", (400, {"error": "tool choice requires --enable-auto-tool-choice"}))

        result = check_tool_invocation(service.url, "stub")

        assert not result.success
        assert "enable-auto-tool-choice" in result.detail


class TestToolAvoidance:
    def test_passes_when_regular_chat_returns_prose(self, service):
        service.route("/v1/chat/completions", (200, chat_body("I am a helpful assistant.")))

        result = check_tool_avoidance(service.url, "stub")

        assert result.success
        assert not result.warnings
        assert len(service.requests) == 2

    def test_unexpected_tool_call_warns_but_passes(self, service):
        service.route("/v1/chat/completions", (200, tool_call_body()))

        result = check_tool_avoidance(service.url, "stub")

        assert result.success
        assert any("unexpectedly returned tool calls" in warning for warning in result.warnings)

    def test_leaked_tool_token_warns_but_passes(self, service):
        service.route("/v1/chat/completions", (200, chat_body("[TOOL_CALLS]get_weather{}")))

        result = check_tool_avoidance(service.url, "stub")

        assert result.success
        assert any("[TOOL_CALLS]" in warning for warning in result.warnings)

    def test_empty_content_fails(self, service):
        service.route("/v1/chat/completions", (200, chat_body("   ")))

        result = check_tool_avoidance(service.url, "stub")

        assert not result.success
        assert "empty or non-string content" in result.detail

    def test_every_request_is_evaluated_before_failing(self, service):
        """CI scores each regular-chat request independently; both must be reported."""
        service.route("/v1/chat/completions", (200, chat_body("")))

        result = check_tool_avoidance(service.url, "stub")

        assert not result.success
        assert "Request #0" in result.detail and "Request #1" in result.detail
        assert len(service.requests) == 2


# ---------------------------------------------------------------------------
# Structured outputs
# ---------------------------------------------------------------------------


class TestStructuredOutput:
    def test_passes_on_schema_conforming_json(self, service):
        event = {"name": "Science Fair", "date": "Friday", "participants": ["Alice", "Bob"]}
        service.route("/v1/chat/completions", (200, chat_body(json.dumps(event))))

        assert check_structured_output(service.url, "stub").success

    def test_requests_json_schema_response_format(self, service):
        event = {"name": "Science Fair", "date": "Friday", "participants": ["Alice"]}
        service.route("/v1/chat/completions", (200, chat_body(json.dumps(event))))

        check_structured_output(service.url, "stub")

        _, payload = service.requests[0]
        assert payload["response_format"]["type"] == "json_schema"
        assert payload["response_format"]["json_schema"]["name"] == "calendar-event"

    def test_fails_on_non_json_content(self, service):
        service.route("/v1/chat/completions", (200, chat_body("Sure! Here is the event.")))

        result = check_structured_output(service.url, "stub")

        assert not result.success
        assert "not valid JSON" in result.detail

    def test_fails_when_schema_is_violated(self, service):
        service.route("/v1/chat/completions", (200, chat_body(json.dumps({"name": "Fair"}))))

        result = check_structured_output(service.url, "stub")

        assert not result.success
        assert "violates schema" in result.detail

    def test_nested_schema_passes(self, service):
        invoice = {
            "invoice_id": "INV-001",
            "status": "sent",
            "customer": {"name": "Alice", "email": "alice@example.com"},
            "line_items": [{"description": "widget", "quantity": 3, "unit_price": 9.99}],
        }
        service.route("/v1/chat/completions", (200, chat_body(json.dumps(invoice))))

        assert check_structured_output_nested(service.url, "stub").success

    def test_nested_schema_rejects_bad_literal(self, service):
        invoice = {
            "invoice_id": "INV-001",
            "status": "posted",
            "customer": {"name": "Alice", "email": "alice@example.com"},
            "line_items": [],
        }
        service.route("/v1/chat/completions", (200, chat_body(json.dumps(invoice))))

        result = check_structured_output_nested(service.url, "stub")

        assert not result.success
        assert "violates schema" in result.detail

    def test_choice_passes_on_an_offered_label(self, service):
        service.route("/v1/chat/completions", (200, chat_body("Positive")))

        assert check_structured_output_choice(service.url, "stub").success

    def test_choice_sends_the_constraint(self, service):
        service.route("/v1/chat/completions", (200, chat_body("positive")))

        check_structured_output_choice(service.url, "stub")

        _, payload = service.requests[0]
        assert payload["structured_outputs"]["choice"] == ["positive", "negative"]

    def test_choice_fails_on_unconstrained_prose(self, service):
        service.route("/v1/chat/completions", (200, chat_body("The sentiment is clearly good!")))

        result = check_structured_output_choice(service.url, "stub")

        assert not result.success
        assert "not one of positive, negative" in result.detail


# ---------------------------------------------------------------------------
# Reasoning
# ---------------------------------------------------------------------------


class TestReasoning:
    def test_passes_on_reasoning_content(self, service):
        service.route("/v1/chat/completions", (200, chat_body(None, reasoning_content="25 * 17 = 425")))

        result = check_reasoning(service.url, "stub")

        assert result.success
        assert not result.warnings

    def test_content_only_passes_but_warns_about_missing_parser(self, service):
        service.route("/v1/chat/completions", (200, chat_body("425")))

        result = check_reasoning(service.url, "stub")

        assert result.success
        assert any("reasoning_content" in warning for warning in result.warnings)

    def test_fails_on_empty_responses(self, service):
        service.route("/v1/chat/completions", (200, chat_body("")))

        result = check_reasoning(service.url, "stub")

        assert not result.success
        assert "empty response" in result.detail

    def test_runs_every_case(self, service):
        service.route("/v1/chat/completions", (200, chat_body("425")))

        check_reasoning(service.url, "stub")

        assert len(service.requests) == 5

    def test_reports_failing_case_domain(self, service):
        service.route("/v1/chat/completions", (400, {"error": "context length"}))

        result = check_reasoning(service.url, "stub")

        assert not result.success
        assert "arithmetic" in result.detail
