# Copyright © Advanced Micro Devices, Inc., or its affiliates.
#
# SPDX-License-Identifier: MIT

"""CPU unit tier: every request size uses the compiled modules, and real compile-event counting.

``torch.compile`` is replaced by a version with the "eager" Dynamo backend, so real Dynamo compiles
(and real compile callbacks) happen on CPU with no C++ compiler.
"""

from __future__ import annotations

import functools

import compilation
import pytest
import torch
from compilation import (
    COMPILE_INDUCTOR,
    COMPILE_NONE,
    CompileConfig,
    CompileStatus,
    compile_status,
    install_compilation,
)


class _TargetA(torch.nn.Module):
    def forward(self, x):
        return x * 2


class _TargetB(torch.nn.Module):
    def forward(self, x):
        return x + 1


class _FakeBoltz(torch.nn.Module):
    """Stands in for Boltz2: one forward per request, taking a feature dict, calling both compiled modules."""

    def __init__(self):
        super().__init__()
        self.transition = _TargetA()
        self.token_transformer = _TargetB()

    def forward(self, feats, recycling_steps=0):
        n = feats["token_pad_mask"].shape[1]
        out = self.transition(torch.ones(1, n, 4)) + self.token_transformer(torch.ones(1, n, 4))
        return out


@pytest.fixture
def eager_backend_compile(monkeypatch):
    """torch.compile -> Dynamo with the eager backend; returns a counter of calls into compiled functions."""
    monkeypatch.setitem(compilation._TARGETS, "transition", (__name__, "_TargetA"))
    monkeypatch.setitem(compilation._TARGETS, "token_transformer", (__name__, "_TargetB"))
    real_compile = torch.compile
    calls = {"compiled_fn_calls": 0, "dynamic": True}

    def fake_compile(fn, *args, **kwargs):
        compiled = real_compile(fn, backend="eager", dynamic=calls["dynamic"])

        @functools.wraps(fn)
        def counted(*a, **k):
            calls["compiled_fn_calls"] += 1
            return compiled(*a, **k)

        return counted

    monkeypatch.setattr(torch, "compile", fake_compile)
    torch._dynamo.reset()
    yield calls
    torch._dynamo.reset()


def _feats(n_tokens: int) -> dict:
    return {"token_pad_mask": torch.ones(1, n_tokens)}


def _install(model, *, transition=COMPILE_INDUCTOR, token_transformer=COMPILE_INDUCTOR):
    install_compilation(model, CompileConfig(transition=transition, token_transformer=token_transformer))


def test_dynamo_compiles_are_counted_exactly_with_static_shapes(eager_backend_compile):
    eager_backend_compile["dynamic"] = False  # each new shape recompiles
    model = _FakeBoltz()
    _install(model)
    model(_feats(4))
    model(_feats(4))  # same shape: no new compile
    model(_feats(6))  # new shape: recompile
    status = compile_status(model)["transition"]
    assert status["dynamo_compile_callbacks"] == 2 and status["successful_compiles"] == 2
    assert status["forward_frame_compiles"] == 2 and status["forward_recompiles"] == 1
    assert status["graph_break_frames"] == 0 and status["failed_compiles"] == 0
    assert status["compiled_calls"] == 3 and status["distinct_input_shapes_seen"] == 2
    log = status["compile_event_log_tail"]
    assert [e["index"] for e in log] == [1, 2]
    assert log[0]["input_shapes"] == "[[1, 4, 4]]" and log[1]["input_shapes"] == "[[1, 6, 4]]"
    assert all(e["call_seconds"] >= 0 for e in log)
    assert status["compile_event_source"] == "dynamo-callback"


def test_dynamic_shapes_serve_new_lengths_without_a_new_compile_event(eager_backend_compile):
    model = _FakeBoltz()
    _install(model)
    for n in (4, 6, 8, 10):
        model(_feats(n))
    status = compile_status(model)["transition"]
    assert status["distinct_input_shapes_seen"] == 4
    assert status["successful_compiles"] < status["distinct_input_shapes_seen"]  # a dynamic dimension reused a graph
    assert status["successful_compiles"] >= 1


def test_compiles_outside_the_wrappers_are_not_credited_to_a_status(eager_backend_compile):
    model = _FakeBoltz()
    _install(model)
    other = torch.compile(lambda x: x - 1)
    other(torch.ones(3))
    assert compile_status(model)["transition"]["dynamo_compile_callbacks"] == 0


def test_unavailable_compile_callbacks_are_reported_not_hidden(eager_backend_compile, monkeypatch):
    monkeypatch.setattr(compilation, "_ensure_compile_hooks", lambda: False)
    model = _FakeBoltz()
    _install(model)
    model(_feats(4))
    status = compile_status(model)["transition"]
    assert status["compile_event_source"] == "unavailable"
    assert status["compiled_calls"] == 1 and status["dynamo_compile_callbacks"] == 0


class _GraphBreakTarget(torch.nn.Module):
    def forward(self, x):
        y = x * 2
        torch._dynamo.graph_break()
        return y + 1


class _FailingBackendTarget(torch.nn.Module):
    def forward(self, x):
        return x + 5


class _SingleTargetModel(torch.nn.Module):
    def __init__(self, target, width=None):
        super().__init__()
        self.transition = target
        self.width = width

    def forward(self, feats):
        n = self.width or feats["token_pad_mask"].shape[1]
        return self.transition(torch.ones(1, n, 2))


def test_a_graph_break_is_counted_as_a_graph_break_frame_not_as_a_recompile(monkeypatch):
    monkeypatch.setitem(compilation._TARGETS, "transition", (__name__, "_GraphBreakTarget"))
    real_compile = torch.compile
    monkeypatch.setattr(torch, "compile", lambda fn, *a, **k: real_compile(fn, backend="eager", dynamic=False))
    torch._dynamo.reset()
    model = _SingleTargetModel(_GraphBreakTarget())
    install_compilation(model, CompileConfig(transition=COMPILE_INDUCTOR, token_transformer=COMPILE_NONE))
    model(_feats(4))
    first = compile_status(model)["transition"]
    assert first["dynamo_compile_callbacks"] == 2  # the forward frame and its resume frame
    assert first["successful_compiles"] == 2
    assert first["forward_frame_compiles"] == 1 and first["forward_recompiles"] == 0
    assert first["graph_break_frames"] == 1 and first["graph_break_frame_compiles"] == 1
    model(_feats(6))  # a new static shape recompiles both frames
    second = compile_status(model)["transition"]
    assert second["forward_recompiles"] == 1 and second["graph_break_frame_compiles"] == 2
    torch._dynamo.reset()


def test_a_failed_compile_is_counted_separately_and_keeps_its_shapes(monkeypatch):
    monkeypatch.setitem(compilation._TARGETS, "transition", (__name__, "_FailingBackendTarget"))
    real_compile = torch.compile

    def failing_backend(gm, example_inputs):
        raise RuntimeError("backend exploded")

    monkeypatch.setattr(torch, "compile", lambda fn, *a, **k: real_compile(fn, backend=failing_backend))
    torch._dynamo.reset()
    model = _SingleTargetModel(_FailingBackendTarget(), width=3)
    install_compilation(model, CompileConfig(transition=COMPILE_INDUCTOR, token_transformer=COMPILE_NONE))
    out = model(_feats(3))
    assert out.shape == (1, 3, 2) and torch.equal(out, torch.full((1, 3, 2), 6.0))  # the eager fallback still answers
    status = compile_status(model)["transition"]
    assert status["degraded"] is True and status["eager_fallback_calls"] == 1
    assert status["failed_compiles"] == 1 and status["successful_compiles"] == 0
    assert status["forward_frame_compiles"] == 0 and status["forward_recompiles"] == 0
    event = status["compile_event_log_tail"][-1]
    assert event["failed"] is True and event["input_shapes"] == "[[1, 3, 2]]"
    torch._dynamo.reset()


def test_the_event_log_keeps_the_latest_events_and_counts_the_dropped_ones():
    status = CompileStatus(mode=COMPILE_INDUCTOR, instances=1, invoked=True)
    for i in range(40):
        status.record_compile_callback(f"0/{i}")
    d = status.to_dict()
    assert d["dynamo_compile_callbacks"] == 40 and d["successful_compiles"] == 40
    assert d["forward_frame_compiles"] == 40 and d["forward_recompiles"] == 39
    assert len(status.compile_event_log) == compilation._MAX_COMPILE_EVENT_LOG == 32
    assert d["compile_events_dropped"] == 8
    assert [e["index"] for e in d["compile_event_log_tail"]] == [37, 38, 39, 40]  # the tail follows the latest events


def test_the_distinct_shape_cap_is_reported_not_silent():
    status = CompileStatus(mode=COMPILE_INDUCTOR, instances=1, invoked=True)
    cap = compilation._MAX_DISTINCT_SHAPES
    for i in range(cap + 6):
        status.note_input_shapes(f"[[1, {i}]]")
    status.note_input_shapes("[[1, 0]]")  # an already tracked shape is not an untracked one
    d = status.to_dict()
    assert d["distinct_input_shapes_seen"] == cap and d["distinct_input_shapes_capped"] is True
    assert d["input_shapes_not_tracked_calls"] == 6


def test_compile_ids_are_split_into_frame_and_attempt():
    status = CompileStatus(mode=COMPILE_INDUCTOR, instances=1, invoked=True)
    for cid in ("3/0", "3/1", "7/0", "weird"):
        status.record_compile_callback(cid)
    d = status.to_dict()
    assert d["forward_frame_compiles"] == 2 and d["forward_recompiles"] == 1
    assert d["graph_break_frames"] == 1 and d["graph_break_frame_compiles"] == 1
    assert d["unparsed_compile_ids"] == 1 and d["dynamo_compile_callbacks"] == 4


def test_registering_the_compile_callback_many_times_registers_it_once():
    import threading

    from torch._dynamo.callback import callback_handler

    torch._dynamo.reset()
    threads = [threading.Thread(target=compilation._ensure_compile_hooks) for _ in range(16)]
    for th in threads:
        th.start()
    for th in threads:
        th.join()
    for _ in range(5):
        compilation._ensure_compile_hooks()
    assert callback_handler.end_callbacks.count(compilation._on_compile_end) == 1
    torch._dynamo.reset()


def test_counting_survives_a_dynamo_reset_between_calls(eager_backend_compile):
    eager_backend_compile["dynamic"] = False
    model = _FakeBoltz()
    _install(model)
    model(_feats(4))
    torch._dynamo.reset()  # clears Dynamo's callbacks and caches
    model(_feats(6))
    status = compile_status(model)["transition"]
    assert status["dynamo_compile_callbacks"] == 2 and status["compile_event_source"] == "dynamo-callback"


def test_removed_compile_settings_are_not_read_from_the_environment(monkeypatch):
    pytest.importorskip("bentoml")
    import service

    for name in (
        "BOLTZ2_COMPILE_MAX_TOKENS",
        "BOLTZ2_COMPILE_TRANSITION",
        "BOLTZ2_COMPILE_TOKEN_TRANSFORMER",
        "BOLTZ2_COMPILE_WARMUP",
        "BOLTZ2_COMPILE_WARMUP_TOKENS",
        "BOLTZ2_TRITON_KERNELS",
        "BOLTZ2_NO_KERNELS",
        "BOLTZ2_AOT_ARTIFACT_DIR",
    ):
        monkeypatch.setenv(name, "-3")
    cfg = service.ServiceConfig()
    assert cfg.optimizations == "on"
    assert not {"compile_max_tokens", "compile_transition", "compile_warmup", "triton_kernels", "no_kernels"} & set(
        type(cfg).model_fields
    )


def test_a_very_large_request_still_uses_the_compiled_modules(eager_backend_compile):
    model = _FakeBoltz()
    _install(model)
    model(_feats(5000))
    status = compile_status(model)
    assert status["transition"]["compiled_calls"] == 1 and status["token_transformer"]["compiled_calls"] == 1
    assert status["transition"]["state"] == "compiled-executed"


def test_the_size_routing_api_is_gone():
    for name in (
        "COMPILE_MAX_TOKENS",
        "compile_routing",
        "reset_routing_counters",
        "RoutingStatus",
        "compile_max_tokens",
    ):
        assert not hasattr(compilation, name)
    assert "max_tokens" not in CompileConfig.__dataclass_fields__
