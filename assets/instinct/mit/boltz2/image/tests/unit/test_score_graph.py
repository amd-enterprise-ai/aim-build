# Copyright © Advanced Micro Devices, Inc., or its affiliates.
#
# SPDX-License-Identifier: MIT

"""Tests for the score-model CUDA graph wrapper (the paths that need no GPU)."""

from __future__ import annotations

import contextlib
import logging
import sys
import types
from types import SimpleNamespace

import pytest
import score_graph


class _Tensor:
    def __init__(self, cuda: bool, value: int = 0) -> None:
        self.is_cuda = cuda
        self.shape = (5, 8, 3)
        self.value = value

    def clone(self) -> _Tensor:
        return _Tensor(self.is_cuda, self.value)

    def copy_(self, other: _Tensor) -> _Tensor:
        self.value = other.value
        return self


class _OomTensor(_Tensor):
    def clone(self) -> _Tensor:
        raise RuntimeError("HIP out of memory")


class _Score:
    def __init__(self) -> None:
        self.calls = 0

    def forward(self, r_noisy, times, **conditioning):  # noqa: ANN001, ANN201
        self.calls += 1
        return ("eager", conditioning.get("multiplicity"))


def _model():
    score = _Score()
    seen = []

    def sample(*args, **kwargs):  # noqa: ANN002, ANN003, ANN202
        seen.append(graph.active)
        return score_model_call()

    structure = SimpleNamespace(score_model=score, sample=sample)
    model = SimpleNamespace(structure_module=structure)
    graph = score_graph.install_score_graph(model)

    def score_model_call():  # noqa: ANN202
        return score.forward(_Tensor(False), _Tensor(False), multiplicity=5)

    return model, graph, score, seen


def test_sample_marks_the_run_active_and_resets_after():
    model, graph, _, seen = _model()
    assert not graph.active
    model.structure_module.sample()
    assert seen == [True]
    assert not graph.active


def test_cpu_tensors_run_eager():
    _, graph, score, _ = _model()
    graph.start_run()
    for _ in range(3):
        assert graph(_Tensor(False), _Tensor(False), multiplicity=5) == ("eager", 5)
    assert score.calls == 3
    assert graph.replays == 0


def test_inactive_graph_runs_eager_for_cuda_tensors():
    _, graph, score, _ = _model()
    assert graph(_Tensor(True), _Tensor(True), multiplicity=5) == ("eager", 5)
    assert score.calls == 1
    assert graph.replays == 0


def test_first_call_of_a_run_is_eager_warm_up():
    _, graph, score, _ = _model()
    graph.start_run()
    assert graph(_Tensor(True), _Tensor(True), multiplicity=5) == ("eager", 5)
    assert score.calls == 1


def test_disabled_graph_runs_eager():
    _, graph, score, _ = _model()
    graph.enabled = False
    graph.start_run()
    for _ in range(3):
        graph(_Tensor(True), _Tensor(True), multiplicity=5)
    assert score.calls == 3


def test_key_change_restarts_the_warm_up():
    _, graph, score, _ = _model()
    graph.start_run()
    graph(_Tensor(True), _Tensor(True), multiplicity=5)
    graph(_Tensor(True), _Tensor(True), multiplicity=5, other=object())
    assert score.calls == 2


def test_failed_capture_runs_eager_and_turns_the_graph_off(monkeypatch):
    _, graph, score, _ = _model()
    captures = []

    def failing_capture(run, r_noisy, times, conditioning):  # noqa: ANN001, ANN202
        captures.append(1)
        graph.enabled = False
        return False

    monkeypatch.setattr(graph, "_capture", failing_capture)
    graph.start_run()
    cond = {"multiplicity": 5}
    for _ in range(4):
        assert graph(_Tensor(True), _Tensor(True), **cond) == ("eager", 5)
    assert captures == [1]
    assert score.calls == 4
    assert graph.replays == 0


def test_run_state_is_dropped_when_sample_raises():
    holder = {}

    def failing_sample(*args, **kwargs):  # noqa: ANN002, ANN003, ANN202
        assert holder["graph"].active
        raise RuntimeError("boom")

    structure = SimpleNamespace(score_model=_Score(), sample=failing_sample)
    holder["graph"] = score_graph.install_score_graph(SimpleNamespace(structure_module=structure))
    with pytest.raises(RuntimeError, match="boom"):
        structure.sample()
    assert not holder["graph"].active


def _fake_torch(monkeypatch, fail=None):
    """Put a fake ``torch.cuda`` with CUDAGraph, graph and synchronize in ``sys.modules``; return its event list."""
    events = []

    class CUDAGraph:
        def __init__(self) -> None:
            if fail == "graph":
                raise RuntimeError("HIP out of memory")

        def replay(self) -> None:
            events.append("replay")

    @contextlib.contextmanager
    def graph(cuda_graph, capture_error_mode=None):  # noqa: ANN001, ANN202
        events.append("capture")
        yield
        if fail == "capture":
            raise RuntimeError("capture failed")

    fake = types.ModuleType("torch")
    fake.cuda = SimpleNamespace(CUDAGraph=CUDAGraph, graph=graph, synchronize=lambda: events.append("sync"))
    monkeypatch.setitem(sys.modules, "torch", fake)
    return events


class _Doubling:
    def __init__(self) -> None:
        self.calls = 0

    def forward(self, r_noisy, times, **conditioning):  # noqa: ANN001, ANN201
        self.calls += 1
        return _Tensor(True, 2 * r_noisy.value)


@pytest.mark.parametrize(("fail", "tensor"), [("graph", _Tensor), ("clone", _OomTensor), ("capture", _Tensor)])
def test_a_capture_that_raises_turns_the_graph_off_and_returns_the_eager_result(monkeypatch, caplog, fail, tensor):
    _fake_torch(monkeypatch, fail)
    score = _Doubling()
    graph = score_graph.ScoreGraph(score.forward)
    graph.start_run()
    graph(tensor(True, 1), _Tensor(True), multiplicity=5)
    with caplog.at_level(logging.WARNING, logger="score_graph"):
        out = graph(tensor(True, 3), _Tensor(True), multiplicity=5)
    assert out.value == 6
    assert graph.enabled is False
    assert graph.replays == 0
    assert "CUDA graph capture of the score model failed" in caplog.text
    run = graph._local.run  # noqa: SLF001
    assert run.graph is None
    assert run.out is None
    assert run.r_noisy is None
    assert run.times is None
    assert graph(tensor(True, 4), _Tensor(True), multiplicity=5).value == 8


def test_a_captured_graph_copies_the_new_inputs_and_replays(monkeypatch):
    events = _fake_torch(monkeypatch)
    score = _Doubling()
    graph = score_graph.ScoreGraph(score.forward)
    graph.start_run()
    graph(_Tensor(True, 1), _Tensor(True), multiplicity=5)
    graph(_Tensor(True, 2), _Tensor(True), multiplicity=5)
    assert score.calls == 2
    assert events == ["sync", "capture", "replay"]
    run = graph._local.run  # noqa: SLF001
    out = graph(_Tensor(True, 7), _Tensor(True, 9), multiplicity=5)
    assert score.calls == 2
    assert events[-1] == "replay"
    assert graph.replays == 2
    assert (run.r_noisy.value, run.times.value) == (7, 9)
    assert out is not run.out
    assert out.value == run.out.value


def test_a_capture_waits_for_an_eager_call_that_does_not_compile(monkeypatch):
    events = _fake_torch(monkeypatch)
    compiles = {"n": 0}
    score = _Doubling()

    def forward(r_noisy, times, **conditioning):  # noqa: ANN001, ANN201
        if score.calls < 1:
            compiles["n"] += 1
        return score.forward(r_noisy, times, **conditioning)

    graph = score_graph.ScoreGraph(forward, compile_count=lambda: compiles["n"])
    graph.start_run()
    for _ in range(2):
        graph(_Tensor(True, 1), _Tensor(True), multiplicity=5)
    assert "capture" not in events
    assert graph(_Tensor(True, 3), _Tensor(True), multiplicity=5).value == 6
    assert events.count("capture") == 1


def test_without_a_compile_count_the_graph_captures_at_the_second_call(monkeypatch):
    events = _fake_torch(monkeypatch)
    graph = score_graph.ScoreGraph(_Doubling().forward)
    graph.start_run()
    for _ in range(3):
        graph(_Tensor(True, 1), _Tensor(True), multiplicity=5)
    assert events.count("capture") == 1


def _graph_with_compiling_calls(monkeypatch, compiling):
    """A graph whose forward compiles on the calls (1-based) in compiling; returns the graph, events, queries."""
    events = _fake_torch(monkeypatch)
    state = {"calls": 0, "n": 0, "queries": 0}
    score = _Doubling()

    def forward(r_noisy, times, **conditioning):  # noqa: ANN001, ANN201
        state["calls"] += 1
        if state["calls"] in compiling:
            state["n"] += 1
        return score.forward(r_noisy, times, **conditioning)

    def count():
        state["queries"] += 1
        return state["n"]

    graph = score_graph.ScoreGraph(forward, compile_count=count)
    graph.start_run()
    return graph, events, state


def _run_calls(graph, calls):
    for _ in range(calls):
        graph(_Tensor(True, 1), _Tensor(True), multiplicity=5)


def test_a_run_that_keeps_compiling_stays_eager_after_the_cap_and_warns_once(monkeypatch, caplog):
    graph, events, state = _graph_with_compiling_calls(monkeypatch, set(range(1, 9)))
    with caplog.at_level("WARNING", logger="score_graph"):
        _run_calls(graph, 8)
    assert "capture" not in events
    assert state["calls"] == 8
    assert state["queries"] == 2 * score_graph.MAX_COMPILING_CALLS
    assert caplog.text.count("stays eager") == 1
    assert f"after {score_graph.MAX_COMPILING_CALLS} eager calls" in caplog.text


def test_a_run_that_stops_compiling_after_the_cap_stays_eager(monkeypatch):
    graph, events, _ = _graph_with_compiling_calls(monkeypatch, {1, 2, 3})
    _run_calls(graph, 6)
    assert "capture" not in events


def test_a_run_that_stops_compiling_at_the_cap_captures_at_the_next_call(monkeypatch):
    graph, events, _ = _graph_with_compiling_calls(monkeypatch, {1, 2})
    _run_calls(graph, 3)
    assert "capture" not in events
    _run_calls(graph, 1)
    assert events.count("capture") == 1
