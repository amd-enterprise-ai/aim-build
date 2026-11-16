# Copyright © Advanced Micro Devices, Inc., or its affiliates.
#
# SPDX-License-Identifier: MIT

"""Sampling device of the REINVENT4 service: the resident model runs on ``REINVENT4_DEVICE``, default cuda:0.

service.py needs bentoml/torch, so the helpers are extracted from the AST and run
against a stub torch; no GPU is needed.
"""

from __future__ import annotations

import ast
import logging
import os
import re
from pathlib import Path
from types import SimpleNamespace

import pytest

SRC = (Path(__file__).resolve().parents[1] / "src/service.py").read_text(encoding="utf-8")
CONSTS = {"DEVICE_ENV", "DEFAULT_DEVICE", "_DEVICE_RE"}
SET_CALLS: list[int] = []  # torch.cuda.set_device calls made by the extracted helpers


def _ns(cuda_devices: int):
    body = [
        n
        for n in ast.parse(SRC).body
        if (
            isinstance(n, ast.FunctionDef) and n.name in {"_resolve_device", "_worker_device_index", "_activate_device"}
        )
        or (isinstance(n, ast.Assign) and getattr(n.targets[0], "id", "") in CONSTS)
    ]
    cuda = SimpleNamespace(
        is_available=lambda: cuda_devices > 0,
        device_count=lambda: cuda_devices,
        set_device=lambda k: SET_CALLS.append(k),
    )
    ns = {"os": os, "re": re, "torch": SimpleNamespace(cuda=cuda), "logger": logging.getLogger("t")}
    exec(compile(ast.Module(body=body, type_ignores=[]), "service_device", "exec"), ns)  # noqa: S102
    return ns


@pytest.fixture(autouse=True)
def _clear_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("REINVENT4_DEVICE", raising=False)
    SET_CALLS.clear()


def test_default_is_cuda0_when_a_gpu_is_visible() -> None:
    assert _ns(1)["_resolve_device"]() == "cuda:0"


def test_no_gpu_fails_by_default() -> None:
    with pytest.raises(RuntimeError, match="No GPU found"):
        _ns(0)["_resolve_device"]()


@pytest.mark.parametrize(("value", "expected"), [("cuda", "cuda:0"), ("CUDA:1", "cuda:1"), (" cuda:0 ", "cuda:0")])
def test_override_selects_gpu(monkeypatch: pytest.MonkeyPatch, value: str, expected: str) -> None:
    monkeypatch.setenv("REINVENT4_DEVICE", value)
    assert _ns(2)["_resolve_device"]() == expected


def test_gpu_index_beyond_visible_gpus_fails(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("REINVENT4_DEVICE", "cuda:1")
    with pytest.raises(RuntimeError, match="only 1 GPU"):
        _ns(1)["_resolve_device"]()


def test_cpu_only_when_asked_and_logged(monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture) -> None:
    monkeypatch.setenv("REINVENT4_DEVICE", "cpu")
    with caplog.at_level(logging.WARNING, logger="t"):
        assert _ns(0)["_resolve_device"]() == "cpu"
    assert "CPU" in caplog.text


@pytest.mark.parametrize("value", ["gpu", "cuda:", "cuda:x", "mps", "cuda:0\nrun_type", 'cpu"'])
def test_unknown_value_is_an_error(monkeypatch: pytest.MonkeyPatch, value: str) -> None:
    monkeypatch.setenv("REINVENT4_DEVICE", value)
    with pytest.raises(RuntimeError, match="not valid"):
        _ns(1)["_resolve_device"]()


def test_service_resolves_the_device_once_and_hands_it_to_the_resident_sampler() -> None:
    tree = ast.parse(SRC)
    init = next(
        n
        for c in tree.body
        if isinstance(c, ast.ClassDef) and c.name == "REINVENT4Service"
        for n in c.body
        if isinstance(n, ast.FunctionDef) and n.name == "__init__"
    )
    text = ast.unparse(init)
    assert "device = _resolve_device(bentoml.server_context.worker_index or 1, args.accelerator_count)" in text
    assert "_activate_device(device)" in text
    assert "_ResidentSampler(device, self._validation_pool)" in text


def test_each_worker_defaults_to_its_own_gpu() -> None:
    ns = _ns(2)
    assert [ns["_resolve_device"](w, 2) for w in (1, 2)] == ["cuda:0", "cuda:1"]


def test_explicit_device_override_wins_over_the_worker_default(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("REINVENT4_DEVICE", "cuda:1")
    assert _ns(2)["_resolve_device"](1) == "cuda:1"


def test_fewer_gpus_than_accelerator_count_fails_at_startup() -> None:
    with pytest.raises(
        RuntimeError, match=r"AIM_ACCELERATOR_COUNT=2 requires 2 accelerator\(s\), but this container has 1"
    ):
        _ns(1)["_resolve_device"](2, 2)


def test_explicit_device_override_skips_the_accelerator_count_check(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("REINVENT4_DEVICE", "cuda:0")
    assert _ns(1)["_resolve_device"](2, 2) == "cuda:0"


def test_activate_device_sets_the_workers_current_gpu() -> None:
    ns = _ns(2)
    ns["_activate_device"]("cuda:1")
    assert SET_CALLS == [1]


def test_activate_device_leaves_cpu_and_bare_cuda_alone() -> None:
    ns = _ns(2)
    ns["_activate_device"]("cpu")
    ns["_activate_device"]("cuda")
    assert SET_CALLS == []
