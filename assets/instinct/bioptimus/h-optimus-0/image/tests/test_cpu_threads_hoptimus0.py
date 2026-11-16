# Copyright © Advanced Micro Devices, Inc., or its affiliates.
#
# SPDX-License-Identifier: MIT

"""CPU thread limit: default min(8, usable CPUs), HOPTIMUS0_NUM_THREADS override, OMP/MKL defaults.

Never imports service.py (heavy deps); the helpers are extracted from its source
and executed in isolation with a recording stand-in for ``torch``.
"""

from __future__ import annotations

import ast
import logging
import os
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

IMAGE_DIR = Path(__file__).resolve().parents[1]
SERVICE_SRC = (IMAGE_DIR / "src/service.py").read_text(encoding="utf-8")


def _load(usable_cpus: int | None = None) -> tuple[dict[str, Any], list[int]]:
    tree = ast.parse(SERVICE_SRC)
    wanted_funcs = {"_usable_cpus", "_resolve_num_threads", "_default_thread_env", "_limit_threads"}
    wanted_consts = {"THREADS_ENV", "DEFAULT_MAX_THREADS", "NUM_THREADS"}
    nodes: list[ast.stmt] = []
    for n in tree.body:
        if isinstance(n, ast.FunctionDef) and n.name in wanted_funcs:
            nodes.append(n)
        elif isinstance(n, ast.Assign) and any(getattr(t, "id", None) in wanted_consts for t in n.targets):
            nodes.append(n)
    assert len(nodes) == len(wanted_funcs) + len(wanted_consts)  # NUM_THREADS is checked separately below
    calls: list[int] = []
    torch = SimpleNamespace(set_num_threads=calls.append)
    ns: dict[str, Any] = {"os": os, "Any": Any, "torch": torch, "LOGGER": logging.getLogger("t")}
    nodes = [n for n in nodes if not (isinstance(n, ast.Assign) and getattr(n.targets[0], "id", "") == "NUM_THREADS")]
    exec(compile(ast.Module(body=nodes, type_ignores=[]), "service_threads", "exec"), ns)  # noqa: S102
    if usable_cpus is not None:
        ns["_usable_cpus"] = lambda: usable_cpus
    return ns, calls


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch: pytest.MonkeyPatch) -> None:
    for name in ("HOPTIMUS0_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
        monkeypatch.setenv(name, "x")  # registers the restore of the original state
        monkeypatch.delenv(name)


@pytest.mark.parametrize(("cpus", "expected"), [(1, 1), (4, 4), (8, 8), (48, 8), (384, 8)])
def test_default_is_min_8_and_usable_cpus(cpus: int, expected: int) -> None:
    ns, _ = _load(cpus)
    assert ns["_resolve_num_threads"]({}) == expected


def test_usable_cpus_respects_affinity() -> None:
    ns, _ = _load()
    if hasattr(os, "sched_getaffinity"):
        assert ns["_usable_cpus"]() == len(os.sched_getaffinity(0))
    assert ns["_usable_cpus"]() >= 1


@pytest.mark.parametrize(("raw", "expected"), [("1", 1), (" 4 ", 4), ("48", 48)])
def test_env_override(raw: str, expected: int) -> None:
    ns, _ = _load(48)
    assert ns["_resolve_num_threads"]({"HOPTIMUS0_NUM_THREADS": raw}) == expected


@pytest.mark.parametrize("raw", ["0", "-2", "abc", "1.5", "8 threads"])
def test_invalid_override_stops_startup(raw: str) -> None:
    ns, _ = _load(48)
    with pytest.raises(RuntimeError, match="HOPTIMUS0_NUM_THREADS"):
        ns["_resolve_num_threads"]({"HOPTIMUS0_NUM_THREADS": raw})


def test_empty_override_uses_default() -> None:
    ns, _ = _load(48)
    assert ns["_resolve_num_threads"]({"HOPTIMUS0_NUM_THREADS": "  "}) == 8


def test_default_env_sets_omp_and_mkl_when_unset() -> None:
    ns, calls = _load(48)
    assert ns["_default_thread_env"]() == 8
    assert os.environ["OMP_NUM_THREADS"] == "8"
    assert os.environ["MKL_NUM_THREADS"] == "8"
    assert calls == []  # torch is only touched after the model is loaded


def test_default_env_does_not_override_operator_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("OMP_NUM_THREADS", "3")
    monkeypatch.setenv("MKL_NUM_THREADS", "5")
    monkeypatch.setenv("HOPTIMUS0_NUM_THREADS", "4")
    ns, _ = _load(48)
    assert ns["_default_thread_env"]() == 4
    assert os.environ["OMP_NUM_THREADS"] == "3"
    assert os.environ["MKL_NUM_THREADS"] == "5"


def test_limit_threads_calls_torch_with_resolved_count() -> None:
    ns, calls = _load(48)
    ns["NUM_THREADS"] = 6
    assert ns["_limit_threads"]() == 6
    assert calls == [6]


def test_env_defaults_at_import_and_torch_limit_after_model_load() -> None:
    """Env at import (single-threaded); set_num_threads only after the model is on the device.

    Calling torch.set_num_threads before CUDA initialisation crashed the process on some starts.
    """
    tree = ast.parse(SERVICE_SRC)
    top = [
        n
        for n in tree.body
        if isinstance(n, ast.Assign) and "_default_thread_env()" in (ast.get_source_segment(SERVICE_SRC, n) or "")
    ]
    assert len(top) == 1
    cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "HOptimus0Service")
    init = next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == "__init__")
    src = ast.get_source_segment(SERVICE_SRC, init) or ""
    assert "_limit_threads()" in src
    assert src.index("timm.create_model") < src.index("self._model.to(") < src.index("_limit_threads()")
