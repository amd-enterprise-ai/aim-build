# Copyright © Advanced Micro Devices, Inc., or its affiliates.
#
# SPDX-License-Identifier: MIT

"""AST guards on H-optimus-0 service hardening (never imports service.py)."""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

IMAGE_DIR = Path(__file__).resolve().parents[1]
SERVICE_SRC = (IMAGE_DIR / "src/service.py").read_text(encoding="utf-8")
HARNESS_SRC = (IMAGE_DIR / "src/harness.py").read_text(encoding="utf-8")


@pytest.fixture
def tree() -> ast.Module:
    return ast.parse(SERVICE_SRC)


def _func(tree: ast.AST, name: str) -> ast.FunctionDef:
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name == name:
            return node
    raise AssertionError(f"{name} not found")


def _raised_names(func: ast.FunctionDef) -> set[str]:
    names = set()
    for n in ast.walk(func):
        if isinstance(n, ast.Raise) and isinstance(n.exc, ast.Call):
            f = n.exc.func
            names.add(f.id if isinstance(f, ast.Name) else getattr(f, "attr", ""))
    return names


def test_device_index_is_relative_to_visible_devices_not_env_index(tree: ast.Module) -> None:
    func = _func(tree, "_get_device")
    assert "CUDA_VISIBLE_DEVICES" not in ast.dump(func), "device index must not be parsed from CUDA_VISIBLE_DEVICES"
    called = {n.func.id for n in ast.walk(func) if isinstance(n, ast.Call) and isinstance(n.func, ast.Name)}
    assert "_worker_device_index" in called, "the device index must come from the worker index and the visible count"


def test_missing_gpu_fails_loudly(tree: ast.Module) -> None:
    assert "RuntimeError" in _raised_names(_func(tree, "_get_device"))


def test_no_gpu_env_vars() -> None:
    assert "AIM_GPU_COUNT" not in SERVICE_SRC and "AIM_GPU_MODEL" not in SERVICE_SRC


def test_image_path_restricted_to_input_root(tree: ast.Module) -> None:
    validate = _func(tree, "_validate_path_within")
    calls = {n.func.attr for n in ast.walk(validate) if isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute)}
    assert {"resolve", "relative_to", "is_file"} <= calls
    assert "InvalidArgument" in _raised_names(validate)
    decode = _func(tree, "_decode_image")
    assert any(isinstance(n, ast.Name) and n.id == "_validate_path_within" for n in ast.walk(decode))
    assert "AIM_INPUT_PATH" in SERVICE_SRC


def test_weights_cached_under_aim_cache_path(tree: ast.Module) -> None:
    assert "AIM_CACHE_PATH" in SERVICE_SRC
    create = [
        n
        for n in ast.walk(tree)
        if isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute) and n.func.attr == "create_model"
    ]
    assert create and all(any(k.arg == "cache_dir" for k in c.keywords) for c in create)


def test_client_errors_use_bentoml_exceptions_not_valueerror(tree: ast.Module) -> None:
    for name in ("inference", "_decode_image"):
        raised = _raised_names(_func(tree, name))
        assert "InvalidArgument" in raised
        assert "ValueError" not in raised


def test_responses_never_carry_error_flag_or_exception_text(tree: ast.Module) -> None:
    assert '"error"' not in SERVICE_SRC
    decode = _func(tree, "_decode_image")
    for n in ast.walk(decode):
        if isinstance(n, ast.Raise) and isinstance(n.exc, ast.Call):
            assert not any(isinstance(a, ast.JoinedStr) and "exc" in ast.dump(a) for a in n.exc.args)


def test_harness_has_no_hardcoded_port_and_no_empty_except() -> None:
    htree = ast.parse(HARNESS_SRC)
    assert "BENTOML_PORT" not in HARNESS_SRC
    assert "resolve_service_url" in HARNESS_SRC
    for n in ast.walk(htree):
        if isinstance(n, ast.ExceptHandler):
            assert not (len(n.body) == 1 and isinstance(n.body[0], ast.Pass)), "empty except handler"
