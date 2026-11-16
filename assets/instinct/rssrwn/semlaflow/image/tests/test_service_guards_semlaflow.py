# Copyright © Advanced Micro Devices, Inc., or its affiliates.
#
# SPDX-License-Identifier: MIT

"""AST guards for SemlaFlow service.py safety behaviour (service.py is never imported)."""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

SERVICE_PATH = Path(__file__).resolve().parents[1] / "src/service.py"


@pytest.fixture
def tree() -> ast.Module:
    return ast.parse(SERVICE_PATH.read_text(encoding="utf-8"))


def _func(tree: ast.Module, name: str) -> ast.FunctionDef:
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name == name:
            return node
    raise AssertionError(f"{name} not found")


def _names(node: ast.AST) -> set[str]:
    out = set()
    for n in ast.walk(node):
        if isinstance(n, ast.Name):
            out.add(n.id)
        elif isinstance(n, ast.Attribute):
            out.add(n.attr)
    return out


def test_no_subprocess_per_request(tree: ast.Module) -> None:
    imported = {a.name for n in ast.walk(tree) if isinstance(n, ast.Import) for a in n.names}
    assert "subprocess" not in imported
    assert "run_module" not in SERVICE_PATH.read_text(encoding="utf-8")


def test_timeout_is_env_configurable_and_maps_to_503(tree: ast.Module) -> None:
    src = SERVICE_PATH.read_text(encoding="utf-8")
    assert "SEMLAFLOW_TIMEOUT_SECONDS" in src
    assert "ServiceUnavailable" in src
    assert "GATEWAY_TIMEOUT" not in src  # the old 504 contract is gone: a bounded wait answers 503
    generate = next(
        n
        for c in ast.walk(tree)
        if isinstance(c, ast.ClassDef) and c.name == "_GenerationEngine"
        for n in c.body
        if isinstance(n, ast.FunctionDef) and n.name == "generate"
    )
    assert "timeout" in _names(generate) and "TimeoutError" in _names(generate)


def test_data_path_is_confined_to_input_path(tree: ast.Module) -> None:
    fn = _func(tree, "_resolve_input_dir")
    names = _names(fn)
    assert {"resolve", "is_relative_to", "InvalidArgument"} <= names
    src = SERVICE_PATH.read_text(encoding="utf-8")
    assert "AIM_INPUT_PATH" in src
    inference = _func(tree, "inference")
    assert "_resolve_input_dir" in _names(inference)


def test_failures_use_bentoml_exceptions_and_do_not_echo_output(tree: ast.Module) -> None:
    inference = _func(tree, "inference")
    for n in ast.walk(inference):
        if isinstance(n, ast.Raise) and isinstance(n.exc, ast.Call):
            fn = n.exc.func
            assert not (isinstance(fn, ast.Name) and fn.id == "RuntimeError"), "use bentoml exceptions in inference"
    engine = next(n for n in ast.walk(tree) if isinstance(n, ast.ClassDef) and n.name == "_GenerationEngine")
    for n in ast.walk(engine):
        if isinstance(n, ast.Raise):
            assert not ({"stderr", "stdout", "tail", "combined", "exc_info"} & _names(n)) or n.exc is None


def test_gpu_and_checkpoint_absence_fail_startup(tree: ast.Module) -> None:
    init = next(
        n
        for c in ast.walk(tree)
        if isinstance(c, ast.ClassDef) and c.name == "SemlaFlowService"
        for n in c.body
        if isinstance(n, ast.FunctionDef) and n.name == "__init__"
    )
    assert {"_resolve_device", "is_file", "_fail_startup", "warm_up"} <= _names(init)
    assert "is_available" in _names(_func(tree, "_resolve_device"))
    assert "os._exit(1)" in ast.unparse(_func(tree, "_fail_startup"))


def test_no_gpu_env_vars_and_flat_request_body(tree: ast.Module) -> None:
    src = SERVICE_PATH.read_text(encoding="utf-8")
    assert "AIM_GPU_COUNT" not in src and "AIM_GPU_MODEL" not in src
    args = [a.arg for a in _func(tree, "inference").args.args]
    assert args == ["self", "data_path", "n_molecules", "integration_steps", "dataset_split", "seed"]
    assert (
        _func(tree, "inference").args.defaults[-1].value is None
    )  # seed is optional: omitted means a fresh random seed


def test_cache_default_matches_runtime_and_path_is_absolute(tree: ast.Module) -> None:
    src = SERVICE_PATH.read_text(encoding="utf-8")
    assert '_DEFAULT_CACHE = "/workspace/model-cache"' in src  # aim_runtime.config.DEFAULT_CACHE_PATH
    # Resolved to an absolute path: the subprocess runs with a different cwd.
    assert "resolve" in _names(_func(tree, "_resolve_ckpt_path"))


def test_upstream_fixed_seed_is_overridden_per_request(tree: ast.Module) -> None:
    assert "randbits" in _names(_func(tree, "inference"))
    generate = next(
        n
        for c in ast.walk(tree)
        if isinstance(c, ast.ClassDef) and c.name == "_ResidentGenerator"
        for n in c.body
        if isinstance(n, ast.FunctionDef) and n.name == "generate"
    )
    assert "seed_everything" in _names(generate) and "seed" in _names(generate)
