# Copyright © Advanced Micro Devices, Inc., or its affiliates.
#
# SPDX-License-Identifier: MIT

"""AST guards on the HTTP surface of the SemlaFlow service.

Never imports service.py (bentoml, torch and timm are not installed here).
"""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

IMAGE_DIR = Path(__file__).resolve().parents[1]
SERVICE_SRC = (IMAGE_DIR / "src/service.py").read_text(encoding="utf-8")
HARNESS_SRC = (IMAGE_DIR / "src/harness.py").read_text(encoding="utf-8")
SERVICE_CLASS = "SemlaFlowService"


@pytest.fixture
def tree() -> ast.Module:
    return ast.parse(SERVICE_SRC)


def _func(tree: ast.AST, name: str) -> ast.AsyncFunctionDef | ast.FunctionDef:
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name == name:
            return node
    raise AssertionError(f"{name} not found")


def _service_class(tree: ast.Module) -> ast.ClassDef:
    return next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == SERVICE_CLASS)


def _starlette_routes(tree: ast.Module) -> dict[str, tuple[str, list[str]]]:
    """Map path -> (handler name, methods) for Route(...) entries."""
    routes = {}
    for n in ast.walk(tree):
        if isinstance(n, ast.Call) and isinstance(n.func, ast.Name) and n.func.id == "Route":
            path = n.args[0].value
            handler = n.args[1].id
            methods = next(
                [c.value for c in kw.value.elts] for kw in n.keywords if kw.arg == "methods"  # type: ignore[attr-defined]
            )
            routes[path] = (handler, methods)
    return routes


def _api_routes(tree: ast.Module) -> list[str]:
    routes = []
    for node in _service_class(tree).body:
        if isinstance(node, ast.FunctionDef):
            for d in node.decorator_list:
                if isinstance(d, ast.Call) and getattr(d.func, "attr", "") == "api":
                    routes += [kw.value.value for kw in d.keywords if kw.arg == "route"]
    return routes


def test_inference_is_the_only_model_endpoint(tree: ast.Module) -> None:
    assert _api_routes(tree) == ["/v1/inference"]


def test_predict_endpoint_is_gone(tree: ast.Module) -> None:
    assert "/predict" not in SERVICE_SRC
    assert "/predict" not in HARNESS_SRC
    methods = {n.name for n in _service_class(tree).body if isinstance(n, ast.FunctionDef)}
    assert "predict" not in methods


def test_get_routes_are_mounted_via_asgi_app_at_root(tree: ast.Module) -> None:
    mounts = [
        d
        for d in _service_class(tree).decorator_list
        if isinstance(d, ast.Call) and getattr(d.func, "attr", "") == "asgi_app"
    ]
    assert len(mounts) == 1
    assert isinstance(mounts[0].args[0], ast.Name) and mounts[0].args[0].id == "canonical_app"
    assert [kw.value.value for kw in mounts[0].keywords if kw.arg == "path"] == ["/"]


def test_canonical_routes_are_get_health_and_models(tree: ast.Module) -> None:
    assert _starlette_routes(tree) == {
        "/health": ("health", ["GET"]),
        "/v1/models": ("list_models", ["GET"]),
    }


def test_health_forwards_to_healthz_on_the_bound_port(tree: ast.Module) -> None:
    health = _func(tree, "health")
    dump = ast.dump(health)
    assert "/healthz" in dump
    assert "PORT" in dump and "BENTOML_PORT" not in SERVICE_SRC and "8000" not in SERVICE_SRC
    # PORT is taken from the forwarded --arg port, not from the environment.
    assign = next(
        n for n in tree.body if isinstance(n, ast.Assign) and any(getattr(t, "id", "") == "PORT" for t in n.targets)
    )
    assert isinstance(assign.value, ast.Attribute) and assign.value.attr == "port"
    # 503 when the upstream is unreachable.
    assert any(isinstance(n, ast.Constant) and n.value == 503 for n in ast.walk(health))


def test_models_response_shape(tree: ast.Module) -> None:
    models = _func(tree, "list_models")
    payload = next(n for n in ast.walk(models) if isinstance(n, ast.Dict))
    keys = [k.value for k in payload.keys]
    assert keys == ["object", "data"]
    assert payload.values[0].value == "list"
    entry = payload.values[1].elts[0]
    entry_keys = [k.value for k in entry.keys]
    assert entry_keys[:2] == ["id", "object"]
    assert isinstance(entry.values[0], ast.Name) and entry.values[0].id == "MODEL_ID"
    assert entry.values[1].value == "model"
    assert 'MODEL_ID = "rssrwn/semlaflow"' in SERVICE_SRC
    assert "SERVED_DTYPE" not in SERVICE_SRC  # no invented dtype
    assert [k.value for k in entry.values[2].keys] == ["accelerator"]


def test_inference_body_shape_unchanged(tree: ast.Module) -> None:
    infer = _func(tree, "inference")
    assert [a.arg for a in infer.args.args] == [
        "self",
        "data_path",
        "n_molecules",
        "integration_steps",
        "dataset_split",
        "seed",  # optional, additive: omitted means a fresh random seed
    ]


def test_builtin_health_routes_are_not_shadowed(tree: ast.Module) -> None:
    paths = set(_starlette_routes(tree)) | set(_api_routes(tree))
    assert not paths & {"/healthz", "/livez", "/readyz", "/metrics"}


def test_port_still_required_and_bound(tree: ast.Module) -> None:
    model = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "BentoArgs")
    port = next(n for n in model.body if isinstance(n, ast.AnnAssign) and n.target.id == "port")
    assert port.value is None


def test_harness_uses_new_endpoints() -> None:
    assert 'HEALTH_ENDPOINT = "/health"' in HARNESS_SRC
    assert 'INFERENCE_ENDPOINT = "/v1/inference"' in HARNESS_SRC
    assert "/predict" not in HARNESS_SRC


def test_harness_probes_the_file_the_service_reads() -> None:
    """The service reads <dataset_split>.smol (default split: test); there is no data.smol."""
    harness = ast.parse(HARNESS_SRC)
    resolve = next(n for n in ast.walk(harness) if isinstance(n, ast.FunctionDef) and n.name == "_resolve_data_path")
    returned = [n for n in ast.walk(resolve) if isinstance(n, ast.Return)]
    assert len(returned) == 1
    constants = [c.value for c in ast.walk(returned[0]) if isinstance(c, ast.Constant)]
    assert constants == ["test.smol"]
    assert "data.smol" not in HARNESS_SRC
    service = ast.parse(SERVICE_SRC)
    assert any(
        isinstance(n, ast.Constant) and n.value == "test" for n in ast.walk(service)
    ), "the service default dataset_split must stay 'test' for the harness probe to be right"
