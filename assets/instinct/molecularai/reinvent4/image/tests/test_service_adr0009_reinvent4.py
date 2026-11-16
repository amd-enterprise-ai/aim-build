# Copyright © Advanced Micro Devices, Inc., or its affiliates.
#
# SPDX-License-Identifier: MIT

"""AST guards on the HTTP surface of the REINVENT4 service.

Never imports service.py (bentoml, torch and reinvent4 are not installed here).
"""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

IMAGE_DIR = Path(__file__).resolve().parents[1]
SERVICE_SRC = (IMAGE_DIR / "src/service.py").read_text(encoding="utf-8")
HARNESS_SRC = (IMAGE_DIR / "src/harness.py").read_text(encoding="utf-8")
SERVICE_CLASS = "REINVENT4Service"


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
    assert [k.value for k in payload.keys] == ["object", "data"]
    assert payload.values[0].value == "list"
    entry = payload.values[1].elts[0]
    assert [k.value for k in entry.keys] == ["id", "object", "metadata"]
    assert isinstance(entry.values[0], ast.Name) and entry.values[0].id == "MODEL_ID"
    assert entry.values[1].value == "model"
    # Only facts the code knows: no invented dtype.
    assert [k.value for k in entry.values[2].keys] == ["accelerator", "default_prior", "priors"]
    assert 'MODEL_ID = "molecularai/reinvent4"' in SERVICE_SRC


def test_inference_body_shape_unchanged(tree: ast.Module) -> None:
    infer = _func(tree, "inference")
    assert [a.arg for a in infer.args.args] == ["self", "data"]
    assert ast.unparse(infer.args.args[1].annotation) == "SamplingRequest"
    assert ast.unparse(infer.returns) == "SamplingResponse"


def test_builtin_health_routes_are_not_shadowed(tree: ast.Module) -> None:
    paths = set(_starlette_routes(tree)) | set(_api_routes(tree))
    assert not paths & {"/healthz", "/livez", "/readyz", "/metrics"}


def test_port_still_required_and_bound(tree: ast.Module) -> None:
    model = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "BentoArgs")
    port = next(n for n in model.body if isinstance(n, ast.AnnAssign) and n.target.id == "port")
    assert port.value is None


def test_harness_uses_new_endpoints() -> None:
    assert '("/health", "/healthz")' in HARNESS_SRC
    assert 'f"{service_url}/v1/inference"' in HARNESS_SRC
    assert "_post_predict" not in HARNESS_SRC


def test_harness_wraps_payload_under_data() -> None:
    harness = ast.parse(HARNESS_SRC)
    post = _func(harness, "_post_inference")
    dumps = next(n for n in ast.walk(post) if isinstance(n, ast.Call) and ast.unparse(n.func) == "json.dumps")
    body = dumps.args[0]
    assert isinstance(body, ast.Dict)
    assert [k.value for k in body.keys] == ["data"]
    inner = body.values[0]
    assert isinstance(inner, ast.Dict) and [k.value for k in inner.keys] == ["num_smiles"]
    # every request goes through the wrapped helper; no other body is built
    assert HARNESS_SRC.count("json.dumps(") == 1


def test_startup_fails_fast_on_missing_gpu_or_prior(tree: ast.Module) -> None:
    init = next(n for n in _service_class(tree).body if isinstance(n, ast.FunctionDef) and n.name == "__init__")
    dump = ast.dump(init)
    assert "_resolve_device" in dump
    assert "_preflight_priors" in dump and "RuntimeError" in dump
    # the constructor must end the process, not just raise
    assert "_fail_startup" in dump
    # the model is warmed up before the service reports ready
    assert "warm_up" in dump
    # per-request behaviour kept: a prior that cannot be provided is an error, not a crash
    infer = ast.dump(_func(tree, "inference"))
    assert "_resolve_prior" in infer and "RuntimeError" in infer


def test_fail_startup_ends_worker_with_message(tree: ast.Module) -> None:
    fn = _func(tree, "_fail_startup")
    calls = [n for n in ast.walk(fn) if isinstance(n, ast.Call)]
    exit_calls = [c for c in calls if ast.unparse(c.func) in ("os._exit", "sys.exit")]
    assert exit_calls, "_fail_startup must terminate the process"
    code = exit_calls[0].args[0]
    assert isinstance(code, ast.Constant) and code.value not in (0, None)
    # the message is logged before exiting
    assert any(ast.unparse(c.func) == "logger.critical" for c in calls)
    assert [ast.unparse(c.func) for c in calls].index("logger.critical") < [ast.unparse(c.func) for c in calls].index(
        ast.unparse(exit_calls[0].func)
    )


def test_service_comment_and_docs_state_real_startup_failure_behaviour() -> None:
    api = (IMAGE_DIR.parent / "templates/API.md").read_text(encoding="utf-8")
    flat = " ".join(api.split())
    # the container does not exit; docs must not promise it
    assert "exits with a non-zero status" not in flat and "container process exits" not in flat
    assert "does not exit" in flat and "about every 10 seconds" in flat and "readiness probe" in flat
    assert "stops startup" not in flat and "startup stops" not in flat
    svc = " ".join(SERVICE_SRC.split())
    assert "exits the process with a non-zero status" not in svc
    assert "container does NOT exit" in svc and "container-level decision" in svc


def test_no_aim_gpu_vars() -> None:
    assert "AIM_GPU_" not in SERVICE_SRC and "AIM_GPU_" not in HARNESS_SRC


def test_api_documents_400_shapes_generic_bodies_and_private_prior() -> None:
    flat = " ".join((IMAGE_DIR.parent / "templates/API.md").read_text(encoding="utf-8").split())
    assert "two shapes" in flat and '[{"error": "Unknown prior' in flat and "1 validation error for Input" in flat
    assert "BentoML's generic text" in flat and "only in the server log" in flat
    assert "0600" in flat and "inode" in flat and "INFO level" in flat
