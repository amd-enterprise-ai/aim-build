# Copyright © Advanced Micro Devices, Inc., or its affiliates.
#
# SPDX-License-Identifier: MIT

"""AST guards on the HTTP surface of the SaProt 650M AF2 service.

Never imports service.py (bentoml, torch and timm are not installed here).
"""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

IMAGE_DIR = Path(__file__).resolve().parents[1]
SERVICE_SRC = (IMAGE_DIR / "src/service.py").read_text(encoding="utf-8")
HARNESS_SRC = (IMAGE_DIR / "src/harness.py").read_text(encoding="utf-8")
SERVICE_CLASS = "SaProtEmbeddingService"


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
    assert isinstance(entry.values[0], ast.Name) and entry.values[0].id == "AIM_ID"
    assert entry.values[1].value == "model"
    assert 'AIM_ID = "westlake-repl/saprot-650m-af2"' in SERVICE_SRC
    assert 'SERVED_DTYPE = "fp16"' in SERVICE_SRC


def test_inference_body_shape_unchanged(tree: ast.Module) -> None:
    infer = _func(tree, "inference")
    assert [a.arg for a in infer.args.args] == ["self", "request"]
    assert ast.unparse(infer.args.args[1].annotation) == "EmbedRequest"


def test_builtin_health_routes_are_not_shadowed(tree: ast.Module) -> None:
    paths = set(_starlette_routes(tree)) | set(_api_routes(tree))
    assert not paths & {"/healthz", "/livez", "/readyz", "/metrics"}


def test_port_still_required_and_bound(tree: ast.Module) -> None:
    model = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "BentoArgs")
    port = next(n for n in model.body if isinstance(n, ast.AnnAssign) and n.target.id == "port")
    assert port.value is None


def test_models_accelerator_is_torch_device_name_or_none(tree: ast.Module) -> None:
    assert "get_device_name" in ast.dump(_func(tree, "_accelerator_name"))


def test_harness_reads_dtype_from_v1_models() -> None:
    harness = ast.parse(HARNESS_SRC)
    fn = _func(harness, "_query_model_dtype")
    dump = ast.dump(fn)
    assert "MODELS_ENDPOINT" in dump and "/info" not in dump and "/metadata" not in dump
    assert 'MODELS_ENDPOINT = "/v1/models"' in HARNESS_SRC


def _load_query_dtype():
    """Exec only the harness pieces that _query_model_dtype needs (harness.py is never imported)."""
    import json
    import logging
    from urllib.error import HTTPError, URLError
    from urllib.request import Request

    harness = ast.parse(HARNESS_SRC)
    wanted = {"MODELS_ENDPOINT", "_DTYPE_ALIASES"}
    body = [n for n in harness.body if isinstance(n, ast.Assign) and getattr(n.targets[0], "id", "") in wanted]
    cls = next(n for n in harness.body if isinstance(n, ast.ClassDef) and n.name == "SaProtHarness")
    fn = _func(cls, "_query_model_dtype")
    fn.decorator_list = []
    mod = ast.Module(body=[*body, fn], type_ignores=[])
    ns = {
        "json": json,
        "logger": logging.getLogger("t"),
        "Request": Request,
        "HTTPError": HTTPError,
        "URLError": URLError,
    }
    return ns, compile(ast.fix_missing_locations(mod), "harness_dtype", "exec")


def _run_query(payload: bytes | None, status: int = 200):
    import io

    ns, code = _load_query_dtype()

    class _Resp(io.BytesIO):
        def __init__(self, b: bytes) -> None:
            super().__init__(b)
            self.status = status

        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

    def fake_urlopen(req, timeout=0):
        assert req.full_url.endswith("/v1/models")
        if payload is None:
            raise OSError("down")
        return _Resp(payload)

    ns["urlopen"] = fake_urlopen
    exec(code, ns)
    return ns["_query_model_dtype"]("http://x")


def test_query_model_dtype_normalises_fp16() -> None:
    body = b'{"object":"list","data":[{"id":"a","object":"model","metadata":{"dtype":"fp16","accelerator":null}}]}'
    assert _run_query(body) == "float16"


def test_query_model_dtype_unknown_on_bad_shape_or_unreachable() -> None:
    assert _run_query(b'{"object":"list","data":[]}') is None
    assert _run_query(b"not json") is None
    assert _run_query(None) is None


def test_harness_uses_new_endpoints() -> None:
    assert 'HEALTH_ENDPOINT = "/health"' in HARNESS_SRC
    assert 'INFERENCE_ENDPOINT = "/v1/inference"' in HARNESS_SRC
    assert "/predict" not in HARNESS_SRC


def test_harness_nests_the_request_body_under_request() -> None:
    """The endpoint parameter is `request`, so a flat body is a 400 (seen on a real container)."""
    harness = ast.parse(HARNESS_SRC)
    send = next(n for n in ast.walk(harness) if isinstance(n, ast.FunctionDef) and n.name == "_send_embed")
    dumps = [
        n
        for n in ast.walk(send)
        if isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute) and n.func.attr == "dumps"
    ]
    assert len(dumps) == 1
    body = dumps[0].args[0]
    assert isinstance(body, ast.Dict)
    assert [k.value for k in body.keys if isinstance(k, ast.Constant)] == ["request"]
    inner = body.values[0]
    assert isinstance(inner, ast.Dict)
    assert [k.value for k in inner.keys if isinstance(k, ast.Constant)] == ["sequence"]


def _imported_modules(tree: ast.AST) -> set[str]:
    """Every module name imported anywhere in the tree (top-level or lazy, inside functions)."""
    mods: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            mods.update(a.name for a in node.names)
        elif isinstance(node, ast.ImportFrom):
            mods.add(node.module or "")
            mods.update(a.name for a in node.names)
    return mods


def test_service_does_not_import_harness_or_aim_runtime(tree: ast.Module) -> None:
    """The service runs in the layer-1 image, which has no aim_runtime and no harness."""
    for mod in _imported_modules(tree):
        root = mod.split(".")[0]
        assert root not in {"harness", "aim_runtime"}, f"service.py imports {mod!r}"


def test_service_defines_its_own_get_device(tree: ast.Module) -> None:
    fn = next((n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "_get_device"), None)
    assert fn is not None, "service.py must define _get_device at module level"
    assert any(isinstance(n, ast.Raise) for n in ast.walk(fn)), "_get_device must raise without a GPU"
