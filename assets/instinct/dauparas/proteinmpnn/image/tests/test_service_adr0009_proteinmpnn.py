# Copyright © Advanced Micro Devices, Inc., or its affiliates.
#
# SPDX-License-Identifier: MIT

"""AST guards on the HTTP surface of the ProteinMPNN service.

Never imports service.py (bentoml and torch are not installed here).
"""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

IMAGE_DIR = Path(__file__).resolve().parents[1]
SERVICE_SRC = (IMAGE_DIR / "src/service.py").read_text(encoding="utf-8")
HARNESS_SRC = (IMAGE_DIR / "src/harness.py").read_text(encoding="utf-8")
SERVICE_CLASS = "ProteinMPNNService"


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
                [c.value for c in kw.value.elts]
                for kw in n.keywords
                if kw.arg == "methods"  # type: ignore[attr-defined]
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
    assert 'MODEL_ID = "dauparas/proteinmpnn"' in SERVICE_SRC
    # Only facts the code knows: an accelerator name (None if unreadable), no invented dtype.
    metadata = entry.values[2]
    assert [k.value for k in metadata.keys] == ["accelerator"]
    assert "dtype" not in ast.dump(models)


def test_inference_body_shape_unchanged(tree: ast.Module) -> None:
    infer = _func(tree, "inference")
    # The whole body stays wrapped under the single pydantic parameter ``request``.
    assert [a.arg for a in infer.args.args] == ["self", "request"]
    assert ast.unparse(infer.args.args[1].annotation) == "PredictRequest"


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


def test_hardening_is_kept(tree: ast.Module) -> None:
    infer = _func(tree, "inference")
    dump = ast.dump(infer)
    assert "ALLOWED_MODEL_NAMES" in dump and "_resolve_pdb_path" in dump
    assert "InvalidArgument" in dump and "_engine" in dump  # 400 here; 500/503 are raised by the engine
    engine_dump = ast.dump(next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "_DesignEngine"))
    for exc in ("ServiceUnavailable", "BentoMLException"):
        assert exc in engine_dump
    init = next(n for n in _service_class(tree).body if isinstance(n, ast.FunctionDef) and n.name == "__init__")
    init_dump = ast.dump(init)
    for step in ("_resolve_device", "_preflight_weights", "warm_up"):
        assert step in init_dump


def test_service_is_resident_not_subprocess_per_request() -> None:
    """Regression guard: the model is loaded and warmed up at start, and no request spawns a process."""
    assert "import subprocess" not in SERVICE_SRC
    assert "Popen" not in SERVICE_SRC
    tree = ast.parse(SERVICE_SRC)
    init = _func(_service_class(tree), "__init__")
    calls = [ast.unparse(c.func) for c in ast.walk(init) if isinstance(c, ast.Call)]
    assert calls.index("_preflight_weights") < calls.index("self._engine.warm_up")
    infer_calls = [
        ast.unparse(c.func) for c in ast.walk(_func(_service_class(tree), "inference")) if isinstance(c, ast.Call)
    ]
    assert "self._engine.design" in infer_calls


def test_startup_failures_use_fail_startup_and_never_raise(tree: ast.Module) -> None:
    init = _func(_service_class(tree), "__init__")
    names = [ast.unparse(c.func) for c in ast.walk(init) if isinstance(c, ast.Call)]
    assert names.count("_fail_startup") == 2
    assert not [n for n in ast.walk(init) if isinstance(n, ast.Raise)]


def test_fail_startup_exits_one_and_logs(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture, capsys
) -> None:
    import logging
    import os
    import sys

    fn = _func(ast.parse(SERVICE_SRC), "_fail_startup")
    fn.returns = None
    mod = ast.Module(body=[fn], type_ignores=[])
    exits: list[int] = []
    monkeypatch.setattr(os, "_exit", exits.append)
    ns = {"logger": logging.getLogger("pmpnn-test"), "os": os, "sys": sys}
    exec(compile(ast.fix_missing_locations(mod), "service.py", "exec"), ns)  # noqa: S102
    with caplog.at_level(logging.CRITICAL):
        ns["_fail_startup"]("weights missing under /opt/x")
    assert exits == [1]
    assert "weights missing under /opt/x" in caplog.text
    assert "weights missing under /opt/x" in capsys.readouterr().err


def test_api_md_describes_worker_restart_not_container_exit() -> None:
    api = " ".join((IMAGE_DIR.parent / "templates/API.md").read_text(encoding="utf-8").split())
    assert "exits with a non-zero status" not in api
    assert "container exit code" not in api
    assert "The container does **not** exit" in api and "about every 10 seconds" in api
    assert "readiness probe" in api


def test_fail_startup_docstring_does_not_claim_container_exit() -> None:
    doc = ast.get_docstring(_func(ast.parse(SERVICE_SRC), "_fail_startup")) or ""
    assert "does NOT exit" in doc and "supervisor" in doc
    assert "supervisor sees the failure" not in SERVICE_SRC


def test_seed_range_rejected_by_model() -> None:
    pydantic = pytest.importorskip("pydantic")
    cls = next(n for n in ast.parse(SERVICE_SRC).body if isinstance(n, ast.ClassDef) and n.name == "PredictRequest")
    ns = {"BaseModel": pydantic.BaseModel, "Field": pydantic.Field}
    exec(
        compile(ast.fix_missing_locations(ast.Module(body=[cls], type_ignores=[])), "service.py", "exec"), ns
    )  # noqa: S102
    model = ns["PredictRequest"]
    for ok in (0, 1, 2**31, 2**32 - 1):
        assert model(pdb_path="/x", seed=ok).seed == ok
    for bad in (-1, -5, 2**32, 2**40):
        with pytest.raises(pydantic.ValidationError):
            model(pdb_path="/x", seed=bad)
