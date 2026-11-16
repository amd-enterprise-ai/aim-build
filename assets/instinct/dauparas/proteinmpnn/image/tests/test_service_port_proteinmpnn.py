# Copyright © Advanced Micro Devices, Inc., or its affiliates.
#
# SPDX-License-Identifier: MIT

"""Guards on how the ProteinMPNN BentoML service resolves its listening port.

aim-runtime forwards the resolved ``AIM_PORT`` as ``bentoml serve --arg port=…``
(``BentomlEngine`` is ``ARGS_FORMAT = FORWARDED``), and the async CI path assigns
each workload its own port in 18000-29999. The service therefore has to consume
that argument; binding a fixed port breaks validation and collides on
host-network nodes.

Assertions are made against the parsed AST rather than a live import: the asset's
``service.py`` pulls in bentoml and torch, none of which are installed
in this test environment.
"""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

PMPNN_IMAGE_DIR = Path(__file__).resolve().parents[1]
PMPNN_SERVICE_PATH = PMPNN_IMAGE_DIR / "src/service.py"
PMPNN_DOCKERFILE_PATH = PMPNN_IMAGE_DIR / "Dockerfile"

SERVICE_CLASS = "ProteinMPNNService"
ARGS_MODEL = "BentoArgs"


@pytest.fixture
def service_tree() -> ast.Module:
    """Parse ../src/service.py without importing it."""
    return ast.parse(PMPNN_SERVICE_PATH.read_text(encoding="utf-8"))


def _service_decorator(tree: ast.Module) -> ast.Call:
    """Return the ``@bentoml.service(...)`` call decorating the service class."""
    for node in tree.body:
        if isinstance(node, ast.ClassDef) and node.name == SERVICE_CLASS:
            for decorator in node.decorator_list:
                if (
                    isinstance(decorator, ast.Call)
                    and isinstance(decorator.func, ast.Attribute)
                    and decorator.func.attr == "service"
                ):
                    return decorator
    raise AssertionError(f"no @bentoml.service(...) decorator found on {SERVICE_CLASS}")


def test_dockerfile_does_not_pin_bentoml_port() -> None:
    """BENTOML_PORT must not be set in the image.

    It is a click envvar on ``bentoml serve --port``, which the CLI applies after
    the service config, so it silently overrides ``http={"port": ...}`` and makes
    consuming the forwarded argument a no-op.
    """
    offenders = [
        line.strip()
        for line in PMPNN_DOCKERFILE_PATH.read_text(encoding="utf-8").splitlines()
        if line.strip().startswith("ENV ") and "BENTOML_PORT" in line
    ]
    assert not offenders, f"Dockerfile must not set BENTOML_PORT: {offenders}"


def test_port_argument_has_no_local_default(service_tree: ast.Module) -> None:
    """``BentoArgs.port`` is required, so the forwarded value is the only source.

    A default here would be a second source of truth that can drift from
    ``AIM_PORT`` -- the drift that made the service ignore the forwarded port.
    """
    model = next(
        (n for n in service_tree.body if isinstance(n, ast.ClassDef) and n.name == ARGS_MODEL),
        None,
    )
    assert model is not None, f"{ARGS_MODEL} model is missing from service.py"

    port_field = next(
        (
            n
            for n in model.body
            if isinstance(n, ast.AnnAssign) and isinstance(n.target, ast.Name) and n.target.id == "port"
        ),
        None,
    )
    assert port_field is not None, f"{ARGS_MODEL} must declare a 'port' field"
    assert port_field.value is None, "BentoArgs.port must not have a default; the forwarded --arg is the only source"


def test_forwarded_port_argument_is_consumed(service_tree: ast.Module) -> None:
    """``bentoml.use_arguments(BentoArgs)`` is called at module scope."""
    calls = [
        node
        for node in ast.walk(service_tree)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute) and node.func.attr == "use_arguments"
    ]
    assert calls, "service.py must call bentoml.use_arguments to read the forwarded --arg port"

    consumed = [c for c in calls if any(isinstance(a, ast.Name) and a.id == ARGS_MODEL for a in c.args)]
    assert consumed, f"use_arguments must be given the {ARGS_MODEL} model"


def test_service_binds_the_consumed_port(service_tree: ast.Module) -> None:
    """The decorator wires the parsed argument into the bound port.

    ``--arg`` is generic template data that BentoML does not act on by itself, so
    without ``http={"port": ...}`` the forwarded value is parsed and discarded and
    the service falls back to BentoML's own default.
    """
    decorator = _service_decorator(service_tree)

    http_kwarg = next((kw.value for kw in decorator.keywords if kw.arg == "http"), None)
    assert http_kwarg is not None, "@bentoml.service must set http={'port': ...} to bind the forwarded port"
    assert isinstance(http_kwarg, ast.Dict), "http= must be a dict literal"

    port_keys = [k for k in http_kwarg.keys if isinstance(k, ast.Constant) and k.value == "port"]
    assert port_keys, "http= must carry a 'port' entry"

    port_value = http_kwarg.values[http_kwarg.keys.index(port_keys[0])]
    assert (
        isinstance(port_value, ast.Attribute) and port_value.attr == "port"
    ), "http['port'] must come from the use_arguments result, not a literal"


def test_traffic_timeout_exceeds_request_timeout(service_tree: ast.Module) -> None:
    """BentoML's default 60 s traffic timeout must not fire before the request limit."""
    request_timeout = None
    for node in service_tree.body:
        if (
            isinstance(node, ast.Assign)
            and any(isinstance(t, ast.Name) and t.id == "REQUEST_TIMEOUT_SECONDS" for t in node.targets)
            and isinstance(node.value, ast.Constant)
        ):
            request_timeout = node.value.value
    assert isinstance(request_timeout, int)

    traffic = next((kw.value for kw in _service_decorator(service_tree).keywords if kw.arg == "traffic"), None)
    assert isinstance(traffic, ast.Dict), "@bentoml.service must set traffic={...}"
    entries = {k.value: v for k, v in zip(traffic.keys, traffic.values, strict=True) if isinstance(k, ast.Constant)}
    timeout = entries["timeout"]
    assert isinstance(timeout, ast.BinOp) and isinstance(timeout.op, ast.Add)
    assert isinstance(timeout.left, ast.Name) and timeout.left.id == "REQUEST_TIMEOUT_SECONDS"
    assert isinstance(timeout.right, ast.Constant) and timeout.right.value > 0


def _args_field(tree: ast.Module, name: str) -> ast.AnnAssign:
    model = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == ARGS_MODEL)
    return next(
        n for n in model.body if isinstance(n, ast.AnnAssign) and isinstance(n.target, ast.Name) and n.target.id == name
    )


def test_accelerator_count_argument_defaults_to_one_gpu(service_tree: ast.Module) -> None:
    """aim-runtime forwards ``--arg accelerator_count=<n>`` when the model declares it."""
    assert _args_field(service_tree, "accelerator_count").value is not None


def _is_args_count(node: ast.expr) -> bool:
    return (
        isinstance(node, ast.Attribute)
        and node.attr == "accelerator_count"
        and isinstance(node.value, ast.Name)
        and node.value.id == "args"
    )


def test_resources_and_workers_follow_accelerator_count(service_tree: ast.Module) -> None:
    """One GPU and one data-parallel worker per accelerator, never a literal."""
    decorator = _service_decorator(service_tree)
    kwargs = {kw.arg: kw.value for kw in decorator.keywords}
    resources = kwargs["resources"]
    assert isinstance(resources, ast.Dict)
    gpu = resources.values[[k.value for k in resources.keys].index("gpu")]
    assert _is_args_count(gpu), "resources['gpu'] must come from args.accelerator_count"
    assert _is_args_count(kwargs["workers"]), "workers must come from args.accelerator_count"


def _select_worker_device():
    node = next(
        n
        for n in ast.parse(PMPNN_SERVICE_PATH.read_text(encoding="utf-8")).body
        if isinstance(n, ast.FunctionDef) and n.name == "_select_worker_device"
    )
    err = next(
        n
        for n in ast.parse(PMPNN_SERVICE_PATH.read_text(encoding="utf-8")).body
        if isinstance(n, ast.FunctionDef) and n.name == "_accelerator_count_error"
    )
    ns: dict = {}
    exec(compile(ast.Module([err, node], []), "service.py", "exec"), ns)
    return ns["_select_worker_device"]


def test_two_accelerators_parse_and_each_worker_gets_its_own_gpu() -> None:
    from pydantic import BaseModel, Field

    model_node = next(
        n
        for n in ast.parse(PMPNN_SERVICE_PATH.read_text(encoding="utf-8")).body
        if isinstance(n, ast.ClassDef) and n.name == ARGS_MODEL
    )
    ns: dict = {"BaseModel": BaseModel, "Field": Field}
    exec(compile(ast.Module([model_node], []), "service.py", "exec"), ns)
    args_model = ns[ARGS_MODEL]
    assert args_model(port=3000).accelerator_count == 1
    assert args_model(port=3000, accelerator_count=2).accelerator_count == 2
    with pytest.raises(ValueError):
        args_model(port=3000, accelerator_count=0)

    select = _select_worker_device()
    # Unrestricted: worker k takes GPU k-1 (BentoML worker indices are 1-based).
    assert [select(None, w, 2) for w in (1, 2)] == ["0", "1"]
    # A restricted list is honoured by position.
    assert [select(["4", "5"], w, 2) for w in (1, 2)] == ["4", "5"]
    # One accelerator needs no narrowing.
    assert select(None, 1, 1) is None
    assert select(["3"], 1, 1) is None
    # Fewer visible devices than workers fails at startup with the accelerator-count message, never a shared GPU.
    with pytest.raises(RuntimeError, match=r"AIM_ACCELERATOR_COUNT=2 requires 2 accelerator\(s\), but .*expose 1"):
        select(["3"], 2, 2)
    # Fewer distinct GPUs than workers is a startup error, never a shared GPU.
    with pytest.raises(RuntimeError, match="repeat device ids"):
        select(["1", "1"], 2, 2)
