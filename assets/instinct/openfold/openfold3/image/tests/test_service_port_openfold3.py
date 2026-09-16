# Copyright © Advanced Micro Devices, Inc., or its affiliates.
#
# SPDX-License-Identifier: MIT

"""Guards on how the OpenFold3 BentoML service resolves its listening port.

aim-runtime forwards the resolved ``AIM_PORT`` as ``bentoml serve --arg port=…``
(``BentomlEngine`` is ``ARGS_FORMAT = FORWARDED``), and the async CI path assigns
each workload its own port in 18000-29999. The service therefore has to consume
that argument; binding a fixed port breaks validation and collides on
host-network nodes.

Assertions are made against the parsed AST rather than a live import: the asset's
``service.py`` pulls in bentoml, torch and openfold3, none of which are installed
in this test environment.
"""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

OF3_IMAGE_DIR = Path(__file__).resolve().parents[1]
OF3_SERVICE_PATH = OF3_IMAGE_DIR / "src/service.py"
OF3_DOCKERFILE_PATH = OF3_IMAGE_DIR / "Dockerfile"

SERVICE_CLASS = "OpenFold3Prediction"
ARGS_MODEL = "BentoArgs"


@pytest.fixture
def service_tree() -> ast.Module:
    """Parse ../src/service.py without importing it."""
    return ast.parse(OF3_SERVICE_PATH.read_text(encoding="utf-8"))


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
        for line in OF3_DOCKERFILE_PATH.read_text(encoding="utf-8").splitlines()
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
