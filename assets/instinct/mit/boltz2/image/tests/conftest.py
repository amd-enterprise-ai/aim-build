# Copyright © Advanced Micro Devices, Inc., or its affiliates.
#
# SPDX-License-Identifier: MIT

"""Shared fixtures for the boltz2 AIM's CPU unit and GPU smoke tiers."""

from __future__ import annotations

import ast
import importlib.util
import json
import os
import pathlib
import sys
import tempfile
from urllib.error import HTTPError

import pytest

# The authored modules live at <image>/src/ and are imported flat (service.py
# does `from runner import ...`), mirroring the container layout where
# /workspace/model/src is on sys.path. run_cpu_tests.sh already exports this;
# adding it here keeps a bare `pytest app/tests` working too.
_SRC = pathlib.Path(__file__).resolve().parents[1] / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

# service.py declares its bound port as a REQUIRED bento argument with no
# default, so `import service` fails unless the arguments are populated first.
# This runs before any test module imports service. set_arguments is not
# re-exported from the `bentoml` namespace in 1.4.x. Without BentoML installed
# there is no service to import, and the modules that need it are not collected
# (see pytest_ignore_collect).
try:
    import bentoml  # noqa: F401
except ImportError:
    pass
else:
    try:
        from bentoml._internal.utils.args import set_arguments
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError(
            "BentoML no longer exposes set_arguments at bentoml._internal.utils.args; "
            "the test tiers need it to populate service.py's required `port` bento argument."
        ) from exc
    set_arguments({"port": int(os.environ.get("AIM_PORT", "3000"))}, override=False)

# service.py checks the cache when it is imported, and the default cache path does
# not exist outside the container. A writable temporary directory keeps that check quiet.
os.environ.setdefault("AIM_CACHE_PATH", tempfile.mkdtemp(prefix="boltz2-test-cache-"))
# The same import stops when a GPU is requested and none is visible; the unit tests run on the CPU.
os.environ.setdefault("BOLTZ2_ACCELERATOR", "auto")

_LOCAL_MODULES = {p.stem for p in _SRC.glob("*.py")} | {p.name for p in _SRC.iterdir() if p.is_dir()}
_MISSING_FOR: dict[pathlib.Path, list[str]] = {}
_NOT_COLLECTED: dict[str, list[str]] = {}


def _imported_roots(path: pathlib.Path) -> set[str]:
    """Top-level module names a source file imports at module level."""
    roots: set[str] = set()
    for node in ast.parse(path.read_text()).body:
        if isinstance(node, ast.Import):
            roots.update(alias.name.split(".")[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
            roots.add(node.module.split(".")[0])
    roots.discard("__future__")
    return roots


def _missing_imports(path: pathlib.Path) -> list[str]:
    """Imports of ``path``, and of the image modules it imports, that cannot be resolved here."""
    if path in _MISSING_FOR:
        return _MISSING_FOR[path]
    _MISSING_FOR[path] = []  # an import cycle adds nothing new
    missing: set[str] = set()
    for root in _imported_roots(path):
        if root in _LOCAL_MODULES:
            local = _SRC / root
            files = sorted(local.rglob("*.py")) if local.is_dir() else [_SRC / f"{root}.py"]
            for file in files:
                missing.update(_missing_imports(file))
        elif importlib.util.find_spec(root) is None:
            missing.add(root)
    _MISSING_FOR[path] = sorted(missing)
    return _MISSING_FOR[path]


_ENGINE_ROOTS = frozenset({"torch", "boltz", "bentoml", "pytorch_lightning", "triton"})
_ENGINE_FOR: dict[pathlib.Path, bool] = {}


def _needs_engine(path: pathlib.Path) -> bool:
    """Whether ``path``, or an image module it imports, imports the engine (torch, boltz, BentoML)."""
    if path in _ENGINE_FOR:
        return _ENGINE_FOR[path]
    _ENGINE_FOR[path] = False  # an import cycle adds nothing new
    found = False
    for root in _imported_roots(path):
        if root in _ENGINE_ROOTS:
            found = True
        elif root in _LOCAL_MODULES:
            local = _SRC / root
            files = sorted(local.rglob("*.py")) if local.is_dir() else [_SRC / f"{root}.py"]
            found = found or any(_needs_engine(file) for file in files)
    _ENGINE_FOR[path] = found
    return found


def pytest_collection_modifyitems(items) -> None:
    """Mark the tests that need the engine and no GPU with ``engine``.

    The engine is installed in the base image, so unit-tests.yaml runs them there
    with ``-m engine``. fast-feedback.yaml and a plain checkout exclude them.
    """
    for item in items:
        path = pathlib.Path(str(item.path))
        if path.name.startswith("test_") and _needs_engine(path) and item.get_closest_marker("gpu") is None:
            item.add_marker(pytest.mark.engine)


def pytest_ignore_collect(collection_path: pathlib.Path, config):
    """Do not collect a test module whose imports are not installed here.

    The CI lock files carry neither torch, BentoML nor Boltz, so the tests that
    need them run only where the image's engine is installed. A module that
    imports only what is available (the fixture, recipe and harness tests) is
    collected everywhere.
    """
    if collection_path.suffix != ".py" or not collection_path.name.startswith("test_"):
        return None
    missing = _missing_imports(collection_path)
    if missing:
        _NOT_COLLECTED[collection_path.name] = missing
        return True
    return None


def _not_collected_message() -> str:
    names = sorted({m for missing in _NOT_COLLECTED.values() for m in missing})
    return (
        f"boltz2: {len(_NOT_COLLECTED)} test module(s) not collected, not installed here: {', '.join(names)} "
        f"({', '.join(sorted(_NOT_COLLECTED))})"
    )


def _selects_engine_tests(config) -> bool:
    """Whether the marker expression of this run selects the tests marked ``engine``."""
    expression = config.getoption("markexpr", default="") or ""
    if not expression.strip():
        return False
    from _pytest.mark.expression import Expression

    try:
        return bool(Expression.compile(expression).evaluate(lambda name, **kwargs: name == "engine"))
    except Exception:  # noqa: BLE001 - an expression pytest cannot read is not our concern here
        return False


def pytest_collection_finish(session) -> None:
    """Fail when the run selects the engine tests and a test module could not be collected.

    A run with ``-m engine`` (unit-tests.yaml, inside the base image) has the engine installed, so
    a module that is not collected is a fault, not a lean environment. ``BOLTZ2_REQUIRE_ENGINE=1``
    gives the same check for a run of the whole suite in the image.
    """
    required = _selects_engine_tests(session.config) or os.environ.get("BOLTZ2_REQUIRE_ENGINE") == "1"
    if _NOT_COLLECTED and required:
        raise pytest.UsageError("the engine is required for this run but " + _not_collected_message())


def pytest_terminal_summary(terminalreporter) -> None:
    if _NOT_COLLECTED:
        terminalreporter.write_line(_not_collected_message(), yellow=True)


# Imported rather than repeated as a literal: the default set is what every
# published figure on this asset sums over, so the tests must read the same
# constant the harness selects on.
from harness import DEFAULT_BENCHMARK_SET  # noqa: E402

REFERENCES = pathlib.Path(__file__).resolve().parent / "references"


@pytest.fixture(scope="session")
def request_model():
    """The service's request model; tests that need it are skipped without BentoML."""
    return pytest.importorskip("service").Boltz2Request


@pytest.fixture(scope="session")
def boltz_cli_defaults() -> dict:
    """Upstream ``boltz predict`` defaults, captured from boltz 2.2.1."""
    return json.loads((REFERENCES / "boltz_cli_defaults.json").read_text())


@pytest.fixture(scope="session")
def benchmark_recipe() -> dict:
    """What the shipped benchmark fixtures are, and how they may diverge from
    the serving defaults."""
    return json.loads((REFERENCES / "benchmark_recipe.json").read_text())


@pytest.fixture(scope="session")
def all_benchmark_fixtures() -> dict:
    """Every ``examples/benchmark_sequences/<id>/<id>.json``, by label.

    All eighteen, whichever set they belong to. Use this only for assertions that
    are genuinely about the whole directory; anything describing the AIM's
    published baseline wants ``benchmark_fixtures`` instead.
    """
    root = pathlib.Path(__file__).resolve().parents[1] / "examples" / "benchmark_sequences"
    loaded = {p.parent.name: json.loads(p.read_text()) for p in sorted(root.glob("*/*.json"))}
    assert loaded, f"no benchmark fixtures found under {root}"
    return loaded


@pytest.fixture(scope="session")
def benchmark_fixtures(all_benchmark_fixtures: dict) -> dict:
    """The DEFAULT SET only, by label — the six the harness evaluates unset.

    Filters on ``metadata.sets`` rather than globbing a directory, because the
    directory now holds all eighteen. Keeping this fixture's meaning fixed at
    "the default set" is what stops the assertions written against the six-target
    baseline from silently widening to eighteen when the sets were merged.
    """
    return {
        label: fixture
        for label, fixture in all_benchmark_fixtures.items()
        if DEFAULT_BENCHMARK_SET in fixture["metadata"]["sets"]
    }


@pytest.fixture(scope="session")
def accuracy_gate_fixtures() -> dict:
    """Every ``examples/benchmark_sequences_accuracy/<id>/<id>.json``, by label.

    A SEPARATE directory from ``all_benchmark_fixtures``, baked to a separate
    image path (``/workspace/model/benchmarks-accuracy``) and reached only when
    both ``BOLTZ2_BENCHMARK_DIR`` and ``BOLTZ2_BENCHMARK_SET=accuracy-gate`` are
    set. See tests/unit/test_accuracy_gate_fixtures.py.
    """
    root = pathlib.Path(__file__).resolve().parents[1] / "examples" / "benchmark_sequences_accuracy"
    loaded = {p.parent.name: json.loads(p.read_text()) for p in sorted(root.glob("*/*.json"))}
    assert loaded, f"no accuracy-gate fixtures found under {root}"
    return loaded


@pytest.fixture(scope="module")
def service_url() -> str:
    """Base URL of the running AIM service (GPU smoke tier only)."""
    port = os.environ.get("AIM_PORT") or os.environ.get("BOLTZ2_SMOKE_PORT") or "3000"
    return os.environ.get("AIM_SERVICE_URL") or f"http://localhost:{port}"


class FakeHttp:
    """Stands in for ``harness.urlopen``: records every request, answers like the service."""

    PDB = "ATOM      1  CA  ALA A   1       0.000   0.000   0.000  1.00  0.00           C\nEND\n"

    def __init__(self) -> None:
        self.requests: list[tuple[str, dict | None]] = []
        self.predict_body = self.PDB
        self.predict_status = 200
        self.mode = "structure"
        self.affinity = {"affinity_pred_value": 0.5, "affinity_probability_binary": 0.9}

    def __call__(self, req, timeout=None):
        url = req.full_url
        data = json.loads(req.data) if req.data else None
        self.requests.append((url, data))
        if url.endswith("/v1/inference"):
            if self.predict_status != 200:
                raise HTTPError(url, self.predict_status, "scripted failure", {}, None)
            name = data["data"]["name"]
            asks_affinity = bool(data["data"].get("properties"))
            if asks_affinity and self.mode != "affinity":
                raise HTTPError(url, 400, "affinity is not served by this profile", {}, None)
            content = self.predict_body if data["data"].get("output_format") == "pdb" else "data_x"
            answer = {
                "error": False,
                "structures": [{"record_id": name, "format": "mmcif", "content": content}],
                "confidence": {name: {"complex_plddt": 0.9, "ptm": 0.8}},
                "timing": {"runtime_s": 1.0},
                "runtime": {
                    "mode": self.mode,
                    "optimizations": "on",
                    "warmup": {"state": "done", "since_warmup": {"compiles": {}}},
                },
            }
            if asks_affinity:
                answer["affinity"] = self.affinity
            body = json.dumps(answer)
        else:
            body = "ok"
        return _FakeResponse(body)

    def sent_data(self) -> list[dict]:
        """The ``data`` block of every prediction request sent."""
        return [d["data"] for _, d in self.requests if d is not None]


class _FakeResponse:
    status = 200

    def __init__(self, body: str) -> None:
        self._body = body

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def read(self) -> bytes:
        return self._body.encode()


@pytest.fixture
def fake_http(monkeypatch) -> FakeHttp:
    import harness

    http = FakeHttp()
    monkeypatch.setattr(harness, "urlopen", http)
    return http
