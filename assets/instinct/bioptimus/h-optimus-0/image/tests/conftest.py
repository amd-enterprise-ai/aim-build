# Copyright © Advanced Micro Devices, Inc., or its affiliates.
#
# SPDX-License-Identifier: MIT

"""Shared stubs so tests import the real ``harness.py`` / ``service.py``.

Pattern (copy per asset; only the names in the constants below change):

* ``harness_module``: loads ``src/harness.py`` with importlib under a private
  module name. ``aim_runtime`` is imported for real when available and replaced
  by a minimal stand-in in ``sys.modules`` otherwise.
* ``stub_server``: a throw-away local HTTP server with per-path handlers, so the
  harness runs its real ``urllib`` code against it (no mocking of urllib).
* ``service_module``: loads ``src/service.py`` with the heavy, GPU-only imports
  (``bentoml``, ``torch``, ``timm``, ``torchvision``) stubbed for the duration of
  the import only, so decorators are identity and the class stays callable.

No GPU, no weights, no network beyond 127.0.0.1.
"""

from __future__ import annotations

import importlib.util
import json
import sys
import threading
import types
from dataclasses import dataclass, field
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, Callable, Iterator

import pytest

IMAGE_DIR = Path(__file__).resolve().parents[1]
SRC_DIR = IMAGE_DIR / "src"
HARNESS_FILE = SRC_DIR / "harness.py"
SERVICE_FILE = SRC_DIR / "service.py"
PRIVATE_PREFIX = "hoptimus0_under_test"


def _load(name: str, path: Path) -> types.ModuleType:
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def _stub_aim_runtime(mp: pytest.MonkeyPatch) -> None:
    """Install a minimal ``aim_runtime.harness`` only when the real one is missing."""
    try:
        import aim_runtime.harness  # noqa: F401

        return
    except ImportError:
        pass
    from enum import Enum

    mod = types.ModuleType("aim_runtime.harness")

    class CheckResultType(Enum):
        PASS_FAIL = "pass_fail"
        SCORE = "score"

    class CheckScope(Enum):
        RUNTIME = "runtime"
        OFFLINE = "offline"

    @dataclass
    class CheckInfo:
        name: str
        result_type: Any
        scope: Any
        description: str = ""

    @dataclass
    class CheckResult:
        name: str
        result_type: Any
        success: bool
        value: Any
        detail: str = ""
        skipped: bool = False

    @dataclass
    class HarnessResult:
        success: bool
        summary: str
        checks: list = field(default_factory=list)
        metrics: dict = field(default_factory=dict)

    @dataclass
    class HarnessConfig:
        profile: dict
        service_url: str | None = None
        timeout_seconds: int = 300
        check_scopes: set = field(default_factory=lambda: set(CheckScope))
        extra: dict = field(default_factory=dict)

        def get(self, key: str, default: Any = None) -> Any:
            return self.extra.get(key, self.profile.get(key, default))

        def resolve_service_url(self) -> str:
            return self.service_url or f"http://localhost:{self.get('port', 8000)}"

    def skipped(name: str, detail: str) -> CheckResult:
        return CheckResult(name, CheckResultType.PASS_FAIL, True, None, detail, skipped=True)

    for obj in (CheckInfo, CheckResult, CheckResultType, CheckScope, HarnessConfig, HarnessResult, skipped):
        setattr(mod, obj.__name__, obj)
    mod.ModelHarness = type("ModelHarness", (), {})
    pkg = types.ModuleType("aim_runtime")
    pkg.harness = mod  # type: ignore[attr-defined]
    mp.setitem(sys.modules, "aim_runtime", pkg)
    mp.setitem(sys.modules, "aim_runtime.harness", mod)


@pytest.fixture(scope="session")
def harness_module() -> Iterator[types.ModuleType]:
    with pytest.MonkeyPatch.context() as mp:
        _stub_aim_runtime(mp)
        module = _load(f"{PRIVATE_PREFIX}_harness", HARNESS_FILE)
        yield module
    sys.modules.pop(f"{PRIVATE_PREFIX}_harness", None)


@pytest.fixture
def make_config(harness_module):
    """Build a ``HarnessConfig`` pointing at a URL; extra kwargs override fields."""
    # Use the class the loaded harness already exports: that is the real runtime
    # class when installed and the stand-in otherwise (whose ``__module__`` is
    # this conftest, so resolving it through ``__module__`` would fail).
    cfg_cls = harness_module.HarnessConfig

    def _make(url: str, **kwargs: Any):
        kwargs.setdefault("profile", {})
        kwargs.setdefault("timeout_seconds", 2)
        return cfg_cls(service_url=url, **kwargs)

    return _make


@dataclass
class StubServer:
    url: str
    routes: dict[tuple[str, str], Callable[[dict[str, Any] | None], tuple[int, Any]]] = field(default_factory=dict)
    calls: list[tuple[str, str]] = field(default_factory=list)

    def on(self, method: str, path: str, handler: Callable[[dict[str, Any] | None], tuple[int, Any]]) -> None:
        self.routes[(method, path)] = handler

    def count(self, method: str, path: str) -> int:
        return self.calls.count((method, path))


@pytest.fixture
def stub_server() -> Iterator[StubServer]:
    """Local HTTP server; register handlers with ``stub.on(method, path, fn)``.

    A handler receives the parsed JSON body (or None) and returns
    ``(status, json_serialisable)``. Unregistered routes answer 404.
    """
    state: StubServer

    class Handler(BaseHTTPRequestHandler):
        def _serve(self, method: str) -> None:
            length = int(self.headers.get("Content-Length") or 0)
            body = json.loads(self.rfile.read(length)) if length else None
            state.calls.append((method, self.path))
            handler = state.routes.get((method, self.path))
            status, payload = handler(body) if handler else (404, {"detail": "not found"})
            data = json.dumps(payload).encode()
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def do_GET(self) -> None:  # noqa: N802
            self._serve("GET")

        def do_POST(self) -> None:  # noqa: N802
            self._serve("POST")

        def log_message(self, *_args: Any) -> None:
            pass

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    state = StubServer(url=f"http://127.0.0.1:{server.server_address[1]}")
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield state
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)


class _Identity:
    """Decorator factory stub: ``@stub(...)`` and ``@stub`` both return the target."""

    def __call__(self, *args: Any, **kwargs: Any):
        if len(args) == 1 and not kwargs and callable(args[0]):
            return args[0]
        return lambda target: target


class _Recorder(_Identity):
    """``@bentoml.service(...)`` stand-in that records the decorator keyword arguments."""

    def __init__(self, sink: list[dict[str, Any]]) -> None:
        self._sink = sink

    def __call__(self, *args: Any, **kwargs: Any):
        self._sink.append(kwargs)
        return super().__call__(*args, **kwargs)


def _stub_gpu_stack(
    mp: pytest.MonkeyPatch,
    *,
    cuda_available: bool,
    port: int,
    accelerator_count: int | None = None,
    worker_index: int = 1,
    device_count: int = 1,
    service_kwargs: list[dict[str, Any]] | None = None,
) -> None:
    class InvalidArgument(Exception):
        pass

    bentoml = types.ModuleType("bentoml")
    bentoml.service = _Recorder(service_kwargs if service_kwargs is not None else [])  # type: ignore[attr-defined]
    bentoml.server_context = types.SimpleNamespace(worker_index=worker_index)  # type: ignore[attr-defined]
    bentoml.api = _Identity()  # type: ignore[attr-defined]
    bentoml.asgi_app = _Identity()  # type: ignore[attr-defined]
    extra = {} if accelerator_count is None else {"accelerator_count": accelerator_count}
    bentoml.use_arguments = lambda cls: cls(port=port, **extra)  # type: ignore[attr-defined]
    exceptions = types.ModuleType("bentoml.exceptions")
    exceptions.InvalidArgument = InvalidArgument  # type: ignore[attr-defined]
    bentoml.exceptions = exceptions  # type: ignore[attr-defined]

    torch = types.ModuleType("torch")
    set_device_calls: list[int] = []  # torch.cuda.set_device recorder
    torch.cuda = types.SimpleNamespace(  # type: ignore[attr-defined]
        is_available=lambda: cuda_available,
        get_device_name=lambda _i: "Stub GPU",
        device_count=lambda: device_count,
        set_device=set_device_calls.append,
    )
    torch.set_device_calls = set_device_calls  # type: ignore[attr-defined]
    torch.device = lambda name: name  # type: ignore[attr-defined]
    torch.float16 = "float16"  # type: ignore[attr-defined]
    torch.set_num_threads = lambda n: None  # type: ignore[attr-defined]

    timm = types.ModuleType("timm")
    timm.create_model = lambda *a, **k: (_ for _ in ()).throw(AssertionError("model must not load in tests"))  # type: ignore[attr-defined]

    transforms = types.ModuleType("torchvision.transforms")
    for name in ("Normalize", "Resize", "ToTensor"):
        setattr(transforms, name, lambda *a, **k: (lambda x: x))
    transforms.Compose = lambda steps: (lambda x: x)  # type: ignore[attr-defined]
    torchvision = types.ModuleType("torchvision")
    torchvision.transforms = transforms  # type: ignore[attr-defined]

    for name, mod in {
        "bentoml": bentoml,
        "bentoml.exceptions": exceptions,
        "torch": torch,
        "timm": timm,
        "torchvision": torchvision,
        "torchvision.transforms": transforms,
    }.items():
        mp.setitem(sys.modules, name, mod)


@pytest.fixture
def load_service():
    """Return ``load(cuda_available=False, port=3000)`` -> freshly imported ``service.py``."""
    pytest.importorskip("starlette")
    pytest.importorskip("httpx")
    pytest.importorskip("pydantic")
    pytest.importorskip("PIL")
    names: list[str] = []

    def _load_service(
        *,
        cuda_available: bool = False,
        port: int = 3000,
        accelerator_count: int | None = None,
        worker_index: int = 1,
        device_count: int = 1,
    ) -> types.ModuleType:
        service_kwargs: list[dict[str, Any]] = []
        with pytest.MonkeyPatch.context() as mp:
            _stub_gpu_stack(
                mp,
                cuda_available=cuda_available,
                port=port,
                accelerator_count=accelerator_count,
                worker_index=worker_index,
                device_count=device_count,
                service_kwargs=service_kwargs,
            )
            name = f"{PRIVATE_PREFIX}_service_{len(names)}"
            names.append(name)
            module = _load(name, SERVICE_FILE)
        module.RECORDED_SERVICE_KWARGS = service_kwargs  # type: ignore[attr-defined]
        return module

    yield _load_service
    for name in names:
        sys.modules.pop(name, None)
