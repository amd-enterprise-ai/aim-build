# Copyright © Advanced Micro Devices, Inc., or its affiliates.
#
# SPDX-License-Identifier: MIT

"""Behaviour of the resident-model pieces of service.py with stubs (no GPU, no torch, no semlaflow).

service.py needs bentoml, torch and semlaflow at import time, so the pieces under test are extracted
by AST and executed in a namespace with stand-ins. ``lightning`` and ``torch`` are stubbed in
``sys.modules`` while a test runs.
"""

from __future__ import annotations

import ast
import concurrent.futures
import importlib.util
import logging
import os
import re
import sys
import threading
import time
import types
from collections import OrderedDict
from contextlib import contextmanager
from pathlib import Path
from typing import Any

import pytest

SRC = Path(__file__).resolve().parents[1] / "src"
SERVICE_SRC = (SRC / "service.py").read_text(encoding="utf-8")
WANTED = {
    "REQUEST_TIMEOUT_SECONDS",
    "DEVICE_ENV",
    "DEFAULT_DEVICE",
    "_DEVICE_RE",
    "WARMUP_N_MOLECULES",
    "WARMUP_INTEGRATION_STEPS",
    "_DATASET_CACHE_SIZE",
    "SEMLAFLOW_PATH",
    "_resolve_device",
    "_worker_device_index",
    "_activate_device",
    "GenerationError",
    "_Job",
    "_DatasetCache",
    "_ResidentGenerator",
    "_GenerationEngine",
}


class _ServiceUnavailable(Exception):
    pass


class _InternalServerError(Exception):
    pass


def _predict_output() -> types.ModuleType:
    spec = importlib.util.spec_from_file_location("predict_output", SRC / "predict_output.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _build_namespace() -> dict[str, Any]:
    body = []
    for node in ast.parse(SERVICE_SRC).body:
        names: set[str] = set()
        if isinstance(node, (ast.FunctionDef, ast.ClassDef)):
            names = {node.name}
        elif isinstance(node, (ast.Assign, ast.AnnAssign)):
            targets = node.targets if isinstance(node, ast.Assign) else [node.target]
            names = {t.id for t in targets if isinstance(t, ast.Name)}
        if names & WANTED:
            body.append(node)
    ns: dict[str, Any] = {
        "concurrent": concurrent,
        "time": time,
        "os": os,
        "re": re,
        "sys": sys,
        "types": types,
        "Path": Path,
        "Any": Any,
        "OrderedDict": OrderedDict,
        "Iterator": object,
        "contextlib": __import__("contextlib"),
        "logger": logging.getLogger("service-under-test"),
        "ServiceUnavailable": _ServiceUnavailable,
        "InternalServerError": _InternalServerError,
    }
    exec(
        compile(ast.Module(body, []), "service_part", "exec", flags=__import__("__future__").annotations.compiler_flag),
        ns,
    )  # noqa: S102
    return ns


NS = _build_namespace()
Job = NS["_Job"]
Engine = NS["_GenerationEngine"]
GenerationError = NS["GenerationError"]


class _Gen:
    """Stand-in generator: records calls and the thread they ran on."""

    device = "cuda"

    def __init__(self, behaviour: Any = None) -> None:
        self.behaviour = behaviour or (lambda job: ["CCO"])
        self.threads: list[str] = []
        self.loaded = 0
        self.released = 0

    def load(self) -> None:
        self.loaded += 1
        self.threads.append(threading.current_thread().name)

    def generate(self, job: Any) -> list[str]:
        self.threads.append(threading.current_thread().name)
        return self.behaviour(job)

    def release_gpu_memory(self) -> None:
        self.released += 1

    def memory_note(self) -> str:
        return ""


def _job(**kw: Any) -> Any:
    base = dict(data_dir=Path("/d"), dataset_split="test", n_molecules=3, integration_steps=5, seed=1)
    base.update(kw)
    return Job(**base)


# ---- engine ----


def test_requests_run_on_the_single_worker_thread_after_load() -> None:
    gen = _Gen()
    engine = Engine(gen)
    engine.warm_up()
    assert engine.generate(_job(), timeout=5) == ["CCO"]
    assert engine.generate(_job(), timeout=5) == ["CCO"]
    assert gen.loaded == 1  # the model is loaded once, not per request
    assert len(set(gen.threads)) == 1 and gen.threads[0].startswith("semlaflow")


def test_warm_up_runs_one_small_real_generation() -> None:
    seen = []
    gen = _Gen(lambda job: seen.append(job) or ["C"])
    Engine(gen).warm_up()
    assert len(seen) == 1 and seen[0].warmup is True and seen[0].data_dir is None
    assert seen[0].n_molecules == NS["WARMUP_N_MOLECULES"]
    assert seen[0].integration_steps == NS["WARMUP_INTEGRATION_STEPS"]


def test_warm_up_failure_is_a_runtime_error_naming_the_device() -> None:
    def boom(job: Any) -> list[str]:
        raise ValueError("kaboom")

    with pytest.raises(RuntimeError, match="warm-up failed on device cuda"):
        Engine(_Gen(boom)).warm_up()


def test_failed_run_is_500_without_detail_and_frees_memory() -> None:
    def boom(job: Any) -> list[str]:
        raise ValueError("secret /path/detail")

    gen = _Gen(boom)
    engine = Engine(gen)
    with pytest.raises(_InternalServerError) as err:
        engine.generate(_job(), timeout=5)
    assert "secret" not in str(err.value) and gen.released == 1
    gen.behaviour = lambda job: ["CC"]
    assert engine.generate(_job(), timeout=5) == ["CC"]


def test_queued_request_that_times_out_is_cancelled_and_never_runs() -> None:
    release = threading.Event()
    ran: list[int] = []

    def slow(job: Any) -> list[str]:
        ran.append(job.seed)
        release.wait(5)
        return ["C"]

    engine = Engine(_Gen(slow))
    first = concurrent.futures.ThreadPoolExecutor(1).submit(engine.generate, _job(seed=1), 10)
    while not ran:
        time.sleep(0.01)
    with pytest.raises(_ServiceUnavailable, match="did not start within"):
        engine.generate(_job(seed=2), timeout=0.2)  # queued behind seed=1
    release.set()
    assert first.result(5) == ["C"]
    assert ran == [1]


def test_running_request_that_times_out_is_503_and_blocks_new_ones_until_it_ends() -> None:
    release = threading.Event()
    started = threading.Event()

    def slow(job: Any) -> list[str]:
        started.set()
        release.wait(5)
        return ["C"]

    gen = _Gen(slow)
    engine = Engine(gen)
    with pytest.raises(_ServiceUnavailable, match="exceeded"):
        engine.generate(_job(), timeout=0.2)
    assert started.is_set()
    with pytest.raises(_ServiceUnavailable, match="still running"):
        engine.generate(_job(), timeout=0.2)
    release.set()
    deadline = time.time() + 5
    while engine._abandoned and time.time() < deadline:
        time.sleep(0.01)
    gen.behaviour = lambda job: ["CN"]
    assert engine.generate(_job(), timeout=5) == ["CN"]


# ---- device ----


class _Torch:
    def __init__(self, available: bool, count: int = 1) -> None:
        self.set_device_calls: list[int] = []
        self.cuda = types.SimpleNamespace(
            is_available=lambda: available, device_count=lambda: count, set_device=self.set_device_calls.append
        )


@contextmanager
def _env(value: str | None) -> Any:
    old = os.environ.get("SEMLAFLOW_DEVICE")
    if value is None:
        os.environ.pop("SEMLAFLOW_DEVICE", None)
    else:
        os.environ["SEMLAFLOW_DEVICE"] = value
    try:
        yield
    finally:
        if old is None:
            os.environ.pop("SEMLAFLOW_DEVICE", None)
        else:
            os.environ["SEMLAFLOW_DEVICE"] = old


def test_device_defaults_to_cuda_and_never_falls_back_to_cpu() -> None:
    with _env(None):
        assert NS["_resolve_device"](_Torch(True)) == "cuda"
        with pytest.raises(RuntimeError, match="No GPU detected"):
            NS["_resolve_device"](_Torch(False))


def test_activate_device_sets_the_workers_current_gpu() -> None:
    torch = _Torch(True, 2)
    NS["_activate_device"](torch, "cuda:1")
    assert torch.set_device_calls == [1]


def test_activate_device_leaves_cpu_and_bare_cuda_alone() -> None:
    torch = _Torch(True, 2)
    NS["_activate_device"](torch, "cpu")
    NS["_activate_device"](torch, "cuda")
    assert torch.set_device_calls == []


def test_device_override_still_wins_over_the_accelerator_count_check() -> None:
    with _env("cuda:0"):
        assert NS["_resolve_device"](_Torch(True, 1), 2, 2) == "cuda:0"


def test_each_worker_defaults_to_its_own_gpu_when_several_accelerators() -> None:
    with _env(None):
        assert NS["_resolve_device"](_Torch(True, 2), 1, 2) == "cuda:0"
        assert NS["_resolve_device"](_Torch(True, 2), 2, 2) == "cuda:1"
        # Fewer GPUs than workers: startup fails instead of two workers sharing one GPU.
        with pytest.raises(
            RuntimeError, match=r"AIM_ACCELERATOR_COUNT=2 requires 2 accelerator\(s\), but this container has 1"
        ):
            NS["_resolve_device"](_Torch(True, 1), 2, 2)
        # A single accelerator keeps the plain "cuda" name.
        assert NS["_resolve_device"](_Torch(True, 1), 1, 1) == "cuda"
    with _env("cuda:1"):  # an explicit operator override wins over the worker default
        assert NS["_resolve_device"](_Torch(True, 2), 1, 2) == "cuda:1"


def test_device_override_accepts_cpu_cuda_index_and_rejects_the_rest() -> None:
    with _env("cpu"):
        assert NS["_resolve_device"](_Torch(False)) == "cpu"
    with _env("CUDA:1"):
        assert NS["_resolve_device"](_Torch(True, 2)) == "cuda:1"
        with pytest.raises(RuntimeError, match="only 1 GPU"):
            NS["_resolve_device"](_Torch(True, 1))
    for bad in ("gpu", "cuda:x", "cuda:0;rm", "mps"):
        with _env(bad), pytest.raises(RuntimeError, match="not valid"):
            NS["_resolve_device"](_Torch(True))


# ---- dataset cache ----


def test_dataset_cache_reads_once_rereads_changed_files_and_is_bounded(tmp_path: Path) -> None:
    loads: list[str] = []
    cache = NS["_DatasetCache"](lambda p: loads.append(str(p)) or object(), size=2)
    a, b, c = (tmp_path / n for n in ("a.smol", "b.smol", "c.smol"))
    for f in (a, b, c):
        f.write_bytes(b"x")
    first = cache.get(a)
    assert cache.get(a) is first and len(loads) == 1
    cache.get(b)
    cache.get(c)  # evicts a (least recently used)
    cache.get(a)
    assert len(loads) == 4
    a.write_bytes(b"changed")  # replaced file: parsed again
    cache.get(a)
    assert len(loads) == 5 and len([k for k in cache._items if k[0] == str(a)]) == 1


# ---- generator with a fake upstream ----


class _FakeModel:
    def __init__(self) -> None:
        self.calls: list[tuple[Any, int, str]] = []

    def _generate(self, tensors: Any, steps: int, strategy: str) -> dict:
        self.calls.append((tensors, steps, strategy))
        return {"o": len(self.calls)}

    def _generate_mols(self, output: dict) -> list[str]:
        return [f"C{output['o']}", None]


class _Tensor:
    def __init__(self, device: str | None = None) -> None:
        self.device = device

    def to(self, device: str) -> "_Tensor":
        return _Tensor(device)


def test_generator_uses_upstream_build_dm_resident_model_and_request_settings() -> None:
    gen = NS["_ResidentGenerator"]("cuda:0", Path("/ck/model.ckpt"))
    model = _FakeModel()
    built: list[Any] = []

    class _DM:
        def test_dataloader(self) -> list:
            return [({"coords": _Tensor()},), ({"coords": _Tensor()},)]

    upstream = types.SimpleNamespace(
        DEFAULT_BATCH_COST=8192,
        DEFAULT_BUCKET_COST_SCALE="linear",
        DEFAULT_ODE_SAMPLING_STRATEGY="log",
        GeometricDataset=type("GeometricDataset", (), {}),
        build_dm=lambda args, hparams, vocab: built.append((args, hparams, vocab)) or _DM(),
    )
    gen._upstream, gen._model, gen._vocab, gen._hparams = upstream, model, "VOCAB", {"h": 1}
    gen._datasets = NS["_DatasetCache"](lambda p: object())
    seeds: list[int] = []
    po = _predict_output()
    po.rdkit_smiles = lambda m: m  # no RDKit here: molecules are already strings
    po.rdkit_molblock = lambda m: f"block-{m}"
    lightning = types.ModuleType("lightning")
    lightning.seed_everything = lambda seed, workers=False: seeds.append(seed)
    torch = types.ModuleType("torch")
    saved = {k: sys.modules.get(k) for k in ("lightning", "torch", "predict_output")}
    sys.modules.update({"lightning": lightning, "torch": torch, "predict_output": po})
    try:
        out = gen.generate(_job(n_molecules=4, integration_steps=7, dataset_split="val", seed=99))
    finally:
        for k, v in saved.items():
            if v is None:
                sys.modules.pop(k, None)
            else:
                sys.modules[k] = v
    assert out == [("C1", "block-C1"), ("C2", "block-C2")]  # None (failed sanitisation) left out; both batches used
    assert seeds == [99]
    args, hparams, vocab = built[0]
    assert (args.n_molecules, args.dataset_split, args.dataset, args.data_path) == (4, "val", "geom-drugs", "/d")
    assert hparams == {"h": 1} and vocab == "VOCAB"
    assert [c[1:] for c in model.calls] == [(7, "log"), (7, "log")]  # steps from the request, upstream strategy
    assert all(c[0]["coords"].device == "cuda:0" for c in model.calls)  # tensors moved to the chosen device


def test_dataset_loader_reads_through_the_original_class_while_the_shim_is_installed(tmp_path: Path) -> None:
    class _Dataset:
        reads: list[Any] = []

        def __init__(self, data: Any, transform: Any = None) -> None:
            self._data = data
            self.transform = transform

        @classmethod
        def load(cls, data_path: Any, transform: Any = None, min_size: Any = None) -> Any:
            cls.reads.append(Path(data_path))
            return cls(["mol"], transform=transform)

    class _Model:
        def to(self, device: str) -> "_Model":
            return self

        def eval(self) -> "_Model":
            return self

    upstream = types.ModuleType("semlaflow.predict")
    upstream.GeometricDataset = _Dataset
    upstream.DEFAULT_INTEGRATION_STEPS = 100
    upstream.DEFAULT_ODE_SAMPLING_STRATEGY = "log"
    upstream.DEFAULT_CAT_SAMPLING_NOISE_LEVEL = 1.0
    upstream.load_model = lambda args, vocab: _Model()
    util = types.ModuleType("semlaflow.scriptutil")
    util.disable_lib_stdout = util.configure_fs = lambda: None
    util.build_vocab = lambda: "VOCAB"
    torch = types.ModuleType("torch")
    torch.load = lambda path, *a, **kw: {"hyper_parameters": {}}
    pkg = types.ModuleType("semlaflow")
    mods = {"semlaflow": pkg, "semlaflow.predict": upstream, "semlaflow.scriptutil": util, "torch": torch}
    saved = {k: sys.modules.get(k) for k in mods}
    sys.modules.update(mods)
    try:
        gen = NS["_ResidentGenerator"]("cpu", tmp_path / "model.ckpt")
        gen.load()
        with gen._dataset_source(_job(data_dir=tmp_path)):
            shim = upstream.GeometricDataset
            assert shim is not _Dataset
            first = shim.load(tmp_path)
            second = shim.load(tmp_path)  # served from the cache
    finally:
        for k, v in saved.items():
            if v is None:
                sys.modules.pop(k, None)
            else:
                sys.modules[k] = v
    assert type(first) is _Dataset and first._data == ["mol"] and second._data == ["mol"]
    assert _Dataset.reads == [tmp_path]
    assert upstream.GeometricDataset is _Dataset
