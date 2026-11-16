# Copyright © Advanced Micro Devices, Inc., or its affiliates.
#
# SPDX-License-Identifier: MIT

"""Behaviour of the resident-model engine in service.py, with stubs (no GPU, no torch, no reinvent).

service.py needs bentoml, torch and reinvent4 at import time, so the engine pieces are extracted by
AST and executed in a namespace with stand-ins; ``torch``, ``rdkit`` and ``reinvent`` are stubbed in
``sys.modules`` for the duration of each test.
"""

from __future__ import annotations
import __future__

import ast
import concurrent.futures
import enum
import hashlib
import logging
import os
import sys
import tempfile
import threading
import time
import types
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pytest

SERVICE_SRC = (Path(__file__).resolve().parents[1] / "src/service.py").read_text(encoding="utf-8")
WANTED = {
    "REQUEST_TIMEOUT_SECONDS",
    "WARMUP_NUM_SMILES",
    "SamplingError",
    "_Job",
    "_Outcome",
    "_cache_key",
    "PriorIntegrityError",
    "_snapshot_prior",
    "_ResidentSampler",
    "_SamplingEngine",
    "_start_validation_pool",
    "SAMPLE_BATCH_SIZE_ENV",
    "MAX_SAMPLE_BATCH_SIZE",
    "resolve_sample_batch_size",
}


class _ServiceUnavailable(Exception):
    pass


class _InternalServerError(Exception):
    pass


def _build_namespace() -> dict[str, Any]:
    body = []
    for node in ast.parse(SERVICE_SRC).body:
        names = set()
        if isinstance(node, (ast.FunctionDef, ast.ClassDef)):
            names = {node.name}
        elif isinstance(node, (ast.Assign, ast.AnnAssign)):
            targets = node.targets if isinstance(node, ast.Assign) else [node.target]
            names = {t.id for t in targets if isinstance(t, ast.Name)}
        if names & WANTED:
            body.append(node)
    module = types.ModuleType("service_under_test")
    sys.modules["service_under_test"] = module  # dataclasses looks its module up by name
    ns: dict[str, Any] = module.__dict__
    ns.update(
        {
            "concurrent": concurrent,
            "time": time,
            "Path": Path,
            "Any": Any,
            "os": os,
            "hashlib": hashlib,
            "tempfile": tempfile,
            "_expected_checksum": lambda name: ("sha256", NS["EXPECTED"][name], None),
            "dataclass": dataclass,
            "logger": logging.getLogger("service-under-test"),
            "ServiceUnavailable": _ServiceUnavailable,
            "InternalServerError": _InternalServerError,
        }
    )
    code = compile(
        ast.fix_missing_locations(ast.Module(body=body, type_ignores=[])),
        "service_engine",
        "exec",
        flags=__future__.annotations.compiler_flag,
    )
    exec(code, ns)  # noqa: S102
    return ns


NS = _build_namespace()
NS["EXPECTED"] = {}  # prior name -> sha256 the registry would hold


class FakeUpstream:
    """Stand-ins for torch / rdkit / reinvent that record what the engine calls."""

    def __init__(self, model_type: str = "Reinvent", smiles: list[str | None] | None = None) -> None:
        self.model_type = model_type
        self.smiles = ["CCO", "c1ccccc1", None, " "] if smiles is None else smiles
        self.loads = 0
        self.seeds: list[int] = []
        self.batch_sizes: list[int] = []
        self.default_device_threads: list[tuple[str, int]] = []
        self.sample_threads: list[int] = []
        self.filter_calls = 0
        self.sample_hook = None

    def modules(self) -> dict[str, Any]:
        up = self

        class _State(enum.Enum):
            INVALID = 0
            VALID = 1
            DUPLICATE = 2

        class _Sampled:
            def __init__(self, smilies: list[str | None], states: list[Any] | None = None) -> None:
                self.smilies = smilies
                if states is None:
                    states, seen = [], set()
                    for smi in smilies:
                        if not smi or not smi.strip():
                            states.append(_State.INVALID)
                        else:
                            states.append(_State.DUPLICATE if smi in seen else _State.VALID)
                            seen.add(smi)
                self.states = states

        class _Sampler:
            def __init__(self, batch: int) -> None:
                self.batch = batch

            def sample(self, _dummy: Any) -> _Sampled:
                up.sample_threads.append(threading.get_ident())
                up.batch_sizes.append(self.batch)
                if up.sample_hook:
                    up.sample_hook()
                return _Sampled(list(up.smiles))

        def create_adapter(path: str, mode: str, device: Any) -> tuple[Any, dict, str]:
            assert mode == "inference"
            up.loads += 1
            return object(), {}, up.model_type

        def setup_sampler(model_type: str, params: dict, adapter: Any) -> tuple[_Sampler, int]:
            assert set(params) == {"batch_size"}
            return _Sampler(params["batch_size"]), params["batch_size"]

        def filter_valid(sampled: _Sampled) -> _Sampled:
            up.filter_calls += 1
            return _Sampled([s for s in sampled.smilies if s == "CCO"])

        torch = types.ModuleType("torch")
        torch.device = lambda name: name  # type: ignore[attr-defined]
        torch.set_default_device = lambda name: up.default_device_threads.append((name, threading.get_ident()))  # type: ignore[attr-defined]

        @contextmanager
        def no_grad():
            yield

        torch.no_grad = no_grad  # type: ignore[attr-defined]
        torch.cuda = types.SimpleNamespace(  # type: ignore[attr-defined]
            empty_cache=lambda: None, memory_allocated=lambda: 0, memory_reserved=lambda: 0
        )
        rdkit = types.ModuleType("rdkit")
        rdkit.rdBase = types.SimpleNamespace(DisableLog=lambda _pattern: None)  # type: ignore[attr-defined]
        reinvent = types.ModuleType("reinvent")
        runmodes = types.ModuleType("reinvent.runmodes")
        runmodes.create_adapter = create_adapter  # type: ignore[attr-defined]
        samplers = types.ModuleType("reinvent.runmodes.samplers")
        run_sampling = types.ModuleType("reinvent.runmodes.samplers.run_sampling")
        run_sampling.filter_valid = filter_valid  # type: ignore[attr-defined]
        setup = types.ModuleType("reinvent.runmodes.setup_sampler")
        setup.setup_sampler = setup_sampler  # type: ignore[attr-defined]
        sample_batch = types.ModuleType("reinvent.models.model_factory.sample_batch")
        sample_batch.SmilesState = _State  # type: ignore[attr-defined]
        utils = types.ModuleType("reinvent.utils")
        utils.set_seed = up.seeds.append  # type: ignore[attr-defined]
        return {
            "torch": torch,
            "rdkit": rdkit,
            "reinvent": reinvent,
            "reinvent.runmodes": runmodes,
            "reinvent.runmodes.samplers": samplers,
            "reinvent.runmodes.samplers.run_sampling": run_sampling,
            "reinvent.runmodes.setup_sampler": setup,
            "reinvent.utils": utils,
            "reinvent.models": types.ModuleType("reinvent.models"),
            "reinvent.models.model_factory": types.ModuleType("reinvent.models.model_factory"),
            "reinvent.models.model_factory.sample_batch": sample_batch,
        }


@pytest.fixture
def up(monkeypatch: pytest.MonkeyPatch) -> FakeUpstream:
    fake = FakeUpstream()
    for name, mod in fake.modules().items():
        monkeypatch.setitem(sys.modules, name, mod)
    return fake


@pytest.fixture
def prior(tmp_path: Path) -> Path:
    path = tmp_path / "reinvent.prior"
    path.write_bytes(b"weights")
    NS["EXPECTED"]["reinvent_pubchem"] = hashlib.sha256(b"weights").hexdigest()
    return path


def _job(prior: Path, **kw: Any) -> Any:
    values = dict(
        prior_path=prior,
        prior_name="reinvent_pubchem",
        num_smiles=8,
        unique_molecules=True,
    )
    values.update(kw)
    return NS["_Job"](**values)


def _engine(device: str = "cpu") -> Any:
    return NS["_SamplingEngine"](NS["_ResidentSampler"](device))


def test_outcome_counts_invalid_and_duplicate_before_the_unique_filter(up: FakeUpstream, prior: Path) -> None:
    up.smiles = ["CCO", "CCO", None, " ", "c1ccccc1", "CCO"]
    engine = _engine()
    unique = engine.sample(_job(prior, unique_molecules=True))
    assert (unique.num_invalid, unique.num_duplicates) == (2, 2)  # counted before the (stubbed) filter
    plain = engine.sample(_job(prior, unique_molecules=False))
    assert (plain.num_invalid, plain.num_duplicates) == (2, 2)


def test_model_loaded_once_for_many_requests(up: FakeUpstream, prior: Path) -> None:
    engine = _engine()
    engine.warm_up(prior, "reinvent_pubchem")
    for _ in range(3):
        assert engine.sample(_job(prior)).smiles == ["CCO"]
    assert up.loads == 1
    assert up.batch_sizes == [NS["WARMUP_NUM_SMILES"], 8, 8, 8]


def test_prior_reloaded_after_the_file_changed(up: FakeUpstream, prior: Path) -> None:
    engine = _engine()
    engine.sample(_job(prior))
    engine.sample(_job(prior))
    assert up.loads == 1
    prior.write_bytes(b"other weights, other size")
    NS["EXPECTED"]["reinvent_pubchem"] = hashlib.sha256(b"other weights, other size").hexdigest()
    engine.sample(_job(prior))
    assert up.loads == 2


def test_prior_swapped_after_verification_is_never_loaded(
    up: FakeUpstream, prior: Path, caplog: pytest.LogCaptureFixture
) -> None:
    """A pickle placed on the cache path is rejected: the loaded bytes must match the registered checksum."""
    engine = _engine()
    engine.sample(_job(prior))
    assert up.loads == 1
    prior.write_bytes(b"malicious pickle")  # registry digest unchanged
    with caplog.at_level(logging.ERROR, logger="service-under-test"):
        with pytest.raises(_InternalServerError, match="^Sampling run failed$"):
            engine.sample(_job(prior))
    assert up.loads == 1  # create_adapter never saw the swapped file
    assert "mismatch after snapshot" in caplog.text


def test_load_gets_a_private_copy_not_the_cache_path(
    up: FakeUpstream, prior: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    seen: list[str] = []
    real = sys.modules["reinvent.runmodes"].create_adapter

    def spy(path: str, mode: str, device: Any) -> Any:
        seen.append(path)
        return real(path, mode, device)

    monkeypatch.setattr(sys.modules["reinvent.runmodes"], "create_adapter", spy)
    _engine().sample(_job(prior))
    assert seen and seen[0] != str(prior) and Path(seen[0]).name == "reinvent_pubchem.prior"


def test_everything_runs_on_one_worker_thread_with_the_device_set_there(up: FakeUpstream, prior: Path) -> None:
    engine = _engine("cuda:0")
    engine.warm_up(prior, "reinvent_pubchem")
    engine.sample(_job(prior))
    assert len(set(up.sample_threads)) == 1 and up.sample_threads[0] != threading.get_ident()
    assert up.default_device_threads == [("cuda:0", up.sample_threads[0])]


def test_unique_molecules_filters_and_blank_smiles_are_dropped(up: FakeUpstream, prior: Path) -> None:
    engine = _engine()
    assert engine.sample(_job(prior, unique_molecules=True)).smiles == ["CCO"]
    assert up.filter_calls == 1
    assert engine.sample(_job(prior, unique_molecules=False)).smiles == ["CCO", "c1ccccc1"]
    assert up.filter_calls == 1


def test_seed_is_applied_only_when_given(up: FakeUpstream, prior: Path) -> None:
    engine = _engine()
    engine.sample(_job(prior))
    engine.sample(_job(prior, seed=7))
    assert up.seeds == [7]


def test_failure_is_generic_and_the_service_keeps_working(
    up: FakeUpstream, prior: Path, caplog: pytest.LogCaptureFixture
) -> None:
    engine = _engine()

    def boom() -> None:
        up.sample_hook = None
        raise ValueError("secret detail /path/to/file")

    up.sample_hook = boom
    with caplog.at_level(logging.ERROR, logger="service-under-test"):
        with pytest.raises(_InternalServerError, match="^Sampling run failed$"):
            engine.sample(_job(prior))
    assert "secret detail" in caplog.text  # detail only in the server log
    assert engine.sample(_job(prior)).smiles == ["CCO"]


def test_empty_result_is_an_error(up: FakeUpstream, prior: Path) -> None:
    up.smiles = [None, " "]
    with pytest.raises(_InternalServerError, match="produced no SMILES"):
        _engine().sample(_job(prior))


def test_non_reinvent_prior_is_refused_generically(up: FakeUpstream, prior: Path) -> None:
    up.model_type = "Mol2Mol"
    with pytest.raises(_InternalServerError, match="^Sampling run failed$"):
        _engine().sample(_job(prior))


def test_warm_up_failure_is_a_runtime_error(up: FakeUpstream, prior: Path) -> None:
    up.model_type = "Mol2Mol"
    with pytest.raises(RuntimeError, match="warm-up failed on device cpu"):
        _engine().warm_up(prior, "reinvent_pubchem")


def test_waiting_request_is_cancelled_with_503_and_never_runs(up: FakeUpstream, prior: Path) -> None:
    engine = _engine()
    release = threading.Event()
    up.sample_hook = lambda: release.wait(5)
    first = threading.Thread(target=lambda: engine.sample(_job(prior, num_smiles=1)), daemon=True)
    first.start()
    while not up.batch_sizes:
        time.sleep(0.01)
    # the first run is still going, so this one waits in the queue and times out
    with pytest.raises(_ServiceUnavailable, match="did not start within"):
        engine.sample(_job(prior, num_smiles=2), timeout=0.2)
    release.set()
    first.join(5)
    assert engine.sample(_job(prior, num_smiles=3)).smiles == ["CCO"]
    assert 2 not in up.batch_sizes


def test_running_request_that_times_out_blocks_new_ones_until_it_ends(up: FakeUpstream, prior: Path) -> None:
    engine = _engine()
    release = threading.Event()
    up.sample_hook = lambda: release.wait(5)
    with pytest.raises(_ServiceUnavailable, match="exceeded"):
        engine.sample(_job(prior), timeout=0.2)
    with pytest.raises(_ServiceUnavailable, match="still running"):
        engine.sample(_job(prior))
    up.sample_hook = None
    release.set()
    deadline = time.time() + 5
    while engine._abandoned and time.time() < deadline:
        time.sleep(0.01)
    assert engine.sample(_job(prior)).smiles == ["CCO"]


def test_request_timeout_is_600_seconds() -> None:
    assert NS["REQUEST_TIMEOUT_SECONDS"] == 600
    assert "REQUEST_TIMEOUT_SECONDS + 60" in SERVICE_SRC  # BentoML's own timeout is above the service limit


def _sampling_request_model() -> Any:
    """SamplingRequest extracted from service.py by AST (pydantic is the only thing it needs)."""
    from pydantic import BaseModel, Field  # noqa: PLC0415

    node = next(n for n in ast.parse(SERVICE_SRC).body if isinstance(n, ast.ClassDef) and n.name == "SamplingRequest")
    ns: dict[str, Any] = {"BaseModel": BaseModel, "Field": Field}
    exec(  # noqa: S102
        compile(
            ast.fix_missing_locations(ast.Module(body=[node], type_ignores=[])),
            "sampling_request",
            "exec",
            flags=__future__.annotations.compiler_flag,
        ),
        ns,
    )
    return ns["SamplingRequest"]


def test_removed_fields_are_not_in_the_schema_but_unknown_fields_are_still_ignored() -> None:
    model = _sampling_request_model()
    assert set(model.model_fields) == {"num_smiles", "unique_molecules", "prior"}
    assert model.model_config.get("extra", "ignore") == "ignore"  # not forbidden: no 422 for an old client
    old_client = model.model_validate({"num_smiles": 5, "temperature": 0.5, "randomize_smiles": False})
    assert old_client.model_dump() == {"num_smiles": 5, "unique_molecules": True, "prior": None}


def test_job_has_no_temperature_or_randomize_smiles() -> None:
    assert set(NS["_Job"].__dataclass_fields__) == {
        "prior_path",
        "prior_name",
        "num_smiles",
        "unique_molecules",
        "seed",
    }


# --- parallel validation wired into the sampler ---------------------------------------------------------------

POOL_ALPHABET = ["cco", "CCO", "c1ccccc1", "C1CCCCC1", "XBAD", "(", "", "QQ1", "N#N", "cc(C)O", "o=c", "CCN"]


@pytest.fixture
def pooled(up: FakeUpstream, pool_module: Any, monkeypatch: pytest.MonkeyPatch) -> Any:
    """Realistic Reinvent sampler stubs: rows from a seeded generator, upstream's validation, a real ``filter_valid`` mask."""
    import random

    np = pytest.importorskip("numpy")
    state = pool_module.SmilesState

    class _Batch:
        def __init__(self, rows: list[Any]) -> None:
            self.rows = rows
            self.output = [row.output for row in rows]
            self.items1 = "unset"
            self.smilies: list[str] = []
            self.states: Any = None

        @classmethod
        def from_list(cls, rows: list[Any]) -> "_Batch":
            return cls(rows)

    class _Model:
        def __init__(self) -> None:
            self.rng = random.Random(11)
            self.calls: list[int] = []

        def sample(self, size: int) -> list[Any]:
            self.calls.append(size)
            return [types.SimpleNamespace(output=self.rng.choice(POOL_ALPHABET)) for _ in range(size)]

    class _ReinventSampler:
        """ReinventSampler.sample of upstream v4.8, with upstream's validate_smiles (the pool module's verbatim copy)."""

        def __init__(self, batch_size: int) -> None:
            self.model = _Model()
            self.batch_size = batch_size
            self.isomeric = False

        def sample(self, _dummy: Any) -> _Batch:
            sizes = [1000] * (self.batch_size // 1000) + ([self.batch_size % 1000] if self.batch_size % 1000 else [])
            sampled = _Batch([row for size in sizes for row in self.model.sample(size)])
            sampled.items1 = None
            chem = pool_module.Chem
            mols = [chem.MolFromSmiles(s, sanitize=False) if s else None for s in sampled.output]
            sampled.smilies, sampled.states = pool_module.validate_smiles(mols, sampled.output, isomeric=False)
            return sampled

    created: list[_ReinventSampler] = []

    def setup_sampler(model_type: str, params: dict, adapter: Any) -> tuple[_ReinventSampler, int]:
        sampler = _ReinventSampler(params["batch_size"])
        created.append(sampler)
        return sampler, params["batch_size"]

    def filter_valid(sampled: _Batch) -> _Batch:
        keep = np.array(sampled.states) == state.VALID
        sampled.smilies = [smi for smi, k in zip(sampled.smilies, keep) if k]
        return sampled

    sample_batch = types.ModuleType("reinvent.models.model_factory.sample_batch")
    sample_batch.SampleBatch = _Batch  # type: ignore[attr-defined]
    sample_batch.SmilesState = state  # type: ignore[attr-defined]
    params_mod = types.ModuleType("reinvent.runmodes.samplers.params")
    params_mod.SAMPLE_BATCH_SIZE = 1000  # type: ignore[attr-defined]
    for name, mod in {
        "reinvent.models": types.ModuleType("reinvent.models"),
        "reinvent.models.model_factory": types.ModuleType("reinvent.models.model_factory"),
        "reinvent.models.model_factory.sample_batch": sample_batch,
        "reinvent.runmodes.samplers.params": params_mod,
    }.items():
        monkeypatch.setitem(sys.modules, name, mod)
    monkeypatch.setattr(sys.modules["reinvent.runmodes.setup_sampler"], "setup_sampler", setup_sampler)
    monkeypatch.setattr(sys.modules["reinvent.runmodes.samplers.run_sampling"], "filter_valid", filter_valid)
    return types.SimpleNamespace(created=created, module=pool_module)


@pytest.mark.parametrize("unique", [True, False])
@pytest.mark.parametrize("num_smiles", [1, 7, 512, 2500])
def test_pooled_sampling_equals_in_process(pooled: Any, prior: Path, unique: bool, num_smiles: int) -> None:
    job = _job(prior, num_smiles=num_smiles, unique_molecules=unique)
    inline = NS["_ResidentSampler"]("cpu").sample(job)
    pool = pooled.module.ValidationPool(3)
    pool.start()
    try:
        pooled_result = NS["_ResidentSampler"]("cpu", pool).sample(job)
        assert pool.usable
    finally:
        pool.shutdown()
    assert pooled_result == inline
    assert pooled.created[0].model.calls == pooled.created[1].model.calls  # same batches, same order


def test_pooled_sampling_sends_every_sampled_string_to_the_pool_in_one_call(pooled: Any, prior: Path) -> None:
    calls: list[tuple[int, bool]] = []

    class _Pool:
        usable = True

        def validate(self, smiles: list[str], isomeric: bool) -> Any:
            calls.append((len(smiles), isomeric))
            return pooled.module.merge_validated([pooled.module.validate_chunk(smiles, isomeric)])

    result = NS["_ResidentSampler"]("cpu", _Pool()).sample(_job(prior, num_smiles=40))
    assert calls == [(40, False)]
    assert result == NS["_ResidentSampler"]("cpu").sample(_job(prior, num_smiles=40))


def test_unusable_pool_falls_back_to_the_in_process_path(pooled: Any, prior: Path) -> None:
    class _Dead:
        usable = False

        def validate(self, *_a: Any) -> Any:
            raise AssertionError("an unusable pool must not be called")

    assert NS["_ResidentSampler"]("cpu", _Dead()).sample(_job(prior)) == NS["_ResidentSampler"]("cpu").sample(
        _job(prior)
    )


def test_worker_failure_is_the_generic_500_then_requests_run_in_process(
    pooled: Any, prior: Path, caplog: pytest.LogCaptureFixture, monkeypatch: pytest.MonkeyPatch
) -> None:
    batch = sys.modules["reinvent.models.model_factory.sample_batch"].SampleBatch
    real_from_list = batch.from_list.__func__
    poisoned = {"on": True}  # row 3 kills the worker that validates it, while the flag is on

    def from_list(cls: Any, rows: list[Any]) -> Any:
        if poisoned["on"]:
            rows = [types.SimpleNamespace(output="DIE" if i == 3 else row.output) for i, row in enumerate(rows)]
        return real_from_list(cls, rows)

    monkeypatch.setattr(batch, "from_list", classmethod(from_list))
    pool = pooled.module.ValidationPool(2)
    pool.start()
    engine = NS["_SamplingEngine"](NS["_ResidentSampler"]("cpu", pool))
    try:
        with caplog.at_level(logging.ERROR, logger="service-under-test"):
            with pytest.raises(_InternalServerError, match="^Sampling run failed$"):
                engine.sample(_job(prior, num_smiles=16))
        assert not pool.usable
        poisoned["on"] = False
        after = engine.sample(_job(prior, num_smiles=16))  # the broken pool is skipped: in-process, still a 200
    finally:
        pool.shutdown()
    assert after == NS["_ResidentSampler"]("cpu").sample(_job(prior, num_smiles=16))


@pytest.mark.parametrize("workers", [0, 1])
def test_start_validation_pool_disabled(pool_module: Any, monkeypatch: pytest.MonkeyPatch, workers: int) -> None:
    monkeypatch.setitem(NS, "smiles_validation_pool", pool_module)
    assert NS["_start_validation_pool"](workers) is None


def test_start_validation_pool_starts_and_warms_the_pool(pool_module: Any, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setitem(NS, "smiles_validation_pool", pool_module)
    started = NS["_start_validation_pool"](2)
    try:
        assert started.usable and len(started.worker_reports) == 2
    finally:
        started.shutdown()


def test_pool_that_cannot_start_falls_back_to_in_process(
    pool_module: Any, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    monkeypatch.setitem(NS, "smiles_validation_pool", pool_module)
    monkeypatch.setenv("STUB_RDKIT_FAIL_INIT", "1")
    with caplog.at_level(logging.ERROR, logger="service-under-test"):
        assert NS["_start_validation_pool"](2) is None
    assert "runs in-process" in caplog.text


def test_service_reads_the_worker_count_from_the_environment_and_stops_the_pool_on_shutdown() -> None:
    assert "resolve_worker_count(os.environ.get(smiles_validation_pool.WORKERS_ENV))" in SERVICE_SRC
    assert "@bentoml.on_shutdown" in SERVICE_SRC and "pool.shutdown()" in SERVICE_SRC


@pytest.mark.parametrize(("raw", "expected"), [(None, 1000), ("", 1000), (" 512 ", 512), ("1024", 1024)])
def test_sample_batch_size_env_resolves(raw: str | None, expected: int) -> None:
    assert NS["resolve_sample_batch_size"](raw, 1000) == expected


@pytest.mark.parametrize("raw", ["0", "-1", "abc", "65537", "1.5"])
def test_sample_batch_size_env_rejects_bad_values(raw: str) -> None:
    with pytest.raises(RuntimeError, match="REINVENT4_SAMPLE_BATCH_SIZE"):
        NS["resolve_sample_batch_size"](raw, 1000)
