# Copyright © Advanced Micro Devices, Inc., or its affiliates.
#
# SPDX-License-Identifier: MIT

"""Behaviour of the resident-model engine in service.py, with stubs (no GPU, no torch, no bentoml).

service.py needs bentoml and torch at import time, so the pieces under test are extracted by AST
and executed in a namespace with stand-ins for those modules.
"""

from __future__ import annotations
import __future__

import ast
import concurrent.futures
import contextlib
import io
import logging
import os
import sys
import tempfile
import threading
import time
import types
from pathlib import Path
from typing import Any

import pytest

SERVICE_SRC = (Path(__file__).resolve().parents[1] / "src/service.py").read_text(encoding="utf-8")

WANTED = {
    "REQUEST_TIMEOUT_SECONDS",
    "OUTPUT_TAIL_LINES",
    "DEVICE_ENV",
    "DEFAULT_DEVICE",
    "_DEVICE_RE",
    "_RUN_ARG_DEFAULTS",
    "_resolve_device",
    "_accelerator_count_error",
    "_run_args",
    "_warmup_pdb_text",
    "DesignError",
    "_ResidentRunner",
    "_DesignEngine",
    "_read_designed_sequences",
}


class _ServiceUnavailable(Exception):
    pass


class _BentoMLException(Exception):
    pass


class _PredictResponse:
    def __init__(self, sequences: list[str], scores: list[float]) -> None:
        self.sequences = sequences
        self.scores = scores


def _parse_output_stub(out_dir: str) -> _PredictResponse:
    """Stand-in for service._parse_output that needs no numpy: the first FASTA's designed records."""
    fasta = sorted(Path(out_dir, "seqs").glob("*.fa"))
    seqs = _NS["_read_designed_sequences"](str(fasta[0])) if fasta else []
    return _PredictResponse(seqs, [0.0] * len(seqs))


def _build_namespace() -> dict[str, Any]:
    import re

    tree = ast.parse(SERVICE_SRC)
    body = []
    for node in tree.body:
        names = set()
        if isinstance(node, (ast.FunctionDef, ast.ClassDef)):
            names = {node.name}
        elif isinstance(node, ast.Assign):
            names = {t.id for t in node.targets if isinstance(t, ast.Name)}
        elif isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
            names = {node.target.id}
        if names & WANTED:
            body.append(node)
    ns: dict[str, Any] = {
        "__name__": "service_under_test",
        "concurrent": concurrent,
        "contextlib": contextlib,
        "io": io,
        "os": os,
        "re": re,
        "sys": sys,
        "tempfile": tempfile,
        "time": time,
        "types": types,
        "Path": Path,
        "Any": Any,
        "Iterator": Any,
        "logger": logging.getLogger("service-under-test"),
        "ServiceUnavailable": _ServiceUnavailable,
        "BentoMLException": _BentoMLException,
        "PredictResponse": _PredictResponse,
        "_parse_output": _parse_output_stub,
        "ALLOWED_MODEL_NAMES": ("v_48_002", "v_48_010", "v_48_020", "v_48_030"),
        "WEIGHTS_DIR": Path("/cache/proteinmpnn/vanilla_model_weights"),
        "PROTEINMPNN_PATH": "/opt/proteinmpnn",
    }
    module = ast.Module(body=body, type_ignores=[])
    code = compile(
        ast.fix_missing_locations(module), "service_helpers", "exec", flags=__future__.annotations.compiler_flag
    )
    exec(code, ns)  # noqa: S102
    return ns


_NS = _build_namespace()


def _torch_stub(available: bool = True) -> types.SimpleNamespace:
    return types.SimpleNamespace(cuda=types.SimpleNamespace(is_available=lambda: available))


# ---- device selection ------------------------------------------------------------------------


def test_default_device_is_gpu(monkeypatch) -> None:
    monkeypatch.delenv("PROTEINMPNN_DEVICE", raising=False)
    assert _NS["_resolve_device"](_torch_stub(True)) == "cuda"


def test_no_gpu_fails_instead_of_falling_back(monkeypatch) -> None:
    monkeypatch.delenv("PROTEINMPNN_DEVICE", raising=False)
    with pytest.raises(RuntimeError, match="No GPU visible.*PROTEINMPNN_DEVICE=cpu"):
        _NS["_resolve_device"](_torch_stub(False))


def test_cpu_only_when_requested(monkeypatch, caplog) -> None:
    monkeypatch.setenv("PROTEINMPNN_DEVICE", "CPU")
    _NS["logger"].propagate = True
    with caplog.at_level(logging.WARNING, logger="service-under-test"):
        assert _NS["_resolve_device"](_torch_stub(False)) == "cpu"
    assert "CPU" in caplog.text


@pytest.mark.parametrize("value", ["cuda", "cuda:0"])
def test_gpu_spellings(monkeypatch, value) -> None:
    monkeypatch.setenv("PROTEINMPNN_DEVICE", value)
    assert _NS["_resolve_device"](_torch_stub(True)) == "cuda"


@pytest.mark.parametrize("value", ["cuda:1", "gpu", "cpu:0", "mps", "cuda:"])
def test_invalid_device_values_fail(monkeypatch, value) -> None:
    monkeypatch.setenv("PROTEINMPNN_DEVICE", value)
    with pytest.raises(RuntimeError, match="not valid"):
        _NS["_resolve_device"](_torch_stub(True))


def test_missing_pinned_gpu_names_the_accelerator_count(monkeypatch) -> None:
    """A worker narrowed (HIP/CUDA_VISIBLE_DEVICES) to a GPU id the container lacks: clear startup error."""
    monkeypatch.delenv("PROTEINMPNN_DEVICE", raising=False)
    with pytest.raises(
        RuntimeError, match=r"AIM_ACCELERATOR_COUNT=2 requires 2 accelerator\(s\), but GPU 1 .*not present"
    ):
        _NS["_resolve_device"](_torch_stub(False), "1", 2)


def test_pinned_gpu_that_exists_resolves_to_cuda(monkeypatch) -> None:
    monkeypatch.delenv("PROTEINMPNN_DEVICE", raising=False)
    assert _NS["_resolve_device"](_torch_stub(True), "1", 2) == "cuda"


# ---- arguments handed to upstream main() -----------------------------------------------------


def test_run_args_match_the_old_command_line() -> None:
    args = _NS["_run_args"](
        pdb_path="/workspace/data/a.pdb", out_dir="/tmp/o", num_seq=8, sampling_temp=0.25, model_name="v_48_010", seed=5
    )
    assert args.pdb_path == "/workspace/data/a.pdb"
    assert args.out_folder == "/tmp/o"
    assert args.num_seq_per_target == 8
    assert args.sampling_temp == "0.25"  # upstream splits this string into temperatures
    assert args.model_name == "v_48_010"
    assert args.path_to_model_weights == "/cache/proteinmpnn/vanilla_model_weights" + os.sep
    assert args.seed == 5
    assert args.save_score == 1
    assert args.batch_size == 1
    # everything else stays at upstream's argparse defaults
    assert args.ca_only is False and args.omit_AAs == "X" and args.max_length == 200000
    assert args.save_probs == 0 and args.score_only == 0 and args.backbone_noise == 0.0
    assert args.jsonl_path is None and args.chain_id_jsonl == "" and args.tied_positions_jsonl == ""


def test_run_args_cover_every_upstream_option() -> None:
    """The names upstream's argparse defines at the pinned commit; main() reads each of them."""
    upstream = {
        "ca_only", "path_to_model_weights", "model_name", "seed", "save_score", "save_probs", "score_only",
        "conditional_probs_only", "conditional_probs_only_backbone", "unconditional_probs_only", "backbone_noise",
        "num_seq_per_target", "batch_size", "max_length", "sampling_temp", "out_folder", "pdb_path",
        "pdb_path_chains", "jsonl_path", "chain_id_jsonl", "fixed_positions_jsonl", "omit_AAs", "bias_AA_jsonl",
        "bias_by_res_jsonl", "omit_AA_jsonl", "pssm_jsonl", "pssm_multi", "pssm_threshold", "pssm_log_odds_flag",
        "pssm_bias_flag", "tied_positions_jsonl",
    }  # fmt: skip
    assert set(_NS["_RUN_ARG_DEFAULTS"]) == upstream


def test_warmup_pdb_is_a_valid_backbone() -> None:
    text = _NS["_warmup_pdb_text"](10)
    atoms = [ln for ln in text.splitlines() if ln.startswith("ATOM")]
    assert len(atoms) == 40
    assert {ln[12:16].strip() for ln in atoms} == {"N", "CA", "C", "O"}
    for ln in atoms:  # fixed PDB columns: x 30-38, y 38-46, z 46-54
        float(ln[30:38]), float(ln[38:46]), float(ln[46:54])
    assert text.rstrip().endswith("END")


# ---- resident runner: what is swapped around upstream main() ---------------------------------


@pytest.fixture
def stub_upstream(monkeypatch):
    """Fake torch, protein_mpnn_utils and protein_mpnn_run in sys.modules; main() records what it saw."""
    calls: list[dict[str, Any]] = []
    loads: list[str] = []

    def real_load(path, *a, **k):
        loads.append(str(path))
        return {"num_edges": 48, "path": str(path)}

    torch = types.ModuleType("torch")
    torch.load = real_load  # type: ignore[attr-defined]
    torch.cuda = types.SimpleNamespace(is_available=lambda: True)  # type: ignore[attr-defined]

    constructed: list[int] = []

    class ProteinMPNN:
        def __init__(self, **kw):
            constructed.append(1)

    utils = types.ModuleType("protein_mpnn_utils")
    utils.ProteinMPNN = ProteinMPNN  # type: ignore[attr-defined]
    utils.helper = lambda: "helper"  # type: ignore[attr-defined]

    def main(args):
        import protein_mpnn_utils as seen  # what upstream's `from protein_mpnn_utils import ...` resolves
        import torch as t

        print("Generating sequences for: x")
        ckpt = t.load(args.path_to_model_weights + args.model_name + ".pt", map_location="cuda:0")
        model = seen.ProteinMPNN(k_neighbors=ckpt["num_edges"])
        calls.append({"model": model, "ckpt": ckpt, "cuda": t.cuda.is_available(), "helper": seen.helper()})
        if getattr(args, "boom", False):
            raise ValueError("boom")

    run = types.ModuleType("protein_mpnn_run")
    run.main = main  # type: ignore[attr-defined]
    for name, mod in (("torch", torch), ("protein_mpnn_utils", utils), ("protein_mpnn_run", run)):
        monkeypatch.setitem(sys.modules, name, mod)
    monkeypatch.setattr(sys, "path", list(sys.path))
    return types.SimpleNamespace(calls=calls, loads=loads, constructed=constructed, torch=torch, utils=utils)


def _args(model_name="v_48_020", **extra):
    ns = _NS["_run_args"](pdb_path="/x.pdb", out_dir="/o", num_seq=1, sampling_temp=0.1, model_name=model_name, seed=1)
    for k, v in extra.items():
        setattr(ns, k, v)
    return ns


def test_model_is_built_once_and_shared_by_all_weight_files(stub_upstream) -> None:
    runner = _NS["_ResidentRunner"]("cuda")
    runner.load()
    for name in ("v_48_002", "v_48_020", "v_48_002", "v_48_030"):
        runner.run(_args(name), io.StringIO())
    assert len(stub_upstream.constructed) == 1
    assert len({id(c["model"]) for c in stub_upstream.calls}) == 1
    assert all(c["helper"] == "helper" for c in stub_upstream.calls)  # rest of the utils module is intact


def test_each_checkpoint_is_read_from_disk_once(stub_upstream) -> None:
    runner = _NS["_ResidentRunner"]("cuda")
    runner.load()
    for _ in range(3):
        runner.run(_args("v_48_010"), io.StringIO())
    runner.run(_args("v_48_020"), io.StringIO())
    assert [Path(p).name for p in stub_upstream.loads] == ["v_48_010.pt", "v_48_020.pt"]


def test_everything_is_restored_after_a_run_even_a_failing_one(stub_upstream) -> None:
    real_load = stub_upstream.torch.load
    real_is_available = stub_upstream.torch.cuda.is_available
    runner = _NS["_ResidentRunner"]("cpu")
    runner.load()
    with pytest.raises(ValueError, match="boom"):
        runner.run(_args(boom=True), io.StringIO())
    assert stub_upstream.torch.load is real_load
    assert stub_upstream.torch.cuda.is_available is real_is_available
    assert sys.modules["protein_mpnn_utils"] is stub_upstream.utils


def test_cpu_hides_the_gpu_from_main_only_during_the_run(stub_upstream) -> None:
    cpu = _NS["_ResidentRunner"]("cpu")
    cpu.load()
    cpu.run(_args(), io.StringIO())
    gpu = _NS["_ResidentRunner"]("cuda")
    gpu.load()
    gpu.run(_args(), io.StringIO())
    assert [c["cuda"] for c in stub_upstream.calls] == [False, True]
    assert stub_upstream.torch.cuda.is_available() is True


def test_upstream_stdout_is_captured_not_printed(stub_upstream, capsys) -> None:
    runner = _NS["_ResidentRunner"]("cuda")
    runner.load()
    buf = io.StringIO()
    runner.run(_args(), buf)
    assert "Generating sequences" in buf.getvalue()
    assert capsys.readouterr().out == ""


def test_proteinmpnn_path_is_put_on_sys_path(stub_upstream) -> None:
    runner = _NS["_ResidentRunner"]("cuda")
    runner.load()
    assert "/opt/proteinmpnn" in sys.path


# ---- engine: serialisation, errors, timeouts -------------------------------------------------

FASTA = ">x, score=1\nAAAA\n>T=0.1, sample=1, score=1\nMKV\n>T=0.1, sample=2, score=1\nMKL\n"


class FakeRunner:
    """Stands in for _ResidentRunner: writes a FASTA like upstream, optionally blocks or fails."""

    device = "cuda"

    def __init__(self, fasta: str = FASTA, fail: bool = False) -> None:
        self.fasta = fasta
        self.fail = fail
        self.gate: threading.Event | None = None
        self.started = threading.Event()
        self.active = 0
        self.max_active = 0
        self.runs: list[str] = []
        self.lock = threading.Lock()
        self.loaded = False

    def load(self) -> None:
        self.loaded = True

    def run(self, args, captured) -> None:
        with self.lock:
            self.active += 1
            self.max_active = max(self.max_active, self.active)
            self.runs.append(args.pdb_path)
        try:
            print("some upstream output", file=captured)
            self.started.set()
            if self.gate is not None:
                self.gate.wait(30)
            if self.fail:
                raise RuntimeError("secret path /workspace/data/leak.pdb")
            seqs = Path(args.out_folder, "seqs")
            seqs.mkdir()
            (seqs / "x.fa").write_text(self.fasta)
        finally:
            with self.lock:
                self.active -= 1


def _engine(runner: FakeRunner):
    return _NS["_DesignEngine"](runner)


def test_design_returns_only_designed_sequences() -> None:
    engine = _engine(FakeRunner())
    response = engine.design(_args())
    assert response.sequences == ["MKV", "MKL"]


def test_requests_run_one_at_a_time_in_arrival_order() -> None:
    runner = FakeRunner()
    engine = _engine(runner)
    results: list[Any] = []

    def call(i: int) -> None:
        results.append(engine.design(_args(pdb_path=f"/p{i}.pdb")))

    threads = [threading.Thread(target=call, args=(i,)) for i in range(6)]
    for t in threads:
        t.start()
    for t in threads:
        t.join(30)
    assert len(results) == 6
    assert runner.max_active == 1


def test_failed_run_is_a_generic_500_and_the_detail_stays_in_the_log(caplog) -> None:
    engine = _engine(FakeRunner(fail=True))
    _NS["logger"].propagate = True
    with caplog.at_level(logging.ERROR, logger="service-under-test"):
        with pytest.raises(_BentoMLException) as info:
            engine.design(_args())
    assert str(info.value) == "Sequence design failed"
    assert "leak.pdb" in caplog.text  # logged for the operator
    assert "leak.pdb" not in str(info.value)  # not returned to the client


def test_no_designed_sequences_is_a_500() -> None:
    engine = _engine(FakeRunner(fasta=">x, score=1\nAAAA\n"))
    with pytest.raises(_BentoMLException, match="produced no output"):
        engine.design(_args())


def test_engine_keeps_working_after_a_failed_run() -> None:
    runner = FakeRunner(fail=True)
    engine = _engine(runner)
    with pytest.raises(_BentoMLException):
        engine.design(_args())
    runner.fail = False
    assert engine.design(_args()).sequences == ["MKV", "MKL"]


def test_running_request_that_times_out_returns_503_and_blocks_new_requests_until_it_ends() -> None:
    runner = FakeRunner()
    runner.gate = threading.Event()
    engine = _engine(runner)
    with pytest.raises(_ServiceUnavailable, match="exceeded 0.3 seconds"):
        engine.design(_args(), timeout=0.3)
    assert runner.started.is_set()  # it was running: cannot be cancelled, only waited out
    t0 = time.monotonic()
    with pytest.raises(_ServiceUnavailable, match="still running"):
        engine.design(_args(), timeout=30)
    assert time.monotonic() - t0 < 1.0  # answered at once, not after a wait
    runner.gate.set()  # the stuck run ends on its own
    deadline = time.monotonic() + 10
    while engine._abandoned and time.monotonic() < deadline:
        time.sleep(0.02)
    assert not engine._abandoned
    assert engine.design(_args(), timeout=10).sequences == ["MKV", "MKL"]


def test_queued_request_that_times_out_is_cancelled_and_never_runs() -> None:
    runner = FakeRunner()
    runner.gate = threading.Event()
    engine = _engine(runner)
    first: list[Any] = []
    t = threading.Thread(target=lambda: first.append(engine.design(_args(pdb_path="/first.pdb"), timeout=30)))
    t.start()
    assert runner.started.wait(10)
    with pytest.raises(_ServiceUnavailable, match="did not start within 0.3 seconds"):
        engine.design(_args(pdb_path="/second.pdb"), timeout=0.3)
    assert not engine._abandoned  # nothing is stuck: the first request is just slow
    runner.gate.set()
    t.join(10)
    assert first and first[0].sequences == ["MKV", "MKL"]
    assert runner.runs == ["/first.pdb"]  # the cancelled one never reached the model


def test_warm_up_loads_and_runs_every_weight_file(tmp_path) -> None:
    runner = FakeRunner()
    engine = _engine(runner)
    engine.warm_up()
    assert runner.loaded
    assert len(runner.runs) == 4  # one design per weight file


def test_warm_up_failure_is_a_runtime_error_naming_the_device() -> None:
    engine = _engine(FakeRunner(fail=True))
    with pytest.raises(RuntimeError, match="warm-up failed on device cuda"):
        engine.warm_up()


def test_request_timeout_is_ten_minutes() -> None:
    assert _NS["REQUEST_TIMEOUT_SECONDS"] == 600
