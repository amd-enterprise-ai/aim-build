# Copyright © Advanced Micro Devices, Inc., or its affiliates.
#
# SPDX-License-Identifier: MIT

"""The parallel SMILES-validation pool (``src/smiles_validation_pool.py``), with a stub rdkit.

The pool starts real ``spawn`` worker processes; ``conftest.py`` puts a deterministic stub ``rdkit`` on
``sys.path`` (the workers inherit it), so no RDKit, torch or GPU is needed. The stub understands a few magic
SMILES: ``BOOM`` raises in the worker, ``DIE`` kills the worker process, ``HANG`` blocks it, ``(...`` does not parse,
``X...`` fails sanitisation, ``QQ...`` cannot be re-read after canonicalisation; canonical form is upper case.
"""

from __future__ import annotations

import ast
import multiprocessing
import os
import random
import signal
import subprocess
import sys
import textwrap
import time
from pathlib import Path
from types import ModuleType
from typing import Any, Iterator

import pytest

SRC = Path(__file__).resolve().parents[1] / "src/smiles_validation_pool.py"

ALPHABET = ["cco", "CCO", "c1ccccc1", "C1CCCCC1", "XBAD", "(", "", "QQ1", "N#N", "cc(C)O", "o=c", "CCN"]


def _random_smiles(rng: random.Random, n: int) -> list[str]:
    return [rng.choice(ALPHABET) for _ in range(n)]


def _reference(module: ModuleType, smiles: list[str], isomeric: bool) -> tuple[list[str], list[int]]:
    """What one in-process pass (upstream's ReinventSampler.sample tail) gives."""
    mols = [module.Chem.MolFromSmiles(s, sanitize=False) if s else None for s in smiles]
    smilies, states = module.validate_smiles(mols, smiles, isomeric=isomeric)
    return smilies, [state.value for state in states]


@pytest.fixture
def pool(pool_module: ModuleType) -> Iterator[Any]:
    started = pool_module.ValidationPool(3)
    started.start()
    try:
        yield started
    finally:
        started.shutdown()


def _alive(pids: list[int]) -> list[int]:
    alive = []
    for pid in pids:
        try:
            os.kill(pid, 0)
        except ProcessLookupError:
            continue
        # a zombie still answers kill(0); a reaped child does not exist at all
        alive.append(pid)
    return alive


# --- configuration -------------------------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("raw", "cpus", "expected"),
    [
        (None, 48, 8),
        ("", 48, 8),
        (None, 4, 4),
        (None, 2, 2),
        (None, 1, 0),
        ("0", 48, 0),
        ("1", 48, 0),
        ("2", 1, 2),
        ("12", 4, 12),
        (" 5 ", 4, 5),
        ("64", 4, 64),
    ],
)
def test_worker_count(pool_module: ModuleType, raw: str | None, cpus: int, expected: int) -> None:
    assert pool_module.resolve_worker_count(raw, cpus) == expected


@pytest.mark.parametrize("raw", ["abc", "-1", "2.5", "65", "1e1", "٣", "0x4"])
def test_invalid_worker_count_names_the_variable(pool_module: ModuleType, raw: str) -> None:
    with pytest.raises(RuntimeError, match="REINVENT4_VALIDATION_WORKERS"):
        pool_module.resolve_worker_count(raw, 8)


def test_default_uses_the_cpu_affinity_not_the_host_core_count(
    pool_module: ModuleType, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(os, "sched_getaffinity", lambda _pid: {0, 1, 2}, raising=False)
    monkeypatch.setattr(os, "cpu_count", lambda: 96)
    assert pool_module.usable_cpus() == 3
    assert pool_module.resolve_worker_count(None) == 3
    monkeypatch.delattr(os, "sched_getaffinity")
    assert pool_module.usable_cpus() == 96  # no affinity API: the core count
    assert pool_module.resolve_worker_count(None) == 8


def test_env_name_and_default_cap(pool_module: ModuleType) -> None:
    assert pool_module.WORKERS_ENV == "REINVENT4_VALIDATION_WORKERS"
    assert pool_module.DEFAULT_MAX_WORKERS == 8


def test_a_pool_needs_two_workers(pool_module: ModuleType) -> None:
    with pytest.raises(ValueError):
        pool_module.ValidationPool(1)


# --- equality with the in-process path -----------------------------------------------------------------------


def test_merge_validated_equals_one_pass_for_any_split(pool_module: ModuleType) -> None:
    rng = random.Random(7)
    for _ in range(200):
        smiles = _random_smiles(rng, rng.randint(0, 40))
        cuts = sorted(rng.sample(range(len(smiles) + 1), k=min(len(smiles) + 1, rng.randint(0, 5))))
        bounds = [0, *cuts, len(smiles)]
        parts = [
            pool_module.validate_chunk(smiles[a:b], False) for a, b in zip(bounds, bounds[1:], strict=False) if a < b
        ]
        assert pool_module.merge_validated(parts) == _reference(pool_module, smiles, False)


@pytest.mark.parametrize("isomeric", [False, True])
@pytest.mark.parametrize("size", [0, 1, 2, 3, 7, 50, 511])
def test_pool_equals_in_process(pool_module: ModuleType, pool: Any, size: int, isomeric: bool) -> None:
    smiles = _random_smiles(random.Random(size), size)
    assert pool.validate(smiles, isomeric) == _reference(pool_module, smiles, isomeric)


def test_duplicates_across_chunk_boundaries_are_marked_in_order(pool_module: ModuleType, pool: Any) -> None:
    smiles = ["cco", "CCO", "c1ccccc1", "cco", "C1CCCCC1", "c1ccccc1", "CCO", "XBAD", "(", ""]
    smilies, states = pool.validate(smiles, False)
    assert (smilies, states) == _reference(pool_module, smiles, False)
    valid, invalid, duplicate = (pool_module.SmilesState[name].value for name in ("VALID", "INVALID", "DUPLICATE"))
    # the stub canonicalises by upper-casing, so "C1CCCCC1" repeats "c1ccccc1"
    assert states == [valid, duplicate, valid, duplicate, duplicate, duplicate, duplicate, invalid, invalid, invalid]


def test_state_stand_in_matches_upstream_values(pool_module: ModuleType) -> None:
    assert {s.name: s.value for s in pool_module.SmilesState} == {"INVALID": 0, "VALID": 1, "DUPLICATE": 2}


def test_validate_smiles_is_a_verbatim_copy_of_upstream() -> None:
    """With a REINVENT4 clone (``REINVENT4_SRC``), our copy and upstream's function have the same AST."""
    clone = os.environ.get("REINVENT4_SRC")
    upstream = Path(clone or "/nonexistent") / "reinvent/runmodes/samplers/sampler.py"
    if not upstream.is_file():
        pytest.skip("REINVENT4_SRC not set (no offline upstream clone)")

    def function(path: Path) -> str:
        tree = ast.parse(path.read_text(encoding="utf-8"))
        node = next(n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef) and n.name == "validate_smiles")
        return ast.dump(node)

    assert function(SRC) == function(upstream)


# --- life cycle ----------------------------------------------------------------------------------------------


def test_start_warms_every_worker_without_accelerators(pool_module: ModuleType, pool: Any) -> None:
    reports = pool.worker_reports
    assert len({r["pid"] for r in reports}) == 3 and os.getpid() not in {r["pid"] for r in reports}
    assert all(r["torch_imported"] is False for r in reports)
    assert all(
        r["accelerator_env"] == {"CUDA_VISIBLE_DEVICES": "", "HIP_VISIBLE_DEVICES": "", "ROCR_VISIBLE_DEVICES": ""}
        for r in reports
    )
    assert pool.usable


def test_accelerator_environment_of_the_parent_is_restored(
    pool_module: ModuleType, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "1")
    monkeypatch.delenv("HIP_VISIBLE_DEVICES", raising=False)
    started = pool_module.ValidationPool(2)
    started.start()
    try:
        assert os.environ["CUDA_VISIBLE_DEVICES"] == "1"
        assert "HIP_VISIBLE_DEVICES" not in os.environ
    finally:
        started.shutdown()


def test_shutdown_reaps_all_workers_and_is_idempotent(pool_module: ModuleType) -> None:
    started = pool_module.ValidationPool(3)
    started.start()
    pids = [r["pid"] for r in started.worker_reports]
    started.validate(["CCO"] * 10, False)
    started.shutdown()
    started.shutdown()
    assert not started.usable
    assert _alive(pids) == []
    assert multiprocessing.active_children() == []
    with pytest.raises(pool_module.ValidationPoolError):
        started.validate(["CCO"], False)


def test_start_failure_is_an_error_and_leaves_no_process(
    pool_module: ModuleType, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("STUB_RDKIT_FAIL_INIT", "1")
    failing = pool_module.ValidationPool(2)
    with pytest.raises(pool_module.ValidationPoolError, match="failed to start"):
        failing.start()
    assert not failing.usable
    assert multiprocessing.active_children() == []


# --- failures ------------------------------------------------------------------------------------------------


def test_exception_in_a_worker_propagates_and_the_pool_keeps_working(pool_module: ModuleType, pool: Any) -> None:
    with pytest.raises(ValueError, match="boom from the worker"):
        pool.validate(["CCO", "BOOM", "CCN", "cco", "c1ccccc1", "CCO"], False)
    assert pool.usable
    assert pool.validate(["cco", "CCO"], False) == _reference(pool_module, ["cco", "CCO"], False)


def test_a_dead_worker_makes_the_pool_unusable_and_everything_is_reaped(pool_module: ModuleType) -> None:
    broken = pool_module.ValidationPool(3)
    broken.start()
    pids = [r["pid"] for r in broken.worker_reports]
    with pytest.raises(pool_module.ValidationPoolError, match="validation pool failed"):
        broken.validate(["CCO", "DIE", "CCN", "cco", "c1ccccc1", "CCO"], False)
    assert not broken.usable
    assert _alive(pids) == []
    with pytest.raises(pool_module.ValidationPoolError):
        broken.validate(["CCO"], False)


def test_a_hung_worker_times_out_and_is_killed(pool_module: ModuleType, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(pool_module, "CHUNK_TIMEOUT_SECONDS", 1.0)
    hung = pool_module.ValidationPool(2)
    hung.start()
    pids = [r["pid"] for r in hung.worker_reports]
    started = time.monotonic()
    with pytest.raises(pool_module.ValidationPoolError):
        hung.validate(["CCO", "HANG", "CCN", "cco"], False)
    assert time.monotonic() - started < 30
    assert not hung.usable
    assert _alive(pids) == []


# --- process exit --------------------------------------------------------------------------------------------


def _run_script(stub_dir: Path, body: str) -> subprocess.CompletedProcess[str]:
    header = (
        "import sys\n"
        f"sys.path[:0] = [{str(stub_dir)!r}, {str(SRC.parent)!r}]\n"
        "import smiles_validation_pool as m\n\n"
        'if __name__ == "__main__":\n'
    )
    script = header + textwrap.indent(textwrap.dedent(body), "    ")
    return subprocess.run(
        [sys.executable, "-I", "-c", script], capture_output=True, text=True, timeout=120, check=False
    )


def test_exit_after_explicit_shutdown_is_silent(pool_module: ModuleType, stub_rdkit_dir: Path) -> None:
    result = _run_script(
        stub_rdkit_dir,
        """
        p = m.ValidationPool(2)
        p.start()
        assert p.validate(["cco", "CCO"], False)[1] == [1, 2]
        p.shutdown()
        print("done")
        """,
    )
    assert (result.returncode, result.stdout.strip(), result.stderr) == (0, "done", "")


def test_exit_without_shutdown_is_silent_and_leaves_no_worker(pool_module: ModuleType, stub_rdkit_dir: Path) -> None:
    """The atexit hook stops the pool: no 'NoneType' has no attribute 'dumps' noise, no orphan process."""
    result = _run_script(
        stub_rdkit_dir,
        """
        p = m.ValidationPool(2)
        p.start()
        assert p.validate(["cco", "CCO"], False)[1] == [1, 2]
        print(*[r["pid"] for r in p.worker_reports])
        """,
    )
    assert result.returncode == 0 and result.stderr == "", result.stderr
    assert _alive([int(pid) for pid in result.stdout.split()]) == []


def test_workers_do_not_outlive_a_killed_parent(pool_module: ModuleType, stub_rdkit_dir: Path) -> None:
    script = textwrap.dedent(f"""
        import os, signal, sys
        sys.path[:0] = [{str(stub_rdkit_dir)!r}, {str(SRC.parent)!r}]
        import smiles_validation_pool as m

        if __name__ == "__main__":
            p = m.ValidationPool(2)
            p.start()
            print(*[r["pid"] for r in p.worker_reports], flush=True)
            os.kill(os.getpid(), signal.SIGKILL)
        """)
    result = subprocess.run(
        [sys.executable, "-I", "-c", script], capture_output=True, text=True, timeout=120, check=False
    )
    pids = [int(pid) for pid in result.stdout.split()]
    assert result.returncode == -signal.SIGKILL and len(pids) == 2
    deadline = time.monotonic() + 15
    while _alive(pids) and time.monotonic() < deadline:
        time.sleep(0.2)
    assert _alive(pids) == []
