# Copyright © Advanced Micro Devices, Inc., or its affiliates.
#
# SPDX-License-Identifier: MIT

"""Parallel RDKit validation of sampled SMILES for the REINVENT4 service.

On the GPU more than half of a request is not the model but the host-side RDKit parse and
canonicalisation of the sampled SMILES (upstream's ``validate_smiles``), which is single threaded.
This module runs that step in a pool of worker processes.

* This module imports only the standard library, ``rdkit`` and ``numpy``; it never imports
  ``torch`` or ``reinvent``, so a worker holds no accelerator context and no accelerator memory
  (the pool is also started with the accelerator-visibility variables emptied, see ``ValidationPool.start``).
* ``validate_smiles`` below is a verbatim copy of ``reinvent.runmodes.samplers.sampler.validate_smiles`` of the
  pinned REINVENT4 (only ``SmilesState`` is a local stand-in with the same names and values); a test compares
  its AST with the upstream source whenever a clone is available (``REINVENT4_SRC``).
* The pool is created and warmed up before the service opens its port, with the ``spawn`` start method
  (no forked CUDA/ROCm state, no forked threads). ``shutdown`` is idempotent and also runs at exit.
* A request's SMILES are split into contiguous chunks, one per worker, and put back in the original
  order. Each chunk is validated independently, so a duplicate that spans two chunks is only known after the
  merge; ``merge_validated`` marks duplicates globally, in order, exactly as one in-process pass would.
"""

from __future__ import annotations

import atexit
import concurrent.futures
import contextlib
import logging
import multiprocessing
import os
import sys
import threading
import time
from concurrent.futures.process import BrokenProcessPool
from enum import Enum
from typing import Any, Iterator, List, Sequence, Tuple

import numpy as np
from rdkit import Chem, rdBase

# A child of the service logger ("service"), which has the stderr handler that ``docker logs`` shows.
logger = logging.getLogger("service.validation_pool")

WORKERS_ENV = "REINVENT4_VALIDATION_WORKERS"
DEFAULT_MAX_WORKERS = 8
MAX_WORKERS = 64

#: Seconds the start-up warm-up may take (all workers started, imports done).
START_TIMEOUT_SECONDS = 120.0
#: Seconds one chunk may take. The limit is generous for a healthy worker, so this only trips
#: on a wedged worker, which then makes the pool unusable instead of blocking the service for good.
CHUNK_TIMEOUT_SECONDS = 120.0

#: Environment variables emptied while the workers are spawned, so that nothing in a worker can see an accelerator.
_ACCELERATOR_ENV = ("CUDA_VISIBLE_DEVICES", "HIP_VISIBLE_DEVICES", "ROCR_VISIBLE_DEVICES")


class ValidationPoolError(Exception):
    """The worker pool could not start, or broke (a worker died or hung); the pool is unusable afterwards."""


# ---------------------------------------------------------------------------
# Validation (runs in the workers, and is the reference for the in-process path)
# ---------------------------------------------------------------------------


class SmilesState(Enum):
    """Same names and values as ``reinvent.models.model_factory.sample_batch.SmilesState``."""

    INVALID = 0
    VALID = 1
    DUPLICATE = 2


def validate_smiles(
    mols: List[Chem.Mol], smilies, isomeric: bool = False, return_original_smiles: bool = False
) -> Tuple[List, np.ndarray]:
    """Basic validation of sampled or joined SMILES

    The molecules are converted to canonical SMILES.  Each SMILES state is
    determined to be invalid, valid or duplicate.

    :mols: molecules
    :smilies: SMILES of molecules including invalid ones
    :isomeric: whether to use isomeric SMILES
    :return_original_smiles: whether to return the original SMILES instead of canonical_smiles
    :returns: validated SMILES and their states
    """

    validated_smilies = []
    smilies_states = []  # valid, invalid, duplicate
    seen_before = set()

    for mol, sampled_smiles in zip(mols, smilies):
        if mol:
            failed = Chem.SanitizeMol(mol, catchErrors=True)

            if not failed:
                canonical_smiles = Chem.MolToSmiles(mol, canonical=True, isomericSmiles=isomeric)

                # NOTE: Canonicalization above may still lead to invalid SMILES when later read
                #       with RDKit (e.g. scoring components), so we use this defensive check here
                try:
                    canonical_smiles = Chem.MolToSmiles(
                        Chem.MolFromSmiles(canonical_smiles),
                        canonical=True,
                        isomericSmiles=isomeric,
                    )
                except Exception:  # FIXME: not sure what to catch here
                    logger.warning(f"Sanitized SMILES {canonical_smiles} found to be invalid.")
                    smilies_states.append(SmilesState.INVALID)
                    validated_smilies.append(sampled_smiles)
                    continue

                if canonical_smiles in seen_before:
                    smilies_states.append(SmilesState.DUPLICATE)
                else:
                    smilies_states.append(SmilesState.VALID)

                if return_original_smiles:
                    validated_smilies.append(sampled_smiles)
                else:
                    validated_smilies.append(canonical_smiles)
                seen_before.add(canonical_smiles)
            else:
                validated_smilies.append(sampled_smiles)
                smilies_states.append(SmilesState.INVALID)

                # typically it is kekuleization that fails but log all failures
                logger.info(f"Sampled SMILES {sampled_smiles} failed to sanitize.")
        else:
            validated_smilies.append(sampled_smiles)
            smilies_states.append(SmilesState.INVALID)

    smilies_states = np.array(smilies_states)

    return validated_smilies, smilies_states


def validate_chunk(smiles: Sequence[str], isomeric: bool) -> Tuple[List[str], List[int]]:
    """Parse and validate one chunk exactly as ``ReinventSampler.sample`` does; states as plain ints."""
    mols = [Chem.MolFromSmiles(s, sanitize=False) if s else None for s in smiles]
    smilies, states = validate_smiles(mols, smiles, isomeric=isomeric)
    return smilies, [state.value for state in states]


def merge_validated(parts: Sequence[Tuple[List[str], List[int]]]) -> Tuple[List[str], List[int]]:
    """Concatenate the validated chunks, in order, and mark duplicates across chunk boundaries.

    Within a chunk ``validate_smiles`` already marked repeats of an earlier valid SMILES as DUPLICATE. A
    VALID entry whose canonical SMILES occurred in an earlier chunk becomes DUPLICATE here; INVALID entries are
    never touched and never enter the seen set, as in a single pass. The result equals one ``validate_smiles``
    call over the concatenated input.
    """
    smilies: List[str] = []
    states: List[int] = []
    seen: set[str] = set()
    invalid = SmilesState.INVALID.value
    for chunk_smilies, chunk_states in parts:
        for smi, state in zip(chunk_smilies, chunk_states):
            smilies.append(smi)
            if state == invalid:
                states.append(invalid)
                continue
            states.append(SmilesState.DUPLICATE.value if smi in seen else SmilesState.VALID.value)
            seen.add(smi)
    return smilies, states


# ---------------------------------------------------------------------------
# Worker configuration
# ---------------------------------------------------------------------------


def usable_cpus() -> int:
    """CPUs this process may run on (the container's cpuset/affinity, not the host's core count)."""
    try:
        return max(1, len(os.sched_getaffinity(0)))
    except (AttributeError, OSError):
        return max(1, os.cpu_count() or 1)


def resolve_worker_count(raw: str | None, cpus: int | None = None) -> int:
    """Number of validation workers: 0 means 'validate in-process'.

    Unset or empty: ``min(8, usable CPUs)``. ``REINVENT4_VALIDATION_WORKERS=<n>`` overrides it (0 to 64). A result of
    0 or 1 disables the pool: one worker is the in-process path plus the process hop. Anything that is not an integer
    from 0 to 64 raises ``RuntimeError`` (the service then fails startup, like an invalid ``REINVENT4_DEVICE``).
    """
    text = (raw or "").strip()
    if not text:
        count = min(DEFAULT_MAX_WORKERS, usable_cpus() if cpus is None else cpus)
    else:
        if not text.isascii() or not text.isdigit() or int(text) > MAX_WORKERS:
            raise RuntimeError(
                f"{WORKERS_ENV}={raw!r}: expected an integer from 0 to {MAX_WORKERS} (0 or 1 disables the pool)"
            )
        count = int(text)
    return 0 if count <= 1 else count


# ---------------------------------------------------------------------------
# Worker process side
# ---------------------------------------------------------------------------

_BARRIER: Any = None


def _init_worker(barrier: Any) -> None:
    """Process initialiser: quiet RDKit (sanitisation failures of sampled SMILES are expected), keep the start barrier."""
    global _BARRIER
    rdBase.DisableLog("rdApp.*")
    _BARRIER = barrier
    parent = multiprocessing.parent_process()
    if parent is not None:
        # A worker never outlives the service: if the parent is killed (no atexit, no shutdown hook) nothing
        # wakes an idle worker, so wait on the parent's sentinel and leave with it.
        threading.Thread(target=_exit_with_parent, args=(parent,), name="exit-with-parent", daemon=True).start()


def _exit_with_parent(parent: Any) -> None:
    parent.join()
    os._exit(0)


def _warm_up_task(_index: int) -> dict[str, Any]:
    """Warm-up task: meet every other worker at the barrier (so each worker runs exactly one), then parse something."""
    _BARRIER.wait(timeout=START_TIMEOUT_SECONDS)
    validate_chunk(["CCO", "c1ccccc1", "C(", ""], False)
    return {
        "pid": os.getpid(),
        "torch_imported": "torch" in sys.modules,
        "accelerator_env": {name: os.environ.get(name) for name in _ACCELERATOR_ENV},
    }


# ---------------------------------------------------------------------------
# Parent process side
# ---------------------------------------------------------------------------


@contextlib.contextmanager
def _accelerators_hidden() -> Iterator[None]:
    """Empty the accelerator-visibility variables while the workers are spawned (they inherit the environment)."""
    saved = {name: os.environ.get(name) for name in _ACCELERATOR_ENV}
    try:
        for name in _ACCELERATOR_ENV:
            os.environ[name] = ""
        yield
    finally:
        for name, value in saved.items():
            if value is None:
                os.environ.pop(name, None)
            else:
                os.environ[name] = value


class ValidationPool:
    """A fixed pool of spawn-started worker processes that validate SMILES chunks.

    ``start`` blocks until every worker is running and warm. ``validate`` is safe to call from one
    thread at a time per request stream (the service calls it from its single sampling thread). If a worker
    dies or hangs, the call raises ``ValidationPoolError``, the pool shuts itself down and ``usable`` becomes False;
    the caller then falls back to in-process validation. An exception raised by the validation code itself is
    propagated as it is and leaves the pool usable.
    """

    def __init__(self, workers: int) -> None:
        if workers < 2:
            raise ValueError("a validation pool needs at least 2 workers")
        self.workers = workers
        self.worker_reports: list[dict[str, Any]] = []  # one warm-up report per worker (pid, accelerator env, ...)
        self._executor: concurrent.futures.ProcessPoolExecutor | None = None
        self._lock = threading.Lock()
        self._closed = False

    @property
    def usable(self) -> bool:
        return self._executor is not None and not self._closed

    def start(self) -> None:
        """Spawn and warm up all workers (``ValidationPoolError`` on failure, after cleaning up)."""
        started = time.perf_counter()
        context = multiprocessing.get_context("spawn")
        barrier = context.Barrier(self.workers)
        executor = concurrent.futures.ProcessPoolExecutor(
            max_workers=self.workers, mp_context=context, initializer=_init_worker, initargs=(barrier,)
        )
        self._executor = executor
        try:
            with _accelerators_hidden():
                # One task per worker, all waiting for each other at the barrier: the executor has to start
                # `workers` distinct processes for them to finish, so a return means every worker is up and warm.
                futures = [executor.submit(_warm_up_task, i) for i in range(self.workers)]
                reports = [future.result(timeout=START_TIMEOUT_SECONDS + 30) for future in futures]
        except Exception as exc:
            self.shutdown()
            raise ValidationPoolError(f"validation pool failed to start: {type(exc).__name__}: {exc}") from None
        self.worker_reports = reports
        if len({report["pid"] for report in reports}) != self.workers:
            self.shutdown()
            raise ValidationPoolError("validation pool failed to start: workers did not all start")
        if any(report["torch_imported"] for report in reports):
            logger.warning("validation workers imported torch (no accelerator context is created, but memory is used)")
        atexit.register(self.shutdown)
        logger.info("validation pool ready: %d workers (spawn) in %.2f s", self.workers, time.perf_counter() - started)

    def validate(self, smiles: Sequence[str], isomeric: bool) -> Tuple[List[str], List[int]]:
        """Validate ``smiles`` like ``validate_smiles`` over the whole list; results in input order."""
        executor = self._executor
        if executor is None or self._closed:
            raise ValidationPoolError("validation pool is not running")
        if not smiles:
            return [], []
        size = -(-len(smiles) // self.workers)
        chunks = [smiles[i : i + size] for i in range(0, len(smiles), size)]
        futures: list[concurrent.futures.Future[Any]] = []
        try:
            try:
                futures = [executor.submit(validate_chunk, chunk, isomeric) for chunk in chunks]
            except RuntimeError:  # the executor was shut down or broke between the check above and the submit
                raise BrokenProcessPool("pool not accepting work") from None
            parts = [future.result(timeout=CHUNK_TIMEOUT_SECONDS) for future in futures]
        except (BrokenProcessPool, concurrent.futures.TimeoutError) as exc:
            # A worker died or hung. An exception raised by the validation code itself is not caught here:
            # it propagates as it is and leaves the pool usable.
            logger.error("validation pool broke (%s: %s); validating in-process from now on", type(exc).__name__, exc)
            self.shutdown()
            raise ValidationPoolError("validation pool failed") from None
        except BaseException:
            for future in futures:
                future.cancel()
            raise
        return merge_validated(parts)

    def shutdown(self) -> None:
        """Stop the workers and reap them (idempotent; kills a worker that does not stop within a few seconds)."""
        with self._lock:
            if self._closed:
                return
            self._closed = True
            executor, self._executor = self._executor, None
        with contextlib.suppress(Exception):
            atexit.unregister(self.shutdown)
        if executor is None:
            return
        # ``_processes`` is private, but it is the only way to reap workers that ignore the shutdown request.
        processes = list((getattr(executor, "_processes", None) or {}).values())
        executor.shutdown(wait=False, cancel_futures=True)
        deadline = time.monotonic() + 5.0
        for process in processes:
            process.join(max(0.0, deadline - time.monotonic()))
        for process in processes:
            if process.is_alive():
                process.terminate()
                process.join(2.0)
            if process.is_alive():
                process.kill()
                process.join(2.0)
        logger.info("validation pool stopped")
