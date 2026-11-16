# Copyright © Advanced Micro Devices, Inc., or its affiliates.
#
# SPDX-License-Identifier: MIT

"""Start-up warm-up prediction for the TorchInductor-compiled Boltz-2 modules.

The first prediction after start-up pays all Dynamo/Inductor compiles of the
wrapped modules (about 100 s). :func:`run_warmup` runs synthetic predictions
before the service accepts requests, so no user request pays them.

The warm-up goes through the same ``run_boltz2_prediction`` path as a user
request (same precision, autocast and recipe). It needs no network: the inputs
use the empty MSA (``msa: empty``) or a synthetic A3M file written to a temporary
directory. Its result is discarded.

After the predictions, the service runs extra steps for compiles that no
prediction input reaches (see ``run_warmup``).

Four inputs are used by default: a size on the plain Transition path (at most
``CHUNK_SIZE_THRESHOLD`` tokens) and a size on the chunked path
(``boltz.data.const.chunk_size_threshold`` is 384; above it Boltz runs the
Transition layers chunked, which is a different graph), each with the empty MSA
and with a multi-row MSA.

The compile counters of ``compilation.compile_status`` are cumulative and count
the warm-up. The warm-up records a baseline, so the report can separate compiles
of the warm-up from compiles caused by user traffic.
"""

from __future__ import annotations

import logging
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

logger = logging.getLogger(__name__)

# Boltz runs Transition chunked when the token count is above this value
# (boltz.data.const.chunk_size_threshold).
CHUNK_SIZE_THRESHOLD = 384

# One input on the plain path, one on the chunked path.
WARMUP_TOKENS = (128, 400)

# Each size runs with the empty MSA (one row, the single-sequence path) and with a
# multi-row MSA. Dynamo gives a dimension of size 1 its own graph, so the MSA row
# count 1 and any count above 1 are different graphs; a user request with an
# MSA would otherwise compile them.
WARMUP_MSA_ROWS = (1, 3)

# Compiled graphs do not depend on the step count, so the warm-up keeps the
# diffusion short. Recycling keeps the default: the trunk graphs are the
# expensive ones and the first recycling pass differs from the later ones.
WARMUP_SAMPLING_STEPS = 2
WARMUP_RECYCLING_STEPS = 3

# Affinity sampling steps of the warm-up when the affinity model is compiled: the score-model CUDA graph
# captures after an eager call that did not compile (see score_graph). Three steps compile, then capture.
AFFINITY_COMPILE_WARMUP_SAMPLING_STEPS = 3

# Affinity warm-up complex: a protein of this many residues and aspirin as the binder.
AFFINITY_WARMUP_TOKENS = 100
AFFINITY_WARMUP_SMILES = "CC(=O)OC1=CC=CC=C1C(=O)O"

# Second affinity warm-up complex, run when the affinity model is compiled: a protein longer than the 200 residues
# that the affinity cropper keeps, and imatinib (37 heavy atoms), a large binder, so the affinity pass runs at 237 tokens.
AFFINITY_CROP_WARMUP_TOKENS = 300
AFFINITY_CROP_WARMUP_SMILES = "Cc1ccc(NC(=O)c2ccc(CN3CCN(C)CC3)cc2)cc1Nc1nccc(-c2cccnc2)n1"

# Fixed synthetic protein; repeated and cut to the requested length.
_MOTIF = "MKTAYIAKQRQISFVKSHFSRQLEERLGLIEVQAPILSRVGDGTQDNLSGAEKAVQVKVKALPDAQFEVV"

STATE_DISABLED = "disabled"
STATE_PENDING = "pending"
STATE_RUNNING = "running"
STATE_DONE = "done"
STATE_FAILED = "failed"

# Counters that describe compiles, read from compilation.compile_status().
_COMPILE_KEYS = ("successful_compiles", "failed_compiles")


def synthetic_sequence(tokens: int) -> str:
    """A fixed protein sequence of ``tokens`` residues (one residue is one token)."""
    repeats = -(-tokens // len(_MOTIF))
    return (_MOTIF * repeats)[:tokens]


def synthetic_a3m(sequence: str, rows: int) -> str:
    """An A3M text: the query plus ``rows - 1`` fixed variants of it (every 7th residue changed)."""
    lines = [">query", sequence]
    for row in range(1, rows):
        residues = list(sequence)
        for i in range(row, len(residues), 7):
            residues[i] = "A" if residues[i] != "A" else "G"
        lines += [f">row{row}", "".join(residues)]
    return "\n".join(lines) + "\n"


def warmup_requests(
    token_counts: tuple[int, ...] = WARMUP_TOKENS, msa_rows: tuple[int, ...] = WARMUP_MSA_ROWS
) -> list[dict[str, Any]]:
    """Keyword arguments of ``run_boltz2_prediction`` for each warm-up input.

    ``msa_rows`` above 1 is kept in the key ``msa_rows``; :func:`run_warmup` turns it into
    inline A3M text and removes the key before the call.
    """
    requests = []
    for tokens in token_counts:
        for rows in msa_rows:
            request: dict[str, Any] = {
                "sequences": [{"protein": {"id": "A", "sequence": synthetic_sequence(tokens)}}],
                "name": f"warmup_{tokens}" if rows == 1 else f"warmup_{tokens}_msa{rows}",
                "recycling_steps": WARMUP_RECYCLING_STEPS,
                "sampling_steps": WARMUP_SAMPLING_STEPS,
                "diffusion_samples": 1,
            }
            if rows > 1:
                request["msa_rows"] = rows
            requests.append(request)
    return requests


def affinity_warmup_request(
    tokens: int = AFFINITY_WARMUP_TOKENS,
    smiles: str = AFFINITY_WARMUP_SMILES,
    name: str = "warmup_affinity",
    diffusion_samples_affinity: int | None = None,
    sampling_steps_affinity: int = WARMUP_SAMPLING_STEPS,
) -> dict[str, Any]:
    """Keyword arguments of ``run_boltz2_prediction`` for an affinity warm-up: ``tokens`` residues and one binder.

    ``diffusion_samples_affinity`` is the batch of the affinity diffusion (the service default when ``None``);
    Dynamo gives a batch of 1 its own graph.

    The structure pass of this input runs the wrapped modules (compiled when optimizations are on) with a ligand in the
    complex. The affinity pass loads the kernels and allocator pools of the affinity model before the first request;
    with optimizations on, it also takes the triangle kernels in its trunk and captures the score-model CUDA graph.
    """
    return {
        "sequences": [
            {"protein": {"id": "A", "sequence": synthetic_sequence(tokens)}},
            {"ligand": {"id": "B", "smiles": smiles}},
        ],
        "properties": [{"affinity": {"binder": "B"}}],
        "name": name,
        "recycling_steps": WARMUP_RECYCLING_STEPS,
        "sampling_steps": WARMUP_SAMPLING_STEPS,
        "diffusion_samples": 1,
        "sampling_steps_affinity": sampling_steps_affinity,
        **({} if diffusion_samples_affinity is None else {"diffusion_samples_affinity": diffusion_samples_affinity}),
    }


def _materialize(request: dict[str, Any]) -> dict[str, Any]:
    """The call arguments of ``request``: inline A3M text for ``msa_rows`` and no ``msa_rows`` key.

    The text goes through the same input check as the MSA of a client request.
    """
    kwargs = {k: v for k, v in request.items() if k != "msa_rows"}
    rows = request.get("msa_rows", 1)
    if rows > 1:
        protein = dict(kwargs["sequences"][0]["protein"])
        protein["msa"] = synthetic_a3m(protein["sequence"], rows)
        kwargs["sequences"] = [{"protein": protein}]
    return kwargs


def _compile_totals(status: dict[str, dict[str, Any]]) -> dict[str, dict[str, int]]:
    return {label: {key: int(entry.get(key, 0)) for key in _COMPILE_KEYS} for label, entry in status.items()}


@dataclass
class WarmupReport:
    """What the start-up warm-up did. Reported by ``_runtime_info``."""

    state: str = STATE_PENDING
    reason: str | None = None
    duration_s: float | None = None
    error: str | None = None
    steps: list[dict[str, Any]] = field(default_factory=list)
    compiles_during_warmup: dict[str, dict[str, int]] = field(default_factory=dict)
    _baseline_compiles: dict[str, dict[str, int]] = field(default_factory=dict, repr=False)

    @property
    def settled(self) -> bool:
        """True once the warm-up is over (or was never going to run)."""
        return self.state not in (STATE_PENDING, STATE_RUNNING)

    def to_dict(self, status: dict[str, dict[str, Any]] | None = None) -> dict[str, Any]:
        """The report, plus the compiles caused by user traffic since the warm-up.

        ``compile_status`` counts warm-up compiles too. ``since_warmup`` is the
        difference to the baseline taken after the warm-up.
        """
        out: dict[str, Any] = {
            "state": self.state,
            "reason": self.reason,
            "duration_s": self.duration_s,
            "error": self.error,
            "steps": self.steps,
            "compiles_during_warmup": self.compiles_during_warmup,
        }
        if status is not None:
            totals = _compile_totals(status)
            out["since_warmup"] = {
                "compiles": {
                    label: {
                        key: max(now[key] - self._baseline_compiles.get(label, {}).get(key, 0), 0)
                        for key in _COMPILE_KEYS
                    }
                    for label, now in totals.items()
                },
            }
        return out


def run_warmup(
    predict: Callable[..., dict[str, Any]],
    requests: list[dict[str, Any]],
    status_fn: Callable[[], dict[str, dict[str, Any]]],
    report: WarmupReport,
    extra_steps: list[tuple[str, Callable[[], Any]]] | None = None,
) -> WarmupReport:
    """Run ``requests`` through ``predict``, then ``extra_steps``, and fill ``report``.

    An extra step is a ``(name, callable)`` pair for a compile that no prediction
    input reaches (for example the long-sequence triangle multiply graph).

    ``report.state`` is ``running`` while this runs and ``done`` or ``failed``
    after. A failure is logged and recorded and does not raise: the compiled
    modules fall back to compiling on the first user request, which is slow but
    correct.
    """
    report.state = STATE_RUNNING
    before = _compile_totals(status_fn())
    started = time.monotonic()
    try:
        for request in requests:
            step_before = _compile_totals(status_fn())
            step_started = time.monotonic()
            result = predict(**_materialize(request))
            if result.get("error"):
                raise RuntimeError(result.get("message", "warm-up prediction failed"))
            step_after = _compile_totals(status_fn())
            report.steps.append(
                {
                    "name": request["name"],
                    "seconds": round(time.monotonic() - step_started, 3),
                    "compiles": {
                        label: step_after[label]["successful_compiles"]
                        - step_before.get(label, {}).get("successful_compiles", 0)
                        for label in step_after
                    },
                }
            )
        for name, step in extra_steps or []:
            step_started = time.monotonic()
            step()
            report.steps.append({"name": name, "seconds": round(time.monotonic() - step_started, 3)})
        report.state = STATE_DONE
    except Exception as exc:  # noqa: BLE001 - recorded, see docstring
        report.state = STATE_FAILED
        report.error = f"{type(exc).__name__}: {exc}"
        logger.error(
            "Start-up warm-up failed; the first user request will compile instead: %s", report.error, exc_info=True
        )
    report.duration_s = round(time.monotonic() - started, 3)
    after = _compile_totals(status_fn())
    report.compiles_during_warmup = {
        label: {key: now[key] - before.get(label, {}).get(key, 0) for key in _COMPILE_KEYS}
        for label, now in after.items()
    }
    report._baseline_compiles = after
    return report
