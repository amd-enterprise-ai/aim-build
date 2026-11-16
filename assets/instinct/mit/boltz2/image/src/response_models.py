# Copyright © Advanced Micro Devices, Inc., or its affiliates.
#
# SPDX-License-Identifier: MIT

"""Pydantic models of the JSON the service returns.

The endpoints build their answer as a dict, validate it with one of these models
and return ``model_dump(exclude_unset=True)``. A key that the dict does not hold
stays out of the answer, and a key that holds ``None`` stays in it as ``null``.
Models for output that the service owns refuse unknown keys; models for the
files of ``boltz predict`` keep them.
"""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict

# ``float | int`` keeps an int as an int, so the JSON text does not change.
Number = float | int


class _Strict(BaseModel):
    model_config = ConfigDict(extra="forbid")


class _Open(BaseModel):
    model_config = ConfigDict(extra="allow")


class Structure(_Strict):
    record_id: str
    format: Literal["pdb", "mmcif"]
    content: str


class ConfidenceSummary(_Open):
    """``confidence_<record_id>_model_<rank>.json`` of ``boltz predict``."""

    confidence_score: Number | None = None
    ptm: Number | None = None
    iptm: Number | None = None
    ligand_iptm: Number | None = None
    protein_iptm: Number | None = None
    complex_plddt: Number | None = None
    complex_iplddt: Number | None = None
    complex_pde: Number | None = None
    complex_ipde: Number | None = None
    chains_ptm: dict[str, Any] | None = None
    pair_chains_iptm: dict[str, Any] | None = None


class Timing(_Strict):
    runtime_s: float
    affinity_s: float | None = None


class Affinity(_Open):
    """``affinity_<record_id>.json`` of ``boltz predict``."""

    affinity_pred_value: Number | None = None
    affinity_probability_binary: Number | None = None
    affinity_pred_value1: Number | None = None
    affinity_probability_binary1: Number | None = None
    affinity_pred_value2: Number | None = None
    affinity_probability_binary2: Number | None = None


class Triton(_Strict):
    attention: bool
    mul_impl: str | None


class CompileStatusEntry(_Strict):
    """Observed compile state of one selector; see ``compilation.CompileStatus``."""

    mode: str
    instances: int
    state: str
    compiled_calls: int
    eager_fallback_calls: int
    degraded: bool
    dynamo_compile_callbacks: int
    failed_compiles: int
    successful_compiles: int
    forward_frame_compiles: int
    forward_recompiles: int
    graph_break_frames: int
    graph_break_frame_compiles: int
    unparsed_compile_ids: int
    compile_event_source: str
    distinct_input_shapes_seen: int
    distinct_input_shapes_capped: bool
    input_shapes_not_tracked_calls: int
    compile_events_dropped: int
    compile_event_log_tail: list[dict[str, Any]]


class WarmupStep(_Strict):
    name: str
    seconds: float
    compiles: dict[str, int] | None = None


class SinceWarmup(_Strict):
    compiles: dict[str, dict[str, int]]


class Warmup(_Strict):
    state: str
    reason: str | None
    duration_s: float | None
    error: str | None
    steps: list[WarmupStep]
    compiles_during_warmup: dict[str, dict[str, int]]
    since_warmup: SinceWarmup | None = None


class RuntimeInfo(_Strict):
    mode: Literal["structure", "affinity"]
    optimizations: Literal["on", "off"]
    triton: Triton
    precision: str
    compile_status: dict[str, CompileStatusEntry]
    warmup: Warmup
    affinity_compile_status: dict[str, CompileStatusEntry] | None = None
    cuda_peak_gb: float | None = None


class InferenceResponse(_Strict):
    structures: list[Structure]
    confidence: dict[str, ConfidenceSummary]
    timing: Timing
    affinity: Affinity | None = None
    seed: int | None
    runtime: RuntimeInfo


class ErrorResponse(_Strict):
    error: str


class HealthResponse(_Strict):
    status: Literal["ok", "unavailable"]


class ModelCard(_Strict):
    id: str
    object: Literal["model"]


class ModelList(_Strict):
    object: Literal["list"]
    data: list[ModelCard]


def dump(model: type[BaseModel], payload: dict[str, Any]) -> dict[str, Any]:
    """Validate ``payload`` against ``model`` and return it as a dict with only the keys it held."""
    return model.model_validate(payload).model_dump(exclude_unset=True)
