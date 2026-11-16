# Copyright © Advanced Micro Devices, Inc., or its affiliates.
#
# SPDX-License-Identifier: MIT

"""Per-module TorchInductor compilation for Boltz-2.

Two module classes are wrapped, selected by name because each sees a distinct
call shape and recompiles independently under Dynamo's own shape guards:

``transition``
    ``boltz.model.layers.transition.Transition`` — a two-layer MLP
    (``norm -> SiLU-gated fc1/fc2 -> fc3``), pure tensor in/out. Instantiated
    many times (``PairformerLayer.transition_s``/``transition_z``,
    ``PairformerNoSeqLayer.transition_z``, the MSA module's ``msa_transition``),
    called at the same fixed shape every trunk/recycling step — the textbook
    repeated-call-same-shape case Inductor amortises its compile cost over.

``token_transformer``
    ``boltz.model.modules.transformersv2.DiffusionTransformer`` — called as
    ``DiffusionModule.token_transformer`` (depth 24) once per diffusion step,
    and again at a different depth/head count inside each ``AtomTransformer``
    used by the atom attention encoder/decoder (``atom_encoder``/
    ``atom_decoder``, each holding its own ``diffusion_transformer``
    attribute). For Inductor, selecting by class, not by one named attribute,
    covers all of them; each gets its own independently compiled graph against
    its own real weights, so sharing the class is safe.

Ordering, and why it is the OPPOSITE of ``rocm_triangle.install``'s
--------------------------------------------------------------------
``rocm_triangle.install`` rebinds a class's ``forward`` and MUST run before any
instance is constructed, or an already-built instance keeps the unpatched
method. Compilation wraps already-built INSTANCES and must run AFTER
``Boltz2.load_from_checkpoint`` and ``model.eval()`` — there is nothing to
specialize against until the real weights and eval-mode buffers exist.
:func:`install_compilation` is called from ``runner.load_boltz2_model`` after
both.

Per-module selectors, not a whole-model compile
------------------------------------------------
``torch.compile(model)`` would pull boltz's own custom Triton dispatch (the
``use_kernels`` seam ``rocm_triangle`` installs into) through Dynamo, which has
no graph-break rule for a custom kernel of ours. Wrapping two named module
classes keeps the compiled region to the plain-tensor MLP/transformer math and
leaves every custom op interpreted exactly as today.

Compile counting
----------------
``CompileStatus`` counts Dynamo compiles seen through Dynamo's own compile-end callback,
credited to the selector whose call was running in that thread; it is never inferred from call
counts. Callbacks are told apart by their compile id (``frame_id/attempt``): successful
compiles of the wrapped forward's frame (the first is the compile, later ones are
recompiles), compiles of other frames (graph-break resume frames), and failed compiles (the
call then raised a Dynamo error). The latest events are kept with the input shapes and call
time of each compiling call. ``distinct_input_shapes_seen`` above the number of successful
compiles means calls with new shapes reused a compiled graph.

Fallback contract
------------------
Compilation can fail to produce a graph for a shape or control-flow pattern
Dynamo does not support, which is a property of the compiler, not a bug in the
model. ``_wrap_with_eager_fallback`` catches ONLY
``torch._dynamo.exc.TorchDynamoException`` (compilation/guard failures) and
falls back to the original eager ``forward`` for the rest of the process,
logging ONCE at ``ERROR`` (not ``WARNING``) and recording the degradation on
:class:`CompileStatus` — a per-MODEL-INSTANCE, queryable fact (see
:func:`compile_status`), so the service or a health check can tell this ran
degraded without relying on a log line a reader could miss, and without that
fact leaking into a later model reload or a different model instance. Every
other exception — a real shape mismatch, a NaN, anything the model itself
would have raised anyway — propagates unchanged; it must never read as
"fallback succeeded."

"""

from __future__ import annotations

import collections
import json
import logging
import re
import threading
import time
from dataclasses import dataclass, field
from importlib import import_module
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    import torch

logger = logging.getLogger(__name__)

# The CompileStatus whose compiled call is running in this thread; Dynamo compile callbacks
# fire in the compiling thread and credit that status.
_tls = threading.local()

COMPILE_NONE = "none"
COMPILE_INDUCTOR = "inductor"
VALID_COMPILE_MODES = (COMPILE_NONE, COMPILE_INDUCTOR)

# selector name -> (module path, class name), resolved lazily so the CPU unit
# tier can exercise mode parsing with no boltz import.
_TARGETS: dict[str, tuple[str, str]] = {
    "transition": ("boltz.model.layers.transition", "Transition"),
    "token_transformer": ("boltz.model.modules.transformersv2", "DiffusionTransformer"),
}

# Bounds on the per-selector compile bookkeeping reported in the compile status.
_MAX_COMPILE_EVENT_LOG = 32
_MAX_DISTINCT_SHAPES = 64

# Marks an instance whose forward has already been wrapped, so a second
# install_compilation() call (e.g. a resumed/retried startup) skips it instead
# of compiling a compiled forward.
_WRAPPED_ATTR = "_aim_compile_wrapped"

# Marks an instance that fell back to eager after a real Dynamo/Inductor
# compile failure. The authoritative, queryable record is the per-model
# CompileStatus (see compile_status()) — this attribute is a cheap in-process
# marker on the submodule itself, handy for direct inspection in tests.
_DEGRADED_ATTR = "_aim_compile_degraded"

# Attribute name under which a model's per-selector CompileStatus dict lives.
# Attached to the MODEL INSTANCE (not a module global), so the fact is
# worker-local/model-specific: a fresh load_boltz2_model(...) call produces a
# fresh model object with no attribute yet, which naturally resets the state —
# nothing to leak across models, reloads, or requests.
_STATUS_ATTR = "_aim_compile_status"


@dataclass
class CompileStatus:
    """Actual, observed runtime compile state for one selector, on one model.

    Mutable and attached to the model instance ``install_compilation`` ran
    against (see :func:`compile_status`), never a module-level dict — a
    module global would leak state across model reloads and unrelated model
    instances, which is exactly the thing this must not do.

    Installing a wrapper is not evidence of a successful compile — Dynamo
    compiles lazily, on first call — so every field here reflects something
    that was actually OBSERVED to happen, not inferred from the mode string
    or from wrapper installation alone.
    """

    mode: str
    instances: int = 0
    invoked: bool = False
    compiled_calls: int = 0
    # A Dynamo compile failure that fell back to eager: "ran, but not as configured".
    eager_fallback_calls: int = 0
    # Inductor-only, sticky: once True, this selector is permanently degraded
    # for the rest of the process (see the module docstring's Fallback
    # contract).
    degraded: bool = False
    # Inductor-only: Dynamo compiles observed through Dynamo's own compile-end callback, never
    # inferred from call counts. The callback fires once per compiled Dynamo FRAME attempt, including
    # failed compiles and the resume frames a graph break creates, so these are kept apart:
    #   compile_callbacks    every callback firing
    #   failed_compiles      callbacks of a call that then raised a Dynamo error
    #   frame_compiles       successful compiles per Dynamo frame id
    # The "primary" frame is the one of the first callback (the wrapped forward). More than one
    # successful compile of the primary frame is a recompile; other frame ids are graph-break frames.
    compile_callbacks: int = 0
    failed_compiles: int = 0
    frame_compiles: dict[int, int] = field(default_factory=dict, repr=False)
    primary_frame_id: int | None = None
    unparsed_compile_ids: int = 0
    # "dynamo-callback", or "unavailable" if the callbacks could not be registered (then the compile
    # counts stay 0 and say nothing).
    compile_event_source: str = "dynamo-callback"
    distinct_input_shapes: set[str] = field(default_factory=set, repr=False)
    # Calls whose input shapes were not tracked because distinct_input_shapes was full.
    input_shapes_not_tracked_calls: int = 0
    # The latest compile events (bounded); older ones are counted in compile_events_dropped.
    compile_event_log: collections.deque = field(
        default_factory=lambda: collections.deque(maxlen=_MAX_COMPILE_EVENT_LOG), repr=False
    )
    compile_events_dropped: int = 0
    _lock: threading.Lock = field(default_factory=threading.Lock, repr=False)

    def record_invocation(self) -> None:
        self.invoked = True

    def record_compile_callback(self, compile_id: Any) -> None:
        match = re.match(r"^(\d+)/(\d+)", str(compile_id))
        with self._lock:
            self.compile_callbacks += 1
            event: dict[str, Any] = {"index": self.compile_callbacks, "compile_id": str(compile_id), "failed": False}
            if match:
                frame_id, attempt = int(match.group(1)), int(match.group(2))
                event.update(frame_id=frame_id, frame_attempt=attempt)
                if self.primary_frame_id is None:
                    self.primary_frame_id = frame_id
                self.frame_compiles[frame_id] = self.frame_compiles.get(frame_id, 0) + 1
            else:
                self.unparsed_compile_ids += 1
            if len(self.compile_event_log) == self.compile_event_log.maxlen:
                self.compile_events_dropped += 1
            self.compile_event_log.append(event)

    def mark_failed_compiles(self, first_new_index: int) -> None:
        """The call that produced callbacks after ``first_new_index`` raised a Dynamo error: those compiles failed."""
        with self._lock:
            failed = self.compile_callbacks - first_new_index
            self.failed_compiles += failed
            for event in self.compile_event_log:
                if event["index"] > first_new_index:
                    event["failed"] = True
            # Failed compiles are not successful ones: undo their frame credit, newest first.
            for event in reversed(self.compile_event_log):
                if event["index"] > first_new_index and "frame_id" in event:
                    self.frame_compiles[event["frame_id"]] -= 1

    def note_input_shapes(self, shapes: str) -> None:
        with self._lock:
            if shapes in self.distinct_input_shapes:
                return
            if len(self.distinct_input_shapes) < _MAX_DISTINCT_SHAPES:
                self.distinct_input_shapes.add(shapes)
            else:
                self.input_shapes_not_tracked_calls += 1

    def annotate_new_events(self, first_new_index: int, shapes: str, seconds: float) -> None:
        with self._lock:
            for event in self.compile_event_log:
                if event["index"] > first_new_index and "input_shapes" not in event:
                    event["input_shapes"] = shapes
                    event["call_seconds"] = round(seconds, 3)

    def to_dict(self) -> dict[str, Any]:
        return {
            "mode": self.mode,
            "instances": self.instances,
            "state": self._state(),
            "compiled_calls": self.compiled_calls,
            "eager_fallback_calls": self.eager_fallback_calls,
            "degraded": self.degraded,
            "dynamo_compile_callbacks": self.compile_callbacks,
            "failed_compiles": self.failed_compiles,
            "successful_compiles": sum(self.frame_compiles.values()),
            "forward_frame_compiles": (
                self.frame_compiles.get(self.primary_frame_id, 0) if self.primary_frame_id is not None else 0
            ),
            "forward_recompiles": (
                max(self.frame_compiles.get(self.primary_frame_id, 0) - 1, 0)
                if self.primary_frame_id is not None
                else 0
            ),
            "graph_break_frames": len(
                [f for f, n in self.frame_compiles.items() if f != self.primary_frame_id and n > 0]
            ),
            "graph_break_frame_compiles": sum(n for f, n in self.frame_compiles.items() if f != self.primary_frame_id),
            "unparsed_compile_ids": self.unparsed_compile_ids,
            "compile_event_source": self.compile_event_source,
            "distinct_input_shapes_seen": len(self.distinct_input_shapes),
            "distinct_input_shapes_capped": len(self.distinct_input_shapes) >= _MAX_DISTINCT_SHAPES,
            "input_shapes_not_tracked_calls": self.input_shapes_not_tracked_calls,
            "compile_events_dropped": self.compile_events_dropped,
            "compile_event_log_tail": list(self.compile_event_log)[-4:],
        }

    def _state(self) -> str:
        """Derive a state string from OBSERVED call counters, never from ``degraded`` alone.

        ``degraded`` is a sticky per-instance marker (see its field docstring)
        but many instances can share one ``CompileStatus`` under one selector
        label, so one instance degrading must not make the aggregate read as
        uniformly degraded when other instances are still compiling and
        executing successfully — see ``"mixed-partial-degradation"`` below.
        Likewise, ``"compiled-executed"`` requires an actual
        successful-call counter above zero, not just ``invoked``: a call whose
        wrapped forward raised something other than the documented fallback
        exception (a real runtime error) still sets ``invoked`` but increments
        neither counter, and must never read as a clean success.
        """
        if self.mode == COMPILE_NONE:
            return "eager-disabled"
        if self.instances == 0:
            return "no-instances"
        if not self.invoked:
            return "not-yet-invoked"
        # inductor: derive from the call counters, not the sticky `degraded`
        # flag, so a label covering several instances can report a genuine
        # mix rather than one instance's degradation tainting the whole
        # aggregate.
        has_compiled = self.compiled_calls > 0
        has_fallback = self.eager_fallback_calls > 0
        if has_compiled and has_fallback:
            return "mixed-partial-degradation"
        if has_fallback:
            return "degraded-fallback"
        if has_compiled:
            return "compiled-executed"
        return "invoked-uncaught-error"


def _status_map(model: "torch.nn.Module") -> dict[str, CompileStatus]:
    """The model's per-selector status dict, creating it on first use.

    Attaching to the model object (rather than keying a module-level dict by
    some derived model identity) is what makes this worker-local and
    model-specific: the dict lives exactly as long as the model instance
    does, and a different model instance — even of the same class — starts
    with none.
    """
    status = getattr(model, _STATUS_ATTR, None)
    if status is None:
        status = {}
        setattr(model, _STATUS_ATTR, status)
    return status


def compile_status(model: "torch.nn.Module") -> dict[str, dict[str, Any]]:
    """Selector label -> observed runtime compile state, for THIS model instance.

    Empty if :func:`install_compilation` has not run against ``model`` yet.
    This is the ONLY supported way to learn what actually happened at
    runtime — never infer success from wrapper installation, and never read
    a log line as a substitute. See ``CompileStatus`` for the state values.
    """
    return {label: status.to_dict() for label, status in _status_map(model).items()}


def compile_count(model: "torch.nn.Module") -> int:
    """The number of Dynamo compile callbacks of all selectors of ``model`` so far."""
    return sum(status.compile_callbacks for status in _status_map(model).values())


def compile_events_available(model: "torch.nn.Module") -> bool:
    """Whether :func:`compile_count` of ``model`` counts the Dynamo compiles (no selector is "unavailable")."""
    return all(status.compile_event_source != "unavailable" for status in _status_map(model).values())


def compile_mode(value: Any) -> str:
    """Validate a compile mode (``none`` or ``inductor``). Raises ``ValueError`` on anything else.

    Pure, so the CPU unit tier can exercise every branch with no GPU and no
    ``boltz``/``torch`` import.
    """
    normalized = str(value or COMPILE_INDUCTOR).strip().lower()
    if normalized not in VALID_COMPILE_MODES:
        raise ValueError(f"compile mode must be one of {VALID_COMPILE_MODES}, got {value!r}")
    return normalized


@dataclass(frozen=True)
class CompileConfig:
    """Compile mode (``none`` or ``inductor``) per target class."""

    transition: str = COMPILE_INDUCTOR
    token_transformer: str = COMPILE_INDUCTOR


def _on_compile_end(*cb_args: Any) -> None:
    """Dynamo compile-end callback; runs in the compiling thread and credits that thread's status."""
    status = getattr(_tls, "status", None)
    if status is None:
        return
    info = cb_args[0] if cb_args else None
    trigger = getattr(getattr(info, "callback_trigger", None), "name", "DYNAMO")
    if trigger == "DYNAMO":
        status.record_compile_callback(getattr(info, "compile_id", None))


_compile_callbacks_unavailable_logged = False
_hooks_lock = threading.Lock()


def _ensure_compile_hooks() -> bool:
    """Make sure :func:`_on_compile_end` is registered with Dynamo (``torch._dynamo.reset()`` clears callbacks).

    Returns False if the callback API is missing, so callers can report the counting as unavailable
    instead of reporting zero compiles.
    """
    global _compile_callbacks_unavailable_logged
    try:
        from torch._dynamo.callback import callback_handler

        with _hooks_lock:
            if _on_compile_end not in callback_handler.end_callbacks:
                callback_handler.register_end_callback(_on_compile_end)
        return True
    except (ImportError, AttributeError):
        if not _compile_callbacks_unavailable_logged:
            _compile_callbacks_unavailable_logged = True
            logger.warning("Dynamo compile callbacks are not available; Dynamo compiles will not be counted")
        return False


def _shape_summary(args: tuple, kwargs: dict) -> str:
    shapes = [list(a.shape) for a in (*args, *kwargs.values()) if hasattr(a, "shape")]
    return json.dumps(shapes[:4])


def _wrap_with_eager_fallback(module: "torch.nn.Module", label: str, status: CompileStatus) -> None:
    """Replace ``module.forward`` with a compiled version that can fall back to eager.

    Assigning ``module.forward`` as an instance attribute shadows the class
    method without touching the class, so uncompiled instances of the same
    class (if any share it) are unaffected. ``status`` is the model-instance-
    scoped :class:`CompileStatus` for ``label`` — every outcome below is
    recorded on it, not inferred later from wrapper installation.
    """
    import torch

    if getattr(module, _WRAPPED_ATTR, False):
        return

    original_forward = module.forward
    compiled_forward = torch.compile(original_forward, dynamic=True)
    class_name = type(module).__name__
    if not _ensure_compile_hooks():
        status.compile_event_source = "unavailable"

    def _forward(*args: Any, **kwargs: Any) -> Any:
        status.record_invocation()
        shapes = _shape_summary(args, kwargs)
        status.note_input_shapes(shapes)
        events_before = status.compile_callbacks
        started = time.perf_counter()
        _ensure_compile_hooks()
        _tls.status = status
        try:
            result = compiled_forward(*args, **kwargs)
        except torch._dynamo.exc.TorchDynamoException as exc:
            # ERROR, not WARNING: this is a permanent degradation to eager for
            # the rest of the process, not a transient notice — a reader
            # scanning for errors must not be able to miss it. The module is
            # also marked and the per-model CompileStatus updated, so the
            # fact is queryable (compile_status(model)), not only present in
            # a log line that could scroll past unread.
            logger.error(
                "compile %s: TorchInductor could not compile %s (%s); "
                "falling back to eager for this module for the rest of the "
                "process. This is a degraded run, not a clean success — this "
                "path is only entered for a Dynamo/Inductor compilation "
                "failure, never for an ordinary runtime error.",
                label.upper(),
                class_name,
                exc,
            )
            status.mark_failed_compiles(events_before)
            status.annotate_new_events(events_before, shapes, time.perf_counter() - started)
            module.forward = original_forward
            setattr(module, _DEGRADED_ATTR, True)
            status.degraded = True
            status.eager_fallback_calls += 1
            return original_forward(*args, **kwargs)
        finally:
            _tls.status = None
        if status.compile_callbacks != events_before:
            status.annotate_new_events(events_before, shapes, time.perf_counter() - started)
        status.compiled_calls += 1
        return result

    module.forward = _forward
    setattr(module, _WRAPPED_ATTR, True)


def install_compilation(
    model: "torch.nn.Module",
    config: CompileConfig,
) -> dict[str, int]:
    """Wrap each configured target class's instances in ``model``.

    Must run AFTER ``Boltz2.load_from_checkpoint`` and ``model.eval()`` — see
    the module docstring for why this ordering is the reverse of
    ``rocm_triangle.install``'s. Returns the number of instances wrapped per
    selector, so the caller can log or assert on it; ``0`` for a selector set
    to ``none``, and also ``0`` if the target class has no instances in this
    model (not an error — it means the selector found nothing to do).

    Attaches a per-selector :class:`CompileStatus` to ``model`` (see
    ``_STATUS_ATTR``) for every selector, INCLUDING ``"none"`` — so
    :func:`compile_status` always reports what was actually configured, not
    only what was wrapped. Because the status dict lives on ``model`` itself,
    calling this against a DIFFERENT model instance starts clean; there is no
    module-global counter to leak between loads.

    A repeated call against the SAME model instance (e.g. a resumed or
    retried startup) does NOT discard an existing selector's
    ``CompileStatus`` — only instances not yet wrapped (``_WRAPPED_ATTR``
    unset) get a new wrapper, and already-tracked invocation/degradation
    history is preserved, not reset to zero. ``status.instances`` always
    reflects the TOTAL number of matching instances currently tracked
    (previously wrapped plus newly wrapped this call), never only the delta
    from this one call.

    A repeated call that asks for a DIFFERENT mode than the one already
    recorded for a selector raises ``ValueError`` instead of silently
    reconfiguring it. There is no live-reconfiguration path: a submodule
    already wrapped for ``inductor`` cannot be switched to ``none`` (or
    vice versa) by calling this again, since the earlier wrapper, its
    tracked history, and any captured originals would be discarded with no
    record of the change. A genuine mode change needs a fresh model
    instance.
    """
    status_map = _status_map(model)
    counts: dict[str, int] = {}
    for selector, mode in (
        ("transition", config.transition),
        ("token_transformer", config.token_transformer),
    ):
        mode = compile_mode(mode)
        existing = status_map.get(selector)
        if existing is not None and existing.mode != mode:
            raise ValueError(
                f"install_compilation: {selector!r} is already configured as "
                f"{existing.mode!r} on this model instance; it cannot be "
                f"reconfigured to {mode!r} by calling install_compilation again. "
                "Build a fresh model instance to change a selector's mode."
            )
        if existing is not None:
            status = existing
        else:
            status = CompileStatus(mode=mode)
            status_map[selector] = status

        if mode == COMPILE_NONE:
            counts[selector] = 0
            continue

        module_path, class_name = _TARGETS[selector]
        target_cls = getattr(import_module(module_path), class_name)

        newly_wrapped = 0
        total_instances = 0
        for submodule in model.modules():
            if isinstance(submodule, target_cls):
                total_instances += 1
                if not getattr(submodule, _WRAPPED_ATTR, False):
                    _wrap_with_eager_fallback(submodule, selector, status)
                    newly_wrapped += 1
        counts[selector] = newly_wrapped
        status.instances = total_instances
        logger.info(
            "compile %s=inductor: wrapped %d new %s instance(s) (%d tracked total)",
            selector.upper(),
            newly_wrapped,
            class_name,
            total_instances,
        )
    return counts
