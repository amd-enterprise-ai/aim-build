# Copyright © Advanced Micro Devices, Inc., or its affiliates.
#
# SPDX-License-Identifier: MIT

"""CPU unit tier for per-module TorchInductor compilation (``compilation.py``).

Unlike ``rocm_triangle``, Inductor needs no GPU — it has a CPU codegen backend
too — so this tier can execute a REAL ``torch.compile`` wrap-and-call, not just
exercise shape/algebra facts about it. What it cannot do is exercise the real
``boltz`` module classes (``boltz`` is not installed in the CPU tier), so
``install_compilation``'s class-resolution path is exercised only for the
``none`` selector, which is specified to never import ``boltz``.
"""

from __future__ import annotations

import inspect

import pytest
import torch
from compilation import (
    COMPILE_INDUCTOR,
    COMPILE_NONE,
    VALID_COMPILE_MODES,
    CompileConfig,
    CompileStatus,
    _wrap_with_eager_fallback,
    compile_mode,
    compile_status,
    install_compilation,
)

# --------------------------------------------------------------------------- #
# Mode parsing
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        (COMPILE_INDUCTOR, COMPILE_INDUCTOR),
        (COMPILE_NONE, COMPILE_NONE),
        ("  INDUCTOR ", COMPILE_INDUCTOR),
        ("None", COMPILE_NONE),
    ],
)
def test_compile_mode_normalizes_case_and_whitespace(value, expected):
    assert compile_mode(value) == expected


@pytest.mark.parametrize("unset", [None, ""])
def test_compile_mode_defaults_to_inductor_when_unset(unset):
    assert compile_mode(unset) == COMPILE_INDUCTOR


@pytest.mark.parametrize("bad", ["on", "true", "triton", "eager", "1", "aot"])
def test_compile_mode_rejects_undeclared_values(bad):
    with pytest.raises(ValueError, match="compile mode"):
        compile_mode(bad)


def test_valid_compile_modes_is_exactly_two():
    assert set(VALID_COMPILE_MODES) == {COMPILE_NONE, COMPILE_INDUCTOR}


# --------------------------------------------------------------------------- #
# --------------------------------------------------------------------------- #
# install_compilation — the "none" selector must never import boltz
# --------------------------------------------------------------------------- #


class _TinyModel(torch.nn.Module):
    """Stand-in for Boltz2 in these tests — boltz is not installed here."""

    def __init__(self):
        super().__init__()
        self.lin = torch.nn.Linear(4, 4)

    def forward(self, x):
        return self.lin(x)


def test_install_compilation_none_is_a_no_op_without_importing_boltz():
    import sys

    model = _TinyModel()
    before = "boltz" in sys.modules
    counts = install_compilation(model, CompileConfig(transition=COMPILE_NONE, token_transformer=COMPILE_NONE))
    after = "boltz" in sys.modules
    assert counts == {"transition": 0, "token_transformer": 0}
    assert after == before, "install_compilation(..., none, none) imported boltz; it must be inert"

    # compile_status must report the ACTUAL configured state ("eager-disabled"),
    # not merely absence of data, and must be scoped to this model instance.
    status = compile_status(model)
    assert status["transition"]["state"] == "eager-disabled"
    assert status["token_transformer"]["state"] == "eager-disabled"

    fresh_model = _TinyModel()
    assert compile_status(fresh_model) == {}, (
        "a model install_compilation has never touched must report no compile_status "
        "at all — a fresh model/reload must not inherit another instance's state"
    )


def test_install_compilation_inductor_without_boltz_raises_module_not_found(monkeypatch):
    """Opting in when the target class cannot be imported fails as
    ModuleNotFoundError, not silently as 0 wrapped.

    Forces the absent-boltz condition by patching ``compilation.import_module``
    rather than relying on the test venv actually lacking boltz: venv312 (the
    CPU unit tier's real environment) has boltz installed, so without this
    patch this test would only pass by accident of ambient install state, not
    because ``install_compilation`` actually fails closed on an unresolvable
    target.
    """
    import compilation as compilation_module

    def _raise(name):
        raise ModuleNotFoundError(f"No module named {name!r}")

    monkeypatch.setattr(compilation_module, "import_module", _raise)

    model = _TinyModel()
    with pytest.raises(ModuleNotFoundError):
        install_compilation(model, CompileConfig(transition=COMPILE_INDUCTOR, token_transformer=COMPILE_NONE))


def test_install_compilation_inductor_resolves_the_real_target_class_when_boltz_is_installed():
    """Positive path: when boltz IS installed (venv312's actual state), the
    inductor selector must resolve the real target class and proceed to
    instance-scanning rather than raising.

    Skipped (not assumed) when boltz is genuinely absent. ``_TinyModel`` has
    no ``Transition``/``DiffusionTransformer`` instances, so the expected
    outcome is "resolved the class, found nothing to wrap" — distinct from
    the mocked-absence test above, which never gets past the import.
    """
    pytest.importorskip("boltz.model.layers.transition")
    pytest.importorskip("boltz.model.modules.transformersv2")

    model = _TinyModel()
    counts = install_compilation(model, CompileConfig(transition=COMPILE_INDUCTOR, token_transformer=COMPILE_NONE))
    assert counts == {"transition": 0, "token_transformer": 0}
    status = compile_status(model)
    assert status["transition"]["state"] == "no-instances"


# --------------------------------------------------------------------------- #
# _wrap_with_eager_fallback — real torch.compile, executed
# --------------------------------------------------------------------------- #


def test_wrap_with_eager_fallback_preserves_the_eager_result():
    """Real compile-or-fallback round trip: whichever path runs, the answer matches eager.

    This exercises an actual ``torch.compile`` call on CPU. ``_TinyModel`` is a
    single ``nn.Linear``, which Inductor lowers to one ``extern_calls``
    passthrough with no C++ kernel ever built — confirmed empirically in this
    venv (``torch._dynamo.utils.counters["inductor"]`` shows
    ``extern_calls: 1`` and no codegen failure). So this test's compiled path
    always succeeds here; it does not exercise the except-and-fallback branch
    at all. See ``test_wrap_with_eager_fallback_catches_a_real_cpp_compile_failure``
    below for a model that forces genuine C++ codegen and proves the fallback
    branch itself works, not just that this trivial shape round-trips.
    """
    model = _TinyModel().eval()
    x = torch.randn(2, 4)
    with torch.no_grad():
        expected = model(x)

    status = CompileStatus(mode=COMPILE_INDUCTOR)
    _wrap_with_eager_fallback(model, "test", status)
    assert getattr(model, "_aim_compile_wrapped") is True

    with torch.no_grad():
        out = model(x)
    assert torch.allclose(out, expected, atol=1e-5)
    assert status.invoked is True
    assert status.compiled_calls >= 1
    assert status.degraded is False


def _compile_that_fails_on_fused_graphs(monkeypatch):
    """Make ``torch.compile`` fail, in any build environment, for a graph with a ``relu``.

    The failure comes from a custom Dynamo backend, so it does not depend on a C++
    toolchain being absent or present. Dynamo wraps an exception from a backend in
    ``BackendCompilerFailed``, which is the same ``TorchDynamoException`` type that
    a genuine Inductor failure raises. Graphs without ``relu`` use Inductor.
    """
    from torch._inductor.compile_fx import compile_fx

    def _backend(graph_module, example_inputs):
        if any(node.target is torch.relu for node in graph_module.graph.nodes):
            raise RuntimeError("forced compile failure")
        return compile_fx(graph_module, example_inputs)

    real_compile = torch.compile

    def _compile(fn, *args, **kwargs):
        kwargs["backend"] = _backend
        return real_compile(fn, *args, **kwargs)

    monkeypatch.setattr(torch, "compile", _compile)


class _FusedModel(torch.nn.Module):
    """A model whose graph holds a ``relu``, which ``_compile_that_fails_on_fused_graphs`` fails."""

    def __init__(self):
        super().__init__()
        self.lin = torch.nn.Linear(4, 4)

    def forward(self, x):
        y = self.lin(x)
        return torch.relu(y) * 2.0 + 1.0


def test_wrap_with_eager_fallback_catches_a_compile_failure(caplog, monkeypatch):
    """Prove the except-and-fallback branch fires on a compile failure.

    The failure is forced with a Dynamo backend (see
    ``_compile_that_fails_on_fused_graphs``), so the test does not depend on the
    C++ toolchain of the environment. ``BackendCompilerFailed`` is a
    ``torch._dynamo.exc.TorchDynamoException``, as is a genuine Inductor
    failure, so ``_wrap_with_eager_fallback``'s
    except clause must catch it, log the documented error, reset
    ``module.forward`` to the original, and still answer correctly. This is
    the "default Inductor failure semantics" contract item 7 requires: a real
    compile failure never silently returns a wrong answer, recovers to eager
    rather than propagating or going uncaught, and is left queryable on the
    model instance's own ``CompileStatus`` (see ``compilation.compile_status``)
    rather than only a log line a reader could miss — it must never read as a
    clean success. The status lives on the model instance, not a module
    global: an unrelated model/status pair must not see this one's
    degradation (asserted below).
    """
    _compile_that_fails_on_fused_graphs(monkeypatch)
    model = _FusedModel().eval()
    x = torch.randn(2, 4)
    with torch.no_grad():
        expected = model(x)

    status = CompileStatus(mode=COMPILE_INDUCTOR, instances=1)
    with caplog.at_level("ERROR", logger="compilation"):
        _wrap_with_eager_fallback(model, "test", status)
        with torch.no_grad():
            out = model(x)

    assert torch.allclose(out, expected, atol=1e-5)
    assert any(
        "could not compile" in r.message for r in caplog.records
    ), "expected the documented eager-fallback error to fire for the forced compile failure"
    assert getattr(model, "_aim_compile_wrapped") is True
    assert getattr(model, "_aim_compile_degraded") is True
    assert status.degraded is True
    assert status.eager_fallback_calls >= 1
    assert status.to_dict()["state"] == "degraded-fallback"

    # A fresh model/status must not see this model's degradation — proves the
    # state is per-instance, not a process-wide counter.
    other_model = _TinyModel().eval()
    other_status = CompileStatus(mode=COMPILE_INDUCTOR, instances=1)
    _wrap_with_eager_fallback(other_model, "test", other_status)
    assert other_status.degraded is False


def test_wrap_with_eager_fallback_is_idempotent():
    """A second wrap call on an already-wrapped module must be a no-op.

    Re-wrapping a compiled forward would compile the compiled function,
    which is the duplicate/conflicting-wrap the plan calls out to prevent.
    """
    model = _TinyModel().eval()
    status = CompileStatus(mode=COMPILE_INDUCTOR)
    _wrap_with_eager_fallback(model, "test", status)
    forward_after_first = model.forward
    _wrap_with_eager_fallback(model, "test", status)
    assert model.forward is forward_after_first


def test_eager_fallback_only_catches_dynamo_failures():
    """Pin the fallback contract: only Dynamo/Inductor failures are swallowed.

    A bare ``except Exception`` here would silently convert an ordinary
    runtime error (bad shape, NaN, a real bug) into "fallback succeeded",
    which is exactly what the plan prohibits. Assert against the source, the
    same way test_triton_adapter.py pins adapter.py's hazards statically.
    """
    source = inspect.getsource(_wrap_with_eager_fallback)
    assert "except torch._dynamo.exc.TorchDynamoException" in source
    assert "except Exception" not in source
    assert "except BaseException" not in source


# --------------------------------------------------------------------------- #
# CompileStatus._state() — "invoked-uncaught-error" must not read as success
# --------------------------------------------------------------------------- #


def test_compile_status_state_is_invoked_uncaught_error_when_no_counter_incremented():
    """``invoked`` alone is not evidence of success.

    A wrapped forward whose underlying call raises something other than the
    documented fallback exception still sets ``invoked``, but increments
    neither ``compiled_calls`` nor
    ``eager_fallback_calls``. That combination must read as a distinct,
    clearly-not-success state, not fall through to one of the success labels.
    """
    inductor_status = CompileStatus(mode=COMPILE_INDUCTOR, instances=1, invoked=True)
    assert inductor_status.to_dict()["state"] == "invoked-uncaught-error"


def test_wrap_with_eager_fallback_leaves_an_unrelated_runtime_error_uncaught():
    """A real runtime error (not a Dynamo compile failure) must propagate, and
    must not be miscounted as either a compiled success or a fallback."""

    class _RaisesAtRuntime(torch.nn.Module):
        def forward(self, x):
            _ = x + 1
            raise RuntimeError("boom - not a compile failure")

    model = _RaisesAtRuntime().eval()
    status = CompileStatus(mode=COMPILE_INDUCTOR, instances=1)
    _wrap_with_eager_fallback(model, "test", status)

    with pytest.raises(RuntimeError, match="boom"):
        model(torch.randn(2, 4))

    assert status.invoked is True
    assert status.compiled_calls == 0
    assert status.eager_fallback_calls == 0
    assert status.to_dict()["state"] == "invoked-uncaught-error"


# --------------------------------------------------------------------------- #
# install_compilation — reuse-not-reset on a repeated call, and total-vs-delta
# instance counting (item C's fix)
# --------------------------------------------------------------------------- #


def test_install_compilation_reinstall_preserves_history_and_reports_total_instances(monkeypatch):
    """A second install_compilation() call against the SAME model and the SAME
    mode (a resumed/retried startup) must not reset invocation history, and
    must report the TOTAL tracked instance count, not just this call's delta."""
    import compilation

    monkeypatch.setitem(compilation._TARGETS, "transition", (__name__, "_TinyModel"))

    class _Container(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.a = _TinyModel()

    model = _Container().eval()
    config = CompileConfig(transition=COMPILE_INDUCTOR, token_transformer=COMPILE_NONE)

    counts_first = install_compilation(model, config)
    assert counts_first["transition"] == 1
    with torch.no_grad():
        model.a(torch.randn(2, 4))

    status_after_first = compile_status(model)["transition"]
    assert status_after_first["instances"] == 1
    compiled_calls_after_first = status_after_first["compiled_calls"]
    assert compiled_calls_after_first >= 1

    # Simulate a resumed startup: a second matching instance appears, and
    # install_compilation runs again with the SAME mode.
    model.b = _TinyModel()
    counts_second = install_compilation(model, config)

    # Only the new instance is newly wrapped this call...
    assert counts_second["transition"] == 1
    # ...but the tracked total reflects both, and the first instance's history
    # survived rather than being reset to zero.
    status_after_second = compile_status(model)["transition"]
    assert status_after_second["instances"] == 2
    assert status_after_second["compiled_calls"] == compiled_calls_after_first


def test_install_compilation_rejects_a_mode_change_on_an_already_installed_selector(monkeypatch):
    """A second install_compilation() call against the SAME model with a
    DIFFERENT mode for a selector already configured must raise, not silently
    discard the first mode's CompileStatus and wrapper. There is no live
    reconfiguration feature — a genuine mode change needs a fresh model."""
    import compilation

    monkeypatch.setitem(compilation._TARGETS, "transition", (__name__, "_TinyModel"))

    class _Container(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.a = _TinyModel()

    model = _Container().eval()
    install_compilation(model, CompileConfig(transition=COMPILE_INDUCTOR, token_transformer=COMPILE_NONE))

    with pytest.raises(ValueError, match="transition.*already configured.*inductor.*none"):
        install_compilation(model, CompileConfig(transition=COMPILE_NONE, token_transformer=COMPILE_NONE))

    # The original status must be untouched by the rejected attempt.
    status = compile_status(model)["transition"]
    assert status["state"] != "eager-disabled"


def test_install_compilation_same_mode_recall_is_still_allowed(monkeypatch):
    """The rejection above must not regress same-mode idempotence: a second
    call with the SAME mode is the resumed-startup case and stays allowed."""
    import compilation

    monkeypatch.setitem(compilation._TARGETS, "transition", (__name__, "_TinyModel"))

    class _Container(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.a = _TinyModel()

    model = _Container().eval()
    config = CompileConfig(transition=COMPILE_INDUCTOR, token_transformer=COMPILE_NONE)
    install_compilation(model, config)
    install_compilation(model, config)  # must not raise


class _SwitchableModel(torch.nn.Module):
    """Same class, two instances with different forward behavior, so one
    selector label can cover a genuine mix of compiled and degraded instances
    (``force_fusion=True`` has a ``relu``, which ``_compile_that_fails_on_fused_graphs``
    fails, while ``force_fusion=False`` behaves like ``_TinyModel`` and compiles
    cleanly)."""

    def __init__(self, force_fusion: bool):
        super().__init__()
        self.lin = torch.nn.Linear(4, 4)
        self.force_fusion = force_fusion

    def forward(self, x):
        y = self.lin(x)
        if self.force_fusion:
            return torch.relu(y) * 2.0 + 1.0
        return y


def test_install_compilation_reports_mixed_partial_degradation(monkeypatch):
    """One selector label, two instances: one degrades to eager, the other
    keeps compiling and executing successfully. The aggregate state must say
    so explicitly, not collapse to a uniform degraded or uniform success."""
    import compilation

    _compile_that_fails_on_fused_graphs(monkeypatch)
    monkeypatch.setitem(compilation._TARGETS, "transition", (__name__, "_SwitchableModel"))

    class _Container(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.ok = _SwitchableModel(force_fusion=False)
            self.bad = _SwitchableModel(force_fusion=True)

    model = _Container().eval()
    install_compilation(model, CompileConfig(transition=COMPILE_INDUCTOR, token_transformer=COMPILE_NONE))

    x = torch.randn(2, 4)
    with torch.no_grad():
        model.ok(x)
        model.bad(x)

    status = compile_status(model)["transition"]
    assert status["instances"] == 2
    assert status["compiled_calls"] >= 1
    assert status["eager_fallback_calls"] >= 1
    assert status["state"] == "mixed-partial-degradation"


# --------------------------------------------------------------------------- #
# The loader receives the one setting, and the off path installs nothing
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize(("value", "expected"), [("on", True), ("off", False)])
def test_service_init_forwards_the_optimizations_flag_to_the_loader(tmp_path, monkeypatch, value, expected):
    import service

    cache_dir = tmp_path / "cache"
    cache_dir.mkdir()
    monkeypatch.setattr(service.CONFIG, "cache", cache_dir)
    monkeypatch.setattr(service.CONFIG, "download_assets", False)
    monkeypatch.setattr(service, "assets_present", lambda cache, *args: True)
    monkeypatch.setattr(service.CONFIG, "optimizations", value)
    monkeypatch.setattr(service, "resolve_device", lambda accelerator: "the-device")

    received = {}

    def _fake_load(cache, **kwargs):
        received.update(kwargs)
        return object()

    monkeypatch.setattr(service, "load_boltz2_model", _fake_load)
    service.Boltz2Prediction.inner()
    assert received == {"optimizations": expected, "device": "the-device"}


def test_loader_maps_the_flag_onto_seam_install_then_model_then_compile(monkeypatch):
    """``load_boltz2_model`` installs the seams BEFORE building the model and compiles AFTER."""
    import sys
    import types
    from pathlib import Path

    import compilation
    import runner

    events: list = []
    fake_triangle = types.ModuleType("rocm_triangle")
    fake_triangle.install = lambda mode: events.append(("seams", mode))
    fake_triangle.prepare_model = lambda model: events.append(("pack",))
    monkeypatch.setitem(sys.modules, "rocm_triangle", fake_triangle)

    fake_main = types.ModuleType("boltz.main")
    for name in ("Boltz2DiffusionParams", "BoltzSteeringParams", "MSAModuleArgs", "PairformerArgsV2"):
        setattr(fake_main, name, type(name, (), {"__init__": lambda self, *a, **k: None, "step_scale": 1.5}))
    fake_boltz2 = types.ModuleType("boltz.model.models.boltz2")

    class _Model(torch.nn.Module):
        structure_module = None

    class _Boltz2:
        @staticmethod
        def load_from_checkpoint(*args, **kwargs):
            events.append(("model", kwargs["use_kernels"]))
            return _Model()

    fake_boltz2.Boltz2 = _Boltz2
    for name in ("boltz", "boltz.model", "boltz.model.models"):
        monkeypatch.setitem(sys.modules, name, types.ModuleType(name))
    monkeypatch.setitem(sys.modules, "boltz.main", fake_main)
    monkeypatch.setitem(sys.modules, "boltz.model.models.boltz2", fake_boltz2)
    monkeypatch.setattr(runner, "asdict", lambda obj: {}, raising=False)
    monkeypatch.setattr(
        compilation, "install_compilation", lambda model, config: events.append(("compile", config)) or {}
    )

    for flag, seam_mode, compile_mode_ in ((False, "off", COMPILE_NONE), (True, "both", COMPILE_INDUCTOR)):
        events.clear()
        runner.load_boltz2_model(Path("/nonexistent"), optimizations=flag)
        # Weights are packed after the model exists and before compilation, only when optimized.
        assert [e[0] for e in events] == ["seams", "model", *(["pack"] if flag else []), "compile"]
        assert events[0] == ("seams", seam_mode)
        assert events[1] == ("model", flag)
        config = events[-1][1]
        assert (config.transition, config.token_transformer) == (compile_mode_, compile_mode_)


def test_install_compilation_none_compiles_nothing_and_inductor_wraps(monkeypatch):
    import compilation

    monkeypatch.setitem(compilation._TARGETS, "transition", (__name__, "_TinyModel"))

    class _Container(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.a = _TinyModel()

    off = _Container().eval()
    assert install_compilation(off, CompileConfig(transition=COMPILE_NONE, token_transformer=COMPILE_NONE)) == {
        "transition": 0,
        "token_transformer": 0,
    }
    assert compile_status(off)["transition"]["state"] == "eager-disabled"

    on = _Container().eval()
    monkeypatch.setitem(compilation._TARGETS, "token_transformer", (__name__, "_TinyModel"))
    counts = install_compilation(on, CompileConfig(transition=COMPILE_INDUCTOR, token_transformer=COMPILE_NONE))
    assert counts["transition"] == 1


def test_the_deterministic_setting_costs_one_recompile_for_each_state_and_then_none():
    """A seeded request turns deterministic mode on; Dynamo guards on that setting.

    The first seeded request therefore compiles every wrapped module once more.
    After that both states are cached, so a seeded and an unseeded request in
    any order compile nothing. The service documents this in ``seeded_prediction``.
    """
    from torch._dynamo.testing import CompileCounter

    counter = CompileCounter()

    @torch.compile(backend=counter, dynamic=True)
    def fn(x):
        return torch.relu(x @ x.T) + 1

    x = torch.randn(4, 4)
    before = (torch.are_deterministic_algorithms_enabled(), torch.is_deterministic_algorithms_warn_only_enabled())
    try:
        torch.use_deterministic_algorithms(False)
        fn(x)
        assert counter.frame_count == 1
        torch.use_deterministic_algorithms(True, warn_only=True)
        fn(x)
        assert counter.frame_count == 2
        torch.use_deterministic_algorithms(False)
        fn(x)
        torch.use_deterministic_algorithms(True, warn_only=True)
        fn(x)
        assert counter.frame_count == 2
    finally:
        torch.use_deterministic_algorithms(before[0], warn_only=before[1])
