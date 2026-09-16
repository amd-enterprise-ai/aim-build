# Copyright © Advanced Micro Devices, Inc., or its affiliates.
#
# SPDX-License-Identifier: MIT

"""Guards on how the OpenFold3 service bounds and sheds work.

Three defects combined to soft-lock the service under a throttled ColabFold MSA
server: the MSA client retries forever, BentoML's request cap cancels the client
but not the worker thread running the prediction, and nothing turned excess work
away. The wiring asserted here is what keeps each of those bounded — none of it
is exercised by a normal single-request run, so it is easy to remove by accident.

Assertions are made against the parsed AST and the patch text rather than a live
import: the asset's ``service.py`` pulls in bentoml, torch and openfold3, none of
which are installed in this test environment.
"""

from __future__ import annotations

import ast
import textwrap
from pathlib import Path

import pytest

OF3_IMAGE_DIR = Path(__file__).resolve().parents[1]
OF3_SERVICE_PATH = OF3_IMAGE_DIR / "src/service.py"
OF3_RUNNER_PATH = OF3_IMAGE_DIR / "src/runner.py"
OF3_DOCKERFILE_PATH = OF3_IMAGE_DIR / "Dockerfile"
OF3_DEADLINE_PATCH_PATH = OF3_IMAGE_DIR / "patches/of3_msa_server_deadline.patch"

SERVICE_CLASS = "OpenFold3Prediction"


@pytest.fixture
def service_tree() -> ast.Module:
    return ast.parse(OF3_SERVICE_PATH.read_text(encoding="utf-8"))


@pytest.fixture
def service_source() -> str:
    return OF3_SERVICE_PATH.read_text(encoding="utf-8")


@pytest.fixture
def patch_text() -> str:
    return OF3_DEADLINE_PATCH_PATH.read_text(encoding="utf-8")


def _service_class(tree: ast.Module) -> ast.ClassDef:
    for node in tree.body:
        if isinstance(node, ast.ClassDef) and node.name == SERVICE_CLASS:
            return node
    raise AssertionError(f"{SERVICE_CLASS} not found")


def _traffic_config(tree: ast.Module) -> ast.Dict:
    for decorator in _service_class(tree).decorator_list:
        if not (
            isinstance(decorator, ast.Call)
            and isinstance(decorator.func, ast.Attribute)
            and decorator.func.attr == "service"
        ):
            continue
        for kw in decorator.keywords:
            if kw.arg == "traffic":
                assert isinstance(kw.value, ast.Dict), "traffic must be a dict literal"
                return kw.value
    raise AssertionError("no traffic config on @bentoml.service")


def _traffic_keys(tree: ast.Module) -> set[str]:
    return {k.value for k in _traffic_config(tree).keys if isinstance(k, ast.Constant)}


def _method(tree: ast.Module, name: str) -> ast.FunctionDef:
    for node in _service_class(tree).body:
        if isinstance(node, ast.FunctionDef) and node.name == name:
            return node
    raise AssertionError(f"{SERVICE_CLASS}.{name} not found")


# --- the request cap must never be what stops a request ---------------------


def test_the_serving_cap_is_configurable_rather_than_hardcoded(service_tree):
    """Operators need to raise it for MSA-heavy work without rebuilding the image."""
    traffic = _traffic_config(service_tree)
    timeout = next(v for k, v in zip(traffic.keys, traffic.values) if getattr(k, "value", None) == "timeout")

    assert not isinstance(timeout, ast.Constant), "traffic.timeout is hardcoded again"


def test_requests_carry_their_own_deadline(service_tree):
    """Without one, the cap fires instead — and cancelling it does not stop the work."""
    predict = _method(service_tree, "predict")
    calls = [n for n in ast.walk(predict) if isinstance(n, ast.Call)]

    assert any(
        isinstance(c.func, ast.Name) and c.func.id == "build_budget" for c in calls
    ), "predict() no longer builds a budget"

    run_calls = [c for c in calls if isinstance(c.func, ast.Name) and c.func.id == "run_openfold3_prediction"]
    assert run_calls, "predict() no longer calls run_openfold3_prediction"

    msa_deadline_kwargs = [kw for c in run_calls for kw in c.keywords if kw.arg == "msa_deadline"]
    assert msa_deadline_kwargs, "the MSA stage is no longer given a deadline"
    assert any(
        isinstance(kw.value, ast.Attribute) and kw.value.attr == "msa_deadline" for kw in msa_deadline_kwargs
    ), "msa_deadline is no longer read off the request's own budget"


def test_the_deadline_is_anchored_to_arrival(service_tree, service_source):
    """Queue time is already spent against the cap; ignoring it re-opens the gap."""
    assert "ArrivalStampMiddleware" in service_source
    assert "add_asgi_middleware" in service_source

    predict = _method(service_tree, "predict")
    build_budget_calls = [
        n
        for n in ast.walk(predict)
        if isinstance(n, ast.Call) and isinstance(n.func, ast.Name) and n.func.id == "build_budget"
    ]
    assert build_budget_calls, "predict() no longer builds a budget"

    arrived_at_kwargs = [kw for c in build_budget_calls for kw in c.keywords if kw.arg == "arrived_at"]
    assert arrived_at_kwargs, "build_budget() is no longer given an arrived_at"
    assert any(
        "ARRIVAL_SCOPE_KEY" in ast.dump(kw.value) for kw in arrived_at_kwargs
    ), "arrived_at is no longer derived from the arrival stamp"


def test_work_that_cannot_finish_is_refused_before_it_starts(service_tree):
    predict = _method(service_tree, "predict")
    calls = [n for n in ast.walk(predict) if isinstance(n, ast.Call)]

    ensure_calls = [c for c in calls if isinstance(c.func, ast.Name) and c.func.id == "ensure_startable"]
    assert ensure_calls, "predict() no longer calls ensure_startable"

    needs_msa_kwargs = [kw for c in ensure_calls for kw in c.keywords if kw.arg == "needs_msa"]
    assert needs_msa_kwargs, "ensure_startable is no longer told whether MSA is needed"
    assert any(
        not isinstance(kw.value, ast.Constant) and "use_msa_server" in ast.dump(kw.value) for kw in needs_msa_kwargs
    ), "needs_msa is hardcoded rather than reflecting the request's use_msa_server"


# --- overload is turned away, not queued ------------------------------------


def test_the_service_sheds_load_instead_of_queueing_indefinitely(service_tree):
    assert "max_concurrency" in _traffic_keys(service_tree)


def test_shedding_scales_with_the_accelerator_count(service_tree, service_source):
    """A fixed number would starve a multi-GPU deployment or overfill a single-GPU one."""
    assert "effective_max_concurrency" in service_source
    assert "max_concurrency_for" in service_source


# --- transient failures are distinguishable from real ones ------------------


def test_transient_failures_answer_503_with_a_reason(service_tree, service_source):
    """503 plus a usable message is why the status is set via the context, not raised."""
    unavailable = _method(service_tree, "_unavailable")
    dumped = ast.dump(unavailable)

    assert "503" in dumped
    assert "Retry-After" in dumped
    assert "message" in dumped


def test_a_throttled_msa_server_is_reported_as_transient(service_tree):
    predict = _method(service_tree, "predict")
    calls = [n for n in ast.walk(predict) if isinstance(n, ast.Call)]

    assert any(isinstance(c.func, ast.Name) and c.func.id == "is_msa_server_timeout" for c in calls)
    assert any(
        isinstance(h.type, ast.Name) and h.type.id == "DeadlineExceeded"
        for h in ast.walk(predict)
        if isinstance(h, ast.ExceptHandler)
    )


def test_ordinary_prediction_failures_keep_answering_200(service_tree):
    """Existing clients and harness.py read the error out of a 200 body."""
    predict = _method(service_tree, "predict")
    handlers = [h for h in ast.walk(predict) if isinstance(h, ast.ExceptHandler)]

    catch_all = [h for h in handlers if isinstance(h.type, ast.Name) and h.type.id == "Exception"]
    assert catch_all, "the catch-all handler is gone"
    assert any(
        isinstance(n, ast.Return) and isinstance(n.value, ast.Dict) for h in catch_all for n in ast.walk(h)
    ), "generic failures no longer return the 200 error body"


# --- the bound cannot silently disappear ------------------------------------


def _is_negated_hook_check(test: ast.expr) -> bool:
    return (
        isinstance(test, ast.UnaryOp)
        and isinstance(test.op, ast.Not)
        and isinstance(test.operand, ast.Call)
        and isinstance(test.operand.func, ast.Name)
        and test.operand.func.id == "msa_deadline_hook_available"
    )


def test_startup_fails_if_the_msa_deadline_patch_is_missing(service_tree):
    """An upstream bump that defeats the patch must not quietly serve unbounded."""
    init = _method(service_tree, "__init__")

    guards = [n for n in ast.walk(init) if isinstance(n, ast.If) and _is_negated_hook_check(n.test)]
    assert guards, "no `if not msa_deadline_hook_available():` guard found"
    assert any(
        isinstance(n, ast.Raise) for guard in guards for n in ast.walk(guard)
    ), "the guard against a missing MSA deadline patch does not raise"


def test_the_image_applies_the_msa_deadline_patch():
    dockerfile = OF3_DOCKERFILE_PATH.read_text(encoding="utf-8")

    assert "of3_msa_server_deadline.patch" in dockerfile
    assert dockerfile.count("of3_msa_server_deadline.patch") >= 2, "--check pass is missing"


def test_every_unbounded_retry_loop_gained_a_deadline_check(patch_text):
    """Six loops in the MSA client plus the template fetch; missing one re-opens the hang."""
    added = [line for line in patch_text.splitlines() if line.startswith("+")]

    call_sites = [line for line in added if "check_deadline(" in line and "def check_deadline(" not in line]
    assert len(call_sites) >= 7, "a check_deadline() call site was dropped from a retry loop"


def test_the_patch_makes_the_dead_retry_cap_reachable(patch_text):
    """The template retry counter must be hoisted out of its loop."""
    added = [line[1:].strip() for line in patch_text.splitlines() if line.startswith("+")]
    removed = [line[1:].strip() for line in patch_text.splitlines() if line.startswith("-")]

    removed_hoists = sum(1 for line in removed if line == "error_count = 0")
    added_hoists = sum(1 for line in added if line == "error_count = 0")

    assert removed_hoists == 1, "expected the in-loop reset removed from the template retry loop"
    assert added_hoists == removed_hoists, "the hoisted error_count = 0 was not re-added for every loop"


def test_exhausting_the_retry_cap_is_reported_as_transient(patch_text):
    """Hoisting error_count made the cap reachable; a bare re-raise then surfaces a
    requests error, which the service cannot tell from a real bug — so it answers
    200-with-error-body instead of the 503 the deadline path gives for the same
    condition. Both must raise the type the service classifies as transient."""
    added = [line[1:] for line in patch_text.splitlines() if line.startswith("+")]
    removed = [line[1:].strip() for line in patch_text.splitlines() if line.startswith("-")]

    assert removed.count("raise") == 4, "expected the bare re-raise removed from all four retry loops"
    assert (
        sum("raise MsaServerTimeout(" in line for line in added) >= 5
    ), "every retry-cap exhaustion must raise MsaServerTimeout, alongside the deadline check"
    assert any("from e" in line for line in added), "the underlying error must stay on the cause chain"


def test_the_patch_backs_off_instead_of_spinning_on_timeouts(patch_text):
    """Upstream `continue`s with no sleep, turning a timeout into a hot retry loop."""
    added = [line for line in patch_text.splitlines() if line.startswith("+")]

    assert sum("time.sleep(5)" in line for line in added) >= 4


def test_the_patch_exposes_what_the_service_relies_on(patch_text):
    """service.py's startup guard and runner.py's plumbing look these up by name."""
    for symbol in ("msa_deadline", "MsaServerTimeout", "_resolve_msa_deadline"):
        assert symbol in patch_text


def test_the_msa_budget_env_var_is_read_by_both_sides(patch_text):
    """The client falls back to it if a request ever reaches it without a deadline."""
    service_source = OF3_SERVICE_PATH.read_text(encoding="utf-8")

    assert "OPENFOLD3_MSA_TIMEOUT_SECONDS" in patch_text
    assert "OPENFOLD3_MSA_TIMEOUT_SECONDS" in service_source


def test_the_runner_bounds_the_call_that_reaches_the_msa_server():
    """MSA runs inside expt_runner.run(), so the bound has to wrap that call."""
    tree = ast.parse(OF3_RUNNER_PATH.read_text(encoding="utf-8"))

    withs = [n for n in ast.walk(tree) if isinstance(n, ast.With)]
    bounded = [
        w
        for w in withs
        if any(
            isinstance(item.context_expr, ast.Call)
            and isinstance(item.context_expr.func, ast.Name)
            and item.context_expr.func.id == "bounded_msa_server"
            for item in w.items
        )
    ]
    assert bounded, "expt_runner.run() is no longer wrapped in an MSA deadline"

    wrapped = ast.dump(bounded[0])
    assert "'run'" in wrapped or '"run"' in wrapped


# --- template preprocessing must not fork on the served path ----------------

OF3_TEMPLATE_PATCH_PATH = OF3_IMAGE_DIR / "patches/of3_template_no_fork.patch"


@pytest.fixture
def template_patch_text() -> str:
    return OF3_TEMPLATE_PATCH_PATH.read_text(encoding="utf-8")


def test_the_image_applies_the_template_no_fork_patch():
    dockerfile = OF3_DOCKERFILE_PATH.read_text(encoding="utf-8")

    assert "of3_template_no_fork.patch" in dockerfile
    assert dockerfile.count("of3_template_no_fork.patch") >= 2, "--check pass is missing"


def _diff_hunks(patch_text: str) -> list[list[str]]:
    """Split a unified diff into hunks; each is its raw context/+/- lines."""
    hunks: list[list[str]] = []
    current: list[str] | None = None
    for line in patch_text.splitlines():
        if line.startswith("@@"):
            current = []
            hunks.append(current)
        elif current is not None:
            current.append(line)
    return hunks


def _patched_lines(hunk: list[str]) -> list[str]:
    """A hunk's lines as they read *after* applying it: context plus additions, with
    removed lines dropped. This reconstructs real post-patch source straight from the
    diff, so the tests below exercise parsed code rather than the diff's own wording."""
    return [line[1:] for line in hunk if line.startswith("+") or line.startswith(" ")]


def _call_site_source(hunk: list[str]) -> str | None:
    """Parseable source for the `__call__` method touched by this hunk, or None."""
    lines = _patched_lines(hunk)
    starts = [i for i, line in enumerate(lines) if line.lstrip().startswith("def __call__(")]
    if not starts:
        return None
    body = textwrap.dedent("\n".join(lines[starts[0] :])).splitlines()
    end = next(
        (i for i, line in enumerate(body) if i and (line.startswith("def ") or line.startswith("class "))),
        len(body),
    )
    return "\n".join(body[:end])


def _contains_pool_call(nodes: list[ast.AST]) -> bool:
    return any(
        isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute) and n.func.attr == "Pool"
        for node in nodes
        for n in ast.walk(node)
    )


def _contains_n_processes_call(nodes: list[ast.AST]) -> bool:
    return any(
        isinstance(n, ast.Call) and any(kw.arg == "n_processes" for kw in n.keywords)
        for node in nodes
        for n in ast.walk(node)
    )


def _is_n_processes(node: ast.expr) -> bool:
    return (isinstance(node, ast.Name) and node.id == "n_processes") or (
        isinstance(node, ast.Attribute) and node.attr == "n_processes"
    )


def _tests_n_processes_le_1(test: ast.expr) -> bool:
    return (
        isinstance(test, ast.Compare)
        and _is_n_processes(test.left)
        and len(test.ops) == 1
        and isinstance(test.ops[0], ast.LtE)
        and len(test.comparators) == 1
        and isinstance(test.comparators[0], ast.Constant)
        and test.comparators[0].value == 1
    )


def test_the_served_template_call_sites_no_longer_fork(template_patch_text):
    """The patch takes three class __call__ sites off mp.Pool; TemplatePreprocessor
    is the one a served request reaches, so a hunk that regresses to building its
    own pool can deadlock the worker again. Each hunk is reconstructed into the
    real post-patch method body and parsed, so a rename of the helper does not
    spuriously fail this."""
    fork_site_hunks = [
        hunk
        for hunk in _diff_hunks(template_patch_text)
        if any(line.startswith("-") and "mp.Pool(" in line for line in hunk) and _call_site_source(hunk)
    ]
    assert len(fork_site_hunks) == 3, "expected all three unconditional-pool call sites to be touched"

    for hunk in fork_site_hunks:
        call_site = ast.parse(_call_site_source(hunk)).body[0]
        assert not _contains_pool_call([call_site]), "a patched call site still builds its own mp.Pool"
        assert _contains_n_processes_call(
            [call_site]
        ), "a patched call site no longer routes through anything driven by n_processes"


def test_the_manager_is_skipped_when_running_in_process(template_patch_text):
    """mp.Manager forks a server process of its own; it exists only to share state
    with forked workers, so a serial run must not start one."""
    manager_hunks = [hunk for hunk in _diff_hunks(template_patch_text) if any("mp.Manager()" in line for line in hunk)]
    assert len(manager_hunks) == 1, "expected exactly one hunk touching the Manager"

    call_site = ast.parse(_call_site_source(manager_hunks[0])).body[0]
    guards = [n for n in ast.walk(call_site) if isinstance(n, ast.If) and _tests_n_processes_le_1(n.test)]
    assert guards, "no `if self.n_processes <= 1:` guard found around the manager"
    guard = guards[0]

    assert not any(
        isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute) and n.func.attr == "Manager"
        for stmt in guard.body
        for n in ast.walk(stmt)
    ), "the in-process branch still creates a Manager"
    assert any(
        isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute) and n.func.attr == "Manager"
        for stmt in guard.orelse
        for n in ast.walk(stmt)
    ), "the Manager must still be created when forking is asked for"
    assert any(
        isinstance(n, ast.Dict) and not n.keys for stmt in guard.body for n in ast.walk(stmt)
    ), "no plain-dict path for the serial case"


def test_the_helper_only_forks_when_more_than_one_process_is_asked_for(template_patch_text):
    """The one function this patch adds decides serial-vs-pool for every call site;
    an off-by-one guard (`n_processes < 1`) or a pool built inside the serial branch
    would reopen the fork for every caller that trusts it."""
    helper_hunks = [hunk for hunk in _diff_hunks(template_patch_text) if not _call_site_source(hunk)]
    assert len(helper_hunks) == 1, "expected exactly one hunk introducing the helper"

    added_only = textwrap.dedent("\n".join(line[1:] for line in helper_hunks[0] if line.startswith("+")))
    helpers = [n for n in ast.parse(added_only).body if isinstance(n, ast.FunctionDef)]
    assert len(helpers) == 1, "expected exactly one new top-level function"
    helper = helpers[0]

    guards = [n for n in ast.walk(helper) if isinstance(n, ast.If) and _tests_n_processes_le_1(n.test)]
    assert guards, "no `if n_processes <= 1:` branch in the helper"
    guard = guards[0]

    assert not _contains_pool_call(guard.body), "the in-process branch still constructs a pool"
    assert _contains_pool_call(helper.body), "the helper no longer forks when parallelism is genuinely requested"
