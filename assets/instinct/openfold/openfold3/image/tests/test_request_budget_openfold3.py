# Copyright © Advanced Micro Devices, Inc., or its affiliates.
#
# SPDX-License-Identifier: MIT

"""Guards on the OpenFold3 request budget.

The budget exists because BentoML's ``traffic.timeout`` cannot stop a running
prediction: ``predict`` is a sync endpoint, so it runs in a worker thread via
``anyio.to_thread.run_sync(..., abandon_on_cancel=False)`` and a cancelled task
does not unwind until that thread returns. The client gets a 504, the thread
keeps its slot, and the server accepts requests it cannot start. So every
request has to give up on its own, before the cap fires.

Loaded from the asset path; the module is deliberately pure stdlib so these run
without torch, bentoml or OpenFold3 installed.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

OF3_BUDGET_PATH = Path(__file__).resolve().parents[1] / "src/request_budget.py"


def _load_budget():
    name = "_of3_budget_under_test"
    spec = importlib.util.spec_from_file_location(name, OF3_BUDGET_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    # @dataclass resolves annotations through sys.modules[cls.__module__].
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


budget_mod = _load_budget()


def _budget(*, arrived_at=1000.0, now=None, timeout=600, msa=300, reserve=120):
    return budget_mod.build_budget(
        arrived_at=arrived_at,
        now=arrived_at if now is None else now,
        request_timeout_s=timeout,
        msa_timeout_s=msa,
        inference_reserve_s=reserve,
    )


# --- the deadline stays inside traffic.timeout ------------------------------


def test_deadline_lands_before_the_serving_cap():
    """Returning after the cap fires is the failure mode being designed out."""
    b = _budget(arrived_at=1000.0, timeout=600)

    assert b.deadline < 1000.0 + 600


def test_deadline_is_measured_from_arrival_not_from_pickup():
    """A queued request must not be handed a fresh full budget."""
    queued = _budget(arrived_at=1000.0, now=1400.0, timeout=600)
    prompt = _budget(arrived_at=1000.0, now=1000.0, timeout=600)

    assert queued.deadline == prompt.deadline
    assert queued.remaining(1400.0) < prompt.remaining(1000.0)


def test_short_timeouts_still_leave_a_usable_budget():
    """Smoke tests dial the timeout right down; slack must not swallow it -- a
    nanosecond of remaining budget is as useless as none."""
    b = _budget(arrived_at=0.0, timeout=20, msa=5, reserve=5)

    assert b.remaining(0.0) >= 0.5 * 20


# --- the MSA phase cannot consume the whole request -------------------------


def test_msa_budget_is_capped_by_its_own_timeout():
    b = _budget(arrived_at=1000.0, timeout=600, msa=300, reserve=120)

    assert b.msa_remaining(1000.0) == pytest.approx(300)


def test_msa_budget_shrinks_to_protect_the_inference_reserve():
    """A generous MSA cap must not eat the time inference still needs."""
    b = _budget(arrived_at=1000.0, timeout=600, msa=10_000, reserve=120)

    assert b.msa_deadline == pytest.approx(b.deadline - 120)


def test_msa_budget_shrinks_further_when_the_request_queued():
    early = _budget(arrived_at=1000.0, now=1000.0, msa=10_000)
    late = _budget(arrived_at=1000.0, now=1300.0, msa=10_000)

    assert late.msa_remaining(1300.0) < early.msa_remaining(1000.0)


# --- work that cannot finish is never started -------------------------------


def test_a_prompt_request_is_startable():
    b = _budget(arrived_at=1000.0, now=1000.0)

    budget_mod.ensure_startable(b, needs_msa=True, now=1000.0)


def test_a_request_that_queued_past_its_budget_is_refused():
    b = _budget(arrived_at=1000.0, now=1550.0, timeout=600, reserve=120)

    with pytest.raises(budget_mod.DeadlineExceeded):
        budget_mod.ensure_startable(b, needs_msa=False, now=1550.0)


def test_refusal_explains_the_queue_wait():
    """The caller needs to know it was turned away for waiting, not for failing.

    ``arrived_at``/``now`` are chosen so the queue-wait figure (455) cannot be
    confused with either input value, unlike the earlier 1000.0/1550.0 pair
    where the raw ``now`` already contained the substring "550"."""
    b = _budget(arrived_at=200.0, now=655.0)

    with pytest.raises(budget_mod.DeadlineExceeded) as excinfo:
        budget_mod.ensure_startable(b, needs_msa=False, now=655.0)

    assert "waited 455s" in str(excinfo.value)


def test_a_request_with_no_msa_budget_left_is_refused_before_starting():
    """Inference alone would fit, but the MSA stage it needs no longer would."""
    b = _budget(arrived_at=1000.0, now=1000.0, timeout=600, msa=0, reserve=120)

    with pytest.raises(budget_mod.DeadlineExceeded):
        budget_mod.ensure_startable(b, needs_msa=True, now=1000.0)


def test_the_same_request_is_allowed_when_it_brings_its_own_msas():
    """No server round trip needed, so an exhausted MSA budget is irrelevant."""
    b = _budget(arrived_at=1000.0, now=1000.0, timeout=600, msa=0, reserve=120)

    budget_mod.ensure_startable(b, needs_msa=False, now=1000.0)


# --- admission control ------------------------------------------------------


def test_concurrency_is_premultiplied_by_the_worker_count():
    """BentoML divides the service-wide value by workers; this service runs one per GPU."""
    assert budget_mod.max_concurrency_for(8, per_worker=3) == 24


def test_default_concurrency_leaves_room_for_one_running_and_two_queued():
    assert budget_mod.max_concurrency_for(1) == budget_mod.DEFAULT_CONCURRENCY_PER_WORKER


@pytest.mark.parametrize("accelerators", [0, -1])
def test_concurrency_never_drops_to_zero(accelerators):
    """A bad accelerator count must not wedge the service into refusing everything."""
    assert budget_mod.max_concurrency_for(accelerators) >= 1


@pytest.mark.parametrize("per_worker", [0, -1])
def test_concurrency_never_drops_to_zero_from_a_bad_per_worker(per_worker):
    """A misconfigured per-worker value must not wedge the service either.

    Asserts the clamped-to-1 result (4) rather than merely `>= 1`: the
    surrounding `max(1, ...)` on the whole product would keep any `>= 1`
    assertion passing even if the per-worker clamp were dropped entirely."""
    assert budget_mod.max_concurrency_for(4, per_worker=per_worker) == 4


# --- a deadline that has already passed is refused outright -----------------


def test_an_expired_deadline_is_refused_even_with_a_nonsensical_reserve():
    """The absolute guard must fire before the reserve check even looks at a
    broken reserve: remaining=-50 with inference_reserve_s=-1000 satisfies
    `remaining < inference_reserve_s`'s negation, so only a dedicated check on
    remaining <= 0 (not the reserve comparison) catches this."""
    b = _budget(arrived_at=1000.0, now=1620.0, timeout=600, reserve=-1000)
    assert b.remaining(1620.0) == pytest.approx(-50.0)

    with pytest.raises(budget_mod.DeadlineExceeded) as excinfo:
        budget_mod.ensure_startable(b, needs_msa=False, now=1620.0)

    message = str(excinfo.value)
    assert "deadline has already passed" in message
    assert "reserved for inference" not in message


def test_an_expired_deadline_is_refused_with_a_sane_reserve_too():
    b = _budget(arrived_at=1000.0, now=1620.0, timeout=600, reserve=120)

    with pytest.raises(budget_mod.DeadlineExceeded) as excinfo:
        budget_mod.ensure_startable(b, needs_msa=False, now=1620.0)

    assert "deadline has already passed" in str(excinfo.value)


# --- response_slack's proportional clamp ------------------------------------


def test_response_slack_is_proportional_below_the_flat_cap():
    """Below the crossover (timeout * 0.1 < RESPONSE_SLACK_SECONDS), the
    proportional share applies, not the flat cap."""
    assert budget_mod.response_slack(299) == pytest.approx(29.9)


def test_response_slack_is_the_flat_cap_above_the_crossover():
    """Above the crossover, the flat cap wins so slack stops growing with a
    large timeout."""
    assert budget_mod.response_slack(301) == pytest.approx(30.0)
    assert budget_mod.response_slack(600) == pytest.approx(30.0)
