# Copyright © Advanced Micro Devices, Inc., or its affiliates.
#
# SPDX-License-Identifier: MIT

"""Wall-clock budget arithmetic for a single ``/predict`` request.

BentoML caps a request with ``traffic.timeout``, but that cap only cancels the
*client* side.  ``predict`` is a sync endpoint, so BentoML runs it through
``anyio.to_thread.run_sync(..., abandon_on_cancel=False)``, and a cancelled task
does not unwind until the worker thread returns.  The thread keeps its
``CapacityLimiter`` slot, the prediction runs to completion, and the result is
thrown away: the server accepts new requests but starts none of them.

So the cap must never fire.  Each request carries a deadline set slightly inside
``traffic.timeout`` and gives up on its own, releasing the thread.  The deadline
is measured from *arrival*, not from when a thread picked the request up —
otherwise a request that queued for most of the timeout would still be handed a
full budget and would trip the cap anyway.

Kept free of bentoml/torch/openfold3 imports so it is unit-testable on its own.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

# Total wall clock a request is allowed, matching the historical hardcoded value.
DEFAULT_REQUEST_TIMEOUT_SECONDS = 600
# Cap on the ColabFold MSA phase. Generous next to a healthy server (tens of
# seconds) while still failing twice as fast as the request timeout when the
# public server throttles us.
DEFAULT_MSA_TIMEOUT_SECONDS = 300
# Budget held back for GPU work so a slow MSA cannot consume the whole request.
DEFAULT_INFERENCE_RESERVE_SECONDS = 120
# In-flight requests per worker: one running plus two queued.
DEFAULT_CONCURRENCY_PER_WORKER = 3
# Headroom between our deadline and traffic.timeout, to serialise the response
# (mmCIF payloads are large) before BentoML's timer fires.
RESPONSE_SLACK_SECONDS = 30


class DeadlineExceeded(Exception):
    """Raised when a request has too little budget left to be worth starting."""


@dataclass(frozen=True)
class RequestBudget:
    """Monotonic deadlines derived from a request's arrival time."""

    arrived_at: float
    deadline: float
    msa_deadline: float
    inference_reserve_s: float

    def remaining(self, now: float) -> float:
        return self.deadline - now

    def msa_remaining(self, now: float) -> float:
        return self.msa_deadline - now


def response_slack(request_timeout_s: float) -> float:
    """Slack to reserve for sending the response.

    Proportional for short timeouts so a deliberately small
    ``OPENFOLD3_REQUEST_TIMEOUT_SECONDS`` (smoke tests do this) still leaves a
    usable budget instead of going negative.
    """
    return min(RESPONSE_SLACK_SECONDS, request_timeout_s * 0.1)


def build_budget(
    *,
    arrived_at: float,
    now: float,
    request_timeout_s: float = DEFAULT_REQUEST_TIMEOUT_SECONDS,
    msa_timeout_s: float = DEFAULT_MSA_TIMEOUT_SECONDS,
    inference_reserve_s: float = DEFAULT_INFERENCE_RESERVE_SECONDS,
) -> RequestBudget:
    """Build the deadlines for a request that arrived at ``arrived_at``.

    All times are ``time.monotonic()`` readings. ``now`` is when a worker thread
    picked the request up; the gap from ``arrived_at`` is queue wait, which is
    already spent against ``traffic.timeout``.

    The MSA deadline is whichever comes first: its own cap, or the point where
    continuing would eat the inference reserve.
    """
    deadline = arrived_at + request_timeout_s - response_slack(request_timeout_s)
    msa_deadline = min(now + msa_timeout_s, deadline - inference_reserve_s)
    return RequestBudget(
        arrived_at=arrived_at,
        deadline=deadline,
        msa_deadline=msa_deadline,
        inference_reserve_s=inference_reserve_s,
    )


def ensure_startable(budget: RequestBudget, *, needs_msa: bool, now: float) -> None:
    """Reject a request that queued so long it cannot finish in what remains.

    Starting it anyway would produce work nobody can receive: the answer would
    land after the cap has already returned an error to the client, and the
    thread would be held for the whole run.
    """
    queued_for = now - budget.arrived_at
    remaining = budget.remaining(now)

    if remaining <= 0:
        raise DeadlineExceeded(
            f"Server busy: request waited {queued_for:.0f}s in the queue, leaving no budget "
            f"at all — the deadline has already passed. Retry when the server is less loaded."
        )

    if remaining < budget.inference_reserve_s:
        raise DeadlineExceeded(
            f"Server busy: request waited {queued_for:.0f}s in the queue, leaving "
            f"{max(remaining, 0.0):.0f}s of its budget — under the {budget.inference_reserve_s:.0f}s "
            f"reserved for inference. Retry when the server is less loaded."
        )

    if needs_msa and budget.msa_remaining(now) <= 0:
        raise DeadlineExceeded(
            f"Server busy: request waited {queued_for:.0f}s in the queue, leaving no "
            f"budget for the MSA stage. Retry when the server is less loaded, or supply "
            f"inline MSAs with use_msa_server=false."
        )


def max_concurrency_for(
    accelerator_count: int,
    per_worker: int = DEFAULT_CONCURRENCY_PER_WORKER,
) -> int:
    """Service-wide ``traffic.max_concurrency`` giving ``per_worker`` per worker.

    BentoML divides the service-wide value by the worker count
    (``ceil(max_concurrency / workers)``), and this service runs one worker per
    accelerator, so the value has to be pre-multiplied.

    Beyond this, requests get an immediate ``429`` instead of queueing into a
    timeout — the queue is bounded, so waiting no longer consumes a request's
    whole budget.
    """
    return max(1, math.ceil(max(1, accelerator_count) * max(1, per_worker)))
