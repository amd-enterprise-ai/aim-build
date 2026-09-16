# Copyright © Advanced Micro Devices, Inc., or its affiliates.
#
# SPDX-License-Identifier: MIT

"""
BentoML service for OpenFold3 structure predictions.

Accepts a JSON payload (OpenFold3 query format: queries with chains)
and returns structure predictions (coordinates, confidence, mmCIF/PDB string).
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any

import bentoml
from gpu_affinity import pin_worker_device
from pydantic import BaseModel, Field, field_validator, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict
from request_budget import (
    DEFAULT_INFERENCE_RESERVE_SECONDS,
    DEFAULT_MSA_TIMEOUT_SECONDS,
    DEFAULT_REQUEST_TIMEOUT_SECONDS,
    DeadlineExceeded,
    build_budget,
    ensure_startable,
    max_concurrency_for,
    response_slack,
)
from runner import (
    _ensure_model_parameters,
    _load_model,
    find_request_conflicts,
    is_msa_server_timeout,
    msa_deadline_hook_available,
    run_openfold3_prediction,
)

from aim_runtime.config import DEFAULT_CACHE_PATH


class OpenFold3Request(BaseModel):
    queries: dict[str, dict[str, Any]]
    seeds: list[int] = Field(default_factory=lambda: [42])
    num_diffusion_samples: int = 1
    # Mirrors OF3's run_openfold.py: defaults to None so the explicit `seeds`
    # list is used; when set, OF3 overrides seeds with generate_seeds(42, N).
    num_model_seeds: int | None = None
    use_msa_server: bool = True
    use_templates: bool = True
    output_format: str = "mmcif"
    include_atom_confidences: bool = False

    @model_validator(mode="after")
    def _reject_conflicting_inputs(self) -> "OpenFold3Request":
        """Reject mutually-exclusive inputs. The raised ValueError surfaces as a
        pydantic ValidationError -> HTTP 400, before any compute runs."""
        conflicts = find_request_conflicts(
            self.queries,
            use_msa_server=self.use_msa_server,
            use_templates=self.use_templates,
            num_model_seeds=self.num_model_seeds,
            seeds_explicit="seeds" in self.model_fields_set,
        )
        if conflicts:
            raise ValueError(" ".join(conflicts))
        return self


class ServiceConfig(BaseSettings):
    """Schema for service-level configuration injected via environment variables.

    When ``aim-runtime serve`` launches ``bentoml serve``, it exports each
    entry in the active profile's ``env_vars`` block into the subprocess
    environment.  ``BaseSettings`` reads those variables at service startup,
    so swapping the active profile (``AIM_PROFILE_ID``) is sufficient to
    reconfigure the service without touching this file.

    Each field declares its env var name explicitly via ``validation_alias``.
    ``cache`` reads the canonical ``AIM_CACHE_PATH`` so the checkpoint lands
    where every other AIM caches its weights, and a deployment redirects it the
    same way it would for any AIM.  Nothing sets that variable in the image and
    aim-runtime resolves it in-memory without re-exporting, so the subprocess
    usually sees it unset, hence the default.

    ``accelerator_count`` reuses ``AIM_ACCELERATOR_COUNT`` — the canonical
    aim-runtime env var that the operator already has to set so profile
    selection matches ``metadata.accelerator_count`` (see
    ``aim_runtime.config._read_accelerator_count``).  Reusing it avoids an
    OF3-only duplicate env var that would need to be kept in lockstep with
    the operator-supplied one.  It also fixes the number of data-parallel
    workers, so a profile can never request more workers than accelerators.

    Note: aim-runtime accepts ``AIM_ACCELERATOR_COUNT=auto`` and resolves it
    via accelerator detection, but the resolved int is kept in-memory only
    and never re-exported.  The bentoml subprocess therefore sees the literal
    operator-supplied string.  This service does not support ``auto`` — the
    operator must pin an integer (typed as ``int`` here so pydantic raises
    a clear ValidationError at startup if ``auto`` leaks through).
    """

    model_config = SettingsConfigDict()

    accelerator_count: int = Field(default=1, validation_alias="AIM_ACCELERATOR_COUNT")
    num_workers: int = Field(default=0, validation_alias="OPENFOLD3_NUM_WORKERS")
    accelerator: str = Field(default="gpu", validation_alias="OPENFOLD3_ACCELERATOR")
    cache: Path = Field(
        default=Path(DEFAULT_CACHE_PATH),
        validation_alias="AIM_CACHE_PATH",
    )
    request_timeout_seconds: int = Field(
        default=DEFAULT_REQUEST_TIMEOUT_SECONDS,
        validation_alias="OPENFOLD3_REQUEST_TIMEOUT_SECONDS",
        gt=0,
    )
    # Also read by the patched OF3 MSA client as its own fallback, so the two
    # agree even if a request reaches the client without a deadline set.
    msa_timeout_seconds: int = Field(
        default=DEFAULT_MSA_TIMEOUT_SECONDS,
        validation_alias="OPENFOLD3_MSA_TIMEOUT_SECONDS",
        gt=0,
    )
    inference_reserve_seconds: int = Field(
        default=DEFAULT_INFERENCE_RESERVE_SECONDS,
        validation_alias="OPENFOLD3_INFERENCE_RESERVE_SECONDS",
        ge=0,
    )
    # 0 means "derive from accelerator_count"; see effective_max_concurrency.
    max_concurrency: int = Field(default=0, validation_alias="OPENFOLD3_MAX_CONCURRENCY", ge=0)

    @field_validator("cache", mode="before")
    @classmethod
    def _expand_cache(cls, v: Any) -> Path:
        return Path(v).expanduser().resolve()

    @model_validator(mode="after")
    def _reject_unstartable_reserve(self) -> ServiceConfig:
        """A reserve at or past the usable budget would refuse every request at
        arrival, with ensure_startable's "Retry when the server is less loaded"
        message — misleading, since retrying never helps a fixed misconfiguration."""
        usable_s = self.request_timeout_seconds - response_slack(self.request_timeout_seconds)
        if self.inference_reserve_seconds >= usable_s:
            raise ValueError(
                f"OPENFOLD3_INFERENCE_RESERVE_SECONDS ({self.inference_reserve_seconds}s) leaves no "
                f"startable budget: OPENFOLD3_REQUEST_TIMEOUT_SECONDS ({self.request_timeout_seconds}s) "
                f"gives {usable_s:.0f}s usable after response slack, and the reserve must be smaller "
                f"than that or every request is refused on arrival."
            )
        return self

    @property
    def effective_max_concurrency(self) -> int:
        """In-flight requests the service accepts before answering 429."""
        return self.max_concurrency or max_concurrency_for(self.accelerator_count)


CONFIG = ServiceConfig()

# Scope key carrying the request's arrival time. Read back in predict() to
# measure the budget from arrival rather than from when a thread picked the
# request up.
ARRIVAL_SCOPE_KEY = "aim_arrived_at"


class ArrivalStampMiddleware:
    """Record when each request arrived, before it queues for a worker thread.

    ``traffic.timeout`` is counted from arrival, but ``predict`` only starts
    running once a worker thread frees up. Without this, a request that spent
    most of the timeout queued would still be handed a full budget and would
    trip the cap anyway — the outcome this whole change exists to avoid.
    """

    def __init__(self, app: Any) -> None:
        self.app = app

    async def __call__(self, scope: Any, receive: Any, send: Any) -> Any:
        if scope["type"] == "http":
            scope[ARRIVAL_SCOPE_KEY] = time.monotonic()
        return await self.app(scope, receive, send)


class BentoArgs(BaseModel):
    """Template arguments AIM forwards via ``bentoml serve --arg``.

    ``aim-runtime`` resolves ``AIM_PORT`` and injects it as ``--arg port=…``
    (an unconditional system override in ``CommandGenerator``), so this is the
    only source of the bound port.  ``--arg`` is generic template data that
    BentoML does not act on by itself, hence the ``http={"port": …}`` below.

    Deliberately no default: a bare ``bentoml serve`` is not a supported entry
    point, and a second default here could silently drift from ``AIM_PORT``.
    """

    port: int


args = bentoml.use_arguments(BentoArgs)


@bentoml.service(
    resources={"gpu": CONFIG.accelerator_count},
    # One data-parallel worker per accelerator: OpenFold3 has no tensor
    # parallelism, and a worker serves one request at a time (threads defaults
    # to 1), so worker count is the ceiling on concurrency.
    # https://docs.bentoml.com/en/latest/build-with-bentoml/parallelize-requests.html
    workers=CONFIG.accelerator_count,
    # timeout is a backstop only: predict() carries a deadline set inside it and
    # returns on its own. Letting the middleware fire instead would 504 the
    # client while the worker thread runs on (sync endpoints go through
    # anyio.to_thread with abandon_on_cancel=False, and a thread cannot be
    # killed), so the server would keep accepting work it could not start.
    #
    # max_concurrency bounds the queue: BentoML divides it by the worker count,
    # so this is three in flight per worker. Past that a client gets an
    # immediate 429 rather than a slot in a queue it will time out in.
    traffic={
        "timeout": CONFIG.request_timeout_seconds,
        "max_concurrency": CONFIG.effective_max_concurrency,
    },
    http={"port": args.port},
)
class OpenFold3Prediction:
    """BentoML service that serves OpenFold3 structure predictions from JSON input."""

    def __init__(self) -> None:
        # Must precede every torch/ROCm touch below: the runtime reads the
        # visibility env vars once, at initialisation.
        pin_worker_device(bentoml.server_context.worker_index or 1, CONFIG.accelerator_count)

        # Eagerly initialise the CUDA/ROCm context at service startup.  Without
        # this, CUDA init is deferred to the first model.to('cuda') inside
        # Lightning's trainer.predict(), pushing ~24s of cold-start cost onto
        # the first /predict request instead of absorbing it during service
        # init.
        # Lazy import: must run before any other torch/CUDA touch.
        # Order and unconditional tf32 match OF3's `predict` entry point
        # (use_tf32 defaults to True).
        from openfold3.entry_points.import_utils import (
            _configure_torch_backend,
            _enable_tf32,
        )

        _configure_torch_backend()
        _enable_tf32()

        # Fail at startup rather than serve with the only bound on MSA-server
        # work silently gone: without the patch, a throttled server keeps this
        # worker busy indefinitely and no timeout can reclaim it.
        if not msa_deadline_hook_available():
            raise RuntimeError(
                "OpenFold3's ColabFold client is missing the deadline hook installed by "
                "patches/of3_msa_server_deadline.patch. Rebuild the image with the patch "
                "applied; see the Dockerfile's patch step."
            )

        self._cache = CONFIG.cache
        self._cache.mkdir(parents=True, exist_ok=True)
        _ensure_model_parameters(self._cache)
        # Cache the LightningModule once at startup; per-request runners reuse
        # it via __dict__ injection in run_openfold3_prediction().
        self._model = _load_model(self._cache)

    @staticmethod
    def _unavailable(ctx: bentoml.Context, message: str, retry_after_s: int = 60) -> dict[str, Any]:
        """Answer 503 while keeping the usual error body.

        Raising ``ServiceUnavailable`` would also give 503, but BentoML replaces
        the body of any 5xx with a generic "an unexpected error has occurred",
        dropping the one detail the caller needs. Setting the status through the
        request context keeps our own payload.
        """
        ctx.response.status_code = 503
        ctx.response.headers["Retry-After"] = str(retry_after_s)
        return {
            "error": True,
            "message": message,
            "structures": [],
            "confidence": {},
            "timing": {},
        }

    @bentoml.api
    def predict(self, data: OpenFold3Request, ctx: bentoml.Context) -> dict[str, Any]:
        """
        Run OpenFold3 prediction given a JSON payload.

        Accepts a structured request describing biomolecular chains
        (protein, DNA, RNA, ligands) and returns structure predictions
        with confidence scores.

        Parameters
        ----------
        data : OpenFold3Request
            The input request with queries, seeds, and prediction settings.
        ctx : bentoml.Context
            Request context, used to set the response status and to read the
            arrival time stamped by ``ArrivalStampMiddleware``.

        Returns
        -------
        dict
            On success:
                - "structures": list of {record_id, format, content}
                - "confidence": dict of confidence metrics per structure
                - "timing": dict of {"runtime_s": ...} per structure (keyed by
                  sample_id like "confidence"); {} when OF3 emits no timing
                - "atom_confidence": per-atom confidences per structure,
                  only when include_atom_confidences=True
                - "error": False

            On failure:
                - "error": True
                - "message": error description
                - "structures": []
                - "confidence": {}
                - "timing": {}

        Mutually-exclusive inputs (e.g. inline MSAs with use_msa_server=True) are
        rejected during request validation with HTTP 400 before reaching here.

        Prediction failures answer HTTP 200 with the error body above. The two
        transient conditions answer HTTP 503 with the same body plus Retry-After:
        the MSA server not finishing within its budget, and a request that queued
        so long it can no longer finish in what remains of the request timeout.
        """
        now = time.monotonic()
        budget = build_budget(
            arrived_at=ctx.request.scope.get(ARRIVAL_SCOPE_KEY, now),
            now=now,
            request_timeout_s=CONFIG.request_timeout_seconds,
            msa_timeout_s=CONFIG.msa_timeout_seconds,
            inference_reserve_s=CONFIG.inference_reserve_seconds,
        )

        try:
            ensure_startable(budget, needs_msa=data.use_msa_server, now=now)
            result = run_openfold3_prediction(
                body=data.model_dump(),
                cache=self._cache,
                model=self._model,
                num_diffusion_samples=data.num_diffusion_samples,
                num_model_seeds=data.num_model_seeds,
                seeds=data.seeds,
                use_msa_server=data.use_msa_server,
                use_templates=data.use_templates,
                output_format=data.output_format,
                num_workers=CONFIG.num_workers,
                accelerator=CONFIG.accelerator,
                include_atom_confidences=data.include_atom_confidences,
                msa_deadline=budget.msa_deadline,
            )
            return result
        except DeadlineExceeded as e:
            return self._unavailable(ctx, str(e))
        except ValueError as e:
            # Checked here too: is_msa_server_timeout walks the cause chain
            # because the failure surfaces wrapped, and nothing guarantees the
            # wrapper is not a ValueError.
            if is_msa_server_timeout(e):
                return self._unavailable(ctx, str(e))
            return {
                "error": True,
                "message": str(e),
                "structures": [],
                "confidence": {},
                "timing": {},
            }
        except Exception as e:
            if is_msa_server_timeout(e):
                return self._unavailable(ctx, str(e))
            return {
                "error": True,
                "message": f"Prediction failed: {e!s}",
                "structures": [],
                "confidence": {},
                "timing": {},
            }


OpenFold3Prediction.add_asgi_middleware(ArrivalStampMiddleware)  # type: ignore[attr-defined]
