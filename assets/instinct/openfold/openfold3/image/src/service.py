# Copyright © Advanced Micro Devices, Inc., or its affiliates.
#
# SPDX-License-Identifier: MIT

"""
BentoML service for OpenFold3 structure predictions.

Accepts a JSON payload (OpenFold3 query format: queries with chains)
and returns structure predictions (coordinates, confidence, mmCIF/PDB string).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import bentoml
from pydantic import BaseModel, Field, field_validator, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict
from runner import (
    _ensure_model_parameters,
    _load_model,
    find_request_conflicts,
    run_openfold3_prediction,
)


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
    ``accelerator_count`` reuses ``AIM_ACCELERATOR_COUNT`` — the canonical
    aim-runtime env var that the operator already has to set so profile
    selection matches ``metadata.gpu_count`` (see
    ``aim_runtime.config._read_accelerator_count``).  Reusing it avoids an
    OF3-only duplicate env var that would need to be kept in lockstep with
    the operator-supplied one.

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
        default=Path("~/.openfold3").expanduser(),
        validation_alias="OPENFOLD3_CACHE",
    )

    @field_validator("cache", mode="before")
    @classmethod
    def _expand_cache(cls, v: Any) -> Path:
        return Path(v).expanduser().resolve()


CONFIG = ServiceConfig()


@bentoml.service(
    resources={"gpu": CONFIG.accelerator_count},
    traffic={"timeout": 600},
)
class OpenFold3Prediction:
    """BentoML service that serves OpenFold3 structure predictions from JSON input."""

    def __init__(self) -> None:
        # Eagerly initialise the CUDA/ROCm context at service startup.  Without
        # this, CUDA init is deferred to the first model.to('cuda') inside
        # Lightning's trainer.predict(), pushing ~24s of cold-start cost onto
        # the first /predict request instead of absorbing it during service
        # init.
        # Lazy import: must run before any other torch/CUDA touch.
        from openfold3.entry_points.import_utils import _torch_gpu_setup

        _torch_gpu_setup()

        self._cache = CONFIG.cache
        self._cache.mkdir(parents=True, exist_ok=True)
        _ensure_model_parameters(self._cache)
        # Cache the LightningModule once at startup; per-request runners reuse
        # it via __dict__ injection in run_openfold3_prediction().
        self._model = _load_model(self._cache)

    @bentoml.api
    def predict(self, data: OpenFold3Request) -> dict[str, Any]:
        """
        Run OpenFold3 prediction given a JSON payload.

        Accepts a structured request describing biomolecular chains
        (protein, DNA, RNA, ligands) and returns structure predictions
        with confidence scores.

        Parameters
        ----------
        data : OpenFold3Request
            The input request with queries, seeds, and prediction settings.

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
        """
        try:
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
            )
            return result
        except ValueError as e:
            return {
                "error": True,
                "message": str(e),
                "structures": [],
                "confidence": {},
                "timing": {},
            }
        except Exception as e:
            return {
                "error": True,
                "message": f"Prediction failed: {e!s}",
                "structures": [],
                "confidence": {},
                "timing": {},
            }
