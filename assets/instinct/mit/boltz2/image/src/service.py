# Copyright © Advanced Micro Devices, Inc., or its affiliates.
#
# SPDX-License-Identifier: MIT

"""
BentoML service for Boltz-2 biomolecular structure prediction.

Routes
-----------------
``GET  /health``        Mounted ASGI app; forwards to BentoML's ``/healthz``.
``GET  /v1/models``     Mounted ASGI app; OpenAI-shaped list with the served model.
``POST /v1/inference``  Predicts a structure. Returns JSON with the structure,
                        the confidence summary (pLDDT / pTM / ipTM) and a
                        ``runtime`` block.
``GET  /healthz``       BentoML's built-in readiness probe (not authored here).
"""

from __future__ import annotations

import json
import logging
import re
import time
from http import HTTPStatus
from pathlib import Path
from typing import Any, Optional

import bentoml
import httpx
from bentoml.exceptions import BentoMLException, InternalServerError, InvalidArgument
from compilation import compile_status
from pydantic import BaseModel, Field, ValidationError, field_validator, model_validator
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
from resident_strategy import resolve_device
from response_models import ErrorResponse, HealthResponse, InferenceResponse, ModelList, RuntimeInfo, dump
from rocm_triangle import seam_status, warm_up_large
from runner import (
    DEFAULT_MSA_SERVER_URL,
    MODE_STRUCTURE,
    MODES,
    assets_present,
    cache_is_writable,
    ensure_boltz2_assets,
    is_msa_server_timeout,
    load_affinity_model,
    load_boltz2_model,
    needs_affinity_model,
    reject_file_references,
    run_boltz2_prediction,
    seeded_prediction,
)
from starlette.applications import Starlette
from starlette.requests import Request
from starlette.responses import JSONResponse
from starlette.routing import Route
from warmup import (
    AFFINITY_COMPILE_WARMUP_SAMPLING_STEPS,
    AFFINITY_CROP_WARMUP_SMILES,
    AFFINITY_CROP_WARMUP_TOKENS,
    STATE_DISABLED,
    STATE_PENDING,
    WARMUP_SAMPLING_STEPS,
    WarmupReport,
    affinity_warmup_request,
    run_warmup,
    warmup_requests,
)

from aim_runtime.config import DEFAULT_CACHE_PATH

logger = logging.getLogger(__name__)


# Request limits. A request above one is refused with HTTP 422 before any compute runs.
# Above this many tokens the token transformer compiles once during the request.
MAX_TOKENS = 2364
# 5 or more samples compile once during the request.
MAX_DIFFUSION_SAMPLES = 4
# Upstream default is 3; the limit keeps one request to a small multiple of the default run time.
MAX_RECYCLING_STEPS = 10
# Upstream default is 200.
MAX_SAMPLING_STEPS = 500
# Upstream default; samples beyond one parallel batch run in further batches.
MAX_PARALLEL_SAMPLES = 5
# Upstream default of ``--diffusion_samples_affinity``.
MAX_DIFFUSION_SAMPLES_AFFINITY = 5
# Boltz refuses a larger ligand for affinity; its training maximum is 56.
MAX_AFFINITY_LIGAND_ATOMS = 128

# Values Boltz accepts for ``method`` (the keys of ``boltz.data.const.method_types_ids`` at the
# pinned BOLTZ_COMMIT, lower case; Boltz lower-cases the request value). Any other value makes
# the featurizer fail on every retry, so it is refused here. A test compares this tuple with Boltz.
BOLTZ_METHODS = (
    "md",
    "x-ray diffraction",
    "electron microscopy",
    "solution nmr",
    "solid-state nmr",
    "neutron diffraction",
    "electron crystallography",
    "fiber diffraction",
    "powder diffraction",
    "infrared spectroscopy",
    "fluorescence transfer",
    "epr",
    "theoretical model",
    "solution scattering",
    "other",
    "afdb",
    "boltz-1",
    "future1",
    "future2",
    "future3",
    "future4",
    "future5",
)
# Entity types that the Boltz input parser accepts.
BOLTZ_ENTITY_TYPES = ("protein", "dna", "rna", "ligand")

# An atom of a SMILES string: a bracket atom, or an atom of the organic subset (aromatic ones are
# lower case; "Br" and "Cl" come first so that they are not read as "B" and "C").
_SMILES_ATOM = re.compile(r"\[([^\]]*)\]|Br|Cl|[BCNOPSFIbcnops*]")
_BRACKET_ELEMENT = re.compile(r"\d*([A-Z][a-z]?|[a-z]{1,2}|\*)")


class UnprocessableRequest(BentoMLException):
    """The request is well formed but above a service limit (HTTP 422)."""

    error_code = HTTPStatus.UNPROCESSABLE_ENTITY


def validation_summary(exc: ValidationError) -> str:
    """One short line per invalid field: its location and the message, without URL or input."""
    parts = []
    for error in exc.errors(include_url=False, include_context=False, include_input=False):
        location = ".".join(str(part) for part in error["loc"]) or "request"
        message = error["msg"].removeprefix("Value error, ")
        parts.append(f"{location}: {message}")
    return "Invalid request. " + "; ".join(parts)


INVALID_REQUEST_PREFIX = "Invalid request. "


def normalize_error_body(status: int, body: bytes) -> bytes | None:
    """The error body of a 4xx or 5xx answer in the one shape ``{"error": "<text>"}``, or ``None`` to keep it.

    BentoML answers a failure of its own request parsing with the pydantic detail (error URLs
    and, for some errors, the input), and an exception of the service with a list that holds the
    object. This keeps one object with one text. A pydantic detail becomes one entry for each
    field (location and message). The text of a 400 or 422 answer starts with "Invalid request. ".
    """
    try:
        document = json.loads(body)
    except ValueError:
        return None
    if isinstance(document, list) and len(document) == 1:
        document = document[0]
    if not isinstance(document, dict):
        return None
    detail = document.get("detail")
    if isinstance(detail, list):
        parts = []
        for error in detail:
            if not isinstance(error, dict):
                return None
            location = ".".join(str(part) for part in error.get("loc", ())) or "request"
            parts.append(f"{location}: {str(error.get('msg', 'invalid')).removeprefix('Value error, ')}")
        logger.warning("Invalid request body: %s", document)
        text = "; ".join(parts)
    elif isinstance(document.get("error"), str):
        text = document["error"]
    elif isinstance(document.get("message"), str):
        text = document["message"]
    else:
        return None
    if status in (400, 422) and not text.startswith(INVALID_REQUEST_PREFIX):
        text = INVALID_REQUEST_PREFIX + text
    return json.dumps({"error": text}).encode()


def smiles_heavy_atoms(smiles: str) -> int:
    """Number of heavy atoms of a SMILES string, which is the number of tokens Boltz gives the ligand.

    Boltz reads the SMILES with RDKit, adds the hydrogens, removes them again, and makes one
    token of each remaining atom. RDKit gives the same count here. Where RDKit is not
    installed, or does not read the string, a tokenizer of the SMILES atoms counts them:
    bracket atoms, organic-subset atoms and aromatic atoms (lower case), without hydrogens.
    """
    try:
        from rdkit import Chem
    except ImportError:
        molecule = None
    else:
        molecule = Chem.MolFromSmiles(smiles)
    if molecule is not None:
        return sum(1 for atom in molecule.GetAtoms() if atom.GetAtomicNum() != 1)
    count = 0
    for match in _SMILES_ATOM.finditer(smiles):
        inner = match.group(1)
        if inner is not None:
            element = _BRACKET_ELEMENT.match(inner)
            if element is None or element.group(1) == "H":
                continue
        count += 1
    return count


def largest_fragment_atoms(smiles: str) -> int:
    """Heavy atoms of the largest fragment of a SMILES string; Boltz keeps only that fragment for affinity."""
    try:
        from rdkit import Chem
    except ImportError:
        molecule = None
    else:
        molecule = Chem.MolFromSmiles(smiles)
    if molecule is None:
        return max(smiles_heavy_atoms(part) for part in smiles.split("."))
    fragments = Chem.GetMolFrags(molecule, asMols=True)
    return max((sum(1 for atom in f.GetAtoms() if atom.GetAtomicNum() != 1) for f in fragments), default=0)


def estimate_tokens(sequences: list[dict[str, Any]]) -> int:
    """Count the tokens of a request: one per polymer residue, one per SMILES heavy atom.

    Chains named by a list of ids count once per id. Ligands given as CCD codes
    are not counted, so the result can be lower than Boltz's own count.
    """
    total = 0
    for entry in sequences:
        for kind, chain in entry.items():
            if not isinstance(chain, dict):
                continue
            ids = chain.get("id")
            copies = len(ids) if isinstance(ids, list) else 1
            if kind == "ligand":
                size = smiles_heavy_atoms(str(chain.get("smiles", "")))
            else:
                size = len(str(chain.get("sequence", "")))
            total += size * copies
    return total


def _affinity_entries(properties: Optional[list[dict[str, Any]]]) -> list[dict[str, Any]]:
    """The properties with an ``affinity`` key in any case.

    Boltz lower-cases the key to detect the property, but reads its value with the exact key
    ``affinity``.
    """
    return [e for e in properties or [] if isinstance(e, dict) and any(str(k).lower() == "affinity" for k in e)]


def affinity_binder(properties: Optional[list[dict[str, Any]]]) -> Optional[Any]:
    """The binder of the ``affinity`` property of a request, or ``None`` when it has none."""
    for entry in _affinity_entries(properties):
        affinity = entry.get("affinity")
        return affinity.get("binder") if isinstance(affinity, dict) else None
    return None


def has_affinity_property(properties: Optional[list[dict[str, Any]]]) -> bool:
    return bool(_affinity_entries(properties))


def _ligand_content(chain: dict[str, Any]) -> str:
    """What Boltz groups ligand entries by: the SMILES string or the CCD code."""
    return str(chain["smiles"] if "smiles" in chain else chain.get("ccd"))


def check_affinity_request(data: "Boltz2Request") -> None:
    """Raise ``InvalidArgument`` or ``UnprocessableRequest`` for an affinity property Boltz cannot serve.

    Boltz allows one binder: a ligand with one chain id and one residue, and no
    other entry with the same SMILES or CCD code. The checks that need no RDKit read
    the request only; the atom count uses the SMILES.
    """
    if any("affinity" not in e for e in _affinity_entries(data.properties)):
        raise InvalidArgument("the property key must be written 'affinity' in lower case")
    binder = affinity_binder(data.properties)
    if not isinstance(binder, str):
        raise InvalidArgument("properties.affinity.binder must be the id of one ligand chain")
    if len(_affinity_entries(data.properties)) > 1:
        raise InvalidArgument("only one affinity property is supported")
    ligands = [e["ligand"] for e in data.sequences if isinstance(e.get("ligand"), dict)]
    chosen = [c for c in ligands if c.get("id") == binder or (isinstance(c.get("id"), list) and binder in c["id"])]
    if not chosen:
        raise InvalidArgument("properties.affinity.binder must be the id of a ligand in sequences")
    ligand = chosen[0]
    copies = [c for c in ligands if _ligand_content(c) == _ligand_content(ligand)]
    if sum(len(c["id"]) if isinstance(c.get("id"), list) else 1 for c in copies) > 1:
        raise InvalidArgument("the affinity binder must be a ligand with one copy")
    if isinstance(ligand.get("ccd"), list) and len(ligand["ccd"]) > 1:
        raise InvalidArgument("the affinity binder must be a ligand of one residue")
    if "smiles" in ligand:
        atoms = largest_fragment_atoms(str(ligand["smiles"]))
        if atoms == 0:
            raise InvalidArgument("the affinity binder must have a SMILES with at least one heavy atom")
        if atoms > MAX_AFFINITY_LIGAND_ATOMS:
            raise UnprocessableRequest(f"affinity ligand atoms is {atoms}; the limit is {MAX_AFFINITY_LIGAND_ATOMS}")


def check_limits(data: "Boltz2Request") -> None:
    """Raise ``UnprocessableRequest`` when ``data`` is above a service limit."""
    limits = (
        ("sampling_steps_affinity", data.sampling_steps_affinity, MAX_SAMPLING_STEPS),
        ("diffusion_samples_affinity", data.diffusion_samples_affinity, MAX_DIFFUSION_SAMPLES_AFFINITY),
        ("recycling_steps", data.recycling_steps, MAX_RECYCLING_STEPS),
        ("sampling_steps", data.sampling_steps, MAX_SAMPLING_STEPS),
        ("diffusion_samples", data.diffusion_samples, MAX_DIFFUSION_SAMPLES),
        ("max_parallel_samples", data.max_parallel_samples, MAX_PARALLEL_SAMPLES),
        ("tokens", estimate_tokens(data.sequences), MAX_TOKENS),
    )
    for name, value, maximum in limits:
        if value > maximum:
            raise UnprocessableRequest(f"{name} is {value}; the limit is {maximum}")


class Boltz2Request(BaseModel):
    """One structure-prediction request.

    ``sequences`` / ``constraints`` / ``templates`` / ``properties`` are Boltz's
    own YAML schema (``boltz.data.parse.schema.parse_boltz_schema``), passed
    through unchanged so the AIM tracks upstream's input format rather than
    inventing a parallel one::

        {"data": {"sequences": [{"protein": {"id": "A", "sequence": "NLYIQ..."}}]}}
    """

    version: int = 1
    sequences: list[dict[str, Any]]
    constraints: Optional[list[dict[str, Any]]] = None
    templates: Optional[list[dict[str, Any]]] = None
    properties: Optional[list[dict[str, Any]]] = None

    name: str = "prediction"
    output_format: str = "mmcif"

    recycling_steps: int = Field(default=3, ge=0)
    sampling_steps: int = Field(default=200, ge=1)
    diffusion_samples: int = Field(default=1, ge=1)
    max_parallel_samples: int = Field(default=5, ge=1)
    step_scale: Optional[float] = None
    method: Optional[str] = None

    use_msa_server: bool = False
    msa_pairing_strategy: str = "greedy"

    # Affinity pass; the defaults are those of ``boltz predict``.
    affinity_mw_correction: bool = False
    sampling_steps_affinity: int = Field(default=200, ge=1)
    diffusion_samples_affinity: int = Field(default=5, ge=1)

    # Seeds all random number generators and uses deterministic algorithms for this request.
    seed: Optional[int] = Field(default=None, ge=0, lt=2**32)

    @field_validator("output_format")
    @classmethod
    def _known_output_format(cls, v: str) -> str:
        if v not in ("pdb", "mmcif"):
            raise ValueError("output_format must be 'pdb' or 'mmcif'")
        return v

    @field_validator("method")
    @classmethod
    def _known_method(cls, v: Optional[str]) -> Optional[str]:
        if v is not None and v.lower() not in BOLTZ_METHODS:
            raise ValueError(f"method must be one of {list(BOLTZ_METHODS)}")
        return v

    @field_validator("version")
    @classmethod
    def _known_version(cls, v: int) -> int:
        if v != 1:
            raise ValueError("version must be 1")
        return v

    @field_validator("msa_pairing_strategy")
    @classmethod
    def _known_pairing_strategy(cls, v: str) -> str:
        if v not in ("greedy", "complete"):
            raise ValueError("msa_pairing_strategy must be 'greedy' or 'complete'")
        return v

    @model_validator(mode="after")
    def _non_empty_sequences(self) -> "Boltz2Request":
        """Reject an empty payload up front.

        Boltz would otherwise process it into an empty manifest and only fail
        after the featurization pass — an HTTP 400 here costs nothing and says
        exactly what is wrong.
        """
        if not self.sequences:
            raise ValueError("sequences must contain at least one entity (protein, dna, rna or ligand)")
        for entry in self.sequences:
            if not isinstance(entry, dict) or len(entry) != 1:
                raise ValueError(
                    "each sequences entry must be a single-key mapping such as "
                    '{"protein": {"id": "A", "sequence": "..."}}'
                )
        for entry in self.sequences:
            kind = next(iter(entry))
            if kind not in BOLTZ_ENTITY_TYPES:
                raise ValueError(f"sequences entry type must be one of {list(BOLTZ_ENTITY_TYPES)}")
            chain = entry[kind]
            if kind == "ligand" and isinstance(chain, dict) and not isinstance(chain.get("smiles", ""), str):
                raise ValueError("ligand smiles must be a string")
        reject_file_references(self.sequences, self.templates)
        return self

    @model_validator(mode="wrap")
    @classmethod
    def _short_error_message(cls, value: Any, handler: Any) -> "Boltz2Request":
        """Answer an invalid request with one short line per field.

        The pydantic detail (error URLs, the input that was sent) goes to the
        server log only. ``InvalidArgument`` gives HTTP 400.
        """
        try:
            return handler(value)
        except ValidationError as exc:
            logger.warning("Invalid request: %s", exc)
            raise InvalidArgument(validation_summary(exc)) from None


class ServiceConfig(BaseSettings):
    """Service-level configuration injected via the active profile's ``env_vars``.

    ``aim-runtime serve`` exports each entry of the selected profile's
    ``env_vars`` into the ``bentoml serve`` subprocess, so switching
    ``AIM_PROFILE_ID`` reconfigures the service without touching this file. Each
    field names its variable explicitly via ``validation_alias``.

    ``cache`` reads the canonical ``AIM_CACHE_PATH`` so Boltz's checkpoint lands
    where every other AIM caches weights. Nothing sets that variable in the image
    and aim-runtime resolves it in-memory without re-exporting, so the subprocess
    usually sees it unset — hence the default.
    """

    model_config = SettingsConfigDict()

    accelerator_count: int = Field(default=1, validation_alias="AIM_ACCELERATOR_COUNT")
    accelerator: str = Field(default="gpu", validation_alias="BOLTZ2_ACCELERATOR")
    # Hardcoded to 0 and deliberately NOT a Tuner knob. Two independent reasons,
    # which are separate arguments:
    #
    # HAZARD — /v1/inference below is a SYNC `def` endpoint.
    # BentoML 1.4.35 dispatches those off the event loop via
    # `anyio.to_thread.run_sync` (_bentoml_impl/server/app.py:826, :655-666),
    # so this code runs on an "AnyIO worker thread", not the main thread —
    # confirmed by printing thread identity from a live server. torch's
    # DataLoader spawns workers with the default multiprocessing context (boltz
    # passes none), which on Linux + CPython 3.12 is `fork`; forking from a
    # non-main thread of a multi-threaded process is the documented deadlock
    # shape, and CPython 3.12 warns about exactly it. There is no recovery:
    # traffic.timeout cancels an asyncio task, which cannot kill an OS thread —
    # the client gets a 504 while the handler keeps running — and the freed
    # capacity token lets the next request barge in (bentoml/BentoML#5642), so
    # wedged work accumulates instead of blocking. On a shared node that is a
    # GPU held by a process that will not die.
    #
    # VALUE — boltz's Boltz2InferenceDataModule.predict_dataloader builds a
    # batch_size=1 DataLoader over a dataset whose length is the manifest record
    # count, and runner.py submits exactly one input per request. One item means
    # extra workers have nothing to overlap; every value >= 1 pays fork + IPC to
    # fetch a single batch. torch only avoids the fork at exactly 0
    # (_SingleProcessDataLoaderIter), so 1 is not a safer middle ground than 4.
    #
    # Kept as a Field rather than a literal so a deployment can still override
    # it deliberately, and so the recipe-parity tier can assert the deviation
    # from upstream's num_workers=2.
    num_workers: int = Field(default=0, validation_alias="BOLTZ2_NUM_WORKERS", ge=0)
    cache: Path = Field(default=Path(DEFAULT_CACHE_PATH), validation_alias="AIM_CACHE_PATH")

    # BOLTZ2_OPTIMIZATIONS is the one performance setting. "on" (default) turns on:
    #   - the ROCm Triton triangle kernels (src/rocm_triangle: attention plus the
    #     Inductor-compiled triangle multiply), entered through boltz's `use_kernels` seam;
    #   - TorchInductor on the Transition and token-transformer modules;
    #   - a start-up warm-up prediction (warmup.py) that does the compiles before
    #     the service accepts requests, so no user request waits for them.
    # "off" runs upstream Boltz unchanged: no Triton seams, no compile, no warm-up.
    # The default lives here and not in the profile: a profile `env_vars` entry
    # overrides `docker run -e` without a warning.
    # cuEquivariance (`boltz[cuda]`) has no ROCm build, so it is always off.
    optimizations: str = Field(default="on", validation_alias="BOLTZ2_OPTIMIZATIONS")

    @field_validator("optimizations", mode="before")
    @classmethod
    def _check_optimizations(cls, v: Any) -> str:
        normalized = str(v if v is not None else "on").strip().lower()
        if normalized not in ("on", "off"):
            raise ValueError(f"BOLTZ2_OPTIMIZATIONS must be 'on' or 'off', got {v!r}")
        return normalized

    @property
    def optimized(self) -> bool:
        return self.optimizations == "on"

    # BOLTZ2_AFFINITY_COMPILE: "on" also compiles the affinity model with TorchInductor. It is off by
    # default and not in the profile, so `docker run -e` can set it. It acts only with
    # BOLTZ2_MODE=affinity and BOLTZ2_OPTIMIZATIONS=on (see `affinity_compiled`).
    affinity_compile: str = Field(default="off", validation_alias="BOLTZ2_AFFINITY_COMPILE")

    @field_validator("affinity_compile", mode="before")
    @classmethod
    def _check_affinity_compile(cls, v: Any) -> str:
        normalized = str(v if v is not None else "off").strip().lower()
        if normalized not in ("on", "off"):
            raise ValueError(f"BOLTZ2_AFFINITY_COMPILE must be 'on' or 'off', got {v!r}")
        return normalized

    # What the service serves, and so which checkpoints it loads:
    #   structure - structure prediction; a request for affinity is refused;
    #   affinity  - structure prediction, and affinity for requests that ask for it.
    mode: str = Field(default=MODE_STRUCTURE, validation_alias="BOLTZ2_MODE")

    @field_validator("mode", mode="before")
    @classmethod
    def _check_mode(cls, v: Any) -> str:
        normalized = str(v if v is not None else MODE_STRUCTURE).strip().lower()
        if normalized not in MODES:
            raise ValueError(f"BOLTZ2_MODE must be one of {list(MODES)}, got {v!r}")
        return normalized

    @property
    def affinity_compiled(self) -> bool:
        return self.affinity_compile == "on" and self.optimized and needs_affinity_model(self.mode)

    # Server that answers `use_msa_server` requests. With it on, the protein
    # sequences of the request are sent to this host; the default is the public
    # ColabFold API. Point it at a private server to keep sequences on-premises,
    # or set it to an empty string to refuse `use_msa_server` requests.
    msa_server_url: str = Field(default=DEFAULT_MSA_SERVER_URL, validation_alias="BOLTZ2_MSA_SERVER_URL")

    request_timeout_seconds: int = Field(
        default=DEFAULT_REQUEST_TIMEOUT_SECONDS, validation_alias="BOLTZ2_REQUEST_TIMEOUT_SECONDS", gt=0
    )
    # Cap on the MSA-server work of one request.
    msa_timeout_seconds: int = Field(
        default=DEFAULT_MSA_TIMEOUT_SECONDS, validation_alias="BOLTZ2_MSA_TIMEOUT_SECONDS", gt=0
    )
    # Budget held back for the prediction, so a slow MSA server cannot use the whole request.
    inference_reserve_seconds: int = Field(
        default=DEFAULT_INFERENCE_RESERVE_SECONDS, validation_alias="BOLTZ2_INFERENCE_RESERVE_SECONDS", ge=0
    )
    # 0 means "derive from accelerator_count"; see effective_max_concurrency.
    max_concurrency: int = Field(default=0, validation_alias="BOLTZ2_MAX_CONCURRENCY", ge=0)

    # Downloads about 6.0 GB (structure mode) or 8.0 GB (affinity mode) on a cold cache. Off means a cold
    # cache is a startup error instead of a long silent stall — which is what a
    # deployment with pre-staged weights wants.
    download_assets: bool = Field(default=True, validation_alias="BOLTZ2_DOWNLOAD_ASSETS")

    @field_validator("cache", mode="before")
    @classmethod
    def _expand_cache(cls, v: Any) -> Path:
        return Path(v).expanduser().resolve()

    @model_validator(mode="after")
    def _reject_unstartable_reserve(self) -> "ServiceConfig":
        """A reserve at or past the usable budget would refuse every request on arrival."""
        usable_s = self.request_timeout_seconds - response_slack(self.request_timeout_seconds)
        if self.inference_reserve_seconds >= usable_s:
            raise ValueError(
                f"BOLTZ2_INFERENCE_RESERVE_SECONDS ({self.inference_reserve_seconds}s) leaves no startable "
                f"budget: BOLTZ2_REQUEST_TIMEOUT_SECONDS ({self.request_timeout_seconds}s) gives "
                f"{usable_s:.0f}s usable after response slack, and the reserve must be smaller than that."
            )
        return self

    @property
    def effective_max_concurrency(self) -> int:
        """In-flight requests the service accepts before answering 429."""
        return self.max_concurrency or max_concurrency_for(self.accelerator_count)


CONFIG = ServiceConfig()


def require_staged_assets(config: ServiceConfig) -> None:
    """Raise when weights are missing and cannot be had: downloads are off, or the cache is not writable.

    This runs when the module is imported, in the process of ``bentoml serve``, so
    the start stops once with this error. The same check in ``Boltz2Prediction.__init__``
    would run in a BentoML worker, and BentoML starts a failed worker again and again.
    The writability check runs before any download starts.
    """
    if assets_present(config.cache, config.mode):
        return
    if not config.download_assets:
        raise RuntimeError(
            f"BOLTZ2_DOWNLOAD_ASSETS is off and {config.cache} is missing the Boltz-2 "
            f"checkpoint or CCD data that BOLTZ2_MODE={config.mode} needs. Pre-stage them, or set BOLTZ2_DOWNLOAD_ASSETS=true."
        )
    if not cache_is_writable(config.cache):
        raise RuntimeError(
            f"{config.cache} is not writable and is missing the Boltz-2 checkpoint or CCD data that "
            f"BOLTZ2_MODE={config.mode} needs, so they cannot be downloaded. Mount the cache directory writable, "
            "or stage the weights in it."
        )


def require_accelerator(config: ServiceConfig) -> None:
    """Raise when ``BOLTZ2_ACCELERATOR`` asks for a GPU and torch finds none.

    ``auto`` and ``cpu`` never raise. The check runs at import, like ``require_staged_assets``.
    """
    if config.accelerator not in ("auto", "cpu") and resolve_device(config.accelerator) is None:
        raise RuntimeError(
            f"BOLTZ2_ACCELERATOR={config.accelerator} needs a GPU, but torch.cuda.is_available() is false. "
            "Pass the GPU devices to the container (/dev/kfd and /dev/dri)."
        )


require_staged_assets(CONFIG)
require_accelerator(CONFIG)

# Scope key carrying the arrival time of a request. ``inference`` reads it back to
# measure its budget from arrival and not from when a worker thread picked it up.
ARRIVAL_SCOPE_KEY = "aim_arrived_at"


class ArrivalStampMiddleware:
    """Record when each request arrived, before it waits for a worker thread.

    ``traffic.timeout`` counts from arrival, but ``inference`` runs only when the
    worker thread is free. Without the stamp, a request that waited most of the
    timeout would get a full budget and would reach the timeout anyway.
    """

    def __init__(self, app: Any) -> None:
        self.app = app

    async def __call__(self, scope: Any, receive: Any, send: Any) -> Any:
        if scope["type"] == "http":
            scope[ARRIVAL_SCOPE_KEY] = time.monotonic()
        return await self.app(scope, receive, send)


class ErrorBodyMiddleware:
    """Give every 4xx and 5xx answer of the service the body ``{"error": "<text>"}``.

    A body that is not valid JSON or has no ``data`` object fails in BentoML before
    ``Boltz2Request`` is built, so its validator cannot shorten that message. The 429 and 504
    answers come from BentoML's own layers, outside this middleware, and already have this shape.
    """

    def __init__(self, app: Any) -> None:
        self.app = app

    async def __call__(self, scope: Any, receive: Any, send: Any) -> Any:
        if scope["type"] != "http":
            return await self.app(scope, receive, send)

        start: dict[str, Any] | None = None
        chunks: list[bytes] = []

        async def _send(message: Any) -> None:
            nonlocal start
            if start is None and message["type"] == "http.response.start" and message["status"] >= 400:
                start = message
                return
            if start is None:
                await send(message)
                return
            chunks.append(message.get("body", b""))
            if message.get("more_body", False):
                return
            body = b"".join(chunks)
            normalized = normalize_error_body(start["status"], body)
            body = body if normalized is None else normalized
            headers = [(k, v) for k, v in start["headers"] if k.lower() != b"content-length"]
            headers.append((b"content-length", str(len(body)).encode()))
            await send({**start, "headers": headers})
            await send({"type": "http.response.body", "body": body})

        return await self.app(scope, receive, _send)


class BentoArgs(BaseModel):
    """Template arguments AIM forwards via ``bentoml serve --arg``.

    ``aim-runtime`` resolves ``AIM_PORT`` and injects it as ``--arg port=…`` (an
    unconditional system override in ``CommandGenerator``), so this is the only
    source of the bound port. Deliberately no default: a bare ``bentoml serve``
    is not a supported entry point, and a second default here could silently
    drift from ``AIM_PORT``.
    """

    port: int


args = bentoml.use_arguments(BentoArgs)

MODEL_ID = "mit/boltz2"
_HEALTH_TIMEOUT_S = 5.0


async def health(_request: Request) -> JSONResponse:
    """``GET /health``: forward to BentoML's ``/healthz`` on this service's own port.

    Returns the upstream status code, or 503 if the upstream cannot be reached.
    """
    try:
        async with httpx.AsyncClient(timeout=_HEALTH_TIMEOUT_S) as client:
            upstream = await client.get(f"http://127.0.0.1:{args.port}/healthz")
    except httpx.HTTPError:
        return JSONResponse(dump(HealthResponse, {"status": "unavailable"}), status_code=503)
    return JSONResponse(
        dump(HealthResponse, {"status": "ok" if upstream.is_success else "unavailable"}),
        status_code=upstream.status_code,
    )


async def list_models(_request: Request) -> JSONResponse:
    """``GET /v1/models``: OpenAI-shaped list with the one served model."""
    return JSONResponse(dump(ModelList, {"object": "list", "data": [{"id": MODEL_ID, "object": "model"}]}))


canonical_app = Starlette(
    routes=[
        Route("/health", health, methods=["GET"]),
        Route("/v1/models", list_models, methods=["GET"]),
    ]
)


@bentoml.service(
    name="boltz2-prediction",
    resources={"gpu": CONFIG.accelerator_count},
    # One worker per accelerator: Boltz-2 has no tensor parallelism, and a worker
    # serves one prediction at a time (one thread), so the settings that a request
    # changes on the model never change during another prediction.
    workers=CONFIG.accelerator_count,
    # timeout is a backstop only: inference() carries a deadline inside it and
    # returns on its own. If the timeout fired first, the client would get a 504
    # while the worker thread kept running (a sync endpoint runs through
    # anyio.to_thread, and a thread cannot be killed).
    #
    # max_concurrency bounds the queue. BentoML divides it by the worker count, so
    # the default is three in flight per worker: one running and two waiting.
    # Past that, a client gets an immediate 429.
    traffic={
        "timeout": CONFIG.request_timeout_seconds,
        "max_concurrency": CONFIG.effective_max_concurrency,
    },
    http={"port": args.port},
)
@bentoml.asgi_app(canonical_app, path="/")
class Boltz2Prediction:
    """Serves Boltz-2 structure predictions from a JSON payload."""

    def __init__(self) -> None:
        self._cache = CONFIG.cache
        self._cache.mkdir(parents=True, exist_ok=True)

        if CONFIG.download_assets:
            ensure_boltz2_assets(self._cache, CONFIG.mode)

        # Load the 2.3 GB checkpoint once; every request reuses this module.
        # The loaders install the kernels before they build any module.
        device = resolve_device(CONFIG.accelerator)
        self._model = load_boltz2_model(self._cache, optimizations=CONFIG.optimized, device=device)
        # The affinity model is a second checkpoint, loaded once and kept on the GPU.
        self._affinity_model = (
            load_affinity_model(
                self._cache, optimizations=CONFIG.optimized, device=device, compile_affinity=CONFIG.affinity_compiled
            )
            if needs_affinity_model(CONFIG.mode)
            else None
        )
        if CONFIG.affinity_compile == "on" and not CONFIG.affinity_compiled:
            logger.info(
                "BOLTZ2_AFFINITY_COMPILE=on is ignored: it needs BOLTZ2_MODE=affinity and BOLTZ2_OPTIMIZATIONS=on"
            )
        self._warmup = WarmupReport()
        if not CONFIG.optimized and self._affinity_model is None:
            self._warmup.state = STATE_DISABLED
            self._warmup.reason = "BOLTZ2_OPTIMIZATIONS=off"
        logger.info(
            "Boltz-2 loaded from %s (mode=%s, optimizations=%s, accelerator=%s)",
            self._cache,
            CONFIG.mode,
            CONFIG.optimizations,
            CONFIG.accelerator,
        )

    @bentoml.on_startup
    def _warm_up(self) -> None:
        """Compile the wrapped modules before the first request.

        With optimizations off nothing is compiled; only the affinity step runs, in affinity mode.

        BentoML runs this hook inside the server lifespan, so the HTTP port is not
        bound and no probe answers until it returns. ``__is_ready__`` repeats that
        guarantee for any future change in the hook order. A failed warm-up is
        logged and reported in ``runtime.warmup``; the service then starts anyway,
        because the compiled modules still work and compile on the first request.
        """
        if self._warmup.state != STATE_PENDING:
            return
        requests = warmup_requests() if CONFIG.optimized else []
        logger.info("Start-up warm-up: %d inputs", len(requests))
        extra_steps = (
            [("triangle_multiply_long", warm_up_large)] if CONFIG.optimized and seam_status()["mul_impl"] else []
        )
        if self._affinity_model is not None:
            extra_steps.append(("affinity", self._warm_up_affinity))
            if CONFIG.affinity_compiled:
                extra_steps.append(("affinity_crop", self._warm_up_affinity_crop))
        run_warmup(
            self._predict_for_warmup,
            requests,
            self._all_compile_status,
            self._warmup,
            extra_steps,
        )
        logger.info("Start-up warm-up %s in %s s", self._warmup.state, self._warmup.duration_s)

    def _all_compile_status(self) -> dict[str, dict[str, Any]]:
        """The compile status of the structure model, plus the affinity model's under ``affinity_<selector>``."""
        status = compile_status(self._model)
        if CONFIG.affinity_compiled and self._affinity_model is not None:
            status.update({f"affinity_{k}": v for k, v in compile_status(self._affinity_model).items()})
        return status

    def _warm_up_affinity(self) -> None:
        """Run the affinity warm-up input; it comes after the structure inputs and the long-sequence step."""
        steps = AFFINITY_COMPILE_WARMUP_SAMPLING_STEPS if CONFIG.affinity_compiled else WARMUP_SAMPLING_STEPS
        self._run_affinity_warmup(affinity_warmup_request(sampling_steps_affinity=steps))

    def _warm_up_affinity_crop(self) -> None:
        """Run the affinity warm-up input at the crop size."""
        self._run_affinity_warmup(
            affinity_warmup_request(
                AFFINITY_CROP_WARMUP_TOKENS,
                AFFINITY_CROP_WARMUP_SMILES,
                name="warmup_affinity_crop",
                diffusion_samples_affinity=1,
                sampling_steps_affinity=AFFINITY_COMPILE_WARMUP_SAMPLING_STEPS,
            )
        )

    def _run_affinity_warmup(self, request: dict[str, Any]) -> None:
        result = self._predict_for_warmup(**request)
        if result.get("error"):
            raise RuntimeError(result.get("message", "affinity warm-up prediction failed"))

    def _predict_for_warmup(self, **request: Any) -> dict[str, Any]:
        return run_boltz2_prediction(
            cache=self._cache,
            model=self._model,
            affinity_model=self._affinity_model,
            num_workers=CONFIG.num_workers,
            accelerator=CONFIG.accelerator,
            devices=1,
            **request,
        )

    def __is_ready__(self) -> bool:
        """Not ready while the warm-up runs."""
        return self._warmup.settled

    @staticmethod
    def _unavailable(ctx: bentoml.Context, message: str, retry_after_s: int = 60) -> dict[str, Any]:
        """Answer 503 with ``Retry-After`` and a short message.

        Raising ``ServiceUnavailable`` would also give 503, but BentoML replaces
        the body of any 5xx with a generic "an unexpected error has occurred",
        dropping the one detail the caller needs.
        """
        ctx.response.status_code = 503
        ctx.response.headers["Retry-After"] = str(retry_after_s)
        return dump(ErrorResponse, {"error": message})

    @staticmethod
    def _check_request(data: Boltz2Request) -> None:
        if has_affinity_property(data.properties):
            if CONFIG.mode == MODE_STRUCTURE:
                raise InvalidArgument("affinity is not served by this profile")
            check_affinity_request(data)
        check_limits(data)
        if data.use_msa_server and not CONFIG.msa_server_url:
            raise InvalidArgument("use_msa_server is not available: this deployment has no MSA server")

    def _run(self, data: Boltz2Request, msa_deadline: float) -> dict[str, Any]:
        """Execute one prediction and return the runner's result dict."""
        with seeded_prediction(data.seed):
            result = self._predict(data, msa_deadline)
        result["seed"] = data.seed
        return result

    def _predict(self, data: Boltz2Request, msa_deadline: float) -> dict[str, Any]:
        return run_boltz2_prediction(
            cache=self._cache,
            model=self._model,
            affinity_model=self._affinity_model,
            affinity_mw_correction=data.affinity_mw_correction,
            sampling_steps_affinity=data.sampling_steps_affinity,
            diffusion_samples_affinity=data.diffusion_samples_affinity,
            sequences=data.sequences,
            name=data.name,
            version=data.version,
            constraints=data.constraints,
            templates=data.templates,
            properties=data.properties,
            recycling_steps=data.recycling_steps,
            sampling_steps=data.sampling_steps,
            diffusion_samples=data.diffusion_samples,
            max_parallel_samples=data.max_parallel_samples,
            step_scale=data.step_scale,
            output_format=data.output_format,
            use_msa_server=data.use_msa_server,
            msa_server_url=CONFIG.msa_server_url,
            msa_pairing_strategy=data.msa_pairing_strategy,
            num_workers=CONFIG.num_workers,
            accelerator=CONFIG.accelerator,
            devices=1,
            method=data.method,
            msa_deadline=msa_deadline,
        )

    @bentoml.api(route="/v1/inference")
    def inference(self, data: Boltz2Request, ctx: bentoml.Context) -> dict[str, Any]:
        """Predict a structure and return it as JSON with confidence metrics.

        Response on success::

            {"structures": [{"record_id": ..., "format": "pdb"|"mmcif", "content": ...}],
             "confidence": {"<record_id>": {"confidence_score", "ptm", "iptm",
                                            "complex_plddt", ...}},
             "timing": {"runtime_s": <float>, "affinity_s": <float>},
             "affinity": {"affinity_pred_value", "affinity_probability_binary", ...},
             "seed": <int>|null,
             "runtime": {"mode": "structure"|"affinity", "optimizations": "on"|"off",
                         "triton": {"attention": <bool>, "mul_impl": "l1"|null},
                         "precision": "bf16-mixed",
                         "compile_status": {"transition": {...}, "token_transformer": {...}},
                         "affinity_compile_status": {...},  # BOLTZ2_AFFINITY_COMPILE=on only; same keys
                         "warmup": {"state", "duration_s", "compiles_during_warmup", "since_warmup", ...},
                         "cuda_peak_gb": <float>|null}}

        ``affinity`` and ``timing.affinity_s`` are present only when the request has an
        ``affinity`` property and ``BOLTZ2_MODE`` is ``affinity``; the keys are those of the
        ``affinity_<id>.json`` file of ``boltz predict``.

        Errors use the HTTP status and carry no exception text; the server log
        holds the detail. 400: the input is invalid or cannot be parsed. 422:
        the request is above a limit (``MAX_*``). 429:
        too many requests in flight. 500: the prediction failed. 503 with
        ``Retry-After``: the MSA server did not answer within its budget.
        504: the request timeout (``BOLTZ2_REQUEST_TIMEOUT_SECONDS``) passed.

        The request carries its own deadline, measured from its arrival and set
        inside that timeout. A request that waited so long that it can no longer
        finish gets 503 with ``Retry-After`` before it starts.

        ``runtime`` reports the kernel configuration the SERVING PROCESS actually
        resolved at startup. It exists because nothing else made that
        observable: the startup ``logger.info`` banner does not survive BentoML's
        logging configuration (checked against four captured serve logs — zero
        occurrences), so a client, a smoke test or an operator had no way to
        confirm which triangle implementation answered their request. Inferring
        it from latency is not a check. Additive and backward compatible: the
        harness reads ``structures`` / ``confidence`` / ``timing`` and is
        unaffected.

        ``runtime.compile_status`` is cumulative and includes the warm-up compiles;
        ``runtime.warmup.since_warmup`` holds the compiles of user traffic, also those of the
        affinity model (labels ``affinity_<selector>``).
        ``runtime.cuda_peak_gb`` is the process peak of allocated CUDA memory
        during this request, in GB. It includes the resident model weights.
        It is null when the request does not run on a CUDA device.

        ``runtime.compile_status`` is the ACTUAL, observed per-selector compile
        state for THIS worker's model instance — see ``compilation.CompileStatus``
        for the state values (``"not-yet-invoked"``, ``"compiled-executed"``,
        ``"eager-disabled"``, ``"degraded-fallback"``, ``"mixed-partial-degradation"``,
        ``"invoked-uncaught-error"``, ``"no-instances"``). Turning optimizations on only
        means a wrapper was installed; it is NOT evidence that any call has
        actually compiled, or that a real compile failure did not silently
        degrade the module to eager. ``compile_status`` is the only field that
        reports what actually ran, which is why it is read fresh off this
        worker's own model on every response rather than derived from
        ``CONFIG`` — a degraded-fallback module can still answer correctly, so
        this must never be read as, or conflated with, liveness/readiness.
        """
        self._check_request(data)
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
            result = self._run(data, budget.msa_deadline)
        except DeadlineExceeded as exc:
            logger.warning("Request not started: %s", exc)
            return self._unavailable(ctx, str(exc))
        except Exception as exc:  # noqa: BLE001 — mapped to a status below, detail goes to the log
            if is_msa_server_timeout(exc):
                logger.warning("MSA server budget exceeded: %s", exc)
                return self._unavailable(ctx, "The MSA server did not answer in time. Retry later.")
            logger.exception("Prediction failed")
            if isinstance(exc, ValueError):
                raise InvalidArgument("The request could not be processed as a Boltz-2 input.") from None
            raise InternalServerError("The prediction failed.") from None
        if result.get("error"):
            raise InvalidArgument(result.get("message", "The input produced no structure."))
        result.pop("error", None)
        peak_gb = result.pop("cuda_peak_gb", None)
        result["runtime"] = {**self._runtime_info(), "cuda_peak_gb": peak_gb}
        return dump(InferenceResponse, result)

    def _runtime_info(self) -> dict[str, Any]:
        """The optimization state this worker resolved, for client-side assertion.

        Instance method, not ``@staticmethod``: ``compile_status`` reads
        ``self._model``'s own observed compile state, which is per-model-
        instance by design (see ``compilation.compile_status``) — a fresh
        model load starts this at ``"not-yet-invoked"`` for every configured
        selector, never carrying over a prior model's state.
        """
        seams = seam_status()
        info: dict[str, Any] = {
            "mode": CONFIG.mode,
            "optimizations": CONFIG.optimizations,
            "triton": {"attention": seams["attention"], "mul_impl": seams["mul_impl"]},
            "precision": "bf16-mixed",
            "compile_status": compile_status(self._model),
            "warmup": self._warmup.to_dict(self._all_compile_status()),
        }
        if CONFIG.affinity_compiled:
            info["affinity_compile_status"] = compile_status(self._affinity_model)
        return dump(RuntimeInfo, info)


Boltz2Prediction.add_asgi_middleware(ArrivalStampMiddleware)  # type: ignore[attr-defined]
Boltz2Prediction.add_asgi_middleware(ErrorBodyMiddleware)  # type: ignore[attr-defined]
