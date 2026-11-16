# Copyright © Advanced Micro Devices, Inc., or its affiliates.
#
# SPDX-License-Identifier: MIT

"""
BentoML service for the Biohub ESMC 300M protein language model.

Serves the shared specialized-AIM HTTP surface:

  POST /v1/inference  protein sequence in, mean-pooled embedding out
                      (960-dimensional for the 300M model); with
                      ``"mode": "masked_lm"`` the masked-token loss and
                      perplexity of the sequence instead
  GET  /health        readiness, forwarded to BentoML ``/healthz``
  GET  /v1/models     OpenAI-shaped list with the served model id
  GET  /healthz, /readyz, /livez   BentoML built-ins (unchanged)

The masked-LM scoring is a model-specific mode of ``/v1/inference`` (the request
field ``mode``), so the service exposes only the certain endpoints. The harness
uses it as a correctness check of the language-model head.

The model is loaded at container startup through the native ``transformers``
ESMC port (``trust_remote_code`` is NOT used, so no code from the Hub repository
is executed and the ``esm`` package is not required).  Weights are read from and,
when absent, downloaded into ``AIM_CACHE_PATH`` only (Hugging Face hub cache
layout).  No HF token is required: ``biohub/ESMC-300M`` is not gated.
"""

import logging
import math
import os
import sys
from pathlib import Path
from typing import Annotated, Any, Literal, NoReturn

import bentoml
import httpx
import torch
from bentoml.exceptions import InvalidArgument
from pydantic import BaseModel, Field
from starlette.applications import Starlette
from starlette.requests import Request
from starlette.responses import JSONResponse
from starlette.routing import Route
from transformers import AutoTokenizer, EsmcForMaskedLM

# Service-specific override; AIM_MODEL_ID is reserved by the AIM runtime (catalog model selector).
MODEL_ID = os.environ.get("ESMC_HF_REPO", "biohub/ESMC-300M")

LOGGER = logging.getLogger(__name__)


def _configure_logging(target: logging.Logger) -> None:
    """Send this module's INFO lines to stderr, unbuffered, so ``docker logs`` shows them.

    BentoML's worker logging does not show application loggers, so the weights
    pre-flight line was invisible. Idempotent: adds one handler only.
    """
    if any(getattr(h, "_esmc_stderr", False) for h in target.handlers):
        return
    handler = logging.StreamHandler(sys.stderr)  # StreamHandler flushes after every record
    handler.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] [esmc-300m] %(message)s"))
    handler._esmc_stderr = True  # type: ignore[attr-defined]
    target.addHandler(handler)
    target.setLevel(logging.INFO)
    target.propagate = False


_configure_logging(LOGGER)


def _fail_startup(message: str) -> NoReturn:
    """Log a clear error naming the cause, then exit this BentoML worker with code 1.

    Behaviour of BentoML 1.x: ``os._exit(1)`` ends only the
    worker process. BentoML's supervisor restarts it about every 10 s, so the
    message repeats, the container keeps running and never becomes ready. The
    container does NOT exit. Making the container exit is a runtime/entrypoint
    concern (design team), not this service's; operators must use a readiness
    probe and watch the logs.
    """
    LOGGER.critical("ESMC-300M startup failed: %s", message)  # stderr handler, see _configure_logging
    os._exit(1)


class BentoArgs(BaseModel):
    """Template arguments AIM forwards via ``bentoml serve --arg``.

    ``aim-runtime`` resolves ``AIM_PORT`` and injects it as ``--arg port=...``, so
    this is the only source of the bound port. Deliberately no default.

    ``accelerator_count`` is the profile's ``metadata.accelerator_count``, which
    aim-runtime forwards as ``--arg accelerator_count=...`` to every BentoML AIM whose
    argument model declares the field. It sets the GPU resource request and the number
    of data-parallel workers (one per GPU). Single-GPU when absent.
    """

    port: int
    accelerator_count: int = Field(default=1, ge=1)


args = bentoml.use_arguments(BentoArgs)

# max_position_embeddings (2048) counts the <cls> and <eos> special tokens.
SPECIAL_TOKENS_PER_SEQUENCE = 2


# ---------------------------------------------------------------------------
# Shared GET endpoints (BentoML custom routes are POST-only, so these live in
# a Starlette app mounted at "/" on the service).
# ---------------------------------------------------------------------------


async def _upstream_healthz() -> httpx.Response:
    """Call BentoML's own readiness endpoint on this container."""
    async with httpx.AsyncClient(timeout=5) as client:
        return await client.get(f"http://127.0.0.1:{args.port}/healthz")


async def health(request: Request) -> JSONResponse:
    """GET /health — forwards BentoML /healthz readiness (200 once the model is loaded)."""
    try:
        upstream = await _upstream_healthz()
    except httpx.HTTPError:
        return JSONResponse({"status": "unavailable"}, status_code=503)
    return JSONResponse(
        {"status": "ok" if upstream.is_success else "unavailable"},
        status_code=upstream.status_code,
    )


async def models(request: Request) -> JSONResponse:
    """GET /v1/models — OpenAI-shaped list; data[0].id is the served model."""
    return JSONResponse({"object": "list", "data": [{"id": MODEL_ID, "object": "model"}]})


canonical_app = Starlette(
    routes=[
        Route("/health", health, methods=["GET"]),
        Route("/v1/models", models, methods=["GET"]),
    ]
)


# Request bodies are FLAT JSON objects ({"sequence": "..."}).  The routes take
# the fields as individual parameters on purpose: a single pydantic-model
# parameter would make BentoML 1.4 expect the body nested under the parameter
# name ({"request": {"sequence": "..."}}).
Sequence = Annotated[str, Field(min_length=1)]
Mode = Literal["embedding", "masked_lm"]
MaskFraction = Annotated[float, Field(gt=0.0, lt=1.0)]


class EmbedResponse(BaseModel):
    embedding: list[float]
    embedding_dim: int
    model: str


class MaskedLMResponse(BaseModel):
    masked_positions: int
    sequence_length: int
    loss: float
    perplexity: float
    model: str


def _hub_snapshot_dir(cache_root: Path) -> Path:
    """Directory where the HF hub layout keeps snapshots of MODEL_ID under cache_root."""
    return cache_root / ("models--" + MODEL_ID.replace("/", "--")) / "snapshots"


def _preflight_weights(cache_root: Path) -> bool:
    """Log the weights location and token need; return True when weights are already cached.

    Weights live only under ``cache_root`` (AIM_CACHE_PATH), in the HF hub cache
    layout the AIM runtime model-cache resolver understands. ``biohub/ESMC-300M``
    is not gated, so HF_TOKEN is not needed to download it.
    """
    snapshots = _hub_snapshot_dir(cache_root)
    cached = snapshots.is_dir() and any(snapshots.iterdir())
    has_token = bool(os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN"))
    LOGGER.info(
        "Weights pre-flight: AIM_CACHE_PATH=%s, %s cached, HF_TOKEN %s (not required: %s is not gated).",
        cache_root,
        "weights already" if cached else "no weights",
        "set" if has_token else "not set",
        MODEL_ID,
    )
    return cached


def _worker_device_index(worker_index: int, device_count: int, accelerator_count: int = 1) -> int:
    """Index of the visible GPU that BentoML worker ``worker_index`` (1-based) uses: worker k takes GPU k-1.

    One data-parallel worker per GPU, so ``accelerator_count`` GPUs must be visible. With
    fewer, the workers would silently share a GPU; raise instead (RuntimeError, so startup
    fails loudly).
    """
    if device_count < accelerator_count:
        raise RuntimeError(
            f"AIM_ACCELERATOR_COUNT={accelerator_count} requires {accelerator_count} accelerator(s), "
            f"but this container has {device_count}. Lower AIM_ACCELERATOR_COUNT (and use the "
            f"matching tp{device_count} profile), or give the container more accelerators."
        )
    return max(worker_index, 1) - 1


@bentoml.service(
    resources={"gpu": args.accelerator_count},
    # One data-parallel worker per GPU: the model fits on a single GPU, so more GPUs
    # raise throughput by serving requests concurrently, not by sharding the model.
    workers=args.accelerator_count,
    http={"port": args.port},
    traffic={"timeout": 60},
)
@bentoml.asgi_app(canonical_app, path="/")
class ESMCEmbeddingService:
    """BentoML service wrapping ESMC for per-sequence embeddings and masked-LM scoring."""

    def __init__(self) -> None:
        if not torch.cuda.is_available():
            _fail_startup("No GPU found. ESMC requires a CUDA/ROCm GPU.")
        worker_index = bentoml.server_context.worker_index or 1
        try:
            index = _worker_device_index(worker_index, torch.cuda.device_count(), args.accelerator_count)
        except RuntimeError as exc:
            _fail_startup(str(exc))
        # Make it the current device too, so any bare "cuda" or .cuda() lands on this worker's GPU.
        torch.cuda.set_device(index)
        device = f"cuda:{index}"
        self._device = device
        cache_root = Path(os.environ.get("AIM_CACHE_PATH", "/workspace/model-cache")).resolve()
        cache_root.mkdir(parents=True, exist_ok=True)
        _preflight_weights(cache_root)
        try:
            self._tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, cache_dir=str(cache_root))
            model = EsmcForMaskedLM.from_pretrained(MODEL_ID, cache_dir=str(cache_root), dtype=torch.bfloat16)
        except Exception as exc:
            _fail_startup(
                f"Could not load {MODEL_ID} weights under AIM_CACHE_PATH={cache_root} "
                "(pre-stage them there in the Hugging Face hub layout, or allow network access to the "
                f"Hugging Face Hub; HF_TOKEN is not required for this public model): {type(exc).__name__}: {exc}"
            )
        model = model.to(device)
        model.eval()
        self._model = model
        self._model_id = MODEL_ID
        self._residue_letters = frozenset(t for t in self._tokenizer.get_vocab() if len(t) == 1 and t.isalpha())
        self._max_residues = int(model.config.max_position_embeddings) - SPECIAL_TOKENS_PER_SEQUENCE

    @bentoml.api(route="/v1/inference")
    def v1_inference(
        self,
        sequence: Sequence,
        mode: Mode = "embedding",
        mask_fraction: MaskFraction = 0.15,
        seed: int = 0,
    ) -> dict[str, Any]:
        """POST /v1/inference — canonical AIM API.

        ``mode`` "embedding" (default) returns the mean-pooled embedding; "masked_lm" returns the
        masked-token loss and perplexity (``mask_fraction`` and ``seed`` apply to that mode only).
        """
        if mode == "masked_lm":
            return self._masked_lm(sequence, mask_fraction, seed).model_dump()
        return self._embed(sequence).model_dump()

    def _tokenize(self, sequence: str) -> dict[str, Any]:
        sequence = "".join(sequence.split()).upper()
        if not sequence:
            raise InvalidArgument("sequence is empty")
        if len(sequence) > self._max_residues:
            raise InvalidArgument(
                f"sequence has {len(sequence)} residues; ESMC supports at most {self._max_residues} "
                f"(context window {self._max_residues + SPECIAL_TOKENS_PER_SEQUENCE} tokens incl. <cls>/<eos>)"
            )
        unsupported = [c for c in dict.fromkeys(sequence) if c not in self._residue_letters]
        if unsupported:
            raise InvalidArgument(
                "sequence contains unsupported characters: "
                + ", ".join(repr(c) for c in unsupported[:5])
                + ("" if len(unsupported) <= 5 else ", ...")
            )
        return self._tokenizer(sequence, return_tensors="pt")

    def _embed(self, sequence: str) -> EmbedResponse:
        """Embed a single protein sequence.

        Returns the mean-pooled final-layer (post-norm) hidden state over the
        residue tokens only: the ``<cls>`` / ``<eos>`` special tokens are
        excluded from the mean.
        """
        inputs = self._tokenize(sequence)
        input_ids = inputs["input_ids"].to(self._device)
        attention_mask = inputs["attention_mask"].to(self._device)

        with torch.inference_mode():
            hidden = self._model.esmc(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state

        tok = self._tokenizer
        residue = attention_mask.bool()
        for special_id in (tok.cls_token_id, tok.eos_token_id, tok.pad_token_id):
            residue = residue & (input_ids != special_id)
        mask = residue.unsqueeze(-1).to(hidden.dtype)
        embedding = (hidden * mask).sum(1) / mask.sum(1)
        embedding_list: list[float] = embedding.squeeze(0).float().cpu().tolist()

        return EmbedResponse(
            embedding=embedding_list,
            embedding_dim=len(embedding_list),
            model=self._model_id,
        )

    def _masked_lm(self, sequence: str, mask_fraction: float, seed: int) -> MaskedLMResponse:
        """Mask a seeded random ``mask_fraction`` of residues; score the originals."""
        inputs = self._tokenize(sequence)
        input_ids = inputs["input_ids"]
        n_residues = input_ids.shape[1] - SPECIAL_TOKENS_PER_SEQUENCE
        n_mask = max(1, round(mask_fraction * n_residues))

        generator = torch.Generator().manual_seed(seed)
        # Residue positions are 1..n_residues (0 is <cls>, last is <eos>).
        positions = torch.randperm(n_residues, generator=generator)[:n_mask] + 1

        labels = torch.full_like(input_ids, -100)
        labels[0, positions] = input_ids[0, positions]
        masked_ids = input_ids.clone()
        masked_ids[0, positions] = self._tokenizer.mask_token_id

        with torch.inference_mode():
            logits = self._model(
                input_ids=masked_ids.to(self._device),
                attention_mask=inputs["attention_mask"].to(self._device),
            ).logits
        loss = torch.nn.functional.cross_entropy(
            logits.float().view(-1, logits.shape[-1]),
            labels.to(self._device).view(-1),
            ignore_index=-100,
        ).item()

        return MaskedLMResponse(
            masked_positions=int(n_mask),
            sequence_length=int(n_residues),
            loss=loss,
            perplexity=math.exp(loss),
            model=self._model_id,
        )
