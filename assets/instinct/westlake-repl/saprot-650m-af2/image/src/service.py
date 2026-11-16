# Copyright © Advanced Micro Devices, Inc., or its affiliates.
#
# SPDX-License-Identifier: MIT

"""
BentoML service for SaProt 650M AF2 structure-aware protein language model.

Canonical AIM API endpoints:
  GET  /health        - mounted ASGI app; forwards to BentoML's /healthz
  GET  /v1/models     - mounted ASGI app; OpenAI-shaped list of the served model
  POST /v1/inference  - accepts a structure-aware protein sequence string and
                        returns the mean-pooled embedding vector (1280-dimensional)
  GET  /healthz       - BentoML built-in, unchanged (also /livez, /readyz, /metrics)

SaProt uses an extended ESM-2 vocabulary of 446 entries (441 tokens + 5 specials) that interleaves amino
acid (AA) residue tokens with 3Di structural tokens derived from Foldseek.
Input sequences should alternate AA and 3Di tokens, e.g. 'M#A#K#T#' where '#'
is the unknown-structure placeholder used when no 3D structure is available.
One residue (AA + 3Di pair) is one token; at most 1022 residues are accepted
(1024-token context minus BOS/EOS), longer input is rejected with HTTP 400.

The model is loaded from the Hugging Face Hub at container startup.
``westlake-repl/SaProt_650M_AF2`` is publicly accessible without a HF token.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Any

import bentoml
import httpx
import torch
from bentoml.exceptions import InternalServerError, InvalidArgument
from pydantic import BaseModel, Field
from starlette.applications import Starlette
from starlette.requests import Request
from starlette.responses import JSONResponse
from starlette.routing import Route
from transformers import EsmForMaskedLM, EsmTokenizer

logger = logging.getLogger(__name__)

MODEL_ID = os.environ.get("AIM_MODEL_ID", "westlake-repl/SaProt_650M_AF2")
#: Identifier reported by GET /v1/models (the AIM's canonical name).
AIM_ID = "westlake-repl/saprot-650m-af2"
# Weights are cache content: downloaded at first start and kept under AIM_CACHE_PATH
# so the operator can mount a persistent volume there.
CACHE_PATH = Path(os.environ.get("AIM_CACHE_PATH", "/workspace/model-cache")).resolve()


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


def _get_device() -> torch.device:
    """Return this worker's CUDA/ROCm device, or raise if none is available.

    Raises RuntimeError rather than silently falling back to CPU so that a missing
    GPU attachment surfaces at startup. Defined here (not imported from harness.py)
    because the service must run in the layer-1 image, which has no aim_runtime.
    """
    if not torch.cuda.is_available():
        raise RuntimeError(
            "SaProt 650M AF2 requires a ROCm GPU but torch.cuda.is_available() "
            "returned False. Ensure the container is launched with a GPU device "
            "and the ROCm stack is initialised correctly."
        )
    worker_index = bentoml.server_context.worker_index or 1
    index = _worker_device_index(worker_index, torch.cuda.device_count(), args.accelerator_count)
    # Make it the current device too, so any bare "cuda" or .cuda() lands on this worker's GPU.
    torch.cuda.set_device(index)
    return torch.device(f"cuda:{index}")


class EmbedRequest(BaseModel):
    sequence: str


class EmbedResponse(BaseModel):
    embedding: list[float]
    embedding_dim: int
    model: str


class BentoArgs(BaseModel):
    """Template arguments AIM forwards via ``bentoml serve --arg``.

    ``aim-runtime`` resolves ``AIM_PORT`` and injects it as ``--arg port=...``, so
    this is the only source of the bound port. ``--arg`` is generic template data
    that BentoML does not act on by itself, hence the ``http={"port": ...}`` below.

    Deliberately no default: a bare ``bentoml serve`` is not a supported entry
    point, and a second default here could silently drift from ``AIM_PORT``.

    ``accelerator_count`` is the profile's ``metadata.accelerator_count``, which
    aim-runtime forwards as ``--arg accelerator_count=...`` to every BentoML AIM whose
    argument model declares the field. It sets the GPU resource request and the number
    of data-parallel workers (one per GPU). Single-GPU when the argument is absent.
    """

    port: int
    accelerator_count: int = Field(default=1, ge=1)


args = bentoml.use_arguments(BentoArgs)

#: Port this service binds (the forwarded ``--arg port``); the /health handler
#: calls BentoML's own /healthz on it.
PORT = args.port

SERVED_DTYPE = "fp16"

#: Model context is 1024 tokens. SaProt's vocabulary has one token per
#: (amino acid, 3Di structure token) pair, so one residue is one token and two input
#: characters. BOS and EOS take two positions, leaving 1022 residues (2044 characters).
MAX_TOKENS = 1024
MAX_RESIDUES = MAX_TOKENS - 2
_HEALTH_TIMEOUT_S = 5.0


#: Real SaProt vocabulary (vocab.txt, 446 entries = 5 specials + 441 tokens): amino-acid
#: letters and 3Di letters outside these sets (B J O U X Z / b j o u x z) are not in it
#: and would silently become <unk>.
AA_LETTERS = frozenset("ACDEFGHIKLMNPQRSTVWY")
STRUCT_LETTERS = frozenset("acdefghiklmnpqrstvwy#")


def _check_sequence_format(sequence: str) -> None:
    """Reject input that is not strictly paired structure-aware SaProt text (HTTP 400).

    Each residue is an amino-acid letter from ``AA_LETTERS`` (uppercase) followed by its
    3Di token, a letter from ``STRUCT_LETTERS`` (lowercase) or '#' (no structure). The
    stripped string must therefore have an even length. Characters outside the real
    vocabulary are rejected by name instead of silently becoming <unk>. Tokens with '#'
    first ('#a', '##') are valid vocabulary entries but are not accepted.
    """
    text = sequence.strip()
    if len(text) % 2 != 0:
        raise InvalidArgument(
            f"'sequence' must be structure-aware SaProt text: an amino-acid letter followed by a 3Di "
            f"token ('#' if none) for every residue, so its length must be even; got {len(text)} characters."
        )
    for i in range(0, len(text), 2):
        aa, struct = text[i], text[i + 1]
        if aa in AA_LETTERS and struct in STRUCT_LETTERS:
            continue
        pair = text[i : i + 2]
        where = f"characters {i + 1}-{i + 2} are {pair!r}"
        if aa in AA_LETTERS and struct.isupper():
            raise InvalidArgument(
                f"'sequence' must be structure-aware SaProt text: {where}. The 3Di (structure) letters "
                "must be lowercase; Foldseek prints them in uppercase, so lowercase its 3Di string before "
                "interleaving it with the amino-acid letters. Use '#' for an unknown structure."
            )
        raise InvalidArgument(
            f"'sequence' must be structure-aware SaProt text: {where}, expected one uppercase amino-acid "
            f"letter from {''.join(sorted(AA_LETTERS))} followed by one lowercase 3Di letter from "
            f"{''.join(sorted(STRUCT_LETTERS - {'#'}))} or '#'. Letters outside these sets "
            "(B, J, O, U, X, Z; lowercase b, j, o, u, x, z) are not in the model vocabulary, and tokens "
            "starting with '#' are not accepted. A plain amino-acid sequence is not accepted; "
            "append '#' to each residue."
        )


def _check_sequence_length(sequence: str) -> None:
    """Reject sequences longer than MAX_RESIDUES residues, before any model call.

    A residue is one amino-acid letter plus its 3Di structure token (two characters),
    so the count is half the stripped character length (the format check has already
    rejected odd lengths). Over-long input
    is never truncated: it raises InvalidArgument (HTTP 400).
    """
    residues = len(sequence.strip()) // 2
    if residues > MAX_RESIDUES:
        raise InvalidArgument(
            f"'sequence' is too long: {residues} residues received, the maximum is {MAX_RESIDUES} "
            f"residues ({MAX_TOKENS} tokens including start and end tokens). Shorten the sequence; "
            "it is not truncated."
        )


def _check_token_count(n_tokens: int) -> None:
    """Reject input whose real tokenized length exceeds the model maximum (HTTP 400).

    Defensive guard: after the format and length checks a string tokenizes to at most
    MAX_TOKENS ids, so this cannot trigger today. It is kept so a future vocabulary or
    tokenizer change cannot silently overrun the context.
    """
    if n_tokens > MAX_TOKENS:
        raise InvalidArgument(
            f"'sequence' is too long: it tokenizes to {n_tokens} tokens, the maximum is {MAX_TOKENS} "
            "tokens including start and end tokens. Shorten the sequence; it is not truncated."
        )


async def health(_request: Request) -> JSONResponse:
    """``GET /health``: forward to BentoML's ``/healthz`` on this service's own port.

    Returns the upstream status code, or 503 if the upstream cannot be reached.
    """
    try:
        async with httpx.AsyncClient(timeout=_HEALTH_TIMEOUT_S) as client:
            upstream = await client.get(f"http://127.0.0.1:{PORT}/healthz")
    except httpx.HTTPError:
        return JSONResponse({"status": "unavailable"}, status_code=503)
    return JSONResponse(
        {"status": "ok" if upstream.is_success else "unavailable"},
        status_code=upstream.status_code,
    )


def _accelerator_name() -> str | None:
    """Name of the GPU the service runs on, or None if it cannot be determined."""
    try:
        if torch.cuda.is_available():
            return str(torch.cuda.get_device_name(0))
    except Exception:  # noqa: BLE001 -- informational field only, never fail /v1/models
        logger.debug("Could not read the accelerator name", exc_info=True)
    return None


async def list_models(_request: Request) -> JSONResponse:
    """``GET /v1/models``: OpenAI-shaped list with the one served model."""
    return JSONResponse(
        {
            "object": "list",
            "data": [
                {
                    "id": AIM_ID,
                    "object": "model",
                    "metadata": {"dtype": SERVED_DTYPE, "accelerator": _accelerator_name()},
                }
            ],
        }
    )


canonical_app = Starlette(
    routes=[
        Route("/health", health, methods=["GET"]),
        Route("/v1/models", list_models, methods=["GET"]),
    ]
)


@bentoml.service(
    resources={"gpu": args.accelerator_count},
    # One data-parallel worker per GPU: the model fits on a single GPU, so more GPUs
    # raise throughput by serving requests concurrently, not by sharding the model.
    workers=args.accelerator_count,
    traffic={"timeout": 60},
    http={"port": args.port},
)
@bentoml.asgi_app(canonical_app, path="/")
class SaProtEmbeddingService:
    """BentoML service wrapping SaProt 650M AF2 for per-sequence mean-pooled embeddings.

    SaProt uses EsmForMaskedLM with a 446-entry vocabulary that interleaves amino
    acid tokens with 3Di structural tokens. Input sequences must alternate residue
    and 3Di tokens (e.g. 'M#A#K#T#'). The '#' token represents unknown structure
    and can be used for all positions when no AlphaFold2 prediction is available.
    Output is the mean-pooled last hidden state over residue positions (BOS/EOS excluded),
    producing a 1280-dimensional embedding identical in shape to ESM-2 650M.
    """

    def __init__(self) -> None:
        device = _get_device()
        self._device = device
        cache_dir = str(CACHE_PATH)
        self._tokenizer: EsmTokenizer = EsmTokenizer.from_pretrained(MODEL_ID, cache_dir=cache_dir)
        model: EsmForMaskedLM = EsmForMaskedLM.from_pretrained(MODEL_ID, cache_dir=cache_dir)
        model = model.half().to(device)
        model.eval()
        self._model = model
        self._model_id = MODEL_ID

    @bentoml.api(route="/v1/inference")
    def inference(self, request: EmbedRequest) -> EmbedResponse:
        """Embed a single structure-aware protein sequence.

        The sequence must use SaProt's interleaved AA+3Di format.  Use '#' as
        the 3Di placeholder for residues without structural information.

        Input that is not strictly paired (AA + 3Di) or over 1022 residues / 1024 tokens is rejected with HTTP 400 before the model runs;
        input is never truncated.

        Returns the mean-pooled last hidden state over the residue positions only.
        The BOS and EOS tokens are not part of the average.
        """
        if not request.sequence.strip():
            raise InvalidArgument("'sequence' must be a non-empty string")
        _check_sequence_format(request.sequence)
        _check_sequence_length(request.sequence)

        try:
            tokenized: dict[str, Any] = self._tokenizer(
                request.sequence,
                return_tensors="pt",
                padding=True,
                truncation=False,
            )
        except Exception as exc:
            # The tokenizer only sees validated input, so a failure here is caused by the input.
            logger.warning("SaProt tokenizer rejected the input: %s", exc)
            raise InvalidArgument("'sequence' could not be tokenized; check the structure-aware format") from exc
        _check_token_count(int(tokenized["input_ids"].shape[-1]))

        try:
            inputs = {k: v.to(self._device) for k, v in tokenized.items()}

            with torch.no_grad():
                outputs = self._model(**inputs, output_hidden_states=True)

            hidden = outputs.hidden_states[-1]
            # Single unpadded sequence: drop BOS (first) and EOS (last) before averaging.
            embedding = hidden[0, 1:-1].float().mean(0)
            embedding_list: list[float] = embedding.cpu().tolist()
        except Exception as exc:
            # Log the detail server-side; never echo internals to the client.
            logger.exception("SaProt inference failed")
            raise InternalServerError("Embedding failed due to a server error") from exc

        return EmbedResponse(
            embedding=embedding_list,
            embedding_dim=len(embedding_list),
            model=self._model_id,
        )
