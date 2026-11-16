# Copyright © Advanced Micro Devices, Inc., or its affiliates.
#
# SPDX-License-Identifier: MIT

"""
BentoML service for H-optimus-0 (Bioptimus pathology foundation model).

H-optimus-0 is a 1.1B-parameter Vision Transformer (ViT) trained on
histopathology images. It produces 1536-dimensional CLS-token embeddings
from 224x224 image tiles. Loaded via timm (not the HuggingFace transformers
pipeline, which is not supported for this model architecture).

Canonical AIM API endpoints:
  GET  /health        — mounted ASGI app; forwards to BentoML's /healthz
  GET  /v1/models     — mounted ASGI app; OpenAI-shaped list of the served model
  POST /v1/inference  — embed one image tile, returns 1536-dim vector
  GET  /healthz       — BentoML built-in, unchanged (also /livez, /readyz, /metrics)

Request body for /v1/inference (exactly one of image_path or image_base64):
  {"image_path": "/path/to/tile.png"}
  {"image_base64": "<base64-encoded PNG or JPEG>"}

Response:
  {"embedding": [...], "embedding_dim": 1536, "model": "bioptimus/H-optimus-0"}

Images must be exactly 224x224 H&E patches extracted at 0.5 microns-per-pixel;
any other size is rejected with HTTP 400 (the service does not resize).
The service normalises with mean=(0.707223, 0.578729, 0.703617) and
std=(0.211883, 0.230117, 0.177517) as specified in the model card.
"""

from __future__ import annotations

import base64
import binascii
import io
import logging
import os
from pathlib import Path
from typing import Any

import bentoml
import httpx
import timm
import torch
from bentoml.exceptions import InvalidArgument
from PIL import Image, UnidentifiedImageError
from pydantic import BaseModel, Field
from starlette.applications import Starlette
from starlette.requests import Request
from starlette.responses import JSONResponse
from starlette.routing import Route
from torchvision import transforms

LOGGER = logging.getLogger(__name__)

MODEL_ID = "bioptimus/H-optimus-0"
EMBEDDING_DIM = 1536
TILE_SIZE = 224

#: Environment variable that overrides the CPU thread count (see ``_resolve_num_threads``).
THREADS_ENV = "HOPTIMUS0_NUM_THREADS"
#: Upper bound of the default thread count. The default PyTorch pool (one thread per usable
#: CPU) stalls the CPU preprocessing in some requests on large hosts; a small fixed count is
#: stable.
DEFAULT_MAX_THREADS = 8

_NORMALIZE = transforms.Normalize(
    mean=(0.707223, 0.578729, 0.703617),
    std=(0.211883, 0.230117, 0.177517),
)

_TRANSFORM = transforms.Compose(
    [
        transforms.ToTensor(),
        _NORMALIZE,
    ]
)


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

    Raises RuntimeError rather than silently falling back to CPU so that
    deployment misconfiguration (missing GPU attachment) surfaces immediately
    at startup rather than producing silently slow/incorrect inference.
    """
    if not torch.cuda.is_available():
        raise RuntimeError(
            "H-optimus-0 requires a ROCm GPU but torch.cuda.is_available() "
            "returned False. Ensure the container is launched with a GPU device "
            "and the ROCm stack is initialised correctly."
        )
    # CUDA_VISIBLE_DEVICES remaps the visible GPUs to indices 0..N-1, so torch
    # indices are relative to the visible set (a host index such as GPU 5 would be
    # out of range): worker k takes visible device k-1.
    worker_index = bentoml.server_context.worker_index or 1
    index = _worker_device_index(worker_index, torch.cuda.device_count(), args.accelerator_count)
    # Make it the current device too, so any bare "cuda" or .cuda() lands on this worker's GPU.
    torch.cuda.set_device(index)
    return torch.device(f"cuda:{index}")


def _usable_cpus() -> int:
    """Number of CPUs this process may run on (cpuset/affinity aware), at least 1."""
    try:
        return max(1, len(os.sched_getaffinity(0)))
    except (AttributeError, OSError):
        return max(1, os.cpu_count() or 1)


def _resolve_num_threads(environ: Any = os.environ) -> int:
    """CPU thread count for PyTorch: ``HOPTIMUS0_NUM_THREADS`` or ``min(8, usable CPUs)``.

    Raises ``RuntimeError`` on a value that is not a positive integer, so a typo
    surfaces at startup instead of silently keeping the slow default.
    """
    raw = environ.get(THREADS_ENV, "").strip()
    if not raw:
        return min(DEFAULT_MAX_THREADS, _usable_cpus())
    try:
        value = int(raw)
    except ValueError:
        value = 0
    if value < 1:
        raise RuntimeError(f"{THREADS_ENV}={raw!r} is not a positive integer.")
    return value


def _default_thread_env() -> int:
    """Resolve the thread count and default ``OMP_NUM_THREADS``/``MKL_NUM_THREADS`` to it (never overriding the operator).

    Called once at import time, while the process is still single-threaded:
    changing ``os.environ`` later, with CUDA/UCX threads running, is unsafe
    (``setenv`` is not safe against a concurrent ``getenv``).
    """
    n = _resolve_num_threads()
    os.environ.setdefault("OMP_NUM_THREADS", str(n))
    os.environ.setdefault("MKL_NUM_THREADS", str(n))
    return n


NUM_THREADS = _default_thread_env()


def _limit_threads() -> int:
    """Limit the PyTorch CPU thread pool to ``NUM_THREADS`` (``torch.set_num_threads``).

    Image decode and normalisation run on the CPU in every request; with one
    thread per CPU on a large host they stall on thread-pool synchronisation.
    Must run AFTER the model is loaded onto the device: calling it before the
    CUDA runtime is initialised crashed the process on some starts; after the load
    it does not.
    """
    torch.set_num_threads(NUM_THREADS)
    LOGGER.info(
        "CPU threads: %d (%s overrides; default min(%d, usable CPUs))", NUM_THREADS, THREADS_ENV, DEFAULT_MAX_THREADS
    )
    return NUM_THREADS


def _validate_path_within(path: Path, base: Path, label: str) -> Path:
    """Resolve *path* (following symlinks) and require an existing file under *base*.

    Raises ``InvalidArgument`` (HTTP 400). Containment and existence share one
    message so a caller cannot probe the filesystem outside *base*.
    """
    resolved = path.resolve()
    base_resolved = base.resolve()
    try:
        resolved.relative_to(base_resolved)
        contained = True
    except ValueError:
        contained = False
    if not contained or not resolved.is_file():
        raise InvalidArgument(f"{label} must be an existing file under {base_resolved}")
    return resolved


class BentoArgs(BaseModel):
    """Template arguments AIM forwards via ``bentoml serve --arg``.

    ``aim-runtime`` resolves ``AIM_PORT`` and injects it as ``--arg port=...``,
    so this is the only source of the bound port. ``--arg`` is generic template
    data that BentoML does not act on by itself, hence the ``http={"port": ...}``
    below.

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
_HEALTH_TIMEOUT_S = 5.0


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
        LOGGER.debug("Could not read the accelerator name", exc_info=True)
    return None


async def list_models(_request: Request) -> JSONResponse:
    """``GET /v1/models``: OpenAI-shaped list with the one served model."""
    return JSONResponse(
        {
            "object": "list",
            "data": [
                {
                    "id": MODEL_ID,
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


def _hub_snapshot_dir(cache_root: Path) -> Path:
    """Directory where the HF hub layout keeps snapshots of MODEL_ID under cache_root."""
    return cache_root / ("models--" + MODEL_ID.replace("/", "--")) / "snapshots"


def _preflight_weights(cache_root: Path) -> None:
    """Log the weights location and token need; stop startup if weights cannot be obtained.

    Weights live only under ``cache_root`` (AIM_CACHE_PATH), in the HF hub cache
    layout that the AIM runtime model-cache resolver understands. The repository
    is gated, so a download needs HF_TOKEN.
    """
    snapshots = _hub_snapshot_dir(cache_root)
    cached = snapshots.is_dir() and any(snapshots.iterdir())
    has_token = bool(os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN"))
    LOGGER.info(
        "Weights pre-flight: AIM_CACHE_PATH=%s, %s cached, HF_TOKEN %s (gated repository %s; needed only to download).",
        cache_root,
        "weights already" if cached else "no weights",
        "set" if has_token else "not set",
        MODEL_ID,
    )
    if not cached and not has_token:
        raise RuntimeError(
            f"{MODEL_ID} weights are not present under AIM_CACHE_PATH={cache_root} "
            "and HF_TOKEN is not set. The repository is gated: set HF_TOKEN to a token with "
            "access, or pre-stage the weights in the Hugging Face hub layout under that path."
        )


@bentoml.service(
    name="h-optimus-0",
    resources={"gpu": args.accelerator_count},
    # One data-parallel worker per GPU: the model fits on a single GPU, so more GPUs
    # raise throughput by serving requests concurrently, not by sharding the model.
    workers=args.accelerator_count,
    traffic={"timeout": 120},
    http={"port": args.port},
)
@bentoml.asgi_app(canonical_app, path="/")
class HOptimus0Service:
    """BentoML service wrapping Bioptimus H-optimus-0.

    Loads the model once at startup from AIM_CACHE_PATH, downloading it from the
    HuggingFace Hub into that path when absent (requires HF_TOKEN for the gated
    repository; startup stops with a clear message otherwise). All /v1/inference
    calls are served in fp16 via torch.autocast for ROCm efficiency.
    """

    def __init__(self) -> None:
        LOGGER.info("Initialising H-optimus-0 service ...")
        self._device = _get_device()
        LOGGER.info("Using device: %s", self._device)

        cache_root = Path(os.environ.get("AIM_CACHE_PATH", "/workspace/model-cache")).resolve()
        cache_root.mkdir(parents=True, exist_ok=True)
        self._input_root = Path(os.environ.get("AIM_INPUT_PATH", "/workspace/data")).resolve()
        _preflight_weights(cache_root)
        try:
            self._model = timm.create_model(
                f"hf-hub:{MODEL_ID}",
                pretrained=True,
                cache_dir=str(cache_root),
                init_values=1e-5,
                dynamic_img_size=False,
            )
        except Exception as exc:
            raise RuntimeError(
                f"Could not load {MODEL_ID} weights under AIM_CACHE_PATH={cache_root} "
                f"(check HF_TOKEN and network access): {type(exc).__name__}: {exc}"
            ) from exc
        self._model.to(self._device)
        self._model.eval()
        self._num_threads = _limit_threads()
        LOGGER.info("H-optimus-0 loaded and ready on %s.", self._device)

    def _decode_image(self, image_path: str | None, image_base64: str | None) -> Image.Image:
        """Load a PIL Image from a file path (under AIM_INPUT_PATH) or base64 payload.

        Every client-side failure raises ``InvalidArgument`` (HTTP 400) with a
        fixed message; the underlying exception text is never returned.
        """
        try:
            if image_path is not None:
                resolved = _validate_path_within(Path(image_path), self._input_root, "image_path")
                image = Image.open(resolved).convert("RGB")
            elif image_base64 is not None:
                raw = base64.b64decode(image_base64, validate=True)
                image = Image.open(io.BytesIO(raw)).convert("RGB")
            else:
                raise InvalidArgument("Provide exactly one of 'image_path' or 'image_base64'.")
        except (binascii.Error, UnidentifiedImageError, OSError, ValueError, Image.DecompressionBombError):
            raise InvalidArgument("The input is not a readable PNG or JPEG image.") from None
        if image.size != (TILE_SIZE, TILE_SIZE):
            raise InvalidArgument(
                f"The tile must be exactly {TILE_SIZE}x{TILE_SIZE} pixels (H&E, 0.5 microns per pixel); "
                f"got {image.width}x{image.height}. The service does not resize."
            )
        return image

    def _embed(self, image: Image.Image) -> list[float]:
        """Run one image through the model and return the embedding as a list."""
        tensor = _TRANSFORM(image).unsqueeze(0).to(self._device)
        with torch.autocast(device_type="cuda", dtype=torch.float16):
            with torch.inference_mode():
                features = self._model(tensor)
        embedding: list[float] = features.squeeze(0).float().cpu().tolist()
        return embedding

    @bentoml.api(route="/v1/inference")
    def inference(
        self,
        image_path: str | None = None,
        image_base64: str | None = None,
    ) -> dict[str, Any]:
        """Embed one 224x224 pathology tile.

        Provide exactly one of image_path (file under AIM_INPUT_PATH) or
        image_base64 (base64-encoded PNG or JPEG bytes).

        Returns a dict with keys:
          embedding      — list of 1536 floats (CLS-token representation)
          embedding_dim  — 1536
          model          — "bioptimus/H-optimus-0"
        """
        if (image_path is None) == (image_base64 is None):
            raise InvalidArgument("Provide exactly one of 'image_path' or 'image_base64', not both or neither.")
        image = self._decode_image(image_path, image_base64)
        embedding = self._embed(image)
        return {
            "embedding": embedding,
            "embedding_dim": EMBEDDING_DIM,
            "model": MODEL_ID,
        }
