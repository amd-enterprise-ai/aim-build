# Copyright © Advanced Micro Devices, Inc., or its affiliates.
#
# SPDX-License-Identifier: MIT

"""
BentoML service for SemlaFlow small-molecule generation (SMILES output).

Requires an accelerator. The model and its checkpoint are loaded ONCE at service start and a
warm-up generation is run before the port opens, so ``/health`` answers only when the service is
ready and the one-time GPU start-up cost is not paid per request. Requests are then served in
this process, one at a time, on one worker thread (see ``_GenerationEngine``). Exposes the
Canonical AIM API endpoints:
  GET  /health        -> mounted ASGI app; forwards to BentoML's /healthz
  GET  /v1/models     -> mounted ASGI app; OpenAI-shaped list of the served model
  POST /v1/inference  -> {"smiles": [...], "n_molecules": int, "num_requested": int,
                         "num_valid": int, "num_invalid": int, "elapsed_s": float}
  GET  /healthz       -> BentoML built-in, unchanged (also /livez, /readyz, /metrics)

The checkpoint is operator-staged; AIM_CACHE_PATH must point to a directory
containing the checkpoint file (default: model.ckpt). See the AIM API
documentation (section 4.7) for the expected mount layout.
"""

from __future__ import annotations

import concurrent.futures
import contextlib
import logging
import os
import re
import secrets
import sys
import time
import types
from collections import OrderedDict
from collections.abc import Iterator
from pathlib import Path
from typing import Any, NoReturn

import bentoml
import httpx
from bentoml.exceptions import InternalServerError, InvalidArgument, ServiceUnavailable
from pydantic import BaseModel, Field
from starlette.applications import Starlette
from starlette.requests import Request
from starlette.responses import JSONResponse
from starlette.routing import Route


def _configure_logging(target: logging.Logger) -> None:
    """Send this service's INFO lines to stderr, unbuffered, so ``docker logs`` shows them.

    BentoML's worker logging does not show application loggers, so the model-load and
    warm-up lines would be invisible. Idempotent: adds one handler only.
    """
    if any(getattr(h, "_semlaflow_stderr", False) for h in target.handlers):
        return
    handler = logging.StreamHandler(sys.stderr)  # StreamHandler flushes after every record
    handler.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] [semlaflow] %(message)s"))
    handler._semlaflow_stderr = True  # type: ignore[attr-defined]
    target.addHandler(handler)
    target.setLevel(logging.INFO)
    target.propagate = False


class BentoArgs(BaseModel):
    """Template arguments AIM forwards via ``bentoml serve --arg``.

    ``accelerator_count`` is the profile's ``metadata.accelerator_count``, which
    aim-runtime forwards as ``--arg accelerator_count=...`` to every BentoML AIM whose
    argument model declares the field. It sets the GPU resource request and the number
    of data-parallel workers (one resident model per GPU). Single-GPU when absent.
    """

    port: int
    accelerator_count: int = Field(default=1, ge=1)


args = bentoml.use_arguments(BentoArgs)

#: Port this service binds (the forwarded ``--arg port``); the /health handler
#: calls BentoML's own /healthz on it.
PORT = args.port

MODEL_ID = "rssrwn/semlaflow"
_HEALTH_TIMEOUT_S = 5.0

_DEFAULT_CACHE = "/workspace/model-cache"  # aim_runtime.config.DEFAULT_CACHE_PATH
_DEFAULT_CKPT_NAME = "model.ckpt"
_DEFAULT_INPUT = "/workspace/data"
_VALID_SPLITS = ("train", "val", "test")

logger = logging.getLogger(__name__)
_configure_logging(logger)

#: Upstream checkout (see the Dockerfile); put on ``sys.path`` when the model is loaded.
SEMLAFLOW_PATH = os.environ.get("SEMLAFLOW_PATH", "/opt/semlaflow")

#: One bound on how long a request may take, counted from its arrival (queue time included).
#: A request that does not finish in time is answered 503; see ``_GenerationEngine``.
REQUEST_TIMEOUT_SECONDS = int(os.environ.get("SEMLAFLOW_TIMEOUT_SECONDS", "1800"))

DEVICE_ENV = "SEMLAFLOW_DEVICE"
DEFAULT_DEVICE = "cuda"
_DEVICE_RE = re.compile(r"(cpu|cuda|cuda:\d+)")

#: Device the resident model runs on (``cuda``, ``cuda:<n>`` or ``cpu``); set once at startup.
_ACTIVE_DEVICE = "cuda"

#: Molecules / ODE steps of the start-up warm-up generation (one real batch through the real path).
WARMUP_N_MOLECULES = 4
WARMUP_INTEGRATION_STEPS = 2
#: Splits kept in memory after their first read (a split file is read and parsed once, not per request).
_DATASET_CACHE_SIZE = 3


def _resolve_input_dir(data_path: str) -> Path:
    """Resolve ``data_path`` (following symlinks) and require it under AIM_INPUT_PATH."""
    root = Path(os.environ.get("AIM_INPUT_PATH", _DEFAULT_INPUT)).resolve()
    resolved = Path(data_path).resolve()
    if not resolved.is_relative_to(root) or not resolved.is_dir():
        raise InvalidArgument(f"data_path must be an existing directory under {root}")
    return resolved


def _worker_device_index(worker_index: int, device_count: int, accelerator_count: int = 1) -> int:
    """Index of the visible GPU that BentoML worker ``worker_index`` (1-based) uses: worker k takes GPU k-1.

    One resident model per GPU, so ``accelerator_count`` GPUs must be visible. With fewer, the
    workers would silently share a GPU; raise instead (RuntimeError, so startup fails loudly).
    """
    if device_count < accelerator_count:
        raise RuntimeError(
            f"AIM_ACCELERATOR_COUNT={accelerator_count} requires {accelerator_count} accelerator(s), "
            f"but this container has {device_count}. Lower AIM_ACCELERATOR_COUNT (and use the "
            f"matching tp{device_count} profile), or give the container more accelerators."
        )
    return max(worker_index, 1) - 1


def _activate_device(torch_module: Any, device: str) -> None:
    """Make ``cuda:<k>`` this worker's current CUDA device, so a bare ``"cuda"`` or ``.cuda()`` lands on it."""
    if device.startswith("cuda:"):
        torch_module.cuda.set_device(int(device.partition(":")[2]))


def _resolve_device(torch_module: Any, worker_index: int = 1, accelerator_count: int = 1) -> str:
    """Return the torch device the resident model must use, or raise RuntimeError.

    Default ``cuda`` (ROCm builds of torch also use the ``cuda`` name); with
    ``accelerator_count`` above 1 the default is this worker's own ``cuda:<k-1>`` (fewer visible GPUs than
    ``accelerator_count`` raises).
    ``SEMLAFLOW_DEVICE``
    overrides it with ``cpu``, ``cuda`` or ``cuda:<index>``; any other value is an error. A GPU
    request with no visible GPU, or an index beyond the visible GPUs, raises instead of falling
    back, so a mistake never silently runs on the CPU. ``cpu`` is used only when asked for, with a
    warning.
    """
    configured = os.environ.get(DEVICE_ENV, "").strip().lower()
    raw = configured or DEFAULT_DEVICE
    if not _DEVICE_RE.fullmatch(raw):
        raise RuntimeError(f"{DEVICE_ENV}={raw!r} is not valid; use 'cpu', 'cuda' or 'cuda:<index>'.")
    if raw == "cpu":
        logger.warning("%s=cpu: generation runs on the CPU, which is far slower than a GPU", DEVICE_ENV)
        return "cpu"
    if not torch_module.cuda.is_available():
        raise RuntimeError(
            "No GPU detected (torch.cuda.is_available() is False). SemlaFlow requires an AMD GPU "
            f"(set {DEVICE_ENV}=cpu to run on the CPU on purpose)."
        )
    if ":" in raw and int(raw.split(":", 1)[1]) >= torch_module.cuda.device_count():
        raise RuntimeError(f"{DEVICE_ENV}={raw!r}: only {torch_module.cuda.device_count()} GPU(s) visible.")
    if not configured and accelerator_count > 1:
        return f"cuda:{_worker_device_index(worker_index, torch_module.cuda.device_count(), accelerator_count)}"
    return raw


def _resolve_ckpt_path() -> Path:
    cache = Path(os.environ.get("AIM_CACHE_PATH", _DEFAULT_CACHE)).resolve()
    return cache / _DEFAULT_CKPT_NAME


class GenerationError(Exception):
    """A generation run failed or produced no molecule; the message is safe to log, not to return."""


class _Job:
    """One generation request: the validated inputs, plus the per-request random seed."""

    __slots__ = ("data_dir", "dataset_split", "integration_steps", "n_molecules", "seed", "warmup")

    def __init__(
        self,
        data_dir: Path | None,
        dataset_split: str,
        n_molecules: int,
        integration_steps: int,
        seed: int,
        warmup: bool = False,
    ) -> None:
        self.data_dir = data_dir
        self.dataset_split = dataset_split
        self.n_molecules = n_molecules
        self.integration_steps = integration_steps
        self.seed = seed
        self.warmup = warmup


class _DatasetCache:
    """Small LRU of parsed ``.smol`` datasets keyed by (path, size, mtime_ns); replaced files are re-read."""

    def __init__(self, loader: Any, size: int = _DATASET_CACHE_SIZE) -> None:
        self._loader = loader
        self._size = size
        self._items: OrderedDict[tuple[str, int, int], Any] = OrderedDict()

    def get(self, path: Path) -> Any:
        st = os.stat(path)
        key = (str(path), st.st_size, st.st_mtime_ns)
        if key in self._items:
            self._items.move_to_end(key)
            return self._items[key]
        for stale in [k for k in self._items if k[0] == key[0]]:
            del self._items[stale]
        data = self._loader(path)
        self._items[key] = data
        while len(self._items) > self._size:
            self._items.popitem(last=False)
        return data


class _ResidentGenerator:
    """Upstream ``semlaflow.predict`` building blocks run in this process on one resident model.

    ``predict.main`` rebuilds the vocabulary, reloads the checkpoint and re-reads the dataset on
    every call. Here the model is built once with upstream's own ``load_model`` and moved to the
    chosen device, and each request calls upstream's ``build_dm`` (datamodule, sampling of the
    conditioning molecules, interpolant) and the same ``_generate`` / ``_generate_mols`` calls
    that ``scriptutil.generate_molecules`` makes. Differences from the old subprocess run: the
    device is not hard-coded to ``cuda``; the metrics set-up is skipped (it is not needed to
    return SMILES and is what the old run had to stub out); the molecules are turned into SMILES
    in memory instead of through the SDF file; parsed dataset splits are kept (``_DatasetCache``).
    The seed is set per request, as the subprocess run did.
    """

    def __init__(self, device: str, ckpt_path: Path) -> None:
        self.device = device
        self.ckpt_path = ckpt_path
        self._upstream: Any = None
        self._util: Any = None
        self._model: Any = None
        self._vocab: Any = None
        self._hparams: dict[str, Any] = {}
        self._datasets: _DatasetCache | None = None
        self._warmup_data: Any = None

    def load(self) -> None:
        """Import upstream (needs ``SEMLAFLOW_PATH`` on ``sys.path``), build the model, put it on the device."""
        if SEMLAFLOW_PATH not in sys.path:
            sys.path.insert(0, SEMLAFLOW_PATH)
        import semlaflow.predict as upstream  # noqa: PLC0415 (upstream checkout, importable once on sys.path)
        import semlaflow.scriptutil as util  # noqa: PLC0415
        import torch  # noqa: PLC0415 (heavy import, only needed in the serving process)

        util.disable_lib_stdout()
        util.configure_fs()
        self._upstream = upstream
        self._util = util
        self._vocab = util.build_vocab()
        load_args = types.SimpleNamespace(
            ckpt_path=str(self.ckpt_path),
            integration_steps=upstream.DEFAULT_INTEGRATION_STEPS,
            ode_sampling_strategy=upstream.DEFAULT_ODE_SAMPLING_STRATEGY,
            cat_sampling_noise_level=upstream.DEFAULT_CAT_SAMPLING_NOISE_LEVEL,
        )
        # Upstream reads the checkpoint with a bare ``torch.load``; on a GPU-saved file that would
        # place tensors on cuda:0 whatever device was chosen (and fail on the CPU).
        real_load = torch.load

        def load_on_device(path: Any, *a: Any, **kw: Any) -> Any:
            kw.setdefault("map_location", self.device)
            return real_load(path, *a, **kw)

        torch.load = load_on_device
        try:
            model = upstream.load_model(load_args, self._vocab)
            self._hparams = torch.load(str(self.ckpt_path))["hyper_parameters"]
        finally:
            torch.load = real_load
        self._model = model.to(self.device).eval()
        # Bound to the original class now: while a request runs ``upstream.GeometricDataset`` is the
        # shim below, whose ``load`` reads this cache.
        read_split = upstream.GeometricDataset.load
        self._datasets = _DatasetCache(lambda path: read_split(path))

    @contextlib.contextmanager
    def _dataset_source(self, job: _Job) -> Iterator[None]:
        """Make ``predict.build_dm`` read parsed data from memory (or the synthetic warm-up set)."""
        upstream = self._upstream
        real_cls = upstream.GeometricDataset
        generator = self

        class _Shim(real_cls):  # type: ignore[valid-type, misc]
            @classmethod
            def load(cls, data_path: Any, transform: Any = None, min_size: Any = None) -> Any:
                if job.warmup:
                    data = generator._warmup_dataset()
                else:
                    data = generator._datasets.get(Path(data_path))  # type: ignore[union-attr]
                return real_cls(data._data, transform=transform)

        upstream.GeometricDataset = _Shim
        try:
            yield
        finally:
            upstream.GeometricDataset = real_cls

    def _warmup_dataset(self) -> Any:
        """A one-molecule dataset (ethanol with an embedded 3D conformer); needs no operator data."""
        if self._warmup_data is None:
            from rdkit import Chem  # noqa: PLC0415
            from rdkit.Chem import AllChem  # noqa: PLC0415
            from semlaflow.util.molrepr import GeometricMol, GeometricMolBatch  # noqa: PLC0415

            mol = Chem.AddHs(Chem.MolFromSmiles("CCO"))
            if AllChem.EmbedMolecule(mol, randomSeed=7) != 0:
                raise RuntimeError("could not embed the warm-up molecule")
            batch = GeometricMolBatch.from_list([GeometricMol.from_rdkit(mol)])
            self._warmup_data = self._upstream.GeometricDataset(batch)
        return self._warmup_data

    def generate(self, job: _Job) -> list[tuple[str, str]]:
        """Run one generation; return ``(canonical SMILES, 3D mol block)`` pairs of the molecules that passed sanitisation."""
        import lightning  # noqa: PLC0415
        from predict_output import collect_structures, rdkit_molblock, rdkit_smiles  # noqa: PLC0415

        upstream = self._upstream
        # Fresh seed per request (upstream's fixed seed would replay one random sequence).
        lightning.seed_everything(job.seed, workers=True)
        run_args = types.SimpleNamespace(
            dataset="geom-drugs",
            dataset_split=job.dataset_split,
            data_path=str(job.data_dir) if job.data_dir is not None else "",
            n_molecules=job.n_molecules,
            batch_cost=upstream.DEFAULT_BATCH_COST,
            bucket_cost_scale=upstream.DEFAULT_BUCKET_COST_SCALE,
        )
        strategy = upstream.DEFAULT_ODE_SAMPLING_STRATEGY
        with self._dataset_source(job):
            dm = upstream.build_dm(run_args, self._hparams, self._vocab)
        mols: list[Any] = []
        for batch in dm.test_dataloader():
            tensors = {k: v.to(self.device) for k, v in batch[0].items()}
            output = self._model._generate(tensors, job.integration_steps, strategy)
            mols.extend(self._model._generate_mols(output))
        smiles, molblocks, _n_invalid = collect_structures(mols, rdkit_smiles, rdkit_molblock, job.n_molecules)
        return list(zip(smiles, molblocks))

    def release_gpu_memory(self) -> None:
        if self.device == "cpu":
            return
        try:
            import torch  # noqa: PLC0415

            torch.cuda.empty_cache()
        except Exception:  # noqa: BLE001 -- best effort after a failed run
            logger.debug("empty_cache failed", exc_info=True)

    def memory_note(self) -> str:
        """``; accelerator memory: X MiB allocated, Y MiB reserved`` for the ready log line ('' on the CPU)."""
        if self.device == "cpu":
            return ""
        try:
            import torch  # noqa: PLC0415

            mib = 2**20
            return (
                f"; accelerator memory: {torch.cuda.memory_allocated() / mib:.0f} MiB allocated, "
                f"{torch.cuda.memory_reserved() / mib:.0f} MiB reserved"
            )
        except Exception:  # noqa: BLE001 -- informational only
            return ""


class _GenerationEngine:
    """One resident model, one worker thread, bounded waits.

    All accelerator work runs on a single worker thread, so there is exactly one model instance and
    requests are served one at a time in arrival order (the serialisation the subprocess design had
    too). The caller waits for its result for ``REQUEST_TIMEOUT_SECONDS`` counted from arrival.

    Timeout semantics. A thread cannot be killed the way the old subprocess was. When the wait runs
    out: a request that has not started yet is cancelled and never runs; a request that is already
    running keeps the worker (and the accelerator) until it ends on its own, the caller gets ``503``
    at once, and every new request is answered ``503`` immediately ("still running") until that run
    ends. Only a restart of the container stops a run that never ends.
    """

    def __init__(self, generator: _ResidentGenerator) -> None:
        self._generator = generator
        self._executor = concurrent.futures.ThreadPoolExecutor(max_workers=1, thread_name_prefix="semlaflow")
        self._abandoned: set[concurrent.futures.Future[Any]] = set()

    def _run(self, job: _Job) -> list[tuple[str, str]]:
        started = time.perf_counter()
        try:
            smiles = self._generator.generate(job)
        except Exception as exc:
            # Detail stays in the server log; it can hold paths and tracebacks.
            logger.exception("generation failed (%s)", type(exc).__name__)
            self._generator.release_gpu_memory()
            raise GenerationError("generation failed") from None
        logger.info(
            "generated %d of %d requested molecule(s), %d step(s), in %.3f s",
            len(smiles),
            job.n_molecules,
            job.integration_steps,
            time.perf_counter() - started,
        )
        return smiles

    def warm_up(self) -> None:
        """Load the model and run one small real generation. Raises RuntimeError on any failure.

        Runs on the worker thread, so the accelerator context, kernels and caches that later
        requests use are the ones created here.
        """
        started = time.perf_counter()

        def work() -> None:
            self._generator.load()
            self._run(
                _Job(None, "test", WARMUP_N_MOLECULES, WARMUP_INTEGRATION_STEPS, seed=37, warmup=True),
            )

        try:
            self._executor.submit(work).result()
        except Exception as exc:
            raise RuntimeError(f"Model warm-up failed on device {self._generator.device}: {exc}") from exc
        logger.info(
            "SemlaFlow resident model ready on %s after warm-up in %.2f s%s",
            self._generator.device,
            time.perf_counter() - started,
            self._generator.memory_note(),
        )

    def generate(self, job: _Job, timeout: float = REQUEST_TIMEOUT_SECONDS) -> list[tuple[str, str]]:
        """Run one job; ``ServiceUnavailable`` (503) if it cannot finish within *timeout* seconds."""
        if self._abandoned:
            raise ServiceUnavailable("A previous request that timed out is still running; try again later")
        future = self._executor.submit(self._run, job)
        try:
            return future.result(timeout=timeout)
        except concurrent.futures.TimeoutError:
            if future.cancel():
                logger.error("request waited %ss in the queue and was cancelled", timeout)
                raise ServiceUnavailable(
                    f"Generation did not start within {timeout:g} seconds (service busy)"
                ) from None
            self._abandoned.add(future)
            future.add_done_callback(self._abandoned.discard)
            logger.error("generation exceeded %ss; it cannot be interrupted and keeps running until it ends", timeout)
            raise ServiceUnavailable(f"Generation exceeded {timeout:g} seconds") from None
        except GenerationError as exc:
            raise InternalServerError(str(exc)) from None


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
        import torch  # deferred: heavy import, informational field only

        if _ACTIVE_DEVICE != "cpu" and torch.cuda.is_available():
            index = int(_ACTIVE_DEVICE.split(":", 1)[1]) if ":" in _ACTIVE_DEVICE else 0
            return str(torch.cuda.get_device_name(index))
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
                    "id": MODEL_ID,
                    "object": "model",
                    "metadata": {"accelerator": _accelerator_name()},
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


def _fail_startup(message: str) -> NoReturn:
    """Log a clear error naming the cause, then exit this BentoML worker with code 1.

    ``os._exit(1)`` ends only the worker process; BentoML's supervisor restarts it, so the failure
    message repeats and the container never becomes ready. Making the container exit is a
    runtime/entrypoint concern; operators must use a readiness probe and watch the logs.
    """
    logger.critical("SemlaFlow startup failed: %s", message)
    print(f"SemlaFlow startup failed: {message}", file=sys.stderr, flush=True)
    os._exit(1)


# BentoML's default request timeout is 60 s, which would answer 504 long before the service's own
# limit. The traffic timeout must exceed REQUEST_TIMEOUT_SECONDS so that a slow run ends in the
# service's own 503.
@bentoml.service(
    name="semlaflow",
    resources={"gpu": args.accelerator_count},
    # One data-parallel worker per GPU, each with its own resident model: more GPUs raise
    # throughput by serving requests concurrently.
    workers=args.accelerator_count,
    http={"port": args.port},
    traffic={"timeout": REQUEST_TIMEOUT_SECONDS + 60},
)
@bentoml.asgi_app(canonical_app, path="/")
class SemlaFlowService:
    """BentoML service wrapping SemlaFlow molecule generation on a resident model.

    The model is loaded and warmed up once in ``__init__`` (BentoML opens the port only after it
    returns, so ``/health`` answers only once the service is ready). If the device or checkpoint is
    unusable or the warm-up fails, the constructor logs a CRITICAL message and ends its worker
    (see ``_fail_startup``).
    """

    def __init__(self) -> None:
        global _ACTIVE_DEVICE  # noqa: PLW0603 -- read by GET /v1/models
        import torch  # noqa: PLC0415 (heavy import, only needed in the serving process)

        try:
            device = _resolve_device(torch, bentoml.server_context.worker_index or 1, args.accelerator_count)
            _activate_device(torch, device)
            ckpt_path = _resolve_ckpt_path()
            if not ckpt_path.is_file():
                raise RuntimeError(
                    f"SemlaFlow checkpoint not found at {ckpt_path}. "
                    f"Weights are read from AIM_CACHE_PATH only; mount the checkpoint directory there "
                    f"(currently: {os.environ.get('AIM_CACHE_PATH', _DEFAULT_CACHE)})."
                )
        except RuntimeError as exc:
            _fail_startup(str(exc))
        _ACTIVE_DEVICE = device
        logger.info("SemlaFlow device: %s; checkpoint: %s", device, ckpt_path)
        self._engine = _GenerationEngine(_ResidentGenerator(device, ckpt_path))
        try:
            self._engine.warm_up()
        except RuntimeError as exc:
            _fail_startup(str(exc))

    @bentoml.api(route="/v1/inference")
    def inference(
        self,
        data_path: str,
        n_molecules: int = 100,
        integration_steps: int = 100,
        dataset_split: str = "test",
        seed: int | None = None,
    ) -> dict:
        """Generate molecules with the resident model.

        Args:
            data_path: Directory containing <dataset_split>.smol (train.smol,
                       val.smol or test.smol); train.smol is read only for
                       the train split.
            n_molecules: Number of molecules to generate.
            integration_steps: Number of ODE integration steps.
            dataset_split: Which split the generation script samples the molecule
                           sizes from ("train", "val", or "test"); it is not a
                           conditioning signal for the model.
            seed: Optional integer from 0 to 2147483647 (2**31 - 1). When omitted, each request
                  draws a fresh random seed.

        Returns:
            dict with keys: smiles (list[str], canonical SMILES of the molecules
            that passed RDKit sanitisation), molblocks (list[str], the 3D structure of
            each molecule in smiles as MolBlock text, same order and length), n_molecules (int, len(smiles)),
            num_requested (int), num_valid (int, len(smiles)), num_invalid (int,
            requested molecules not in smiles), elapsed_s (float, wall time of this call
            including any wait for an earlier request).

        Failures: 400 for bad input, 500 for a failed run or no valid molecule, 503 when the
        request cannot finish within SEMLAFLOW_TIMEOUT_SECONDS or an earlier timed-out run is
        still occupying the accelerator.
        """
        data_dir = _resolve_input_dir(data_path)
        if n_molecules < 1 or integration_steps < 1:
            raise InvalidArgument("n_molecules and integration_steps must be positive")
        if dataset_split not in _VALID_SPLITS:
            raise InvalidArgument(f"dataset_split must be one of {list(_VALID_SPLITS)}")
        if seed is not None and not 0 <= seed < 2**31:
            raise InvalidArgument("seed must be an integer from 0 to 2147483647")
        t0 = time.monotonic()
        # A fresh seed per request unless the caller gives one, so equal-sized requests are independent batches.
        job_seed = secrets.randbits(31) if seed is None else seed
        job = _Job(data_dir, dataset_split, n_molecules, integration_steps, seed=job_seed)
        results = self._engine.generate(job)
        smiles = [smi for smi, _block in results]
        molblocks = [block for _smi, block in results]
        n_invalid = n_molecules - len(smiles)
        return {
            "smiles": smiles,
            "molblocks": molblocks,
            "n_molecules": len(smiles),
            "num_requested": n_molecules,
            "num_valid": len(smiles),
            "num_invalid": n_invalid,
            "elapsed_s": round(time.monotonic() - t0, 3),
        }
