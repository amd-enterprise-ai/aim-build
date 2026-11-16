# Copyright © Advanced Micro Devices, Inc., or its affiliates.
#
# SPDX-License-Identifier: MIT

"""
ProteinMPNN BentoML service.

Exposes (via BentoML's class-based ``@bentoml.service`` SDK), surface:
  GET  /health        -> mounted ASGI app; forwards to BentoML's /healthz
  GET  /v1/models     -> mounted ASGI app; OpenAI-shaped list of the served model
  POST /v1/inference  -> {"sequences": [...], "scores": [...]}
  GET  /healthz       -> BentoML built-in, unchanged (also /livez, /readyz, /metrics)

The service runs ``protein_mpnn_run.main`` from the cloned ProteinMPNN repo
(at ``PROTEINMPNN_PATH``) IN-PROCESS, with one resident model: the weights are
loaded and a warm-up design is run once at service start, before the service
accepts connections, so the one-time GPU start-up cost
is not paid per request. Requests run one at a time on a single worker thread.
The weights are not part of the image: they live under
``$AIM_CACHE_PATH/proteinmpnn/vanilla_model_weights`` and a pre-flight at startup
downloads any missing file. Inputs are a PDB file path and sampling
parameters; outputs are the designed sequences and their log-prob scores.

How AIM launches it
-------------------
The BentoML engine entry in ``engines.yaml`` runs::

    python -m bentoml serve --working-dir /workspace/model --arg port=<AIM_PORT>

``bentoml serve`` reads ``bentofile.yaml`` (``service: "service:ProteinMPNNService"``).
AIM forwards the serving port as a BentoML template argument (``--arg port=…``).
"""

from __future__ import annotations

import concurrent.futures
import contextlib
import glob
import hashlib
import io
import logging
import os
import re
import sys
import tempfile
import threading
import time
import types
from collections.abc import Iterator
from pathlib import Path
from typing import Any, NoReturn

import bentoml
import httpx
from bentoml.exceptions import BentoMLException, InvalidArgument, ServiceUnavailable
from pydantic import BaseModel, Field
from starlette.applications import Starlette
from starlette.requests import Request
from starlette.responses import JSONResponse
from starlette.routing import Route

logger = logging.getLogger(__name__)


def _configure_logging(target: logging.Logger) -> None:
    """Send this service's INFO lines to stderr, unbuffered, so ``docker logs`` shows them.

    BentoML's worker logging does not show application loggers, so the pre-flight
    and weight-download lines (the cold download takes tens of seconds) were
    invisible. Idempotent: adds one handler only.
    """
    if any(getattr(h, "_pmpnn_stderr", False) for h in target.handlers):
        return
    handler = logging.StreamHandler(sys.stderr)  # StreamHandler flushes after every record
    handler.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] [proteinmpnn] %(message)s"))
    handler._pmpnn_stderr = True  # type: ignore[attr-defined]
    target.addHandler(handler)
    target.setLevel(logging.INFO)
    target.propagate = False


_configure_logging(logger)

PROTEINMPNN_PATH = os.environ.get("PROTEINMPNN_PATH", "/opt/proteinmpnn")

#: Model id reported by ``GET /v1/models`` (the AIM id).
MODEL_ID = "dauparas/proteinmpnn"

_HEALTH_TIMEOUT_S = 5.0

# Caller-supplied ``pdb_path`` must resolve (symlinks followed) to a file under
# AIM_INPUT_PATH or under the image's baked fixture directory, which the
# harness uses for its smoke and benchmark structures.
INPUT_ROOT = Path(os.environ.get("AIM_INPUT_PATH", "/workspace/data"))
FIXTURE_ROOT = Path(os.environ.get("PROTEINMPNN_FIXTURE_DIR", "/workspace/model/fixtures"))

# Weights are the four vanilla files of the pinned upstream commit. They are
# never baked into the image: they live under AIM_CACHE_PATH and a
# startup pre-flight downloads any missing file. ``model_name`` is limited to these.
ALLOWED_MODEL_NAMES = ("v_48_002", "v_48_010", "v_48_020", "v_48_030")

#: Upstream commit (tag v1.0.1) the weights are fetched from and checked against.
WEIGHTS_SOURCE_COMMIT = "905b0081f3aa3404d7ac22174bc25710f9995ff8"
WEIGHTS_URL_TEMPLATE = (
    "https://raw.githubusercontent.com/dauparas/ProteinMPNN/"
    + WEIGHTS_SOURCE_COMMIT
    + "/vanilla_model_weights/{name}.pt"
)
WEIGHT_SIZE = 6681301
#: sha256 of vanilla_model_weights/<name>.pt at WEIGHTS_SOURCE_COMMIT.
WEIGHT_SHA256 = {
    "v_48_002": "925f2ca1007bf9b02e0e7f420ff00eb91f50fcc2722f64b42e644ae95adaa131",
    "v_48_010": "db866fae956a28661f926053d630610c55e9fc4bc03922f2aeeb98a37435ccce",
    "v_48_020": "c9cb4a671d79604111231f8dbfc7c590e06f1197453b7a6854ac6661a642f5bd",
    "v_48_030": "c34b7bfb38418ea30989fda3314f4781ac4e3920f9825731cf555f1fed44ac66",
}
_DOWNLOAD_TIMEOUT_S = 120.0
_DOWNLOAD_LOCK = threading.Lock()
#: (path, size, mtime_ns) of files already checksum-verified in this process.
_VERIFIED_WEIGHTS: set[tuple[str, int, int]] = set()


def _cache_root() -> Path:
    """Cache root: ``AIM_CACHE_PATH`` (default ``/workspace/model-cache``)."""
    return Path(os.environ.get("AIM_CACHE_PATH", "/workspace/model-cache")).resolve()


#: Directory passed to ``protein_mpnn_run.py --path_to_model_weights``.
WEIGHTS_DIR = _cache_root() / "proteinmpnn" / "vanilla_model_weights"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _check_weight(path: Path, name: str) -> None:
    """Raise RuntimeError unless ``path`` has the pinned size and sha256."""
    size = path.stat().st_size
    if size != WEIGHT_SIZE:
        raise RuntimeError(f"size {size} != expected {WEIGHT_SIZE}")
    if _sha256(path) != WEIGHT_SHA256[name]:
        raise RuntimeError("sha256 checksum mismatch")


def _ensure_weight(name: str) -> None:
    """Make sure ``WEIGHTS_DIR/<name>.pt`` exists and is verified, downloading it if missing.

    The download goes to a temporary file in the same directory, is verified
    (size and sha256), then renamed atomically. Raises RuntimeError naming the
    path and the URL on any failure.
    """
    dest = WEIGHTS_DIR / f"{name}.pt"
    url = WEIGHTS_URL_TEMPLATE.format(name=name)
    with _DOWNLOAD_LOCK:
        tmp_name = ""
        try:
            if dest.is_file():
                st = dest.stat()
                if (str(dest), st.st_size, st.st_mtime_ns) not in _VERIFIED_WEIGHTS:
                    _check_weight(dest, name)
                    _VERIFIED_WEIGHTS.add((str(dest), st.st_size, st.st_mtime_ns))
                return
            logger.info("ProteinMPNN weight missing at %s; downloading from %s (no token needed)", dest, url)
            dest.parent.mkdir(parents=True, exist_ok=True)
            fd, tmp_name = tempfile.mkstemp(dir=dest.parent, prefix=f".{dest.name}.", suffix=".part")
            with (
                os.fdopen(fd, "wb") as out,
                httpx.stream("GET", url, follow_redirects=True, timeout=_DOWNLOAD_TIMEOUT_S) as resp,
            ):
                resp.raise_for_status()
                for chunk in resp.iter_bytes(1 << 20):
                    out.write(chunk)
            _check_weight(Path(tmp_name), name)
            os.replace(tmp_name, dest)
            st = dest.stat()
            _VERIFIED_WEIGHTS.add((str(dest), st.st_size, st.st_mtime_ns))
        except (OSError, httpx.HTTPError, RuntimeError) as exc:
            if tmp_name:
                Path(tmp_name).unlink(missing_ok=True)
            raise RuntimeError(
                f"Could not provide ProteinMPNN weight {dest} (source {url}): {exc}. "
                f"Pre-stage the four vanilla files under {WEIGHTS_DIR} (AIM_CACHE_PATH={_cache_root()})."
            ) from exc


def _preflight_weights() -> None:
    """Log the cache path, then verify or download every weight file (RuntimeError on failure)."""
    logger.info("ProteinMPNN weights directory: %s (AIM_CACHE_PATH=%s)", WEIGHTS_DIR, _cache_root())
    for name in ALLOWED_MODEL_NAMES:
        _ensure_weight(name)


#: Upper bound, in seconds, on how long one request waits for its result (queue wait plus run).
#: A running design cannot be killed (it runs on a thread): see ``_DesignEngine``.
REQUEST_TIMEOUT_SECONDS = 600

#: Lines of captured upstream stdout kept for the error log when a run fails.
OUTPUT_TAIL_LINES = 50

DEVICE_ENV = "PROTEINMPNN_DEVICE"
DEFAULT_DEVICE = "cuda"
_DEVICE_RE = re.compile(r"(cpu|cuda|cuda:0)")

#: Device the resident model runs on (``cuda`` or ``cpu``); set once at startup.
_ACTIVE_DEVICE = "cuda"


#: PyTorch-ROCm honours either variable; setting only one leaves the other free to disagree.
_VISIBILITY_ENV_VARS = ("HIP_VISIBLE_DEVICES", "CUDA_VISIBLE_DEVICES")


def _accelerator_count_error(accelerator_count: int, found: str) -> str:
    """Startup message when the container cannot give every worker its own accelerator."""
    return (
        f"AIM_ACCELERATOR_COUNT={accelerator_count} requires {accelerator_count} accelerator(s), but {found}. "
        "Lower AIM_ACCELERATOR_COUNT (and use the matching tp profile), or give the container more accelerators."
    )


def _select_worker_device(visible: list[str] | None, worker_index: int, accelerator_count: int) -> str | None:
    """Device id BentoML worker ``worker_index`` (1-based) must be restricted to, or None to leave it alone.

    Upstream ``protein_mpnn_run.py`` always uses ``cuda:0``, so a worker is given its own GPU by narrowing the
    visibility variables to that one device (a ``torch.device`` index would not reach upstream). ``visible`` is the
    device list already imposed on the process (BentoML sets ``CUDA_VISIBLE_DEVICES`` to the same list for every worker),
    or None when unrestricted. A single accelerator needs no pinning.

    Raises RuntimeError when fewer devices are visible than workers, so two workers never share a GPU. When
    unrestricted the device count is not known yet (asking torch would initialise the GPU runtime, after which
    the narrowing no longer takes effect); a missing id then shows up as no visible GPU, see ``_resolve_device``.
    """
    if accelerator_count <= 1:
        return None
    if visible is None:
        return str(max(worker_index, 1) - 1)
    if len(visible) < accelerator_count:
        raise RuntimeError(
            _accelerator_count_error(
                accelerator_count, f"HIP_VISIBLE_DEVICES/CUDA_VISIBLE_DEVICES expose {len(visible)}: {visible}"
            )
        )
    if len(set(visible)) < accelerator_count:
        raise RuntimeError(
            f"HIP_VISIBLE_DEVICES/CUDA_VISIBLE_DEVICES repeat device ids ({visible}), so "
            f"{accelerator_count} worker(s) cannot each get their own accelerator"
        )
    return visible[max(worker_index, 1) - 1]


def _pin_worker_device(worker_index: int, accelerator_count: int) -> str | None:
    """Restrict this worker process to its own GPU and return its id (None: left alone).

    Call before anything initialises the GPU runtime.
    """
    visible = None
    for name in _VISIBILITY_ENV_VARS:
        raw = os.environ.get(name)
        if raw is not None:
            visible = [part.strip() for part in raw.split(",") if part.strip()]
            break
    device = _select_worker_device(visible, worker_index, accelerator_count)
    if device is not None:
        for name in _VISIBILITY_ENV_VARS:
            os.environ[name] = device
        logger.info("worker %d of %d uses GPU %s", worker_index, accelerator_count, device)
    return device


def _resolve_device(torch_module: Any, pinned_device: str | None = None, accelerator_count: int = 1) -> str:
    """Return ``cuda`` or ``cpu`` for the resident model, or raise RuntimeError.

    Default ``cuda`` (ROCm builds of torch also use the ``cuda`` name). ``PROTEINMPNN_DEVICE``
    overrides it with ``cpu``, ``cuda`` or ``cuda:0``; any other value is an error. Upstream
    ``protein_mpnn_run.py`` always uses ``cuda:0``, so a worker's own GPU is selected with
    ``HIP_VISIBLE_DEVICES`` / ``CUDA_VISIBLE_DEVICES`` (see ``_pin_worker_device``). A GPU request with no visible GPU raises
    instead of falling back, so a mistake never silently runs on the CPU. ``cpu`` is used only
    when asked for, with a warning. When this worker was narrowed to ``pinned_device`` and that GPU does not exist
    in the container, the error says that ``AIM_ACCELERATOR_COUNT`` is too high.
    """
    raw = os.environ.get(DEVICE_ENV, "").strip().lower() or DEFAULT_DEVICE
    if not _DEVICE_RE.fullmatch(raw):
        raise RuntimeError(f"{DEVICE_ENV}={raw!r} is not valid; use 'cpu' or 'cuda'.")
    if raw == "cpu":
        logger.warning("%s=cpu: designs run on the CPU, which is slow", DEVICE_ENV)
        return "cpu"
    if not torch_module.cuda.is_available():
        if pinned_device is not None:
            raise RuntimeError(
                _accelerator_count_error(
                    accelerator_count, f"GPU {pinned_device} (this worker's) is not present in this container"
                )
            )
        raise RuntimeError(
            "No GPU visible to PyTorch (torch.cuda.is_available() is False); this AIM requires an accelerator "
            f"(set {DEVICE_ENV}=cpu to run on the CPU on purpose)."
        )
    return "cuda"


# Defaults of upstream protein_mpnn_run.py's argparse at the pinned commit (WEIGHTS_SOURCE_COMMIT).
# ``main(args)`` reads every one of these; only the per-request fields are overridden.
_RUN_ARG_DEFAULTS: dict[str, Any] = {
    "ca_only": False,
    "path_to_model_weights": "",
    "model_name": "v_48_020",
    "seed": 0,
    "save_score": 0,
    "save_probs": 0,
    "score_only": 0,
    "conditional_probs_only": 0,
    "conditional_probs_only_backbone": 0,
    "unconditional_probs_only": 0,
    "backbone_noise": 0.0,
    "num_seq_per_target": 1,
    "batch_size": 1,
    "max_length": 200000,
    "sampling_temp": "0.1",
    "out_folder": None,
    "pdb_path": "",
    "pdb_path_chains": "",
    "jsonl_path": None,
    "chain_id_jsonl": "",
    "fixed_positions_jsonl": "",
    "omit_AAs": "X",
    "bias_AA_jsonl": "",
    "bias_by_res_jsonl": "",
    "omit_AA_jsonl": "",
    "pssm_jsonl": "",
    "pssm_multi": 0.0,
    "pssm_threshold": 0.0,
    "pssm_log_odds_flag": 0,
    "pssm_bias_flag": 0,
    "tied_positions_jsonl": "",
}


def _run_args(
    *, pdb_path: str, out_dir: str, num_seq: int, sampling_temp: float, model_name: str, seed: int
) -> types.SimpleNamespace:
    """The argparse namespace that ``protein_mpnn_run.main`` expects (see ``_RUN_ARG_DEFAULTS``)."""
    values = dict(_RUN_ARG_DEFAULTS)
    values.update(
        pdb_path=pdb_path,
        out_folder=out_dir,
        num_seq_per_target=num_seq,
        sampling_temp=str(sampling_temp),
        model_name=model_name,
        path_to_model_weights=str(WEIGHTS_DIR) + os.sep,
        seed=seed,
        save_score=1,
        batch_size=1,
    )
    return types.SimpleNamespace(**values)


def _warmup_pdb_text(n_residues: int = 64) -> str:
    """A synthetic alpha-helix backbone (N, CA, C, O per residue) used only for the start-up warm-up."""
    import math  # noqa: PLC0415

    lines = []
    serial = 1
    for i in range(n_residues):
        angle = math.radians(100.0 * i)
        cx, cy, cz = 2.3 * math.cos(angle), 2.3 * math.sin(angle), 1.5 * i
        for name, element, (dx, dy, dz) in (
            ("N", "N", (-1.2, -0.6, -0.6)),
            ("CA", "C", (0.0, 0.0, 0.0)),
            ("C", "C", (1.2, 0.6, 0.6)),
            ("O", "O", (1.4, 1.8, 0.7)),
        ):
            lines.append(
                f"ATOM  {serial:5d}  {name:<3s} ALA A{i + 1:4d}    "
                f"{cx + dx:8.3f}{cy + dy:8.3f}{cz + dz:8.3f}  1.00  0.00           {element}"
            )
            serial += 1
    lines.append("END")
    return "\n".join(lines) + "\n"


class DesignError(Exception):
    """A design run failed or produced nothing; the detail is only in the server log."""


class _ResidentRunner:
    """Runs upstream ``protein_mpnn_run.main`` in this process with a resident model.

    ``main`` builds a ProteinMPNN, loads a checkpoint and writes FASTA/npz files on every call.
    To keep its code path and output format byte for byte, it is called unchanged, and three
    things are swapped for the duration of one call (the engine runs one call at a time):

    * ``protein_mpnn_utils.ProteinMPNN`` (imported by ``main`` at call time) is a factory that
      returns one cached model object per configuration; ``main`` still copies the requested
      checkpoint into it with ``load_state_dict``, so every ``model_name`` is served by the
      same resident object;
    * ``torch.load`` returns each checkpoint from memory after its first read;
    * for ``cpu`` only, ``torch.cuda.is_available`` reports False (``main`` picks its device from it).

    Seeding is done by ``main`` itself on every call, so the same request gives the same sequences.
    """

    def __init__(self, device: str) -> None:
        self.device = device
        self._main: Any = None
        self._shim: Any = None
        self._models: dict[str, Any] = {}
        self._checkpoints: dict[str, Any] = {}

    def load(self) -> None:
        """Import upstream's modules (needs ``PROTEINMPNN_PATH`` on ``sys.path``)."""
        if PROTEINMPNN_PATH not in sys.path:
            sys.path.insert(0, PROTEINMPNN_PATH)
        import protein_mpnn_run  # noqa: PLC0415 (upstream script, only importable once PROTEINMPNN_PATH is on sys.path)
        import protein_mpnn_utils  # noqa: PLC0415

        real_class = protein_mpnn_utils.ProteinMPNN
        models = self._models

        def factory(*args: Any, **kwargs: Any) -> Any:
            key = repr(sorted(kwargs.items()))
            if key not in models:
                models[key] = real_class(*args, **kwargs)
            return models[key]

        # Replacing the name inside the real module would break the class's own
        # ``super(ProteinMPNN, self)``; main() gets a shim module instead.
        shim = types.ModuleType("protein_mpnn_utils")
        shim.__dict__.update({k: v for k, v in vars(protein_mpnn_utils).items() if not k.startswith("__")})
        shim.ProteinMPNN = factory  # type: ignore[attr-defined]
        self._shim = shim
        self._real_utils = protein_mpnn_utils
        self._main = protein_mpnn_run.main

    @contextlib.contextmanager
    def _patched(self) -> Iterator[None]:
        import torch  # noqa: PLC0415

        real_load = torch.load
        real_available = torch.cuda.is_available
        real_utils = sys.modules.get("protein_mpnn_utils")
        checkpoints = self._checkpoints

        def cached_load(path: Any, *args: Any, **kwargs: Any) -> Any:
            key = str(path)
            if key not in checkpoints:
                checkpoints[key] = real_load(path, *args, **kwargs)
            return checkpoints[key]

        sys.modules["protein_mpnn_utils"] = self._shim
        torch.load = cached_load
        if self.device == "cpu":
            torch.cuda.is_available = lambda: False  # type: ignore[assignment]
        try:
            yield
        finally:
            torch.load = real_load
            torch.cuda.is_available = real_available  # type: ignore[assignment]
            if real_utils is not None:
                sys.modules["protein_mpnn_utils"] = real_utils
            else:
                sys.modules.pop("protein_mpnn_utils", None)

    def run(self, args: types.SimpleNamespace, captured: io.StringIO) -> None:
        """Run upstream ``main(args)``, collecting what it prints in *captured*."""
        with self._patched(), contextlib.redirect_stdout(captured):
            self._main(args)


class _DesignEngine:
    """One resident model, one worker thread, bounded waits.

    All GPU work runs on a single worker thread, so there is exactly one model instance and
    requests are served one at a time in arrival order. ``submit`` returns a future; the caller
    waits for it with ``REQUEST_TIMEOUT_SECONDS``.

    Timeout semantics. A thread cannot be killed. When the wait
    runs out: a request that has not started yet is cancelled and never runs; a request that is
    already running keeps the worker (and the GPU) until it ends on its own, the caller gets
    ``503`` at once, and every new request is answered ``503`` immediately ("still busy") until
    that run ends. Only a restart of the container stops a run that never ends.
    """

    def __init__(self, runner: _ResidentRunner) -> None:
        self._runner = runner
        self._executor = concurrent.futures.ThreadPoolExecutor(max_workers=1, thread_name_prefix="proteinmpnn")
        self._abandoned: set[concurrent.futures.Future[Any]] = set()

    def _design(self, args: types.SimpleNamespace) -> "PredictResponse":
        started = time.perf_counter()
        try:
            with tempfile.TemporaryDirectory() as out_dir:
                args.out_folder = out_dir
                captured = io.StringIO()
                try:
                    self._runner.run(args, captured)
                except Exception as exc:
                    # Output stays in the server log; it can hold paths and tracebacks.
                    tail = "\n".join(captured.getvalue().splitlines()[-OUTPUT_TAIL_LINES:])
                    logger.exception("protein_mpnn_run.main failed (%s); last output:\n%s", type(exc).__name__, tail)
                    self._release_gpu_memory()
                    raise DesignError("Sequence design failed") from None
                response = _parse_output(out_dir)
        except DesignError:
            raise
        except Exception:
            logger.exception("Could not read the ProteinMPNN output")
            raise DesignError("Sequence design failed") from None
        if not response.sequences:
            logger.error("protein_mpnn_run.main produced no designed sequences")
            raise DesignError("Sequence design produced no output")
        logger.info(
            "designed %d sequence(s) with %s in %.3f s",
            len(response.sequences),
            args.model_name,
            time.perf_counter() - started,
        )
        return response

    def _release_gpu_memory(self) -> None:
        if self._runner.device == "cpu":
            return
        try:
            import torch  # noqa: PLC0415

            torch.cuda.empty_cache()
        except Exception:  # noqa: BLE001 -- best effort after a failed run
            logger.debug("empty_cache failed", exc_info=True)

    def warm_up(self) -> None:
        """Load every model and run one design with each (scoring forward + sampling).

        Raises RuntimeError on any failure. Runs on the worker thread, so the GPU context,
        kernels and caches that later requests use are the ones created here.
        """
        started = time.perf_counter()
        with tempfile.TemporaryDirectory() as tmp:
            pdb = Path(tmp) / "warmup.pdb"
            pdb.write_text(_warmup_pdb_text())

            def work() -> None:
                self._runner.load()
                for name in ALLOWED_MODEL_NAMES:
                    t0 = time.perf_counter()
                    self._design(
                        _run_args(pdb_path=str(pdb), out_dir="", num_seq=2, sampling_temp=0.1, model_name=name, seed=37)
                    )
                    logger.info("warm-up %s done in %.2f s", name, time.perf_counter() - t0)

            try:
                self._executor.submit(work).result()
            except Exception as exc:
                raise RuntimeError(f"Model warm-up failed on device {self._runner.device}: {exc}") from exc
        logger.info(
            "ProteinMPNN resident model ready on %s after warm-up of %d weight files in %.2f s%s",
            self._runner.device,
            len(ALLOWED_MODEL_NAMES),
            time.perf_counter() - started,
            self._memory_note(),
        )

    def _memory_note(self) -> str:
        """``; accelerator memory: X MiB allocated, Y MiB reserved`` for the ready log line ('' on the CPU)."""
        if self._runner.device == "cpu":
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

    def design(self, args: types.SimpleNamespace, timeout: float = REQUEST_TIMEOUT_SECONDS) -> "PredictResponse":
        """Run one design; ``ServiceUnavailable`` (503) if it cannot finish within *timeout* seconds."""
        if self._abandoned:
            raise ServiceUnavailable("A previous request that timed out is still running on the GPU; try again later")
        future = self._executor.submit(self._design, args)
        try:
            return future.result(timeout=timeout)
        except concurrent.futures.TimeoutError:
            if future.cancel():
                logger.error("request waited %ss in the queue and was cancelled", timeout)
                raise ServiceUnavailable(
                    f"Sequence design did not start within {timeout:g} seconds (service busy)"
                ) from None
            self._abandoned.add(future)
            future.add_done_callback(self._abandoned.discard)
            logger.error("design exceeded %ss; it cannot be interrupted and keeps running until it ends", timeout)
            raise ServiceUnavailable(f"Sequence design exceeded {timeout:g} seconds") from None
        except DesignError as exc:
            raise BentoMLException(str(exc)) from None


def _resolve_pdb_path(raw: str) -> Path:
    """Resolve *raw* and require an existing file under an allowed root.

    Raises ``InvalidArgument`` (HTTP 400). Containment is checked before
    existence, with one shared message, so callers cannot probe the filesystem.
    """
    try:
        resolved = Path(raw).resolve()
    except (OSError, RuntimeError, ValueError):
        raise InvalidArgument("pdb_path is not a usable path") from None
    roots = [root.resolve() for root in (INPUT_ROOT, FIXTURE_ROOT)]
    if not any(resolved.is_relative_to(root) for root in roots) or not resolved.is_file():
        raise InvalidArgument(f"pdb_path must be an existing file under {INPUT_ROOT.resolve()}")
    return resolved


class BentoArgs(BaseModel):
    """Template arguments forwarded by AIM through ``bentoml serve --arg``.

    Deliberately no default: ``aim-runtime`` injects ``--arg port=<AIM_PORT>``
    and that is the only source of the bound port.

    ``accelerator_count`` is the profile's ``metadata.accelerator_count``, which
    aim-runtime forwards as ``--arg accelerator_count=...`` to every BentoML AIM whose
    argument model declares the field. It sets the GPU resource request and the number
    of data-parallel workers (one resident model per GPU). Single-GPU when absent.
    """

    port: int
    accelerator_count: int = Field(default=1, ge=1)


class PredictRequest(BaseModel):
    """Request body for /v1/inference."""

    pdb_path: str = Field(
        ..., description="Absolute path to the input PDB file inside the container, under AIM_INPUT_PATH."
    )
    num_seq_per_target: int = Field(default=4, ge=1, le=64, description="Number of sequences to design per chain.")
    sampling_temp: float = Field(
        default=0.1, gt=0.0, le=1.0, description="Sampling temperature (diversity); must be greater than 0."
    )
    model_name: str = Field(
        default="v_48_020", description="ProteinMPNN model variant: v_48_002, v_48_010, v_48_020 or v_48_030."
    )
    seed: int = Field(
        default=37,
        ge=0,
        le=2**32 - 1,
        description="Random seed, 0 to 2**32 - 1 (0 = random per call); out-of-range values are rejected with 400.",
    )


class PredictResponse(BaseModel):
    """Response body for /v1/inference."""

    sequences: list[str]
    scores: list[float]
    seq_recovery: list[float] = Field(default_factory=list)
    # default_factory, not a NaN literal default: a NaN default is written into the OpenAPI schema
    # and makes GET /docs.json fail (JSON has no NaN). The value in responses is unchanged.
    native_score: float = Field(default_factory=lambda: float("nan"))


args = bentoml.use_arguments(BentoArgs)

#: Port this service binds (the forwarded ``--arg port``); the /health handler
#: calls BentoML's own /healthz on it.
PORT = args.port


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
    if _ACTIVE_DEVICE == "cpu":
        return None
    try:
        import torch  # noqa: PLC0415 (heavy import, only needed in the serving process)

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

    Behaviour of BentoML 1.x: ``os._exit(1)``
    ends only the worker process. BentoML's supervisor restarts it about every
    10 s, so the failure message repeats, the container keeps running and never
    becomes ready. The container does NOT exit. Making the container exit is a
    runtime/entrypoint concern (decided by the design team), not this service's;
    operators must use a readiness probe and watch the logs.
    """
    logger.critical("ProteinMPNN startup failed: %s", message)
    print(f"ProteinMPNN startup failed: {message}", file=sys.stderr, flush=True)
    os._exit(1)


# BentoML's default request timeout is 60 s, which would answer 504 long before the
# service's own limit below.  The traffic timeout must exceed REQUEST_TIMEOUT_SECONDS so
# that a slow run ends in the service's own 503 instead.
@bentoml.service(
    name="proteinmpnn",
    resources={"gpu": args.accelerator_count},
    # One data-parallel worker per GPU, each with its own resident model: more GPUs raise
    # throughput by serving requests concurrently.
    workers=args.accelerator_count,
    http={"port": args.port},
    traffic={"timeout": REQUEST_TIMEOUT_SECONDS + 60},
)
@bentoml.asgi_app(canonical_app, path="/")
class ProteinMPNNService:
    """BentoML service wrapping ProteinMPNN protein sequence design.

    The model is loaded and warmed up once in ``__init__`` (BentoML opens the port only
    after it returns, so ``/health`` answers only once the service is ready). Each request
    runs upstream ``protein_mpnn_run.main`` in-process, with a temporary output directory;
    results are read back from the emitted FASTA and score files.
    """

    def __init__(self) -> None:
        global _ACTIVE_DEVICE  # noqa: PLW0603 -- read by GET /v1/models
        # Fail at startup, with a clear message, instead of letting upstream
        # silently fall back to CPU (it selects "cpu" when CUDA/HIP is absent).
        import torch  # noqa: PLC0415 (heavy import, only needed in the serving process)

        try:
            # Before the first GPU call: the runtime reads the visibility variables once.
            pinned = _pin_worker_device(bentoml.server_context.worker_index or 1, args.accelerator_count)
            device = _resolve_device(torch, pinned, args.accelerator_count)
            _preflight_weights()
        except RuntimeError as exc:
            _fail_startup(str(exc))
        _ACTIVE_DEVICE = device
        logger.info("ProteinMPNN device: %s", device)
        self._engine = _DesignEngine(_ResidentRunner(device))
        try:
            self._engine.warm_up()
        except RuntimeError as exc:
            _fail_startup(str(exc))

    @bentoml.api(route="/v1/inference")
    def inference(self, request: PredictRequest) -> PredictResponse:
        """Design sequences for the given PDB backbone.

        Runs upstream ``protein_mpnn_run.main`` in-process on the resident model and parses
        its output. Errors are raised as BentoML exceptions so the HTTP status reflects the
        failure: 400 for bad input, 500 for a failed run, 503 on timeout or when busy.
        """
        if request.model_name not in ALLOWED_MODEL_NAMES:
            raise InvalidArgument(f"model_name must be one of: {', '.join(ALLOWED_MODEL_NAMES)}")
        pdb_path = _resolve_pdb_path(request.pdb_path)
        run_args = _run_args(
            pdb_path=str(pdb_path),
            out_dir="",
            num_seq=request.num_seq_per_target,
            sampling_temp=request.sampling_temp,
            model_name=request.model_name,
            seed=request.seed,
        )
        return self._engine.design(run_args)


def _read_designed_sequences(fasta_path: str) -> list[str]:
    """Return the designed sequences of one ProteinMPNN FASTA file.

    Upstream writes the native (input) sequence as the first record, with a
    header like ``>1ubq, score=..., fixed_chains=...``, followed by one record
    per sample with a header starting ``>T=<temp>, sample=<n>, ...``. Only the
    sample records are returned, so they align one-to-one with the entries of
    the ``score`` array in the matching ``scores/<name>.npz``.
    """
    seqs: list[str] = []
    is_sample = False
    current: list[str] = []

    def flush() -> None:
        if is_sample and current:
            seqs.append("".join(current))

    with open(fasta_path) as fh:
        for line in fh:
            line = line.strip()
            if line.startswith(">"):
                flush()
                current = []
                is_sample = line[1:].startswith("T=")
            elif line:
                current.append(line)
    flush()
    return seqs


def _header_number(value: str | None) -> float:
    """Parse a ``key=value`` number from a FASTA header; NaN when it is absent or not a number."""
    try:
        return float(value) if value is not None else float("nan")
    except ValueError:
        return float("nan")


def _read_header_values(fasta_path: str) -> tuple[float, list[float]]:
    """Return the native ``score`` and the per-sample ``seq_recovery`` of one ProteinMPNN FASTA file.

    The native record header looks like ``>1ubq, score=..., global_score=...``; each
    sample record header like ``>T=0.1, sample=1, score=..., seq_recovery=...``. The
    sample values align one-to-one with ``_read_designed_sequences``. A value that is
    missing or not a number is NaN.
    """
    native_score = float("nan")
    seen_native = False
    recoveries: list[float] = []
    with open(fasta_path) as fh:
        for line in fh:
            line = line.strip()
            if not line.startswith(">"):
                continue
            header = line[1:]
            fields = dict(part.strip().split("=", 1) for part in header.split(",") if "=" in part)
            if header.startswith("T="):
                recoveries.append(_header_number(fields.get("seq_recovery")))
            elif not seen_native:
                seen_native = True
                native_score = _header_number(fields.get("score"))
    return native_score, recoveries


def _parse_output(out_dir: str) -> PredictResponse:
    """Read sequences and scores from the ProteinMPNN output directory.

    ProteinMPNN (``--save_score 1``) writes, per input structure ``<name>``,
    ``seqs/<name>.fa`` and ``scores/<name>.npz`` whose ``score`` key holds one
    value per designed sample (lower is better). Files are paired by name and
    aggregated in sorted order; the native-sequence record is skipped for the
    sequences, and its ``score`` is returned as ``native_score`` (first file only).
    ``seq_recovery`` comes from the sample record headers.
    """
    import numpy as np  # lazy import; numpy ships with the ROCm pytorch base image

    seqs: list[str] = []
    scores: list[float] = []
    recoveries: list[float] = []
    native_score = float("nan")

    for index, fasta_path in enumerate(sorted(glob.glob(os.path.join(out_dir, "seqs", "*.fa")))):
        file_seqs = _read_designed_sequences(fasta_path)
        file_native_score, file_recoveries = _read_header_values(fasta_path)
        if index == 0:
            native_score = file_native_score
        file_recoveries = (file_recoveries + [float("nan")] * len(file_seqs))[: len(file_seqs)]
        score_path = os.path.join(out_dir, "scores", Path(fasta_path).stem + ".npz")
        file_scores: list[float] = []
        try:
            with np.load(score_path) as data:
                file_scores = [float(v) for v in np.asarray(data["score"]).flatten()]
        except (OSError, KeyError, ValueError) as exc:
            logger.warning("Could not load score file %s: %s", score_path, exc)
        if len(file_scores) != len(file_seqs):
            logger.warning(
                "%s: %d sequences but %d scores; padding or truncating to match",
                fasta_path,
                len(file_seqs),
                len(file_scores),
            )
            file_scores = (file_scores + [float("nan")] * len(file_seqs))[: len(file_seqs)]
        seqs.extend(file_seqs)
        scores.extend(file_scores)
        recoveries.extend(file_recoveries)

    return PredictResponse(sequences=seqs, scores=scores, seq_recovery=recoveries, native_score=native_score)
