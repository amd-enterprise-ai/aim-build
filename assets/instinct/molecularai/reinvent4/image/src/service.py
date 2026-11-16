# Copyright © Advanced Micro Devices, Inc., or its affiliates.
#
# SPDX-License-Identifier: MIT

"""
BentoML service for REINVENT4 molecular de novo generation.

Runs REINVENT4's sampling in-process on a resident model (loaded and warmed
up once, before the service accepts connections; the host-side RDKit validation of the sampled SMILES runs in a
pool of worker processes, see ``smiles_validation_pool``) and exposes a single model
endpoint:

    POST /v1/inference  ->  {"smiles": [...], "num_generated": N, "num_requested": R,
                             "num_invalid": I, "num_duplicates": D}

plus the canonical GET /health and GET /v1/models, served by a small mounted
ASGI app. BentoML's own /healthz, /livez, /readyz and /metrics are unchanged.
The former predict route is removed (no alias).

Request body (all fields optional, sensible defaults applied):
    {
      "num_smiles":         int   = 100,    # how many SMILES to sample
      "unique_molecules":   bool  = true    # deduplicate generated molecules
    }

Response body:
    {
      "smiles":        list[str],  # generated SMILES strings (one per line)
      "num_generated": int,        # len(smiles) — convenience field for harness
      "num_requested": int,        # the requested num_smiles
      "num_invalid":   int,        # sampled strings RDKit could not parse or sanitise
      "num_duplicates": int        # valid sampled strings that repeat an earlier canonical SMILES
    }

How AIM launches it
-------------------
The BentoML engine entry in ``engines.yaml`` runs::

    python -m bentoml serve --working-dir /workspace/model

``bentoml serve`` reads ``bentofile.yaml`` from ``/workspace/model`` and loads
the service class declared there (``service: "service:REINVENT4Service"``).
aim-runtime resolves ``AIM_PORT`` and forwards it as ``--arg port=<AIM_PORT>``;
the service reads it via ``bentoml.use_arguments`` and binds it with
``http={"port": ...}``.
"""

from __future__ import annotations

import concurrent.futures
import hashlib
import json
import logging
import os
import re
import sys
import tempfile
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, NoReturn

import bentoml
import httpx
import smiles_validation_pool
import torch
from bentoml.exceptions import BadInput, InternalServerError, ServiceUnavailable
from pydantic import BaseModel, Field
from starlette.applications import Starlette
from starlette.requests import Request
from starlette.responses import JSONResponse
from starlette.routing import Route

# ---------------------------------------------------------------------------
# Request / response schemas
# ---------------------------------------------------------------------------

logger = logging.getLogger(__name__)


def _configure_logging(target: logging.Logger) -> None:
    """Send this module's INFO lines to stderr, unbuffered, so ``docker logs`` shows them.

    BentoML's worker logging does not show application loggers, so the pre-flight
    lines (prior path, download URL, "verified") were invisible. Idempotent: adds
    one handler only.
    """
    if any(getattr(h, "_reinvent4_stderr", False) for h in target.handlers):
        return
    handler = logging.StreamHandler(sys.stderr)  # StreamHandler flushes after every record
    handler.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] [reinvent4] %(message)s"))
    handler._reinvent4_stderr = True  # type: ignore[attr-defined]
    target.addHandler(handler)
    target.setLevel(logging.INFO)
    target.propagate = False


_configure_logging(logger)

# Weights live under AIM_CACHE_PATH, never in the image. Priors are pickled
# torch checkpoints: loading one executes code, so a file is only ever handed to
# reinvent if it is a pinned built-in (size+md5 verified before every hand-off,
# result cached by path/size/mtime/inode/ctime) or an operator-declared staged prior (sha256 verified, see REINVENT4_EXTRA_PRIORS).
PRIOR_FILENAME = "reinvent.prior"
PRIOR_URL = "https://zenodo.org/records/20701824/files/reinvent_pubchem.prior?download=1"
# Zenodo record 20701824 metadata for reinvent_pubchem.prior.
PRIOR_MD5 = "75052d5e93b2470e9c6684d369704f5f"
PRIOR_SIZE = 23528840
DEFAULT_PRIOR = "reinvent_pubchem"
# Env vars: service-level default prior, and staged priors as {"name": "sha256"}.
PRIOR_ENV = "REINVENT4_PRIOR"
EXTRA_PRIORS_ENV = "REINVENT4_EXTRA_PRIORS"
# Names are identifiers, never paths: no separators, no "..", no leading dot.
_PRIOR_NAME_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9_-]{0,63}")
_SHA256_RE = re.compile(r"[0-9a-f]{64}")
_DOWNLOAD_TIMEOUT_S = 600.0
_DOWNLOAD_LOCK = threading.Lock()
# (path, size, mtime_ns, inode, ctime_ns) of built-in prior files that already passed size+md5.
# ctime cannot be set from user space and a replaced file has a new inode, so a tamper that
# restores size and mtime still invalidates the entry.
_VERIFIED_BUILTINS = set()  # type: set[tuple[str, int, int, int, int]]
# A key is only conclusive once the file's ctime is older than the timestamp granularity: on a
# coarse-clock or inode-recycling filesystem (tmpfs, overlayfs) a same-size rewrite with a
# restored mtime, done within the same tick as the verification, would produce an identical key.
# Files changed more recently than this are re-hashed on every call (the "racy git" rule).
_RACY_WINDOW_NS = 2_000_000_000


def _cache_key(path: Path, st: os.stat_result) -> tuple[str, int, int, int, int]:
    """Cache key of a verified built-in prior file."""
    return (str(path), st.st_size, st.st_mtime_ns, st.st_ino, st.st_ctime_ns)


def _remember_verified(path: Path, st: os.stat_result) -> None:
    """Record a verified built-in file, unless its key is not yet conclusive (see _RACY_WINDOW_NS)."""
    if time.time_ns() - max(st.st_ctime_ns, st.st_mtime_ns) >= _RACY_WINDOW_NS:
        _VERIFIED_BUILTINS.add(_cache_key(path, st))


class PriorIntegrityError(RuntimeError):
    """A prior file does not match its pinned checksum: permanent, no retry fixes it (HTTP 500)."""


def _cache_root() -> Path:
    """Cache root: ``AIM_CACHE_PATH`` (default ``/workspace/model-cache``)."""
    return Path(os.environ.get("AIM_CACHE_PATH", "/workspace/model-cache")).resolve()


def _priors_dir() -> Path:
    """Directory holding all priors: ``$AIM_CACHE_PATH/reinvent4/priors``."""
    return _cache_root() / "reinvent4" / "priors"


def _prior_path() -> Path:
    """Location of the default built-in prior: ``<priors dir>/reinvent.prior``."""
    return _priors_dir() / PRIOR_FILENAME


def _md5(path: Path) -> str:
    digest = hashlib.md5(usedforsecurity=False)
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _builtin_priors() -> dict[str, dict]:
    """Built-in allow-list: public priors valid for the ``sampling`` run mode.

    Only ``reinvent_pubchem`` (checkpoint model_type "Reinvent") is clearly
    compatible; the other files of Zenodo record 20701824 are Lib/Link/Pep-Invent
    and Mol2Mol models, which need other run modes and inputs.
    """
    return {
        DEFAULT_PRIOR: {
            "filename": PRIOR_FILENAME,
            "url": PRIOR_URL,
            "md5": PRIOR_MD5,
            "size": PRIOR_SIZE,
        }
    }


def _staged_priors() -> dict[str, str]:
    """Parse ``REINVENT4_EXTRA_PRIORS`` ({name: sha256}); raise RuntimeError if invalid."""
    raw = os.environ.get(EXTRA_PRIORS_ENV, "").strip()
    if not raw:
        return {}
    try:
        declared = json.loads(raw)
    except ValueError as exc:
        raise RuntimeError(f"{EXTRA_PRIORS_ENV} is not valid JSON: {exc}") from exc
    if not isinstance(declared, dict):
        raise RuntimeError(f'{EXTRA_PRIORS_ENV} must be a JSON object {{"name": "sha256"}}')
    builtin_files = {e["filename"] for e in _builtin_priors().values()}
    for name, digest in declared.items():
        if not isinstance(name, str) or not _PRIOR_NAME_RE.fullmatch(name):
            raise RuntimeError(f"{EXTRA_PRIORS_ENV}: invalid prior name {name!r} (allowed: {_PRIOR_NAME_RE.pattern})")
        if name in _builtin_priors():
            raise RuntimeError(f"{EXTRA_PRIORS_ENV}: {name!r} collides with a built-in prior name")
        if f"{name}.prior" in builtin_files:
            raise RuntimeError(f"{EXTRA_PRIORS_ENV}: {name!r} would use {name}.prior, the file of a built-in prior")
        if not isinstance(digest, str) or not _SHA256_RE.fullmatch(digest.lower()):
            raise RuntimeError(f"{EXTRA_PRIORS_ENV}: {name!r} needs a 64-hex-digit sha256")
    return {name: digest.lower() for name, digest in declared.items()}


def _prior_file(filename: str) -> Path:
    """Resolve ``filename`` inside the priors directory; refuse anything that escapes it."""
    root = _priors_dir().resolve()
    path = (root / filename).resolve()
    if path.parent != root:
        raise RuntimeError(f"Prior file {filename!r} resolves outside {root}")
    return path


def _verify_staged(name: str, sha256: str) -> Path:
    """Return the staged prior's path after verifying its sha256 (RuntimeError otherwise)."""
    path = _prior_file(f"{name}.prior")
    if not path.is_file():
        raise RuntimeError(f"Staged prior {name!r} declared in {EXTRA_PRIORS_ENV} but {path} does not exist")
    actual = _sha256(path)
    if actual != sha256:
        raise PriorIntegrityError(
            f"Staged prior file {path}: sha256 mismatch (expected {sha256}, actual {actual}); refusing to load"
        )
    return path


def _verify_builtin(path: Path, entry: dict) -> None:
    """Verify an existing built-in prior's size and md5 (PriorIntegrityError otherwise).

    A passing result is cached by (path, size, mtime_ns, inode, ctime_ns), so the 23 MB file is only
    re-hashed when it changes on disk (and always while the file is younger than _RACY_WINDOW_NS).
    """
    st = path.stat()
    key = _cache_key(path, st)
    if key in _VERIFIED_BUILTINS:
        return
    if st.st_size != entry["size"]:
        raise PriorIntegrityError(f"Prior file {path}: size {st.st_size} != expected {entry['size']}; refusing to load")
    actual = _md5(path)
    if actual != entry["md5"]:
        raise PriorIntegrityError(
            f"Prior file {path}: md5 mismatch (expected {entry['md5']}, actual {actual}); refusing to load"
        )
    _remember_verified(path, st)


def _allowed_names() -> list[str]:
    """All selectable prior names (built-in first, then staged)."""
    return list(_builtin_priors()) + sorted(_staged_priors())


def _default_prior_name() -> str:
    """Service-level default: ``REINVENT4_PRIOR`` or ``reinvent_pubchem``."""
    return os.environ.get(PRIOR_ENV, "").strip() or DEFAULT_PRIOR


def _ensure_prior(entry: dict | None = None) -> Path:
    """Return the prior path, downloading it into the cache first if missing.

    ``entry`` is a built-in registry entry (default: reinvent_pubchem). The
    download goes to a temporary file in the same directory, is verified (size
    and md5), then renamed atomically; concurrent callers are serialised by a
    lock. Raises RuntimeError naming the path and the URL on any failure.
    """
    entry = entry or _builtin_priors()[DEFAULT_PRIOR]
    dest = _prior_file(entry["filename"])
    url = entry["url"]
    with _DOWNLOAD_LOCK:
        if dest.is_file():
            _verify_builtin(dest, entry)
            logger.info("REINVENT4 prior found at %s and verified (no download needed)", dest)
            return dest
        logger.info("REINVENT4 prior missing at %s; downloading from %s (no token needed)", dest, url)
        tmp_name = ""
        try:
            dest.parent.mkdir(parents=True, exist_ok=True)
            fd, tmp_name = tempfile.mkstemp(dir=dest.parent, prefix=f".{dest.name}.", suffix=".part")
            with (
                os.fdopen(fd, "wb") as out,
                httpx.stream("GET", url, follow_redirects=True, timeout=_DOWNLOAD_TIMEOUT_S) as resp,
            ):
                resp.raise_for_status()
                for chunk in resp.iter_bytes(1 << 20):
                    out.write(chunk)
            tmp = Path(tmp_name)
            if tmp.stat().st_size != entry["size"]:
                raise RuntimeError(f"size {tmp.stat().st_size} != expected {entry['size']}")
            if _md5(tmp) != entry["md5"]:
                raise RuntimeError("md5 checksum mismatch")
            os.replace(tmp, dest)
            # Least privilege: mkstemp created the file 0600 (owner = the service user, the only
            # reader that needs it). The mode is pinned here rather than widened for operators.
            os.chmod(dest, 0o600)
            _remember_verified(dest, dest.stat())
        except (OSError, httpx.HTTPError, RuntimeError) as exc:
            if tmp_name:
                Path(tmp_name).unlink(missing_ok=True)
            raise RuntimeError(
                f"Could not fetch the REINVENT4 prior to {dest} from {url}: {exc}. "
                f"Pre-stage the file at that path (AIM_CACHE_PATH={_cache_root()})."
            ) from exc
    logger.info("REINVENT4 prior downloaded to %s", dest)
    return dest


class UnknownPriorError(ValueError):
    """The requested prior name is not in the registry (HTTP 400)."""


def _resolve_prior(name: str | None) -> Path:
    """Map a prior NAME (never a path) to a verified file; None selects the default.

    Raises UnknownPriorError for an unknown/invalid name (message lists the
    allowed names) and RuntimeError when the file cannot be provided or verified.
    """
    name = name if name is not None else _default_prior_name()
    allowed = _allowed_names()
    if not isinstance(name, str) or not _PRIOR_NAME_RE.fullmatch(name) or name not in allowed:
        raise UnknownPriorError(f"Unknown prior {str(name)[:80]!r}. Allowed priors: {', '.join(allowed)}")
    builtin = _builtin_priors()
    if name in builtin:
        return _ensure_prior(builtin[name])
    # Re-hash on every use so a file swapped after startup is never loaded.
    return _verify_staged(name, _staged_priors()[name])


def _expected_checksum(name: str) -> tuple[str, str, int | None]:
    """(algorithm, hex digest, size or None) the registered prior ``name`` must have."""
    builtin = _builtin_priors().get(name)
    if builtin is not None:
        return "md5", builtin["md5"], builtin["size"]
    return "sha256", _staged_priors()[name], None


def _snapshot_prior(name: str, src: Path, dest_dir: Path) -> Path:
    """Copy prior ``name`` from ``src`` into the private ``dest_dir`` and verify the COPY.

    Priors are pickled checkpoints, so a file that changes between its verification and the
    moment torch opens it would run unverified code. The cache directory is a mounted volume
    that others may write to, so ``src`` is opened once, streamed into a 0600 file in
    ``dest_dir`` (a per-request 0700 directory) while hashing, and the digest of exactly those
    bytes is compared with the registered one. The caller hands only the returned path to
    REINVENT4. Raises PriorIntegrityError on a mismatch, OSError if ``src`` cannot be read.
    """
    algo, expected, size = _expected_checksum(name)
    digest = hashlib.md5(usedforsecurity=False) if algo == "md5" else hashlib.sha256()
    dest = dest_dir / f"{name}.prior"
    total = 0
    src_fd = os.open(src, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0))
    dest_fd = os.open(dest, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(src_fd, "rb") as fin, os.fdopen(dest_fd, "wb") as fout:
        for chunk in iter(lambda: fin.read(1 << 20), b""):
            digest.update(chunk)
            total += len(chunk)
            fout.write(chunk)
    if (size is not None and total != size) or digest.hexdigest() != expected:
        dest.unlink(missing_ok=True)
        raise PriorIntegrityError(
            f"Prior file {src}: {algo} mismatch after snapshot (expected {expected}, size {total}); refusing to load"
        )
    return dest


def _preflight_priors() -> None:
    """Startup check: validate config, verify staged priors, fetch the default prior.

    Raises RuntimeError (the caller stops startup) naming the file and, for a
    sha256 mismatch, the expected and actual hash.
    """
    staged = _staged_priors()
    for name, digest in staged.items():
        logger.info("REINVENT4 staged prior %r: verifying %s", name, _prior_file(f"{name}.prior"))
        _verify_staged(name, digest)
    default = _default_prior_name()
    logger.info("REINVENT4 default prior %r; priors dir %s; selectable: %s", default, _priors_dir(), _allowed_names())
    try:
        _resolve_prior(default)
    except UnknownPriorError as exc:
        raise RuntimeError(f"{PRIOR_ENV}: {exc}") from exc


def _list_priors() -> list[dict]:
    """Registry view for /v1/models; never raises."""
    items = [
        {
            "name": name,
            "source": "built-in",
            "checksum": f"md5:{e['md5']}",
            "present": _prior_file(e["filename"]).is_file(),
        }
        for name, e in _builtin_priors().items()
    ]
    try:
        staged = _staged_priors()
    except RuntimeError:
        staged = {}
    items += [
        {
            "name": n,
            "source": "staged",
            "checksum": f"sha256:{d}",
            "present": _prior_file(f"{n}.prior").is_file(),
        }
        for n, d in sorted(staged.items())
    ]
    return items


SAMPLE_BATCH_SIZE_ENV = "REINVENT4_SAMPLE_BATCH_SIZE"
MAX_SAMPLE_BATCH_SIZE = 65536


def resolve_sample_batch_size(raw: str | None, default: int) -> int:
    """Rows sampled per model call: ``REINVENT4_SAMPLE_BATCH_SIZE`` (1 to 65536), else upstream's value."""
    text = (raw or "").strip()
    if not text:
        return default
    try:
        value = int(text)
    except ValueError:
        value = 0
    if not 1 <= value <= MAX_SAMPLE_BATCH_SIZE:
        raise RuntimeError(f"{SAMPLE_BATCH_SIZE_ENV}={raw!r}: expected an integer from 1 to {MAX_SAMPLE_BATCH_SIZE}")
    return value


DEVICE_ENV = "REINVENT4_DEVICE"
DEFAULT_DEVICE = "cuda:0"
_DEVICE_RE = re.compile(r"(cpu|cuda(:[0-9]{1,2})?)")


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


def _activate_device(device: str) -> None:
    """Make ``cuda:<k>`` this worker's current CUDA device, so a bare ``"cuda"`` or ``.cuda()`` lands on it."""
    if device.startswith("cuda:"):
        torch.cuda.set_device(int(device.partition(":")[2]))


def _resolve_device(worker_index: int = 1, accelerator_count: int = 1) -> str:
    """Return the torch device the resident model must use, or raise RuntimeError.

    Default is this worker's own GPU, ``cuda:<k-1>`` for BentoML worker ``k`` (``cuda:0``
    for a single accelerator; fewer visible GPUs than ``accelerator_count`` raises; ROCm builds of torch also use the ``cuda`` name). REINVENT4
    itself falls back to the CPU when no device is given, which gives slow but
    "successful" runs, so the device is always set explicitly. ``REINVENT4_DEVICE`` overrides it: ``cpu``, ``cuda`` or
    ``cuda:<n>``. Any other value, a GPU device with no visible GPU, or an index
    beyond the visible GPUs raises, so a mistake fails loudly and never runs on
    the wrong device. ``cpu`` is only used when asked for, with a warning.
    """
    configured = os.environ.get(DEVICE_ENV, "").strip().lower()
    raw = configured or DEFAULT_DEVICE
    if not _DEVICE_RE.fullmatch(raw):
        raise RuntimeError(f"{DEVICE_ENV}={raw!r} is not valid; use 'cpu', 'cuda' or 'cuda:<index>'.")
    if raw == "cpu":
        logger.warning("%s=cpu: sampling runs on the CPU, which is slow", DEVICE_ENV)
        return raw
    if not torch.cuda.is_available():
        raise RuntimeError(
            f"No GPU found. REINVENT4 requires a CUDA/ROCm GPU (set {DEVICE_ENV}=cpu to run on the CPU on purpose)."
        )
    index = int(raw.partition(":")[2] or 0)
    if not configured:
        index = _worker_device_index(worker_index, torch.cuda.device_count(), accelerator_count)
    if index >= torch.cuda.device_count():
        raise RuntimeError(f"{DEVICE_ENV}={raw!r}: only {torch.cuda.device_count()} GPU(s) visible.")
    return f"cuda:{index}"


class SamplingRequest(BaseModel):
    """Parameters for a REINVENT4 sampling run.

    All fields are optional; defaults produce a straightforward sampling call
    with the default prior model (``prior`` selects another registered one by name).
    """

    num_smiles: int = Field(default=100, ge=1, le=10_000, description="Number of SMILES to generate")
    unique_molecules: bool = Field(default=True, description="Remove duplicate molecules from output")
    prior: str | None = Field(
        default=None,
        max_length=64,
        description="Prior NAME from GET /v1/models (never a path); default: the service default prior",
    )


class SamplingResponse(BaseModel):
    """Result of a REINVENT4 sampling run."""

    smiles: list[str]
    num_generated: int
    num_requested: int
    num_invalid: int
    num_duplicates: int


# ---------------------------------------------------------------------------
# Resident sampling engine
# ---------------------------------------------------------------------------

#: Upper bound, in seconds, on how long one request waits for its result (queue wait plus run).
#: A run is not a subprocess any more, so it cannot be killed: see ``_SamplingEngine``.
REQUEST_TIMEOUT_SECONDS = 600

#: Molecules sampled by the start-up warm-up (one call, before the port opens).
WARMUP_NUM_SMILES = 128


class SamplingError(Exception):
    """A sampling run failed or produced nothing; the detail is only in the server log."""


@dataclass(frozen=True)
class _Outcome:
    """What one sampling call returns: the SMILES and the validation counts of the whole sampled batch."""

    smiles: list[str]
    num_invalid: int
    num_duplicates: int


@dataclass(frozen=True)
class _Job:
    """One sampling call (``seed`` is for tests and benchmarks only; the HTTP API has no seed)."""

    prior_path: Path
    prior_name: str
    num_smiles: int
    unique_molecules: bool
    seed: int | None = None


class _ResidentSampler:
    """Loads REINVENT4 priors once and samples from them in this process.

    Mirrors what ``python -m reinvent`` does for a ``sampling`` run, without the process start
    and without re-reading the checkpoint on every call (``reinvent.runmodes.samplers.run_sampling``
    reads it on every call, so it cannot keep a model resident; its steps are done here instead):
    ``create_adapter`` once per prior file, then per request ``setup_sampler`` + ``sampler.sample``
    under ``torch.no_grad()`` and, for ``unique_molecules``, upstream's own ``filter_valid``.
    Every method runs on the engine's single worker thread.
    """

    def __init__(self, device: str, validation_pool: Any = None) -> None:
        self.device_name = device
        self._validation_pool = validation_pool  # smiles_validation_pool.ValidationPool, or None: validate in-process
        self._models: dict[Path, tuple[tuple, Any, str]] = {}

    def init_thread(self) -> None:
        """Worker-thread initialiser: default device, RDKit logging (as REINVENT4's ``main``)."""
        import torch  # noqa: PLC0415 (heavy import, only needed in the serving process)
        from rdkit import rdBase  # noqa: PLC0415

        rdBase.DisableLog("rdApp.*")  # sanitisation failures of sampled SMILES are expected
        torch.set_default_device(self.device_name)

    def _load(self, job: _Job) -> tuple[Any, str]:
        """Return (adapter, model_type) of a prior, loading it on first use or after its file changed.

        A prior is a pickled checkpoint, so it is never loaded from the shared cache path: the file
        is copied into a private directory and the copy is verified against the registered checksum
        (``_snapshot_prior``), and only that copy is handed to REINVENT4. A file swapped after the
        service verified it is therefore rejected, not loaded.
        """
        import torch  # noqa: PLC0415
        from reinvent.runmodes import create_adapter  # noqa: PLC0415

        prior_path = job.prior_path
        key = _cache_key(prior_path, prior_path.stat())
        cached = self._models.get(prior_path)
        if cached is not None and cached[0] == key:
            return cached[1], cached[2]
        started = time.perf_counter()
        with tempfile.TemporaryDirectory() as private_dir:
            snapshot = _snapshot_prior(job.prior_name, prior_path, Path(private_dir))
            adapter, _, model_type = create_adapter(str(snapshot), "inference", torch.device(self.device_name))
        if model_type != "Reinvent":
            raise SamplingError(f"prior {prior_path.name} is a {model_type} model; only Reinvent priors are supported")
        self._models[prior_path] = (key, adapter, model_type)
        logger.info("prior %s loaded on %s in %.2f s", prior_path.name, self.device_name, time.perf_counter() - started)
        return adapter, model_type

    def sample(self, job: _Job) -> _Outcome:
        """Sample ``job.num_smiles`` molecules; returns the non-empty SMILES strings in sampling order.

        ``num_invalid`` and ``num_duplicates`` count the states upstream's validation gave the sampled batch, before
        ``filter_valid`` drops them.
        """
        import torch  # noqa: PLC0415
        from reinvent.models.model_factory.sample_batch import SmilesState  # noqa: PLC0415
        from reinvent.runmodes.samplers.run_sampling import filter_valid  # noqa: PLC0415
        from reinvent.runmodes.setup_sampler import setup_sampler  # noqa: PLC0415
        from reinvent.utils import set_seed  # noqa: PLC0415

        adapter, model_type = self._load(job)
        # Upstream's de novo Reinvent sampler has no use for ``temperature`` (only applied to transformer
        # models) or ``randomize_smiles`` (only matters for samplers with SMILES input), so only the batch size is passed.
        params = {"batch_size": job.num_smiles}
        if job.seed is not None:
            set_seed(job.seed)
        with torch.no_grad():
            sampler, _ = setup_sampler(model_type, params, adapter)
            pool = self._validation_pool
            if pool is not None and pool.usable:
                sampled = self._sample_with_pool(sampler, pool)
            else:
                sampled = sampler.sample(None)
        states = list(sampled.states)
        num_invalid = states.count(SmilesState.INVALID)
        num_duplicates = states.count(SmilesState.DUPLICATE)
        if job.unique_molecules:
            sampled = filter_valid(sampled)
        smiles = [smi.strip() for smi in sampled.smilies if smi and smi.strip()]
        return _Outcome(smiles, num_invalid, num_duplicates)

    @staticmethod
    def _sample_with_pool(sampler: Any, pool: Any) -> Any:
        """``ReinventSampler.sample`` (upstream v4.8) with the RDKit validation done by the worker pool.

        The steps are upstream's: sample in batches of ``SAMPLE_BATCH_SIZE`` on the model, build the
        ``SampleBatch``, then parse and validate every sampled string. Only that last step moves to the pool
        (``smiles_validation_pool``, which holds upstream's ``validate_smiles`` and merges duplicates in order), so
        ``smilies`` and ``states`` are what the in-process call would return.
        """
        import numpy as np  # noqa: PLC0415
        from reinvent.models.model_factory.sample_batch import SampleBatch, SmilesState  # noqa: PLC0415
        from reinvent.runmodes.samplers.params import SAMPLE_BATCH_SIZE as UPSTREAM_BATCH_SIZE  # noqa: PLC0415

        SAMPLE_BATCH_SIZE = resolve_sample_batch_size(
            os.environ.get(SAMPLE_BATCH_SIZE_ENV), UPSTREAM_BATCH_SIZE
        )  # noqa: N806
        batch_sizes = [SAMPLE_BATCH_SIZE for _ in range(sampler.batch_size // SAMPLE_BATCH_SIZE)]
        if remainder := sampler.batch_size % SAMPLE_BATCH_SIZE:
            batch_sizes += [remainder]
        sequences = []
        for batch_size in batch_sizes:
            for batch_row in sampler.model.sample(batch_size):
                sequences.append(batch_row)
        sampled = SampleBatch.from_list(sequences)
        sampled.items1 = None
        smilies, states = pool.validate(list(sampled.output), sampler.isomeric)
        sampled.smilies, sampled.states = smilies, np.array([SmilesState(state) for state in states])
        return sampled

    def release_gpu_memory(self) -> None:
        if self.device_name == "cpu":
            return
        try:
            import torch  # noqa: PLC0415

            torch.cuda.empty_cache()
        except Exception:  # noqa: BLE001 -- best effort after a failed run
            logger.debug("empty_cache failed", exc_info=True)

    def memory_note(self) -> str:
        """``; accelerator memory: X MiB allocated, Y MiB reserved`` for the ready log line ('' on the CPU)."""
        if self.device_name == "cpu":
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


class _SamplingEngine:
    """One resident model, one worker thread, bounded waits.

    All sampling runs on a single worker thread, so there is one model instance per prior and
    requests are served one at a time in arrival order (the serialisation the one-process-per-request
    design had too, through BentoML's single worker). ``sample`` waits for the result for at most
    ``REQUEST_TIMEOUT_SECONDS``.

    Timeout semantics. A thread cannot be killed the way the old subprocess was. When the wait runs
    out: a request that has not started yet is cancelled and never runs; a request that is already
    running keeps the worker (and the GPU) until it ends on its own, the caller gets ``503`` at
    once, and every new request is answered ``503`` immediately ("still busy") until that run ends.
    Only a restart of the container stops a run that never ends.
    """

    def __init__(self, sampler: _ResidentSampler) -> None:
        self._sampler = sampler
        self._executor = concurrent.futures.ThreadPoolExecutor(
            max_workers=1, thread_name_prefix="reinvent4", initializer=sampler.init_thread
        )
        self._abandoned: set[concurrent.futures.Future[Any]] = set()

    def _run(self, job: _Job) -> _Outcome:
        started = time.perf_counter()
        try:
            outcome = self._sampler.sample(job)
        except (SamplingError, PriorIntegrityError) as exc:
            logger.error("REINVENT4 sampling refused: %s", exc)
            raise SamplingError("Sampling run failed") from None
        except Exception:
            logger.exception("REINVENT4 sampling failed")
            self._sampler.release_gpu_memory()
            raise SamplingError("Sampling run failed") from None
        if not outcome.smiles:
            logger.error("REINVENT4 sampling returned no SMILES")
            raise SamplingError("Sampling run produced no SMILES")
        logger.info(
            "sampled %d of %d SMILES (%d invalid, %d duplicates) in %.3f s",
            len(outcome.smiles),
            job.num_smiles,
            outcome.num_invalid,
            outcome.num_duplicates,
            time.perf_counter() - started,
        )
        return outcome

    def warm_up(self, prior_path: Path, prior_name: str) -> None:
        """Load the prior and sample once on the worker thread (RuntimeError on failure).

        The accelerator context, kernels and allocator state later requests use are the ones created here.
        """
        started = time.perf_counter()
        job = _Job(prior_path, prior_name, WARMUP_NUM_SMILES, True)
        try:
            self._executor.submit(self._sampler.sample, job).result()
        except Exception as exc:
            raise RuntimeError(f"Model warm-up failed on device {self._sampler.device_name}: {exc}") from exc
        logger.info(
            "REINVENT4 resident model ready on %s after warm-up in %.2f s%s",
            self._sampler.device_name,
            time.perf_counter() - started,
            self._sampler.memory_note(),
        )

    def sample(self, job: _Job, timeout: float = REQUEST_TIMEOUT_SECONDS) -> _Outcome:
        """Run one job; ``ServiceUnavailable`` (503) if it cannot finish within *timeout* seconds."""
        if self._abandoned:
            raise ServiceUnavailable("A previous request that timed out is still running; try again later")
        future = self._executor.submit(self._run, job)
        try:
            return future.result(timeout=timeout)
        except concurrent.futures.TimeoutError:
            if future.cancel():
                logger.error("request waited %ss in the queue and was cancelled", timeout)
                raise ServiceUnavailable(f"Sampling did not start within {timeout:g} seconds (service busy)") from None
            self._abandoned.add(future)
            future.add_done_callback(self._abandoned.discard)
            logger.error("sampling exceeded %ss; it cannot be interrupted and keeps running until it ends", timeout)
            raise ServiceUnavailable(f"Sampling exceeded {timeout:g} seconds") from None
        except SamplingError as exc:
            raise InternalServerError(str(exc)) from None


# ---------------------------------------------------------------------------
# BentoML service
# ---------------------------------------------------------------------------


class BentoArgs(BaseModel):
    """Template arguments AIM forwards via ``bentoml serve --arg``.

    ``aim-runtime`` injects the resolved ``AIM_PORT`` as ``--arg port=...``, so
    this is the only source of the bound port. Deliberately no default: a bare
    ``bentoml serve`` is not a supported entry point.

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


def _fail_startup(message: str) -> NoReturn:
    """Log a clear error and end this worker process with exit code 1.

    Verified on a real node: this only terminates the BentoML *worker*. The
    ``bentoml serve`` arbiter and aim-runtime's supervisor stay alive and respawn
    the worker about every 10 s, so the container keeps running, never becomes
    ready, and logs this message again on every restart. Making the container
    itself exit is a container-level decision (supervisor / orchestrator
    policy) that belongs to the AIM runtime design, not to this service, so the
    helper deliberately stays as is. Operators must use a readiness probe.
    """
    logger.critical("REINVENT4 startup failed: %s", message)  # stderr handler, see _configure_logging
    os._exit(1)


def _start_validation_pool(workers: int) -> Any:
    """Start the RDKit validation worker pool (before the port opens); None means validate in-process.

    ``workers`` below 2 disables the pool. A pool that cannot start is not fatal: the service logs the reason and
    validates in-process, which gives the same results more slowly.
    """
    if workers < 2:
        logger.info(
            "SMILES validation runs in-process (%s=%s)",
            smiles_validation_pool.WORKERS_ENV,
            os.environ.get(smiles_validation_pool.WORKERS_ENV, "unset"),
        )
        return None
    pool = smiles_validation_pool.ValidationPool(workers)
    try:
        pool.start()
    except smiles_validation_pool.ValidationPoolError as exc:
        logger.error("%s; SMILES validation runs in-process", exc)
        return None
    return pool


MODEL_ID = "molecularai/reinvent4"
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
                    "metadata": {
                        "accelerator": _accelerator_name(),
                        "default_prior": _default_prior_name(),
                        "priors": _list_priors(),
                    },
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


# BentoML's default request timeout is 60 s, which would answer 504 long before the service's own
# limit. The traffic timeout must exceed REQUEST_TIMEOUT_SECONDS so that a slow run ends in the
# service's own 503.
@bentoml.service(
    name="reinvent4",
    resources={"gpu": args.accelerator_count},
    # One data-parallel worker per GPU, each with its own resident model and sampler
    # thread: more GPUs raise throughput by serving requests concurrently. The RDKit
    # validation pool is per worker and unchanged.
    workers=args.accelerator_count,
    traffic={"timeout": REQUEST_TIMEOUT_SECONDS + 60},
    http={"port": args.port},
)
@bentoml.asgi_app(canonical_app, path="/")
class REINVENT4Service:
    """BentoML service that drives REINVENT4 de novo molecule generation.

    The model is loaded and warmed up once in ``__init__`` (BentoML opens the port only after it
    returns, so ``/health`` answers only once the service is ready). When no GPU is visible or a
    prior cannot be found, verified or downloaded into ``AIM_CACHE_PATH``, or the
    warm-up fails, the constructor logs a CRITICAL message and ends its worker
    (``_fail_startup``). The container does NOT exit: the worker is restarted about every 10 s
    and the service never becomes ready.

    Each call to ``inference`` samples in-process on the resident model (one request at a time). The RDKit
    validation of the sampled SMILES runs in a pool of worker processes (``smiles_validation_pool``), which is created
    and warmed up here before the port opens, or in-process when the pool is disabled or could not start.
    """

    def __init__(self) -> None:
        # Fail at startup (worker ends with a clear log message, is restarted
        # about every 10 s, never becomes ready) instead of becoming ready and
        # answering every request with a generic 500.
        try:
            device = _resolve_device(bentoml.server_context.worker_index or 1, args.accelerator_count)
            _activate_device(device)
            logger.info("REINVENT4 sampling device: %s", device)
            _preflight_priors()
            default_prior = _resolve_prior(None)
            workers = smiles_validation_pool.resolve_worker_count(os.environ.get(smiles_validation_pool.WORKERS_ENV))
            resolve_sample_batch_size(os.environ.get(SAMPLE_BATCH_SIZE_ENV), 1)
        except RuntimeError as exc:
            _fail_startup(str(exc))
        self._validation_pool = _start_validation_pool(workers)
        self._engine = _SamplingEngine(_ResidentSampler(device, self._validation_pool))
        try:
            self._engine.warm_up(default_prior, _default_prior_name())
        except RuntimeError as exc:
            self._stop_validation_pool()  # _fail_startup ends the process without running exit handlers
            _fail_startup(str(exc))

    @bentoml.on_shutdown
    def _stop_validation_pool(self) -> None:
        """Stop and reap the validation workers when the service shuts down."""
        pool = getattr(self, "_validation_pool", None)
        if pool is not None:
            pool.shutdown()

    @bentoml.api(route="/v1/inference")
    def inference(self, data: SamplingRequest) -> SamplingResponse:
        """Run REINVENT4 sampling on the resident model and return generated SMILES.

        Parameters
        ----------
        data : SamplingRequest
            Sampling parameters (num_smiles, unique_molecules, prior).

        Returns
        -------
        SamplingResponse
            ``smiles``: list of generated SMILES strings.
            ``num_generated``: ``len(smiles)``.
            ``num_requested``: the requested ``num_smiles``.
            ``num_invalid``: sampled strings RDKit could not parse or sanitise.
            ``num_duplicates``: valid sampled strings that repeat an earlier canonical SMILES.

        Raises
        ------
        bentoml.exceptions.BadInput
            (HTTP 400) Unknown prior name.
        bentoml.exceptions.InternalServerError
            (HTTP 500) A prior failed its integrity check, or the sampling run failed or
            produced no SMILES. Details are only logged server-side, never returned to the client.
        bentoml.exceptions.ServiceUnavailable
            (HTTP 503) The default prior download failed, the request did not finish within
            ``REQUEST_TIMEOUT_SECONDS``, or an earlier timed-out run is still running.
        """
        prior_name = data.prior if data.prior is not None else _default_prior_name()
        try:
            prior_path = _resolve_prior(data.prior)
        except UnknownPriorError as exc:
            raise BadInput(str(exc)) from exc
        except PriorIntegrityError as exc:
            # Permanent server-side fault (checksum mismatch): no retry fixes it, so 500 not 503.
            logger.error("REINVENT4 prior failed integrity check: %s", exc)
            raise InternalServerError("The selected prior failed its integrity check; see the server log") from exc
        except RuntimeError as exc:
            # Lazy download failed; details (path, URL) are in the log.
            logger.error("REINVENT4 prior unavailable: %s", exc)
            raise ServiceUnavailable("The requested prior could not be downloaded; see the server log") from exc

        outcome = self._engine.sample(
            _Job(
                prior_path=prior_path,
                prior_name=prior_name,
                num_smiles=data.num_smiles,
                unique_molecules=data.unique_molecules,
            )
        )
        return SamplingResponse(
            smiles=outcome.smiles,
            num_generated=len(outcome.smiles),
            num_requested=data.num_smiles,
            num_invalid=outcome.num_invalid,
            num_duplicates=outcome.num_duplicates,
        )
