# Copyright © Advanced Micro Devices, Inc., or its affiliates.
#
# SPDX-License-Identifier: MIT

"""
Runner for Boltz-2 prediction from a JSON payload.

Bridges the BentoML service to Boltz-2's inference pipeline. Everything here
*imports* upstream ``boltz`` and drives it the way ``boltz predict`` does — the
upstream source is never modified.

Why not shell out to ``boltz predict``?
--------------------------------------
The CLI reloads the 2.3 GB checkpoint on every invocation. A served AIM has to
amortise that across requests, so the checkpoint is loaded once at service start
(:func:`load_boltz2_model`) and each request rebuilds only the cheap parts of the
CLI's flow: process the input, build a data module, run ``trainer.predict``, read
the written files back.

Kernels
-------
Boltz-2 reaches for cuEquivariance kernels when ``use_kernels`` is true
(``boltz.model.layers.triangular_mult.kernel_triangular_mult`` imports
``cuequivariance_torch`` at call time). Those are the ``boltz[cuda]`` extra and
have no ROCm build, so the cuEquivariance path is unreachable here — the same
reason ``boltz predict --no_kernels`` exists. Boltz's own auto-disable in
``Boltz2.setup`` only fires below compute capability 8.0, and MI300X reports
major=9, so it does *not* cover us.

That argument is about cuEquivariance, not about kernels in general.
``src/rocm_triangle`` vendors OpenFold3 0.4.1's Triton triangle kernels — which
the sibling ``openfold3`` AIM already runs on ROCm — and installs them over
boltz's triangle layers, so ``use_kernels=True`` becomes reachable on an AMD GPU
without cuEquivariance. ``load_boltz2_model``'s ``optimizations`` flag selects it;
``False`` leaves boltz completely unpatched.
"""

from __future__ import annotations

import contextlib
import fcntl
import functools
import json
import logging
import os
import shutil
import tempfile
import time
from collections.abc import Callable, Iterator
from dataclasses import asdict
from pathlib import Path
from typing import TYPE_CHECKING, Any, Optional

import yaml
from resident_strategy import resident_strategy, resolve_device

if TYPE_CHECKING:
    import pytorch_lightning as pl
    import torch

logger = logging.getLogger(__name__)

# Boltz's structure-prediction checkpoint, relative to the cache dir. Written by
# boltz.main.download_boltz2 alongside mols/ (the CCD components it needs).
CHECKPOINT_NAME = "boltz2_conf.ckpt"
AFFINITY_CHECKPOINT_NAME = "boltz2_aff.ckpt"
MOLS_DIR_NAME = "mols"

# What the service serves; see ``ServiceConfig.mode``.
MODE_STRUCTURE = "structure"
MODE_AFFINITY = "affinity"
MODES = (MODE_STRUCTURE, MODE_AFFINITY)


def needs_affinity_model(mode: str) -> bool:
    return mode == MODE_AFFINITY


# boltz's mmseqs2 client posts to this by default; overridable per request so a
# deployment can point at a private MSA server.
DEFAULT_MSA_SERVER_URL = "https://api.colabfold.com"

# Where load_boltz2_model stashes the checkpoint's own diffusion step scale, so a
# request that omits step_scale can be restored to it. The service loads the
# checkpoint once and reuses that object for every request, so a bare assignment
# would make one caller's step_scale the default for every later caller.
DEFAULT_STEP_SCALE_ATTR = "_aim_default_step_scale"

# Single-sequence sentinel understood by boltz's schema parser: a protein chain
# with `msa: empty` skips MSA generation entirely. Without it, a chain that has
# no `msa` key defaults to "auto-generate", and boltz raises
# "Missing MSA's in input and --use_msa_server flag not set."
MSA_EMPTY = "empty"

_POLYMER_KEYS = ("protein",)


class MsaServerTimeout(RuntimeError):
    """The MSA server did not finish within the request's budget."""


# --------------------------------------------------------------------------- #
# Bounding MSA-server work
# --------------------------------------------------------------------------- #


class _DeadlineSleep:
    """Stand-in for the ``time`` module that refuses to sleep past a deadline.

    ``boltz.data.msa.mmseqs2`` polls the MSA server in four ``while True`` /
    ``while status in (...)`` loops, and the RATELIMIT branch resubmits forever.
    Every one of those loops backs off through ``time.sleep``, so replacing the
    module's ``time`` reference with this object puts a wall clock on all of them
    at once — and on nothing else.

    This is deliberately a runtime shim rather than a source patch: iron rule 2
    keeps upstream files off-limits, and the guard is a serving-layer concern
    (a wedged BentoML worker), not a model fix. ``traffic.timeout`` cannot do the
    job: a sync endpoint's thread cannot be cancelled, so the client 504s while
    the worker keeps polling.
    """

    def __init__(self, inner: Any, deadline: float) -> None:
        self._inner = inner
        self._deadline = deadline
        self.timeout: Optional["MsaServerTimeout"] = None

    def __getattr__(self, item: str) -> Any:
        return getattr(self._inner, item)

    def sleep(self, seconds: float) -> None:
        remaining = self._deadline - self._inner.monotonic()
        if remaining <= 0:
            # boltz catches an exception per input and goes on, so the caller
            # cannot rely on this one reaching it. The timeout is kept here.
            self.timeout = MsaServerTimeout(
                "MSA server did not return within the request's MSA budget. "
                "Retry later, supply a precomputed MSA, or set use_msa_server=false "
                "to run in single-sequence mode."
            )
            raise self.timeout
        self._inner.sleep(min(seconds, remaining))


@contextlib.contextmanager
def bounded_msa_server(deadline: Optional[float]) -> Iterator[_DeadlineSleep | None]:
    """Give MSA-server polling a ``time.monotonic()`` deadline; no-op when None.

    One deadline covers the whole prediction: boltz queries the server once per
    unique protein entity (and again for pairing), and those queries have to
    share a budget rather than each getting a fresh one.

    Yields the shim, whose ``timeout`` attribute holds the budget-exceeded error
    once it was raised (``None`` when there is no deadline).
    """
    if deadline is None:
        yield None
        return

    from boltz.data.msa import mmseqs2

    original = mmseqs2.time
    shim = _DeadlineSleep(original, deadline)
    mmseqs2.time = shim
    try:
        yield shim
    finally:
        mmseqs2.time = original


@contextlib.contextmanager
def seeded_prediction(seed: Optional[int]) -> Iterator[None]:
    """Seed every random number generator and use deterministic algorithms for one prediction.

    ``None`` changes nothing. The seeding is the one of ``boltz predict --seed``
    (Lightning ``seed_everything``: python, numpy, torch CPU and GPU). The
    deterministic setting is for this prediction only and is restored on exit,
    also when the prediction raises. One worker runs one prediction at a time, so
    no other request sees it.

    A change of this setting makes Dynamo compile the compiled modules once more
    for the new state (the setting is part of its global-state guard); both states
    stay cached after that.
    """
    if seed is None:
        yield
        return

    import torch
    from pytorch_lightning import seed_everything

    was_enabled = torch.are_deterministic_algorithms_enabled()
    was_warn_only = torch.is_deterministic_algorithms_warn_only_enabled()
    seed_everything(seed)
    torch.use_deterministic_algorithms(True, warn_only=True)
    try:
        yield
    finally:
        torch.use_deterministic_algorithms(was_enabled, warn_only=was_warn_only)


def raise_if_msa_timeout_left_no_records(records: Any, msa_budget: Optional[_DeadlineSleep]) -> None:
    """Raise the MSA budget error when it was hit and the input produced no record.

    boltz catches the exception per input ("Failed to process ... Skipping") and
    returns an empty manifest, so without this the caller would report bad input
    for what is a slow MSA server.
    """
    if not records and msa_budget is not None and msa_budget.timeout is not None:
        raise msa_budget.timeout


def is_msa_server_timeout(exc: BaseException) -> bool:
    """Whether ``exc`` (or something it wraps) is the MSA budget-exceeded error.

    The chain is walked because the failure surfaces from inside boltz's
    preprocessing, which may re-raise it wrapped.
    """
    seen: set[int] = set()
    current: Optional[BaseException] = exc
    while current is not None and id(current) not in seen:
        if isinstance(current, MsaServerTimeout):
            return True
        seen.add(id(current))
        current = current.__cause__ or current.__context__
    return False


# --------------------------------------------------------------------------- #
# Asset acquisition
# --------------------------------------------------------------------------- #


@contextlib.contextmanager
def _download_lock(cache: Path) -> Iterator[None]:
    """Serialize asset downloads across processes sharing ``cache``.

    A multi-worker deployment starts N processes that all want the same ~4 GB of
    checkpoint + CCD data; without this they race on the same paths. The lock
    holder downloads, the rest wait and then find everything cached.

    A cache that cannot hold a lock file (read-only mount) yields unlocked:
    nothing can be downloaded there anyway, so there is nothing to serialize.
    """
    try:
        cache.mkdir(parents=True, exist_ok=True)
        lock_file = (cache / ".boltz-assets.lock").open("w")
    except OSError:
        logger.debug("Cannot create a lock file under %s; proceeding unlocked", cache)
        yield
        return

    with lock_file:
        fcntl.flock(lock_file, fcntl.LOCK_EX)
        yield


def _download_checkpoint(path: Path, urls: list[str]) -> None:
    """Fetch ``path`` from the first of ``urls`` that works (upstream's loop in ``download_boltz2``)."""
    import urllib.request

    if path.exists():
        return
    for i, url in enumerate(urls):
        try:
            urllib.request.urlretrieve(url, str(path))  # noqa: S310
            return
        except Exception as exc:  # noqa: BLE001
            if i == len(urls) - 1:
                raise RuntimeError(f"Failed to download {path.name} from all URLs. Last error: {exc}") from exc


def _extract_mols(archive: Path, mols: Path) -> None:
    """Extract ``archive`` so that ``mols`` appears whole or not at all."""
    import tarfile

    for stale in mols.parent.glob(".mols-*"):
        shutil.rmtree(stale, ignore_errors=True)
    staging = Path(tempfile.mkdtemp(dir=mols.parent, prefix=".mols-"))
    try:
        with tarfile.open(str(archive), "r") as tar:
            tar.extractall(staging, filter="data")
        if not (staging / mols.name).is_dir():
            raise RuntimeError(f"{archive.name} has no '{mols.name}/' directory")
        os.replace(staging / mols.name, mols)
    finally:
        shutil.rmtree(staging, ignore_errors=True)


def ensure_boltz2_assets(cache: Path, mode: str = MODE_STRUCTURE) -> None:
    """Download the CCD data and the checkpoints that ``mode`` needs, if absent (idempotent).

    Uses upstream's URLs and fallbacks. ``download_boltz2`` fetches both
    checkpoints, so the structure and the affinity checkpoint are fetched here
    one by one and only when the mode loads them. The cache holds about 6.0 GB
    (structure mode) or 8.0 GB (affinity mode).
    """
    import urllib.request

    from boltz.main import BOLTZ2_AFFINITY_URL_WITH_FALLBACK, BOLTZ2_URL_WITH_FALLBACK, MOL_URL

    with _download_lock(cache):
        mols = cache / MOLS_DIR_NAME
        tar_mols = cache / "mols.tar"
        if not mols.exists():
            if not tar_mols.exists():
                urllib.request.urlretrieve(MOL_URL, str(tar_mols))  # noqa: S310
            _extract_mols(tar_mols, mols)
        _download_checkpoint(cache / CHECKPOINT_NAME, BOLTZ2_URL_WITH_FALLBACK)
        if needs_affinity_model(mode):
            _download_checkpoint(cache / AFFINITY_CHECKPOINT_NAME, BOLTZ2_AFFINITY_URL_WITH_FALLBACK)


def cache_is_writable(cache: Path) -> bool:
    """Whether a file can be created in ``cache`` (the directory is made if it is absent)."""
    try:
        cache.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryFile(dir=cache):
            return True
    except OSError:
        return False


def assets_present(cache: Path, mode: str = MODE_STRUCTURE) -> bool:
    """Whether ``cache`` already holds everything a prediction in ``mode`` needs."""
    if not (cache / MOLS_DIR_NAME).is_dir():
        return False
    if not (cache / CHECKPOINT_NAME).is_file():
        return False
    return not needs_affinity_model(mode) or (cache / AFFINITY_CHECKPOINT_NAME).is_file()


# --------------------------------------------------------------------------- #
# Model loading
# --------------------------------------------------------------------------- #


def load_boltz2_model(
    cache: Path, *, optimizations: bool = True, device: "torch.device | None" = None
) -> "pl.LightningModule":
    """Load the Boltz-2 checkpoint into a LightningModule once at startup.

    Mirrors the argument set ``boltz predict`` builds for ``model == "boltz2"``.
    ``predict_args`` is seeded with upstream's defaults and re-set per request by
    :func:`run_boltz2_prediction`, which is safe because ``Boltz2`` keeps it as a
    plain instance attribute read at predict time.

    ``optimizations=True`` installs the ROCm Triton triangle kernels in
    ``src/rocm_triangle`` (attention and the Inductor-compiled triangle multiply) and, after
    loading, TorchInductor on the Transition and token-transformer modules (see
    ``compilation.py``). ``False`` leaves upstream Boltz untouched.

    Installing the kernels rebinds ``forward`` on boltz's triangle classes, so it
    MUST happen before ``load_from_checkpoint`` constructs any module — a module
    built first would hold bound methods pointing at the unpatched implementation.
    Compilation wraps built instances, so it runs after ``load_from_checkpoint``
    and ``model.eval()``, once real weights and eval-mode buffers exist.

    ``device`` moves the model there once, before compilation and the warm-up; requests
    then keep it resident (see ``resident_strategy``).
    """
    from rocm_triangle import install as install_triton_kernels

    install_triton_kernels("both" if optimizations else "off")

    from boltz.main import Boltz2DiffusionParams, BoltzSteeringParams, MSAModuleArgs, PairformerArgsV2
    from boltz.model.models.boltz2 import Boltz2

    diffusion_params = Boltz2DiffusionParams()
    pairformer_args = PairformerArgsV2()
    # subsample_msa mirrors what `boltz predict` actually passes, not the help
    # text. `--subsample_msa` is an is_flag option with no default, so the CLI
    # hands the model False even though its help says "Default is True" (and the
    # python signature default is True). Matching the CLI keeps this AIM's
    # predictions reproducible against upstream; see
    # tests/references/boltz_cli_defaults.json.
    msa_args = MSAModuleArgs(subsample_msa=False, num_subsampled_msa=1024, use_paired_feature=True)
    steering_args = BoltzSteeringParams()
    steering_args.fk_steering = False
    steering_args.physical_guidance_update = False

    model = Boltz2.load_from_checkpoint(
        cache / CHECKPOINT_NAME,
        strict=True,
        map_location="cpu",
        predict_args={
            "recycling_steps": 3,
            "sampling_steps": 200,
            "diffusion_samples": 1,
            "max_parallel_samples": 5,
            "write_confidence_summary": True,
            "write_full_pae": False,
            "write_full_pde": False,
        },
        diffusion_process_args=asdict(diffusion_params),
        ema=False,
        use_kernels=optimizations,
        pairformer_args=asdict(pairformer_args),
        msa_args=asdict(msa_args),
        steering_args=asdict(steering_args),
    )
    model.eval()
    if device is not None:
        model.to(device)
    if optimizations:
        from rocm_triangle import prepare_model

        prepare_model(model)
    record_default_step_scale(model, diffusion_params.step_scale)

    from compilation import CompileConfig, install_compilation

    mode = "inductor" if optimizations else "none"
    install_compilation(
        model,
        CompileConfig(transition=mode, token_transformer=mode),
    )
    return model


def _force_eager_triangles(module: Any) -> None:
    """Make ``module`` ignore the ``use_kernels`` flag of its caller."""
    forward = module.forward

    def eager_forward(*args: Any, **kwargs: Any) -> Any:
        kwargs["use_kernels"] = False
        return forward(*args, **kwargs)

    module.forward = eager_forward


# Settings of upstream's affinity pass (``boltz predict``, boltz/main.py).
AFFINITY_RECYCLING_STEPS = 5
AFFINITY_MAX_PARALLEL_SAMPLES = 1


def load_affinity_model(
    cache: Path, *, optimizations: bool = False, device: "torch.device | None" = None, compile_affinity: bool = False
) -> "pl.LightningModule":
    """Load the Boltz-2 affinity checkpoint once at startup, the way ``boltz predict`` does.

    Upstream loads it without ``use_kernels``, with steering off and with the
    diffusion, pairformer and MSA arguments of the structure model. With
    ``optimizations`` the trunk takes the triangle kernels, as in the structure model.
    The affinity module runs in fp32 and keeps boltz's own triangle layers. With
    ``optimizations`` and a device, the diffusion score model also runs through a CUDA graph
    (see ``score_graph``).
    With ``compile_affinity`` (needs ``optimizations``), TorchInductor also compiles the Transition and
    token-transformer modules, as in :func:`load_boltz2_model`, and the graph captures after those compiles.
    ``affinity_mw_correction`` is set per request by :func:`run_affinity_prediction`.

    With ``optimizations``, the kernels are installed before the model is built, also when
    :func:`load_boltz2_model` did not run first.
    """
    if optimizations:
        from rocm_triangle import install as install_triton_kernels

        install_triton_kernels("both")

    from boltz.main import Boltz2DiffusionParams, BoltzSteeringParams, MSAModuleArgs, PairformerArgsV2
    from boltz.model.models.boltz2 import Boltz2

    diffusion_params = Boltz2DiffusionParams()
    steering_args = BoltzSteeringParams()
    steering_args.fk_steering = False
    steering_args.physical_guidance_update = False
    steering_args.contact_guidance_update = False
    msa_args = MSAModuleArgs(subsample_msa=False, num_subsampled_msa=1024, use_paired_feature=True)

    model = Boltz2.load_from_checkpoint(
        cache / AFFINITY_CHECKPOINT_NAME,
        strict=True,
        map_location="cpu",
        predict_args={
            "recycling_steps": AFFINITY_RECYCLING_STEPS,
            "sampling_steps": 200,
            "diffusion_samples": 5,
            "max_parallel_samples": AFFINITY_MAX_PARALLEL_SAMPLES,
            "write_confidence_summary": False,
            "write_full_pae": False,
            "write_full_pde": False,
        },
        diffusion_process_args=asdict(diffusion_params),
        ema=False,
        use_kernels=optimizations,
        pairformer_args=asdict(PairformerArgsV2()),
        msa_args=asdict(msa_args),
        steering_args=asdict(steering_args),
        affinity_mw_correction=False,
    )
    model.eval()
    if device is not None:
        model.to(device)
    if optimizations:
        from rocm_triangle import prepare_model

        prepare_model(model)
        for name in ("affinity_module", "affinity_module1", "affinity_module2"):
            if hasattr(model, name):
                _force_eager_triangles(getattr(model, name))
    record_default_step_scale(model, diffusion_params.step_scale)
    compile_counter = None
    if optimizations and compile_affinity:
        import compilation

        compilation.install_compilation(
            model, compilation.CompileConfig(transition="inductor", token_transformer="inductor")
        )
        if not compilation.compile_events_available(model):
            logger.warning(
                "Dynamo compile events are not available, so a capture cannot wait for the compiles: "
                "the affinity score model runs without a CUDA graph"
            )
            return model
        compile_counter = functools.partial(compilation.compile_count, model)
    if optimizations and device is not None:
        from score_graph import install_score_graph

        install_score_graph(model, compile_count=compile_counter)
    return model


# --------------------------------------------------------------------------- #
# Per-request diffusion step scale
# --------------------------------------------------------------------------- #


def record_default_step_scale(model: Any, fallback: float) -> Optional[float]:
    """Remember ``model``'s load-time diffusion step scale and return it.

    Read back off the loaded module rather than assumed, so it stays whatever
    boltz resolved; ``fallback`` (``Boltz2DiffusionParams.step_scale``, 1.5 in
    boltz 2.2.1) covers a module that does not expose the attribute.
    """
    structure_module = getattr(model, "structure_module", None)
    current = getattr(structure_module, "step_scale", None) if structure_module is not None else None
    default = float(current if current is not None else fallback)
    setattr(model, DEFAULT_STEP_SCALE_ATTR, default)
    return default


def apply_step_scale(model: Any, step_scale: Optional[float]) -> Optional[float]:
    """Set the diffusion step scale for ONE request; return the value now in effect.

    ``step_scale`` is the only per-request knob that is not re-set on every
    prediction — ``predict_args`` is rebuilt each time, this lives on the
    structure module. Assigning it only when the request supplies one therefore
    leaks: the value survives into every later request on the same worker, which
    silently moves the AIM's *served* step scale off boltz's default. Restoring
    the recorded default when the request omits one keeps the knob per-request.

    Returns ``None`` for a model that exposes no ``structure_module.step_scale``.
    """
    structure_module = getattr(model, "structure_module", None)
    if structure_module is None or not hasattr(structure_module, "step_scale"):
        return None
    default = getattr(model, DEFAULT_STEP_SCALE_ATTR, None)
    if default is None:
        default = structure_module.step_scale
        setattr(model, DEFAULT_STEP_SCALE_ATTR, default)
    effective = default if step_scale is None else step_scale
    structure_module.step_scale = effective
    return effective


# --------------------------------------------------------------------------- #
# Request → boltz input
# --------------------------------------------------------------------------- #


def apply_single_sequence_mode(sequences: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Return ``sequences`` with every un-annotated protein chain set to ``msa: empty``.

    A request that does not want the MSA server has to say so per chain: boltz's
    schema parser treats a missing ``msa`` key as "auto-generate", which then
    hard-fails with "Missing MSA's in input and --use_msa_server flag not set."
    An explicit ``msa`` the caller supplied (a path, or ``empty`` already) is left
    alone, so this only fills the gap rather than overriding intent.
    """
    patched: list[dict[str, Any]] = []
    for entry in sequences:
        if not isinstance(entry, dict):
            patched.append(entry)
            continue
        new_entry = {k: (dict(v) if isinstance(v, dict) else v) for k, v in entry.items()}
        for key in _POLYMER_KEYS:
            chain = new_entry.get(key)
            if isinstance(chain, dict) and not chain.get("msa"):
                chain["msa"] = MSA_EMPTY
        patched.append(new_entry)
    return patched


def build_boltz_schema(
    sequences: list[dict[str, Any]],
    *,
    version: int = 1,
    constraints: Optional[list[dict[str, Any]]] = None,
    templates: Optional[list[dict[str, Any]]] = None,
    properties: Optional[list[dict[str, Any]]] = None,
    use_msa_server: bool = False,
) -> dict[str, Any]:
    """Build the boltz YAML schema dict for one prediction request."""
    schema: dict[str, Any] = {
        "version": version,
        "sequences": sequences if use_msa_server else apply_single_sequence_mode(sequences),
    }
    if constraints:
        schema["constraints"] = constraints
    if templates:
        schema["templates"] = templates
    if properties:
        schema["properties"] = properties
    return schema


_TEMPLATE_FILE_KEYS = ("cif", "pdb")


def _is_inline_msa(value: Any) -> bool:
    return isinstance(value, str) and value.lstrip().startswith(">")


def _is_inline_template(value: Any) -> bool:
    return isinstance(value, str) and "\n" in value


# Upper-case letters of ``boltz.data.const.prot_letter_to_token``, the only non-gap, non-insert
# characters that Boltz's A3M parser accepts (any other character raises ``KeyError``).
_A3M_RESIDUE_LETTERS = frozenset("ARNDCEQGHILKMFPSTWYVXJBZOU")


def check_inline_msa(msa: str, sequence: str) -> None:
    """Raise ``ValueError`` for an inline A3M that Boltz's parser or featurizer would fail on.

    Boltz skips header, empty and ``#`` lines, treats ``-`` as a gap and a lower-case letter as
    an insert, and raises ``KeyError`` for any other character that is not an upper-case residue
    letter. An A3M without a sequence line gives an empty MSA, and a row whose aligned length
    differs from the chain makes the feature step fail; Boltz then retries for minutes.
    """
    rows = 0
    for line in msa.splitlines():
        line = line.strip()
        if not line or line.startswith(("#", ">")):
            continue
        rows += 1
        aligned = 0
        for char in line:
            if char != "-" and char.islower():
                continue
            if char != "-" and char not in _A3M_RESIDUE_LETTERS:
                raise ValueError("msa has a character that is not a residue letter, '-' or an insert letter")
            aligned += 1
        if aligned != len(sequence):
            raise ValueError("msa has a row whose aligned length differs from the protein sequence")
    if rows == 0:
        raise ValueError("msa has no sequence line after its header")


def reject_file_references(sequences: list[dict[str, Any]], templates: Optional[list[dict[str, Any]]]) -> None:
    """Raise ``ValueError`` unless every ``msa`` and template file is inline content.

    Boltz reads these fields as file paths. A request must carry the content
    itself (an A3M document for ``msa``, mmCIF or PDB text for a template), so a
    caller cannot make the service read or probe files in the container.
    """
    for entry in sequences:
        for key in _POLYMER_KEYS:
            chain = entry.get(key) if isinstance(entry, dict) else None
            msa = chain.get("msa") if isinstance(chain, dict) else None
            if msa is not None and msa != MSA_EMPTY and not _is_inline_msa(msa):
                raise ValueError(
                    f"{key}.msa must be 'empty' or inline A3M text starting with '>'; file paths are not accepted"
                )
            if _is_inline_msa(msa):
                check_inline_msa(msa, str(chain.get("sequence", "")))
    for template in templates or []:
        for key in _TEMPLATE_FILE_KEYS:
            if key in template and not _is_inline_template(template[key]):
                raise ValueError(f"templates[].{key} must be inline mmCIF or PDB text; file paths are not accepted")


def materialize_inline_inputs(
    sequences: list[dict[str, Any]], templates: Optional[list[dict[str, Any]]], directory: Path
) -> tuple[list[dict[str, Any]], Optional[list[dict[str, Any]]]]:
    """Write inline MSA and template content to ``directory`` and point the schema at the files.

    Returns copies; the request-owned inputs are not changed. Call only after
    ``reject_file_references`` accepted them.
    """
    count = 0
    out_sequences: list[dict[str, Any]] = []
    for entry in sequences:
        new_entry = {k: (dict(v) if isinstance(v, dict) else v) for k, v in entry.items()}
        for key in _POLYMER_KEYS:
            chain = new_entry.get(key)
            if isinstance(chain, dict) and _is_inline_msa(chain.get("msa")):
                path = directory / f"msa_{count}.a3m"
                path.write_text(chain["msa"], encoding="utf-8")
                chain["msa"] = str(path)
                count += 1
        out_sequences.append(new_entry)
    out_templates: Optional[list[dict[str, Any]]] = None
    if templates:
        out_templates = []
        for template in templates:
            new_template = dict(template)
            for key in _TEMPLATE_FILE_KEYS:
                if key in new_template:
                    path = directory / f"template_{count}.{key}"
                    path.write_text(new_template[key], encoding="utf-8")
                    new_template[key] = str(path)
                    count += 1
            out_templates.append(new_template)
    return out_sequences, out_templates


def sanitize_record_name(name: str) -> str:
    """Reduce ``name`` to a filesystem-safe record id.

    boltz derives the record id from the input file's stem and writes output
    under a directory of that name, so a caller-supplied name reaches the
    filesystem. Restricting it to a known-safe alphabet keeps a request from
    escaping the per-request temp dir.
    """
    cleaned = "".join(c if (c.isalnum() or c in "-_") else "_" for c in name).strip("_")
    return cleaned or "prediction"


# --------------------------------------------------------------------------- #
# Output parsing
# --------------------------------------------------------------------------- #


def _parse_predictions(predictions_dir: Path) -> dict[str, Any]:
    """Parse boltz's ``predictions/`` tree into a JSON-serializable result dict.

    BoltzWriter writes, per record::

        predictions/<record_id>/<record_id>_model_<rank>.{cif,pdb}
        predictions/<record_id>/confidence_<record_id>_model_<rank>.json
        predictions/<record_id>/plddt_<record_id>_model_<rank>.npz

    Ranks are assigned by descending confidence, so ``_model_0`` is the best
    sample. The returned shape ("structures" / "confidence" / "error") matches
    the sibling openfold3 AIM's response so downstream consumers can read either.
    """
    result: dict[str, Any] = {"structures": [], "confidence": {}, "error": False}
    if not predictions_dir.is_dir():
        return result

    for record_dir in sorted(predictions_dir.iterdir()):
        if not record_dir.is_dir():
            continue
        for model_file in sorted(record_dir.glob("*_model_*")):
            suffix = model_file.suffix.lstrip(".")
            if suffix not in ("cif", "pdb"):
                continue
            sample_id = model_file.stem
            result["structures"].append(
                {
                    "record_id": sample_id,
                    "format": "mmcif" if suffix == "cif" else "pdb",
                    "content": model_file.read_text(),
                }
            )
            conf_file = record_dir / f"confidence_{sample_id}.json"
            if conf_file.exists():
                try:
                    result["confidence"][sample_id] = json.loads(conf_file.read_text())
                except (OSError, json.JSONDecodeError):
                    logger.warning("Unreadable confidence summary at %s", conf_file)

    # Best sample first: _model_0 is boltz's highest-confidence rank.
    result["structures"].sort(key=lambda s: s["record_id"])
    return result


# --------------------------------------------------------------------------- #
# Prediction
# --------------------------------------------------------------------------- #


def predict_tracking_cuda_peak(predict: Callable[[], None], device: "torch.device | None") -> Optional[float]:
    """Run ``predict`` and return the process peak of allocated CUDA memory during it.

    The peak is in GB (10^9 bytes) and includes the resident model weights.
    It is ``None`` when the request does not run on a CUDA device.
    """
    if device is None or device.type != "cuda":
        predict()
        return None
    import torch

    torch.cuda.reset_peak_memory_stats(device)
    predict()
    return round(torch.cuda.max_memory_allocated(device) / 1e9, 2)


def run_affinity_prediction(
    *,
    cache: Path,
    model: "pl.LightningModule",
    processed: Any,
    manifest: Any,
    predictions_dir: Path,
    out_dir: Path,
    sampling_steps: int,
    diffusion_samples: int,
    mw_correction: bool,
    step_scale: Optional[float],
    num_workers: int,
    accelerator: str,
    devices: int,
    trainer_dir: Path,
) -> dict[str, Any]:
    """Run upstream's affinity pass on the records of ``manifest`` and return the written values.

    Follows the second half of ``boltz predict``: the data module reads
    ``<predictions_dir>/<id>/pre_affinity_<id>.npz`` (the rank-0 structure), the
    model runs with recycling 5 and one parallel sample, and ``BoltzAffinityWriter``
    writes ``affinity_<id>.json``. Returns the content of that file for the one
    binder record.
    """
    from boltz.data.module.inferencev2 import Boltz2InferenceDataModule
    from boltz.data.write.writer import BoltzAffinityWriter
    from pytorch_lightning import Trainer

    pred_writer = BoltzAffinityWriter(data_dir=processed.targets_dir, output_dir=predictions_dir)
    data_module = Boltz2InferenceDataModule(
        manifest=manifest,
        target_dir=predictions_dir,
        msa_dir=processed.msa_dir,
        mol_dir=cache / MOLS_DIR_NAME,
        num_workers=num_workers,
        constraints_dir=processed.constraints_dir,
        template_dir=processed.template_dir,
        extra_mols_dir=processed.extra_mols_dir,
        override_method="other",
        affinity=True,
    )
    model.predict_args = {
        "recycling_steps": AFFINITY_RECYCLING_STEPS,
        "sampling_steps": sampling_steps,
        "diffusion_samples": diffusion_samples,
        "max_parallel_samples": AFFINITY_MAX_PARALLEL_SAMPLES,
        "write_confidence_summary": False,
        "write_full_pae": False,
        "write_full_pde": False,
    }
    model.affinity_mw_correction = mw_correction
    apply_step_scale(model, step_scale)

    device = resolve_device(accelerator) if devices == 1 else None
    trainer = Trainer(
        default_root_dir=trainer_dir,
        strategy=resident_strategy(device) if device is not None else "auto",
        callbacks=[pred_writer],
        accelerator=accelerator,
        devices=devices,
        precision="bf16-mixed",
        logger=False,
        enable_progress_bar=False,
        enable_model_summary=False,
    )
    model._trainer = None  # noqa: SLF001
    trainer.predict(model, datamodule=data_module, return_predictions=False)

    return _parse_affinity(predictions_dir, [r.id for r in manifest.records])


def _parse_affinity(predictions_dir: Path, record_ids: list[str]) -> dict[str, Any]:
    """The content of ``affinity_<id>.json`` of the first record that has one; empty when none was written."""
    for record_id in record_ids:
        path = predictions_dir / record_id / f"affinity_{record_id}.json"
        if path.is_file():
            return json.loads(path.read_text())
    return {}


def run_boltz2_prediction(
    *,
    cache: Path,
    model: "pl.LightningModule | None",
    sequences: list[dict[str, Any]],
    name: str = "prediction",
    version: int = 1,
    constraints: Optional[list[dict[str, Any]]] = None,
    templates: Optional[list[dict[str, Any]]] = None,
    properties: Optional[list[dict[str, Any]]] = None,
    recycling_steps: int = 3,
    sampling_steps: int = 200,
    diffusion_samples: int = 1,
    max_parallel_samples: int = 5,
    step_scale: Optional[float] = None,
    output_format: str = "mmcif",
    use_msa_server: bool = False,
    msa_server_url: str = DEFAULT_MSA_SERVER_URL,
    msa_pairing_strategy: str = "greedy",
    max_msa_seqs: int = 8192,
    num_workers: int = 0,
    accelerator: str = "gpu",
    devices: int = 1,
    method: Optional[str] = None,
    msa_deadline: Optional[float] = None,
    affinity_model: "pl.LightningModule | None" = None,
    sampling_steps_affinity: int = 200,
    diffusion_samples_affinity: int = 5,
    affinity_mw_correction: bool = False,
) -> dict[str, Any]:
    """Run one Boltz-2 structure prediction and return the parsed result.

    Follows ``boltz predict``'s flow for ``model == "boltz2"`` but reuses the
    caller-supplied ``model`` instead of reloading the checkpoint. Every
    intermediate (processed features, MSAs, written structures) lives in a
    per-request temp dir, so concurrent requests cannot see each other's work —
    boltz keys its "already processed, skipping" shortcut on the output tree, and
    a shared tree would silently return a previous request's structure.

    Returns a dict with "structures", "confidence", "timing" and "error".
    """
    from boltz.data.module.inferencev2 import Boltz2InferenceDataModule
    from boltz.data.types import Manifest
    from boltz.data.write.writer import BoltzWriter
    from boltz.main import BoltzProcessedInput, process_inputs
    from pytorch_lightning import Trainer

    if output_format not in ("pdb", "mmcif"):
        raise ValueError(f"output_format must be 'pdb' or 'mmcif', got {output_format!r}")

    record_name = sanitize_record_name(name)
    reject_file_references(sequences, templates)

    started = time.monotonic()
    with tempfile.TemporaryDirectory(prefix="boltz2_run_") as work_dir:
        work_path = Path(work_dir)
        sequences, templates = materialize_inline_inputs(sequences, templates, work_path)
        schema = build_boltz_schema(
            sequences,
            version=version,
            constraints=constraints,
            templates=templates,
            properties=properties,
            use_msa_server=use_msa_server,
        )
        input_path = work_path / f"{record_name}.yaml"
        input_path.write_text(yaml.safe_dump(schema, sort_keys=False), encoding="utf-8")

        out_dir = work_path / "out"
        out_dir.mkdir()

        # preprocessing_threads=1 keeps boltz on its sequential path. Its Pool is
        # guarded by `if preprocessing_threads > 1 and len(data) > 1`, and we always
        # submit exactly one input — so nothing forks. That matters here: BentoML
        # runs a sync endpoint on an AnyIO worker thread, and forking a Pool off a
        # non-main thread is what wedges a worker for good.
        with bounded_msa_server(msa_deadline if use_msa_server else None) as msa_budget:
            process_inputs(
                data=[input_path],
                out_dir=out_dir,
                ccd_path=cache / "ccd.pkl",  # unused for boltz2; it loads canonicals from mol_dir
                mol_dir=cache / MOLS_DIR_NAME,
                use_msa_server=use_msa_server,
                msa_server_url=msa_server_url,
                msa_pairing_strategy=msa_pairing_strategy,
                max_msa_seqs=max_msa_seqs,
                boltz2=True,
                preprocessing_threads=1,
            )

        manifest = Manifest.load(out_dir / "processed" / "manifest.json")
        raise_if_msa_timeout_left_no_records(manifest.records, msa_budget)
        if not manifest.records:
            return {
                "error": True,
                "message": "Input produced no records; check the sequences payload.",
                "structures": [],
                "confidence": {},
                "timing": {},
            }

        processed_dir = out_dir / "processed"
        processed = BoltzProcessedInput(
            manifest=manifest,
            targets_dir=processed_dir / "structures",
            msa_dir=processed_dir / "msa",
            constraints_dir=(processed_dir / "constraints" if (processed_dir / "constraints").exists() else None),
            template_dir=(processed_dir / "templates" if (processed_dir / "templates").exists() else None),
            extra_mols_dir=(processed_dir / "mols" if (processed_dir / "mols").exists() else None),
        )

        data_module = Boltz2InferenceDataModule(
            manifest=processed.manifest,
            target_dir=processed.targets_dir,
            msa_dir=processed.msa_dir,
            mol_dir=cache / MOLS_DIR_NAME,
            num_workers=num_workers,
            constraints_dir=processed.constraints_dir,
            template_dir=processed.template_dir,
            extra_mols_dir=processed.extra_mols_dir,
            override_method=method,
        )

        pred_writer = BoltzWriter(
            data_dir=processed.targets_dir,
            output_dir=out_dir / "predictions",
            output_format=output_format,
            boltz2=True,
        )

        device = resolve_device(accelerator) if devices == 1 else None
        timing: dict[str, float] = {}

        def predict_structure() -> None:
            # Per-request knobs live on the module, which reads them inside
            # predict_step. Upstream builds a fresh module per run; we reuse one, so
            # they are re-set here rather than baked in at load time.
            model.predict_args = {
                "recycling_steps": recycling_steps,
                "sampling_steps": sampling_steps,
                "diffusion_samples": diffusion_samples,
                "max_parallel_samples": max_parallel_samples,
                "write_confidence_summary": True,
                "write_full_pae": False,
                "write_full_pde": False,
            }
            apply_step_scale(model, step_scale)
            trainer = Trainer(
                default_root_dir=out_dir,
                strategy=resident_strategy(device) if device is not None else "auto",
                callbacks=[pred_writer],
                accelerator=accelerator,
                devices=devices,
                precision="bf16-mixed",
                logger=False,
                enable_progress_bar=False,
                enable_model_summary=False,
            )
            # Lightning attaches the module to this trainer; drop the previous
            # request's so a stale log_dir cannot leak into this prediction.
            model._trainer = None  # noqa: SLF001
            trainer.predict(model, datamodule=data_module, return_predictions=False)

        affinity: dict[str, Any] = {}

        def predict_all() -> None:
            predict_structure()
            if affinity_model is None or not any(r.affinity for r in manifest.records):
                return
            affinity_started = time.monotonic()
            affinity.update(
                run_affinity_prediction(
                    cache=cache,
                    model=affinity_model,
                    processed=processed,
                    manifest=manifest,
                    predictions_dir=out_dir / "predictions",
                    out_dir=out_dir,
                    sampling_steps=sampling_steps_affinity,
                    diffusion_samples=diffusion_samples_affinity,
                    mw_correction=affinity_mw_correction,
                    step_scale=step_scale,
                    num_workers=num_workers,
                    accelerator=accelerator,
                    devices=devices,
                    trainer_dir=out_dir,
                )
            )
            timing["affinity_s"] = round(time.monotonic() - affinity_started, 3)

        cuda_peak_gb = predict_tracking_cuda_peak(predict_all, device)

        result = _parse_predictions(out_dir / "predictions")
        result["timing"] = {"runtime_s": round(time.monotonic() - started, 3), **timing}
        result["cuda_peak_gb"] = cuda_peak_gb
        if affinity:
            result["affinity"] = affinity

        if not result["structures"]:
            failed = getattr(pred_writer, "failed", 0)
            return {
                "error": True,
                "message": (
                    f"Boltz-2 produced no structure ({failed} record(s) failed during prediction). "
                    "This usually means the input could not be featurized."
                ),
                "structures": [],
                "confidence": {},
                "timing": result["timing"],
            }
        if affinity_model is not None and any(r.affinity for r in manifest.records) and not affinity:
            return {
                "error": True,
                "message": "Boltz-2 produced no affinity; the complex could not be featurized for the affinity model.",
                "structures": [],
                "confidence": {},
                "timing": result["timing"],
            }
        return result


def resolve_cache_dir(default: str) -> Path:
    """Resolve the Boltz asset cache, honouring BOLTZ_CACHE then the AIM default."""
    raw = os.environ.get("BOLTZ_CACHE") or default
    return Path(raw).expanduser().resolve()
