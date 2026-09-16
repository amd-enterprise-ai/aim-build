# Copyright © Advanced Micro Devices, Inc., or its affiliates.
#
# SPDX-License-Identifier: MIT

"""
Runner for OpenFold3 prediction from JSON payload.

Bridges the BentoML service to OpenFold3's inference pipeline.
Constructs the inference config, runs prediction via the OpenFold3 API,
and parses output files into a JSON-serializable response.
"""

from __future__ import annotations

import contextlib
import fcntl
import json
import logging
import os
import tempfile
from collections.abc import Iterator
from pathlib import Path
from typing import TYPE_CHECKING, Any, Optional

if TYPE_CHECKING:
    import pytorch_lightning as pl

logger = logging.getLogger(__name__)

# Must match a key in OF3's OPENFOLD_MODEL_CHECKPOINT_REGISTRY that is valid
# for the pinned OpenFold3 version (0.5.0 → OpenBind 174k). Used both to
# pre-download at startup and as inference_ckpt_name so load cannot silently
# pick a different default than we fetched.
DEFAULT_CHECKPOINT_REGISTRY_NAME = "openbind-2025-06-30-174k"

# ColabFold's client warns that an unset user agent "will become an error in the
# future" and asks for "toolname/version contact". OF3 defaults to a bare
# "openfold"; identify the AIM instead so the operators of the shared public
# server can tell who is calling.
DEFAULT_MSA_USER_AGENT = "aim-openfold3 (+https://github.com/amd-enterprise-ai/aim-build)"


# OF3 only parses MSA files whose basename is a key in MSASettings.max_seq_counts;
# other names are silently skipped. These match OF3's own MSA-server output.
_RECOGNIZED_MSA_BASENAME = {
    "main_msa": "colabfold_main",
    "paired_msa": "colabfold_paired",
}


def _write_msa_file(content: str | list[str], dest_dir: Path, basename: str) -> str:
    """Write a3m MSA ``content`` to ``dest_dir/{basename}.a3m``; return the path.

    ``content`` is a3m text (str) or list[str]; other types raise TypeError,
    empty/whitespace-only content raises ValueError. A list is concatenated into
    one file (OF3 only reads recognized basenames, and concatenation is what it
    does with a chain's MSA files anyway).
    """
    if isinstance(content, str):
        items = [content]
    elif isinstance(content, list) and all(isinstance(t, str) for t in content):
        items = content
    else:
        raise TypeError("inline MSA content must be a3m text (str) or list[str]")
    parts: list[str] = []
    for text in items:
        if not text.strip():
            raise ValueError("inline MSA content is empty")
        parts.append(text if text.endswith("\n") else text + "\n")
    dest_dir.mkdir(parents=True, exist_ok=True)
    path = dest_dir / f"{basename}.a3m"
    path.write_text("".join(parts), encoding="utf-8")
    return str(path)


def _materialize_inline_msas(queries: dict[str, Any], msa_dir: Path) -> None:
    """Convert inline chain MSAs into on-disk .a3m paths OF3 will read.

    Each chain's ``main_msa`` / ``paired_msa`` (str or list[str]) is written to
    its own subdir ``msa_dir/q{q}_c{c}/`` with a recognized basename, and the
    chain's ``*_msa_file_paths`` set accordingly. The per-chain subdir is
    required: OF3 keys a chain's MSA on the file's parent-dir name, so a flat
    layout would collapse all chains onto one. Inline keys are always dropped
    (OF3's query parser forbids unknown fields); explicit ``*_msa_file_paths``
    win. No-op when no inline content is present.
    """
    for q_idx, query in enumerate(queries.values()):
        chains = query.get("chains", [])
        if not isinstance(chains, list):
            continue
        for c_idx, chain in enumerate(chains):
            if not isinstance(chain, dict):
                continue
            chain_dir = msa_dir / f"q{q_idx}_c{c_idx}"
            for inline_key, path_key in (
                ("main_msa", "main_msa_file_paths"),
                ("paired_msa", "paired_msa_file_paths"),
            ):
                content = chain.pop(inline_key, None)
                if not content:
                    continue
                if chain.get(path_key):
                    continue
                chain[path_key] = [_write_msa_file(content, chain_dir, _RECOGNIZED_MSA_BASENAME[inline_key])]


def find_request_conflicts(
    queries: dict[str, Any],
    *,
    use_msa_server: bool,
    use_templates: bool,
    num_model_seeds: int | None,
    seeds_explicit: bool,
) -> list[str]:
    """Return error messages for mutually-exclusive request inputs.

    Each case is a conflict where one input would silently override another, so
    the request is rejected rather than guessing intent:
      - inline MSAs with use_msa_server (ColabFold MSAs would overwrite inline)
      - both seeds and num_model_seeds (num_model_seeds would regenerate, ignoring seeds)
      - use_templates without use_msa_server (templates are only server-fetched)
    """
    conflicts: list[str] = []

    chains = [
        c for q in queries.values() if isinstance(q, dict) for c in (q.get("chains") or []) if isinstance(c, dict)
    ]
    has_inline_msa = any(c.get("main_msa") or c.get("paired_msa") for c in chains)

    if use_msa_server and has_inline_msa:
        conflicts.append(
            "Inline main_msa/paired_msa cannot be combined with use_msa_server=True "
            "(server MSAs would override them). Set use_msa_server=false to use "
            "inline MSAs."
        )
    if num_model_seeds is not None and seeds_explicit:
        conflicts.append(
            "seeds and num_model_seeds cannot both be set: num_model_seeds generates "
            "seeds while seeds lists them explicitly. Provide only one."
        )
    if use_templates and not use_msa_server:
        conflicts.append(
            "use_templates=True requires use_msa_server=True (templates are fetched " "via the ColabFold server)."
        )

    return conflicts


@contextlib.contextmanager
def _download_lock(cache: Path) -> Iterator[None]:
    """Serialize the checkpoint download across processes sharing ``cache``.

    A multi-worker deployment starts N processes that all want the same 2.3 GB
    file; without this they race on it. The lock holder downloads, the rest wait
    and then find it cached.

    A cache that cannot hold a lock file (read-only mount) yields unlocked
    rather than failing: nothing can be downloaded there anyway, so there is
    nothing to serialize.
    """
    try:
        cache.mkdir(parents=True, exist_ok=True)
        lock_file = (cache / ".parameters.lock").open("w")
    except OSError:
        logger.debug("Cannot create a lock file under %s; proceeding unlocked", cache)
        yield
        return

    # Closing the file releases the lock, on the exception path too, so no
    # explicit unlock is needed — but the close is then load-bearing.
    with lock_file:
        fcntl.flock(lock_file, fcntl.LOCK_EX)
        yield


def _ensure_model_parameters(cache: Path) -> None:
    """Download model checkpoint if not already cached (idempotent, multi-process safe)."""
    from openfold3.entry_points.parameters import download_model_parameters

    with _download_lock(cache):
        download_model_parameters(
            download_dir=cache,
            parameter_name=DEFAULT_CHECKPOINT_REGISTRY_NAME,
            skip_confirmation=True,
        )


def _scratch_runner_args(work_path: Path) -> dict[str, Any]:
    """Runner args placing OF3's template scratch under ``work_path``.

    ``output_directory`` defaults to ``get_of3_tmpdir(...)``, which is keyed only
    on the OS user, so every worker in a container resolves to the same tree. OF3
    then treats an already-staged file as a cache and deletes the tree after each
    run, letting concurrent requests read each other's template data or lose
    their own mid-run.

    This has to be passed at construction, not assigned afterwards: the template
    validator derives structure/cache/precache/array/log directories from
    ``output_directory`` while building the model, and a later assignment would
    leave those five behind on the shared tree.

    ``work_path`` is the request's ``mkdtemp`` directory, so this is unique per
    request. Kept under its own subdirectory to stay clearly distinct from
    ``work_path/msas``, which holds caller-supplied inline alignments.

    The MSA side needs no counterpart: OpenFold3 0.5.0 derives its
    workspace from a per-run directory name and saves records under
    ``output_dir/msas``, both already unique per request.
    """
    return {
        "template_preprocessor_settings": {"output_directory": work_path / "of3_scratch" / "template_data"},
    }


def _msa_server_args() -> dict[str, Any]:
    """``msa_computation_settings`` overrides for talking to the MSA server.

    Kept separate from ``_scratch_runner_args`` because OpenFold3 manages
    its own per-request MSA workspace.
    """
    return {"server_user_agent": os.environ.get("OPENFOLD3_MSA_USER_AGENT") or DEFAULT_MSA_USER_AGENT}


def _colabfold_client() -> Any:
    """OF3's ColabFold client module, patched in this image with a deadline hook."""
    from openfold3.core.data.tools import colabfold_msa_server

    return colabfold_msa_server


def msa_deadline_hook_available() -> bool:
    """Whether this image's MSA deadline patch is in place.

    Called at service startup so that an upstream bump which silently defeats
    ``of3_msa_server_deadline.patch`` fails loudly, rather than serving with the
    only bound on MSA-server work quietly removed.
    """
    try:
        client = _colabfold_client()
    except ImportError:
        return False
    return hasattr(client, "msa_deadline") and hasattr(client, "MsaServerTimeout")


@contextlib.contextmanager
def bounded_msa_server(deadline: float | None) -> Iterator[None]:
    """Give MSA-server work a ``time.monotonic()`` deadline; no-op when None.

    One deadline covers the whole prediction: OF3 makes several server queries
    (main MSAs, then one per unique complex for paired MSAs, then templates), and
    they have to share a budget rather than each getting a fresh one.
    """
    if deadline is None:
        yield
        return
    with _colabfold_client().msa_deadline(deadline):
        yield


def is_msa_server_timeout(exc: BaseException) -> bool:
    """Whether ``exc`` (or something it wraps) is the MSA budget-exceeded error.

    The chain is walked because the failure surfaces from inside Lightning's
    predict loop, which may re-raise it wrapped.
    """
    try:
        timeout_cls = _colabfold_client().MsaServerTimeout
    except (ImportError, AttributeError):
        return False

    seen: set[int] = set()
    current: BaseException | None = exc
    while current is not None and id(current) not in seen:
        if isinstance(current, timeout_cls):
            return True
        seen.add(id(current))
        current = current.__cause__ or current.__context__
    return False


def _get_rocm_runner_args() -> dict[str, Any]:
    """Return runner config dict optimized for ROCm GPUs.

    Enables Triton triangle kernels (ROCm-compatible) and disables
    NVIDIA-specific DeepSpeed evo attention and cuEquivariance kernels.
    """
    return {
        "model_update": {
            "presets": ["predict"],
            "custom": {
                "settings": {
                    "memory": {
                        "eval": {
                            "use_triton_triangle_kernels": True,
                            "use_deepspeed_evo_attention": False,
                            "use_cueq_triangle_kernels": False,
                        }
                    }
                }
            },
        }
    }


def _load_model(cache: Path) -> "pl.LightningModule":
    """Load the OF3 checkpoint into a LightningModule once at startup.

    Per-request runners reuse the returned model (injected into their
    ``lightning_module`` slot) so the 2.3 GB checkpoint isn't reloaded each call;
    their other cached_properties (data/trainer/etc.) are rebuilt per request.

    ``output_dir`` must be a service-lifetime path, NOT a TemporaryDirectory: the
    cached model holds a ``_trainer`` whose ``log_dir`` points here, so a temp
    dir deleted on return causes ENOENT on the first request.
    """
    from openfold3.entry_points.experiment_runner import InferenceExperimentRunner
    from openfold3.entry_points.validator import InferenceExperimentConfig

    runner_args = _get_rocm_runner_args()
    runner_args["data_module_args"] = {"num_workers": 0}
    runner_args["output_writer_settings"] = {"structure_format": "cif"}

    expt_config = InferenceExperimentConfig(
        cache_path=cache,
        inference_ckpt_name=DEFAULT_CHECKPOINT_REGISTRY_NAME,
        **runner_args,
    )

    warmup_output = cache / "warmup_output"
    warmup_output.mkdir(parents=True, exist_ok=True)

    runner = InferenceExperimentRunner(expt_config, output_dir=warmup_output)
    runner.setup()
    return runner.lightning_module


def _read_seed_timing(seed_dir: Path) -> dict[str, Any] | None:
    """Return the seed's ``timing.json`` (``{"runtime_s": ...}``), or None.

    timing is incidental metadata, so a missing or corrupt file is tolerated
    rather than failing the prediction.
    """
    timing_file = seed_dir / "timing.json"
    if not timing_file.exists():
        return None
    try:
        with timing_file.open() as f:
            return json.load(f)
    except (OSError, json.JSONDecodeError):
        return None


def _parse_output_dir(output_dir: Path, *, include_atom_confidences: bool = False) -> dict[str, Any]:
    """Parse OpenFold3 output directory into a JSON-serializable result dict.

    OF3OutputWriter writes files in this structure:
        output_dir/{query_id}/seed_{seed}/{query_id}_seed_{seed}_sample_{n}_model.{cif|pdb}
        output_dir/{query_id}/seed_{seed}/{query_id}_seed_{seed}_sample_{n}_confidences_aggregated.json
        output_dir/{query_id}/seed_{seed}/timing.json

    Returns a dict with "structures", "confidence", "error", and "timing" keys
    (the first three match the Boltz2 response format). ``timing`` is a dict
    keyed by ``sample_id``; each sample carries its seed's ``timing.json``
    (``{"runtime_s": ...}``). When ``include_atom_confidences`` is set, also
    reads each sample's per-atom ``*_confidences.json`` into an
    ``atom_confidence`` map.
    """
    result: dict[str, Any] = {
        "structures": [],
        "confidence": {},
        "error": False,
        "timing": {},
    }
    if include_atom_confidences:
        result["atom_confidence"] = {}

    for query_dir in sorted(output_dir.iterdir()):
        if not query_dir.is_dir():
            continue
        for seed_dir in sorted(query_dir.iterdir()):
            if not seed_dir.is_dir() or not seed_dir.name.startswith("seed_"):
                continue

            seed_timing = _read_seed_timing(seed_dir)

            for model_file in sorted(seed_dir.glob("*_model.*")):
                sample_id = model_file.stem.removesuffix("_model")
                fmt = model_file.suffix.lstrip(".")
                if fmt == "cif":
                    fmt = "mmcif"

                result["structures"].append(
                    {
                        "record_id": sample_id,
                        "format": fmt,
                        "content": model_file.read_text(),
                    }
                )

                conf_file = seed_dir / f"{sample_id}_confidences_aggregated.json"
                if conf_file.exists():
                    with conf_file.open() as f:
                        conf_data = json.load(f)
                    result["confidence"][sample_id] = conf_data

                if include_atom_confidences:
                    atom_file = seed_dir / f"{sample_id}_confidences.json"
                    if atom_file.exists():
                        with atom_file.open() as f:
                            result["atom_confidence"][sample_id] = json.load(f)

                if seed_timing is not None:
                    result["timing"][sample_id] = seed_timing

    return result


def run_openfold3_prediction(
    body: dict[str, Any],
    cache: Path,
    *,
    model: Optional["pl.LightningModule"] = None,
    num_diffusion_samples: int = 1,
    num_model_seeds: int | None = None,
    seeds: list[int] | None = None,
    use_msa_server: bool = True,
    use_templates: bool = True,
    output_format: str = "mmcif",
    num_workers: int = 0,
    accelerator: str = "gpu",
    include_atom_confidences: bool = False,
    msa_deadline: float | None = None,
) -> dict[str, Any]:
    """Run OpenFold3 prediction from a JSON request body.

    ``body["queries"]`` is OF3's query format (chains may carry inline
    ``main_msa``/``paired_msa``). ``model`` is the shared startup-loaded model,
    reused to skip the checkpoint reload. Returns a JSON-serializable dict with
    "structures", "confidence", and "error". When ``include_atom_confidences``
    is set, each sample's per-atom ``*_confidences.json`` is also included under
    an ``atom_confidence`` map.

    The result dict carries "structures", "confidence", "error", and "timing"
    keys; when ``include_atom_confidences`` is True it also includes
    "atom_confidence".

    Seeds: ``seeds`` is used as-is unless ``num_model_seeds`` is set, which makes
    OF3 regenerate them (mirrors run_openfold.py). Total structures =
    len(seeds) * num_diffusion_samples.

    ``msa_deadline`` is a ``time.monotonic()`` reading past which MSA-server work
    gives up with ``MsaServerTimeout`` instead of retrying indefinitely. Only
    meaningful with ``use_msa_server``.
    """
    from openfold3.entry_points.experiment_runner import InferenceExperimentRunner
    from openfold3.entry_points.validator import InferenceExperimentConfig
    from openfold3.projects.of3_all_atom.config.inference_query_format import (
        InferenceQuerySet,
    )

    if seeds is None:
        seeds = [42]

    query_data = {
        "seeds": seeds,
        "queries": body.get("queries", {}),
    }

    with tempfile.TemporaryDirectory(prefix="openfold3_run_") as work_dir:
        work_path = Path(work_dir)
        output_dir = work_path / "output"
        output_dir.mkdir()

        _materialize_inline_msas(query_data["queries"], work_path / "msas")

        query_json = work_path / "query.json"
        with query_json.open("w") as f:
            json.dump(query_data, f)

        query_set = InferenceQuerySet.from_json(query_json)

        structure_format = "pdb" if output_format == "pdb" else "cif"

        runner_args = _get_rocm_runner_args()
        runner_args["output_writer_settings"] = {
            "structure_format": structure_format,
        }
        runner_args["data_module_args"] = {
            "num_workers": num_workers,
        }
        runner_args.update(_scratch_runner_args(work_path))
        runner_args.setdefault("msa_computation_settings", {}).update(_msa_server_args())

        expt_config = InferenceExperimentConfig(
            cache_path=cache,
            **runner_args,
        )
        # The runner reads seeds from experiment_settings.seeds (not the query
        # set); num_model_seeds, if set, overrides them. See run_openfold.py.
        expt_config.experiment_settings.seeds = seeds

        expt_runner = InferenceExperimentRunner(
            expt_config,
            num_diffusion_samples=num_diffusion_samples,
            num_model_seeds=num_model_seeds,
            use_msa_server=use_msa_server,
            use_templates=use_templates,
            output_dir=output_dir,
        )

        if model is not None:
            # Drop the previous request's trainer so the new one takes over
            # (avoids leaking a stale log_dir into this prediction).
            if hasattr(model, "_trainer"):
                model._trainer = None
            # The reused model reads no_full_rollout_samples from its own config,
            # which the per-request runner doesn't reach -- set it here.
            model.config.update(
                {"architecture": {"shared": {"diffusion": {"no_full_rollout_samples": num_diffusion_samples}}}}
            )
            expt_runner.__dict__["lightning_module"] = model
        else:
            expt_runner.setup()

        # The MSA server is queried from inside run(), on this thread, so the
        # deadline has to span the whole call rather than a preceding stage.
        with bounded_msa_server(msa_deadline if use_msa_server else None):
            expt_runner.run(query_set)
        expt_runner.cleanup()

        result = _parse_output_dir(output_dir, include_atom_confidences=include_atom_confidences)

        if not result["structures"]:
            return {
                "error": True,
                "message": "No predictions returned.",
                "structures": [],
                "confidence": {},
                "timing": result["timing"],
            }

        return result
