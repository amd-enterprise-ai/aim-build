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

import json
import logging
import tempfile
from pathlib import Path
from typing import TYPE_CHECKING, Any, Optional

if TYPE_CHECKING:
    import pytorch_lightning as pl

logger = logging.getLogger(__name__)

# Registry name used by openfold3 to look up the checkpoint filename (of3-p2-155k.pt).
# Available checkpoints: https://huggingface.co/OpenFold/OpenFold3/tree/main/checkpoints
DEFAULT_CHECKPOINT_REGISTRY_NAME = "openfold3-p2-155k"


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


def _ensure_model_parameters(cache: Path) -> None:
    """Download model checkpoint if not already cached (idempotent)."""
    from openfold3.entry_points.parameters import download_model_parameters

    cache.mkdir(parents=True, exist_ok=True)
    download_model_parameters(
        download_dir=cache,
        parameter_name=DEFAULT_CHECKPOINT_REGISTRY_NAME,
        skip_confirmation=True,
    )


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

    expt_config = InferenceExperimentConfig(cache_path=cache, **runner_args)

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
