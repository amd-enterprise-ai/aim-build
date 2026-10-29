# Copyright © Advanced Micro Devices, Inc., or its affiliates.
#
# SPDX-License-Identifier: MIT

"""llama.cpp engine (``llama-server``, ZenDNN backend) for EPYC.

``llama-server -m`` needs one ``.gguf`` file. The model directory holds either
GGUF files or a Hugging Face checkpoint, which is converted to GGUF once on first
launch. A model not yet in the AIM cache is downloaded on ``serve``, as vLLM
downloads its own.
"""

from __future__ import annotations

import glob
import json
import logging
import os
import re
import subprocess
import sys
from typing import TYPE_CHECKING, Any, ClassVar, Literal, Optional

from aim_common.object_model import Precision
from aim_runtime.cpu_detector import CpuDetector
from aim_runtime.engines.base import BaseEngine
from aim_runtime.engines.engine_args_models import EngineArgsFormat, EngineArgsModel
from aim_runtime.model_cache_resolver import ModelCacheResolver

if TYPE_CHECKING:
    from aim_runtime.model_cache_resolver import ResolvedModelPath
    from aim_runtime.object_model import Profile

logger = logging.getLogger(__name__)

# From the same llama.cpp commit as llama-server (docker/Dockerfile.aim-epyc-base).
CONVERTER_PATH = "/opt/llamacpp/convert/convert_hf_to_gguf.py"

# GGUF type the converter writes for each profile precision.
GGUF_OUTTYPE_BY_PRECISION: dict[Precision, str] = {
    Precision.BF16: "bf16",
    Precision.FP16: "f16",
    Precision.FP32: "f32",
    Precision.INT8: "q8_0",
}

KvCacheType = Literal["f32", "f16", "bf16", "q8_0", "q4_0", "q4_1", "iq4_nl", "q5_0", "q5_1"]

# Bytes per element for each KV cache type (block-quantized types: block bytes / 32).
KV_CACHE_TYPE_BYTES: dict[str, float] = {
    "f32": 4,
    "f16": 2,
    "bf16": 2,
    "q8_0": 34 / 32,
    "q4_0": 18 / 32,
    "q4_1": 20 / 32,
    "iq4_nl": 18 / 32,
    "q5_0": 22 / 32,
    "q5_1": 24 / 32,
}

# Later parts of a split GGUF (llama-server loads them all from the first part).
SPLIT_GGUF_LATER_PART = re.compile(r"-(?!00001-)\d{5}-of-\d{5}\.gguf$")

# llama.cpp pads the context to a multiple of this.
CTX_SIZE_ALIGNMENT = 256

# llama-server's short option names (pinned commit) and their long forms. AIM
# serializes every key as --<key>, which llama-server rejects for short names.
SHORT_FLAG_NAMES: dict[str, str] = {
    "a": "alias",
    "b": "batch-size",
    "c": "ctx-size",
    "C": "cpu-mask",
    "cb": "cont-batching",
    "Cr": "cpu-range",
    "ctk": "cache-type-k",
    "ctv": "cache-type-v",
    "fa": "flash-attn",
    "kvu": "kv-unified",
    "lm": "load-mode",
    "n": "n-predict",
    "np": "parallel",
    "t": "threads",
    "tb": "threads-batch",
    "to": "timeout",
    "ub": "ubatch-size",
}


class LlamaCppEngineArgsModel(EngineArgsModel):
    """Pydantic model for ``llama-server`` CLI arguments.

    Pydantic-only validation (no Python CLI parser to delegate to). Fields cover
    the options AIM sets or rewrites, as of the llama.cpp commit pinned in
    ``docker/Dockerfile.aim-epyc-base``; other flags pass through unvalidated.
    """

    # Serving
    host: str | None = None
    port: int | None = None
    timeout: int | None = None
    alias: str | None = None

    # AIM-only, not llama-server flags: the .gguf file to load, and the KV cache
    # budget in GiB (as VLLM_CPU_KVCACHE_SPACE) that ctx-size is derived from.
    gguf_filename: str | None = None
    kv_cache_space: float | None = None

    # Context / batching
    ctx_size: int | None = None
    batch_size: int | None = None
    ubatch_size: int | None = None
    n_predict: int | None = None
    parallel: int | None = None
    cont_batching: bool | None = None
    kv_unified: bool | None = None
    cache_type_k: KvCacheType | None = None
    cache_type_v: KvCacheType | None = None
    flash_attn: Literal["on", "off", "auto"] | None = None

    # CPU / memory placement
    threads: int | None = None
    threads_batch: int | None = None
    cpu_mask: str | None = None
    cpu_range: str | None = None
    cpu_strict: Literal[0, 1] | None = None
    load_mode: Literal["auto", "none", "mmap", "mlock", "mmap+mlock"] | None = None


class LlamaCppEngine(BaseEngine):
    """llama.cpp-backed serving engine (``llama-server``)."""

    ARGS_MODEL: ClassVar[Optional[type[EngineArgsModel]]] = LlamaCppEngineArgsModel
    ARGS_FORMAT: ClassVar[EngineArgsFormat] = EngineArgsFormat.STANDARD

    def apply_engine_defaults(self, engine_args: dict[str, Any], served_model_names: list[str]) -> None:
        """Serve under AIM's model names; one thread per available core, pinned when the cpuset is known.

        As for vLLM: otherwise llama-server reports the .gguf file name as the
        model and sizes its pool from the host's core count. Explicit values from
        the profile or ``AIM_ENGINE_ARGS`` win.
        """
        if "alias" not in engine_args and served_model_names:
            engine_args["alias"] = ",".join(served_model_names)

        detector = CpuDetector()
        if "threads" not in engine_args:
            engine_args["threads"] = detector.available_cores

        pinned = any(key in engine_args for key in ("cpu-mask", "cpu_mask", "cpu-range", "cpu_range"))
        if not pinned and detector.cpuset_bind:
            cpus = CpuDetector._parse_cpuset_to_set(detector.cpuset_bind)
            engine_args["cpu-mask"] = format(sum(1 << cpu for cpu in cpus), "x")
            if "cpu-strict" not in engine_args and "cpu_strict" not in engine_args:
                engine_args["cpu-strict"] = 1

    def resolve_model_path(self, resolved: "ResolvedModelPath", engine_args: dict[str, Any], profile: "Profile") -> str:
        """Resolve the model directory to the ``.gguf`` file to load.

        The directory is the cached one, or a local path given as the model id
        (which may also name a ``.gguf`` file directly). In order:
        ``gguf_filename`` (popped from ``engine_args``); for a Hugging Face
        checkpoint, ``<model>-<outtype>.gguf`` for the profile's precision;
        otherwise the directory's ``.gguf`` file (see :func:`_pick_gguf`).
        ``kv_cache_space`` (also popped) then sets ``ctx-size``, unless
        ``ctx-size`` is given explicitly. Short option names (e.g. ``ctk``) are
        first rewritten to their long form.

        Raises:
            ValueError: if the model is not in the cache, the file cannot be
                determined unambiguously, or the KV cache size cannot be derived.
        """
        if _expand_short_flags(engine_args):
            # Values given under short names skipped validation; check them now.
            self.validate_engine_args(engine_args)
        gguf_filename = _pop_arg(engine_args, "gguf_filename")
        kv_cache_space = _pop_arg(engine_args, "kv_cache_space")

        model_path = self._resolve_gguf_path(resolved, gguf_filename, profile)

        if kv_cache_space is not None:
            if _get_arg(engine_args, "ctx_size") is not None:
                logger.info("ctx-size is set explicitly; ignoring kv_cache_space")
            else:
                engine_args["ctx-size"] = self._ctx_size_for_kv_space(
                    float(kv_cache_space), model_path, resolved.path, engine_args
                )
        return model_path

    def _resolve_gguf_path(
        self, resolved: "ResolvedModelPath", gguf_filename: Optional[str], profile: "Profile"
    ) -> str:
        """Return the ``.gguf`` file to load; see :meth:`resolve_model_path`."""
        # Outside the AIM cache, the model id may itself be a local directory or .gguf file.
        if os.path.isfile(resolved.path) and resolved.path.endswith(".gguf"):
            return resolved.path
        if not os.path.isdir(resolved.path):
            raise ValueError(
                f"Model '{resolved.model_id}' is not in the AIM model cache yet; `serve` downloads it, or run "
                f"`./entrypoint.py download-to-cache --model-id hf://{resolved.model_id}` (without --use-hf-cache), "
                f"mount it at <AIM_CACHE_PATH>/{resolved.model_id}, or set the model id to a local directory or "
                f".gguf file."
            )

        if gguf_filename:
            candidate = os.path.join(resolved.path, gguf_filename)
            if not os.path.isfile(candidate):
                raise ValueError(f"gguf_filename '{gguf_filename}' not found under '{resolved.path}'.")
            return candidate

        if self._is_hf_checkpoint(resolved.path):
            return self._converted_gguf_path(resolved.path, profile)

        names = sorted(
            os.path.relpath(path, resolved.path)
            for path in glob.glob(os.path.join(resolved.path, "**", "*.gguf"), recursive=True)
        )
        if not names:
            raise ValueError(f"No .gguf file or Hugging Face checkpoint found under '{resolved.path}'.")
        return os.path.join(resolved.path, _pick_gguf(names, profile.metadata.precision, resolved.path))

    def prepare_model(self, model_id: str, profile: "Profile") -> None:
        """Make sure a local GGUF exists for ``model_id``: download it, and convert a checkpoint.

        A model neither in the AIM cache nor a local path is downloaded from
        Hugging Face into the cache, as ``download-to-cache`` does: only the
        ``.gguf`` the precision selects (see :func:`_pick_gguf`), or a checkpoint
        without Meta's ``original/`` weights. A checkpoint is then converted to
        GGUF once, next to it, and renamed into place only when complete.

        Raises:
            ValueError: if the model cannot be found or downloaded as a GGUF or checkpoint.
            RuntimeError: if the download or the conversion fails.
        """
        resolved = ModelCacheResolver(self.config.cache_path).resolve_model_path(model_id)
        model_dir = resolved.path if resolved else model_id
        if not os.path.exists(model_dir):
            model_dir = self._download(model_id, profile)
        if not (os.path.isdir(model_dir) and self._is_hf_checkpoint(model_dir)):
            return
        target = self._converted_gguf_path(model_dir, profile)
        if os.path.isfile(target):
            return
        if not os.access(model_dir, os.W_OK):
            raise RuntimeError(
                f"Cannot convert '{model_dir}' to GGUF: the directory is not writable. Mount the "
                f"model cache read-write for the first launch, or provide a .gguf file in it."
            )

        outtype = GGUF_OUTTYPE_BY_PRECISION[profile.metadata.precision]
        partial = f"{target}.partial"
        command = [sys.executable, CONVERTER_PATH, model_dir, "--outtype", outtype, "--outfile", partial]
        logger.info(f"Converting Hugging Face checkpoint to GGUF ({outtype}): {target}")
        try:
            subprocess.run(command, check=True)
        except (OSError, subprocess.CalledProcessError) as e:
            if os.path.exists(partial):
                os.remove(partial)
            raise RuntimeError(f"GGUF conversion of '{model_dir}' failed: {e}") from e
        os.replace(partial, target)
        logger.info(f"GGUF conversion complete: {target}")

    def _download(self, model_id: str, profile: "Profile") -> str:
        """Download ``model_id`` from Hugging Face into the AIM cache and return its directory."""
        from huggingface_hub import HfApi

        from aim_runtime.model_storage import StorageBackendRegistry

        if model_id.startswith("/") or model_id.count("/") != 1:
            raise ValueError(f"Model '{model_id}' was not found and is not a Hugging Face model id.")
        files = HfApi(token=os.environ.get("HF_TOKEN")).list_repo_files(model_id)

        allow_patterns, ignore_patterns = None, None
        ggufs = [name for name in files if name.endswith(".gguf")]
        if ggufs:
            engine_args = {**(profile.engine_args or {}), **(self.config.engine_args_override or {})}
            name = _get_arg(engine_args, "gguf_filename") or _pick_gguf(ggufs, profile.metadata.precision, model_id)
            if name not in ggufs:
                raise ValueError(f"gguf_filename '{name}' not found in Hugging Face repository '{model_id}'.")
            # A split GGUF downloads every part.
            allow_patterns = [name.replace("-00001-of-", "-*-of-")]
        elif "config.json" in files and any(
            name.endswith(".safetensors") or re.fullmatch(r"pytorch_model.*\.bin", name) for name in files
        ):
            ignore_patterns = ["original/*"]
        else:
            raise ValueError(
                f"Hugging Face repository '{model_id}' has neither .gguf files nor a checkpoint llama.cpp can convert."
            )

        logger.info(f"Model '{model_id}' is not in the AIM cache; downloading it to {self.config.cache_path}")
        return StorageBackendRegistry().download(
            f"hf://{model_id}", self.config.cache_path, allow_patterns=allow_patterns, ignore_patterns=ignore_patterns
        )

    @staticmethod
    def _ctx_size_for_kv_space(
        kv_cache_space_gib: float, model_path: str, model_dir: str, engine_args: dict[str, Any]
    ) -> int:
        """Number of context tokens whose KV cache fits in ``kv_cache_space_gib``.

        Uses the GGUF metadata, or ``config.json`` for a checkpoint not yet
        converted. Every layer is counted as full attention, so hybrid and
        sliding-window models get a conservative (smaller) context.
        """
        k_bytes = KV_CACHE_TYPE_BYTES[_get_arg(engine_args, "cache_type_k") or "f16"]
        v_bytes = KV_CACHE_TYPE_BYTES[_get_arg(engine_args, "cache_type_v") or "f16"]
        try:
            if os.path.isfile(model_path):
                layers = _kv_layers_from_gguf(model_path)
            else:
                layers = _kv_layers_from_hf_config(os.path.join(model_dir, "config.json"))
            bytes_per_token = sum(heads * (k_len * k_bytes + v_len * v_bytes) for heads, k_len, v_len in layers)
        except (KeyError, TypeError, ValueError, ZeroDivisionError) as e:
            raise ValueError(
                f"Cannot derive the KV cache size per token for '{model_path}' ({e!r}); set ctx-size instead."
            ) from e
        if bytes_per_token <= 0:
            raise ValueError(f"Cannot derive the KV cache size per token for '{model_path}'; set ctx-size instead.")
        ctx_size = int(kv_cache_space_gib * 1024**3 // bytes_per_token) // CTX_SIZE_ALIGNMENT * CTX_SIZE_ALIGNMENT
        if ctx_size < CTX_SIZE_ALIGNMENT:
            raise ValueError(f"kv_cache_space of {kv_cache_space_gib} GiB is too small for '{model_path}'.")
        logger.info(
            f"KV cache: {kv_cache_space_gib} GiB at {bytes_per_token / 1024:.1f} KiB/token -> ctx-size {ctx_size}"
        )
        return ctx_size

    @staticmethod
    def _is_hf_checkpoint(path: str) -> bool:
        """Whether ``path`` holds a Hugging Face checkpoint in a format the converter reads."""
        return os.path.isfile(os.path.join(path, "config.json")) and bool(
            glob.glob(os.path.join(path, "*.safetensors")) or glob.glob(os.path.join(path, "pytorch_model*.bin"))
        )

    def _converted_gguf_path(self, checkpoint_dir: str, profile: "Profile") -> str:
        """Return the GGUF converted (by :meth:`prepare_model`) from ``checkpoint_dir`` for the profile's precision."""
        precision = profile.metadata.precision
        outtype = GGUF_OUTTYPE_BY_PRECISION.get(precision)
        if outtype is None:
            supported = ", ".join(p.value for p in GGUF_OUTTYPE_BY_PRECISION)
            raise ValueError(
                f"Cannot convert a Hugging Face checkpoint to GGUF for precision '{precision}'. "
                f"Supported: {supported}; or provide a .gguf file and set `gguf_filename`."
            )
        return os.path.join(checkpoint_dir, f"{os.path.basename(checkpoint_dir)}-{outtype}.gguf")


def _pick_gguf(names: list[str], precision: Precision, where: str) -> str:
    """Pick the model ``.gguf`` among ``names``: the only one, else the one named for ``precision``.

    Later parts of a split GGUF and vision projectors (``mmproj``) are not
    candidates. With several, the file whose name has the precision's GGUF type
    as a token (e.g. ``q8_0`` for ``int8``) is chosen.
    """
    candidates = [
        name
        for name in names
        if not SPLIT_GGUF_LATER_PART.search(name) and "mmproj" not in os.path.basename(name).lower()
    ]
    outtype = GGUF_OUTTYPE_BY_PRECISION.get(precision)
    if len(candidates) > 1 and outtype:
        token = re.compile(rf"(?<![a-z0-9]){re.escape(outtype)}(?![a-z0-9])")
        candidates = [name for name in candidates if token.search(os.path.basename(name).lower())]
    if len(candidates) == 1:
        return candidates[0]
    if not candidates:
        raise ValueError(f"No .gguf file for precision '{precision}' in '{where}'. Set `gguf_filename` to pick one.")
    raise ValueError(
        f"Multiple .gguf files in '{where}': {candidates}. Set `gguf_filename` in engine_args to pick one."
    )


def _expand_short_flags(engine_args: dict[str, Any]) -> bool:
    """Rewrite llama-server short option names to their long form, in place.

    Keys keep the merge order (profile, then AIM_ENGINE_ARGS), so when a short
    and a long name set the same option the later one wins, as in the merge.
    Returns whether any short name was found.
    """
    if not any(key in SHORT_FLAG_NAMES for key in engine_args):
        return False
    long_names = set(SHORT_FLAG_NAMES.values())
    items = list(engine_args.items())
    engine_args.clear()
    for key, value in items:
        long = SHORT_FLAG_NAMES.get(key) or key.replace("_", "-")
        if long in long_names:
            engine_args.pop(long.replace("-", "_"), None)
            engine_args.pop(long, None)
            engine_args[long] = value
        else:
            engine_args[key] = value
    return True


def _get_arg(engine_args: dict[str, Any], name: str) -> Any:
    """Value of an engine argument given in either snake_case or kebab-case."""
    return engine_args.get(name, engine_args.get(name.replace("_", "-")))


def _pop_arg(engine_args: dict[str, Any], name: str) -> Any:
    """Remove an engine argument in either spelling and return its value."""
    snake, kebab = engine_args.pop(name, None), engine_args.pop(name.replace("_", "-"), None)
    return snake if snake is not None else kebab


def _kv_layers_from_gguf(path: str) -> list[tuple[int, int, int]]:
    """(KV heads, key length, value length) per layer, from GGUF metadata."""
    from gguf import GGUFReader

    fields = GGUFReader(path).fields

    def value(key: str, default: Any = None) -> Any:
        field = fields.get(key)
        return default if field is None else field.contents()

    arch = value("general.architecture")
    n_layer = value(f"{arch}.block_count")
    n_head = value(f"{arch}.attention.head_count")
    n_head_kv = value(f"{arch}.attention.head_count_kv", n_head)
    head_dim = value(f"{arch}.embedding_length", 0) // (max(n_head) if isinstance(n_head, list) else n_head or 1)
    k_len = value(f"{arch}.attention.key_length", head_dim)
    v_len = value(f"{arch}.attention.value_length", head_dim)
    heads_per_layer = n_head_kv if isinstance(n_head_kv, list) else [n_head_kv] * n_layer
    return [(heads, k_len, v_len) for heads in heads_per_layer]


def _kv_layers_from_hf_config(path: str) -> list[tuple[int, int, int]]:
    """(KV heads, key length, value length) per layer, from a Hugging Face ``config.json``."""
    with open(path) as f:
        config = json.load(f)
    config = config.get("text_config", config)
    n_head = config["num_attention_heads"]
    n_head_kv = config.get("num_key_value_heads") or n_head
    head_dim = config.get("head_dim") or config["hidden_size"] // n_head
    return [(n_head_kv, head_dim, head_dim)] * config["num_hidden_layers"]
