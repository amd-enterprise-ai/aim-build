# Copyright © Advanced Micro Devices, Inc., or its affiliates.
#
# SPDX-License-Identifier: MIT

"""vLLM engine.

Owns vLLM-specific runtime behavior: the ``served-model-name`` default,
``VLLM_*`` env-var validation, and (in a later change) the LoRA adapter
contract. Note the run-vs-check split: ``VllmEngine`` is *how to run* a vLLM
model; a future ``EngineChecks`` provider is *how to check* it.

The vLLM argument-validation model (:class:`VllmEngineArgsModel`) and the
``validate_vllm_env_vars`` helper live here too, next to the engine they
describe; they are re-exported from ``aim_runtime.engines``.
"""

from __future__ import annotations

import functools
import logging
from typing import TYPE_CHECKING, Any, ClassVar, Literal, Optional

from pydantic import model_validator
from pydantic.functional_validators import ModelWrapValidatorHandler

from aim_common.object_model import AdapterToken
from aim_runtime.config import AdapterMode
from aim_runtime.engines.adapters import enumerate_adapters
from aim_runtime.engines.base import BaseEngine
from aim_runtime.engines.engine_args_models import (
    EngineArgsFormat,
    EngineArgsModel,
    engine_args_to_cli_list,
)

if TYPE_CHECKING:
    from aim_runtime.object_model import Profile

logger = logging.getLogger(__name__)


class VllmEngineArgsModel(EngineArgsModel):
    """Pydantic model mirroring vLLM EngineArgs for offline validation.

    When vLLM is installed the ``@model_validator`` delegates to vLLM's full
    CLI arg parser (engine + frontend args) for authoritative validation.
    When vLLM is not installed, Pydantic's own field-type checking is used
    as a fallback.

    Class attributes:
        _vllm_parser: Lazily builds and caches the vLLM CLI arg parser
            (or ``None`` when vLLM is not installed or its imports are broken).
            Lazy so that importing this module does not trigger a vLLM import
            and its logging side-effects until validation is first requested.
            Intended for use by tests and callers that need to gate behaviour
            on vLLM presence without importing vLLM themselves.
        _vllm_cli_argv_prefix: Argv tokens prepended before engine flags when
            delegating to a vLLM CLI parser (subclasses override, e.g. Omni).
        _vllm_validation_label: Short label for :class:`ValueError` messages
            when native CLI validation fails.
    """

    _vllm_cli_argv_prefix: ClassVar[tuple[str, ...]] = ()
    _vllm_validation_label: ClassVar[str] = "vLLM"

    @staticmethod
    @functools.lru_cache(maxsize=None)
    def _vllm_parser() -> Any:
        """Return the vLLM CLI arg parser, or None if unavailable.

        Cached via lru_cache so the import and parser construction happen at
        most once per process. Performs a real import so that a broken vLLM
        install (e.g. mismatched transitive dependencies) is treated as
        unavailable rather than silently falling back to Pydantic-only
        validation.
        """
        try:
            from vllm.entrypoints.openai.cli_args import make_arg_parser
            from vllm.utils.argparse_utils import FlexibleArgumentParser

            def _raise_on_error(message: str) -> None:
                raise ValueError(message)

            parser = make_arg_parser(FlexibleArgumentParser())
            parser.error = _raise_on_error  # type: ignore[assignment]
            return parser
        except Exception:
            return None

    @model_validator(mode="wrap")
    @classmethod
    def _validate_with_vllm_if_available(cls, data: Any, handler: ModelWrapValidatorHandler) -> Any:
        """Delegate to vLLM's CLI arg parser when the library is present.

        When vLLM is absent ``handler(data)`` is called, which runs the full
        Pydantic field validation pipeline. This acts as a best-effort
        fallback when the authoritative validator is unavailable.

        When vLLM *is* present, the input dict is only validated by vLLM's own
        argument parser, i.e., handler function is not called. On success,
        ``model_construct()`` is used to build the model instance directly,
        bypassing Pydantic's field validation.

        This is intentional: vLLM is the authoritative source of truth, and
        running Pydantic's ``Literal[...]`` annotations afterwards could
        reject values that vLLM has already accepted (e.g. a dtype added in a
        newer vLLM release that is not yet reflected in the field definitions).

        See: https://docs.pydantic.dev/latest/concepts/validators/#model-validators
        """
        parser = cls._vllm_parser()
        if not isinstance(data, dict) or parser is None:
            return handler(data)
        # Normalize keys using the model's alias_generator so that snake_case
        # keys from YAML (e.g. "distributed_executor_backend") become the CLI
        # flag format vLLM's argparse expects ("--distributed-executor-backend").
        # Kebab-case keys are already correct and pass through unchanged.
        alias_gen = cls.model_config.get("alias_generator")
        normalized = {alias_gen(k): v for k, v in data.items()} if alias_gen else data
        cli_args = engine_args_to_cli_list(normalized)
        label = cls._vllm_validation_label
        argv = [*cls._vllm_cli_argv_prefix, *cli_args]
        try:
            parser.parse_args(argv)
        except SystemExit as exc:
            raise ValueError(f"{label} engine_args validation failed (parser exit code {exc.code})") from exc
        except ValueError as exc:
            raise ValueError(f"{label} engine_args validation failed: {exc}") from exc
        # vLLM validated successfully. Build the model without re-validating
        return cls.model_construct(**normalized)

    disable_log_stats: bool | None = None
    model: str | None = None
    runner: Literal["auto", "draft", "generate", "pooling"] | None = None
    convert: Literal["auto", "classify", "embed", "none", "reward"] | None = None
    task: (
        Literal["auto", "classify", "draft", "embed", "embedding", "generate", "reward", "score", "transcription"]
        | None
    ) = None
    tokenizer: str | None = None
    tokenizer_mode: Literal["auto", "custom", "mistral", "slow"] | None = None
    tokenizer_revision: str | None = None
    trust_remote_code: bool | None = None
    dtype: Literal["auto", "bfloat16", "float", "float16", "float32", "half"] | None = None
    seed: int | None = None
    hf_config_path: str | None = None
    allowed_local_media_path: str | None = None
    revision: str | None = None
    code_revision: str | None = None
    rope_scaling: dict[str, Any] | None = None
    rope_theta: float | None = None
    max_model_len: int | None = None
    quantization: str | None = None
    enforce_eager: bool | None = None
    max_logprobs: int | None = None
    logprobs_mode: Literal["raw_logits", "raw_logprobs", "processed_logits", "processed_logprobs"] | None = None
    disable_sliding_window: bool | None = None
    disable_cascade_attn: bool | None = None
    skip_tokenizer_init: bool | None = None
    enable_prompt_embeds: bool | None = None
    served_model_name: str | list[str] | None = None
    disable_async_output_proc: bool | None = None
    config_format: Literal["auto", "hf", "mistral"] | None = None
    hf_token: str | None = None
    hf_overrides: dict[str, Any] | None = None
    override_neuron_config: dict[str, Any] | None = None
    override_pooler_config: dict[str, Any] | None = None
    logits_processor_pattern: str | None = None
    generation_config: str | None = None
    override_generation_config: dict[str, Any] | None = None
    enable_sleep_mode: bool | None = None
    model_impl: Literal["auto", "vllm", "transformers"] | None = None
    override_attention_dtype: str | None = None
    logits_processors: str | None = None
    load_format: (
        Literal[
            "auto",
            "pt",
            "safetensors",
            "npcache",
            "dummy",
            "tensorizer",
            "runai_streamer",
            "bitsandbytes",
            "sharded_state",
            "gguf",
            "mistral",
        ]
        | None
    ) = None
    download_dir: str | None = None
    model_loader_extra_config: dict[str, Any] | None = None
    ignore_patterns: str | None = None
    use_tqdm_on_load: bool | None = None
    pt_load_map_location: str | None = None
    guided_decoding_backend: Literal["auto", "guidance", "outlines", "xgrammar"] | None = None
    guided_decoding_disable_fallback: bool | None = None
    guided_decoding_disable_any_whitespace: bool | None = None
    guided_decoding_disable_additional_properties: bool | None = None
    reasoning_parser: str | None = None
    distributed_executor_backend: str | None = None
    block_size: Literal[1, 8, 16, 32, 64, 128] | None = None
    gpu_memory_utilization: float | None = None
    kv_cache_dtype: Literal["auto", "fp8", "fp8_e4m3", "fp8_e5m2", "fp8_inc"] | None = None
    prefix_caching_hash_algo: Literal["builtin", "sha256", "sha256_cbor_64bit"] | None = None
    mm_encoder_tp_mode: Literal["data", "weights"] | None = None
    lora_dtype: Literal["auto", "bfloat16", "float16"] | None = None
    collect_detailed_traces: (
        Literal[
            "all",
            "model",
            "worker",
            "model,worker",
            "model,all",
            "worker,model",
            "worker,all",
            "all,model",
            "all,worker",
        ]
        | None
    ) = None
    preemption_mode: Literal["recompute", "swap"] | None = None
    scheduling_policy: Literal["fcfs", "priority"] | None = None
    data_parallel_backend: Literal["mp", "ray"] | None = None
    compilation_config: dict[str, Any] | None = None
    additional_config: dict[str, Any] | None = None
    pipeline_parallel_size: int | None = None
    tensor_parallel_size: int | None = None
    data_parallel_size: int | None = None
    data_parallel_rank: int | None = None
    data_parallel_start_rank: int | None = None
    data_parallel_size_local: int | None = None
    data_parallel_address: str | None = None
    data_parallel_rpc_port: int | None = None
    data_parallel_hybrid_lb: bool | None = None
    enable_expert_parallel: bool | None = None
    enable_eplb: bool | None = None
    eplb_config: dict[str, Any] | None = None
    num_redundant_experts: int | None = None
    eplb_window_size: int | None = None
    eplb_step_interval: int | None = None
    eplb_log_balancedness: bool | None = None
    max_parallel_loading_workers: int | None = None
    ray_workers_use_nsight: bool | None = None
    disable_custom_all_reduce: bool | None = None
    worker_cls: str | None = None
    worker_extension_cls: str | None = None
    enable_multimodal_encoder_data_parallel: bool | None = None
    swap_space: int | None = None
    num_gpu_blocks_override: int | None = None
    enable_prefix_caching: bool | None = None
    cpu_offload_gb: int | None = None
    calculate_kv_scales: bool | None = None
    kv_sharing_fast_prefill: bool | None = None
    mamba_cache_dtype: Literal["auto", "float32"] | None = None
    mamba_ssm_cache_dtype: Literal["auto", "float32"] | None = None
    limit_mm_per_prompt: dict[str, Any] | None = None
    media_io_kwargs: dict[str, Any] | None = None
    mm_processor_kwargs: dict[str, Any] | None = None
    mm_processor_cache_gb: int | None = None
    disable_mm_preprocessor_cache: bool | None = None
    interleave_mm_strings: bool | None = None
    skip_mm_profiling: bool | None = None
    enable_lora: bool | None = None
    enable_lora_bias: bool | None = None
    max_loras: int | None = None
    max_lora_rank: int | None = None
    lora_extra_vocab_size: int | None = None
    max_cpu_loras: int | None = None
    fully_sharded_loras: bool | None = None
    # --lora-modules accepts a list of "name=path" (or JSON) tokens; vLLM's
    # LoRAParserAction parses each. Serialized as `--lora-modules n1=p1 n2=p2`.
    lora_modules: list[str] | None = None
    default_mm_loras: dict[str, Any] | None = None
    show_hidden_metrics_for_version: str | None = None
    otlp_traces_endpoint: str | None = None
    max_num_batched_tokens: int | None = None
    max_num_seqs: int | None = None
    max_num_partial_prefills: int | None = None
    max_long_partial_prefills: int | None = None
    cuda_graph_sizes: list[Any] | None = None
    long_prefill_token_threshold: int | None = None
    num_lookahead_slots: int | None = None
    scheduler_delay_factor: float | None = None
    enable_chunked_prefill: bool | None = None
    disable_chunked_mm_input: bool | None = None
    scheduler_cls: str | None = None
    disable_hybrid_kv_cache_manager: bool | None = None
    async_scheduling: bool | None = None
    speculative_config: dict[str, Any] | None = None
    kv_transfer_config: dict[str, Any] | None = None
    kv_events_config: dict[str, Any] | None = None
    enable_log_requests: bool | None = None
    disable_log_requests: bool | None = None

    # CLI negation flags — these are passed as `--no-<flag>` with null value in YAML.
    # They must only accept null (None); any other value is invalid.
    no_async_scheduling: None = None
    no_calculate_kv_scales: None = None
    no_data_parallel_hybrid_lb: None = None
    no_disable_cascade_attn: None = None
    no_disable_chunked_mm_input: None = None
    no_disable_custom_all_reduce: None = None
    no_disable_hybrid_kv_cache_manager: None = None
    no_disable_log_requests: None = None
    no_disable_sliding_window: None = None
    no_enable_chunked_prefill: None = None
    no_enable_eplb: None = None
    no_enable_expert_parallel: None = None
    no_enable_log_requests: None = None
    no_enable_lora: None = None
    no_enable_lora_bias: None = None
    no_enable_prefix_caching: None = None
    no_enable_prompt_embeds: None = None
    no_enable_sleep_mode: None = None
    no_enforce_eager: None = None
    no_eplb_log_balancedness: None = None
    no_fully_sharded_loras: None = None
    no_guided_decoding_disable_additional_properties: None = None
    no_guided_decoding_disable_any_whitespace: None = None
    no_guided_decoding_disable_fallback: None = None
    no_interleave_mm_strings: None = None
    no_kv_sharing_fast_prefill: None = None
    no_ray_workers_use_nsight: None = None
    no_skip_mm_profiling: None = None
    no_skip_tokenizer_init: None = None
    no_trust_remote_code: None = None
    no_use_tqdm_on_load: None = None


def validate_vllm_env_vars(env_vars: dict[str, str], source: str = "") -> None:
    """Validate VLLM_* env vars against vllm.envs.environment_variables.

    Raises ValueError for any VLLM_*-prefixed key not found in the registry.
    Only runs when vLLM is importable; otherwise does nothing.

    Args:
        env_vars: Dictionary of environment variable names to values.
        source: Optional source identifier (e.g. profile path) for error messages.

    Raises:
        ValueError: If any VLLM_*-prefixed env var is not in the vLLM registry.
    """
    if VllmEngineArgsModel._vllm_parser() is None:
        return

    from vllm.envs import environment_variables

    known = set(environment_variables)
    vllm_vars = {k for k in env_vars if k.startswith("VLLM_")}
    unrecognized = sorted(vllm_vars - known)

    if unrecognized:
        source_msg = f" in {source}" if source else ""
        raise ValueError(
            f"Unrecognized vLLM environment variable(s){source_msg}: "
            f"{', '.join(unrecognized)}. "
            f"Check for typos in env_vars."
        )


class VllmEngine(BaseEngine):
    """vLLM-backed LLM engine."""

    ARGS_MODEL: ClassVar[Optional[type[EngineArgsModel]]] = VllmEngineArgsModel
    ARGS_FORMAT: ClassVar[EngineArgsFormat] = EngineArgsFormat.STANDARD
    requires_aiter_kernels: ClassVar[bool] = True

    def apply_engine_defaults(self, engine_args: dict[str, Any], served_model_names: list[str]) -> None:
        """Set the vLLM ``served-model-name`` OpenAI-compatibility flag."""
        engine_args["served-model-name"] = served_model_names
        logger.info(f"Setting served-model-name to: {served_model_names}")

    @classmethod
    def validate_env_vars(cls, env_vars: dict[str, str], source: str = "") -> None:
        """Validate ``VLLM_*`` env vars against vLLM's known environment registry."""
        validate_vllm_env_vars(env_vars, source=source)

    def _runtime_lora_updating(self, profile: "Profile | None") -> bool:
        """Whether this serve uses vLLM runtime LoRA add/remove (resolver + watcher).

        True only when ``AIM_ADAPTER_MODE=dynamic`` **and** the profile declares
        the full ``adapters`` token. A profile with ``adapters-scale-only`` has
        no runtime add/remove support (ADR-0004), so it stays on the static path
        even in dynamic mode. ``profile is None`` (no profile context, e.g. unit
        tests) defers to the configured mode alone.
        """
        if self.config.adapter_mode != AdapterMode.DYNAMIC:
            return False
        if profile is None or profile.metadata is None:
            return True
        return profile.metadata.adapter_token == AdapterToken.ADAPTERS

    def build_adapter_runtime(self, profile: "Profile | None") -> tuple[dict[str, Any], dict[str, str]]:
        """Translate the AIM_ADAPTER_* contract into vLLM flags + env (ADR-0004).

        Static path (static mode, or a scale-only profile in dynamic mode):
        enable LoRA + caps **only when adapters are actually present** on disk,
        and pass them as ``--lora-modules name=path``; the filesystem resolver
        stays off so vLLM rejects runtime load/unload (ADR static-mode refusal).
        When no adapters are mounted, ``--enable-lora`` is left off so the base
        model keeps its full performance — notably, ``--enable-lora`` forces
        vLLM off the AITER fused-MoE backend on ROCm (it is skipped for LoRA),
        so always enabling it would penalise MoE base serving even with zero
        adapters.

        Dynamic path (dynamic mode + full ``adapters`` token): enable LoRA +
        caps unconditionally (adapters can appear later via the watcher), leave
        ``--lora-modules`` off, and set the resolver triplet so adapters resolve
        lazily and the in-pod watcher can load/unload.
        """
        config = self.config
        lora_caps: dict[str, Any] = {
            "enable-lora": True,
            "max-loras": config.adapter_max_count,
            "max-cpu-loras": config.adapter_max_cpu_count,
            "max-lora-rank": config.adapter_max_rank,
        }
        engine_args: dict[str, Any] = {}
        env: dict[str, str] = {}

        if self._runtime_lora_updating(profile):
            # Dynamic mode: adapters may be added at runtime, so LoRA must be
            # enabled now even if the source is currently empty.
            engine_args.update(lora_caps)
            env = {
                "VLLM_ALLOW_RUNTIME_LORA_UPDATING": "True",
                "VLLM_LORA_RESOLVER_CACHE_DIR": config.adapter_source,
                "VLLM_PLUGINS": "lora_filesystem_resolver",
            }
            logger.info("Dynamic adapter mode: enabled vLLM filesystem resolver + runtime LoRA updating.")
        else:
            adapters = enumerate_adapters(config.adapter_source, config.adapter_max_rank)
            if adapters:
                engine_args.update(lora_caps)
                engine_args["lora-modules"] = [f"{a.name}={a.path}" for a in adapters]
                logger.info(f"Loaded {len(adapters)} static LoRA adapter(s): {[a.name for a in adapters]}")
            else:
                logger.info(
                    "No LoRA adapters found under %s; leaving --enable-lora off so the base "
                    "model keeps full performance (e.g. AITER fused-MoE on ROCm). To enable "
                    "LoRA: mount adapters in the adapter source dir, or (for full `adapters` "
                    "profiles only) switch to dynamic mode.",
                    config.adapter_source,
                )

        return engine_args, env

    def needs_runtime_supervisor(self, profile: "Profile") -> bool:
        """True when the profile needs the in-pod watcher (runtime LoRA updates).

        Only full ``adapters`` profiles in dynamic mode need supervision;
        ``adapters-scale-only`` profiles run on the static path with no watcher.
        """
        return self._runtime_lora_updating(profile)
