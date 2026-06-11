# Copyright © Advanced Micro Devices, Inc., or its affiliates.
#
# SPDX-License-Identifier: MIT

"""
Pydantic models for engine arguments validation.

Each model mirrors the CLI arguments of its respective serving engine.
Validation is performed by calling ``Model.model_validate(engine_args)``,
where *engine_args* is a raw ``dict[str, Any]`` whose keys may be either
kebab-case (as they appear in YAML / ``AIM_ENGINE_ARGS``) or snake_case.
Both forms are accepted because the base model is configured with
``alias_generator`` (underscore → hyphen) and ``populate_by_name=True``.

Adding support for a new engine
--------------------------------
1. Create a subclass of ``EngineArgsModel`` in this file.
2. Declare fields with standard Python snake_case names; the kebab-case alias
   is generated automatically.
3. If the engine ships its own native argument class, add a
   ``@model_validator(mode="wrap")`` that delegates to it when available,
   mirroring the pattern e.g. in ``VllmEngineArgsModel``.
4. Register it in the ``ENGINE_ARGS_MODELS`` dict at the bottom of this file:
       ENGINE_ARGS_MODELS["engine_name"] = NewEngineArgsModel
"""

import functools
import json
from typing import Any, ClassVar, Literal

from pydantic import BaseModel, ConfigDict, model_validator
from pydantic.functional_validators import ModelWrapValidatorHandler

from aim_common.compat import StrEnum


class EngineArgsFormat(StrEnum):
    """How engine_args are serialized on the command line.

    STANDARD:   --key value  (default, used by vLLM, sglang, llama.cpp, …)
    FORWARDED:  --arg key=value  (used by engines that accept forwarded KV pairs,
                e.g. BentoML ``serve --arg …``)
    """

    STANDARD = "standard"
    FORWARDED = "forwarded"


class EngineArgsModel(BaseModel):
    """Base class for all engine argument Pydantic models.

    Subclasses inherit ``model_config`` so they accept both the kebab-case
    CLI alias (auto-generated from the snake_case field name) and the Python
    snake_case field name, and silently pass through any extra keys that are
    not declared as fields (``extra="allow"``).
    """

    model_config = ConfigDict(
        extra="allow",
        populate_by_name=True,
        alias_generator=lambda field_name: field_name.replace("_", "-"),
    )


def engine_args_to_cli_list(
    engine_args: dict[str, Any],
    args_format: EngineArgsFormat = EngineArgsFormat.STANDARD,
) -> list[str]:
    """Convert an engine_args dict to a CLI argument list.

    Args:
        engine_args: Engine arguments dict whose keys may be snake_case or kebab-case.
        args_format: Serialization format. ``STANDARD`` (default) produces ``--key value``
            flags. ``FORWARDED`` produces ``--arg key=value`` pairs for engines that
            accept forwarded KV pairs (e.g. BentoML ``serve --arg …``).

    Keys are normalized (underscore → hyphen) for STANDARD format so both
    snake_case and kebab-case inputs produce the same ``--kebab-case`` flags.
    """
    cli_args: list[str] = []
    for key, value in engine_args.items():
        if args_format == EngineArgsFormat.FORWARDED:
            cli_args.extend(["--arg", f"{key}={True if value is None else value}"])
            continue
        flag = f"--{key.replace('_', '-')}"
        if value is None:
            cli_args.append(flag)
        elif isinstance(value, bool):
            if value:
                cli_args.append(flag)
        elif isinstance(value, (list, tuple)):
            cli_args.append(flag)
            for item in value:
                cli_args.append(str(item))
        elif isinstance(value, dict):
            cli_args.extend([flag, json.dumps(value)])
        else:
            cli_args.extend([flag, str(value)])
    return cli_args


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


class BentomlEngineArgsModel(EngineArgsModel):
    """Pydantic model for BentoML ``bentoml serve`` CLI arguments.

    Unlike vLLM, BentoML does **not** expose a native ``EngineArgs`` class or
    argparse-based CLI parser that can be hooked for authoritative validation.
    BentoML is a serving *framework* (wrapping inference engines) rather than
    an inference engine itself:

    * Its CLI is built with Click, not argparse, and only accepts
      framework-level flags (host, port, workers, SSL, etc.).
    * Model-specific and inference-engine configuration lives inside Python
      service code (``@bentoml.service()`` decorator) and ``bentofile.yaml``,
      not as CLI arguments.
    * BentoML's internal ``BentoMLConfiguration`` validates BentoML's own
      runtime config (tracing, logging, api_server settings) — not engine
      arguments in the AIM sense.

    BentoML *does* offer validation surfaces, but none function as a
    centralized engine-args validator:

    * ``bentoml.use_arguments()`` (v1.4.8+) lets service authors define a
      Pydantic schema for **per-service template arguments** (passed via
      ``--arg``). The schema is user-defined per service, not a built-in
      class that BentoML ships.
    * ``_bentoml_sdk.validators`` (``TensorSchema``, ``DataframeSchema``,
      ``PILImageEncoder``, etc.) validate API **request/response payloads**
      (tensors, images, dataframes), not CLI or engine configuration.

    Consequently this model uses **Pydantic-only validation** — there is no
    ``@model_validator(mode="wrap")`` delegation to a native parser. The
    fields below mirror the ``bentoml serve`` Click options as of BentoML
    v1.4.x so that AIM profiles can be validated at merge time.
    """

    port: int | None = None
    host: str | None = None
    api_workers: int | None = None
    timeout: int | None = None
    backlog: int | None = None
    reload: bool | None = None
    working_dir: str | None = None
    development: bool | None = None

    # SSL
    ssl_certfile: str | None = None
    ssl_keyfile: str | None = None
    ssl_keyfile_password: str | None = None
    ssl_version: int | None = None
    ssl_cert_reqs: int | None = None
    ssl_ca_certs: str | None = None
    ssl_ciphers: str | None = None

    # Timeouts
    timeout_keep_alive: int | None = None
    timeout_graceful_shutdown: int | None = None


class VllmOmniEngineArgsModel(VllmEngineArgsModel):
    """Pydantic model mirroring vLLM-Omni ``vllm serve --omni`` CLI arguments.

    When vLLM-Omni is installed, validation delegates to the same argparse tree
    built by ``OmniServeCommand.subparser_init`` (engine + frontend + Omni
    flags). When it is not installed, Pydantic field validation is used as a
    fallback, identical in spirit to :class:`VllmEngineArgsModel`.
    """

    @staticmethod
    @functools.lru_cache(maxsize=None)
    def _vllm_parser() -> Any:
        """Return the vLLM-Omni ``serve`` subcommand parser, or None if unavailable."""
        try:
            from vllm.utils.argparse_utils import FlexibleArgumentParser
            from vllm_omni.entrypoints.cli.serve import OmniServeCommand

            def _raise_on_error(message: str) -> None:
                raise ValueError(message)

            parser = FlexibleArgumentParser()
            subparsers = parser.add_subparsers(dest="vllm_subcommand")
            OmniServeCommand().subparser_init(subparsers)
            parser.error = _raise_on_error  # type: ignore[assignment]
            return parser
        except Exception:
            return None

    # Mirrors CommandGenerator order: serve --omni --model <path> <engine_args>
    _vllm_cli_argv_prefix: ClassVar[tuple[str, ...]] = (
        "serve",
        "--omni",
        "--model",
        "aim-engine-args-validation-placeholder",
    )
    _vllm_validation_label: ClassVar[str] = "vLLM-Omni"

    usp: int | None = None
    vae_patch_parallel_size: int | None = None
    vae_use_tiling: bool | None = None


# ---------------------------------------------------------------------------
# Engine model registry
#
# Maps engine names (matching the ``validator`` field in engines.yaml) to
# their EngineArgsModel subclass. Callers use ModelClass.model_validate(args)
# directly. Add new engines here after defining their model class above.
# ---------------------------------------------------------------------------

ENGINE_ARGS_MODELS: dict[str, type[EngineArgsModel]] = {
    "vllm": VllmEngineArgsModel,
    "bentoml": BentomlEngineArgsModel,
    "vllm_omni": VllmOmniEngineArgsModel,
}


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
