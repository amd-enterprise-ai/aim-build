# Copyright © Advanced Micro Devices, Inc., or its affiliates.
#
# SPDX-License-Identifier: MIT

"""
Pydantic model for vLLM engine arguments validation.

Used as fallback validation when vLLM is not installed.
"""

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field


class VllmEngineArgsModel(BaseModel):
    """Pydantic model mirroring vLLM EngineArgs for offline validation."""

    model_config = ConfigDict(extra="allow", populate_by_name=True)

    disable_log_stats: bool | None = Field(default=None, alias="disable-log-stats")
    model: str | None = None
    runner: Literal["auto", "draft", "generate", "pooling"] | None = None
    convert: Literal["auto", "classify", "embed", "none", "reward"] | None = None
    task: (
        Literal["auto", "classify", "draft", "embed", "embedding", "generate", "reward", "score", "transcription"]
        | None
    ) = None
    tokenizer: str | None = None
    tokenizer_mode: Literal["auto", "custom", "mistral", "slow"] | None = Field(default=None, alias="tokenizer-mode")
    tokenizer_revision: str | None = Field(default=None, alias="tokenizer-revision")
    trust_remote_code: bool | None = Field(default=None, alias="trust-remote-code")
    dtype: Literal["auto", "bfloat16", "float", "float16", "float32", "half"] | None = None
    seed: int | None = None
    hf_config_path: str | None = Field(default=None, alias="hf-config-path")
    allowed_local_media_path: str | None = Field(default=None, alias="allowed-local-media-path")
    revision: str | None = None
    code_revision: str | None = Field(default=None, alias="code-revision")
    rope_scaling: dict[str, Any] | None = Field(default=None, alias="rope-scaling")
    rope_theta: float | None = Field(default=None, alias="rope-theta")
    max_model_len: int | None = Field(default=None, alias="max-model-len")
    quantization: str | None = None
    enforce_eager: bool | None = Field(default=None, alias="enforce-eager")
    max_logprobs: int | None = Field(default=None, alias="max-logprobs")
    logprobs_mode: Literal["raw_logits", "raw_logprobs", "processed_logits", "processed_logprobs"] | None = Field(
        default=None, alias="logprobs-mode"
    )
    disable_sliding_window: bool | None = Field(default=None, alias="disable-sliding-window")
    disable_cascade_attn: bool | None = Field(default=None, alias="disable-cascade-attn")
    skip_tokenizer_init: bool | None = Field(default=None, alias="skip-tokenizer-init")
    enable_prompt_embeds: bool | None = Field(default=None, alias="enable-prompt-embeds")
    served_model_name: str | list[str] | None = Field(default=None, alias="served-model-name")
    disable_async_output_proc: bool | None = Field(default=None, alias="disable-async-output-proc")
    config_format: Literal["auto", "hf", "mistral"] | None = Field(default=None, alias="config-format")
    hf_token: str | None = Field(default=None, alias="hf-token")
    hf_overrides: dict[str, Any] | None = Field(default=None, alias="hf-overrides")
    override_neuron_config: dict[str, Any] | None = Field(default=None, alias="override-neuron-config")
    override_pooler_config: dict[str, Any] | None = Field(default=None, alias="override-pooler-config")
    logits_processor_pattern: str | None = Field(default=None, alias="logits-processor-pattern")
    generation_config: str | None = Field(default=None, alias="generation-config")
    override_generation_config: dict[str, Any] | None = Field(default=None, alias="override-generation-config")
    enable_sleep_mode: bool | None = Field(default=None, alias="enable-sleep-mode")
    model_impl: Literal["auto", "vllm", "transformers"] | None = Field(default=None, alias="model-impl")
    override_attention_dtype: str | None = Field(default=None, alias="override-attention-dtype")
    logits_processors: str | None = Field(default=None, alias="logits-processors")
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
    ) = Field(default=None, alias="load-format")
    download_dir: str | None = Field(default=None, alias="download-dir")
    model_loader_extra_config: dict[str, Any] | None = Field(default=None, alias="model-loader-extra-config")
    ignore_patterns: str | None = Field(default=None, alias="ignore-patterns")
    use_tqdm_on_load: bool | None = Field(default=None, alias="use-tqdm-on-load")
    pt_load_map_location: str | None = Field(default=None, alias="pt-load-map-location")
    guided_decoding_backend: Literal["auto", "guidance", "outlines", "xgrammar"] | None = Field(
        default=None, alias="guided-decoding-backend"
    )
    guided_decoding_disable_fallback: bool | None = Field(default=None, alias="guided-decoding-disable-fallback")
    guided_decoding_disable_any_whitespace: bool | None = Field(
        default=None, alias="guided-decoding-disable-any-whitespace"
    )
    guided_decoding_disable_additional_properties: bool | None = Field(
        default=None, alias="guided-decoding-disable-additional-properties"
    )
    reasoning_parser: str | None = Field(default=None, alias="reasoning-parser")
    distributed_executor_backend: str | None = Field(default=None, alias="distributed-executor-backend")
    block_size: Literal[1, 8, 16, 32, 64, 128] | None = Field(default=None, alias="block-size")
    gpu_memory_utilization: float | None = Field(default=None, alias="gpu-memory-utilization")
    kv_cache_dtype: Literal["auto", "fp8", "fp8_e4m3", "fp8_e5m2", "fp8_inc"] | None = Field(
        default=None, alias="kv-cache-dtype"
    )
    prefix_caching_hash_algo: Literal["builtin", "sha256", "sha256_cbor_64bit"] | None = Field(
        default=None, alias="prefix-caching-hash-algo"
    )
    mm_encoder_tp_mode: Literal["data", "weights"] | None = Field(default=None, alias="mm-encoder-tp-mode")
    lora_dtype: Literal["auto", "bfloat16", "float16"] | None = Field(default=None, alias="lora-dtype")
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
    ) = Field(default=None, alias="collect-detailed-traces")
    preemption_mode: Literal["recompute", "swap"] | None = Field(default=None, alias="preemption-mode")
    scheduling_policy: Literal["fcfs", "priority"] | None = Field(default=None, alias="scheduling-policy")
    data_parallel_backend: Literal["mp", "ray"] | None = Field(default=None, alias="data-parallel-backend")
    compilation_config: dict[str, Any] | None = Field(default=None, alias="compilation-config")
    additional_config: dict[str, Any] | None = Field(default=None, alias="additional-config")
    pipeline_parallel_size: int | None = Field(default=None, alias="pipeline-parallel-size")
    tensor_parallel_size: int | None = Field(default=None, alias="tensor-parallel-size")
    data_parallel_size: int | None = Field(default=None, alias="data-parallel-size")
    data_parallel_rank: int | None = Field(default=None, alias="data-parallel-rank")
    data_parallel_start_rank: int | None = Field(default=None, alias="data-parallel-start-rank")
    data_parallel_size_local: int | None = Field(default=None, alias="data-parallel-size-local")
    data_parallel_address: str | None = Field(default=None, alias="data-parallel-address")
    data_parallel_rpc_port: int | None = Field(default=None, alias="data-parallel-rpc-port")
    data_parallel_hybrid_lb: bool | None = Field(default=None, alias="data-parallel-hybrid-lb")
    enable_expert_parallel: bool | None = Field(default=None, alias="enable-expert-parallel")
    enable_eplb: bool | None = Field(default=None, alias="enable-eplb")
    eplb_config: dict[str, Any] | None = Field(default=None, alias="eplb-config")
    num_redundant_experts: int | None = Field(default=None, alias="num-redundant-experts")
    eplb_window_size: int | None = Field(default=None, alias="eplb-window-size")
    eplb_step_interval: int | None = Field(default=None, alias="eplb-step-interval")
    eplb_log_balancedness: bool | None = Field(default=None, alias="eplb-log-balancedness")
    max_parallel_loading_workers: int | None = Field(default=None, alias="max-parallel-loading-workers")
    ray_workers_use_nsight: bool | None = Field(default=None, alias="ray-workers-use-nsight")
    disable_custom_all_reduce: bool | None = Field(default=None, alias="disable-custom-all-reduce")
    worker_cls: str | None = Field(default=None, alias="worker-cls")
    worker_extension_cls: str | None = Field(default=None, alias="worker-extension-cls")
    enable_multimodal_encoder_data_parallel: bool | None = Field(
        default=None, alias="enable-multimodal-encoder-data-parallel"
    )
    swap_space: int | None = Field(default=None, alias="swap-space")
    num_gpu_blocks_override: int | None = Field(default=None, alias="num-gpu-blocks-override")
    enable_prefix_caching: bool | None = Field(default=None, alias="enable-prefix-caching")
    cpu_offload_gb: int | None = Field(default=None, alias="cpu-offload-gb")
    calculate_kv_scales: bool | None = Field(default=None, alias="calculate-kv-scales")
    kv_sharing_fast_prefill: bool | None = Field(default=None, alias="kv-sharing-fast-prefill")
    mamba_cache_dtype: Literal["auto", "float32"] | None = Field(default=None, alias="mamba-cache-dtype")
    mamba_ssm_cache_dtype: Literal["auto", "float32"] | None = Field(default=None, alias="mamba-ssm-cache-dtype")
    limit_mm_per_prompt: dict[str, Any] | None = Field(default=None, alias="limit-mm-per-prompt")
    media_io_kwargs: dict[str, Any] | None = Field(default=None, alias="media-io-kwargs")
    mm_processor_kwargs: dict[str, Any] | None = Field(default=None, alias="mm-processor-kwargs")
    mm_processor_cache_gb: int | None = Field(default=None, alias="mm-processor-cache-gb")
    disable_mm_preprocessor_cache: bool | None = Field(default=None, alias="disable-mm-preprocessor-cache")
    interleave_mm_strings: bool | None = Field(default=None, alias="interleave-mm-strings")
    skip_mm_profiling: bool | None = Field(default=None, alias="skip-mm-profiling")
    enable_lora: bool | None = Field(default=None, alias="enable-lora")
    enable_lora_bias: bool | None = Field(default=None, alias="enable-lora-bias")
    max_loras: int | None = Field(default=None, alias="max-loras")
    max_lora_rank: int | None = Field(default=None, alias="max-lora-rank")
    lora_extra_vocab_size: int | None = Field(default=None, alias="lora-extra-vocab-size")
    max_cpu_loras: int | None = Field(default=None, alias="max-cpu-loras")
    fully_sharded_loras: bool | None = Field(default=None, alias="fully-sharded-loras")
    default_mm_loras: dict[str, Any] | None = Field(default=None, alias="default-mm-loras")
    show_hidden_metrics_for_version: str | None = Field(default=None, alias="show-hidden-metrics-for-version")
    otlp_traces_endpoint: str | None = Field(default=None, alias="otlp-traces-endpoint")
    max_num_batched_tokens: int | None = Field(default=None, alias="max-num-batched-tokens")
    max_num_seqs: int | None = Field(default=None, alias="max-num-seqs")
    max_num_partial_prefills: int | None = Field(default=None, alias="max-num-partial-prefills")
    max_long_partial_prefills: int | None = Field(default=None, alias="max-long-partial-prefills")
    cuda_graph_sizes: list[Any] | None = Field(default=None, alias="cuda-graph-sizes")
    long_prefill_token_threshold: int | None = Field(default=None, alias="long-prefill-token-threshold")
    num_lookahead_slots: int | None = Field(default=None, alias="num-lookahead-slots")
    scheduler_delay_factor: float | None = Field(default=None, alias="scheduler-delay-factor")
    enable_chunked_prefill: bool | None = Field(default=None, alias="enable-chunked-prefill")
    disable_chunked_mm_input: bool | None = Field(default=None, alias="disable-chunked-mm-input")
    scheduler_cls: str | None = Field(default=None, alias="scheduler-cls")
    disable_hybrid_kv_cache_manager: bool | None = Field(default=None, alias="disable-hybrid-kv-cache-manager")
    async_scheduling: bool | None = Field(default=None, alias="async-scheduling")
    speculative_config: dict[str, Any] | None = Field(default=None, alias="speculative-config")
    kv_transfer_config: dict[str, Any] | None = Field(default=None, alias="kv-transfer-config")
    kv_events_config: dict[str, Any] | None = Field(default=None, alias="kv-events-config")
    enable_log_requests: bool | None = Field(default=None, alias="enable-log-requests")
    disable_log_requests: bool | None = Field(default=None, alias="disable-log-requests")

    # CLI negation flags — these are passed as `--no-<flag>` with null value in YAML.
    # They must only accept null (None); any other value is invalid.
    no_async_scheduling: None = Field(default=None, alias="no-async-scheduling")
    no_calculate_kv_scales: None = Field(default=None, alias="no-calculate-kv-scales")
    no_data_parallel_hybrid_lb: None = Field(default=None, alias="no-data-parallel-hybrid-lb")
    no_disable_cascade_attn: None = Field(default=None, alias="no-disable-cascade-attn")
    no_disable_chunked_mm_input: None = Field(default=None, alias="no-disable-chunked-mm-input")
    no_disable_custom_all_reduce: None = Field(default=None, alias="no-disable-custom-all-reduce")
    no_disable_hybrid_kv_cache_manager: None = Field(default=None, alias="no-disable-hybrid-kv-cache-manager")
    no_disable_log_requests: None = Field(default=None, alias="no-disable-log-requests")
    no_disable_sliding_window: None = Field(default=None, alias="no-disable-sliding-window")
    no_enable_chunked_prefill: None = Field(default=None, alias="no-enable-chunked-prefill")
    no_enable_eplb: None = Field(default=None, alias="no-enable-eplb")
    no_enable_expert_parallel: None = Field(default=None, alias="no-enable-expert-parallel")
    no_enable_log_requests: None = Field(default=None, alias="no-enable-log-requests")
    no_enable_lora: None = Field(default=None, alias="no-enable-lora")
    no_enable_lora_bias: None = Field(default=None, alias="no-enable-lora-bias")
    no_enable_prefix_caching: None = Field(default=None, alias="no-enable-prefix-caching")
    no_enable_prompt_embeds: None = Field(default=None, alias="no-enable-prompt-embeds")
    no_enable_sleep_mode: None = Field(default=None, alias="no-enable-sleep-mode")
    no_enforce_eager: None = Field(default=None, alias="no-enforce-eager")
    no_eplb_log_balancedness: None = Field(default=None, alias="no-eplb-log-balancedness")
    no_fully_sharded_loras: None = Field(default=None, alias="no-fully-sharded-loras")
    no_guided_decoding_disable_additional_properties: None = Field(
        default=None, alias="no-guided-decoding-disable-additional-properties"
    )
    no_guided_decoding_disable_any_whitespace: None = Field(
        default=None, alias="no-guided-decoding-disable-any-whitespace"
    )
    no_guided_decoding_disable_fallback: None = Field(default=None, alias="no-guided-decoding-disable-fallback")
    no_interleave_mm_strings: None = Field(default=None, alias="no-interleave-mm-strings")
    no_kv_sharing_fast_prefill: None = Field(default=None, alias="no-kv-sharing-fast-prefill")
    no_ray_workers_use_nsight: None = Field(default=None, alias="no-ray-workers-use-nsight")
    no_skip_mm_profiling: None = Field(default=None, alias="no-skip-mm-profiling")
    no_skip_tokenizer_init: None = Field(default=None, alias="no-skip-tokenizer-init")
    no_trust_remote_code: None = Field(default=None, alias="no-trust-remote-code")
    no_use_tqdm_on_load: None = Field(default=None, alias="no-use-tqdm-on-load")
