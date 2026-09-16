#!/usr/bin/env python3

# Copyright © Advanced Micro Devices, Inc., or its affiliates.
#
# SPDX-License-Identifier: MIT
"""
AIM Benchmarking Script

Benchmarking script for AIM LLM service. Uses vLLM bench
and collects detailed performance metrics.
"""

import csv
import json
import logging
import os
import re
import shlex
import subprocess
import time
from functools import cached_property
from pathlib import Path
from typing import Any, Dict, Mapping, Optional
from urllib.parse import urlparse

import requests

from aim_common.engine_args import (
    ENGINE_ARG_TRUST_REMOTE_CODE,
    engine_flag_enabled,
    tokenizer_mode_from_engine_args,
)
from aim_runtime.utils import read_yaml

logger = logging.getLogger(__name__)


# Metric patterns for parsing vLLM benchmark output
# From Jenkins pipeline, with p90 percentiles added
METRIC_PATTERNS = {
    "successful_reqs": re.compile(r"Successful requests:\s*(\d+)"),
    "duration": re.compile(r"Benchmark duration.*?:\s*([\d.]+)"),
    "total_input_tokens": re.compile(r"Total input tokens:\s*(\d+)"),
    "total_generated_tokens": re.compile(r"Total generated tokens:\s*(\d+)"),
    "req_throughput": re.compile(r"Request throughput.*?:\s*([\d.]+)"),
    "output_tok_throughput": re.compile(r"Output token throughput.*?:\s*([\d.]+)"),
    "total_tok_throughput": re.compile(r"Total Token throughput.*?:\s*([\d.]+)"),
    "mean_ttft": re.compile(r"Mean TTFT.*?:\s*([\d.]+)"),
    "median_ttft": re.compile(r"Median TTFT.*?:\s*([\d.]+)"),
    "p75_ttft": re.compile(r"P75 TTFT.*?:\s*([\d.]+)"),
    "p90_ttft": re.compile(r"P90 TTFT.*?:\s*([\d.]+)"),
    "p99_ttft": re.compile(r"P99 TTFT.*?:\s*([\d.]+)"),
    "mean_tpot": re.compile(r"Mean TPOT.*?:\s*([\d.]+)"),
    "median_tpot": re.compile(r"Median TPOT.*?:\s*([\d.]+)"),
    "p75_tpot": re.compile(r"P75 TPOT.*?:\s*([\d.]+)"),
    "p90_tpot": re.compile(r"P90 TPOT.*?:\s*([\d.]+)"),
    "p99_tpot": re.compile(r"P99 TPOT.*?:\s*([\d.]+)"),
    "mean_itl": re.compile(r"Mean ITL.*?:\s*([\d.]+)"),
    "median_itl": re.compile(r"Median ITL.*?:\s*([\d.]+)"),
    "p75_itl": re.compile(r"P75 ITL.*?:\s*([\d.]+)"),
    "p90_itl": re.compile(r"P90 ITL.*?:\s*([\d.]+)"),
    "p99_itl": re.compile(r"P99 ITL.*?:\s*([\d.]+)"),
    "mean_e2el": re.compile(r"Mean E2EL.*?:\s*([\d.]+)"),
    "median_e2el": re.compile(r"Median E2EL.*?:\s*([\d.]+)"),
    "p75_e2el": re.compile(r"P75 E2EL.*?:\s*([\d.]+)"),
    "p90_e2el": re.compile(r"P90 E2EL.*?:\s*([\d.]+)"),
    "p99_e2el": re.compile(r"P99 E2EL.*?:\s*([\d.]+)"),
}

# CSV header matching Jenkins pipeline output
CSV_HEADER = [
    "config_name",
    "model_name",
    "profile_id",
    "concurrency",
    "input_seq_len",
    "output_seq_len",
    "num_prompts",
    "successful_reqs",
    "duration",
    "manual_time_durations",
    "total_input_tokens",
    "total_generated_tokens",
    "req_throughput",
    "output_tok_throughput",
    "total_tok_throughput",
    "tok_per_user_per_second",
    "mean_ttft",
    "median_ttft",
    "p75_ttft",
    "p90_ttft",
    "p99_ttft",
    "mean_tpot",
    "median_tpot",
    "p75_tpot",
    "p90_tpot",
    "p99_tpot",
    "mean_itl",
    "median_itl",
    "p75_itl",
    "p90_itl",
    "p99_itl",
    "mean_e2el",
    "median_e2el",
    "p75_e2el",
    "p90_e2el",
    "p99_e2el",
]


class AIMBenchmark:
    """Benchmark runner for AIM LLM service."""

    def __init__(
        self,
        service_url: str,
        timeout_seconds: int = 30,
        config_file: str = None,
        engine_args: Optional[Mapping[str, Any]] = None,
    ):
        self.timeout_seconds = timeout_seconds
        self.profile_id = os.getenv("AIM_PROFILE_ID") or os.getenv("PROFILE_ID")
        self.accelerator_count = None
        if self.profile_id:
            match = re.search(r"-tp(\d+)-", self.profile_id)
            self.accelerator_count = int(match.group(1)) if match else None
        self.config = self._load_config(config_file)
        self._engine_args = engine_args
        self._engine_args_resolved = engine_args is not None

        parsed_url = urlparse(service_url)

        if not parsed_url.hostname:
            raise ValueError(f"Invalid service URL: {service_url}")

        if not parsed_url.port:
            raise ValueError(f"Port must be explicitly specified in service URL: {service_url}")

        service_host = parsed_url.hostname
        service_port = parsed_url.port

        # Store in settings for use by benchmark methods
        settings = self.config.get("settings", {})
        settings["service_host"] = service_host
        settings["service_port"] = service_port
        self.config["settings"] = settings

        self.service_url = service_url

        self._validate_settings()

    def _load_config(self, config_file: str = None) -> Dict[str, Any]:
        """Load benchmark configuration from YAML file."""
        if config_file is None:
            config_path = Path(__file__).parent / "benchmark-config.yaml"
        else:
            config_path = Path(config_file)

        if not config_path.exists():
            raise FileNotFoundError(f"Config file not found: {config_path}")

        raw_config = read_yaml(config_path)

        logger.info(f"Loaded benchmark config from {config_path}")

        # Parse config_suites format
        # Allow environment variable to override active_config from YAML
        active_config = os.getenv("ACTIVE_SUITE")

        # If active_config is empty or not provided, auto-select based on accelerator count from profile
        if not active_config:
            acc_count_suite_map = raw_config.get("accelerator_count_suite_map", {})

            if self.accelerator_count and acc_count_suite_map:
                active_config = acc_count_suite_map.get(self.accelerator_count)
                if active_config:
                    logger.info(
                        "Auto-selected suite '%s' based on accelerator count %s",
                        active_config,
                        self.accelerator_count,
                    )
                else:
                    logger.warning(
                        "No suite mapping found for accelerator count %s, using default", self.accelerator_count
                    )

            # Fall back to active_config from YAML if auto-selection didn't work
            if not active_config:
                active_config = raw_config.get("active_config", "default_opt")
                logger.info(f"Using default suite from config: '{active_config}'")
        else:
            logger.info(f"Using explicitly provided suite: '{active_config}'")

        config_suites = raw_config.get("config_suites", {})

        if active_config not in config_suites:
            raise ValueError(
                f"Active config '{active_config}' not found in config_suites. Available: {list(config_suites.keys())}"
            )

        logger.info(f"Final config suite: '{active_config}'")

        # Convert tuples to benchmark_configs
        suite = config_suites[active_config]
        benchmark_configs = []

        for i, params in enumerate(suite, 1):
            if not isinstance(params, list) or len(params) != 4:
                raise ValueError(
                    f"Invalid config tuple at index {i}: {params}. Expected [ISL, OSL, concurrency, num_prompts]"
                )

            isl, osl, conc, np = params
            config = {
                "name": f"isl{isl}_osl{osl}_conc{conc}_np{np}",
                "input_seq_len": isl,
                "output_seq_len": osl,
                "concurrency": conc,
                "num_prompts": np,
            }
            benchmark_configs.append(config)

        # Build final config structure
        config = {
            "benchmark_configs": benchmark_configs,
            "settings": raw_config.get("settings", {}),
        }

        logger.info(f"Found {len(config['benchmark_configs'])} benchmark configurations:")

        # Log each configuration
        for i, bench_config in enumerate(config["benchmark_configs"], 1):
            logger.info(
                f"  {i}. {bench_config['name']}: "
                f"concurrency={bench_config['concurrency']}, "
                f"input_len={bench_config['input_seq_len']}, "
                f"output_len={bench_config['output_seq_len']}, "
                f"prompts={bench_config['num_prompts']}"
            )

        # Log global settings
        settings = config.get("settings", {})
        logger.info(
            f"Global settings: timeout_seconds_per_config={settings.get('timeout_seconds_per_config', 300)}s, "
            f"ignore_eos={settings.get('ignore_eos', True)}"
        )

        return config

    @property
    def engine_args(self) -> Mapping[str, Any]:
        """Engine args of the profile the benchmarked service was started with.

        Callers that have already resolved the profile pass them in; callers
        that construct this class directly fall back to selecting the profile
        the same way the server did. An empty mapping means no profile-derived
        flags are added.
        """
        if not self._engine_args_resolved:
            self._engine_args = self._resolve_profile_engine_args()
            self._engine_args_resolved = True
        return self._engine_args or {}

    @cached_property
    def tokenizer_mode(self) -> Optional[str]:
        """The tokenizer mode the benchmark client has to match the engine on.

        Resolved once per run rather than per configuration: the engine args
        cannot change between configurations, so a profile declaring a mode the
        engine would reject should say so once instead of once per point in the
        sweep.
        """
        return tokenizer_mode_from_engine_args(self.engine_args)

    @staticmethod
    def _resolve_profile_engine_args() -> Dict[str, Any]:
        from aim_runtime.aim_runtime import AIMRuntime
        from aim_runtime.config import AIMConfig

        try:
            runtime = AIMRuntime(AIMConfig.from_environment())
            profile = runtime.profile_selector.find_profile()
        except Exception as e:
            logger.warning(
                "Could not resolve the profile for benchmark flags (%s); "
                "tokenizer mode and trust-remote-code will not be derived from it",
                e,
            )
            return {}

        logger.info("Resolved engine args for benchmark flags from profile '%s'", profile.profile_id)
        return dict(profile.engine_args or {})

    def _parse_benchmark_output(self, output: str) -> Dict[str, Any]:
        """Parse vLLM benchmark output using regex patterns from Jenkins pipeline."""
        logger.info("Parsing benchmark metrics...")
        results = {}
        for key, pattern in METRIC_PATTERNS.items():
            match = pattern.search(output)
            results[key] = match.group(1) if match else None
        return results

    def get_model_info(self) -> Dict[str, Any] | None:
        """Get model information from the service."""
        try:
            response = requests.get(f"{self.service_url}/v1/models", timeout=self.timeout_seconds)
            response.raise_for_status()
            model_data = response.json()
            logger.info("Retrieved model information")

            if "data" in model_data and model_data["data"]:
                model_names = [model["id"] for model in model_data["data"]]
                logger.info(f"Available models: {model_names}")
            else:
                logger.warning("No models found in response")

            return model_data
        # requests.JSONDecodeError is also a RequestException, so it must be caught first.
        except requests.JSONDecodeError as e:
            logger.error(f"Failed to parse model info JSON: {e}")
            return None
        except requests.RequestException as e:
            logger.error(f"Failed to get model info: {e}")
            return None

    @staticmethod
    def _trust_remote_code_setting(settings: Dict[str, Any]) -> Optional[bool]:
        """The boolean the settings force, or None to follow the profile.

        Only a real boolean forces it, and anything else is rejected rather than
        interpreted. ``bool()`` reads ``"false"`` and ``"no"`` as True, which is
        the wrong way round for the strings a YAML author would reach for to turn
        this off, and quietly falling back to the profile would leave someone who
        wrote ``"true"`` with the opposite of what they asked for. Both mistakes
        are silent in a benchmark that then reports plausible numbers, so the run
        stops instead.
        """
        setting = settings.get("trust_remote_code")
        if setting is None or isinstance(setting, bool):
            return setting
        raise ValueError(
            f"Benchmark setting trust_remote_code must be a boolean or unset, got {setting!r}. "
            f"Write it unquoted as true or false to force the flag, or leave it unset "
            f"(null) to follow the profile's engine args."
        )

    @staticmethod
    def _num_warmups(settings: Dict[str, Any]) -> int:
        """How many warmup requests precede the measured window.

        A missing key takes the packaged default, because ``--config-file``
        replaces the settings block rather than layering over it and a custom
        config silent about warmups would otherwise measure its first
        configuration cold. An explicit null and 0 both mean no warmup.

        A count is rejected the same way a malformed ``trust_remote_code`` is,
        rather than left to ``int()``: a bare conversion turns 2.7 into 2 and
        reports a string as an unhelpful "invalid literal", and a negative count
        would silently drop the flag and reintroduce the very skew it removes.
        """
        setting = settings.get("num_warmups", 5)
        if setting is None:
            return 0
        if isinstance(setting, bool) or not isinstance(setting, int) or setting < 0:
            raise ValueError(
                f"Benchmark setting num_warmups must be a non-negative integer or unset, "
                f"got {setting!r}. Use 0 to measure without warmup."
            )
        return setting

    def _validate_settings(self) -> None:
        """Reject malformed benchmark settings before anything is measured.

        Reading them at construction means a typo surfaces immediately, rather
        than after the service has been started and probed and the first
        configuration is about to run.
        """
        settings = self.config.get("settings", {})
        self._trust_remote_code_setting(settings)
        self._num_warmups(settings)

    def _trust_remote_code(self, settings: Dict[str, Any]) -> bool:
        """Whether the benchmark client may execute remote tokenizer code.

        Unset follows the profile, so the client and the served engine agree;
        an explicit boolean in the settings forces it either way.
        """
        setting = self._trust_remote_code_setting(settings)
        if setting is not None:
            return setting
        return engine_flag_enabled(self.engine_args, ENGINE_ARG_TRUST_REMOTE_CODE)

    def run_vllm_benchmark(self, model_name: str, config: Dict[str, Any]) -> Dict[str, Any]:
        """Run vLLM benchmark for a specific configuration."""
        logger.info(f"Running vLLM benchmark: {config['name']}")

        # Extract benchmark settings
        settings = self.config.get("settings", {})
        host = settings["service_host"]
        port = settings["service_port"]

        cmd = [
            "vllm",
            "bench",
            "serve",
            "--model",
            model_name,
            "--dataset-name",
            settings.get("dataset_name", "random"),
            "--random-input-len",
            str(config["input_seq_len"]),
            "--random-output-len",
            str(config["output_seq_len"]),
            "--max-concurrency",
            str(config["concurrency"]),
            "--num-prompts",
            str(config["num_prompts"]),
            "--percentile-metrics",
            settings.get("percentile_metrics", "ttft,tpot,itl,e2el"),
            "--metric-percentiles",
            settings.get("metric_percentiles", "75,90,99"),
            "--host",
            host,
            "--port",
            str(port),
        ]

        if settings.get("ignore_eos", True):
            cmd.append("--ignore-eos")

        num_warmups = self._num_warmups(settings)
        if num_warmups > 0:
            cmd.extend(["--num-warmups", str(num_warmups)])

        if self.tokenizer_mode:
            cmd.extend(["--tokenizer-mode", self.tokenizer_mode])

        extra_args = shlex.split(os.getenv("VLLM_BENCH_EXTRA_ARGS", "").strip())

        if self._trust_remote_code(settings) and "--trust-remote-code" not in extra_args:
            cmd.append("--trust-remote-code")

        if extra_args:
            cmd.extend(extra_args)
            logger.info(f"Added extra vllm bench args: {shlex.join(extra_args)}")

        logger.info(f"Running command: {shlex.join(cmd)}")

        start_time = time.time()
        try:
            # Run vLLM benchmark
            result = subprocess.run(
                cmd, capture_output=True, text=True, timeout=settings.get("timeout_seconds_per_config", 300)
            )

            manual_duration = time.time() - start_time

            if result.returncode != 0:
                logger.error(f"vLLM benchmark failed with return code {result.returncode}")
                logger.error(f"Failed command: {shlex.join(cmd)}")
                logger.error(f"stderr: {result.stderr}")
                return {"success": False, "error": result.stderr, "manual_time_durations": manual_duration}

            # Parse the output
            metrics = self._parse_benchmark_output(result.stdout)
            metrics["success"] = True
            metrics["manual_time_durations"] = manual_duration
            metrics["config_name"] = config["name"]

            # Calculate per-user throughput
            if metrics.get("output_tok_throughput"):
                try:
                    metrics["tok_per_user_per_second"] = float(metrics["output_tok_throughput"]) / config["concurrency"]
                except (ValueError, ZeroDivisionError):
                    metrics["tok_per_user_per_second"] = None

            logger.info(f"Benchmark completed successfully in {manual_duration:.2f}s")
            return metrics

        except subprocess.TimeoutExpired:
            logger.error(f"Benchmark timed out after {settings.get('timeout_seconds_per_config', 300)}s")
            return {"success": False, "error": "Timeout", "manual_time_durations": time.time() - start_time}
        except Exception as e:
            logger.error(f"Benchmark failed: {e}")
            return {"success": False, "error": str(e), "manual_time_durations": time.time() - start_time}

    def run_benchmark_suite(self) -> Dict[str, Any]:
        """Run the complete benchmark suite."""
        logger.info("Starting AIM Benchmark Suite")

        results: Dict[str, Any] = {
            "timestamp": time.time(),
            "model_name": None,
            "service_host": self.config["settings"]["service_host"],
            "service_port": self.config["settings"]["service_port"],
            "profile_id": self.profile_id,
            "model_info": None,
            "benchmark_configs": [],
            "overall_success": False,
        }

        # Start benchmarking
        # Model discovery
        logger.info("Discovering model from AIM service...")
        results["model_info"] = self.get_model_info()

        model_name = None
        if results["model_info"] and "data" in results["model_info"]:
            models = results["model_info"]["data"]
            if models and len(models) > 0:
                model_name = models[0]["id"]
                results["model_name"] = model_name
                logger.info(f"Discovered model: {model_name}")
            else:
                logger.error("No models found in service response")
                return results
        else:
            logger.error("Could not discover models from service")
            return results

        # Run benchmarks for each configuration
        benchmark_configs = self.config["benchmark_configs"]
        successful_configs = 0

        for config in benchmark_configs:
            logger.info(f"Running benchmark configuration: {config['name']}")

            config_result = self.run_vllm_benchmark(model_name, config)
            config_result.update(
                {
                    "model_name": model_name,
                    "profile_id": self.profile_id,
                    "concurrency": config["concurrency"],
                    "input_seq_len": config["input_seq_len"],
                    "output_seq_len": config["output_seq_len"],
                    "num_prompts": config["num_prompts"],
                }
            )

            results["benchmark_configs"].append(config_result)

            if config_result.get("success", False):
                successful_configs += 1
                logger.info(f"Configuration '{config['name']}' completed successfully")
            else:
                logger.error(f"Configuration '{config['name']}' failed: {config_result.get('error', 'Unknown error')}")

        # Overall success if model discovery worked and at least one benchmark succeeded
        results["overall_success"] = results["model_info"] is not None and successful_configs > 0

        if results["overall_success"]:
            logger.info(
                f"Benchmark suite completed! {successful_configs}/{len(benchmark_configs)} configurations successful"
            )
        else:
            logger.error("Benchmark suite failed!")

        return results

    def export_results(self, results: Dict[str, Any], output_dir: str = ".") -> list[Path]:
        """Export benchmark results to both CSV and JSON formats."""
        # Always export both formats - use environment variables or defaults
        json_filename = os.getenv("BENCHMARK_JSON_FILE", "benchmark_results.json")
        csv_filename = os.getenv("BENCHMARK_CSV_FILE", "benchmark_results.csv")

        Path(output_dir).mkdir(parents=True, exist_ok=True)

        # Export JSON
        json_path = Path(output_dir) / json_filename
        with open(json_path, "w") as f:
            json.dump(results, f, indent=2)
        logger.info(f"JSON results saved to {json_path}")

        # Export CSV
        csv_path = Path(output_dir) / csv_filename
        self._export_csv(results, csv_path)
        logger.info(f"CSV results saved to {csv_path}")

        return [json_path, csv_path]

    def _export_csv(self, results: Dict[str, Any], csv_path: Path) -> None:
        """Export results to CSV format matching Jenkins pipeline."""
        with open(csv_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=CSV_HEADER)
            writer.writeheader()

            for config_result in results.get("benchmark_configs", []):
                # Create row with all required fields
                row = {}
                for field in CSV_HEADER:
                    row[field] = config_result.get(field, "")

                writer.writerow(row)
