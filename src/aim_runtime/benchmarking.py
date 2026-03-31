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
from pathlib import Path
from typing import Any, Dict
from urllib.error import URLError
from urllib.parse import urlparse
from urllib.request import Request, urlopen

import yaml

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
    "p90_ttft": re.compile(r"P90 TTFT.*?:\s*([\d.]+)"),
    "p99_ttft": re.compile(r"P99 TTFT.*?:\s*([\d.]+)"),
    "mean_tpot": re.compile(r"Mean TPOT.*?:\s*([\d.]+)"),
    "median_tpot": re.compile(r"Median TPOT.*?:\s*([\d.]+)"),
    "p90_tpot": re.compile(r"P90 TPOT.*?:\s*([\d.]+)"),
    "p99_tpot": re.compile(r"P99 TPOT.*?:\s*([\d.]+)"),
    "mean_itl": re.compile(r"Mean ITL.*?:\s*([\d.]+)"),
    "median_itl": re.compile(r"Median ITL.*?:\s*([\d.]+)"),
    "p90_itl": re.compile(r"P90 ITL.*?:\s*([\d.]+)"),
    "p99_itl": re.compile(r"P99 ITL.*?:\s*([\d.]+)"),
    "mean_e2el": re.compile(r"Mean E2EL.*?:\s*([\d.]+)"),
    "median_e2el": re.compile(r"Median E2EL.*?:\s*([\d.]+)"),
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
    "total_input_tokens",
    "total_generated_tokens",
    "req_throughput",
    "output_tok_throughput",
    "total_tok_throughput",
    "tok_per_user_per_second",
    "mean_ttft",
    "median_ttft",
    "p90_ttft",
    "p99_ttft",
    "mean_tpot",
    "median_tpot",
    "p90_tpot",
    "p99_tpot",
    "mean_itl",
    "median_itl",
    "p90_itl",
    "p99_itl",
    "mean_e2el",
    "median_e2el",
    "p90_e2el",
    "p99_e2el",
]


class AIMBenchmark:
    """Benchmark runner for AIM LLM service."""

    def __init__(self, service_url: str, timeout_seconds: int = 30, config_file: str = None):
        self.timeout_seconds = timeout_seconds
        self.profile_id = os.getenv("AIM_PROFILE_ID") or os.getenv("PROFILE_ID")
        self.gpu_count = None
        if self.profile_id:
            match = re.search(r"-tp(\d+)-", self.profile_id)
            self.gpu_count = int(match.group(1)) if match else None
        self.config = self._load_config(config_file)

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

    def _load_config(self, config_file: str = None) -> Dict[str, Any]:
        """Load benchmark configuration from YAML file."""
        if config_file is None:
            config_path = Path(__file__).parent / "benchmark-config.yaml"
        else:
            config_path = Path(config_file)

        if not config_path.exists():
            raise FileNotFoundError(f"Config file not found: {config_path}")

        with open(config_path, "r") as f:
            raw_config = yaml.safe_load(f)

        logger.info(f"Loaded benchmark config from {config_path}")

        # Parse config_suites format
        # Allow environment variable to override active_config from YAML
        active_config = os.getenv("ACTIVE_SUITE")

        # If active_config is empty or not provided, auto-select based on GPU count from profile
        if not active_config:
            gpu_count_suite_map = raw_config.get("gpu_count_suite_map", {})

            if self.gpu_count and gpu_count_suite_map:
                active_config = gpu_count_suite_map.get(self.gpu_count)
                if active_config:
                    logger.info(
                        "Auto-selected suite '%s' based on GPU count %s",
                        active_config,
                        self.gpu_count,
                    )
                else:
                    logger.warning("No suite mapping found for GPU count %s, using default", self.gpu_count)

            # Fall back to active_config from YAML if auto-selection didn't work
            if not active_config:
                active_config = raw_config.get("active_config", "default_dev")
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
            req = Request(f"{self.service_url}/v1/models")
            with urlopen(req, timeout=self.timeout_seconds) as response:
                model_data = json.loads(response.read().decode())
                logger.info("Retrieved model information")

                if "data" in model_data and model_data["data"]:
                    model_names = [model["id"] for model in model_data["data"]]
                    logger.info(f"Available models: {model_names}")
                else:
                    logger.warning("No models found in response")

                return model_data
        except URLError as e:
            logger.error(f"Failed to get model info: {e}")
            return None
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse model info JSON: {e}")
            return None

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
            settings.get("metric_percentiles", "90,99"),
            "--host",
            host,
            "--port",
            str(port),
        ]

        if settings.get("ignore_eos", True):
            cmd.append("--ignore-eos")

        # Add extra vllm bench kwargs if provided via environment variable
        extra_args = os.getenv("VLLM_BENCH_EXTRA_ARGS", "").strip()
        if extra_args:
            cmd.extend(shlex.split(extra_args))
            logger.info(f"Added extra vllm bench args: {extra_args}")

        logger.info(f"Running command: {' '.join(cmd)}")

        start_time = time.time()
        try:
            # Run vLLM benchmark
            result = subprocess.run(
                cmd, capture_output=True, text=True, timeout=settings.get("timeout_seconds_per_config", 300)
            )

            manual_duration = time.time() - start_time

            if result.returncode != 0:
                logger.error(f"vLLM benchmark failed with return code {result.returncode}")
                logger.error(f"Failed command: {' '.join(cmd)}")
                logger.error(f"stderr: {result.stderr}")
                return {"success": False, "error": result.stderr, "manual_duration": manual_duration}

            # Parse the output
            metrics = self._parse_benchmark_output(result.stdout)
            metrics["success"] = True
            metrics["manual_duration"] = manual_duration
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
            return {"success": False, "error": "Timeout", "manual_duration": time.time() - start_time}
        except Exception as e:
            logger.error(f"Benchmark failed: {e}")
            return {"success": False, "error": str(e), "manual_duration": time.time() - start_time}

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

    def export_results(self, results: Dict[str, Any], output_dir: str = ".") -> None:
        """Export benchmark results to both CSV and JSON formats."""
        # Always export both formats - use environment variables or defaults
        json_filename = os.getenv("BENCHMARK_JSON_FILE", "benchmark_results.json")
        csv_filename = os.getenv("BENCHMARK_CSV_FILE", "benchmark_results.csv")

        # Export JSON
        json_path = Path(output_dir) / json_filename
        with open(json_path, "w") as f:
            json.dump(results, f, indent=2)
        logger.info(f"JSON results saved to {json_path}")

        # Export CSV
        csv_path = Path(output_dir) / csv_filename
        self._export_csv(results, csv_path)
        logger.info(f"CSV results saved to {csv_path}")

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
