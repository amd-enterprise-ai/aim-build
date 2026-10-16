# Copyright © Advanced Micro Devices, Inc., or its affiliates.
#
# SPDX-License-Identifier: MIT

"""Benchmark harness for the vLLM-served Whisper Large v3 AIM."""

from __future__ import annotations

import csv
import io
import json
import logging
import math
import os
import re
import shlex
import struct
import subprocess
import time
import wave
from pathlib import Path
from typing import Any, ClassVar
from urllib.parse import urlparse

import requests
import yaml

from aim_runtime.harness import (
    CheckInfo,
    CheckResult,
    CheckResultType,
    CheckScope,
    HarnessConfig,
    HarnessResult,
)
from aim_runtime.harness.vllm_harness import VLLMHarness

logger = logging.getLogger(__name__)
DEFAULT_CONFIG_PATH = Path("/workspace/model/benchmark-config.yaml")

METRIC_PATTERNS: dict[str, re.Pattern[str]] = {
    "successful_reqs": re.compile(r"Successful requests:\s*(\d+)"),
    "duration": re.compile(r"Benchmark duration.*?:\s*([\d.]+)"),
    "req_throughput": re.compile(r"Request throughput.*?:\s*([\d.]+)"),
    "output_token_throughput": re.compile(r"Output token throughput.*?:\s*([\d.]+)"),
    "total_token_throughput": re.compile(r"Total token throughput.*?:\s*([\d.]+)"),
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

CSV_FIELDS = [
    "config_name",
    "model_name",
    "profile_id",
    "suite",
    "concurrency",
    "num_prompts",
    "min_audio_sec",
    "max_audio_sec",
    "successful_reqs",
    "duration",
    "req_throughput",
    "output_token_throughput",
    "total_token_throughput",
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
    "success",
    "error",
]


class WhisperHarness(VLLMHarness):
    """Run vLLM's OpenAI-audio benchmark against Whisper Large v3."""

    SUPPORTED_ENGINES: ClassVar[frozenset[str]] = frozenset({"vllm", "vllm_omni"})

    CHECKS: ClassVar[list[CheckInfo]] = [
        CheckInfo(
            "audio_request_throughput",
            CheckResultType.SCORE,
            CheckScope.OFFLINE,
            "Successful audio transcription requests per second",
        )
    ]

    def list_checks(self) -> list[CheckInfo]:
        return list(self.CHECKS)

    def validate(self, config: HarnessConfig) -> HarnessResult:
        service_url = config.resolve_service_url()
        healthy = self.health_check(service_url, timeout_seconds=config.timeout_seconds)
        health_check = CheckResult(
            name="health",
            result_type=CheckResultType.PASS_FAIL,
            success=healthy,
            value=healthy,
            detail="" if healthy else f"Whisper service not reachable at {service_url}",
        )
        if not healthy:
            return HarnessResult(
                success=False,
                summary="Whisper health check failed",
                checks=[health_check],
            )

        checks = [health_check]
        if CheckScope.RUNTIME in config.check_scopes:
            checks.append(self._check_smoke_transcription(service_url, config))

        success = all(check.success for check in checks)
        return HarnessResult(
            success=success,
            summary=(
                "Whisper ASR validation: " f"{sum(check.success for check in checks)}/{len(checks)} checks passed"
            ),
            checks=checks,
        )

    def evaluate(self, config: HarnessConfig) -> HarnessResult:
        return HarnessResult(
            success=True,
            summary="Whisper ASR quality evaluation is not configured",
            checks=[
                CheckResult(
                    name="asr_quality_evaluation",
                    result_type=CheckResultType.PASS_FAIL,
                    success=True,
                    value=True,
                    detail="Skipped: ASR quality evaluation requires reference transcripts.",
                )
            ],
        )

    def benchmark(self, config: HarnessConfig) -> HarnessResult:
        service_url = config.resolve_service_url()
        if not self.health_check(service_url, timeout_seconds=config.timeout_seconds):
            return HarnessResult(
                success=False,
                summary="Service not reachable for audio benchmarking",
            )

        try:
            benchmark_config = self._load_config(config.extra)
            model_name = self._get_model_name(service_url, config.timeout_seconds)
            if not benchmark_config["settings"].get("skip_preflight", False):
                self._run_preflight_smoke(service_url, model_name, benchmark_config["settings"])
        except (OSError, ValueError, RuntimeError, requests.RequestException, yaml.YAMLError) as exc:
            return HarnessResult(
                success=False,
                summary=f"Unable to prepare audio benchmark: {exc}",
            )

        results: list[dict[str, Any]] = []
        for benchmark in benchmark_config["benchmark_configs"]:
            logger.info("Running audio benchmark configuration: %s", benchmark["config_name"])
            result = self._run_benchmark(
                service_url=service_url,
                model_name=model_name,
                suite_name=benchmark_config["suite_name"],
                settings=benchmark_config["settings"],
                benchmark=benchmark,
            )
            results.append(result)

        passed = sum(1 for result in results if result.get("success"))
        total = len(results)
        overall_success = self._suite_overall_success(
            benchmark_config["suite_name"],
            passed,
            total,
            benchmark_config["settings"],
        )
        peak_throughput = max(
            (self._parse_float(result.get("req_throughput")) for result in results if result.get("success")),
            default=0.0,
        )
        failures = [f"{result['config_name']}: {result['error']}" for result in results if not result.get("success")]

        payload = {
            "timestamp": time.time(),
            "suite": benchmark_config["suite_name"],
            "model_name": model_name,
            "profile_id": benchmark_config["profile_id"],
            "service_url": service_url,
            "overall_success": overall_success,
            "passed_configs": passed,
            "total_configs": total,
            "results": results,
        }
        artifacts = self._export_results(payload, results, config.get("output_dir", "."))

        return HarnessResult(
            success=overall_success,
            summary=f"Whisper audio benchmark: {passed}/{total} configurations successful",
            checks=[
                CheckResult(
                    name="audio_request_throughput",
                    result_type=CheckResultType.SCORE,
                    success=overall_success,
                    value=peak_throughput,
                    detail="; ".join(failures),
                )
            ],
            metrics={
                "suite": benchmark_config["suite_name"],
                "model_name": model_name,
                "profile_id": benchmark_config["profile_id"],
                "passed_configs": passed,
                "total_configs": total,
                "benchmark_configs": results,
            },
            artifacts=artifacts,
        )

    @staticmethod
    def _load_config(overrides: dict[str, Any]) -> dict[str, Any]:
        with DEFAULT_CONFIG_PATH.open() as config_stream:
            raw_config = yaml.safe_load(config_stream) or {}

        settings = dict(raw_config.get("settings", {}))
        override_settings = overrides.get("settings")
        if isinstance(override_settings, dict):
            settings.update(override_settings)

        settings.update(
            {
                key: value
                for key, value in overrides.items()
                if key not in {"active_config", "config_suites", "settings", "output_dir"}
            }
        )

        suite_name = os.getenv(
            "ACTIVE_SUITE",
            overrides.get("active_config", raw_config.get("active_config", "default_dev")),
        )
        suite_map = dict(raw_config.get("config_suites", {}))
        override_suites = overrides.get("config_suites")
        if isinstance(override_suites, dict):
            suite_map.update(override_suites)
        suite = suite_map.get(suite_name)
        if not isinstance(suite, list) or not suite:
            raise ValueError(f"Invalid audio benchmark suite '{suite_name}'")

        benchmarks = [WhisperHarness._normalize_benchmark_row(entry, settings) for entry in suite]

        logger.info("Loaded audio benchmark config from %s", DEFAULT_CONFIG_PATH)
        logger.info("Final audio benchmark suite: '%s'", suite_name)

        return {
            "suite_name": suite_name,
            "profile_id": os.environ.get("AIM_PROFILE_ID", ""),
            "benchmark_configs": benchmarks,
            "settings": settings,
        }

    @staticmethod
    def _normalize_benchmark_row(entry: Any, settings: dict[str, Any]) -> dict[str, Any]:
        if isinstance(entry, dict):
            required = {"max_concurrency", "num_prompts", "min_audio_sec", "max_audio_sec"}
            missing = required - set(entry)
            if missing:
                raise ValueError(f"Benchmark entry missing required keys: {', '.join(sorted(missing))}")

            concurrency = int(entry["max_concurrency"])
            num_prompts = int(entry["num_prompts"])
            min_audio_sec = int(entry["min_audio_sec"])
            max_audio_sec = int(entry["max_audio_sec"])
            config_name = str(
                entry.get("name") or WhisperHarness._config_name(concurrency, num_prompts, min_audio_sec, max_audio_sec)
            )
            return {
                "config_name": config_name,
                "max_concurrency": concurrency,
                "num_prompts": num_prompts,
                "min_audio_sec": min_audio_sec,
                "max_audio_sec": max_audio_sec,
            }

        if not isinstance(entry, list) or len(entry) != 4:
            raise ValueError(f"Invalid benchmark entry: {entry!r}")

        values = [int(value) for value in entry]
        if str(settings.get("suite_row_format", "asr")).lower() == "asr":
            concurrency, num_prompts, min_audio_sec, max_audio_sec = values
        else:
            min_audio_sec, max_audio_sec, concurrency, num_prompts = values

        return {
            "config_name": WhisperHarness._config_name(concurrency, num_prompts, min_audio_sec, max_audio_sec),
            "max_concurrency": concurrency,
            "num_prompts": num_prompts,
            "min_audio_sec": min_audio_sec,
            "max_audio_sec": max_audio_sec,
        }

    @staticmethod
    def _config_name(concurrency: int, num_prompts: int, min_audio_sec: int, max_audio_sec: int) -> str:
        return f"a{min_audio_sec}-{max_audio_sec}_conc{concurrency}_np{num_prompts}"

    @staticmethod
    def _get_model_name(service_url: str, timeout_seconds: int) -> str:
        response = requests.get(f"{service_url}/v1/models", timeout=timeout_seconds)
        response.raise_for_status()

        models = response.json().get("data", [])
        if not models:
            raise ValueError("No models returned from the service")

        model_id = models[0].get("id")
        if not model_id:
            raise ValueError("Model id missing in /v1/models response")
        return str(model_id)

    @classmethod
    def _check_smoke_transcription(cls, service_url: str, config: HarnessConfig) -> CheckResult:
        try:
            model_name = cls._get_model_name(service_url, config.timeout_seconds)
            response = requests.post(
                f"{service_url}/v1/audio/transcriptions",
                data={"model": model_name, "response_format": "json"},
                files={"file": ("smoke.wav", cls._make_smoke_wav(), "audio/wav")},
                timeout=config.timeout_seconds,
            )
            response.raise_for_status()
            text = response.json().get("text")
        except (OSError, ValueError, requests.RequestException) as exc:
            return CheckResult(
                name="smoke_transcription",
                result_type=CheckResultType.PASS_FAIL,
                success=False,
                value=False,
                detail=f"Audio transcription request failed: {exc}",
            )

        if not isinstance(text, str):
            return CheckResult(
                name="smoke_transcription",
                result_type=CheckResultType.PASS_FAIL,
                success=False,
                value=False,
                detail="Audio transcription response did not contain a text field",
            )

        return CheckResult(
            name="smoke_transcription",
            result_type=CheckResultType.PASS_FAIL,
            success=True,
            value=True,
            detail="Audio transcription endpoint returned a valid response",
        )

    @staticmethod
    def _make_smoke_wav() -> bytes:
        sample_rate = 16_000
        duration_seconds = 1
        frequency_hz = 440
        frames = bytearray()
        for index in range(int(sample_rate * duration_seconds)):
            sample = 0.2 * math.sin(2 * math.pi * frequency_hz * index / sample_rate)
            frames.extend(struct.pack("<h", int(sample * 32767)))

        buffer = io.BytesIO()
        with wave.open(buffer, "wb") as wav_file:
            wav_file.setnchannels(1)
            wav_file.setsampwidth(2)
            wav_file.setframerate(sample_rate)
            wav_file.writeframes(frames)
        return buffer.getvalue()

    @staticmethod
    def _build_vllm_command(
        *,
        service_url: str,
        model_name: str,
        settings: dict[str, Any],
        benchmark: dict[str, Any],
    ) -> list[str]:
        parsed_url = urlparse(service_url)
        if not parsed_url.hostname or not parsed_url.port:
            raise ValueError(f"Service URL must include a host and port: {service_url}")

        command = [
            "vllm",
            "bench",
            "serve",
            "--model",
            model_name,
            "--host",
            parsed_url.hostname,
            "--port",
            str(parsed_url.port),
            "--backend",
            str(settings.get("backend", "openai-audio")),
            "--endpoint",
            str(settings.get("endpoint", "/v1/audio/transcriptions")),
            "--dataset-name",
            str(settings.get("dataset_name", "hf")),
            "--dataset-path",
            str(settings.get("dataset_path", "openslr/librispeech_asr")),
        ]

        if str(settings.get("dataset_name", "hf")) == "hf":
            command.extend(
                [
                    "--hf-subset",
                    str(settings.get("hf_subset", "clean")),
                    "--hf-split",
                    str(settings.get("hf_split", "test")),
                ]
            )

        if settings.get("no_stream", True):
            command.append("--no-stream")
        if settings.get("no_oversample", True):
            command.append("--no-oversample")

        percentile_metrics = str(settings.get("percentile_metrics", "ttft,tpot,itl,e2el"))
        metric_percentiles = str(settings.get("metric_percentiles", "75,90,99"))

        command.extend(
            [
                "--max-concurrency",
                str(benchmark["max_concurrency"]),
                "--num-prompts",
                str(benchmark["num_prompts"]),
                "--asr-min-audio-len-sec",
                str(benchmark["min_audio_sec"]),
                "--asr-max-audio-len-sec",
                str(benchmark["max_audio_sec"]),
                "--ready-check-timeout-sec",
                str(settings.get("ready_check_timeout_sec", 600)),
                "--percentile-metrics",
                percentile_metrics,
                "--metric-percentiles",
                metric_percentiles,
            ]
        )

        if settings.get("ignore_eos", False):
            command.append("--ignore-eos")
        if settings.get("trust_remote_code", False):
            command.append("--trust-remote-code")

        extra_args = str(settings.get("extra_vllm_bench_args", os.getenv("VLLM_BENCH_EXTRA_ARGS", ""))).strip()
        if extra_args:
            command.extend(shlex.split(extra_args))

        return command

    @staticmethod
    def _parse_benchmark_output(output: str) -> dict[str, str | None]:
        metrics: dict[str, str | None] = {}
        for name, pattern in METRIC_PATTERNS.items():
            match = pattern.search(output)
            metrics[name] = match.group(1) if match else None
        return metrics

    @staticmethod
    def _truncate_error(message: str, limit: int = 2000) -> str:
        message = message.strip()
        if len(message) <= limit:
            return message
        return message[: limit - 3] + "..."

    @staticmethod
    def _config_successful(successful_reqs: str | None, num_prompts: int) -> bool:
        if successful_reqs is None:
            return False
        try:
            return int(successful_reqs) >= int(0.95 * num_prompts)
        except ValueError:
            return False

    @staticmethod
    def _suite_overall_success(suite_name: str, passed: int, total: int, settings: dict[str, Any]) -> bool:
        if total == 0:
            return False
        required_fraction = settings.get("required_success_fraction")
        if required_fraction is not None:
            return passed / total >= float(required_fraction)
        if suite_name == "asr_throughput":
            return passed / total >= 0.8
        return passed == total

    @staticmethod
    def _parse_float(value: Any) -> float:
        try:
            return float(value)
        except (TypeError, ValueError):
            return 0.0

    def _run_preflight_smoke(self, service_url: str, model_name: str, settings: dict[str, Any]) -> None:
        benchmark = {
            "config_name": "preflight",
            "max_concurrency": int(settings.get("preflight_concurrency", 1)),
            "num_prompts": int(settings.get("preflight_num_prompts", 1)),
            "min_audio_sec": int(settings.get("preflight_min_audio_sec", 0)),
            "max_audio_sec": int(settings.get("preflight_max_audio_sec", 30)),
        }
        result = self._run_benchmark(
            service_url=service_url,
            model_name=model_name,
            suite_name="preflight",
            settings=settings,
            benchmark=benchmark,
        )
        if not result.get("success"):
            raise RuntimeError(f"Preflight smoke failed: {result.get('error', 'unknown error')}")

    def _run_benchmark(
        self,
        *,
        service_url: str,
        model_name: str,
        suite_name: str,
        settings: dict[str, Any],
        benchmark: dict[str, Any],
    ) -> dict[str, Any]:
        command = self._build_vllm_command(
            service_url=service_url,
            model_name=model_name,
            settings=settings,
            benchmark=benchmark,
        )
        environment = os.environ.copy()
        environment["HF_DATASETS_CACHE"] = str(settings.get("hf_datasets_cache", "/ai_models"))
        command_str = shlex.join(command)

        result: dict[str, Any] = {
            "config_name": benchmark["config_name"],
            "model_name": model_name,
            "profile_id": os.environ.get("AIM_PROFILE_ID", ""),
            "suite": suite_name,
            "concurrency": benchmark["max_concurrency"],
            "num_prompts": benchmark["num_prompts"],
            "min_audio_sec": benchmark["min_audio_sec"],
            "max_audio_sec": benchmark["max_audio_sec"],
            "successful_reqs": None,
            "duration": None,
            "req_throughput": None,
            "output_token_throughput": None,
            "total_token_throughput": None,
            "mean_ttft": None,
            "median_ttft": None,
            "p75_ttft": None,
            "p90_ttft": None,
            "p99_ttft": None,
            "mean_tpot": None,
            "median_tpot": None,
            "p75_tpot": None,
            "p90_tpot": None,
            "p99_tpot": None,
            "mean_itl": None,
            "median_itl": None,
            "p75_itl": None,
            "p90_itl": None,
            "p99_itl": None,
            "mean_e2el": None,
            "median_e2el": None,
            "p75_e2el": None,
            "p90_e2el": None,
            "p99_e2el": None,
            "success": False,
            "error": "",
            "command": command_str,
        }

        logger = logging.getLogger("aim_runtime.whisper_harness")
        logger.info("Running command: %s", command_str)

        started_at = time.monotonic()
        try:
            completed = subprocess.run(
                command,
                capture_output=True,
                text=True,
                timeout=int(settings.get("timeout_seconds_per_config", 14400)),
                env=environment,
                check=False,
            )
        except subprocess.TimeoutExpired:
            result["duration_seconds"] = time.monotonic() - started_at
            result["error"] = f"Timeout after {settings.get('timeout_seconds_per_config', 14400)}s"
            return result

        output = (completed.stdout or "") + "\n" + (completed.stderr or "")
        result.update(self._parse_benchmark_output(output))
        result["duration_seconds"] = time.monotonic() - started_at

        if completed.returncode != 0:
            result["error"] = self._truncate_error(
                completed.stderr or completed.stdout or f"exit code {completed.returncode}"
            )
            return result

        if not self._config_successful(result.get("successful_reqs"), int(benchmark["num_prompts"])):
            result["error"] = self._truncate_error(output or "Benchmark finished but successful_reqs below threshold")
            return result

        result["success"] = True
        return result

    @staticmethod
    def _export_results(payload: dict[str, Any], results: list[dict[str, Any]], output_dir: str) -> list[str]:
        destination = Path(output_dir)
        destination.mkdir(parents=True, exist_ok=True)

        json_path = destination / os.getenv("BENCHMARK_JSON_FILE", "benchmark_results.json")
        csv_path = destination / os.getenv("BENCHMARK_CSV_FILE", "benchmark_results.csv")

        json_path.write_text(json.dumps(payload, indent=2) + "\n")

        with csv_path.open("w", newline="") as csv_stream:
            writer = csv.DictWriter(csv_stream, fieldnames=CSV_FIELDS, extrasaction="ignore")
            writer.writeheader()
            writer.writerows(results)

        return [str(json_path), str(csv_path)]
