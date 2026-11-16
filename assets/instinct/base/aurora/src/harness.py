# Copyright © Advanced Micro Devices, Inc., or its affiliates.
#
# SPDX-License-Identifier: MIT

# Harness for the Aurora 0.25 degree pretrained checkpoint.
#
# Two import rules, both needed for harness discovery to find this class:
#   1. Import the base module, not the class. `from weather_forecast_harness
#      import WeatherForecastHarness` puts the base class in this namespace,
#      where subclass enumeration can pick it instead of this one.
#   2. No `from __future__ import annotations`; it breaks discovery.

import os

import weather_forecast_harness

from aim_runtime.harness import CheckInfo, CheckResult, CheckResultType, CheckScope

# Aurora's variable set; input and output carry the same one. Must match
# model_runner.py, or forecast_sanity fails on a correct forecast.
SURFACE_VARS = ("2t", "10u", "10v", "msl")
ATMOS_VARS = ("z", "u", "v", "t", "q")

# `time` is the initialisation time, a length-1 dimension, not the per-step
# axis. `lead_time` is the forecast horizon in hours and keeps every step.
# `level` is checked so upper-air fields cannot quietly become surface-only.
EXPECTED_DIMS = ("time", "lead_time", "level", "lat", "lon")
EXPECTED_VARIABLES = SURFACE_VARS + ATMOS_VARS


class AuroraLargeLeafHarness(weather_forecast_harness.WeatherForecastHarness):
    ENV_PREFIX = "AURORALG"

    CHECKS = [
        CheckInfo(
            name="accelerator_visible",
            result_type=CheckResultType.PASS_FAIL,
            scope=CheckScope.RUNTIME,
            description="torch.cuda.is_available() is True inside the container, and the device name is reported.",
        ),
        CheckInfo(
            name="weights_resolvable",
            result_type=CheckResultType.PASS_FAIL,
            scope=CheckScope.RUNTIME,
            description="Reports where aurora-0.25-pretrained.ckpt will come from: a HuggingFace "
            "download at startup, or an AURORALG_WEIGHTS_DIR override. Does not fetch the "
            "4.7 GiB checkpoint itself.",
        ),
        CheckInfo(
            name="service_health",
            result_type=CheckResultType.PASS_FAIL,
            scope=CheckScope.OFFLINE,
            description="HTTP GET /health, which forwards to BentoML's /healthz.",
        ),
        CheckInfo(
            name="predict_smoke",
            result_type=CheckResultType.PASS_FAIL,
            scope=CheckScope.OFFLINE,
            description="HTTP POST /v1/inference for one 1-step forecast on the mounted input; asserts a non-empty "
            "response carrying an output, dims and all_finite.",
        ),
        CheckInfo(
            name="forecast_sanity",
            result_type=CheckResultType.PASS_FAIL,
            scope=CheckScope.OFFLINE,
            description="Parses the /v1/inference response: (time, lead_time, level, lat, lon) are present, "
            "the four surface and five upper-air variables are present, and every value is finite.",
        ),
        CheckInfo(
            name="profile_facts",
            result_type=CheckResultType.PASS_FAIL,
            scope=CheckScope.RUNTIME,
            description="The input file exists under AIM_INPUT_PATH: AIM_DATA_PATH if the environment or "
            "profile sets it, else the default filename in this harness. Grid, variables and the other "
            "model facts come from this harness, not the profile.",
        ),
        CheckInfo(
            name="grid_matches",
            result_type=CheckResultType.PASS_FAIL,
            scope=CheckScope.OFFLINE,
            description="Response dims are exactly the declared grid: time=1, level=13, lat=720, lon=1440.",
        ),
        CheckInfo(
            name="bad_input",
            result_type=CheckResultType.PASS_FAIL,
            scope=CheckScope.OFFLINE,
            description="Ten malformed /v1/inference requests. Fails if unreadable input comes back as a "
            "forecast, a request gets no answer, or the service stops answering; a 500 rather than a 4xx "
            "is a warning.",
        ),
        CheckInfo(
            name="concurrent_requests",
            result_type=CheckResultType.PASS_FAIL,
            scope=CheckScope.OFFLINE,
            description="Three rollouts of differing length at once: each answer matches its own request, "
            "each writes its own file, and /health stays 200 throughout.",
        ),
        CheckInfo(
            name="request_fields",
            result_type=CheckResultType.SCORE,
            scope=CheckScope.OFFLINE,
            description="Coverage table of the request fields the model honours (forecast_steps, output_dir, "
            "return_data_base64) and the fields it accepts and ignores. Does not gate.",
        ),
        CheckInfo(
            name="lead_zero",
            result_type=CheckResultType.PASS_FAIL,
            scope=CheckScope.OFFLINE,
            description="Lead 0 of the forecast is the initial condition handed back. Skipped unless "
            "AIM_INPUT_PATH holds the file the profile names.",
        ),
        CheckInfo(
            name="latency",
            result_type=CheckResultType.SCORE,
            scope=CheckScope.OFFLINE,
            description="Wall-clock p50/p95 over N /v1/inference requests, reported with the active "
            "autocast and compile settings.",
        ),
    ]

    # The framework-specific check: torch's view of the GPU.
    def check_accelerator_visible(self):
        try:
            import torch  # noqa: PLC0415

            ok = bool(torch.cuda.is_available())
            detail = "torch.cuda: " + (torch.cuda.get_device_name(0) if ok else "no GPU visible")
            return CheckResult("accelerator_visible", CheckResultType.PASS_FAIL, ok, ok, detail)
        except Exception as exc:  # noqa: BLE001
            return CheckResult("accelerator_visible", CheckResultType.PASS_FAIL, False, False, str(exc))

    def _check_weights_resolvable(self):
        """Report where the checkpoint will come from, without fetching it.

        Reads the same config.weights_candidates() list the runner loads from,
        so the two cannot disagree. Passes either way.
        """
        import config  # noqa: PLC0415

        for path in config.weights_candidates():
            if os.path.isfile(path):
                size = os.path.getsize(path)
                return CheckResult(
                    "weights_resolvable",
                    CheckResultType.PASS_FAIL,
                    True,
                    True,
                    "AURORALG_WEIGHTS_DIR override in use: %s (%d bytes)" % (path, size),
                )
        return CheckResult(
            "weights_resolvable",
            CheckResultType.PASS_FAIL,
            True,
            True,
            "will download %s (4.7 GiB) from HuggingFace microsoft/aurora at startup" % config.CHECKPOINT,
        )

    def prereq_checks(self, config):
        if CheckScope.OFFLINE not in config.check_scopes and CheckScope.RUNTIME not in config.check_scopes:
            return []
        return [self.check_accelerator_visible(), self._check_weights_resolvable()]

    def smoke_request(self):
        # Every check and benchmark request starts from this body, with single
        # fields overridden.
        name = self._input_name or self._facts.get("input_filename")
        return {"input_netcdf_path": name, "forecast_steps": 1}

    # What the checks compare the service against; input_filename is the fallback when AIM_DATA_PATH is unset.
    def default_facts(self):
        return {
            "dims": EXPECTED_DIMS,
            "grid": {"time": 1, "level": 13, "lat": 720, "lon": 1440},
            "variables": EXPECTED_VARIABLES,
            "ignored_fields": {"num_samples": 4, "conditioning_netcdf_b64": ""},
            "concurrent_requests": 3,
            "input_filename": "aurora_input.nc",
        }
