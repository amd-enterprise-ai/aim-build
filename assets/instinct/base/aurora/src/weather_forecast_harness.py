# Copyright © Advanced Micro Devices, Inc., or its affiliates.
#
# SPDX-License-Identifier: MIT

# No `from __future__ import annotations` here; postponed annotations break
# harness discovery.

"""Shared harness base for weather-forecasting models served over HTTP.

A per-model subclass declares ``CHECKS`` and ``ENV_PREFIX``, and implements:

  * ``check_accelerator_visible`` — the framework-specific part; torch, jax and
    onnxruntime each see the GPU differently.
  * ``smoke_request`` — the JSON body for one 1-step ``/v1/inference``. Every
    check and benchmark request starts from it.
  * ``default_facts`` — the grid, variables and other per-model values the
    checks compare against. A profile may name another input file with
    ``AIM_DATA_PATH``.

Everything else is shared. Checks run against a live service over HTTP rather
than re-running the model in process. The server is started through the AIM
layer, so this file holds no engine knowledge, and starting is a no-op when a
server is already running. Readiness polls ``GET /health``.

Per-run values such as ``benchmark_n_requests`` and ``service_url`` come from
the active profile.
"""

import atexit
import base64
import json
import logging
import os
import statistics
import subprocess
import tempfile
import time
from concurrent.futures import ThreadPoolExecutor

# Any/Optional/Tuple appear only in `# type:` comments, which linters do not
# parse; the noqa stops them being reported as unused and stripped.
from typing import Any, ClassVar, List, Optional, Tuple  # noqa: F401
from urllib.error import HTTPError, URLError
from urllib.parse import urlparse
from urllib.request import Request, urlopen

from aim_runtime.harness import (
    CheckInfo,
    CheckResult,
    CheckResultType,
    CheckScope,
    HarnessResult,
    ModelHarness,
    failed,
    passed,
    skipped,
)

logger = logging.getLogger(__name__)

# The HTTP surface. /health is the readiness path everything outside the
# container polls; it forwards to BentoML's own /healthz.
HEALTH_PATH = "/health"
READYZ_PATH = "/healthz"
PREDICT_PATH = "/v1/inference"
# Default port for a hand-run service; AIM_PORT overrides it.
DEFAULT_PORT = int(os.environ.get("AIM_PORT", "8000"))
DEFAULT_BENCH_REQUESTS = 10
DEFAULT_HEALTH_POLL_S = 2.0


def latency_stats(latencies_ms: List[float]) -> dict:
    """Summary statistics over a list of wall-clock millisecond samples.

    Returns ``{}`` for empty input.
    """
    if not latencies_ms:
        return {}
    s = sorted(latencies_ms)
    n = len(s)
    p50 = s[n // 2]
    p95 = s[min(n - 1, int(n * 0.95))]
    total_s = sum(s) / 1000.0
    return {
        "mean_ms": round(statistics.fmean(s), 1),
        "p50_ms": round(p50, 1),
        "p95_ms": round(p95, 1),
        "min_ms": round(min(s), 1),
        "max_ms": round(max(s), 1),
        "throughput_req_per_sec": round((n / total_s), 4) if total_s > 0 else 0.0,
    }


class WeatherForecastHarness(ModelHarness):
    """HTTP harness base for weather-forecasting models served via BentoML.

    A subclass declares ``ENV_PREFIX`` and ``CHECKS``, implements
    ``check_accelerator_visible``, and returns its ``smoke_request`` body and
    ``default_facts``. The HTTP framing, response handling, server lifecycle,
    ``/health`` polling, validate/benchmark/evaluate orchestration and latency
    statistics are inherited unchanged.
    """

    ENGINE: ClassVar[str] = "bentoml"

    # Namespaces the model's environment variables, and matches the ENV_PREFIX
    # in its config.py.
    ENV_PREFIX: ClassVar[str] = ""

    # The per-model check list. Names and scopes vary by model; the
    # orchestration below does not.
    CHECKS: ClassVar[List[CheckInfo]] = []

    DEFAULT_PORT: ClassVar[int] = DEFAULT_PORT
    DEFAULT_BENCH_REQUESTS: ClassVar[int] = DEFAULT_BENCH_REQUESTS

    def __init__(self):
        self._server_proc = None  # type: Optional[subprocess.Popen]
        self._facts = {}  # type: dict
        self._input = None  # type: Optional[str]
        self._input_name = ""
        self._input_reason = ""

    # ----- discovery / introspection ------------------------------------ #

    def list_checks(self):
        return list(self.CHECKS)

    # ----- health ------------------------------------------------------- #

    def health_check(self, service_url, timeout_seconds=60):
        """Poll ``GET /health`` until it answers 200, or the timeout expires.

        Overrides the default ``/v1/models``, which BentoML does not serve.
        """
        deadline = time.monotonic() + timeout_seconds
        while time.monotonic() < deadline:
            try:
                with urlopen(Request(service_url + HEALTH_PATH), timeout=5) as resp:
                    if resp.status == 200:
                        return True
            except (HTTPError, URLError, OSError) as exc:
                logger.debug("health_check retry after transient error: %s", exc)
            time.sleep(DEFAULT_HEALTH_POLL_S)
        return False

    # ----- validate ----------------------------------------------------- #

    def validate(self, config):
        """Run the checks against the service, starting it first if it is down.

        Runs the prerequisite checks, then health, then one inference and the
        checks that read its result. Stops the server only if this call started
        it. The inference checks are skipped when no input is mounted, and the
        service is not touched unless the offline scope is requested.
        """
        self._load_run_context(config)
        url = self._service_url(config)
        checks = [self.check_profile_facts(config)] + list(self.prereq_checks(config))
        if CheckScope.OFFLINE not in config.check_scopes:
            return HarnessResult(
                success=all(c.success for c in checks),
                summary=self._slug() + " validation: service checks skipped, offline scope not requested",
                checks=checks,
            )

        started = self._ensure_server(url, startup_timeout=config.timeout_seconds)
        try:
            healthy = self.health_check(url, timeout_seconds=config.timeout_seconds)
            checks.append(
                CheckResult(
                    "service_health",
                    CheckResultType.PASS_FAIL,
                    healthy,
                    healthy,
                    "" if healthy else "Service unreachable at " + url + HEALTH_PATH,
                )
            )
            if not healthy:
                return HarnessResult(success=False, summary="Health check failed", checks=checks)

            if self._input is None:
                checks.append(skipped("predict_smoke", "no input: " + self._input_reason))
                return HarnessResult(
                    success=all(c.success for c in checks),
                    summary=self._slug() + " validation: inference checks skipped, " + self._input_reason,
                    checks=checks,
                )

            smoke_check, smoke_context = self.run_predict_smoke(url, config.timeout_seconds)
            checks.append(smoke_check)

            if smoke_check.success:
                checks.append(self.check_forecast_sanity(smoke_context))
                checks.append(self.check_grid(smoke_context, config))
                checks.append(self.check_bad_input(url, config))
                checks.append(self.check_concurrent_requests(url, config))
                checks.append(self.check_request_fields(url, config))
                checks.append(self.check_lead_zero(url, config))
        finally:
            if started:
                self._stop_server()

        success = all(c.success for c in checks)
        passed = sum(c.success for c in checks)
        return HarnessResult(
            success=success,
            summary=self._slug() + " validation: %d/%d checks passed" % (passed, len(checks)),
            checks=checks,
        )

    # ----- benchmark ---------------------------------------------------- #

    def benchmark(self, config):
        """Wall-clock p50/p95 of ``/v1/inference`` over N requests.

        Times the same ``smoke_request`` body the smoke check uses. N comes from
        the profile's ``benchmark_n_requests``, then ``AIM_HARNESS_NUM_REQUESTS``,
        then the default.
        """
        self._load_run_context(config)
        if self._input is None:
            return HarnessResult(success=False, summary="Cannot benchmark — " + self._input_reason)
        url = self._service_url(config)
        started = self._ensure_server(url, startup_timeout=config.timeout_seconds)
        try:
            if not self.health_check(url, timeout_seconds=config.timeout_seconds):
                return HarnessResult(success=False, summary="Cannot benchmark — service unreachable")

            n = self._bench_requests(config)
            latencies_ms = []  # type: List[float]
            failures = []  # type: List[str]
            for i in range(n):
                t0 = time.perf_counter()
                try:
                    self.post_predict(url, self.smoke_request(), config.timeout_seconds)
                    latencies_ms.append((time.perf_counter() - t0) * 1000.0)
                except Exception as exc:  # noqa: BLE001
                    failures.append("req %d: %s" % (i, exc))
                    logger.warning("benchmark request %d/%d failed: %s", i + 1, n, exc)
        finally:
            if started:
                self._stop_server()

        if not latencies_ms:
            return HarnessResult(
                success=False,
                summary="All %d benchmark requests failed" % n,
                metrics={"failures": failures},
            )

        stats = latency_stats(latencies_ms)
        p50 = stats["p50_ms"]
        p95 = stats["p95_ms"]
        ok = len(failures) == 0 and n > 0
        check = CheckResult(
            "latency",
            CheckResultType.SCORE,
            ok,
            p50,
            "%d/%d ok; p50=%.1fms p95=%.1fms" % (len(latencies_ms), n, p50, p95),
        )
        return HarnessResult(
            success=ok,
            summary=self._slug() + " benchmark: p50=%.1fms over %d/%d requests" % (p50, len(latencies_ms), n),
            checks=[check],
            metrics=dict(
                stats,
                n_requests=n,
                succeeded=len(latencies_ms),
                failures=failures,
                optimizations=self._optimizations(),
            ),
        )

    # ----- evaluate ----------------------------------------------------- #

    def evaluate(self, config):
        """Output-structure sanity on a fresh forecast.

        Structure only; there is no scoring against a reference forecast.
        """
        self._load_run_context(config)
        if self._input is None:
            return HarnessResult(
                success=True,
                summary=self._slug() + " evaluation skipped: " + self._input_reason,
                checks=[skipped("forecast_sanity", "no input: " + self._input_reason)],
            )
        url = self._service_url(config)
        started = self._ensure_server(url, startup_timeout=config.timeout_seconds)
        try:
            if not self.health_check(url, timeout_seconds=config.timeout_seconds):
                return HarnessResult(success=False, summary="Cannot evaluate — service unreachable")
            smoke_check, smoke_context = self.run_predict_smoke(url, config.timeout_seconds)
            if not smoke_check.success:
                return HarnessResult(success=False, summary="Evaluate predict failed", checks=[smoke_check])
            sanity = self.check_forecast_sanity(smoke_context)
        finally:
            if started:
                self._stop_server()
        return HarnessResult(
            success=sanity.success,
            summary=self._slug() + " evaluation: output sanity only (no ground truth)",
            checks=[sanity],
        )

    # ----- prerequisites, run before the server starts ------------------ #

    def prereq_checks(self, config):
        """Checks that need no service. Override to add more."""
        return [self.check_accelerator_visible()]

    # ----- implemented by the subclass ---------------------------------- #

    def check_accelerator_visible(self):
        # type: () -> CheckResult
        """Whether the accelerator is visible to the model's framework.

        A torch model checks ``torch.cuda.is_available()``, a jax model checks
        ``jax.devices()``, an ONNX model checks the active execution providers.
        """
        raise NotImplementedError("%s must implement check_accelerator_visible()" % type(self).__name__)

    def smoke_request(self):
        # type: () -> dict
        """The JSON body for one 1-step ``/v1/inference`` request.

        Only the field names are model-specific. Every check and benchmark
        request starts from this body with single fields overridden.
        """
        raise NotImplementedError("%s must implement smoke_request()" % type(self).__name__)

    def default_facts(self):
        # type: () -> dict
        """Per-model values the checks compare the service against.

        Keys: ``dims``, ``grid``, ``variables``, ``ignored_fields``,
        ``concurrent_requests``, ``input_filename``.
        """
        return {}

    def expected_dims(self):
        # type: () -> Tuple[str, ...]
        return tuple(self._facts.get("dims") or ())

    def expected_variables(self):
        # type: () -> Tuple[str, ...]
        return tuple(self._facts.get("variables") or ())

    # ----- facts and input ---------------------------------------------- #

    def _load_run_context(self, config):
        # type: (Any) -> None
        """Resolve the facts and the mounted input once per run."""
        self._facts = self.facts(config)
        self._input, self._input_reason = self.real_input_path(config)
        self._input_name = self._request_name(self._input)

    def _request(self, **overrides):
        # type: (Any) -> dict
        """The smoke body with fields overridden."""
        body = dict(self.smoke_request())
        body.update(overrides)
        return body

    def facts(self, config):
        # type: (Any) -> dict
        """The per-model values, from ``default_facts``."""
        return dict(self.default_facts())

    def _declared_data_path(self, config):
        # type: (Any) -> str
        """``AIM_DATA_PATH`` from the environment, else from the profile."""
        from_env = os.environ.get("AIM_DATA_PATH", "").strip()
        if from_env:
            return from_env
        env_vars = (getattr(config, "profile", None) or {}).get("env_vars") or {}
        return str(env_vars.get("AIM_DATA_PATH") or "").strip()

    def check_profile_facts(self, config):
        # type: (Any) -> CheckResult
        """Require the input file, from AIM_DATA_PATH or the default name, to exist."""
        path, reason = self.real_input_path(config)
        if not path:
            return failed("profile_facts", reason)
        source = "AIM_DATA_PATH" if self._declared_data_path(config) else "default input filename"
        return passed("profile_facts", "input %s (%s)" % (path, source))

    def real_input_path(self, config):
        # type: (Any) -> Tuple[Optional[str], str]
        """The input file named by ``AIM_DATA_PATH`` under ``AIM_INPUT_PATH``, or ``(None, reason)``."""
        declared = self._declared_data_path(config) or self.facts(config).get("input_filename") or ""
        if not declared:
            return None, "neither AIM_DATA_PATH nor a default input filename is set"
        root = os.environ.get("AIM_INPUT_PATH", "").strip()
        if not root:
            return None, "AIM_INPUT_PATH is unset, so %s cannot be resolved" % declared
        path = os.path.join(root, declared)
        if not os.path.realpath(path).startswith(os.path.realpath(root) + os.sep):
            return None, "%s is outside AIM_INPUT_PATH %s" % (path, root)
        if not os.path.isfile(path):
            return None, "%s does not exist" % path
        return path, ""

    def _request_name(self, path):
        # type: (Optional[str]) -> str
        """The input as the service names it: a path under ``AIM_INPUT_PATH``."""
        if not path:
            return ""
        root = os.environ.get("AIM_INPUT_PATH", "").strip()
        if not root:
            return os.path.basename(path)
        base = os.path.realpath(root)
        resolved = os.path.realpath(path)
        if resolved.startswith(base + os.sep):
            return os.path.relpath(resolved, base)
        return os.path.basename(path)

    # ----- inference smoke and sanity ----------------------------------- #

    def run_predict_smoke(self, service_url, timeout_seconds=1800):
        # type: (str, int) -> Tuple[CheckResult, Any]
        """POST one ``/v1/inference`` and check the response.

        The response carries ``output_path`` or ``data_base64``, plus ``dims``,
        ``variables`` and ``all_finite``. Returns the check and the decoded
        response, which later checks read.
        """
        try:
            raw = self.post_predict(service_url, self.smoke_request(), timeout_seconds)
            payload = json.loads(raw.decode())
        except Exception as exc:  # noqa: BLE001
            return CheckResult("predict_smoke", CheckResultType.PASS_FAIL, False, False, str(exc)), {}
        has_output = bool(payload.get("output_path") or payload.get("data_base64"))
        ok = has_output and bool(payload.get("all_finite")) and bool(payload.get("dims"))
        detail = "output=%s dims=%s all_finite=%s" % (
            payload.get("output_path") or ("data_base64" if payload.get("data_base64") else None),
            payload.get("dims"),
            payload.get("all_finite"),
        )
        return CheckResult("predict_smoke", CheckResultType.PASS_FAIL, ok, ok, detail), payload

    def check_forecast_sanity(self, smoke_context):
        # type: (Any) -> CheckResult
        """Check the response has the expected dims and variables, and is finite."""
        payload = smoke_context or {}
        dims = payload.get("dims", {})
        present = set(dims.keys()) if isinstance(dims, dict) else set(dims or [])
        missing = [d for d in self.expected_dims() if d not in present]
        variables = list(payload.get("variables", []) or [])
        missing_vars = [v for v in self.expected_variables() if v not in variables]
        ok = not missing and not missing_vars and len(variables) > 0 and bool(payload.get("all_finite"))
        detail = "dims=%s n_vars=%d missing_dims=%s missing_vars=%s all_finite=%s" % (
            sorted(present),
            len(variables),
            missing,
            missing_vars,
            payload.get("all_finite"),
        )
        return CheckResult("forecast_sanity", CheckResultType.PASS_FAIL, ok, ok, detail)

    # ----- checks against the declared facts ----------------------------- #

    def check_grid(self, smoke_context, config):
        # type: (Any, Any) -> CheckResult
        """Response dims must be exactly the declared grid."""
        grid = self.facts(config).get("grid") or {}
        if not grid:
            return skipped("grid_matches", "no grid declared")
        dims = (smoke_context or {}).get("dims") or {}
        wrong = {key: dims.get(key) for key, size in grid.items() if dims.get(key) != size}
        if wrong:
            return failed("grid_matches", "declared %s, got %s" % (grid, wrong))
        return passed("grid_matches", "dims match the declared grid %s" % grid)

    def check_bad_input(self, service_url, config):
        # type: (str, Any) -> CheckResult
        """Unreadable input must not come back as a forecast, or stop the service."""
        not_netcdf = base64.b64encode(b"this is plain text, not a NetCDF file").decode()
        truncated = base64.b64encode(b"CDF\x01" + b"\x00" * 64).decode()
        cases = [
            ("invalid base64", {"input_netcdf_b64": "!!!not base64!!!"}, True),
            ("valid base64, not NetCDF", {"input_netcdf_b64": not_netcdf}, True),
            ("truncated NetCDF", {"input_netcdf_b64": truncated}, True),
            ("path outside the input dir", {"input_netcdf_path": "../../etc/hostname"}, True),
            ("empty body", {}, False),
            ("unknown field", self._request(no_such_field=7), False),
            ("forecast_steps=0", self._request(forecast_steps=0), False),
            ("forecast_steps=-1", self._request(forecast_steps=-1), False),
            ("forecast_steps='four'", self._request(forecast_steps="four"), False),
            ("unwritable output_dir", self._request(output_dir="/proc/nope"), False),
        ]
        fabricated, unhandled, unanswered = [], [], []
        for name, body, unreadable in cases:
            status, _ = self._post_status(service_url, body, config.timeout_seconds)
            if status == 0:
                unanswered.append(name)
            elif unreadable and status == 200:
                fabricated.append(name)
            elif status >= 500:
                unhandled.append("%s -> %d" % (name, status))
        if not self.health_check(service_url, timeout_seconds=30):
            return failed("bad_input", "service stopped answering after bad input")
        if fabricated:
            return failed("bad_input", "forecast returned for unreadable input: " + "; ".join(fabricated))
        if unanswered:
            return failed("bad_input", "no answer for: " + "; ".join(unanswered))
        # A 4xx and a 500 both refuse the request, so the shape of the refusal
        # is reported rather than failed.
        warnings = ["refused with 500 rather than 4xx: " + "; ".join(unhandled)] if unhandled else []
        return passed("bad_input", "%d bad requests, none answered with a forecast" % len(cases), warnings=warnings)

    def check_concurrent_requests(self, service_url, config):
        # type: (str, Any) -> CheckResult
        """Rollouts of differing length at once: no swapped answers, no shared file."""
        n = int(self.facts(config).get("concurrent_requests") or 3)
        steps = list(range(1, n + 1))
        timeout = config.timeout_seconds
        with ThreadPoolExecutor(max_workers=n) as pool:
            futures = [
                pool.submit(self._post_status, service_url, self._request(forecast_steps=s), timeout) for s in steps
            ]
            healthy = True
            while not all(f.done() for f in futures):
                healthy = self.health_check(service_url, timeout_seconds=10) and healthy
                time.sleep(DEFAULT_HEALTH_POLL_S)
            results = [(s, f.result()) for s, f in zip(steps, futures)]

        problems, paths = [], []
        for want, (status, payload) in results:
            if status != 200 or not isinstance(payload, dict):
                problems.append("steps=%d -> %s" % (want, status))
                continue
            got = (payload.get("dims") or {}).get("lead_time")
            if got != want + 1:
                problems.append("steps=%d answered with lead_time=%s" % (want, got))
            paths.append(payload.get("output_path"))
        if len(set(paths)) != len(paths):
            problems.append("concurrent calls shared an output_path")
        if not healthy:
            problems.append("health was not 200 throughout")
        if problems:
            return failed("concurrent_requests", "; ".join(problems))
        return passed("concurrent_requests", "%d concurrent rollouts, each answered with its own" % n)

    def check_request_fields(self, service_url, config):
        # type: (str, Any) -> CheckResult
        """Coverage table of which request fields the model honours. Does not gate."""
        facts = self.facts(config)
        timeout = config.timeout_seconds
        status, baseline = self._post_status(service_url, self._request(), timeout)
        if status != 200 or not isinstance(baseline, dict):
            return failed("request_fields", "baseline request -> %s" % status)
        dims = baseline.get("dims") or {}
        coverage = {}

        rollout = {}
        for s in (1, 2, 5):
            code, body = self._post_status(service_url, self._request(forecast_steps=s), timeout)
            rollout[s] = (body.get("dims") or {}).get("lead_time") if code == 200 and isinstance(body, dict) else None
        coverage["forecast_steps"] = all(rollout[s] == s + 1 for s in rollout)

        code, body = self._post_status(service_url, self._request(return_data_base64=True), timeout)
        raw = base64.b64decode(body.get("data_base64", "")) if code == 200 and isinstance(body, dict) else b""
        coverage["return_data_base64"] = raw[:3] in (b"CDF", b"\x89HD")

        sub = "aim_harness_out"
        code, body = self._post_status(service_url, self._request(output_dir=sub, output_format="netcdf"), timeout)
        path = body.get("output_path", "") if code == 200 and isinstance(body, dict) else ""
        coverage["output_dir"] = os.path.basename(os.path.dirname(path)) == sub

        # An ignored field must be accepted and change nothing: accepting
        # num_samples and then growing a sample axis is not ignoring it.
        for field, value in (facts.get("ignored_fields") or {}).items():
            code, body = self._post_status(service_url, self._request(**{field: value}), timeout)
            after = (body.get("dims") or {}) if code == 200 and isinstance(body, dict) else {}
            coverage[field] = code == 200 and after == dims

        honoured = sum(1 for ok in coverage.values() if ok)
        detail = "; ".join("%s=%s" % (f, "yes" if ok else "no") for f, ok in sorted(coverage.items()))
        return CheckResult(
            "request_fields",
            CheckResultType.SCORE,
            True,
            round(honoured / float(len(coverage)), 3),
            detail,
            gating=False,
        )

    def check_lead_zero(self, service_url, config):
        # type: (str, Any) -> CheckResult
        """Lead 0 must be the initial condition handed back."""
        if self._input is None:
            return skipped("lead_zero", "no input: " + self._input_reason)
        status, payload = self._post_status(service_url, self._request(return_data_base64=True), config.timeout_seconds)
        if status != 200 or not isinstance(payload, dict) or "data_base64" not in payload:
            return failed("lead_zero", "request -> %s %s" % (status, str(payload)[:160]))
        try:
            return self._compare_lead_zero(payload, self._input)
        except Exception as exc:  # noqa: BLE001
            return failed("lead_zero", "could not read the forecast back: %s" % exc)

    @staticmethod
    def _compare_lead_zero(payload, input_path):
        # type: (dict, str) -> CheckResult
        """Lead 0 against the input, per variable, scaled by the field's own spread.

        Level variables are skipped: the input may order its pressure levels
        differently, and a reordering would read as a mismatch.
        """
        import numpy as np  # noqa: PLC0415
        import xarray as xr  # noqa: PLC0415

        with tempfile.NamedTemporaryFile(suffix=".nc") as fh:
            fh.write(base64.b64decode(payload["data_base64"]))
            fh.flush()
            forecast = xr.open_dataset(fh.name)
            source = xr.open_dataset(input_path)
            try:
                mismatched, compared = [], 0
                for var in payload.get("variables") or []:
                    # An Aurora batch prefixes surface fields (surf_2t); an
                    # ERA5-shaped file names the same field t2m.
                    suffixed = [v for v in source.data_vars if v.endswith("_" + var)]
                    src = var if var in source else (suffixed[0] if len(suffixed) == 1 else None)
                    if src is None or var not in forecast:
                        continue
                    if "lead_time" not in forecast[var].dims or "level" in forecast[var].dims:
                        continue
                    got = forecast[var].isel(lead_time=0).values.astype("float64").squeeze()
                    want = source[src].values.astype("float64").squeeze()
                    while want.ndim > got.ndim:
                        want = want[-1]
                    if want.shape != got.shape:
                        crop = tuple(slice(0, min(a, b)) for a, b in zip(got.shape, want.shape))
                        got, want = got[crop], want[crop]
                    diff = float(np.nanmean(np.abs(got - want)))
                    spread = float(np.nanstd(want))
                    compared += 1
                    if spread > 0 and diff > 0.5 * spread:
                        mismatched.append("%s diff=%.5g spread=%.5g" % (var, diff, spread))
            finally:
                forecast.close()
                source.close()

        if not compared:
            return skipped("lead_zero", "no input variable name matched the output")
        if mismatched:
            return failed("lead_zero", "lead 0 is not the input: " + "; ".join(mismatched))
        return passed("lead_zero", "lead 0 matches the input for %d variables" % compared)

    @staticmethod
    def _optimizations():
        # type: () -> dict
        """The active knob settings, reported alongside benchmark numbers.

        Reads ``config.OPT_SUMMARY``, or ``{}`` if there is none.
        """
        try:
            import config  # noqa: PLC0415

            opt = getattr(config, "OPT_SUMMARY", {})
            return dict(opt) if isinstance(opt, dict) else {}
        except Exception:  # noqa: BLE001
            return {}

    # ----- HTTP helpers -------------------------------------------------- #

    @staticmethod
    def post_predict(service_url, body, timeout_seconds=1800):
        # type: (str, dict, int) -> bytes
        """POST a JSON body to ``/v1/inference`` and return the raw response bytes."""
        data = json.dumps(body).encode()
        req = Request(service_url + PREDICT_PATH, data=data, headers={"Content-Type": "application/json"})
        with urlopen(req, timeout=timeout_seconds) as resp:
            return resp.read()

    @staticmethod
    def _post_status(service_url, body, timeout_seconds=600):
        # type: (str, dict, int) -> Tuple[int, Any]
        """POST ``/v1/inference`` and return ``(status, decoded body)`` without raising.

        The checks that send bad requests need the status of a refusal, which
        :meth:`post_predict` raises away. Status 0 means no answer at all.
        """
        data = json.dumps(body).encode()
        req = Request(service_url + PREDICT_PATH, data=data, headers={"Content-Type": "application/json"})
        try:
            with urlopen(req, timeout=timeout_seconds) as resp:
                raw, status = resp.read(), resp.status
        except HTTPError as exc:
            raw, status = exc.read(), exc.code
        except (URLError, OSError) as exc:
            return 0, str(exc)
        try:
            return status, json.loads(raw.decode())
        except ValueError:
            return status, raw[:200].decode(errors="replace")

    # ----- server lifecycle ---------------------------------------------- #

    @staticmethod
    def aim_serve_command():
        # type: () -> Tuple[List[str], dict]
        """The argv and environment that start the model service.

        Asks the AIM layer to select the active profile and generate the launch
        command, rather than restating it here, so the service under test is
        configured the same way a deployment is.

        Profile selection has to succeed, so a standalone caller needs a
        selectable profile via ``AIM_PROFILE_ID`` or
        ``AIM_ALLOW_UNOPTIMIZED=true``.
        """
        from aim_runtime.aim_runtime import AIMRuntime  # noqa: PLC0415
        from aim_runtime.config import AIMConfig  # noqa: PLC0415

        runtime = AIMRuntime(AIMConfig.from_environment())
        profile = runtime.profile_selector.find_profile()
        logger.info("AIM layer selected profile: %s", profile.profile_handling.path)
        return runtime.command_generator.generate_execution_params(profile)

    def _ensure_server(self, url, startup_timeout=300):
        """Start the model service if ``url`` is not already healthy.

        Returns True only if this call started it, so the caller knows to stop
        it again.
        """
        if self.health_check(url, timeout_seconds=3):
            return False
        # Pin AIM_PORT to the port about to be health-checked, so the process
        # that starts binds it.
        os.environ["AIM_PORT"] = str(urlparse(url).port or self.DEFAULT_PORT)
        cmd, env_vars = self.aim_serve_command()
        env = os.environ.copy()
        env.update({key: str(value) for key, value in env_vars.items()})
        logger.info("Starting model service via the AIM layer: %s", " ".join(cmd))
        self._server_proc = subprocess.Popen(cmd, env=env)
        atexit.register(self._stop_server)
        if not self.health_check(url, timeout_seconds=startup_timeout):
            self._stop_server()
            raise RuntimeError("Model service did not become healthy at " + url)
        return True

    def _stop_server(self):
        proc = self._server_proc
        if proc is None:
            return
        self._server_proc = None
        if proc.poll() is None:
            proc.terminate()
            try:
                proc.wait(timeout=15)
            except subprocess.TimeoutExpired:
                proc.kill()

    # ----- internal helpers ---------------------------------------------- #

    def _bench_requests(self, config):
        """How many benchmark requests to send.

        Reads the profile's ``benchmark_n_requests`` first, then
        ``AIM_HARNESS_NUM_REQUESTS``, then the default. Both routes are in use:
        the profile carries the static value, and benchmark tooling passes the
        environment variable.
        """
        profile = getattr(config, "profile", None) or {}
        sources = (
            ("profile benchmark_n_requests", profile.get("benchmark_n_requests")),
            ("AIM_HARNESS_NUM_REQUESTS", os.environ.get("AIM_HARNESS_NUM_REQUESTS")),
        )
        for label, raw in sources:
            if raw is None or str(raw).strip() == "":
                continue
            try:
                n = int(raw)
            except (TypeError, ValueError):
                logger.warning("Ignoring non-integer %s=%r", label, raw)
                continue
            if n < 1:
                logger.warning("Ignoring %s=%r (must be >= 1)", label, raw)
                continue
            return n
        return self.DEFAULT_BENCH_REQUESTS

    def _service_url(self, config):
        return config.resolve_service_url()

    def _slug(self):
        """Short label for summary strings; defaults to the class name."""
        prefix = (self.ENV_PREFIX or type(self).__name__).strip()
        return prefix.lower() or type(self).__name__
