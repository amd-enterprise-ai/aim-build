# Copyright © Advanced Micro Devices, Inc., or its affiliates.
#
# SPDX-License-Identifier: MIT

"""BentoML service for ECMWF AIFS Single 2.0 or AIFS ENS 2.0 (``AIFS_VARIANT``).

Transport only: load the runner, write the forecast file, and return the
response envelope. ``/health`` and ``/v1/models`` come from the mounted
``canonical_api`` app.
"""

import base64
import binascii
import datetime
import os
import secrets
import tempfile
import time

import bentoml
import config
import data_utils
from bentoml.exceptions import InvalidArgument
from canonical_api import build_canonical_app
from pydantic import BaseModel


class BentoArgs(BaseModel):
    # Overridden by AIM_PORT, or AIFS_PORT for a hand-run service.
    port: int = config.PORT


args = bentoml.use_arguments(BentoArgs)


# A 10-day N320 roll-out with all output fields takes minutes, far past
# BentoML's 60 s default.
@bentoml.service(
    name="aifs-single-2-0-inference",
    traffic={"timeout": 3600},
    http={"port": args.port},
)
# The mounted app needs the same port the service binds, because /health
# forwards to /healthz on it.
@bentoml.asgi_app(build_canonical_app(args.port, config.MODEL_ID), path="/")
class AifsSingleService:
    def __init__(self) -> None:
        from model_runner import AifsRunner  # noqa: PLC0415 -- heavy import, kept lazy

        self._runner = AifsRunner()

        # Puts the checkpoint load inside the readiness window rather than the
        # first request.
        if config.WARMUP_ON_STARTUP:
            self._runner.ensure_loaded()

    @bentoml.api(route="/v1/inference")
    def predict(
        self,
        # The initial condition: one GRIB2 file with the step-0 0.25-degree fields
        # at the initial time and 6 hours earlier, inline or as a file under
        # AIM_INPUT_PATH.
        input_grib_b64: str = "",
        input_grib_path: str = "",
        # Initial time, YYYY-MM-DDTHH:MM (UTC). Optional with a GRIB file, where it
        # defaults to the latest time in the file; required with input_dir.
        init_time: str = "",
        # Instead of a file: a subdirectory of AIM_INPUT_PATH with one
        # YYYYMMDDHH.grib2 file per time, such as the image's own test data.
        input_dir: str = "",
        forecast_steps: int = 1,
        # Output fields to write; empty writes every model output.
        variables: list[str] = [],  # noqa: B006 -- BentoML reads the default for the schema
        output_format: str = "",
        output_dir: str = "",
        # Ensemble members (AIFS_VARIANT=ens); more than 1 adds a member dimension.
        # Ignored by the deterministic Single model.
        num_samples: int = 1,
        # Also return the forecast bytes inline, which inflates the response by
        # about a third.
        return_data_base64: bool = False,
        # Ensemble only: member i uses torch seed seed + i; empty draws a random seed.
        seed: int | None = None,
        # Ensemble only: "analysis" starts every member from the same input files;
        # "perturbed" starts member i from input_dir's members/pf<i+1>.
        initial_states: str = "analysis",
    ) -> dict:
        """Run an AIFS forecast and return the response envelope.

        The initial condition arrives as ``input_grib_b64`` or ``input_grib_path``,
        or as ``init_time`` with ``input_dir``; the three are mutually exclusive.
        ``forecast_steps`` is the number of 6-hour steps. ``lead_time`` in the
        output keeps every step, including the initial condition at 0 h.
        """
        supplied = [
            name
            for name, value in (
                ("input_grib_b64", input_grib_b64),
                ("input_grib_path", input_grib_path),
                ("input_dir", input_dir),
            )
            if value
        ]
        if len(supplied) > 1:
            raise InvalidArgument("%s are mutually exclusive" % " and ".join(supplied))
        from_file = bool(input_grib_b64 or input_grib_path)
        if not from_file and not init_time:
            raise InvalidArgument(
                "an initial condition is required: send input_grib_b64 or input_grib_path, "
                "or init_time with the files under input_dir"
            )
        steps = int(forecast_steps)
        if steps < 1 or steps > config.MAX_FORECAST_STEPS:
            raise InvalidArgument("forecast_steps must be between 1 and %d" % config.MAX_FORECAST_STEPS)
        members = 1
        seeds = None
        start = (initial_states or "analysis").lower()
        if config.ENSEMBLE:
            members = int(num_samples)
            if members < 1 or members > config.MAX_MEMBERS:
                raise InvalidArgument("num_samples must be between 1 and %d" % config.MAX_MEMBERS)
            if start not in ("analysis", "perturbed"):
                raise InvalidArgument("initial_states must be 'analysis' or 'perturbed'")
            if start == "perturbed" and from_file:
                raise InvalidArgument("initial_states=perturbed needs input_dir holding members/pfNN")
            base = int(seed) if seed is not None else secrets.randbelow(2**31 - 2**16)
            seeds = [base + i for i in range(members)]
        elif start != "analysis":
            raise InvalidArgument("initial_states=%r needs AIFS_VARIANT=ens" % initial_states)
        fmt = (output_format or config.OUTPUT_FORMAT or "netcdf").lower()
        if fmt != "netcdf":
            raise InvalidArgument("output_format %r is not supported; only 'netcdf'" % fmt)
        out_dir = _resolve_output_dir(output_dir)

        n_vars = len(variables) if variables else len(self._runner.output_variables)
        size_gb = members * (steps + 1) * n_vars * data_utils.N320_POINTS * 4 / 1e9
        if size_gb > config.MAX_OUTPUT_GB:
            raise InvalidArgument(
                "forecast would hold %.1f GB (> %.0f GB); request fewer variables, steps or members"
                % (size_gb, config.MAX_OUTPUT_GB)
            )

        tmp_path = None
        member_dirs = None
        try:
            if input_grib_b64:
                source = tmp_path = _write_inline_grib(input_grib_b64)
            elif input_grib_path:
                source = _resolve_input_file(input_grib_path)
            else:
                source = _resolve_input_dir(input_dir)
            if from_file:
                when = _file_init_time(source, init_time)
            else:
                when = _parse_init_time(init_time)
                if start == "perturbed":
                    member_dirs = [_member_dir(source, i + 1) for i in range(members)]
                for d in member_dirs or [source]:
                    _require_init_inputs(d, when)
            os.makedirs(out_dir, exist_ok=True)
            out_path = os.path.join(out_dir, "forecast_%d_%s.nc" % (int(time.time() * 1000), secrets.token_hex(4)))
            result = self._runner.forecast_to_file(
                source, when, steps, out_path, variables or None, seeds=seeds, member_dirs=member_dirs
            )
        except ValueError as exc:
            raise InvalidArgument(str(exc)) from exc
        finally:
            if tmp_path:
                os.unlink(tmp_path)

        response = {
            "output_path": out_path,
            "output_format": fmt,
            "dims": result["dims"],
            "variables": result["variables"],
            "all_finite": bool(result["all_finite"]),
            "optimizations": config.OPT_SUMMARY,
            "init_time": when.strftime("%Y-%m-%dT%H:%M:%SZ"),
            "lead_time_hours": result["lead_time_hours"],
            "timings_s": result["timings_s"],
        }
        if config.ENSEMBLE:
            response.update(num_samples=members, seeds=seeds, initial_states=start)
        if return_data_base64:
            with open(out_path, "rb") as fh:
                response["data_base64"] = base64.b64encode(fh.read()).decode("ascii")
        return response


def _parse_init_time(value):
    try:
        return data_utils.parse_time(value)
    except ValueError as exc:
        raise InvalidArgument(str(exc)) from exc


def _write_inline_grib(b64):
    """Decode ``input_grib_b64`` to a temporary file and return its path; the caller removes it."""
    try:
        data = base64.b64decode(b64, validate=True)
    except (binascii.Error, ValueError) as exc:
        raise InvalidArgument("input_grib_b64 is not valid base64: %s" % exc) from exc
    if not data.startswith(b"GRIB"):
        raise InvalidArgument("input_grib_b64 is not a GRIB file")
    fd, path = tempfile.mkstemp(suffix=".grib2", prefix="aifs_input_")
    with os.fdopen(fd, "wb") as fh:
        fh.write(data)
    return path


def _resolve_input_file(requested):
    """Resolve ``input_grib_path`` to a file inside AIM_INPUT_PATH, rejecting anything that leaves it."""
    if not config.INPUT_DIR:
        raise InvalidArgument(
            "input_grib_path requires a mounted input directory; set AIM_INPUT_PATH to one, "
            "or send the file inline as input_grib_b64"
        )
    base = os.path.realpath(config.INPUT_DIR)
    resolved = os.path.realpath(os.path.join(base, requested))
    if resolved != base and not resolved.startswith(base + os.sep):
        raise InvalidArgument("input_grib_path must resolve within %s" % base)
    if not os.path.isfile(resolved):
        raise InvalidArgument("input_grib_path %r does not name a file under %s" % (requested, base))
    return resolved


def _file_init_time(path, init_time):
    """The initial time of a GRIB input: ``init_time`` if given, else its latest time; both it and 6 h earlier must be in the file."""
    try:
        times = data_utils.file_times(path)
    except Exception as exc:  # noqa: BLE001 -- any decode failure is the caller's file
        raise InvalidArgument("cannot read the GRIB input: %s" % exc) from exc
    if not times:
        raise InvalidArgument("the GRIB input holds no fields")
    when = _parse_init_time(init_time) if init_time else times[-1]
    prev = when - datetime.timedelta(hours=data_utils.TIMESTEP_HOURS)
    if when not in times or prev not in times:
        raise InvalidArgument(
            "the GRIB input must hold fields at %s and %s; it holds %s"
            % (prev.isoformat(), when.isoformat(), ", ".join(t.isoformat() for t in times))
        )
    return when


def _resolve_input_dir(requested):
    """Resolve ``input_dir`` inside AIM_INPUT_PATH, rejecting anything that leaves it."""
    if not config.INPUT_DIR:
        raise InvalidArgument("no input directory; set AIM_INPUT_PATH")
    base = os.path.realpath(config.INPUT_DIR)
    resolved = os.path.realpath(os.path.join(base, requested)) if requested else base
    if resolved != base and not resolved.startswith(base + os.sep):
        raise InvalidArgument("input_dir must resolve within %s" % base)
    if not os.path.isdir(resolved):
        raise InvalidArgument("input_dir %r is not a directory under %s" % (requested, base))
    return resolved


def _require_init_inputs(in_dir, when):
    """Raise InvalidArgument unless ``in_dir`` holds inputs at ``when`` and 6 h earlier."""
    times = data_utils.available_times(in_dir)
    prev = when - datetime.timedelta(hours=data_utils.TIMESTEP_HOURS)
    if when not in times or prev not in times:
        raise InvalidArgument(
            "init_time %s needs inputs at %s and %s in %s; available %s to %s"
            % (
                when.isoformat(),
                prev.isoformat(),
                when.isoformat(),
                in_dir,
                times[0].isoformat() if times else "none",
                times[-1].isoformat() if times else "none",
            )
        )


def _member_dir(in_dir, number):
    """Directory of perturbed member ``number`` under ``in_dir``."""
    path = os.path.join(in_dir, config.MEMBERS_SUBDIR, "pf%02d" % number)
    if not os.path.isdir(path):
        base = os.path.join(in_dir, config.MEMBERS_SUBDIR)
        have = sorted(os.listdir(base)) if os.path.isdir(base) else []
        raise InvalidArgument("no perturbed member %d under %s (have %s)" % (number, base, have or "none"))
    return path


def _resolve_output_dir(requested):
    """Resolve ``output_dir`` under the configured output base, rejecting anything outside it."""
    base = os.path.realpath(config.DEFAULT_OUTPUT_DIR)
    if not requested:
        return base
    resolved = os.path.realpath(os.path.join(base, requested))
    if resolved != base and not resolved.startswith(base + os.sep):
        raise InvalidArgument("output_dir must resolve within %s" % base)
    return resolved
