# Copyright © Advanced Micro Devices, Inc., or its affiliates.
#
# SPDX-License-Identifier: MIT

"""BentoML service for the Aurora 0.25 degree pretrained checkpoint.

Transport only: load the runner, write the forecast file, and return the
response envelope. ``/health`` and ``/v1/models`` come from the mounted
``canonical_api`` app.
"""

import base64
import os
import time

import bentoml
import config
from bentoml.exceptions import InvalidArgument
from canonical_api import build_canonical_app
from pydantic import BaseModel


class BentoArgs(BaseModel):
    # Overridden by AIM_PORT, or AURORALG_PORT for a hand-run service.
    port: int = config.PORT


args = bentoml.use_arguments(BentoArgs)


# The timeout is 1800 s because a multi-step 0.25 degree roll-out at this
# parameter count takes far longer than BentoML's 60 s default.
@bentoml.service(
    name="aurora-v1-0p25deg-pretrained-large-inference",
    traffic={"timeout": 1800},
    http={"port": args.port},
)
# The mounted app needs the same port the service binds, because /health
# forwards to /healthz on it.
@bentoml.asgi_app(build_canonical_app(args.port, config.MODEL_ID), path="/")
class AuroraLargeService:
    def __init__(self) -> None:
        from model_runner import AuroraLargeRunner  # noqa: PLC0415 -- heavy import, kept lazy

        self._runner = AuroraLargeRunner()

        # Loading here puts the 4.7 GiB weight download inside the readiness
        # window rather than inside the first request.
        if config.WARMUP_ON_STARTUP:
            self._runner.ensure_loaded()

    @bentoml.api(route="/v1/inference")
    def predict(
        self,
        input_netcdf_b64: str = "",
        # A file already visible to the container, named relative to the mounted
        # input directory. Preferred over base64 for real initial conditions,
        # which run to hundreds of megabytes.
        input_netcdf_path: str = "",
        forecast_steps: int = 1,
        output_format: str = "",
        output_dir: str = "",
        # Ignored by this global forecaster; accepted so the request shape is
        # the same across weather models.
        conditioning_netcdf_b64: str = "",
        num_samples: int = 1,
        # Also return the forecast bytes inline, which inflates the response by
        # about a third.
        return_data_base64: bool = False,
    ) -> dict:
        """Run an Aurora forecast and return the response envelope.

        The initial condition arrives either inline as ``input_netcdf_b64`` or
        as ``input_netcdf_path`` naming a file under the mounted input
        directory. One is required; both together are rejected.

        ``forecast_steps`` is the number of roll-out steps, each one model time
        step of 6 hours. ``lead_time`` in the response keeps every step,
        including the initial condition at 0 h.

        Precision and graph compilation come from ``AURORALG_AUTOCAST`` and
        ``AURORALG_COMPILE``, and are reported back under ``optimizations``.
        """
        supplied = [
            name
            for name, value in (
                ("input_netcdf_b64", input_netcdf_b64),
                ("input_netcdf_path", input_netcdf_path),
            )
            if value
        ]
        if len(supplied) > 1:
            raise InvalidArgument("input_netcdf_b64 and input_netcdf_path are mutually exclusive")
        if not supplied:
            raise InvalidArgument("an initial condition is required: set input_netcdf_path or input_netcdf_b64")
        # Resolved before the model is touched, so a bad path fails here rather
        # than several frames into the runner.
        input_path = _resolve_input_path(input_netcdf_path) if input_netcdf_path else ""
        fmt = (output_format or config.OUTPUT_FORMAT or "netcdf").lower()
        if fmt != "netcdf":
            raise InvalidArgument("output_format %r is not supported; only 'netcdf'" % fmt)
        out_dir = _resolve_output_dir(output_dir)
        os.makedirs(out_dir, exist_ok=True)
        out_path = os.path.join(out_dir, "forecast_%d.nc" % int(time.time() * 1000))

        result = self._runner.forecast_to_file(
            input_netcdf_b64=input_netcdf_b64,
            input_netcdf_path=input_path,
            forecast_steps=max(1, int(forecast_steps)),
            conditioning_netcdf_b64=conditioning_netcdf_b64,
            num_samples=max(1, int(num_samples)),
            out_path=out_path,
        )

        response = {
            "output_path": out_path,
            "output_format": fmt,
            "dims": result["dims"],
            "variables": result["variables"],
            "all_finite": bool(result["all_finite"]),
            "optimizations": config.OPT_SUMMARY,
        }
        if return_data_base64:
            with open(out_path, "rb") as fh:
                response["data_base64"] = base64.b64encode(fh.read()).decode("ascii")
        return response


def _resolve_input_path(requested):
    """Resolve ``input_netcdf_path`` inside the mounted input directory.

    The request names a file relative to ``config.INPUT_DIR``. An absolute path,
    a ``..`` segment or a symlink leading outside that directory is rejected,
    and containment is checked on the resolved path before the file is opened.
    """
    if not config.INPUT_DIR:
        raise InvalidArgument(
            "input_netcdf_path requires a mounted input directory; set "
            "AIM_INPUT_PATH to one, or send the file inline as input_netcdf_b64"
        )
    base = os.path.realpath(config.INPUT_DIR)
    resolved = os.path.realpath(os.path.join(base, requested))
    if resolved != base and not resolved.startswith(base + os.sep):
        raise InvalidArgument("input_netcdf_path must resolve within %s" % base)
    if not os.path.isfile(resolved):
        raise InvalidArgument("input_netcdf_path %r does not name a file under %s" % (requested, base))
    return resolved


def _resolve_output_dir(requested):
    """Resolve ``output_dir`` under the configured output base.

    A request picks a subdirectory of ``AIM_OUTPUT_PATH``; anything resolving
    outside it is rejected, before any filesystem access.
    """
    base = os.path.realpath(config.DEFAULT_OUTPUT_DIR)
    if not requested:
        return base
    resolved = os.path.realpath(os.path.join(base, requested))
    if resolved != base and not resolved.startswith(base + os.sep):
        raise InvalidArgument("output_dir must resolve within %s" % base)
    return resolved
