# Copyright © Advanced Micro Devices, Inc., or its affiliates.
#
# SPDX-License-Identifier: MIT

"""AuroraLargeRunner — loads the model and runs the forecast.

Loads ``aurora.AuroraPretrained`` (the 0.25 degree pretrained checkpoint, 1.26 B
parameters) once, builds a Batch from the supplied input, rolls it forward with
``aurora.rollout.rollout()``, and writes the forecast with
``data_utils.save_netcdf()``.

Aurora normalises and crops the grid internally, so this module passes raw
physical-unit values on the native 721x1440 grid.
"""

import base64
import contextlib
import os
import tempfile
from datetime import timezone

import config
import data_utils
import numpy as np
import torch

SURFACE_VARS = ("2t", "10u", "10v", "msl")
STATIC_VARS = ("lsm", "z", "slt")
ATMOS_VARS = ("z", "u", "v", "t", "q")
PRESSURE_LEVELS = data_utils.DEFAULT_PRESSURE_LEVELS  # 13 standard ERA5 levels (hPa)

HISTORY = 2  # Aurora max_history_size; the input time dim must equal this

_HF_REPO = "microsoft/aurora"


def _as_naive_utc(value):
    """Normalise a datetime to naive UTC for a numpy datetime64 coordinate."""
    if getattr(value, "tzinfo", None) is not None:
        return value.astimezone(timezone.utc).replace(tzinfo=None)
    return value


def _resolve_checkpoint_path():
    """A configured checkpoint path, or None to download it."""
    for path in config.weights_candidates():
        if os.path.isfile(path):
            return path
    return None


class AuroraLargeRunner:
    """Loads the model once; construction is cheap and the load is lazy."""

    def __init__(self):
        self._model = None
        self._timestep_hours = None
        self._patch_size = None
        if not torch.cuda.is_available():
            raise RuntimeError(
                "No GPU visible to torch. This AIM requires a ROCm GPU; check that /dev/kfd and "
                "/dev/dri are mapped into the container and that torch is the ROCm build."
            )
        self._device = "cuda"

    def ensure_loaded(self):
        """Load the model up front. Idempotent."""
        return self._load_model()

    def _load_model(self):
        if self._model is not None:
            return self._model
        from aurora import AuroraPretrained  # noqa: PLC0415 -- heavy import, kept lazy

        model = AuroraPretrained()
        local_ckpt = _resolve_checkpoint_path()
        if local_ckpt:
            # Loads strictly, so a checkpoint that does not match this
            # architecture fails here rather than producing a wrong forecast.
            model.load_checkpoint_local(local_ckpt)
        else:
            model.load_checkpoint(repo=_HF_REPO, name=config.CHECKPOINT)

        model = model.to(self._device)
        model.eval()

        # Read from the uncompiled module, so nothing depends on torch.compile
        # forwarding attributes.
        self._timestep_hours = model.timestep.total_seconds() / 3600.0
        self._patch_size = model.patch_size

        if config.COMPILE:
            # Expect a long warmup on the first call at this parameter count.
            model = torch.compile(model)
        self._model = model
        return model

    def _build_batch_from_netcdf_b64(self, input_netcdf_b64):
        """Decode an inline NetCDF into an ``aurora.Batch``."""
        raw = base64.b64decode(input_netcdf_b64)
        with tempfile.NamedTemporaryFile(suffix=".nc") as fh:
            fh.write(raw)
            fh.flush()
            return self._build_batch_from_netcdf_file(fh.name, "input_netcdf_b64")

    def _build_batch_from_netcdf_file(self, path, source):
        """Read a NetCDF file into an ``aurora.Batch``.

        ``aurora.Batch.from_netcdf`` is the inverse of ``Batch.to_netcdf``, not a
        general ERA5 reader: it expects variables prefixed ``surf_`` / ``static_``
        / ``atmos_``, dims ``(batch, history, latitude, longitude)`` and
        ``(batch, history, level, latitude, longitude)``, and coords
        ``latitude`` / ``longitude`` / ``time`` / ``level``.

        A file that does not match reads as empty variable dicts instead of
        raising, which ``_validate_batch`` then reports. ``source`` names the
        request field the file came from.
        """
        from aurora import Batch  # noqa: PLC0415

        batch = Batch.from_netcdf(path)
        self._validate_batch(batch, source)
        return batch

    @staticmethod
    def _validate_batch(batch, source):
        """Report a mis-shaped input here rather than as a shape error in the encoder."""
        missing_surf = [v for v in SURFACE_VARS if v not in batch.surf_vars]
        missing_static = [v for v in STATIC_VARS if v not in batch.static_vars]
        missing_atmos = [v for v in ATMOS_VARS if v not in batch.atmos_vars]
        if missing_surf or missing_static or missing_atmos:
            raise ValueError(
                "%s read as an incomplete aurora.Batch "
                "(missing surf=%s static=%s atmos=%s). It must be produced by "
                "aurora.Batch.to_netcdf(), which prefixes variables surf_/"
                "static_/atmos_; a plain ERA5-shaped NetCDF reads as empty "
                "variable dicts. Required: surf=%s static=%s atmos=%s."
                % (
                    source,
                    missing_surf,
                    missing_static,
                    missing_atmos,
                    list(SURFACE_VARS),
                    list(STATIC_VARS),
                    list(ATMOS_VARS),
                )
            )

        history = next(iter(batch.surf_vars.values())).shape[1]
        if history != HISTORY:
            raise ValueError(
                "%s carries history=%d time steps, but this model "
                "requires exactly %d (Aurora max_history_size)." % (source, history, HISTORY)
            )

        levels = tuple(int(lv) for lv in batch.metadata.atmos_levels)
        if levels != tuple(int(lv) for lv in PRESSURE_LEVELS):
            raise ValueError(
                "%s carries atmos_levels=%s, but this AIM reports "
                "level=%s in its response contract. Supply the 13 standard ERA5 "
                "pressure levels." % (source, list(levels), list(PRESSURE_LEVELS))
            )

    def forecast_to_file(
        self,
        input_netcdf_b64,
        input_netcdf_path,
        forecast_steps,
        conditioning_netcdf_b64,  # unused by a global forecaster
        num_samples,  # unused by a global forecaster
        out_path,
    ):
        from aurora.rollout import rollout  # noqa: PLC0415

        model = self._load_model()
        if input_netcdf_b64:
            batch = self._build_batch_from_netcdf_b64(input_netcdf_b64)
        else:
            batch = self._build_batch_from_netcdf_file(input_netcdf_path, "input_netcdf_path")
        batch = batch.to(self._device)

        # Crop as rollout() does internally, so the initial-condition frame sits
        # on the same grid as every predicted frame. rollout() crops its own
        # copy, so this is not a double crop.
        cropped_ic = batch.crop(self._patch_size)
        lat_out = cropped_ic.metadata.lat.detach().cpu().numpy()
        lon_out = cropped_ic.metadata.lon.detach().cpu().numpy()
        # One response time per batch member (initial condition).
        valid_times = [_as_naive_utc(t) for t in cropped_ic.metadata.time]

        # Lead 0 h is the initial condition: the last history step. Every step
        # is kept.
        frames_surf = {k: [cropped_ic.surf_vars[k][:, -1].detach().float().cpu().numpy()] for k in SURFACE_VARS}
        frames_atmos = {k: [cropped_ic.atmos_vars[k][:, -1].detach().float().cpu().numpy()] for k in ATMOS_VARS}
        lead_hours = [0.0]

        # bf16 rather than fp16: the roll-out feeds its own output back in, so
        # exponent range matters more than mantissa bits.
        autocast_ctx = (
            torch.autocast(device_type="cuda", dtype=torch.bfloat16) if config.AUTOCAST else contextlib.nullcontext()
        )
        with torch.no_grad(), autocast_ctx:
            for step_i, pred in enumerate(rollout(model, batch, steps=max(1, int(forecast_steps))), start=1):
                lead_hours.append(step_i * self._timestep_hours)
                # .float() before .numpy(): under autocast the predictions come
                # back bf16, which numpy cannot represent.
                for k in SURFACE_VARS:
                    frames_surf[k].append(pred.surf_vars[k][:, -1].detach().float().cpu().numpy())
                for k in ATMOS_VARS:
                    frames_atmos[k].append(pred.atmos_vars[k][:, -1].detach().float().cpu().numpy())

        data_vars = {}
        for k in SURFACE_VARS:
            arr = np.stack(frames_surf[k], axis=1)  # (time=batch, lead_time, lat, lon)
            data_vars[k] = (("time", "lead_time", "lat", "lon"), arr)
        for k in ATMOS_VARS:
            arr = np.stack(frames_atmos[k], axis=1)  # (time=batch, lead_time, level, lat, lon)
            data_vars[k] = (("time", "lead_time", "level", "lat", "lon"), arr)

        out_coords = {
            # datetime64 has no timezone, and warns on a tz-aware input.
            "time": np.array(valid_times, dtype="datetime64[s]"),
            "lead_time": np.asarray(lead_hours, dtype="float32"),
            "lat": lat_out,
            "lon": lon_out,
            "level": np.asarray(PRESSURE_LEVELS),
        }
        return data_utils.save_netcdf(out_path, data_vars, out_coords)
