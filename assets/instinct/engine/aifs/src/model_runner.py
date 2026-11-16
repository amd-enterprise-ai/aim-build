# Copyright © Advanced Micro Devices, Inc., or its affiliates.
#
# SPDX-License-Identifier: MIT

"""AifsRunner — loads the configured AIFS checkpoint once and runs forecasts.

Builds the input state from a GRIB2 file or a directory of per-time files with
``data_utils`` (the upstream notebooks' preparation), runs ``anemoi.inference.runners.simple.SimpleRunner``
once per ensemble member and writes every 6-hour step to NetCDF, with the initial
condition at lead 0 h.
"""

import datetime
import os
import time

import config
import data_utils
import numpy as np
import torch


def _resolve_file(name):
    """Local path of an upstream repo file: WEIGHTS_DIR, else the HF cache (download if missing)."""
    for path in config.weights_candidates(name):
        if os.path.isfile(path):
            return path
    from huggingface_hub import hf_hub_download  # noqa: PLC0415

    return hf_hub_download(repo_id=config.HF_REPO, filename=name, revision=config.HF_REVISION)


class AifsRunner:
    """Loads the model once; construction is cheap and the load is lazy."""

    def __init__(self):
        self._runner = None
        self._lsm_path = None
        self._needed = None
        if not torch.cuda.is_available():
            raise RuntimeError(
                "No GPU visible to torch. This AIM requires a ROCm GPU; check that /dev/kfd and "
                "/dev/dri are mapped into the container and that torch is the ROCm build."
            )

    def ensure_loaded(self):
        """Resolve the checkpoint and load the model. Idempotent."""
        if self._runner is not None:
            return self._runner
        from anemoi.inference.runners.simple import SimpleRunner  # noqa: PLC0415

        checkpoint = _resolve_file(config.CHECKPOINT)
        self._lsm_path = _resolve_file(config.LSM_FILE)
        runner = SimpleRunner(checkpoint, device="cuda", precision=config.PRECISION or None)
        # The model is loaded lazily by anemoi; touch it so the load happens here.
        model = runner.model
        found = type(getattr(model, "model", model)).__name__
        if found != config.MODEL_CLASS:
            raise RuntimeError(
                "%s holds %s, but AIFS_VARIANT=%s expects %s"
                % (config.CHECKPOINT, found, config.VARIANT, config.MODEL_CLASS)
            )
        self._needed, _ = runner.checkpoint.select_variables_and_masks(
            include=["prognostic", "forcing"], exclude=["computed"]
        )
        self._runner = runner
        return runner

    @property
    def output_variables(self):
        runner = self.ensure_loaded()
        return list(runner.checkpoint.output_tensor_index_to_variable.values())

    def input_state(self, input_dir, init_time):
        self.ensure_loaded()
        return data_utils.build_input_state(
            input_dir, init_time, self._lsm_path, config.REGRID_DB or None, needed=self._needed
        )

    def forecast(self, input_dir, init_time, forecast_steps, variables=None, seeds=None, member_dirs=None):
        """Run ``forecast_steps`` 6-hour steps; returns lead hours, fields, lat/lon and timings.

        ``seeds`` holds one torch seed per ensemble member (None: one member, unseeded);
        ``member_dirs`` holds each member's input directory (default ``input_dir``).
        Fields are (lead_time, values) for one member, (member, lead_time, values) for more.
        """
        runner = self.ensure_loaded()
        seeds = list(seeds) if seeds else [None]
        dirs = list(member_dirs) if member_dirs else [input_dir] * len(seeds)
        names = list(dict.fromkeys(variables)) if variables else self.output_variables
        unknown = sorted(set(names) - set(self.output_variables))
        if unknown:
            raise ValueError("unknown output variables: %s" % unknown)

        states = {}
        t0 = time.perf_counter()
        for d in dict.fromkeys(dirs):
            states[d] = self.input_state(d, init_time)
        t_prep = time.perf_counter() - t0

        npoints = data_utils.N320_POINTS
        lead = datetime.timedelta(hours=data_utils.TIMESTEP_HOURS * forecast_steps)
        members = []
        lead_hours = [0]
        lats = lons = None
        member_times = []
        torch.cuda.synchronize()
        t_run = time.perf_counter()
        for seed, d in zip(seeds, dirs):
            initial = states[d]["fields"]
            series = {
                n: [
                    (
                        np.asarray(initial[n][-1], dtype=np.float32)
                        if n in initial
                        else np.full(npoints, np.nan, dtype=np.float32)
                    )
                ]
                for n in names
            }
            if seed is not None:
                torch.manual_seed(int(seed))
            hours = [0]
            t_member = time.perf_counter()
            state = {"date": states[d]["date"], "fields": dict(initial)}
            for out in runner.run(input_state=state, lead_time=lead):
                fields = out["fields"]
                for n in names:
                    series[n].append(np.array(fields[n], dtype=np.float32, copy=True))
                hours.append(int(out["step"].total_seconds() // 3600))
                if lats is None:
                    lats = np.asarray(out["latitudes"])
                    lons = np.asarray(out["longitudes"])
            member_times.append(time.perf_counter() - t_member)
            lead_hours = hours
            members.append({n: np.stack(v) for n, v in series.items()})
        t_total = time.perf_counter() - t_run

        if len(members) == 1:
            fields = members[0]
        else:
            fields = {n: np.stack([m[n] for m in members]) for n in names}
        return {
            "init_time": data_utils.parse_time(init_time),
            "lead_hours": lead_hours,
            "fields": fields,
            "latitudes": lats,
            "longitudes": lons,
            "timings_s": {"prepare_input": t_prep, "forecast": t_total, "per_member": member_times},
        }

    def forecast_to_file(
        self, input_dir, init_time, forecast_steps, out_path, variables=None, seeds=None, member_dirs=None
    ):
        """Run a forecast, write it to ``out_path`` and return the envelope fields."""
        result = self.forecast(input_dir, init_time, forecast_steps, variables, seeds, member_dirs)
        dims = data_utils.save_netcdf(
            out_path,
            result["init_time"],
            result["lead_hours"],
            result["fields"],
            result["latitudes"],
            result["longitudes"],
            model=config.HF_REPO,
        )
        timings = {k: v for k, v in result["timings_s"].items() if k != "per_member"}
        if len(result["timings_s"]["per_member"]) > 1:
            timings["per_member"] = result["timings_s"]["per_member"]
        return {
            "dims": dims,
            "variables": sorted(result["fields"]),
            "all_finite": _all_finite(result["fields"]),
            "lead_time_hours": result["lead_hours"],
            "timings_s": timings,
        }


# Wave fields are undefined over land and sd/swvl/ro/snowc over sea in AIFS output, so
# "finite" means no NaN or inf outside those masked points.
_MASKED_OK = {
    "swh",
    "mwp",
    "cos_mwd",
    "sin_mwd",
    "cdww",
    "wmb",
    "h1012",
    "h1214",
    "h1417",
    "h1721",
    "h2125",
    "h2530",
    "mwd",
    "sd",
    "swvl1",
    "swvl2",
    "ro",
    "snowc",
}


def _all_finite(fields):
    """True when no forecast step holds inf, NaN outside the masked fields, or only NaN."""
    for name, values in fields.items():
        v = values[..., 1:, :]
        if np.isinf(v).any():
            return False
        if name not in _MASKED_OK and np.isnan(v).any():
            return False
        if np.isnan(v).all(axis=-1).any():
            return False
    return True
