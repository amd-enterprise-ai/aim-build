# Copyright © Advanced Micro Devices, Inc., or its affiliates.
#
# SPDX-License-Identifier: MIT

"""Score an AIFS forecast file against the analyses in the input directory.

Usage: python -m scoring <forecast.nc> [--input-dir DIR] [--lsm PATH] [--variables 2t,msl,...]

Prints JSON: area-weighted RMSE per variable and lead time for every lead
whose valid time has an analysis, plus mean / min / max and NaN counts. For a
file with a member dimension it adds per-member RMSE, ensemble-mean RMSE, fair
CRPS, spread and the spread/skill ratio (see ``data_utils.ensemble_scores``).
"""

import argparse
import datetime
import json
import sys

import config
import data_utils
import numpy as np


def summarise(values, weights):
    """Weighted mean, min, max and NaN fraction of one field."""
    v = np.asarray(values, dtype=np.float64)
    finite = np.isfinite(v)
    return {
        "mean": data_utils.weighted_mean(v, weights) if finite.any() else None,
        "min": float(np.nanmin(v)) if finite.any() else None,
        "max": float(np.nanmax(v)) if finite.any() else None,
        "nan_fraction": float(1.0 - finite.mean()),
        "inf_count": int(np.isinf(v).sum()),
    }


def score_file(path, input_dir, lsm_path, variables=None, matrix_source=None):
    """RMSE and summary stats of the forecast in ``path``."""
    import xarray as xr  # noqa: PLC0415

    ds = xr.open_dataset(path)
    if "member" in ds.dims:
        return _score_ensemble(ds, input_dir, lsm_path, variables, matrix_source)
    names = list(variables) if variables else [n for n in data_utils.SCORE_VARIABLES if n in ds]
    weights = data_utils.area_weights(ds["latitude"].values)
    init = data_utils.parse_time(str(ds["time"].values[0])[:19])
    leads = [int(h) for h in ds["lead_time"].values]

    out = {
        "init_time": init.strftime("%Y-%m-%dT%H:%M:%SZ"),
        "units": {},
        "rmse": {},
        "persistence_rmse": {},
        "stats": {},
        "skipped_leads": [],
    }
    for n in names:
        out["units"][n] = data_utils.SCORE_VARIABLES.get(n, "")
        out["rmse"][n] = {}
        out["persistence_rmse"][n] = {}
        out["stats"][n] = {}
    # Lead 0 holds the initial condition, which is the persistence forecast.
    initial = {n: ds[n].values[0, 0] for n in names}
    for i, h in enumerate(leads):
        valid = init + datetime.timedelta(hours=h)
        for n in names:
            out["stats"][n][str(h)] = summarise(ds[n].values[0, i], weights)
        truth = data_utils.load_analysis(input_dir, valid, lsm_path, names, matrix_source)
        if truth is None:
            out["skipped_leads"].append(h)
            continue
        for n in names:
            out["rmse"][n][str(h)] = data_utils.weighted_rmse(ds[n].values[0, i], truth[n], weights)
            out["persistence_rmse"][n][str(h)] = data_utils.weighted_rmse(initial[n], truth[n], weights)
    ds.close()
    return out


def _score_ensemble(ds, input_dir, lsm_path, variables, matrix_source):
    """Per-member and ensemble scores of a (time, member, lead_time, values) file.

    Persistence is the analysis at the initial time, since members may start from
    different perturbed states.
    """
    names = list(variables) if variables else [n for n in data_utils.SCORE_VARIABLES if n in ds]
    weights = data_utils.area_weights(ds["latitude"].values)
    init = data_utils.parse_time(str(ds["time"].values[0])[:19])
    leads = [int(h) for h in ds["lead_time"].values]
    n_members = int(ds.sizes["member"])
    out = {
        "init_time": init.strftime("%Y-%m-%dT%H:%M:%SZ"),
        "members": n_members,
        "units": {n: data_utils.SCORE_VARIABLES.get(n, "") for n in names},
        "skipped_leads": [],
    }
    keys = ("rmse", "rmse_members", "ens_mean_rmse", "crps", "spread", "spread_skill", "persistence_rmse", "stats")
    for k in keys:
        out[k] = {n: {} for n in names}
    values = {n: ds[n].values[0] for n in names}
    ds.close()
    start = data_utils.load_analysis(input_dir, init, lsm_path, names, matrix_source)
    for i, h in enumerate(leads):
        valid = init + datetime.timedelta(hours=h)
        for n in names:
            out["stats"][n][str(h)] = [summarise(values[n][m, i], weights) for m in range(n_members)]
        truth = data_utils.load_analysis(input_dir, valid, lsm_path, names, matrix_source)
        if truth is None:
            out["skipped_leads"].append(h)
            continue
        for n in names:
            per_member = [data_utils.weighted_rmse(values[n][m, i], truth[n], weights) for m in range(n_members)]
            ens = data_utils.ensemble_scores(values[n][:, i], truth[n], weights)
            out["rmse_members"][n][str(h)] = per_member
            out["rmse"][n][str(h)] = float(np.mean(per_member))
            for k in ("ens_mean_rmse", "crps", "spread", "spread_skill"):
                out[k][n][str(h)] = ens[k]
            if start is not None:
                out["persistence_rmse"][n][str(h)] = data_utils.weighted_rmse(start[n], truth[n], weights)
    return out


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("forecast")
    p.add_argument("--input-dir", default=config.INPUT_DIR)
    p.add_argument("--lsm", default="", help="lsm.grib; default resolves it like the service does")
    p.add_argument("--variables", default="")
    a = p.parse_args(argv)
    lsm = a.lsm
    if not lsm:
        from model_runner import _resolve_file  # noqa: PLC0415

        lsm = _resolve_file(config.LSM_FILE)
    variables = [v for v in a.variables.split(",") if v] or None
    json.dump(score_file(a.forecast, a.input_dir, lsm, variables, config.REGRID_DB or None), sys.stdout, indent=1)
    print()
    return 0


if __name__ == "__main__":
    sys.exit(main())
