# Copyright © Advanced Micro Devices, Inc., or its affiliates.
#
# SPDX-License-Identifier: MIT

"""Input preparation, forecast files and scores for AIFS Single 2.0 and AIFS ENS 2.0.

Inputs are step-0 0.25-degree GRIB2 fields at two times 6 h apart: one file sent
with the request, or a directory of per-time files such as the ones
``inputs/fetch_open_data.py`` writes for the image. They are turned into the
model's N320 input state the same way as upstream's ``run_AIFS_v2.0.ipynb`` and
``run_AIFS_ENS_v2.0.ipynb``; each checkpoint then takes the fields it needs. The
same files, prepared the same way, are the analyses forecasts are scored against.
"""

import datetime
import json
import os

import numpy as np

PARAM_SFC = ["10u", "10v", "2d", "2t", "msl", "skt", "sp", "tcw", "lsm", "z", "slor", "sdor", "sd"]
PARAM_SOIL = ["vsw", "sot"]
SOIL_LEVELS = [1, 2]
PARAM_WAVE = ["wmb", "h1012", "h1214", "h1417", "h1721", "h2125", "h2530", "mwd", "cdww", "mwp", "swh"]
PARAM_PL = ["gh", "t", "u", "v", "w", "q"]
LEVELS = [1000, 925, 850, 700, 600, 500, 400, 300, 250, 200, 150, 100, 50, 10]

SOIL_RENAME = {"sot_1": "stl1", "sot_2": "stl2", "vsw_1": "swvl1", "vsw_2": "swvl2"}
LAND_ONLY = ("sd", "swvl1", "swvl2")
# Input-file fields ``transform`` always reads.
TRANSFORM_INPUTS = {"mwd", "sd"} | set(SOIL_RENAME) | {"gh_%d" % lev for lev in LEVELS}
# Prepared fields each model does not read; the checkpoint's own list wins when known.
NOT_INPUT = {
    "single": {"q_10", "q_50"} | {"w_%d" % lev for lev in LEVELS},
    "ens": {"q_10"},
}

G = 9.80665  # m s-2, gh to z
N320_POINTS = 542080
TIMESTEP_HOURS = 6

# Default fields scored against the analyses: name -> unit.
SCORE_VARIABLES = {"2t": "K", "msl": "Pa", "z_500": "m2 s-2", "t_850": "K", "10u": "m s-1", "10v": "m s-1"}


def input_file(input_dir, when):
    """Path of the GRIB2 file for ``when`` under ``input_dir``."""
    return os.path.join(input_dir, when.strftime("%Y%m%d%H") + ".grib2")


def available_times(input_dir):
    """Sorted valid times found in ``input_dir``'s manifest (or its file names)."""
    manifest = os.path.join(input_dir, "manifest.json")
    if os.path.isfile(manifest):
        with open(manifest) as fh:
            names = [e["file"] for e in json.load(fh)["times"]]
    else:
        names = [n for n in os.listdir(input_dir) if n.endswith(".grib2")]
    out = []
    for n in names:
        try:
            out.append(datetime.datetime.strptime(n[:10], "%Y%m%d%H"))
        except ValueError:
            continue
    return sorted(out)


def parse_time(value):
    """Parse ``YYYY-MM-DDTHH[:MM[:SS]][Z]`` (or a datetime) to a naive UTC datetime."""
    if isinstance(value, datetime.datetime):
        return value.replace(tzinfo=None)
    text = str(value).strip().rstrip("Z")
    for fmt in ("%Y-%m-%dT%H:%M:%S", "%Y-%m-%dT%H:%M", "%Y-%m-%dT%H", "%Y%m%d%H"):
        try:
            return datetime.datetime.strptime(text, fmt)
        except ValueError:
            pass
    raise ValueError("cannot parse time %r; use YYYY-MM-DDTHH:MM" % (value,))


def _regrid(values, matrix_source):
    """0.25 deg (-180..180) to N320, as in the upstream notebook."""
    import earthkit.regrid as ekr  # noqa: PLC0415

    assert values.shape == (721, 1440), values.shape
    values = np.roll(values, -values.shape[1] // 2, axis=1)
    kwargs = {"matrix_source": matrix_source} if matrix_source else {}
    return ekr.interpolate(values, {"grid": (0.25, 0.25)}, {"grid": "N320"}, **kwargs)


def _valid_time(field):
    """Valid time of a GRIB message: its date and time plus its step in hours."""
    date = str(field.metadata("dataDate"))
    hhmm = int(field.metadata("dataTime"))
    step = int(str(field.metadata("step", default=0)).split("-")[-1] or 0)
    return datetime.datetime.strptime(date, "%Y%m%d") + datetime.timedelta(hours=hhmm // 100 + step, minutes=hhmm % 100)


def file_times(path):
    """Sorted valid times of the messages in one GRIB file."""
    import earthkit.data as ekd  # noqa: PLC0415

    return sorted({_valid_time(f) for f in ekd.from_source("file", path)})


def read_time(path, matrix_source=None, only=None, when=None):
    """Fields in one input file on N320, keyed like the notebook (``gh_850``, ``2t``, ...).

    ``only`` limits the keys; ``when`` keeps only the messages valid at that time.
    """
    import earthkit.data as ekd  # noqa: PLC0415

    fields = {}
    for f in ekd.from_source("file", path):
        if when is not None and _valid_time(f) != when:
            continue
        param = f.metadata("param")
        levtype = f.metadata("levtype", default="")
        if levtype in ("pl", "isobaricInhPa", "sol", "soilLayer") or param in PARAM_SOIL:
            name = "%s_%s" % (param, f.metadata("levelist"))
        else:
            name = param
        if only is not None and name not in only:
            continue
        fields[name] = _regrid(f.to_numpy(), matrix_source)
    return fields


def _raw_name(name):
    """The input-file key a model field name is derived from."""
    if name.startswith("z_"):
        return "gh_" + name[2:]
    if name in ("cos_mwd", "sin_mwd"):
        return "mwd"
    for raw, model in SOIL_RENAME.items():
        if model == name:
            return raw
    return name


def land_sea_mask(lsm_path):
    """True where ``lsm.grib`` (already on N320) is sea."""
    import earthkit.data as ekd  # noqa: PLC0415

    return np.equal(ekd.from_source("file", lsm_path)[0].to_numpy(flatten=True), 0)


def grid_points(lsm_path):
    """(latitudes, longitudes) of the N320 points, in model order, from ``lsm.grib``."""
    import earthkit.data as ekd  # noqa: PLC0415

    return ekd.from_source("file", lsm_path)[0].grid_points()


def transform(fields, sea):
    """Apply the notebook's transforms to a dict of (n_times, N320) arrays, in place."""
    mwd = fields.pop("mwd")
    rad = np.deg2rad(mwd)
    fields["cos_mwd"] = np.cos(rad)
    fields["sin_mwd"] = np.sin(rad)
    for src, dst in SOIL_RENAME.items():
        fields[dst] = fields.pop(src)
    for name in LAND_ONLY:
        fields[name][:, sea] = np.nan
    for level in LEVELS:
        fields["z_%d" % level] = fields.pop("gh_%d" % level) * G
    return fields


def prepared_names():
    """Every field name ``transform`` produces from a full input file."""
    names = list(PARAM_SFC) + [p for p in PARAM_WAVE if p != "mwd"] + ["cos_mwd", "sin_mwd"]
    names += list(SOIL_RENAME.values())
    names += ["%s_%d" % (p if p != "gh" else "z", lev) for p in PARAM_PL for lev in LEVELS]
    return sorted(names)


def expected_names(variant="single"):
    """Field names the variant's notebook hands to the runner."""
    return sorted(n for n in prepared_names() if n not in NOT_INPUT[variant])


def build_input_state(source, init_time, lsm_path, matrix_source=None, needed=None):
    """The notebook's ``input_state``: fields at ``init_time - 6 h`` and ``init_time``.

    ``source`` is one GRIB file holding both times, or a directory of per-time
    files. ``needed`` is the checkpoint's input list; default is the Single 2.0 list.
    """
    init_time = parse_time(init_time)
    needed = set(needed) if needed is not None else set(expected_names())
    times = (init_time - datetime.timedelta(hours=TIMESTEP_HOURS), init_time)
    stacked = {}
    for when in times:
        if os.path.isdir(source):
            path, pick = input_file(source, when), None
            if not os.path.isfile(path):
                raise FileNotFoundError("no input for %s (%s)" % (when.isoformat(), path))
        else:
            path, pick = source, when
        fields_at = read_time(path, matrix_source, when=pick)
        if not fields_at:
            raise FileNotFoundError("no input for %s in %s" % (when.isoformat(), path))
        for name, values in fields_at.items():
            stacked.setdefault(name, []).append(values)
    raw = {_raw_name(n) for n in needed} | TRANSFORM_INPUTS
    incomplete = sorted(n for n in raw if len(stacked.get(n, ())) != len(times))
    if incomplete:
        raise ValueError(
            "input files lack fields at %s or %s: %s" % (times[0].isoformat(), times[1].isoformat(), incomplete)
        )
    fields = {k: np.stack(v) for k, v in stacked.items()}
    transform(fields, land_sea_mask(lsm_path))
    missing = sorted(needed - set(fields))
    if missing:
        raise ValueError("input files are missing fields: %s" % missing)
    return {"date": init_time, "fields": {k: v for k, v in fields.items() if k in needed}}


def load_analysis(input_dir, valid_time, lsm_path, names, matrix_source=None):
    """Analysis fields at ``valid_time`` on N320, named like the forecast output."""
    path = input_file(input_dir, parse_time(valid_time))
    if not os.path.isfile(path):
        return None
    raw = read_time(path, matrix_source, only={_raw_name(n) for n in names})
    sea = land_sea_mask(lsm_path) if set(names) & set(LAND_ONLY) else None
    out = {}
    for n in names:
        src = raw.get(_raw_name(n))
        if src is None:
            continue
        if n.startswith("z_"):
            out[n] = src * G
        elif n == "cos_mwd":
            out[n] = np.cos(np.deg2rad(src))
        elif n == "sin_mwd":
            out[n] = np.sin(np.deg2rad(src))
        elif n in LAND_ONLY:
            out[n] = np.where(sea, np.nan, src)
        else:
            out[n] = src
    return out


def area_weights(latitudes):
    """Per-point area weights, mean 1: cos(latitude) divided by the number of points on that latitude row."""
    lat = np.asarray(latitudes, dtype=np.float64)
    rows, inverse, counts = np.unique(lat, return_inverse=True, return_counts=True)
    w = (np.cos(np.deg2rad(rows)) / counts)[inverse]
    return w / w.mean()


def weighted_rmse(forecast, truth, weights):
    """Area-weighted RMSE over the points where both values are finite."""
    f = np.asarray(forecast, dtype=np.float64)
    t = np.asarray(truth, dtype=np.float64)
    ok = np.isfinite(f) & np.isfinite(t)
    w = weights[ok]
    return float(np.sqrt(np.sum(w * (f[ok] - t[ok]) ** 2) / np.sum(w)))


def weighted_mean(values, weights):
    """Area-weighted mean over finite points."""
    v = np.asarray(values, dtype=np.float64)
    ok = np.isfinite(v)
    return float(np.sum(weights[ok] * v[ok]) / np.sum(weights[ok]))


def save_netcdf(path, init_time, lead_hours, fields, latitudes, longitudes, model="ecmwf/aifs-single-2.0"):
    """Write ``fields`` with dims (time, lead_time, values), or (time, member, lead_time, values).

    ``fields`` maps name -> (lead_time, values), or (member, lead_time, values) for an ensemble.
    """
    import xarray as xr  # noqa: PLC0415

    data_vars = {}
    n_members = None
    for name, arr in fields.items():
        arr = np.asarray(arr, dtype=np.float32)
        if arr.ndim == 3:
            n_members = arr.shape[0]
            data_vars[name] = (("time", "member", "lead_time", "values"), arr[None, ...])
        else:
            data_vars[name] = (("time", "lead_time", "values"), arr[None, ...])
    coords = {
        "time": ("time", [np.datetime64(parse_time(init_time), "ns")]),
        "lead_time": ("lead_time", np.asarray(lead_hours, dtype=np.int32), {"units": "hours"}),
        "latitude": ("values", np.asarray(latitudes, dtype=np.float32), {"units": "degrees_north"}),
        "longitude": ("values", np.asarray(longitudes, dtype=np.float32), {"units": "degrees_east"}),
    }
    if n_members is not None:
        coords["member"] = ("member", np.arange(n_members, dtype=np.int32))
    ds = xr.Dataset(data_vars, coords=coords, attrs={"grid": "N320", "model": model})
    tmp = path + ".partial"
    ds.to_netcdf(tmp, engine="netcdf4")
    os.replace(tmp, path)
    return {k: int(v) for k, v in ds.sizes.items()}


def ensemble_scores(members, truth, weights):
    """Area-weighted ensemble scores of ``members`` (M, points) against ``truth`` (points).

    Uses points where the truth and every member are finite. Returns
    ``ens_mean_rmse``; ``crps``, the fair CRPS (Ferro 2014, Zamo and Naveau 2018),
    per point (1/M) sum_i |x_i - y| - 1/(2M(M-1)) sum_i sum_j |x_i - x_j|, area-averaged;
    ``spread``, the root of the area-mean unbiased member variance; and
    ``spread_skill``, sqrt((M+1)/M) * spread / ens_mean_rmse (Fortin et al. 2014),
    which is 1 for a reliable ensemble. With M = 1, ``crps`` is the absolute error
    and the spread terms are None.
    """
    x = np.asarray(members, dtype=np.float64)
    y = np.asarray(truth, dtype=np.float64)
    m = x.shape[0]
    ok = np.isfinite(y) & np.isfinite(x).all(axis=0)
    x, y, w = x[:, ok], y[ok], np.asarray(weights, dtype=np.float64)[ok]
    wsum = np.sum(w)
    mean = x.mean(axis=0)
    out = {"members": m, "ens_mean_rmse": float(np.sqrt(np.sum(w * (mean - y) ** 2) / wsum))}
    mae = np.abs(x - y).mean(axis=0)
    if m == 1:
        out.update(crps=float(np.sum(w * mae) / wsum), spread=None, spread_skill=None)
        return out
    # sum_i sum_j |x_i - x_j| = 2 sum_k (2k - M + 1) x_(k) over the sorted members, k = 0..M-1.
    ranks = (2 * np.arange(m) - m + 1).astype(np.float64)[:, None]
    pair_sum = 2.0 * np.sum(ranks * np.sort(x, axis=0), axis=0)
    crps = mae - pair_sum / (2.0 * m * (m - 1))
    spread = float(np.sqrt(np.sum(w * x.var(axis=0, ddof=1)) / wsum))
    out.update(
        crps=float(np.sum(w * crps) / wsum),
        spread=spread,
        spread_skill=(
            float(np.sqrt((m + 1.0) / m) * spread / out["ens_mean_rmse"]) if out["ens_mean_rmse"] > 0 else None
        ),
    )
    return out
