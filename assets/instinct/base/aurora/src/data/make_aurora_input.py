#!/usr/bin/env python3

# Copyright © Advanced Micro Devices, Inc., or its affiliates.
#
# SPDX-License-Identifier: MIT

"""Turn the files fetch_weatherbench2.py writes into Aurora input files, one per day.

Writes <outdir>/<YYYY-MM-DD>_input.nc in the format `aurora.Batch.from_netcdf`
reads. Each file holds the day's first two times, about 345 MB.

Usage:
    python3 make_aurora_input.py --source-dir era5 --outdir inputs
"""

import argparse
import pathlib
import sys

import numpy as np
import xarray as xr

LEVELS = (50, 100, 150, 200, 250, 300, 400, 500, 600, 700, 850, 925, 1000)

# Aurora's name -> the WeatherBench 2 name for the same field.
SURF_MAP = {
    "2t": "2m_temperature",
    "10u": "10m_u_component_of_wind",
    "10v": "10m_v_component_of_wind",
    "msl": "mean_sea_level_pressure",
}
STATIC_MAP = {"lsm": "land_sea_mask", "z": "geopotential_at_surface", "slt": "soil_type"}
ATMOS_MAP = {
    "t": "temperature",
    "u": "u_component_of_wind",
    "v": "v_component_of_wind",
    "q": "specific_humidity",
    "z": "geopotential",
}

SURF_DIMS = ("batch", "history", "latitude", "longitude")
STATIC_DIMS = ("latitude", "longitude")
ATMOS_DIMS = ("batch", "history", "level", "latitude", "longitude")


def f32(arr):
    return np.nan_to_num(arr, nan=0.0).astype(np.float32)


def build(day_path, static):
    day = xr.open_dataset(day_path)
    try:
        if day.sizes["time"] < 2:
            sys.exit(f"{day_path} has {day.sizes['time']} time(s); Aurora needs two")
        # Aurora wants latitude from north to south.
        if day.latitude.values[0] < day.latitude.values[-1]:
            day = day.isel(latitude=slice(None, None, -1))
        steps = [0, 1]

        data_vars = {}
        for name, src in SURF_MAP.items():
            data_vars[f"surf_{name}"] = (SURF_DIMS, f32(day[src].values[steps][None]))
        for name, src in STATIC_MAP.items():
            arr = f32(static[src].sel(latitude=day.latitude).values)
            # Soil type is a class number, stored with small rounding noise.
            data_vars[f"static_{name}"] = (STATIC_DIMS, np.rint(arr) if name == "slt" else arr)
        for name, src in ATMOS_MAP.items():
            arr = day[src].sel(level=list(LEVELS)).values[steps][None]
            data_vars[f"atmos_{name}"] = (ATMOS_DIMS, f32(arr))

        times = day.time.values.astype("datetime64[ns]")
        ds = xr.Dataset(
            data_vars=data_vars,
            coords={
                "latitude": ("latitude", day.latitude.values.astype(np.float32)),
                "longitude": ("longitude", day.longitude.values.astype(np.float32)),
                # Aurora dates the batch at the latest history step.
                "time": ("time", np.array([times[steps[1]]], dtype="datetime64[ns]")),
                "level": ("level", np.array(LEVELS, dtype=np.int64)),
                "rollout_step": np.int64(0),
            },
        )
        return ds, times[steps[0]], times[steps[1]]
    finally:
        day.close()


def self_check(path):
    """Check the names and dims the reader keys off, so a bad file fails here."""
    ds = xr.open_dataset(path)
    problems = []
    for name in SURF_MAP:
        if f"surf_{name}" not in ds.data_vars:
            problems.append(f"missing surf_{name}")
        elif ds[f"surf_{name}"].dims != SURF_DIMS:
            problems.append(f"surf_{name} dims {ds[f'surf_{name}'].dims}")
    for name in STATIC_MAP:
        if f"static_{name}" not in ds.data_vars:
            problems.append(f"missing static_{name}")
    for name in ATMOS_MAP:
        if f"atmos_{name}" not in ds.data_vars:
            problems.append(f"missing atmos_{name}")
        elif ds[f"atmos_{name}"].dims != ATMOS_DIMS:
            problems.append(f"atmos_{name} dims {ds[f'atmos_{name}'].dims}")
    for coord in ("latitude", "longitude", "time", "level", "rollout_step"):
        if coord not in ds.coords:
            problems.append(f"missing coord {coord}")
    if ds.sizes.get("history") != 2:
        problems.append(f"history is {ds.sizes.get('history')}, Aurora needs 2")
    if ds.sizes.get("latitude") != 721 or ds.sizes.get("longitude") != 1440:
        problems.append(f"grid is {ds.sizes.get('latitude')}x{ds.sizes.get('longitude')}, Aurora needs 721x1440")
    ds.close()
    return problems


def verify_with_aurora(path):
    """Read the file back with Aurora's own reader."""
    try:
        from aurora import Batch
    except ImportError:
        sys.exit("FAIL  --verify-with-aurora: the aurora package is not installed here")
    batch = Batch.from_netcdf(str(path))
    ok = (
        set(batch.surf_vars) == set(SURF_MAP)
        and set(batch.static_vars) == set(STATIC_MAP)
        and set(batch.atmos_vars) == set(ATMOS_MAP)
        and tuple(int(lv) for lv in batch.metadata.atmos_levels) == LEVELS
    )
    print("  Batch.from_netcdf " + ("OK" if ok else "MISMATCH"))
    return ok


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--source-dir", default="era5", help="directory written by fetch_weatherbench2.py")
    ap.add_argument("--outdir", default="inputs")
    ap.add_argument("--overwrite", action="store_true", help="rewrite files that already exist")
    ap.add_argument("--verify-with-aurora", action="store_true", help="also read each file with aurora.Batch")
    args = ap.parse_args()

    source = pathlib.Path(args.source_dir)
    static_path = source / "static.nc"
    days = sorted(p for p in source.glob("????-??-??.nc"))
    if not static_path.is_file() or not days:
        sys.exit(f"{source} must hold static.nc and at least one <YYYY-MM-DD>.nc")

    outdir = pathlib.Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    static = xr.open_dataset(static_path)

    written = 0
    for day_path in days:
        out = outdir / f"{day_path.stem}_input.nc"
        if out.exists() and not args.overwrite:
            print(f"skip {out} (already present)")
            continue
        ds, t0, t1 = build(day_path, static)
        # Compressed, as Batch.to_netcdf writes it.
        encoding = {v: {"zlib": True, "shuffle": True, "complevel": 4} for v in ds.data_vars}
        ds.to_netcdf(out, encoding=encoding)
        print(f"wrote {out} ({out.stat().st_size / 1e6:.1f} MB), dated {t1}, history from {t0}")

        problems = self_check(out)
        if problems:
            sys.exit("FAIL  schema self-check: " + "; ".join(problems))
        if args.verify_with_aurora and not verify_with_aurora(out):
            sys.exit(f"FAIL  aurora.Batch read {out} differently")
        written += 1

    static.close()
    print(f"\n{written} file(s) written, {len(days) - written} skipped")


if __name__ == "__main__":
    main()
