#!/usr/bin/env python3

# Copyright © Advanced Micro Devices, Inc., or its affiliates.
#
# SPDX-License-Identifier: MIT

"""Download ERA5 days from the public WeatherBench 2 bucket. No API key needed.

Writes static.nc (fixed fields) and one <YYYY-MM-DD>.nc per day under --outdir.
The bucket holds 1959-01-01 to 2023-01-10.

Usage:
    python3 fetch_weatherbench2.py --start 2022-06-01 --days 10 --outdir era5
"""

import argparse
import datetime as dt
import pathlib
import sys
import time

import numpy as np
import xarray as xr

STORE = "gs://weatherbench2/datasets/era5/1959-2023_01_10-wb13-6h-1440x721_with_derived_variables.zarr"

SURFACE_VARS = (
    "2m_temperature",
    "10m_u_component_of_wind",
    "10m_v_component_of_wind",
    "mean_sea_level_pressure",
)
ATMOS_VARS = (
    "temperature",
    "u_component_of_wind",
    "v_component_of_wind",
    "specific_humidity",
    "geopotential",
)
STATIC_VARS = ("land_sea_mask", "geopotential_at_surface", "soil_type")

RETRIES = 4


def open_store(store):
    return xr.open_zarr(store, storage_options={"token": "anon"}, chunks={})


def write(ds, path):
    """Write under a temporary name, so an interrupted run leaves no half file."""
    partial = path.with_suffix(".partial")
    encoding = {v: {"zlib": True, "complevel": 1} for v in ds.data_vars}
    for attempt in range(1, RETRIES + 1):
        try:
            ds.load().to_netcdf(partial, encoding=encoding)
            break
        except OSError as exc:
            if attempt == RETRIES:
                raise
            print(f"    read failed ({exc}); retry {attempt}/{RETRIES - 1}")
            time.sleep(10 * attempt)
    partial.replace(path)
    print(f"    wrote {path.name} ({path.stat().st_size / 1e6:.1f} MB)")


def fetch_static(ds, outdir):
    path = outdir / "static.nc"
    if path.exists():
        print(f"skip {path.name} (already present)")
        return
    print(f"--> {path.name}")
    write(ds[list(STATIC_VARS)], path)


def fetch_day(ds, outdir, date, hours):
    path = outdir / f"{date.isoformat()}.nc"
    if path.exists():
        print(f"skip {path.name} (already present)")
        return
    stamps = [np.datetime64(f"{date.isoformat()}T{h:02d}:00") for h in hours]
    missing = [str(s) for s in stamps if s not in ds.indexes["time"]]
    if missing:
        sys.exit(f"{STORE} has no data at {', '.join(missing)}; it ends at {ds.time.values[-1]}")
    print(f"--> {path.name} at {', '.join(f'{h:02d}:00' for h in hours)}")
    write(ds[list(SURFACE_VARS + ATMOS_VARS)].sel(time=stamps), path)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--start", default="2022-06-01", help="first date, YYYY-MM-DD")
    ap.add_argument("--days", type=int, default=10, help="how many consecutive days")
    ap.add_argument("--times", default="0,6", help="hours of the day to keep, comma separated, multiples of 6")
    ap.add_argument("--outdir", default="era5")
    ap.add_argument("--store", default=STORE, help="WeatherBench 2 ERA5 zarr store")
    args = ap.parse_args()

    hours = sorted({int(h) for h in args.times.split(",")})
    if any(h % 6 or not 0 <= h < 24 for h in hours):
        sys.exit(f"--times must be hours from 0, 6, 12, 18; got {args.times}")

    outdir = pathlib.Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    ds = open_store(args.store)

    fetch_static(ds, outdir)
    first = dt.date.fromisoformat(args.start)
    for offset in range(args.days):
        fetch_day(ds, outdir, first + dt.timedelta(days=offset), hours)

    total = sum(f.stat().st_size for f in outdir.glob("*.nc"))
    print(f"\nERA5 snapshot ready in {outdir} ({total / 1e9:.2f} GB, {args.days} day(s))")


if __name__ == "__main__":
    main()
