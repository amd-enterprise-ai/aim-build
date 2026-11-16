#!/usr/bin/env python3

# Copyright © Advanced Micro Devices, Inc., or its affiliates.
#
# SPDX-License-Identifier: MIT

"""Download an ERA5 snapshot covering what the global weather models take as input.

One directory per day under --outdir, each holding:

  era5_static.nc           time-invariant fields
  era5_surface_instant.nc  instantaneous single-level fields, 4 analysis times
  era5_surface_accum.nc    accumulated single-level fields, 4 analysis times
  era5_atmospheric.nc      upper-air fields on the 13 standard pressure levels

Roughly 600 MB per day, of which the pressure-level file is about 550 MB.

Times are 00/06/12/18 UTC, which covers every model's history length: Aurora
needs two consecutive 6-hourly steps, GenCast two steps 12 h apart, Pangu one.

Field union, and which model needs what:

  Surface        Aurora  Pangu  GenCast
  2t / t2m         x       x       x
  10u / u10        x       x       x
  10v / v10        x       x       x
  msl              x       x       x
  sst              .       .       x
  sp               .       .       .    (commonly useful)
  tp               .       .       .    (GenCast target, not an input)
  tisr             .       .       .    (GraphCast forcing)

  Static
  lsm              x       x       x
  z (surface)      x       x       x
  slt              x       x       .

  Pressure levels (50 100 150 200 250 300 400 500 600 700 850 925 1000 hPa)
  t u v q z        x       x       x
  w                .       .       x

Credentials come from a .cdsapirc-style file holding `url:` and `key:` lines.
Register at https://cds.climate.copernicus.eu to get a key, and accept the
licence for both ERA5 datasets on that account before the first run.

Requests are queued server-side and can take minutes to hours under load. Files
already present are skipped, so re-running resumes an interrupted download.

Usage:
    python3 fetch_era5.py --start 2024-06-01 --days 10 --outdir era5
"""

import argparse
import datetime as dt
import pathlib
import shutil
import sys
import zipfile

import cdsapi

PRESSURE_LEVELS = [
    "50",
    "100",
    "150",
    "200",
    "250",
    "300",
    "400",
    "500",
    "600",
    "700",
    "850",
    "925",
    "1000",
]

TIMES = ["00:00", "06:00", "12:00", "18:00"]

SURFACE_VARS = [
    "2m_temperature",
    "10m_u_component_of_wind",
    "10m_v_component_of_wind",
    "mean_sea_level_pressure",
    "sea_surface_temperature",
    "surface_pressure",
    "total_precipitation",
    "toa_incident_solar_radiation",
]

STATIC_VARS = [
    "land_sea_mask",
    "geopotential",
    "soil_type",
]

ATMOS_VARS = [
    "temperature",
    "u_component_of_wind",
    "v_component_of_wind",
    "specific_humidity",
    "geopotential",
    "vertical_velocity",
]

# A request mixing instantaneous and accumulated variables comes back as a zip
# of one file per step type, whatever download_format asks for.
ZIP_MEMBERS = {
    "data_stream-oper_stepType-instant.nc": "era5_surface_instant.nc",
    "data_stream-oper_stepType-accum.nc": "era5_surface_accum.nc",
}


def read_credentials(env_path):
    creds = {}
    for line in pathlib.Path(env_path).read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or ":" not in line:
            continue
        name, _, value = line.partition(":")
        creds[name.strip()] = value.strip()
    if "url" not in creds or "key" not in creds:
        sys.exit(f'{env_path} must contain "url:" and "key:" lines')
    return creds["url"], creds["key"]


def retrieve(client, dataset, request, target):
    target = pathlib.Path(target)
    if target.exists() and target.stat().st_size > 0:
        print(f"skip {target.name} (already present, {target.stat().st_size / 1e6:.1f} MB)")
        return False
    print(f"--> {target.name}: {dataset}")
    client.retrieve(dataset, request, str(target))
    print(f"    done, {target.stat().st_size / 1e6:.1f} MB")
    return True


def split_surface_zip(path):
    """Unpack the instant/accum pair CDS returns for a mixed surface request."""
    path = pathlib.Path(path)
    if not zipfile.is_zipfile(path):
        return
    with zipfile.ZipFile(path) as archive:
        for member, out_name in ZIP_MEMBERS.items():
            if member not in archive.namelist():
                continue
            out = path.parent / out_name
            with archive.open(member) as src, open(out, "wb") as dst:
                shutil.copyfileobj(src, dst)
            print(f"    extracted {out.name} ({out.stat().st_size / 1e6:.1f} MB)")
    path.unlink()


def download_day(client, outdir, date):
    """Fetch the static, surface and pressure-level files for one date."""
    year, month, day = date.split("-")
    outdir = pathlib.Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    common = {
        "product_type": ["reanalysis"],
        "year": [year],
        "month": [month],
        "day": [day],
        "data_format": "netcdf",
        "download_format": "unarchived",
    }

    retrieve(
        client,
        "reanalysis-era5-single-levels",
        {**common, "variable": STATIC_VARS, "time": ["00:00"]},
        outdir / "era5_static.nc",
    )

    surface_raw = outdir / "era5_surface.nc"
    if not (outdir / "era5_surface_instant.nc").exists():
        retrieve(
            client,
            "reanalysis-era5-single-levels",
            {**common, "variable": SURFACE_VARS, "time": TIMES},
            surface_raw,
        )
        split_surface_zip(surface_raw)
    else:
        print("skip era5_surface_*.nc (already present)")

    retrieve(
        client,
        "reanalysis-era5-pressure-levels",
        {**common, "variable": ATMOS_VARS, "pressure_level": PRESSURE_LEVELS, "time": TIMES},
        outdir / "era5_atmospheric.nc",
    )


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--env", default=str(pathlib.Path.home() / ".cdsapirc"), help='file with "url:" and "key:" lines')
    ap.add_argument("--outdir", default="era5")
    ap.add_argument("--start", default="2024-06-01", help="first date, YYYY-MM-DD")
    ap.add_argument("--days", type=int, default=1, help="how many consecutive days to fetch")
    args = ap.parse_args()

    url, key = read_credentials(args.env)
    client = cdsapi.Client(url=url, key=key)

    first = dt.date.fromisoformat(args.start)
    root = pathlib.Path(args.outdir)
    for offset in range(args.days):
        date = (first + dt.timedelta(days=offset)).isoformat()
        print(f"\n=== {date} ===")
        download_day(client, root / date, date)

    total = sum(f.stat().st_size for f in root.rglob("*.nc"))
    print(f"\nERA5 snapshot ready in {root} ({total / 1e9:.2f} GB across {args.days} day(s))")


if __name__ == "__main__":
    main()
