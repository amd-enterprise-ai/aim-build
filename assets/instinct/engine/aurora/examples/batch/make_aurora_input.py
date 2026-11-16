#!/usr/bin/env python3

# Copyright © Advanced Micro Devices, Inc., or its affiliates.
#
# SPDX-License-Identifier: MIT

"""Convert the ERA5 snapshot into input files for the Aurora AIM, one per day.

The Aurora service reads its input with `aurora.Batch.from_netcdf`, which
expects Aurora's own schema rather than plain ERA5: variables prefixed `surf_`,
`static_` and `atmos_`, over dims `(batch, history, ...)`. A plain ERA5 file
decodes to empty variable dicts and the service then reports a shape error that
says nothing about the real cause, so this conversion is required.

The schema is a plain xarray Dataset (float32 arrays, five coords, no
attributes), so this writes it directly and needs only numpy, xarray and
netCDF4. Going through `aurora.Batch.to_netcdf` would pull in torch and the
model package.

Verified against microsoft-aurora 2.0.1: the output matches the library's own
serialisation and produces an identical forecast. `--verify-with-aurora` checks
that round trip when the package happens to be installed.

Each file holds the two history steps Aurora conditions on, about 220 MB. A day
has four analysis times, so --step0/--step1 choose which pair to use.

Usage:
    python3 make_aurora_input.py --source-dir era5 --outdir out
    python3 make_aurora_input.py --source-dir era5/2024-06-01 --out out/aurora_input.nc
"""

import argparse
import pathlib
import sys

import numpy as np
import xarray as xr

LEVELS = (50, 100, 150, 200, 250, 300, 400, 500, 600, 700, 850, 925, 1000)

# Aurora's name -> the short name CDS uses for the same field.
SURF_MAP = {"2t": "t2m", "10u": "u10", "10v": "v10", "msl": "msl"}
STATIC_VARS = ("lsm", "z", "slt")
ATMOS_VARS = ("t", "u", "v", "q", "z")

SURF_DIMS = ("batch", "history", "latitude", "longitude")
STATIC_DIMS = ("latitude", "longitude")
ATMOS_DIMS = ("batch", "history", "level", "latitude", "longitude")


def build(era5_dir, step0, step1):
    era5 = pathlib.Path(era5_dir)
    surf = xr.open_dataset(era5 / "era5_surface_instant.nc")
    static = xr.open_dataset(era5 / "era5_static.nc")
    atmos = xr.open_dataset(era5 / "era5_atmospheric.nc")
    steps = [step0, step1]

    def f32(arr):
        return np.nan_to_num(arr, nan=0.0).astype(np.float32)

    data_vars = {}
    for name, cds_name in SURF_MAP.items():
        # (2, H, W) -> (1, 2, H, W): a leading batch dim of one.
        data_vars[f"surf_{name}"] = (SURF_DIMS, f32(surf[cds_name].values[steps][None]))
    for name in STATIC_VARS:
        data_vars[f"static_{name}"] = (STATIC_DIMS, f32(static[name].values[0]))
    for name in ATMOS_VARS:
        # CDS returns levels 1000..50; Aurora wants them in its own order.
        arr = atmos[name].sel(pressure_level=list(LEVELS)).values[steps][None]
        data_vars[f"atmos_{name}"] = (ATMOS_DIMS, f32(arr))

    valid_time = surf.valid_time.values.astype("datetime64[ns]")
    ds = xr.Dataset(
        data_vars=data_vars,
        coords={
            "latitude": ("latitude", surf.latitude.values.astype(np.float32)),
            "longitude": ("longitude", surf.longitude.values.astype(np.float32)),
            # Aurora anchors the batch at the latest history step.
            "time": ("time", np.array([valid_time[steps[1]]], dtype="datetime64[ns]")),
            "level": ("level", np.array(LEVELS, dtype=np.int64)),
            "rollout_step": np.int64(0),
        },
    )
    return ds, valid_time[steps[0]], valid_time[steps[1]]


def self_check(path):
    """Check the names and dims the reader keys off, so a bad file fails here."""
    ds = xr.open_dataset(path)
    problems = []
    for name in SURF_MAP:
        if f"surf_{name}" not in ds.data_vars:
            problems.append(f"missing surf_{name}")
        elif ds[f"surf_{name}"].dims != SURF_DIMS:
            problems.append(f'surf_{name} dims {ds[f"surf_{name}"].dims}')
    for name in STATIC_VARS:
        if f"static_{name}" not in ds.data_vars:
            problems.append(f"missing static_{name}")
    for name in ATMOS_VARS:
        if f"atmos_{name}" not in ds.data_vars:
            problems.append(f"missing atmos_{name}")
        elif ds[f"atmos_{name}"].dims != ATMOS_DIMS:
            problems.append(f'atmos_{name} dims {ds[f"atmos_{name}"].dims}')
    for coord in ("latitude", "longitude", "time", "level", "rollout_step"):
        if coord not in ds.coords:
            problems.append(f"missing coord {coord}")
    if ds.sizes.get("history") != 2:
        problems.append(f'history is {ds.sizes.get("history")}, Aurora needs 2')
    ds.close()
    return problems


def verify_with_aurora(path):
    """Round-trip through Aurora's own reader, when it is importable."""
    try:
        from aurora import Batch
    except ImportError:
        print("SKIP  --verify-with-aurora: the aurora package is not installed here")
        return True
    batch = Batch.from_netcdf(str(path))
    ok = (
        set(batch.surf_vars) == set(SURF_MAP)
        and set(batch.static_vars) == set(STATIC_VARS)
        and set(batch.atmos_vars) == set(ATMOS_VARS)
    )
    print("round-trip Batch.from_netcdf " + ("OK" if ok else "MISMATCH"))
    return ok


def day_dirs(source_dir):
    """The day directories under source_dir, or source_dir itself if it is one."""
    root = pathlib.Path(source_dir)
    if (root / "era5_surface_instant.nc").is_file():
        return [root]
    return sorted(d for d in root.iterdir() if (d / "era5_surface_instant.nc").is_file())


def convert(source, out, step0, step1, check_with_aurora):
    ds, t0, t1 = build(source, step0, step1)

    out.parent.mkdir(parents=True, exist_ok=True)
    # Compressed, matching what Batch.to_netcdf writes; uncompressed files are much larger.
    encoding = {v: {"zlib": True, "shuffle": True, "complevel": 4} for v in ds.data_vars}
    ds.to_netcdf(out, encoding=encoding)

    print(f"wrote {out} ({out.stat().st_size / 1e6:.1f} MB)")
    print(f"  init time  : {t1} (history from {t0})")

    problems = self_check(out)
    if problems:
        sys.exit("FAIL  schema self-check: " + "; ".join(problems))
    if check_with_aurora and not verify_with_aurora(out):
        sys.exit("FAIL  aurora round-trip disagreed with the written file")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument(
        "--source-dir",
        "--era5-dir",
        dest="source_dir",
        default="era5",
        help="one day of ERA5 files, or a directory of such directories",
    )
    ap.add_argument("--outdir", default="out", help="written as <outdir>/<day>/<name>")
    ap.add_argument("--name", default="aurora_input.nc", help="filename the AIM profile expects")
    ap.add_argument("--out", help="explicit output path; single day only")
    ap.add_argument("--step0", type=int, default=0, help="index of the earlier history step")
    ap.add_argument("--step1", type=int, default=1, help="index of the initial condition")
    ap.add_argument("--overwrite", action="store_true", help="rewrite files that already exist")
    ap.add_argument(
        "--verify-with-aurora", action="store_true", help="additionally round-trip through aurora.Batch, if installed"
    )
    args = ap.parse_args()

    days = day_dirs(args.source_dir)
    if not days:
        sys.exit(f"no ERA5 files under {args.source_dir}")
    if args.out and len(days) > 1:
        sys.exit(f"--out takes a single day, but {args.source_dir} holds {len(days)}; use --outdir")

    print(f"{len(days)} day(s) from {args.source_dir}")
    print(f"  surf_vars  : {sorted(SURF_MAP)}")
    print(f"  static_vars: {sorted(STATIC_VARS)}")
    print(f"  atmos_vars : {sorted(ATMOS_VARS)} on {len(LEVELS)} levels")

    written = 0
    for day in days:
        if args.out:
            out = pathlib.Path(args.out)
        elif len(days) == 1:
            out = pathlib.Path(args.outdir) / args.name
        else:
            out = pathlib.Path(args.outdir) / day.name / args.name
        if out.exists() and not args.overwrite:
            print(f"skip {out} (already present, {out.stat().st_size / 1e6:.1f} MB)")
            continue
        convert(day, out, args.step0, args.step1, args.verify_with_aurora)
        written += 1

    print(f"\n{written} file(s) written, {len(days) - written} skipped")


if __name__ == "__main__":
    main()
