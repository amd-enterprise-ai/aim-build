# Copyright © Advanced Micro Devices, Inc., or its affiliates.
#
# SPDX-License-Identifier: MIT

"""Download the AIFS input fields from ECMWF Open Data on AWS.

For every 6-hourly time in the range, writes ``<outdir>/<YYYYMMDDHH>.grib2``
holding the 112 step-0 GRIB2 messages AIFS Single 2.0 and AIFS ENS 2.0 read
(surface, soil, pressure levels and wave), fetched by byte range from the ECMWF
Open Data buckets. With ``--members``, writes the perturbed ensemble members'
fields instead, to ``<outdir>/pfNN/<YYYYMMDDHH>.grib2``. Each output directory
gets a ``manifest.json`` with the sha256 of every file. Standard library only.
"""

import argparse
import concurrent.futures
import datetime
import hashlib
import json
import os
import random
import sys
import time
import urllib.error
import urllib.request

# Byte-identical ECMWF mirrors of s3://ecmwf-forecasts, tried in order.
MIRRORS = (
    "https://storage.googleapis.com/ecmwf-open-data",
    "https://ecmwf-forecasts.s3.eu-central-1.amazonaws.com",
)

PARAM_SFC = ["10u", "10v", "2d", "2t", "msl", "skt", "sp", "tcw", "lsm", "z", "slor", "sdor", "sd"]
# Constant fields; ensemble members take them from the oper run, as the ENS notebook does.
PARAM_CONST = ["lsm", "z", "slor", "sdor"]
PARAM_SOIL = ["vsw", "sot"]
SOIL_LEVELS = ["1", "2"]
PARAM_WAVE = ["wmb", "h1012", "h1214", "h1417", "h1721", "h2125", "h2530", "mwd", "cdww", "mwp", "swh"]
PARAM_PL = ["gh", "t", "u", "v", "w", "q"]
LEVELS = ["1000", "925", "850", "700", "600", "500", "400", "300", "250", "200", "150", "100", "50", "10"]

FIELDS_PER_TIME = len(PARAM_SFC) + len(PARAM_SOIL) * len(SOIL_LEVELS) + len(PARAM_WAVE) + len(PARAM_PL) * len(LEVELS)

RETRIES = 8


def _get(path, byte_range=None):
    """GET ``path`` from the first mirror that answers, with backoff; S3 answers 503 SlowDown under load."""
    headers = {"Range": "bytes=%d-%d" % byte_range} if byte_range else {}
    errors = []
    for attempt in range(RETRIES):
        for mirror in MIRRORS:
            url = "%s/%s" % (mirror, path)
            try:
                with urllib.request.urlopen(urllib.request.Request(url, headers=headers), timeout=120) as resp:
                    data = resp.read()
                if byte_range and len(data) != byte_range[1] - byte_range[0] + 1:
                    raise OSError("short read %d bytes" % len(data))
                return data
            except (urllib.error.URLError, OSError) as exc:
                errors.append("%s: %s" % (url, exc))
        time.sleep(min(60, 2**attempt) + random.random())
    raise RuntimeError("GET %s failed after %d rounds: %s" % (path, RETRIES, errors[-len(MIRRORS) :]))


def _file_path(t, stream):
    kind = "ef" if stream in ("enfo", "waef") else "fc"
    return "%s/%02dz/ifs/0p25/%s/%s%02d0000-0h-%s-%s.grib2" % (
        t.strftime("%Y%m%d"),
        t.hour,
        stream,
        t.strftime("%Y%m%d"),
        t.hour,
        stream,
        kind,
    )


def _wanted(rec, stream, member=None):
    param, levtype, level = rec.get("param"), rec.get("levtype"), str(rec.get("levelist", ""))
    if member is not None:
        if stream == "oper":
            return levtype == "sfc" and param in PARAM_CONST
        if str(rec.get("number")) != str(member):
            return False
    if stream in ("wave", "waef"):
        return param in PARAM_WAVE
    if levtype == "sfc":
        if member is None:
            return param in PARAM_SFC
        return param in PARAM_SFC and param not in PARAM_CONST
    if levtype == "sol":
        return param in PARAM_SOIL and level in SOIL_LEVELS
    if levtype == "pl":
        return param in PARAM_PL and level in LEVELS
    return False


def _key(rec):
    return "%s_%s" % (rec["param"], rec["levelist"]) if rec.get("levelist") else rec["param"]


def _streams(member):
    if member is None:
        return ("oper", "wave")
    return ("enfo", "waef", "oper")


def fetch_time(t, outdir, member=None):
    """Write the 112 fields for time ``t`` (of ensemble ``member`` if set); return the manifest entry."""
    chunks = []
    keys = []
    sources = []
    for stream in _streams(member):
        path = _file_path(t, stream)
        index = path[: -len(".grib2")] + ".index"
        records = [json.loads(line) for line in _get(index).decode().splitlines() if line.strip()]
        selected = {}
        for rec in records:
            if _wanted(rec, stream, member) and _key(rec) not in selected:
                selected[_key(rec)] = rec
        for key, rec in sorted(selected.items(), key=lambda kv: kv[1]["_offset"]):
            start = int(rec["_offset"])
            chunks.append(_get(path, (start, start + int(rec["_length"]) - 1)))
            keys.append(key)
        sources.append(path)
    if len(keys) != FIELDS_PER_TIME or len(set(keys)) != len(keys):
        raise RuntimeError("%s: expected %d distinct fields, got %d" % (t.isoformat(), FIELDS_PER_TIME, len(keys)))
    data = b"".join(chunks)
    name = t.strftime("%Y%m%d%H") + ".grib2"
    tmp = os.path.join(outdir, name + ".partial")
    with open(tmp, "wb") as fh:
        fh.write(data)
    os.replace(tmp, os.path.join(outdir, name))
    entry = {
        "file": name,
        "valid_time": t.strftime("%Y-%m-%dT%H:%M:%SZ"),
        "fields": len(keys),
        "bytes": len(data),
        "sha256": hashlib.sha256(data).hexdigest(),
        "sources": sources,
    }
    if member is not None:
        entry["number"] = member
    return entry


def _write_manifest(outdir, entries, member=None):
    source = "ECMWF Open Data (IFS 0.25 deg, step 0), s3://ecmwf-forecasts mirrors %s" % ", ".join(MIRRORS)
    if member is not None:
        source += "; enfo/waef perturbed member %d, constants %s from oper" % (member, ",".join(PARAM_CONST))
    year = min(e["valid_time"][:4] for e in entries)
    manifest = {
        "source": source,
        "license": "CC-BY-4.0",
        "attribution": (
            "ECMWF IFS forecast data (Open Data, 0.25 degree), (c) European Centre for Medium-Range "
            "Weather Forecasts (ECMWF), %s, accessed via Google Cloud and Amazon Web Services. "
            "Licensed under CC-BY-4.0. Modified: selected GRIB messages joined into one file per time." % year
        ),
        "fields_per_time": FIELDS_PER_TIME,
        "times": entries,
        "total_bytes": sum(e["bytes"] for e in entries),
    }
    with open(os.path.join(outdir, "manifest.json"), "w") as fh:
        json.dump(manifest, fh, indent=1)
    return manifest


def _parse(text):
    t = datetime.datetime.strptime(text, "%Y-%m-%dT%H")
    if t.hour % 6:
        raise ValueError("%s is not 00, 06, 12 or 18 UTC" % text)
    return t


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--start", help="first time, YYYY-MM-DDTHH")
    p.add_argument("--end", help="last time, YYYY-MM-DDTHH (inclusive)")
    p.add_argument("--times", default="", help="comma-separated times instead of --start/--end")
    p.add_argument("--members", default="", help="comma-separated perturbed member numbers, e.g. 1,2")
    p.add_argument("--outdir", required=True)
    p.add_argument("--workers", type=int, default=4)
    a = p.parse_args(argv)

    if a.times:
        times = [_parse(x) for x in a.times.split(",") if x]
    else:
        if not (a.start and a.end):
            p.error("give --start and --end, or --times")
        start, end = _parse(a.start), _parse(a.end)
        if end < start:
            p.error("--end must be >= --start")
        times = []
        t = start
        while t <= end:
            times.append(t)
            t += datetime.timedelta(hours=6)
    members = [int(m) for m in a.members.split(",") if m] or [None]

    for member in members:
        outdir = a.outdir if member is None else os.path.join(a.outdir, "pf%02d" % member)
        os.makedirs(outdir, exist_ok=True)
        with concurrent.futures.ThreadPoolExecutor(a.workers) as pool:
            entries = list(pool.map(lambda t: fetch_time(t, outdir, member), times))
        for e in entries:
            print("%s %s %3d fields %6.1f MB" % (outdir, e["valid_time"], e["fields"], e["bytes"] / 1e6), flush=True)
        manifest = _write_manifest(outdir, entries, member)
        print("%s: total %.1f MB in %d files" % (outdir, manifest["total_bytes"] / 1e6, len(entries)))
    return 0


if __name__ == "__main__":
    sys.exit(main())
