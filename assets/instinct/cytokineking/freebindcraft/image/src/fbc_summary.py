# Copyright © Advanced Micro Devices, Inc., or its affiliates.
#
# SPDX-License-Identifier: MIT

"""Result publishing: copy finished results out of the work tree, write summary.json.

FreeBindCraft keeps everything in one design directory and has no notion of a
partial result. This module makes a capped or interrupted run usable: it
publishes the CSVs and the accepted/ranked PDBs while the run is in progress
(every file via a temporary name and rename, so a reader never sees half a
file) and synthesizes ``summary.json`` from ``trajectory_stats.csv``,
``mpnn_design_stats.csv`` and ``failure_csv.csv``.

Pure standard library.
"""

from __future__ import annotations

import csv
import json
import os
import re
import shutil
from pathlib import Path
from typing import Any, Iterable, Optional

SCHEMA_VERSION = 1

# Files copied to the top of the output directory when they exist.
RESULT_CSVS = (
    "trajectory_stats.csv",
    "mpnn_design_stats.csv",
    "final_design_stats.csv",
    "failure_csv.csv",
    "rejected_mpnn_full_stats.csv",
)

_TIME_RE = re.compile(r"(?:(\d+)\s*hours?)?[, ]*(?:(\d+)\s*minutes?)?[, ]*(?:(\d+)\s*seconds?)?")

# Prefix of the columns in failure_csv.csv that count trajectories rejected
# before any MPNN redesign (see upstream generate_filter_pass_csv).
_TRAJECTORY_FAILURE_PREFIX = "Trajectory_"


def parse_time_text(text: str) -> Optional[int]:
    """Seconds from upstream's ``'1 hours, 2 minutes, 3 seconds'``; None if unparsable."""
    match = _TIME_RE.fullmatch((text or "").strip())
    if not match or not any(match.groups()):
        return None
    hours, minutes, seconds = (int(group or 0) for group in match.groups())
    return hours * 3600 + minutes * 60 + seconds


def atomic_write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name("." + path.name + ".tmp")
    tmp.write_text(text, encoding="utf-8")
    os.replace(tmp, path)


def atomic_copy(src: Path, dst: Path) -> bool:
    """Copy ``src`` to ``dst`` through a temporary name; skip when unchanged.

    Returns True when ``dst`` was (re)written and False when nothing needed copying (the source is gone, or the
    destination is already current). A failed copy removes the temporary file and re-raises the ``OSError``, so a
    write error is never mistaken for an unchanged file.
    """
    try:
        stat = src.stat()
    except OSError:
        return False
    try:
        old = dst.stat()
        if old.st_size == stat.st_size and old.st_mtime >= stat.st_mtime:
            return False
    except OSError:
        pass  # no destination yet (or unreadable): copy it
    dst.parent.mkdir(parents=True, exist_ok=True)
    tmp = dst.with_name("." + dst.name + ".tmp")
    try:
        shutil.copyfile(src, tmp)
        shutil.copystat(src, tmp)
        os.replace(tmp, dst)
    except OSError:
        try:
            tmp.unlink()
        except OSError:
            pass  # best-effort cleanup of the temporary file; the copy error is re-raised below
        raise
    return True


def _pdbs(directory: Path) -> list[Path]:
    if not directory.is_dir():
        return []
    return sorted(p for p in directory.iterdir() if p.suffix == ".pdb" and not p.name.startswith("."))


def publish_results(design_dir: Path, out_dir: Path) -> None:
    """Copy CSVs, Accepted/*.pdb and Accepted/Ranked/*.pdb into ``out_dir`` (atomic per file)."""
    for name in RESULT_CSVS:
        atomic_copy(design_dir / name, out_dir / name)
    for sub in ("Accepted", "Accepted/Ranked"):
        wanted = {p.name for p in _pdbs(design_dir / sub)}
        for pdb in _pdbs(design_dir / sub):
            atomic_copy(pdb, out_dir / sub / pdb.name)
        # Mirror removals: upstream clears and rewrites Ranked when it re-ranks.
        for stale in _pdbs(out_dir / sub):
            if stale.name not in wanted:
                try:
                    stale.unlink()
                except OSError:
                    pass  # stale copy could not be removed; the next publish retries


def read_rows(path: Path) -> list[dict[str, str]]:
    """Rows of a CSV; tolerant of a missing, empty or mid-rewrite file."""
    try:
        with path.open("r", encoding="utf-8", errors="replace", newline="") as handle:
            return [row for row in csv.DictReader(handle) if row]
    except (OSError, csv.Error):
        return []


def trajectory_times(rows: Iterable[dict[str, str]]) -> list[int]:
    times = []
    for row in rows:
        seconds = parse_time_text(row.get("TrajectoryTime", ""))
        if seconds is not None:
            times.append(seconds)
    return times


def _failure_count(rows: list[dict[str, str]], prefix: str) -> int:
    """Sum the counters of failure_csv columns with ``prefix`` (single counter row)."""
    total = 0
    for row in rows[:1]:
        for column, value in row.items():
            if column and column.startswith(prefix):
                try:
                    total += int(float(value))
                except (TypeError, ValueError):
                    continue  # non-numeric cell in upstream's CSV: not a counter
    return total


def build_summary(
    *,
    design_dir: Path,
    settings: dict[str, Any],
    status: str,
    finished: bool,
    exit_code: Optional[int],
    started_at: str,
    finished_at: Optional[str],
    wall_time_s: float,
    first_trajectory_s: Optional[float],
    accelerators: Optional[int],
    resumed: bool,
    notes: Iterable[str] = (),
    restarts: int = 0,
    crash_causes: Iterable[dict[str, Any]] = (),
) -> dict[str, Any]:
    """Synthesize the summary from the upstream CSVs and result directories."""
    trajectories = read_rows(design_dir / "trajectory_stats.csv")
    mpnn = read_rows(design_dir / "mpnn_design_stats.csv")
    rejected = read_rows(design_dir / "rejected_mpnn_full_stats.csv")
    failures = read_rows(design_dir / "failure_csv.csv")
    accepted = _pdbs(design_dir / "Accepted")
    ranked = _pdbs(design_dir / "Accepted" / "Ranked")
    relaxed = _pdbs(design_dir / "Trajectory" / "Relaxed")
    times = trajectory_times(trajectories)
    return {
        "schema_version": SCHEMA_VERSION,
        "status": status,
        "finished": finished,
        "exit_code": exit_code,
        "started_at": started_at,
        "finished_at": finished_at,
        "wall_time_s": round(wall_time_s, 1),
        "time_to_first_trajectory_s": None if first_trajectory_s is None else round(first_trajectory_s, 1),
        "trajectories": len(trajectories),
        "trajectories_relaxed": len(relaxed),
        "trajectories_rejected_early": _failure_count(failures, _TRAJECTORY_FAILURE_PREFIX),
        "per_trajectory_times_s": times,
        "mean_trajectory_time_s": round(sum(times) / len(times), 1) if times else None,
        "mpnn_designs_scored": len(mpnn),
        "mpnn_designs_rejected": len(rejected),
        "accepted_designs": len(accepted),
        "ranked_designs": len(ranked),
        "requested_designs": settings.get("num_designs"),
        "resumed": resumed,
        "restarts": restarts,
        "crash_causes": list(crash_causes),
        "accelerators": accelerators,
        "settings": settings,
        "notes": list(notes),
    }


def write_summary(out_dir: Path, summary: dict[str, Any]) -> None:
    atomic_write_text(out_dir / "summary.json", json.dumps(summary, indent=2, sort_keys=False) + "\n")
