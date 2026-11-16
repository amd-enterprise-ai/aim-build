#!/usr/bin/env python3
# Copyright © Advanced Micro Devices, Inc., or its affiliates.
#
# SPDX-License-Identifier: MIT

"""FreeBindCraft AIM entrypoint: long-running job container.

Reads ``target.pdb`` (and an optional ``settings.json``) from ``AIM_INPUT_PATH``,
runs FreeBindCraft (``bindcraft.py --no-pyrosetta``) and writes results, CSVs,
``summary.json`` and ``bindcraft.log`` under ``AIM_OUTPUT_PATH``.

Exit status:
    0    finished: the requested number of designs was reached, or the
         ``max_trajectories`` cap was reached
    1    the run failed (including a crashing upstream whose restarts are exhausted)
    2    invalid input or settings (nothing was started)
    3    not enough accelerators
    4    the run stopped early with fewer designs than requested and no cap hit
         (upstream's low-acceptance-rate guard)
    0    also: AIM_FBC_PREFLIGHT_ONLY=1 and the settings, target and GPU checks passed (nothing was designed)
    124  the wall-clock ``time_limit`` was reached (partial results are published)
    128+n  killed by signal n (143 for SIGTERM)

Settings: see fbc_settings.py and the image README. This module owns the
contract; the science stays in the upstream tool.
"""

from __future__ import annotations

import json
import os
import re
import shutil
import signal
import sys
import time
from pathlib import Path
from typing import Mapping, Optional, Sequence

sys.path.insert(0, str(Path(__file__).resolve().parent))

from fbc_preflight import (  # noqa: E402
    PreflightError,
    check_resume_guard,
    hotspot_report,
    input_fingerprint,
    require_accelerators,
    resolve_af_params_dir,
    validate_target,
)
from fbc_runner import (  # noqa: E402
    DEFAULT_GRACE_S,
    DEFAULT_TICK_S,
    Log,
    RunResult,
    SignalLatch,
    exit_status,
    run_supervised,
)
from fbc_settings import ENGINE_ARGS_VAR, Settings, SettingsError, resolve_settings  # noqa: E402
from fbc_summary import build_summary, publish_results, read_rows, write_summary  # noqa: E402

LOG_NAME = "bindcraft.log"
WORK_DIR_NAME = ".work"
DEFAULT_BINDCRAFT_HOME = "/app"
NO_PROGRESS_LIMIT = 3  # consecutive crashes without any new trajectory result: give up
ADVANCED_TEMPLATE = "settings_advanced/default_4stage_multimer.json"
FILTERS_TEMPLATE = "settings_filters/default_filters.json"

# Markers in upstream's output.
_MARKERS = {
    "first_trajectory": re.compile(r"Starting trajectory took:"),
    "no_gpu": re.compile(r"No GPU device found, terminating"),
}

USAGE = """FreeBindCraft AIM. Run the image with no arguments.

  AIM_INPUT_PATH   directory with target.pdb and optional settings.json (read-only)
  AIM_OUTPUT_PATH  directory for results, summary.json and bindcraft.log
  AIM_FBC_TARGET_CHAINS, AIM_FBC_HOTSPOTS, AIM_FBC_BINDER_NAME, AIM_FBC_MIN_LENGTH,
  AIM_FBC_MAX_LENGTH, AIM_FBC_NUM_DESIGNS, AIM_FBC_MAX_TRAJECTORIES, AIM_FBC_TIME_LIMIT,
  AIM_FBC_MAX_RESTARTS
  AIM_FBC_PREFLIGHT_ONLY=1
                   check-only mode: validate settings, target and GPU, write the log and summary.json,
                   exit 0 without designing anything
                   run settings (also keys of settings.json without the prefix)
See the image README for the full contract.
"""


def _utc_now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def prepare_configs(
    settings: Settings, target: Path, work_dir: Path, bindcraft_home: Path, af_params_dir: str
) -> tuple[Path, Path, Path]:
    """Write the target-settings and advanced-settings JSON the upstream CLI reads.

    Upstream has no CLI flag for the trajectory cap, so ``max_trajectories`` goes
    into a copy of the advanced-settings JSON; the shipped file is never edited.
    Returns ``(settings_json, advanced_json, design_dir)``.
    """
    config_dir = work_dir / "config"
    design_dir = work_dir / "design"
    config_dir.mkdir(parents=True, exist_ok=True)
    design_dir.mkdir(parents=True, exist_ok=True)

    target_copy = config_dir / "target.pdb"
    target_copy.write_bytes(target.read_bytes())

    target_settings = {
        "design_path": str(design_dir),
        "binder_name": settings.binder_name,
        "starting_pdb": str(target_copy),
        "chains": settings.target_chains,
        "target_hotspot_residues": settings.hotspots,
        "lengths": [settings.min_length, settings.max_length],
        "number_of_final_designs": settings.num_designs,
    }
    settings_json = config_dir / "target_settings.json"
    settings_json.write_text(json.dumps(target_settings, indent=2) + "\n", encoding="utf-8")

    advanced = json.loads((bindcraft_home / ADVANCED_TEMPLATE).read_text(encoding="utf-8"))
    advanced["max_trajectories"] = settings.max_trajectories if settings.max_trajectories else False
    advanced["af_params_dir"] = af_params_dir
    # Design animations render an HTML5 video through ffmpeg, which the image does not ship (GPL-licensed build,
    # removed after the licence review). Upstream defaults the setting to true, so switch it off explicitly; designs
    # and scores do not depend on it.
    advanced["save_design_animations"] = False
    # The image ships conda-forge's mkdssp (BSD-2) instead of a bundled binary; point upstream at it so
    # secondary-structure metrics never depend on the (removed) functions/dssp.
    mkdssp = shutil.which("mkdssp")
    if mkdssp:
        advanced["dssp_path"] = mkdssp
    advanced_json = config_dir / "advanced_settings.json"
    advanced_json.write_text(json.dumps(advanced, indent=2) + "\n", encoding="utf-8")
    return settings_json, advanced_json, design_dir


def classify(result: RunResult, settings: Settings, design_dir: Path, no_gpu_seen: bool) -> tuple[str, int, list[str]]:
    """Map the finished run to ``(status, exit_code, notes)``."""
    code = exit_status(result)
    if result.signal_received is not None:
        return "interrupted", code, [f"stopped by signal {result.signal_received}; partial results published"]
    if result.timed_out:
        return "time_limit", 124, ["wall-clock time_limit reached; partial results published"]
    if code != 0:
        return "failed", code, [f"bindcraft.py exited with status {code}"]
    if no_gpu_seen:
        return "failed", 1, ["upstream reported no usable GPU (it exits 0 in that case)"]
    trajectories = read_rows(design_dir / "trajectory_stats.csv")
    accepted = [p for p in (design_dir / "Accepted").glob("*.pdb") if not p.name.startswith(".")]
    relaxed = [p for p in (design_dir / "Trajectory" / "Relaxed").glob("*.pdb") if not p.name.startswith(".")]
    if not trajectories:
        return "failed", 1, ["bindcraft.py exited 0 without scoring a single trajectory"]
    if len(accepted) >= settings.num_designs:
        return "completed", 0, []
    if settings.max_trajectories and len(relaxed) >= settings.max_trajectories:
        return (
            "max_trajectories_reached",
            0,
            [f"{len(accepted)} of {settings.num_designs} designs accepted within the trajectory cap"],
        )
    return "stopped_early", 4, ["upstream stopped before reaching the requested designs (low acceptance rate)"]


def _counter_total(path: Path) -> int:
    """Sum of the integer counters in ``failure_csv.csv``.

    Upstream (functions/generic_utils.py) writes one header row and a single row of per-filter failure counters
    and increments them in place, so the row count never changes; the counter values do.
    """
    total = 0
    for row in read_rows(path):
        for value in row.values():
            try:
                total += int(float(value))
            except (TypeError, ValueError):
                continue  # not a counter (empty or malformed cell)
    return total


def progress_count(design_dir: Path) -> int:
    """Grows whenever a trajectory finishes or is rejected: rows of the result CSVs plus the failure counters."""
    names = ("trajectory_stats.csv", "mpnn_design_stats.csv", "rejected_mpnn_full_stats.csv")
    rows = sum(len(read_rows(design_dir / name)) for name in names)
    return rows + _counter_total(design_dir / "failure_csv.csv")


def main(
    argv: Optional[Sequence[str]] = None,
    env: Optional[Mapping[str, str]] = None,
    dev_root: Path = Path("/dev"),
    grace_s: float = DEFAULT_GRACE_S,
    tick_s: float = DEFAULT_TICK_S,
) -> int:
    # One latch for the whole run: a signal outside the supervised child (preflight,
    # restart gap, final publish) is kept and honoured instead of killing the process.
    with SignalLatch() as latch:
        return _run(latch, argv, env, dev_root, grace_s, tick_s)


def _run(
    latch: SignalLatch,
    argv: Optional[Sequence[str]],
    env: Optional[Mapping[str, str]],
    dev_root: Path,
    grace_s: float,
    tick_s: float,
) -> int:
    args = list(sys.argv[1:] if argv is None else argv)
    environ = dict(os.environ if env is None else env)
    if args in (["-h"], ["--help"]):
        sys.stdout.write(USAGE)
        return 0
    if args:
        sys.stderr.write(
            f"error: unexpected arguments {args}; settings come from AIM_* variables and settings.json\n{USAGE}"
        )
        return 2

    out_raw = environ.get("AIM_OUTPUT_PATH", "").strip()
    if not out_raw:
        sys.stderr.write("error: AIM_OUTPUT_PATH is not set\n")
        return 2
    out_dir = Path(out_raw)
    try:
        out_dir.mkdir(parents=True, exist_ok=True)
        log = Log(out_dir / LOG_NAME)
    except OSError as exc:
        sys.stderr.write(f"error: cannot write under AIM_OUTPUT_PATH ({out_dir}): {exc}\n")
        return 2

    started_at = _utc_now()
    t0 = time.monotonic()
    log.info(f"FreeBindCraft AIM job starting (input={environ.get('AIM_INPUT_PATH', '')!r}, output={str(out_dir)!r})")

    settings_dict: dict = {}
    accelerators: Optional[int] = None

    def fail(status_code: int, message: str, state: str = "failed") -> int:
        log.info(f"ERROR: {message}")
        summary = build_summary(
            design_dir=out_dir / WORK_DIR_NAME / "design",
            settings=settings_dict,
            status=state,
            finished=True,
            exit_code=status_code,
            started_at=started_at,
            finished_at=_utc_now(),
            wall_time_s=time.monotonic() - t0,
            first_trajectory_s=None,
            accelerators=accelerators,
            resumed=False,
            notes=[message],
        )
        try:
            write_summary(out_dir, summary)
        except OSError as exc:
            # Already failing: the original error is the one to report. Best effort only.
            log.info(f"could not write summary.json: {exc}")
        log.close()
        return status_code

    try:
        in_raw = environ.get("AIM_INPUT_PATH", "").strip()
        if not in_raw:
            raise PreflightError("AIM_INPUT_PATH is not set")
        input_dir = Path(in_raw)
        settings = resolve_settings(environ, input_dir) if input_dir.is_dir() else None
        if settings is None:
            raise PreflightError(f"AIM_INPUT_PATH is not a directory: {input_dir}")
        settings_dict = settings.as_dict()
        target = validate_target(input_dir, settings)
        accelerators = require_accelerators(environ, dev_root)
    except SettingsError as exc:
        return fail(2, f"invalid settings: {exc}")
    except PreflightError as exc:
        return fail(exc.status, str(exc))

    model = environ.get("AIM_ACCELERATOR_MODEL", "")
    log.info(f"settings: {json.dumps(settings_dict)}")
    log.info(f"accelerators: {accelerators}" + (f" ({model})" if model else ""))
    hotspot_line = hotspot_report(target, settings)
    if hotspot_line:
        log.info(hotspot_line)
    if environ.get(ENGINE_ARGS_VAR, "").strip():
        log.info(f"WARNING: {ENGINE_ARGS_VAR} is set but this job has no engine arguments; it is ignored")

    if settings.preflight_only:
        # Check-only mode (CI and diagnostics): everything above passed. No AlphaFold2 parameters are needed, no work
        # directory is created and nothing is designed.
        note = "AIM_FBC_PREFLIGHT_ONLY: settings, target and accelerators are valid; no designs were run"
        log.info(note)
        summary = build_summary(
            design_dir=out_dir / WORK_DIR_NAME / "design",
            settings=settings_dict,
            status="preflight_ok",
            finished=True,
            exit_code=0,
            started_at=started_at,
            finished_at=_utc_now(),
            wall_time_s=time.monotonic() - t0,
            first_trajectory_s=None,
            accelerators=accelerators,
            resumed=False,
            notes=[note],
        )
        try:
            write_summary(out_dir, summary)
        except OSError as exc:
            return fail(1, f"cannot write summary.json under {out_dir}: {exc}")
        log.close()
        return 0

    bindcraft_home = Path(environ.get("BINDCRAFT_HOME", DEFAULT_BINDCRAFT_HOME))
    work_dir = out_dir / WORK_DIR_NAME
    try:
        resumed = check_resume_guard(work_dir, input_fingerprint(target, settings))
        af_dir = resolve_af_params_dir(environ, bindcraft_home)
        settings_json, advanced_json, design_dir = prepare_configs(settings, target, work_dir, bindcraft_home, af_dir)
    except PreflightError as exc:
        return fail(exc.status, str(exc))
    except OSError as exc:
        return fail(1, f"cannot prepare the work directory under {out_dir}: {exc}")
    if resumed:
        log.info(f"resuming from the existing work directory {work_dir}")

    def publish(
        status: str = "running",
        finished: bool = False,
        exit_code: Optional[int] = None,
        first: Optional[float] = None,
        notes: Sequence[str] = (),
    ) -> None:
        publish_results(design_dir, out_dir)
        write_summary(
            out_dir,
            build_summary(
                design_dir=design_dir,
                settings=settings_dict,
                status=status,
                finished=finished,
                exit_code=exit_code,
                started_at=started_at,
                finished_at=_utc_now() if finished else None,
                wall_time_s=time.monotonic() - t0,
                first_trajectory_s=first,
                accelerators=accelerators,
                resumed=resumed,
                notes=notes,
                restarts=restart_count[0],
                crash_causes=crash_causes,
            ),
        )

    crash_causes: list[dict] = []
    restart_count = [0]
    no_gpu_seen = False
    child_env = dict(environ)
    child_env["PYTHONUNBUFFERED"] = "1"
    cmd = [
        sys.executable,
        str(bindcraft_home / "bindcraft.py"),
        "--settings",
        str(settings_json),
        "--filters",
        str(bindcraft_home / FILTERS_TEMPLATE),
        "--advanced",
        str(advanced_json),
        "--no-pyrosetta",
    ]
    log.info("command: " + " ".join(cmd))
    publish()

    first_mark: dict[str, float] = {}
    attempt_marks: dict[str, float] = {}
    offset = 0.0  # seconds of the supervised run spent before the current attempt

    def first_seen() -> Optional[float]:
        if "first_trajectory" in first_mark:
            return first_mark["first_trajectory"]
        if "first_trajectory" in attempt_marks:
            return offset + attempt_marks["first_trajectory"]
        return None

    def tick() -> None:
        publish(first=first_seen())

    run_start = time.monotonic()
    no_progress = 0
    gave_up: Optional[str] = None
    result = RunResult()
    while True:
        if latch.signum is not None:  # signalled outside a supervised run (preflight or restart gap)
            result.signal_received = latch.signum
            log.info(f"received {signal.Signals(latch.signum).name} before the next start; not starting bindcraft.py")
            break
        remaining = None if settings.time_limit_s is None else settings.time_limit_s - (time.monotonic() - run_start)
        attempt_marks.clear()
        before = progress_count(design_dir)
        result = run_supervised(
            cmd,
            cwd=bindcraft_home,
            env=child_env,
            log=log,
            time_limit_s=remaining,
            grace_s=grace_s,
            markers=_MARKERS,
            marks=attempt_marks,
            tick=tick,
            tick_s=tick_s,
        )
        if "first_trajectory" not in first_mark and "first_trajectory" in attempt_marks:
            first_mark["first_trajectory"] = offset + attempt_marks["first_trajectory"]
        offset += result.elapsed_s
        if "no_gpu" in attempt_marks:
            no_gpu_seen = True
        crashed = (
            result.returncode != 0
            and result.signal_received is None
            and not result.timed_out
            and "no_gpu" not in attempt_marks
        )
        if not crashed:
            break
        progressed = progress_count(design_dir) - before
        no_progress = 0 if progressed > 0 else no_progress + 1
        crash_causes.append(
            {
                "attempt": len(crash_causes) + 1,
                "exit_code": exit_status(result),
                "last_line": result.last_line,
                "new_result_rows": progressed,
                "elapsed_s": round(result.elapsed_s, 1),
            }
        )
        log.info(
            f"bindcraft.py crashed (attempt {len(crash_causes)}, exit code {exit_status(result)}, "
            f"last line: {result.last_line!r})"
        )
        if len(crash_causes) > settings.max_restarts:
            if settings.max_restarts:
                gave_up = f"bindcraft.py crashed {len(crash_causes)} times; AIM_FBC_MAX_RESTARTS={settings.max_restarts} exhausted"
            break
        if no_progress >= NO_PROGRESS_LIMIT:
            gave_up = f"bindcraft.py crashed {no_progress} times in a row without any new trajectory result; giving up"
            break
        if settings.time_limit_s is not None and settings.time_limit_s - (time.monotonic() - run_start) < 1:
            result.timed_out = True
            break
        restart_count[0] += 1
        log.info(f"restarting bindcraft.py (restart {restart_count[0]} of at most {settings.max_restarts})")
        try:
            publish(first=first_seen())
        except OSError as exc:
            # Interim publish between restarts; the final publish reports failures.
            log.info(f"could not publish interim results: {exc}")

    first = first_seen()
    status, code, notes = classify(result, settings, design_dir, no_gpu_seen)
    if gave_up:
        status, code, notes = "failed", 1, [gave_up] + notes
    elif crash_causes and status != "failed":
        notes = [f"bindcraft.py was restarted {restart_count[0]} time(s) after crashes"] + notes
    log.info(f"run ended: status={status} exit_code={code} wall_time_s={offset:.1f} restarts={restart_count[0]}")
    try:
        publish(status=status, finished=True, exit_code=code, first=first, notes=notes)
    except OSError as exc:
        log.info(f"ERROR: could not publish final results: {exc}")
        code = code or 1
    log.close()
    return code


if __name__ == "__main__":
    sys.exit(main())
