#!/usr/bin/env python3
# Copyright © Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""
GROMACS AIM entrypoint - long-running job container contract.

Reads ``run.tpr`` (or ``--tpr``) from AIM_INPUT_PATH, runs ``gmx mdrun``, writes results
to AIM_OUTPUT_PATH. Exit status: 0 success; 2 invalid settings; 1 runtime/input failure;
127 GROMACS cannot be started; 128+N when terminated by signal N (143 for SIGTERM).

Settings (environment; see image/README.md):
  AIM_INPUT_PATH         directory with the .tpr (required)
  AIM_OUTPUT_PATH        output directory (required)
  AIM_ENGINE_ARGS        JSON object, same grammar as the platform, e.g. {"ntomp": 24}
                         keys: ntomp, ntmpi, nstlist, nsteps, gpu-id, tpr, pme-tune, bonded-cpu
  AIM_ACCELERATOR_COUNT  GPUs to use; overrides device-node detection
  AIM_ACCELERATOR_MODEL  informational only (logged)
Any other AIM_* variable is rejected (exit 2).

Precedence for each tunable: command-line flag > AIM_ENGINE_ARGS > default.
"""

from __future__ import annotations

import argparse
import glob
import json
import logging
import os
import sys
from pathlib import Path
from typing import Any, Mapping, Optional

from runner import run_mdrun

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

EXIT_USAGE = 2
KNOWN_AIM_VARS = frozenset(
    {
        "AIM_INPUT_PATH",
        "AIM_OUTPUT_PATH",
        "AIM_ENGINE_ARGS",
        "AIM_ACCELERATOR_COUNT",
        "AIM_ACCELERATOR_MODEL",
    }
)
# AIM_ENGINE_ARGS keys (kebab or snake case accepted) -> (argparse dest, type)
ENGINE_ARG_KEYS: dict[str, type] = {
    "ntomp": int,
    "ntmpi": int,
    "nstlist": int,
    "nsteps": int,
    "gpu_id": str,
    "tpr": str,
    "pme_tune": bool,
    "bonded_cpu": bool,
}
MAX_DEFAULT_OMP_THREADS = 24  # total OpenMP threads across all ranks, see default_ntomp
# ntomp has no static default: when unset it is derived from the usable hardware threads (default_ntomp).
DEFAULTS: dict[str, Any] = {
    "nstlist": 200,
    "nsteps": -2,
    "gpu_id": "",
    "tpr": "run.tpr",
    "pme_tune": False,
    "bonded_cpu": True,  # bonded forces on the CPU; False = -bonded gpu
}


def usable_cpus() -> int:
    """Hardware threads this process may run on (honours the container cpuset)."""
    try:
        return len(os.sched_getaffinity(0)) or 1
    except (AttributeError, OSError):
        return os.cpu_count() or 1


def default_ntomp(ntmpi: int, cpus: int) -> int:
    """OpenMP threads per rank: at most 24 in total, and always one hardware thread free.

    ``min(24, usable CPUs - 1)`` threads are split across the ranks (floor, at least 1).
    One OpenMP thread per hardware thread starves the HIP runtime thread, and more threads
    than physical cores do not help.
    """
    return max(1, min(MAX_DEFAULT_OMP_THREADS, cpus - 1) // ntmpi)


class SettingsError(ValueError):
    """Invalid user-supplied settings (exit status 2)."""


def check_aim_env(environ: "os._Environ[str] | dict[str, str]") -> list[str]:
    """Warn about unknown ``AIM_*`` variables and ignore them; return their names.

    A launcher may set variables for every AIM (``AIM_PROFILE_ID``, ``AIM_LOG_LEVEL``, ...) that this job does not
    use, so they must not stop the container. Bad ``AIM_ENGINE_ARGS`` keys are still rejected (``parse_engine_args``).
    """
    unknown = sorted(k for k in environ if k.startswith("AIM_") and k not in KNOWN_AIM_VARS)
    if unknown:
        logger.warning(
            "Ignoring unknown setting(s): %s. Supported AIM_* variables: %s",
            ", ".join(unknown),
            ", ".join(sorted(KNOWN_AIM_VARS)),
        )
    return unknown


def parse_engine_args(raw: str) -> dict[str, Any]:
    """Parse AIM_ENGINE_ARGS (a JSON object) into validated, snake_case settings."""
    raw = raw.strip()
    if not raw:
        return {}
    try:
        obj = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise SettingsError(f'AIM_ENGINE_ARGS must be a JSON object such as {{"ntomp": 24}}: {exc}') from exc
    if not isinstance(obj, dict):
        raise SettingsError(f"AIM_ENGINE_ARGS must be a JSON object, got {type(obj).__name__}")
    out: dict[str, Any] = {}
    for key, value in obj.items():
        norm = str(key).lstrip("-").replace("-", "_")
        if norm not in ENGINE_ARG_KEYS:
            raise SettingsError(f"Unknown AIM_ENGINE_ARGS key {key!r}. Supported: {', '.join(sorted(ENGINE_ARG_KEYS))}")
        typ = ENGINE_ARG_KEYS[norm]
        if typ is bool:
            if isinstance(value, str) and value.strip().lower() in ("true", "false"):
                value = value.strip().lower() == "true"
            if not isinstance(value, bool):
                raise SettingsError(f"AIM_ENGINE_ARGS[{key!r}] must be true or false, got {value!r}")
        elif typ is int:
            if isinstance(value, bool) or not isinstance(value, (int, str)):
                raise SettingsError(f"AIM_ENGINE_ARGS[{key!r}] must be an integer, got {value!r}")
            try:
                value = int(value)
            except ValueError as exc:
                raise SettingsError(f"AIM_ENGINE_ARGS[{key!r}] must be an integer, got {value!r}") from exc
        else:
            value = str(value)
        out[norm] = value
    return out


def _listed_devices(value: str) -> int:
    """Number of devices in a ROCR/HIP_VISIBLE_DEVICES list; a negative id (e.g. -1) ends the list."""
    count = 0
    for token in value.split(","):
        token = token.strip()
        if not token or token.startswith("-"):
            break
        count += 1
    return count


def detect_gpu_count(dev_root: str = "/dev", environ: "Optional[Mapping[str, str]]" = None) -> int:
    """Number of AMD GPUs usable by this job.

    Starts from /dev/dri/renderD* when /dev/kfd exists, then honours ROCR_VISIBLE_DEVICES and HIP_VISIBLE_DEVICES
    (a scheduler may expose every render node but restrict ROCm to the allocated ones): when set and non-empty, the
    number of listed devices caps the count. Unset or empty variables change nothing.
    """
    if not os.path.exists(os.path.join(dev_root, "kfd")):
        return 0
    count = len(glob.glob(os.path.join(dev_root, "dri", "renderD*")))
    environ = os.environ if environ is None else environ
    for var in ("ROCR_VISIBLE_DEVICES", "HIP_VISIBLE_DEVICES"):
        value = environ.get(var, "").strip()
        if value:
            count = min(count, _listed_devices(value))
    return count


def _count_override(environ: "os._Environ[str] | dict[str, str]") -> Optional[int]:
    raw = environ.get("AIM_ACCELERATOR_COUNT", "").strip()
    if not raw:
        return None
    try:
        n = int(raw)
    except ValueError as exc:
        raise SettingsError(f"AIM_ACCELERATOR_COUNT must be an integer, got {raw!r}") from exc
    if n < 1 or n > 9:
        raise SettingsError(f"AIM_ACCELERATOR_COUNT must be between 1 and 9, got {n}")
    return n


def build_parser(environ: "os._Environ[str] | dict[str, str]") -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="GROMACS AIM - long-running job")
    p.add_argument("--input", default=environ.get("AIM_INPUT_PATH", ""), help="Input directory containing .tpr file")
    p.add_argument("--output", default=environ.get("AIM_OUTPUT_PATH", ""), help="Output directory for all results")
    p.add_argument("--tpr", help="tpr filename relative to --input (default run.tpr)")
    p.add_argument(
        "--ntomp",
        type=int,
        help="OpenMP threads per rank (default: usable hardware threads minus one, split across ranks)",
    )
    p.add_argument("--ntmpi", type=int, help="GPUs / MPI ranks (default: detected or AIM_ACCELERATOR_COUNT)")
    p.add_argument("--nstlist", type=int, help="Neighbor-list update interval (default 200)")
    p.add_argument("--nsteps", type=int, help="Steps to run (-2 = .tpr default)")
    p.add_argument("--gpu-id", dest="gpu_id", help="GPU device string, e.g. 0123 (default: first ntmpi GPUs)")
    p.add_argument(
        "--pme-tune",
        dest="pme_tune",
        action=argparse.BooleanOptionalAction,
        help="Let mdrun auto-tune the PME grid (default off: -notunepme is passed)",
    )
    p.add_argument(
        "--bonded-cpu",
        dest="bonded_cpu",
        action=argparse.BooleanOptionalAction,
        help="Compute bonded interactions on the CPU (-bonded cpu; default on; --no-bonded-cpu computes them on the GPU)",
    )
    return p


def resolve_settings(
    argv: list[str], environ: "os._Environ[str] | dict[str, str]", detected_gpus: int
) -> argparse.Namespace:
    """Merge CLI > AIM_ENGINE_ARGS > defaults and resolve the GPU count. Raises SettingsError."""
    check_aim_env(environ)
    p = build_parser(environ)
    try:
        args = p.parse_args(argv)
    except SystemExit as exc:  # argparse already printed the message
        raise SettingsError("invalid command-line arguments") from exc
    env_args = parse_engine_args(environ.get("AIM_ENGINE_ARGS", ""))

    for key in ENGINE_ARG_KEYS:
        if getattr(args, key, None) is None:
            setattr(args, key, env_args.get(key, DEFAULTS.get(key)))

    override = _count_override(environ)
    available = override if override is not None else detected_gpus
    if override is not None and 0 < detected_gpus < override:
        raise SettingsError(f"AIM_ACCELERATOR_COUNT={override} but only {detected_gpus} GPU device node(s) are visible")
    if args.ntmpi is None:
        if available < 1:
            raise SettingsError(
                "No AMD GPU found (/dev/kfd and /dev/dri/renderD* missing). Pass --device /dev/kfd --device "
                "/dev/dri, or set AIM_ACCELERATOR_COUNT to override detection."
            )
        args.ntmpi = available
    if args.ntmpi < 1 or args.ntmpi > 9:
        raise SettingsError(f"ntmpi must be between 1 and 9, got {args.ntmpi}")
    if available >= 1 and args.ntmpi > available:
        raise SettingsError(f"ntmpi={args.ntmpi} requested but only {available} GPU(s) available")
    if args.ntomp is None:
        args.ntomp = default_ntomp(args.ntmpi, usable_cpus())
    if args.ntomp < 1:
        raise SettingsError(f"ntomp must be >= 1, got {args.ntomp}")
    args.gpu_id = args.gpu_id or "".join(str(i) for i in range(args.ntmpi))
    args.accelerator_model = environ.get("AIM_ACCELERATOR_MODEL", "")
    return args


def main(argv: Optional[list[str]] = None, environ: Optional[dict[str, str]] = None) -> int:
    argv = sys.argv[1:] if argv is None else argv
    environ = os.environ if environ is None else environ  # type: ignore[assignment]
    try:
        args = resolve_settings(argv, environ, detect_gpu_count())  # type: ignore[arg-type]
    except SettingsError as exc:
        logger.error("%s", exc)
        return EXIT_USAGE

    if not args.input:
        logger.error("AIM_INPUT_PATH is not set and --input was not provided")
        return EXIT_USAGE
    if not args.output:
        logger.error("AIM_OUTPUT_PATH is not set and --output was not provided")
        return EXIT_USAGE

    # Absolute paths: mdrun runs in <output>/.work, so relative -s / -deffnm paths would point to nothing.
    input_dir = Path(os.path.abspath(args.input))
    output_dir = Path(os.path.abspath(args.output))
    tpr_path = input_dir / args.tpr

    # Create gromacs.log before the input checks so a broken input is diagnosed in it (README: "always").
    output_dir.mkdir(parents=True, exist_ok=True)
    gmx_log = output_dir / "gromacs.log"
    gmx_log.touch()

    def input_error(msg: str, *margs: object) -> int:
        logger.error(msg, *margs)
        with open(gmx_log, "a", errors="replace") as fh:
            fh.write("ERROR: " + (msg % margs) + "\n")
        return 1

    if not input_dir.is_dir():
        return input_error("Input directory does not exist: %s", input_dir)
    if not tpr_path.is_file():
        return input_error("TPR file not found: %s", tpr_path)

    logger.info(
        "GROMACS AIM starting: tpr=%s ntmpi=%d ntomp=%d nstlist=%d nsteps=%d gpu_id=%s pme_tune=%s bonded_cpu=%s accelerator_model=%s",
        tpr_path,
        args.ntmpi,
        args.ntomp,
        args.nstlist,
        args.nsteps,
        args.gpu_id,
        args.pme_tune,
        args.bonded_cpu,
        args.accelerator_model or "unset",
    )

    returncode = run_mdrun(
        tpr_path=tpr_path,
        output_dir=output_dir,
        ntmpi=args.ntmpi,
        ntomp=args.ntomp,
        nstlist=args.nstlist,
        nsteps=args.nsteps,
        gpu_id=args.gpu_id,
        pme_tune=args.pme_tune,
        bonded_cpu=args.bonded_cpu,
    )

    if returncode == 0:
        logger.info("GROMACS run completed successfully. Output in %s", output_dir)
    else:
        logger.error("GROMACS run failed with exit code %d. Logs in %s", returncode, output_dir)
    return returncode


if __name__ == "__main__":
    sys.exit(main())
