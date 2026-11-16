#!/usr/bin/env python3
# Copyright © Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""
GROMACS mdrun subprocess wrapper - long-running job container.

Behaviour (see image/README.md for the user-facing contract):

* ``gmx mdrun`` is started with ``Popen`` in its own session/process group and its
  combined stdout+stderr is teed line by line to this process's stdout (so
  ``docker logs`` / ``kubectl logs`` / Slurm see progress live) and to
  ``<output>/gromacs.log``. The log file is opened before anything else runs, so it
  exists even when the binary is missing or cannot be started.
* SIGTERM / SIGINT received by this process are forwarded to the child process group
  (mpirun and its ranks). If the child does not exit within ``KILL_GRACE_SECONDS`` it
  is killed. A signalled run always exits ``128 + signal`` (143 for SIGTERM).
* A child killed by a signal (negative ``returncode``) is reported as ``128 + signal``
  like a shell would, not as ``256 - signal``.
* mdrun works in ``<output>/.work`` (GROMACS writes md.* files in place, so a reader
  of ``<output>`` must never see a half-written file). Only on success are the finished
  files moved into ``<output>`` (``os.replace``, same filesystem) and a ``DONE`` marker
  is written last. On failure or interruption the partial files stay in ``.work``,
  including the checkpoint, so the next run in the same output directory resumes.
* A successful run reusing an output directory removes the ``md.*`` files of an earlier run that this run did not
  produce (e.g. an old ``md.xtc``), after the new set is in place and before ``DONE`` is written, so the directory
  never mixes two simulations. ``gromacs.log`` and any other file are left alone.
"""

from __future__ import annotations

import logging
import os
import signal
import subprocess
import sys
import threading
from collections import deque
from pathlib import Path
from typing import IO, Optional

from utils import parse_performance

logger = logging.getLogger(__name__)

# GROMACS binaries
_GMX_TMPI = "/opt/gromacs-tmpi/bin/gmx"  # thread-MPI build (single rank, no system MPI)
_GMX_MPI = "/opt/gromacs-mpi/bin/gmx_mpi"  # OpenMPI build (multi-rank, requires mpirun)

# Seconds between forwarding a termination signal and SIGKILL of the child group.
KILL_GRACE_SECONDS = 60.0
# Exit code when the mdrun/mpirun executable cannot be started (shell convention).
EXIT_CANNOT_EXEC = 127
WORK_DIR_NAME = ".work"
DONE_MARKER = "DONE"
_TAIL_LINES = 30


def exit_code_from_returncode(returncode: int) -> int:
    """Map a ``subprocess`` returncode to a process exit status (128+N for signal N)."""
    return 128 - returncode if returncode < 0 else returncode


def _build_gmx_env(ntomp: int) -> dict[str, str]:
    """Return an environment dict with GPU/MPI performance flags set."""
    env = os.environ.copy()

    # Ensure both GROMACS bin dirs are on PATH
    gromacs_paths = "/opt/gromacs-mpi/bin:/opt/gromacs-tmpi/bin"
    current_path = env.get("PATH", "/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin")
    if "/opt/gromacs" not in current_path:
        env["PATH"] = gromacs_paths + ":" + current_path

    # GPU + MPI performance flags (setdefault: respect caller overrides)
    env.setdefault("GMX_ENABLE_DIRECT_GPU_COMM", "1")
    env.setdefault("GMX_FORCE_GPU_AWARE_MPI", "1")
    env.setdefault("GMX_GPU_DISABLE_COMPATIBILITY_CHECK", "1")
    env.setdefault("ROC_ACTIVE_WAIT_TIMEOUT", "0")
    env.setdefault("AMD_DIRECT_DISPATCH", "1")
    env.setdefault("OMPI_ALLOW_RUN_AS_ROOT", "1")
    env.setdefault("OMPI_ALLOW_RUN_AS_ROOT_CONFIRM", "1")
    env.setdefault("OMPI_MCA_btl_vader_single_copy_mechanism", "none")
    env["OMP_NUM_THREADS"] = str(ntomp)

    return env


def _select_binary(ntmpi: int) -> str:
    """Select the gmx binary based on rank count.

    For ntmpi > 1: use gmx_mpi (OpenMPI build, invoked via mpirun).
    For ntmpi == 1: use gmx (thread-MPI build, no system MPI needed), falling
    back to gmx_mpi. Raises RuntimeError if no suitable binary exists.
    """
    if ntmpi > 1:
        gmx_bin = _GMX_MPI
        if not Path(gmx_bin).is_file():
            raise RuntimeError(
                f"MPI binary not found: {gmx_bin}. Build with -DGMX_MPI=ON -DCMAKE_INSTALL_PREFIX=/opt/gromacs-mpi."
            )
    else:
        gmx_bin = _GMX_TMPI
        if not Path(gmx_bin).is_file():
            gmx_bin = _GMX_MPI
            if not Path(gmx_bin).is_file():
                raise RuntimeError(f"Neither {_GMX_TMPI} nor {_GMX_MPI} found. Check Dockerfile install paths.")
    return gmx_bin


class _SignalForwarder:
    """Context manager: forward SIGTERM/SIGINT to the child's process group.

    Must be entered from the main thread (signal.signal restriction). Records the
    first signal received so the caller can report ``128 + signal``.
    """

    _SIGNALS = (signal.SIGTERM, signal.SIGINT)

    def __init__(self) -> None:
        self.proc: Optional[subprocess.Popen] = None
        self.received: Optional[int] = None
        self._previous: dict[int, object] = {}
        self._killer: Optional[threading.Timer] = None

    def __enter__(self) -> "_SignalForwarder":
        for sig in self._SIGNALS:
            self._previous[sig] = signal.signal(sig, self._handler)
        return self

    def __exit__(self, *exc: object) -> None:
        if self._killer is not None:
            self._killer.cancel()
        for sig, prev in self._previous.items():
            signal.signal(sig, prev)  # type: ignore[arg-type]

    def attach(self, proc: subprocess.Popen) -> None:
        """Set the child; forward a signal that arrived before it was started."""
        self.proc = proc
        if self.received is not None:
            self._forward(self.received)

    def _handler(self, signum: int, _frame: object) -> None:
        if self.received is None:
            self.received = signum
        logger.warning("Received signal %d; forwarding to the GROMACS process group", signum)
        self._forward(signum)

    def _forward(self, signum: int) -> None:
        proc = self.proc
        if proc is None or proc.poll() is not None:
            return
        try:
            os.killpg(proc.pid, signum)
        except ProcessLookupError:
            return
        if self._killer is None:
            self._killer = threading.Timer(KILL_GRACE_SECONDS, self._kill, args=(proc.pid,))
            self._killer.daemon = True
            self._killer.start()

    @staticmethod
    def _kill(pgid: int) -> None:
        try:
            os.killpg(pgid, signal.SIGKILL)
        except ProcessLookupError:
            pass


def _tee(proc: subprocess.Popen, log_fh: IO[str], tail: "deque[str]") -> None:
    assert proc.stdout is not None
    for line in proc.stdout:
        sys.stdout.write(line)
        sys.stdout.flush()
        log_fh.write(line)
        log_fh.flush()
        tail.append(line.rstrip("\n"))


def _publish(work_dir: Path, output_dir: Path) -> list[str]:
    """Move the finished mdrun files from work_dir into output_dir (atomic per file)."""
    moved = []
    for src in sorted(work_dir.iterdir()):
        if src.is_file():
            os.replace(src, output_dir / src.name)
            moved.append(src.name)
    try:
        work_dir.rmdir()
    except OSError:
        pass
    return moved


def _remove_stale_outputs(output_dir: Path, produced: list[str]) -> list[str]:
    """Remove ``md.*`` files left by an earlier run in output_dir that this run did not produce."""
    keep = set(produced)
    removed = []
    for old in sorted(output_dir.glob("md.*")):
        if old.is_file() and old.name not in keep:
            old.unlink()
            removed.append(old.name)
    return removed


def run_mdrun(
    tpr_path: Path,
    output_dir: Path,
    ntmpi: int = 1,
    ntomp: int = 48,
    nstlist: int = 200,
    nsteps: int = -2,
    gpu_id: str = "0",
    pme_tune: bool = False,
    bonded_cpu: bool = True,
) -> int:
    """Run gmx mdrun; return the process exit status (0 success, 128+N if signalled).

    Output lands in ``output_dir`` only for a successful run (see module docstring);
    ``output_dir/gromacs.log`` is always written (live).
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    gmx_log = output_dir / "gromacs.log"
    work_dir = output_dir / WORK_DIR_NAME
    work_dir.mkdir(exist_ok=True)
    (output_dir / DONE_MARKER).unlink(missing_ok=True)

    tail: "deque[str]" = deque(maxlen=_TAIL_LINES)
    returncode: int  # always assigned inside the try/except below before note() reads it
    forwarder = _SignalForwarder()

    with open(gmx_log, "a", errors="replace") as log_fh, forwarder:

        def note(msg: str) -> None:
            logger.error(msg) if returncode != 0 else logger.info(msg)
            log_fh.write(msg + "\n")
            log_fh.flush()

        try:
            gmx_bin = _select_binary(ntmpi)
            env = _build_gmx_env(ntomp)

            mdrun_args = [
                gmx_bin,
                "mdrun",
                "-v",
                "-s",
                str(tpr_path),
                "-deffnm",
                str(work_dir / "md"),
                "-ntomp",
                str(ntomp),
                "-pin",
                "on",
                "-nb",
                "gpu",
                "-bonded",
                "cpu" if bonded_cpu else "gpu",
                "-pme",
                "gpu",
                "-update",
                "gpu",
                "-gpu_id",
                gpu_id,
                "-nstlist",
                str(nstlist),
            ]
            if not pme_tune:
                # PME auto-tuning replans the GPU FFT for ~9 grid trials; on short runs this costs more
                # than it gains (preliminary evidence, see templates/API.md). Opt back in with pme_tune.
                mdrun_args.append("-notunepme")
            if nsteps >= 0:
                mdrun_args += ["-nsteps", str(nsteps)]
            checkpoint = work_dir / "md.cpt"
            if checkpoint.is_file():
                # Resume explicitly: do not rely on mdrun picking up md.cpt from -deffnm alone.
                mdrun_args += ["-cpi", str(checkpoint)]

            if ntmpi > 1:
                # --bind-to none: without it OpenMPI pins each rank to 1 core and all
                # OpenMP threads fight for it (throughput drops noticeably)
                cmd = ["mpirun", "--bind-to", "none", "-np", str(ntmpi)] + mdrun_args
            else:
                cmd = mdrun_args + ["-ntmpi", "1"]

            logger.info("Running: %s", " ".join(cmd))
            log_fh.write("$ " + " ".join(cmd) + "\n")
            log_fh.flush()

            proc = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                errors="replace",
                env=env,
                cwd=str(work_dir),
                start_new_session=True,
            )
            forwarder.attach(proc)
            _tee(proc, log_fh, tail)
            returncode = exit_code_from_returncode(proc.wait())
        except FileNotFoundError as exc:
            returncode = EXIT_CANNOT_EXEC
            note(f"Cannot start GROMACS: {exc}")
        except Exception as exc:  # noqa: BLE001 - always leave a log and a non-zero status
            returncode = 1
            logger.exception("Unexpected error running gmx mdrun")
            note(f"Unexpected error running gmx mdrun: {exc}")

        if forwarder.received is not None:
            # mdrun exits 0 after a graceful SIGTERM stop; the run is still incomplete.
            returncode = 128 + forwarder.received
            note(
                f"Interrupted by signal {forwarder.received}; partial results and checkpoint kept in "
                f"{work_dir}. Re-run with the same output directory to resume."
            )

        md_log = work_dir / "md.log"
        log_text = md_log.read_text(errors="replace") if md_log.is_file() else "\n".join(tail)

        if returncode == 0:
            moved = _publish(work_dir, output_dir)
            stale = _remove_stale_outputs(output_dir, moved)
            if stale:
                note(f"Removed output(s) of an earlier run that this run did not produce: {', '.join(stale)}")
            (output_dir / DONE_MARKER).write_text("0\n")
            note(f"Published {len(moved)} file(s) to {output_dir}: {', '.join(moved)}")
        else:
            note(
                f"gmx mdrun exited {returncode}. Last lines:\n" + "\n".join(tail)
                if tail
                else f"gmx mdrun exited {returncode}."
            )

    ns_per_day, hours_per_ns = parse_performance(log_text)
    if ns_per_day > 0.0:
        logger.info("GROMACS performance: %.3f ns/day  (%.4f hours/ns)", ns_per_day, hours_per_ns)
    elif returncode == 0:
        logger.warning("Could not parse performance line from GROMACS log")

    return returncode
