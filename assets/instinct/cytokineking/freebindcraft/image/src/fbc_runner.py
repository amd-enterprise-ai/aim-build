# Copyright © Advanced Micro Devices, Inc., or its affiliates.
#
# SPDX-License-Identifier: MIT

"""Supervised run of the upstream process (logging, signals, exit codes).

* Output is streamed live: the child's stdout/stderr are read as they arrive and
  written to this process's stdout and to the log file at once (a tee), never
  buffered until the child exits.
* The child runs in its own process group. SIGTERM/SIGINT/SIGHUP received by the
  wrapper are only recorded by the handler (a handler must not take the log lock, which a
  write it interrupted may hold); the read loop logs and forwards them to that whole group; if it has not exited after a grace
  period it is killed. The caller maps a received signal to ``128 + signal``.
* A wall-clock limit is enforced here (upstream has none): SIGTERM to the group
  at the deadline, SIGKILL after the grace period.

Pure standard library.
"""

from __future__ import annotations

import os
import re
import select
import signal
import subprocess
import sys
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Optional, Sequence

FORWARDED_SIGNALS = (signal.SIGTERM, signal.SIGINT, signal.SIGHUP)
DEFAULT_GRACE_S = 20.0
DEFAULT_TICK_S = 30.0
READ_CHUNK = 65536
POLL_S = 0.1  # read-loop wake-up: bounds how late a recorded signal is forwarded and a gone leader noticed
TICK_JOIN_S = 10.0  # bound on waiting for a running periodic publish before returning


class Log:
    """Tee to stdout and an append-mode log file, flushed on every write."""

    def __init__(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        self._file = path.open("ab", buffering=0)
        self._lock = threading.Lock()
        self.path = path

    def write_bytes(self, data: bytes) -> None:
        with self._lock:
            try:
                self._file.write(data)
            except OSError:
                # Log-file write failure (disk full, etc.) must not kill the run; the
                # same output is still streamed to stdout by the caller.
                pass
            try:
                sys.stdout.buffer.write(data)
                sys.stdout.buffer.flush()
            except (OSError, ValueError):
                pass  # a closed stdout must not kill the run

    def info(self, message: str) -> None:
        stamp = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        self.write_bytes(f"[aim-fbc {stamp}] {message}\n".encode("utf-8", "replace"))

    def close(self) -> None:
        with self._lock:
            try:
                self._file.close()
            except OSError:
                # Closing at shutdown: nothing useful to do if the flush fails.
                pass


@dataclass
class RunResult:
    returncode: int = 0  # raw: negative means killed by that signal
    signal_received: Optional[int] = None
    timed_out: bool = False
    elapsed_s: float = 0.0
    last_line: str = ""  # last non-empty line of the child's output
    marks: dict[str, float] = field(default_factory=dict)  # first-seen time of each marker, seconds


class SignalLatch:
    """Record the first SIGTERM/SIGINT/SIGHUP for the whole life of ``main()``.

    ``run_supervised`` installs its own forwarding handlers only while a child
    runs. Outside that window (preflight, the gap between restarts, the final
    publish) this latch keeps the signal instead of letting the default action
    kill the process with ``summary.json`` still ``running``. The caller checks
    ``signum`` between steps.
    """

    def __init__(self) -> None:
        self.signum: Optional[int] = None
        self._previous: dict[int, Any] = {}

    def _on_signal(self, signum: int, _frame: object) -> None:
        if self.signum is None:
            self.signum = signum

    def __enter__(self) -> "SignalLatch":
        self._previous = {sig: signal.signal(sig, self._on_signal) for sig in FORWARDED_SIGNALS}
        return self

    def __exit__(self, *_exc: object) -> None:
        for sig, handler in self._previous.items():
            signal.signal(sig, handler)


def _kill_group(pgid: int, sig: int) -> None:
    try:
        os.killpg(pgid, sig)
    except (ProcessLookupError, PermissionError):
        pass  # the group already exited; nothing left to signal


def run_supervised(
    cmd: Sequence[str],
    *,
    cwd: Path,
    env: dict[str, str],
    log: Log,
    time_limit_s: Optional[float] = None,
    grace_s: float = DEFAULT_GRACE_S,
    markers: Optional[dict[str, "re.Pattern[str]"]] = None,
    marks: Optional[dict[str, float]] = None,
    tick: Optional[Callable[[], None]] = None,
    tick_s: float = DEFAULT_TICK_S,
) -> RunResult:
    """Run ``cmd`` to completion under the rules in the module docstring."""
    result = RunResult()
    if marks is not None:
        result.marks = marks  # filled live so a periodic publisher can read it
    state: dict[str, Any] = {"signal": None, "timed_out": False}
    stop = threading.Event()
    timers: list[threading.Timer] = []
    proc_holder: dict[str, subprocess.Popen] = {}

    def escalate() -> None:
        proc = proc_holder.get("proc")
        if proc is not None and proc.poll() is None:
            log.info(f"child still running {grace_s:.0f}s after the stop request; sending SIGKILL")
            _kill_group(proc.pid, signal.SIGKILL)

    def request_stop(sig: int) -> None:
        proc = proc_holder.get("proc")
        if proc is None or proc.poll() is not None:
            return
        _kill_group(proc.pid, sig)
        timer = threading.Timer(grace_s, escalate)
        timer.daemon = True
        timer.start()
        timers.append(timer)

    def on_signal(signum: int, _frame: object) -> None:
        # Record only: no logging (the tee lock is not reentrant and the interrupted code may hold it),
        # no thread or timer creation. The read loop forwards the signal.
        if state["signal"] is None:
            state["signal"] = signum

    def forward_signal() -> None:
        signum = state["signal"]
        if signum is None or state.get("forwarded"):
            return
        state["forwarded"] = True
        log.info(f"received {signal.Signals(signum).name}; forwarding to the run and shutting down")
        request_stop(signal.SIGTERM if signum != signal.SIGINT else signal.SIGINT)

    def on_time_limit() -> None:
        if state["signal"] is None and not state["timed_out"]:
            state["timed_out"] = True
            log.info(f"wall-clock limit of {time_limit_s:.0f}s reached; stopping the run")
            request_stop(signal.SIGTERM)

    previous = {sig: signal.signal(sig, on_signal) for sig in FORWARDED_SIGNALS}
    started = time.monotonic()
    tick_thread: Optional[threading.Thread] = None
    try:
        proc = subprocess.Popen(
            list(cmd),
            cwd=str(cwd),
            env=env,
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
        proc_holder["proc"] = proc
        forward_signal()  # a signal that arrived before the child existed
        if time_limit_s:
            limit_timer = threading.Timer(time_limit_s, on_time_limit)
            limit_timer.daemon = True
            limit_timer.start()
            timers.append(limit_timer)

        def ticker() -> None:
            while not stop.wait(tick_s):
                try:
                    tick()  # type: ignore[misc]
                except Exception as exc:  # noqa: BLE001 - publishing must never kill the run
                    log.info(f"periodic publish failed: {exc}")

        if tick is not None:
            tick_thread = threading.Thread(target=ticker, daemon=True)
            tick_thread.start()

        pending = b""
        tail = b""
        seen: set[str] = set()
        assert proc.stdout is not None
        fd = proc.stdout.fileno()
        while True:
            forward_signal()
            # Poll instead of a blocking read: a descendant that keeps the pipe open after the leader
            # exits must not hold this loop (and the final publish) forever.
            ready, _, _ = select.select([fd], [], [], POLL_S)
            if not ready:
                if proc.poll() is not None:
                    break  # leader gone and nothing left to read; stragglers are killed below
                continue
            chunk = os.read(fd, READ_CHUNK)
            if not chunk:
                break
            log.write_bytes(chunk)
            tail = (tail + chunk)[-4096:]
            if markers:
                pending += chunk
                *lines, pending = re.split(rb"[\r\n]", pending)
                pending = pending[-4096:]
                for raw in lines:
                    text = raw.decode("utf-8", "replace")
                    for name, pattern in markers.items():
                        if name not in seen and pattern.search(text):
                            seen.add(name)
                            result.marks[name] = time.monotonic() - started
        proc.wait()
        lines_tail = [x.strip() for x in re.split(rb"[\r\n]", tail) if x.strip()]
        result.last_line = lines_tail[-1].decode("utf-8", "replace")[:300] if lines_tail else ""
        result.returncode = proc.returncode
        _kill_group(proc.pid, signal.SIGKILL)  # reap stray children of the run
    finally:
        stop.set()
        if tick_thread is not None:
            tick_thread.join(timeout=TICK_JOIN_S)  # a running publish must finish before the caller publishes
        for timer in timers:
            timer.cancel()
        for sig, handler in previous.items():
            signal.signal(sig, handler)
    result.elapsed_s = time.monotonic() - started
    result.signal_received = state["signal"]
    result.timed_out = bool(state["timed_out"])
    return result


def exit_status(result: RunResult) -> int:
    """Shell-style status: 128+signal when the wrapper was signalled, 124 on the time limit."""
    if result.signal_received is not None:
        return 128 + int(result.signal_received)
    if result.timed_out:
        return 124
    if result.returncode < 0:
        return 128 + (-result.returncode)
    return result.returncode
