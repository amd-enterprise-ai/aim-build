# Copyright © Advanced Micro Devices, Inc., or its affiliates.
#
# SPDX-License-Identifier: MIT

"""Supervised-run behaviour of fbc_runner: signal handling, stray descendants, the periodic publisher."""

from __future__ import annotations

import os
import signal
import sys
import threading
import time
from pathlib import Path

import fbc_runner as runner
import pytest


@pytest.fixture()
def watchdog():
    """Fail instead of hanging forever if the code under test deadlocks (signal handlers run in the main thread)."""

    def _expired(_signum, _frame):
        raise TimeoutError("run_supervised did not return: deadlock or hang")

    previous = signal.signal(signal.SIGALRM, _expired)
    signal.setitimer(signal.ITIMER_REAL, 15)
    yield
    signal.setitimer(signal.ITIMER_REAL, 0)
    signal.signal(signal.SIGALRM, previous)


def _run(tmp_path: Path, cmd: list[str], log: runner.Log | None = None, **kw) -> tuple[runner.RunResult, Path]:
    log_path = tmp_path / "run.log"
    log = log or runner.Log(log_path)
    try:
        result = runner.run_supervised(cmd, cwd=tmp_path, env=dict(os.environ), log=log, **kw)
    finally:
        log.close()
    return result, log_path


def test_signal_arriving_during_a_log_write_does_not_deadlock(tmp_path: Path, watchdog) -> None:
    class InterruptedWrite(runner.Log):
        fired = False

        def write_bytes(self, data: bytes) -> None:
            with self._lock:  # the lock a handler calling log.info() would need
                if not self.fired and b"ready" in data:
                    self.fired = True
                    os.kill(os.getpid(), signal.SIGTERM)
                    time.sleep(0.05)  # let the handler run while this write holds the lock
                self._file.write(data)

    log = InterruptedWrite(tmp_path / "run.log")
    child = [sys.executable, "-c", "import time; print('ready', flush=True); time.sleep(60)"]
    result, log_path = _run(tmp_path, child, log=log, grace_s=2)
    assert result.signal_received == signal.SIGTERM
    assert runner.exit_status(result) == 143
    assert "received SIGTERM" in log_path.read_text()  # logged by the read loop, not the handler


def _alive(pid: int) -> bool:
    try:
        state = Path(f"/proc/{pid}/stat").read_text().rsplit(")", 1)[1].split()[0]
    except OSError:
        return False
    return state != "Z"


def test_descendant_holding_stdout_does_not_block_after_the_leader_exits(tmp_path: Path, watchdog) -> None:
    child = ["sh", "-c", "sleep 300 & echo $!; exit 0"]
    started = time.monotonic()
    result, log_path = _run(tmp_path, child)
    assert time.monotonic() - started < 10
    assert result.returncode == 0
    pid = int(log_path.read_text().split()[0])
    for _ in range(50):
        if not _alive(pid):
            break
        time.sleep(0.1)
    assert not _alive(pid), "the remaining group member was not killed"


def test_periodic_publish_is_finished_before_run_supervised_returns(tmp_path: Path, watchdog) -> None:
    running = 0
    lock = threading.Lock()

    def tick() -> None:
        nonlocal running
        with lock:
            running += 1
        time.sleep(0.4)
        with lock:
            running -= 1

    child = [sys.executable, "-c", "import time; time.sleep(0.2)"]
    _run(tmp_path, child, tick=tick, tick_s=0.01)
    with lock:
        assert running == 0
