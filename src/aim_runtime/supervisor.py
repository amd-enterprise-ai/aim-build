# Copyright © Advanced Micro Devices, Inc., or its affiliates.
#
# SPDX-License-Identifier: MIT

"""Process lifecycle supervision for the AIM runtime."""

from __future__ import annotations

import logging
import os
import signal
import subprocess
import threading
from collections.abc import Callable, Sequence
from typing import Any

logger = logging.getLogger(__name__)

_SHUTDOWN_GRACE_SECONDS = 20


class _SupervisorTermination(Exception):
    """Interrupt the blocking engine wait after forwarding shutdown."""


class EngineSupervisor:
    """Launch and supervise one engine process group."""

    def __init__(
        self,
        command_list: Sequence[str],
        *,
        watcher: Any = None,
        on_ready: Callable[[Any], None] | None = None,
        reap_orphans: Callable[..., None] | None = None,
    ) -> None:
        self.command_list = command_list
        self.watcher = watcher
        self.on_ready = on_ready
        self.reap_orphans = reap_orphans or self._reap_orphans
        self.process: subprocess.Popen[bytes] | None = None
        self.pgid: int | None = None
        self.terminating = threading.Event()
        self.interrupt_wait = threading.Event()

    def run(self) -> int:
        """Run the engine until it exits and return its subprocess status."""
        self._start_engine()
        self._install_signal_handlers()
        self._start_watcher()
        returncode = self._wait_for_engine_exit()
        self._kill_process_group(signal.SIGKILL, "after engine exit")
        return returncode

    def _start_engine(self) -> None:
        logger.info("Starting engine under runtime supervision...")
        # Keep the child's fds inherited so engine logs stream byte-for-byte and
        # cannot block on a full pipe.
        self.process = subprocess.Popen(self.command_list, env=os.environ.copy(), process_group=0)
        process = self.process
        try:
            self.pgid = os.getpgid(process.pid)
        except ProcessLookupError:
            # The engine may exit before its process group can be queried.
            self.pgid = process.pid

    def _install_signal_handlers(self) -> None:
        signal.signal(signal.SIGTERM, self._forward_signal)
        signal.signal(signal.SIGINT, self._forward_signal)
        signal.signal(signal.SIGCHLD, self._reap_children)

    def _start_watcher(self) -> None:
        watcher = self.watcher
        if watcher is None:
            return
        assert self.process is not None
        process = self.process

        def watch() -> None:
            if self.on_ready is not None:
                self.on_ready(process)
            watcher.run_forever(lambda: process.poll() is not None)

        threading.Thread(target=watch, daemon=True).start()

    def _forward_signal(self, signum: int, _frame: Any) -> None:
        assert self.process is not None
        assert self.pgid is not None
        logger.info("Forwarding signal %s to engine process group (pgid %s)", signum, self.pgid)
        self.terminating.set()
        try:
            os.killpg(self.pgid, signum)
        except ProcessLookupError:
            # The engine group already exited; there is nothing to forward.
            pass
        if self.interrupt_wait.is_set():
            raise _SupervisorTermination

    def _reap_children(self, _signum: int, _frame: Any) -> None:
        assert self.process is not None
        self.reap_orphans(skip_pid=self.process.pid)

    def _wait_for_engine_exit(self) -> int:
        assert self.process is not None
        assert self.pgid is not None
        return self.wait_for_engine_exit(
            self.process,
            self.terminating,
            self.pgid,
            self.interrupt_wait,
        )

    @staticmethod
    def wait_for_engine_exit(
        proc: Any,
        terminating: threading.Event,
        pgid: int,
        interrupt_wait: threading.Event | None = None,
    ) -> int:
        """Wait for an engine, bounding the drain after termination."""
        if not terminating.is_set():
            if interrupt_wait is None:
                interrupt_wait = threading.Event()
            interrupt_wait.set()
            try:
                return proc.wait()
            except _SupervisorTermination:
                pass
            finally:
                interrupt_wait.clear()
        try:
            return proc.wait(timeout=_SHUTDOWN_GRACE_SECONDS)
        except subprocess.TimeoutExpired:
            logger.warning(
                "Engine did not exit within %ss of termination signal; escalating to SIGKILL (pid %s).",
                _SHUTDOWN_GRACE_SECONDS,
                proc.pid,
            )
            try:
                os.killpg(pgid, signal.SIGKILL)
            except ProcessLookupError:
                # The engine group already exited during the grace period.
                pass
            return proc.wait()

    def _kill_process_group(self, signum: int, reason: str) -> None:
        assert self.pgid is not None
        try:
            os.killpg(self.pgid, signum)
        except ProcessLookupError:
            logger.debug("Engine process group already exited %s.", reason)

    @staticmethod
    def _reap_orphans(*, skip_pid: int) -> None:
        """Reap exited children without stealing the supervised engine status."""
        while True:
            try:
                info = os.waitid(os.P_ALL, 0, os.WEXITED | os.WNOHANG | os.WNOWAIT)
            except ChildProcessError:
                return
            if info is None or info.si_pid == skip_pid:
                return
            try:
                os.waitpid(info.si_pid, 0)
            except ChildProcessError:
                # Another reaper won the race after waitid() observed the child.
                pass
