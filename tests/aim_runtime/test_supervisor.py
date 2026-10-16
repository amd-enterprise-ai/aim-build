# Copyright © Advanced Micro Devices, Inc., or its affiliates.
#
# SPDX-License-Identifier: MIT

"""Unit tests for the AIM engine supervisor."""

import signal
import subprocess
import threading
from unittest.mock import Mock, call, patch

import pytest

from aim_runtime.supervisor import EngineSupervisor


class TestEngineSupervisor:
    def _run_with_returncode(self, returncode: int) -> int:
        proc = Mock()
        proc.pid = 1234
        proc.wait.return_value = returncode
        supervisor = EngineSupervisor(["python", "-m", "vllm"])

        with (
            patch("subprocess.Popen", return_value=proc),
            patch("os.getpgid", return_value=proc.pid),
            patch("signal.signal"),
            patch("os.killpg"),
        ):
            return supervisor.run()

    def test_clean_exit_code_passthrough(self):
        assert self._run_with_returncode(0) == 0

    def test_nonzero_exit_code_passthrough(self):
        assert self._run_with_returncode(1) == 1

    def test_signal_exit_code_passthrough(self):
        assert self._run_with_returncode(-15) == -15

    def test_immediate_engine_exit_during_pgid_lookup(self):
        proc = Mock()
        proc.pid = 1234
        proc.wait.return_value = 17
        supervisor = EngineSupervisor(["python", "-m", "vllm"])

        with (
            patch("subprocess.Popen", return_value=proc),
            patch("os.getpgid", side_effect=ProcessLookupError),
            patch("signal.signal"),
            patch("os.killpg"),
        ):
            assert supervisor.run() == 17

    def test_engine_exit_kills_remaining_process_group(self):
        proc = Mock()
        proc.pid = 1234
        proc.wait.return_value = 1
        supervisor = EngineSupervisor(["python", "-m", "vllm"])

        with (
            patch("subprocess.Popen", return_value=proc),
            patch("os.getpgid", return_value=5678),
            patch("signal.signal"),
            patch("os.killpg") as killpg,
        ):
            assert supervisor.run() == 1

        killpg.assert_called_once_with(5678, signal.SIGKILL)

    def test_shutdown_signals_forward_to_engine_process_group(self):
        proc = Mock()
        proc.pid = 1234
        proc.wait.return_value = 0
        handlers = {}

        def register_handler(signum, handler):
            handlers[signum] = handler

        supervisor = EngineSupervisor(["python", "-m", "vllm"])
        with (
            patch("subprocess.Popen", return_value=proc),
            patch("os.getpgid", return_value=5678),
            patch("signal.signal", side_effect=register_handler),
            patch("os.killpg") as killpg,
            patch.object(supervisor, "_reap_children") as reap,
        ):
            assert supervisor.run() == 0
            handlers[signal.SIGTERM](signal.SIGTERM, None)
            handlers[signal.SIGINT](signal.SIGINT, None)
            handlers[signal.SIGCHLD](signal.SIGCHLD, None)

        assert killpg.call_args_list == [
            call(5678, signal.SIGKILL),
            call(5678, signal.SIGTERM),
            call(5678, signal.SIGINT),
        ]
        reap.assert_called_once()
        proc.send_signal.assert_not_called()


class TestEngineSupervisorWait:
    def test_normal_exit_returns_code(self):
        proc = Mock()
        proc.wait.return_value = 0
        assert EngineSupervisor.wait_for_engine_exit(proc, threading.Event(), pgid=1234) == 0

    def test_terminating_wait_returns_clean_exit(self):
        terminating = threading.Event()
        terminating.set()
        proc = Mock()
        proc.wait.return_value = -15

        assert EngineSupervisor.wait_for_engine_exit(proc, terminating, pgid=1234) == -15
        proc.wait.assert_called_once()

    def test_hung_engine_escalates_to_sigkill(self):
        terminating = threading.Event()
        terminating.set()
        proc = Mock()
        proc.pid = 4321
        proc.wait.side_effect = [subprocess.TimeoutExpired(cmd="vllm", timeout=20), -9]

        with patch("os.killpg") as killpg:
            assert EngineSupervisor.wait_for_engine_exit(proc, terminating, pgid=4321) == -9

        killpg.assert_called_once_with(4321, signal.SIGKILL)
        assert proc.wait.call_count == 2


class TestReapOrphans:
    @staticmethod
    def _run_isolated(test):
        import functools
        import os
        import traceback

        @functools.wraps(test)
        def wrapper(self):
            child_pid = os.fork()
            if child_pid == 0:
                try:
                    test(self)
                except Exception:
                    traceback.print_exc()
                    os._exit(1)
                os._exit(0)

            _, status = os.waitpid(child_pid, 0)
            assert os.WIFEXITED(status)
            assert os.WEXITSTATUS(status) == 0

        return wrapper

    @_run_isolated
    def test_reaps_an_exited_child_not_matching_skip_pid(self):
        import os

        child_pid = os.fork()
        if child_pid == 0:
            os._exit(0)
        os.waitid(os.P_PID, child_pid, os.WEXITED | os.WNOWAIT)
        EngineSupervisor._reap_orphans(skip_pid=-1)
        with pytest.raises(ChildProcessError):
            os.waitpid(child_pid, 0)

    @_run_isolated
    def test_leaves_skip_pid_for_the_caller_to_reap(self):
        import os

        child_pid = os.fork()
        if child_pid == 0:
            os._exit(0)
        os.waitid(os.P_PID, child_pid, os.WEXITED | os.WNOWAIT)
        EngineSupervisor._reap_orphans(skip_pid=child_pid)
        pid, _ = os.waitpid(child_pid, 0)
        assert pid == child_pid

    @_run_isolated
    def test_missing_children_is_a_no_op(self):
        with patch("os.waitid", side_effect=ChildProcessError):
            EngineSupervisor._reap_orphans(skip_pid=1)
