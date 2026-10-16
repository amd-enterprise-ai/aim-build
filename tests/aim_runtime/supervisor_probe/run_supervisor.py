# Copyright © Advanced Micro Devices, Inc., or its affiliates.
#
# SPDX-License-Identifier: MIT

"""Run the real ``AIMRuntime._serve_supervised`` against ``fake_engine.py``.

Profile selection and hardware detection are bypassed, so this runs anywhere;
everything else — process group, signal handlers, readiness probe, exit-code
translation — is the production code path. Never exits normally: it terminates
via the supervisor's own ``sys.exit``.

Usage (also handy by hand, outside pytest):
    python run_supervisor.py <engine-status> <port> [--crash-exit-code CODE]
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "src"))

from aim_runtime.aim_runtime import AIMRuntime  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("status", type=int, help="HTTP status the fake engine returns")
    parser.add_argument("port", type=int)
    parser.add_argument("--crash-exit-code", type=int, default=None)
    args = parser.parse_args()

    runtime = AIMRuntime.__new__(AIMRuntime)
    runtime.config = SimpleNamespace(port=args.port)

    engine = Path(__file__).with_name("fake_engine.py")
    command = [sys.executable, str(engine), str(args.status), str(args.port)]
    if args.crash_exit_code is not None:
        command += ["--exit-code", str(args.crash_exit_code)]
    runtime._serve_supervised(command, watcher=None)


if __name__ == "__main__":
    main()
