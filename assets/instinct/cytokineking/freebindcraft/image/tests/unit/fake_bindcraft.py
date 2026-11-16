# Copyright © Advanced Micro Devices, Inc., or its affiliates.
#
# SPDX-License-Identifier: MIT

"""A stand-in for upstream bindcraft.py used to test the wrapper on a CPU.

It reads the same ``--settings/--advanced/--filters`` arguments, writes the same
directory layout and CSV files (with upstream's column names for the fields the
wrapper reads) and behaves according to ``FAKE_MODE``:

    ok       one trajectory, one accepted design, exits 0
    slow     one trajectory, then sleeps (starts a grandchild ``sleep`` too)
    crash    prints and exits 3
    nogpu    prints upstream's no-GPU message and exits 0
    silent   exits 0 without producing anything
    flaky    exits 1 (like an AF2 NaN crash) until FAKE_COUNTER holds FAKE_CRASHES attempts, then behaves as ok
    flaky_progress  like flaky, but each crashing attempt first bumps a failure counter in failure_csv.csv (one row)
"""

import csv
import json
import os
import subprocess
import sys
import time
from pathlib import Path


def main() -> int:
    argv = sys.argv[1:]
    settings = json.loads(Path(argv[argv.index("--settings") + 1]).read_text())
    advanced = json.loads(Path(argv[argv.index("--advanced") + 1]).read_text())
    design = Path(settings["design_path"])
    mode = os.environ.get("FAKE_MODE", "ok")
    record = os.environ.get("FAKE_RECORD")
    if record:
        Path(record).write_text(json.dumps({"argv": argv, "advanced": advanced, "settings": settings}))

    if mode == "nogpu":
        print("No GPU device found, terminating.")
        return 0
    if mode == "silent":
        return 0
    if mode in ("flaky", "flaky_progress"):
        counter = Path(os.environ["FAKE_COUNTER"])
        done = int(counter.read_text()) if counter.exists() else 0
        if done < int(os.environ.get("FAKE_CRASHES", "1")):
            counter.write_text(str(done + 1))
            if mode == "flaky_progress":
                design.mkdir(parents=True, exist_ok=True)
                # Upstream's format: a header and ONE row of counters, rewritten in place (the row count stays 1).
                with (design / "failure_csv.csv").open("w", newline="") as handle:
                    writer = csv.writer(handle)
                    writer.writerow(["Trajectory_Clashes", "Trajectory_Contacts", "pLDDT"])
                    writer.writerow([done + 1, 0, 0])
            print("ValueError: cannot convert float NaN to integer", flush=True)
            return 1
        mode = "ok"
    if mode == "crash":
        print("boom: simulated failure", flush=True)
        return 3

    for sub in ("Accepted/Ranked", "Trajectory/Relaxed"):
        (design / sub).mkdir(parents=True, exist_ok=True)
    print("Starting trajectory: demo_l35_s1", flush=True)
    time.sleep(0.2)
    print("Starting trajectory took: 0 hours, 1 minutes, 5 seconds", flush=True)
    (design / "Trajectory/Relaxed/demo_l35_s1.pdb").write_text("ATOM\n")
    with (design / "trajectory_stats.csv").open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["Design", "Length", "TrajectoryTime"])
        writer.writerow(["demo_l35_s1", 35, "0 hours, 1 minutes, 5 seconds"])
    with (design / "mpnn_design_stats.csv").open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["Design", "Sequence"])
        writer.writerow(["demo_l35_s1_mpnn1", "MKTAYIAKQR"])
    with (design / "failure_csv.csv").open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["Trajectory_Clashes", "Trajectory_Contacts", "pLDDT"])
        writer.writerow([2, 1, 4])

    if mode == "slow":
        grandchild = subprocess.Popen(["sleep", "300"])
        ready = os.environ.get("FAKE_READY")
        if ready:
            Path(ready).write_text(str(grandchild.pid))
        time.sleep(300)
        return 0

    (design / "Accepted/demo_l35_s1_mpnn1_model1.pdb").write_text("ATOM\n")
    (design / "Accepted/Ranked/1_demo_l35_s1_mpnn1_model1.pdb").write_text("ATOM\n")
    print("Target number of designs reached!", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
