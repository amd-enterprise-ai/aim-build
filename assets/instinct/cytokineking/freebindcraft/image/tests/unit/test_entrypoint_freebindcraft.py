# Copyright © Advanced Micro Devices, Inc., or its affiliates.
#
# SPDX-License-Identifier: MIT

"""The contract of the entrypoint, end to end, against a fake upstream CLI.

Covers: inputs/outputs, settings written into the copied advanced JSON, log
always present, unknown settings rejected, exit codes (0, 1, 2, 3, 124, 143),
partial results when capped or interrupted, live log streaming and signal
forwarding to the whole process group.
"""

from __future__ import annotations

import json
import os
import shutil
import signal
import subprocess
import sys
import time
from pathlib import Path

import pytest

HERE = Path(__file__).resolve().parent
ENTRYPOINT = HERE.parents[1] / "src" / "entrypoint.py"
FAKE = HERE / "fake_bindcraft.py"
ADVANCED_TEMPLATE = {
    "max_trajectories": False,
    "af_params_dir": "",
    "enable_mpnn": True,
    "save_design_animations": True,
}


@pytest.fixture()
def home(tmp_path: Path) -> Path:
    """A BINDCRAFT_HOME that looks like the upstream checkout."""
    h = tmp_path / "bc"
    (h / "settings_advanced").mkdir(parents=True)
    (h / "settings_filters").mkdir()
    shutil.copy(FAKE, h / "bindcraft.py")
    (h / "settings_advanced/default_4stage_multimer.json").write_text(json.dumps(ADVANCED_TEMPLATE))
    (h / "settings_filters/default_filters.json").write_text("{}")
    return h


@pytest.fixture()
def dirs(tmp_path: Path, target_pdb: Path) -> tuple[Path, Path]:
    inp, out = tmp_path / "in", tmp_path / "out"
    inp.mkdir()
    shutil.copy(target_pdb, inp / "target.pdb")
    return inp, out


# Variables the interpreter itself needs to start. The entrypoint hands this
# environment to the fake bindcraft.py child (started with sys.executable); on a
# Python whose libpython is a shared library found via LD_LIBRARY_PATH (e.g. the
# CI runner's toolcache Python) a bare environment makes the child exit 127.
_INTERPRETER_ENV = ("LD_LIBRARY_PATH", "LD_PRELOAD", "PYTHONHOME", "PYTHONPATH", "PYTHONUSERBASE", "VIRTUAL_ENV")


def _env(home: Path, dirs: tuple[Path, Path], **extra: str) -> dict[str, str]:
    params = home / "params"  # AF2 weights are mounted, not baked: BINDCRAFT_HOME/params stands in for the mount
    params.mkdir(exist_ok=True)
    (params / "params_model_1_multimer_v3.npz").write_text("")
    env = {
        **{k: os.environ[k] for k in _INTERPRETER_ENV if k in os.environ},
        "AIM_INPUT_PATH": str(dirs[0]),
        "AIM_OUTPUT_PATH": str(dirs[1]),
        "AIM_ACCELERATOR_COUNT": "1",
        "BINDCRAFT_HOME": str(home),
        "PATH": os.environ.get("PATH", ""),
    }
    env.update(extra)
    return env


def _summary(out: Path) -> dict:
    return json.loads((out / "summary.json").read_text())


def test_successful_run(entrypoint, home: Path, dirs, tmp_path: Path) -> None:
    record = tmp_path / "record.json"
    inp, out = dirs
    (inp / "settings.json").write_text(
        json.dumps({"min_length": 31, "max_length": 40, "num_designs": 1, "binder_name": "demo"})
    )
    code = entrypoint.main(
        [], _env(home, dirs, FAKE_RECORD=str(record), AIM_FBC_MAX_TRAJECTORIES="1", AIM_FBC_HOTSPOTS="A20")
    )
    assert code == 0
    s = _summary(out)
    assert s["status"] == "completed" and s["finished"] is True and s["exit_code"] == 0
    assert (s["trajectories"], s["accepted_designs"], s["ranked_designs"]) == (1, 1, 1)
    assert s["per_trajectory_times_s"] == [65] and s["time_to_first_trajectory_s"] is not None
    assert (out / "Accepted/Ranked").is_dir() and any((out / "Accepted/Ranked").glob("*.pdb"))
    assert (
        (out / "trajectory_stats.csv").is_file()
        and (out / "mpnn_design_stats.csv").is_file()
        and (out / "failure_csv.csv").is_file()
    )
    assert "Starting trajectory took" in (out / "bindcraft.log").read_text()
    assert not [p for p in out.rglob("*") if p.name.endswith(".tmp")]
    # The cap and the settings reach the upstream CLI through the copied JSON; the shipped template is untouched.
    seen = json.loads(record.read_text())
    assert seen["advanced"]["max_trajectories"] == 1
    # The image ships no ffmpeg, so the optional design animations are off, whatever upstream's template says.
    assert seen["advanced"]["save_design_animations"] is False
    assert seen["settings"]["lengths"] == [31, 40] and seen["settings"]["binder_name"] == "demo"
    assert seen["settings"]["target_hotspot_residues"] == "A20" and seen["settings"]["chains"] == "A"
    assert seen["settings"]["number_of_final_designs"] == 1
    assert "--no-pyrosetta" in seen["argv"]
    assert json.loads((home / "settings_advanced/default_4stage_multimer.json").read_text()) == ADVANCED_TEMPLATE
    assert (inp / "settings.json").exists() and sorted(p.name for p in inp.iterdir()) == [
        "settings.json",
        "target.pdb",
    ]  # input untouched


def test_uncapped_advanced_json_has_no_trajectory_limit(entrypoint, home: Path, dirs, tmp_path: Path) -> None:
    record = tmp_path / "r.json"
    assert entrypoint.main([], _env(home, dirs, FAKE_RECORD=str(record), AIM_FBC_NUM_DESIGNS="1")) == 0
    assert json.loads(record.read_text())["advanced"]["max_trajectories"] is False


def test_cap_reached_with_fewer_designs_exits_zero_with_status(entrypoint, home: Path, dirs) -> None:
    code = entrypoint.main([], _env(home, dirs, AIM_FBC_NUM_DESIGNS="5", AIM_FBC_MAX_TRAJECTORIES="1"))
    assert code == 0
    s = _summary(dirs[1])
    assert s["status"] == "max_trajectories_reached" and s["accepted_designs"] == 1 and s["requested_designs"] == 5


def test_fewer_designs_without_a_cap_is_not_success(entrypoint, home: Path, dirs) -> None:
    code = entrypoint.main([], _env(home, dirs, AIM_FBC_NUM_DESIGNS="5"))
    assert code == 4 and _summary(dirs[1])["status"] == "stopped_early"


def test_broken_input_fails_non_zero_with_a_log(entrypoint, home: Path, dirs) -> None:
    (dirs[0] / "target.pdb").write_text("not a pdb")
    code = entrypoint.main([], _env(home, dirs))
    assert code == 2
    out = dirs[1]
    assert (out / "bindcraft.log").stat().st_size > 0 and "no protein ATOM" in (out / "bindcraft.log").read_text()
    s = _summary(out)
    assert s["status"] == "failed" and s["exit_code"] == 2 and s["finished"] is True


def test_missing_target_and_missing_input_dir(entrypoint, home: Path, dirs) -> None:
    (dirs[0] / "target.pdb").unlink()
    assert entrypoint.main([], _env(home, dirs)) == 2
    env = _env(home, dirs)
    env["AIM_INPUT_PATH"] = str(dirs[0] / "absent")
    assert entrypoint.main([], env) == 2
    assert "not a directory" in (dirs[1] / "bindcraft.log").read_text()


def test_unset_paths(entrypoint, home: Path, dirs, capsys) -> None:
    env = _env(home, dirs)
    del env["AIM_OUTPUT_PATH"]
    assert entrypoint.main([], env) == 2
    assert "AIM_OUTPUT_PATH" in capsys.readouterr().err
    env = _env(home, dirs)
    del env["AIM_INPUT_PATH"]
    assert entrypoint.main([], env) == 2
    assert "AIM_INPUT_PATH is not set" in (dirs[1] / "bindcraft.log").read_text()


def test_unknown_setting_is_an_error_with_a_log(entrypoint, home: Path, dirs) -> None:
    assert entrypoint.main([], _env(home, dirs, AIM_FBC_NUM_DESGINS="1")) == 2
    assert "AIM_FBC_NUM_DESGINS" in (dirs[1] / "bindcraft.log").read_text()
    assert not (dirs[1] / ".work" / "design").exists(), "nothing was started"


def test_unexpected_arguments_are_rejected(entrypoint, home: Path, dirs, capsys) -> None:
    assert entrypoint.main(["serve"], _env(home, dirs)) == 2
    assert entrypoint.main(["--help"], _env(home, dirs)) == 0
    assert "AIM_INPUT_PATH" in capsys.readouterr().out


def test_too_few_accelerators_exits_3(entrypoint, home: Path, dirs, tmp_path: Path) -> None:
    env = _env(home, dirs)
    del env["AIM_ACCELERATOR_COUNT"]
    (tmp_path / "devroot" / "dri").mkdir(parents=True)  # no kfd, no render nodes
    assert entrypoint.main([], env, dev_root=tmp_path / "devroot") == 3
    assert "needs 1 AMD GPU" in (dirs[1] / "bindcraft.log").read_text()


def test_upstream_failure_propagates_and_log_keeps_its_output(entrypoint, home: Path, dirs) -> None:
    code = entrypoint.main([], _env(home, dirs, FAKE_MODE="crash", AIM_FBC_MAX_RESTARTS="0"))
    assert code == 3
    assert "boom: simulated failure" in (dirs[1] / "bindcraft.log").read_text()
    assert _summary(dirs[1])["status"] == "failed"


def test_upstream_no_gpu_exit_zero_is_a_failure(entrypoint, home: Path, dirs) -> None:
    assert entrypoint.main([], _env(home, dirs, FAKE_MODE="nogpu")) == 1
    assert _summary(dirs[1])["status"] == "failed"


def test_exit_zero_without_any_trajectory_is_a_failure(entrypoint, home: Path, dirs) -> None:
    assert entrypoint.main([], _env(home, dirs, FAKE_MODE="silent")) == 1


def test_resume_against_other_input_is_refused(entrypoint, home: Path, dirs) -> None:
    assert entrypoint.main([], _env(home, dirs, AIM_FBC_NUM_DESIGNS="1")) == 0
    assert entrypoint.main([], _env(home, dirs, AIM_FBC_NUM_DESIGNS="1", AIM_FBC_HOTSPOTS="A20")) == 2
    assert entrypoint.main([], _env(home, dirs, AIM_FBC_NUM_DESIGNS="1")) == 0  # same input resumes
    assert _summary(dirs[1])["resumed"] is True


def test_time_limit_publishes_partial_results_and_exits_124(entrypoint, home: Path, dirs, tmp_path: Path) -> None:
    ready = tmp_path / "ready"
    code = entrypoint.main(
        [], _env(home, dirs, FAKE_MODE="slow", FAKE_READY=str(ready), AIM_FBC_TIME_LIMIT="2"), grace_s=2.0
    )
    assert code == 124
    s = _summary(dirs[1])
    assert s["status"] == "time_limit" and s["finished"] is True
    assert s["trajectories"] == 1, "the trajectory finished before the cap is kept"
    assert (dirs[1] / "trajectory_stats.csv").is_file()
    assert _gone(int(ready.read_text())), "the whole process group was stopped"


def _gone(pid: int, timeout: float = 5.0) -> bool:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        try:
            os.kill(pid, 0)
        except ProcessLookupError:
            return True
        except PermissionError:
            return False
        # A reaped-by-init zombie also counts as gone.
        try:
            if Path(f"/proc/{pid}/stat").read_text().split()[2] == "Z":
                return True
        except OSError:
            return True
        time.sleep(0.1)
    return False


def _spawn(home: Path, dirs, tmp_path: Path, **extra: str) -> subprocess.Popen:
    env = _env(home, dirs, FAKE_MODE="slow", FAKE_READY=str(tmp_path / "ready"), **extra)
    return subprocess.Popen(
        [sys.executable, str(ENTRYPOINT)], env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT
    )


def _wait_for(path: Path, timeout: float = 20.0) -> None:
    deadline = time.monotonic() + timeout
    while not path.exists():
        assert time.monotonic() < deadline, f"{path} did not appear"
        time.sleep(0.1)


def test_sigterm_exits_143_forwards_to_the_group_and_leaves_usable_output(home: Path, dirs, tmp_path: Path) -> None:
    proc = _spawn(home, dirs, tmp_path)
    try:
        _wait_for(tmp_path / "ready")
        proc.send_signal(signal.SIGTERM)
        assert proc.wait(timeout=30) == 143
    finally:
        if proc.poll() is None:
            proc.kill()
    out = dirs[1]
    s = _summary(out)
    assert s["status"] == "interrupted" and s["exit_code"] == 143 and s["finished"] is True
    assert s["trajectories"] == 1 and (out / "trajectory_stats.csv").is_file()
    assert s["restarts"] == 0, "our own SIGTERM is never retried"
    assert "received SIGTERM" in (out / "bindcraft.log").read_text()
    assert _gone(int((tmp_path / "ready").read_text())), "grandchild of the run was stopped with the group"


def test_log_is_streamed_while_the_run_is_in_progress(home: Path, dirs, tmp_path: Path) -> None:
    proc = _spawn(home, dirs, tmp_path)
    try:
        _wait_for(tmp_path / "ready")  # the fake is now sleeping for minutes
        log = (dirs[1] / "bindcraft.log").read_text()
        assert "Starting trajectory took" in log, "output must reach the log before the run ends"
        assert _summary(dirs[1])["status"] == "running" and _summary(dirs[1])["finished"] is False
    finally:
        proc.send_signal(signal.SIGTERM)
        proc.wait(timeout=30)
    streamed = proc.stdout.read().decode() if proc.stdout else ""
    assert "Starting trajectory took" in streamed, "and to stdout"


def test_sigint_exits_130(home: Path, dirs, tmp_path: Path) -> None:
    proc = _spawn(home, dirs, tmp_path)
    try:
        _wait_for(tmp_path / "ready")
        proc.send_signal(signal.SIGINT)
        assert proc.wait(timeout=30) == 130
    finally:
        if proc.poll() is None:
            proc.kill()


# --- restart of a crashing upstream process ---------------------------------


def test_crash_then_success_restarts_and_records_the_cause(entrypoint, home: Path, dirs, tmp_path: Path) -> None:
    env = _env(
        home, dirs, FAKE_MODE="flaky", FAKE_COUNTER=str(tmp_path / "n"), FAKE_CRASHES="2", AIM_FBC_NUM_DESIGNS="1"
    )
    assert entrypoint.main([], env) == 0
    s = _summary(dirs[1])
    assert s["status"] == "completed" and s["exit_code"] == 0
    assert s["restarts"] == 2 and len(s["crash_causes"]) == 2
    assert s["crash_causes"][0]["exit_code"] == 1 and "NaN" in s["crash_causes"][0]["last_line"]
    assert [c["attempt"] for c in s["crash_causes"]] == [1, 2]
    log = (dirs[1] / "bindcraft.log").read_text()
    assert "restarting bindcraft.py (restart 2 of at most 20)" in log and "crashed (attempt 1, exit code 1" in log


def test_restarts_exhausted_exits_1_and_keeps_partial_results(entrypoint, home: Path, dirs, tmp_path: Path) -> None:
    env = _env(
        home,
        dirs,
        FAKE_MODE="flaky_progress",
        FAKE_COUNTER=str(tmp_path / "n"),
        FAKE_CRASHES="99",
        AIM_FBC_MAX_RESTARTS="2",
    )
    assert entrypoint.main([], env) == 1
    s = _summary(dirs[1])
    assert s["status"] == "failed" and s["exit_code"] == 1 and s["finished"] is True
    assert s["restarts"] == 2 and len(s["crash_causes"]) == 3
    assert "exhausted" in " ".join(s["notes"])
    assert (dirs[1] / "failure_csv.csv").is_file(), "partial results kept"


def test_repeated_crash_without_progress_gives_up_after_three(entrypoint, home: Path, dirs, tmp_path: Path) -> None:
    env = _env(home, dirs, FAKE_MODE="flaky", FAKE_COUNTER=str(tmp_path / "n"), FAKE_CRASHES="99")
    assert entrypoint.main([], env) == 1
    s = _summary(dirs[1])
    assert s["status"] == "failed" and s["restarts"] == 2 and len(s["crash_causes"]) == 3
    assert (tmp_path / "n").read_text() == "3"
    assert "without any new trajectory result" in " ".join(s["notes"])


def test_progressing_crashes_are_not_cut_short_by_the_no_progress_rule(
    entrypoint, home: Path, dirs, tmp_path: Path
) -> None:
    env = _env(
        home,
        dirs,
        FAKE_MODE="flaky_progress",
        FAKE_COUNTER=str(tmp_path / "n"),
        FAKE_CRASHES="5",
        AIM_FBC_NUM_DESIGNS="1",
    )
    assert entrypoint.main([], env) == 0
    assert _summary(dirs[1])["restarts"] == 5


def test_progress_count_follows_the_failure_counters_of_the_one_row_csv(entrypoint, tmp_path: Path) -> None:
    design = tmp_path / "design"
    design.mkdir()
    failure = design / "failure_csv.csv"
    failure.write_text("Trajectory_Clashes,Trajectory_Contacts,pLDDT\n0,0,0\n")
    before = entrypoint.progress_count(design)
    failure.write_text("Trajectory_Clashes,Trajectory_Contacts,pLDDT\n1,0,0\n")  # still one row, one more failure
    assert entrypoint.progress_count(design) == before + 1
    failure.write_text("Trajectory_Clashes,Trajectory_Contacts,pLDDT\n1,2,x\n")  # a malformed cell is ignored
    assert entrypoint.progress_count(design) == before + 3
    assert entrypoint.progress_count(tmp_path / "missing") == 0


def test_zero_max_restarts_disables_restarting(entrypoint, home: Path, dirs, tmp_path: Path) -> None:
    env = _env(home, dirs, FAKE_MODE="flaky", FAKE_COUNTER=str(tmp_path / "n"), AIM_FBC_MAX_RESTARTS="0")
    assert entrypoint.main([], env) == 1
    s = _summary(dirs[1])
    assert s["restarts"] == 0 and (tmp_path / "n").read_text() == "1"


def test_time_limit_is_not_retried_and_counts_across_attempts(entrypoint, home: Path, dirs, tmp_path: Path) -> None:
    ready = tmp_path / "ready"
    env = _env(home, dirs, FAKE_MODE="slow", FAKE_READY=str(ready), AIM_FBC_TIME_LIMIT="2")
    assert entrypoint.main([], env, grace_s=2.0) == 124
    s = _summary(dirs[1])
    assert s["status"] == "time_limit" and s["restarts"] == 0 and s["crash_causes"] == []


def test_invalid_input_is_not_retried(entrypoint, home: Path, dirs, tmp_path: Path) -> None:
    (dirs[0] / "target.pdb").write_text("not a pdb\n")
    env = _env(home, dirs, FAKE_MODE="flaky", FAKE_COUNTER=str(tmp_path / "n"), FAKE_CRASHES="99")
    assert entrypoint.main([], env) == 2
    assert not (tmp_path / "n").exists(), "upstream was never started"
    assert _summary(dirs[1])["restarts"] == 0


def test_max_restarts_setting_is_validated(entrypoint, home: Path, dirs) -> None:
    assert entrypoint.main([], _env(home, dirs, AIM_FBC_MAX_RESTARTS="-1")) == 2
    assert entrypoint.main([], _env(home, dirs, AIM_FBC_MAX_RESTARTS="many")) == 2


# --- signals outside the supervised child -----------------------------------


def test_sigterm_in_the_restart_gap_stops_cleanly_with_a_final_summary(
    entrypoint, home: Path, dirs, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    out = dirs[1]
    counter = tmp_path / "n"
    real_publish = entrypoint.publish_results

    def publish_then_signal(design_dir: Path, out_dir: Path) -> None:
        real_publish(design_dir, out_dir)
        log = out / "bindcraft.log"
        if log.exists() and "restarting bindcraft.py" in log.read_text() and not (tmp_path / "sent").exists():
            (tmp_path / "sent").write_text("1")
            os.kill(os.getpid(), signal.SIGTERM)  # lands in the gap between two attempts

    monkeypatch.setattr(entrypoint, "publish_results", publish_then_signal)
    env = _env(home, dirs, FAKE_MODE="flaky", FAKE_COUNTER=str(counter), FAKE_CRASHES="5", AIM_FBC_NUM_DESIGNS="1")
    assert entrypoint.main([], env) == 143
    s = _summary(out)
    assert s["status"] == "interrupted" and s["exit_code"] == 143 and s["finished"] is True
    assert counter.read_text() == "1", "bindcraft.py must not be started again after the signal"
    assert "not starting bindcraft.py" in (out / "bindcraft.log").read_text()


def test_signal_handlers_are_restored_after_main(entrypoint, home: Path, dirs) -> None:
    before = {s: signal.getsignal(s) for s in (signal.SIGTERM, signal.SIGINT, signal.SIGHUP)}
    assert entrypoint.main([], _env(home, dirs, FAKE_MODE="ok", AIM_FBC_NUM_DESIGNS="1")) == 0
    assert {s: signal.getsignal(s) for s in before} == before


def test_missing_af2_params_fail_before_the_run_with_status_2(entrypoint, home: Path, dirs) -> None:
    env = _env(home, dirs)
    (home / "params" / "params_model_1_multimer_v3.npz").unlink()
    assert entrypoint.main([], env) == 2
    assert "AlphaFold2 parameters not found" in (dirs[1] / "bindcraft.log").read_text()


def test_mkdssp_on_path_is_written_into_the_advanced_settings(
    entrypoint, home: Path, dirs, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    bindir = tmp_path / "bin"
    bindir.mkdir()
    mk = bindir / "mkdssp"
    mk.write_text("#!/bin/sh\n")
    mk.chmod(0o755)
    record = tmp_path / "record.json"
    env = _env(home, dirs, FAKE_RECORD=str(record), AIM_FBC_MAX_TRAJECTORIES="1")
    monkeypatch.setenv("PATH", f"{bindir}{os.pathsep}{os.environ.get('PATH', '')}")
    assert entrypoint.main([], env) == 0
    assert json.loads(record.read_text())["advanced"]["dssp_path"] == str(mk)


def test_preflight_only_checks_and_exits_zero_without_designing(entrypoint, home: Path, dirs) -> None:
    inp, out = dirs
    env = _env(home, dirs, AIM_FBC_PREFLIGHT_ONLY="1", AIM_FBC_HOTSPOTS="A20", AIM_FBC_MIN_LENGTH="31")
    (home / "params" / "params_model_1_multimer_v3.npz").unlink()  # check-only mode needs no AlphaFold2 parameters
    (home / "bindcraft.py").unlink()  # and never starts the design process
    assert entrypoint.main([], env) == 0
    s = _summary(out)
    assert s["status"] == "preflight_ok" and s["finished"] is True and s["exit_code"] == 0
    assert (s["trajectories"], s["accepted_designs"], s["accelerators"]) == (0, 0, 1)
    assert s["settings"]["preflight_only"] is True and s["settings"]["min_length"] == 31
    log = (out / "bindcraft.log").read_text()
    assert "AIM_FBC_PREFLIGHT_ONLY" in log and "hotspots:" in log
    assert not (out / ".work").exists(), "no work directory, no designs"
    assert sorted(p.name for p in out.iterdir()) == ["bindcraft.log", "summary.json"]


def test_preflight_only_still_rejects_bad_input_and_settings(entrypoint, home: Path, dirs) -> None:
    inp, out = dirs
    base = {"AIM_FBC_PREFLIGHT_ONLY": "1"}
    assert entrypoint.main([], _env(home, dirs, AIM_FBC_TARGET_CHAINS="Z", **base)) == 2
    assert entrypoint.main([], _env(home, dirs, AIM_FBC_MIN_LENGTH="5", **base)) == 2
    assert entrypoint.main([], _env(home, dirs, AIM_FBC_TYPO="1", **base)) == 2
    (inp / "target.pdb").write_text("not a pdb\n")
    assert entrypoint.main([], _env(home, dirs, **base)) == 2
    assert _summary(out)["status"] == "failed"
    assert "bindcraft.log" in {p.name for p in out.iterdir()}


def test_preflight_only_without_a_gpu_exits_3(entrypoint, home: Path, dirs, tmp_path: Path) -> None:
    env = _env(home, dirs, AIM_FBC_PREFLIGHT_ONLY="1")
    del env["AIM_ACCELERATOR_COUNT"]
    (tmp_path / "devroot" / "dri").mkdir(parents=True)  # no kfd, no render nodes
    assert entrypoint.main([], env, dev_root=tmp_path / "devroot") == 3
    assert "needs 1 AMD GPU" in (dirs[1] / "bindcraft.log").read_text()


def test_unused_engine_args_is_tolerated_with_a_warning(entrypoint, home: Path, dirs) -> None:
    env = _env(home, dirs, AIM_FBC_PREFLIGHT_ONLY="1", AIM_ENGINE_ARGS='{"nsteps": 10}')
    assert entrypoint.main([], env) == 0
    assert "WARNING: AIM_ENGINE_ARGS is set" in (dirs[1] / "bindcraft.log").read_text()
    out2 = dirs[1].parent / "out2"
    full = _env(
        home, (dirs[0], out2), AIM_ENGINE_ARGS='{"nsteps": 10}', AIM_FBC_NUM_DESIGNS="1", AIM_FBC_MAX_TRAJECTORIES="1"
    )
    assert entrypoint.main([], full) == 0
    assert "WARNING: AIM_ENGINE_ARGS is set" in (out2 / "bindcraft.log").read_text()
