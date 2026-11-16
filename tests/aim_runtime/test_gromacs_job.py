# Copyright © Advanced Micro Devices, Inc., or its affiliates.
#
# SPDX-License-Identifier: MIT

"""CPU-only tests for the GROMACS job AIM using a fake ``gmx`` child.

The real ``entrypoint.py`` / ``runner.py`` are imported from the asset; no GROMACS,
GPU or container is needed.
"""

from __future__ import annotations

import json
import os
import re
import signal
import stat
import subprocess
import sys
import textwrap
import threading
import time
from pathlib import Path

import pytest
import yaml

ASSET = Path(__file__).resolve().parents[2] / "assets/instinct/gromacs/gromacs"
SRC = ASSET / "image/src"

# The asset imports a top-level ``utils`` module that would shadow ``ci/utils`` for other tests in the
# same process, so import it with a temporary sys.path and drop the generic module names afterwards.
_saved = {name: sys.modules.pop(name, None) for name in ("utils", "runner", "entrypoint")}
sys.path.insert(0, str(SRC))
try:
    import runner  # noqa: E402

    import entrypoint  # noqa: E402
finally:
    sys.path.remove(str(SRC))
    for _name, _mod in _saved.items():
        sys.modules.pop(_name, None)
        if _mod is not None:
            sys.modules[_name] = _mod

FAKE_GMX = textwrap.dedent("""\
    #!/usr/bin/env python3
    import os, signal, sys, time
    a = sys.argv
    deffnm = a[a.index("-deffnm") + 1]
    mode = os.environ.get("FAKE_MODE", "ok")
    print("fake-gmx start", flush=True)
    if mode == "ok":
        open(deffnm + ".log", "w").write("Performance:       12.5        1.920\\n")
        open(deffnm + ".xtc", "w").write("x")
        open(deffnm + ".cpt", "w").write("c")
        print("step 1", flush=True)
        sys.exit(0)
    if mode == "argv":  # record the command line, for the resume tests
        open(deffnm + ".argv", "w").write(" ".join(a[1:]))
        sys.exit(0)
    if mode == "tprcheck":  # like mdrun: reject an input that is not a run-input file (real tpr: 00 00 00 <ver>)
        tpr = a[a.index("-s") + 1]
        if not open(tpr, "rb").read(4).startswith(bytes(3)):
            print("Fatal error: Failed to read run input " + tpr, file=sys.stderr, flush=True)
            sys.exit(1)
        open(deffnm + ".log", "w").write("Performance:       12.5        1.920\\n")
        open(deffnm + ".edr", "w").write("e")
        open(deffnm + ".argv", "w").write(" ".join(a[1:]))
        sys.exit(0)
    if mode == "fail":
        print("fatal: boom", file=sys.stderr, flush=True)
        sys.exit(3)
    if mode == "selfkill":
        os.kill(os.getpid(), signal.SIGKILL)
    if mode == "graceful":  # like mdrun: stop cleanly on SIGTERM, exit 0
        signal.signal(signal.SIGTERM, lambda *_: sys.exit(0))
        open(deffnm + ".cpt", "w").write("partial")
        print("ready", flush=True)
        time.sleep(30)
    if mode == "hang":  # ignore SIGTERM
        signal.signal(signal.SIGTERM, signal.SIG_IGN)
        print("ready", flush=True)
        time.sleep(30)
    """)


@pytest.fixture
def fake_gmx(tmp_path, monkeypatch):
    exe = tmp_path / "fakebin" / "gmx"
    exe.parent.mkdir()
    exe.write_text(FAKE_GMX)
    exe.chmod(exe.stat().st_mode | stat.S_IEXEC)
    monkeypatch.setattr(runner, "_select_binary", lambda ntmpi: str(exe))
    monkeypatch.setattr(runner, "KILL_GRACE_SECONDS", 2.0)
    return exe


def _run(tmp_path, mode, **kw):
    os.environ["FAKE_MODE"] = mode
    tpr = tmp_path / "run.tpr"
    tpr.write_text("tpr")
    out = tmp_path / "out"
    rc = runner.run_mdrun(tpr, out, ntmpi=1, **kw)
    return rc, out


@pytest.fixture(autouse=True)
def _clean_mode():
    yield
    os.environ.pop("FAKE_MODE", None)


# ---------------------------------------------------------------- exit codes / outputs


def test_exit_code_mapping():
    assert runner.exit_code_from_returncode(0) == 0
    assert runner.exit_code_from_returncode(3) == 3
    assert runner.exit_code_from_returncode(-15) == 143
    assert runner.exit_code_from_returncode(-9) == 137


def test_success_publishes_files_and_done_marker(tmp_path, fake_gmx, capfd):
    rc, out = _run(tmp_path, "ok")
    assert rc == 0
    assert sorted(p.name for p in out.iterdir()) == ["DONE", "gromacs.log", "md.cpt", "md.log", "md.xtc"]
    assert not (out / ".work").exists()
    log = (out / "gromacs.log").read_text()
    assert "fake-gmx start" in log and "step 1" in log
    assert "fake-gmx start" in capfd.readouterr().out  # tee'd to stdout


def test_failure_keeps_log_no_done_no_published_outputs(tmp_path, fake_gmx):
    rc, out = _run(tmp_path, "fail")
    assert rc == 3
    assert "fatal: boom" in (out / "gromacs.log").read_text()  # stderr merged
    assert not (out / "DONE").exists() and not (out / "md.xtc").exists()


def test_child_killed_by_signal_maps_to_128_plus_n(tmp_path, fake_gmx):
    rc, _ = _run(tmp_path, "selfkill")
    assert rc == 137


def test_missing_binary_still_writes_log(tmp_path, monkeypatch):
    def boom(ntmpi):
        raise RuntimeError("Neither /opt/gromacs-tmpi/bin/gmx nor mpi found")

    monkeypatch.setattr(runner, "_select_binary", boom)
    tpr = tmp_path / "run.tpr"
    tpr.write_text("t")
    rc = runner.run_mdrun(tpr, tmp_path / "out", ntmpi=1)
    assert rc == 1
    assert "Neither" in (tmp_path / "out/gromacs.log").read_text()


def test_unexecutable_binary_is_127_and_logged(tmp_path, monkeypatch):
    monkeypatch.setattr(runner, "_select_binary", lambda n: str(tmp_path / "does-not-exist"))
    tpr = tmp_path / "run.tpr"
    tpr.write_text("t")
    rc = runner.run_mdrun(tpr, tmp_path / "out", ntmpi=1)
    assert rc == 127
    assert "Cannot start GROMACS" in (tmp_path / "out/gromacs.log").read_text()


# ---------------------------------------------------------------- resume


def test_mdrun_is_not_given_cpi_without_a_checkpoint(tmp_path, fake_gmx):
    rc, out = _run(tmp_path, "argv")
    assert rc == 0
    assert "-cpi" not in (out / "md.argv").read_text().split()


def test_mdrun_is_given_cpi_when_a_checkpoint_is_in_work(tmp_path, fake_gmx):
    work = tmp_path / "out" / ".work"
    work.mkdir(parents=True)
    (work / "md.cpt").write_text("state")
    rc, out = _run(tmp_path, "argv")
    assert rc == 0
    argv = (out / "md.argv").read_text().split()
    assert argv[argv.index("-cpi") + 1] == str(work / "md.cpt")


# ---------------------------------------------------------------- signals / streaming


def _send_after_ready(log: Path, sig: int):
    def go():
        for _ in range(200):
            if log.exists() and "ready" in log.read_text():
                break
            time.sleep(0.05)
        os.kill(os.getpid(), sig)

    t = threading.Thread(target=go)
    t.start()
    return t


@pytest.mark.parametrize("sig", [signal.SIGTERM, signal.SIGINT])
def test_signal_is_forwarded_and_exit_is_128_plus_n(tmp_path, fake_gmx, sig):
    t = _send_after_ready(tmp_path / "out/gromacs.log", sig)
    start = time.time()
    rc, out = _run(tmp_path, "graceful")  # child exits 0 on SIGTERM, but run is incomplete
    t.join()
    assert rc == 128 + sig
    assert time.time() - start < 20  # forwarded, not waited out
    assert not (out / "DONE").exists()
    assert (out / ".work/md.cpt").exists()  # checkpoint kept for resume


def test_stuck_child_is_killed_after_grace(tmp_path, fake_gmx):
    t = _send_after_ready(tmp_path / "out/gromacs.log", signal.SIGTERM)
    rc, _ = _run(tmp_path, "hang")
    t.join()
    assert rc == 128 + signal.SIGTERM


def test_signal_handlers_restored(tmp_path, fake_gmx):
    before = signal.getsignal(signal.SIGTERM)
    _run(tmp_path, "ok")
    assert signal.getsignal(signal.SIGTERM) is before


def test_logs_stream_live(tmp_path, fake_gmx):
    """gromacs.log gains lines while the child is still running."""
    seen = []

    def watcher():
        log = tmp_path / "out/gromacs.log"
        for _ in range(200):
            if log.exists() and "ready" in log.read_text():
                seen.append(True)
                break
            time.sleep(0.05)
        os.kill(os.getpid(), signal.SIGTERM)

    t = threading.Thread(target=watcher)
    t.start()
    _run(tmp_path, "graceful")
    t.join()
    assert seen


# ---------------------------------------------------------------- settings


def test_engine_args_json_object_kebab_and_snake():
    assert entrypoint.parse_engine_args('{"ntomp": 24, "gpu-id": "01", "nstlist": "150"}') == {
        "ntomp": 24,
        "gpu_id": "01",
        "nstlist": 150,
    }


@pytest.mark.parametrize("raw", ["--ntomp 24", '["a"]', '{"bogus": 1}', '{"ntomp": "x"}', '{"ntomp": true}'])
def test_engine_args_rejects_bad_input(raw):
    with pytest.raises(entrypoint.SettingsError):
        entrypoint.parse_engine_args(raw)


def test_unknown_aim_var_is_warned_and_ignored(caplog):
    """A launcher may set AIM_* variables this job does not use: warn and carry on, do not exit 2."""
    with caplog.at_level("WARNING"):
        args = entrypoint.resolve_settings([], {"AIM_FOO": "1", "AIM_PROFILE_ID": "p"}, 1)
    assert args.ntmpi == 1
    assert "AIM_FOO" in caplog.text and "AIM_PROFILE_ID" in caplog.text


def test_check_aim_env_returns_only_unknown_names():
    assert entrypoint.check_aim_env({"AIM_INPUT_PATH": "/i", "AIM_B": "1", "AIM_A": "1", "PATH": "/bin"}) == [
        "AIM_A",
        "AIM_B",
    ]
    assert entrypoint.check_aim_env({}) == []


def test_precedence_cli_over_engine_args_over_default():
    env = {"AIM_ENGINE_ARGS": '{"ntomp": 24, "nstlist": 100}'}
    a = entrypoint.resolve_settings(["--ntomp", "8"], env, 2)
    assert (a.ntomp, a.nstlist, a.nsteps, a.ntmpi, a.gpu_id) == (8, 100, -2, 2, "01")


@pytest.mark.parametrize(
    "ntmpi,cpus,expected",
    [(1, 48, 24), (2, 48, 12), (4, 96, 6), (1, 16, 15), (2, 16, 7), (1, 1, 1), (1, 2, 1), (4, 3, 1)],
)
def test_default_ntomp_is_capped_at_24_and_leaves_a_hardware_thread_free(ntmpi, cpus, expected):
    assert entrypoint.default_ntomp(ntmpi, cpus) == expected


def test_ntomp_default_is_derived_and_overridable(monkeypatch):
    monkeypatch.setattr(entrypoint, "usable_cpus", lambda: 48)
    assert entrypoint.resolve_settings([], {}, 1).ntomp == 24
    assert entrypoint.resolve_settings([], {"AIM_ENGINE_ARGS": '{"ntomp": 48}'}, 1).ntomp == 48
    assert entrypoint.resolve_settings(["--ntomp", "12"], {"AIM_ENGINE_ARGS": '{"ntomp": 48}'}, 1).ntomp == 12
    with pytest.raises(entrypoint.SettingsError, match="ntomp"):
        entrypoint.resolve_settings(["--ntomp", "0"], {}, 1)


def test_gpu_count_detection_and_override(tmp_path):
    (tmp_path / "kfd").write_text("")
    (tmp_path / "dri").mkdir()
    for n in (128, 129, 130):
        (tmp_path / "dri" / f"renderD{n}").write_text("")
    (tmp_path / "dri" / "card0").write_text("")
    assert entrypoint.detect_gpu_count(str(tmp_path)) == 3
    assert entrypoint.detect_gpu_count(str(tmp_path / "nowhere")) == 0
    assert entrypoint.resolve_settings([], {}, 3).ntmpi == 3
    assert entrypoint.resolve_settings([], {"AIM_ACCELERATOR_COUNT": "2"}, 3).ntmpi == 2
    assert entrypoint.resolve_settings([], {"AIM_ACCELERATOR_COUNT": "2"}, 0).ntmpi == 2  # detection overridden


def test_rejects_more_gpus_than_available_and_no_gpu():
    with pytest.raises(entrypoint.SettingsError, match="only 2"):
        entrypoint.resolve_settings([], {"AIM_ACCELERATOR_COUNT": "4"}, 2)
    with pytest.raises(entrypoint.SettingsError, match="only 2"):
        entrypoint.resolve_settings(["--ntmpi", "4"], {}, 2)
    with pytest.raises(entrypoint.SettingsError, match="No AMD GPU"):
        entrypoint.resolve_settings([], {}, 0)


def test_main_exit_codes(tmp_path, monkeypatch, fake_gmx):
    monkeypatch.setattr(entrypoint, "detect_gpu_count", lambda: 1)
    inp = tmp_path / "in"
    inp.mkdir()
    env = {"AIM_INPUT_PATH": str(inp), "AIM_OUTPUT_PATH": str(tmp_path / "out")}
    assert entrypoint.main([], {**env, "AIM_ENGINE_ARGS": "--ntomp 2"}) == 2
    assert entrypoint.main([], env) == 1  # no run.tpr
    (inp / "run.tpr").write_text("t")
    os.environ["FAKE_MODE"] = "ok"
    assert entrypoint.main([], env) == 0
    os.environ["FAKE_MODE"] = "fail"
    assert entrypoint.main([], env) == 3


def test_entrypoint_as_pid1_style_subprocess_sigterm(tmp_path):
    """End to end: run the entrypoint as a real process, SIGTERM it, expect status 143."""
    fakebin = tmp_path / "gmx"
    fakebin.write_text(FAKE_GMX)
    fakebin.chmod(0o755)
    inp = tmp_path / "in"
    inp.mkdir()
    (inp / "run.tpr").write_text("t")
    shim = tmp_path / "shim.py"
    shim.write_text(
        f"import sys; sys.path.insert(0, {str(SRC)!r})\n"
        "import runner, entrypoint\n"
        f"runner._select_binary = lambda n: {str(fakebin)!r}\n"
        "entrypoint.detect_gpu_count = lambda: 1\n"
        "sys.exit(entrypoint.main([]))\n"
    )
    # Drop ambient AIM_* variables (CI exports AIM_REGISTRY_* etc.) so the run only sees the settings under test.
    env = {
        **{k: v for k, v in os.environ.items() if not k.startswith("AIM_")},
        "FAKE_MODE": "graceful",
        "AIM_INPUT_PATH": str(inp),
        "AIM_OUTPUT_PATH": str(tmp_path / "out"),
    }
    p = subprocess.Popen([sys.executable, str(shim)], env=env, stdout=subprocess.PIPE, text=True)
    deadline = time.time() + 20
    while time.time() < deadline:
        log = tmp_path / "out/gromacs.log"
        if log.exists() and "ready" in log.read_text():
            break
        time.sleep(0.05)
    p.send_signal(signal.SIGTERM)
    assert p.wait(timeout=20) == 143


# ---------------------------------------------------------------- shipped profiles


@pytest.mark.parametrize("path", sorted((ASSET / "profiles").glob("*.yaml")), ids=lambda p: p.name)
def test_profile_engine_args_are_accepted_settings(path):
    prof = yaml.safe_load(path.read_text())
    assert prof["metadata"]["job_type"] == "long_running"  # declared per profile
    assert "hardware_requirements" not in prof
    parsed = entrypoint.parse_engine_args(json.dumps(prof["engine_args"]))  # same keys as AIM_ENGINE_ARGS
    assert parsed["ntmpi"] == prof["metadata"]["accelerator_count"]
    # profile env_vars are applied verbatim by the platform; they must not be rejected as AIM_* settings
    entrypoint.check_aim_env(prof["env_vars"])


def test_only_single_gpu_profiles_ship():
    """Multi-GPU (ntmpi > 1) has not been run on this image, so no profile for it ships."""
    counts = {
        yaml.safe_load(p.read_text())["metadata"]["accelerator_count"] for p in (ASSET / "profiles").glob("*.yaml")
    }
    assert counts == {1}


def test_builder_stages_install_rocm_thrust_packages():
    dockerfile = (ASSET / "image/Dockerfile").read_text()
    stage_mpi = dockerfile.split("AS gromacs-mpi-builder")[1]
    stage_tmpi = dockerfile.split("AS gromacs-tmpi-builder")[1].split("AS gromacs-mpi-builder")[0]
    for stage in (stage_tmpi, stage_mpi):
        for pkg in ("rocprim-dev", "rocthrust-dev", "hipcub-dev"):
            assert pkg in stage


def test_job_type_not_in_metadata_yaml():
    meta = yaml.safe_load((ASSET / "metadata.yaml").read_text())
    assert "job_type" not in meta["com"]["amd"]["aim"]  # job_type lives in the profile YAML


def test_runtime_image_is_slim_and_has_no_unused_python_deps():
    dockerfile = (ASSET / "image/Dockerfile").read_text()
    runtime = dockerfile.split("# ---- Stage 2")[1]
    assert "openmpi-bin" in runtime
    assert "libopenmpi-dev" not in runtime and "git \\" not in runtime
    assert "/usr/local/bin/python" not in dockerfile
    # the job code is stdlib-only, so no requirements file / pip install is needed
    assert not (ASSET / "image/requirements.txt").exists()
    assert "requirements.txt" not in dockerfile


def test_dockerfile_targets_mi300x_only():
    """GROMACS is validated on MI300X (gfx942) only; both build stages must offload to gfx942 alone."""
    text = (ASSET / "image/Dockerfile").read_text()
    archs = re.findall(r'-DCMAKE_HIP_ARCHITECTURES="([^"]*)"', text)
    assert len(archs) == 2  # both build stages
    assert set(archs) == {"gfx942"}
    assert "-DGMX_SIMD=AVX2_256" in text
    for profile in (ASSET / "profiles").glob("*.yaml"):
        assert "mi300x" in profile.name


def test_base_image_and_source_are_pinned():
    text = (ASSET / "image/Dockerfile").read_text()
    tag = re.search(r"^ARG BASE_IMAGE=rocm/dev-ubuntu-24\.04:(\S+)$", text, re.M).group(1)
    assert tag != "latest" and re.fullmatch(r"7\.2\.\d+", tag)
    cfg = yaml.safe_load((ASSET / "config.yaml").read_text())["base_image"]
    assert cfg["base_tag"] == tag
    sha = "924b96a7ac69a2e84cf2b73374f82d1f2eeab68a"
    assert re.findall(r"^ARG GROMACS_REF=(\S+)$", text, re.M) == [sha, sha]  # both build stages


def test_dockerfile_records_apt_package_manifest():
    text = (ASSET / "image/Dockerfile").read_text()
    assert "dpkg-query -W" in text
    assert "/opt/aim-build-packages.txt" in text
    assert "/opt/gromacs-build-packages.txt" in text


# ---------------------------------------------------------------- pme_tune / bonded_cpu


def _cmd_line(out):
    return next(ln for ln in (out / "gromacs.log").read_text().splitlines() if ln.startswith("$ ")).split()


def test_mdrun_defaults_pass_notunepme_and_bonded_cpu(tmp_path, fake_gmx):
    rc, out = _run(tmp_path, "ok")
    assert rc == 0
    cmd = _cmd_line(out)
    assert "-notunepme" in cmd
    assert cmd[cmd.index("-bonded") + 1] == "cpu"


def test_mdrun_pme_tune_opt_in_and_bonded_gpu_opt_out(tmp_path, fake_gmx):
    rc, out = _run(tmp_path, "ok", pme_tune=True, bonded_cpu=False)
    assert rc == 0
    cmd = _cmd_line(out)
    assert "-notunepme" not in cmd
    assert cmd[cmd.index("-bonded") + 1] == "gpu"


def test_new_settings_defaults_and_validation():
    a = entrypoint.resolve_settings([], {}, 1)
    assert (a.pme_tune, a.bonded_cpu) == (False, True)
    env = {"AIM_ENGINE_ARGS": '{"pme-tune": true, "bonded_cpu": "false"}'}
    a = entrypoint.resolve_settings([], env, 1)
    assert (a.pme_tune, a.bonded_cpu) == (True, False)
    a = entrypoint.resolve_settings(["--no-pme-tune", "--bonded-cpu"], env, 1)
    assert (a.pme_tune, a.bonded_cpu) == (False, True)
    a = entrypoint.resolve_settings(["--no-bonded-cpu"], {}, 1)
    assert a.bonded_cpu is False


@pytest.mark.parametrize("raw", ['{"pme_tune": 1}', '{"bonded_cpu": "maybe"}', '{"pme_tune": null}'])
def test_bool_settings_reject_non_booleans(raw):
    with pytest.raises(entrypoint.SettingsError):
        entrypoint.parse_engine_args(raw)


def test_bad_engine_args_key_still_exits_2(tmp_path):
    (tmp_path / "run.tpr").write_text("t")
    env = {"AIM_INPUT_PATH": str(tmp_path), "AIM_OUTPUT_PATH": str(tmp_path / "o"), "AIM_ACCELERATOR_COUNT": "1"}
    assert entrypoint.main([], {**env, "AIM_ENGINE_ARGS": '{"pme_tunee": true}'}) == 2


def test_v1_build_has_no_fftw_and_no_colvars():
    """Licence variant V1: GROMACS links the bundled fftpack, FFTW packages are neither built against nor shipped."""
    text = (ASSET / "image/Dockerfile").read_text()
    assert "libfftw3" not in text and "GMX_BUILD_OWN_FFTW" not in text
    assert text.count("-DGMX_FFT_LIBRARY=fftpack") == 2  # both build stages
    assert text.count("-DGMX_USE_COLVARS=NONE") == 2
    assert text.count("ldd $bins | grep -i fftw") == 2  # build-time check in both stages
    assert "GMX_USE_TNG" not in text  # TNG stays at its default (ON)
    assert "libhwloc15" in text and "openmpi-bin" in text


def test_runner_never_requests_cpu_pme_or_cpu_fft():
    """The AIM forces GPU PME, so the fftpack CPU FFT is not on its default path (documented in API.md)."""
    src = (ASSET / "image/src/runner.py").read_text()
    assert '"-pme",\n                "gpu"' in src
    assert "pmefft" not in src and '"-pme",\n                "cpu"' not in src
    api = (ASSET / "templates/API.md").read_text()
    assert "fftpack" in api and "-pme cpu" in api and "colvars" in api


# ---------------------------------------------------------------- output directory reuse


def test_rerun_removes_stale_md_outputs_but_not_the_log_or_foreign_files(tmp_path, fake_gmx):
    out = tmp_path / "out"
    out.mkdir()
    (out / "md.trr").write_text("old trajectory")
    (out / "md.xtc").write_text("old")  # the new run also produces md.xtc: replaced, not removed
    (out / "notes.txt").write_text("mine")
    (out / "gromacs.log").write_text("earlier run\n")
    rc, out = _run(tmp_path, "ok")
    assert rc == 0
    assert not (out / "md.trr").exists()
    assert (out / "md.xtc").read_text() == "x"
    assert (out / "notes.txt").read_text() == "mine"
    assert (out / "gromacs.log").read_text().startswith("earlier run\n")
    assert (out / "DONE").is_file()


def test_failed_rerun_keeps_earlier_outputs(tmp_path, fake_gmx):
    out = tmp_path / "out"
    out.mkdir()
    (out / "md.xtc").write_text("old")
    rc, out = _run(tmp_path, "fail")
    assert rc == 3
    assert (out / "md.xtc").read_text() == "old"  # nothing is removed unless the new set is complete


def test_resume_into_same_directory_still_publishes_its_set_and_cleans_stale(tmp_path, fake_gmx):
    out = tmp_path / "out"
    # interrupted first run: checkpoint left in .work, an md.trr from an even earlier finished run
    os.environ["FAKE_MODE"] = "argv"
    (out / ".work").mkdir(parents=True)
    (out / ".work" / "md.cpt").write_text("state")
    (out / "md.trr").write_text("older run")
    rc, out = _run(tmp_path, "argv")
    assert rc == 0
    assert "-cpi" in (out / "md.argv").read_text().split()  # resumed
    assert (out / "md.cpt").read_text() == "state" and (out / "DONE").is_file()
    assert not (out / "md.trr").exists()


# ---------------------------------------------------------------- log exists for broken input


def test_missing_tpr_is_diagnosed_in_gromacs_log(tmp_path, monkeypatch):
    monkeypatch.setattr(entrypoint, "detect_gpu_count", lambda: 1)
    inp = tmp_path / "in"
    inp.mkdir()
    out = tmp_path / "out"
    assert entrypoint.main([], {"AIM_INPUT_PATH": str(inp), "AIM_OUTPUT_PATH": str(out)}) == 1
    assert "TPR file not found" in (out / "gromacs.log").read_text()
    assert not (out / "DONE").exists()


def test_missing_input_directory_is_diagnosed_in_gromacs_log(tmp_path, monkeypatch):
    monkeypatch.setattr(entrypoint, "detect_gpu_count", lambda: 1)
    out = tmp_path / "out"
    env = {"AIM_INPUT_PATH": str(tmp_path / "nope"), "AIM_OUTPUT_PATH": str(out)}
    assert entrypoint.main([], env) == 1
    assert "Input directory does not exist" in (out / "gromacs.log").read_text()


# ---------------------------------------------------------------- relative paths


def test_relative_input_and_output_paths_work(tmp_path, monkeypatch, fake_gmx):
    monkeypatch.setattr(entrypoint, "detect_gpu_count", lambda: 1)
    monkeypatch.chdir(tmp_path)
    (tmp_path / "input").mkdir()
    (tmp_path / "input" / "run.tpr").write_text("t")
    os.environ["FAKE_MODE"] = "argv"
    assert entrypoint.main(["--input", "input", "--output", "output"], {}) == 0
    argv = (tmp_path / "output" / "md.argv").read_text().split()
    assert argv[argv.index("-s") + 1] == str(tmp_path / "input" / "run.tpr")
    assert argv[argv.index("-deffnm") + 1] == str(tmp_path / "output" / ".work" / "md")
    assert (tmp_path / "output" / "DONE").is_file()


# ---------------------------------------------------------------- GPU visibility


def _dev_with_render_nodes(tmp_path, n):
    (tmp_path / "kfd").write_text("")
    (tmp_path / "dri").mkdir()
    for i in range(n):
        (tmp_path / "dri" / f"renderD{128 + i}").write_text("")
    return str(tmp_path)


def test_gpu_count_respects_rocr_and_hip_visible_devices(tmp_path):
    dev = _dev_with_render_nodes(tmp_path, 8)
    assert entrypoint.detect_gpu_count(dev, {}) == 8  # unset: unchanged
    assert entrypoint.detect_gpu_count(dev, {"ROCR_VISIBLE_DEVICES": ""}) == 8  # empty: ignored
    assert entrypoint.detect_gpu_count(dev, {"ROCR_VISIBLE_DEVICES": "0"}) == 1
    assert entrypoint.detect_gpu_count(dev, {"HIP_VISIBLE_DEVICES": "2,3,5"}) == 3
    assert entrypoint.detect_gpu_count(dev, {"ROCR_VISIBLE_DEVICES": "0,1,2,3", "HIP_VISIBLE_DEVICES": "1,2"}) == 2
    assert entrypoint.detect_gpu_count(dev, {"ROCR_VISIBLE_DEVICES": "-1"}) == 0
    assert (
        entrypoint.detect_gpu_count(dev, {"ROCR_VISIBLE_DEVICES": "0,1,2,3,4,5,6,7,8,9"}) == 8
    )  # never above the nodes


def test_single_gpu_default_unchanged(tmp_path):
    dev = _dev_with_render_nodes(tmp_path, 1)
    assert entrypoint.detect_gpu_count(dev, {}) == 1
    assert entrypoint.resolve_settings([], {}, entrypoint.detect_gpu_count(dev, {})).ntmpi == 1
    one_of_eight = _dev_with_render_nodes(tmp_path / "x", 8) if (tmp_path / "x").mkdir() is None else ""
    n = entrypoint.detect_gpu_count(one_of_eight, {"ROCR_VISIBLE_DEVICES": "0"})
    assert entrypoint.resolve_settings([], {}, n).ntmpi == 1


# ---------------------------------------------------------------- CI inputs (job-profile CI)

CI_INPUT = ASSET / "image/tests/ci-input"
_ci_saved = sys.path[:]
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "ci"))
try:
    import job_profile_check  # noqa: E402  (standard library and yaml only, so no module shadowing)
finally:
    sys.path[:] = _ci_saved


def _ci_env(case: str) -> dict[str, str]:
    """Environment the CI gives the image for a case: profile env_vars and engine_args, then the case env."""
    profile = ASSET / "profiles/job-mi300x-fp32-tp1-throughput.yaml"
    env_file = CI_INPUT / f"{case}.env.yaml"
    case_env = (
        {k: str(v) for k, v in (yaml.safe_load(env_file.read_text()) or {}).items()} if env_file.is_file() else {}
    )
    env = job_profile_check.job_settings_env(profile, case_env)
    env.update({"AIM_ACCELERATOR_COUNT": "1", "AIM_ACCELERATOR_MODEL": "MI300X"})
    return env


def test_ci_small_env_overrides_nsteps_key_by_key():
    """docs/engine-args-override.md: the case's AIM_ENGINE_ARGS wins per key, the profile's other keys stay."""
    env = _ci_env("small")
    merged = json.loads(env["AIM_ENGINE_ARGS"])
    assert merged["nstlist"] == 200 and merged["ntmpi"] == 1  # from the profile
    assert 0 <= merged["nsteps"] <= 1000  # from small.env.yaml, a few steps (the profile has -2 = whole .tpr run)
    args = entrypoint.resolve_settings([], env, detected_gpus=0)
    assert (args.nsteps, args.nstlist, args.ntmpi) == (merged["nsteps"], 200, 1)
    # the command line still wins over AIM_ENGINE_ARGS
    assert entrypoint.resolve_settings(["--nsteps", "7"], env, detected_gpus=0).nsteps == 7


def test_ci_profile_env_vars_do_not_conflict_with_runner_defaults(monkeypatch):
    """runner.py uses setdefault, so a profile env_var wins; check the profile values equal the defaults."""
    env = _ci_env("small")
    for key, value in yaml.safe_load((ASSET / "profiles/job-mi300x-fp32-tp1-throughput.yaml").read_text())[
        "env_vars"
    ].items():
        monkeypatch.delenv(key, raising=False)
        assert env[key] == value
        assert runner._build_gmx_env(24)[key] == value


def _stage_and_run(tmp_path, monkeypatch, case, input_dir):
    """Run the real entrypoint on a staged input with a fake gmx, as the CI wrapper would; return (rc, outcome)."""
    out = tmp_path / "artifacts/output"
    os.environ["FAKE_MODE"] = "tprcheck"
    environ = {**_ci_env(case), "AIM_INPUT_PATH": str(input_dir), "AIM_OUTPUT_PATH": str(out)}
    rc = entrypoint.main([], environ)
    files = sorted(p.relative_to(out).as_posix() for p in out.rglob("*") if p.is_file())
    return rc, job_profile_check.CaseOutcome(exit_code=rc, output_files=files), out


def test_ci_broken_case_exits_nonzero_with_log(tmp_path, fake_gmx, monkeypatch):
    # A fake gmx stands in for mdrun, which rejects a file that is not a run input.
    assert (CI_INPUT / "broken").is_dir()
    rc, outcome, out = _stage_and_run(tmp_path, monkeypatch, "broken", CI_INPUT / "broken")
    assert 1 <= rc < 128
    assert all(c["success"] for c in job_profile_check.evaluate_case("broken", outcome)), outcome
    assert "Fatal error" in (out / "gromacs.log").read_text()


def test_ci_broken_case_without_tpr_exits_1_with_log(tmp_path, fake_gmx, monkeypatch):
    empty = tmp_path / "in"
    empty.mkdir()
    rc, outcome, _ = _stage_and_run(tmp_path, monkeypatch, "broken", empty)
    assert rc == 1
    assert all(c["success"] for c in job_profile_check.evaluate_case("broken", outcome))


def test_ci_small_case_runs_few_steps_and_populates_output(tmp_path, fake_gmx, monkeypatch):
    staged = tmp_path / "in"
    staged.mkdir()
    (staged / "run.tpr").write_bytes(b"\x00\x00\x00\x0dsynthetic stand-in for the real run.tpr")
    rc, outcome, out = _stage_and_run(tmp_path, monkeypatch, "small", staged)
    assert rc == 0
    assert all(c["success"] for c in job_profile_check.evaluate_case("small", outcome)), outcome
    nsteps = json.loads(_ci_env("small")["AIM_ENGINE_ARGS"])["nsteps"]
    assert f"-nsteps {nsteps}" in (out / "md.argv").read_text()


def test_ci_input_layout():
    assert (CI_INPUT / "small.env.yaml").is_file() and (CI_INPUT / "broken/run.tpr").is_file()
    size = (CI_INPUT / "small/run.tpr").stat().st_size if (CI_INPUT / "small/run.tpr").is_file() else 0
    # small/run.tpr is generated on a node with gmx (see ci-input/README.md); once present it must fit the CI limit
    assert size <= 140_000
