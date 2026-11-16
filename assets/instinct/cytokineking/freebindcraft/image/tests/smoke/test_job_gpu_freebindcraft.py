# Copyright © Advanced Micro Devices, Inc., or its affiliates.
#
# SPDX-License-Identifier: MIT

"""GPU smoke of the built image against the check (not run in the CPU pre-gate).

Runs the real image with no arguments on a small, real input (1PGB chain A, one
design, one trajectory, short binder) and asserts exit 0, a populated output
directory and a log; then a deliberately broken input (non-zero exit, log still
present) and SIGTERM during a run (exit 143). Needs ``docker`` on a node with an
AMD GPU, the image tag in ``FBC_IMAGE`` and ``FBC_AF2_CACHE_PATH``: a host directory that holds
``params/params_model_*_multimer_v3.npz`` (the AlphaFold2 parameters, which the image does not bundle). It is mounted
read-only at ``/af2`` with ``AIM_CACHE_PATH=/af2`` and is validated before Docker starts. ``FBC_SMOKE_TIME_LIMIT`` (default
``2h``) caps the real run; a capped run exits 124 and fails this test on purpose,
so a smoke that could not finish is never reported as green.
"""

from __future__ import annotations

import json
import os
import subprocess
import time
from pathlib import Path

import pytest

pytestmark = [pytest.mark.gpu, pytest.mark.integration]

REFERENCES = Path(__file__).resolve().parents[1] / "references"
GPU_FLAGS = [
    "--device=/dev/kfd",
    "--device=/dev/dri",
    "--group-add",
    "video",
    "--group-add",
    "render",
    "--security-opt",
    "seccomp=unconfined",
]


@pytest.fixture(scope="module")
def image() -> str:
    tag = os.environ.get("FBC_IMAGE", "")
    if not tag:
        pytest.fail("FBC_IMAGE is not set: pass the tag of the built image")
    return tag


AF2_MOUNT = "/af2"
AF2_MARKER = Path("params") / "params_model_1_multimer_v3.npz"


def validated_af2_cache(raw: str | None) -> Path:
    """The operator's AlphaFold2 parameter directory, or a ``ValueError`` that says what is wrong."""
    if not raw or not raw.strip():
        raise ValueError(
            "FBC_AF2_CACHE_PATH is not set: pass a host directory that holds params/params_model_*_multimer_v3.npz"
        )
    path = Path(raw.strip()).expanduser()
    if not path.is_absolute():
        raise ValueError(f"FBC_AF2_CACHE_PATH must be an absolute path (docker mounts it): {raw!r}")
    if not path.is_dir():
        raise ValueError(f"FBC_AF2_CACHE_PATH is not a directory: {path}")
    if not (path / AF2_MARKER).is_file():
        raise ValueError(f"FBC_AF2_CACHE_PATH holds no {AF2_MARKER}: {path}")
    return path


@pytest.fixture(scope="module")
def af2_cache() -> Path:
    try:
        return validated_af2_cache(os.environ.get("FBC_AF2_CACHE_PATH"))
    except ValueError as exc:
        pytest.fail(str(exc))


def _stage(tmp_path: Path, settings: dict | None = None) -> tuple[Path, Path]:
    inp, out = tmp_path / "in", tmp_path / "out"
    inp.mkdir()
    out.mkdir()
    out.chmod(0o777)
    # The reference file carries a license comment header; hand the job a plain PDB.
    lines = (REFERENCES / "1PGB_A.pdb").read_text().splitlines()
    (inp / "target.pdb").write_text("\n".join(line for line in lines if not line.startswith("#")).lstrip("\n") + "\n")
    if settings is not None:
        (inp / "settings.json").write_text(json.dumps(settings))
    return inp, out


def _docker_run(
    image: str, af2_cache: Path, inp: Path, out: Path, *extra: str, detach: bool = False, timeout: float | None = None
):
    cmd = [
        "docker",
        "run",
        *(["-d"] if detach else ["--rm"]),
        *GPU_FLAGS,
        "-e",
        "ROCR_VISIBLE_DEVICES",
        "-e",
        "AIM_INPUT_PATH=/input",
        "-e",
        "AIM_OUTPUT_PATH=/output",
        "-e",
        f"AIM_CACHE_PATH={AF2_MOUNT}",
        "-v",
        f"{af2_cache}:{AF2_MOUNT}:ro",
        "-v",
        f"{inp}:/input:ro",
        "-v",
        f"{out}:/output",
        *extra,
        image,
    ]
    return subprocess.run(cmd, capture_output=True, text=True, timeout=timeout, check=False)


def _send_sigterm(cid: str) -> None:
    """SIGTERM to the container's PID 1. Some hosts refuse `docker kill` (runc: unable to signal init:
    permission denied); signalling from inside the container's PID namespace is the fallback."""
    killed = subprocess.run(["docker", "kill", "--signal", "TERM", cid], capture_output=True, text=True, check=False)
    if killed.returncode != 0:
        python = "/miniforge3/envs/BindCraft/bin/python"
        subprocess.run(
            ["docker", "exec", cid, python, "-c", "import os; os.kill(1, 15)"], check=True, capture_output=True
        )


def test_real_run_exits_zero_with_outputs_and_log(image: str, af2_cache: Path, tmp_path: Path) -> None:
    inp, out = _stage(
        tmp_path,
        {
            "binder_name": "smoke",
            "min_length": 31,
            "max_length": 40,
            "num_designs": 1,
            "max_trajectories": 1,
            "time_limit": os.environ.get("FBC_SMOKE_TIME_LIMIT", "2h"),
        },
    )
    done = _docker_run(image, af2_cache, inp, out)
    summary = json.loads((out / "summary.json").read_text())
    assert done.returncode == 0, f"exit {done.returncode}; summary={summary}; tail={done.stdout[-2000:]}"
    assert summary["finished"] is True and summary["status"] in ("completed", "max_trajectories_reached")
    assert summary["trajectories"] >= 1 and summary["time_to_first_trajectory_s"] is not None
    assert (out / "bindcraft.log").stat().st_size > 0
    assert (out / "trajectory_stats.csv").is_file()


def test_broken_input_exits_non_zero_and_still_logs(image: str, af2_cache: Path, tmp_path: Path) -> None:
    inp, out = _stage(tmp_path)
    (inp / "target.pdb").write_text("not a pdb\n")
    done = _docker_run(image, af2_cache, inp, out)
    assert done.returncode != 0
    assert (out / "bindcraft.log").stat().st_size > 0
    assert json.loads((out / "summary.json").read_text())["status"] == "failed"


def test_sigterm_exits_143(image: str, af2_cache: Path, tmp_path: Path) -> None:
    inp, out = _stage(tmp_path, {"min_length": 31, "max_length": 40, "num_designs": 1})
    started = _docker_run(image, af2_cache, inp, out, detach=True)
    assert started.returncode == 0, started.stderr
    cid = started.stdout.strip()
    try:
        deadline = time.monotonic() + 600
        while not (out / "bindcraft.log").exists() or "command:" not in (out / "bindcraft.log").read_text():
            assert time.monotonic() < deadline, "the run did not start"
            time.sleep(1)
        _send_sigterm(cid)
        waited = subprocess.run(["docker", "wait", cid], capture_output=True, text=True, timeout=120, check=True)
        assert waited.stdout.strip() == "143"
    finally:
        subprocess.run(["docker", "rm", "-f", cid], capture_output=True, check=False)
