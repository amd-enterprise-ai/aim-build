# Copyright © Advanced Micro Devices, Inc., or its affiliates.
#
# SPDX-License-Identifier: MIT

"""The shipped CI inputs (tests/ci-input) behave as the run-to-completion check expects.

Each case is staged into a temporary AIM_INPUT_PATH and run through the entrypoint with the environment CI sets
(AIM_INPUT_PATH, AIM_OUTPUT_PATH, AIM_ACCELERATOR_COUNT, the case's .env.yaml). No AlphaFold2 parameters and no
upstream checkout exist here, which is the situation on a CI runner for the check-only ``small`` case.
"""

from __future__ import annotations

import base64
import gzip
import io
import json
import shutil
import tarfile
from pathlib import Path

import yaml

CI_INPUT = Path(__file__).resolve().parents[1] / "ci-input"
MAX_ENCODED_CHARS = 190_000  # ci/docs/aim_conventions.md: base64 of the gzip tar, per case


def _env(case: str, tmp_path: Path) -> tuple[dict[str, str], Path]:
    inp, out = tmp_path / "in", tmp_path / "out"
    shutil.copytree(CI_INPUT / case, inp)
    case_env = yaml.safe_load((CI_INPUT / f"{case}.env.yaml").read_text()) or {}
    env = {
        "AIM_INPUT_PATH": str(inp),
        "AIM_OUTPUT_PATH": str(out),
        "AIM_ACCELERATOR_COUNT": "1",
        "AIM_ACCELERATOR_MODEL": "MI300X",
        "BINDCRAFT_HOME": str(tmp_path / "no-upstream"),
        **{k: str(v) for k, v in case_env.items()},
    }
    return env, out


def test_small_case_passes_in_check_only_mode(entrypoint, tmp_path: Path) -> None:
    env, out = _env("small", tmp_path)
    assert env["AIM_FBC_PREFLIGHT_ONLY"] == "1"
    assert entrypoint.main([], env) == 0
    names = sorted(p.name for p in out.iterdir())
    assert names == ["bindcraft.log", "summary.json"]  # one *.log and one other file
    assert json.loads((out / "summary.json").read_text())["status"] == "preflight_ok"


def test_broken_case_fails_through_input_validation_with_a_log(entrypoint, tmp_path: Path) -> None:
    env, out = _env("broken", tmp_path)
    code = entrypoint.main([], env)
    assert code == 2  # exit codes 1 to 127 only: never a signal
    assert (out / "bindcraft.log").is_file()
    assert (
        "not found" in (out / "bindcraft.log").read_text() or "no protein ATOM" in (out / "bindcraft.log").read_text()
    )


def test_cases_fit_the_workload_size_limit() -> None:
    for case in ("small", "broken"):
        buf = io.BytesIO()
        with tarfile.open(fileobj=buf, mode="w") as tar:
            tar.add(CI_INPUT / case, arcname=".")
        encoded = base64.b64encode(gzip.compress(buf.getvalue()))
        assert len(encoded) <= MAX_ENCODED_CHARS, case
