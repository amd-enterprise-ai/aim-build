# Copyright © Advanced Micro Devices, Inc., or its affiliates.
#
# SPDX-License-Identifier: MIT

"""The mkdssp wrapper drops free-text REMARK records (which libcifpp rejects) and passes everything else on."""

from __future__ import annotations

import os
import stat
import subprocess
from pathlib import Path

import pytest

SHIM = Path(__file__).resolve().parents[2] / "src" / "mkdssp_shim.sh"

PDB = (
    "REMARK   Copyright (c) Example\n"
    "REMARK   SPDX-License-Identifier: MIT\n"
    "REMARK 2 RESOLUTION. 1.00 ANGSTROMS.\n"
    "REMARK\n"
    "ATOM      1  N   MET A   1      12.969  18.506  30.954  1.00 15.93           N\n"
    "END\n"
)


@pytest.fixture
def bindir(tmp_path: Path) -> Path:
    d = tmp_path / "bin"
    d.mkdir()
    shim = d / "mkdssp"
    shim.write_bytes(SHIM.read_bytes())
    shim.chmod(shim.stat().st_mode | stat.S_IXUSR)
    real = d / "mkdssp.real"
    # fake real binary: echoes its arguments, then the contents of the last argument (the input file)
    real.write_text('#!/bin/sh\necho "ARGS: $*"\nfor a; do f=$a; done\n[ -f "$f" ] && cat "$f"\nexit "${FAKE_RC:-0}"\n')
    real.chmod(real.stat().st_mode | stat.S_IXUSR)
    return d


def _run(bindir: Path, *args: str, env_extra: dict[str, str] | None = None) -> subprocess.CompletedProcess[str]:
    env = {**os.environ, **(env_extra or {})}
    return subprocess.run([str(bindir / "mkdssp"), *args], capture_output=True, text=True, env=env, check=False)


def test_free_text_remarks_are_dropped_and_numeric_remarks_kept(bindir: Path, tmp_path: Path) -> None:
    pdb = tmp_path / "in.pdb"
    pdb.write_text(PDB)
    r = _run(bindir, "--output-format=dssp", str(pdb))
    assert r.returncode == 0
    assert "REMARK   Copyright" not in r.stdout and "SPDX" not in r.stdout
    assert "REMARK 2 RESOLUTION" in r.stdout
    assert "ATOM      1  N   MET A" in r.stdout and "END" in r.stdout
    assert "--output-format=dssp" in r.stdout.splitlines()[0]
    assert pdb.read_text() == PDB  # the caller's file is never modified


def test_version_and_missing_input_are_passed_through(bindir: Path) -> None:
    r = _run(bindir, "--version")
    assert r.returncode == 0 and "ARGS: --version" in r.stdout


def test_exit_status_of_the_real_binary_is_returned(bindir: Path, tmp_path: Path) -> None:
    pdb = tmp_path / "in.pdb"
    pdb.write_text(PDB)
    assert _run(bindir, str(pdb), env_extra={"FAKE_RC": "3"}).returncode == 3


def test_mmcif_input_is_not_rewritten(bindir: Path, tmp_path: Path) -> None:
    cif = tmp_path / "in.cif"
    cif.write_text("data_x\nREMARK keep me\n")
    r = _run(bindir, str(cif))
    assert "data_x" in r.stdout and "REMARK keep me" in r.stdout


def test_temp_copy_is_removed(bindir: Path, tmp_path: Path) -> None:
    pdb = tmp_path / "in.pdb"
    pdb.write_text(PDB)
    tmpdir = tmp_path / "tmp"
    tmpdir.mkdir()
    _run(bindir, str(pdb), env_extra={"TMPDIR": str(tmpdir)})
    assert list(tmpdir.iterdir()) == []
