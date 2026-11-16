# Copyright © Advanced Micro Devices, Inc., or its affiliates.
#
# SPDX-License-Identifier: MIT

"""The upstream ProteinMPNN ref in the Dockerfile must be a full commit SHA."""

from __future__ import annotations

import re
from pathlib import Path

DOCKERFILE = (Path(__file__).resolve().parents[1] / "Dockerfile").read_text(encoding="utf-8")


def test_proteinmpnn_ref_is_commit_sha() -> None:
    match = re.search(r"^ARG PROTEINMPNN_REF=(\S+)$", DOCKERFILE, re.MULTILINE)
    assert match, "ARG PROTEINMPNN_REF not found"
    assert re.fullmatch(r"[0-9a-f]{40}", match.group(1)), match.group(1)


def test_clone_checks_out_the_pinned_ref() -> None:
    assert "checkout ${PROTEINMPNN_REF}" in DOCKERFILE
    assert "--branch" not in DOCKERFILE


def test_clone_pruned_to_runtime_files() -> None:
    """Licence-driven V1: only the run script, its utils module and LICENSE remain."""
    assert "! -name protein_mpnn_run.py ! -name protein_mpnn_utils.py ! -name LICENSE" in DOCKERFILE
    assert "protein_mpnn_run.py --help" in DOCKERFILE
    assert "apt-get purge -y --auto-remove git" in DOCKERFILE
