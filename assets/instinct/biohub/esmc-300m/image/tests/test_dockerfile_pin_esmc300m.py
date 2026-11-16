# Copyright © Advanced Micro Devices, Inc., or its affiliates.
#
# SPDX-License-Identifier: MIT

"""Text guard: the upstream FROM of the Layer 1 image is pinned by digest (.github/README-CI.md)."""

from __future__ import annotations

import re
from pathlib import Path

DOCKERFILE = Path(__file__).resolve().parents[1] / "Dockerfile"


def test_upstream_from_is_pinned_by_digest() -> None:
    froms = [line for line in DOCKERFILE.read_text(encoding="utf-8").splitlines() if line.startswith("FROM ")]
    assert froms, "no FROM line"
    for line in froms:
        assert re.fullmatch(r"FROM \S+:\S+@sha256:[0-9a-f]{64}", line.strip()), line
