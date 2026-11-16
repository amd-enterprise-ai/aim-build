# Copyright © Advanced Micro Devices, Inc., or its affiliates.
#
# SPDX-License-Identifier: MIT

"""The image puts ``/workspace/model/src`` on ``sys.path``; do the same so ``harness.py`` can import its siblings."""

from __future__ import annotations

import sys
from pathlib import Path

SRC_DIR = str(Path(__file__).resolve().parents[1] / "src")
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)
