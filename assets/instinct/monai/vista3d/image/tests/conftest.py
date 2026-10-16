# Copyright © Advanced Micro Devices, Inc., or its affiliates.
#
# SPDX-License-Identifier: MIT

import os
import pathlib
import sys

import pytest

# Make src/ importable when running tests from the image/ directory.
_SRC = pathlib.Path(__file__).parents[1] / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))


@pytest.fixture(scope="module")
def aim_data_dir():
    p = pathlib.Path(os.environ.get("AIM_DATA_DIR", "/workspace/data"))
    if not p.exists():
        pytest.skip(f"AIM_DATA_DIR missing: {p}")
    return p


@pytest.fixture(scope="module")
def service_url():
    return os.environ.get("AIM_SERVICE_URL", "http://localhost:8000")
