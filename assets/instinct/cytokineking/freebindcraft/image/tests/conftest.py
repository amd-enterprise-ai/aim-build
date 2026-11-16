# Copyright © Advanced Micro Devices, Inc., or its affiliates.
#
# SPDX-License-Identifier: MIT

"""Shared fixtures for the FreeBindCraft job tests.

The helper modules are named ``fbc_*`` so they can be imported plainly. The
entrypoint is loaded by file path under a unique name: ``entrypoint`` is also the
name of a module in the aim-build ``src`` tree, so a plain import could pick the
wrong one.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import ModuleType

import pytest

TESTS_DIR = Path(__file__).resolve().parent
SRC_DIR = TESTS_DIR.parent / "src"
REFERENCES = TESTS_DIR / "references"


def pytest_configure(config: pytest.Config) -> None:
    config.addinivalue_line(
        "markers", "gpu: needs the built image and an AMD GPU (run through smoke_gpu, not the CPU pre-gate)"
    )
    config.addinivalue_line("markers", "integration: GPU environment required (registered here for standalone runs)")
    if str(SRC_DIR) not in sys.path:
        sys.path.insert(0, str(SRC_DIR))  # the fbc_* modules have collision-free names


def _load_entrypoint() -> ModuleType:
    if str(SRC_DIR) not in sys.path:
        sys.path.insert(0, str(SRC_DIR))
    spec = importlib.util.spec_from_file_location("fbc_job_entrypoint", SRC_DIR / "entrypoint.py")
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules["fbc_job_entrypoint"] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="session")
def entrypoint() -> ModuleType:
    return _load_entrypoint()


@pytest.fixture(scope="session")
def src_dir() -> Path:
    return SRC_DIR


@pytest.fixture()
def target_pdb() -> Path:
    return REFERENCES / "1PGB_A.pdb"
