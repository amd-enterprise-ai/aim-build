# Copyright © Advanced Micro Devices, Inc., or its affiliates.
#
# SPDX-License-Identifier: MIT

"""Fixtures for the REINVENT4 image tests: ``smiles_validation_pool`` imported against a stub ``rdkit``.

The pool spawns real worker processes. A spawned worker imports its function by module name and takes its
``sys.path`` from the parent, so while a test module runs, a stub ``rdkit`` package (deterministic, no chemistry) and the
service's ``src`` directory are put first on ``sys.path``. Everything is restored when the module's tests are done.
"""

from __future__ import annotations

import importlib
import sys
import textwrap
from pathlib import Path
from types import ModuleType
from typing import Iterator

import pytest

SRC_DIR = Path(__file__).resolve().parents[1] / "src"

STUB_CHEM = """
import os
import time


class Mol:
    def __init__(self, smiles):
        self.smiles = smiles

    def __bool__(self):
        return True


def MolFromSmiles(smiles, sanitize=True):
    if smiles == "BOOM":
        raise ValueError("boom from the worker")
    if smiles == "DIE":
        os._exit(3)
    if smiles == "HANG":
        time.sleep(300)
    if smiles.startswith("(") or (sanitize and smiles.startswith("QQ")):
        return None
    return Mol(smiles)


def SanitizeMol(mol, catchErrors=False):
    return 1 if mol.smiles.startswith("X") else 0


def MolToSmiles(mol, canonical=True, isomericSmiles=False):
    return mol.smiles.upper() + ("!" if isomericSmiles else "")
"""

STUB_RDBASE = """
import os


def DisableLog(pattern):
    if os.environ.get("STUB_RDKIT_FAIL_INIT"):
        raise RuntimeError("stub rdkit init failure")
"""


@pytest.fixture(scope="module")
def stub_rdkit_dir(tmp_path_factory: pytest.TempPathFactory) -> Path:
    root = tmp_path_factory.mktemp("stub_rdkit")
    package = root / "rdkit"
    package.mkdir()
    (package / "__init__.py").write_text("from . import Chem, rdBase\n")
    (package / "Chem.py").write_text(textwrap.dedent(STUB_CHEM))
    (package / "rdBase.py").write_text(textwrap.dedent(STUB_RDBASE))
    return root


@pytest.fixture(scope="module")
def pool_module(stub_rdkit_dir: Path) -> Iterator[ModuleType]:
    """``smiles_validation_pool`` bound to the stub rdkit; spawned workers see the same ``sys.path``."""
    pytest.importorskip("numpy")
    saved_modules = {name: mod for name, mod in sys.modules.items() if name == "rdkit" or name.startswith("rdkit.")}
    saved_modules.update(
        {"smiles_validation_pool": sys.modules["smiles_validation_pool"]}
        if "smiles_validation_pool" in sys.modules
        else {}
    )
    for name in saved_modules:
        del sys.modules[name]
    saved_path = list(sys.path)
    sys.path[:0] = [str(stub_rdkit_dir), str(SRC_DIR)]
    try:
        yield importlib.import_module("smiles_validation_pool")
    finally:
        sys.path[:] = saved_path
        for name in [n for n in sys.modules if n == "rdkit" or n.startswith("rdkit.") or n == "smiles_validation_pool"]:
            del sys.modules[name]
        sys.modules.update(saved_modules)
