# Copyright © Advanced Micro Devices, Inc., or its affiliates.
#
# SPDX-License-Identifier: MIT

"""The licence-driven removal of the GPL-2.0 openbabel wheel from the SemlaFlow image.

Checks that the Dockerfile no longer installs it, that ``patches/drop_openbabel.py`` rewrites the two
upstream modules so neither imports it, and that nothing the AIM ships imports it. Pure text/AST work;
no upstream checkout, RDKit or torch is needed.
"""

from __future__ import annotations

import ast
import importlib.util
from pathlib import Path

import pytest

IMAGE = Path(__file__).resolve().parents[1]
PATCH = IMAGE / "patches" / "drop_openbabel.py"

UPSTREAM_SCRIPTUTIL = """import numpy as np
import torch
from openbabel import pybel
from rdkit import RDLogger


def disable_lib_stdout():
    pybel.ob.obErrorLog.StopLogging()
    RDLogger.DisableLog("rdApp.*")


def other():
    return 1
"""

UPSTREAM_RDKIT = """import numpy as np
from openbabel import pybel
from rdkit import Chem


def mol_from_atoms(coords, bonds=None):
    if bonds is None:
        return _infer_bonds(coords)
    return coords


def _infer_bonds(mol: Chem.rdchem.Mol):
    pybel_mol = pybel.readstring("xyz", "x")
    return pybel_mol.write("mol")
"""


@pytest.fixture(scope="module")
def patch_mod():
    spec = importlib.util.spec_from_file_location("drop_openbabel", PATCH)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _imports_openbabel(tree: ast.AST) -> bool:
    for n in ast.walk(tree):
        if isinstance(n, ast.Import) and any(a.name.split(".")[0] == "openbabel" for a in n.names):
            return True
        if isinstance(n, ast.ImportFrom) and (n.module or "").split(".")[0] == "openbabel":
            return True
    return False


def test_dockerfile_no_longer_installs_openbabel_pin() -> None:
    lines = (IMAGE / "Dockerfile").read_text(encoding="utf-8").splitlines()
    pins = [ln for ln in lines if "openbabel-wheel" in ln and not ln.lstrip().startswith("#")]
    assert not pins, pins


def test_dockerfile_applies_patch_after_checkout_and_asserts_absence() -> None:
    d = (IMAGE / "Dockerfile").read_text(encoding="utf-8")
    assert d.index("git checkout -q FETCH_HEAD") < d.index("drop_openbabel.py /opt/semlaflow")
    assert "COPY patches/drop_openbabel.py" in d
    assert "! grep -rnE '(from|import) +openbabel|pybel' /opt/semlaflow" in d
    assert "sys.modules['openbabel'] = None" in d and "semlaflow.predict" in d


def test_scriptutil_patch_removes_import_and_stoplogging(patch_mod) -> None:
    out = patch_mod.patch_scriptutil(UPSTREAM_SCRIPTUTIL)
    tree = ast.parse(out)
    assert not _imports_openbabel(tree)
    assert "pybel" not in out and "openbabel" not in out
    fn = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "disable_lib_stdout")
    assert [ast.unparse(s) for s in fn.body] == ["RDLogger.DisableLog('rdApp.*')"]
    assert "def other" in out


def test_rdkit_util_patch_uses_rdkit_lazily(patch_mod) -> None:
    out = patch_mod.patch_rdkit_util(UPSTREAM_RDKIT)
    tree = ast.parse(out)
    assert not _imports_openbabel(tree)
    assert "pybel" not in out and "openbabel" not in out
    assert not any(
        isinstance(n, ast.ImportFrom) and n.module == "rdkit.Chem" and n.names[0].name == "rdDetermineBonds"
        for n in tree.body
    ), "rdDetermineBonds must be imported lazily"
    fn = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "_infer_bonds")
    assert any(isinstance(n, ast.ImportFrom) and n.module == "rdkit.Chem" for n in ast.walk(fn))
    assert "DetermineBonds" in ast.unparse(fn)
    assert "mol_from_atoms" in out  # rest of the module kept


def test_patch_fails_loudly_when_upstream_changed(patch_mod) -> None:
    with pytest.raises(SystemExit):
        patch_mod.patch_scriptutil("import torch\n")
    with pytest.raises(SystemExit):
        patch_mod.patch_rdkit_util("from openbabel import pybel\n")


def test_patch_main_rewrites_tree(patch_mod, tmp_path: Path) -> None:
    (tmp_path / "semlaflow" / "util").mkdir(parents=True)
    (tmp_path / "semlaflow" / "scriptutil.py").write_text(UPSTREAM_SCRIPTUTIL)
    (tmp_path / "semlaflow" / "util" / "rdkit.py").write_text(UPSTREAM_RDKIT)
    patch_mod.main(tmp_path)
    for p in tmp_path.rglob("*.py"):
        assert "openbabel" not in p.read_text() and "pybel" not in p.read_text(), p


def test_nothing_the_aim_ships_imports_openbabel() -> None:
    for p in (IMAGE / "src").rglob("*.py"):
        assert not _imports_openbabel(ast.parse(p.read_text(encoding="utf-8"))), p
