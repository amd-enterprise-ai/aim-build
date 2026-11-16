# Copyright © Advanced Micro Devices, Inc., or its affiliates.
#
# SPDX-License-Identifier: MIT

"""Make upstream SemlaFlow (rssrwn/semla-flow) work without the GPL-2.0 openbabel wheel.

Run at image build time on the pinned upstream checkout:

    python3 drop_openbabel.py /opt/semlaflow

Upstream imports ``openbabel.pybel`` at module level in two files. The only functional
use is ``_infer_bonds`` (reached only when ``mol_from_atoms`` gets ``bonds=None``; the
served ``predict`` path always passes predicted bonds) and one log-silencing call in
``disable_lib_stdout``. The patch:

* scriptutil.py: drops the import; ``disable_lib_stdout`` only calls ``RDLogger.DisableLog``.
* util/rdkit.py: drops the import; ``_infer_bonds`` uses ``rdkit.Chem.rdDetermineBonds``
  (BSD-3, imported lazily).

Every replacement is an exact-string match and raises if upstream changed, so a SHA bump
fails the build instead of silently skipping the patch.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

IMPORT_LINE = "from openbabel import pybel\n"

DISABLE_OLD = 'def disable_lib_stdout():\n    pybel.ob.obErrorLog.StopLogging()\n    RDLogger.DisableLog("rdApp.*")\n'
DISABLE_NEW = 'def disable_lib_stdout():\n    RDLogger.DisableLog("rdApp.*")\n'

INFER_NEW = '''def _infer_bonds(mol: Chem.rdchem.Mol):
    """Infer bonds from the 3D coordinates with RDKit (RDKit rdDetermineBonds)."""
    from rdkit.Chem import rdDetermineBonds

    mol = Chem.Mol(mol)
    try:
        rdDetermineBonds.DetermineBonds(mol, charge=0)
        Chem.SanitizeMol(mol)
    except Exception:
        return None
    return mol
'''


def _replace_once(text: str, old: str, new: str, what: str) -> str:
    if text.count(old) != 1:
        raise SystemExit(f"drop_openbabel: expected exactly one {what}, found {text.count(old)}")
    return text.replace(old, new)


def patch_scriptutil(text: str) -> str:
    text = _replace_once(text, IMPORT_LINE, "", "openbabel import in scriptutil.py")
    return _replace_once(text, DISABLE_OLD, DISABLE_NEW, "disable_lib_stdout body")


def patch_rdkit_util(text: str) -> str:
    text = _replace_once(text, IMPORT_LINE, "", "openbabel import in util/rdkit.py")
    m = re.search(r"^def _infer_bonds\(.*?(?=^def |^class |\Z)", text, flags=re.S | re.M)
    if m is None:
        raise SystemExit("drop_openbabel: _infer_bonds not found")
    rest = text[m.end() :].lstrip("\n")
    return text[: m.start()] + INFER_NEW + ("\n\n" + rest if rest else "")


def main(root: Path) -> None:
    for rel, fn in (("semlaflow/scriptutil.py", patch_scriptutil), ("semlaflow/util/rdkit.py", patch_rdkit_util)):
        p = root / rel
        p.write_text(fn(p.read_text(encoding="utf-8")), encoding="utf-8")
        print(f"patched {p}")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        raise SystemExit("usage: drop_openbabel.py <semlaflow checkout>")
    main(Path(sys.argv[1]))
