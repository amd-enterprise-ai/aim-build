# Copyright © Advanced Micro Devices, Inc., or its affiliates.
#
# SPDX-License-Identifier: MIT

"""Turn the molecules the resident SemlaFlow model generates into SMILES strings.

``model._generate_mols`` returns RDKit molecules (``None`` where upstream could not build or
sanitise one). Upstream's own ``predict`` script writes them to an SDF file and the old service read
that file back with ``removeHs=True, sanitize=True``; ``rdkit_smiles`` does the same conversion in
memory (a mol-block round trip), so the SMILES are the ones the SDF route gave. RDKit is imported
inside the function that needs it so the pure logic is testable without RDKit.
"""

from __future__ import annotations

import logging
from collections.abc import Callable, Iterable
from typing import Any

logger = logging.getLogger(__name__)


class PredictOutputError(Exception):
    """No generated molecule yields a SMILES string."""


def collect_smiles(
    mols: Iterable[Any],
    to_smiles: Callable[[Any], str | None],
    n_requested: int,
) -> tuple[list[str], int]:
    """Return ``(smiles, n_invalid)``.

    ``n_invalid`` counts requested molecules that are not in the result: molecules upstream could
    not build (``None``) plus molecules that do not give a SMILES string.
    """
    smiles: list[str] = []
    for mol in mols:
        try:
            smi = to_smiles(mol) if mol is not None else None
        except Exception:  # noqa: BLE001 -- one bad molecule must not fail the request
            logger.debug("molecule conversion raised", exc_info=True)
            smi = None
        if isinstance(smi, str) and smi:
            smiles.append(smi)
    if not smiles:
        raise PredictOutputError("no generated molecule converts to SMILES")
    n_invalid = max(n_requested - len(smiles), 0)
    if n_invalid:
        logger.warning(
            "%d of %d requested molecules did not yield a valid SMILES and were left out", n_invalid, n_requested
        )
    return smiles, n_invalid


def collect_structures(
    mols: Iterable[Any],
    to_smiles: Callable[[Any], str | None],
    to_molblock: Callable[[Any], str | None],
    n_requested: int,
) -> tuple[list[str], list[str], int]:
    """Return ``(smiles, molblocks, n_invalid)``, the two lists aligned one to one.

    A molecule is kept only when it gives both a SMILES string and a mol block. ``n_invalid`` counts
    requested molecules that are not in the result, as in ``collect_smiles``.
    """
    smiles: list[str] = []
    molblocks: list[str] = []
    for mol in mols:
        try:
            smi = to_smiles(mol) if mol is not None else None
            block = to_molblock(mol) if smi else None
        except Exception:  # noqa: BLE001 -- one bad molecule must not fail the request
            logger.debug("molecule conversion raised", exc_info=True)
            smi = block = None
        if isinstance(smi, str) and smi and isinstance(block, str) and block:
            smiles.append(smi)
            molblocks.append(block)
    if not smiles:
        raise PredictOutputError("no generated molecule converts to SMILES")
    n_invalid = max(n_requested - len(smiles), 0)
    if n_invalid:
        logger.warning(
            "%d of %d requested molecules did not yield a valid SMILES and were left out", n_invalid, n_requested
        )
    return smiles, molblocks, n_invalid


def rdkit_molblock(mol: Any) -> str:
    """MolBlock text of a generated molecule with its 3D coordinates, as the model generated it (needs rdkit)."""
    from rdkit import Chem

    return Chem.MolToMolBlock(mol)


def rdkit_smiles(mol: Any) -> str | None:
    """Canonical SMILES of a generated molecule, hydrogens removed (needs rdkit).

    The mol-block round trip is what writing an SDF and reading it back does: it perceives
    stereochemistry from the 3D coordinates and removes explicit hydrogens with sanitisation.
    """
    from rdkit import Chem

    clean = Chem.MolFromMolBlock(Chem.MolToMolBlock(mol), sanitize=True, removeHs=True)
    return Chem.MolToSmiles(clean) if clean is not None else None
