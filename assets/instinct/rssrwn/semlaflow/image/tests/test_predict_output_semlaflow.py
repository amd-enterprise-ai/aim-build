# Copyright © Advanced Micro Devices, Inc., or its affiliates.
#
# SPDX-License-Identifier: MIT

"""predictions.smol -> SMILES conversion (service.py and harness.py are never imported)."""

from __future__ import annotations

import ast
import importlib.util
import re
from pathlib import Path

import pytest

SRC = Path(__file__).resolve().parents[1] / "src"
SERVICE_TREE = ast.parse((SRC / "service.py").read_text(encoding="utf-8"))
HARNESS_SRC = (SRC / "harness.py").read_text(encoding="utf-8")

_spec = importlib.util.spec_from_file_location("predict_output", SRC / "predict_output.py")
predict_output = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(predict_output)


def _inference() -> ast.FunctionDef:
    return next(n for n in ast.walk(SERVICE_TREE) if isinstance(n, ast.FunctionDef) and n.name == "inference")


# ---- service.py (AST) ----


def test_service_converts_in_memory_not_from_sdf_or_smol_text() -> None:
    src = ast.unparse(SERVICE_TREE)
    assert "collect_structures" in src and "rdkit_smiles" in src and "rdkit_molblock" in src
    assert "sdf_smiles" not in src and "SDF_FILE" not in src
    assert "splitlines" not in src and ".glob(" not in src and "read_bytes" not in src


def test_service_maps_run_failure_to_internal_server_error_without_details() -> None:
    engine = next(n for n in ast.walk(SERVICE_TREE) if isinstance(n, ast.ClassDef) and n.name == "_GenerationEngine")
    run = next(n for n in engine.body if isinstance(n, ast.FunctionDef) and n.name == "_run")
    handler_src = "\n".join(ast.unparse(h) for t in ast.walk(run) if isinstance(t, ast.Try) for h in t.handlers)
    assert "logger.exception" in handler_src and "GenerationError" in handler_src
    assert "str(" not in handler_src and "from None" in handler_src  # no details to the client
    generate = next(n for n in engine.body if isinstance(n, ast.FunctionDef) and n.name == "generate")
    assert "InternalServerError" in ast.unparse(generate)


def test_service_response_keys_are_existing_plus_counts() -> None:
    ret = next(n for n in ast.walk(_inference()) if isinstance(n, ast.Return))
    keys = [k.value for k in ret.value.keys]
    assert keys == ["smiles", "molblocks", "n_molecules", "num_requested", "num_valid", "num_invalid", "elapsed_s"]
    values = {k.value: ast.unparse(v) for k, v in zip(ret.value.keys, ret.value.values)}
    assert values["num_requested"] == "n_molecules"
    assert values["num_valid"] == "len(smiles)"
    assert values["num_invalid"] == "n_invalid"


def test_collect_structures_keeps_smiles_and_molblocks_aligned() -> None:
    smiles, blocks, n_invalid = predict_output.collect_structures(
        ["CCO", None, "boom", "", "c1ccccc1"], _to_smiles, lambda m: f"block-{m}", 7
    )
    assert smiles == ["CCO", "c1ccccc1"] and blocks == ["block-CCO", "block-c1ccccc1"]
    assert n_invalid == 5


def test_collect_structures_drops_a_molecule_without_a_molblock() -> None:
    smiles, blocks, n_invalid = predict_output.collect_structures(
        ["CC", "CN"], _to_smiles, lambda m: None if m == "CN" else "blk", 2
    )
    assert smiles == ["CC"] and blocks == ["blk"] and n_invalid == 1
    with pytest.raises(predict_output.PredictOutputError):
        predict_output.collect_structures([None], _to_smiles, lambda m: "b", 1)


def test_rdkit_molblock_writes_the_generated_molecule() -> None:
    src = (SRC / "predict_output.py").read_text(encoding="utf-8")
    assert "def rdkit_molblock" in src and "Chem.MolToMolBlock(mol)" in src


def test_count_identity_valid_plus_invalid_is_requested() -> None:
    for requested, records in ((5, ["CC", None, "CN"]), (3, ["CC", "CN", "CO"])):
        smiles, n_invalid = predict_output.collect_smiles(iter(records), _to_smiles, requested)
        assert len(smiles) + n_invalid == requested


def test_dockerfile_copies_src_dir_with_helper() -> None:
    dockerfile = (SRC.parent / "Dockerfile").read_text(encoding="utf-8")
    assert "COPY src/ /workspace/model/src/" in dockerfile and "/workspace/model/src" in dockerfile.split("printf")[1]


# ---- helper logic with fake molecules ----


def _to_smiles(mol):
    if mol == "boom":
        raise ValueError("bad")
    return mol


def test_collect_smiles_counts_missing_and_unconvertible() -> None:
    smiles, n_invalid = predict_output.collect_smiles(["CCO", None, "boom", "", "c1ccccc1"], _to_smiles, 7)
    assert smiles == ["CCO", "c1ccccc1"] and n_invalid == 5  # 7 requested, 2 returned


def test_all_requested_returned_has_no_invalid() -> None:
    smiles, n_invalid = predict_output.collect_smiles(iter(["CC", "CN"]), _to_smiles, 2)
    assert smiles == ["CC", "CN"] and n_invalid == 0


def test_no_convertible_molecule_raises() -> None:
    with pytest.raises(predict_output.PredictOutputError):
        predict_output.collect_smiles([], _to_smiles, 1)
    with pytest.raises(predict_output.PredictOutputError):
        predict_output.collect_smiles([None, "boom"], _to_smiles, 2)


def test_rdkit_smiles_matches_the_sdf_read_settings() -> None:
    src = (SRC / "predict_output.py").read_text(encoding="utf-8")
    # Same RDKit settings the old SDMolSupplier read used: sanitise, remove hydrogens.
    assert "sanitize=True" in src and "removeHs=True" in src and "MolToMolBlock" in src


# ---- harness SMILES checks (extracted by AST; harness imports aim_runtime) ----


@pytest.fixture(scope="module")
def harness_ns() -> dict:
    tree = ast.parse(HARNESS_SRC)
    keep = [
        n
        for n in tree.body
        if (isinstance(n, ast.FunctionDef) and n.name in {"_count_smiles_like", "_smiles_problem"})
        or (isinstance(n, ast.Assign) and getattr(n.targets[0], "id", "") == "_SMILES_RE")
    ]
    ns: dict = {"re": re}
    exec(compile(ast.Module(keep, []), "harness_part", "exec"), ns)  # noqa: S102
    return ns


def test_harness_accepts_real_smiles(harness_ns) -> None:
    ok = {"smiles": ["CC(=O)Oc1ccccc1C(=O)O", "C[N+](C)(C)C", "[H]/C=C/Cl", "c1ccccc1"], "n_molecules": 4}
    assert harness_ns["_smiles_problem"](ok) is None


def test_harness_rejects_binary_garbage_and_count_mismatch(harness_ns) -> None:
    bad = {"smiles": ["�\x04]�(B~@\x01", "torch._utils _rebuild", "CCO"], "n_molecules": 3}
    assert "not SMILES" in harness_ns["_smiles_problem"](bad)
    assert harness_ns["_smiles_problem"]({"smiles": ["CCO"], "n_molecules": 2}) is not None
    assert harness_ns["_smiles_problem"]({"smiles": [""], "n_molecules": 1}) is not None
    assert harness_ns["_smiles_problem"]({"n_molecules": 1}) is not None
