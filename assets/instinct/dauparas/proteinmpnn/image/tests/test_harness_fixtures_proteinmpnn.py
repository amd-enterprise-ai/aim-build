# Copyright © Advanced Micro Devices, Inc., or its affiliates.
#
# SPDX-License-Identifier: MIT

"""Benchmark fixture guards: declared residue counts are computed, codes are valid."""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pytest

pytest.importorskip("aim_runtime")

HARNESS_PATH = Path(__file__).resolve().parents[1] / "src/harness.py"
STANDARD_RESIDUES = set("ALA ARG ASN ASP CYS GLN GLU GLY HIS ILE LEU LYS MET PHE PRO SER THR TRP TYR VAL".split())


@pytest.fixture(scope="module")
def harness():
    spec = importlib.util.spec_from_file_location("proteinmpnn_harness_under_test", HARNESS_PATH)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_declared_length_matches_fixture(harness) -> None:
    for name, declared, pdb in harness.BENCHMARK_PDBS:
        assert declared == harness.count_residues(pdb) > 0, name


def test_incomplete_trailing_residue_not_counted(harness) -> None:
    pdb = (
        "ATOM      1  N   GLY A   1       0.000   0.000   0.000  1.00  0.00           N\n"
        "ATOM      2  CA  GLY A   1       1.000   0.000   0.000  1.00  0.00           C\n"
        "ATOM      3  C   GLY A   1       2.000   0.000   0.000  1.00  0.00           C\n"
        "ATOM      4  N   GLY A   2       3.000   0.000   0.000  1.00  0.00           N\n"
    )
    assert harness.count_residues(pdb) == 1


def test_fixtures_use_standard_residue_names(harness) -> None:
    for name, _, pdb in harness.BENCHMARK_PDBS:
        names = {line[17:20] for line in pdb.splitlines() if line.startswith("ATOM")}
        assert names <= STANDARD_RESIDUES, (name, names - STANDARD_RESIDUES)


def test_inference_payload_is_nested_under_request(harness, monkeypatch: pytest.MonkeyPatch) -> None:
    seen = {}

    class _Resp:
        status = 200

        def __enter__(self):
            return self

        def __exit__(self, *exc):
            return False

        def read(self):
            return b"{}"

    def fake_urlopen(req, timeout=None):
        seen["body"] = json.loads(req.data)
        return _Resp()

    monkeypatch.setattr(harness, "urlopen", fake_urlopen)
    harness._post_inference("http://x", "/p.pdb", num_seq_per_target=2)
    assert seen["body"] == {"request": {"pdb_path": "/p.pdb", "num_seq_per_target": 2, "sampling_temp": 0.1}}


def _fixture_texts(harness) -> dict[str, str]:
    """Every module-level ``*_PDB`` / ``*_PDB_CONTENT`` string constant in the harness and its fixtures module."""
    import benchmark_fixtures  # noqa: PLC0415 (sibling of harness.py; tests/conftest.py puts src on sys.path)

    return {
        name: value
        for name, value in {**vars(benchmark_fixtures), **vars(harness)}.items()
        if isinstance(value, str) and (name.endswith("_PDB") or name.endswith("_PDB_CONTENT")) and "ATOM" in value
    }


def test_all_pdb_fixtures_are_discovered(harness) -> None:
    texts = _fixture_texts(harness)
    assert {
        "SYNTHETIC_A_PDB",
        "SYNTHETIC_B_PDB",
        "SYNTHETIC_C_PDB",
        "SYNTHETIC_D_PDB",
        "_SMOKE_PDB_CONTENT",
    } <= set(texts)


def test_all_pdb_fixtures_parse_with_fixed_width_columns(harness) -> None:
    """Mirror the float/int conversions of ProteinMPNN's parser; misaligned columns fail here."""
    for fixture, text in _fixture_texts(harness).items():
        residues: dict[tuple[str, int], set[str]] = {}
        for line in text.splitlines():
            if not line.startswith("ATOM"):
                continue
            where = (fixture, line)
            for start, end in ((30, 38), (38, 46), (46, 54), (54, 60), (60, 66)):
                try:
                    float(line[start:end])
                except ValueError:
                    pytest.fail(f"unparsable columns {start + 1}-{end}: {where}")
            chain = line[21]
            assert chain != " ", where
            resnum = int(line[22:26])
            residues.setdefault((chain, resnum), set()).add(line[12:16].strip())
        assert residues, fixture
        for key, atoms in residues.items():
            assert {"N", "CA", "C"} <= atoms, (fixture, key, atoms)


def test_fixtures_live_in_their_own_module_not_in_the_harness_source() -> None:
    harness_src = HARNESS_PATH.read_text(encoding="utf-8")
    assert "SYNTHETIC_A_PDB = " not in harness_src
    assert "def count_residues" not in harness_src
    fixtures_src = (HARNESS_PATH.parent / "benchmark_fixtures.py").read_text(encoding="utf-8")
    assert "SYNTHETIC_A_PDB = " in fixtures_src and "def count_residues" in fixtures_src
