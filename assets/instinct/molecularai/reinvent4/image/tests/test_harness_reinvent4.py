# Copyright © Advanced Micro Devices, Inc., or its affiliates.
#
# SPDX-License-Identifier: MIT

"""Static guards for the REINVENT4 harness and the service source.

Neither ``harness.py`` nor ``service.py`` is imported: they need aim_runtime,
bentoml, torch and reinvent4. The harness is inspected through its AST and the
service source is only read as text.
"""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

IMAGE_DIR = Path(__file__).resolve().parents[1]
HARNESS_PATH = IMAGE_DIR / "src/harness.py"
SERVICE_PATH = IMAGE_DIR / "src/service.py"

EXPECTED_CHECKS = [
    "cli_importable",
    "cli_executable",
    "gpu_visible",
    "service_health",
    "sampling_smoke",
    "smiles_validity",
    "throughput_smiles_per_min",
]


@pytest.fixture
def harness_tree() -> ast.Module:
    return ast.parse(HARNESS_PATH.read_text(encoding="utf-8"))


@pytest.fixture
def service_source() -> str:
    return SERVICE_PATH.read_text(encoding="utf-8")


def test_checks_names(harness_tree: ast.Module) -> None:
    cls = next(n for n in harness_tree.body if isinstance(n, ast.ClassDef) and n.name == "REINVENT4Harness")
    checks = next(
        n.value
        for n in cls.body
        if isinstance(n, ast.AnnAssign) and isinstance(n.target, ast.Name) and n.target.id == "CHECKS"
    )
    names = [c.args[0].value for c in checks.elts]
    assert names == EXPECTED_CHECKS


def test_every_emitted_check_is_declared_in_checks(harness_tree: ast.Module) -> None:
    """CI dashboards drop results whose name is not in CHECKS, so every CheckResult name must be declared."""
    emitted = {
        kw.value.value
        for node in ast.walk(harness_tree)
        if isinstance(node, ast.Call) and getattr(node.func, "id", None) == "CheckResult"
        for kw in node.keywords
        if kw.arg == "name" and isinstance(kw.value, ast.Constant)
    }
    assert emitted, "no CheckResult(name=...) found: the guard would pass vacuously"
    assert emitted <= set(EXPECTED_CHECKS), f"emitted but not declared: {sorted(emitted - set(EXPECTED_CHECKS))}"


def test_health_check_polls_health_then_healthz() -> None:
    assert '("/health", "/healthz")' in HARNESS_PATH.read_text(encoding="utf-8")


def test_no_silent_except_pass(harness_tree: ast.Module) -> None:
    for node in ast.walk(harness_tree):
        if isinstance(node, ast.ExceptHandler):
            assert not (len(node.body) == 1 and isinstance(node.body[0], ast.Pass)), f"line {node.lineno}"


def test_docstrings_use_unique_molecules(service_source: str) -> None:
    assert "unique_sequences" not in HARNESS_PATH.read_text(encoding="utf-8").split('"""')[1]
    assert "unique_sequences" not in service_source.split('"""')[1]


def test_service_samples_in_process_without_a_subprocess_or_csv(service_source: str) -> None:
    assert "import subprocess" not in service_source and "subprocess.run" not in service_source
    assert "samples.csv" not in service_source and "_parse_smiles_csv" not in service_source
    assert "produced no SMILES" in service_source
    assert "stderr" not in service_source.split("raise InternalServerError")[1].split("\n")[0]
