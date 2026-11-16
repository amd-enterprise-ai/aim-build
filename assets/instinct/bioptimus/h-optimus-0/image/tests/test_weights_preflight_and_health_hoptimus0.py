# Copyright © Advanced Micro Devices, Inc., or its affiliates.
#
# SPDX-License-Identifier: MIT

"""Weights pre-flight and /health-only readiness guards.

Never imports service.py (heavy deps); the pre-flight function is extracted
from its source and executed in isolation.
"""

from __future__ import annotations

import ast
import logging
import os
from pathlib import Path

import pytest

IMAGE_DIR = Path(__file__).resolve().parents[1]
SERVICE_SRC = (IMAGE_DIR / "src/service.py").read_text(encoding="utf-8")
HARNESS_SRC = (IMAGE_DIR / "src/harness.py").read_text(encoding="utf-8")


def _load_preflight():
    tree = ast.parse(SERVICE_SRC)
    wanted = {"_hub_snapshot_dir", "_preflight_weights"}
    nodes = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name in wanted]
    assert {n.name for n in nodes} == wanted
    ns = {"os": os, "Path": Path, "LOGGER": logging.getLogger("t"), "MODEL_ID": "bioptimus/H-optimus-0"}
    exec(compile(ast.Module(body=nodes, type_ignores=[]), "service_preflight", "exec"), ns)  # noqa: S102
    return ns["_preflight_weights"]


@pytest.fixture(autouse=True)
def _no_token(monkeypatch):
    monkeypatch.delenv("HF_TOKEN", raising=False)
    monkeypatch.delenv("HUGGING_FACE_HUB_TOKEN", raising=False)


def test_missing_weights_and_token_stops_naming_path_and_token(tmp_path):
    with pytest.raises(RuntimeError) as err:
        _load_preflight()(tmp_path)
    assert str(tmp_path) in str(err.value)
    assert "HF_TOKEN" in str(err.value)
    assert "AIM_CACHE_PATH" in str(err.value)


def test_token_set_allows_download(tmp_path, monkeypatch, caplog):
    monkeypatch.setenv("HF_TOKEN", "x")
    with caplog.at_level(logging.INFO, logger="t"):
        _load_preflight()(tmp_path)
    assert str(tmp_path) in caplog.text


def test_cached_snapshot_needs_no_token(tmp_path):
    (tmp_path / "models--bioptimus--H-optimus-0" / "snapshots" / "abc").mkdir(parents=True)
    _load_preflight()(tmp_path)


def test_service_runs_preflight_and_wraps_download_failure():
    tree = ast.parse(SERVICE_SRC)
    init = next(n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef) and n.name == "__init__")
    calls = [c.func.id for c in ast.walk(init) if isinstance(c, ast.Call) and isinstance(c.func, ast.Name)]
    assert "_preflight_weights" in calls
    tries = [t for t in ast.walk(init) if isinstance(t, ast.Try)]
    assert tries and any("RuntimeError" in ast.dump(t) for t in tries)
    assert "cache_dir=str(cache_root)" in SERVICE_SRC


def test_harness_readiness_polls_only_health():
    assert "/healthz" not in HARNESS_SRC.replace("/healthz)", "")
    assert "BENTOML_HEALTHZ_ENDPOINT" not in HARNESS_SRC
    assert 'HEALTH_ENDPOINT = "/health"' in HARNESS_SRC
