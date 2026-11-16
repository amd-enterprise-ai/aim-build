# Copyright © Advanced Micro Devices, Inc., or its affiliates.
#
# SPDX-License-Identifier: MIT

"""Naming contract: canonicalName, variants, aim_id and model_id equal the asset path (lowercase)."""

from __future__ import annotations

from pathlib import Path

import yaml

ASSET = Path(__file__).resolve().parents[2]
ASSET_PATH = "molecularai/reinvent4"


def test_asset_path_matches_contract() -> None:
    assert ASSET.parent.name + "/" + ASSET.name == ASSET_PATH


def test_metadata_names_equal_asset_path() -> None:
    meta = yaml.safe_load((ASSET / "metadata.yaml").read_text(encoding="utf-8"))
    model = next(iter(meta.values()))
    while "canonicalName" not in model:
        model = next(v for v in model.values() if isinstance(v, dict))
    assert model["canonicalName"] == ASSET_PATH
    assert model["variants"] == [ASSET_PATH]
    # the upstream capitalisation stays in `source`
    assert "MolecularAI/REINVENT4" in model["source"]


def test_profile_ids_equal_asset_path() -> None:
    profiles = sorted((ASSET / "profiles").glob("*.yaml"))
    assert profiles
    for p in profiles:
        d = yaml.safe_load(p.read_text(encoding="utf-8"))
        assert d["aim_id"] == ASSET_PATH, p.name
        assert d["model_id"] == ASSET_PATH, p.name
