# Copyright © Advanced Micro Devices, Inc., or its affiliates.
#
# SPDX-License-Identifier: MIT

"""Input validation, accelerator detection, weights location and the resume guard."""

from __future__ import annotations

import shutil
from pathlib import Path

import fbc_preflight as preflight
import fbc_settings as settings
import pytest


def _input_dir(tmp_path: Path, target_pdb: Path) -> Path:
    d = tmp_path / "in"
    d.mkdir()
    shutil.copy(target_pdb, d / "target.pdb")
    return d


def test_reference_target_is_a_real_folded_chain(target_pdb: Path) -> None:
    # 1PGB chain A: 56 residues, one chain.
    assert preflight.target_chains(target_pdb) == {"A"}
    assert (
        sum(
            1 for line in target_pdb.read_text().splitlines() if line.startswith("ATOM") and line[12:16].strip() == "CA"
        )
        == 56
    )


def test_valid_target_is_returned(tmp_path: Path, target_pdb: Path) -> None:
    d = _input_dir(tmp_path, target_pdb)
    assert preflight.validate_target(d, settings.Settings()).name == "target.pdb"


def test_missing_target_fails(tmp_path: Path) -> None:
    with pytest.raises(preflight.PreflightError, match="target.pdb not found"):
        preflight.validate_target(tmp_path, settings.Settings())


def test_empty_and_non_protein_targets_fail(tmp_path: Path) -> None:
    (tmp_path / "target.pdb").write_text("")
    with pytest.raises(preflight.PreflightError, match="empty"):
        preflight.validate_target(tmp_path, settings.Settings())
    (tmp_path / "target.pdb").write_text("this is not a pdb\n")
    with pytest.raises(preflight.PreflightError, match="no protein ATOM"):
        preflight.validate_target(tmp_path, settings.Settings())


def test_missing_chain_and_hotspot_chain_fail(tmp_path: Path, target_pdb: Path) -> None:
    d = _input_dir(tmp_path, target_pdb)
    with pytest.raises(preflight.PreflightError, match="chain\\(s\\) B"):
        preflight.validate_target(d, settings.Settings(target_chains="A,B"))
    with pytest.raises(preflight.PreflightError, match="hotspot 'C5'"):
        preflight.validate_target(d, settings.Settings(hotspots="C5"))
    assert preflight.validate_target(d, settings.Settings(hotspots="A5,12"))  # bare numbers carry no chain


def _dev(tmp_path: Path, kfd: bool, render: int) -> Path:
    dev = tmp_path / "dev"
    (dev / "dri").mkdir(parents=True)
    if kfd:
        (dev / "kfd").write_text("")
    for i in range(render):
        (dev / "dri" / f"renderD{128 + i}").write_text("")
    (dev / "dri" / "card0").write_text("")  # card nodes are not counted
    return dev


def test_accelerators_are_counted_from_device_nodes(tmp_path: Path) -> None:
    assert preflight.detect_accelerators({}, _dev(tmp_path, True, 2)) == 2


def test_no_kfd_means_no_accelerator(tmp_path: Path) -> None:
    assert preflight.detect_accelerators({}, _dev(tmp_path, False, 2)) == 0


def test_count_override_wins_over_detection(tmp_path: Path) -> None:
    assert preflight.detect_accelerators({"AIM_ACCELERATOR_COUNT": "4"}, _dev(tmp_path, False, 0)) == 4
    with pytest.raises(preflight.PreflightError):
        preflight.detect_accelerators({"AIM_ACCELERATOR_COUNT": "many"}, tmp_path)
    with pytest.raises(preflight.PreflightError):
        preflight.detect_accelerators({"AIM_ACCELERATOR_COUNT": "-1"}, tmp_path)


def test_too_few_accelerators_fails_with_status_3(tmp_path: Path) -> None:
    with pytest.raises(preflight.PreflightError) as info:
        preflight.require_accelerators({}, _dev(tmp_path, True, 0))
    assert info.value.status == 3 and "found 0" in str(info.value)


def test_af_params_dir_prefers_a_populated_cache(tmp_path: Path) -> None:
    cache = tmp_path / "cache"
    (cache / "params").mkdir(parents=True)
    with pytest.raises(preflight.PreflightError):  # an empty cache is not usable
        preflight.resolve_af_params_dir({"AIM_CACHE_PATH": str(cache)}, tmp_path / "app")
    (cache / "params" / "params_model_1_multimer_v3.npz").write_text("")
    assert preflight.resolve_af_params_dir({"AIM_CACHE_PATH": str(cache)}, tmp_path / "app") == str(cache)


def test_af_params_missing_is_an_error_that_names_the_mount(tmp_path: Path) -> None:
    for env in ({}, {"AIM_CACHE_PATH": str(tmp_path / "nowhere")}):
        with pytest.raises(preflight.PreflightError) as info:
            preflight.resolve_af_params_dir(env, tmp_path / "app")
        assert info.value.status == 2
        assert "params/params_model_1_multimer_v3.npz" in str(info.value) and "AIM_CACHE_PATH" in str(info.value)


def test_af_params_baked_copy_is_still_honoured(tmp_path: Path) -> None:
    (tmp_path / "app" / "params").mkdir(parents=True)
    (tmp_path / "app" / "params" / "params_model_1_multimer_v3.npz").write_text("")
    assert preflight.resolve_af_params_dir({}, tmp_path / "app") == str(tmp_path / "app")


def test_resume_guard(tmp_path: Path, target_pdb: Path) -> None:
    work = tmp_path / ".work"
    fp = preflight.input_fingerprint(target_pdb, settings.Settings())
    assert preflight.check_resume_guard(work, fp) is False  # first run records it
    assert preflight.check_resume_guard(work, fp) is True  # same input resumes
    other = preflight.input_fingerprint(target_pdb, settings.Settings(hotspots="A5"))
    with pytest.raises(preflight.PreflightError, match="different target"):
        preflight.check_resume_guard(work, other)


def test_hotspot_report_counts_residue_numbers_present_in_the_target(target_pdb: Path) -> None:
    # 1PGB_A has residues 2 to 56 on chain A; the report is informational and never raises.
    assert preflight.hotspot_report(target_pdb, settings.Settings()) is None
    assert preflight.hotspot_report(target_pdb, settings.Settings(hotspots="A")) is None  # a chain letter alone
    assert (
        preflight.hotspot_report(target_pdb, settings.Settings(hotspots="A5,10-12"))
        == "hotspots: 4 of 4 residue numbers found in target.pdb"
    )
    line = preflight.hotspot_report(target_pdb, settings.Settings(hotspots="A5,A900,901-902"))
    assert line == "hotspots: 1 of 4 residue numbers found in target.pdb; not found: A900, 901, 902"
