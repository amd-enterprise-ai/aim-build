# Copyright © Advanced Micro Devices, Inc., or its affiliates.
#
# SPDX-License-Identifier: MIT

"""Parsing and input-validation tests for the ProteinMPNN service.

Fixtures follow the output format of upstream ``protein_mpnn_run.py`` at tag
v1.0.1: ``seqs/<name>.fa`` starts with the native sequence record (header
without ``T=``), then one ``>T=<temp>, sample=<n>, ...`` record per sample;
``scores/<name>.npz`` holds a ``score`` array with one value per sample only.
No model is run.
"""

from __future__ import annotations

import importlib.util
import math
import sys
import types
from pathlib import Path

import numpy as np
import pytest

bentoml = pytest.importorskip("bentoml")
from bentoml.exceptions import InvalidArgument  # noqa: E402

SERVICE_PATH = Path(__file__).resolve().parents[1] / "src/service.py"

NATIVE = ">1abc, score=1.2000, global_score=1.3000, fixed_chains=[], designed_chains=['A'], model_name=v_48_020, git_hash=x, seed=37"
FASTA = (
    f"{NATIVE}\nNATIVESEQ\n"
    ">T=0.1, sample=1, score=0.9000, global_score=1.0000, seq_recovery=0.5000\nAAAA\n"
    ">T=0.1, sample=2, score=0.8000, global_score=0.9000, seq_recovery=0.2500\nCC\nCC\n"
)


@pytest.fixture
def service(monkeypatch: pytest.MonkeyPatch):
    """Import service.py with the forwarded ``--arg port`` stubbed out."""
    monkeypatch.setattr(bentoml, "use_arguments", lambda _cls: types.SimpleNamespace(port=3000, accelerator_count=1))
    spec = importlib.util.spec_from_file_location("proteinmpnn_service_under_test", SERVICE_PATH)
    module = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, spec.name, module)
    spec.loader.exec_module(module)
    return module


def _write_output(out_dir: Path, name: str, fasta: str, scores: list[float] | None) -> None:
    (out_dir / "seqs").mkdir(exist_ok=True)
    (out_dir / "scores").mkdir(exist_ok=True)
    (out_dir / "seqs" / f"{name}.fa").write_text(fasta)
    if scores is not None:
        np.savez(out_dir / "scores" / f"{name}.npz", score=np.array(scores, np.float32), global_score=np.zeros(2))


def test_native_record_is_skipped_and_scores_align(service, tmp_path: Path) -> None:
    _write_output(tmp_path, "1abc", FASTA, [0.9, 0.8])
    response = service._parse_output(str(tmp_path))
    assert response.sequences == ["AAAA", "CCCC"]
    assert response.scores == pytest.approx([0.9, 0.8])
    assert response.seq_recovery == pytest.approx([0.5, 0.25])
    assert response.native_score == pytest.approx(1.2)


def test_multiple_structures_aggregate_in_sorted_order(service, tmp_path: Path) -> None:
    _write_output(tmp_path, "b", FASTA.replace("AAAA", "GGGG"), [0.1, 0.2])
    _write_output(tmp_path, "a", FASTA, [0.9, 0.8])
    response = service._parse_output(str(tmp_path))
    assert response.sequences == ["AAAA", "CCCC", "GGGG", "CCCC"]
    assert response.scores == pytest.approx([0.9, 0.8, 0.1, 0.2])
    assert response.seq_recovery == pytest.approx([0.5, 0.25, 0.5, 0.25])
    assert response.native_score == pytest.approx(1.2)  # first structure in file-name order ("a")


def test_missing_score_file_gives_nan_not_misalignment(service, tmp_path: Path) -> None:
    _write_output(tmp_path, "1abc", FASTA, None)
    response = service._parse_output(str(tmp_path))
    assert response.sequences == ["AAAA", "CCCC"]
    assert len(response.scores) == 2 and all(math.isnan(s) for s in response.scores)
    assert response.seq_recovery == pytest.approx([0.5, 0.25])  # parsed from the FASTA headers, not the score file


def test_empty_output_dir_returns_nothing(service, tmp_path: Path) -> None:
    response = service._parse_output(str(tmp_path))
    assert response.sequences == [] and response.scores == []
    assert response.seq_recovery == [] and math.isnan(response.native_score)


def test_header_values_missing_or_malformed_are_nan(service, tmp_path: Path) -> None:
    fasta = (
        ">1abc, fixed_chains=[], designed_chains=['A', 'B']\nNATIVESEQ\n"
        ">T=0.1, sample=1, score=0.9000, global_score=1.0000\nAAAA\n"
        ">T=0.1, sample=2, score=0.8000, seq_recovery=oops\nCCCC\n"
        ">T=0.1, sample=3, score=0.7000, seq_recovery=0.7500\nGGGG\n"
    )
    path = tmp_path / "x.fa"
    path.write_text(fasta)
    native_score, recoveries = service._read_header_values(str(path))
    assert math.isnan(native_score)
    assert len(recoveries) == 3 and math.isnan(recoveries[0]) and math.isnan(recoveries[1])
    assert recoveries[2] == pytest.approx(0.75)


def test_response_defaults_keep_old_clients_working(service) -> None:
    response = service.PredictResponse(sequences=["AAAA"], scores=[0.5])
    assert response.seq_recovery == [] and math.isnan(response.native_score)


def test_sampling_temp_must_be_positive(service) -> None:
    from pydantic import ValidationError

    with pytest.raises(ValidationError):
        service.PredictRequest(pdb_path="/x.pdb", sampling_temp=0.0)
    assert service.PredictRequest(pdb_path="/x.pdb", sampling_temp=0.1).sampling_temp == 0.1


def test_pdb_path_allow_list(service, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    root = tmp_path / "input"
    root.mkdir()
    outside = tmp_path / "secret.pdb"
    outside.write_text("ATOM\n")
    good = root / "a.pdb"
    good.write_text("ATOM\n")
    (root / "link.pdb").symlink_to(outside)
    monkeypatch.setattr(service, "INPUT_ROOT", root)
    monkeypatch.setattr(service, "FIXTURE_ROOT", tmp_path / "no-fixtures")

    assert service._resolve_pdb_path(str(good)) == good.resolve()
    for bad in (
        str(outside),
        str(root / ".." / "secret.pdb"),
        str(root / "link.pdb"),
        str(root / "missing.pdb"),
        str(root),
    ):
        with pytest.raises(InvalidArgument):
            service._resolve_pdb_path(bad)


def test_response_schema_is_json_compliant(service) -> None:
    """GET /docs.json serialises the response schema with allow_nan=False; a NaN default broke it."""
    import json

    for model in (
        (service.PredictResponse, service.PredictRequest)
        if hasattr(service, "PredictRequest")
        else (service.PredictResponse,)
    ):
        json.dumps(model.model_json_schema(), allow_nan=False)
    # Content is unchanged: an omitted native_score is still NaN.
    assert math.isnan(service.PredictResponse(sequences=[], scores=[]).native_score)
