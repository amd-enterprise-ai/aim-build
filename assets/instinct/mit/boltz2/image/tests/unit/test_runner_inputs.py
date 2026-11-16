# Copyright © Advanced Micro Devices, Inc., or its affiliates.
#
# SPDX-License-Identifier: MIT

"""Input shaping and output parsing in ``runner.py`` — no model, no GPU."""

from __future__ import annotations

import json
import time
from pathlib import Path

import pytest
from runner import (
    DEFAULT_STEP_SCALE_ATTR,
    MSA_EMPTY,
    MsaServerTimeout,
    _DeadlineSleep,
    _parse_predictions,
    apply_single_sequence_mode,
    apply_step_scale,
    build_boltz_schema,
    is_msa_server_timeout,
    materialize_inline_inputs,
    record_default_step_scale,
    reject_file_references,
    sanitize_record_name,
)

# --------------------------------------------------------------------------- #
# Single-sequence mode
# --------------------------------------------------------------------------- #


def test_protein_without_msa_is_switched_to_single_sequence():
    """A bare protein chain defaults to "auto-generate MSA" in boltz's parser,
    which hard-fails when the MSA server is off. We fill in the sentinel."""
    out = apply_single_sequence_mode([{"protein": {"id": "A", "sequence": "NLYIQWLKDGGPSSGRPPPS"}}])
    assert out[0]["protein"]["msa"] == MSA_EMPTY


def test_explicit_msa_is_preserved():
    msa = ">query\nAAA\n>hit\nAAA\n"
    out = apply_single_sequence_mode([{"protein": {"id": "A", "sequence": "AAA", "msa": msa}}])
    assert out[0]["protein"]["msa"] == msa


# --------------------------------------------------------------------------- #
# Inline content only
# --------------------------------------------------------------------------- #

INLINE_MSA = ">query\nAAA\n>hit\nAAA\n"


def _protein(msa):
    return [{"protein": {"id": "A", "sequence": "AAA", "msa": msa}}]


@pytest.mark.parametrize("msa", ["/etc/passwd", "data/a.a3m", "../a.a3m", "file:///etc/hosts"])
def test_msa_file_paths_are_rejected(msa):
    with pytest.raises(ValueError, match="file paths are not accepted"):
        reject_file_references(_protein(msa), None)


def test_inline_and_empty_msa_are_accepted():
    reject_file_references(_protein(INLINE_MSA), None)
    reject_file_references(_protein(MSA_EMPTY), None)


@pytest.mark.parametrize("key", ["cif", "pdb"])
def test_template_file_paths_are_rejected(key):
    with pytest.raises(ValueError, match="file paths are not accepted"):
        reject_file_references(_protein(MSA_EMPTY), [{key: "/data/t.cif", "chain_id": "A"}])


def test_inline_template_is_accepted():
    reject_file_references(_protein(MSA_EMPTY), [{"cif": "data_x\n#\n", "chain_id": "A"}])


def test_inline_content_is_written_to_files_and_inputs_are_not_mutated(tmp_path):
    sequences = _protein(INLINE_MSA)
    templates = [{"cif": "data_x\n#\n", "chain_id": "A"}]
    out_seq, out_tpl = materialize_inline_inputs(sequences, templates, tmp_path)
    assert Path(out_seq[0]["protein"]["msa"]).read_text(encoding="utf-8") == INLINE_MSA
    assert Path(out_tpl[0]["cif"]).read_text(encoding="utf-8") == "data_x\n#\n"
    assert out_tpl[0]["chain_id"] == "A"
    assert sequences[0]["protein"]["msa"] == INLINE_MSA
    assert templates[0]["cif"] == "data_x\n#\n"


def test_non_protein_entities_are_untouched():
    """Only protein chains take an MSA; boltz sets ligand/dna/rna msa_id to -1."""
    entries = [{"ligand": {"id": "L", "smiles": "CC1=CC=CC=C1"}}, {"rna": {"id": "R", "sequence": "GCAUAGC"}}]
    assert apply_single_sequence_mode(entries) == entries


def test_input_is_not_mutated_in_place():
    """The request object is pydantic-owned; shaping must not write back into it."""
    original = [{"protein": {"id": "A", "sequence": "AAA"}}]
    apply_single_sequence_mode(original)
    assert "msa" not in original[0]["protein"]


# --------------------------------------------------------------------------- #
# Schema assembly
# --------------------------------------------------------------------------- #


def test_msa_server_requests_pass_sequences_through_unchanged():
    seqs = [{"protein": {"id": "A", "sequence": "AAA"}}]
    schema = build_boltz_schema(seqs, use_msa_server=True)
    assert schema["sequences"] == seqs
    assert schema["version"] == 1


def test_optional_blocks_are_omitted_when_absent():
    schema = build_boltz_schema([{"protein": {"id": "A", "sequence": "AAA"}}])
    assert set(schema) == {"version", "sequences"}


def test_optional_blocks_are_forwarded_when_present():
    schema = build_boltz_schema(
        [{"protein": {"id": "A", "sequence": "AAA"}}, {"ligand": {"id": "L", "smiles": "CCO"}}],
        constraints=[{"pocket": {"binder": "L", "contacts": [["A", 1]]}}],
        properties=[{"affinity": {"binder": "L"}}],
    )
    assert schema["constraints"][0]["pocket"]["binder"] == "L"
    assert schema["properties"][0]["affinity"]["binder"] == "L"


# --------------------------------------------------------------------------- #
# Record names reach the filesystem
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("smoke_test", "smoke_test"),
        ("../../etc/passwd", "etc_passwd"),
        ("a/b", "a_b"),
        ("", "prediction"),
        ("///", "prediction"),
        ("case-1", "case-1"),
    ],
)
def test_record_name_is_filesystem_safe(raw, expected):
    """boltz names the output directory after the input file's stem, so a
    caller-supplied name reaches the filesystem."""
    assert sanitize_record_name(raw) == expected


# --------------------------------------------------------------------------- #
# Output parsing
# --------------------------------------------------------------------------- #


def _write_prediction(root, record_id, rank, suffix, confidence=None):
    d = root / record_id
    d.mkdir(parents=True, exist_ok=True)
    stem = f"{record_id}_model_{rank}"
    (d / f"{stem}.{suffix}").write_text(f"ATOM  {record_id} rank{rank}\nEND\n")
    if confidence is not None:
        (d / f"confidence_{stem}.json").write_text(json.dumps(confidence))
    return stem


def test_parses_structures_and_confidence(tmp_path):
    preds = tmp_path / "predictions"
    stem = _write_prediction(preds, "case1", 0, "cif", {"confidence_score": 0.87, "ptm": 0.8})

    out = _parse_predictions(preds)

    assert out["error"] is False
    assert len(out["structures"]) == 1
    assert out["structures"][0]["record_id"] == stem
    assert out["structures"][0]["format"] == "mmcif"
    assert "ATOM" in out["structures"][0]["content"]
    assert out["confidence"][stem]["confidence_score"] == 0.87


def test_pdb_output_is_reported_as_pdb(tmp_path):
    preds = tmp_path / "predictions"
    _write_prediction(preds, "case1", 0, "pdb")
    assert _parse_predictions(preds)["structures"][0]["format"] == "pdb"


def test_best_ranked_sample_comes_first(tmp_path):
    """boltz ranks by descending confidence, so _model_0 is the best sample and
    must be the one /v1/inference returns."""
    preds = tmp_path / "predictions"
    _write_prediction(preds, "case1", 1, "cif")
    _write_prediction(preds, "case1", 0, "cif")
    assert _parse_predictions(preds)["structures"][0]["record_id"].endswith("_model_0")


def test_missing_confidence_file_is_tolerated(tmp_path):
    preds = tmp_path / "predictions"
    _write_prediction(preds, "case1", 0, "cif", confidence=None)
    out = _parse_predictions(preds)
    assert out["structures"] and out["confidence"] == {}


def test_absent_predictions_dir_yields_no_structures(tmp_path):
    out = _parse_predictions(tmp_path / "nope")
    assert out["structures"] == [] and out["confidence"] == {}


# --------------------------------------------------------------------------- #
# Per-request diffusion step scale
# --------------------------------------------------------------------------- #


class _FakeStructureModule:
    def __init__(self, step_scale):
        self.step_scale = step_scale


class _FakeModel:
    """Stand-in for the long-lived Boltz2 LightningModule the service reuses."""

    def __init__(self, step_scale=1.5):
        self.structure_module = _FakeStructureModule(step_scale)


def test_load_time_step_scale_is_recorded_from_the_model_not_assumed():
    model = _FakeModel(step_scale=1.5)
    assert record_default_step_scale(model, fallback=99.0) == 1.5
    assert getattr(model, DEFAULT_STEP_SCALE_ATTR) == 1.5


def test_recorded_default_falls_back_when_the_module_exposes_no_step_scale():
    """Boltz2DiffusionParams.step_scale (1.5 in boltz 2.2.1) is the fallback."""

    class _Bare:
        pass

    model = _Bare()
    assert record_default_step_scale(model, fallback=1.5) == 1.5


def test_request_step_scale_is_applied():
    model = _FakeModel(step_scale=1.5)
    record_default_step_scale(model, 1.5)
    assert apply_step_scale(model, 1.638) == 1.638
    assert model.structure_module.step_scale == 1.638


def test_step_scale_does_not_leak_into_the_next_request():
    """THE REGRESSION THIS GUARDS.

    The service loads the checkpoint once and reuses that object for every
    request, so assigning ``structure_module.step_scale`` only when a request
    supplies one leaves the previous request's value in place. The benchmark
    fixtures now send ``step_scale=1.638``; without this restore, one
    ``aim-runtime evaluate`` would permanently move the AIM's *served* step
    scale off boltz's own 1.5 for every later caller on that worker.
    """
    model = _FakeModel(step_scale=1.5)
    record_default_step_scale(model, 1.5)

    apply_step_scale(model, 1.638)  # a benchmark fixture request
    assert model.structure_module.step_scale == 1.638

    apply_step_scale(model, None)  # an ordinary serving request
    assert model.structure_module.step_scale == 1.5


def test_default_is_self_healed_when_it_was_never_recorded():
    """A model built by some other path still gets per-request semantics."""
    model = _FakeModel(step_scale=1.5)
    assert not hasattr(model, DEFAULT_STEP_SCALE_ATTR)

    apply_step_scale(model, 1.638)
    assert getattr(model, DEFAULT_STEP_SCALE_ATTR) == 1.5

    apply_step_scale(model, None)
    assert model.structure_module.step_scale == 1.5


def test_apply_step_scale_is_a_no_op_on_a_model_without_the_attribute():
    class _Bare:
        pass

    assert apply_step_scale(_Bare(), 1.638) is None


# --------------------------------------------------------------------------- #
# MSA deadline
# --------------------------------------------------------------------------- #


def test_deadline_sleep_raises_once_the_budget_is_gone():
    guard = _DeadlineSleep(time, time.monotonic() - 1.0)
    with pytest.raises(MsaServerTimeout):
        guard.sleep(5)


def test_deadline_sleep_clamps_to_the_remaining_budget():
    slept = []
    fake = type("T", (), {"monotonic": staticmethod(time.monotonic), "sleep": staticmethod(slept.append)})()
    guard = _DeadlineSleep(fake, time.monotonic() + 0.2)
    guard.sleep(30)
    assert slept and slept[0] <= 0.2


def test_deadline_sleep_proxies_other_time_attributes():
    """boltz's MSA client uses more of the time module than sleep."""
    guard = _DeadlineSleep(time, time.monotonic() + 60)
    assert isinstance(guard.monotonic(), float)


def test_msa_timeout_is_detected_through_a_wrapping_exception():
    """The failure surfaces from inside boltz's preprocessing, possibly wrapped."""
    try:
        try:
            raise MsaServerTimeout("budget gone")
        except MsaServerTimeout as inner:
            raise RuntimeError("preprocessing failed") from inner
    except RuntimeError as outer:
        assert is_msa_server_timeout(outer)


def test_unrelated_exception_is_not_an_msa_timeout():
    assert not is_msa_server_timeout(ValueError("something else"))


# --------------------------------------------------------------------------- #
# Accuracy metric
# --------------------------------------------------------------------------- #


def _helix(n, radius=2.3, rise=1.5, turn=1.75):
    """C-alpha coordinates on an alpha-helix-like curve.

    Deliberately not a straight line: collinear points are a degenerate case for
    rigid-body superposition, and gemmi's solver returns NaN for them (which is
    why _ca_rmsd screens non-finite results).
    """
    import math

    return [(radius * math.cos(i * turn), radius * math.sin(i * turn), i * rise) for i in range(n)]


def _synthetic_mmcif(coords, chain="A"):
    """Build a minimal poly-alanine mmCIF from a list of C-alpha coordinates."""
    gemmi = pytest.importorskip("gemmi")
    st = gemmi.Structure()
    st.name = "TEST"
    model = gemmi.Model("1")
    ch = gemmi.Chain(chain)
    for i, (x, y, z) in enumerate(coords, start=1):
        res = gemmi.Residue()
        res.name = "ALA"
        res.seqid = gemmi.SeqId(i, " ")
        atom = gemmi.Atom()
        atom.name = "CA"
        atom.element = gemmi.Element("C")
        atom.pos = gemmi.Position(x, y, z)
        atom.occ = 1.0
        res.add_atom(atom)
        ch.add_residue(res)
    model.add_chain(ch)
    st.add_model(model)
    st.setup_entities()
    doc = st.make_mmcif_document()
    return doc.as_string()


def test_ca_rmsd_is_zero_for_an_identical_structure(tmp_path):
    """A metric whose self-identity is not zero cannot be trusted for anything."""
    pytest.importorskip("gemmi")
    from harness import Boltz2Harness

    coords = _helix(12)
    text = _synthetic_mmcif(coords)
    ref = tmp_path / "ref.cif"
    ref.write_text(text)

    rmsd, aligned = Boltz2Harness._ca_rmsd(text, ref)
    assert rmsd == pytest.approx(0.0, abs=1e-6)
    assert aligned == len(coords)


def test_ca_rmsd_is_invariant_under_rigid_motion(tmp_path):
    """Superposition must remove translation, or every score is meaningless."""
    pytest.importorskip("gemmi")
    from harness import Boltz2Harness

    coords = _helix(12)
    ref = tmp_path / "ref.cif"
    ref.write_text(_synthetic_mmcif(coords))
    shifted = _synthetic_mmcif([(x + 100.0, y + 50.0, z - 25.0) for x, y, z in coords])

    rmsd, _ = Boltz2Harness._ca_rmsd(shifted, ref)
    assert rmsd == pytest.approx(0.0, abs=1e-6)


def test_ca_rmsd_grows_with_real_displacement(tmp_path):
    """A genuinely different conformation must not superpose away."""
    pytest.importorskip("gemmi")
    from harness import Boltz2Harness

    coords = _helix(12)
    ref = tmp_path / "ref.cif"
    ref.write_text(_synthetic_mmcif(coords))
    # Bend the chain: every other residue pushed off-axis.
    bent = _synthetic_mmcif([(x, y + (5.0 if i % 2 else -5.0), z) for i, (x, y, z) in enumerate(coords)])

    rmsd, _ = Boltz2Harness._ca_rmsd(bent, ref)
    assert rmsd > 1.0


def test_ca_rmsd_rejects_a_structure_with_no_polymer(tmp_path):
    pytest.importorskip("gemmi")
    from harness import Boltz2Harness

    ref = tmp_path / "ref.cif"
    ref.write_text(_synthetic_mmcif([(i * 3.8, 0.0, 0.0) for i in range(12)]))
    with pytest.raises(ValueError):
        Boltz2Harness._ca_rmsd("data_EMPTY\n#\n", ref)


def test_ca_rmsd_keeps_the_solver_limit_message_off_the_streams_and_reports_it(tmp_path, capfd):
    """A collinear chain makes gemmi print "More than 50 iterations needed!" from C++."""
    pytest.importorskip("gemmi")
    from harness import Boltz2Harness

    line = [(3.8 * i, 0.0, 0.0) for i in range(6)]
    text = _synthetic_mmcif(line)
    ref = tmp_path / "ref.cif"
    ref.write_text(text)
    notes: list[str] = []

    with pytest.raises(ValueError, match="no residues could be aligned"):
        Boltz2Harness._ca_rmsd(text, ref, notes=notes)

    captured = capfd.readouterr()
    assert "iterations needed" not in captured.out + captured.err
    assert len(notes) == 1 and "50-iteration limit for 1 chain pair(s)" in notes[0]


def test_ca_rmsd_gives_no_note_for_a_converging_pair(tmp_path, capfd):
    pytest.importorskip("gemmi")
    from harness import Boltz2Harness

    text = _synthetic_mmcif(_helix(12))
    ref = tmp_path / "ref.cif"
    ref.write_text(text)
    notes: list[str] = []

    Boltz2Harness._ca_rmsd(text, ref, notes=notes)

    assert notes == [] and capfd.readouterr().out == ""


def test_ca_rmsd_passes_other_native_output_on_to_stderr(tmp_path, capfd, monkeypatch):
    gemmi = pytest.importorskip("gemmi")
    import os

    from harness import Boltz2Harness

    real = gemmi.calculate_superposition

    def noisy(*args, **kwargs):
        os.write(2, b"some other native message\n")
        return real(*args, **kwargs)

    monkeypatch.setattr(gemmi, "calculate_superposition", noisy)
    text = _synthetic_mmcif(_helix(12))
    ref = tmp_path / "ref.cif"
    ref.write_text(text)
    notes: list[str] = []

    Boltz2Harness._ca_rmsd(text, ref, notes=notes)

    captured = capfd.readouterr()
    assert "some other native message" in captured.err and captured.out == ""
    assert notes == []


def test_every_shipped_fixture_passes_the_inline_content_check(all_benchmark_fixtures, accuracy_gate_fixtures):
    """The harness sends these bodies to the service, so none may name a file path."""
    for label, fixture in {
        **all_benchmark_fixtures,
        **{f"acc-{k}": v for k, v in accuracy_gate_fixtures.items()},
    }.items():
        data = fixture["data"]
        reject_file_references(data["sequences"], data.get("templates"))


# --------------------------------------------------------------------------- #
# An MSA timeout that boltz catches per input still reaches the caller
# --------------------------------------------------------------------------- #


def _fake_boltz_mmseqs2(monkeypatch):
    import sys
    import types

    module = types.ModuleType("boltz.data.msa.mmseqs2")
    module.time = time
    for name in ("boltz", "boltz.data", "boltz.data.msa"):
        monkeypatch.setitem(sys.modules, name, types.ModuleType(name))
    monkeypatch.setitem(sys.modules, "boltz.data.msa.mmseqs2", module)
    sys.modules["boltz.data.msa"].mmseqs2 = module
    return module


def _boltz_like_processing(mmseqs2):
    """What boltz does: the MSA step raises, and the per-input handler catches it and skips the input."""
    records = []
    try:
        mmseqs2.time.sleep(0.01)
        records.append("record")
    except Exception:  # noqa: BLE001 - boltz: "Failed to process ... Skipping"
        pass
    return records


def test_a_budget_timeout_that_boltz_catches_is_raised_when_no_record_remains(monkeypatch):
    from runner import bounded_msa_server, raise_if_msa_timeout_left_no_records

    mmseqs2 = _fake_boltz_mmseqs2(monkeypatch)
    with bounded_msa_server(time.monotonic() - 1) as budget:
        records = _boltz_like_processing(mmseqs2)

    assert records == []
    with pytest.raises(MsaServerTimeout) as raised:
        raise_if_msa_timeout_left_no_records(records, budget)
    assert is_msa_server_timeout(raised.value)
    assert mmseqs2.time is time  # the shim is removed again


def test_an_empty_result_without_a_timeout_is_not_reported_as_one(monkeypatch):
    from runner import bounded_msa_server, raise_if_msa_timeout_left_no_records

    _fake_boltz_mmseqs2(monkeypatch)
    with bounded_msa_server(time.monotonic() + 60) as budget:
        pass
    raise_if_msa_timeout_left_no_records([], budget)
    raise_if_msa_timeout_left_no_records([], None)


def test_a_record_that_survived_a_timeout_is_not_replaced_by_the_error(monkeypatch):
    from runner import bounded_msa_server, raise_if_msa_timeout_left_no_records

    mmseqs2 = _fake_boltz_mmseqs2(monkeypatch)
    with bounded_msa_server(time.monotonic() - 1) as budget:
        _boltz_like_processing(mmseqs2)
    raise_if_msa_timeout_left_no_records(["record"], budget)


# --------------------------------------------------------------------------- #
# Seed and deterministic mode, for one prediction
# --------------------------------------------------------------------------- #


@pytest.fixture
def fake_lightning(monkeypatch):
    """A ``pytorch_lightning`` with a recording ``seed_everything``."""
    import sys
    import types

    calls = []
    module = types.ModuleType("pytorch_lightning")
    module.seed_everything = lambda seed, *a, **k: calls.append(seed) or seed
    monkeypatch.setitem(sys.modules, "pytorch_lightning", module)
    return calls


@pytest.fixture
def deterministic_state():
    torch = pytest.importorskip("torch")
    before = (torch.are_deterministic_algorithms_enabled(), torch.is_deterministic_algorithms_warn_only_enabled())
    yield torch
    torch.use_deterministic_algorithms(before[0], warn_only=before[1])


def test_no_seed_changes_nothing(fake_lightning, deterministic_state):
    from runner import seeded_prediction

    torch = deterministic_state
    with seeded_prediction(None):
        assert torch.are_deterministic_algorithms_enabled() is False
    assert fake_lightning == []


def test_a_seed_seeds_everything_and_sets_deterministic_mode_for_the_prediction(fake_lightning, deterministic_state):
    from runner import seeded_prediction

    torch = deterministic_state
    with seeded_prediction(123):
        assert torch.are_deterministic_algorithms_enabled() is True
        assert torch.is_deterministic_algorithms_warn_only_enabled() is True
    assert fake_lightning == [123]
    assert torch.are_deterministic_algorithms_enabled() is False


def test_a_seed_of_zero_is_a_seed(fake_lightning, deterministic_state):
    from runner import seeded_prediction

    with seeded_prediction(0):
        pass
    assert fake_lightning == [0]


def test_the_previous_setting_comes_back_when_the_prediction_fails(fake_lightning, deterministic_state):
    from runner import seeded_prediction

    torch = deterministic_state
    torch.use_deterministic_algorithms(True, warn_only=False)
    with pytest.raises(RuntimeError, match="boom"):
        with seeded_prediction(7):
            assert torch.is_deterministic_algorithms_warn_only_enabled() is True
            raise RuntimeError("boom")
    assert torch.are_deterministic_algorithms_enabled() is True
    assert torch.is_deterministic_algorithms_warn_only_enabled() is False


def test_the_setting_is_off_after_a_failure_when_it_was_off_before(fake_lightning, deterministic_state):
    from runner import seeded_prediction

    torch = deterministic_state
    with pytest.raises(RuntimeError):
        with seeded_prediction(7):
            raise RuntimeError("boom")
    assert torch.are_deterministic_algorithms_enabled() is False


# --------------------------------------------------------------------------- #
# The content of an inline A3M
# --------------------------------------------------------------------------- #

QUERY = "NLYIQWLKDG"


def _msa_request(msa):
    return [{"protein": {"id": "A", "sequence": QUERY, "msa": msa}}]


@pytest.mark.parametrize("msa", [">", "> \n", ">query\n", ">query\n>hit\n", ">query\n# comment\n\n"])
def test_an_a3m_without_a_sequence_line_is_refused(msa):
    with pytest.raises(ValueError, match="no sequence line"):
        reject_file_references(_msa_request(msa), None)


@pytest.mark.parametrize(
    "msa",
    [
        f">query\n{QUERY}\n>hit\nNLYIQWLK\n",  # shorter than the chain
        f">query\n{QUERY}A\n",  # longer than the chain
        f">q\n{QUERY[:5]}\n>h\n{QUERY[:5]}\n",
    ],
)
def test_an_a3m_row_of_the_wrong_length_is_refused(msa):
    with pytest.raises(ValueError, match="aligned length"):
        reject_file_references(_msa_request(msa), None)


@pytest.mark.parametrize("bad", ["*", "1", ".", "?"])
def test_an_a3m_character_that_boltz_cannot_parse_is_refused(bad):
    msa = f">query\n{QUERY[:-1]}{bad}\n"
    with pytest.raises(ValueError, match="not a residue letter") as raised:
        reject_file_references(_msa_request(msa), None)
    assert bad not in str(raised.value).replace("'-'", "")


def test_a_valid_a3m_with_gaps_and_inserts_is_accepted():
    msa = f">query\n{QUERY}\n>hit1\nNLY-QWLKdDG\n>hit2\nnnNLYIQWLKDGaa\n>hit3 desc\nXXBZJOUXXX\n"
    reject_file_references(_msa_request(msa), None)


def test_a_duplicate_row_and_a_comment_line_are_accepted():
    reject_file_references(_msa_request(f">q\n#comment\n{QUERY}\n>q2\n{QUERY}\n"), None)
