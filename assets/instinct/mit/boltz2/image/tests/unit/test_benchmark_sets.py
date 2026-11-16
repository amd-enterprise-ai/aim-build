# Copyright © Advanced Micro Devices, Inc., or its affiliates.
#
# SPDX-License-Identifier: MIT

"""One fixture directory, two named sets, and the seam between them.

``examples/benchmark_sequences/`` holds all eighteen cases. Which of them an
evaluate run uses is decided by ``metadata.sets`` against
``harness.BENCHMARK_SET``, not by which directory a file sits in:

* ``fast`` — six targets. What the harness selects with nothing set, and the set
  every timing, accuracy and gate-abstention figure this asset has published is
  a sum over.
* ``timing-18`` — all eighteen targets of the public timing table.

The six are members of BOTH sets, so the sets OVERLAP. That is the point of
naming them: before this, covering the eighteen meant a second directory with
six files duplicated into it, and the duplicate copies could drift apart.

Four failure modes are specific to having two overlapping sets in one directory,
and the tests below exist one per mode rather than as a general "the fixtures
look right" sweep:

1. **Silent re-baselining.** If a case joins ``fast``, every future evaluate
   sums over more targets than every historical figure did — same key, same
   code, different meaning, no signal.
2. **Silent exclusion.** A case that declares no ``sets`` at all belongs to
   nothing and is skipped. That must be loud, and it must not fall back to the
   default set.
3. **A methodology change smuggled in as a coverage change.** The sets exist to
   differ in COVERAGE. If their recipes also differed, no figure from one could
   be read against the other.
4. **A set selected from a profile.** Exporting the selector in a profile makes
   the wider set the served default, which is failure mode 1 by another route.

Everything here is pure file reading. No GPU, no service, no weights.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import yaml

ASSET = Path(__file__).resolve().parents[3]
IMAGE = Path(__file__).resolve().parents[2]
BENCH_DIR = IMAGE / "examples" / "benchmark_sequences"

FAST_SET = "fast"
REFERENCE_SET = "timing-18"
FAST_IDS = ["7qsj", "8are", "8c4d", "8cpk", "8eil", "8gpp"]

# boltz 2.2.1's own token maps, transcribed from src/boltz/data/const.py. The
# point of holding them here is that the non-standard-letter test DERIVES the
# offending set by differencing against them, instead of comparing against a
# list of letters someone remembered to write down.
BOLTZ_TOKEN_LETTERS = {
    "protein": set("ARNDCEQGHILKMFPSTWYV"),
    "dna": set("AGCTN"),
    "rna": set("AGCUN"),
}
BOLTZ_UNK_TOKEN = {"protein": "UNK", "dna": "DN", "rna": "N"}


def _entity(entry: dict) -> tuple[str, dict]:
    kind = next(iter(entry))
    return kind, entry[kind]


def _chain_count(body: dict) -> int:
    ids = body["id"]
    return len(ids) if isinstance(ids, list) else 1


def _in_set(fixtures: dict, name: str) -> dict:
    return {lab: f for lab, f in fixtures.items() if name in f["metadata"]["sets"]}


def _reference_block(recipe: dict) -> dict:
    return recipe["fixture_sets"]["sets"][REFERENCE_SET]


# --------------------------------------------------------------------------- #
# Mode 1 — the default set must not widen
# --------------------------------------------------------------------------- #


def test_the_default_set_is_exactly_the_six_published_targets(all_benchmark_fixtures):
    """Pinned by id, not by count.

    A count would pass if one target were swapped for another, which changes
    every published figure just as thoroughly as adding one. This is the single
    most load-bearing assertion in the file: the six named here are the set that
    every timing and accuracy number on this asset sums over.
    """
    fast = sorted(_in_set(all_benchmark_fixtures, FAST_SET))
    assert fast == FAST_IDS, (
        f"the default set is now {fast}, not {FAST_IDS}. Every published figure on this asset "
        f"is a sum over those six; changing the membership silently re-baselines all of them. "
        f"If this is intended, the figures in tuning.yaml and .build/ must be retired in the "
        f"same change."
    )
    assert len(all_benchmark_fixtures) == 18, "the directory must hold all eighteen cases"


def test_the_default_set_is_a_strict_subset_of_the_reference_set(all_benchmark_fixtures):
    """The sets overlap deliberately; the six are the fast third of the eighteen."""
    fast = set(_in_set(all_benchmark_fixtures, FAST_SET))
    reference_set = set(_in_set(all_benchmark_fixtures, REFERENCE_SET))
    assert len(reference_set) == 18
    assert fast < reference_set, (
        f"the default set must stay a strict subset of {REFERENCE_SET!r}. Members of {FAST_SET!r} "
        f"missing from it: {sorted(fast - reference_set)}"
    )


def test_every_fixture_declares_at_least_one_set(all_benchmark_fixtures):
    """A case declaring no set belongs to nothing and would never be evaluated.

    The loader skips it with a warning rather than defaulting it in — see
    ``test_a_case_declaring_no_set_is_skipped_rather_than_defaulted_in``. That is
    the safe direction, but silently shipping a case nothing ever runs is still a
    defect, so it is caught here at authoring time.
    """
    for label, fixture in all_benchmark_fixtures.items():
        sets = fixture["metadata"].get("sets")
        assert sets, f"{label} declares no metadata.sets, so no evaluate run would ever include it"
        assert isinstance(sets, list), f"{label}: metadata.sets must be a list, got {type(sets).__name__}"
        assert set(sets) <= {
            FAST_SET,
            REFERENCE_SET,
        }, f"{label} names an unknown set: {sorted(set(sets) - {FAST_SET, REFERENCE_SET})}"


def test_the_harness_default_set_and_dir_are_unchanged(monkeypatch):
    """With nothing exported, the harness selects the six from the default path.

    The environment is scrubbed and the module reloaded rather than read as
    imported. Reading the constants as-is would test whatever the variables
    happened to hold when the suite started — and this suite is expected to run
    inside the very GPU session that exports them to the wider set, so the naive
    form would fail there for the wrong reason and pass elsewhere for no reason.
    """
    import importlib

    import harness

    monkeypatch.delenv("BOLTZ2_BENCHMARK_DIR", raising=False)
    monkeypatch.delenv("BOLTZ2_BENCHMARK_SET", raising=False)
    reloaded = importlib.reload(harness)
    try:
        assert str(reloaded.BENCHMARK_DIR) == "/workspace/model/benchmarks"
        assert reloaded.DEFAULT_BENCHMARK_SET == FAST_SET
        assert reloaded.BENCHMARK_SET == FAST_SET, (
            f"harness.BENCHMARK_SET resolved to {reloaded.BENCHMARK_SET!r} with "
            f"BOLTZ2_BENCHMARK_SET unset. The default must remain {FAST_SET!r}."
        )
    finally:
        # Leave the module in whatever state the ambient environment implies, so
        # a later test in the same session is not silently affected by this one.
        importlib.reload(harness)


def test_the_loader_returns_exactly_the_six_by_default_and_eighteen_when_asked(monkeypatch):
    """Exercises the real selector against the real shipped files.

    Every other test here reads metadata and reasons about it. This one calls
    ``_load_eval_cases`` itself, pointed at the shipped directory, so the
    filtering CODE is covered rather than only the data it filters on. Without
    it, a loader that ignored ``metadata.sets`` entirely would pass this suite.
    """
    import harness

    monkeypatch.setattr(harness, "BENCHMARK_DIR", BENCH_DIR)

    monkeypatch.setattr(harness, "BENCHMARK_SET", FAST_SET)
    default = harness.Boltz2Harness._load_eval_cases()
    assert sorted(c["metadata"]["label"] for c in default) == FAST_IDS

    monkeypatch.setattr(harness, "BENCHMARK_SET", REFERENCE_SET)
    wide = harness.Boltz2Harness._load_eval_cases()
    assert len(wide) == 18

    monkeypatch.setattr(harness, "BENCHMARK_SET", "no-such-set")
    assert harness.Boltz2Harness._load_eval_cases() == [], "an unknown set must select nothing, not everything"


def test_a_case_declaring_no_set_is_skipped_rather_than_defaulted_in(monkeypatch, tmp_path, caplog):
    """The negative control for mode 2.

    A case with no ``sets`` key must NOT be swept into the current set. Falling
    back to the default is the one behaviour that could widen the published
    baseline by accident, so it is proven absent rather than assumed.
    """
    import logging

    import harness

    (tmp_path / "7qsj").mkdir()
    (tmp_path / "7qsj" / "7qsj.json").write_text(
        json.dumps({"metadata": {"label": "7qsj", "sets": [FAST_SET]}, "data": {}})
    )
    (tmp_path / "zzzz").mkdir()
    (tmp_path / "zzzz" / "zzzz.json").write_text(json.dumps({"metadata": {"label": "zzzz"}, "data": {}}))

    monkeypatch.setattr(harness, "BENCHMARK_DIR", tmp_path)
    monkeypatch.setattr(harness, "BENCHMARK_SET", FAST_SET)
    with caplog.at_level(logging.WARNING):
        cases = harness.Boltz2Harness._load_eval_cases()

    assert [c["metadata"]["label"] for c in cases] == ["7qsj"], "a set-less case must not be included"
    assert any(
        "declares no metadata.sets" in r.message for r in caplog.records
    ), "skipping a case must be logged; a silent skip is how a case stops being evaluated unnoticed"


def test_the_dockerfile_bakes_one_directory_holding_both_sets():
    """The merge must survive into the image.

    One COPY to the path ``harness.BENCHMARK_DIR`` defaults to, for the
    'fast'/'timing-18' pair this file is about. A lingering second
    destination for THOSE sets would mean the image still carried the old
    table2 split and the two could disagree about which files exist.

    ``examples/benchmark_sequences_accuracy/`` is a DIFFERENT, deliberate
    second directory (a recipe difference, not a coverage split of this pair)
    and is exempted below rather than counted here — see
    test_accuracy_gate_fixtures.py::test_the_dockerfile_bakes_the_accuracy_gate_directory_to_its_own_path.
    """
    text = (IMAGE / "Dockerfile").read_text()
    assert "COPY --link examples/benchmark_sequences /workspace/model/benchmarks\n" in text, (
        "the fixture directory must be COPYed to /workspace/model/benchmarks — that exact path is "
        "harness.BENCHMARK_DIR's default"
    )
    assert "benchmarks-table2" not in text, (
        "the Dockerfile still bakes a separate table2 path. The sets are selected by "
        "metadata.sets now; a second directory would reintroduce the duplication this replaced."
    )
    copy_lines = [
        line
        for line in text.splitlines()
        if line.startswith("COPY --link examples/benchmark_sequences")
        and not line.startswith("COPY --link examples/benchmark_sequences_accuracy")
    ]
    assert len(copy_lines) == 1


def test_no_profile_claims_either_benchmark_selection_seam():
    """The wider set stays opt-in.

    A profile exporting ``BOLTZ2_BENCHMARK_SET`` would make the eighteen-target
    set the served default for `aim-runtime evaluate`, which is the change this
    whole seam exists to prevent. ``BOLTZ2_BENCHMARK_DIR`` is checked too: it
    still overrides the directory, so it is a second route to the same failure.
    Checked against the profiles rather than trusted, because profiles are what
    aim-runtime actually exports.
    """
    profiles = sorted((ASSET / "profiles").glob("*.yaml"))
    assert profiles, "no profiles found"
    for path in profiles:
        env_vars = yaml.safe_load(path.read_text()).get("env_vars") or {}
        for seam in ("BOLTZ2_BENCHMARK_SET", "BOLTZ2_BENCHMARK_DIR"):
            assert seam not in env_vars, (
                f"{path.name} sets {seam}={env_vars[seam]!r}. Those variables are what keep the "
                "wider set opt-in; setting either in a profile re-baselines every default evaluate."
            )


# --------------------------------------------------------------------------- #
# Mode 3 — coverage differs, methodology does not
# --------------------------------------------------------------------------- #


def _recipe_view(data: dict, request_model) -> dict:
    """Every request field that configures the recipe, resolved against the
    service defaults so an absent field compares as what would be applied."""
    identity = {"version", "sequences", "constraints", "templates", "properties", "name"}
    fields = sorted(set(request_model.model_fields) - identity)
    return {f: data.get(f, request_model.model_fields[f].default) for f in fields}


def test_every_fixture_requests_one_identical_recipe(all_benchmark_fixtures, request_model):
    """All eighteen files resolve to ONE recipe, whichever set they are in.

    The sets are a coverage distinction. If the recipe varied with the set, no
    figure from one could be read against the other, and merging them into one
    directory would have hidden the difference rather than removed it.
    """
    views = {label: _recipe_view(f["data"], request_model) for label, f in all_benchmark_fixtures.items()}
    assert len(views) == 18
    distinct = {json.dumps(v, sort_keys=True) for v in views.values()}
    assert len(distinct) == 1, (
        f"the directory holds {len(distinct)} different recipes across {len(views)} files. "
        "Coverage may differ between sets; methodology may not. Recipes seen: " + "; ".join(sorted(distinct))[:600]
    )


def test_every_fixture_matches_the_declared_timing_recipe(benchmark_recipe, all_benchmark_fixtures):
    expected = benchmark_recipe["timing_recipe"]["fields"]
    for label, fixture in all_benchmark_fixtures.items():
        for field, value in expected.items():
            assert fixture["data"].get(field) == value, f"{label}.{field}"
        assert fixture["data"].get("use_msa_server") is False


def test_no_fixture_sets_a_field_the_request_schema_would_silently_ignore(all_benchmark_fixtures, request_model):
    """A typo'd field is dropped by pydantic, not rejected — catch it here."""
    declared = set(request_model.model_fields)
    for label, fixture in all_benchmark_fixtures.items():
        unknown = set(fixture["data"]) - declared
        assert not unknown, f"fixture {label} sets unknown request field(s) {sorted(unknown)}"


def test_every_fixture_is_a_valid_request(all_benchmark_fixtures, request_model):
    """Every one of the eighteen parses through the real request model.

    Not a shape check on a dict: this constructs ``Boltz2Request``, so the
    single-key-mapping rule, the output_format validator and the non-empty
    sequences rule all run. Ten of the eighteen are entity shapes the six-target
    set never had (multi-entity complexes, DNA, RNA), and "the JSON is well
    formed" would not have covered any of them.
    """
    for label, fixture in all_benchmark_fixtures.items():
        req = request_model(**fixture["data"])
        assert req.name == label
        assert req.sequences
        for entry in req.sequences:
            kind, body = _entity(entry)
            assert kind in ("protein", "dna", "rna", "ligand"), f"{label}: unexpected entity {kind}"
            assert body["sequence"], f"{label}: empty sequence on {kind} {body['id']}"


def test_neither_set_is_gradeable_and_the_wider_one_does_not_close_that_gap(benchmark_recipe, all_benchmark_fixtures):
    """The eighteen do NOT close the accuracy-gate coverage gap.

    It looks like they might — eighteen targets, fifteen with references. They do
    not, and pinning that here stops the gap from being quietly considered
    closed: every fixture runs the same recipe by design, so all eighteen
    mismatch ``ACCURACY_GATE_RECIPE`` on the same three fields and the gate
    abstains on all of them.
    """
    from harness import accuracy_recipe_mismatch

    declared = set(benchmark_recipe["accuracy_gate_relationship"]["mismatching_fields"])
    assert _reference_block(benchmark_recipe)["gradeable_by_the_in_image_accuracy_gate"] is False
    for label, fixture in all_benchmark_fixtures.items():
        mismatch = accuracy_recipe_mismatch(fixture["data"])
        assert mismatch, f"{label} is gradeable — the gate would now judge it; update tuning.yaml too"
        assert set(mismatch) == declared, f"{label} mismatches {sorted(mismatch)}, declared {sorted(declared)}"


# --------------------------------------------------------------------------- #
# The table the wider set exists to cover
# --------------------------------------------------------------------------- #


def test_the_eighteen_are_exactly_the_declared_timing_targets(benchmark_recipe, all_benchmark_fixtures):
    """Membership is checked against the ids declared in ``benchmark_recipe.json``."""
    declared = _reference_block(benchmark_recipe)["ids"]
    reference_set = _in_set(all_benchmark_fixtures, REFERENCE_SET)
    assert len(declared) == 18
    assert set(declared) == set(reference_set), (
        f"fixtures {sorted(set(reference_set) - set(declared))} are not declared; declared ids "
        f"{sorted(set(declared) - set(reference_set))} have no fixture"
    )
    for label, fixture in all_benchmark_fixtures.items():
        assert "reference_timings" not in fixture["metadata"], f"{label}: reference times live outside the repo"


def test_declared_residue_counts_match_the_sequences_actually_present(all_benchmark_fixtures):
    """A fixture may not claim a size it does not carry.

    Recomputed from the request body — length times chain count, summed over
    entities — rather than read back from the same metadata that states it.
    """
    total = 0
    for label, fixture in all_benchmark_fixtures.items():
        meta = fixture["metadata"]
        residues = chains = 0
        for entry in fixture["data"]["sequences"]:
            _, body = _entity(entry)
            n = _chain_count(body)
            residues += len(body["sequence"]) * n
            chains += n
        assert (
            meta["residues"] == residues
        ), f"{label}: metadata says {meta['residues']} residues, sequences carry {residues}"
        assert meta["chains"] == chains, f"{label}: metadata says {meta['chains']} chains, sequences carry {chains}"
        assert meta["entities"] == len(fixture["data"]["sequences"]), f"{label}: entity count"
        total += residues
    assert total == 16888, (
        f"the directory now carries {total} residues, not 16888. If a sequence legitimately changed, "
        "update this number with it."
    )


# --------------------------------------------------------------------------- #
# Provenance the fixtures must carry
# --------------------------------------------------------------------------- #


def test_exactly_three_targets_are_timing_only_and_say_why(benchmark_recipe, all_benchmark_fixtures):
    """msc1 / bcor / evpl carry no accuracy claim, and the fixture says so.

    They are UniProt accessions, not depositions, so no experimental structure
    exists to superpose against — the limit is in the target, not in the
    staging. A reader must learn that from the fixture rather than from a
    missing file and a harness error string.
    """
    declared = set(_reference_block(benchmark_recipe)["timing_only_ids"])
    assert declared == {"msc1", "bcor", "evpl"}

    observed = {lab for lab, f in all_benchmark_fixtures.items() if f["metadata"]["accuracy"]["timing_only"]}
    assert observed == declared, f"timing-only set is {sorted(observed)}, declared {sorted(declared)}"
    assert not (declared & set(_in_set(all_benchmark_fixtures, FAST_SET))), (
        "no timing-only target may join the default set — it would put an ungradeable case in the "
        "set the accuracy figures are summed over"
    )

    for label in sorted(declared):
        meta = all_benchmark_fixtures[label]["metadata"]
        assert meta["accuracy"]["scoreable"] is False
        assert meta["reference_pdb_id"] is None
        assert meta["reference_url"] is None
        assert meta["reference_file"] is None
        reason = meta["accuracy"]["reason"]
        assert "TIMING ONLY" in reason
        assert "no deposition" in reason.lower()
        assert "in principle" in reason.lower(), f"{label}: must say the limit is in principle, not a staging gap"
        # And the accession that replaces the PDB id.
        up = meta["uniprot"]
        assert up["accession"] in ("Q9UT79", "Q6W2J9", "Q92817")
        assert up["fetched_at"] and up["fasta_sha256"] and up["sequence_sha256"]
        assert up["fasta_url"].startswith("https://rest.uniprot.org/uniprotkb/")

    for label in sorted(set(all_benchmark_fixtures) - declared):
        meta = all_benchmark_fixtures[label]["metadata"]
        assert meta["accuracy"]["scoreable"] is True, f"{label} should be scoreable"
        assert meta["reference_pdb_id"] == label.upper()
        assert len(meta["reference_sha256"]) == 64, f"{label}: reference digest must be recorded"
        assert meta["reference_fetched_at"]
        assert meta["reference_license"].startswith("CC0")


def test_the_uniprot_accessions_and_lengths_are_pinned(all_benchmark_fixtures):
    """The accession and the length identify these three targets; both are pinned."""
    expect = {"msc1": ("Q9UT79", 1588), "bcor": ("Q6W2J9", 1755), "evpl": ("Q92817", 2033)}
    for label, (accession, length) in expect.items():
        meta = all_benchmark_fixtures[label]["metadata"]
        assert meta["uniprot"]["accession"] == accession
        assert meta["residues"] == length


def test_non_standard_letters_are_found_from_the_token_maps_and_declared(all_benchmark_fixtures):
    """Derive the offending set; do not look it up.

    boltz maps an unrecognised sequence letter to its unknown token SILENTLY
    (``schema.py:1162``: ``token_map.get(c, unk_token)``), so nothing upstream
    will ever complain. Any fixture carrying such a letter must declare it, with
    the letter, its position and what boltz turns it into. Because the set is
    recomputed here from boltz's own three token maps, a THIRD such letter
    arriving in a future edit fails this test rather than being swallowed.
    """
    offenders: dict[str, list] = {}
    for label, fixture in all_benchmark_fixtures.items():
        found = []
        for entry in fixture["data"]["sequences"]:
            kind, body = _entity(entry)
            allowed = BOLTZ_TOKEN_LETTERS[kind]
            bad = sorted({c for c in body["sequence"] if c not in allowed})
            if bad:
                found.append((kind, body["id"], bad))
        if found:
            offenders[label] = found

    assert set(offenders) == {"8clz", "8pso"}, (
        f"targets carrying letters outside boltz's token maps are {sorted(offenders)}; expected "
        "{'8clz', '8pso'}. boltz would map any new one to its unknown token silently, so a new "
        "entry here needs a metadata.non_standard_residues declaration and an update to "
        "benchmark_recipe.json's non_standard_residues block."
    )
    assert not (
        set(offenders) & set(_in_set(all_benchmark_fixtures, FAST_SET))
    ), "a target carrying a silently-remapped residue must not be in the default set"

    for label, found in offenders.items():
        block = all_benchmark_fixtures[label]["metadata"].get("non_standard_residues")
        assert block, f"{label} carries a non-standard letter but declares nothing"
        assert "SILENTLY" in block["note"], f"{label}: the declaration must state that boltz does not complain"
        declared = {(e["entity"], json.dumps(e["id"]), tuple(e["letters"])) for e in block["found"]}
        observed = {(kind, json.dumps(cid), tuple(bad)) for kind, cid, bad in found}
        assert declared == observed, f"{label}: declared {sorted(declared)} but sequences carry {sorted(observed)}"
        for e in block["found"]:
            assert e["boltz_maps_to"] == BOLTZ_UNK_TOKEN[e["entity"]]
            assert e["positions_1based"], f"{label}: positions must be recorded"

    total_bad = sum(
        len(e["positions_1based"])
        for lab in offenders
        for e in all_benchmark_fixtures[lab]["metadata"]["non_standard_residues"]["found"]
    )
    assert total_bad == 2, f"{total_bad} non-standard residues across the set; benchmark_recipe.json declares 2"


def test_the_nucleic_acid_and_multi_entity_targets_are_present_and_shaped_as_expected(all_benchmark_fixtures):
    """Pins what makes the wider set harder than the default one.

    The default set is protein-only and tops out at three chains from one entity.
    The twelve that complete the eighteen introduce DNA, RNA, an eight-chain
    homo-oligomer and a five-entity nucleoprotein complex. If a future edit
    flattened any of those back to a single protein chain, the set would still
    pass every recipe check and would quietly stop covering the thing it was
    added for.
    """
    kinds = {
        label: sorted({_entity(e)[0] for e in f["data"]["sequences"]}) for label, f in all_benchmark_fixtures.items()
    }
    assert kinds["8ibx"] == ["dna", "protein", "rna"], "8ibx must keep protein + DNA + RNA"
    assert kinds["8pso"] == ["dna", "protein", "rna"], "8pso must keep protein + DNA + RNA"

    nucleic = {lab for lab, k in kinds.items() if {"dna", "rna"} & set(k)}
    assert nucleic == {"8ibx", "8pso"}, f"nucleic-acid targets are {sorted(nucleic)}"

    assert all_benchmark_fixtures["8gi1"]["metadata"]["chains"] == 8
    assert all_benchmark_fixtures["8gi1"]["metadata"]["entities"] == 1, "8gi1 is one entity across eight chains"
    assert all_benchmark_fixtures["8sm6"]["metadata"]["chains"] == 4
    assert all_benchmark_fixtures["8clz"]["metadata"]["chains"] == 2
    assert all_benchmark_fixtures["8pso"]["metadata"]["entities"] == 5, "8pso is the five-entity complex"

    largest = max(all_benchmark_fixtures.values(), key=lambda f: f["metadata"]["residues"])
    assert largest["metadata"]["label"] == "evpl"
    assert (
        largest["metadata"]["residues"] == 2033
    ), "evpl at 2033 residues is the set's size ceiling and the reason a truncation check exists"


@pytest.mark.parametrize("label", sorted(["8owf", "7tpu", "7ylz", "8clz", "8k7x", "8ibx", "8gi1", "8sm6", "8pso"]))
def test_the_nine_translated_targets_say_they_were_translated_not_copied(label, all_benchmark_fixtures):
    """The sibling uses a different request schema; the fixture must not pretend
    otherwise. A reader who assumes these were copied would look for a
    ``molecule_type`` key that is not there and conclude the fixture is wrong."""
    prov = all_benchmark_fixtures[label]["metadata"]["provenance"]
    assert "openfold3" in prov
    assert "TRANSLATED" in prov, f"{label}: the schema translation must be stated"
    assert "molecule_type" in prov
    assert "VERBATIM" in prov, f"{label}: the sequence-reuse claim must be stated"


def test_the_six_shared_targets_record_both_memberships(all_benchmark_fixtures):
    """The six are the six because the sibling openfold3 AIM already scores them; the file must say so."""
    for label in FAST_IDS:
        meta = all_benchmark_fixtures[label]["metadata"]
        assert sorted(meta["sets"]) == sorted([FAST_SET, REFERENCE_SET]), f"{label} must be in both sets"
        prov = meta["provenance"]
        assert "public performance table" in prov, f"{label}: the timing-table membership must be stated"
        assert "OpenFold3 reference set" in prov, f"{label}: the OpenFold3 membership must be stated"


def test_every_fixture_tells_a_reader_what_it_is_for_and_which_sets_it_is_in(all_benchmark_fixtures):
    """Opening one file must be enough to learn both facts."""
    for label, fixture in all_benchmark_fixtures.items():
        meta = fixture["metadata"]
        rel = meta["set_relationship"]
        assert "BOLTZ2_BENCHMARK_SET" in rel, f"{label}: must name how a set is selected"
        assert "metadata.sets" in rel, f"{label}: must name where membership is declared"
        assert "never add their totals" in rel, f"{label}: the overlap caveat must be stated"
        recipe = meta["recipe"]
        assert "Timing instrument, not an accuracy eval set" in recipe["purpose"]
        assert recipe["source"].strip()
        assert recipe["diverges_from_serving_defaults"].strip()
        assert "bound" in recipe["msa"], f"{label}: the comparison caveat must survive"


def test_no_fixture_points_at_a_builder_working_directory():
    """A file baked into the image must not reference ``.build/``, which is
    Builder scratch a container reader cannot reach and which may never be
    committed."""
    paths = sorted(BENCH_DIR.glob("*/*.json"))
    assert len(paths) == 18
    for path in paths:
        text = path.read_text()
        for dangling in (".build/", "build_report.md", "state.yaml"):
            assert dangling not in text, f"{path.name} references the Builder working directory ({dangling})"


def test_no_fixture_still_describes_the_retired_two_directory_layout():
    """No fixture may describe a two-directory layout.

    Text that names ``benchmark_sequences_table2`` or the ``benchmarks-table2``
    image path would send a reader looking for a directory that does not exist,
    and would contradict the seam the loader actually uses.
    """
    assert not (IMAGE / "examples" / "benchmark_sequences_table2").exists()
    for path in sorted(BENCH_DIR.glob("*/*.json")):
        text = path.read_text()
        for retired in ("benchmark_sequences_table2", "benchmarks-table2", "BOLTZ2_BENCHMARK_DIR to that path"):
            assert retired not in text, f"{path.name} still describes the retired two-directory layout ({retired})"


def test_what_ci_runs_by_default_never_calls_the_msa_server(fake_http, monkeypatch):
    """CI runs ``validate``, ``benchmark`` and ``evaluate`` with no ``--config``.

    The requests are recorded from the harness itself, so a new request in any of
    those paths is covered without a list here. Only the separate accuracy-gate
    fixtures request the MSA server, and only when selected.
    """
    import harness

    from aim_runtime.harness import HarnessConfig

    monkeypatch.setattr(harness, "BENCHMARK_DIR", IMAGE / "examples" / "benchmark_sequences")
    monkeypatch.setattr(harness, "BENCHMARK_SET", harness.DEFAULT_BENCHMARK_SET)
    config = HarnessConfig(profile={}, service_url="http://svc")
    smoke = HarnessConfig(profile={}, service_url="http://svc", extra={"benchmark_set": "smoke"})

    harness.Boltz2Harness().validate(config)
    harness.Boltz2Harness().benchmark(config)
    harness.Boltz2Harness().benchmark(smoke)
    harness.Boltz2Harness().evaluate(config)

    sent = fake_http.sent_data()
    paths = {url.rsplit("/", 1)[1] for url, data in fake_http.requests if data is not None}
    assert paths == {"inference"}
    # validate 1 + default benchmark 18 targets x 3 blocks + smoke 10 + sanity 1 + default set 6
    assert len(sent) == 1 + 54 + 10 + 1 + 6
    served_default = harness.SERVED_REQUEST_DEFAULTS["use_msa_server"]
    offenders = [d["name"] for d in sent if d.get("use_msa_server", served_default) is not False]
    assert not offenders, f"requests that call, or may call, the MSA server: {offenders}"
