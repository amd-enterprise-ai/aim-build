# Copyright © Advanced Micro Devices, Inc., or its affiliates.
#
# SPDX-License-Identifier: MIT

"""The accuracy-gate case set, and why it is a separate directory.

``examples/benchmark_sequences/`` (all eighteen cases, any ``metadata.sets``
membership) runs ONE recipe — the published performance-testing recipe — and every
one of those cases mismatches ``harness.ACCURACY_GATE_RECIPE``, so
``harness._check_structure_accuracy`` abstains on all of them
(``test_neither_set_is_gradeable_and_the_wider_one_does_not_close_that_gap`` in
test_benchmark_sets.py pins that). That gap is a RECIPE difference, not a
coverage difference, so it cannot be closed by adding another name to the same
files' ``metadata.sets`` — widening coverage there would still request the
same recipe. It needs cases that request the gate recipe itself, which is why
they live in ``examples/benchmark_sequences_accuracy/``, a directory baked to
its own image path and reached only when BOTH ``BOLTZ2_BENCHMARK_DIR`` and
``BOLTZ2_BENCHMARK_SET=accuracy-gate`` are set.

Everything here is pure file reading plus one call into the real harness
functions. No GPU, no service, no weights.
"""

from __future__ import annotations

ACCURACY_GATE_SET = "accuracy-gate"
ACCURACY_IDS = ["7qsj", "8are", "8c4d", "8cpk", "8eil", "8gpp"]


def test_the_accuracy_gate_set_is_exactly_the_six_floor_derivation_targets(accuracy_gate_fixtures):
    """Pinned by id, matching the six ``tuning.yaml`` names as the floor-derivation run.

    The complex_plddt floors in tuning.yaml were derived on these six targets
    at the gate recipe. Grading a different six would not be a regression check
    against that derivation.
    """
    assert sorted(accuracy_gate_fixtures) == ACCURACY_IDS
    assert len(accuracy_gate_fixtures) == 6


def test_every_accuracy_gate_fixture_declares_only_the_accuracy_gate_set(accuracy_gate_fixtures):
    """These cases must not also claim ``fast`` or ``timing-18``.

    Those names select the OTHER directory's files
    (``harness._load_eval_cases`` globs ``BENCHMARK_DIR``, one directory at a
    time); a fixture here claiming one of them would just be a label that
    cannot do anything, which is a sign the file was copied rather than
    authored for this directory.
    """
    for label, fixture in accuracy_gate_fixtures.items():
        sets = fixture["metadata"].get("sets")
        assert sets == [ACCURACY_GATE_SET], f"{label}: metadata.sets is {sets!r}, expected {[ACCURACY_GATE_SET]!r}"


def test_every_accuracy_gate_fixture_matches_the_gate_recipe_exactly(accuracy_gate_fixtures):
    """The whole point of the directory: zero mismatch against ACCURACY_GATE_RECIPE.

    Calls the real harness function rather than re-deriving the comparison
    here, so a change to ``ACCURACY_GATE_RECIPE`` or to the mismatch logic
    itself is exercised against these fixtures, not just against the function's
    own unit tests.
    """
    from harness import accuracy_recipe_mismatch

    for label, fixture in accuracy_gate_fixtures.items():
        mismatch = accuracy_recipe_mismatch(fixture["data"])
        assert mismatch == {}, f"{label} is NOT gradeable: {mismatch}. The gate would abstain on it."


def test_every_accuracy_gate_fixture_requests_use_msa_server_true(accuracy_gate_fixtures):
    """The one field that must be set explicitly; everything else rides the service default.

    ``ACCURACY_GATE_RECIPE`` is the served defaults plus ``use_msa_server=true``
    — the only departure. Spelling that out here means a fixture that silently
    regained ``use_msa_server=false`` (and so stopped being gradeable) fails on
    a clear, specific assertion rather than only on the mismatch-dict check
    above.
    """
    for label, fixture in accuracy_gate_fixtures.items():
        assert fixture["data"].get("use_msa_server") is True, f"{label}: use_msa_server must be true"


def test_accuracy_gate_fixtures_reuse_the_same_targets_and_references_as_the_fast_set(
    all_benchmark_fixtures, accuracy_gate_fixtures
):
    """Same six RCSB targets, same staged reference structure, different recipe only.

    No new reference fetch is needed: ``harness._check_structure_accuracy``
    resolves the reference path from ``metadata.label`` alone
    (``data/references/<label>.cif``), so reusing the label is what reuses the
    reference.
    """
    fast = {label: f for label, f in all_benchmark_fixtures.items() if label in ACCURACY_IDS}
    assert sorted(fast) == ACCURACY_IDS
    for label in ACCURACY_IDS:
        assert accuracy_gate_fixtures[label]["metadata"]["reference_file"] == fast[label]["metadata"]["reference_file"]
        assert (
            accuracy_gate_fixtures[label]["metadata"]["reference_sha256"] == fast[label]["metadata"]["reference_sha256"]
        )


def test_the_default_evaluate_run_does_not_see_the_accuracy_gate_set(monkeypatch):
    """No leakage: with nothing exported, ``_load_eval_cases`` never reaches this directory.

    The environment is scrubbed and the module reloaded rather than read as
    imported, mirroring
    ``test_the_harness_default_set_and_dir_are_unchanged`` in
    test_benchmark_sets.py — the suite may run inside a session that already
    exports the wider selection, and reading the live constants would pass or
    fail for the wrong reason depending on that ambient state.
    """
    import importlib

    import harness

    monkeypatch.delenv("BOLTZ2_BENCHMARK_DIR", raising=False)
    monkeypatch.delenv("BOLTZ2_BENCHMARK_SET", raising=False)
    reloaded = importlib.reload(harness)
    try:
        assert str(reloaded.BENCHMARK_DIR) == "/workspace/model/benchmarks", (
            "the default directory must stay the timing-fixture path; the accuracy-gate " "directory is opt-in only"
        )
        default_cases = reloaded.Boltz2Harness._load_eval_cases()
        assert all(ACCURACY_GATE_SET not in c["metadata"]["sets"] for c in default_cases), (
            "a default evaluate run picked up a case from the accuracy-gate set — that set "
            "must only be reached by explicitly exporting both BOLTZ2_BENCHMARK_DIR and "
            "BOLTZ2_BENCHMARK_SET=accuracy-gate"
        )
    finally:
        importlib.reload(harness)


def test_the_loader_returns_the_six_only_when_both_env_vars_are_set(monkeypatch):
    """Exercises the real selector pointed at the real accuracy-gate directory.

    Setting only one of the two seams must not be enough — the directory holds
    only accuracy-gate cases, so pointing BENCHMARK_DIR there with the default
    BENCHMARK_SET ('fast') selects nothing, which is itself worth pinning: it
    is the failure mode of someone exporting BOLTZ2_BENCHMARK_DIR alone and
    getting a silent empty evaluate instead of the cases they meant to grade.
    """
    import pathlib

    import harness

    accuracy_dir = pathlib.Path(__file__).resolve().parents[2] / "examples" / "benchmark_sequences_accuracy"

    monkeypatch.setattr(harness, "BENCHMARK_DIR", accuracy_dir)
    monkeypatch.setattr(harness, "BENCHMARK_SET", harness.DEFAULT_BENCHMARK_SET)
    assert (
        harness.Boltz2Harness._load_eval_cases() == []
    ), "BENCHMARK_DIR alone, without BENCHMARK_SET=accuracy-gate, must select nothing"

    monkeypatch.setattr(harness, "BENCHMARK_SET", ACCURACY_GATE_SET)
    cases = harness.Boltz2Harness._load_eval_cases()
    assert sorted(c["metadata"]["label"] for c in cases) == ACCURACY_IDS


def test_the_dockerfile_bakes_the_accuracy_gate_directory_to_its_own_path():
    """The merge must survive into the image, at a path distinct from the timing fixtures."""
    import pathlib

    image = pathlib.Path(__file__).resolve().parents[2]
    text = (image / "Dockerfile").read_text()
    assert "COPY --link examples/benchmark_sequences_accuracy /workspace/model/benchmarks-accuracy\n" in text
    assert (
        "COPY --link examples/benchmark_sequences /workspace/model/benchmarks\n" in text
    ), "the timing-fixture COPY must still be present and unchanged — this directory is additive"


def test_no_profile_sets_the_accuracy_gate_selector(accuracy_gate_fixtures):
    """The accuracy-gate set stays opt-in, same requirement as the timing sets.

    A profile exporting these would make grading the served default, instead
    of the deliberate choice an operator (or the Tuner, invoked explicitly)
    makes.
    """
    import pathlib

    import yaml

    asset = pathlib.Path(__file__).resolve().parents[3]
    profiles = sorted((asset / "profiles").glob("*.yaml"))
    assert profiles, "no profiles found"
    for path in profiles:
        env_vars = yaml.safe_load(path.read_text()).get("env_vars") or {}
        assert env_vars.get("BOLTZ2_BENCHMARK_SET") != ACCURACY_GATE_SET
        assert "benchmarks-accuracy" not in str(env_vars.get("BOLTZ2_BENCHMARK_DIR", ""))
