# Copyright © Advanced Micro Devices, Inc., or its affiliates.
#
# SPDX-License-Identifier: MIT

"""The service's defaults must equal upstream ``boltz predict``'s, except where
the AIM deliberately deviates — and every deviation must be declared.

This is the recipe-parity tier: it compares the *constructed* request defaults
against a reference captured from boltz 2.2.1's CLI, without running the model.
A boltz version bump that quietly changes a sampling default breaks here rather
than in a prediction nobody re-checked.

Two recipes live in this AIM and they are deliberately different:

* the **serving** recipe — ``Boltz2Request``'s defaults, which must track
  ``boltz predict``. Reference: ``tests/references/boltz_cli_defaults.json``.
* the **benchmark** recipe — what the six cases under
  ``examples/benchmark_sequences/`` request. They are a timing instrument
  matched to a published performance-test recipe, so they diverge from the
  serving recipe on purpose. Reference:
  ``tests/references/benchmark_recipe.json``.

The tests below keep those two failure modes apart. "A fixture drifted off the
published methodology" and "a serving default drifted off upstream boltz" must
never surface as the same assertion, because the fix is different.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from service import Boltz2Request, ServiceConfig

# Request fields that must track the upstream CLI exactly.
_TRACKED = (
    "recycling_steps",
    "sampling_steps",
    "diffusion_samples",
    "max_parallel_samples",
    "step_scale",
    "output_format",
    "msa_pairing_strategy",
    "use_msa_server",
)


@pytest.mark.parametrize("field", _TRACKED)
def test_request_default_matches_upstream_cli(boltz_cli_defaults, field):
    expected = boltz_cli_defaults["cli_defaults"][field]
    actual = Boltz2Request.model_fields[field].default
    assert actual == expected, (
        f"Boltz2Request.{field} default is {actual!r} but `boltz predict` uses "
        f"{expected!r}. Either restore parity or declare the deviation in "
        f"tests/references/boltz_cli_defaults.json under aim_deviations."
    )


def test_msa_server_url_is_a_deployment_setting_with_the_upstream_default(boltz_cli_defaults):
    assert "msa_server_url" not in Boltz2Request.model_fields
    assert ServiceConfig().msa_server_url == boltz_cli_defaults["cli_defaults"]["msa_server_url"]


def test_declared_deviations_are_exactly_the_ones_we_take(boltz_cli_defaults):
    """Every deviation is declared, and every declaration is real."""
    deviations = boltz_cli_defaults["aim_deviations"]
    assert set(deviations) == {"num_workers", "optimizations"}

    config = ServiceConfig()
    assert config.num_workers == deviations["num_workers"]["aim_value"]
    assert config.optimizations == deviations["optimizations"]["aim_value"]

    for name, entry in deviations.items():
        assert entry["reason"].strip(), f"deviation {name} has no recorded reason"


def test_optimizations_default_on_and_only_on_or_off_are_accepted(monkeypatch):
    """The default comes from code; any value other than on/off fails at config load."""
    monkeypatch.delenv("BOLTZ2_OPTIMIZATIONS", raising=False)
    assert ServiceConfig().optimizations == "on"
    for value, expected in (("off", "off"), ("OFF", "off"), (" on ", "on")):
        monkeypatch.setenv("BOLTZ2_OPTIMIZATIONS", value)
        assert ServiceConfig().optimizations == expected
    for bad in ("both", "attn", "true", "1", ""):
        monkeypatch.setenv("BOLTZ2_OPTIMIZATIONS", bad)
        with pytest.raises(ValueError, match="BOLTZ2_OPTIMIZATIONS must be 'on' or 'off'"):
            ServiceConfig()


def test_profiles_do_not_pin_the_optimizations_or_removed_settings():
    """A profile env_var overrides ``docker run -e`` without a warning.

    The defaults of ``BOLTZ2_OPTIMIZATIONS`` and ``BOLTZ2_AFFINITY_COMPILE`` live in code, so no profile
    may set them, and the deleted settings must not come back through a profile.
    """
    import yaml

    profiles_dir = Path(__file__).resolve().parents[3] / "profiles"
    profiles = sorted(profiles_dir.glob("*.yaml"))
    assert profiles, f"no profiles found under {profiles_dir}"

    removed = {
        "BOLTZ2_OPTIMIZATIONS",
        "BOLTZ2_AFFINITY_COMPILE",
        "BOLTZ2_TRITON_KERNELS",
        "BOLTZ2_NO_KERNELS",
        "BOLTZ2_COMPILE_TRANSITION",
        "BOLTZ2_COMPILE_TOKEN_TRANSFORMER",
        "BOLTZ2_COMPILE_MAX_TOKENS",
        "BOLTZ2_COMPILE_WARMUP",
        "BOLTZ2_COMPILE_WARMUP_TOKENS",
        "BOLTZ2_AOT_ARTIFACT_DIR",
    }
    for path in profiles:
        env_vars = yaml.safe_load(path.read_text()).get("env_vars") or {}
        assert not removed & set(env_vars), f"{path.name} sets {sorted(removed & set(env_vars))}"


def test_dataloader_workers_default_to_in_process():
    """0 workers keeps torch from forking off the BentoML worker thread.

    ``num_workers=0`` is the ONLY value that creates no child process: torch
    uses ``_SingleProcessDataLoaderIter`` at exactly 0 and
    ``_MultiProcessingDataLoaderIter`` for every value >= 1. A sync
    ``@bentoml.api`` runs on an anyio worker thread, not the main thread, so any
    non-zero value forks from a non-main thread — the documented deadlock shape,
    and one ``traffic.timeout`` cannot reclaim. See the field comment in
    ``service.py`` for the full argument.
    """
    assert ServiceConfig().num_workers == 0


def test_num_workers_is_not_offered_to_the_tuner():
    """The knob is fixed at 0, so it must not appear as a sweep axis.

    Guards the exact inconsistency this asset shipped with once: ``tuning.yaml``
    declared ``num_workers`` as a swept enum while the profile pinned it to 0 as
    a hazard. Sweeping it would ask the Tuner to run the arm the AIM declares
    unsafe.
    """
    import yaml

    tuning_path = Path(__file__).resolve().parents[3] / "tuning.yaml"
    spec = yaml.safe_load(tuning_path.read_text())
    declared = {knob["name"] for knob in spec.get("knobs", [])}

    assert "num_workers" not in declared, (
        f"num_workers is declared as a tunable knob in {tuning_path}, but it is "
        "fixed at 0 for safety by the profile and the ServiceConfig default. "
        "Remove it from tuning.yaml or lift the restriction — do not ship both."
    )
    assert "optimizations" not in declared, "BOLTZ2_OPTIMIZATIONS is not a sweep axis"


# --------------------------------------------------------------------------- #
# The benchmark fixtures — a timing instrument, deliberately NOT the serving
# recipe. Everything below is about keeping that divergence declared rather than
# discovered.
# --------------------------------------------------------------------------- #

# Request fields that identify a CASE rather than configure the recipe. This is
# the ONLY hardcoded list, and it is the safe direction: forgetting to exclude a
# field makes the divergence check noisier, never blinder.
#
# Everything else ``Boltz2Request`` declares is a recipe knob and is compared.
# DERIVED FROM THE SCHEMA, NOT WHITELISTED: a list of the fields expected to
# matter cannot see a divergence on any other field. Deriving the set from the
# schema covers a field nobody thought of the moment it exists.
_CASE_IDENTITY_FIELDS = frozenset({"version", "sequences", "constraints", "templates", "properties", "name"})


def _recipe_fields() -> tuple[str, ...]:
    """Every ``Boltz2Request`` field that configures the recipe, from the schema."""
    return tuple(sorted(set(Boltz2Request.model_fields) - _CASE_IDENTITY_FIELDS))


def test_the_recipe_field_set_is_derived_and_covers_the_whole_request_schema():
    """The derivation itself, checked — a silently-empty set would pass everything."""
    derived = set(_recipe_fields())
    declared = set(Boltz2Request.model_fields)

    assert _CASE_IDENTITY_FIELDS <= declared, (
        f"_CASE_IDENTITY_FIELDS names fields Boltz2Request does not declare: "
        f"{sorted(_CASE_IDENTITY_FIELDS - declared)}"
    )
    assert derived | _CASE_IDENTITY_FIELDS == declared, "every request field must be identity or recipe"
    assert derived >= {"recycling_steps", "sampling_steps", "diffusion_samples", "step_scale", "output_format"}
    # Fields nobody enumerated but which a fixture could still set.
    assert derived >= {"max_parallel_samples", "method", "msa_pairing_strategy"}


def test_fixtures_set_no_field_the_request_schema_would_silently_ignore(benchmark_fixtures):
    """A typo in a fixture is dropped by the schema, not rejected — so it has to
    be caught here or the case quietly runs a different recipe than it reads."""
    declared = set(Boltz2Request.model_fields)
    for label, fixture in benchmark_fixtures.items():
        unknown = set(fixture["data"]) - declared
        assert not unknown, f"benchmark fixture {label} sets unknown request field(s) {sorted(unknown)}"


def _assert_serving_defaults_intact(benchmark_recipe) -> None:
    """Arm 1 of the two-arm discrimination: has a SERVING default moved?

    Run before any fixture comparison, because a serving default sliding toward
    a fixture value would otherwise show up as "the divergence declaration is
    stale" — the opposite diagnosis, with the opposite fix.
    """
    for field, value in benchmark_recipe["serving_request_defaults"]["fields"].items():
        actual = Boltz2Request.model_fields[field].default
        assert actual == value, (
            f"A SERVING DEFAULT DRIFTED: Boltz2Request.{field} is {actual!r}, but "
            f"benchmark_recipe.json records the served default as {value!r}. The benchmark "
            "fixtures are allowed to diverge from the serving defaults; the serving defaults "
            "are not allowed to move. Restore the default (it must track upstream "
            "`boltz predict` — see boltz_cli_defaults.json), or, if upstream itself moved, "
            "update BOTH reference files together."
        )


def test_serving_request_defaults_have_not_drifted(benchmark_recipe):
    _assert_serving_defaults_intact(benchmark_recipe)


def test_every_fixture_matches_the_declared_benchmark_recipe(benchmark_recipe, benchmark_fixtures):
    """The fixtures ARE the declared timing recipe, field by field, all six.

    Fails when a FIXTURE drifts. It says nothing about the serving defaults —
    that is the next test's job, and the two are kept apart on purpose.
    """
    expected = benchmark_recipe["timing_recipe"]["fields"]
    for label, fixture in benchmark_fixtures.items():
        data = fixture["data"]
        for field, value in expected.items():
            assert data.get(field) == value, (
                f"benchmark fixture {label} has {field}={data.get(field)!r} but the declared "
                f"timing recipe says {value!r}. Either restore the fixture or update "
                f"tests/references/benchmark_recipe.json ."
            )
        assert (
            data.get("use_msa_server") == benchmark_recipe["not_from_timing_recipe"]["use_msa_server"]["fixture_value"]
        )


def test_fixture_divergence_from_the_serving_defaults_is_exactly_what_is_declared(benchmark_recipe, benchmark_fixtures):
    """The core assertion of the fixture/serving split.

    Computes the divergence rather than trusting the declaration, then compares
    the two. Deliberately two separate assertions with two separate messages:

    * the SET check fires when a fixture moved, or when a declaration is
      phantom;
    * the ``serving_value`` check fires when a SERVING default moved — the
      declaration records what the service defaults to, so a drift there
      shows up as the declaration no longer describing reality.
    """
    _assert_serving_defaults_intact(benchmark_recipe)

    declared = benchmark_recipe["declared_divergences"]

    observed: set[str] = set()
    for fixture in benchmark_fixtures.values():
        data = fixture["data"]
        for field in _recipe_fields():
            served = Boltz2Request.model_fields[field].default
            if data.get(field, served) != served:
                observed.add(field)

    assert observed == set(declared), (
        f"benchmark fixtures diverge from the serving defaults on {sorted(observed)}, but "
        f"tests/references/benchmark_recipe.json declares {sorted(declared)}. "
        "Undeclared divergence means a FIXTURE drifted off the published methodology; a "
        "declaration with no matching divergence means the declaration is stale. Fix the "
        "fixture or the declaration — do not widen this assertion."
    )

    for field, entry in declared.items():
        assert Boltz2Request.model_fields[field].default == entry["serving_value"], (
            f"benchmark_recipe.json records the SERVING default for {field} as "
            f"{entry['serving_value']!r}, but Boltz2Request.{field} now defaults to "
            f"{Boltz2Request.model_fields[field].default!r}. A SERVING DEFAULT DRIFTED. The "
            "fixtures are allowed to diverge; the serving defaults are not — they must keep "
            "tracking upstream `boltz predict`."
        )
        assert entry["reason"].strip(), f"divergence {field} has no recorded reason"


def test_fields_declared_non_divergent_really_are(benchmark_recipe, benchmark_fixtures):
    """Where the timing recipe and boltz agree, the fixtures and the service must too."""
    for field, value in benchmark_recipe["fields_that_must_NOT_diverge"].items():
        if field == "note":
            continue
        assert Boltz2Request.model_fields[field].default == value
        for label, fixture in benchmark_fixtures.items():
            assert fixture["data"].get(field) == value, f"{label}.{field}"


def test_all_six_fixtures_request_the_same_recipe(benchmark_fixtures):
    """A timing instrument whose cases disagree is not one instrument."""
    assert len(benchmark_fixtures) == 6
    recipes = {
        label: tuple((f, fixture["data"].get(f)) for f in _recipe_fields())
        for label, fixture in benchmark_fixtures.items()
    }
    distinct = set(recipes.values())
    assert len(distinct) == 1, f"fixtures request {len(distinct)} different recipes: {recipes}"


def test_the_1638_step_scale_is_a_real_deviation_not_a_no_op(benchmark_recipe):
    """Pins the established fact so it cannot quietly be re-described.

    boltz resolves ``step_scale`` per model: 1.5 for boltz2, 1.638 for boltz1
    (``boltz/main.py:1229-1237``). This AIM runs boltz2, so the timing recipe's 1.638 is the
    Boltz-1 constant and setting it changes the sampler.
    """
    upstream = benchmark_recipe["upstream_boltz2_defaults"]["fields"]["step_scale"]
    fixture = benchmark_recipe["timing_recipe"]["fields"]["step_scale"]
    assert upstream == 1.5
    assert fixture == 1.638
    assert fixture != upstream, "if these ever become equal, the divergence declaration must be removed"


def test_shipped_fixtures_are_not_gradeable_by_the_accuracy_gate(benchmark_recipe, benchmark_fixtures):
    """Pins the dormancy so it cannot become invisible.

    The in-image accuracy floors were derived at the serving recipe WITH an MSA.
    The fixtures run none of that, so the gate abstains on them. That is
    intended, and it is also a coverage gap — asserting it here means the gap
    stays a reviewable fact instead of something a future reader rediscovers as
    a silently-green build.
    """
    from harness import accuracy_recipe_mismatch

    declared = set(benchmark_recipe["accuracy_gate_relationship"]["mismatching_fields"])
    assert benchmark_recipe["accuracy_gate_relationship"]["gradeable_by_the_in_image_accuracy_gate"] is False

    for label, fixture in benchmark_fixtures.items():
        mismatch = accuracy_recipe_mismatch(fixture["data"])
        assert mismatch, f"{label} is gradeable — update benchmark_recipe.json and tuning.yaml"
        assert set(mismatch) == declared, f"{label} mismatches {sorted(mismatch)}, declared {sorted(declared)}"


def test_no_shipped_file_points_at_a_builder_working_directory(benchmark_fixtures):
    """Files baked into the image must not cross-reference an agent artifact.

    ``.build/`` is Builder scratch that may never be committed, and a reader
    inside the container has no way to reach it.
    """
    root = Path(__file__).resolve().parents[2]
    shipped = sorted(root.glob("examples/benchmark_sequences/*/*.json")) + [
        root / "tests" / "references" / "benchmark_recipe.json",
        root / "tests" / "references" / "boltz_cli_defaults.json",
    ]
    # 18 fixtures + 2 references. Was 8 while the directory held only the six;
    # the count is here so a glob that silently matches nothing cannot pass.
    assert len(shipped) == 20
    for path in shipped:
        text = path.read_text()
        for dangling in (".build/", "build_report.md", "state.yaml"):
            assert dangling not in text, f"{path.name} references the Builder working directory ({dangling})"
    # And the fixtures still say the thing that pointer was carrying.
    for label, fixture in benchmark_fixtures.items():
        assert "bound" in fixture["metadata"]["recipe"]["msa"], f"{label} lost the comparison caveat"


def test_the_benchmark_recipe_reference_keeps_its_source_caveat(benchmark_recipe):
    """The published material never states that the script parameters produced a timing table.
    Deleting that sentence would turn a bounded alignment claim into an unbounded one."""
    caveat = benchmark_recipe["timing_recipe"]["caveat"]
    assert "does not state" in caveat


def test_fixture_metadata_tells_a_reader_what_the_fixture_is_for():
    """A reader opening one fixture must not have to find this test tier to
    learn that it is not the serving configuration."""
    root = Path(__file__).resolve().parents[2] / "examples" / "benchmark_sequences"
    for path in sorted(root.glob("*/*.json")):
        recipe = json.loads(path.read_text())["metadata"]["recipe"]
        assert "Timing instrument, not an accuracy eval set" in recipe["purpose"]
        assert recipe["source"].strip()
        assert recipe["diverges_from_serving_defaults"].strip()


def test_profile_pins_num_workers_to_zero():
    """The served profile must agree with ServiceConfig's default.

    The profile is the thing aim-runtime actually exports into the serving
    subprocess, so a drift here silently re-enables the fork path. This is
    checked separately from the ``ServiceConfig`` default because the repo's
    ``yaml-key-order`` pre-commit hook rewrites files under ``profiles/`` and
    strips their comments — the reasoning cannot live there, so the assertion
    has to.
    """
    import yaml

    profiles_dir = Path(__file__).resolve().parents[3] / "profiles"
    profiles = sorted(profiles_dir.glob("*.yaml"))
    assert profiles, f"no profiles found under {profiles_dir}"

    for path in profiles:
        env_vars = yaml.safe_load(path.read_text()).get("env_vars") or {}
        assert str(env_vars.get("BOLTZ2_NUM_WORKERS")) == "0", (
            f"{path.name} sets BOLTZ2_NUM_WORKERS={env_vars.get('BOLTZ2_NUM_WORKERS')!r}; "
            "it must be '0' — see the field comment in service.py"
        )
