# Copyright © Advanced Micro Devices, Inc., or its affiliates.
#
# SPDX-License-Identifier: MIT

"""The service's request schema, and the harness ↔ service contract.

The harness and the service are authored as a pair: the harness POSTs a body it
builds itself, and only the service validates it. Nothing else checks that the
two agree, so drift between them would surface as a 400 during a GPU smoke that
costs an allocation. These tests catch it for free.
"""

from __future__ import annotations

import harness as harness_mod
import pytest
import service as service_mod
import torch
from bentoml.exceptions import InvalidArgument
from harness import EVAL_SEQUENCE, FAST_PREDICT_PARAMS, SMOKE_SEQUENCE, Boltz2Harness, _build_predict_payload
from pydantic import ValidationError
from service import Boltz2Request

from aim_runtime.harness import CheckInfo, CheckScope, ModelHarness

MINIMAL = {"sequences": [{"protein": {"id": "A", "sequence": "NLYIQWLKDGGPSSGRPPPS"}}]}


# --------------------------------------------------------------------------- #
# Schema accept / reject
# --------------------------------------------------------------------------- #


def test_minimal_payload_is_accepted():
    req = Boltz2Request(**MINIMAL)
    assert req.sequences[0]["protein"]["id"] == "A"


def test_empty_sequences_is_rejected():
    with pytest.raises(InvalidArgument):
        Boltz2Request(sequences=[])


def test_missing_sequences_is_rejected():
    with pytest.raises(InvalidArgument):
        Boltz2Request()


@pytest.mark.parametrize("bad", ["cif", "PDB", "", "json"])
def test_unknown_output_format_is_rejected(bad):
    with pytest.raises(InvalidArgument):
        Boltz2Request(**MINIMAL, output_format=bad)


@pytest.mark.parametrize("good", ["pdb", "mmcif"])
def test_known_output_formats_are_accepted(good):
    assert Boltz2Request(**MINIMAL, output_format=good).output_format == good


def test_unknown_msa_pairing_strategy_is_rejected():
    with pytest.raises(InvalidArgument):
        Boltz2Request(**MINIMAL, msa_pairing_strategy="clever")


@pytest.mark.parametrize("bad_entry", [{"protein": {}, "ligand": {}}, "protein", 42, []])
def test_sequence_entries_must_be_single_key_mappings(bad_entry):
    with pytest.raises(InvalidArgument):
        Boltz2Request(sequences=[bad_entry])


@pytest.mark.parametrize("smiles", [None, 5, ["CCO"]])
def test_a_ligand_smiles_that_is_not_a_string_is_refused_without_an_affinity_property(smiles):
    with pytest.raises(InvalidArgument, match="ligand smiles must be a string"):
        Boltz2Request(sequences=[{"ligand": {"id": "L", "smiles": smiles}}])


@pytest.mark.parametrize("field", ["sampling_steps", "diffusion_samples", "max_parallel_samples"])
def test_step_counts_must_be_positive(field):
    with pytest.raises(InvalidArgument):
        Boltz2Request(**MINIMAL, **{field: 0})


def test_recycling_steps_may_be_zero_but_not_negative():
    assert Boltz2Request(**MINIMAL, recycling_steps=0).recycling_steps == 0
    with pytest.raises(InvalidArgument):
        Boltz2Request(**MINIMAL, recycling_steps=-1)


# --------------------------------------------------------------------------- #
# Harness ↔ service contract
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("sequence", [SMOKE_SEQUENCE, EVAL_SEQUENCE])
def test_harness_payload_validates_against_the_service_schema(sequence):
    """Every body the harness builds must be accepted by the service."""
    body = _build_predict_payload(sequence, name="t")
    req = Boltz2Request(**body["data"])
    assert req.sequences[0]["protein"]["sequence"] == sequence


def test_harness_fast_params_are_real_request_fields():
    """A typo in FAST_PREDICT_PARAMS would be silently ignored by a lenient
    schema; these must land on declared fields."""
    unknown = set(FAST_PREDICT_PARAMS) - set(Boltz2Request.model_fields)
    assert not unknown, f"harness sends unknown fields: {sorted(unknown)}"


def test_harness_fast_params_reach_the_request():
    req = Boltz2Request(**_build_predict_payload(SMOKE_SEQUENCE)["data"])
    for key, value in FAST_PREDICT_PARAMS.items():
        assert getattr(req, key) == value


def test_harness_payload_builder_can_request_the_msa_server():
    """The builder passes the flag through. No default CI path sets it (test_benchmark_sets.py)."""
    body = _build_predict_payload(EVAL_SEQUENCE, use_msa_server=True)
    assert Boltz2Request(**body["data"]).use_msa_server is True


def test_harness_smoke_payload_is_network_free():
    """The smoke and benchmark paths must not depend on an external service."""
    assert Boltz2Request(**_build_predict_payload(SMOKE_SEQUENCE)["data"]).use_msa_server is False


def test_harness_targets_the_routes_the_service_publishes():
    """The one authored API route is POST /v1/inference.

    Reads the routes off the constructed Service rather than the class:
    @bentoml.service replaces the class with a Service object whose `apis`
    mapping is the authoritative list of what is actually served.
    """
    published = {api.route for api in service_mod.Boltz2Prediction.apis.values()}
    assert published == {"/v1/inference"}, f"service publishes {sorted(published)}"
    assert harness_mod.BENTOML_INFERENCE_ENDPOINT in published


def test_harness_health_endpoint_is_the_adr_0009_path():
    assert harness_mod.BENTOML_HEALTH_ENDPOINT == "/health"


def test_the_adr_0009_get_routes_are_mounted():
    routes = {r.path: sorted(r.methods - {"HEAD"}) for r in service_mod.canonical_app.routes}
    assert routes == {"/health": ["GET"], "/v1/models": ["GET"]}


# --------------------------------------------------------------------------- #
# Harness wiring
# --------------------------------------------------------------------------- #


def test_harness_subclasses_the_runtime_abc():
    assert issubclass(Boltz2Harness, ModelHarness)


def test_harness_declares_the_bentoml_engine():
    assert Boltz2Harness.ENGINE == "bentoml"


def test_every_check_declares_a_scope_and_description():
    checks = Boltz2Harness().list_checks()
    assert checks
    for check in checks:
        assert isinstance(check, CheckInfo)
        assert isinstance(check.scope, CheckScope)
        assert check.description.strip()


def test_check_names_are_unique():
    names = [c.name for c in Boltz2Harness().list_checks()]
    assert len(names) == len(set(names))


def test_harness_exposes_at_least_one_runtime_and_one_offline_check():
    """RUNTIME checks gate every deploy; OFFLINE ones carry the heavy work."""
    scopes = {c.scope for c in Boltz2Harness().list_checks()}
    assert CheckScope.RUNTIME in scopes
    assert CheckScope.OFFLINE in scopes


# --------------------------------------------------------------------------- #
# Service configuration
# --------------------------------------------------------------------------- #


def test_max_concurrency_defaults_to_three_per_accelerator():
    assert service_mod.ServiceConfig(AIM_ACCELERATOR_COUNT=1).effective_max_concurrency == 3
    assert service_mod.ServiceConfig(AIM_ACCELERATOR_COUNT=2).effective_max_concurrency == 6


def test_explicit_max_concurrency_wins():
    assert service_mod.ServiceConfig(BOLTZ2_MAX_CONCURRENCY=2).effective_max_concurrency == 2


def test_max_concurrency_zero_means_derive():
    assert service_mod.ServiceConfig(BOLTZ2_MAX_CONCURRENCY=0, AIM_ACCELERATOR_COUNT=2).effective_max_concurrency == 6


def test_max_concurrency_cannot_be_negative():
    with pytest.raises(ValidationError):
        service_mod.ServiceConfig(BOLTZ2_MAX_CONCURRENCY=-1)


def _request(**overrides):
    return service_mod.Boltz2Request(**{**MINIMAL, **overrides})


@pytest.mark.parametrize(
    "overrides",
    [
        {"recycling_steps": service_mod.MAX_RECYCLING_STEPS + 1},
        {"sampling_steps": service_mod.MAX_SAMPLING_STEPS + 1},
        {"diffusion_samples": service_mod.MAX_DIFFUSION_SAMPLES + 1},
        {"max_parallel_samples": service_mod.MAX_PARALLEL_SAMPLES + 1},
        {"sequences": [{"protein": {"id": "A", "sequence": "A" * (service_mod.MAX_TOKENS + 1)}}]},
        {"sequences": [{"protein": {"id": ["A", "B"], "sequence": "A" * service_mod.MAX_TOKENS}}]},
    ],
)
def test_a_request_above_a_limit_is_a_422(overrides):
    with pytest.raises(service_mod.UnprocessableRequest) as raised:
        service_mod.check_limits(_request(**overrides))
    assert raised.value.error_code == 422


def test_a_request_at_every_limit_is_accepted():
    service_mod.check_limits(
        _request(
            recycling_steps=service_mod.MAX_RECYCLING_STEPS,
            sampling_steps=service_mod.MAX_SAMPLING_STEPS,
            diffusion_samples=service_mod.MAX_DIFFUSION_SAMPLES,
            max_parallel_samples=service_mod.MAX_PARALLEL_SAMPLES,
            sequences=[{"protein": {"id": "A", "sequence": "A" * service_mod.MAX_TOKENS}}],
        )
    )


def test_the_default_recipe_is_inside_the_limits():
    service_mod.check_limits(_request())


def test_ligand_heavy_atoms_count_as_tokens():
    assert service_mod.estimate_tokens([{"ligand": {"id": "L", "smiles": "CC(=O)Cl"}}]) == 4


def test_request_timeout_must_be_positive():
    with pytest.raises(ValidationError):
        service_mod.ServiceConfig(BOLTZ2_REQUEST_TIMEOUT_SECONDS=0)


def test_an_inference_reserve_that_leaves_no_budget_is_refused():
    with pytest.raises(ValidationError):
        service_mod.ServiceConfig(BOLTZ2_REQUEST_TIMEOUT_SECONDS=600, BOLTZ2_INFERENCE_RESERVE_SECONDS=600)


def test_the_default_budget_leaves_room_for_inference():
    cfg = service_mod.ServiceConfig()
    assert cfg.inference_reserve_seconds < cfg.request_timeout_seconds - service_mod.response_slack(
        cfg.request_timeout_seconds
    )
    assert cfg.msa_timeout_seconds < cfg.request_timeout_seconds


# --------------------------------------------------------------------------- #
# complex_plddt gate — the accuracy criterion tuning.yaml hands the Tuner.
#
# grade_confidence is pure, so every branch is exercised here rather than on a
# GPU. The floors themselves are derived in harness.py's constant comments.
# --------------------------------------------------------------------------- #

# Six healthy complex_plddt values, one per default-set target.
BUILD_PLDDT = {
    "7qsj": 0.93,
    "8are": 0.96,
    "8c4d": 0.90,
    "8cpk": 0.95,
    "8eil": 0.92,
    "8gpp": 0.93,
}


def _cases(values: dict[str, float]) -> list[dict]:
    return [{"label": k, "rmsd_a": 0.5, "complex_plddt": v} for k, v in values.items()]


def test_gate_floors_are_ordered_and_inside_the_plddt_range():
    assert 0.0 < harness_mod.PLDDT_PER_CASE_FLOOR < harness_mod.PLDDT_MEAN_FLOOR < 1.0


def test_healthy_values_pass_the_floors():
    """A threshold that fails healthy values is useless."""
    passed, mean, detail = harness_mod.grade_confidence(_cases(BUILD_PLDDT))
    assert passed, detail
    assert mean == pytest.approx(0.9317, abs=5e-5)


def test_a_sagging_mean_fails_even_when_no_single_case_is_catastrophic():
    """Every case stays well inside 'confident', but the mean leaves 'very
    high' — the broad, uniform degradation a wrong precision would produce."""
    sagged = {k: 0.85 for k in BUILD_PLDDT}
    passed, mean, detail = harness_mod.grade_confidence(_cases(sagged))
    assert not passed, detail
    assert mean == pytest.approx(0.85)
    assert all(v > harness_mod.PLDDT_PER_CASE_FLOOR for v in sagged.values())


def test_one_collapsed_case_fails_even_when_the_mean_survives():
    """The reason the per-case floor exists: at n=6 a single collapse can hide
    behind five healthy targets and still clear the mean floor."""
    collapsed = dict.fromkeys(BUILD_PLDDT, 1.0)
    collapsed["8c4d"] = 0.65
    passed, mean, detail = harness_mod.grade_confidence(_cases(collapsed))
    assert not passed, detail
    assert mean > harness_mod.PLDDT_MEAN_FLOOR, "this fixture must fail on the per-case floor, not the mean"


def test_cases_without_confidence_are_ignored_not_failed():
    """A case can be unscored for reasons this gate has no opinion about; the
    accuracy check already reports those."""
    records = _cases(BUILD_PLDDT) + [{"label": "8xyz", "rmsd_a": None, "error": "reference not staged"}]
    passed, _, detail = harness_mod.grade_confidence(records)
    assert passed, detail
    assert "6/7" in detail


def test_no_confidence_at_all_fails_rather_than_passing_vacuously():
    passed, mean, detail = harness_mod.grade_confidence([{"label": "a", "rmsd_a": None}])
    assert not passed
    assert mean is None
    assert "cannot be applied" in detail


def test_confidence_regression_is_a_declared_check():
    names = {c.name for c in Boltz2Harness().list_checks()}
    assert "confidence_regression" in names


# --------------------------------------------------------------------------- #
# Which configuration the gate is allowed to judge.
#
# The floors are properties of the model RUN A PARTICULAR WAY, so the gate has to
# know that way and abstain outside it. These tests own that logic; the shipped
# fixtures' side of the same contract lives in test_recipe_parity.py.
# --------------------------------------------------------------------------- #


def test_served_request_defaults_mirror_the_real_request_schema():
    """harness.py duplicates the service's defaults instead of importing them —
    importing service.py would drag BentoML into every aim-runtime harness
    import. This is the pin that keeps the copy from drifting."""
    for field, value in harness_mod.SERVED_REQUEST_DEFAULTS.items():
        assert Boltz2Request.model_fields[field].default == value, (
            f"harness.SERVED_REQUEST_DEFAULTS[{field!r}] is {value!r} but "
            f"Boltz2Request.{field} defaults to {Boltz2Request.model_fields[field].default!r}"
        )


def test_the_gate_recipe_is_the_serving_defaults_plus_an_msa():
    """One departure, and it must stay one: an MSA is a first-order accuracy
    input and the floors are derived with one."""
    recipe = harness_mod.ACCURACY_GATE_RECIPE
    served = harness_mod.SERVED_REQUEST_DEFAULTS
    differing = {k for k in recipe if recipe[k] != served[k]}
    assert differing == {"use_msa_server"}
    assert recipe["use_msa_server"] is True


def test_a_case_at_the_gate_recipe_is_gradeable():
    assert harness_mod.accuracy_recipe_mismatch(dict(harness_mod.ACCURACY_GATE_RECIPE)) == {}


def test_an_absent_field_resolves_to_the_served_default_not_to_a_match():
    """The trap this guards: treating "the case says nothing about the MSA" as
    "the case has an MSA" would grade a single-sequence prediction against
    MSA-derived floors."""
    mismatch = harness_mod.accuracy_recipe_mismatch({"sampling_steps": 200})
    assert set(mismatch) == {"use_msa_server"}
    assert mismatch["use_msa_server"] == {"gate": True, "case": False}


@pytest.mark.parametrize(
    ("override", "expected_case"),
    [
        ({"sampling_steps": 50}, 50),
        ({"step_scale": 1.638}, 1.638),
        ({"recycling_steps": 1}, 1),
        ({"diffusion_samples": 5}, 5),
    ],
)
def test_each_recipe_field_is_actually_checked(override, expected_case):
    case = {**harness_mod.ACCURACY_GATE_RECIPE, **override}
    mismatch = harness_mod.accuracy_recipe_mismatch(case)
    field = next(iter(override))
    assert set(mismatch) == {field}
    assert mismatch[field]["case"] == expected_case


def test_the_timing_recipe_is_not_gradeable_and_says_why():
    """The shipped benchmark fixtures' recipe, spelled out here so the two
    divergences are visible at the point the gate rejects them."""
    timing = {
        "use_msa_server": False,
        "sampling_steps": 50,
        "recycling_steps": 3,
        "diffusion_samples": 1,
        "step_scale": 1.638,
        "output_format": "mmcif",
    }
    mismatch = harness_mod.accuracy_recipe_mismatch(timing)
    assert set(mismatch) == {"use_msa_server", "sampling_steps", "step_scale"}

    rendered = harness_mod.describe_recipe_mismatch(mismatch)
    for token in ("use_msa_server", "sampling_steps", "step_scale"):
        assert token in rendered


def test_abstention_detail_names_the_fields_and_refuses_to_call_it_a_regression():
    per_case = [
        {
            "label": "8eil",
            "rmsd_a": 12.8,
            "complex_plddt": 0.53,
            "accuracy_gradeable": False,
            "accuracy_recipe_mismatch": {"use_msa_server": {"gate": True, "case": False}},
        }
    ]
    detail = Boltz2Harness._gate_abstention_detail(per_case)
    assert "GATE NOT APPLIED" in detail
    assert "use_msa_server" in detail
    assert "ungraded_" in detail


def test_summarize_writes_nothing_for_an_empty_record_set():
    """An absent key must mean "no number of this kind was produced", never 0."""
    metrics: dict = {}
    harness_mod._summarize(metrics, [], "")
    assert metrics == {}


def test_summarize_separates_graded_from_ungraded_means():
    metrics: dict = {}
    harness_mod._summarize(metrics, [{"rmsd_a": 0.5, "complex_plddt": 0.93}], "")
    harness_mod._summarize(metrics, [{"rmsd_a": 12.8, "complex_plddt": 0.53}], "ungraded_")
    assert metrics["mean_ca_rmsd_a"] == 0.5
    assert metrics["mean_complex_plddt"] == 0.93
    assert metrics["ungraded_mean_ca_rmsd_a"] == 12.8
    assert metrics["ungraded_mean_complex_plddt"] == 0.53


def test_declared_gate_matches_the_thresholds_the_harness_enforces():
    """tuning.yaml is what the Tuner reads; harness.py is what actually fails a
    run. If they drift, the Tuner is judging against a number nothing enforces.
    """
    import pathlib

    import yaml

    spec = yaml.safe_load((pathlib.Path(__file__).resolve().parents[3] / "tuning.yaml").read_text())
    accuracy = spec["accuracy"]

    assert accuracy["metric"] == "complex_plddt"
    assert accuracy["min_mean"] == harness_mod.PLDDT_MEAN_FLOOR
    assert accuracy["min_per_case"] == harness_mod.PLDDT_PER_CASE_FLOOR


def test_tuning_yaml_tells_the_truth_about_whether_the_gate_covers_the_shipped_set():
    """``tuning.yaml`` must not name a set its floors do not describe.

    The shipped fixtures run the timing recipe, not the configuration the floors
    were derived under, so grading them would report a ``complex_plddt``
    regression that is only a configuration difference. The two claims below are
    computed from the fixtures rather than asserted in prose.
    """
    import json
    import pathlib

    import yaml

    asset = pathlib.Path(__file__).resolve().parents[3]
    accuracy = yaml.safe_load((asset / "tuning.yaml").read_text())["accuracy"]

    fixtures = sorted((asset / "image" / "examples" / "benchmark_sequences").glob("*/*.json"))
    assert fixtures
    any_gradeable = any(not harness_mod.accuracy_recipe_mismatch(json.loads(p.read_text())["data"]) for p in fixtures)

    assert accuracy["gate_applies_to_shipped_benchmark_set"] is any_gradeable, (
        "tuning.yaml claims gate_applies_to_shipped_benchmark_set="
        f"{accuracy['gate_applies_to_shipped_benchmark_set']}, but computing it from the "
        f"shipped fixtures gives {any_gradeable}."
    )
    assert (
        accuracy["eval_set_baked_in_image"] is any_gradeable
    ), "eval_set_baked_in_image must equal whether a gradeable case set is actually baked."
    if not any_gradeable:
        assert "NO case set matching this recipe is currently baked" in accuracy["eval_set"]


# --------------------------------------------------------------------------- #
# inference / _runtime_info — the response must report the REAL observed
# compile state of this worker's model, not a default or a stale carry-over.
# --------------------------------------------------------------------------- #


class _FakeContext:
    """Stand-in for ``bentoml.Context``: only ``response`` is touched on the
    success path these tests exercise."""

    def __init__(self, arrived_at: float | None = None):
        self.response = type("Resp", (), {"status_code": 200, "headers": {}})()
        scope = {} if arrived_at is None else {service_mod.ARRIVAL_SCOPE_KEY: arrived_at}
        self.request = type("Req", (), {"scope": scope})()


def _install_fake_model(monkeypatch, tmp_path, *, status_map=None):
    """Build the service with a fake, non-BentoML model object standing in
    for the real Boltz-2 module, and a fake ``run_boltz2_prediction`` that
    returns a minimal success result. ``status_map``, if given, is attached to
    the fake model so ``compilation.compile_status`` reports it verbatim —
    this is what lets a test assert on a REAL, specific observed state rather
    than whatever a fresh model happens to default to.
    """
    import compilation
    import service

    cache_dir = tmp_path / "cache"
    cache_dir.mkdir()
    monkeypatch.setattr(service.CONFIG, "cache", cache_dir)
    monkeypatch.setattr(service.CONFIG, "download_assets", False)
    monkeypatch.setattr(service, "assets_present", lambda cache, *args: True)

    fake_model = type("FakeModel", (), {})()
    if status_map is not None:
        setattr(fake_model, compilation._STATUS_ATTR, status_map)  # noqa: SLF001

    monkeypatch.setattr(service, "load_boltz2_model", lambda cache, **kwargs: fake_model)

    def _fake_run_boltz2_prediction(**kwargs):
        return {
            "error": False,
            "structures": [{"record_id": "A", "format": "pdb", "content": "ATOM..."}],
            "confidence": {},
            "timing": {"runtime_s": 0.01},
        }

    monkeypatch.setattr(service, "run_boltz2_prediction", _fake_run_boltz2_prediction)
    return service, fake_model


def _minimal_request(service):
    return service.Boltz2Request(**MINIMAL)


def test_msa_server_request_is_refused_when_the_deployment_has_no_msa_server(tmp_path, monkeypatch):
    from bentoml.exceptions import InvalidArgument

    service, _ = _install_fake_model(monkeypatch, tmp_path)
    monkeypatch.setattr(service.CONFIG, "msa_server_url", "")
    request = service.Boltz2Request(**{**MINIMAL, "use_msa_server": True})
    with pytest.raises(InvalidArgument):
        service.Boltz2Prediction.inner().inference(request, _FakeContext())


def _inference_raising(monkeypatch, tmp_path, exc):
    from bentoml.exceptions import BentoMLException

    service, _ = _install_fake_model(monkeypatch, tmp_path)

    def _boom(**kwargs):
        raise exc

    monkeypatch.setattr(service, "run_boltz2_prediction", _boom)
    ctx = _FakeContext()
    try:
        result = service.Boltz2Prediction.inner().inference(_minimal_request(service), ctx)
    except BentoMLException as raised:
        return raised, ctx
    return result, ctx


# --------------------------------------------------------------------------- #
# Request budget: the deadline is set inside traffic.timeout and counted from arrival
# --------------------------------------------------------------------------- #


def test_a_request_that_waited_too_long_is_refused_before_it_starts(tmp_path, monkeypatch):
    import time

    service, _ = _install_fake_model(monkeypatch, tmp_path)
    started = []
    monkeypatch.setattr(service, "run_boltz2_prediction", lambda **kwargs: started.append(1))
    arrived_at = time.monotonic() - service.CONFIG.request_timeout_seconds

    ctx = _FakeContext(arrived_at=arrived_at)
    result = service.Boltz2Prediction.inner().inference(_minimal_request(service), ctx)

    assert started == []
    assert ctx.response.status_code == 503
    assert ctx.response.headers["Retry-After"] == "60"
    assert "Server busy" in result["error"]


def test_the_msa_deadline_given_to_the_runner_is_inside_the_request_budget(tmp_path, monkeypatch):
    import time

    service, _ = _install_fake_model(monkeypatch, tmp_path)
    seen = {}

    def _capture(**kwargs):
        seen.update(kwargs)
        return {
            "structures": [{"record_id": "A", "format": "pdb", "content": "x"}],
            "confidence": {},
            "timing": {"runtime_s": 0.0},
        }

    monkeypatch.setattr(service, "run_boltz2_prediction", _capture)
    arrived_at = time.monotonic()

    service.Boltz2Prediction.inner().inference(_minimal_request(service), _FakeContext(arrived_at=arrived_at))

    cfg = service.CONFIG
    assert seen["msa_deadline"] <= arrived_at + cfg.request_timeout_seconds - cfg.inference_reserve_seconds + 1
    assert seen["msa_deadline"] <= arrived_at + cfg.msa_timeout_seconds + 1


def test_the_arrival_stamp_is_set_before_the_app_runs():
    import asyncio

    seen = {}

    async def app(scope, receive, send):
        seen.update(scope)

    asyncio.run(service_mod.ArrivalStampMiddleware(app)({"type": "http"}, None, None))
    assert isinstance(seen[service_mod.ARRIVAL_SCOPE_KEY], float)


def test_the_service_deadline_is_set_inside_the_serving_timeout():
    """The BentoML timeout is a backstop; the request returns on its own before it."""
    cfg = service_mod.ServiceConfig()
    budget = service_mod.build_budget(arrived_at=0.0, now=0.0, request_timeout_s=cfg.request_timeout_seconds)
    assert budget.deadline < cfg.request_timeout_seconds


def test_an_unexpected_failure_is_a_500_without_the_exception_text(tmp_path, monkeypatch):
    from bentoml.exceptions import InternalServerError

    raised, _ = _inference_raising(monkeypatch, tmp_path, RuntimeError("/secret/path CUDA out of memory"))
    assert isinstance(raised, InternalServerError)
    assert "secret" not in str(raised) and "CUDA" not in str(raised)


def test_a_value_error_from_the_input_is_a_400_without_the_exception_text(tmp_path, monkeypatch):
    from bentoml.exceptions import InvalidArgument

    raised, _ = _inference_raising(monkeypatch, tmp_path, ValueError("/secret/path parse error"))
    assert isinstance(raised, InvalidArgument)
    assert "secret" not in str(raised)


def test_an_msa_budget_timeout_is_a_503_with_retry_after(tmp_path, monkeypatch):
    from runner import MsaServerTimeout

    result, ctx = _inference_raising(monkeypatch, tmp_path, MsaServerTimeout("host colabfold took 9999 s"))
    assert ctx.response.status_code == 503
    assert ctx.response.headers["Retry-After"] == "60"
    assert "colabfold" not in result["error"]


def test_a_result_without_records_is_a_400(tmp_path, monkeypatch):
    from bentoml.exceptions import InvalidArgument

    service, _ = _install_fake_model(monkeypatch, tmp_path)
    monkeypatch.setattr(
        service,
        "run_boltz2_prediction",
        lambda **kwargs: {"error": True, "message": "Input produced no records.", "structures": []},
    )
    with pytest.raises(InvalidArgument):
        service.Boltz2Prediction.inner().inference(_minimal_request(service), _FakeContext())


def test_a_successful_answer_has_no_error_field(tmp_path, monkeypatch):
    service, _ = _install_fake_model(monkeypatch, tmp_path)
    result = service.Boltz2Prediction.inner().inference(_minimal_request(service), _FakeContext())
    assert "error" not in result


def test_inference_reports_the_models_actual_observed_compile_state(tmp_path, monkeypatch):
    """A real degraded/mixed state attached to the fake model must come back
    verbatim in the response, not a default or empty dict."""
    import compilation

    degraded = compilation.CompileStatus(
        mode=compilation.COMPILE_INDUCTOR,
        instances=2,
        invoked=True,
        compiled_calls=3,
        eager_fallback_calls=1,
    )
    service, _ = _install_fake_model(
        monkeypatch, tmp_path, status_map={"transition": degraded, "token_transformer": degraded}
    )

    import rocm_triangle.adapter as adapter

    monkeypatch.setattr(adapter, "_MUL_ENABLED", True)
    monkeypatch.setattr(adapter, "_ATTN_ENABLED", True)
    monkeypatch.setattr(adapter, "_MUL_IMPL", adapter.MUL_IMPL_L1)
    svc = service.Boltz2Prediction.inner()
    result = svc.inference(_minimal_request(service), _FakeContext())

    runtime = result["runtime"]
    assert runtime["optimizations"] == "on"
    assert runtime["triton"] == {"attention": True, "mul_impl": "l1"}
    assert not {
        "triton_kernels",
        "use_kernels",
        "compile_transition",
        "compile_token_transformer",
        "compile_max_tokens",
        "compile_routing",
    } & set(runtime)
    status = result["runtime"]["compile_status"]
    assert status["transition"]["state"] == "mixed-partial-degradation"
    assert status["token_transformer"]["state"] == "mixed-partial-degradation"
    assert status["transition"]["instances"] == 2


def test_inference_reports_the_cuda_peak_inside_runtime_only(tmp_path, monkeypatch):
    service, _ = _install_fake_model(monkeypatch, tmp_path)
    monkeypatch.setattr(
        service,
        "run_boltz2_prediction",
        lambda **kwargs: {
            "error": False,
            "structures": [{"record_id": "A", "format": "pdb", "content": "ATOM..."}],
            "confidence": {},
            "timing": {"runtime_s": 0.01},
            "cuda_peak_gb": 7.5,
        },
    )

    result = service.Boltz2Prediction.inner().inference(_minimal_request(service), _FakeContext())

    assert result["runtime"]["cuda_peak_gb"] == 7.5
    assert "cuda_peak_gb" not in result


def test_inference_reports_a_null_cuda_peak_when_the_runner_gives_none(tmp_path, monkeypatch):
    service, _ = _install_fake_model(monkeypatch, tmp_path)

    result = service.Boltz2Prediction.inner().inference(_minimal_request(service), _FakeContext())

    assert "cuda_peak_gb" in result["runtime"] and result["runtime"]["cuda_peak_gb"] is None


def test_inference_reports_eager_disabled_when_compile_is_turned_off(tmp_path, monkeypatch):
    """A model installed with compile mode ``none`` must report
    "eager-disabled" for both selectors — not "not-yet-invoked" or any
    compiled-looking state."""
    import compilation

    off = {
        "transition": compilation.CompileStatus(mode=compilation.COMPILE_NONE),
        "token_transformer": compilation.CompileStatus(mode=compilation.COMPILE_NONE),
    }
    service, _ = _install_fake_model(monkeypatch, tmp_path, status_map=off)

    svc = service.Boltz2Prediction.inner()
    result = svc.inference(_minimal_request(service), _FakeContext())

    status = result["runtime"]["compile_status"]
    assert status["transition"]["state"] == "eager-disabled"
    assert status["token_transformer"]["state"] == "eager-disabled"


def test_inference_does_not_leak_a_prior_models_compile_state(tmp_path, monkeypatch):
    """A fresh model instance — one install_compilation has never touched —
    must report an empty compile_status, never a state copied from some
    other model instance this process previously served."""
    service, fake_model = _install_fake_model(monkeypatch, tmp_path, status_map=None)

    svc = service.Boltz2Prediction.inner()
    result = svc.inference(_minimal_request(service), _FakeContext())

    # No install_compilation ever ran against fake_model, so there is no
    # per-selector status to report at all — this must read as "nothing
    # observed yet", not as some other instance's leftover state.
    assert result["runtime"]["compile_status"] == {}


def test_inference_reports_a_state_a_real_install_compilation_call_actually_computed(tmp_path, monkeypatch):
    """The three tests above hand-attach a CompileStatus directly onto a bare
    fake object, which only proves inference reads whatever is attached —
    it says nothing about whether the REAL install_compilation/
    _wrap_with_eager_fallback/compile_status code actually produces that
    value. This test drives that real code: a real ``boltz.Transition``
    instance is wrapped by the real ``install_compilation``, called once for
    real, and the resulting state is read back through inference exactly
    as a real deployment would observe it. ``torch.compile`` itself is
    stubbed to a passthrough so this runs on CPU with no real Dynamo/Inductor
    build — everything around that one call is the genuine code path.
    """
    pytest.importorskip("boltz.model.layers.transition")
    import compilation
    import service
    import torch
    from boltz.model.layers.transition import Transition

    monkeypatch.setattr(torch, "compile", lambda fn, *args, **kwargs: fn)

    class _RealEnoughModel(torch.nn.Module):
        """A real target-class instance, not a stand-in shaped like one —
        install_compilation's isinstance scan must match it for real."""

        def __init__(self):
            super().__init__()
            self.transition = Transition(dim=8, hidden=16)

        def forward(self, x):
            return self.transition(x)

    cache_dir = tmp_path / "cache"
    cache_dir.mkdir()
    monkeypatch.setattr(service.CONFIG, "cache", cache_dir)
    monkeypatch.setattr(service.CONFIG, "download_assets", False)
    monkeypatch.setattr(service, "assets_present", lambda cache, *args: True)

    def _fake_load_boltz2_model(cache, **kwargs):
        # Mirrors the real load_boltz2_model's documented ordering — install
        # the real compilation wrapper AFTER the model is built and in eval()
        # — against a lightweight model instead of a full checkpoint load.
        model = _RealEnoughModel().eval()
        compilation.install_compilation(
            model,
            compilation.CompileConfig(
                transition=compilation.COMPILE_INDUCTOR,
                token_transformer=compilation.COMPILE_NONE,
            ),
        )
        # One real forward call through the wrapped transition, so the
        # reported state reflects an ACTUAL observed call, not just that a
        # wrapper was installed.
        with torch.no_grad():
            model.transition(torch.randn(1, 4, 8))
        return model

    monkeypatch.setattr(service, "load_boltz2_model", _fake_load_boltz2_model)

    def _fake_run_boltz2_prediction(**kwargs):
        return {
            "error": False,
            "structures": [{"record_id": "A", "format": "pdb", "content": "ATOM..."}],
            "confidence": {},
            "timing": {"runtime_s": 0.01},
        }

    monkeypatch.setattr(service, "run_boltz2_prediction", _fake_run_boltz2_prediction)

    svc = service.Boltz2Prediction.inner()
    result = svc.inference(_minimal_request(service), _FakeContext())

    status = result["runtime"]["compile_status"]
    assert status["transition"]["state"] == "compiled-executed"
    assert status["transition"]["instances"] == 1
    assert status["transition"]["compiled_calls"] >= 1
    assert status["token_transformer"]["state"] == "eager-disabled"

    # Model isolation: a second, independently-built model that
    # install_compilation has not yet seen must report its own fresh state —
    # never the first model's history.
    second_model = _RealEnoughModel().eval()
    compilation.install_compilation(
        second_model,
        compilation.CompileConfig(transition=compilation.COMPILE_INDUCTOR, token_transformer=compilation.COMPILE_NONE),
    )
    second_status = compilation.compile_status(second_model)
    assert second_status["transition"]["state"] == "not-yet-invoked"
    assert second_status["transition"]["compiled_calls"] == 0


# --------------------------------------------------------------------------- #
# Free-text fields that Boltz looks up
# --------------------------------------------------------------------------- #


def test_a_method_value_boltz_does_not_know_is_refused_at_once():
    with pytest.raises(InvalidArgument, match="method must be one of"):
        _request(method="xxxxx")


@pytest.mark.parametrize("method", ["x-ray diffraction", "X-RAY DIFFRACTION", "md", "afdb"])
def test_a_method_value_boltz_knows_is_accepted_in_any_case(method):
    assert _request(method=method).method == method


def test_the_method_values_are_the_ones_boltz_has():
    const = pytest.importorskip("boltz.data.const")
    assert set(service_mod.BOLTZ_METHODS) == set(const.method_types_ids)


def test_an_unknown_entity_type_and_version_are_refused():
    with pytest.raises(InvalidArgument, match="entry type"):
        _request(sequences=[{"protien": {"id": "A", "sequence": "AAA"}}])
    with pytest.raises(InvalidArgument, match="version must be 1"):
        _request(version=2)


def test_an_invalid_request_gets_a_short_message_per_field_without_url_or_input(caplog):
    secret = "SECRET-VALUE-9"
    with caplog.at_level("WARNING"), pytest.raises(InvalidArgument) as raised:
        _request(recycling_steps=secret, method="nope" + secret, sampling_steps=0)

    message = str(raised.value)
    assert "recycling_steps:" in message and "method:" in message and "sampling_steps:" in message
    assert "pydantic.dev" not in message and secret not in message and "http" not in message
    assert any(secret in r.getMessage() for r in caplog.records), "the full detail goes to the server log"


# --------------------------------------------------------------------------- #
# Errors that BentoML builds before Boltz2Request runs
# --------------------------------------------------------------------------- #

BENTOML_400 = {
    "error": "1 validation error for Input",
    "detail": [
        {
            "type": "missing",
            "loc": ["data"],
            "msg": "Field required",
            "url": "https://errors.pydantic.dev/2.14/v/missing",
            "input": {"secret": "SECRET-VALUE-9"},
        }
    ],
}


def _normalized(status, document):
    import json

    body = service_mod.normalize_error_body(status, json.dumps(document).encode())
    return None if body is None else json.loads(body)


def test_the_bentoml_validation_body_is_shortened():
    assert _normalized(400, BENTOML_400) == {"error": "Invalid request. data: Field required"}


def test_a_list_body_from_a_service_exception_becomes_one_object():
    assert _normalized(400, [{"error": "Invalid request. method: bad"}]) == {"error": "Invalid request. method: bad"}
    assert _normalized(503, [{"error": "Server busy"}]) == {"error": "Server busy"}


def test_a_message_key_becomes_error():
    assert _normalized(503, {"message": "Retry later"}) == {"error": "Retry later"}


def test_the_text_of_a_400_and_a_422_starts_with_invalid_request_once():
    assert _normalized(422, {"error": "tokens is 2400; the limit is 2364"}) == {
        "error": "Invalid request. tokens is 2400; the limit is 2364"
    }
    assert _normalized(400, {"error": "Invalid request. a: b"}) == {"error": "Invalid request. a: b"}


@pytest.mark.parametrize("status", [429, 500, 503, 504])
def test_the_text_of_other_errors_is_kept(status):
    assert _normalized(status, {"error": "Too many requests"}) == {"error": "Too many requests"}


@pytest.mark.parametrize("body", [b"not json", b"[]", b"[1]", b'{"x": 1}', b'{"detail": "x"}', b'{"detail": [1]}'])
def test_other_bodies_are_left_alone(body):
    assert service_mod.normalize_error_body(400, body) is None


def _run_through_middleware(status, body):
    import asyncio

    sent = []

    async def app(scope, receive, send):
        await send({"type": "http.response.start", "status": status, "headers": [(b"content-length", b"1")]})
        await send({"type": "http.response.body", "body": body[:5], "more_body": True})
        await send({"type": "http.response.body", "body": body[5:]})

    async def send(message):
        sent.append(message)

    asyncio.run(service_mod.ErrorBodyMiddleware(app)({"type": "http"}, None, send))
    return sent


@pytest.mark.parametrize("status", [400, 422, 500, 503])
def test_the_middleware_rewrites_an_error_answer_in_chunks(status):
    import json

    sent = _run_through_middleware(status, json.dumps(BENTOML_400).encode())

    start, body = sent
    assert start["status"] == status
    assert dict(start["headers"])[b"content-length"] == str(len(body["body"])).encode()
    assert b"pydantic.dev" not in body["body"] and b"SECRET" not in body["body"]
    assert list(json.loads(body["body"])) == ["error"]


def test_the_middleware_leaves_a_success_answer_unchanged():
    sent = _run_through_middleware(200, b'{"detail": [{"loc": [], "msg": "x"}]}')
    assert [m["type"] for m in sent] == ["http.response.start", "http.response.body", "http.response.body"]
    assert b"".join(m["body"] for m in sent[1:]) == b'{"detail": [{"loc": [], "msg": "x"}]}'


# --------------------------------------------------------------------------- #
# seed
# --------------------------------------------------------------------------- #


def test_seed_is_optional_and_defaults_to_none():
    assert _request().seed is None


@pytest.mark.parametrize("seed", [0, 1, 2**32 - 1])
def test_a_seed_in_range_is_accepted(seed):
    assert _request(seed=seed).seed == seed


@pytest.mark.parametrize("seed", [-1, 2**32, "abc", 1.5])
def test_a_seed_out_of_range_is_refused(seed):
    with pytest.raises(InvalidArgument, match="seed"):
        _request(seed=seed)


def test_the_seed_is_applied_during_the_prediction_and_reported(tmp_path, monkeypatch):
    import sys
    import types

    torch = pytest.importorskip("torch")
    service, _ = _install_fake_model(monkeypatch, tmp_path)
    seeded = []
    lightning = types.ModuleType("pytorch_lightning")
    lightning.seed_everything = lambda seed, *a, **k: seeded.append(seed) or seed
    monkeypatch.setitem(sys.modules, "pytorch_lightning", lightning)
    during = []

    def _run(**kwargs):
        during.append(torch.are_deterministic_algorithms_enabled())
        return {
            "structures": [{"record_id": "A", "format": "pdb", "content": "x"}],
            "confidence": {},
            "timing": {"runtime_s": 0.0},
        }

    monkeypatch.setattr(service, "run_boltz2_prediction", _run)
    before = torch.are_deterministic_algorithms_enabled()
    try:
        seeded_result = service.Boltz2Prediction.inner().inference(_request(seed=42), _FakeContext())
        plain_result = service.Boltz2Prediction.inner().inference(_request(), _FakeContext())
        after = torch.are_deterministic_algorithms_enabled()
    finally:
        torch.use_deterministic_algorithms(before)

    assert seeded == [42]
    assert during == [True, before]
    assert after is before
    assert seeded_result["seed"] == 42 and plain_result["seed"] is None


def test_a_header_only_msa_is_a_short_400_at_the_schema():
    with pytest.raises(InvalidArgument, match="msa has no sequence line") as raised:
        _request(sequences=[{"protein": {"id": "A", "sequence": "NLYIQWLKDG", "msa": ">"}}])
    assert "pydantic.dev" not in str(raised.value)


# --------------------------------------------------------------------------- #
# Start-up without the weights
# --------------------------------------------------------------------------- #


def test_start_up_stops_when_downloads_are_off_and_the_cache_is_empty(tmp_path, monkeypatch):
    monkeypatch.setattr(service_mod, "assets_present", lambda cache, *args: False)
    config = service_mod.ServiceConfig(BOLTZ2_DOWNLOAD_ASSETS="false", AIM_CACHE_PATH=str(tmp_path))
    with pytest.raises(RuntimeError, match="BOLTZ2_DOWNLOAD_ASSETS is off"):
        service_mod.require_staged_assets(config)


def test_start_up_goes_on_when_the_weights_are_staged_or_downloads_are_on(tmp_path, monkeypatch):
    monkeypatch.setattr(service_mod, "assets_present", lambda cache, *args: True)
    service_mod.require_staged_assets(
        service_mod.ServiceConfig(BOLTZ2_DOWNLOAD_ASSETS="false", AIM_CACHE_PATH=str(tmp_path))
    )
    monkeypatch.setattr(service_mod, "assets_present", lambda cache, *args: False)
    service_mod.require_staged_assets(
        service_mod.ServiceConfig(BOLTZ2_DOWNLOAD_ASSETS="true", AIM_CACHE_PATH=str(tmp_path))
    )


def test_start_up_stops_when_the_cache_is_not_writable_and_the_weights_are_missing(tmp_path, monkeypatch):
    monkeypatch.setattr(service_mod, "assets_present", lambda cache, *args: False)
    monkeypatch.setattr(service_mod, "cache_is_writable", lambda cache: False)
    config = service_mod.ServiceConfig(BOLTZ2_DOWNLOAD_ASSETS="true", AIM_CACHE_PATH=str(tmp_path))
    with pytest.raises(RuntimeError, match="not writable") as raised:
        service_mod.require_staged_assets(config)
    assert str(tmp_path) in str(raised.value)
    assert "Mount the cache directory writable" in str(raised.value)


def test_an_unwritable_cache_is_fine_when_the_weights_are_staged(tmp_path, monkeypatch):
    monkeypatch.setattr(service_mod, "assets_present", lambda cache, *args: True)
    monkeypatch.setattr(service_mod, "cache_is_writable", lambda cache: False)
    service_mod.require_staged_assets(
        service_mod.ServiceConfig(BOLTZ2_DOWNLOAD_ASSETS="true", AIM_CACHE_PATH=str(tmp_path))
    )


def test_cache_is_writable_tells_a_writable_directory_from_a_blocked_one(tmp_path):
    from runner import cache_is_writable

    assert cache_is_writable(tmp_path / "new" / "cache")
    blocker = tmp_path / "file"
    blocker.write_text("x")
    assert not cache_is_writable(blocker / "cache")


def test_the_check_runs_when_the_module_is_imported_not_in_the_worker():
    """A failure in ``Boltz2Prediction.__init__`` runs in a BentoML worker, which BentoML starts again and again."""
    import ast
    import inspect

    tree = ast.parse(inspect.getsource(service_mod))
    top_level_calls = [
        n.value.func.id
        for n in tree.body
        if isinstance(n, ast.Expr) and isinstance(n.value, ast.Call) and isinstance(n.value.func, ast.Name)
    ]
    assert "require_staged_assets" in top_level_calls


# --------------------------------------------------------------------------- #
# Ligand tokens: the heavy atoms of the SMILES, aromatic ones included
# --------------------------------------------------------------------------- #

HEAVY_ATOMS = [
    ("C", 1),
    ("CC(=O)Cl", 4),
    ("c1ccccc1", 6),  # benzene, aromatic atoms are lower case
    ("c1ccncc1", 6),  # pyridine
    ("Clc1ccccc1", 7),
    ("Brc1ccccc1", 7),
    ("[nH]1cccc1", 5),  # pyrrole, a bracket atom with a hydrogen
    ("[NH4+]", 1),  # a charged bracket atom
    ("C[N+](C)(C)C", 5),
    ("[Na+].[Cl-]", 2),
    ("[se]1cccc1", 5),  # a two-letter aromatic bracket atom
    ("[H][H]", 0),  # hydrogens are not tokens
    ("[2H]C", 1),
    ("OC(=O)c1ccccc1O", 10),  # salicylic acid
]


@pytest.mark.parametrize(("smiles", "expected"), HEAVY_ATOMS)
def test_the_fallback_counts_the_heavy_atoms_of_a_smiles(monkeypatch, smiles, expected):
    import sys

    monkeypatch.setitem(sys.modules, "rdkit", None)  # makes "from rdkit import Chem" fail
    assert service_mod.smiles_heavy_atoms(smiles) == expected


@pytest.mark.parametrize(("smiles", "expected"), HEAVY_ATOMS)
def test_rdkit_and_the_fallback_agree_when_rdkit_is_installed(smiles, expected):
    pytest.importorskip("rdkit")
    assert service_mod.smiles_heavy_atoms(smiles) == expected


def test_a_request_of_2365_tokens_with_an_aromatic_ligand_is_above_the_limit(monkeypatch):
    import sys

    monkeypatch.setitem(sys.modules, "rdkit", None)
    sequence = "A" * (service_mod.MAX_TOKENS - 6)
    at_limit = [{"protein": {"id": "A", "sequence": sequence}}, {"ligand": {"id": "L", "smiles": "c1ccccc1"}}]
    service_mod.check_limits(_request(sequences=at_limit))
    above = [{"protein": {"id": "A", "sequence": sequence + "A"}}, {"ligand": {"id": "L", "smiles": "c1ccccc1"}}]
    with pytest.raises(service_mod.UnprocessableRequest):
        service_mod.check_limits(_request(sequences=above))


# --------------------------------------------------------------------------- #
# The example request bodies of API.md
# --------------------------------------------------------------------------- #


def _code_blocks(language):
    import pathlib
    import re

    text = (pathlib.Path(__file__).resolve().parents[3] / "templates" / "API.md").read_text(encoding="utf-8")
    return re.findall(rf"```{language}\n(.*?)```", text, re.S)


def test_every_json_example_of_the_api_document_is_a_valid_request():
    import json

    bodies = [json.loads(b) for b in _code_blocks("json") if '"data"' in b and '"sequences"' in b]
    used = set()
    for body in bodies:
        for field in ("constraints", "properties"):
            for entry in body["data"].get(field, []):
                used.update(entry)
        request = _request(**body["data"])
        service_mod.check_limits(request)
        if service_mod.has_affinity_property(request.properties):
            service_mod.check_affinity_request(request)
    assert {"contact", "affinity"} <= used


def test_the_template_example_of_the_api_document_is_valid_python_and_a_valid_request():
    import ast

    blocks = [b for b in _code_blocks("python") if "templates" in b]
    assert len(blocks) == 1
    ast.parse(blocks[0])
    pdb = "ATOM      1  N   ASN A   1       0.000   0.000   0.000  1.00  0.00           N\nEND\n"
    request = _request(
        sequences=[{"protein": {"id": "A", "sequence": "NLYIQW"}}], templates=[{"pdb": pdb, "chain_id": "A"}]
    )
    assert request.templates[0]["chain_id"] == "A"


# --------------------------------------------------------------------------- #
# Start-up without a GPU
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("accelerator", ["gpu", "cuda", "rocm"])
def test_start_up_stops_when_a_gpu_is_requested_and_none_is_found(accelerator, monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    with pytest.raises(RuntimeError, match="BOLTZ2_ACCELERATOR=" + accelerator) as raised:
        service_mod.require_accelerator(service_mod.ServiceConfig(BOLTZ2_ACCELERATOR=accelerator))
    assert "/dev/kfd" in str(raised.value)


def test_start_up_goes_on_without_a_gpu_for_auto_and_cpu_and_with_a_gpu_for_gpu(monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    service_mod.require_accelerator(service_mod.ServiceConfig(BOLTZ2_ACCELERATOR="auto"))
    service_mod.require_accelerator(service_mod.ServiceConfig(BOLTZ2_ACCELERATOR="cpu"))
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    service_mod.require_accelerator(service_mod.ServiceConfig(BOLTZ2_ACCELERATOR="gpu"))


# --------------------------------------------------------------------------- #
# Response models — the JSON of every answer
# --------------------------------------------------------------------------- #

_CONFIDENCE = {
    "confidence_score": 0.87,
    "ptm": 0.9,
    "iptm": 0.0,
    "ligand_iptm": 0,
    "protein_iptm": 0.0,
    "complex_plddt": 0.91,
    "complex_iplddt": 0.88,
    "complex_pde": 0.4,
    "complex_ipde": 0.5,
    "chains_ptm": {"0": 0.9, "1": 0.8},
    "pair_chains_iptm": {"0": {"0": 0.9, "1": 0.7}, "1": {"0": 0.7, "1": 0.8}},
}
_AFFINITY = {
    "affinity_pred_value": 0.42,
    "affinity_probability_binary": 0.87,
    "affinity_pred_value1": 0.4,
    "affinity_probability_binary1": 0.85,
    "affinity_pred_value2": 0.44,
    "affinity_probability_binary2": 0.89,
}


def _status_with_compiles():
    import compilation

    status = compilation.CompileStatus(mode="inductor", instances=2)
    status.record_invocation()
    status.record_compile_callback("0/0")
    status.record_compile_callback("1/0")
    status.record_compile_callback("odd")
    status.compiled_calls = 3
    return {"transition": status, "token_transformer": compilation.CompileStatus(mode="none")}


def _legacy_answer(svc, result, *, affinity_status=False):
    """The expected answer: the result and the runtime info as plain dicts, with no response model."""
    result = dict(result)
    result.pop("error", None)
    peak_gb = result.pop("cuda_peak_gb", None)
    seams = service_mod.seam_status()
    info = {
        "mode": service_mod.CONFIG.mode,
        "optimizations": service_mod.CONFIG.optimizations,
        "triton": {"attention": seams["attention"], "mul_impl": seams["mul_impl"]},
        "precision": "bf16-mixed",
        "compile_status": service_mod.compile_status(svc._model),
        "warmup": svc._warmup.to_dict(svc._all_compile_status()),
    }
    if affinity_status:
        info["affinity_compile_status"] = service_mod.compile_status(svc._affinity_model)
    result["runtime"] = {**info, "cuda_peak_gb": peak_gb}
    return result


def _warmup_report(state):
    import warmup

    report = warmup.WarmupReport(state=state)
    if state == warmup.STATE_DISABLED:
        report.reason = "BOLTZ2_OPTIMIZATIONS=off"
    if state in (warmup.STATE_DONE, warmup.STATE_FAILED):
        report.duration_s = 12.345
        report.steps = [
            {"name": "warmup_128", "seconds": 3.5, "compiles": {"transition": 2}},
            {"name": "affinity", "seconds": 1.25},
        ]
        report.compiles_during_warmup = {"transition": {"successful_compiles": 2, "failed_compiles": 0}}
    if state == warmup.STATE_FAILED:
        report.error = "RuntimeError: no memory"
    return report


_STRUCTURE_ONLY = {
    "error": False,
    "structures": [{"record_id": "A_model_0", "format": "pdb", "content": "ATOM  1\n"}],
    "confidence": {"A_model_0": _CONFIDENCE},
    "timing": {"runtime_s": 3.456},
    "cuda_peak_gb": 12.34,
}
_WITH_AFFINITY = {
    "error": False,
    "structures": [
        {"record_id": "A_model_0", "format": "mmcif", "content": "data_A\n"},
        {"record_id": "A_model_1", "format": "mmcif", "content": "data_B\n"},
    ],
    "confidence": {"A_model_0": _CONFIDENCE, "A_model_1": {**_CONFIDENCE, "ptm": 1}},
    "timing": {"runtime_s": 9.0, "affinity_s": 4.5},
    "affinity": _AFFINITY,
    "cuda_peak_gb": 20.5,
}
_NO_CONFIDENCE_OR_PEAK = {
    "structures": [{"record_id": "A", "format": "pdb", "content": "x"}],
    "confidence": {},
    "timing": {"runtime_s": 0.0},
    "cuda_peak_gb": None,
}


@pytest.mark.parametrize("state", ["pending", "disabled", "done", "failed"])
@pytest.mark.parametrize(
    ("result", "seed", "mode", "affinity_compile"),
    [
        (_STRUCTURE_ONLY, None, "structure", "off"),
        (_WITH_AFFINITY, 7, "affinity", "off"),
        (_WITH_AFFINITY, 0, "affinity", "on"),
        (_NO_CONFIDENCE_OR_PEAK, None, "structure", "off"),
    ],
)
def test_the_response_models_leave_the_inference_json_unchanged(
    tmp_path, monkeypatch, state, result, seed, mode, affinity_compile
):
    import contextlib
    import copy
    import json

    service, _ = _install_fake_model(monkeypatch, tmp_path, status_map=_status_with_compiles())
    monkeypatch.setattr(service.CONFIG, "mode", mode)
    monkeypatch.setattr(service.CONFIG, "affinity_compile", affinity_compile)
    monkeypatch.setattr(service, "load_affinity_model", lambda cache, **kwargs: type("FakeAffinity", (), {})())
    monkeypatch.setattr(service, "seeded_prediction", lambda seed: contextlib.nullcontext())
    monkeypatch.setattr(service, "run_boltz2_prediction", lambda **kwargs: copy.deepcopy(result))
    svc = service.Boltz2Prediction.inner()
    svc._warmup = _warmup_report(state)
    if service.CONFIG.affinity_compiled:
        import compilation

        setattr(svc._affinity_model, compilation._STATUS_ATTR, _status_with_compiles())  # noqa: SLF001

    answer = svc.inference(_request(seed=seed), _FakeContext())

    expected = _legacy_answer(
        svc, {**copy.deepcopy(result), "seed": seed}, affinity_status=service.CONFIG.affinity_compiled
    )
    assert answer == expected
    assert json.dumps(answer) == json.dumps(expected)
    assert ("affinity" in answer) == ("affinity" in result)
    assert ("affinity_s" in answer["timing"]) == ("affinity_s" in result["timing"])
    assert ("affinity_compile_status" in answer["runtime"]) == service.CONFIG.affinity_compiled
    assert answer["seed"] == seed


def test_a_result_that_breaks_the_response_contract_is_not_returned(tmp_path, monkeypatch):
    service, _ = _install_fake_model(monkeypatch, tmp_path)
    bad = {"structures": [{"record_id": "A", "format": "pdf", "content": "x"}], "confidence": {}, "timing": {}}
    monkeypatch.setattr(service, "run_boltz2_prediction", lambda **kwargs: bad)
    with pytest.raises(ValidationError):
        service.Boltz2Prediction.inner().inference(_request(), _FakeContext())


def test_the_runtime_info_keeps_its_keys_and_their_order(tmp_path, monkeypatch):
    service, _ = _install_fake_model(monkeypatch, tmp_path, status_map=_status_with_compiles())
    svc = service.Boltz2Prediction.inner()
    svc._warmup = _warmup_report("done")
    info = svc._runtime_info()
    assert list(info) == ["mode", "optimizations", "triton", "precision", "compile_status", "warmup"]
    assert list(info["warmup"]) == [
        "state",
        "reason",
        "duration_s",
        "error",
        "steps",
        "compiles_during_warmup",
        "since_warmup",
    ]
    assert info["warmup"]["reason"] is None


def test_the_503_answer_is_the_error_object(tmp_path, monkeypatch):
    service, _ = _install_fake_model(monkeypatch, tmp_path)
    ctx = _FakeContext()
    answer = service.Boltz2Prediction.inner()._unavailable(ctx, "The MSA server did not answer in time. Retry later.")
    assert answer == {"error": "The MSA server did not answer in time. Retry later."}
    assert ctx.response.status_code == 503


def test_the_models_answer_is_the_openai_list():
    import asyncio

    response = asyncio.run(service_mod.list_models(None))
    assert response.body == b'{"object":"list","data":[{"id":"mit/boltz2","object":"model"}]}'


@pytest.mark.parametrize(("upstream_status", "status"), [(200, "ok"), (503, "unavailable")])
def test_the_health_answer_keeps_the_upstream_status(monkeypatch, upstream_status, status):
    import asyncio

    import httpx

    class _Client:
        def __init__(self, **kwargs):
            pass

        async def __aenter__(self):
            return self

        async def __aexit__(self, *exc):
            return False

        async def get(self, url):
            return httpx.Response(upstream_status, request=httpx.Request("GET", url))

    monkeypatch.setattr(service_mod.httpx, "AsyncClient", _Client)
    response = asyncio.run(service_mod.health(None))
    assert response.status_code == upstream_status
    assert response.body == f'{{"status":"{status}"}}'.encode()


def test_the_health_answer_is_503_when_the_upstream_is_unreachable(monkeypatch):
    import asyncio

    import httpx

    class _Client:
        def __init__(self, **kwargs):
            pass

        async def __aenter__(self):
            raise httpx.ConnectError("down")

        async def __aexit__(self, *exc):
            return False

    monkeypatch.setattr(service_mod.httpx, "AsyncClient", _Client)
    response = asyncio.run(service_mod.health(None))
    assert (response.status_code, response.body) == (503, b'{"status":"unavailable"}')
