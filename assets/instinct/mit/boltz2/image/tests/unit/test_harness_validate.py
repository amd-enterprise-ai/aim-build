# Copyright © Advanced Micro Devices, Inc., or its affiliates.
#
# SPDX-License-Identifier: MIT

"""A failed prediction must never read as a pass.

The service answers a failed prediction with a non-2xx status, and a 2xx answer
with no atom records is still not a structure.
"""

from __future__ import annotations

from harness import Boltz2Harness

from aim_runtime.harness import HarnessConfig

FAILED_STATUS = 500


def _config(**extra) -> HarnessConfig:
    return HarnessConfig(profile={}, service_url="http://svc", extra=extra)


def _check(result, name):
    return next(c for c in result.checks if c.name == name)


def test_validate_passes_on_a_structure(fake_http):
    result = Boltz2Harness().validate(_config())

    smoke = _check(result, "predict_smoke")
    assert result.success and smoke.success and "1 atom records" in smoke.detail


def test_validate_fails_on_a_failed_prediction_status(fake_http):
    fake_http.predict_status = FAILED_STATUS

    result = Boltz2Harness().validate(_config())

    smoke = _check(result, "predict_smoke")
    assert not result.success and not smoke.success
    assert "HTTP Error 500" in smoke.detail


def test_validate_fails_on_a_body_with_no_atom_records(fake_http):
    fake_http.predict_body = "data_only_a_header\n"

    smoke = _check(Boltz2Harness().validate(_config()), "predict_smoke")

    assert not smoke.success and "no ATOM/HETATM record" in smoke.detail


def test_validate_fails_on_an_empty_body(fake_http):
    fake_http.predict_body = ""

    smoke = _check(Boltz2Harness().validate(_config()), "predict_smoke")

    assert not smoke.success and "empty response" in smoke.detail


def test_validate_accepts_hetatm_only_structures(fake_http):
    fake_http.predict_body = "HETATM    1  C1  LIG A   1       0.0   0.0   0.0  1.00  0.00           C\n"

    assert _check(Boltz2Harness().validate(_config()), "predict_smoke").success


def test_the_smoke_benchmark_does_not_count_failed_predictions_as_structures(fake_http):
    fake_http.predict_status = FAILED_STATUS

    result = Boltz2Harness().benchmark(_config(benchmark_set="smoke", benchmark_n_requests=3))

    assert not result.success and "All benchmark requests failed" in result.summary


def test_evaluate_structure_sanity_fails_on_a_failed_prediction_status(fake_http):
    fake_http.predict_status = FAILED_STATUS

    result = Boltz2Harness().evaluate(_config())

    assert not _check(result, "structure_sanity").success
    assert "HTTP Error 500" in _check(result, "structure_sanity").detail


def test_validate_in_structure_mode_sends_no_affinity_request(fake_http):
    result = Boltz2Harness().validate(_config())

    assert result.success
    assert all("properties" not in data for data in fake_http.sent_data())
    assert not [c for c in result.checks if c.name == "predict_affinity_smoke"]


def test_validate_in_affinity_mode_sends_one_affinity_request(fake_http):
    fake_http.mode = "affinity"

    result = Boltz2Harness().validate(_config())

    assert result.success and _check(result, "predict_affinity_smoke").success
    affinity_requests = [data for data in fake_http.sent_data() if data.get("properties")]
    assert len(affinity_requests) == 1
    assert affinity_requests[0]["properties"] == [{"affinity": {"binder": "B"}}]


def test_validate_in_affinity_mode_fails_on_an_answer_without_affinity_values(fake_http):
    fake_http.mode = "affinity"
    fake_http.affinity = {"affinity_pred_value": float("nan")}

    result = Boltz2Harness().validate(_config())

    smoke = _check(result, "predict_affinity_smoke")
    assert not result.success and not smoke.success
    assert "affinity_pred_value" in smoke.detail and "affinity_probability_binary" in smoke.detail


def test_validate_skips_the_affinity_request_when_the_structure_smoke_fails(fake_http):
    fake_http.mode = "affinity"
    fake_http.predict_body = ""

    result = Boltz2Harness().validate(_config())

    assert not result.success
    assert not [c for c in result.checks if c.name == "predict_affinity_smoke"]
