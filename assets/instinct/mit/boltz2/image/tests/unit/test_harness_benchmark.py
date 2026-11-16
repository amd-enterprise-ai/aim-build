# Copyright © Advanced Micro Devices, Inc., or its affiliates.
#
# SPDX-License-Identifier: MIT

"""Fixture-set benchmark and case-set evaluation in the Boltz-2 harness.

The service is faked: ``_post_json`` returns scripted ``/v1/inference`` bodies
and a fake clock gives each request an exact client time, so every ratio below
is computed by hand.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import harness as harness_mod
import pytest
from harness import Boltz2Harness, reference_ratios

from aim_runtime.harness import CheckResult, CheckResultType, HarnessConfig

SET = "timing-18"
FIXTURE_RECIPE = {"sampling_steps": 50, "step_scale": 1.638, "use_msa_server": False}


def _write_case(root: Path, label: str, tokens: int, ref: float, alt: float, sets: tuple[str, ...] = (SET,)) -> None:
    case_dir = root / label
    case_dir.mkdir(parents=True)
    case = {
        "metadata": {
            "label": label,
            "sets": list(sets),
            "residues": tokens,
        },
        "data": {"sequences": [{"protein": {"id": "A", "sequence": "A" * tokens}}], "name": label, **FIXTURE_RECIPE},
    }
    (case_dir / f"{label}.json").write_text(json.dumps(case))
    reference_path = root.parent / "reference.json"
    document = json.loads(reference_path.read_text()) if reference_path.exists() else {"targets": {}}
    document["targets"][label] = {"first_s": ref, "warm_s": ref, "alt_warm_s": alt}
    reference_path.write_text(json.dumps(document))


@pytest.fixture
def cases_dir(tmp_path: Path) -> Path:
    root = tmp_path / "cases"
    # Written big-first so the test proves the sort by token count.
    _write_case(root, "big", 200, ref=4.0, alt=20.0)
    _write_case(root, "small", 100, ref=2.0, alt=10.0)
    return root


class FakeService:
    """Scripted /v1/inference: one duration and one cumulative compile count per request."""

    def __init__(self, monkeypatch: pytest.MonkeyPatch, durations: dict[tuple[str, int], float]):
        self.clock = 0.0
        self.durations = durations
        self.seen: dict[str, int] = {}
        self.payloads: list[dict[str, Any]] = []
        self.served: list[tuple[str, int]] = []
        self.compiles: dict[tuple[str, int], int] = {}
        # Cumulative (successful, failed) compiles of the "transition" selector, per request.
        self.failed_compiles: dict[tuple[str, int], int] = {}
        self.confidence = {"complex_plddt": 0.91, "ptm": 0.8}
        self.cuda_peak_gb: float | None = 12.5
        self.peaks: dict[tuple[str, int], float] = {}
        self.optimizations: str | None = "on"
        self.warmup_state: str | None = "done"
        self.report_since_warmup = True
        self.malformed_since_warmup = False
        self.optimizations_by: dict[tuple[str, int], str] = {}
        self.fail_all = False
        monkeypatch.setattr(harness_mod.time, "perf_counter", lambda: self.clock)
        monkeypatch.setattr(Boltz2Harness, "_post_json", staticmethod(self._post))
        monkeypatch.setattr(Boltz2Harness, "health_check", lambda self, url, timeout_seconds=60: True)

    def _post(self, url: str, payload: dict[str, Any], timeout: int = 3600) -> str:
        assert url.endswith("/v1/inference")
        self.payloads.append(payload)
        label = payload["data"]["name"]
        n = self.seen[label] = self.seen.get(label, 0) + 1
        self.clock += self.durations[(label, n)]
        self.served.append((label, n))
        cumulative = max([self.compiles.get(key, 0) for key in self.served])
        failed = max([self.failed_compiles.get(key, 0) for key in self.served])
        if self.fail_all:
            raise OSError("connection refused")
        warmup: dict[str, Any] = {"state": self.warmup_state}
        if self.malformed_since_warmup:
            warmup["since_warmup"] = "n/a"
        elif self.report_since_warmup:
            warmup["since_warmup"] = {
                "compiles": {"transition": {"successful_compiles": cumulative, "failed_compiles": failed}}
            }
        body = {
            "error": False,
            "structures": [{"record_id": label, "format": "mmcif", "content": "data_x"}],
            "confidence": {label: self.confidence},
            "timing": {"runtime_s": 1.0},
            "runtime": {
                "cuda_peak_gb": self.peaks.get((label, n), self.cuda_peak_gb),
                "optimizations": self.optimizations_by.get((label, n), self.optimizations),
                "warmup": warmup,
            },
        }
        return json.dumps(body)


def _config(tmp_path: Path, cases_dir: Path, **extra: Any) -> HarnessConfig:
    return HarnessConfig(
        profile={},
        service_url="http://svc",
        extra={
            "benchmark_set": SET,
            "benchmark_dir": str(cases_dir),
            "benchmark_recipe": "fixture",
            "output_dir": str(tmp_path / "out"),
            **(
                {"benchmark_reference_file": str(cases_dir.parent / "reference.json")}
                if (cases_dir.parent / "reference.json").exists()
                else {}
            ),
            **extra,
        },
    )


def _check(result, name: str) -> CheckResult:
    return next(c for c in result.checks if c.name == name)


# --------------------------------------------------------------------------- #
# Scenario 1: no options times the default set; the smoke timing is explicit
# --------------------------------------------------------------------------- #


def test_default_benchmark_times_the_default_set_with_the_fixture_recipe(monkeypatch, tmp_path, cases_dir):
    service = FakeService(monkeypatch, _durations())
    monkeypatch.setattr(harness_mod, "BENCHMARK_DIR", cases_dir)

    result = Boltz2Harness().benchmark(HarnessConfig(profile={}, service_url="http://svc"))

    assert harness_mod.DEFAULT_TIMING_SET == "timing-18"
    assert result.metrics["benchmark_set"] == "timing-18"
    assert harness_mod.DEFAULT_BENCHMARK_RECIPE == "fixture"
    assert result.metrics["recipe_name"] == "fixture" and result.metrics["blocks"] == 3
    assert result.metrics["requests_total"] == 6
    assert service.payloads[0]["data"]["sampling_steps"] == 50
    assert _check(result, "reference_ratio_warm").skipped
    # 6 requests in 24 s of client time; warm requests are 1, 3, 4 and 4 s.
    assert _check(result, "throughput").value == 15.0
    assert _check(result, "p50_ms").value == 3500.0


def test_elapsed_seconds_are_rounded_to_the_millisecond(monkeypatch):
    monkeypatch.setattr(harness_mod.time, "perf_counter", lambda: 10.123456789)

    assert harness_mod._elapsed_s(5.0) == 5.123


def _smoke_config(**extra):
    return HarnessConfig(profile={}, service_url="http://svc", extra={"benchmark_set": "smoke", **extra})


def test_smoke_benchmark_is_an_explicit_choice(monkeypatch):
    monkeypatch.setattr(Boltz2Harness, "health_check", lambda self, url, timeout_seconds=60: True)
    sent = []
    monkeypatch.setattr(Boltz2Harness, "_send_predict", staticmethod(lambda url, seq, **kw: sent.append(seq) or "PDB"))
    monkeypatch.setattr(Boltz2Harness, "_post_json", staticmethod(lambda *a, **k: pytest.fail("fixture path used")))

    result = Boltz2Harness().benchmark(_smoke_config(benchmark_n_requests=3))

    assert sent == [harness_mod.SMOKE_SEQUENCE] * 3
    assert result.success
    assert result.metrics["requests_total"] == 3
    assert _check(result, "throughput").value is not None and _check(result, "p50_ms").value is not None
    for name in (
        "reference_ratio_warm",
        "reference_ratio_first",
        "reference_alt_ratio_warm",
        "requests_ok",
        "compiles_after_ready",
    ):
        assert _check(result, name).skipped


def test_smoke_benchmark_latencies_are_rounded_to_the_millisecond(monkeypatch):
    monkeypatch.setattr(Boltz2Harness, "health_check", lambda self, url, timeout_seconds=60: True)
    clock = iter([0.0, 1.23456, 10.0, 11.23456])
    monkeypatch.setattr(harness_mod.time, "perf_counter", lambda: next(clock))
    monkeypatch.setattr(Boltz2Harness, "_send_predict", staticmethod(lambda url, seq, **kw: "PDB"))

    result = Boltz2Harness().benchmark(_smoke_config(benchmark_n_requests=2))

    assert result.metrics["avg_latency_s"] == 1.235
    assert _check(result, "p50_ms").value == 1235.0


def test_smoke_benchmark_n_requests_falls_back_to_the_profile(monkeypatch):
    monkeypatch.setattr(Boltz2Harness, "health_check", lambda self, url, timeout_seconds=60: True)
    monkeypatch.setattr(Boltz2Harness, "_send_predict", staticmethod(lambda url, seq, **kw: "PDB"))

    result = Boltz2Harness().benchmark(
        HarnessConfig(profile={"benchmark_n_requests": 2}, service_url="http://svc", extra={"benchmark_set": "smoke"})
    )

    assert result.metrics["requests_total"] == 2


# --------------------------------------------------------------------------- #
# Scenario 2: first and warm timing against both reference times
# --------------------------------------------------------------------------- #


def _durations() -> dict[tuple[str, int], float]:
    # small: first 4 s, warm 1 s and 3 s (median 2 s); big: first 8 s, warm 4 s and 4 s.
    return {("small", 1): 4.0, ("small", 2): 1.0, ("small", 3): 3.0, ("big", 1): 8.0, ("big", 2): 4.0, ("big", 3): 4.0}


def test_reference_first_and_warm_ratios(monkeypatch, tmp_path, cases_dir):
    service = FakeService(monkeypatch, _durations())

    result = Boltz2Harness().benchmark(_config(tmp_path, cases_dir, benchmark_blocks=3))

    assert result.success
    # first: 2/4 and 4/8 -> 0.5; warm: 2/2 and 4/4 -> 1.0; warm alt: 10/2 and 20/4 -> 5.0
    assert _check(result, "reference_ratio_first").value == 0.5
    assert _check(result, "reference_ratio_warm").value == 1.0
    assert _check(result, "reference_alt_ratio_warm").value == 5.0
    assert "median of blocks 2..3 (n=2 per target)" in _check(result, "reference_ratio_warm").detail
    assert "2/2 targets" in _check(result, "reference_ratio_warm").detail
    assert [p["data"]["name"] for p in service.payloads[:2]] == ["small", "big"]
    assert result.metrics["requests_total"] == 6 and result.metrics["requests_succeeded"] == 6
    assert result.metrics["cuda_peak_gb_max"] == 12.5
    assert "clean" in result.summary


def test_reference_records_are_saved(monkeypatch, tmp_path, cases_dir):
    FakeService(monkeypatch, _durations())

    result = Boltz2Harness().benchmark(_config(tmp_path, cases_dir, benchmark_blocks=3))

    saved = json.loads((tmp_path / "out" / "benchmark_records.json").read_text())
    assert result.artifacts == [str(tmp_path / "out" / "benchmark_records.json")]
    assert len(saved["records"]) == 6
    first = saved["records"][0]
    assert (first["block"], first["label"], first["wall_s"], first["tokens"]) == (1, "small", 4.0, 100)
    assert first["reference"] == {"first_s": 2.0, "warm_s": 2.0, "alt_warm_s": 10.0}


def test_single_block_skips_the_warm_checks(monkeypatch, tmp_path, cases_dir):
    FakeService(monkeypatch, _durations())

    result = Boltz2Harness().benchmark(_config(tmp_path, cases_dir, benchmark_blocks=1))

    assert result.success
    assert _check(result, "reference_ratio_warm").skipped
    assert _check(result, "reference_alt_ratio_warm").skipped
    assert _check(result, "reference_ratio_first").value == 0.5


def test_labels_filter_and_unknown_label(monkeypatch, tmp_path, cases_dir):
    FakeService(monkeypatch, _durations())

    only = Boltz2Harness().benchmark(_config(tmp_path, cases_dir, benchmark_blocks=2, benchmark_labels="small"))
    unknown = Boltz2Harness().benchmark(_config(tmp_path, cases_dir, benchmark_labels=["small", "nope"]))

    assert only.metrics["targets_total"] == 1
    assert not unknown.success and "nope" in unknown.summary


def test_a_failed_request_fails_the_ratio_and_names_the_target(monkeypatch, tmp_path, cases_dir):
    service = FakeService(monkeypatch, _durations())
    original = service._post

    def flaky(url, payload, timeout=3600):
        if payload["data"]["name"] == "big" and service.seen.get("big", 0) == 0:
            service.seen["big"] = 1
            raise OSError("connection reset")
        return original(url, payload, timeout)

    monkeypatch.setattr(Boltz2Harness, "_post_json", staticmethod(flaky))

    result = Boltz2Harness().benchmark(_config(tmp_path, cases_dir, benchmark_blocks=1))

    first = _check(result, "reference_ratio_first")
    assert not first.success and "no ratio for big" in first.detail
    assert result.metrics["requests_failed"] == 1
    assert _check(result, "compiles_after_ready").warnings


def test_a_failed_warm_request_fails_the_run_and_reports_the_real_n(monkeypatch, tmp_path, cases_dir):
    service = FakeService(monkeypatch, _durations())
    original = service._post

    def flaky(url, payload, timeout=3600):
        if payload["data"]["name"] == "big" and service.seen.get("big", 0) == 1:
            service.seen["big"] = 2
            raise OSError("connection reset")
        return original(url, payload, timeout)

    monkeypatch.setattr(Boltz2Harness, "_post_json", staticmethod(flaky))

    result = Boltz2Harness().benchmark(_config(tmp_path, cases_dir, benchmark_blocks=3))

    warm = _check(result, "reference_ratio_warm")
    assert not warm.success and not result.success
    assert "n=1-2 per target" in warm.detail and "big block 2" in warm.detail


# --------------------------------------------------------------------------- #
# Scenario 3: the served recipe
# --------------------------------------------------------------------------- #


def test_served_recipe_uses_the_service_defaults(monkeypatch, tmp_path, cases_dir):
    service = FakeService(monkeypatch, _durations())

    result = Boltz2Harness().benchmark(_config(tmp_path, cases_dir, benchmark_blocks=3, benchmark_recipe="served"))

    for payload in service.payloads:
        assert not set(payload["data"]) & set(harness_mod.SERVED_RECIPE_FIELDS)
    assert result.metrics["recipe_name"] == "served"
    assert result.metrics["recipe"]["sampling_steps"] == 200
    assert result.metrics["recipe"]["use_msa_server"] is False
    assert "not a speed claim" in result.metrics["recipe_note"]
    assert "not a speed claim" in result.summary


def test_the_reference_ratios_are_skipped_for_a_recipe_that_is_not_the_fixture_recipe(monkeypatch, tmp_path, cases_dir):
    FakeService(monkeypatch, _durations())

    result = Boltz2Harness().benchmark(_config(tmp_path, cases_dir, benchmark_blocks=3, benchmark_recipe="served"))

    assert result.success
    for name in ("reference_ratio_warm", "reference_ratio_first", "reference_alt_ratio_warm"):
        check = _check(result, name)
        assert check.skipped and "reference times are for the fixture recipe" in check.detail
        assert check.value is None


def test_fixture_recipe_sends_the_fixture_request(monkeypatch, tmp_path, cases_dir):
    service = FakeService(monkeypatch, _durations())

    result = Boltz2Harness().benchmark(_config(tmp_path, cases_dir, benchmark_blocks=1))

    assert service.payloads[0]["data"]["sampling_steps"] == 50
    assert result.metrics["recipe"]["step_scale"] == 1.638
    assert "SERVED RECIPE" not in _check(result, "reference_ratio_first").detail


def test_unknown_recipe_is_rejected(monkeypatch, tmp_path, cases_dir):
    FakeService(monkeypatch, _durations())

    result = Boltz2Harness().benchmark(_config(tmp_path, cases_dir, benchmark_recipe="fast"))

    assert not result.success and "benchmark_recipe" in result.summary


# --------------------------------------------------------------------------- #
# Scenario 4: a compile after ready
# --------------------------------------------------------------------------- #


def test_compile_after_ready_marks_the_run_not_clean(monkeypatch, tmp_path, cases_dir):
    service = FakeService(monkeypatch, _durations())
    service.compiles[("big", 2)] = 1  # cumulative count rises on big's second request

    result = Boltz2Harness().benchmark(_config(tmp_path, cases_dir, benchmark_blocks=3))

    check = _check(result, "compiles_after_ready")
    assert not check.success and not result.success
    assert "1/6 timed requests" in check.detail and "block 2 big" in check.detail
    assert result.metrics["compiles_after_ready_requests"] == 1
    assert "NOT CLEAN" in result.summary


def test_compile_baseline_excludes_earlier_traffic(monkeypatch, tmp_path, cases_dir):
    service = FakeService(monkeypatch, _durations())
    service.compiles[("small", 1)] = 3  # compiled by an earlier run on the same service

    dirty = Boltz2Harness().benchmark(_config(tmp_path, cases_dir, benchmark_blocks=1))
    clean = Boltz2Harness().benchmark(_config(tmp_path, cases_dir, benchmark_blocks=1, benchmark_compiles_baseline=3))

    assert not _check(dirty, "compiles_after_ready").success
    assert _check(clean, "compiles_after_ready").success


# --------------------------------------------------------------------------- #
# Pure helpers
# --------------------------------------------------------------------------- #


def test_reference_ratios_skip_errors_and_use_the_warm_median():
    nv = {"first_s": 10.0, "warm_s": 10.0, "alt_warm_s": 40.0}
    records = [
        {"block": 1, "label": "a", "wall_s": 20.0, "reference": nv},
        {"block": 2, "label": "a", "wall_s": 5.0, "reference": nv},
        {"block": 3, "label": "a", "wall_s": 100.0, "reference": nv, "error": "boom"},
        {"block": 4, "label": "a", "wall_s": 15.0, "reference": nv},
        {"block": 1, "label": "b", "wall_s": 9.0, "reference": None},
    ]

    ratios = reference_ratios(records)

    assert ratios["first"] == {"a": 0.5}
    assert ratios["warm_s"] == {"a": 10.0}
    assert ratios["warm"] == {"a": 1.0} and ratios["alt_warm"] == {"a": 4.0}


def test_eval_cases_set_and_directory_override_the_environment_defaults(tmp_path, cases_dir):
    _write_case(cases_dir, "other", 50, ref=1.0, alt=1.0, sets=("elsewhere",))

    in_set = Boltz2Harness._load_eval_cases(SET, cases_dir)
    elsewhere = Boltz2Harness._load_eval_cases("elsewhere", str(cases_dir))

    assert sorted(c["metadata"]["label"] for c in in_set) == ["big", "small"]
    assert [c["metadata"]["label"] for c in elsewhere] == ["other"]
    assert Boltz2Harness._load_eval_cases(SET, tmp_path / "missing") == []


# --------------------------------------------------------------------------- #
# Scenario 5: evaluate any case set
# --------------------------------------------------------------------------- #


@pytest.fixture
def evaluation(monkeypatch, tmp_path, cases_dir):
    """Two cases at the timing recipe; only ``small`` has a staged reference."""
    refs = tmp_path / "data" / "references"
    refs.mkdir(parents=True)
    (refs / "small.cif").write_text("data_ref")
    monkeypatch.setattr(harness_mod, "DATA_DIR", tmp_path / "data")
    monkeypatch.setattr(Boltz2Harness, "_ca_rmsd", staticmethod(lambda mmcif, ref, notes=None: (1.5, 90)))
    monkeypatch.setattr(
        Boltz2Harness,
        "_check_structure_sanity",
        classmethod(lambda cls, url: CheckResult("structure_sanity", CheckResultType.PASS_FAIL, True, True, "")),
    )
    service = FakeService(monkeypatch, {("small", 1): 2.0, ("big", 1): 3.0})
    config = HarnessConfig(
        profile={},
        service_url="http://svc",
        extra={"eval_set": SET, "eval_dir": str(cases_dir), "output_dir": str(tmp_path / "out")},
    )
    return service, config


def test_evaluate_predicts_a_case_without_a_reference(evaluation, tmp_path):
    service, config = evaluation

    result = Boltz2Harness().evaluate(config)

    assert sorted(p["data"]["name"] for p in service.payloads) == ["big", "small"]
    by_label = {c["label"]: c for c in result.metrics["cases"]}
    assert by_label["small"]["rmsd_a"] == 1.5
    assert by_label["big"]["rmsd_a"] is None
    assert by_label["big"]["complex_plddt"] == 0.91 and by_label["big"]["runtime_s"] == 1.0
    assert "RMSD not available" in by_label["big"]["reference"]
    assert "error" not in by_label["big"]
    assert result.metrics["cases_predicted"] == 2 and result.metrics["cases_scored"] == 1


def test_evaluate_says_the_gate_did_not_grade_this_recipe(evaluation):
    _, config = evaluation

    result = Boltz2Harness().evaluate(config)

    assert result.success
    assert result.metrics["accuracy_gate"]["applied"] is False
    assert "GATE NOT APPLIED" in result.metrics["accuracy_gate"]["detail"]
    assert result.metrics["ungraded_mean_ca_rmsd_a"] == 1.5
    assert result.metrics["ungraded_mean_complex_plddt"] == 0.91
    assert "mean_complex_plddt" not in result.metrics
    assert _check(result, "structure_accuracy").skipped and _check(result, "confidence_regression").skipped


def test_evaluate_saves_the_per_case_records(evaluation, tmp_path):
    _, config = evaluation

    result = Boltz2Harness().evaluate(config)

    saved = json.loads((tmp_path / "out" / "evaluate_records.json").read_text())
    assert result.artifacts == [str(tmp_path / "out" / "evaluate_records.json")]
    assert {c["label"] for c in saved["cases"]} == {"small", "big"}


def test_evaluate_a_set_with_no_reference_at_all_is_skipped_not_failed(evaluation, tmp_path):
    _, config = evaluation
    (tmp_path / "data" / "references" / "small.cif").unlink()

    result = Boltz2Harness().evaluate(config)

    accuracy = _check(result, "structure_accuracy")
    assert accuracy.skipped and "0/2 cases have a staged reference" in accuracy.detail
    assert result.success
    assert result.metrics["ungraded_mean_complex_plddt"] == 0.91


def test_evaluate_without_options_keeps_the_environment_set(monkeypatch, tmp_path, cases_dir):
    monkeypatch.setattr(harness_mod, "BENCHMARK_DIR", cases_dir)
    monkeypatch.setattr(harness_mod, "BENCHMARK_SET", "elsewhere")
    monkeypatch.setattr(Boltz2Harness, "health_check", lambda self, url, timeout_seconds=60: True)
    monkeypatch.setattr(
        Boltz2Harness,
        "_check_structure_sanity",
        classmethod(lambda cls, url: CheckResult("structure_sanity", CheckResultType.PASS_FAIL, True, True, "")),
    )

    result = Boltz2Harness().evaluate(HarnessConfig(profile={}, service_url="http://svc"))

    accuracy = _check(result, "structure_accuracy")
    assert accuracy.skipped and "set 'elsewhere'" in accuracy.detail


def test_results_round_trip_through_json(monkeypatch, tmp_path, cases_dir):
    FakeService(monkeypatch, _durations())

    result = Boltz2Harness().benchmark(_config(tmp_path, cases_dir, benchmark_blocks=3))

    assert json.loads(json.dumps(result.to_dict()))["success"] is True
    assert {c.name for c in result.checks} <= {c.name for c in Boltz2Harness().list_checks()}


def test_evaluate_fails_when_a_reference_that_should_exist_is_not_staged(evaluation, tmp_path, cases_dir):
    _, config = evaluation
    (tmp_path / "data" / "references" / "small.cif").unlink()
    case_file = cases_dir / "small" / "small.json"
    case = json.loads(case_file.read_text())
    case["metadata"]["reference_pdb_id"] = "1ABC"
    case_file.write_text(json.dumps(case))

    result = Boltz2Harness().evaluate(config)

    accuracy = _check(result, "structure_accuracy")
    assert not accuracy.skipped and not accuracy.success
    assert "not staged" in accuracy.detail
    assert any("small" in w for w in accuracy.warnings)


def test_evaluate_ignores_the_confidence_of_a_case_whose_rmsd_failed(evaluation, monkeypatch):
    _, config = evaluation

    def broken(mmcif, ref, notes=None):
        raise ValueError("no polymer chains to superpose")

    monkeypatch.setattr(Boltz2Harness, "_ca_rmsd", staticmethod(broken))

    result = Boltz2Harness().evaluate(config)

    small = next(c for c in result.metrics["cases"] if c["label"] == "small")
    assert "ValueError" in small["error"] and "complex_plddt" not in small
    assert result.metrics["cases_predicted"] == 1


def test_a_run_without_a_reference_file_is_timed_with_the_ratios_skipped(monkeypatch, tmp_path, cases_dir):
    (cases_dir.parent / "reference.json").unlink()
    FakeService(monkeypatch, _durations())

    result = Boltz2Harness().benchmark(_config(tmp_path, cases_dir, benchmark_blocks=2))

    assert result.success
    assert result.metrics["requests_succeeded"] == 4
    for name in ("reference_ratio_warm", "reference_ratio_first", "reference_alt_ratio_warm"):
        check = _check(result, name)
        assert check.skipped and "no benchmark_reference_file" in check.detail


def test_a_missing_reference_file_fails_the_run_before_any_request(monkeypatch, tmp_path, cases_dir):
    service = FakeService(monkeypatch, _durations())

    result = Boltz2Harness().benchmark(
        _config(tmp_path, cases_dir, benchmark_reference_file=str(tmp_path / "absent.json"))
    )

    assert not result.success and "benchmark_reference_file" in result.summary
    assert service.payloads == []


def test_a_reference_file_without_an_alt_time_skips_only_the_alt_ratio(monkeypatch, tmp_path, cases_dir):
    path = cases_dir.parent / "reference.json"
    document = json.loads(path.read_text())
    for entry in document["targets"].values():
        del entry["alt_warm_s"]
    path.write_text(json.dumps(document))
    FakeService(monkeypatch, _durations())

    result = Boltz2Harness().benchmark(_config(tmp_path, cases_dir, benchmark_blocks=3))

    assert _check(result, "reference_ratio_warm").value == 1.0
    assert _check(result, "reference_ratio_first").value == 0.5
    assert not _check(result, "reference_alt_ratio_warm").success


def test_an_empty_benchmark_set_selects_the_default(monkeypatch, cases_dir):
    FakeService(monkeypatch, _durations())
    monkeypatch.setattr(harness_mod, "BENCHMARK_DIR", cases_dir)

    result = Boltz2Harness().benchmark(
        HarnessConfig(profile={}, service_url="http://svc", extra={"benchmark_set": None})
    )

    assert result.metrics["benchmark_set"] == "timing-18"


def test_ratio_details_and_summary_name_the_reference(monkeypatch, tmp_path, cases_dir):
    FakeService(monkeypatch, _durations())

    result = Boltz2Harness().benchmark(_config(tmp_path, cases_dir, benchmark_blocks=3))

    assert "above 1 = faster than the reference" in _check(result, "reference_ratio_warm").detail
    assert "reference ratio warm" in result.summary


# --------------------------------------------------------------------------- #
# Statistics: different ratios per target, an outlier, mixed peaks
# --------------------------------------------------------------------------- #


@pytest.fixture
def stats_dir(tmp_path: Path) -> Path:
    root = tmp_path / "stats_cases"
    _write_case(root, "small", 100, ref=2.0, alt=10.0)
    _write_case(root, "mid", 150, ref=6.0, alt=12.0)
    _write_case(root, "big", 200, ref=5.0, alt=28.0)
    return root


def _stats_durations() -> dict[tuple[str, int], float]:
    # block 1 is first; blocks 2..4 are warm. small has an outlier (30 s), big has one (10 s).
    return {
        **{("small", n): d for n, d in zip((1, 2, 3, 4), (4.0, 1.0, 2.0, 30.0))},
        **{("mid", n): d for n, d in zip((1, 2, 3, 4), (3.0, 2.0, 3.0, 3.0))},
        **{("big", n): d for n, d in zip((1, 2, 3, 4), (8.0, 4.0, 4.0, 10.0))},
    }


def test_ratio_statistics_are_geometric_means_of_per_target_medians(monkeypatch, tmp_path, stats_dir):
    FakeService(monkeypatch, _stats_durations())

    result = Boltz2Harness().benchmark(_config(tmp_path, stats_dir, benchmark_blocks=4))

    # warm medians: small 2, mid 3, big 4 (the means would be 11, 2.67, 6).
    # warm ratios 1.0, 2.0, 1.25 -> geometric mean 2.5 ** (1/3) = 1.357 (arithmetic 1.417, min 1.0)
    warm_trt = _check(result, "reference_ratio_warm")
    assert warm_trt.value == 1.357
    assert "n=3 per target" in warm_trt.detail and "min 1.000 (small), max 2.000 (mid)" in warm_trt.detail
    # first ratios 0.5, 2.0, 0.625 -> 0.625 ** (1/3) = 0.855 (arithmetic 1.042, min 0.5)
    first_trt = _check(result, "reference_ratio_first")
    assert first_trt.value == 0.855
    assert "min 0.500 (small), max 2.000 (mid)" in first_trt.detail
    # OSS warm ratios 5, 4, 7 -> 140 ** (1/3) = 5.192 (arithmetic 5.333, min 4.0)
    warm_oss = _check(result, "reference_alt_ratio_warm")
    assert warm_oss.value == 5.192
    assert "min 4.000 (mid), max 7.000 (big)" in warm_oss.detail


def test_throughput_and_p50_cover_the_populations_their_details_state(monkeypatch, tmp_path, stats_dir):
    FakeService(monkeypatch, _stats_durations())

    result = Boltz2Harness().benchmark(_config(tmp_path, stats_dir, benchmark_blocks=4))

    # 12 requests in 15 s (first) + 59 s (warm) = 74 s -> 9.73 structures/min
    throughput = _check(result, "throughput")
    assert throughput.value == 9.73 and "12/12 successful timed requests of all blocks" in throughput.detail
    # the 9 warm requests sorted: 1 2 2 3 3 4 4 10 30 -> median 3 s, p95 30 s
    p50 = _check(result, "p50_ms")
    assert p50.value == 3000.0
    assert "9 successful warm requests (blocks 2..4" in p50.detail and "p95 = 30000.0 ms" in p50.detail


def test_p50_needs_a_warm_block(monkeypatch, tmp_path, stats_dir):
    FakeService(monkeypatch, _stats_durations())

    result = Boltz2Harness().benchmark(_config(tmp_path, stats_dir, benchmark_blocks=1))

    assert _check(result, "p50_ms").skipped and _check(result, "throughput").value is not None


def test_peak_memory_is_the_largest_of_the_requests(monkeypatch, tmp_path, stats_dir):
    service = FakeService(monkeypatch, _stats_durations())
    service.peaks = {("small", 1): 5.0, ("big", 2): 30.5, ("mid", 3): 18.0}

    result = Boltz2Harness().benchmark(_config(tmp_path, stats_dir, benchmark_blocks=4))

    assert result.metrics["cuda_peak_gb_max"] == 30.5


def test_one_warm_request_per_target_is_reported_as_n_1(monkeypatch, tmp_path, cases_dir):
    FakeService(monkeypatch, _durations())

    result = Boltz2Harness().benchmark(_config(tmp_path, cases_dir, benchmark_blocks=2))

    warm = _check(result, "reference_ratio_warm")
    # one warm request per target: small 2/1, big 4/4 -> geometric mean sqrt(2)
    assert warm.value == 1.414 and "n=1 per target" in warm.detail


def test_failed_compiles_count_as_compiles(monkeypatch, tmp_path, stats_dir):
    service = FakeService(monkeypatch, _stats_durations())
    service.failed_compiles[("mid", 2)] = 1

    result = Boltz2Harness().benchmark(_config(tmp_path, stats_dir, benchmark_blocks=4))

    check = _check(result, "compiles_after_ready")
    assert not check.success and "block 2 mid {'transition': 1}" in check.detail


def test_the_first_request_detail_does_not_blame_the_baseline(monkeypatch, tmp_path, stats_dir):
    service = FakeService(monkeypatch, _stats_durations())
    service.compiles[("small", 1)] = 5

    result = Boltz2Harness().benchmark(_config(tmp_path, stats_dir, benchmark_blocks=1, benchmark_compiles_baseline=3))

    check = _check(result, "compiles_after_ready")
    assert not check.success and "block 1 small {'above_baseline': 2}" in check.detail


def test_sums_name_the_targets_they_cover(monkeypatch, tmp_path, stats_dir):
    FakeService(monkeypatch, _stats_durations())

    result = Boltz2Harness().benchmark(_config(tmp_path, stats_dir, benchmark_blocks=4))

    assert result.metrics["first_s_sum"] == 15.0 and result.metrics["targets_in_first_sum"] == 3
    assert result.metrics["warm_median_s_sum"] == 9.0 and result.metrics["targets_in_warm_sum"] == 3


# --------------------------------------------------------------------------- #
# A set without reference timings, and failing requests
# --------------------------------------------------------------------------- #


def _no_reference_set(cases_dir: Path) -> None:
    (cases_dir.parent / "reference.json").unlink()
    for label in ("big", "small"):
        case_file = cases_dir / label / f"{label}.json"
        case = json.loads(case_file.read_text())
        case["metadata"]["sets"].append("fast")
        case_file.write_text(json.dumps(case))


def test_a_set_without_references_still_reports_times(monkeypatch, tmp_path, cases_dir):
    _no_reference_set(cases_dir)
    FakeService(monkeypatch, _durations())

    result = Boltz2Harness().benchmark(_config(tmp_path, cases_dir, benchmark_set="fast", benchmark_blocks=3))

    assert result.success
    assert _check(result, "p50_ms").value == 3500.0 and _check(result, "throughput").value == 15.0
    assert _check(result, "reference_ratio_warm").skipped
    assert result.metrics["first_s_sum"] == 12.0 and result.metrics["warm_median_s_sum"] == 6.0


def test_a_set_without_references_fails_when_every_request_fails(monkeypatch, tmp_path, cases_dir):
    _no_reference_set(cases_dir)
    service = FakeService(monkeypatch, _durations())
    service.fail_all = True

    result = Boltz2Harness().benchmark(_config(tmp_path, cases_dir, benchmark_set="fast", benchmark_blocks=2))

    requests = _check(result, "requests_ok")
    assert not result.success and not requests.success and "0/4 timed requests succeeded" in requests.detail
    assert "REQUEST(S) FAILED" in result.summary and "clean" not in result.summary
    assert "first_s_sum" not in result.metrics and "warm_median_s_sum" not in result.metrics
    assert not _check(result, "throughput").success


def test_any_failed_request_fails_requests_ok(monkeypatch, tmp_path, cases_dir):
    service = FakeService(monkeypatch, _durations())
    original = service._post

    def flaky(url, payload, timeout=3600):
        if payload["data"]["name"] == "big" and service.seen.get("big", 0) == 1:
            service.seen["big"] = 2
            raise OSError("connection reset")
        return original(url, payload, timeout)

    monkeypatch.setattr(Boltz2Harness, "_post_json", staticmethod(flaky))

    result = Boltz2Harness().benchmark(_config(tmp_path, cases_dir, benchmark_blocks=3))

    requests = _check(result, "requests_ok")
    assert not requests.success and "big block 2: OSError: connection reset" in requests.detail
    assert "1 REQUEST(S) FAILED" in result.summary


# --------------------------------------------------------------------------- #
# The serving mode is recorded, and the compile check says when it cannot apply
# --------------------------------------------------------------------------- #


def test_the_mode_is_in_the_records_metrics_and_summary(monkeypatch, tmp_path, cases_dir):
    FakeService(monkeypatch, _durations())

    result = Boltz2Harness().benchmark(_config(tmp_path, cases_dir, benchmark_blocks=2))

    saved = json.loads((tmp_path / "out" / "benchmark_records.json").read_text())
    assert (saved["optimizations"], saved["warmup_state"]) == ("on", "done")
    assert saved["records"][0]["optimizations"] == "on" and saved["records"][0]["warmup_state"] == "done"
    assert (result.metrics["optimizations"], result.metrics["warmup_state"]) == ("on", "done")
    assert "optimizations=on" in result.summary


def test_compiles_after_ready_is_skipped_when_optimizations_are_off(monkeypatch, tmp_path, cases_dir):
    service = FakeService(monkeypatch, _durations())
    service.optimizations, service.warmup_state = "off", "disabled"

    result = Boltz2Harness().benchmark(_config(tmp_path, cases_dir, benchmark_blocks=2))

    check = _check(result, "compiles_after_ready")
    assert check.skipped and "optimizations=off" in check.detail
    assert result.metrics["optimizations"] == "off" and result.metrics["compiles_after_ready_requests"] is None
    assert "optimizations=off" in result.summary and "compile check skipped" in result.summary


def test_a_missing_compile_count_is_unknown_not_clean(monkeypatch, tmp_path, cases_dir):
    service = FakeService(monkeypatch, _durations())
    service.report_since_warmup = False

    result = Boltz2Harness().benchmark(_config(tmp_path, cases_dir, benchmark_blocks=2))

    check = _check(result, "compiles_after_ready")
    assert check.skipped and "no request reported" in check.detail
    saved = json.loads((tmp_path / "out" / "benchmark_records.json").read_text())
    assert all(r["compiled"] is None for r in saved["records"])


# --------------------------------------------------------------------------- #
# Branches: baseline over several requests, each skip condition, failures
# --------------------------------------------------------------------------- #


def _fail_when(service: FakeService, monkeypatch, should_fail):
    original = service._post

    def flaky(url, payload, timeout=3600):
        label = payload["data"]["name"]
        if should_fail(label, service.seen.get(label, 0) + 1):
            service.seen[label] = service.seen.get(label, 0) + 1
            raise OSError("connection reset")
        return original(url, payload, timeout)

    monkeypatch.setattr(Boltz2Harness, "_post_json", staticmethod(flaky))


def test_only_the_first_request_uses_the_baseline_rule(monkeypatch, tmp_path, stats_dir):
    service = FakeService(monkeypatch, _stats_durations())
    service.compiles[("small", 1)] = 5  # 2 above the baseline
    service.compiles[("mid", 1)] = 6  # one more, by the second request; big adds none

    result = Boltz2Harness().benchmark(_config(tmp_path, stats_dir, benchmark_blocks=1, benchmark_compiles_baseline=3))

    check = _check(result, "compiles_after_ready")
    assert "2/3 timed requests compiled" in check.detail
    assert "block 1 small {'above_baseline': 2}" in check.detail and "block 1 mid {'transition': 1}" in check.detail
    assert result.metrics["compiles_after_ready_requests"] == 2
    assert result.metrics["compiles_since_warmup_final"] == 6


@pytest.mark.parametrize(("optimizations", "state"), [("off", "done"), ("on", "disabled")])
def test_either_signal_alone_skips_the_compile_check(monkeypatch, tmp_path, cases_dir, optimizations, state):
    service = FakeService(monkeypatch, _durations())
    service.optimizations, service.warmup_state = optimizations, state

    result = Boltz2Harness().benchmark(_config(tmp_path, cases_dir, benchmark_blocks=2))

    assert _check(result, "compiles_after_ready").skipped


def test_mixed_modes_are_reported_as_mixed_and_the_compile_check_still_runs(monkeypatch, tmp_path, cases_dir):
    service = FakeService(monkeypatch, _durations())
    service.optimizations_by = {("big", 2): "off"}

    result = Boltz2Harness().benchmark(_config(tmp_path, cases_dir, benchmark_blocks=2))

    assert result.metrics["optimizations"] == "mixed"
    assert not _check(result, "compiles_after_ready").skipped


def test_a_malformed_compile_report_is_unknown_and_does_not_crash(monkeypatch, tmp_path, cases_dir):
    service = FakeService(monkeypatch, _durations())
    service.malformed_since_warmup = True

    result = Boltz2Harness().benchmark(_config(tmp_path, cases_dir, benchmark_blocks=2))

    assert result.metrics["requests_succeeded"] == 4
    assert _check(result, "compiles_after_ready").skipped
    assert result.metrics["compiles_since_warmup_final"] is None


def test_p50_fails_when_no_warm_request_succeeds(monkeypatch, tmp_path, cases_dir):
    service = FakeService(monkeypatch, _durations())
    _fail_when(service, monkeypatch, lambda label, n: n > 1)

    result = Boltz2Harness().benchmark(_config(tmp_path, cases_dir, benchmark_blocks=2))

    p50 = _check(result, "p50_ms")
    assert not p50.success and p50.value is None and "no warm request succeeded" in p50.detail
    assert _check(result, "throughput").value is not None


def test_throughput_counts_only_the_successful_requests(monkeypatch, tmp_path, cases_dir):
    service = FakeService(monkeypatch, _durations())
    _fail_when(service, monkeypatch, lambda label, n: (label, n) == ("big", 2))

    result = Boltz2Harness().benchmark(_config(tmp_path, cases_dir, benchmark_blocks=3))

    # 5 successful requests: small 4 + 1 + 3 s, big 8 + 4 s = 20 s -> 15 structures/min (6 would give 18)
    assert _check(result, "throughput").value == 15.0
    assert "5/6 successful" in _check(result, "throughput").detail


def test_sum_targets_count_only_targets_with_a_successful_request(monkeypatch, tmp_path, stats_dir):
    service = FakeService(monkeypatch, _stats_durations())
    _fail_when(service, monkeypatch, lambda label, n: (label, n) == ("mid", 1))

    result = Boltz2Harness().benchmark(_config(tmp_path, stats_dir, benchmark_blocks=4))

    assert result.metrics["targets_in_first_sum"] == 2 and result.metrics["first_s_sum"] == 12.0
    assert result.metrics["targets_in_warm_sum"] == 3


def test_a_request_without_a_compile_count_is_named_in_the_summary(monkeypatch, tmp_path, cases_dir):
    service = FakeService(monkeypatch, _durations())
    original = service._post

    def partly_silent(url, payload, timeout=3600):
        body = json.loads(original(url, payload, timeout))
        if payload["data"]["name"] == "big":
            del body["runtime"]["warmup"]["since_warmup"]
        return json.dumps(body)

    monkeypatch.setattr(Boltz2Harness, "_post_json", staticmethod(partly_silent))

    result = Boltz2Harness().benchmark(_config(tmp_path, cases_dir, benchmark_blocks=2))

    assert "compile state unknown for 2 request(s)" in result.summary and "clean" not in result.summary


# --------------------------------------------------------------------------- #
# Progress goes to stderr; stdout stays free for the JSON result
# --------------------------------------------------------------------------- #


def test_benchmark_prints_one_progress_line_per_request_on_stderr(monkeypatch, tmp_path, cases_dir, capsys):
    FakeService(monkeypatch, _durations())

    Boltz2Harness().benchmark(_config(tmp_path, cases_dir, benchmark_blocks=3))

    captured = capsys.readouterr()
    assert captured.out == ""
    lines = captured.err.splitlines()
    assert len(lines) == 1 + 6
    assert lines[0] == "[benchmark] set timing-18, fixture recipe: 2 targets x 3 blocks = 6 requests"
    # first request: small, 100 tokens, 4.0 s against a reference time of 2.0 s -> ratio 0.500
    assert lines[1] == (
        "[benchmark] block 1/3 small  100 tok  wall 4.00 s  ratio 0.500 vs reference 2.00 s  compiled=no"
    )


def test_benchmark_progress_line_names_a_compile_and_an_error():
    rec = {"block": 2, "label": "big", "tokens": 200, "wall_s": 4.0, "compiled": True, "reference": None}
    failed = {"block": 1, "label": "big", "wall_s": 0.5, "compiled": None, "error": "OSError: reset"}

    assert harness_mod._benchmark_line(rec, 3, "s") == "[benchmark] block 2/3 big  200 tok  wall 4.00 s  compiled=yes"
    assert harness_mod._benchmark_line(failed, 3, "s").endswith("compiled=unknown  ERROR: OSError: reset")


def test_evaluate_prints_one_progress_line_per_case_on_stderr(evaluation, capsys):
    _, config = evaluation

    Boltz2Harness().evaluate(config)

    captured = capsys.readouterr()
    assert captured.out == ""
    lines = [line for line in captured.err.splitlines() if line.startswith("[evaluate]")]
    assert lines == [
        "[evaluate] 1/2 big  RMSD n/a  pLDDT 0.9100  wall 3.00 s",
        "[evaluate] 2/2 small  RMSD 1.500 A  pLDDT 0.9100  wall 2.00 s",
    ]


def test_native_output_is_captured_and_does_not_reach_the_streams(capfd):
    import os

    with harness_mod._native_output_captured() as native:
        os.write(1, b"to stdout\n")
        os.write(2, b"More than 50 iterations needed!\n")

    captured = capfd.readouterr()
    assert captured.out == "" and captured.err == ""
    assert native == ["to stdout", "More than 50 iterations needed!"]
    os.write(1, b"after\n")
    assert capfd.readouterr().out == "after\n"


def test_evaluate_turns_a_solver_limit_note_into_a_warning(evaluation, monkeypatch):
    _, config = evaluation

    def noted(mmcif, ref, notes=None):
        notes.append("gemmi's superposition solver reached its 50-iteration limit for 1 chain pair(s)")
        return 1.5, 90

    monkeypatch.setattr(Boltz2Harness, "_ca_rmsd", staticmethod(noted))

    result = Boltz2Harness().evaluate(config)

    warnings = _check(result, "structure_accuracy").warnings
    assert any(w.startswith("small: gemmi's superposition solver") for w in warnings)
    small = next(c for c in result.metrics["cases"] if c["label"] == "small")
    assert len(small["rmsd_notes"]) == 1


def test_the_served_recipe_progress_line_says_the_ratio_is_a_different_workload():
    rec = {
        "block": 1,
        "label": "small",
        "tokens": 100,
        "wall_s": 4.0,
        "compiled": False,
        "reference": {"first_s": 2.0, "warm_s": 2.0, "alt_warm_s": 10.0},
    }

    assert "(served recipe: different workload)" in harness_mod._benchmark_line(rec, 2, "s", served=True)
    assert "different workload" not in harness_mod._benchmark_line(rec, 2, "s")


def test_evaluate_keeps_the_solver_note_when_the_rmsd_step_then_fails(evaluation, monkeypatch):
    _, config = evaluation

    def noted_then_failing(mmcif, ref, notes=None):
        notes.append("gemmi's superposition solver reached its 50-iteration limit for 1 chain pair(s)")
        raise ValueError("no residues could be aligned between prediction and reference")

    monkeypatch.setattr(Boltz2Harness, "_ca_rmsd", staticmethod(noted_then_failing))

    result = Boltz2Harness().evaluate(config)

    small = next(c for c in result.metrics["cases"] if c["label"] == "small")
    assert "no residues could be aligned" in small["error"] and len(small["rmsd_notes"]) == 1
    assert any(
        w.startswith("small: gemmi's superposition solver") for w in _check(result, "structure_accuracy").warnings
    )


def test_a_zero_client_time_is_left_out_of_the_ratios_and_does_not_divide():
    nv = {"first_s": 10.0, "warm_s": 10.0, "alt_warm_s": 40.0}
    records = [
        {"block": 1, "label": "a", "wall_s": 0.0, "reference": nv},
        {"block": 2, "label": "a", "wall_s": 0.0, "reference": nv},
        {"block": 1, "label": "b", "wall_s": 5.0, "reference": nv},
    ]

    ratios = reference_ratios(records)

    assert ratios["first"] == {"b": 2.0} and ratios["warm"] == {}


# --------------------------------------------------------------------------- #
# Another benchmark in the same container
# --------------------------------------------------------------------------- #


def _proc(root: Path, pid: int, ppid: int, argv: list[str]) -> None:
    (root / str(pid)).mkdir(parents=True)
    (root / str(pid) / "stat").write_text(f"{pid} (python) S {ppid} {pid} {pid} 0 -1 4194560\n")
    (root / str(pid) / "cmdline").write_bytes(b"\0".join(a.encode() for a in argv) + b"\0")


def test_other_benchmark_processes_are_found_and_this_one_and_its_shell_are_not(tmp_path):
    proc = tmp_path / "proc"
    proc.mkdir()
    _proc(proc, 1, 0, ["tini"])
    _proc(proc, 90, 1, ["python3", "entrypoint.py", "benchmark", "--output-dir", "/artifacts"])  # our parent
    _proc(proc, 100, 90, ["python3", "/workspace/entrypoint.py", "benchmark"])  # this process
    _proc(proc, 200, 1, ["python3", "/workspace/entrypoint.py", "benchmark", "--service-url", "http://x:8000"])
    _proc(proc, 300, 1, ["python3", "/workspace/entrypoint.py", "serve"])
    _proc(proc, 400, 1, ["python3", "/workspace/entrypoint.py", "evaluate"])
    _proc(proc, 500, 1, ["python3", "-m", "pytest", "tests/benchmark"])
    _proc(proc, 600, 1, ["grep", "entrypoint.py", "benchmark"])  # not the command
    (proc / "self").mkdir()

    found = harness_mod.concurrent_benchmark_processes(str(proc), own_pid=100)

    assert found == ["200: python3 /workspace/entrypoint.py benchmark --service-url http://x:8000"]


def test_an_unreadable_proc_gives_no_processes(tmp_path):
    assert harness_mod.concurrent_benchmark_processes(str(tmp_path / "missing"), own_pid=1) == []


def test_a_concurrent_benchmark_is_warned_about_on_stderr_checks_and_summary(monkeypatch, tmp_path, cases_dir, capsys):
    FakeService(monkeypatch, _durations())
    other = "200: python3 /workspace/entrypoint.py benchmark"
    monkeypatch.setattr(harness_mod, "concurrent_benchmark_processes", lambda *a, **k: [other])

    result = Boltz2Harness().benchmark(_config(tmp_path, cases_dir, benchmark_blocks=2))

    assert f"WARNING: another benchmark is running in this container ({other})" in capsys.readouterr().err
    for name in ("requests_ok", "compiles_after_ready"):
        assert any("200: python3" in w for w in _check(result, name).warnings), name
    assert "1 CONCURRENT BENCHMARK PROCESS(ES)" in result.summary and "clean" not in result.summary


def test_a_benchmark_that_starts_during_the_run_is_reported(monkeypatch, tmp_path, cases_dir, capsys):
    FakeService(monkeypatch, _durations())
    answers = iter([[], ["300: python3 entrypoint.py benchmark"]])
    monkeypatch.setattr(harness_mod, "concurrent_benchmark_processes", lambda *a, **k: next(answers))

    result = Boltz2Harness().benchmark(_config(tmp_path, cases_dir, benchmark_blocks=2))

    assert "another benchmark started during this run (300:" in capsys.readouterr().err
    assert any("300:" in w for w in _check(result, "requests_ok").warnings)


def test_no_concurrent_benchmark_leaves_no_warning(monkeypatch, tmp_path, cases_dir):
    FakeService(monkeypatch, _durations())
    monkeypatch.setattr(harness_mod, "concurrent_benchmark_processes", lambda *a, **k: [])

    result = Boltz2Harness().benchmark(_config(tmp_path, cases_dir, benchmark_blocks=2))

    assert _check(result, "requests_ok").warnings == [] and "CONCURRENT" not in result.summary


def test_health_check_waits_on_the_monotonic_clock_and_gives_up_at_the_deadline(monkeypatch):
    ticks = iter(range(0, 1000, 3))
    monkeypatch.setattr(harness_mod.time, "monotonic", lambda: next(ticks))
    monkeypatch.setattr(harness_mod.time, "sleep", lambda s: None)
    monkeypatch.setattr(harness_mod.time, "time", lambda: pytest.fail("wall clock used"))

    def refuse(req, timeout=None):
        raise OSError("connection refused")

    monkeypatch.setattr(harness_mod, "urlopen", refuse)

    assert Boltz2Harness().health_check("http://svc", timeout_seconds=10) is False


# --------------------------------------------------------------------------- #
# Reference-times file
# --------------------------------------------------------------------------- #


def test_a_reference_file_is_read_per_target(tmp_path):
    path = tmp_path / "ref.json"
    path.write_text(
        json.dumps(
            {
                "targets": {
                    "a": {"first_s": 2.0, "warm_s": 1.5, "alt_warm_s": 4},
                    "b": {"first_s": 3, "warm_s": 3},
                }
            }
        )
    )

    assert harness_mod.load_reference_times(path) == {
        "a": {"first_s": 2.0, "warm_s": 1.5, "alt_warm_s": 4.0},
        "b": {"first_s": 3.0, "warm_s": 3.0},
    }


@pytest.mark.parametrize(
    "document",
    [
        "not json",
        json.dumps([]),
        json.dumps({"targets": {}}),
        json.dumps({"targets": {"a": {"first_s": 1.0}}}),
        json.dumps({"targets": {"a": {"first_s": 1.0, "warm_s": 0}}}),
        json.dumps({"targets": {"a": {"first_s": "1", "warm_s": 1.0}}}),
    ],
)
def test_a_malformed_reference_file_is_refused(tmp_path, document):
    path = tmp_path / "ref.json"
    path.write_text(document)
    with pytest.raises(ValueError, match="benchmark_reference_file"):
        harness_mod.load_reference_times(path)


def test_the_second_reference_ratio_is_left_out_when_a_target_has_no_alt_time():
    records = [
        {"block": 1, "label": "a", "wall_s": 4.0, "reference": {"first_s": 2.0, "warm_s": 2.0}},
        {"block": 2, "label": "a", "wall_s": 2.0, "reference": {"first_s": 2.0, "warm_s": 2.0}},
    ]

    ratios = reference_ratios(records)

    assert ratios["first"] == {"a": 0.5} and ratios["warm"] == {"a": 1.0} and ratios["alt_warm"] == {}
