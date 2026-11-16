# Copyright © Advanced Micro Devices, Inc., or its affiliates.
#
# SPDX-License-Identifier: MIT

"""CPU unit tier: start-up warm-up settings, inputs, readiness ordering and reporting."""

from __future__ import annotations

import compilation
import pytest
import warmup
from warmup import (
    CHUNK_SIZE_THRESHOLD,
    STATE_DISABLED,
    STATE_DONE,
    STATE_FAILED,
    STATE_PENDING,
    WARMUP_TOKENS,
    WarmupReport,
    run_warmup,
    synthetic_sequence,
    warmup_requests,
)

# --------------------------------------------------------------------------- #
# Pure logic
# --------------------------------------------------------------------------- #


def test_default_inputs_cover_the_plain_and_the_chunked_transition_path():
    assert min(WARMUP_TOKENS) <= CHUNK_SIZE_THRESHOLD < max(WARMUP_TOKENS)


def test_synthetic_sequence_is_fixed_and_exact_length():
    assert synthetic_sequence(400) == synthetic_sequence(400)
    assert len(synthetic_sequence(1)) == 1
    assert len(synthetic_sequence(400)) == 400
    assert set(synthetic_sequence(400)) <= set("ACDEFGHIKLMNPQRSTVWY")


def test_requests_cover_both_sizes_with_the_empty_and_a_multi_row_msa_and_use_the_default_recipe():
    reqs = warmup_requests()
    assert [(len(r["sequences"][0]["protein"]["sequence"]), r.get("msa_rows", 1)) for r in reqs] == [
        (128, 1),
        (128, 3),
        (400, 1),
        (400, 3),
    ]
    assert len({r["name"] for r in reqs}) == 4
    for r in reqs:
        assert "msa" not in r["sequences"][0]["protein"]  # the A3M path is added by run_warmup
        assert "use_msa_server" not in r
        assert r["recycling_steps"] == 3
        assert r["diffusion_samples"] == 1


def test_synthetic_a3m_has_the_query_first_and_the_requested_number_of_rows():
    seq = synthetic_sequence(50)
    lines = warmup.synthetic_a3m(seq, 3).splitlines()
    assert lines[:2] == [">query", seq]
    assert len(lines) == 6 and lines[3] != seq and lines[5] != lines[3]
    assert all(len(lines[i]) == 50 for i in (1, 3, 5))


def test_run_warmup_gives_the_a3m_inline_for_multi_row_requests_and_removes_the_internal_key():
    seen = []

    def predict(**kw):
        assert "msa_rows" not in kw
        msa = kw["sequences"][0]["protein"].get("msa")
        seen.append((kw["name"], msa and msa.count(">")))
        return {"error": False}

    fake = _FakeCompiler()
    run_warmup(predict, warmup_requests(), fake.status_fn, WarmupReport())
    assert seen == [("warmup_128", None), ("warmup_128_msa3", 3), ("warmup_400", None), ("warmup_400_msa3", 3)]


def test_every_warmup_input_passes_the_check_that_a_client_request_passes():
    """The warm-up calls the runner like a client does: inline content only, no file paths."""
    from runner import reject_file_references

    inputs = []
    for request in warmup_requests():
        kwargs = warmup._materialize(request)
        reject_file_references(kwargs["sequences"], kwargs.get("templates"))
        inputs.append(kwargs)
    assert any(k["sequences"][0]["protein"].get("msa", "").startswith(">") for k in inputs)


def test_the_synthetic_a3m_parses_with_boltz_when_boltz_is_installed(tmp_path):
    a3m = pytest.importorskip("boltz.data.parse.a3m")
    seq = synthetic_sequence(40)
    path = tmp_path / "x.a3m"
    path.write_text(warmup.synthetic_a3m(seq, 3))
    msa = a3m.parse_a3m(path, None, 100)
    assert len(msa.sequences) == 3


# --------------------------------------------------------------------------- #
# run_warmup and the report
# --------------------------------------------------------------------------- #


class _FakeCompiler:
    """A predict function that 'compiles' on its first call of each warm-up size."""

    def __init__(self, fail_on=None, error_result_on=None):
        self.status = {
            "transition": compilation.CompileStatus(mode="inductor"),
            "token_transformer": compilation.CompileStatus(mode="inductor"),
        }
        self.seen = set()
        self.fail_on = fail_on
        self.error_result_on = error_result_on

    def predict(self, **request):
        if request["name"] == self.fail_on:
            raise RuntimeError("boom")
        if request["name"] == self.error_result_on:
            return {"error": True, "message": "no structure"}
        if request["name"] not in self.seen:
            self.seen.add(request["name"])
            self.status["transition"].frame_compiles[1] = self.status["transition"].frame_compiles.get(1, 0) + 2
        return {"error": False}

    def status_fn(self):
        return {k: v.to_dict() for k, v in self.status.items()}


def test_run_warmup_records_steps_compiles_and_a_baseline():
    fake = _FakeCompiler()
    report = run_warmup(fake.predict, warmup_requests(), fake.status_fn, WarmupReport())

    assert report.state == STATE_DONE and report.settled
    assert [s["name"] for s in report.steps] == ["warmup_128", "warmup_128_msa3", "warmup_400", "warmup_400_msa3"]
    assert report.steps[0]["compiles"]["transition"] == 2
    assert report.compiles_during_warmup["transition"]["successful_compiles"] == 8
    assert report.duration_s is not None

    # A user request after the warm-up: no compile.
    since = report.to_dict(fake.status_fn())["since_warmup"]
    assert since["compiles"]["transition"] == {"successful_compiles": 0, "failed_compiles": 0}

    # A user request that compiles is visible as a compile since the warm-up.
    fake.status["transition"].frame_compiles[1] += 1
    since = report.to_dict(fake.status_fn())["since_warmup"]
    assert since["compiles"]["transition"]["successful_compiles"] == 1


def test_extra_steps_run_after_the_predictions_and_are_reported():
    fake = _FakeCompiler()
    calls = []
    report = run_warmup(
        fake.predict, warmup_requests(), fake.status_fn, WarmupReport(), [("long", lambda: calls.append(1))]
    )
    assert report.state == STATE_DONE and calls == [1]
    assert [s["name"] for s in report.steps] == [
        "warmup_128",
        "warmup_128_msa3",
        "warmup_400",
        "warmup_400_msa3",
        "long",
    ]


def test_a_failing_extra_step_fails_the_warmup_without_raising():
    fake = _FakeCompiler()

    def boom():
        raise RuntimeError("no memory")

    report = run_warmup(fake.predict, warmup_requests(), fake.status_fn, WarmupReport(), [("long", boom)])
    assert report.state == STATE_FAILED and "no memory" in report.error


@pytest.mark.parametrize("mode", ["raise", "error_result"])
def test_failed_warmup_is_recorded_and_does_not_raise(mode):
    fake = _FakeCompiler(
        fail_on="warmup_400_msa3" if mode == "raise" else None,
        error_result_on="warmup_400_msa3" if mode == "error_result" else None,
    )
    report = run_warmup(fake.predict, warmup_requests(), fake.status_fn, WarmupReport())
    assert report.state == STATE_FAILED and report.settled
    assert report.error
    assert [s["name"] for s in report.steps] == ["warmup_128", "warmup_128_msa3", "warmup_400"]


def test_report_is_not_settled_until_the_warmup_ends():
    assert not WarmupReport().settled  # pending
    assert WarmupReport(state=STATE_DISABLED).settled
    assert not WarmupReport(state=warmup.STATE_RUNNING).settled


# --------------------------------------------------------------------------- #
# Service wiring: settings, readiness ordering, reporting
# --------------------------------------------------------------------------- #


def _service(monkeypatch, tmp_path, *, optimizations="on"):
    pytest.importorskip("bentoml")
    pytest.importorskip("torch")
    import service

    cache_dir = tmp_path / "cache"
    cache_dir.mkdir(exist_ok=True)
    monkeypatch.setattr(service.CONFIG, "cache", cache_dir)
    monkeypatch.setattr(service.CONFIG, "download_assets", False)
    monkeypatch.setattr(service, "assets_present", lambda cache, *args: True)
    monkeypatch.setattr(service.CONFIG, "optimizations", optimizations)

    model = type("FakeModel", (), {})()
    monkeypatch.setattr(service, "load_boltz2_model", lambda cache, **kwargs: model)
    calls = []

    def fake_run(**kwargs):
        calls.append(kwargs)
        return {"error": False, "structures": [], "confidence": {}, "timing": {}}

    monkeypatch.setattr(service, "run_boltz2_prediction", fake_run)
    return service, calls


def test_service_is_not_ready_before_the_warmup_and_ready_after(monkeypatch, tmp_path):
    service, calls = _service(monkeypatch, tmp_path)
    svc = service.Boltz2Prediction.inner()

    assert svc._warmup.state == STATE_PENDING  # noqa: SLF001
    assert svc.__is_ready__() is False
    assert calls == []

    svc._warm_up()  # noqa: SLF001
    assert svc.__is_ready__() is True
    assert [len(c["sequences"][0]["protein"]["sequence"]) for c in calls] == [128, 128, 400, 400]
    assert all(c["accelerator"] == service.CONFIG.accelerator and c["devices"] == 1 for c in calls)
    info = svc._runtime_info()["warmup"]  # noqa: SLF001
    assert info["state"] == STATE_DONE
    assert info["duration_s"] is not None


def test_service_is_not_ready_while_the_warmup_runs(monkeypatch, tmp_path):
    service, _ = _service(monkeypatch, tmp_path)
    svc = service.Boltz2Prediction.inner()
    seen = []

    def predict(**kwargs):
        seen.append((svc.__is_ready__(), svc._warmup.state))  # noqa: SLF001
        return {"error": False}

    monkeypatch.setattr(service, "run_boltz2_prediction", predict)
    svc._warm_up()  # noqa: SLF001
    assert seen == [(False, warmup.STATE_RUNNING)] * 4


def test_startup_hook_is_registered_with_bentoml(monkeypatch, tmp_path):
    service, _ = _service(monkeypatch, tmp_path)
    assert getattr(service.Boltz2Prediction.inner._warm_up, "__bentoml_startup_hook__", False)  # noqa: SLF001


def test_no_warmup_when_optimizations_are_off(monkeypatch, tmp_path):
    service, calls = _service(monkeypatch, tmp_path, optimizations="off")
    svc = service.Boltz2Prediction.inner()
    assert svc.__is_ready__() is True
    svc._warm_up()  # noqa: SLF001
    assert calls == []
    info = svc._runtime_info()["warmup"]  # noqa: SLF001
    assert info["state"] == STATE_DISABLED and info["reason"]


def test_only_the_affinity_warmup_runs_when_optimizations_are_off_in_affinity_mode(monkeypatch, tmp_path):
    service, calls = _service(monkeypatch, tmp_path, optimizations="off")
    monkeypatch.setattr(service, "needs_affinity_model", lambda mode: True)
    loaded = {}

    def load(cache, **kwargs):
        loaded.update(kwargs)
        return type("FakeAffinity", (), {})()

    monkeypatch.setattr(service, "load_affinity_model", load)
    svc = service.Boltz2Prediction.inner()
    assert loaded["optimizations"] is False
    assert svc.__is_ready__() is False
    svc._warm_up()  # noqa: SLF001
    assert [c["name"] for c in calls] == ["warmup_affinity"]
    assert svc.__is_ready__() is True
    info = svc._runtime_info()["warmup"]  # noqa: SLF001
    assert info["state"] == STATE_DONE
    assert [s["name"] for s in info["steps"]] == ["affinity"]


def test_affinity_warmup_follows_the_structure_inputs_when_optimizations_are_on(monkeypatch, tmp_path):
    service, calls = _service(monkeypatch, tmp_path)
    monkeypatch.setattr(service, "needs_affinity_model", lambda mode: True)
    monkeypatch.setattr(service, "load_affinity_model", lambda cache, **kwargs: object())
    svc = service.Boltz2Prediction.inner()
    svc._warm_up()  # noqa: SLF001
    assert calls[-1]["name"] == "warmup_affinity"
    assert len(calls) == 5


def test_the_affinity_model_is_loaded_with_the_optimizations_setting(monkeypatch, tmp_path):
    service, _ = _service(monkeypatch, tmp_path)
    seen = {}

    def load(cache, **kwargs):
        seen.update(kwargs)
        return type("FakeAffinity", (), {})()

    monkeypatch.setattr(service.CONFIG, "mode", "affinity")
    monkeypatch.setattr(service, "load_affinity_model", load)
    service.Boltz2Prediction.inner()
    assert seen["optimizations"] is True


def test_failed_warmup_keeps_the_service_ready_and_reports_the_error(monkeypatch, tmp_path):
    service, _ = _service(monkeypatch, tmp_path)

    def failing(**kwargs):
        raise RuntimeError("gpu gone")

    monkeypatch.setattr(service, "run_boltz2_prediction", failing)
    svc = service.Boltz2Prediction.inner()
    svc._warm_up()  # noqa: SLF001
    assert svc.__is_ready__() is True
    info = svc._runtime_info()["warmup"]  # noqa: SLF001
    assert info["state"] == STATE_FAILED and "gpu gone" in info["error"]


def test_affinity_mode_adds_one_complex_warmup_that_reaches_the_affinity_model(monkeypatch, tmp_path):
    service, calls = _service(monkeypatch, tmp_path)
    affinity_model = type("FakeAffinity", (), {})()
    monkeypatch.setattr(service.CONFIG, "mode", "affinity")
    monkeypatch.setattr(service, "load_affinity_model", lambda cache, **kwargs: affinity_model)
    svc = service.Boltz2Prediction.inner()
    svc._warm_up()  # noqa: SLF001
    assert len(calls) == 5
    assert all(c["affinity_model"] is affinity_model for c in calls)
    last = calls[-1]
    assert last["properties"] == [{"affinity": {"binder": "B"}}]
    assert "ligand" in last["sequences"][1]
    assert svc._runtime_info()["warmup"]["state"] == STATE_DONE  # noqa: SLF001


def test_structure_mode_loads_no_affinity_model_and_warms_no_affinity_input(monkeypatch, tmp_path):
    service, calls = _service(monkeypatch, tmp_path)
    monkeypatch.setattr(service, "load_affinity_model", lambda cache, **kwargs: pytest.fail("loaded"))
    svc = service.Boltz2Prediction.inner()
    svc._warm_up()  # noqa: SLF001
    assert len(calls) == 4
    assert all(c["affinity_model"] is None and "properties" not in c for c in calls)


def test_a_failed_affinity_warmup_is_reported_after_the_structure_steps(monkeypatch, tmp_path):
    service, _ = _service(monkeypatch, tmp_path)
    monkeypatch.setattr(service.CONFIG, "mode", "affinity")
    monkeypatch.setattr(service, "load_affinity_model", lambda cache, **kwargs: object())

    def predict(**kwargs):
        if "properties" in kwargs:
            return {"error": True, "message": "no affinity"}
        return {"error": False}

    monkeypatch.setattr(service, "run_boltz2_prediction", predict)
    svc = service.Boltz2Prediction.inner()
    svc._warm_up()  # noqa: SLF001
    info = svc._runtime_info()["warmup"]  # noqa: SLF001
    assert info["state"] == STATE_FAILED and "no affinity" in info["error"]
    assert len(info["steps"]) == 4
    assert svc.__is_ready__() is True


def _affinity_service(monkeypatch, tmp_path, *, compile_affinity):
    service, calls = _service(monkeypatch, tmp_path)
    monkeypatch.setattr(service, "needs_affinity_model", lambda mode: True)
    monkeypatch.setattr(service.CONFIG, "affinity_compile", "on" if compile_affinity else "off")
    loaded = {}

    def load(cache, **kwargs):
        loaded.update(kwargs)
        return type("FakeAffinity", (), {})()

    monkeypatch.setattr(service, "load_affinity_model", load)
    return service, calls, loaded


def test_the_affinity_warmup_is_unchanged_when_the_affinity_compile_is_off(monkeypatch, tmp_path):
    service, calls, loaded = _affinity_service(monkeypatch, tmp_path, compile_affinity=False)
    svc = service.Boltz2Prediction.inner()
    svc._warm_up()  # noqa: SLF001
    assert loaded["compile_affinity"] is False
    assert [c["name"] for c in calls][-1:] == ["warmup_affinity"]
    assert len(calls) == 5
    assert calls[-1]["sampling_steps_affinity"] == warmup.WARMUP_SAMPLING_STEPS
    info = svc._runtime_info()  # noqa: SLF001
    assert "affinity_compile_status" not in info
    assert [s["name"] for s in info["warmup"]["steps"]][-1:] == ["affinity"]


def test_the_affinity_warmup_compiles_at_the_crop_size_when_the_affinity_compile_is_on(monkeypatch, tmp_path):
    service, calls, loaded = _affinity_service(monkeypatch, tmp_path, compile_affinity=True)
    svc = service.Boltz2Prediction.inner()
    svc._warm_up()  # noqa: SLF001
    assert loaded["compile_affinity"] is True
    assert [c["name"] for c in calls][-2:] == ["warmup_affinity", "warmup_affinity_crop"]
    assert calls[-2]["sampling_steps_affinity"] == warmup.AFFINITY_COMPILE_WARMUP_SAMPLING_STEPS
    assert calls[-1]["sampling_steps_affinity"] == warmup.AFFINITY_COMPILE_WARMUP_SAMPLING_STEPS
    assert calls[-1]["diffusion_samples_affinity"] == 1
    info = svc._runtime_info()  # noqa: SLF001
    assert [s["name"] for s in info["warmup"]["steps"]][-2:] == ["affinity", "affinity_crop"]


@pytest.mark.parametrize(("mode", "optimizations"), [("structure", "on"), ("affinity", "off")])
def test_the_affinity_compile_is_ignored_and_logged_once(monkeypatch, tmp_path, caplog, mode, optimizations):
    service, calls, loaded = _affinity_service(monkeypatch, tmp_path, compile_affinity=True)
    monkeypatch.setattr(service.CONFIG, "mode", mode)
    monkeypatch.setattr(service.CONFIG, "optimizations", optimizations)
    monkeypatch.setattr(service, "needs_affinity_model", lambda m: m == "affinity")
    with caplog.at_level("INFO", logger="service"):
        svc = service.Boltz2Prediction.inner()
    assert caplog.text.count("BOLTZ2_AFFINITY_COMPILE=on is ignored") == 1
    assert loaded.get("compile_affinity", False) is False
    svc._warm_up()  # noqa: SLF001
    assert "warmup_affinity_crop" not in [c.get("name") for c in calls]
