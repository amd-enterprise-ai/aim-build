# Copyright © Advanced Micro Devices, Inc., or its affiliates.
#
# SPDX-License-Identifier: MIT

"""Service tests: port argument, AIM_CACHE_PATH pre-flight.

No GPU or weights: torch/transformers are stubbed and the upstream BentoML
``/healthz`` call is replaced.
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path
from unittest.mock import MagicMock

import pytest

IMAGE_DIR = Path(__file__).resolve().parents[1]
# Optional deps: CI test environments may lack them, so skip the module before touching them.
httpx = pytest.importorskip("httpx")
pytest.importorskip("pydantic")
TestClient = pytest.importorskip("starlette.testclient").TestClient


def _upstream(status: int):
    async def fake() -> httpx.Response:
        return httpx.Response(status)

    return fake


def test_v1_models_lists_served_model(load_service):
    svc = load_service()
    body = TestClient(svc.canonical_app).get("/v1/models").json()
    assert body == {"object": "list", "data": [{"id": "biohub/ESMC-300M", "object": "model"}]}


@pytest.mark.parametrize(("upstream", "status", "body"), [(200, 200, "ok"), (503, 503, "unavailable")])
def test_health_forwards_upstream(load_service, monkeypatch, upstream, status, body):
    svc = load_service()
    monkeypatch.setattr(svc, "_upstream_healthz", _upstream(upstream))
    r = TestClient(svc.canonical_app).get("/health")
    assert (r.status_code, r.json()) == (status, {"status": body})


def test_health_unreachable_upstream_is_503(load_service, monkeypatch):
    svc = load_service()

    async def boom():
        raise httpx.ConnectError("down")

    monkeypatch.setattr(svc, "_upstream_healthz", boom)
    assert TestClient(svc.canonical_app).get("/health").status_code == 503


def test_get_only(load_service):
    c = TestClient(load_service().canonical_app)
    assert c.post("/health").status_code == 405
    assert c.post("/v1/models").status_code == 405


def test_routes_and_flat_bodies(load_bentoml_service):
    from pydantic import ValidationError

    svc = load_bentoml_service()
    apis = svc.ESMCEmbeddingService.apis
    # /v1/inference is the only inference route (no /predict alias, no /v1/masked_lm)
    assert {a.route for a in apis.values()} == {"/v1/inference"}
    assert set(apis["v1_inference"].input_spec.model_fields) == {"sequence", "mode", "mask_fraction", "seed"}
    inference = apis["v1_inference"].input_spec
    with pytest.raises(ValidationError):
        inference(sequence="")
    with pytest.raises(ValidationError):
        inference(sequence="MKV", mode="nope")
    for bad in (0.0, 1.0):
        with pytest.raises(ValidationError):
            inference(sequence="MKV", mode="masked_lm", mask_fraction=bad)
    assert inference(sequence="MKV").mode == "embedding"
    assert inference(sequence="MKV").mask_fraction == 0.15


class _Exited(Exception):
    """Raised by the os._exit stub so execution stops where the real process would."""


def _stub_exit(monkeypatch, svc) -> list[int]:
    exits: list[int] = []

    def fake_exit(code: int) -> None:
        exits.append(code)
        raise _Exited

    monkeypatch.setattr(svc.os, "_exit", fake_exit)
    return exits


def test_no_gpu_fails_loudly(load_service, monkeypatch, caplog):
    svc = load_service(cuda_available=False)
    exits = _stub_exit(monkeypatch, svc)
    with pytest.raises(_Exited):
        svc.ESMCEmbeddingService()
    assert exits == [1]
    assert "No GPU found" in caplog.text


def test_bentofile_and_dockerfile_wiring():
    bentofile = (IMAGE_DIR / "bentofile.yaml").read_text(encoding="utf-8")
    assert 'service: "service:ESMCEmbeddingService"' in bentofile
    dockerfile = (IMAGE_DIR / "Dockerfile").read_text(encoding="utf-8")
    assert "COPY bentofile.yaml /workspace/model/bentofile.yaml" in dockerfile
    assert "BENTOML_PORT" not in dockerfile
    assert "BENTOML_PORT" not in (IMAGE_DIR / "src/service.py").read_text(encoding="utf-8")


def test_port_comes_only_from_bento_arguments(load_service):
    svc = load_service(port=4321)
    assert svc.args.port == 4321
    assert svc.BentoArgs.model_fields["port"].is_required()


def test_single_gpu_by_default(load_service):
    svc = load_service(cuda_available=True)
    assert svc.args.accelerator_count == 1
    (kwargs,) = svc.RECORDED_SERVICE_KWARGS
    assert kwargs["resources"] == {"gpu": 1}
    assert kwargs["workers"] == 1


def test_two_accelerators_request_two_gpus_and_two_workers(load_service):
    svc = load_service(cuda_available=True, accelerator_count=2, device_count=2)
    (kwargs,) = svc.RECORDED_SERVICE_KWARGS
    assert kwargs["resources"] == {"gpu": 2}
    assert kwargs["workers"] == 2


def test_accelerator_count_must_be_positive(load_service):
    with pytest.raises(ValueError):
        load_service(accelerator_count=0)


def test_each_worker_takes_its_own_gpu(load_service):
    both_visible = load_service(cuda_available=True, accelerator_count=2, device_count=2)
    assert [both_visible._worker_device_index(w, 2, 2) for w in (1, 2)] == [0, 1]
    # Fewer visible GPUs than accelerators would make workers share one: startup fails.
    with pytest.raises(
        RuntimeError, match=r"AIM_ACCELERATOR_COUNT=2 requires 2 accelerator\(s\), but this container has 1"
    ):
        both_visible._worker_device_index(2, 1, 2)


def test_worker_device_is_derived_from_worker_index(load_service, tmp_path, monkeypatch):
    svc = load_service(cuda_available=True, worker_index=2, device_count=2)
    monkeypatch.setenv("AIM_CACHE_PATH", str(tmp_path))
    svc.EsmcForMaskedLM.from_pretrained.return_value = MagicMock()
    instance = svc.ESMCEmbeddingService()
    assert instance._device == "cuda:1"
    assert svc.torch.set_device_calls == [1]


def test_single_gpu_still_uses_cuda0(load_service, tmp_path, monkeypatch):
    svc = load_service(cuda_available=True, worker_index=1, device_count=1)
    monkeypatch.setenv("AIM_CACHE_PATH", str(tmp_path))
    svc.EsmcForMaskedLM.from_pretrained.return_value = MagicMock()
    instance = svc.ESMCEmbeddingService()
    assert instance._device == "cuda:0"
    assert svc.torch.set_device_calls == [0]


def test_fewer_gpus_than_workers_ends_startup_before_binding_a_device(load_service, tmp_path, monkeypatch):
    svc = load_service(cuda_available=True, worker_index=2, accelerator_count=2, device_count=1)
    monkeypatch.setenv("AIM_CACHE_PATH", str(tmp_path))
    fail = MagicMock(side_effect=SystemExit(1))
    monkeypatch.setattr(svc, "_fail_startup", fail)
    with pytest.raises(SystemExit):
        svc.ESMCEmbeddingService()
    assert "requires 2 accelerator(s), but this container has 1" in fail.call_args.args[0]
    assert svc.torch.set_device_calls == []


def test_loopback_health_uses_forwarded_port(load_service, monkeypatch):
    svc = load_service(port=4321)
    seen: list[str] = []

    class FakeClient:
        def __init__(self, *a, **k):
            pass

        async def __aenter__(self):
            return self

        async def __aexit__(self, *a):
            return False

        async def get(self, url):
            seen.append(url)
            return httpx.Response(200)

    monkeypatch.setattr(svc.httpx, "AsyncClient", FakeClient)
    TestClient(svc.canonical_app).get("/health")
    assert seen == ["http://127.0.0.1:4321/healthz"]


def test_preflight_logs_cache_path_and_token_need(load_service, tmp_path, monkeypatch, caplog):
    svc = load_service()
    monkeypatch.delenv("HF_TOKEN", raising=False)
    svc.LOGGER.addHandler(caplog.handler)  # the module logger does not propagate to root
    with caplog.at_level(logging.INFO, logger=svc.LOGGER.name):
        assert svc._preflight_weights(tmp_path) is False
    assert str(tmp_path) in caplog.text
    assert "AIM_CACHE_PATH" in caplog.text and "HF_TOKEN" in caplog.text
    (tmp_path / "models--biohub--ESMC-300M" / "snapshots" / "abc").mkdir(parents=True)
    assert svc._preflight_weights(tmp_path) is True


def test_failed_load_stops_startup_naming_cache_path_and_token(load_service, tmp_path, monkeypatch, caplog):
    monkeypatch.setenv("AIM_CACHE_PATH", str(tmp_path))
    svc = load_service(cuda_available=True)
    svc.AutoTokenizer.from_pretrained.side_effect = OSError("offline")
    exits = _stub_exit(monkeypatch, svc)
    svc.LOGGER.addHandler(caplog.handler)
    with pytest.raises(_Exited):
        svc.ESMCEmbeddingService()
    assert exits == [1]
    msg = caplog.text
    assert str(tmp_path) in msg and "AIM_CACHE_PATH" in msg and "HF_TOKEN" in msg
    assert svc.AutoTokenizer.from_pretrained.call_args.kwargs["cache_dir"] == str(tmp_path)


def test_both_loaders_receive_cache_dir(load_service, tmp_path, monkeypatch):
    monkeypatch.setenv("AIM_CACHE_PATH", str(tmp_path))
    svc = load_service(cuda_available=True)
    svc.EsmcForMaskedLM.from_pretrained.return_value = MagicMock()
    svc.ESMCEmbeddingService()
    assert svc.AutoTokenizer.from_pretrained.call_args.kwargs["cache_dir"] == str(tmp_path)
    assert svc.EsmcForMaskedLM.from_pretrained.call_args.kwargs["cache_dir"] == str(tmp_path)


def test_profiles_declare_job_type_service():
    import yaml

    profiles = sorted((IMAGE_DIR.parent / "profiles").glob("*.yaml"))
    assert profiles
    for path in profiles:
        profile = yaml.safe_load(path.read_text(encoding="utf-8"))
        assert profile["metadata"]["job_type"] == "service", path.name


def test_service_does_not_read_runtime_reserved_aim_model_id():
    source = (IMAGE_DIR / "src" / "service.py").read_text(encoding="utf-8")
    assert 'environ.get("AIM_MODEL_ID"' not in source
    assert "ESMC_HF_REPO" in source


def test_info_logging_goes_to_stderr(load_service, capsys):
    svc = load_service()
    handlers = [h for h in svc.LOGGER.handlers if getattr(h, "_esmc_stderr", False)]
    assert len(handlers) == 1
    svc._configure_logging(svc.LOGGER)  # idempotent
    assert len([h for h in svc.LOGGER.handlers if getattr(h, "_esmc_stderr", False)]) == 1
    assert svc.LOGGER.level == logging.INFO and svc.LOGGER.propagate is False
    handlers[0].stream = sys.stderr  # pytest swaps sys.stderr per test
    svc.LOGGER.info("pre-flight probe line")
    assert "pre-flight probe line" in capsys.readouterr().err


def test_fail_startup_docstring_does_not_claim_container_exit(load_service):
    doc = load_service()._fail_startup.__doc__ or ""
    assert "does NOT exit" in doc and "supervisor" in doc


def test_api_md_describes_worker_restart_not_container_stop():
    api = " ".join((IMAGE_DIR.parent / "templates/API.md").read_text(encoding="utf-8").split())
    assert "the service stops and the error names" not in api
    assert "The container does **not** exit" in api and "about every 10 seconds" in api
    assert "readiness probe" in api
    assert "1.5559" in api and "4.7396" in api and "2.8833" not in api


_VOCAB = {
    "<cls>": 0,
    "<pad>": 1,
    "<eos>": 2,
    "<unk>": 3,
    "L": 4,
    "A": 5,
    "K": 6,
    "M": 7,
    "T": 8,
    "X": 9,
    ".": 10,
    "-": 11,
}


def _tokenizing_service(load_service):
    svc = load_service()
    inst = svc.ESMCEmbeddingService.__new__(svc.ESMCEmbeddingService)
    inst._max_residues = 2046
    inst._tokenizer = MagicMock()
    inst._tokenizer.get_vocab.return_value = _VOCAB
    inst._residue_letters = frozenset(t for t in _VOCAB if len(t) == 1 and t.isalpha())
    return svc, inst


@pytest.mark.parametrize(
    "bad, listed", [("MKT1AA", "'1'"), ("MKT*AA", "'*'"), ("MK-TAA", "'-'"), ("MK1*2.3A", "'1', '*', '2', '.'")]
)
def test_tokenize_rejects_characters_outside_vocabulary(load_service, bad, listed):
    svc, inst = _tokenizing_service(load_service)
    with pytest.raises(svc.InvalidArgument, match="unsupported characters") as exc:
        inst._tokenize(bad)
    assert listed in str(exc.value)
    inst._tokenizer.assert_not_called()


def test_tokenize_lists_only_first_offenders(load_service):
    svc, inst = _tokenizing_service(load_service)
    with pytest.raises(svc.InvalidArgument) as exc:
        inst._tokenize("1234567")
    assert "'5'" in str(exc.value) and "'6'" not in str(exc.value) and "..." in str(exc.value)


def test_tokenize_accepts_vocabulary_letters_including_x(load_service):
    svc, inst = _tokenizing_service(load_service)
    inst._tokenize("mk t\nlXa")
    inst._tokenizer.assert_called_once_with("MKTLXA", return_tensors="pt")
