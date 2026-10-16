# Copyright © Advanced Micro Devices, Inc., or its affiliates.
#
# SPDX-License-Identifier: MIT

import json

import dinov2_harness as harness_module
import pytest
from dinov2_harness import Dinov2Vitl14Harness

from aim_runtime.harness import ModelHarness


def test_harness_subclasses_model_harness():
    harness = Dinov2Vitl14Harness()

    assert isinstance(harness, ModelHarness)
    assert harness.ENGINE == "bentoml"


def test_post_predict_wraps_payload_for_bentoml(monkeypatch):
    captured = {}

    class Response:
        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

        def read(self):
            return b'{"embedding": []}'

    def fake_urlopen(request, timeout):
        captured["body"] = json.loads(request.data.decode("utf-8"))
        captured["timeout"] = timeout
        return Response()

    monkeypatch.setattr(harness_module, "urlopen", fake_urlopen)

    Dinov2Vitl14Harness()._post_predict(
        "http://localhost:8000", {"image_path": "/workspace/data/x.jpg"}, timeout_seconds=45
    )

    assert captured["body"] == {"request": {"image_path": "/workspace/data/x.jpg"}}
    assert captured["timeout"] == 45


def test_runtime_optimization_env_defaults_and_overrides(monkeypatch):
    monkeypatch.delenv("DINOV2_PRECISION", raising=False)
    monkeypatch.delenv("DINOV2_TORCH_COMPILE", raising=False)
    monkeypatch.delenv("DINOV2_GPU_PREPROCESS", raising=False)
    default_harness = Dinov2Vitl14Harness()

    assert default_harness.precision == "fp16"
    assert default_harness.torch_compile is True
    assert default_harness.gpu_preprocess is True
    assert default_harness.compile_mode == "default"

    monkeypatch.setenv("DINOV2_PRECISION", "fp16")
    monkeypatch.setenv("DINOV2_TORCH_COMPILE", "false")
    monkeypatch.setenv("DINOV2_GPU_PREPROCESS", "false")
    monkeypatch.setenv("DINOV2_COMPILE_MODE", "reduce-overhead")
    tuned_harness = Dinov2Vitl14Harness()

    assert tuned_harness.precision == "fp16"
    assert tuned_harness.torch_compile is False
    assert tuned_harness.gpu_preprocess is False
    assert tuned_harness.compile_mode == "reduce-overhead"


def test_rejects_unsupported_precision(monkeypatch):
    monkeypatch.setenv("DINOV2_PRECISION", "bf16")

    with pytest.raises(ValueError, match="DINOV2_PRECISION"):
        Dinov2Vitl14Harness()
