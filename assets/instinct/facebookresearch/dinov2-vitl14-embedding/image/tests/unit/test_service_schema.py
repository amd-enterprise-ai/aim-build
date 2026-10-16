# Copyright © Advanced Micro Devices, Inc., or its affiliates.
#
# SPDX-License-Identifier: MIT

import importlib.util
import sys
from pathlib import Path

import dinov2_inference
import pytest
from pydantic import ValidationError

bentoml = pytest.importorskip("bentoml")
# Another AIM's tests may leave a bare stub module registered as bentoml in the shared session.
if getattr(bentoml, "__file__", None) is None:
    pytest.skip("bentoml is a stub module, not the installed package", allow_module_level=True)

SERVICE_PATH = Path(__file__).resolve().parents[2] / "src" / "service.py"


def _load_service():
    name = "dinov2_service"
    spec = importlib.util.spec_from_file_location(name, SERVICE_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    previous = sys.modules.get("inference")
    sys.modules["inference"] = dinov2_inference
    try:
        spec.loader.exec_module(module)
    finally:
        if previous is None:
            sys.modules.pop("inference", None)
        else:
            sys.modules["inference"] = previous
    return module


service = _load_service()
EmbeddingRequest = service.EmbeddingRequest
EmbeddingResponse = service.EmbeddingResponse


def test_embedding_request_accepts_one_image_source():
    request = EmbeddingRequest(
        image_path="/workspace/data/inputs/000000000139.jpg",
        include_timings=True,
    )
    assert request.image_path.endswith(".jpg")
    assert request.image_base64 is None
    assert request.include_timings is True


def test_embedding_request_rejects_missing_or_duplicate_sources():
    with pytest.raises(ValidationError):
        EmbeddingRequest()
    with pytest.raises(ValidationError):
        EmbeddingRequest(image_path="x.jpg", image_base64="abcd")


def test_embedding_response_shape():
    response = EmbeddingResponse(
        model="dinov2_vitl14",
        source="unit",
        embedding_dim=2,
        embedding=[0.1, 0.2],
        norm=0.22,
        timings={"preprocessing_ms": 1.5, "gpu_inference_ms": 2.5},
    )
    assert response.embedding_dim == 2
    assert response.timings == {"preprocessing_ms": 1.5, "gpu_inference_ms": 2.5}
