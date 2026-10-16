# Copyright © Advanced Micro Devices, Inc., or its affiliates.
#
# SPDX-License-Identifier: MIT

"""BentoML service for DINOv2 image embeddings."""

import bentoml
from bentoml.exceptions import InvalidArgument
from inference import Dinov2Predictor
from pydantic import BaseModel, Field, model_validator


class EmbeddingRequest(BaseModel):
    """Request for one image embedding."""

    image_path: str | None = Field(
        default=None, description="Path visible inside the container, e.g. /workspace/data/inputs/x.jpg"
    )
    image_base64: str | None = Field(default=None, description="Base64-encoded image bytes")
    include_timings: bool = Field(
        default=False,
        description="Return server-side preprocessing and GPU inference timings",
    )

    @model_validator(mode="after")
    def exactly_one_image_source(self) -> "EmbeddingRequest":
        if (self.image_path is None) == (self.image_base64 is None):
            raise ValueError("provide exactly one of image_path or image_base64")
        return self


class EmbeddingResponse(BaseModel):
    """DINOv2 embedding response."""

    model: str
    source: str
    embedding_dim: int
    embedding: list[float]
    norm: float
    timings: dict[str, float] | None = None


class BentoArgs(BaseModel):
    """Template arguments AIM forwards via ``bentoml serve --arg``.

    ``aim-runtime`` resolves ``AIM_PORT`` and injects it as ``--arg port=…``, so this
    is the only source of the bound port. Deliberately no default: a second default
    here could silently drift from ``AIM_PORT``.
    """

    port: int


args = bentoml.use_arguments(BentoArgs)


@bentoml.service(
    name="dinov2-vitl14-embedding-inference",
    http={"port": args.port},
    traffic={"timeout": 300},
)
class Dinov2Vitl14Service:
    """Served DINOv2 ViT-L/14 embedding model."""

    def __init__(self) -> None:
        self._predictor = Dinov2Predictor()
        self._predictor.warmup()

    @bentoml.api
    def predict(self, request: EmbeddingRequest) -> EmbeddingResponse:
        try:
            result = self._predictor.predict_request(request.model_dump())
        except (ValueError, OSError) as exc:
            raise InvalidArgument(str(exc)) from exc
        return EmbeddingResponse(**result)
