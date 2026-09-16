# Copyright © Advanced Micro Devices, Inc., or its affiliates.
#
# SPDX-License-Identifier: MIT

"""
Pydantic request/response models for the MONAI AIM service.

This file is fully shared — identical for all MONAI models.
The canonical AIM API uses a uniform response schema.
"""

from typing import Any

from pydantic import BaseModel


class PredictResponse(BaseModel):
    """Internal response from the _run_predict method.

    wholeBrainSeg_Large_UNEST_segmentation is a segmentation AIM and
    populates ``output_path`` with the saved 133-class label NIfTI;
    ``predictions`` stays None.
    """

    output_path: str | None = None
    predictions: dict[str, Any] | None = None
    device: str
    used_amp: bool


class InferenceResponse(BaseModel):
    """POST /v1/inference response — canonical AIM API."""

    model: str
    output: dict[str, Any]
    metadata: dict[str, Any]
