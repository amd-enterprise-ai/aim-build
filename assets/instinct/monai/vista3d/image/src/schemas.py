# Copyright © Advanced Micro Devices, Inc., or its affiliates.
#
# SPDX-License-Identifier: MIT

"""
Pydantic request/response models for the MONAI VISTA-3D AIM service.

VISTA-3D is a segmentation AIM — its output is a NIfTI mask written to disk,
so ``PredictResponse.output_path`` is the populated field. ``predictions``
is kept for parity with the broader template (classification AIMs use that
slot) but is always ``None`` for VISTA-3D.
"""

from typing import Any

from pydantic import BaseModel


class PredictResponse(BaseModel):
    """Internal response from the _run_predict method."""

    output_path: str | None = None
    predictions: dict[str, Any] | None = None
    device: str
    used_amp: bool


class InferenceResponse(BaseModel):
    """POST /v1/inference response — canonical AIM API."""

    model: str
    output: dict[str, Any]
    metadata: dict[str, Any]
