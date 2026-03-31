# Copyright © Advanced Micro Devices, Inc., or its affiliates.
#
# SPDX-License-Identifier: MIT

"""Pydantic models for AIM metadata.yaml validation.

These models use ConfigDict(extra="forbid") to enforce strict field checking,
ensuring no unexpected fields are present in metadata files.
"""

from __future__ import annotations

from typing import List, Literal, Optional, Union

from pydantic import BaseModel, ConfigDict, Field

from aim_common.object_model import GPUModel, Metric, Precision


class OciImage(BaseModel):
    model_config = ConfigDict(extra="forbid")

    vendor: str
    authors: str
    licenses: str
    description: str = Field(max_length=160)
    documentation: str
    source: str


class OciOpencontainers(BaseModel):
    model_config = ConfigDict(extra="forbid")

    image: OciImage


class OrgMetadata(BaseModel):
    model_config = ConfigDict(extra="forbid")

    opencontainers: OciOpencontainers


class AimRelease(BaseModel):
    model_config = ConfigDict(extra="forbid")

    notes: str


class AimDescription(BaseModel):
    model_config = ConfigDict(extra="forbid")

    full: str


class BaseAimSection(BaseModel):
    model_config = ConfigDict(extra="forbid")

    title: str
    release: Optional[AimRelease] = None
    description: Optional[AimDescription] = None


class BaseAmdSection(BaseModel):
    model_config = ConfigDict(extra="forbid")

    aim: BaseAimSection


class BaseComSection(BaseModel):
    model_config = ConfigDict(extra="forbid")

    amd: BaseAmdSection


class BaseMetadataModel(BaseModel):
    """Pydantic model for base metadata.yaml validation."""

    model_config = ConfigDict(extra="forbid")

    com: BaseComSection
    org: OrgMetadata


class RecommendedDeployment(BaseModel):
    model_config = ConfigDict(extra="forbid")

    gpuModel: Union[GPUModel, Literal["NONE"]]
    gpuCount: int = Field(ge=0, le=8)
    precision: Optional[Precision] = None
    metric: Optional[Metric] = None
    description: Optional[str] = None
    profileId: Optional[str] = None


class AimModel(BaseModel):
    model_config = ConfigDict(extra="forbid")

    canonicalName: str
    publisher: str
    source: Optional[str] = None
    tags: Optional[List[str]] = None
    variants: Optional[List[str]] = None
    recommendedDeployments: Optional[List[RecommendedDeployment]] = None


class HfToken(BaseModel):
    model_config = ConfigDict(extra="forbid")

    required: bool


class ModelAimSection(BaseModel):
    """com.amd.aim section for model metadata — extends base with model and hfToken."""

    model_config = ConfigDict(extra="forbid")

    title: str
    model: AimModel
    hfToken: Optional[HfToken] = None
    release: Optional[AimRelease] = None
    description: Optional[AimDescription] = None


class ModelAmdSection(BaseModel):
    model_config = ConfigDict(extra="forbid")

    aim: ModelAimSection


class ModelComSection(BaseModel):
    model_config = ConfigDict(extra="forbid")

    amd: ModelAmdSection


class ModelMetadataModel(BaseModel):
    """Pydantic model for model metadata.yaml validation."""

    model_config = ConfigDict(extra="forbid")

    com: ModelComSection
    org: OrgMetadata
