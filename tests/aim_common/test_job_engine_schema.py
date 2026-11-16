# Copyright © Advanced Micro Devices, Inc., or its affiliates.
#
# SPDX-License-Identifier: MIT

"""Tests for the job engine and the profile-level ``job_type`` field."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from aim_common.metadata_models import ModelAimSection
from aim_common.object_model import (
    AcceleratorModel,
    AcceleratorType,
    Engine,
    JobType,
    Metric,
    Precision,
    ProfileMetadata,
    ProfileType,
)
from aim_runtime.config import AIMConfig
from aim_runtime.engine_config import EngineConfig
from aim_runtime.engines import ENGINE_CLASSES, JobEngine, engine_class_for

_PROFILE = dict(
    accelerator_type=AcceleratorType.GPU,
    engine=Engine.JOB,
    accelerator_model=AcceleratorModel.MI300X,
    precision=Precision.FP32,
    accelerator_count=1,
    metric=Metric.THROUGHPUT,
    type=ProfileType.GENERAL,
)


class TestJobType:
    def test_job_type_defaults_to_service(self):
        assert ProfileMetadata(**_PROFILE).job_type == JobType.SERVICE

    @pytest.mark.parametrize("value", ["long_running", "service"])
    def test_job_type_accepted(self, value):
        assert ProfileMetadata(**_PROFILE, job_type=value).job_type == JobType(value)

    def test_job_type_unknown_value_rejected(self):
        with pytest.raises(ValidationError):
            ProfileMetadata(**_PROFILE, job_type="batch")

    def test_job_type_round_trips_through_to_dict(self):
        assert "job_type" not in ProfileMetadata(**_PROFILE).to_dict()
        data = ProfileMetadata(**_PROFILE, job_type="long_running").to_dict()
        assert data["job_type"] == "long_running"
        assert ProfileMetadata.from_dict(data).job_type == JobType.LONG_RUNNING

    def test_job_type_is_not_a_metadata_yaml_field(self):
        """The job_type lives in the profile YAML, not in metadata.yaml."""
        assert "job_type" not in ModelAimSection.model_fields
        data = {"title": "Job", "model": {"canonicalName": "org/job", "publisher": "Org"}, "job_type": "long_running"}
        with pytest.raises(ValidationError):
            ModelAimSection.model_validate(data)


class TestJobEngine:
    def test_job_engine_value(self):
        assert Engine.JOB == "job"

    def test_job_engine_mapped(self):
        assert ENGINE_CLASSES[Engine.JOB] is JobEngine
        assert engine_class_for(Engine.JOB) is JobEngine

    def test_serve_raises_not_implemented(self):
        engine = JobEngine(
            AIMConfig(aim_id="org/job", engine=Engine.JOB),
            EngineConfig(engine=Engine.JOB, launch="x", model_arg="--model"),
        )
        with pytest.raises(NotImplementedError, match="long-running job"):
            engine.serve(engine.config, engine.engine_config)

    def test_profile_with_job_engine_validates(self):
        profile = ProfileMetadata(
            accelerator_type=AcceleratorType.GPU,
            engine=Engine.JOB,
            accelerator_model=AcceleratorModel.MI300X,
            precision=Precision.FP32,
            accelerator_count=2,
            metric=Metric.THROUGHPUT,
            type=ProfileType.GENERAL,
        )
        assert str(profile) == "job-mi300x-fp32-tp2-throughput"
