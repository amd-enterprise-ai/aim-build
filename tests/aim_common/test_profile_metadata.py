# Copyright © Advanced Micro Devices, Inc., or its affiliates.
#
# SPDX-License-Identifier: MIT

"""Tests for the ProfileMetadata dataclass."""

import sys
from pathlib import Path

# Add ci and src directories to path
ci_dir = Path(__file__).parent.parent.parent / "ci"
src_dir = Path(__file__).parent.parent.parent / "src"
sys.path.insert(0, str(ci_dir))
sys.path.insert(0, str(src_dir))

from aim_common.object_model import (  # noqa: E402
    Engine,
    GPUModel,
    Metric,
    Precision,
    ProfileMetadata,
    ProfileType,
)


class TestProfileMetadata:
    """Test ProfileMetadata dataclass functionality."""

    def test_profile_str_representation(self):
        """Test that ProfileMetadata generates correct string representation."""
        profile = ProfileMetadata(
            engine=Engine.VLLM,
            gpu=GPUModel.MI300X,
            precision=Precision.FP16,
            gpu_count=1,
            metric=Metric.THROUGHPUT,
            manual_selection_only=False,
            type=ProfileType.GENERAL,
        )
        assert str(profile) == "vllm-mi300x-fp16-tp1-throughput"

    def test_profile_id_property(self):
        """Test that profile_id property returns the same as str()."""
        profile = ProfileMetadata(
            engine=Engine.VLLM,
            gpu=GPUModel.MI325X,
            precision=Precision.FP8,
            gpu_count=2,
            metric=Metric.LATENCY,
            manual_selection_only=False,
            type=ProfileType.GENERAL,
        )
        assert profile.profile_id == str(profile)
        assert profile.profile_id == "vllm-mi325x-fp8-tp2-latency"

    def test_profile_to_dict(self):
        """Test ProfileMetadata serialization to dictionary."""
        profile = ProfileMetadata(
            engine=Engine.VLLM,
            gpu=GPUModel.MI300X,
            precision=Precision.FP16,
            gpu_count=1,
            metric=Metric.THROUGHPUT,
            manual_selection_only=False,
            type=ProfileType.GENERAL,
        )
        result = profile.to_dict()
        assert result == {
            "engine": "vllm",
            "gpu": "MI300X",
            "precision": "fp16",
            "gpu_count": 1,
            "metric": "throughput",
            "manual_selection_only": False,
            "type": "general",
        }

    def test_profile_from_dict(self):
        """Test ProfileMetadata deserialization from dictionary."""
        data = {"engine": "vllm", "gpu": "mi325x", "precision": "fp8", "gpu_count": 2, "metric": "latency"}
        profile = ProfileMetadata.from_dict(data)
        assert profile.engine == Engine.VLLM
        assert profile.gpu == GPUModel.MI325X
        assert profile.precision == Precision.FP8
        assert profile.gpu_count == 2
        assert profile.metric == Metric.LATENCY

    def test_profile_equality(self):
        """Test that Profiles with same values are equal."""
        profile1 = ProfileMetadata(
            engine=Engine.VLLM,
            gpu=GPUModel.MI300X,
            precision=Precision.FP16,
            gpu_count=1,
            metric=Metric.THROUGHPUT,
            manual_selection_only=False,
            type=ProfileType.GENERAL,
        )
        profile2 = ProfileMetadata(
            engine=Engine.VLLM,
            gpu=GPUModel.MI300X,
            precision=Precision.FP16,
            gpu_count=1,
            metric=Metric.THROUGHPUT,
            manual_selection_only=False,
            type=ProfileType.GENERAL,
        )
        assert profile1 == profile2

    def test_profile_inequality(self):
        """Test that Profiles with different values are not equal."""
        profile1 = ProfileMetadata(
            engine=Engine.VLLM,
            gpu=GPUModel.MI300X,
            precision=Precision.FP16,
            gpu_count=1,
            metric=Metric.THROUGHPUT,
            manual_selection_only=False,
            type=ProfileType.GENERAL,
        )
        profile2 = ProfileMetadata(
            engine=Engine.VLLM,
            gpu=GPUModel.MI300X,
            precision=Precision.FP16,
            gpu_count=2,
            metric=Metric.THROUGHPUT,
            manual_selection_only=False,
            type=ProfileType.GENERAL,
        )
        assert profile1 != profile2
