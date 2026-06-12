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
    AcceleratorModel,
    AcceleratorType,
    Engine,
    Metric,
    Precision,
    ProfileCapabilities,
    ProfileMetadata,
    ProfileType,
)


class TestProfileMetadata:
    """Test ProfileMetadata dataclass functionality."""

    def test_profile_str_representation(self):
        """Test that str(ProfileMetadata) returns the accelerator_label."""
        profile = ProfileMetadata(
            engine=Engine.VLLM,
            accelerator_model=AcceleratorModel.MI300X,
            precision=Precision.FP16,
            accelerator_count=1,
            metric=Metric.THROUGHPUT,
            manual_selection_only=False,
            type=ProfileType.GENERAL,
        )
        assert str(profile) == "vllm-mi300x-fp16-tp1-throughput"

    def test_accelerator_label_property(self):
        """Test that accelerator_label property returns the same as str()."""
        profile = ProfileMetadata(
            engine=Engine.VLLM,
            accelerator_model=AcceleratorModel.MI325X,
            precision=Precision.FP8,
            accelerator_count=2,
            metric=Metric.LATENCY,
            manual_selection_only=False,
            type=ProfileType.GENERAL,
        )
        assert profile.accelerator_label == str(profile)
        assert profile.accelerator_label == "vllm-mi325x-fp8-tp2-latency"

    def test_accelerator_label_none_accelerator(self):
        """Test accelerator_label when accelerator is None (e.g. CPU-only profile)."""
        profile = ProfileMetadata(
            engine=Engine.VLLM,
            accelerator_model=None,
            precision=Precision.BF16,
            accelerator_count=0,
            metric=Metric.LATENCY,
            manual_selection_only=False,
            type=ProfileType.GENERAL,
        )
        assert profile.accelerator_label == "vllm-none-bf16-tp0-latency"

    def test_profile_to_dict(self):
        """Test ProfileMetadata serialization to dictionary."""
        profile = ProfileMetadata(
            engine=Engine.VLLM,
            accelerator_type=AcceleratorType.GPU,
            accelerator_model=AcceleratorModel.MI300X,
            precision=Precision.FP16,
            accelerator_count=1,
            metric=Metric.THROUGHPUT,
            manual_selection_only=False,
            type=ProfileType.GENERAL,
        )
        result = profile.to_dict()
        assert result == {
            "engine": "vllm",
            "accelerator_type": "gpu",
            "accelerator_model": "MI300X",
            "precision": "fp16",
            "accelerator_count": 1,
            "metric": "throughput",
            "manual_selection_only": False,
            "type": "general",
        }

    def test_profile_from_dict(self):
        """Test ProfileMetadata deserialization from dictionary."""
        data = {
            "engine": "vllm",
            "accelerator_model": "mi325x",
            "precision": "fp8",
            "accelerator_count": 2,
            "metric": "latency",
            "manual_selection_only": False,
            "type": "general",
        }
        profile = ProfileMetadata.from_dict(data)
        assert profile.engine == Engine.VLLM
        assert profile.accelerator_model == AcceleratorModel.MI325X
        assert profile.precision == Precision.FP8
        assert profile.accelerator_count == 2
        assert profile.metric == Metric.LATENCY
        assert profile.manual_selection_only is False
        assert profile.type == ProfileType.GENERAL

    def test_profile_to_dict_includes_capabilities_when_any_enabled(self):
        """Test that capabilities is serialized only when at least one flag is enabled."""
        profile = ProfileMetadata(
            engine=Engine.VLLM,
            accelerator_model=AcceleratorModel.MI300X,
            precision=Precision.FP16,
            accelerator_count=1,
            metric=Metric.THROUGHPUT,
            manual_selection_only=False,
            type=ProfileType.GENERAL,
            capabilities=ProfileCapabilities(reasoning=True),
        )

        dumped = profile.to_dict()
        assert dumped["capabilities"] == {
            "tool_calling": False,
            "structured_outputs": False,
            "reasoning": True,
        }

    def test_profile_from_dict_with_old_field_names(self):
        """Test that old YAML field names (gpu, gpu_count) still parse correctly."""
        data = {
            "engine": "vllm",
            "gpu": "MI300X",
            "precision": "fp16",
            "gpu_count": 1,
            "metric": "throughput",
            "manual_selection_only": False,
            "type": "general",
        }
        profile = ProfileMetadata.from_dict(data)
        assert profile.accelerator_model == AcceleratorModel.MI300X
        assert profile.accelerator_count == 1

    def test_profile_from_dict_with_none_sentinel(self):
        """Test that legacy 'NONE' sentinel value in YAML parses as Python None."""
        data = {
            "engine": "vllm",
            "gpu": "NONE",
            "precision": "bf16",
            "gpu_count": 0,
            "metric": "latency",
            "manual_selection_only": False,
            "type": "general",
        }
        profile = ProfileMetadata.from_dict(data)
        assert profile.accelerator_model is None

    def test_profile_equality(self):
        """Test that Profiles with same values are equal."""
        profile1 = ProfileMetadata(
            engine=Engine.VLLM,
            accelerator_model=AcceleratorModel.MI300X,
            precision=Precision.FP16,
            accelerator_count=1,
            metric=Metric.THROUGHPUT,
            manual_selection_only=False,
            type=ProfileType.GENERAL,
        )
        profile2 = ProfileMetadata(
            engine=Engine.VLLM,
            accelerator_model=AcceleratorModel.MI300X,
            precision=Precision.FP16,
            accelerator_count=1,
            metric=Metric.THROUGHPUT,
            manual_selection_only=False,
            type=ProfileType.GENERAL,
        )
        assert profile1 == profile2

    def test_profile_inequality(self):
        """Test that Profiles with different values are not equal."""
        profile1 = ProfileMetadata(
            engine=Engine.VLLM,
            accelerator_model=AcceleratorModel.MI300X,
            precision=Precision.FP16,
            accelerator_count=1,
            metric=Metric.THROUGHPUT,
            manual_selection_only=False,
            type=ProfileType.GENERAL,
        )
        profile2 = ProfileMetadata(
            engine=Engine.VLLM,
            accelerator_model=AcceleratorModel.MI300X,
            precision=Precision.FP16,
            accelerator_count=2,
            metric=Metric.THROUGHPUT,
            manual_selection_only=False,
            type=ProfileType.GENERAL,
        )
        assert profile1 != profile2
