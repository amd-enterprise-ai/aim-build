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

import pytest  # noqa: E402
from pydantic import ValidationError  # noqa: E402

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


class TestProfileMetadataVariant:
    """Tests for the variant field on ProfileMetadata."""

    def _base_profile(self, **kwargs) -> ProfileMetadata:
        defaults = dict(
            engine=Engine.VLLM,
            accelerator_model=AcceleratorModel.MI300X,
            precision=Precision.FP16,
            accelerator_count=1,
            metric=Metric.LATENCY,
            manual_selection_only=False,
            type=ProfileType.OPTIMIZED,
        )
        defaults.update(kwargs)
        return ProfileMetadata(**defaults)

    def test_accelerator_label_without_variant_unchanged(self):
        """Existing profiles that omit variant must produce the same five-segment label."""
        profile = self._base_profile()
        assert profile.accelerator_label == "vllm-mi300x-fp16-tp1-latency"

    def test_accelerator_label_with_variant_appends_suffix(self):
        """When variant is set, accelerator_label gains a sixth segment."""
        profile = self._base_profile(variant="inductor-diff")
        assert profile.accelerator_label == "vllm-mi300x-fp16-tp1-latency-inductor-diff"

    def test_str_uses_accelerator_label_with_variant(self):
        """str() should reflect the variant-extended accelerator_label."""
        profile = self._base_profile(variant="short")
        assert str(profile) == "vllm-mi300x-fp16-tp1-latency-short"

    def test_hash_differs_between_variant_and_no_variant(self):
        """Two profiles that differ only by variant must hash differently."""
        p1 = self._base_profile(variant=None)
        p2 = self._base_profile(variant="inductor-diff")
        assert hash(p1) != hash(p2)

    def test_hash_equal_for_same_variant(self):
        """Two profiles with the same variant must hash the same."""
        p1 = self._base_profile(variant="inductor-diff")
        p2 = self._base_profile(variant="inductor-diff")
        assert hash(p1) == hash(p2)

    def test_from_dict_parses_variant(self):
        """ProfileMetadata.from_dict should accept and parse the variant key."""
        data = {
            "engine": "vllm",
            "accelerator_model": "MI300X",
            "precision": "fp16",
            "accelerator_count": 1,
            "metric": "latency",
            "manual_selection_only": False,
            "type": "optimized",
            "variant": "inductor-diff",
        }
        profile = ProfileMetadata.from_dict(data)
        assert profile.variant == "inductor-diff"
        assert profile.accelerator_label == "vllm-mi300x-fp16-tp1-latency-inductor-diff"

    def test_from_dict_without_variant_defaults_to_none(self):
        """YAML profiles without variant key must parse to variant=None."""
        data = {
            "engine": "vllm",
            "accelerator_model": "MI300X",
            "precision": "fp16",
            "accelerator_count": 1,
            "metric": "latency",
            "manual_selection_only": False,
            "type": "optimized",
        }
        profile = ProfileMetadata.from_dict(data)
        assert profile.variant is None

    def test_to_dict_excludes_variant_when_none(self):
        """to_dict() should not emit a variant key when variant is None (preserve YAML round-trip)."""
        profile = self._base_profile(variant=None)
        result = profile.to_dict()
        assert "variant" not in result

    def test_to_dict_includes_variant_when_set(self):
        """to_dict() should include variant when it is not None."""
        profile = self._base_profile(variant="inductor-diff")
        result = profile.to_dict()
        assert result["variant"] == "inductor-diff"

    @pytest.mark.parametrize(
        "bad_variant",
        [
            "",  # empty string
            " ",  # whitespace
            "UPPER",  # uppercase
            "Mixed-Case",  # mixed case
            "-leading-hyphen",  # leading hyphen
            "1leading-digit",  # leading digit
            "has space",  # whitespace inside
            "under_score",  # underscores not in slug pattern
        ],
    )
    def test_invalid_variant_raises_validation_error(self, bad_variant):
        """Invalid variant slugs (empty, whitespace, uppercase, etc.) must raise ValidationError."""
        with pytest.raises(ValidationError):
            self._base_profile(variant=bad_variant)

    def test_valid_variant_accepted(self):
        """A correctly formatted slug variant should validate without error."""
        profile = self._base_profile(variant="inductor-diff")
        assert profile.variant == "inductor-diff"

    def test_none_variant_accepted(self):
        """variant=None should always pass validation (pattern not applied to None)."""
        profile = self._base_profile(variant=None)
        assert profile.variant is None
