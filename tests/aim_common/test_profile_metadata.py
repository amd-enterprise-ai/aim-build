# Copyright © Advanced Micro Devices, Inc., or its affiliates.
#
# SPDX-License-Identifier: MIT

"""Tests for the ProfileMetadata dataclass."""

import logging
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
            accelerator_type=AcceleratorType.GPU,
            engine=Engine.VLLM,
            accelerator_model=AcceleratorModel.MI300X,
            precision=Precision.FP16,
            accelerator_count=1,
            metric=Metric.THROUGHPUT,
            type=ProfileType.GENERAL,
        )
        assert str(profile) == "vllm-mi300x-fp16-tp1-throughput"

    def test_accelerator_label_property(self):
        """Test that accelerator_label property returns the same as str()."""
        profile = ProfileMetadata(
            accelerator_type=AcceleratorType.GPU,
            engine=Engine.VLLM,
            accelerator_model=AcceleratorModel.MI325X,
            precision=Precision.FP8,
            accelerator_count=2,
            metric=Metric.LATENCY,
            type=ProfileType.GENERAL,
        )
        assert profile.accelerator_label == str(profile)
        assert profile.accelerator_label == "vllm-mi325x-fp8-tp2-latency"

    def test_accelerator_label_cpu_profile(self):
        """CPU-only profiles name their accelerator explicitly rather than omitting it."""
        profile = ProfileMetadata(
            accelerator_type=AcceleratorType.CPU,
            engine=Engine.VLLM,
            accelerator_model=AcceleratorModel.CPU,
            precision=Precision.BF16,
            accelerator_count=124,
            metric=Metric.LATENCY,
            type=ProfileType.GENERAL,
        )
        # CPU profiles collapse to tp1 regardless of the core count they request.
        assert profile.accelerator_label == "vllm-cpu-bf16-tp1-latency"

    def test_accelerator_model_none_rejected(self):
        """accelerator_model is required: None is no longer an accepted value."""
        with pytest.raises(ValidationError):
            ProfileMetadata(
                accelerator_type=AcceleratorType.GPU,
                engine=Engine.VLLM,
                accelerator_model=None,
                precision=Precision.BF16,
                accelerator_count=1,
                metric=Metric.LATENCY,
                type=ProfileType.GENERAL,
            )

    def test_accelerator_count_zero_rejected(self):
        """A profile must claim at least one accelerator; zero is not a deployment target."""
        with pytest.raises(ValidationError) as exc_info:
            ProfileMetadata(
                accelerator_type=AcceleratorType.GPU,
                engine=Engine.VLLM,
                accelerator_model=AcceleratorModel.MI300X,
                precision=Precision.BF16,
                accelerator_count=0,
                metric=Metric.LATENCY,
                type=ProfileType.GENERAL,
            )
        assert "accelerator_count" in str(exc_info.value)

    def test_profile_to_dict(self):
        """Test ProfileMetadata serialization to dictionary."""
        profile = ProfileMetadata(
            engine=Engine.VLLM,
            accelerator_type=AcceleratorType.GPU,
            accelerator_model=AcceleratorModel.MI300X,
            precision=Precision.FP16,
            accelerator_count=1,
            metric=Metric.THROUGHPUT,
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
            "type": "general",
        }

    def test_profile_from_dict(self):
        """Test ProfileMetadata deserialization from dictionary."""
        data = {
            "engine": "vllm",
            "accelerator_type": "gpu",
            "accelerator_model": "mi325x",
            "precision": "fp8",
            "accelerator_count": 2,
            "metric": "latency",
            "type": "general",
        }
        profile = ProfileMetadata.from_dict(data)
        assert profile.engine == Engine.VLLM
        assert profile.accelerator_model == AcceleratorModel.MI325X
        assert profile.precision == Precision.FP8
        assert profile.accelerator_count == 2
        assert profile.metric == Metric.LATENCY
        assert profile.type == ProfileType.GENERAL

    def test_profile_to_dict_includes_capabilities_when_any_enabled(self):
        """Test that capabilities is serialized only when at least one flag is enabled."""
        profile = ProfileMetadata(
            accelerator_type=AcceleratorType.GPU,
            engine=Engine.VLLM,
            accelerator_model=AcceleratorModel.MI300X,
            precision=Precision.FP16,
            accelerator_count=1,
            metric=Metric.THROUGHPUT,
            type=ProfileType.GENERAL,
            capabilities=ProfileCapabilities(reasoning=True),
        )

        dumped = profile.to_dict()
        assert dumped["capabilities"] == {
            "tool_calling": False,
            "structured_outputs": False,
            "reasoning": True,
        }

    def test_profile_from_dict_accepts_old_field_names_with_warning(self, caplog):
        """Deprecated accelerator fields remain compatible during the warning period."""
        data = {
            "engine": "vllm",
            "gpu": "MI300X",
            "precision": "fp16",
            "gpu_count": 1,
            "metric": "throughput",
            "type": "general",
        }
        with caplog.at_level(logging.WARNING):
            profile = ProfileMetadata.from_dict(data, source="profiles/legacy.yaml")

        assert profile.accelerator_model == AcceleratorModel.MI300X
        assert profile.accelerator_count == 1
        assert profile.accelerator_type == AcceleratorType.GPU
        assert "'gpu' (use 'accelerator_model')" in caplog.text
        assert "'gpu_count' (use 'accelerator_count')" in caplog.text
        assert "profiles/legacy.yaml" in caplog.text
        assert "will stop working in a future release" in caplog.text

    def test_profile_from_dict_accepts_unknown_field_with_warning(self, caplog):
        """Unknown metadata keys are accepted temporarily but produce a migration warning."""
        data = {
            "engine": "vllm",
            "accelerator_type": "gpu",
            "accelerator_model": "MI300X",
            "precision": "fp16",
            "accelerator_count": 1,
            "metric": "throughput",
            "type": "general",
            "tensor_parallel_size": 1,
        }
        with caplog.at_level(logging.WARNING):
            profile = ProfileMetadata.from_dict(data)

        assert profile.tensor_parallel_size == 1
        assert "'tensor_parallel_size' (unsupported)" in caplog.text
        assert "will stop working in a future release" in caplog.text

    def test_profile_from_dict_accepts_retired_field_with_warning(self, caplog):
        """Retired metadata remains loadable but no longer affects runtime behavior."""
        data = {
            "engine": "vllm",
            "accelerator_type": "gpu",
            "accelerator_model": "MI300X",
            "precision": "fp16",
            "accelerator_count": 1,
            "metric": "throughput",
            "type": "general",
            "manual_selection_only": True,
        }
        with caplog.at_level(logging.WARNING):
            profile = ProfileMetadata.from_dict(data)

        assert profile.manual_selection_only is True
        assert "'manual_selection_only' (retired and ignored)" in caplog.text
        assert "will stop working in a future release" in caplog.text

    def test_canonical_fields_win_over_deprecated_fields(self):
        """Dual-spelled metadata uses canonical values while retaining compatibility."""
        data = {
            "engine": "vllm",
            "accelerator_type": "gpu",
            "accelerator_model": "MI325X",
            "gpu": "MI300X",
            "precision": "fp16",
            "accelerator_count": 2,
            "gpu_count": 1,
            "metric": "throughput",
            "type": "general",
        }
        profile = ProfileMetadata.from_dict(data)
        assert profile.accelerator_model == AcceleratorModel.MI325X
        assert profile.accelerator_count == 2

    def test_profile_from_dict_rejects_none_sentinel(self):
        """The legacy 'NONE' sentinel is no longer accepted now that the field is required."""
        data = {
            "engine": "vllm",
            "accelerator_type": "gpu",
            "accelerator_model": "NONE",
            "precision": "bf16",
            "accelerator_count": 1,
            "metric": "latency",
            "type": "general",
        }
        with pytest.raises(ValidationError):
            ProfileMetadata.from_dict(data)

    def test_profile_equality(self):
        """Test that Profiles with same values are equal."""
        profile1 = ProfileMetadata(
            accelerator_type=AcceleratorType.GPU,
            engine=Engine.VLLM,
            accelerator_model=AcceleratorModel.MI300X,
            precision=Precision.FP16,
            accelerator_count=1,
            metric=Metric.THROUGHPUT,
            type=ProfileType.GENERAL,
        )
        profile2 = ProfileMetadata(
            accelerator_type=AcceleratorType.GPU,
            engine=Engine.VLLM,
            accelerator_model=AcceleratorModel.MI300X,
            precision=Precision.FP16,
            accelerator_count=1,
            metric=Metric.THROUGHPUT,
            type=ProfileType.GENERAL,
        )
        assert profile1 == profile2

    def test_profile_inequality(self):
        """Test that Profiles with different values are not equal."""
        profile1 = ProfileMetadata(
            accelerator_type=AcceleratorType.GPU,
            engine=Engine.VLLM,
            accelerator_model=AcceleratorModel.MI300X,
            precision=Precision.FP16,
            accelerator_count=1,
            metric=Metric.THROUGHPUT,
            type=ProfileType.GENERAL,
        )
        profile2 = ProfileMetadata(
            accelerator_type=AcceleratorType.GPU,
            engine=Engine.VLLM,
            accelerator_model=AcceleratorModel.MI300X,
            precision=Precision.FP16,
            accelerator_count=2,
            metric=Metric.THROUGHPUT,
            type=ProfileType.GENERAL,
        )
        assert profile1 != profile2


class TestProfileMetadataVariant:
    """Tests for the variant field on ProfileMetadata."""

    def _base_profile(self, **kwargs) -> ProfileMetadata:
        defaults = dict(
            engine=Engine.VLLM,
            accelerator_type=AcceleratorType.GPU,
            accelerator_model=AcceleratorModel.MI300X,
            precision=Precision.FP16,
            accelerator_count=1,
            metric=Metric.LATENCY,
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
            "accelerator_type": "gpu",
            "accelerator_model": "MI300X",
            "precision": "fp16",
            "accelerator_count": 1,
            "metric": "latency",
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
            "accelerator_type": "gpu",
            "accelerator_model": "MI300X",
            "precision": "fp16",
            "accelerator_count": 1,
            "metric": "latency",
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
