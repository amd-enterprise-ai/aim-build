# Copyright © Advanced Micro Devices, Inc., or its affiliates.
#
# SPDX-License-Identifier: MIT

"""Tests for profile utilities."""

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from aim_common.object_model import AcceleratorModel, Engine, Metric, Precision, ProfileMetadata, ProfileType
from aim_utils.profile_utils import (
    FileNameFormatMismatch,
    ProfileFileValueResolver,
    ProfileTypeEvaluator,
    _get_metadata_profile_id_mismatches,
)


@pytest.fixture
def profile_data_file(test_root: Path) -> Path:
    """Return the path to the committed Excel test data file."""
    return test_root / "aim_utils" / "profile_data.xlsx"


class TestProfileTypeEvaluator:
    """Test suite for ProfileTypeEvaluator class."""

    def _make_evaluator(self, is_general: bool, type_value, profile_path: Path = Path("test.yaml")):
        with (
            patch("aim_utils.profile_utils.read_yaml") as mock_read_yaml,
            patch("aim_utils.profile_utils.ProfileHandling") as mock_profile_handling,
        ):
            mock_read_yaml.return_value = {}
            instance = MagicMock()
            instance.is_general = is_general
            instance.profile_name = "test-profile"
            mock_profile_handling.return_value = instance

            mock_resolver = MagicMock()
            mock_resolver.get_type_value.return_value = type_value

            return ProfileTypeEvaluator(profile_path, mock_resolver)

    @pytest.mark.parametrize(
        ("profile_path", "expected_image_name"),
        [
            (Path("assets/instinct/base/profiles/general/test-profile.yaml"), "aim-instinct-base"),
            (Path("assets/radeon/base/profiles/general/test-profile.yaml"), "aim-radeon-base"),
            (Path("assets/epyc/base/profiles/general/test-profile.yaml"), "aim-epyc-base"),
        ],
        ids=["instinct", "radeon", "epyc"],
    )
    def test_get_image_name_uses_expected_general_repository_name(self, profile_path, expected_image_name):
        """General profiles should follow the expected per-accelerator base repository naming."""
        evaluator = self._make_evaluator(is_general=True, type_value=None, profile_path=profile_path)

        assert evaluator._get_image_name() == expected_image_name

    @pytest.mark.parametrize(
        ("profile_path", "expected_image_name"),
        [
            (
                Path("assets/instinct/meta-llama/Llama-3.1-8B-Instruct/profiles/test-profile.yaml"),
                "aim-instinct-meta-llama-llama-3-1-8b-instruct",
            ),
            (
                Path("assets/radeon/meta-llama/Llama-3.1-8B-Instruct/profiles/test-profile.yaml"),
                "aim-radeon-meta-llama-llama-3-1-8b-instruct",
            ),
            (
                Path("assets/epyc/meta-llama/Llama-3.1-8B-Instruct/profiles/test-profile.yaml"),
                "aim-epyc-meta-llama-llama-3-1-8b-instruct",
            ),
        ],
        ids=["instinct", "radeon", "epyc"],
    )
    def test_get_image_name_uses_expected_model_repository_prefix(self, profile_path, expected_image_name):
        """Model-specific profiles should follow the expected per-accelerator repository naming."""

        with (
            patch("aim_utils.profile_utils.read_yaml") as mock_read_yaml,
            patch("aim_utils.profile_utils.ProfileHandling") as mock_profile_handling,
        ):
            mock_read_yaml.return_value = {"aim_id": "meta-llama/Llama-3.1-8B-Instruct"}
            instance = MagicMock()
            instance.is_general = False
            instance.profile_name = "test-profile"
            mock_profile_handling.return_value = instance

            mock_resolver = MagicMock()
            mock_resolver.get_type_value.return_value = None

            evaluator = ProfileTypeEvaluator(profile_path, mock_resolver)

        assert evaluator._get_image_name() == expected_image_name

    @patch("aim_utils.profile_utils.ProfileHandling")
    @patch("aim_utils.profile_utils.read_yaml")
    def test_evaluate_optimized_profile(self, mock_read_yaml, mock_profile_handling):
        """Evaluate returns OPTIMIZED when type column is 'optimized'."""
        mock_read_yaml.return_value = {"aim_id": "meta-llama/Llama-3.1-8B-Instruct"}
        instance = MagicMock()
        instance.is_general = False
        instance.profile_name = "test-profile"
        mock_profile_handling.return_value = instance

        mock_resolver = MagicMock()
        mock_resolver.get_type_value.return_value = "optimized"

        evaluator = ProfileTypeEvaluator(Path("test.yaml"), mock_resolver)
        result = evaluator.evaluate()
        assert result.profile_type == ProfileType.OPTIMIZED
        assert result.manual_selection_only is False

    @patch("aim_utils.profile_utils.ProfileHandling")
    @patch("aim_utils.profile_utils.read_yaml")
    def test_evaluate_preview_profile(self, mock_read_yaml, mock_profile_handling):
        """Evaluate returns PREVIEW when type column is 'preview'."""
        mock_read_yaml.return_value = {"aim_id": "meta-llama/Llama-3.1-8B-Instruct"}
        instance = MagicMock()
        instance.is_general = False
        instance.profile_name = "test-profile"
        mock_profile_handling.return_value = instance

        mock_resolver = MagicMock()
        mock_resolver.get_type_value.return_value = "preview"

        evaluator = ProfileTypeEvaluator(Path("test.yaml"), mock_resolver)
        result = evaluator.evaluate()
        assert result.profile_type == ProfileType.PREVIEW
        assert result.manual_selection_only is False

    @patch("aim_utils.profile_utils.ProfileHandling")
    @patch("aim_utils.profile_utils.read_yaml")
    def test_evaluate_unoptimized_profile(self, mock_read_yaml, mock_profile_handling):
        """Evaluate returns UNOPTIMIZED when type column is 'unoptimized'."""
        mock_read_yaml.return_value = {"aim_id": "meta-llama/Llama-3.1-8B-Instruct"}
        instance = MagicMock()
        instance.is_general = False
        instance.profile_name = "test-profile"
        mock_profile_handling.return_value = instance

        mock_resolver = MagicMock()
        mock_resolver.get_type_value.return_value = "unoptimized"

        evaluator = ProfileTypeEvaluator(Path("test.yaml"), mock_resolver)
        result = evaluator.evaluate()
        assert result.profile_type == ProfileType.UNOPTIMIZED
        assert result.manual_selection_only is True

    @patch("aim_utils.profile_utils.ProfileHandling")
    @patch("aim_utils.profile_utils.read_yaml")
    def test_evaluate_no_type_value(self, mock_read_yaml, mock_profile_handling):
        """Evaluate returns None profile type when no type value is present."""
        mock_read_yaml.return_value = {"aim_id": "meta-llama/Llama-3.1-8B-Instruct"}
        instance = MagicMock()
        instance.is_general = False
        instance.profile_name = "test-profile"
        mock_profile_handling.return_value = instance

        mock_resolver = MagicMock()
        mock_resolver.get_type_value.return_value = None

        evaluator = ProfileTypeEvaluator(Path("test.yaml"), mock_resolver)
        result = evaluator.evaluate()
        assert result.profile_type is None
        assert result.manual_selection_only is True

    @patch("aim_utils.profile_utils.ProfileHandling")
    @patch("aim_utils.profile_utils.read_yaml")
    def test_evaluate_invalid_type_raises(self, mock_read_yaml, mock_profile_handling):
        """Evaluate raises ValueError for an unrecognised type string."""
        mock_read_yaml.return_value = {"aim_id": "meta-llama/Llama-3.1-8B-Instruct"}
        instance = MagicMock()
        instance.is_general = False
        instance.profile_name = "test-profile"
        mock_profile_handling.return_value = instance

        mock_resolver = MagicMock()
        mock_resolver.get_type_value.return_value = "not-a-real-type"

        with pytest.raises(ValueError):
            ProfileTypeEvaluator(Path("test.yaml"), mock_resolver)


class TestProfileFileValueResolver:
    """Test suite for ProfileFileValueResolver class."""

    def test_profile_data_file_value_resolver_initialization(self, profile_data_file: Path):
        """ProfileFileValueResolver initializes correctly with Excel file."""
        resolver = ProfileFileValueResolver(profile_data_file)
        assert resolver.df is not None
        assert len(resolver.df) > 0
        assert "aim" in resolver.df.columns
        assert "profile" in resolver.df.columns
        assert "type" in resolver.df.columns

    def test_get_type_value_found(self, profile_data_file: Path):
        """Get type value returns correct value when found."""
        resolver = ProfileFileValueResolver(profile_data_file)
        value = resolver.get_type_value("vllm-mi300x-fp16-tp1-latency", "aim-model-b")
        assert value == "preview"

    def test_get_type_value_optimized(self, profile_data_file: Path):
        """Get type value returns 'optimized' for an optimized entry."""
        resolver = ProfileFileValueResolver(profile_data_file)
        value = resolver.get_type_value("vllm-mi300x-fp16-tp1-latency", "aim-model-a")
        assert value == "optimized"

    def test_get_type_value_unoptimized(self, profile_data_file: Path):
        """Get type value returns 'unoptimized' for an unoptimized entry."""
        resolver = ProfileFileValueResolver(profile_data_file)
        value = resolver.get_type_value("vllm-mi300x-fp8-tp1-latency", "aim-model-c")
        assert value == "unoptimized"

    def test_get_type_value_not_found(self, profile_data_file: Path):
        """Get type value returns None when profile/aim combination not found."""
        resolver = ProfileFileValueResolver(profile_data_file)
        value = resolver.get_type_value("nonexistent-profile", "nonexistent-model")
        assert value is None


def _make_metadata(**overrides) -> ProfileMetadata:
    """Build a valid ProfileMetadata with sensible defaults, accepting field overrides."""
    defaults = dict(
        engine=Engine.VLLM,
        accelerator_model=AcceleratorModel.MI300X,
        precision=Precision.FP8,
        accelerator_count=1,
        metric=Metric.LATENCY,
        manual_selection_only=False,
        type=ProfileType.OPTIMIZED,
    )
    defaults.update(overrides)
    return ProfileMetadata.model_validate(defaults)


class TestGetMetadataProfileIdMismatches:
    """Tests for _get_metadata_profile_id_mismatches."""

    PROFILE_PATH = Path("assets/instinct/meta-llama/Llama-3.1-8B-Instruct/profiles/vllm-mi300x-fp8-tp1-latency.yaml")

    def test_no_mismatches_when_profile_id_matches_metadata(self):
        """Returns empty list when profile filename exactly matches all metadata fields."""
        metadata = _make_metadata()
        result = _get_metadata_profile_id_mismatches(self.PROFILE_PATH, metadata, "vllm-mi300x-fp8-tp1-latency")
        assert result == []

    def test_engine_mismatch_detected(self):
        """Returns a MetadataMismatch when engine in filename differs from metadata."""
        metadata = _make_metadata(engine=Engine.BENTOML)
        result = _get_metadata_profile_id_mismatches(self.PROFILE_PATH, metadata, "vllm-mi300x-fp8-tp1-latency")
        names = [m.field_name for m in result]
        assert "engine" in names
        engine_mismatch = next(m for m in result if m.field_name == "engine")
        assert engine_mismatch.expected_value == "vllm"
        assert engine_mismatch.actual_value == "bentoml"

    def test_precision_mismatch_detected(self):
        """Returns a MetadataMismatch when precision in filename differs from metadata."""
        metadata = _make_metadata(precision=Precision.FP16)
        result = _get_metadata_profile_id_mismatches(self.PROFILE_PATH, metadata, "vllm-mi300x-fp8-tp1-latency")
        names = [m.field_name for m in result]
        assert "precision" in names

    def test_tp_mismatch_detected(self):
        """Returns a MetadataMismatch when tensor-parallel count in filename differs from metadata."""
        metadata = _make_metadata(accelerator_count=8)
        result = _get_metadata_profile_id_mismatches(self.PROFILE_PATH, metadata, "vllm-mi300x-fp8-tp1-latency")
        names = [m.field_name for m in result]
        assert "accelerator_count" in names
        tp_mismatch = next(m for m in result if m.field_name == "accelerator_count")
        assert tp_mismatch.expected_value == "tp1"
        assert tp_mismatch.actual_value == "tp8"

    def test_metric_mismatch_detected(self):
        """Returns a MetadataMismatch when metric in filename differs from metadata."""
        metadata = _make_metadata(metric=Metric.THROUGHPUT)
        result = _get_metadata_profile_id_mismatches(self.PROFILE_PATH, metadata, "vllm-mi300x-fp8-tp1-latency")
        names = [m.field_name for m in result]
        assert "metric" in names

    def test_multiple_mismatches_returned(self):
        """Returns one MetadataMismatch entry per mismatched field."""
        metadata = _make_metadata(engine=Engine.BENTOML, precision=Precision.FP16)
        result = _get_metadata_profile_id_mismatches(self.PROFILE_PATH, metadata, "vllm-mi300x-fp8-tp1-latency")
        mismatch_fields = {m.field_name for m in result}
        assert {"engine", "precision"}.issubset(mismatch_fields)

    def test_mxfp4_normalised_to_fp4(self):
        """mxfp4 in the profile ID is normalised to fp4 before comparison."""
        metadata = _make_metadata(precision=Precision.FP4)
        result = _get_metadata_profile_id_mismatches(self.PROFILE_PATH, metadata, "vllm-mi300x-mxfp4-tp1-latency")
        assert result == []

    def test_invalid_profile_id_format_returns_file_name_format_mismatch(self):
        """A profile ID that cannot be split into 5 parts returns a FileNameFormatMismatch."""
        metadata = _make_metadata()
        result = _get_metadata_profile_id_mismatches(self.PROFILE_PATH, metadata, "bad-profile-id")
        assert len(result) == 1
        assert isinstance(result[0], FileNameFormatMismatch)

    def test_file_name_format_mismatch_has_mismatch_true(self):
        """FileNameFormatMismatch.has_mismatch is True when error message is non-empty."""
        metadata = _make_metadata()
        result = _get_metadata_profile_id_mismatches(self.PROFILE_PATH, metadata, "only-three-parts")
        assert result[0].has_mismatch is True

    def test_metadata_mismatch_profile_path_is_preserved(self):
        """MetadataMismatch entries carry the original profile path."""
        metadata = _make_metadata(engine=Engine.BENTOML)
        result = _get_metadata_profile_id_mismatches(self.PROFILE_PATH, metadata, "vllm-mi300x-fp8-tp1-latency")
        assert all(m.profile_path == self.PROFILE_PATH for m in result)

    def test_variant_suffix_matches_metadata(self):
        """A variant-suffixed filename matches when metadata.variant equals the suffix."""
        metadata = _make_metadata(variant="inductor-diff")
        result = _get_metadata_profile_id_mismatches(
            self.PROFILE_PATH, metadata, "vllm-mi300x-fp8-tp1-latency-inductor-diff"
        )
        assert result == []

    def test_variant_in_filename_but_not_metadata_detected(self):
        """A variant suffix in the filename with no metadata.variant is a mismatch."""
        metadata = _make_metadata()  # variant defaults to None
        result = _get_metadata_profile_id_mismatches(
            self.PROFILE_PATH, metadata, "vllm-mi300x-fp8-tp1-latency-shortseqs"
        )
        variant_mismatch = next(m for m in result if m.field_name == "variant")
        assert variant_mismatch.expected_value == "shortseqs"
        assert variant_mismatch.actual_value is None

    def test_variant_in_metadata_but_not_filename_detected(self):
        """metadata.variant set while the filename omits the suffix is a mismatch."""
        metadata = _make_metadata(variant="shortseqs")
        result = _get_metadata_profile_id_mismatches(self.PROFILE_PATH, metadata, "vllm-mi300x-fp8-tp1-latency")
        variant_mismatch = next(m for m in result if m.field_name == "variant")
        assert variant_mismatch.expected_value is None
        assert variant_mismatch.actual_value == "shortseqs"

    def test_no_variant_in_filename_or_metadata_no_mismatch(self):
        """A five-segment filename with variant=None metadata reports no variant mismatch."""
        metadata = _make_metadata()
        result = _get_metadata_profile_id_mismatches(self.PROFILE_PATH, metadata, "vllm-mi300x-fp8-tp1-latency")
        assert "variant" not in {m.field_name for m in result}
