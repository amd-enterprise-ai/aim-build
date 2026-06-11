# Copyright © Advanced Micro Devices, Inc., or its affiliates.
#
# SPDX-License-Identifier: MIT

"""Tests for profile utilities."""

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from aim_common.object_model import ProfileType
from aim_utils.profile_utils import (
    ProfileFileValueResolver,
    ProfileTypeEvaluator,
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
