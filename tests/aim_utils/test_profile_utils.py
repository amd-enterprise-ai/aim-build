# Copyright © Advanced Micro Devices, Inc., or its affiliates.
#
# SPDX-License-Identifier: MIT

"""Tests for profile utilities."""

import tempfile
from pathlib import Path
from typing import Generator
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

from aim_common.object_model import ProfileType
from aim_utils.profile_utils import (
    ProfileFileValueResolver,
    ProfileTypeEvaluator,
)


@pytest.fixture
def profile_data_file(test_root: Path) -> Generator[Path, None, None]:
    """Create a temporary Excel file from CSV data."""
    csv_file = test_root / "aim_utils" / "profile_data.csv"

    with tempfile.NamedTemporaryFile(suffix=".xlsx", delete=False) as temp_file:
        xlsx_file = Path(temp_file.name)

        df = pd.read_csv(csv_file)
        df.to_excel(xlsx_file, index=False, sheet_name="profile_data")

        yield xlsx_file

        xlsx_file.unlink(missing_ok=True)


class TestProfileTypeEvaluator:
    """Test suite for ProfileTypeEvaluator class."""

    def _make_evaluator(self, is_general: bool, type_value):
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

            return ProfileTypeEvaluator(Path("test.yaml"), mock_resolver)

    @patch("aim_utils.profile_utils.ProfileHandling")
    @patch("aim_utils.profile_utils.read_yaml")
    def test_evaluate_optimized_profile(self, mock_read_yaml, mock_profile_handling):
        """Evaluate returns OPTIMIZED when type column is 'optimized'."""
        mock_read_yaml.return_value = {}
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
        mock_read_yaml.return_value = {}
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
        mock_read_yaml.return_value = {}
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
        mock_read_yaml.return_value = {}
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
        mock_read_yaml.return_value = {}
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

    @pytest.mark.xfail(reason="EAI-1965: openpyxl _NoValueType leaks into DataFrame on some CI environments")
    def test_get_type_value_found(self, profile_data_file: Path):
        """Get type value returns correct value when found."""
        resolver = ProfileFileValueResolver(profile_data_file)
        value = resolver.get_type_value("vllm-mi300x-fp16-tp1-latency", "aim-model-b")
        assert value == "preview"

    @pytest.mark.xfail(reason="EAI-1965: openpyxl _NoValueType leaks into DataFrame on some CI environments")
    def test_get_type_value_optimized(self, profile_data_file: Path):
        """Get type value returns 'optimized' for an optimized entry."""
        resolver = ProfileFileValueResolver(profile_data_file)
        value = resolver.get_type_value("vllm-mi300x-fp16-tp1-latency", "aim-model-a")
        assert value == "optimized"

    @pytest.mark.xfail(reason="EAI-1965: openpyxl _NoValueType leaks into DataFrame on some CI environments")
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
