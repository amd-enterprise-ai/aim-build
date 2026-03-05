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
    BenchmarkFileValueResolver,
    ProfileTypeEvaluator,
)


@pytest.fixture
def benchmark_file(test_root: Path) -> Generator[Path, None, None]:
    """Create a temporary benchmark Excel file from CSV data."""
    csv_file = test_root / "aim_utils" / "benchmark_data.csv"

    with tempfile.NamedTemporaryFile(suffix=".xlsx", delete=False) as temp_file:
        xlsx_file = Path(temp_file.name)

        # Convert CSV to XLSX
        df = pd.read_csv(csv_file)
        df.to_excel(xlsx_file, index=False, sheet_name="benchmark_data")

        yield xlsx_file

        xlsx_file.unlink(missing_ok=True)


class TestProfileTypeEvaluator:
    """Test suite for ProfileTypeEvaluator class."""

    @patch("aim_utils.profile_utils.ProfileHandling")
    @patch("aim_utils.profile_utils.read_yaml")
    def test_is_optimized_general_profile(self, mock_read_yaml, mock_profile_handling):
        """General profiles cannot be optimized."""
        mock_read_yaml.return_value = {}
        mock_profile_handling_instance = MagicMock()
        mock_profile_handling_instance.is_general = True
        mock_profile_handling_instance.profile_name = "test-profile"
        mock_profile_handling.return_value = mock_profile_handling_instance

        mock_resolver = MagicMock()
        mock_resolver.get_gate_value.return_value = 0.95
        mock_resolver.get_type_value.return_value = None

        evaluator = ProfileTypeEvaluator(
            Path("test.yaml"), mock_resolver, preview_threshold=0.5, optimized_threshold=0.8
        )
        assert evaluator.is_optimized() is False

    @patch("aim_utils.profile_utils.ProfileHandling")
    @patch("aim_utils.profile_utils.read_yaml")
    def test_is_optimized_with_nim_ratio_above_threshold(self, mock_read_yaml, mock_profile_handling):
        """Profile is optimized when NIM ratio > 0.8."""
        mock_read_yaml.return_value = {}
        mock_profile_handling_instance = MagicMock()
        mock_profile_handling_instance.is_general = False
        mock_profile_handling_instance.profile_name = "test-profile"
        mock_profile_handling.return_value = mock_profile_handling_instance

        mock_resolver = MagicMock()
        mock_resolver.get_gate_value.return_value = 0.95
        mock_resolver.get_type_value.return_value = None

        evaluator = ProfileTypeEvaluator(
            Path("test.yaml"), mock_resolver, preview_threshold=0.5, optimized_threshold=0.8
        )
        assert evaluator.is_optimized() is True

    @patch("aim_utils.profile_utils.ProfileHandling")
    @patch("aim_utils.profile_utils.read_yaml")
    def test_is_optimized_with_nim_ratio_at_threshold(self, mock_read_yaml, mock_profile_handling):
        """Profile is not optimized at the 0.8 boundary."""
        mock_read_yaml.return_value = {}
        mock_profile_handling_instance = MagicMock()
        mock_profile_handling_instance.is_general = False
        mock_profile_handling_instance.profile_name = "test-profile"
        mock_profile_handling.return_value = mock_profile_handling_instance

        mock_resolver = MagicMock()
        mock_resolver.get_gate_value.return_value = 0.8
        mock_resolver.get_type_value.return_value = None

        evaluator = ProfileTypeEvaluator(
            Path("test.yaml"), mock_resolver, preview_threshold=0.5, optimized_threshold=0.8
        )
        assert evaluator.is_optimized() is False

    @patch("aim_utils.profile_utils.ProfileHandling")
    @patch("aim_utils.profile_utils.read_yaml")
    def test_is_optimized_no_ratios(self, mock_read_yaml, mock_profile_handling):
        """Profile without ratios is not optimized."""
        mock_read_yaml.return_value = {}
        mock_profile_handling_instance = MagicMock()
        mock_profile_handling_instance.is_general = False
        mock_profile_handling_instance.profile_name = "test-profile"
        mock_profile_handling.return_value = mock_profile_handling_instance

        mock_resolver = MagicMock()
        mock_resolver.get_gate_value.return_value = None
        mock_resolver.get_type_value.return_value = None

        evaluator = ProfileTypeEvaluator(
            Path("test.yaml"), mock_resolver, preview_threshold=0.5, optimized_threshold=0.8
        )
        assert evaluator.is_optimized() is False

    @patch("aim_utils.profile_utils.ProfileHandling")
    @patch("aim_utils.profile_utils.read_yaml")
    def test_is_preview_with_nim_ratio_in_range(self, mock_read_yaml, mock_profile_handling):
        """Profile is in preview when NIM ratio between 0.5 and 0.8."""
        mock_read_yaml.return_value = {}
        mock_profile_handling_instance = MagicMock()
        mock_profile_handling_instance.is_general = False
        mock_profile_handling_instance.profile_name = "test-profile"
        mock_profile_handling.return_value = mock_profile_handling_instance

        mock_resolver = MagicMock()
        mock_resolver.get_gate_value.return_value = 0.65
        mock_resolver.get_type_value.return_value = None

        evaluator = ProfileTypeEvaluator(
            Path("test.yaml"), mock_resolver, preview_threshold=0.5, optimized_threshold=0.8
        )
        assert evaluator.is_preview() is True

    @patch("aim_utils.profile_utils.ProfileHandling")
    @patch("aim_utils.profile_utils.read_yaml")
    def test_is_preview_with_nim_ratio_at_lower_bound(self, mock_read_yaml, mock_profile_handling):
        """Profile is in preview at 0.7 boundary."""
        mock_read_yaml.return_value = {}
        mock_profile_handling_instance = MagicMock()
        mock_profile_handling_instance.is_general = False
        mock_profile_handling_instance.profile_name = "test-profile"
        mock_profile_handling.return_value = mock_profile_handling_instance

        mock_resolver = MagicMock()
        mock_resolver.get_gate_value.return_value = 0.7
        mock_resolver.get_type_value.return_value = None

        evaluator = ProfileTypeEvaluator(
            Path("test.yaml"), mock_resolver, preview_threshold=0.5, optimized_threshold=0.8
        )
        assert evaluator.is_preview() is True

    @patch("aim_utils.profile_utils.ProfileHandling")
    @patch("aim_utils.profile_utils.read_yaml")
    def test_is_preview_with_nim_ratio_at_upper_bound(self, mock_read_yaml, mock_profile_handling):
        """Profile is in preview at 0.8 boundary."""
        mock_read_yaml.return_value = {}
        mock_profile_handling_instance = MagicMock()
        mock_profile_handling_instance.is_general = False
        mock_profile_handling_instance.profile_name = "test-profile"
        mock_profile_handling.return_value = mock_profile_handling_instance

        mock_resolver = MagicMock()
        mock_resolver.get_gate_value.return_value = 0.8
        mock_resolver.get_type_value.return_value = None

        evaluator = ProfileTypeEvaluator(
            Path("test.yaml"), mock_resolver, preview_threshold=0.5, optimized_threshold=0.8
        )
        assert evaluator.is_preview() is True

    @patch("aim_utils.profile_utils.ProfileHandling")
    @patch("aim_utils.profile_utils.read_yaml")
    def test_is_preview_no_ratios(self, mock_read_yaml, mock_profile_handling):
        """Profile without ratios is not in preview."""
        mock_read_yaml.return_value = {}
        mock_profile_handling_instance = MagicMock()
        mock_profile_handling_instance.is_general = False
        mock_profile_handling_instance.profile_name = "test-profile"
        mock_profile_handling.return_value = mock_profile_handling_instance

        mock_resolver = MagicMock()
        mock_resolver.get_gate_value.return_value = None
        mock_resolver.get_type_value.return_value = None

        evaluator = ProfileTypeEvaluator(
            Path("test.yaml"), mock_resolver, preview_threshold=0.5, optimized_threshold=0.8
        )
        assert evaluator.is_preview() is False

    @patch("aim_utils.profile_utils.ProfileHandling")
    @patch("aim_utils.profile_utils.read_yaml")
    def test_is_unoptimized_with_nim_ratio_below_threshold(self, mock_read_yaml, mock_profile_handling):
        """Profile is unoptimized when NIM ratio < 0.75."""
        mock_read_yaml.return_value = {}
        mock_profile_handling_instance = MagicMock()
        mock_profile_handling_instance.is_general = False
        mock_profile_handling_instance.profile_name = "test-profile"
        mock_profile_handling.return_value = mock_profile_handling_instance

        mock_resolver = MagicMock()
        mock_resolver.get_gate_value.return_value = 0.65
        mock_resolver.get_type_value.return_value = None

        evaluator = ProfileTypeEvaluator(
            Path("test.yaml"), mock_resolver, preview_threshold=0.75, optimized_threshold=0.8
        )
        assert evaluator.is_unoptimized() is True

    @patch("aim_utils.profile_utils.ProfileHandling")
    @patch("aim_utils.profile_utils.read_yaml")
    def test_is_unoptimized_with_nim_ratio_at_threshold(self, mock_read_yaml, mock_profile_handling):
        """Profile is not unoptimized at the 0.5 boundary."""
        mock_read_yaml.return_value = {}
        mock_profile_handling_instance = MagicMock()
        mock_profile_handling_instance.is_general = False
        mock_profile_handling_instance.profile_name = "test-profile"
        mock_profile_handling.return_value = mock_profile_handling_instance

        mock_resolver = MagicMock()
        mock_resolver.get_gate_value.return_value = 0.5
        mock_resolver.get_type_value.return_value = None

        evaluator = ProfileTypeEvaluator(
            Path("test.yaml"), mock_resolver, preview_threshold=0.5, optimized_threshold=0.8
        )
        assert evaluator.is_unoptimized() is False

    @patch("aim_utils.profile_utils.ProfileHandling")
    @patch("aim_utils.profile_utils.read_yaml")
    def test_is_unoptimized_no_ratios(self, mock_read_yaml, mock_profile_handling):
        """Profile without ratios is not unoptimized."""
        mock_read_yaml.return_value = {}
        mock_profile_handling_instance = MagicMock()
        mock_profile_handling_instance.is_general = False
        mock_profile_handling_instance.profile_name = "test-profile"
        mock_profile_handling.return_value = mock_profile_handling_instance

        mock_resolver = MagicMock()
        mock_resolver.get_gate_value.return_value = None
        mock_resolver.get_type_value.return_value = None

        evaluator = ProfileTypeEvaluator(
            Path("test.yaml"), mock_resolver, preview_threshold=0.5, optimized_threshold=0.8
        )
        assert evaluator.is_unoptimized() is False

    @patch("aim_utils.profile_utils.ProfileHandling")
    @patch("aim_utils.profile_utils.read_yaml")
    def test_get_profile_type_optimized(self, mock_read_yaml, mock_profile_handling):
        """Get profile type returns OPTIMIZED for high ratio."""
        mock_read_yaml.return_value = {}
        mock_profile_handling_instance = MagicMock()
        mock_profile_handling_instance.is_general = False
        mock_profile_handling_instance.profile_name = "test-profile"
        mock_profile_handling.return_value = mock_profile_handling_instance

        mock_resolver = MagicMock()
        mock_resolver.get_gate_value.return_value = 0.95
        mock_resolver.get_type_value.return_value = None

        evaluator = ProfileTypeEvaluator(
            Path("test.yaml"), mock_resolver, preview_threshold=0.5, optimized_threshold=0.8
        )
        assert evaluator._get_profile_type() == ProfileType.OPTIMIZED

    @patch("aim_utils.profile_utils.ProfileHandling")
    @patch("aim_utils.profile_utils.read_yaml")
    def test_get_profile_type_preview(self, mock_read_yaml, mock_profile_handling):
        """Get profile type returns PREVIEW for mid-range ratio."""
        mock_read_yaml.return_value = {}
        mock_profile_handling_instance = MagicMock()
        mock_profile_handling_instance.is_general = False
        mock_profile_handling_instance.profile_name = "test-profile"
        mock_profile_handling.return_value = mock_profile_handling_instance

        mock_resolver = MagicMock()
        mock_resolver.get_gate_value.return_value = 0.8
        mock_resolver.get_type_value.return_value = None

        evaluator = ProfileTypeEvaluator(
            Path("test.yaml"), mock_resolver, preview_threshold=0.5, optimized_threshold=0.8
        )
        assert evaluator._get_profile_type() == ProfileType.PREVIEW

    @patch("aim_utils.profile_utils.ProfileHandling")
    @patch("aim_utils.profile_utils.read_yaml")
    def test_get_profile_type_unoptimized(self, mock_read_yaml, mock_profile_handling):
        """Get profile type returns UNOPTIMIZED for low ratio."""
        mock_read_yaml.return_value = {}
        mock_profile_handling_instance = MagicMock()
        mock_profile_handling_instance.is_general = False
        mock_profile_handling_instance.profile_name = "test-profile"
        mock_profile_handling.return_value = mock_profile_handling_instance

        mock_resolver = MagicMock()
        mock_resolver.get_gate_value.return_value = 0.65
        mock_resolver.get_type_value.return_value = None

        evaluator = ProfileTypeEvaluator(
            Path("test.yaml"), mock_resolver, preview_threshold=0.75, optimized_threshold=0.8
        )
        assert evaluator._get_profile_type() == ProfileType.UNOPTIMIZED

    @patch("aim_utils.profile_utils.ProfileHandling")
    @patch("aim_utils.profile_utils.read_yaml")
    def test_get_profile_type_general(self, mock_read_yaml, mock_profile_handling):
        """Get profile type returns GENERAL for general profile."""
        mock_read_yaml.return_value = {}
        mock_profile_handling_instance = MagicMock()
        mock_profile_handling_instance.is_general = True
        mock_profile_handling_instance.profile_name = "test-profile"
        mock_profile_handling.return_value = mock_profile_handling_instance

        mock_resolver = MagicMock()
        mock_resolver.get_gate_value.return_value = None
        mock_resolver.get_type_value.return_value = None

        evaluator = ProfileTypeEvaluator(
            Path("test.yaml"), mock_resolver, preview_threshold=0.5, optimized_threshold=0.8
        )
        assert evaluator._get_profile_type() == ProfileType.GENERAL

    @patch("aim_utils.profile_utils.ProfileHandling")
    @patch("aim_utils.profile_utils.read_yaml")
    def test_get_profile_type_none(self, mock_read_yaml, mock_profile_handling):
        """Get profile type returns None when no metrics available."""
        mock_read_yaml.return_value = {}
        mock_profile_handling_instance = MagicMock()
        mock_profile_handling_instance.is_general = False
        mock_profile_handling_instance.profile_name = "test-profile"
        mock_profile_handling.return_value = mock_profile_handling_instance

        mock_resolver = MagicMock()
        mock_resolver.get_gate_value.return_value = None
        mock_resolver.get_type_value.return_value = None

        evaluator = ProfileTypeEvaluator(
            Path("test.yaml"), mock_resolver, preview_threshold=0.5, optimized_threshold=0.8
        )
        assert evaluator._get_profile_type() is None

    @patch("aim_utils.profile_utils.ProfileHandling")
    @patch("aim_utils.profile_utils.read_yaml")
    def test_evaluate_optimized_profile(self, mock_read_yaml, mock_profile_handling):
        """Evaluate returns correct result for optimized profile."""
        mock_read_yaml.return_value = {}
        mock_profile_handling_instance = MagicMock()
        mock_profile_handling_instance.is_general = False
        mock_profile_handling_instance.profile_name = "test-profile"
        mock_profile_handling.return_value = mock_profile_handling_instance

        mock_resolver = MagicMock()
        mock_resolver.get_gate_value.return_value = 0.95
        mock_resolver.get_type_value.return_value = None

        evaluator = ProfileTypeEvaluator(
            Path("test.yaml"), mock_resolver, preview_threshold=0.5, optimized_threshold=0.8
        )
        result = evaluator.evaluate()
        assert result.profile_type == ProfileType.OPTIMIZED
        assert result.manual_selection_only is False

    @patch("aim_utils.profile_utils.ProfileHandling")
    @patch("aim_utils.profile_utils.read_yaml")
    def test_evaluate_preview_profile(self, mock_read_yaml, mock_profile_handling):
        """Evaluate returns correct result for preview profile."""
        mock_read_yaml.return_value = {}
        mock_profile_handling_instance = MagicMock()
        mock_profile_handling_instance.is_general = False
        mock_profile_handling_instance.profile_name = "test-profile"
        mock_profile_handling.return_value = mock_profile_handling_instance

        mock_resolver = MagicMock()
        mock_resolver.get_gate_value.return_value = 0.8
        mock_resolver.get_type_value.return_value = None

        evaluator = ProfileTypeEvaluator(
            Path("test.yaml"), mock_resolver, preview_threshold=0.5, optimized_threshold=0.8
        )
        result = evaluator.evaluate()
        assert result.profile_type == ProfileType.PREVIEW
        assert result.manual_selection_only is False

    @patch("aim_utils.profile_utils.ProfileHandling")
    @patch("aim_utils.profile_utils.read_yaml")
    def test_evaluate_unoptimized_profile(self, mock_read_yaml, mock_profile_handling):
        """Evaluate returns correct result for unoptimized profile."""
        mock_read_yaml.return_value = {}
        mock_profile_handling_instance = MagicMock()
        mock_profile_handling_instance.is_general = False
        mock_profile_handling_instance.profile_name = "test-profile"
        mock_profile_handling.return_value = mock_profile_handling_instance

        mock_resolver = MagicMock()
        mock_resolver.get_gate_value.return_value = 0.65
        mock_resolver.get_type_value.return_value = None

        evaluator = ProfileTypeEvaluator(
            Path("test.yaml"), mock_resolver, preview_threshold=0.7, optimized_threshold=0.8
        )
        result = evaluator.evaluate()
        assert result.profile_type == ProfileType.UNOPTIMIZED
        assert result.manual_selection_only is True

    @patch("aim_utils.profile_utils.ProfileHandling")
    @patch("aim_utils.profile_utils.read_yaml")
    def test_evaluate_general_profile(self, mock_read_yaml, mock_profile_handling):
        """Evaluate returns correct result for general profile."""
        mock_read_yaml.return_value = {}
        mock_profile_handling_instance = MagicMock()
        mock_profile_handling_instance.is_general = True
        mock_profile_handling_instance.profile_name = "test-profile"
        mock_profile_handling.return_value = mock_profile_handling_instance

        mock_resolver = MagicMock()
        mock_resolver.get_gate_value.return_value = None
        mock_resolver.get_type_value.return_value = None

        evaluator = ProfileTypeEvaluator(
            Path("test.yaml"), mock_resolver, preview_threshold=0.5, optimized_threshold=0.8
        )
        result = evaluator.evaluate()
        assert result.profile_type == ProfileType.GENERAL
        assert result.manual_selection_only is False

    @patch("aim_utils.profile_utils.ProfileHandling")
    @patch("aim_utils.profile_utils.read_yaml")
    def test_evaluate_no_profile_type(self, mock_read_yaml, mock_profile_handling):
        """Evaluate handles case when profile type is None."""
        mock_read_yaml.return_value = {}
        mock_profile_handling_instance = MagicMock()
        mock_profile_handling_instance.is_general = False
        mock_profile_handling_instance.profile_name = "test-profile"
        mock_profile_handling.return_value = mock_profile_handling_instance

        mock_resolver = MagicMock()
        mock_resolver.get_gate_value.return_value = None
        mock_resolver.get_type_value.return_value = None

        evaluator = ProfileTypeEvaluator(
            Path("test.yaml"), mock_resolver, preview_threshold=0.5, optimized_threshold=0.8
        )
        result = evaluator.evaluate()
        assert result.profile_type is None
        assert result.manual_selection_only is True

    @patch("aim_utils.profile_utils.ProfileHandling")
    @patch("aim_utils.profile_utils.read_yaml")
    def test_evaluate_with_manually_specified_type(self, mock_read_yaml, mock_profile_handling):
        """Evaluate uses manually specified type when it differs from calculated type."""
        mock_read_yaml.return_value = {}
        mock_profile_handling_instance = MagicMock()
        mock_profile_handling_instance.is_general = False
        mock_profile_handling_instance.profile_name = "test-profile"
        mock_profile_handling.return_value = mock_profile_handling_instance

        mock_resolver = MagicMock()
        mock_resolver.get_gate_value.return_value = 0.95  # Would be OPTIMIZED
        mock_resolver.get_type_value.return_value = "preview"  # Manually set to PREVIEW

        evaluator = ProfileTypeEvaluator(
            Path("test.yaml"), mock_resolver, preview_threshold=0.5, optimized_threshold=0.8
        )
        result = evaluator.evaluate()
        # Manually specified type takes precedence
        assert result.profile_type == ProfileType.PREVIEW
        assert result.manual_selection_only is False


class TestBenchmarkFileValueResolver:
    """Test suite for BenchmarkFileValueResolver class."""

    def test_benchmark_file_value_resolver_initialization(self, benchmark_file: Path):
        """BenchmarkFileValueResolver initializes correctly with Excel file."""
        resolver = BenchmarkFileValueResolver(benchmark_file)
        assert resolver.df is not None
        assert len(resolver.df) > 0
        assert "aim" in resolver.df.columns
        assert "profile" in resolver.df.columns
        assert "gate value" in resolver.df.columns

    def test_get_gate_value_found(self, benchmark_file: Path):
        """Get gate value returns correct value when found."""
        resolver = BenchmarkFileValueResolver(benchmark_file)
        value = resolver.get_gate_value("vllm-mi300x-fp16-tp1-latency", "aim-qwen-qwen3-32b")
        assert value is not None
        assert isinstance(value, float)
        assert value == 0.95

    def test_get_gate_value_specific_aim_profile(self, benchmark_file: Path):
        """Get gate value retrieves value for specific AIM profile."""
        resolver = BenchmarkFileValueResolver(benchmark_file)
        value = resolver.get_gate_value("vllm-mi300x-fp16-tp1-latency", "aim-meta-llama-llama-3-1-8b-instruct")
        assert value is not None
        assert value == 0.88

    def test_get_gate_value_not_found(self, benchmark_file: Path):
        """Get gate value returns None when not found."""
        resolver = BenchmarkFileValueResolver(benchmark_file)
        value = resolver.get_gate_value("nonexistent-profile", "nonexistent-model")
        assert value is None

    def test_get_gate_value_multiple_aims(self, benchmark_file: Path):
        """Get gate value correctly handles multiple AIM images."""
        resolver = BenchmarkFileValueResolver(benchmark_file)

        # Test aim-meta-llama-llama-3-1-8b-instruct profile
        value1 = resolver.get_gate_value("vllm-mi300x-fp16-tp1-latency", "aim-meta-llama-llama-3-1-8b-instruct")

        # Test aim-meta-llama-llama-3-2-1b-instruct profile
        value2 = resolver.get_gate_value("vllm-mi300x-fp16-tp1-latency", "aim-meta-llama-llama-3-2-1b-instruct")

        assert value1 is not None
        assert value2 is not None
        assert value1 != value2
        assert value1 == 0.88
        assert value2 == 0.65

    def test_get_gate_value_different_profiles_same_aim(self, benchmark_file: Path):
        """Get gate value retrieves different values for different profiles of same AIM."""
        resolver = BenchmarkFileValueResolver(benchmark_file)

        value_fp16 = resolver.get_gate_value("vllm-mi300x-fp16-tp1-latency", "aim-meta-llama-llama-3-1-8b-instruct")
        value_fp8 = resolver.get_gate_value("vllm-mi300x-fp8-tp1-latency", "aim-meta-llama-llama-3-1-8b-instruct")

        assert value_fp16 is not None
        assert value_fp8 is not None
        assert value_fp16 == 0.88
        assert value_fp8 == 0.75

    def test_get_type_value_found(self, benchmark_file: Path):
        """Get type value returns correct value when found."""
        resolver = BenchmarkFileValueResolver(benchmark_file)
        value = resolver.get_type_value("vllm-mi300x-fp16-tp1-latency", "aim-meta-llama-llama-3-1-8b-instruct")
        assert value is not None
        assert isinstance(value, str)

    def test_get_type_value_not_found(self, benchmark_file: Path):
        """Get type value returns None when not found."""
        resolver = BenchmarkFileValueResolver(benchmark_file)
        value = resolver.get_type_value("nonexistent-profile", "nonexistent-model")
        assert value is None
