# Copyright © Advanced Micro Devices, Inc., or its affiliates.
#
# SPDX-License-Identifier: MIT

from copy import deepcopy
from pathlib import Path
from unittest.mock import Mock, patch

import pytest

from aim_common import Engine, GPUModel, Metric, Precision, ProfileType
from aim_runtime.config import AIMConfig
from aim_runtime.object_model import Profile
from aim_runtime.profile_selector import ProfileCompatibilityState, ProfileNotFound, ProfileSelector


@pytest.fixture
def selector_with_mock_gpu(aim_config: AIMConfig) -> ProfileSelector:
    """Create a ProfileSelector with mocked GPU detection to match test profiles."""
    with patch("aim_runtime.profile_selector.GPUDetector") as mock_detector:
        # Mock GPU detector to return values that match our test profiles
        mock_instance = Mock()
        mock_instance.all_gpus_idle = True
        mock_instance.has_gpus = True
        mock_instance.gpu_models = [GPUModel.MI300X]
        mock_instance.gpu_count = 1
        mock_detector.return_value = mock_instance

        selector = ProfileSelector(aim_config)
        return selector


@pytest.fixture
def selector_no_gpu(aim_config: AIMConfig) -> ProfileSelector:
    """Create a ProfileSelector with no GPU detected."""
    with patch("aim_runtime.profile_selector.GPUDetector") as mock_detector:
        mock_instance = Mock()
        mock_instance.all_gpus_idle = True
        mock_instance.has_gpus = False
        mock_instance.gpu_models = []
        mock_instance.gpu_count = 0
        mock_detector.return_value = mock_instance

        selector = ProfileSelector(aim_config)
        return selector


def test_profile_selector_initialization(aim_config: AIMConfig, selector_with_mock_gpu: ProfileSelector) -> None:
    """Test that ProfileSelector initializes correctly."""
    assert selector_with_mock_gpu.config == aim_config
    assert selector_with_mock_gpu.detected_gpu == GPUModel.MI300X
    assert selector_with_mock_gpu.detected_gpu_count == 1
    assert selector_with_mock_gpu.profile_validator is not None


def test_profile_selector_no_gpu_initialization(aim_config: AIMConfig, selector_no_gpu: ProfileSelector) -> None:
    """Test that ProfileSelector handles no GPU correctly."""
    assert selector_no_gpu.detected_gpu is None
    assert selector_no_gpu.detected_gpu_count == 0


def test_build_search_paths_model_specific(
    selector_with_mock_gpu: ProfileSelector,
) -> None:
    """Test that search paths are built correctly for model-specific profiles."""
    search_paths = selector_with_mock_gpu._build_search_paths()

    # Should contain custom, model-specific, and general paths
    assert len(search_paths) >= 2
    assert any("meta-llama/Llama-3.1-8B-Instruct" in path for path in search_paths)
    assert any("general" in path for path in search_paths)


def test_build_search_paths_with_custom_path(custom_profiles_path: str) -> None:
    """Test that custom path is included with proper precedence by convention."""
    config = AIMConfig(
        aim_id="meta-llama/Llama-3.1-8B-Instruct",
        profile_base_path=custom_profiles_path,
    )

    with patch("aim_runtime.profile_selector.GPUDetector"):
        selector = ProfileSelector(config)
        search_paths = selector._build_search_paths()

        # Custom path should come first (by convention at <base_path>/custom)
        assert search_paths[0].endswith("/custom")
        # Model-specific path should be second
        assert "meta-llama/Llama-3.1-8B-Instruct" in search_paths[1]
        # General path should be last
        assert search_paths[2].endswith("/general")


def test_find_profile_success_model_specific(
    selector_with_mock_gpu: ProfileSelector,
) -> None:
    """Test successful profile finding for model-specific case."""
    profile = selector_with_mock_gpu.find_profile()

    assert isinstance(profile, Profile)
    assert Path(profile.profile_handling.path).exists()


def test_find_profile_with_profile_id(aim_config: AIMConfig) -> None:
    """Test finding profile by specific profile ID."""
    # First get a valid profile ID
    with patch("aim_runtime.profile_selector.GPUDetector"):
        selector = ProfileSelector(aim_config)
        registry = selector.registry
        valid_profiles = registry.profiles

        if valid_profiles:
            # Test with valid profile ID
            profile_id = valid_profiles[0].profile_id
            config_with_id = AIMConfig(
                aim_id="meta-llama/Llama-3.1-8B-Instruct",
                profile_base_path=aim_config.profile_base_path,
                profile_id=profile_id,
            )
            selector_with_id = ProfileSelector(config_with_id)
            profile = selector_with_id.find_profile()
            assert isinstance(profile, Profile)
            assert profile.profile_handling.path.endswith(".yaml")


def test_find_profile_with_invalid_profile_id(assets_instinct_path: str) -> None:
    """Test that invalid profile ID raises ProfileNotFound."""
    config = AIMConfig(
        aim_id="meta-llama/Llama-3.1-8B-Instruct",
        profile_base_path=assets_instinct_path,
        profile_id="invalid-profile-id",
    )

    with patch("aim_runtime.profile_selector.GPUDetector"):
        selector = ProfileSelector(config)

        with pytest.raises(ProfileNotFound) as exc_info:
            selector.find_profile()

        assert "Specified profile ID 'invalid-profile-id' not found" in str(exc_info.value)


def test_find_profile_no_suitable_profile_found(assets_instinct_path: str) -> None:
    """Test that no suitable profile raises ProfileNotFound."""
    # Use config that won't match any profiles
    config = AIMConfig(
        aim_id="meta-llama/Llama-3.1-8B-Instruct",  # Need aim_id to pass validation
        profile_base_path=assets_instinct_path,
        precision="unsupported_precision",
        engine="unsupported_engine",
        metric="unsupported_metric",
    )

    with patch("aim_runtime.profile_selector.GPUDetector") as mock_detector:
        mock_instance = Mock()
        mock_instance.all_gpus_idle = True
        mock_instance.has_gpus = True
        mock_instance.gpu_models = [None]
        mock_instance.gpu_count = 999
        mock_detector.return_value = mock_instance

        selector = ProfileSelector(config)

        with pytest.raises(ProfileNotFound) as exc_info:
            selector.find_profile()

        error_message = str(exc_info.value)
        assert "No compatible profile found" in error_message
        assert "Profile breakdown:" in error_message
        # Should NOT mention manual selection when profiles are truly incompatible
        assert "AIM_PROFILE_ID" not in error_message


def test_find_profile_manual_selection_only(selector_with_mock_gpu: ProfileSelector, aim_config: AIMConfig) -> None:
    """Test that profiles with manual_selection_only can be selected with explicit profile_id."""

    manual_profile_id = "test_profile_manual"
    config_with_id = AIMConfig(
        aim_id=aim_config.aim_id,
        profile_base_path=aim_config.profile_base_path,
        profile_id=manual_profile_id,
    )

    selector_with_mock_gpu.config = config_with_id
    profile = selector_with_mock_gpu.find_profile()

    assert profile.metadata.manual_selection_only


def test_find_profile_general_fallback(general_aim_config: AIMConfig) -> None:
    """Test that general profiles are used as fallback when no model-specific profiles exist."""
    with patch("aim_runtime.profile_selector.GPUDetector") as mock_detector:
        mock_instance = Mock()
        mock_instance.all_gpus_idle = True
        mock_instance.has_gpus = True
        mock_instance.gpu_models = [GPUModel.MI300X]
        mock_instance.gpu_count = 1
        mock_detector.return_value = mock_instance

        selector = ProfileSelector(general_aim_config)
        profile = selector.find_profile()

        assert isinstance(profile, Profile)
        assert profile.profile_handling.path.endswith(".yaml")
        # Should be from general directory
        assert profile.profile_handling.is_general


def test_auto_precision_engine_metric_defaults(aim_config: AIMConfig) -> None:
    """Test that 'auto' values are converted to preferred defaults."""
    config = AIMConfig(
        aim_id="meta-llama/Llama-3.1-8B-Instruct",
        profile_base_path=aim_config.profile_base_path,
        precision=None,
        engine=Engine.VLLM,
        metric=Metric.LATENCY,
    )

    with patch("aim_runtime.profile_selector.GPUDetector") as mock_detector:
        mock_instance = Mock()
        mock_instance.all_gpus_idle = True
        mock_instance.has_gpus = True
        mock_instance.gpu_models = [GPUModel.MI300X]
        mock_instance.gpu_count = 1
        mock_detector.return_value = mock_instance

        selector = ProfileSelector(config)
        # Should successfully find a profile even with auto values
        profile = selector.find_profile()
        assert isinstance(profile, Profile)
        assert profile.profile_handling.path.endswith(".yaml")


# Tests for get_categorized_profiles functionality


def test_get_categorized_profiles_returns_categorized_dict(
    selector_with_mock_gpu: ProfileSelector,
) -> None:
    """Test that get_categorized_profiles returns properly categorized profiles."""
    categorized = selector_with_mock_gpu.get_categorized_profiles()

    # Should return a dictionary with all expected states
    assert isinstance(categorized, dict)
    expected_states = set([state for state in ProfileCompatibilityState])
    assert set(categorized.keys()) == expected_states

    # All values should be lists
    for state, profiles in categorized.items():
        assert isinstance(profiles, list)


def test_get_categorized_profiles_with_matching_config(
    selector_with_mock_gpu: ProfileSelector,
) -> None:
    """Test that compatible profiles are correctly categorized as COMPATIBLE."""
    categorized = selector_with_mock_gpu.get_categorized_profiles()
    compatible_profiles = categorized[ProfileCompatibilityState.COMPATIBLE]

    # Should have at least one compatible profile with our test config
    assert len(compatible_profiles) > 0

    # All compatible profiles should have valid paths
    for profile in compatible_profiles:
        assert hasattr(profile, "profile_handling")
        assert hasattr(profile.profile_handling, "path")
        assert hasattr(profile.profile_handling, "filename")
        assert hasattr(profile, "metadata")


def test_get_categorized_profiles_precision_mismatch(assets_instinct_path: str) -> None:
    """Test that profiles with precision mismatch are categorized correctly."""

    # Create config with precision that may not match all profiles
    config = AIMConfig(
        aim_id="meta-llama/Llama-3.1-8B-Instruct",
        profile_base_path=assets_instinct_path,
        precision=Precision.INT4,  # Specific precision that might not match all profiles
        engine=Engine.VLLM,
        metric=Metric.LATENCY,
        gpu_count="1",
    )

    with patch("aim_runtime.profile_selector.GPUDetector") as mock_detector:
        mock_instance = Mock()
        mock_instance.all_gpus_idle = True
        mock_instance.has_gpus = True
        mock_instance.gpu_models = ["MI300X"]
        mock_instance.gpu_count = 1
        mock_detector.return_value = mock_instance

        selector = ProfileSelector(config)
        categorized = selector.get_categorized_profiles()

        # Check that categorization occurred
        assert isinstance(categorized, dict)

        # All categories should be lists
        for profiles in categorized.values():
            assert isinstance(profiles, list)


def test_get_categorized_profiles_model_mismatch(
    selector_with_mock_gpu: ProfileSelector,
) -> None:
    """Test that profiles with model mismatch are categorized correctly."""
    # Create a copy with a model that won't have specific profiles
    selector = deepcopy(selector_with_mock_gpu)
    selector.config.aim_id = "nonexistent/aim"

    categorized = selector.get_categorized_profiles()

    # Should have some model mismatch profiles (model-specific profiles for other models)
    model_mismatch = categorized[ProfileCompatibilityState.MODEL_MISMATCH]

    # Check that categorization is working
    assert isinstance(categorized, dict)
    assert isinstance(model_mismatch, list)


def test_get_categorized_profiles_unknown_engine(assets_instinct_path: str) -> None:
    """Test that profiles with unknown engine/metric are categorized correctly."""
    # Create a profile selector with an unsupported engine combination
    config = AIMConfig(
        aim_id="meta-llama/Llama-3.1-8B-Instruct",
        profile_base_path=assets_instinct_path,
        precision=Precision.FP16,
        engine=Engine.VLLM,  # We'll mock this to not match
        metric=Metric.LATENCY,
        gpu_count="1",
    )

    with patch("aim_runtime.profile_selector.GPUDetector") as mock_detector:
        mock_instance = Mock()
        mock_instance.all_gpus_idle = True
        mock_instance.has_gpus = True
        mock_instance.gpu_models = ["MI300X"]
        mock_instance.gpu_count = 1
        mock_detector.return_value = mock_instance

        selector = ProfileSelector(config)

        # Mock the registry to return profiles that don't match engine
        mock_profile = Mock()
        mock_profile.metadata.engine = "unknown_engine"

        with patch.object(selector, "registry") as mock_registry:
            mock_registry.profiles = [mock_profile]

            categorized = selector.get_categorized_profiles()
            engine_incompatible_profiles = categorized[ProfileCompatibilityState.ENGINE_MISMATCH]

            # The mock profile should be in engine_incompatible category
            assert len(engine_incompatible_profiles) == 1
            assert engine_incompatible_profiles[0] == mock_profile


def test_get_categorized_profiles_with_auto_precision(
    selector_with_mock_gpu: ProfileSelector,
) -> None:
    """Test that AUTO precision is handled correctly."""
    # Create a copy with AUTO precision
    selector = deepcopy(selector_with_mock_gpu)
    selector.config.precision = None

    categorized = selector.get_categorized_profiles()

    # With AUTO precision, profiles should not be rejected for precision mismatch
    # (unless they specifically don't support the detected/configured precision)
    precision_incompatible = categorized[ProfileCompatibilityState.PRECISION_MISMATCH]

    # AUTO should be more permissive
    assert isinstance(precision_incompatible, list)


def test_get_categorized_profiles_ordering_preserved(
    selector_with_mock_gpu: ProfileSelector,
) -> None:
    """Test that compatible profiles maintain priority ordering."""
    categorized = selector_with_mock_gpu.get_categorized_profiles()
    compatible_profiles = categorized[ProfileCompatibilityState.COMPATIBLE]

    if len(compatible_profiles) > 1:
        # Check that profiles are ordered by priority (lower number = higher priority)
        for i in range(len(compatible_profiles) - 1):
            current_priority = getattr(compatible_profiles[i], "priority", 0)
            next_priority = getattr(compatible_profiles[i + 1], "priority", 0)
            assert current_priority <= next_priority


def test_get_categorized_profiles_empty_registry() -> None:
    """Test behavior when no profiles are found."""

    # Create config with path that has no profiles
    config = AIMConfig(
        aim_id="meta-llama/Llama-3.1-8B-Instruct",
        profile_base_path="/nonexistent/path",  # Path with no profiles
        precision=Precision.FP16,
        engine=Engine.VLLM,
        metric=Metric.LATENCY,
        gpu_count="1",
    )

    with patch("aim_runtime.profile_selector.GPUDetector") as mock_detector:
        mock_instance = Mock()
        mock_instance.all_gpus_idle = True
        mock_instance.has_gpus = True
        mock_instance.gpu_models = ["MI300X"]
        mock_instance.gpu_count = 1
        mock_detector.return_value = mock_instance

        selector = ProfileSelector(config)
        categorized = selector.get_categorized_profiles()

        # All categories should be empty lists
        for state, profiles in categorized.items():
            assert isinstance(profiles, list)
            assert len(profiles) == 0


def test_get_categorized_profiles_integration_with_real_profiles(
    selector_with_mock_gpu: ProfileSelector,
) -> None:
    """Integration test with real profile structure."""
    categorized = selector_with_mock_gpu.get_categorized_profiles()

    # Verify the structure matches our enum
    expected_states = set([state for state in ProfileCompatibilityState])
    assert set(categorized.keys()) == expected_states

    # Count total profiles
    total_profiles = sum(len(profiles) for profiles in categorized.values())

    # Should have found some profiles
    assert total_profiles > 0

    # At least one category should have profiles
    assert any(len(profiles) > 0 for profiles in categorized.values())

    # Verify profile objects have expected attributes
    for state, profiles in categorized.items():
        for profile in profiles:
            assert hasattr(profile, "profile_handling")
            assert hasattr(profile.profile_handling, "filename")
            assert hasattr(profile.profile_handling, "path")
            assert hasattr(profile, "metadata")

            # Test that profile paths exist (for compatible profiles)
            if state == ProfileCompatibilityState.COMPATIBLE:
                assert Path(profile.profile_handling.path).exists()


def test_profile_selector_with_gpu_model_override(aim_config: AIMConfig) -> None:
    """Test that ProfileSelector uses AIM_GPU_MODEL when provided."""
    # Create config with GPU model override
    config_with_gpu_model = deepcopy(aim_config)
    config_with_gpu_model.gpu_model = GPUModel.MI325X
    config_with_gpu_model.gpu_count = 2

    # ProfileSelector should not call GPUDetector when gpu_model is set
    with patch("aim_runtime.profile_selector.GPUDetector") as mock_detector:
        selector = ProfileSelector(config_with_gpu_model)

        # GPUDetector should not be instantiated when gpu_model is set
        mock_detector.assert_not_called()

        # Check that the GPU model from config is used
        assert selector.detected_gpu == GPUModel.MI325X
        assert selector.detected_gpu_count == 2


def test_profile_selector_gpu_model_with_auto_gpu_count(aim_config: AIMConfig) -> None:
    """Test that ProfileSelector defaults to 1 GPU when gpu_model is set but gpu_count is auto."""
    config_with_gpu_model = deepcopy(aim_config)
    config_with_gpu_model.gpu_model = GPUModel.MI300A
    config_with_gpu_model.gpu_count = "auto"

    with patch("aim_runtime.profile_selector.GPUDetector") as mock_detector:
        selector = ProfileSelector(config_with_gpu_model)

        mock_detector.assert_not_called()
        assert selector.detected_gpu == GPUModel.MI300A
        assert selector.detected_gpu_count == 1


def test_build_search_paths_with_general_fallback_enabled(assets_instinct_path: str) -> None:
    """Test that general profiles are included when allow_general_profile_fallback is True."""
    config = AIMConfig(
        aim_id="meta-llama/Llama-3.1-8B-Instruct",
        profile_base_path=assets_instinct_path,
        allow_general_profile_fallback=True,
    )

    with patch("aim_runtime.profile_selector.GPUDetector") as mock_detector:
        mock_instance = Mock()
        mock_instance.all_gpus_idle = True
        mock_instance.has_gpus = True
        mock_instance.gpu_models = [GPUModel.MI300X]
        mock_instance.gpu_count = 1
        mock_detector.return_value = mock_instance

        selector = ProfileSelector(config)
        search_paths = selector._build_search_paths()

        # Should include custom, model-specific, and general paths
        assert len(search_paths) == 3
        assert search_paths[0].endswith("/custom")
        assert "meta-llama/Llama-3.1-8B-Instruct" in search_paths[1]
        assert search_paths[2].endswith("/general")


def test_build_search_paths_with_general_fallback_disabled(assets_instinct_path: str) -> None:
    """Test that general profiles are still loaded when allow_general_profile_fallback is False (but marked manual-only)."""
    config = AIMConfig(
        aim_id="meta-llama/Llama-3.1-8B-Instruct",
        profile_base_path=assets_instinct_path,
        allow_general_profile_fallback=False,
    )

    with patch("aim_runtime.profile_selector.GPUDetector") as mock_detector:
        mock_instance = Mock()
        mock_instance.all_gpus_idle = True
        mock_instance.has_gpus = True
        mock_instance.gpu_models = [GPUModel.MI300X]
        mock_instance.gpu_count = 1
        mock_detector.return_value = mock_instance

        selector = ProfileSelector(config)
        search_paths = selector._build_search_paths()

        # Should still include all paths (custom, model-specific, and general)
        assert len(search_paths) == 3
        assert search_paths[0].endswith("/custom")
        assert "meta-llama/Llama-3.1-8B-Instruct" in search_paths[1]
        assert search_paths[2].endswith("/general")


def test_general_profiles_marked_manual_only_when_fallback_disabled(profile_base_path: str) -> None:
    """Test that general profiles are marked as manual_selection_only when fallback is disabled."""
    config = AIMConfig(
        aim_id="meta-llama/Llama-3.1-8B-Instruct",
        profile_base_path=profile_base_path,
        allow_general_profile_fallback=False,
    )

    with patch("aim_runtime.profile_selector.GPUDetector") as mock_detector:
        mock_instance = Mock()
        mock_instance.all_gpus_idle = True
        mock_instance.has_gpus = True
        mock_instance.gpu_models = [GPUModel.MI300X]
        mock_instance.gpu_count = 1
        mock_detector.return_value = mock_instance

        selector = ProfileSelector(config)

        # Check that all general profiles are marked as manual-selection-only
        general_profiles = selector.registry.get_general_profiles()
        assert len(general_profiles) > 0, "Should have some general profiles"

        for profile in general_profiles:
            assert (
                profile.metadata.manual_selection_only is True
            ), f"General profile {profile.profile_id} should be marked as manual-selection-only"


def test_general_profiles_not_marked_manual_only_when_fallback_enabled(profile_base_path: str) -> None:
    """Test that general profiles are NOT marked as manual_selection_only when fallback is enabled."""
    config = AIMConfig(
        aim_id="meta-llama/Llama-3.1-8B-Instruct",
        profile_base_path=profile_base_path,
        allow_general_profile_fallback=True,
    )

    with patch("aim_runtime.profile_selector.GPUDetector") as mock_detector:
        mock_instance = Mock()
        mock_instance.all_gpus_idle = True
        mock_instance.has_gpus = True
        mock_instance.gpu_models = [GPUModel.MI300X]
        mock_instance.gpu_count = 1
        mock_detector.return_value = mock_instance

        selector = ProfileSelector(config)

        # Check that general profiles are NOT marked as manual-selection-only (unless they were in YAML)
        general_profiles = selector.registry.get_general_profiles()
        assert len(general_profiles) > 0, "Should have some general profiles"

        # At least one general profile should be auto-selectable
        auto_selectable = [p for p in general_profiles if not p.metadata.manual_selection_only]
        assert len(auto_selectable) > 0, "Should have at least one auto-selectable general profile"


def test_find_profile_fallback_disabled_excludes_general_from_auto_selection(profile_base_path: str) -> None:
    """Test that general profiles are excluded from automatic selection when fallback is disabled."""
    # Use a model that doesn't have model-specific profiles in test fixtures
    config = AIMConfig(
        aim_id="nonexistent/model",
        profile_base_path=profile_base_path,
        allow_general_profile_fallback=False,
    )

    with patch("aim_runtime.profile_selector.GPUDetector") as mock_detector:
        mock_instance = Mock()
        mock_instance.all_gpus_idle = True
        mock_instance.has_gpus = True
        mock_instance.gpu_models = [GPUModel.MI300X]
        mock_instance.gpu_count = 1
        mock_detector.return_value = mock_instance

        selector = ProfileSelector(config)

        # Should raise ProfileNotFound with informative message about manual-only profiles
        with pytest.raises(ProfileNotFound) as exc_info:
            selector.find_profile()

        error_message = str(exc_info.value)
        # Check that error mentions manual selection requirement
        assert "require explicit selection via AIM_PROFILE_ID" in error_message
        # Check that it lists available profiles
        assert "Available profiles have the following AIM_PROFILE_ID values:" in error_message

        # But general profiles should still be listable
        general_profiles = selector.registry.get_general_profiles()
        assert len(general_profiles) > 0, "General profiles should still be loaded"


def test_find_profile_fallback_enabled_uses_general(profile_base_path: str) -> None:
    """Test that general profiles are used for automatic selection when fallback is enabled."""
    # Use a model that doesn't have model-specific profiles in test fixtures
    config = AIMConfig(
        aim_id="nonexistent/model",
        profile_base_path=profile_base_path,
        allow_general_profile_fallback=True,
        precision=Precision.FP16,
        gpu_count=1,
    )

    with patch("aim_runtime.profile_selector.GPUDetector") as mock_detector:
        mock_instance = Mock()
        mock_instance.all_gpus_idle = True
        mock_instance.has_gpus = True
        mock_instance.gpu_models = [GPUModel.MI300X]
        mock_instance.gpu_count = 1
        mock_detector.return_value = mock_instance

        selector = ProfileSelector(config)
        profile = selector.find_profile()

        # Should find a general profile
        assert profile.profile_handling.is_general
        assert "general" in profile.profile_handling.path


# Tests for unoptimized profile fallback (AIM_ALLOW_UNOPTIMIZED)


def test_find_profile_unoptimized_fallback_disabled_by_default(profile_base_path: str) -> None:
    """Test that unoptimized profiles are NOT auto-selected when allow_unoptimized is False (default)."""
    config = AIMConfig(
        aim_id="meta-llama/Llama-3.1-8B-Instruct",
        profile_base_path=str(profile_base_path),
        precision=Precision.FP16,
        gpu_count=1,
        allow_unoptimized=False,
        allow_general_profile_fallback=False,
    )

    with patch("aim_runtime.profile_selector.GPUDetector") as mock_detector:
        mock_instance = Mock()
        mock_instance.all_gpus_idle = True
        mock_instance.has_gpus = True
        # MI325X only has unoptimized profiles in test fixtures
        mock_instance.gpu_models = [GPUModel.MI325X]
        mock_instance.gpu_count = 1
        mock_detector.return_value = mock_instance

        selector = ProfileSelector(config)

        with pytest.raises(ProfileNotFound) as exc_info:
            selector.find_profile()

        assert "AIM_PROFILE_ID" in str(exc_info.value)


def test_find_profile_unoptimized_fallback_enabled_selects_unoptimized(profile_base_path: str) -> None:
    """Test that unoptimized profiles ARE auto-selected when allow_unoptimized is True."""
    config = AIMConfig(
        aim_id="meta-llama/Llama-3.1-8B-Instruct",
        profile_base_path=str(profile_base_path),
        precision=Precision.FP16,
        gpu_count=1,
        allow_unoptimized=True,
        allow_general_profile_fallback=False,
    )

    with patch("aim_runtime.profile_selector.GPUDetector") as mock_detector:
        mock_instance = Mock()
        mock_instance.all_gpus_idle = True
        mock_instance.has_gpus = True
        # MI325X only has unoptimized profiles in test fixtures
        mock_instance.gpu_models = [GPUModel.MI325X]
        mock_instance.gpu_count = 1
        mock_detector.return_value = mock_instance

        selector = ProfileSelector(config)
        profile = selector.find_profile()

        assert profile.metadata.type == ProfileType.UNOPTIMIZED
        assert profile.metadata.manual_selection_only is True


def test_find_profile_unoptimized_fallback_prefers_optimized(profile_base_path: str) -> None:
    """Test that optimized profiles are preferred over unoptimized even when fallback is enabled."""
    config = AIMConfig(
        aim_id="meta-llama/Llama-3.1-8B-Instruct",
        profile_base_path=str(profile_base_path),
        precision=Precision.FP16,
        gpu_count=1,
        allow_unoptimized=True,
    )

    with patch("aim_runtime.profile_selector.GPUDetector") as mock_detector:
        mock_instance = Mock()
        mock_instance.all_gpus_idle = True
        mock_instance.has_gpus = True
        # MI300X has both optimized and unoptimized profiles
        mock_instance.gpu_models = [GPUModel.MI300X]
        mock_instance.gpu_count = 1
        mock_detector.return_value = mock_instance

        selector = ProfileSelector(config)
        profile = selector.find_profile()

        # Should select the optimized profile, not the unoptimized one
        assert profile.metadata.type == ProfileType.OPTIMIZED


def test_find_profile_unoptimized_fallback_logs_warning(profile_base_path: str) -> None:
    """Test that a warning is logged when unoptimized fallback is used."""
    config = AIMConfig(
        aim_id="meta-llama/Llama-3.1-8B-Instruct",
        profile_base_path=str(profile_base_path),
        precision=Precision.FP16,
        gpu_count=1,
        allow_unoptimized=True,
        allow_general_profile_fallback=False,
    )

    with patch("aim_runtime.profile_selector.GPUDetector") as mock_detector:
        mock_instance = Mock()
        mock_instance.all_gpus_idle = True
        mock_instance.has_gpus = True
        mock_instance.gpu_models = [GPUModel.MI325X]
        mock_instance.gpu_count = 1
        mock_detector.return_value = mock_instance

        selector = ProfileSelector(config)

        import logging

        with patch.object(logging.getLogger("aim_runtime.profile_selector"), "warning") as mock_warning:
            selector.find_profile()
            mock_warning.assert_called_once()
            warning_msg = mock_warning.call_args[0][0]
            assert "UNOPTIMIZED PROFILE FALLBACK" in warning_msg


def test_find_profile_unoptimized_fallback_no_compatible_profiles(profile_base_path: str) -> None:
    """Test that ProfileNotFound is raised even with fallback when no profiles match at all."""
    config = AIMConfig(
        aim_id="meta-llama/Llama-3.1-8B-Instruct",
        profile_base_path=str(profile_base_path),
        precision=Precision.FP16,
        gpu_count=1,
        allow_unoptimized=True,
    )

    with patch("aim_runtime.profile_selector.GPUDetector") as mock_detector:
        mock_instance = Mock()
        mock_instance.all_gpus_idle = True
        mock_instance.has_gpus = True
        # Use a GPU that has no profiles at all
        mock_instance.gpu_models = [GPUModel.MI300A]
        mock_instance.gpu_count = 4
        mock_detector.return_value = mock_instance

        selector = ProfileSelector(config)

        with pytest.raises(ProfileNotFound) as exc_info:
            selector.find_profile()

        assert "No compatible profile found" in str(exc_info.value)


def test_find_profile_unoptimized_and_general_fallback_interaction(profile_base_path: str) -> None:
    """Test that general profiles (pass 1) are preferred over unoptimized fallback (pass 2)."""
    # Use a model that doesn't have model-specific profiles, so only general profiles match
    config = AIMConfig(
        aim_id="nonexistent/model",
        profile_base_path=str(profile_base_path),
        precision=Precision.FP16,
        gpu_count=1,
        allow_unoptimized=True,
        allow_general_profile_fallback=True,
    )

    with patch("aim_runtime.profile_selector.GPUDetector") as mock_detector:
        mock_instance = Mock()
        mock_instance.all_gpus_idle = True
        mock_instance.has_gpus = True
        mock_instance.gpu_models = [GPUModel.MI300X]
        mock_instance.gpu_count = 1
        mock_detector.return_value = mock_instance

        selector = ProfileSelector(config)
        profile = selector.find_profile()

        # General profiles are auto-selectable in pass 1, so they should be selected
        # before the unoptimized fallback in pass 2 even activates
        assert profile.profile_handling.is_general


class TestSerializeProfiles:
    """Tests for profile serialization methods (JSON/YAML output)."""

    def test_serialize_profiles_structure(self, selector_with_mock_gpu: ProfileSelector) -> None:
        """Test that serialize_profiles returns wrapper dicts with correct top-level keys."""
        categorized = selector_with_mock_gpu.get_categorized_profiles()
        serialized = selector_with_mock_gpu.serialize_profiles(categorized)

        assert isinstance(serialized, list)
        assert len(serialized) > 0

        for entry in serialized:
            assert set(entry.keys()) == {"profile_id", "compatibility", "profile"}
            assert isinstance(entry["profile_id"], str)
            assert isinstance(entry["compatibility"], str)
            assert isinstance(entry["profile"], dict)

    def test_serialize_profiles_compatibility_values(self, selector_with_mock_gpu: ProfileSelector) -> None:
        """Test that compatibility values are valid state strings."""
        categorized = selector_with_mock_gpu.get_categorized_profiles()
        serialized = selector_with_mock_gpu.serialize_profiles(categorized)

        valid_states = {state.value for state in ProfileCompatibilityState}
        for entry in serialized:
            assert entry["compatibility"] in valid_states

    def test_serialize_profiles_raw_yaml_content(self, selector_with_mock_gpu: ProfileSelector) -> None:
        """Test that profile content matches raw YAML from disk."""
        categorized = selector_with_mock_gpu.get_categorized_profiles()
        serialized = selector_with_mock_gpu.serialize_profiles(categorized)

        # Find a compatible profile and verify its content has expected YAML fields
        compatible = [e for e in serialized if e["compatibility"] == "compatible"]
        assert len(compatible) > 0

        profile_data = compatible[0]["profile"]
        assert "metadata" in profile_data

    def test_serialize_profiles_filtered_by_state(self, selector_with_mock_gpu: ProfileSelector) -> None:
        """Test serialization with state filter applied."""
        categorized = selector_with_mock_gpu.get_categorized_profiles()

        # Filter to compatible only
        filtered = {ProfileCompatibilityState.COMPATIBLE: categorized[ProfileCompatibilityState.COMPATIBLE]}
        serialized = selector_with_mock_gpu.serialize_profiles(filtered)

        for entry in serialized:
            assert entry["compatibility"] == "compatible"

    def test_serialize_all_profiles_structure(self, selector_with_mock_gpu: ProfileSelector) -> None:
        """Test that serialize_all_profiles returns all profiles with unknown compatibility."""
        serialized = selector_with_mock_gpu.serialize_all_profiles()

        assert isinstance(serialized, list)
        assert len(serialized) > 0

        for entry in serialized:
            assert set(entry.keys()) == {"profile_id", "compatibility", "profile"}
            assert entry["compatibility"] == "unknown"

    def test_serialize_all_profiles_sorted(self, selector_with_mock_gpu: ProfileSelector) -> None:
        """Test that serialize_all_profiles returns profiles sorted by priority then profile_id."""
        serialized = selector_with_mock_gpu.serialize_all_profiles()

        assert len(serialized) >= 2  # We have multiple test profiles

        # Verify ordering matches the registry profiles sorted by (priority, profile_id)
        registry_sorted = sorted(
            selector_with_mock_gpu.registry.profiles,
            key=lambda p: (p.profile_handling.priority, p.profile_id),
        )
        expected_ids = [p.profile_id for p in registry_sorted]
        actual_ids = [e["profile_id"] for e in serialized]
        assert actual_ids == expected_ids

    def test_serialize_profiles_profile_id_matches(self, selector_with_mock_gpu: ProfileSelector) -> None:
        """Test that profile_id in wrapper matches the profile's computed identifier."""
        categorized = selector_with_mock_gpu.get_categorized_profiles()
        serialized = selector_with_mock_gpu.serialize_profiles(categorized)

        # profile_id should be a non-empty string
        for entry in serialized:
            assert len(entry["profile_id"]) > 0
