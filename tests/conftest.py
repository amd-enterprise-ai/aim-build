# Copyright © Advanced Micro Devices, Inc., or its affiliates.
#
# SPDX-License-Identifier: MIT

from copy import deepcopy
from pathlib import Path
from typing import Any

import pytest

from aim_common import Engine, Metric, Precision
from aim_runtime import AIMConfig
from aim_runtime.object_model import Profile
from aim_runtime.profile_registry import ProfileRegistry
from aim_runtime.profile_validator import ProfileValidator


# Common fixtures
@pytest.fixture
def project_root() -> Path:
    """Get the project root directory."""
    return Path(__file__).parent.parent


@pytest.fixture
def test_root() -> Path:
    """Get the test directory root."""
    return Path(__file__).parent


# aim_runtime fixtures
@pytest.fixture
def profile_base_path(test_root: Path) -> Path:
    return test_root / "workspace" / "profiles"


@pytest.fixture
def general_profiles_path(test_root: Path) -> str:
    """Get the test profiles directory path."""
    return str(test_root / "workspace" / "profiles" / "general")


@pytest.fixture
def custom_profiles_path(profile_base_path: Path) -> str:
    return str(profile_base_path / "custom")


@pytest.fixture
def aim_config(profile_base_path: Path) -> AIMConfig:
    """Create a test configuration with known valid parameters."""
    return AIMConfig(
        aim_id="meta-llama/Llama-3.1-8B-Instruct",
        profile_base_path=str(profile_base_path),
        precision=Precision.FP16,
        engine=Engine.VLLM,
        metric=Metric.LATENCY,
        gpu_count="1",
        gpu_model=None,
    )


@pytest.fixture
def general_aim_config(aim_config: AIMConfig, profile_base_path: Path) -> AIMConfig:
    """Create a test configuration for general profiles."""
    config = deepcopy(aim_config)
    config.aim_id = ""  # Clear aim_id
    config.model_id = "meta-llama/Llama-3.1-8B-Instruct"  # Set model_id instead
    config.profile_base_path = str(profile_base_path)
    return config


@pytest.fixture
def faulty_aim_config_with_no_model(aim_config: AIMConfig) -> AIMConfig:
    """Create a test configuration with no model specified."""
    config = deepcopy(aim_config)
    config.aim_id = ""
    config.model_id = ""
    return config


@pytest.fixture
def profile_validator() -> ProfileValidator:
    """Create a profile validator for testing."""
    return ProfileValidator()


@pytest.fixture
def no_op_profile_validator() -> ProfileValidator:
    """Create a no-op profile validator for testing (skips validation)."""

    class NoOpProfileValidator(ProfileValidator):
        def validate(self, profile_data: dict[str, Any], is_general_profile: bool = False) -> None:
            return

    return NoOpProfileValidator()


@pytest.fixture
def model_profile(model_profiles_path: str, profile_validator: ProfileValidator) -> Profile:
    """Create a sample model profile for testing."""
    registry = ProfileRegistry.discover_and_validate(search_paths=[model_profiles_path], validator=profile_validator)
    return registry.find_by_id("test_profile_correct")


@pytest.fixture
def general_profile(general_profiles_path: str, profile_validator: ProfileValidator) -> Profile:
    """Get a valid general test profile."""
    registry = ProfileRegistry.discover_and_validate(search_paths=[general_profiles_path], validator=profile_validator)
    return registry.find_by_id("general/test_profile_correct")


@pytest.fixture
def complex_profile(assets_instinct_path: Path, no_op_profile_validator: ProfileValidator) -> Profile:
    """Get a complex test profile with comprehensive test data."""
    profiles_path = assets_instinct_path / "test" / "model" / "profiles"
    registry = ProfileRegistry.discover_and_validate(
        search_paths=[str(profiles_path)], validator=no_op_profile_validator
    )
    return registry.find_by_id("complex_profile")


# aim_utils fixtures
@pytest.fixture
def model_profiles_path(assets_instinct_path: Path) -> str:
    return str(assets_instinct_path / "meta-llama" / "Llama-3.1-8B-Instruct" / "profiles")


@pytest.fixture
def assets_instinct_path(assets_path: Path) -> Path:
    return assets_path / "instinct"


@pytest.fixture
def assets_path(test_root: Path) -> Path:
    return test_root / "assets"
