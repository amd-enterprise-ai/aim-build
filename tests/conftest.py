# Copyright © Advanced Micro Devices, Inc., or its affiliates.
#
# SPDX-License-Identifier: MIT

import os
from copy import deepcopy
from pathlib import Path
from typing import Any

import pytest

from aim_runtime import AIMConfig
from aim_runtime.object_model import Engine, Metric, Precision, Profile
from aim_runtime.profile_registry import ProfileRegistry
from aim_runtime.profile_validator import ProfileValidator


@pytest.fixture
def project_root() -> Path:
    """Get the project root directory."""
    return Path(__file__).parent.parent


@pytest.fixture
def test_root() -> Path:
    """Get the test directory root."""
    return Path(__file__).parent


@pytest.fixture
def schemas_path(project_root: Path) -> str:
    """Get the schemas directory path."""
    return str(project_root / "schemas")


@pytest.fixture
def profiles_path(test_root: Path) -> str:
    """Get the test profiles directory path."""
    return str(test_root / "profiles")


@pytest.fixture
def general_profiles_path(test_root: Path) -> str:
    """Get the test profiles directory path."""
    return str(test_root / os.path.join("profiles", "general"))


@pytest.fixture
def model_profiles_path(profiles_path: str) -> str:
    return os.path.join(profiles_path, "meta-llama", "Llama-3.1-8B-Instruct")


@pytest.fixture
def metadata_path(test_root: Path) -> str:
    return str(test_root / "metadata")


@pytest.fixture
def aim_config(schemas_path: str, profiles_path: str) -> AIMConfig:
    """Create a test configuration with known valid parameters."""
    from aim_runtime.object_model import GPUModel

    return AIMConfig(
        aim_id="meta-llama/Llama-3.1-8B-Instruct",
        schema_search_path=schemas_path,
        profile_base_path=profiles_path,
        precision=Precision.FP16,
        engine=Engine.VLLM,
        metric=Metric.LATENCY,
        gpu_count="1",
        gpu_model=GPUModel.NONE,
    )


@pytest.fixture
def general_aim_config(aim_config: AIMConfig) -> AIMConfig:
    """Create a test configuration for general profiles."""
    config = deepcopy(aim_config)
    config.aim_id = ""  # Clear aim_id
    config.model_id = "meta-llama/Llama-3.1-8B-Instruct"  # Set model_id instead
    return config


@pytest.fixture
def faulty_aim_config_with_no_model(aim_config: AIMConfig) -> AIMConfig:
    """Create a test configuration with no model specified."""
    config = deepcopy(aim_config)
    config.aim_id = ""
    config.model_id = ""
    return config


@pytest.fixture
def validator(schemas_path: str) -> ProfileValidator:
    return ProfileValidator(schemas_path)


@pytest.fixture
def no_op_profile_validator(schemas_path: str) -> ProfileValidator:
    """Create a no-op profile validator for testing (skips validation)."""

    class NoOpProfileValidator(ProfileValidator):
        def validate(self, profile_data: dict[str, Any], is_general_profile: bool = False) -> None:
            return

    return NoOpProfileValidator(schemas_path)


@pytest.fixture
def profile_validator(schemas_path: str) -> ProfileValidator:
    """Create a profile validator for testing."""
    return ProfileValidator(schemas_path)


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
def complex_profile(profiles_path: str, no_op_profile_validator: ProfileValidator) -> Profile:
    """Get a complex test profile with comprehensive test data."""
    registry = ProfileRegistry.discover_and_validate(search_paths=[profiles_path], validator=no_op_profile_validator)
    return registry.find_by_id("complex_profile")
