# Copyright © Advanced Micro Devices, Inc., or its affiliates.
#
# SPDX-License-Identifier: MIT

import os

import pytest

from aim_common import GPUModel
from aim_runtime.profile_registry import ProfileRegistry


@pytest.fixture
def custom_profiles_path(profiles_path: str) -> str:
    return os.path.join(profiles_path, "custom")


@pytest.fixture
def model_profiles_path(profiles_path: str) -> str:
    return os.path.join(profiles_path, "meta-llama", "Llama-3.1-8B-Instruct")


def test_discover_and_validate_precedence(custom_profiles_path, model_profiles_path, general_profiles_path, validator):
    """Test discovering and validating profiles."""
    registry = ProfileRegistry.discover_and_validate(
        [custom_profiles_path, model_profiles_path, general_profiles_path], validator
    )

    assert registry.total_discovered == 6
    assert len(registry.profiles) == 6

    # Verify the search path is set correctly
    assert registry.search_path == " -> ".join([custom_profiles_path, model_profiles_path, general_profiles_path])

    # Verify profile contents
    general_profiles = registry.get_general_profiles()
    assert len(general_profiles) == 2
    assert general_profiles[0].profile_handling.is_general
    assert general_profiles[0].metadata.gpu == GPUModel.MI300X

    model_profile = registry.find_by_id("custom/meta-llama/Llama-3.1-8B-Instruct/test_profile_correct")
    assert model_profile.metadata.gpu == GPUModel.MI355X
