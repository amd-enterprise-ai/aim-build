# Copyright © Advanced Micro Devices, Inc., or its affiliates.
#
# SPDX-License-Identifier: MIT

from pathlib import Path

import pytest

from aim_common import GPUModel
from aim_runtime.profile_registry import ProfileRegistry


@pytest.fixture
def custom_profiles_path(assets_instinct_path: str) -> str:
    return str(Path(assets_instinct_path) / "custom")


@pytest.fixture
def model_profiles_path(assets_instinct_path: str) -> str:
    return str(Path(assets_instinct_path) / "meta-llama" / "Llama-3.1-8B-Instruct" / "profiles")


def test_discover_and_validate_precedence(
    custom_profiles_path, model_profiles_path, general_profiles_path, profile_validator
):
    """Test discovering and validating profiles."""
    registry = ProfileRegistry.discover_and_validate(
        [custom_profiles_path, model_profiles_path, general_profiles_path], profile_validator
    )

    assert registry.total_discovered == 7
    assert len(registry.profiles) == 7

    # Verify the search path is set correctly
    assert registry.search_path == " -> ".join([custom_profiles_path, model_profiles_path, general_profiles_path])

    # Verify profile contents
    general_profiles = registry.get_general_profiles()
    assert len(general_profiles) == 2
    assert general_profiles[0].profile_handling.is_general
    assert general_profiles[0].metadata.gpu == GPUModel.MI300X

    model_profile = registry.find_by_id("custom/meta-llama/Llama-3.1-8B-Instruct/test_profile_correct")
    assert model_profile.metadata.gpu == GPUModel.MI355X
