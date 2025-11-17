# Copyright © Advanced Micro Devices, Inc., or its affiliates.
#
# SPDX-License-Identifier: MIT

import os

import pytest
import yaml

from aim_runtime.profile_validator import ProfileValidator


def test_validate_model_profile_should_pass(validator: ProfileValidator, model_profiles_path: str) -> None:
    with open(os.path.join(model_profiles_path, "test_profile_correct.yaml"), mode="r") as f:
        profile_data = yaml.safe_load(f)
        validator.validate(profile_data)


def test_validate_general_profile_should_pass(validator: ProfileValidator, general_profiles_path: str) -> None:
    with open(os.path.join(general_profiles_path, "test_profile_correct.yaml"), mode="r") as f:
        profile_data = yaml.safe_load(f)
        validator.validate(profile_data, is_general_profile=True)


def test_validate_profile_with_incorrect_dtype_should_fail(validator: ProfileValidator, model_profiles_path: str):
    with pytest.raises(Exception) as e:
        with open(os.path.join(model_profiles_path, "test_profile_incorrect_dtype.yaml"), mode="r") as f:
            profile_data = yaml.safe_load(f)
            validator.validate(profile_data)
    assert "float16" in str(e.value) and "float17" in str(e.value)


def test_validate_profile_with_missing_model_section_should_fail(validator: ProfileValidator, model_profiles_path: str):
    with pytest.raises(Exception) as e:
        with open(os.path.join(model_profiles_path, "test_profile_missing_model.yaml"), mode="r") as f:
            profile_data = yaml.safe_load(f)
            validator.validate(profile_data)
    assert "model" in str(e.value)
