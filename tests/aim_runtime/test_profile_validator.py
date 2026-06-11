# Copyright © Advanced Micro Devices, Inc., or its affiliates.
#
# SPDX-License-Identifier: MIT

"""
Tests for profile_validator.py.

These tests cover ProfileValidator with Pydantic profile structure validation.
Engine-specific arg validation is handled by engine_config.py and tested in test_engine_config.py.
"""

import os
from pathlib import Path

import pytest

from aim_runtime.profile_validator import ProfileValidator
from aim_utils.yaml_utils import read_yaml


def test_validate_model_profile_should_pass(profile_validator: ProfileValidator, model_profiles_path: str) -> None:
    profile_data = read_yaml(Path(os.path.join(model_profiles_path, "test_profile_correct.yaml")))
    profile_validator.validate(profile_data)


def test_validate_general_profile_should_pass(profile_validator: ProfileValidator, general_profiles_path: str) -> None:
    profile_data = read_yaml(Path(os.path.join(general_profiles_path, "test_profile_correct.yaml")))
    profile_validator.validate(profile_data, is_general_profile=True)


def test_validate_profile_with_missing_model_section_should_fail(
    profile_validator: ProfileValidator, model_profiles_path: str
):
    with pytest.raises(Exception) as e:
        profile_data = read_yaml(Path(os.path.join(model_profiles_path, "test_profile_missing_model.yaml")))
        profile_validator.validate(profile_data)
    assert "model" in str(e.value)


def test_validate_valid_model_profile_dict(profile_validator: ProfileValidator):
    """Test validating a valid model profile from a dict."""
    profile_data = {
        "aim_id": "meta-llama/Llama-3.1-8B-Instruct",
        "model_id": "meta-llama/Llama-3.1-8B-Instruct",
        "metadata": {
            "engine": "vllm",
            "accelerator_model": "MI300X",
            "precision": "fp16",
            "accelerator_count": 1,
            "metric": "latency",
            "manual_selection_only": False,
            "type": "optimized",
        },
        "engine_args": {"dtype": "float16", "tensor-parallel-size": 1},
        "env_vars": {},
    }
    profile_validator.validate(profile_data)


def test_validate_valid_general_profile_dict(profile_validator: ProfileValidator):
    """Test validating a valid general profile from a dict."""
    profile_data = {
        "metadata": {
            "engine": "vllm",
            "accelerator_model": "MI300X",
            "precision": "fp16",
            "accelerator_count": 1,
            "metric": "latency",
            "manual_selection_only": False,
            "type": "unoptimized",
        },
        "engine_args": {"dtype": "float16"},
        "env_vars": {},
    }
    profile_validator.validate(profile_data, is_general_profile=True)
