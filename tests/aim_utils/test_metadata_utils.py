# Copyright © Advanced Micro Devices, Inc., or its affiliates.
#
# SPDX-License-Identifier: MIT

import os
from pathlib import Path
from typing import Any, Dict

import pytest

from aim_utils import metadata_utils, yaml_utils


@pytest.fixture(params=["instinct", "radeon"], ids=["instinct", "radeon"])
def sample_yaml_data(request, assets_path: Path) -> Dict[str, Any]:
    return yaml_utils.read_yaml(
        assets_path / request.param / "TinyLlama" / "TinyLlama-1.1B-Chat-v1.0_case1" / "metadata.yaml"
    )


@pytest.fixture(params=["tests/assets/instinct", "tests/assets/radeon"], ids=["instinct", "radeon"])
def test_metadata_path(request):
    return request.param


def test_get_value_existing_key(sample_yaml_data):
    assert (
        metadata_utils.get_value(sample_yaml_data, "com.amd.aim.model.canonicalName")
        == "TinyLlama/TinyLlama-1.1B-Chat-v1.0_case1"
    )
    assert metadata_utils.get_value(sample_yaml_data, "org.opencontainers.image.vendor") == "AMD"
    assert metadata_utils.get_value(sample_yaml_data, "com.amd.aim.hfToken.required") is False


def test_get_value_missing_key(sample_yaml_data):
    assert metadata_utils.get_value(sample_yaml_data, "com.amd.aim.model.nonexistent") is None
    assert metadata_utils.get_value(sample_yaml_data, "com.amd.aim.nonexistent.key") is None
    assert metadata_utils.get_value(sample_yaml_data, "com.nonexistent") is None


def test_get_value_partial_path(sample_yaml_data):
    # Should return the nested dict if not a leaf
    result = metadata_utils.get_value(sample_yaml_data, "com.amd.aim.model")
    assert isinstance(result, dict)
    assert "canonicalName" in result


def test_set_value_existing_key(sample_yaml_data):
    updated_data = metadata_utils.set_value(sample_yaml_data, "org.opencontainers.image.vendor", "NewVendor")
    assert metadata_utils.get_value(updated_data, "org.opencontainers.image.vendor") == "NewVendor"


def test_set_value_missing_key(sample_yaml_data):
    updated_data = metadata_utils.set_value(sample_yaml_data, "com.amd.aim.newKey", "NewValue", add_if_missing=True)
    assert metadata_utils.get_value(updated_data, "com.amd.aim.newKey") == "NewValue"


def test_set_value_missing_key_no_add(sample_yaml_data):
    updated_data = metadata_utils.set_value(sample_yaml_data, "com.amd.aim.newKey", "NewValue", add_if_missing=False)
    assert metadata_utils.get_value(updated_data, "com.amd.aim.newKey") is None


@pytest.mark.parametrize("assets_accelerator_path", ["instinct", "radeon"], indirect=True, ids=["instinct", "radeon"])
def test_get_model_variants(assets_accelerator_path):
    variants = metadata_utils.get_model_variants(
        Path(assets_accelerator_path) / "meta-llama" / "Llama-3.1-8B-Instruct" / "profiles"
    )
    assert len(variants) == 2
    assert variants[0] == "amd/Llama-3.1-8B-Instruct-FP8-KV"
    assert variants[1] == "meta-llama/Llama-3.1-8B-Instruct"


def test_add_recommended_deployments_basic(tmp_path):
    """Test basic functionality of adding recommended deployments."""
    # Create test metadata file
    metadata_dir = tmp_path / "test-org" / "test-model"
    metadata_dir.mkdir(parents=True)
    metadata_file = metadata_dir / "metadata.yaml"

    metadata = {
        "com": {
            "amd": {
                "aim": {
                    "model": {"canonicalName": "test-org/test-model"},
                }
            }
        }
    }

    yaml_utils.save_yaml(metadata, path=metadata_file)

    # Create test profiles directory
    profiles_dir = tmp_path / "test-org" / "test-model" / "profiles"
    profiles_dir.mkdir(parents=True)

    # Create test profile
    profile = {
        "aim_id": "test-org/test-model",
        "metadata": {
            "gpu": "MI300X",
            "gpu_count": 1,
            "metric": "latency",
            "precision": "fp16",
            "manual_selection_only": False,
        },
    }

    yaml_utils.save_yaml(profile, path=profiles_dir / "profile1.yaml")

    # Change to tmp_path so profiles are found
    original_cwd = os.getcwd()
    try:
        os.chdir(tmp_path)

        # Run the function
        metadata_utils._add_recommended_deployments_for_model(metadata_file)

        # Verify the result
        updated_metadata = yaml_utils.read_yaml(metadata_file)

        deployments = metadata_utils.get_value(updated_metadata, "com.amd.aim.model.recommendedDeployments")
        assert deployments is not None
        assert len(deployments) == 1
        assert deployments[0]["gpuModel"] == "MI300X"
        assert deployments[0]["gpuCount"] == 1
        assert deployments[0]["metric"] == "latency"
        assert deployments[0]["precision"] == "fp16"
    finally:
        os.chdir(original_cwd)


def test_add_recommended_deployments_multiple_profiles(tmp_path):
    """Test with multiple profiles for different GPU models and metrics."""
    # Create test metadata file
    metadata_dir = tmp_path / "test-org" / "test-model"
    metadata_dir.mkdir(parents=True)
    metadata_file = metadata_dir / "metadata.yaml"

    metadata = {
        "com": {
            "amd": {
                "aim": {
                    "model": {"canonicalName": "test-org/test-model"},
                }
            }
        }
    }

    yaml_utils.save_yaml(metadata, path=metadata_file)

    # Create test profiles directory
    profiles_dir = tmp_path / "test-org" / "test-model" / "profiles"
    profiles_dir.mkdir(parents=True)

    # Create multiple test profiles
    profiles = [
        {
            "aim_id": "test-org/test-model",
            "metadata": {"gpu": "MI300X", "gpu_count": 1, "metric": "latency", "precision": "fp16"},
        },
        {
            "aim_id": "test-org/test-model",
            "metadata": {"gpu": "MI300X", "gpu_count": 2, "metric": "latency", "precision": "int8"},
        },
        {
            "aim_id": "test-org/test-model",
            "metadata": {"gpu": "MI300X", "gpu_count": 1, "metric": "throughput", "precision": "fp16"},
        },
        {
            "aim_id": "test-org/test-model",
            "metadata": {"gpu": "MI325X", "gpu_count": 2, "metric": "latency", "precision": "int4"},
        },
    ]

    for i, profile in enumerate(profiles):
        yaml_utils.save_yaml(profile, path=profiles_dir / f"profile{i}.yaml")

    # Change to tmp_path so profiles are found
    original_cwd = os.getcwd()
    try:
        os.chdir(tmp_path)

        # Run the function
        metadata_utils._add_recommended_deployments_for_model(metadata_file)

        # Verify the result
        updated_metadata = yaml_utils.read_yaml(metadata_file)

        deployments = metadata_utils.get_value(updated_metadata, "com.amd.aim.model.recommendedDeployments")
        assert deployments is not None
        assert len(deployments) == 3  # MI300X latency, MI300X throughput, MI325X latency

        # Check MI300X latency - should select int8 with gpu_count=2 (lower precision over lower count)
        mi300x_latency = [d for d in deployments if d["gpuModel"] == "MI300X" and d["metric"] == "latency"][0]
        assert mi300x_latency["precision"] == "int8"
        assert mi300x_latency["gpuCount"] == 2

        # Check MI300X throughput
        mi300x_throughput = [d for d in deployments if d["gpuModel"] == "MI300X" and d["metric"] == "throughput"][0]
        assert mi300x_throughput["precision"] == "fp16"
        assert mi300x_throughput["gpuCount"] == 1

        # Check MI325X latency
        mi325x_latency = [d for d in deployments if d["gpuModel"] == "MI325X" and d["metric"] == "latency"][0]
        assert mi325x_latency["precision"] == "int4"
        assert mi325x_latency["gpuCount"] == 2
    finally:
        os.chdir(original_cwd)


def test_add_recommended_deployments_deprioritize_manual_selection_only(tmp_path):
    """Test that profiles with manual_selection_only=True are deprioritized."""
    # Create test metadata file
    metadata_dir = tmp_path / "test-org" / "test-model"
    metadata_dir.mkdir(parents=True)
    metadata_file = metadata_dir / "metadata.yaml"

    metadata = {
        "com": {
            "amd": {
                "aim": {
                    "model": {"canonicalName": "test-org/test-model"},
                }
            }
        }
    }

    yaml_utils.save_yaml(metadata, path=metadata_file)

    # Create test profiles directory
    profiles_dir = tmp_path / "test-org" / "test-model" / "profiles"
    profiles_dir.mkdir(parents=True)

    # Create test profiles (one manual only, one not)
    profiles = [
        {
            "aim_id": "test-org/test-model",
            "metadata": {
                "gpu": "MI300X",
                "gpu_count": 1,
                "metric": "latency",
                "precision": "fp16",
                "manual_selection_only": True,
            },
        },
        {
            "aim_id": "test-org/test-model",
            "metadata": {
                "gpu": "MI325X",
                "gpu_count": 1,
                "metric": "latency",
                "precision": "fp16",
                "manual_selection_only": True,
            },
        },
        {
            "aim_id": "test-org/test-model",
            "metadata": {"gpu": "MI300X", "gpu_count": 1, "metric": "latency", "precision": "int8"},
        },
    ]

    for i, profile in enumerate(profiles):
        yaml_utils.save_yaml(profile, path=profiles_dir / f"profile{i}.yaml")

    # Change to tmp_path so profiles are found
    original_cwd = os.getcwd()
    try:
        os.chdir(tmp_path)

        # Run the function
        metadata_utils._add_recommended_deployments_for_model(metadata_file)

        # Verify the result
        updated_metadata = yaml_utils.read_yaml(metadata_file)

        deployments = metadata_utils.get_value(updated_metadata, "com.amd.aim.model.recommendedDeployments")
        assert deployments is not None
        assert len(deployments) == 2
        # Should only have the int8 profile, not the manual_selection_only fp16
        assert deployments[0]["precision"] == "int8"
        assert deployments[1]["profileId"] == "profile1"
    finally:
        os.chdir(original_cwd)


def test_add_recommended_deployments_precision_priority(tmp_path):
    """Test that precision priority is correctly applied."""
    # Create test metadata file
    metadata_dir = tmp_path / "test-org" / "test-model"
    metadata_dir.mkdir(parents=True)
    metadata_file = metadata_dir / "metadata.yaml"

    metadata = {
        "com": {
            "amd": {
                "aim": {
                    "model": {"canonicalName": "test-org/test-model"},
                }
            }
        }
    }

    yaml_utils.save_yaml(metadata, path=metadata_file)

    # Create test profiles directory
    profiles_dir = tmp_path / "test-org" / "test-model" / "profiles"
    profiles_dir.mkdir(parents=True)

    # Create profiles with different precisions, same GPU count
    profiles = [
        {
            "aim_id": "test-org/test-model",
            "metadata": {"gpu": "MI300X", "gpu_count": 2, "metric": "latency", "precision": "fp32"},
        },
        {
            "aim_id": "test-org/test-model",
            "metadata": {"gpu": "MI300X", "gpu_count": 2, "metric": "latency", "precision": "int4"},
        },
        {
            "aim_id": "test-org/test-model",
            "metadata": {"gpu": "MI300X", "gpu_count": 2, "metric": "latency", "precision": "fp16"},
        },
    ]

    for i, profile in enumerate(profiles):
        yaml_utils.save_yaml(profile, path=profiles_dir / f"profile{i}.yaml")

    # Change to tmp_path so profiles are found
    original_cwd = os.getcwd()
    try:
        os.chdir(tmp_path)

        # Run the function
        metadata_utils._add_recommended_deployments_for_model(metadata_file)

        # Verify the result
        updated_metadata = yaml_utils.read_yaml(metadata_file)

        deployments = metadata_utils.get_value(updated_metadata, "com.amd.aim.model.recommendedDeployments")
        assert deployments is not None
        assert len(deployments) == 1
        # Should select int4 as it has the highest priority (lowest value)
        assert deployments[0]["precision"] == "int4"
    finally:
        os.chdir(original_cwd)


def test_add_recommended_deployments_no_canonical_name(tmp_path):
    """Test that function skips metadata without canonical name."""
    # Create test metadata file without canonical name
    metadata_dir = tmp_path / "test-org" / "test-model"
    metadata_dir.mkdir(parents=True)
    metadata_file = metadata_dir / "metadata.yaml"

    metadata = {"com": {"amd": {"aim": {"model": {}}}}}

    yaml_utils.save_yaml(metadata, path=metadata_file)

    # Run the function - should skip without error
    metadata_utils._add_recommended_deployments_for_model(metadata_file)

    # Verify nothing was added
    updated_metadata = yaml_utils.read_yaml(metadata_file)

    deployments = metadata_utils.get_value(updated_metadata, "com.amd.aim.model.recommendedDeployments")
    assert deployments is None


def test_add_recommended_deployments_no_profiles_dir(tmp_path):
    """Test that function handles missing profiles directory gracefully."""
    # Create test metadata file
    metadata_dir = tmp_path / "test-org" / "test-model"
    metadata_dir.mkdir(parents=True)
    metadata_file = metadata_dir / "metadata.yaml"

    metadata = {
        "com": {
            "amd": {
                "aim": {
                    "model": {"canonicalName": "test-org/test-model"},
                }
            }
        }
    }

    yaml_utils.save_yaml(metadata, path=metadata_file)

    # Don't create profiles directory

    # Change to tmp_path
    original_cwd = os.getcwd()
    try:
        os.chdir(tmp_path)

        # Run the function - should skip without error
        metadata_utils._add_recommended_deployments_for_model(metadata_file)

        # Verify nothing was added
        updated_metadata = yaml_utils.read_yaml(metadata_file)

        deployments = metadata_utils.get_value(updated_metadata, "com.amd.aim.model.recommendedDeployments")
        assert deployments is None
    finally:
        os.chdir(original_cwd)


def test_extract_all_keys(sample_yaml_data):
    keys = sorted(metadata_utils.extract_all_keys(sample_yaml_data))
    assert keys == [
        "com.amd.aim.description.full",
        "com.amd.aim.hfToken.required",
        "com.amd.aim.model.canonicalName",
        "com.amd.aim.model.publisher",
        "com.amd.aim.model.recommendedDeployments",
        "com.amd.aim.model.source",
        "com.amd.aim.model.tags",
        "com.amd.aim.model.variants",
        "com.amd.aim.release.notes",
        "com.amd.aim.title",
        "org.opencontainers.image.authors",
        "org.opencontainers.image.description",
        "org.opencontainers.image.documentation",
        "org.opencontainers.image.licenses",
        "org.opencontainers.image.source",
        "org.opencontainers.image.vendor",
    ]


def test_validate_metadata_nonexistent_directory():
    """Test validate_metadata with nonexistent metadata directory."""
    result = metadata_utils.validate_metadata("/nonexistent/path")

    assert result == {"total_count": 0, "valid_count": 0, "invalid_count": 0}


def test_validate_metadata_invalid_data(tmp_path):
    """Test validate_metadata with data that doesn't pass validation."""
    # Create a model-specific asset directory with an invalid metadata.yaml
    model_dir = tmp_path / "test-org" / "test-model"
    model_dir.mkdir(parents=True)

    # Create a metadata file with invalid structure
    test_file = model_dir / "metadata.yaml"
    yaml_utils.save_yaml({"test": "data"}, path=test_file)

    result = metadata_utils.validate_metadata(str(tmp_path))
    assert result == {"total_count": 1, "valid_count": 0, "invalid_count": 1}


def test_validate_metadata(test_metadata_path):
    """Test validate_metadata with test metadata files."""
    # Test with the test metadata directory that contains TinyLlama
    result = metadata_utils.validate_metadata(test_metadata_path)

    # Should find and validate the TinyLlama metadata file
    assert result["total_count"] == 4
    assert result["valid_count"] == 3
    assert result["invalid_count"] == 1


def test_validate_metadata_with_canonical_name_filter(test_metadata_path):
    """Test validate_metadata with canonical name filter using test data."""
    # Test filtering by the TinyLlama canonical name
    result = metadata_utils.validate_metadata(
        test_metadata_path, canonical_name="TinyLlama/TinyLlama-1.1B-Chat-v1.0_case1"
    )

    # Should only validate the TinyLlama file
    assert result["total_count"] == 1
    assert result["valid_count"] == 1
    assert result["invalid_count"] == 0


def test_validate_metadata_main_metadata_directory():
    """Test validate_metadata with the main metadata directory."""
    main_metadata_path = "assets/instinct"

    # Test with the main metadata directory containing all model metadata
    result = metadata_utils.validate_metadata(main_metadata_path)

    # Should find multiple metadata files and most should be valid
    assert result["total_count"] > 1
    assert result["valid_count"] > 0
    # Allow for some invalid files in case the main metadata has issues
    assert result["valid_count"] + result["invalid_count"] == result["total_count"]


def test_validate_incorrect_metadata_handling_for_specific_model(test_metadata_path):
    """Test validate_metadata with a specific model from test metadata."""
    # Test with a specific model that should exist in test metadata
    result = metadata_utils.validate_metadata(
        test_metadata_path, canonical_name="TinyLlama/TinyLlama-1.1B-Chat-v1.0_case2"
    )

    # Should find exactly one file for this model
    assert result["total_count"] == 1
    assert result["valid_count"] == 0
    assert result["invalid_count"] == 1


def test_validate_minimal_recommendations_metadata_handling_for_specific_model(test_metadata_path):
    """Test validate_metadata with a specific model from test metadata."""
    # Test with a specific model that should exist in test metadata
    result = metadata_utils.validate_metadata(
        test_metadata_path, canonical_name="TinyLlama/TinyLlama-1.1B-Chat-v1.0_case3"
    )

    # Should find exactly one file for this model
    assert result["total_count"] == 1
    assert result["valid_count"] == 1
    assert result["invalid_count"] == 0


def test_validate_profile_id_recommendations_metadata_handling_for_specific_model(test_metadata_path):
    """Test validate_metadata with a specific model from test metadata."""
    # Test with a specific model that should exist in test metadata
    result = metadata_utils.validate_metadata(
        test_metadata_path, canonical_name="TinyLlama/TinyLlama-1.1B-Chat-v1.0_case4"
    )

    # Should find exactly one file for this model
    assert result["total_count"] == 1
    assert result["valid_count"] == 1
    assert result["invalid_count"] == 0
