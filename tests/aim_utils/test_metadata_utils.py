# Copyright © Advanced Micro Devices, Inc., or its affiliates.
#
# SPDX-License-Identifier: MIT

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


def test_extract_all_keys(sample_yaml_data):
    keys = sorted(metadata_utils.extract_all_keys(sample_yaml_data))
    assert keys == [
        "com.amd.aim.description.full",
        "com.amd.aim.hfToken.required",
        "com.amd.aim.model.canonicalName",
        "com.amd.aim.model.publisher",
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
    with pytest.raises(ValueError):
        metadata_utils.validate_metadata("/nonexistent/path")


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
