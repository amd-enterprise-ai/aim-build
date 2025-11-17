# Copyright © Advanced Micro Devices, Inc., or its affiliates.
#
# SPDX-License-Identifier: MIT

"""Tests for the ModelCacheResolver class."""

import os

import pytest

from aim_runtime.model_cache_resolver import ModelCacheResolver, ResolvedModelPath


@pytest.fixture
def temp_cache_dir(tmp_path):
    """Create a temporary cache directory for testing."""
    return str(tmp_path / "model-cache")


@pytest.fixture
def cache_resolver(temp_cache_dir):
    """Create a ModelCacheResolver instance with temporary cache."""
    return ModelCacheResolver(temp_cache_dir)


def test_resolver_initialization(cache_resolver, temp_cache_dir):
    """Test that ModelCacheResolver initializes correctly."""
    assert cache_resolver.cache_dir == temp_cache_dir
    assert cache_resolver.hf_hub_cache_dir == os.path.join(temp_cache_dir, "hub")


def test_resolve_model_path_hf_cache_exists(cache_resolver, temp_cache_dir):
    """Test resolution when HF Hub cache directory exists (but no local dir)."""
    # Create HF Hub cache directory but no local dir
    hf_cache_dir = os.path.join(temp_cache_dir, "hub")
    os.makedirs(hf_cache_dir, exist_ok=True)

    model_id = "meta-llama/Llama-3.1-8B-Instruct"
    result = cache_resolver.resolve_model_path(model_id)

    # Should return model_id (HF will handle cache lookup transparently)
    assert result is not None
    assert result.path == model_id
    assert result.is_local_dir is False
    assert result.model_id == model_id


def test_resolve_model_path_local_dir_exists(cache_resolver, temp_cache_dir):
    """Test resolution when local directory format exists."""
    # Create local directory structure
    model_id = "meta-llama/Llama-3.1-8B-Instruct"
    local_path = os.path.join(temp_cache_dir, "meta-llama", "Llama-3.1-8B-Instruct")
    os.makedirs(local_path, exist_ok=True)

    result = cache_resolver.resolve_model_path(model_id)

    assert result is not None
    assert result.path == local_path
    assert result.is_local_dir is True
    assert result.model_id == model_id


def test_resolve_model_path_local_dir_takes_precedence(cache_resolver, temp_cache_dir):
    """Test that local directory takes precedence over HF cache."""
    # Create both HF cache and local directory
    hf_cache_dir = os.path.join(temp_cache_dir, "hub")
    os.makedirs(hf_cache_dir, exist_ok=True)

    model_id = "meta-llama/Llama-3.1-8B-Instruct"
    local_path = os.path.join(temp_cache_dir, "meta-llama", "Llama-3.1-8B-Instruct")
    os.makedirs(local_path, exist_ok=True)

    result = cache_resolver.resolve_model_path(model_id)

    # Should use local directory format (takes precedence)
    assert result is not None
    assert result.path == local_path
    assert result.is_local_dir is True


def test_resolve_model_path_no_cache_exists(cache_resolver):
    """Test resolution when no cache exists (should return model_id for download)."""
    model_id = "meta-llama/Llama-3.1-8B-Instruct"
    result = cache_resolver.resolve_model_path(model_id)

    assert result is not None
    assert result.path == model_id  # Should return model_id to allow HF download
    assert result.is_local_dir is False
    assert result.model_id == model_id


def test_resolve_model_path_invalid_model_id(cache_resolver):
    """Test resolution with invalid model_id format."""
    invalid_model_id = "invalid-model-id-without-slash"
    result = cache_resolver.resolve_model_path(invalid_model_id)

    # Should still return model_id for HF to handle
    assert result is not None
    assert result.path == invalid_model_id
    assert result.is_local_dir is False


def test_get_local_dir_path_valid(cache_resolver, temp_cache_dir):
    """Test local directory path generation with valid model_id."""
    model_id = "meta-llama/Llama-3.1-8B-Instruct"
    expected_path = os.path.join(temp_cache_dir, "meta-llama", "Llama-3.1-8B-Instruct")

    result = cache_resolver._get_local_dir_path(model_id)

    assert result == expected_path


def test_get_local_dir_path_invalid(cache_resolver):
    """Test local directory path generation with invalid model_id."""
    invalid_model_id = "invalid-model-id"
    result = cache_resolver._get_local_dir_path(invalid_model_id)

    assert result is None


def test_check_local_dir_exists_true(cache_resolver, temp_cache_dir):
    """Test check_local_dir_exists when directory exists."""
    model_id = "meta-llama/Llama-3.1-8B-Instruct"
    local_path = os.path.join(temp_cache_dir, "meta-llama", "Llama-3.1-8B-Instruct")
    os.makedirs(local_path, exist_ok=True)

    assert cache_resolver.check_local_dir_exists(model_id) is True


def test_check_local_dir_exists_false(cache_resolver):
    """Test check_local_dir_exists when directory doesn't exist."""
    model_id = "meta-llama/Llama-3.1-8B-Instruct"
    assert cache_resolver.check_local_dir_exists(model_id) is False


def test_check_hf_cache_exists_true(cache_resolver, temp_cache_dir):
    """Test check_hf_cache_exists when HF cache exists."""
    hf_cache_dir = os.path.join(temp_cache_dir, "hub")
    os.makedirs(hf_cache_dir, exist_ok=True)

    assert cache_resolver.check_hf_cache_exists() is True


def test_check_hf_cache_exists_false(cache_resolver):
    """Test check_hf_cache_exists when HF cache doesn't exist."""
    assert cache_resolver.check_hf_cache_exists() is False


def test_resolved_model_path_dataclass():
    """Test ResolvedModelPath dataclass creation."""
    resolved = ResolvedModelPath(
        path="/workspace/model-cache/meta-llama/Llama-3.1-8B-Instruct",
        is_local_dir=True,
        model_id="meta-llama/Llama-3.1-8B-Instruct",
    )

    assert resolved.path == "/workspace/model-cache/meta-llama/Llama-3.1-8B-Instruct"
    assert resolved.is_local_dir is True
    assert resolved.model_id == "meta-llama/Llama-3.1-8B-Instruct"
