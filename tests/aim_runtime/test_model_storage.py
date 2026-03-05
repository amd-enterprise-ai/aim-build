# Copyright © Advanced Micro Devices, Inc., or its affiliates.
#
# SPDX-License-Identifier: MIT

"""
Tests for model storage backend functionality.
"""


import pytest

from aim_runtime.model_storage import (
    HuggingFaceStorageBackend,
    ModelStorageBackend,
    StorageBackendRegistry,
)


class TestHuggingFaceStorageBackend:
    """Test suite for HuggingFace storage backend."""

    @pytest.fixture
    def backend(self):
        """Create a HuggingFace storage backend instance."""
        return HuggingFaceStorageBackend()

    def test_supports_huggingface_model(self, backend):
        """Test that backend recognizes HuggingFace model sources."""
        assert backend.supports_model("hf://meta-llama/Llama-3.1-8B-Instruct")
        assert backend.supports_model("hf://Qwen/Qwen3-32B")
        assert not backend.supports_model("local:///path/to/model")

    def test_extract_model_id(self, backend):
        """Test extraction of model ID from HuggingFace URI."""
        assert backend._extract_model_id("hf://meta-llama/Llama-3.1-8B-Instruct") == "meta-llama/Llama-3.1-8B-Instruct"
        assert backend._extract_model_id("meta-llama/Llama-3.1-8B-Instruct") == "meta-llama/Llama-3.1-8B-Instruct"

    def test_estimate_storage_returns_none_when_api_fails(self, backend, monkeypatch):
        """Test that storage estimation returns None when API fails."""

        # Mock _estimate_from_api to return None
        def mock_estimate_from_api(model_id):
            return None

        monkeypatch.setattr(backend, "_estimate_from_api", mock_estimate_from_api)

        result = backend.estimate_storage("meta-llama/Llama-3.1-8B-Instruct")

        assert result is None

    def test_estimate_storage_with_hf_prefix(self, backend, monkeypatch):
        """Test storage estimation with hf:// prefix."""

        # Mock _estimate_from_api to return None
        def mock_estimate_from_api(model_id):
            return None

        monkeypatch.setattr(backend, "_estimate_from_api", mock_estimate_from_api)

        result = backend.estimate_storage("hf://meta-llama/Llama-3.1-8B-Instruct")

        assert result is None

    def test_download_success(self, backend, monkeypatch):
        """Test successful model download."""

        def mock_snapshot_download(
            repo_id, token, resume_download, allow_patterns, ignore_patterns, cache_dir=None, local_dir=None
        ):
            target_dir = local_dir if local_dir else cache_dir
            return f"{target_dir}/snapshots/abc123"

        from aim_runtime import model_storage

        monkeypatch.setattr(model_storage, "snapshot_download", mock_snapshot_download)

        result = backend.download("meta-llama/Llama-3.1-8B-Instruct", "/tmp/cache")
        # When using local_dir (default), the model_id is appended to cache_dir
        assert result == "/tmp/cache/meta-llama/Llama-3.1-8B-Instruct/snapshots/abc123"

    def test_download_with_hf_prefix(self, backend, monkeypatch):
        """Test download with hf:// prefix."""

        def mock_snapshot_download(
            repo_id, token, resume_download, allow_patterns, ignore_patterns, cache_dir=None, local_dir=None
        ):
            target_dir = local_dir if local_dir else cache_dir
            return f"{target_dir}/snapshots/abc123"

        from aim_runtime import model_storage

        monkeypatch.setattr(model_storage, "snapshot_download", mock_snapshot_download)

        result = backend.download("hf://meta-llama/Llama-3.1-8B-Instruct", "/tmp/cache")
        # When using local_dir (default), the model_id is appended to cache_dir
        assert result == "/tmp/cache/meta-llama/Llama-3.1-8B-Instruct/snapshots/abc123"

    def test_download_failure(self, backend, monkeypatch):
        """Test download failure handling."""

        def mock_snapshot_download(repo_id, cache_dir, token, resume_download, allow_patterns, ignore_patterns):
            raise Exception("Network error")

        from aim_runtime import model_storage

        monkeypatch.setattr(model_storage, "snapshot_download", mock_snapshot_download)

        with pytest.raises(RuntimeError, match="Failed to download model"):
            backend.download("meta-llama/Llama-3.1-8B-Instruct", "/tmp/cache")


class TestStorageBackendRegistry:
    """Test suite for storage backend registry."""

    @pytest.fixture
    def registry(self):
        """Create a storage backend registry instance."""
        return StorageBackendRegistry()

    def test_default_backend_registered(self, registry):
        """Test that HuggingFace backend is registered by default."""
        backend = registry.get_backend("hf://meta-llama/Llama-3.1-8B-Instruct")
        assert isinstance(backend, HuggingFaceStorageBackend)

    def test_get_backend_for_unsupported_source(self, registry):
        """Test that unsupported sources return None."""
        backend = registry.get_backend("unsupported://my-bucket/model")
        assert backend is None

    def test_estimate_storage_with_registry(self, registry, monkeypatch):
        """Test storage estimation through registry."""

        # Mock the backend's _estimate_from_api to return test data
        def mock_estimate_from_api(model_id):
            return 16.0

        backend = registry.get_backend("hf://meta-llama/Llama-3.1-8B-Instruct")
        monkeypatch.setattr(backend, "_estimate_from_api", mock_estimate_from_api)

        result = registry.estimate_storage("hf://meta-llama/Llama-3.1-8B-Instruct")

        assert result == 16.0

    def test_estimate_storage_unsupported_source(self, registry):
        """Test storage estimation for unsupported source."""
        result = registry.estimate_storage("protocol://my-bucket/model")

        assert result is None

    def test_register_custom_backend(self, registry):
        """Test registering a custom backend."""

        class CustomBackend(ModelStorageBackend):
            def supports_model(self, model_source: str) -> bool:
                return model_source.startswith("custom://")

            def estimate_storage(self, model_id: str):
                return 10.0

            def download(self, model_id: str, cache_dir: str) -> str:
                return f"{cache_dir}/{model_id}"

        custom_backend = CustomBackend()
        registry.register(custom_backend)

        backend = registry.get_backend("custom://my-model")
        assert isinstance(backend, CustomBackend)

        result = registry.estimate_storage("custom://my-model")
        assert result == 10.0

    def test_download_through_registry(self, registry, monkeypatch):
        """Test downloading through registry.

        Note: By default, HuggingFace backend uses local_dir mode to download directly
        to the target directory (not HF cache structure). The model_id is appended to
        the cache_dir to create a model-specific subdirectory.
        """

        def mock_snapshot_download(
            repo_id, token, resume_download, allow_patterns, ignore_patterns, cache_dir=None, local_dir=None
        ):
            # By default, local_dir mode is used
            target_dir = local_dir if local_dir else (cache_dir if cache_dir else "/default/hf/cache")
            return f"{target_dir}/snapshots/abc123"

        from aim_runtime import model_storage

        monkeypatch.setattr(model_storage, "snapshot_download", mock_snapshot_download)

        # Registry uses local_dir mode by default, with model_id appended to cache_dir
        result = registry.download("hf://meta-llama/Llama-3.1-8B-Instruct", "/tmp/cache")
        assert result == "/tmp/cache/meta-llama/Llama-3.1-8B-Instruct/snapshots/abc123"

    def test_download_unsupported_source(self, registry):
        """Test download with unsupported source."""

        with pytest.raises(ValueError, match="No storage backend available"):
            registry.download("unsupported://my-bucket/model", "/tmp/cache")


class TestAPIBasedEstimation:
    """Test suite for API-based storage estimation."""

    @pytest.fixture
    def backend(self):
        """Create a HuggingFace storage backend."""
        return HuggingFaceStorageBackend()

    def test_estimate_from_api_with_mock(self, backend, monkeypatch):
        """Test API-based estimation with mocked HuggingFace API."""

        # Mock the HfApi
        class MockSibling:
            def __init__(self, size):
                self.size = size

        class MockModelInfo:
            def __init__(self):
                self.siblings = [
                    MockSibling(5 * 1024**3),  # 5 GB
                    MockSibling(10 * 1024**3),  # 10 GB
                    MockSibling(1 * 1024**3),  # 1 GB
                ]

        class MockHfApi:
            def __init__(self, token=None):
                pass

            def model_info(self, model_id, files_metadata=False):
                return MockModelInfo()

        # Patch HfApi in the model_storage module
        from aim_runtime import model_storage

        monkeypatch.setattr(model_storage, "HfApi", MockHfApi)

        # Test the API estimation
        api_result = backend._estimate_from_api("test-model")
        assert api_result == 16.0  # 5 + 10 + 1 = 16 GB

    def test_estimate_storage_uses_api(self, backend, monkeypatch):
        """Test that estimate_storage uses API."""

        def mock_estimate_from_api(model_id):
            return 16.0

        monkeypatch.setattr(backend, "_estimate_from_api", mock_estimate_from_api)
        result = backend.estimate_storage("meta-llama/Llama-3.1-8B-Instruct")
        assert result == 16.0

    def test_estimate_storage_returns_none_when_api_fails(self, backend, monkeypatch):
        # Mock the _estimate_from_api method to return None (API failure)
        def mock_estimate_from_api(model_id):
            return None

        monkeypatch.setattr(backend, "_estimate_from_api", mock_estimate_from_api)
        result = backend.estimate_storage("meta-llama/Llama-3.1-8B-Instruct")
        assert result is None

        """Test that API results are cached."""
        call_count = 0

        def counting_estimate(model_id):
            nonlocal call_count
            call_count += 1
            # Check cache first (mimicking real behavior)
            if model_id in backend._cache:
                return backend._cache[model_id]
            # Simulate API call
            result = 16.0
            backend._cache[model_id] = result
            return result

        monkeypatch.setattr(backend, "_estimate_from_api", counting_estimate)
        # First call should hit the "API"
        result1 = backend.estimate_storage("test-model")
        assert call_count == 1
        # Second call should use cache
        result2 = backend.estimate_storage("test-model")
        assert call_count == 2  # _estimate_from_api is called but returns cached result
        assert result1 == result2

    def test_api_called_each_time(self, backend, monkeypatch):
        """Test that API is called each time (no caching)."""
        call_count = 0

        def counting_estimate(model_id):
            nonlocal call_count
            call_count += 1
            # Simulate API call
            result = 16.0
            return result

        monkeypatch.setattr(backend, "_estimate_from_api", counting_estimate)
        # First call
        result1 = backend.estimate_storage("test-model")
        assert call_count == 1
        # Second call should call API again (no caching)
        result2 = backend.estimate_storage("test-model")
        assert call_count == 2
        assert result1 == result2


class TestEdgeCases:
    """Test edge cases and special scenarios."""

    @pytest.fixture
    def backend(self):
        """Create a HuggingFace storage backend instance."""
        return HuggingFaceStorageBackend()

    def test_empty_model_id(self, backend, monkeypatch):
        """Test handling of empty model ID."""

        # Mock API to return None
        def mock_estimate_from_api(model_id):
            return None

        monkeypatch.setattr(backend, "_estimate_from_api", mock_estimate_from_api)

        result = backend.estimate_storage("")
        assert result is None

    def test_model_id_extraction_with_prefix(self, backend, monkeypatch):
        """Test that hf:// prefix is properly extracted."""

        # Mock API to return None
        def mock_estimate_from_api(model_id):
            return None

        monkeypatch.setattr(backend, "_estimate_from_api", mock_estimate_from_api)

        # Both should behave the same
        result1 = backend.estimate_storage("hf://test-model")
        result2 = backend.estimate_storage("test-model")
        assert result1 is None
        assert result2 is None
        assert result1 == result2
