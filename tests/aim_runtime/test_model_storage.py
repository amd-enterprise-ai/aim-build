# Copyright © Advanced Micro Devices, Inc., or its affiliates.
#
# SPDX-License-Identifier: MIT

"""
Tests for model storage backend functionality.
"""

import os
from unittest.mock import MagicMock, Mock, patch

import pytest

from aim_runtime.model_storage import (
    HuggingFaceStorageBackend,
    ModelStorageBackend,
    S3StorageBackend,
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
        assert not backend.supports_model("s3://my-bucket/model")
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
        result = registry.estimate_storage("s3://my-bucket/model")

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


class TestS3StorageBackend:
    """Test suite for S3 storage backend."""

    @pytest.fixture
    def backend(self):
        """Create an S3 storage backend instance with mocked boto3."""
        # Create a mock boto3 module
        mock_boto3 = MagicMock()
        mock_client = Mock()
        mock_boto3.client.return_value = mock_client

        from aim_runtime import model_storage

        with patch("aim_runtime.model_storage.BOTO3_AVAILABLE", True):
            with patch.dict(
                "sys.modules", {"boto3": mock_boto3, "botocore": MagicMock(), "botocore.exceptions": MagicMock()}
            ):
                # Inject boto3 into the module's namespace (create=True since it doesn't exist)
                with patch.object(model_storage, "boto3", mock_boto3, create=True):
                    backend = S3StorageBackend()
                    yield backend

    def test_supports_s3_model(self, backend):
        """Test that backend recognizes S3 model sources."""
        assert backend.supports_model("s3://my-bucket/models/llama")
        assert backend.supports_model("s3://bucket/path/to/model")
        assert not backend.supports_model("hf://meta-llama/Llama-3.1-8B-Instruct")
        assert not backend.supports_model("local:///path/to/model")

    def test_parse_s3_uri(self, backend):
        """Test parsing of S3 URIs."""
        bucket, key = backend._parse_s3_uri("s3://my-bucket/path/to/model")
        assert bucket == "my-bucket"
        assert key == "path/to/model"

        bucket, key = backend._parse_s3_uri("s3://bucket-name/model")
        assert bucket == "bucket-name"
        assert key == "model"

        bucket, key = backend._parse_s3_uri("s3://bucket-name")
        assert bucket == "bucket-name"
        assert key == ""

    def test_parse_s3_uri_invalid(self, backend):
        """Test parsing of invalid S3 URIs."""
        with pytest.raises(ValueError, match="Invalid S3 URI"):
            backend._parse_s3_uri("hf://not-s3")

        with pytest.raises(ValueError, match="missing bucket name"):
            backend._parse_s3_uri("s3://")

    def test_estimate_storage_success(self, backend):
        """Test successful storage estimation."""
        mock_paginator = Mock()
        backend._s3_client.get_paginator.return_value = mock_paginator

        # Mock paginated response with multiple objects
        mock_paginator.paginate.return_value = [
            {
                "Contents": [
                    {"Key": "path/to/model/file1.bin", "Size": 5 * 1024**3},  # 5 GB
                    {"Key": "path/to/model/file2.bin", "Size": 10 * 1024**3},  # 10 GB
                ]
            },
            {
                "Contents": [
                    {"Key": "path/to/model/file3.bin", "Size": 1 * 1024**3},  # 1 GB
                ]
            },
        ]

        result = backend.estimate_storage("s3://my-bucket/path/to/model")
        assert result == 16.0  # 5 + 10 + 1 = 16 GB

    def test_estimate_storage_empty(self, backend):
        """Test storage estimation with empty S3 path."""
        mock_paginator = Mock()
        backend._s3_client.get_paginator.return_value = mock_paginator
        mock_paginator.paginate.return_value = [{}]

        result = backend.estimate_storage("s3://my-bucket/empty")
        assert result is None

    def test_estimate_storage_client_unavailable(self):
        """Test storage estimation when S3 client is unavailable."""
        with patch("aim_runtime.model_storage.BOTO3_AVAILABLE", False):
            backend = S3StorageBackend()
            result = backend.estimate_storage("s3://my-bucket/model")
            assert result is None

    def test_download_success(self, backend, tmp_path):
        """Test successful model download."""
        mock_paginator = Mock()
        backend._s3_client.get_paginator.return_value = mock_paginator

        # Mock S3 objects
        mock_paginator.paginate.return_value = [
            {
                "Contents": [
                    {"Key": "models/llama/config.json", "Size": 1024},
                    {"Key": "models/llama/model.safetensors", "Size": 1024**3},
                ]
            }
        ]

        # Mock download_file
        backend._s3_client.download_file = Mock()

        cache_dir = str(tmp_path)
        result = backend.download("s3://my-bucket/models/llama", cache_dir)

        # Check that download_file was called twice
        assert backend._s3_client.download_file.call_count == 2

        # Check that result path exists and is correct
        assert result.startswith(cache_dir)
        assert "llama" in result

    def test_download_with_nested_structure(self, backend, tmp_path):
        """Test download with nested directory structure."""
        mock_paginator = Mock()
        backend._s3_client.get_paginator.return_value = mock_paginator

        # Mock S3 objects with nested structure
        mock_paginator.paginate.return_value = [
            {
                "Contents": [
                    {"Key": "models/llama/config.json", "Size": 1024},
                    {"Key": "models/llama/tokenizer/vocab.json", "Size": 2048},
                    {"Key": "models/llama/weights/model.safetensors", "Size": 1024**3},
                ]
            }
        ]

        # Mock download_file
        backend._s3_client.download_file = Mock()

        cache_dir = str(tmp_path)
        backend.download("s3://my-bucket/models/llama", cache_dir)

        # Check that download_file was called for all files
        assert backend._s3_client.download_file.call_count == 3

    def test_download_skip_directory_markers(self, backend, tmp_path):
        """Test that directory markers (keys ending with /) are skipped."""
        mock_paginator = Mock()
        backend._s3_client.get_paginator.return_value = mock_paginator

        # Mock S3 objects including directory marker
        mock_paginator.paginate.return_value = [
            {
                "Contents": [
                    {"Key": "models/llama/", "Size": 0},  # Directory marker
                    {"Key": "models/llama/config.json", "Size": 1024},
                ]
            }
        ]

        # Mock download_file
        backend._s3_client.download_file = Mock()

        cache_dir = str(tmp_path)
        backend.download("s3://my-bucket/models/llama", cache_dir)

        # Check that download_file was called only once (skipped directory marker)
        assert backend._s3_client.download_file.call_count == 1

    def test_download_skip_existing_files(self, backend, tmp_path):
        """Test that existing files with matching sizes are skipped."""
        mock_paginator = Mock()
        backend._s3_client.get_paginator.return_value = mock_paginator

        # Mock S3 objects
        mock_paginator.paginate.return_value = [
            {
                "Contents": [
                    {"Key": "models/llama/config.json", "Size": 1024},
                    {"Key": "models/llama/model.safetensors", "Size": 2048},
                ]
            }
        ]

        # Create cache directory and pre-populate one file with matching size
        cache_dir = str(tmp_path)
        # Note: _get_model_dir_name_from_s3_uri returns "models/llama" (last 2 components)
        model_dir = os.path.join(cache_dir, "models", "llama")
        os.makedirs(model_dir, exist_ok=True)

        # Create config.json with matching size (1024 bytes)
        existing_file = os.path.join(model_dir, "config.json")
        with open(existing_file, "wb") as f:
            f.write(b"x" * 1024)  # Exactly 1024 bytes

        # Mock download_file
        backend._s3_client.download_file = Mock()

        result = backend.download("s3://my-bucket/models/llama", cache_dir)

        # Check that download_file was called only once (skipped existing file)
        assert backend._s3_client.download_file.call_count == 1

        # Verify it was called for the non-existing file only
        call_args = backend._s3_client.download_file.call_args[0]
        assert call_args[1] == "models/llama/model.safetensors"
        assert result == model_dir

    def test_download_redownload_size_mismatch(self, backend, tmp_path):
        """Test that files with mismatched sizes are re-downloaded."""
        mock_paginator = Mock()
        backend._s3_client.get_paginator.return_value = mock_paginator

        # Mock S3 objects
        mock_paginator.paginate.return_value = [
            {
                "Contents": [
                    {"Key": "models/llama/config.json", "Size": 2048},
                ]
            }
        ]

        # Create cache directory with file that has wrong size
        cache_dir = str(tmp_path)
        # Note: _get_model_dir_name_from_s3_uri returns "models/llama" (last 2 components)
        model_dir = os.path.join(cache_dir, "models", "llama")
        os.makedirs(model_dir, exist_ok=True)

        # Create config.json with DIFFERENT size (1024 bytes instead of 2048)
        existing_file = os.path.join(model_dir, "config.json")
        with open(existing_file, "wb") as f:
            f.write(b"x" * 1024)  # Wrong size

        # Mock download_file
        backend._s3_client.download_file = Mock()

        backend.download("s3://my-bucket/models/llama", cache_dir)

        # Check that download_file was called (file needs re-download due to size mismatch)
        assert backend._s3_client.download_file.call_count == 1

    def test_download_no_files(self, backend, tmp_path):
        """Test download with no files found."""
        mock_paginator = Mock()
        backend._s3_client.get_paginator.return_value = mock_paginator
        mock_paginator.paginate.return_value = [{}]

        cache_dir = str(tmp_path)
        with pytest.raises(RuntimeError, match="No files found"):
            backend.download("s3://my-bucket/empty", cache_dir)

    def test_download_client_unavailable(self):
        """Test download when S3 client is unavailable."""
        with patch("aim_runtime.model_storage.BOTO3_AVAILABLE", False):
            backend = S3StorageBackend()
            with pytest.raises(RuntimeError, match="S3 client not available"):
                backend.download("s3://my-bucket/model", "/tmp/cache")

    def test_download_client_error(self, backend, tmp_path):
        """Test download handling of client errors."""
        mock_paginator = Mock()
        backend._s3_client.get_paginator.return_value = mock_paginator

        # Mock S3 objects
        mock_paginator.paginate.return_value = [
            {
                "Contents": [
                    {"Key": "models/llama/config.json", "Size": 1024},
                ]
            }
        ]

        # Mock download_file to raise an error
        mock_error = Exception("NoSuchKey: The specified key does not exist.")
        backend._s3_client.download_file = Mock(side_effect=mock_error)

        cache_dir = str(tmp_path)
        with pytest.raises(RuntimeError, match="Failed to download model from S3"):
            backend.download("s3://my-bucket/models/llama", cache_dir)

    def test_initialization_without_boto3(self):
        """Test initialization when boto3 is not available."""
        with patch("aim_runtime.model_storage.BOTO3_AVAILABLE", False):
            backend = S3StorageBackend()
            assert backend._s3_client is None

    def test_get_model_dir_name_from_s3_uri(self, backend):
        """Test extraction of model directory name from S3 URI."""
        # Test with full path (3+ components) - should return last 2
        result = backend._get_model_dir_name_from_s3_uri("s3://my-bucket/models/org/llama-3.1-8b")
        assert result == os.path.join("org", "llama-3.1-8b")

        # Test with 2 components - should return both
        result = backend._get_model_dir_name_from_s3_uri("s3://my-bucket/org/model")
        assert result == os.path.join("org", "model")

        # Test with single component - should return just the component
        result = backend._get_model_dir_name_from_s3_uri("s3://my-bucket/model-name")
        assert result == "model-name"

        # Test with just bucket - should return bucket name
        result = backend._get_model_dir_name_from_s3_uri("s3://my-bucket")
        assert result == "my-bucket"

        # Test with trailing slash
        result = backend._get_model_dir_name_from_s3_uri("s3://my-bucket/models/org/llama/")
        assert result == os.path.join("org", "llama")

    def test_initialization_with_custom_endpoint(self):
        """Test initialization with custom S3 endpoint."""
        # Create a mock boto3 module
        mock_boto3 = MagicMock()
        mock_client = Mock()
        mock_boto3.client.return_value = mock_client

        from aim_runtime import model_storage

        with patch("aim_runtime.model_storage.BOTO3_AVAILABLE", True):
            with patch.dict(
                "sys.modules", {"boto3": mock_boto3, "botocore": MagicMock(), "botocore.exceptions": MagicMock()}
            ):
                with patch.object(model_storage, "boto3", mock_boto3, create=True):
                    with patch.dict(os.environ, {"AWS_ENDPOINT_URL": "https://minio.example.com"}):
                        S3StorageBackend()

                        # Check that boto3.client was called with custom endpoint
                        mock_boto3.client.assert_called_once()
                        call_kwargs = mock_boto3.client.call_args[1]
                        assert call_kwargs["endpoint_url"] == "https://minio.example.com"


class TestStorageBackendRegistryWithS3:
    """Test storage backend registry with S3 support."""

    @pytest.fixture
    def registry(self):
        """Create a storage backend registry instance."""
        return StorageBackendRegistry()

    def test_s3_backend_registered(self, registry):
        """Test that S3 backend is registered by default."""
        backend = registry.get_backend("s3://my-bucket/model")
        assert isinstance(backend, S3StorageBackend)

    def test_get_correct_backend_for_source(self, registry):
        """Test that registry returns correct backend for different sources."""
        hf_backend = registry.get_backend("hf://meta-llama/Llama-3.1-8B-Instruct")
        assert isinstance(hf_backend, HuggingFaceStorageBackend)

        s3_backend = registry.get_backend("s3://my-bucket/model")
        assert isinstance(s3_backend, S3StorageBackend)


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
