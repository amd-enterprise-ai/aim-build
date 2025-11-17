# Copyright © Advanced Micro Devices, Inc., or its affiliates.
#
# SPDX-License-Identifier: MIT

"""
Model Storage Backend Management

This module provides abstraction for different model storage backends (e.g., HuggingFace Hub)
and functionality to estimate storage requirements for model downloads.
"""

import logging
import os
from abc import ABC, abstractmethod
from typing import Optional

from huggingface_hub import HfApi, snapshot_download

try:
    import boto3

    BOTO3_AVAILABLE = True
except ImportError:
    BOTO3_AVAILABLE = False

logger = logging.getLogger(__name__)


class ModelStorageBackend(ABC):
    """
    Abstract base class for model storage backends.

    A storage backend handles downloading models and estimating storage requirements
    for different model sources (HuggingFace Hub, local files, S3, etc.).
    """

    @abstractmethod
    def estimate_storage(self, model_id: str) -> Optional[float]:
        """
        Estimate storage requirements for a model.

        Args:
            model_id: Model identifier

        Returns:
            Estimated size in GB (float), or None if unavailable
        """

    @abstractmethod
    def supports_model(self, model_source: str) -> bool:
        """
        Check if this backend supports the given model source.

        Args:
            model_source: Model source URI (e.g., "hf://meta-llama/Llama-3.1-8B")

        Returns:
            True if this backend can handle the model source
        """

    @abstractmethod
    def download(self, model_id: str, cache_dir: str = None) -> str:
        """
        Download a model to the cache directory.

        Args:
            model_id: Model identifier
            cache_dir: Optional directory to cache the downloaded model.
                      If None, uses backend's default cache location
                      (backend-specific, may be determined by environment variables).

        Returns:
            Path to the downloaded model

        Raises:
            NotImplementedError: If download is not supported
            RuntimeError: If download fails
        """


class HuggingFaceStorageBackend(ModelStorageBackend):
    """
    HuggingFace Hub storage backend.

    Handles models hosted on HuggingFace Hub and provides storage estimation
    based on model size heuristics.
    """

    def __init__(self):
        """Initialize the HuggingFace storage backend."""
        self._cache = {}  # Simple in-memory cache for API calls
        self._hf_token = os.environ.get("HF_TOKEN")  # Get token if available for gated models

    def supports_model(self, model_source: str) -> bool:
        """
        Check if this is a HuggingFace Hub model.

        Args:
            model_source: Model source URI

        Returns:
            True if the model source is HuggingFace Hub (starts with "hf://")
        """
        return model_source.startswith("hf://")

    def _extract_model_id(self, model_source: str) -> str:
        """
        Extract HuggingFace model ID from source URI.

        Args:
            model_source: Model source URI (e.g., "hf://meta-llama/Llama-3.1-8B")

        Returns:
            Model ID without the "hf://" prefix
        """
        if model_source.startswith("hf://"):
            return model_source[5:]
        return model_source

    def _estimate_from_api(self, model_id: str) -> Optional[float]:
        """
        Estimate storage requirements by querying HuggingFace Hub API.

        This method fetches the actual model files metadata from HuggingFace Hub
        to calculate precise storage requirements. Results are cached to avoid
        redundant API calls.

        Args:
            model_id: HuggingFace model ID (without "hf://" prefix)

        Returns:
            Estimated size in GB (float), or None if API call fails
        """
        # Check cache first
        if model_id in self._cache:
            logger.debug(f"Using cached API result for model: {model_id}")
            return self._cache[model_id]

        # Check if HuggingFace Hub is available
        if HfApi is None:
            logger.debug("huggingface_hub library not available, cannot use API estimation")
            return None

        try:
            logger.debug(f"Querying HuggingFace Hub API for model: {model_id}")

            # Create API client with token if available
            api = HfApi(token=self._hf_token)

            # Get model info including file list
            try:
                model_info = api.model_info(model_id, files_metadata=True)
            except Exception as e:
                logger.debug(f"Failed to get model info from API: {e}")
                return None

            # Calculate total size from all files
            total_size = 0

            if hasattr(model_info, "siblings") and model_info.siblings:
                for file_info in model_info.siblings:
                    if hasattr(file_info, "size") and file_info.size:
                        file_size = file_info.size
                        total_size += file_size

            if total_size == 0:
                logger.debug(f"No size information available from API for model: {model_id}")
                return None

            # Convert bytes to GB
            size_gb = round(total_size / (1024**3), 2)
            # Cache the result
            self._cache[model_id] = size_gb
            logger.debug(f"API result for {model_id}: {size_gb}GB total")
            return size_gb

        except Exception as e:
            logger.debug(f"Failed to estimate from API: {e}")
            return None

    def estimate_storage(self, model_id: str) -> Optional[float]:
        """
        Estimate storage requirements using HuggingFace Hub API.

        Args:
            model_id: HuggingFace model ID (with or without hf:// prefix)

        Returns:
            Estimated size in GB (float), or None if unavailable
        """
        model_id = self._extract_model_id(model_id)
        api_result = self._estimate_from_api(model_id)
        return api_result

    def download(self, model_id: str, cache_dir: str = None, use_hf_cache: bool = False) -> str:
        """
        Download a model from HuggingFace Hub using snapshot_download.

        This downloads the complete model repository including all files
        (weights, config, tokenizer, etc.) to the cache directory.

        Args:
            model_id: HuggingFace model ID (with or without hf:// prefix)
            cache_dir: Optional directory to cache the downloaded model.
                      If None, uses HuggingFace Hub's default cache location
                      (determined by HF_HOME or platform defaults).
            use_hf_cache: If True, uses HuggingFace's cache structure (cache_dir parameter).
                         If False (default), downloads directly to local directory (local_dir parameter).

        Returns:
            Path to the downloaded model directory

        Raises:
            RuntimeError: If download fails
        """
        model_id = self._extract_model_id(model_id)

        if use_hf_cache:
            logger.info(f"Downloading model {model_id} using HuggingFace cache structure")
            if cache_dir:
                logger.info(f"Cache directory: {cache_dir}")
        else:
            logger.info(f"Downloading model {model_id} to local directory")
            if cache_dir:
                logger.info(f"Local directory: {cache_dir}")

        try:
            # Use snapshot_download to download the entire model repository
            # This function handles retries, partial downloads, and caching automatically
            if use_hf_cache:
                # Use cache_dir parameter for HF's cache structure
                downloaded_path = snapshot_download(
                    repo_id=model_id,
                    cache_dir=cache_dir,
                    token=self._hf_token,
                    resume_download=True,
                    allow_patterns=None,
                    ignore_patterns=None,
                )
            else:
                # Use local_dir parameter to download directly to the target directory
                # Append org/model to cache_dir to create model-specific subdirectory
                if cache_dir:
                    local_dir_path = os.path.join(cache_dir, model_id)
                else:
                    local_dir_path = None

                downloaded_path = snapshot_download(
                    repo_id=model_id,
                    local_dir=local_dir_path,
                    token=self._hf_token,
                    resume_download=True,
                    allow_patterns=None,
                    ignore_patterns=None,
                )

            logger.info(f"Successfully downloaded model to: {downloaded_path}")
            return downloaded_path

        except Exception as e:
            error_msg = f"Failed to download model {model_id}: {e}"
            logger.error(error_msg)
            raise RuntimeError(error_msg) from e


class S3StorageBackend(ModelStorageBackend):
    """
    S3 storage backend for downloading models from AWS S3 or S3-compatible storage.

    Handles models stored in S3 buckets. Model source format: "s3://bucket-name/path/to/model"

    Environment variables:
        AWS_ACCESS_KEY_ID: AWS access key (optional if using IAM role)
        AWS_SECRET_ACCESS_KEY: AWS secret key (optional if using IAM role)
        AWS_DEFAULT_REGION: AWS region (default: us-east-1)
        AWS_ENDPOINT_URL: Custom S3 endpoint for S3-compatible storage (optional)
    """

    def __init__(self):
        """Initialize the S3 storage backend."""
        if not BOTO3_AVAILABLE:
            logger.warning("boto3 library not available. S3 storage backend will not be functional.")
            self._s3_client = None
            return

        # Get AWS configuration from environment
        endpoint_url = os.environ.get("AWS_ENDPOINT_URL")
        region = os.environ.get("AWS_DEFAULT_REGION", "us-east-1")

        # Create S3 client
        try:
            self._s3_client = boto3.client(
                "s3",
                region_name=region,
                endpoint_url=endpoint_url,
                # AWS credentials are automatically loaded from environment or IAM role
            )
            logger.debug(f"Initialized S3 client with region: {region}")
            if endpoint_url:
                logger.debug(f"Using custom S3 endpoint: {endpoint_url}")
        except Exception as e:
            logger.warning(f"Failed to initialize S3 client: {e}")
            self._s3_client = None

    def supports_model(self, model_source: str) -> bool:
        """
        Check if this is an S3 model source.

        Args:
            model_source: Model source URI

        Returns:
            True if the model source is S3 (starts with "s3://")
        """
        return model_source.startswith("s3://")

    def _parse_s3_uri(self, s3_uri: str) -> tuple[str, str]:
        """
        Parse S3 URI into bucket and key.

        Args:
            s3_uri: S3 URI (e.g., "s3://bucket-name/path/to/model")

        Returns:
            Tuple of (bucket_name, key_prefix)

        Raises:
            ValueError: If URI format is invalid
        """
        if not s3_uri.startswith("s3://"):
            raise ValueError(f"Invalid S3 URI format: {s3_uri}")

        # Remove s3:// prefix
        path = s3_uri[5:]

        # Split into bucket and key
        parts = path.split("/", 1)
        bucket = parts[0]
        key = parts[1] if len(parts) > 1 else ""

        if not bucket:
            raise ValueError(f"Invalid S3 URI - missing bucket name: {s3_uri}")

        return bucket, key

    def estimate_storage(self, model_id: str) -> Optional[float]:
        """
        Estimate storage requirements by listing S3 objects.

        Args:
            model_id: S3 URI (e.g., "s3://bucket-name/path/to/model")

        Returns:
            Estimated size in GB (float), or None if unavailable
        """
        if self._s3_client is None:
            logger.debug("S3 client not available for storage estimation")
            return None

        try:
            bucket, key_prefix = self._parse_s3_uri(model_id)
            logger.debug(f"Estimating storage for S3 bucket: {bucket}, prefix: {key_prefix}")

            # List all objects with the given prefix
            total_size = 0
            paginator = self._s3_client.get_paginator("list_objects_v2")

            for page in paginator.paginate(Bucket=bucket, Prefix=key_prefix):
                if "Contents" in page:
                    for obj in page["Contents"]:
                        total_size += obj["Size"]

            if total_size == 0:
                logger.debug(f"No objects found or empty S3 path: {model_id}")
                return None

            # Convert bytes to GB
            size_gb = round(total_size / (1024**3), 2)
            logger.debug(f"Estimated size for {model_id}: {size_gb}GB")
            return size_gb

        except ValueError as e:
            logger.error(f"Invalid S3 URI: {e}")
            return None
        except Exception as e:
            # Catch any boto3/botocore exceptions or other errors
            logger.debug(f"Failed to estimate storage from S3: {e}")
            return None

    def _get_model_dir_name_from_s3_uri(self, s3_uri: str) -> str:
        """
        Extract a meaningful directory name from an S3 URI.

        Uses last 2 path components (like org/model) if available to mirror
        HuggingFace's organizational structure. Falls back to last component
        or bucket name if shorter paths are provided.

        Args:
            s3_uri: S3 URI (e.g., "s3://bucket-name/path/to/model")

        Returns:
            Directory name string (e.g., "org/model" or "model" or "bucket-name")

        Examples:
            s3://my-bucket/models/org/llama-3.1-8b → org/llama-3.1-8b
            s3://my-bucket/path/to/model → to/model
            s3://my-bucket/model-name → model-name
            s3://my-bucket → my-bucket
        """
        bucket, key_prefix = self._parse_s3_uri(s3_uri)

        if not key_prefix:
            # Just bucket name - no path provided
            return bucket

        # Remove trailing slashes
        key_prefix = key_prefix.rstrip("/")

        # Split into path components
        parts = key_prefix.split("/")

        if len(parts) >= 2:
            # Return last 2 components to mirror org/model pattern
            return os.path.join(parts[-2], parts[-1])
        else:
            # Return last component only
            return parts[-1]

    def download(self, model_id: str, cache_dir: str = None, custom_model_name: str = None) -> str:
        """
        Download a model from S3 to the specified cache directory.

        This downloads all files under the S3 prefix, maintaining the directory structure.
        Files that already exist locally with matching sizes are skipped to optimize downloads.

        Args:
            model_id: S3 URI (e.g., "s3://bucket-name/path/to/model")
            cache_dir: Directory to cache the downloaded model (required for S3 backend,
                      but signature must match base class)
            custom_model_name: Optional custom directory name (e.g., "org/model" or "custom-name").
                       If not provided, will be extracted from the S3 URI path using
                       intelligent naming logic.

        Returns:
            Path to the downloaded model directory

        Raises:
            RuntimeError: If S3 client is unavailable, cache_dir is None, or download fails
        """
        if self._s3_client is None:
            raise RuntimeError("S3 client not available. Install boto3: pip install boto3")

        if cache_dir is None:
            raise RuntimeError("cache_dir is required for S3 backend downloads")

        try:
            bucket, key_prefix = self._parse_s3_uri(model_id)
            logger.info(f"Downloading model from S3 bucket: {bucket}, prefix: {key_prefix}")

            # Create cache directory if it doesn't exist
            os.makedirs(cache_dir, exist_ok=True)

            # Determine the local model directory path
            if custom_model_name:
                # Use explicitly provided custom model name
                model_dir_name = custom_model_name
                logger.info(f"Using custom model name: {custom_model_name}")
            else:
                # Use enhanced naming logic to extract from S3 URI
                # This extracts last 2 path components (org/model) to mirror HuggingFace pattern
                model_dir_name = self._get_model_dir_name_from_s3_uri(model_id)
                logger.info(f"Auto-detected model name from S3 URI: {model_dir_name}")

            local_model_dir = os.path.join(cache_dir, model_dir_name)
            os.makedirs(local_model_dir, exist_ok=True)

            # List all objects with the given prefix
            paginator = self._s3_client.get_paginator("list_objects_v2")
            downloaded_files = 0
            skipped_files = 0
            total_files = 0

            for page in paginator.paginate(Bucket=bucket, Prefix=key_prefix):
                if "Contents" not in page:
                    continue

                for obj in page["Contents"]:
                    s3_key = obj["Key"]

                    # Skip if this is just a directory marker
                    if s3_key.endswith("/"):
                        continue

                    total_files += 1

                    # Calculate the relative path from the prefix
                    if key_prefix:
                        relative_path = s3_key[len(key_prefix) :].lstrip("/")
                    else:
                        relative_path = s3_key

                    # Construct local file path
                    local_file_path = os.path.join(local_model_dir, relative_path)

                    # Create parent directories if needed
                    os.makedirs(os.path.dirname(local_file_path), exist_ok=True)

                    # Check if file already exists with matching size
                    should_download = True
                    if os.path.exists(local_file_path):
                        local_size = os.path.getsize(local_file_path)
                        remote_size = obj["Size"]

                        if local_size == remote_size:
                            # File exists with same size - skip download
                            # Note: Size-based comparison is much faster than ETag/hash verification
                            # and sufficient for detecting incomplete downloads or corruption
                            logger.debug(
                                f"Skipping {s3_key} - already exists locally with matching size ({remote_size} bytes)"
                            )
                            should_download = False
                            skipped_files += 1
                        else:
                            # File exists but size differs - re-download
                            logger.debug(
                                f"Re-downloading {s3_key} - size mismatch "
                                f"(local: {local_size} bytes, remote: {remote_size} bytes)"
                            )

                    if should_download:
                        # Download the file
                        logger.debug(f"Downloading {s3_key} to {local_file_path}")
                        self._s3_client.download_file(bucket, s3_key, local_file_path)
                        downloaded_files += 1

            if total_files == 0:
                raise RuntimeError(f"No files found in S3 path: {model_id}")

            # Log summary
            if skipped_files > 0:
                logger.info(
                    f"Download complete: {downloaded_files} downloaded, {skipped_files} skipped "
                    f"(already cached), {total_files} total files"
                )
            else:
                logger.info(f"Successfully downloaded {downloaded_files} files to: {local_model_dir}")

            return local_model_dir

        except ValueError as e:
            error_msg = f"Invalid S3 URI {model_id}: {e}"
            logger.error(error_msg)
            raise RuntimeError(error_msg) from e
        except RuntimeError:
            # Re-raise RuntimeError as-is (e.g., "No files found")
            raise
        except Exception as e:
            # Catch any boto3/botocore exceptions or other errors
            error_msg = f"Failed to download model from S3 {model_id}: {e}"
            logger.error(error_msg)
            raise RuntimeError(error_msg) from e


class StorageBackendRegistry:
    """
    Registry for model storage backends.

    Manages multiple storage backends and routes requests to the appropriate backend
    based on the model source.
    """

    def __init__(self):
        """Initialize the registry with default backends."""
        self._backends = []
        self._register_default_backends()

    def _register_default_backends(self):
        """Register default storage backends."""
        self.register(HuggingFaceStorageBackend())
        self.register(S3StorageBackend())

    def register(self, backend: ModelStorageBackend):
        """
        Register a new storage backend.

        Args:
            backend: Storage backend instance to register
        """
        self._backends.append(backend)
        logger.debug(f"Registered storage backend: {backend.__class__.__name__}")

    def get_backend(self, model_source: str) -> Optional[ModelStorageBackend]:
        """
        Get the appropriate backend for a model source.

        Args:
            model_source: Model source URI (e.g., "hf://meta-llama/Llama-3.1-8B")

        Returns:
            Storage backend that supports the model source, or None if not found
        """
        for backend in self._backends:
            if backend.supports_model(model_source):
                return backend

        logger.warning(f"No storage backend found for model source: {model_source}")
        return None

    def estimate_storage(self, model_source: str) -> Optional[float]:
        """
        Estimate storage requirements for a model.

        Args:
            model_source: Model source URI

        Returns:
            Estimated size in GB (float), or None if unavailable
        """
        backend = self.get_backend(model_source)

        if backend is None:
            logger.warning(f"No storage backend available for source: {model_source}")
            return None

        # Extract model ID from source
        if hasattr(backend, "_extract_model_id"):
            model_id = backend._extract_model_id(model_source)
        else:
            model_id = model_source

        return backend.estimate_storage(model_id)

    def download(
        self, model_source: str, cache_dir: str = None, use_hf_cache: bool = False, custom_model_name: str = None
    ) -> str:
        """
        Download a model using the appropriate backend.

        Args:
            model_source: Model source URI (e.g., "hf://meta-llama/Llama-3.1-8B", "s3://bucket/path")
            cache_dir: Optional directory to cache the downloaded model.
                      For HuggingFace backend: If None, uses HF's default cache (HF_HOME).
                      For S3 backend: Required, must be provided.
            use_hf_cache: For HuggingFace backend, whether to use HF's cache structure (True)
                         or download directly to local directory (False, default).
            custom_model_name: For S3 backend, optional custom directory name.
                       If not provided, will be auto-detected from S3 URI path.
                       Ignored for HuggingFace backend.

        Returns:
            Path to the downloaded model

        Raises:
            ValueError: If no backend supports the model source
            RuntimeError: If download fails
        """
        backend = self.get_backend(model_source)

        if backend is None:
            raise ValueError(f"No storage backend available for source: {model_source}")

        # Extract model ID from source
        if hasattr(backend, "_extract_model_id"):
            model_id = backend._extract_model_id(model_source)
        else:
            model_id = model_source

        # HuggingFace backend supports both cache_dir and local_dir modes
        # S3 backend requires an explicit cache_dir and supports custom_model_name override
        if isinstance(backend, HuggingFaceStorageBackend):
            return backend.download(model_id, cache_dir=cache_dir, use_hf_cache=use_hf_cache)
        elif isinstance(backend, S3StorageBackend):
            # S3 backend supports optional custom_model_name parameter
            if cache_dir is None:
                raise ValueError(f"cache_dir is required for {backend.__class__.__name__}")
            return backend.download(model_id, cache_dir=cache_dir, custom_model_name=custom_model_name)
        else:
            # Other backends (fallback)
            if cache_dir is None:
                raise ValueError(f"cache_dir is required for {backend.__class__.__name__}")
            return backend.download(model_id, cache_dir)
