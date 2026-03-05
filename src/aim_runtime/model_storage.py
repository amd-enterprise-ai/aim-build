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

logger = logging.getLogger(__name__)


class ModelStorageBackend(ABC):
    """
    Abstract base class for model storage backends.

    A storage backend handles downloading models and estimating storage requirements
    for different model sources (HuggingFace Hub, local files, etc.).
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

    def download(self, model_source: str, cache_dir: str = None, use_hf_cache: bool = False) -> str:
        """
        Download a model using the appropriate backend.

        Args:
            model_source: Model source URI (e.g., "hf://meta-llama/Llama-3.1-8B")
            cache_dir: Optional directory to cache the downloaded model.
                      For HuggingFace backend: If None, uses HF's default cache (HF_HOME).
            use_hf_cache: For HuggingFace backend, whether to use HF's cache structure (True)
                         or download directly to local directory (False, default).

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
        if isinstance(backend, HuggingFaceStorageBackend):
            return backend.download(model_id, cache_dir=cache_dir, use_hf_cache=use_hf_cache)
        else:
            # Other backends (fallback)
            if cache_dir is None:
                raise ValueError(f"cache_dir is required for {backend.__class__.__name__}")
            return backend.download(model_id, cache_dir)
