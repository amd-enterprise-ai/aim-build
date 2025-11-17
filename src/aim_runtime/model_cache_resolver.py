# Copyright © Advanced Micro Devices, Inc., or its affiliates.
#
# SPDX-License-Identifier: MIT

"""
Model Cache Resolver

This module provides functionality to locate models in different cache formats:
1. HuggingFace Hub cache format (HF_HOME/hub/)
2. Local directory format (cache_dir/org/model/)
"""

import logging
import os
from dataclasses import dataclass
from typing import Optional

logger = logging.getLogger(__name__)


@dataclass
class ResolvedModelPath:
    """Represents a resolved model path with metadata."""

    path: str
    """The resolved path to the model"""

    is_local_dir: bool
    """Whether the model is in local directory format (True) or HF cache format (False)"""

    model_id: str
    """The original model ID (org/model format)"""


class ModelCacheResolver:
    """
    Resolves model locations across different cache formats.

    The resolver checks for models in the following order:
    1. HuggingFace Hub cache format (HF_HOME/hub/) - uses model_id for lookup
    2. Local directory format (cache_dir/org/model/) - uses local path
    """

    def __init__(self, cache_dir: str):
        """
        Initialize the model cache resolver.

        Args:
            cache_dir: Base cache directory (typically /workspace/model-cache)
        """
        self.cache_dir = cache_dir
        self.hf_hub_cache_dir = os.path.join(cache_dir, "hub")

    def resolve_model_path(self, model_id: str) -> Optional[ResolvedModelPath]:
        """
        Resolve the path to a model, checking both cache formats.

        Args:
            model_id: Model identifier in org/model format (e.g., "meta-llama/Llama-3.1-8B-Instruct")

        Returns:
            ResolvedModelPath if model is found in cache, None if not found

        The resolution order is:
        1. Check local directory format (cache_dir/org/model/)
        2. Fall back to model_id (HuggingFace handles cache/download transparently)
        """
        # Check local directory format first (cache_dir/org/model/)
        local_path = self._get_local_dir_path(model_id)
        if local_path and os.path.isdir(local_path):
            logger.info(f"Found model in local directory format: {local_path}")
            return ResolvedModelPath(
                path=local_path,
                is_local_dir=True,
                model_id=model_id,
            )

        # No local directory found, use model_id
        # HuggingFace will handle cache lookup or download transparently
        logger.debug(f"No local directory found for {model_id}, using model_id (HF will handle cache/download)")
        return ResolvedModelPath(
            path=model_id,
            is_local_dir=False,
            model_id=model_id,
        )

    def _get_local_dir_path(self, model_id: str) -> Optional[str]:
        """
        Get the local directory path for a model ID.

        Args:
            model_id: Model identifier in org/model format

        Returns:
            Path to local directory if valid format, None otherwise
        """
        # Parse model_id into org/model
        parts = model_id.split("/")
        if len(parts) != 2:
            logger.debug(f"Invalid model_id format for local dir lookup: {model_id}")
            return None

        org, model = parts
        local_path = os.path.join(self.cache_dir, org, model)
        return local_path

    def check_local_dir_exists(self, model_id: str) -> bool:
        """
        Check if a model exists in local directory format.

        Args:
            model_id: Model identifier in org/model format

        Returns:
            True if model exists in local directory format
        """
        local_path = self._get_local_dir_path(model_id)
        return local_path is not None and os.path.isdir(local_path)

    def check_hf_cache_exists(self) -> bool:
        """
        Check if HuggingFace Hub cache directory exists.

        Returns:
            True if HF Hub cache directory exists
        """
        return os.path.isdir(self.hf_hub_cache_dir)
