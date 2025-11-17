# Copyright © Advanced Micro Devices, Inc., or its affiliates.
#
# SPDX-License-Identifier: MIT

"""
AIM Runtime

This module contains the AIMRuntime class which orchestrates the complete
AIM runtime workflow including profile selection and command generation.
"""

import logging
import os
import shutil
from dataclasses import replace
from typing import Any, Dict, List, Optional

import yaml

from .command_generator import CommandGenerator
from .config import AIMConfig
from .model_storage import StorageBackendRegistry
from .profile_selector import ProfileSelector

logger = logging.getLogger(__name__)


def _extract_models_from_profile(profile_data: Dict[str, Any]) -> List[Dict[str, Any]]:
    """
    Extract required models from a profile template.

    Args:
        profile_data: Parsed YAML profile data

    Returns:
        List of dictionaries with model information. Each dict has 'name' and 'source' keys.
        Uses model_id field for the actual model to load.
        Model sources are prefixed with 'hf://'.
    """
    models: List[Dict[str, Any]] = []

    # Use model_id field (general profiles won't have this)
    if "model_id" not in profile_data or not profile_data["model_id"]:
        # No model specified (e.g., general profile), return empty list
        return models

    model_name = profile_data["model_id"]

    # Prefix with hf:// for the source
    model_source = f"hf://{model_name}"

    # Create model info dict
    model_info = {"name": model_name, "source": model_source}

    models.append(model_info)

    return models


def _add_storage_estimates(models: List[Dict[str, Any]], storage_registry: StorageBackendRegistry) -> None:
    """
    Add storage size estimates to model information dictionaries (in-place).

    Args:
        models: List of model dictionaries with 'source' keys
        storage_registry: Storage backend registry for estimating storage needs
    """
    for model_info in models:
        if "source" in model_info:
            size_gb = storage_registry.estimate_storage(model_info["source"])
            model_info["size_gb"] = size_gb


class AIMRuntime:
    """Main orchestrator for AIM runtime operations."""

    def __init__(self, config: AIMConfig):
        """Initialize the AIM runtime with configuration."""
        self.config = config
        self.profile_selector = ProfileSelector(config)
        self.command_generator = CommandGenerator(config)
        self.storage_registry = StorageBackendRegistry()

    @staticmethod
    def normalize_model_source(model_id: str) -> str:
        """
        Normalize a model ID to include appropriate protocol prefix.

        If the model ID already has a protocol (s3://, hf://), returns it as-is.
        Otherwise, adds the hf:// prefix for HuggingFace models.

        Args:
            model_id: Model identifier (e.g., "meta-llama/Llama-3.1-8B" or "s3://bucket/path")

        Returns:
            Normalized model source with protocol prefix

        Examples:
            normalize_model_source("meta-llama/Llama-3.1-8B") -> "hf://meta-llama/Llama-3.1-8B"
            normalize_model_source("s3://bucket/model") -> "s3://bucket/model"
            normalize_model_source("hf://org/model") -> "hf://org/model"
        """
        if model_id.startswith(("s3://", "hf://")):
            return model_id
        else:
            return f"hf://{model_id}"

    @staticmethod
    def drop_protocol(model_id: str) -> str:
        if model_id.startswith("s3://") or model_id.startswith("hf://"):
            return model_id[5:]
        return model_id

    def download_to_cache(
        self,
        model_id: Optional[str] = None,
        use_hf_cache: bool = False,
        custom_model_name: Optional[str] = None,
    ) -> str:
        """
        Download model to cache directory.

        Downloads the model to the cache directory specified by AIM_CACHE_PATH environment variable.
        By default, downloads directly to the local directory (local-dir mode). Use use_hf_cache
        to download using HuggingFace's default cache structure instead.

        If model_id is not provided, uses the current configuration to determine the model.

        For S3 downloads, use custom_model_name to specify a custom directory name.

        Args:
            model_id: Explicit model id to download (e.g. hf://org/model or s3://path/to/model).
                      Overrides profile selection.
            use_hf_cache: Use HuggingFace's default cache directory structure instead of
                          downloading directly to local directory.
            custom_model_name: Custom directory name for S3 downloads (e.g. 'org/model' or 'custom-name').
                       If not provided, auto-detects from S3 URI. Ignored for HuggingFace downloads.

        Returns:
            Path: The path where the model was downloaded

        Raises:
            ValueError: Configuration error
            FileNotFoundError: If profile or model not found
            RuntimeError: If download fails
        """
        logger.info("AIM Runtime download-to-cache mode...")
        logger.debug(f"Log levels - Root: {self.config.log_level_root}, AIM: {self.config.log_level}")
        logger.info(f"Using cache directory: {self.config.cache_path}")

        if model_id:
            # Use explicit model id (with protocol)
            model_source = self.normalize_model_source(model_id)
            logger.info(f"Using explicit model id: {model_source}")
        else:
            # Select the profile
            logger.info("Selecting profile...")
            profile = self.profile_selector.find_profile()
            logger.info(f"Selected profile: {profile.profile_handling.path}")

            # Read and parse profile to extract model
            with open(profile.profile_handling.path, "r", encoding="utf-8") as f:
                profile_data = yaml.safe_load(f)

            # Get model_id from profile
            if "model_id" in profile_data and profile_data["model_id"]:
                model_id_from_profile = profile_data["model_id"]
                logger.info(f"Using model: {model_id_from_profile}")
                model_source = self.normalize_model_source(model_id_from_profile)
            else:
                raise ValueError(
                    "The 'model_id' field is missing from the selected profile. "
                    "This field is required to identify which model to download. Please ensure your profile "
                    "YAML includes a valid 'model_id' entry."
                )

        # Estimate storage before downloading
        logger.info("Estimating storage requirements...")
        size_gb = self.storage_registry.estimate_storage(model_source)
        if size_gb:
            logger.info(f"Estimated model size: {size_gb} GB")
        else:
            logger.warning("Could not estimate model size")

        # Download the model
        if use_hf_cache:
            logger.info("Downloading model using HuggingFace cache structure")
            logger.info(f"Cache directory: {self.config.cache_path}")
        else:
            logger.info(f"Downloading model to local directory: {self.config.cache_path}")

        if custom_model_name:
            logger.info(f"Using custom model name: {custom_model_name}")

        downloaded_path = self.storage_registry.download(
            model_source, self.config.cache_path, use_hf_cache=use_hf_cache, custom_model_name=custom_model_name
        )

        logger.info(f"✓ Model successfully downloaded to: {downloaded_path}")
        return downloaded_path

    def serve(self) -> None:
        """
        Select profile and execute the inference server (replaces current process).

        This method performs the complete serve workflow:
        1. Select the appropriate profile
        2. Determine actual model and auto-download from S3 if needed
        3. Generate execution parameters
        4. Set environment variables
        5. Execute the inference server command (via os.execv)

        Raises:
            FileNotFoundError: If the executable cannot be found
        """
        self.config.log_debug_info()

        # Step 1: Select the profile
        logger.info("Selecting profile...")
        profile = self.profile_selector.find_profile()
        logger.info(f"Selected profile: {profile.profile_handling.path}")

        # Step 2: Determine the actual model and auto-download if S3
        # Resolve model with fallback chain (same logic as CommandGenerator)
        model_id = profile.model_id or self.config.model_id or self.config.aim_id

        if not model_id:
            raise ValueError("Model not specified in profile or configuration")

        if model_id.startswith("s3://"):
            logger.info(f"Detected S3 model source: {model_id}")
            logger.info("Auto-downloading model from S3 to cache...")

            try:
                downloaded_path = self.download_to_cache(
                    model_id=model_id,
                    use_hf_cache=False,  # Use local directory format
                    custom_model_name=self.config.custom_model_name,  # Use from config or auto-detect
                )

                # Extract model name from downloaded path (e.g., /cache/org/model -> org/model)
                # This allows the model_cache_resolver to find it
                # Validate that the downloaded path is within the cache directory
                abs_downloaded_path = os.path.abspath(os.path.realpath(downloaded_path))
                abs_cache_path = os.path.abspath(os.path.realpath(self.config.cache_path))

                if not abs_downloaded_path.startswith(os.path.join(abs_cache_path, "")):
                    logger.error(
                        f"Downloaded path {abs_downloaded_path} is not within cache directory {abs_cache_path}"
                    )
                    raise ValueError(
                        f"Downloaded model path {abs_downloaded_path} is not within cache directory {abs_cache_path}"
                    )

                relative_path = os.path.relpath(abs_downloaded_path, abs_cache_path)
                logger.info(f"Model identifier updated from S3 URI to: {relative_path}")

                # Update the appropriate field based on where the S3 URI came from
                # Note: aim_id is never an S3 URI (it's the container identifier)
                if profile.model_id and profile.model_id.startswith("s3://"):
                    # Profile had S3 model_id - update the profile using dataclasses.replace
                    profile = replace(profile, model_id=relative_path)
                    logger.debug(f"Updated profile.model_id to: {relative_path}")
                elif self.config.model_id and self.config.model_id.startswith("s3://"):
                    # Config had S3 model_id - update config
                    self.config.model_id = relative_path
                    logger.debug(f"Updated config.model_id to: {relative_path}")

            except Exception as e:
                logger.error(f"Failed to download model from S3: {e}")
                raise RuntimeError(f"Auto-download failed for S3 model {model_id}: {e}") from e

        # Step 3: Generate execution parameters
        logger.info("Generating execution parameters...")
        command_list, env_vars = self.command_generator.generate_execution_params(profile)

        # Step 4: Set environment variables
        logger.info("--- Setting Environment Variables ---")
        for key, value in env_vars.items():
            os.environ[key] = str(value)
            logger.info(f"Set env var: {key}={value}")
        logger.info("-------------------------------------")

        # Step 5: Log command
        logger.info("--- Execution Command ---")
        logger.info(f"Command: {' '.join(command_list)}")
        logger.info("-------------------------")

        # Step 6: Resolve the executable path and execute
        executable_path = shutil.which(command_list[0])
        if not executable_path:
            raise FileNotFoundError(f"Could not find executable: {command_list[0]}")

        # Execute the command directly (replaces current process)
        os.execv(executable_path, command_list)

    def dry_run(self) -> str:
        """
        Perform profile selection and return the selected profile YAML.

        This command shows the complete profile that would be used based on the current configuration.

        Returns:
            str: The profile YAML content with header comment
        """
        self.config.log_debug_info()

        # Step 1: Select the profile
        logger.info("Selecting profile...")
        profile = self.profile_selector.find_profile()

        # Step 2: Generate the command script to show what would be executed
        logger.info("Generating command script for dry-run...")
        script_path = self.command_generator.generate_command_script(profile)

        # Step 3: Build output with profile path and contents
        output_lines = [
            "=" * 80,
            "SELECTED PROFILE",
            "=" * 80,
            f"Path: {profile.profile_handling.path}",
            "",
        ]

        # Read and append the profile YAML
        with open(profile.profile_handling.path, "r", encoding="utf-8") as f:
            output_lines.append(f.read())

        # Step 4: Add the generated script content
        output_lines.extend(
            [
                "",
                "=" * 80,
                "GENERATED SCRIPT",
                "=" * 80,
                "",
            ]
        )
        with open(script_path, "r", encoding="utf-8") as f:
            output_lines.append(f.read())

        logger.info("Dry-run completed successfully")
        return "\n".join(output_lines)

    def dry_run_json(self) -> List[Dict[str, Any]]:
        """
        Get the selected profile as a JSON-serializable list of dictionaries.

        Mimics dry_run logic but returns JSON format with parsed YAML content,
        including storage requirements estimation for the required models.

        Returns:
            List[Dict[str, Any]]: List containing a single dictionary with 'filename', 'profile', and 'models' keys.
                                  Each model includes storage estimation information.
                                  Returns empty list if no compatible profile is found.
        """
        self.config.log_debug_info()

        try:
            # Step 1: Select the profile (same as dry_run)
            logger.info("Selecting profile...")
            profile = self.profile_selector.find_profile()

            # Step 2: Build list with dict containing filename and profile data
            filename = profile.profile_handling.filename

            # Read and parse profile contents as YAML
            with open(profile.profile_handling.path, "r", encoding="utf-8") as f:
                profile_data = yaml.safe_load(f)

            # Step 3: Extract required models from profile
            models = _extract_models_from_profile(profile_data)

            # Step 4: Fallback to config if no models in profile (e.g., general profiles)
            if not models:
                fallback_model_id = self.config.model_id or self.config.aim_id
                if fallback_model_id:
                    models = [{"name": self.drop_protocol(fallback_model_id), "source": fallback_model_id}]

            # Step 5: Normalize model sources using class method
            for model_info in models:
                if "source" in model_info:
                    model_info["source"] = self.normalize_model_source(model_info["name"])

            # Step 6: Add storage estimates to models
            _add_storage_estimates(models, self.storage_registry)

            logger.info(f"Selected profile: {profile.profile_handling.path}")
            return [{"filename": filename, "profile": profile_data, "models": models}]

        except Exception as e:
            # If no profile found or any error occurs, return empty list
            # This includes ProfileNotFound, FileNotFoundError, yaml.YAMLError, etc.
            logger.warning(f"No compatible profile found or error occurred: {e}")
            return []
