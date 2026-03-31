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
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml

from .command_generator import CommandGenerator
from .config import AIMConfig
from .engine_config import load_engine_config
from .gpu_detector import get_gfx_arch
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


def _install_aiter_prebuilt_kernels(gpu_model: str) -> None:
    """Copy pre-built AITER kernels into the system AITER JIT directory.

    Instead of redirecting AITER_JIT_DIR (which requires symlinking all
    52 system modules into each arch directory), this copies only the
    pre-built attention kernels into AITER's default JIT directory.
    AITER then finds both pre-built and system modules in one place.

    This approach is robust against /workspace volume mounts, symlink
    breakage on overlayfs, and AITER package path changes.

    Args:
        gpu_model: GPU model from profile.metadata.gpu (e.g., "MI300X")
    """
    gfx_arch = get_gfx_arch(gpu_model)
    if not gfx_arch:
        return

    prebuilt_dir = Path(f"/workspace/aiter-jit-prebuilt/{gfx_arch}")
    if not prebuilt_dir.is_dir():
        return

    try:
        import aiter

        jit_dir = Path(aiter.__file__).parent / "jit"
    except ImportError:
        logger.debug("AITER package not available, skipping kernel installation")
        return

    if not jit_dir.is_dir():
        logger.warning(f"AITER JIT directory not found at {jit_dir}")
        return

    installed = 0
    for so_file in prebuilt_dir.glob("*.so"):
        dest = jit_dir / so_file.name
        if not dest.exists():
            try:
                shutil.copy2(so_file, dest)
                installed += 1
            except OSError as e:
                logger.warning(f"Failed to install pre-built kernel {so_file.name}: {e}")

    if installed:
        logger.info(f"Installed {installed} pre-built AITER kernel(s) for {gpu_model} ({gfx_arch}) into {jit_dir}")


class AIMRuntime:
    """Main orchestrator for AIM runtime operations."""

    def __init__(self, config: AIMConfig):
        """Initialize the AIM runtime with configuration."""
        self.config = config
        self.profile_selector = ProfileSelector(config)
        engine_config = load_engine_config(config.engine, config.config_path)
        self.command_generator = CommandGenerator(config, engine_config)
        self.storage_registry = StorageBackendRegistry()

    @staticmethod
    def normalize_model_source(model_id: str) -> str:
        """
        Normalize a model ID to include appropriate protocol prefix.

        If the model ID already has a protocol (e.g. hf://), returns it as-is.
        Otherwise, adds the hf:// prefix for HuggingFace models.

        Args:
            model_id: Model identifier (e.g., "meta-llama/Llama-3.1-8B")

        Returns:
            Normalized model source with protocol prefix

        Examples:
            normalize_model_source("meta-llama/Llama-3.1-8B") -> "hf://meta-llama/Llama-3.1-8B"
            normalize_model_source("hf://org/model") -> "hf://org/model"
        """
        if model_id.startswith("hf://"):
            return model_id
        else:
            return f"hf://{model_id}"

    @staticmethod
    def drop_protocol(model_id: str) -> str:
        if model_id.startswith("hf://"):
            return model_id[5:]
        return model_id

    def download_to_cache(
        self,
        model_id: Optional[str] = None,
        use_hf_cache: bool = False,
    ) -> str:
        """
        Download model to cache directory.

        Downloads the model to the cache directory specified by AIM_CACHE_PATH environment variable.
        By default, downloads directly to the local directory (local-dir mode). Use use_hf_cache
        to download using HuggingFace's default cache structure instead.

        If model_id is not provided, uses the current configuration to determine the model.

        Args:
            model_id: Explicit model id to download (e.g. hf://org/model).
                      Overrides profile selection.
            use_hf_cache: Use HuggingFace's default cache directory structure instead of
                          downloading directly to local directory.

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

        downloaded_path = self.storage_registry.download(
            model_source, self.config.cache_path, use_hf_cache=use_hf_cache
        )

        logger.info(f"✓ Model successfully downloaded to: {downloaded_path}")
        return downloaded_path

    def serve(self) -> None:
        """
        Select profile and execute the inference server (replaces current process).

        This method performs the complete serve workflow:
        1. Select the appropriate profile
        2. Determine actual model and auto-download if needed
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

        # Step 2: Determine the actual model
        # Resolve model with fallback chain (same logic as CommandGenerator)
        model_id = profile.model_id or self.config.model_id or self.config.aim_id

        if not model_id:
            raise ValueError("Model not specified in profile or configuration")

        # Install pre-built AITER kernels for the target GPU architecture
        if profile.metadata and profile.metadata.gpu:
            _install_aiter_prebuilt_kernels(profile.metadata.gpu)

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

    def dry_run(self) -> List[Dict[str, Any]]:
        """
        Get the selected profile as a serializable dictionary.

        Returns:
            List[Dict[str, Any]]: A list of dictionaries with 'filename', 'profile', 'models', and 'script' keys.
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
            path = profile.profile_handling.path

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

            script_path = self.command_generator.generate_command_script(profile)
            with open(script_path, "r", encoding="utf-8") as f:
                output_script = f.read()
            # Step 6: Add storage estimates to models
            _add_storage_estimates(models, self.storage_registry)

            logger.info(f"Selected profile: {profile.profile_handling.path}")
            return [
                {
                    "filename": filename,
                    "path": path,
                    "profile": profile_data,
                    "models": models,
                    "script": output_script,
                }
            ]

        except Exception as e:
            # If no profile found or any error occurs, return empty list
            # This includes ProfileNotFound, FileNotFoundError, yaml.YAMLError, etc.
            logger.warning(f"No compatible profile found or error occurred: {e}")
            return []
