# Copyright © Advanced Micro Devices, Inc., or its affiliates.
#
# SPDX-License-Identifier: MIT

"""
AIM Command Generator

This module contains the CommandGenerator class for generating runtime commands
from profile configurations.
"""

import json
import logging
import os
import shlex
import shutil
import stat
import tempfile
from dataclasses import asdict
from typing import Any, Dict, List, Optional

from jsonschema import ValidationError

from .config import AIMConfig
from .model_cache_resolver import ModelCacheResolver
from .object_model import Engine, EngineModule, Profile
from .profile_validator import ProfileValidator

logger = logging.getLogger(__name__)


class CommandGenerator:
    """Generates runtime commands from profile configurations."""

    def __init__(self, config: AIMConfig):
        """Initialize the command generator with configuration."""
        self.config = config
        self.cache_resolver = ModelCacheResolver(config.cache_path)
        self.profile_validator = ProfileValidator(config.schema_search_path)

    def generate_execution_params(self, profile: Profile) -> tuple[List[str], Dict[str, str]]:
        """
        Generate execution parameters from a profile object.

        This returns the command as an argument list and environment variables
        for direct process execution via os.execv(), avoiding shell interpretation
        and injection risks.

        Args:
            profile: Profile object containing profile parameters

        Returns:
            tuple: (command_list, env_vars_dict) where:
                - command_list: List of command arguments for direct execution
                - env_vars_dict: Dictionary of environment variables to set
        """
        # Build command as list (no shell interpretation)
        command_list = self._build_command_list(profile)

        # Get environment variables
        env_vars = profile.env_vars or {}

        logger.info(f"Generated execution parameters: {len(command_list)} args, {len(env_vars)} env vars")
        return command_list, env_vars

    def generate_command_script(self, profile: Profile) -> str:
        """
        Generate a shell script from a profile object (legacy method for dry-run).

        Args:
            profile: Profile object containing profile parameters

        Returns:
            str: Path to the generated shell script
        """
        # Generate the command components
        command = self._build_command(profile)

        # Create the shell script
        script_content = self._create_script_content(command, profile.env_vars)
        script_path = self._write_script_file(script_content)

        logger.info(f"Generated command script: {script_path}")
        return script_path

    def _build_command(self, profile: Profile) -> str:
        """
        Build the command string from profile configuration for shell execution.

        This properly quotes arguments for safe shell interpretation.
        For direct execution via os.execv(), use _build_command_list() instead.
        """
        command_list = self._build_command_list(profile)
        # Use shlex.quote() to properly escape arguments for shell execution
        return " ".join(shlex.quote(arg) for arg in command_list)

    def _build_command_list(self, profile: Profile) -> List[str]:
        """Build the command as a list of arguments."""
        # Resolve model path with fallback chain:
        # 1. Profile model_id (model-specific profiles)
        # 2. Config model_id (base containers with AIM_MODEL_ID)
        # 3. Config aim_id (model-specific containers using general profiles)
        model_id = profile.model_id or self.config.model_id or self.config.aim_id
        if not model_id:
            raise ValueError("Model not specified in profile or configuration")

        # Resolve model path using cache resolver
        resolved_model = self.cache_resolver.resolve_model_path(model_id)
        if resolved_model is None:
            # Fallback to model_id if resolution fails
            logger.warning(f"Could not resolve model path for {model_id}, using model_id directly")
            model_path = model_id
        else:
            model_path = resolved_model.path

        # Build served-model-name as a list: [model_id, aim_id] (deduplicated)
        # Always set served-model-name regardless of cache type
        served_model_name_list = [model_id]
        # Add aim_id if present and different from model_id
        if self.config.aim_id and self.config.aim_id != model_id:
            served_model_name_list.append(self.config.aim_id)

        # Determine engine and corresponding Python module
        metadata = profile.metadata
        engine = metadata.engine
        engine_module = self._get_engine_module(engine)

        # Merge and validate engine arguments
        engine_args = self._merge_and_validate_engine_args(profile, engine)

        # Add system overrides (always take precedence)
        engine_args["port"] = self.config.port

        # Add served-model-name (as a list)
        engine_args["served-model-name"] = served_model_name_list
        logger.info(f"Setting served-model-name to: {served_model_name_list}")

        args_list = self._build_engine_args(engine_args)

        # Use python3 if python is not available
        python_cmd = "python" if shutil.which("python") else "python3"

        # Construct the full command
        command_list = [python_cmd, "-m", engine_module.value, "--model", model_path] + args_list

        return command_list

    def _merge_and_validate_engine_args(self, profile: Profile, engine: Engine) -> Dict[str, Any]:
        """
        Merge engine arguments from profile and user overrides, then validate using ProfileValidator.

        Merge precedence (lowest to highest):
        1. Profile defaults
        2. User overrides (from AIM_ENGINE_ARGS)
        3. System overrides (added by caller, e.g., port)

        Note: Security validation is NOT needed because arguments are passed directly
        to the engine process via os.execv() as an argument list, with no shell
        interpretation. This makes command injection impossible.

        Args:
            profile: Profile containing base engine arguments
            engine: Engine type for validation

        Returns:
            Merged and validated engine arguments dictionary

        Raises:
            ValidationError: If arguments don't conform to profile schema
        """
        # Start with profile defaults
        engine_args = profile.engine_args.copy() if profile.engine_args else {}

        # Apply user overrides if present
        if self.config.engine_args_override:
            logger.info(f"Applying {len(self.config.engine_args_override)} user-provided engine argument overrides")

            # Log what's being overridden
            for key, value in self.config.engine_args_override.items():
                if key in engine_args:
                    logger.debug(f"Overriding engine_arg '{key}': {engine_args[key]} -> {value}")
                else:
                    logger.debug(f"Adding new engine_arg '{key}': {value}")

            # Merge (user values win)
            engine_args.update(self.config.engine_args_override)

            # Validate merged arguments by converting profile to dict and updating engine_args
            try:
                # Convert profile dataclass to dict (handles nested dataclasses automatically)
                profile_data = asdict(profile)

                # Update with merged engine_args
                profile_data["engine_args"] = engine_args

                # Determine if general profile
                is_general_profile = not profile.aim_id  # Empty string for general profiles

                # Validate using ProfileValidator (which validates engine_args via schema)
                self.profile_validator.validate(profile_data, is_general_profile=is_general_profile)
                logger.debug(f"Successfully validated {len(engine_args)} merged engine arguments")

            except ValidationError as e:
                error_msg = f"Engine arguments validation failed: {e.message}"
                logger.error(error_msg)
                logger.error(f"Path: {' -> '.join(str(p) for p in e.path) if e.path else 'root'}")
                logger.error(f"Invalid arguments: {json.dumps(engine_args, indent=2)}")
                raise ValidationError(f"{error_msg}\nPath: {e.path}\nInvalid value: {e.instance}")

        return engine_args

    def _get_engine_module(self, engine: Engine) -> EngineModule:
        try:
            return EngineModule[engine.name]
        except KeyError:
            raise ValueError(f"Unsupported engine: {engine}. Supported engines: {[engine.value for engine in Engine]}")

    def _build_engine_args(self, engine_args: Dict[str, Any]) -> List[str]:
        """Build engine arguments list from the engine_args dictionary."""
        args_list = []

        for key, value in engine_args.items():
            if value is None:
                # Flag without value
                args_list.append(f"--{key}")
            elif isinstance(value, bool):
                # Boolean flags
                if value:
                    args_list.append(f"--{key}")
                # Skip false boolean values
            elif isinstance(value, (list, tuple)):
                # Multiple values - add flag once followed by all values
                # This matches vLLM's nargs="+" behavior for --served-model-name
                args_list.append(f"--{key}")
                for item in value:
                    args_list.append(str(item))
            elif isinstance(value, dict):
                # YAML objects (dictionaries in Python) - JSON string without quotes
                # No quotes needed since we use os.execv() which passes args directly
                args_list.extend([f"--{key}", json.dumps(value)])
            else:
                # Regular key-value pairs
                args_list.extend([f"--{key}", str(value)])

        return args_list

    def _create_script_content(self, command: str, env_vars: Optional[Dict[str, Any]] = None) -> str:
        """Create the shell script content."""
        script_content = "#!/bin/bash\nset -e\n\n"
        script_content += "# Generated by AIM Command Generator\n\n"

        # Add environment variables
        if env_vars:
            script_content += "# Environment variables\n"
            for key, value in env_vars.items():
                script_content += f"export {key}={shlex.quote(str(value))}\n"
            script_content += "\n"

        script_content += "echo '>>> Executing AIM Runtime Command...'\n"
        script_content += f"exec {command}\n"

        return script_content

    def _write_script_file(self, script_content: str) -> str:
        """Write the script content to a temporary file and make it executable."""
        fd, script_path = tempfile.mkstemp(suffix=".sh", prefix="aim-serve-")

        try:
            with os.fdopen(fd, "w", encoding="utf-8") as f:
                f.write(script_content)
        except Exception:
            # Clean up if writing fails
            os.unlink(script_path)
            raise

        # Make it executable
        st = os.stat(script_path)
        os.chmod(script_path, st.st_mode | stat.S_IEXEC)

        return script_path
