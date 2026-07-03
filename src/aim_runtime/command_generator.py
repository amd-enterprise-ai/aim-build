# Copyright © Advanced Micro Devices, Inc., or its affiliates.
#
# SPDX-License-Identifier: MIT

"""
AIM Command Generator

This module contains the CommandGenerator class for generating runtime commands
from profile configurations. Engine-specific details (launch command, model
argument, validation) are configured via engines.yaml rather than hardcoded.
"""

import logging
import os
import shlex
import stat
import tempfile
from typing import Any, Dict, List, Optional

from pydantic import ValidationError as PydanticValidationError

from aim_runtime.engines.base import BaseEngine

from .config import AIMConfig
from .model_cache_resolver import ModelCacheResolver
from .object_model import Profile
from .profile_validator import ProfileValidator

logger = logging.getLogger(__name__)


class CommandGenerator:
    """Generates runtime commands from profile configurations."""

    def __init__(self, config: AIMConfig, engine: BaseEngine):
        """Initialize the command generator with configuration.

        Args:
            config: AIM runtime configuration.
            engine: Concrete engine that owns launch, defaults, validation,
                and CLI serialization.
        """
        self.config = config
        self.engine = engine
        self.cache_resolver = ModelCacheResolver(config.cache_path)
        self.profile_validator = ProfileValidator()
        # Cache the (engine_args, env_vars) adapter runtime per profile so the
        # command (args) and env (vars) halves come from a single enumeration of
        # the adapter source — recomputing risks mismatched CLI vs env if the
        # directory changes between calls, plus redundant disk I/O.
        self._adapter_runtime_cache: dict[int, tuple[Dict[str, Any], Dict[str, str]]] = {}

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

        # Get environment variables (profile env + any adapter env)
        env_vars = self._resolve_env_vars(profile)

        logger.info(f"Generated execution parameters: {len(command_list)} args, {len(env_vars)} env vars")
        return command_list, env_vars

    def generate_command_script(self, profile: Profile, env_vars: Optional[Dict[str, Any]] = None) -> str:
        """
        Generate a shell script from a profile object (legacy method for dry-run).

        Args:
            profile: Profile object containing profile parameters
            env_vars: Optional env var overrides to render in the script. When provided
                (e.g. the cpuset-adjusted CPU env vars), these are used instead of
                ``profile.env_vars`` so the dry-run script matches what ``serve`` runs.

        Returns:
            str: Path to the generated shell script
        """
        # Generate the command components
        command = self._build_command(profile)

        # Create the shell script. Use the explicit override when provided (e.g. the
        # cpuset-adjusted CPU env vars from dry-run) so the script matches what serve
        # runs; otherwise fall back to the resolved profile env + any adapter env.
        script_env_vars = env_vars if env_vars is not None else self._resolve_env_vars(profile)
        script_content = self._create_script_content(command, script_env_vars)
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
        return shlex.join(command_list)

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

        # Merge and validate engine arguments
        engine_args = self._merge_and_validate_engine_args(profile)

        # Merge LoRA adapter engine args when the profile declares support, so
        # they flow through serialization (the env half is added in
        # _resolve_env_vars). The gate is the feature flag, not the engine name;
        # only LLM engines implement the hook.
        adapter_args, _ = self._adapter_runtime(profile)
        if adapter_args:
            engine_args.update(adapter_args)
            # Adapter args are injected after _merge_and_validate_engine_args, so
            # re-validate to keep invalid adapter-derived flags from reaching the
            # launch command undetected.
            self.engine.validate_engine_args(engine_args)

        # Add system overrides (always take precedence)
        engine_args["port"] = self.config.port

        # Apply engine-specific argument defaults (e.g. vLLM served-model-name).
        self.engine.apply_engine_defaults(engine_args, served_model_name_list)

        args_list = self.engine.serialize_engine_args(engine_args)

        # Build launch command from the engine.
        launch = self.engine.launch_prefix()

        # Prepend model path when the engine uses a dedicated model flag
        if self.engine.model_arg:
            command_list = launch + [self.engine.model_arg, model_path] + args_list
        else:
            command_list = launch + args_list

        return command_list

    def _resolve_env_vars(self, profile: Profile) -> Dict[str, str]:
        """Return the env vars to set on the engine process.

        Profile env vars plus any adapter env (e.g. the vLLM dynamic resolver
        triplet) when the profile declares adapter support. Shared by serve and
        dry-run so both surfaces show identical env.
        """
        env_vars: Dict[str, str] = dict(profile.env_vars or {})
        _, adapter_env = self._adapter_runtime(profile)
        env_vars.update(adapter_env)
        return env_vars

    def _adapter_runtime(self, profile: Profile) -> tuple[Dict[str, Any], Dict[str, str]]:
        """Return (engine_args, env_vars) for the profile's LoRA adapters.

        Memoized per profile so the command and env halves share one adapter
        enumeration. Returns empty dicts when the profile declares no adapter
        support or the engine has no adapter hook. The gate is the feature flag,
        not the engine name; only LLM engines implement the hook.
        """
        key = id(profile)
        cached = self._adapter_runtime_cache.get(key)
        if cached is not None:
            return cached
        if profile.metadata and profile.metadata.supports_adapters and hasattr(self.engine, "build_adapter_runtime"):
            result = self.engine.build_adapter_runtime(profile)
        else:
            result = ({}, {})
        self._adapter_runtime_cache[key] = result
        return result

    def _merge_and_validate_engine_args(self, profile: Profile) -> Dict[str, Any]:
        """
        Merge engine arguments from profile and user overrides, then validate.

        Merge precedence (lowest to highest):
        1. Profile defaults
        2. User overrides (from AIM_ENGINE_ARGS)
        3. System overrides (added by caller, e.g., port)

        Note: Security validation is NOT needed because arguments are passed directly
        to the engine process via os.execv() as an argument list, with no shell
        interpretation. This makes command injection impossible.

        Args:
            profile: Profile containing base engine arguments

        Returns:
            Merged and validated engine arguments dictionary

        Raises:
            ValidationError: If arguments fail validation
            ValueError: If native engine arg validation fails.
            pydantic.ValidationError: If profile structure validation fails.
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

            # Validate merged arguments
            try:
                # Validate profile structure (metadata, env_vars, aim_id, model_id)
                profile_data = {
                    "metadata": profile.metadata.to_dict(),
                    "aim_id": profile.aim_id,
                    "model_id": profile.model_id,
                    "engine_args": engine_args,
                    "env_vars": profile.env_vars or {},
                }
                is_general_profile = not profile.aim_id
                self.profile_validator.validate(profile_data, is_general_profile=is_general_profile)

                # Validate engine args via engine-specific validator
                self.engine.validate_engine_args(engine_args)

                logger.debug(f"Successfully validated {len(engine_args)} merged engine arguments")

            except PydanticValidationError as e:
                error_msg = f"Profile validation failed: {e}"
                logger.error(error_msg)
                raise

        return engine_args

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
