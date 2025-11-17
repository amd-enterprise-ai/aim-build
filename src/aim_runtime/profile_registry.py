# Copyright © Advanced Micro Devices, Inc., or its affiliates.
#
# SPDX-License-Identifier: MIT

"""
AIM Profile Registry

This module contains dataclasses and functionality for discovering, loading,
and validating all available profiles before selection.
"""

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

import yaml
from jsonschema import ValidationError

from .object_model import Engine, GPUModel, Metric, Precision, Profile, ProfileHandling, ProfileMetadata, ProfileType
from .profile_validator import ProfileValidator

logger = logging.getLogger(__name__)


@dataclass
class ProfileRegistry:
    """Registry of all discovered and validated profiles."""

    profiles: List[Profile]
    search_path: str
    total_discovered: int

    @classmethod
    def discover_and_validate(cls, search_paths: List[str], validator: ProfileValidator) -> "ProfileRegistry":
        """
        Discover and validate all profiles from the given search paths.

        Args:
            search_paths: List of directories to search for profiles (in precedence order)
            validator: ProfileValidator instance for validation

        Returns:
            ProfileRegistry with all discovered valid profiles
        """
        logger.info(f"Discovering profiles with search precedence: {' -> '.join(search_paths)}")

        all_discovered_profiles = []
        profile_ids_seen = set()  # Track profile IDs to implement precedence

        # Process each search path in precedence order
        for priority, search_path in enumerate(search_paths, 1):  # priority starts at 1
            search_path_obj = Path(search_path)
            if not search_path_obj.exists():
                logger.debug(f"Search path does not exist (skipping): {search_path}")
                continue

            logger.debug(f"Searching in: {search_path} (priority {priority})")
            path_profiles = []

            # Find all YAML files recursively in this path
            for profile_file in search_path_obj.rglob("*.yaml"):
                try:
                    profile = cls._load_and_validate_profile(str(profile_file), validator, priority)

                    # Implement precedence: only add if we haven't seen this profile ID
                    if profile.profile_id not in profile_ids_seen:
                        path_profiles.append(profile)
                        profile_ids_seen.add(profile.profile_id)
                        logger.debug(f"✓ Valid profile: {profile.profile_id} (priority {priority}, from {search_path})")
                    else:
                        logger.debug(f"Profile {profile.profile_id} already found in higher precedence path, skipping")

                except ValidationError as e:
                    logger.warning(f"✗ Invalid profile: {profile_file} - Validation error: {e.message}")
                except Exception as e:
                    logger.warning(f"Failed to process profile file {profile_file}: {e}")

            all_discovered_profiles.extend(path_profiles)
            logger.debug(f"Found {len(path_profiles)} profiles in {search_path}")

        # Calculate summary statistics
        total_discovered = len(all_discovered_profiles)

        logger.info("Profile discovery complete:")
        logger.info(f"  Total valid profiles found: {total_discovered}")

        return cls(
            profiles=all_discovered_profiles,
            search_path=" -> ".join(search_paths),
            total_discovered=total_discovered,
        )

    @staticmethod
    def _load_and_validate_profile(profile_path: str, validator: ProfileValidator, priority: int) -> Profile:
        """
        Load and validate a single profile.

        Args:
            profile_path: Path to the profile file
            validator: ProfileValidator instance
            priority: Priority level (1 = highest priority)

        Returns:
            Profile object (raises ValidationError if invalid)
        """
        profile_file = Path(profile_path)

        # Load the YAML content
        with open(profile_path, "r", encoding="utf-8") as f:
            profile_data = yaml.safe_load(f)

        # Determine if this is a general or model-specific profile
        profile_handling = ProfileHandling(path=profile_path, filename=profile_file.name, priority=priority)
        is_general = profile_handling.is_general

        # Validate the profile using the already-loaded data (this will raise ValidationError if invalid)
        validator.validate(profile_data, is_general_profile=is_general)

        metadata_dict = profile_data["metadata"]
        metadata = ProfileMetadata(
            engine=Engine(metadata_dict["engine"].lower()),
            gpu=GPUModel(metadata_dict["gpu"].upper()),
            precision=Precision(metadata_dict["precision"].lower()),
            gpu_count=metadata_dict["gpu_count"],
            metric=Metric(metadata_dict["metric"].lower()),
            manual_selection_only=metadata_dict["manual_selection_only"],
            type=ProfileType(metadata_dict["type"].lower()),
        )

        # Model-specific profiles have aim_id and model_id
        # General profiles don't have these fields
        if not is_general:
            aim_id = profile_data["aim_id"]
            model_id = profile_data["model_id"]
        else:
            # General profiles don't have aim_id or model_id
            aim_id = ""
            model_id = ""

        return Profile(
            profile_handling=profile_handling,
            aim_id=aim_id,
            model_id=model_id,
            metadata=metadata,
            engine_args=profile_data["engine_args"],
            env_vars=profile_data["env_vars"],
        )

    def find_by_id(self, profile_id: str) -> Optional[Profile]:
        """Find a profile by its ID (filename without extension)."""
        for profile in self.profiles:
            if profile.profile_id == profile_id:
                return profile
        return None

    def get_general_profiles(self) -> List[Profile]:
        """Get all general profiles."""
        return [p for p in self.profiles if p.profile_handling.is_general]

    def log_summary(self) -> None:
        """Log a summary of the profile registry."""
        logger.info("=== Profile Registry Summary ===")
        logger.info(f"Search path: {self.search_path}")
        logger.info(f"Total valid profiles discovered: {self.total_discovered}")

        logger.info("\nProfiles by category:")

        # Group by engine, GPU, etc.
        engines = set(p.metadata.engine for p in self.profiles)
        for engine in sorted(engines):
            engine_profiles = [p for p in self.profiles if p.metadata.engine == engine]
            logger.info(f"  {engine}: {len(engine_profiles)} profiles")

        gpus = set(p.metadata.gpu for p in self.profiles)
        for gpu in sorted(gpus):
            gpu_profiles = [p for p in self.profiles if p.metadata.gpu == gpu]
            logger.info(f"  {gpu}: {len(gpu_profiles)} profiles")

        general_count = len(self.get_general_profiles())
        model_specific_count = len(self.profiles) - general_count
        logger.info(f"  General profiles: {general_count}")
        logger.info(f"  Model-specific profiles: {model_specific_count}")

        logger.info("\nProfile listing:")
        for p in self.profiles:
            manual_flag = " [manual-only]" if p.metadata.manual_selection_only else ""
            logger.info(
                f"    {p.profile_handling.filename}{manual_flag} (engine={p.metadata.engine}, gpu={p.metadata.gpu}, precision={p.metadata.precision}, metric={p.metadata.metric}, gpu_count={p.metadata.gpu_count}, type={p.metadata.type})"
            )

        logger.info("===============================")
