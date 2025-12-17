# Copyright © Advanced Micro Devices, Inc., or its affiliates.
#
# SPDX-License-Identifier: MIT

"""
AIM Profile Selector

This module contains the ProfileSelector class for finding the best profile
based on model and hardware configuration.
"""

import logging
import sys
from dataclasses import dataclass
from enum import Enum

from aim_common import ProfileType

# TODO: Remove this compatibility workaround once the ROCm base image is updated to Python 3.12
if sys.version_info >= (3, 11):
    from enum import StrEnum
else:

    class StrEnum(str, Enum):
        """Minimal StrEnum for Python <3.11."""


from typing import Dict, List, Tuple

from .config import AIMConfig
from .gpu_detector import GPUDetector
from .object_model import Engine, GPUModel, Metric, Precision, ProfileMetadata
from .profile_registry import Profile, ProfileRegistry
from .profile_validator import ProfileValidator

logger = logging.getLogger(__name__)

UNKNOWN_PRIORITY = 999


class ProfileNotFound(Exception):
    """Exception raised when a requested profile cannot be found."""


class ProfileCompatibilityState(StrEnum):
    """State categories for profile compatibility."""

    COMPATIBLE = "compatible"
    GPU_MISMATCH = "gpu_mismatch"
    PRECISION_MISMATCH = "precision_mismatch"
    MODEL_MISMATCH = "model_mismatch"
    ENGINE_MISMATCH = "engine_mismatch"
    METRIC_MISMATCH = "metric_mismatch"
    UNKNOWN = "unknown"


# Initialize ANSI color codes using enum directly as keys
PROFILE_STATE_COLORS = {
    ProfileCompatibilityState.COMPATIBLE: "\033[92m",  # Green
    ProfileCompatibilityState.GPU_MISMATCH: "\033[93m",  # Yellow
    ProfileCompatibilityState.PRECISION_MISMATCH: "\033[94m",  # Blue
    ProfileCompatibilityState.MODEL_MISMATCH: "\033[95m",  # Magenta
    ProfileCompatibilityState.ENGINE_MISMATCH: "\033[96m",  # Cyan
    ProfileCompatibilityState.METRIC_MISMATCH: "\033[91m",  # Red
    ProfileCompatibilityState.UNKNOWN: "\033[97m",  # White
    "reset": "\033[0m",  # Reset (special key for convenience)
}


@dataclass
class ProfileCompatibilityResult:
    """Result of profile compatibility assessment."""

    profile: Profile
    state: ProfileCompatibilityState
    reason: str = ""


class ProfileSelector:
    """Selects the best profile for a given model and hardware configuration."""

    def __init__(self, config: AIMConfig):
        """Initialize the profile selector with configuration."""
        self.config = config
        self.profile_validator = ProfileValidator(config.schema_search_path)

        # Check if GPU model is manually specified via environment variable
        if config.gpu_model is not None and config.gpu_model not in (GPUModel.NONE, GPUModel.UNKNOWN):
            logger.info(f"Using GPU model from AIM_GPU_MODEL: {config.gpu_model}")
            self.detected_gpu = config.gpu_model

            # Determine GPU count
            if config.gpu_count == "auto":
                logger.warning(
                    "AIM_GPU_MODEL is set but AIM_GPU_COUNT is 'auto'. "
                    "Defaulting to 1 GPU. Set AIM_GPU_COUNT explicitly if needed."
                )
                self.detected_gpu_count = 1
            else:
                self.detected_gpu_count = int(config.gpu_count)
        else:
            # Use GPU auto-detection
            gpu_detector = GPUDetector()
            if not gpu_detector.all_gpus_idle:
                logger.warning("Some GPUs are not idle! Check GPU usage.")

            if gpu_detector.has_gpus and gpu_detector.gpu_models:
                self.detected_gpu = gpu_detector.gpu_models[0]
                if config.gpu_count == "auto":
                    self.detected_gpu_count = gpu_detector.gpu_count
                else:
                    self.detected_gpu_count = int(config.gpu_count)
            else:
                self.detected_gpu = GPUModel.NONE
                self.detected_gpu_count = 0

        logger.debug(f"Detected GPU: {self.detected_gpu}, GPU count: {self.detected_gpu_count}")

        # Build registry at construction
        logger.info("Loading and validating all profiles...")
        search_paths = self._build_search_paths()
        self.registry: ProfileRegistry = ProfileRegistry.discover_and_validate(search_paths, self.profile_validator)

        # If general profile fallback is disabled, mark general profiles as manual-selection-only
        if not self.config.allow_general_profile_fallback:
            logger.info("General profile fallback disabled - marking general profiles as manual-selection-only")
            self._mark_general_profiles_manual_only()

        self.registry.log_summary()

    def _mark_general_profiles_manual_only(self) -> None:
        """Mark all general profiles in the registry as manual-selection-only."""
        for i, profile in enumerate(self.registry.profiles):
            if profile.profile_handling.is_general and not profile.metadata.manual_selection_only:
                # Create a new metadata object with manual_selection_only=True
                # Since ProfileMetadata is frozen, we need to create a new Profile object
                new_metadata = ProfileMetadata(
                    engine=profile.metadata.engine,
                    gpu=profile.metadata.gpu,
                    precision=profile.metadata.precision,
                    gpu_count=profile.metadata.gpu_count,
                    metric=profile.metadata.metric,
                    manual_selection_only=True,
                    type=profile.metadata.type,
                )
                # Replace the profile in the registry with updated metadata
                new_profile = Profile(
                    profile_handling=profile.profile_handling,
                    metadata=new_metadata,
                    aim_id=profile.aim_id,
                    model_id=profile.model_id,
                    engine_args=profile.engine_args,
                    env_vars=profile.env_vars,
                )
                # Update in the list
                self.registry.profiles[i] = new_profile

    def _build_search_paths(self) -> List[str]:
        """
        Build search paths with proper precedence according to AIM architecture:
        1. Custom path (<base_path>/custom/)
        2. Model-specific path (<base_path>/<org>/<model>/)
        3. General path (<base_path>/general/)
        """
        search_paths = []

        # 1. Custom path (by convention)
        custom_path = f"{self.config.profile_base_path}/custom"
        search_paths.append(custom_path)

        # 2. Model-specific path (based on aim_id which determines which profile set to use)
        if self.config.aim_id:
            # Extract org and model from aim_id (e.g., "meta-llama/Llama-3.1-8B-Instruct")
            model_parts = self.config.aim_id.split("/")
            if len(model_parts) >= 2:
                org = model_parts[0]
                model = "/".join(model_parts[1:])  # Handle cases with multiple slashes
                model_specific_path = f"{self.config.profile_base_path}/{org}/{model}"
                search_paths.append(model_specific_path)

        # 3. General path (by convention at <base_path>/general)
        general_path = f"{self.config.profile_base_path}/general"
        search_paths.append(general_path)

        return search_paths

    def _order_profiles(self, profiles: List[Profile]) -> List[Profile]:
        """Order profiles by priority first, then precision preference, then by type."""
        precision = self.config.precision

        # Define precision priority (lower number = higher priority)
        precision_priority = {
            Precision.INT4: 1,
            Precision.INT8: 2,
            Precision.FP4: 3,
            Precision.FP8: 4,
            Precision.FP16: 5,
            Precision.BF16: 6,
            Precision.FP32: 7,
        }

        type_priority = {
            ProfileType.OPTIMIZED: 1,
            ProfileType.PREVIEW: 2,
            ProfileType.UNOPTIMIZED: 3,
            ProfileType.GENERAL: 4,
        }

        if precision != Precision.AUTO:
            # If specific precision requested, filter for exact match first
            exact_matches = [p for p in profiles if p.matches_precision(precision)]
            if exact_matches:
                profiles = exact_matches

        # Sort by priority first (lower priority number = higher precedence),
        # then by precision priority (lower precision first for auto)
        # then by profile type priority (optimized over preview)
        def get_sort_key(profile: Profile) -> tuple:
            profile_precision = profile.metadata.precision
            precision_prio = precision_priority.get(profile_precision, UNKNOWN_PRIORITY)
            type_prio = type_priority.get(profile.metadata.type, UNKNOWN_PRIORITY)
            return (profile.profile_handling.priority, precision_prio, type_prio)

        sorted_profiles = sorted(profiles, key=get_sort_key)

        # Log the ordering for debugging
        logger.info(
            f"Ordered profiles by priority and precision: {[f'{p.profile_handling.filename}(prio={p.profile_handling.priority})' for p in sorted_profiles]}"
        )

        return sorted_profiles

    def find_profile(self) -> Profile:
        """
        Find the best profile using the merged assessment approach.

        Returns the highest-priority compatible profile.
        """
        if self.config.profile_id:
            # Handle specific profile ID request
            profile = self.registry.find_by_id(self.config.profile_id)
            if profile:
                logger.info(f"Using specified profile: {profile.profile_handling.path}")
                return profile
            else:
                raise ProfileNotFound(f"Specified profile ID '{self.config.profile_id}' not found")

        # Use merged assessment approach
        results = self.assess_all_profiles()
        compatible_results = results[ProfileCompatibilityState.COMPATIBLE]

        # Exclude profiles with metadata.manual_selection_only == True from automatic selection
        auto_compatible_results = [r for r in compatible_results if not r.profile.metadata.manual_selection_only]

        if not auto_compatible_results:
            # Provide detailed error message with breakdown
            error_parts = []
            for state, results_list in results.items():
                if results_list and state != ProfileCompatibilityState.COMPATIBLE:
                    error_parts.append(f"{len(results_list)} {state.value}")

            error_detail = ", ".join(error_parts) if error_parts else "no profiles found"
            raise ProfileNotFound(
                f"No compatible profile found for AIM {self.config.aim_id}. Profile breakdown: {error_detail}"
            )

        # Return the highest-priority auto-compatible profile
        best_profile = auto_compatible_results[0].profile
        logger.info(f"Selected profile: {best_profile.profile_handling.path}")
        return best_profile

    def get_categorized_profiles(self) -> Dict[ProfileCompatibilityState, List[Profile]]:
        """
        Get profiles categorized by compatibility state.

        This is now a simplified wrapper around assess_all_profiles().
        """
        results = self.assess_all_profiles()

        # Convert results to the expected format (profiles only, no detailed results)
        return {state: [result.profile for result in results_list] for state, results_list in results.items()}

    def assess_all_profiles(self) -> Dict[ProfileCompatibilityState, List[ProfileCompatibilityResult]]:
        """
        Assess all profiles and categorize by compatibility state.

        This replaces both the old filtering logic and the separate compatibility checking.
        Returns detailed results including reasons for incompatibility.
        """

        # Initialize result categories
        categorized_results: Dict[ProfileCompatibilityState, List[ProfileCompatibilityResult]] = {
            state: [] for state in ProfileCompatibilityState
        }

        # Get all valid profiles
        candidates = self.registry.profiles
        if not candidates:
            logger.warning("No valid profiles found in registry")
            return categorized_results

        # Resolve AUTO values once
        resolved_engine, resolved_metric = self._resolve_auto_values()

        logger.info(f"Assessing {len(candidates)} profiles with resolved config:")
        logger.info(f"  Engine: {resolved_engine}")
        logger.info(f"  Precision: {self.config.precision}")
        logger.info(f"  Detected GPU: {self.detected_gpu}")
        logger.info(f"  GPU Count: {self.detected_gpu_count}")
        logger.info(f"  Metric: {resolved_metric}")

        # Assess each profile in a single pass
        for profile in candidates:
            result = self._assess_profile_compatibility(profile, resolved_engine, resolved_metric)
            categorized_results[result.state].append(result)

        # Order compatible profiles by priority and precision preference
        if categorized_results[ProfileCompatibilityState.COMPATIBLE]:
            compatible_results = categorized_results[ProfileCompatibilityState.COMPATIBLE]
            ordered_profiles = self._order_profiles([r.profile for r in compatible_results])
            # Create a new list with the same ProfileCompatibilityResult objects but reordered
            # Create mapping by index since profiles can't be used as dict keys
            profile_id_to_result = {id(r.profile): r for r in compatible_results}
            categorized_results[ProfileCompatibilityState.COMPATIBLE] = [
                profile_id_to_result[id(profile)] for profile in ordered_profiles
            ]

        # Log summary
        for state, results in categorized_results.items():
            if results:
                logger.info(
                    f"{state.value}: {len(results)} profiles - {[r.profile.profile_handling.filename for r in results]}"
                )

        return categorized_results

    def _resolve_auto_values(self) -> Tuple[Engine, Metric]:
        """Resolve AUTO values to concrete preferences."""
        engine = self.config.engine
        if engine == Engine.AUTO:
            engine = Engine.VLLM  # Prefer vllm for auto

        metric = self.config.metric
        if metric == Metric.AUTO:
            metric = Metric.LATENCY  # Prefer latency for auto

        return engine, metric

    def _assess_profile_compatibility(
        self, profile: Profile, resolved_engine: Engine, resolved_metric: Metric
    ) -> ProfileCompatibilityResult:
        """
        Assess a single profile's compatibility with current configuration.

        This method checks compatibility in order of importance and returns the first
        incompatibility found, or COMPATIBLE if all checks pass.
        """
        # Check engine compatibility first (most basic requirement)
        if not profile.matches_engine(resolved_engine):
            return ProfileCompatibilityResult(
                profile=profile,
                state=ProfileCompatibilityState.ENGINE_MISMATCH,
                reason=f"Profile engine {profile.metadata.engine} doesn't match required {resolved_engine}",
            )

        # Check metric compatibility
        if not profile.matches_metric(resolved_metric):
            return ProfileCompatibilityResult(
                profile=profile,
                state=ProfileCompatibilityState.METRIC_MISMATCH,
                reason=f"Profile metric {profile.metadata.metric} doesn't match required {resolved_metric}",
            )

        # Check profile compatibility (profiles match against aim_id, not the deployed model_id)
        if not profile.matches_aim_id(self.config.aim_id):
            return ProfileCompatibilityResult(
                profile=profile,
                state=ProfileCompatibilityState.MODEL_MISMATCH,
                reason=f"Profile doesn't support AIM ID {self.config.aim_id}",
            )

        # Check GPU compatibility (type and count)
        if not profile.matches_gpu(self.detected_gpu):
            return ProfileCompatibilityResult(
                profile=profile,
                state=ProfileCompatibilityState.GPU_MISMATCH,
                reason=f"Profile doesn't support GPU type {self.detected_gpu}",
            )

        if not profile.matches_gpu_count(self.detected_gpu_count):
            return ProfileCompatibilityResult(
                profile=profile,
                state=ProfileCompatibilityState.GPU_MISMATCH,
                reason=f"Profile doesn't support {self.detected_gpu_count} GPUs",
            )

        # Check precision compatibility
        if self.config.precision != Precision.AUTO and not profile.matches_precision(self.config.precision):
            return ProfileCompatibilityResult(
                profile=profile,
                state=ProfileCompatibilityState.PRECISION_MISMATCH,
                reason=f"Profile doesn't support precision {self.config.precision}",
            )

        # All checks passed
        return ProfileCompatibilityResult(
            profile=profile,
            state=ProfileCompatibilityState.COMPATIBLE,
            reason="Profile is compatible with current configuration",
        )

    def format_text_report(self, categorized: Dict[ProfileCompatibilityState, List[Profile]]) -> str:
        """
        Format profile compatibility report as human-readable text.

        Args:
            categorized: Dictionary mapping ProfileCompatibilityState to list of profiles

        Returns:
            Formatted text report as string
        """
        lines = []
        lines.append("AIM Profile Compatibility Report")
        lines.append("=" * 50)
        lines.append(f"AIM ID: {self.config.aim_id}")
        lines.append(f"Model ID: {self.config.model_id}")
        lines.append(f"Precision: {self.config.precision}")
        lines.append(f"Engine: {self.config.engine}")
        lines.append(f"Metric: {self.config.metric}")
        lines.append(f"GPU Count: {self.config.gpu_count}")
        lines.append(f"GPU Model: {self.detected_gpu}")
        lines.append("")

        total_profiles = sum(len(profiles) for profiles in categorized.values())
        lines.append(f"Total profiles analyzed: {total_profiles}")
        lines.append("")

        for state, profiles in categorized.items():
            if not profiles:
                continue

            lines.append(f"{state.value.upper().replace('_', ' ')} ({len(profiles)} profiles):")
            lines.append("-" * 40)

            for profile in profiles:
                manual_flag = " [manual-only]" if profile.metadata.manual_selection_only else ""
                lines.append(f"  • {profile.profile_id}{manual_flag}")
                lines.append(f"    GPU: {profile.metadata.gpu}")
                lines.append(f"    Precision: {profile.metadata.precision}")
                lines.append(f"    Engine: {profile.metadata.engine}")
                lines.append(f"    Type: {profile.metadata.type}")
                lines.append(f"    Priority: {profile.profile_handling.priority}")
            lines.append("")

        return "\n".join(lines)

    def _format_profile_table(
        self, profiles_with_states: List[Tuple[Profile, ProfileCompatibilityState]], include_legend: bool = True
    ) -> List[str]:
        """
        Helper method to format profiles as a table with optional state colors.

        Args:
            profiles_with_states: List of (Profile, State) tuples
            include_legend: Whether to include the color legend at the end

        Returns:
            List of formatted lines
        """
        lines = []

        # Table headers - reordered with State at the end
        headers = [
            "Profile",
            "GPU",
            "Precision",
            "Engine",
            "TP",
            "Metric",
            "Type",
            "Priority",
            "Manual Only",
            "Compatibility",
        ]

        # Calculate column widths
        col_widths = [len(h) for h in headers]

        # Calculate width for each column based on content
        for profile, state in profiles_with_states:
            profile_id = profile.profile_id
            manual_only_display = "Yes" if profile.metadata.manual_selection_only else "No"
            col_widths[0] = max(col_widths[0], len(profile_id))
            col_widths[1] = max(col_widths[1], len(profile.metadata.gpu.value))
            col_widths[2] = max(col_widths[2], len(profile.metadata.precision.value))
            col_widths[3] = max(col_widths[3], len(profile.metadata.engine.value))
            col_widths[4] = max(col_widths[4], len(str(profile.metadata.gpu_count)))
            col_widths[5] = max(col_widths[5], len(profile.metadata.metric.value))
            col_widths[6] = max(col_widths[6], len(profile.metadata.type.value))
            col_widths[7] = max(col_widths[7], len(str(profile.profile_handling.priority)))
            col_widths[8] = max(col_widths[8], len(manual_only_display))
            col_widths[9] = max(col_widths[9], len(state.value.replace("_", " ").title()))

        # Print table header
        header_row = "| " + " | ".join(h.ljust(col_widths[i]) for i, h in enumerate(headers)) + " |"
        lines.append(header_row)
        lines.append("|" + "|".join("-" * (col_widths[i] + 2) for i in range(len(headers))) + "|")

        # Print table rows with colors
        for profile, state in profiles_with_states:
            profile_id = profile.profile_id
            gpu = profile.metadata.gpu.value
            precision = profile.metadata.precision.value
            engine = profile.metadata.engine.value
            tp_size = str(profile.metadata.gpu_count)
            metric = profile.metadata.metric.value
            profile_type = profile.metadata.type.value
            priority = str(profile.profile_handling.priority)
            manual_only_display = "Yes" if profile.metadata.manual_selection_only else "No"
            state_display = state.value

            # Get color for this state
            color = PROFILE_STATE_COLORS.get(state, PROFILE_STATE_COLORS["reset"])
            reset = PROFILE_STATE_COLORS["reset"]

            row_data = [
                profile_id,
                gpu,
                precision,
                engine,
                tp_size,
                metric,
                profile_type,
                priority,
                manual_only_display,
                state_display,
            ]
            row = (
                f"{color}| " + " | ".join(row_data[i].ljust(col_widths[i]) for i in range(len(row_data))) + f" |{reset}"
            )
            lines.append(row)

        if include_legend:
            lines.append("")
            # Print color legend
            lines.append("State Color Legend:")
            for state in ProfileCompatibilityState:
                if state in PROFILE_STATE_COLORS:
                    color_code = PROFILE_STATE_COLORS[state]
                    display_name = state.value.replace("_", " ").title()
                    lines.append(f"  {color_code}■{PROFILE_STATE_COLORS['reset']} {display_name}")

        return lines

    def format_table_report(self, categorized: Dict[ProfileCompatibilityState, List[Profile]]) -> str:
        """
        Format profile compatibility report as table with ANSI colors.

        Args:
            categorized: Dictionary mapping ProfileCompatibilityState to list of profiles

        Returns:
            Formatted table report as string with ANSI color codes
        """
        lines = []
        lines.append("AIM Profile Compatibility Report")
        lines.append("=" * 100)
        lines.append(f"AIM ID: {self.config.aim_id}")
        lines.append(f"Model ID: {self.config.model_id}")
        lines.append(
            f"GPU Count: {self.config.gpu_count} | GPU Model: {self.detected_gpu} | "
            f"Precision: {self.config.precision} | Engine: {self.config.engine} | Metric: {self.config.metric}"
        )
        lines.append("")

        # Collect all profiles with their states
        all_profiles = []
        for state, profiles in categorized.items():
            for profile in profiles:
                all_profiles.append((profile, state))

        if not all_profiles:
            lines.append("No profiles found.")
            return "\n".join(lines)

        lines.append(f"Total profiles analyzed: {len(all_profiles)}")
        lines.append("")

        # Use helper method to format the table
        table_lines = self._format_profile_table(all_profiles, include_legend=True)
        lines.extend(table_lines)
        lines.append("")

        return "\n".join(lines)

    def format_all_profiles_report(self, format_type: str = "text") -> str:
        """
        Format a report of all profiles without GPU detection or compatibility checks.

        Args:
            format_type: Output format ("text" or "table")

        Returns:
            Formatted report as string
        """
        profiles = self.registry.profiles

        if not profiles:
            return "No profiles found."

        if format_type == "table":
            lines = []
            lines.append("AIM Profile Compatibility Report (No Compatibility Checks)")
            lines.append("=" * 100)
            lines.append(f"AIM ID: {self.config.aim_id}")
            lines.append(f"Model ID: {self.config.model_id}")
            lines.append(f"Total profiles discovered: {len(profiles)}")
            lines.append("")

            # Create list of (profile, UNKNOWN state) tuples and sort by profile_id
            profiles_with_state = [
                (p, ProfileCompatibilityState.UNKNOWN)
                for p in sorted(profiles, key=lambda p: (p.profile_handling.priority, p.profile_id))
            ]

            # Reuse the table formatting helper
            table_lines = self._format_profile_table(profiles_with_state, include_legend=True)
            lines.extend(table_lines)

            return "\n".join(lines)

        else:
            # Text format (grouped by GPU model)
            from collections import defaultdict

            lines = []
            lines.append("AIM Profile Compatibility Report (No Compatibility Checks)")
            lines.append("=" * 50)
            lines.append(f"AIM ID: {self.config.aim_id}")
            lines.append(f"Model ID: {self.config.model_id}")
            lines.append(f"Total profiles discovered: {len(profiles)}")
            lines.append("")

            profiles_by_gpu = defaultdict(list)
            for profile in profiles:
                profiles_by_gpu[profile.metadata.gpu].append(profile)

            for gpu_model in sorted(profiles_by_gpu.keys(), key=str):
                gpu_profiles = sorted(profiles_by_gpu[gpu_model], key=lambda p: p.profile_id)
                lines.append(f"{gpu_model} ({len(gpu_profiles)} profiles):")
                lines.append("-" * 40)

                for profile in gpu_profiles:
                    manual_flag = " [manual-only]" if profile.metadata.manual_selection_only else ""
                    lines.append(f"  • {profile.profile_id}{manual_flag}")
                    lines.append(f"    Precision: {profile.metadata.precision}")
                    lines.append(f"    Engine: {profile.metadata.engine}")
                    lines.append(f"    TP: {profile.metadata.gpu_count}")
                    lines.append(f"    Metric: {profile.metadata.metric}")
                    lines.append(f"    Type: {profile.metadata.type}")
                    lines.append(f"    Priority: {profile.profile_handling.priority}")
                lines.append("")

            return "\n".join(lines)
