# Copyright © Advanced Micro Devices, Inc., or its affiliates.
#
# SPDX-License-Identifier: MIT
"""Utilities for managing AIM profiles."""

import logging
import numbers
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Type, Union

import click
import pandas as pd

from aim_common.object_model import CanonicalName, GPUModel, ProfileType
from aim_runtime.object_model import ProfileHandling
from aim_utils.yaml_utils import FileType, get_yamls, read_yaml, resolve_paths, save_yaml, sort_yaml_file

logger = logging.getLogger(__name__)


class BenchmarkFileValueResolver:
    """Resolves gate values from benchmark Excel file."""

    def __init__(self, file_path: Path, sheet_name: Optional[str] = None):
        """Load and parse gate values from an Excel file.

        Args:
            file_path: Path to the Excel file containing benchmark data.
            sheet_name: Optional; Name or index of the sheet to read from the Excel file. Defaults to the first sheet if not specified.
        """

        def image_name_to_aim(name: str) -> str:
            split_result = name.split("/")
            aim_with_version = split_result[-1]
            aim, _ = aim_with_version.split(":")
            return aim

        final_sheet_name: Union[str, int] = 0
        if sheet_name is not None:
            final_sheet_name = 0
        columns_variants = [
            ["Unnamed: 0", "profile", "gate value", "type (manual)"],
            ["AIM", "profile", "gate value", "type (manual)"],
        ]
        for columns in columns_variants:
            try:
                df = pd.read_excel(file_path, sheet_name=final_sheet_name, usecols=columns)
                if not df.empty:
                    df = df.rename(columns={"Unnamed: 0": "aim", "AIM": "aim"})
                    df["aim"] = df["aim"].ffill()
                    df["aim"] = df["aim"].apply(image_name_to_aim)
                    self.df = df
                    break
            except ValueError:
                continue
        else:
            raise ValueError(
                f"Could not parse {file_path} with any of the expected column variants: {columns_variants}"
            )

    def _get_value(
        self, profile_name: str, aim: str, column_name: str, value_type: Type
    ) -> Optional[Union[float, str]]:
        profiles = self.df[self.df["profile"] == profile_name]
        profiles = profiles[profiles["aim"] == aim]
        if not profiles.empty:
            values = profiles[column_name].values
            value = values[0]
            if not isinstance(value, value_type):
                return None
            return value
        return None

    def get_gate_value(self, profile_name: str, aim: str) -> Optional[float]:
        """Get gate value for a given profile and AIM."""
        result = self._get_value(profile_name, aim, "gate value", numbers.Real)
        return float(result) if isinstance(result, numbers.Real) else None

    def get_type_value(self, profile_name: str, aim: str) -> Optional[str]:
        """Get type value for a given profile and AIM."""
        result = self._get_value(profile_name, aim, "type (manual)", str)
        return result if isinstance(result, str) else None


@dataclass
class ProfileTypeEvaluationResult:
    """Result of profile type evaluation."""

    profile_type: Optional[ProfileType] = None
    manual_selection_only: bool = False


class ProfileTypeEvaluator:
    """Evaluates profile optimization level based on performance metrics."""

    def __init__(
        self,
        profile_path: Path,
        value_resolver: BenchmarkFileValueResolver,
        preview_threshold: float,
        optimized_threshold: float,
    ):
        """Initialize evaluator with profile data and value resolver.

        Args:
            profile_path: Path to the profile YAML file.
            value_resolver: Instance of BenchmarkFileValueResolver to fetch values from benchmark file.
            preview_threshold: Performance ratio threshold for classifying profile as "preview".
            optimized_threshold: Performance ratio threshold for classifying profile as "optimized".
        """
        self.preview_threshold = preview_threshold
        self.optimized_threshold = optimized_threshold

        profile_data = read_yaml(profile_path)
        profile_handling = ProfileHandling(str(profile_path), profile_path.name, 1)
        self.is_general = profile_handling.is_general
        self.aim_id = profile_data.get("aim_id")

        image_name = self._get_image_name()

        self.aim_vs_nim = value_resolver.get_gate_value(profile_handling.profile_name, image_name)
        # This value is always set to None in the current implementation
        self.aim_vs_oob_vllm = None

        type_value = value_resolver.get_type_value(profile_handling.profile_name, image_name)
        self.profile_type = None

        if type_value is not None:
            try:
                self.profile_type = ProfileType(type_value.lower())
            except ValueError as e:
                logger.warning(f"Invalid profile type '{type_value}' for profile '{profile_handling.profile_name}'")
                raise e

    def _get_image_name(self) -> str:
        """Get AIM image name based on profile type and ID."""
        if self.is_general:
            return "aim-base"
        if self.aim_id is not None:
            sanitized_name = CanonicalName.from_string(self.aim_id).sanitize
            return f"aim-{sanitized_name}"
        return "aim"

    def is_optimized(self) -> bool:
        if self.is_general:
            return False

        if self.aim_vs_nim is not None:
            return self.aim_vs_nim > self.optimized_threshold

        if self.aim_vs_oob_vllm is not None:
            return self.aim_vs_oob_vllm > self.optimized_threshold

        return False

    def is_preview(self) -> bool:
        """Check if profile is in preview (performance ratio between preview threshold and optimized threshold)."""
        if self.is_general:
            return False

        if self.aim_vs_nim is not None:
            return self.preview_threshold <= self.aim_vs_nim <= self.optimized_threshold

        if self.aim_vs_oob_vllm is not None:
            return self.preview_threshold <= self.aim_vs_oob_vllm <= self.optimized_threshold

        return False

    def is_unoptimized(self) -> bool:
        """Check if profile is unoptimized (performance ratio < self.preview_threshold)."""
        if self.is_general:
            return False

        if self.aim_vs_nim is not None:
            return self.aim_vs_nim < self.preview_threshold

        if self.aim_vs_oob_vllm is not None:
            return self.aim_vs_oob_vllm < self.preview_threshold

        return False

    def evaluate(self) -> ProfileTypeEvaluationResult:
        calculated_profile_type = self._get_profile_type()
        manually_specified_profile_type = self.profile_type

        if (
            calculated_profile_type is not None
            and manually_specified_profile_type is not None
            and calculated_profile_type != manually_specified_profile_type
        ):
            logger.warning(
                f"Calculated profile type '{calculated_profile_type}' does not match manually specified type '{manually_specified_profile_type}'; aim_id: '{self.aim_id}''"
            )

        # Map final profile type based on presence of calculated and manually specified types
        # Key: (calculated exists, manual exists)
        final_profile_type_mapping = {
            (False, False): None,
            (False, True): manually_specified_profile_type,
            (True, False): calculated_profile_type,
            (True, True): manually_specified_profile_type,
        }

        profile_type_manual_selection_mapping: Dict[Optional[ProfileType], bool] = {
            ProfileType.UNOPTIMIZED: True,
            ProfileType.GENERAL: False,
            ProfileType.OPTIMIZED: False,
            ProfileType.PREVIEW: False,
        }

        final_profile_type = final_profile_type_mapping[
            (calculated_profile_type is not None, manually_specified_profile_type is not None)
        ]

        return ProfileTypeEvaluationResult(
            profile_type=final_profile_type,
            manual_selection_only=profile_type_manual_selection_mapping.get(final_profile_type, True),
        )

    def _get_profile_type(self) -> Optional[ProfileType]:
        """Determine profile type based on optimization metrics."""
        if self.is_optimized():
            return ProfileType.OPTIMIZED
        elif self.is_preview():
            return ProfileType.PREVIEW
        elif self.is_unoptimized():
            return ProfileType.UNOPTIMIZED
        if self.is_general:
            return ProfileType.GENERAL
        return None


@click.group(invoke_without_command=True)
@click.pass_context
def cli(ctx):
    pass


@cli.command("sync-profiles-with-benchmark")
@click.argument("preview_threshold", type=float)
@click.argument("optimized_threshold", type=float)
@click.option("--profiles_path", type=str, default="profiles")
@click.option("--benchmark_path", type=str, required=True)
@click.option("--sheet_name", type=str, required=False, default=None)
def sync_profiles_with_benchmark(
    preview_threshold: float,
    optimized_threshold: float,
    profiles_path: str,
    benchmark_path: str,
    sheet_name: Optional[str] = None,
) -> None:
    """Synchronize profile metadata with benchmark gate values."""
    value_resolver = BenchmarkFileValueResolver(Path(benchmark_path), sheet_name=sheet_name)

    profiles = get_yamls(Path(profiles_path))
    for profile in profiles:
        evaluator = ProfileTypeEvaluator(profile, value_resolver, preview_threshold, optimized_threshold)
        evaluation_result = evaluator.evaluate()

        if evaluation_result.profile_type is not None:
            profile_data = read_yaml(profile)
            profile_data["metadata"]["type"] = evaluation_result.profile_type.value
            profile_data["metadata"]["manual_selection_only"] = evaluation_result.manual_selection_only
            save_yaml(
                profile_data,
                path=profile,
                enforce_double_quotes=False,
            )
        else:
            logger.debug(f"Skipped profile '{profile}': profile_type is None.")


@dataclass
class MetadataMismatch:
    profile_path: Path
    field_name: str
    expected_value: Optional[str]
    actual_value: Optional[str]


@cli.command("check-metadata")
@click.option("--profiles_path", type=str, default="profiles")
@click.option("--canonical_name", type=str, help="Filter by model canonical name (format: 'org/model')")
def check_profile_metadata(profiles_path: str = "profiles", canonical_name: Optional[str] = None) -> None:
    """Check that profile metadata matches profile filenames."""

    def check_value(
        profile_path: Path, field_name: str, expected_value: str, actual_value: str
    ) -> Optional[MetadataMismatch]:
        if expected_value is None:
            logger.error(f"Expected value for '{field_name}' not found.")
            return MetadataMismatch(profile_path, field_name, expected_value, actual_value)
        if actual_value is None:
            logger.error(f"Actual value for '{field_name}' not found.")
            return MetadataMismatch(profile_path, field_name, expected_value, actual_value)

        expected_value = expected_value.lower()
        actual_value = actual_value.lower()

        if expected_value != actual_value:
            logger.error(f"Mismatch: expected '{expected_value}', got '{actual_value}'")
            return MetadataMismatch(profile_path, field_name, expected_value, actual_value)

        return None

    def append_if_not_none(list_obj: List[MetadataMismatch], item: Optional[MetadataMismatch]) -> None:
        if item is not None:
            list_obj.append(item)

    profiles = resolve_paths(directory=profiles_path, canonical_name=canonical_name)

    mismatches: List[MetadataMismatch] = []
    for profile in profiles:
        profile_data = read_yaml(profile)
        engine, gpu, precision, gpu_count, metric = profile.stem.split("-")
        precision = precision.replace("mx", "")

        metadata = profile_data.get("metadata", {})
        append_if_not_none(mismatches, check_value(profile, "engine", engine, metadata.get("engine")))
        append_if_not_none(mismatches, check_value(profile, "gpu", gpu, metadata.get("gpu")))
        append_if_not_none(mismatches, check_value(profile, "precision", precision, metadata.get("precision")))
        append_if_not_none(mismatches, check_value(profile, "gpu_count", gpu_count, f'tp{metadata.get("gpu_count")}'))
        append_if_not_none(mismatches, check_value(profile, "metric", metric, metadata.get("metric")))

        if metadata.get("type") == "general":
            if "general" not in profile.parts:
                append_if_not_none(mismatches, check_value(profile, "type", "general", metadata.get("type")))

    if len(mismatches) > 0:
        logger.error(f"❌ Found {len(mismatches)} errors out of {len(profiles)} profiles")
        exit(1)
    else:
        logger.info(f"✅ All {len(profiles)} profiles passed metadata check.")
        exit(0)


@cli.command("clone-profiles-gpu-name")
@click.option("--profile_path", type=str, default="profiles", help="The directory to scan for profiles.")
@click.option("--old-gpu-name", type=str, required=True, help="The old GPU name to replace (e.g., MI300X).")
@click.option(
    "--new-gpu-name", type=str, required=True, help="The new GPU name to use for the cloned profile (e.g., MI325X)."
)
def clone_profiles(profile_path: str, old_gpu_name: str, new_gpu_name: str):
    """
    Clone GPU profiles by updating the GPU name.

    Recursively scans a directory for YAML files with old_gpu_name in their name,
    updates the GPU profile from old_gpu_name to new_gpu_name, and saves a new file.
    """
    root_path = Path(profile_path)
    if not root_path.is_dir():
        logging.error(f"Directory '{profile_path}' not found.")
        return

    try:
        GPUModel.from_string(old_gpu_name)
        GPUModel.from_string(new_gpu_name)
    except ValueError as e:
        logging.error(f"Invalid GPU name: {e}")
        return

    logging.info(f"Scanning directory: '{profile_path}'...")

    yamls = get_yamls(root_path)

    for filepath in yamls:
        # Process only YAML files containing old_gpu_name
        if old_gpu_name.lower() in filepath.name.lower():
            try:
                data = read_yaml(filepath)

                # Check and update the GPU value
                if (
                    data
                    and "metadata" in data
                    and "gpu" in data["metadata"]
                    and data["metadata"]["gpu"] == old_gpu_name
                ):

                    logging.info(f"Found '{old_gpu_name}' in '{filepath.name}'. Updating...")
                    data["metadata"]["gpu"] = new_gpu_name

                    # Create the new filename and path
                    new_filename = filepath.name.lower().replace(old_gpu_name.lower(), new_gpu_name.lower())
                    new_filepath = filepath.with_name(new_filename)

                    # Save the modified data to the new file
                    save_yaml(data, new_filepath)

                    logging.info(f"  -> Saved new profile to '{new_filepath}'")
                else:
                    logging.warning(f"Skipping '{filepath.name}': '{old_gpu_name}' not found in metadata.gpu field.")

            except Exception as e:
                logging.error(f"An error occurred while processing '{filepath.name}': {e}")


@cli.command("sort-profile-keys")
@click.argument("files", nargs=-1, type=str)
def sort_profile_keys(files: List[str]):

    yaml_paths = resolve_paths(files, "profiles")

    modified_count = 0
    error_count = 0

    for yaml_file in yaml_paths:
        try:
            if sort_yaml_file(yaml_file, FileType.PROFILE):
                modified_count += 1
        except Exception as e:
            logger.error(f"Error processing {yaml_file}: {e}")
            error_count += 1

    if modified_count > 0 or error_count > 0:
        if modified_count > 0:
            logger.info(f"✅ {modified_count}/{len(files)} files were sorted")
        if error_count > 0:
            logger.error(f"❌ Encountered {error_count} errors")
        sys.exit(1)

    sys.exit(0)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
    cli()
