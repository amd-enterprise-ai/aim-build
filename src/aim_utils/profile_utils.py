# Copyright © Advanced Micro Devices, Inc., or its affiliates.
#
# SPDX-License-Identifier: MIT
"""Utilities for managing AIM profiles."""

import logging
import sys
from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Type, Union

import click
from pydantic import ValidationError

from aim_common.object_model import CanonicalName, GPUModel, ProfileMetadata, ProfileType
from aim_runtime.object_model import ProfileHandling
from aim_utils.asset_utils import (
    AssetDescriptor,
    AssetManager,
    assets_path_option,
    assets_root_option,
    discover_assets_paths,
)
from aim_utils.dict_utils import get_value, set_value
from aim_utils.image_naming import get_image_name
from aim_utils.yaml_utils import FileType, get_yamls, read_yaml, save_yaml, sort_yaml_file

logger = logging.getLogger(__name__)


COLUMN_NAMES_MAPPING = {
    "Profile": "profile",
    "Type": "type",
    "type (manual)": "type",
    "Unnamed 0": "aim",
    "AIM": "aim",
    "Docker_Image": "aim",
}


class ProfileFileValueResolver:
    """Resolves profile type values from an Excel file."""

    def __init__(self, file_path: Path, sheet_name: Optional[str] = None):
        """Load and parse profile type values from an Excel file.

        Args:
            file_path: Path to the Excel file containing profile data.
            sheet_name: Optional; Name or index of the sheet to read from the Excel file. Defaults to the first sheet if not specified.
        """

        def drop_tag(name: str) -> str:
            return name.split(":")[0]

        def convert_to_repository_name(name: str) -> str:
            split_result = name.split("/")
            aim_with_version = split_result[-1]
            return drop_tag(aim_with_version)

        import pandas as pd

        final_sheet_name: Union[str, int] = sheet_name or 0
        try:
            df = pd.read_excel(file_path, sheet_name=final_sheet_name)
            if not df.empty:
                if "AIM" in df.columns:
                    df = df.rename(columns={"AIM": "aim"})
                    df["aim"] = df["aim"].ffill()
                    df["aim"] = df["aim"].apply(convert_to_repository_name)

                if "Docker_Image" in df.columns:
                    df["Docker_Image"] = df["Docker_Image"].apply(convert_to_repository_name)

                for old_name, new_name in COLUMN_NAMES_MAPPING.items():
                    df = df.rename(columns={old_name: new_name} if old_name in df.columns else {})

                self.df = df
        except Exception as e:
            logger.exception(e)

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

    def get_type_value(self, profile_name: str, aim: str) -> Optional[str]:
        """Get type value for a given profile and AIM."""
        result = self._get_value(profile_name, aim, "type", str)
        return result if isinstance(result, str) else None


@dataclass
class ProfileTypeEvaluationResult:
    """Result of profile type evaluation."""

    profile_type: Optional[ProfileType] = None
    manual_selection_only: bool = False


class ProfileTypeEvaluator:
    """Reads and applies profile type from profile data file."""

    def __init__(
        self,
        profile_path: Path,
        value_resolver: ProfileFileValueResolver,
    ):
        self.profile_path = profile_path
        profile_data = read_yaml(profile_path)
        profile_handling = ProfileHandling(str(profile_path), profile_path.name, 1)
        self.is_general = profile_handling.is_general
        self.aim_id = profile_data.get("aim_id")

        image_name = self._get_image_name()

        type_value = value_resolver.get_type_value(profile_handling.profile_name, image_name)
        self.profile_type = None

        if type_value is not None:
            try:
                self.profile_type = ProfileType(type_value.lower())
            except ValueError:
                logger.warning(f"Invalid profile type '{type_value}' for profile '{profile_handling.profile_name}'")
                raise

    def _get_image_name(self) -> str:
        accelerator_family = self._get_accelerator_family()

        if self.is_general:
            image_name = get_image_name(accelerator_family, is_base=True)
            return image_name.canonical
        elif self.aim_id is not None:
            sanitized_name = CanonicalName.from_string(self.aim_id).sanitize  # type: ignore[union-attr]
            image_name = get_image_name(accelerator_family, canonical_name_sanitized=sanitized_name, is_base=False)
            return image_name.canonical
        else:
            raise ValueError(
                f"Cannot determine image name for profile '{self.profile_path}': "
                "profile is not general and has no aim_id"
            )

    def _get_accelerator_family(self) -> str:
        try:
            assets_index = self.profile_path.parts.index("assets")
            return self.profile_path.parts[assets_index + 1]
        except (ValueError, IndexError):
            logger.warning(
                "Could not infer accelerator family from profile path %s; defaulting to instinct", self.profile_path
            )
            return "instinct"

    def evaluate(self) -> ProfileTypeEvaluationResult:
        profile_type_manual_selection_mapping: Dict[Optional[ProfileType], bool] = {
            ProfileType.UNOPTIMIZED: True,
            ProfileType.GENERAL: False,
            ProfileType.OPTIMIZED: False,
            ProfileType.PREVIEW: False,
        }

        return ProfileTypeEvaluationResult(
            profile_type=self.profile_type,
            manual_selection_only=profile_type_manual_selection_mapping.get(self.profile_type, True),
        )


class ProfileManager(AssetManager):

    def __init__(
        self,
        assets_path: str,
        skip_custom: bool = True,
        skip_base: bool = False,
        skip_model_specific: bool = False,
    ) -> None:
        super().__init__(assets_path, enforce_double_quotes=False)
        self.skip_custom = skip_custom
        self.skip_base = skip_base
        self.skip_model_specific = skip_model_specific

    def get_yamls(self, canonical_name: Optional[CanonicalName] = None) -> List[Path]:
        """Find all profile YAML files in assets/{org}/{model}/profiles folders.

        Args:
            canonical_name: Optional CanonicalName to filter profiles by model (e.g., org/model)

        Returns:
            List of paths to profile YAML files
        """

        descriptors: List[AssetDescriptor] = self.get_descriptors(
            canonical_name=canonical_name,
            skip_custom=self.skip_custom,
            skip_base=self.skip_base,
            skip_model_specific=self.skip_model_specific,
        )

        profile_paths = []

        for descriptor in descriptors:
            resolved_path = descriptor.directory / "profiles"
            if resolved_path.exists():
                profile_paths.extend(get_yamls(resolved_path))

        return profile_paths


@click.group(invoke_without_command=True)
@click.pass_context
def cli(ctx):
    pass


@cli.command("sync-profiles-with-file")
@assets_path_option
@click.option("--file_path", type=str, required=True)
@click.option("--sheet_name", type=str, required=False, default=None)
@click.option("--canonical_name", type=str, required=False, default=None)
def sync_profiles_with_file(
    assets_path: str,
    file_path: str,
    sheet_name: Optional[str] = None,
    canonical_name: Optional[str] = None,
) -> None:
    """Synchronize profile metadata with type values from an Excel file."""
    value_resolver = ProfileFileValueResolver(Path(file_path), sheet_name=sheet_name)

    canonical_name_value = CanonicalName.from_string(canonical_name)
    profiles = ProfileManager(assets_path=assets_path).get_yamls(canonical_name=canonical_name_value)  # type: ignore[arg-type]
    for profile in profiles:
        evaluator = ProfileTypeEvaluator(profile, value_resolver)
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


class Mismatch(ABC):

    @property
    @abstractmethod
    def has_mismatch(self) -> bool:
        """Whether this represents a mismatch."""


@dataclass
class MetadataMismatch(Mismatch):
    profile_path: Path
    field_name: str
    expected_value: Optional[str]
    actual_value: Optional[str]

    def __str__(self):
        return f"Metadata mismatch in {self.field_name}. Expected: {self.expected_value}, Actual: {self.actual_value} for profile {self.profile_path}"

    @property
    def has_mismatch(self) -> bool:
        return self.expected_value != self.actual_value


@dataclass
class FileNameFormatMismatch(Mismatch):
    profile_path: Path
    error_message: str

    def __str__(self):
        return f"File name format mismatch: {self.error_message} for profile {self.profile_path}"

    @property
    def has_mismatch(self) -> bool:
        return len(self.error_message) > 0


@cli.command("check-metadata")
@assets_path_option
@click.option("--canonical_name", type=str, help="Filter by model canonical name (format: 'org/model')")
def check_profile_metadata(assets_path: str, canonical_name: Optional[str] = None) -> None:
    """Check that profile metadata matches profile filenames."""
    error_count = _check_profile_metadata(assets_path, canonical_name)
    sys.exit(1 if error_count > 0 else 0)


@cli.command("check-all-metadata")
@assets_root_option
@click.option("--canonical_name", type=str, help="Filter by model canonical name (format: 'org/model')")
def check_all_profile_metadata(assets_root: str = "assets", canonical_name: Optional[str] = None) -> None:
    """Check profile metadata across all accelerator asset directories."""
    total_errors = 0
    for assets_path in discover_assets_paths(assets_root):
        print()
        logger.info(f"Checking profiles in {assets_path}...")
        total_errors += _check_profile_metadata(assets_path, canonical_name)
    sys.exit(1 if total_errors > 0 else 0)


def _get_metadata_profile_id_mismatches(
    profile_path: Path, metadata: ProfileMetadata, profile_id: str
) -> List[MetadataMismatch | FileNameFormatMismatch]:
    result: List[MetadataMismatch | FileNameFormatMismatch] = []
    try:
        engine, accelerator, precision, tp, metric = profile_id.split("-")

        precision = precision.replace("mxfp4", "fp4")

        result.extend(
            [
                MetadataMismatch(profile_path, "engine", engine, metadata.engine.value),
                MetadataMismatch(
                    profile_path,
                    "accelerator_model",
                    accelerator.lower(),
                    metadata.accelerator_model.value.lower() if metadata.accelerator_model is not None else None,
                ),
                MetadataMismatch(profile_path, "precision", precision, metadata.precision.value),
                MetadataMismatch(profile_path, "accelerator_count", tp, f"tp{metadata.accelerator_count}"),
                MetadataMismatch(profile_path, "metric", metric, metadata.metric.value),
            ]
        )

        return [r for r in result if r.has_mismatch]
    except ValueError as e:
        result.append(FileNameFormatMismatch(profile_path, str(e)))
        return result


def _check_profile_metadata(assets_path: str, canonical_name: Optional[str] = None) -> int:
    """Core logic: returns the number of mismatches found."""
    profiles = ProfileManager(assets_path=assets_path).get_yamls(canonical_name=canonical_name)  # type: ignore[arg-type]

    mismatches: List[MetadataMismatch | FileNameFormatMismatch] = []
    for profile_path in profiles:
        profile_data = read_yaml(profile_path)
        raw_metadata = profile_data.get("metadata", {})

        try:
            metadata = ProfileMetadata.model_validate(raw_metadata)
        except (ValidationError, ValueError) as e:
            logger.error(f"{profile_path}: Failed to parse metadata: {e}")
            mismatches.append(MetadataMismatch(profile_path, "metadata", profile_path.stem, str(e)))
            continue

        expected_stem = profile_path.stem
        mismatches.extend(_get_metadata_profile_id_mismatches(profile_path, metadata, expected_stem))

        if metadata.type == "general" and "general" not in profile_path.parts:
            logger.error(f"{profile_path}: General profile not in a 'general' directory")
            mismatches.append(MetadataMismatch(profile_path, "type", "general directory", str(profile_path.parent)))

    if len(mismatches) > 0:
        logger.error(f"❌ Found {len(mismatches)} errors out of {len(profiles)} profiles")
        failed_profiles = sorted(mismatches, key=lambda m: m.profile_path)
        for p in failed_profiles:
            logger.error(f"  - {p}")
    else:
        logger.info(f"✅ All {len(profiles)} profiles passed metadata check.")

    return len(mismatches)


@cli.command("clone-profiles-gpu-name")
@assets_path_option
@click.option("--old-gpu-name", type=str, required=True, help="The old GPU name to replace (e.g., MI300X).")
@click.option(
    "--new-gpu-name", type=str, required=True, help="The new GPU name to use for the cloned profile (e.g., MI325X)."
)
def clone_profiles(assets_path: str, old_gpu_name: str, new_gpu_name: str):
    """
    Recursively scans a directory for YAML files with old_gpu_name in their name,
    updates the GPU profile from old_gpu_name to new_gpu_name, and saves a new file.
    """

    try:
        GPUModel.from_string(old_gpu_name)
        GPUModel.from_string(new_gpu_name)
    except ValueError as e:
        logging.error(f"Invalid GPU name: {e}")
        return

    yamls = ProfileManager(assets_path=assets_path).get_yamls()

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
@assets_root_option
def sort_all_profile_keys(files: List[str], assets_root: str = "assets"):
    if not files:
        all_profiles = []
        for assets_path in discover_assets_paths(assets_root):
            all_profiles.extend(ProfileManager(assets_path=assets_path).get_yamls())
        _sort_profile_keys(all_profiles)
    else:
        _sort_profile_keys([Path(f) for f in files])


def _sort_profile_keys(yaml_paths: List[Path]):

    if not yaml_paths:
        logger.error("No files provided. Pass profile YAML files as arguments.")
        sys.exit(1)

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
            logger.info(f"✅ {modified_count}/{len(yaml_paths)} files were sorted")
        if error_count > 0:
            logger.error(f"❌ Encountered {error_count} errors")
        sys.exit(1)

    sys.exit(0)


@cli.command(name="set-primary-flags")
@assets_path_option
@click.option("--canonical_name", type=str, default=None, help="Filter by model canonical name (format: 'org/model')")
def set_primary_flags_command(assets_path: str, canonical_name: Optional[str] = None) -> None:
    """
    Add recommended deployment configurations based on available profiles.
    Automatically detects latency and throughput profiles for each GPU model.

    Args:
        canonical_name: If provided, only update files matching this canonical name
    """
    modified_count = _set_primary_flags(assets_path, canonical_name)
    sys.exit(1 if modified_count > 0 else 0)


@cli.command(name="set-all-primary-flags")
@assets_root_option
@click.option("--canonical_name", type=str, default=None, help="Filter by model canonical name (format: 'org/model')")
def set_all_primary_flags_command(assets_root: str = "assets", canonical_name: Optional[str] = None) -> None:
    """Add recommended deployments across all accelerator asset directories."""
    total_modified = 0
    for assets_path in discover_assets_paths(assets_root):
        total_modified += _set_primary_flags(assets_path, canonical_name)
    sys.exit(1 if total_modified > 0 else 0)


def _set_primary_flags(assets_path: str, canonical_name: Optional[str] = None) -> int:
    """Core logic: returns the number of modified files."""
    canonical_name_value = CanonicalName.from_string(canonical_name)
    profile_yamls = ProfileManager(assets_path=assets_path, skip_base=True).get_yamls(canonical_name_value)

    groups = {}
    if canonical_name_value is not None:
        groups[canonical_name_value.canonical] = profile_yamls
    else:
        for profile_yaml in profile_yamls:
            canonical_name_value = CanonicalName.from_profile_yaml_path(profile_yaml)
            if canonical_name_value is not None:
                canonical = canonical_name_value.canonical
                if canonical in groups:
                    groups[canonical].append(profile_yaml)
                else:
                    groups[canonical] = [profile_yaml]

    modified_count = 0
    for group, profile_files in groups.items():
        modified = _set_primary_flags_for_model(group, profile_files)
        modified_count += modified

    if modified_count > 0:
        logger.info(f"✅ Recommended deployments added to {modified_count}/{len(profile_yamls)} metadata files")

    return modified_count


def _set_primary_flags_for_model(canonical_name: str, profile_files: List[Path]) -> int:
    """
    Add recommended deployments for a single model based on its profiles for each GPU and metric.

    Selection criteria:
    1. Prioritizes profiles with manual_selection_only=false
    2. Lowest precision int4 > int8 > fp4 > fp8 > fp16 > bf16 > fp32 (lower is better)
    3. Lowest GPU count (minimal TP):

    Args:
        profile_yaml: Path to the metadata.yaml file
    """
    # Define precision priority matching ProfileSelector logic (lower number = higher priority/lower precision)
    # This matches the priority order in profile_selector.py
    # TODO: Extract this into a common utility to avoid duplication
    precision_priority = {
        "int4": 1,
        "int8": 2,
        "fp4": 3,
        "fp8": 4,
        "fp16": 5,
        "bf16": 6,
        "fp32": 7,
    }

    # Unknown precision constant (matches ProfileSelector)
    UNKNOWN_PRECISION_PRIORITY = 999

    # Parse profiles and organize by GPU model and metric
    # Structure: {gpu_model: {metric: [list of profile info]}}
    profiles_by_gpu: Dict[str, Dict[str, List[Dict[str, Any]]]] = {}

    for profile_file in profile_files:
        profile = read_yaml(profile_file)

        profile_metadata = profile.get("metadata", {})

        manual = profile_metadata.get("manual_selection_only", False)
        profile_id = profile_file.stem
        accelerator_model = profile_metadata.get("gpu")
        if accelerator_model is None:
            accelerator_model = profile_metadata.get("accelerator_model")

        accelerator_count = profile_metadata.get("gpu_count")
        if accelerator_count is None:
            accelerator_count = profile_metadata.get("accelerator_count")

        metric = profile_metadata.get("metric")
        precision = profile_metadata.get("precision")

        if not all([accelerator_model, accelerator_count, metric, precision]):
            logger.debug(f"Skipping {profile_file.name} - missing required metadata")
            continue

        # Initialize nested structure
        if accelerator_model not in profiles_by_gpu:
            profiles_by_gpu[accelerator_model] = {"latency": [], "throughput": []}

        # Store the profile info with precision priority matching ProfileSelector
        profile_info = {
            "accelerator_model": accelerator_model,
            "accelerator_count": accelerator_count,
            "metric": metric,
            "precision": precision,
            "precision_priority": precision_priority.get(precision.lower(), UNKNOWN_PRECISION_PRIORITY),
            "manual_selection_only": manual,
            "profileId": profile_id,
            "profile_file": profile_file,
        }

        profiles_by_gpu[accelerator_model][metric].append(profile_info)

    # Select best profiles for each GPU model and metric
    # Strategy: For each GPU model and metric, select minimal precision and minimal TP
    recommended_deployments = []

    for accelerator_model in sorted(profiles_by_gpu.keys()):
        for metric in ["latency", "throughput"]:
            profiles = profiles_by_gpu[accelerator_model][metric]
            if profiles:
                # Sort by: 1) manual_selection_only (False preferred), 2) precision priority (lower is better), 3) GPU count (lower is better)
                # This heavily prioritizes manual_selection_only=False profiles
                best_profile = min(
                    profiles,
                    key=lambda p: (p["manual_selection_only"], p["precision_priority"], p["accelerator_count"]),
                )

                deployment = {
                    "accelerator_model": best_profile["accelerator_model"],
                    "accelerator_count": best_profile["accelerator_count"],
                    "precision": best_profile["precision"],
                    "metric": metric,
                    "description": f"Optimized for {metric} on {best_profile['accelerator_model']} using {best_profile['precision']} precision",
                    "profile_file": best_profile["profile_file"],
                }

                logger.debug(
                    f"  Selected {metric}: {best_profile['accelerator_model']} tp{best_profile['accelerator_count']} {best_profile['precision']}"
                )

                if best_profile["manual_selection_only"]:
                    deployment["profileId"] = best_profile["profileId"]
                    del deployment["precision"]

                recommended_deployments.append(deployment)

    if not recommended_deployments:
        logger.warning(f"No valid deployment configurations found for {canonical_name}")
        return 0

    primary_profiles = set()
    for deployment in recommended_deployments:
        primary_profiles.add(deployment["profile_file"])

    modified_count = 0
    for profile_file in profile_files:
        profile_data = read_yaml(profile_file)
        primary = get_value(profile_data, "metadata.primary")
        if profile_file in primary_profiles:
            if not primary:
                set_value(profile_data, "metadata.primary", True, add_if_missing=True)
                modified_count += save_yaml(profile_data, path=profile_file, enforce_double_quotes=False)
                sort_yaml_file(profile_file, FileType.PROFILE)
        else:
            if primary:
                set_value(profile_data, "metadata.primary", False, add_if_missing=True)
                modified_count += save_yaml(profile_data, path=profile_file, enforce_double_quotes=False)
                sort_yaml_file(profile_file, FileType.PROFILE)

    logger.debug(f"Updated primary flag in {modified_count} of {len(profile_files)} profiles")
    return modified_count


@cli.command(name="rename-key")
@click.argument("source_key", type=str)
@click.argument("target_key", type=str)
@assets_path_option
@click.option("--canonical_name", type=str, default=None, help="Filter by model canonical name (format: 'org/model')")
def rename_key_command(
    source_key: str,
    target_key: str,
    assets_path: str,
    canonical_name: Optional[str] = None,
) -> None:
    """Rename a key by copying its value to a new key and deleting the original."""
    ProfileManager(assets_path=assets_path).rename_key(source_key, target_key, canonical_name)


@cli.command(name="update-value")
@click.argument("key", type=str)
@click.argument("new_value", type=str, default=None)
@assets_path_option
@click.option("--canonical_name", type=str, default=None, help="Filter by model canonical name (format: 'org/model')")
@click.option("--add_if_missing", is_flag=True, default=False, help="Add the key if it doesn't exist")
def update_value_command(
    key: str,
    assets_path: str,
    new_value: Optional[Any] = None,
    canonical_name: Optional[str] = None,
    add_if_missing: bool = False,
) -> None:
    """
    Update a specific field in YAML file(s)

    Args:
        assets_path: Root directory containing assets
        key: Dot notation path to the key (e.g., "org.opencontainers.image.vendor")
        new_value: New value to set for the key
        add_if_missing: If True, add the key if it doesn't exist
        canonical_name: If provided, only update files matching this canonical name
    """
    ProfileManager(assets_path=assets_path).update_value(key, new_value, canonical_name, add_if_missing)


@cli.command(name="copy-value")
@click.argument("source_key", type=str)
@click.argument("target_key", type=str)
@assets_path_option
@click.option("--canonical_name", type=str, default=None, help="Filter by model canonical name (format: 'org/model')")
@click.option("--prefix", type=str, default=None, help="Prefix to add to the copied value")
@click.option("--postfix", type=str, default=None, help="Postfix to add to the copied value")
@click.option("--separator", type=str, default="", help="Separator between prefix/postfix and value")
@click.option("--add_if_missing", is_flag=True, default=True, help="Add the target key if it doesn't exist")
def copy_value_command(
    source_key: str,
    target_key: str,
    assets_path: str,
    canonical_name: Optional[str] = None,
    prefix: Optional[str] = None,
    postfix: Optional[str] = None,
    separator: str = "",
    add_if_missing: bool = True,
) -> None:
    """
    Copy a value from one key to another in metadata files. Optionally add prefix/postfix to string values with a separator.
    """
    ProfileManager(assets_path=assets_path).copy_value(
        source_key,
        target_key,
        canonical_name,
        prefix,
        postfix,
        separator,
        add_if_missing,
    )


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
    cli()
