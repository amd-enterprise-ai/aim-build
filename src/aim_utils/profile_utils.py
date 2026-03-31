# Copyright © Advanced Micro Devices, Inc., or its affiliates.
#
# SPDX-License-Identifier: MIT
"""Utilities for managing AIM profiles."""

import logging
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Type, Union

import click

from aim_common.object_model import CanonicalName, GPUModel, ProfileType
from aim_runtime.object_model import ProfileHandling
from aim_utils.asset_utils import AssetDescriptor, AssetManager
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
        if self.is_general:
            return "aim-base"
        if self.aim_id is not None:
            sanitized_name = CanonicalName.from_string(self.aim_id).sanitize  # type: ignore[union-attr]
            return f"aim-{sanitized_name}"
        return "aim"

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
        assets_path: str = "assets/instinct",
        skip_custom: bool = True,
        skip_base: bool = False,
        skip_model_specific: bool = False,
    ) -> None:
        super().__init__(assets_path)
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
@click.option(
    "--assets_path",
    type=click.Path(exists=True, dir_okay=True, file_okay=False),
    default="assets/instinct",
    help="Path to the root assets directory",
)
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

    profiles = ProfileManager(assets_path=assets_path).get_yamls(canonical_name=canonical_name)  # type: ignore[arg-type]
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


@dataclass
class MetadataMismatch:
    profile_path: Path
    field_name: str
    expected_value: Optional[str]
    actual_value: Optional[str]


@cli.command("check-metadata")
@click.option(
    "--assets_path",
    type=click.Path(exists=True, dir_okay=True, file_okay=False),
    default="assets/instinct",
    help="Path to the root assets directory",
)
@click.option("--canonical_name", type=str, help="Filter by model canonical name (format: 'org/model')")
def check_profile_metadata(assets_path: str = "assets/instinct", canonical_name: Optional[str] = None) -> None:
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

    profiles = ProfileManager(assets_path=assets_path).get_yamls(canonical_name=canonical_name)  # type: ignore[arg-type]

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
@click.option(
    "--assets_path",
    type=click.Path(exists=True, dir_okay=True, file_okay=False),
    default="assets/instinct",
    help="Path to the root assets directory",
)
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
def sort_profile_keys(files: List[str]):

    if files:
        yaml_paths = [Path(f) for f in files]
    else:
        yaml_paths = ProfileManager(assets_path="assets/instinct").get_yamls()

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


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
    cli()
