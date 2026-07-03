# Copyright © Advanced Micro Devices, Inc., or its affiliates.
#
# SPDX-License-Identifier: MIT
import copy
import logging
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

import click
from pydantic import ValidationError

from aim_common.metadata_models import BaseMetadataModel, ModelMetadataModel
from aim_common.object_model import CanonicalName, ProfileMetadata

from .asset_utils import (
    AssetDescriptor,
    AssetManager,
    Initializer,
    assets_path_option,
    assets_root_option,
    discover_assets_paths,
)
from .dict_utils import get_value, rename_keys, set_value
from .yaml_utils import get_yamls, read_yaml, save_yaml

logger = logging.getLogger(__name__)

Metadata = Dict[str, Any]

# Default metadata template with empty strings
DEFAULT_METADATA: Metadata = {
    "com": {
        "amd": {
            "aim": {
                "description": {"full": ""},
                "hfToken": {"required": False},
                "model": {
                    "canonicalName": "",
                    "publisher": "",
                    "recommendedDeployments": [],
                    "source": "",
                    "tags": "",
                    "variants": [],
                },
                "release": {"notes": ""},
                "title": "",
            }
        }
    },
    "org": {
        "opencontainers": {
            "image": {
                "authors": "",
                "description": "",
                "documentation": "",
                "licenses": "",
                "source": "",
                "vendor": "AMD",
            }
        },
    },
}

BASE_METADATA: Metadata = {
    "org": {
        "opencontainers": {
            "image": {
                "vendor": "AMD",
                "authors": "",
                "licenses": "MIT",
                "description": "Generic image that can run any model in the AIM catalog. Model identifier should be specified using the environment variable AIM_MODEL_ID.",
                "documentation": "",
                "source": "https://github.com/amd-enterprise-ai/aim-build",
            }
        }
    },
    "com": {
        "amd": {
            "aim": {
                "release": {"notes": ""},
                "description": {
                    "full": "Generic image that can run any model in the AIM catalog. Model identifier should be specified using the environment variable AIM_MODEL_ID."
                },
                "title": "AIM Base",
            }
        }
    },
}


class MetadataInitializer(Initializer):

    def __init__(
        self,
        assets_path: str,
        file_name: Optional[str] = None,
        recreate: bool = False,
    ) -> None:
        if file_name is None:
            file_name = "metadata.yaml"
        super().__init__(
            assets_path=assets_path,
            file_name=file_name,
            recreate=recreate,
        )

    def initialize(self, assets_descriptor: AssetDescriptor) -> None:
        output_path = assets_descriptor.directory / self.file_name  # type: ignore

        if output_path.exists() and output_path.stat().st_size > 0:
            if self.recreate:
                logger.warning(f"Metadata already exists and is not empty for '{output_path}', recreating...")
            else:
                logger.info(f"Metadata already exists and is not empty for '{output_path}', skipping...")
                return
        else:
            output_path.parent.mkdir(parents=True, exist_ok=True)

        if assets_descriptor.is_base:
            metadata = copy.deepcopy(BASE_METADATA)
            save_yaml(metadata, path=output_path, enforce_double_quotes=True)
            return

        metadata = copy.deepcopy(DEFAULT_METADATA)

        # Update metadata with info from profiles
        variants = get_model_variants(assets_descriptor.directory)
        set_value(metadata, "com.amd.aim.model.variants", variants)
        canonical_name = CanonicalName(assets_descriptor.org, assets_descriptor.model_name)  # type: ignore[arg-type]
        set_value(metadata, "com.amd.aim.model.canonicalName", canonical_name.canonical)
        set_value(metadata, "org.opencontainers.image.vendor", "AMD")
        set_value(metadata, "com.amd.aim.model.publisher", canonical_name.publisher, add_if_missing=True)
        set_value(metadata, "com.amd.aim.title", canonical_name.title, add_if_missing=True)
        set_value(
            metadata,
            "com.amd.aim.model.source",
            f"https://huggingface.co/{canonical_name.canonical}",
            add_if_missing=True,
        )

        save_yaml(metadata, path=output_path, enforce_double_quotes=True)

        _add_recommended_deployments_for_model(output_path)

        logger.info(f"Generated metadata for {assets_descriptor.directory}")


class MetadataManager(AssetManager):

    def __init__(self, assets_path: str):
        super().__init__(assets_path, enforce_double_quotes=True)

    def get_yamls(self, canonical_name: Optional[CanonicalName] = None) -> List[Path]:
        descriptors: List[AssetDescriptor] = self.get_descriptors(canonical_name=canonical_name)

        metadata_paths = []

        for descriptor in descriptors:
            resolved_path = descriptor.directory / "metadata.yaml"
            if resolved_path.exists():
                metadata_paths.append(resolved_path)

        return metadata_paths

    def update_value(
        self,
        key: str,
        new_value: Optional[Any] = None,
        canonical_name: Optional[str] = None,
        add_if_missing: bool = False,
    ) -> None:
        logger.warning(
            "Currently, this command supports only string values. Non-string values will be set as strings. Use with caution."
        )

        # Mapping of keys to functions that calculate their values dynamically from metadata. Not all keys need this.
        update_mapping = {
            "com.amd.aim.model.publisher": lambda data: CanonicalName.from_string(  # type: ignore[union-attr]
                get_value(data, "com.amd.aim.model.canonicalName")
            ).publisher,
            "com.amd.aim.title": lambda data: CanonicalName.from_string(  # type: ignore[union-attr]
                get_value(data, "com.amd.aim.model.canonicalName")
            ).title,
        }

        # Find all metadata.yaml files
        metadata_files = self.get_yamls(CanonicalName.from_string(canonical_name))
        logger.debug(f"Found {len(metadata_files)} metadata files to update")

        # Process each metadata file
        for file_path in metadata_files:
            logger.info(f"File name: '{file_path}'")
            try:
                metadata = read_yaml(file_path)
                if new_value is None:
                    value_function = update_mapping.get(key)
                    if value_function:
                        calculated_value = value_function(metadata)
                        if calculated_value is not None:
                            metadata = set_value(metadata, key, calculated_value, add_if_missing=add_if_missing)
                            save_yaml(metadata, path=file_path, enforce_double_quotes=self.enforce_double_quotes)
                        else:
                            logger.warning(f"Calculated value for key '{key}' is None, skipping update for {file_path}")
                    else:
                        logger.warning(
                            f"No update mapping found for key '{key}' and no explicit value provided. Skipping {file_path}"
                        )
                else:
                    metadata = set_value(metadata, key, new_value, add_if_missing=add_if_missing)
                    save_yaml(metadata, path=file_path, enforce_double_quotes=self.enforce_double_quotes)
            except Exception as e:
                logger.error(f"Error updating '{file_path}': {str(e)}")
                raise e


def get_model_variants(model_dir: Path) -> List[str]:
    files = get_yamls(model_dir, subfolder=None)
    variants = set()
    for file in files:
        profile = read_yaml(file)

        model_id = profile.get("model_id")
        if model_id:
            variants.add(model_id)

    return sorted(list(variants))


@click.group(invoke_without_command=True)
@click.pass_context
def cli(ctx):
    pass


@cli.command(name="init")
@assets_path_option
def init_command(assets_path: str) -> None:
    """Initialize metadata.yaml files for all models based on their profiles."""
    MetadataInitializer(assets_path=assets_path).initialize_all()


@cli.command(name="delete")
@assets_path_option
def delete_metadata_command(assets_path: str) -> None:
    """
    Delete all metadata.yaml files from the assets directory.
    """
    MetadataManager(assets_path=assets_path).delete_assets()


@cli.command("delete-key")
@click.argument("key", type=str)
@assets_path_option
@click.option("--canonical_name", type=str, default=None, help="Filter by model canonical name (format: 'org/model')")
def delete_key_command(key: str, assets_path: str, canonical_name: Optional[str] = None) -> None:
    """Remove a specific key from metadata files."""
    MetadataManager(assets_path=assets_path).delete_key(key, canonical_name)


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
    Update a specific field in all metadata.yaml files

    Args:
        assets_path: Root directory containing assets
        key: Dot notation path to the key (e.g., "org.opencontainers.image.vendor")
        new_value: New value to set for the key
        add_if_missing: If True, add the key if it doesn't exist
        canonical_name: If provided, only update files matching this canonical name
    """
    MetadataManager(assets_path=assets_path).update_value(key, new_value, canonical_name, add_if_missing)


@cli.command(name="copy-value")
@click.argument("source_key", type=str)
@click.argument("target_key", type=str)
@assets_path_option
@click.option("--canonical_name", type=str, default=None, help="Filter by model canonical name (format: 'org/model')")
@click.option("--prefix", type=str, default=None, help="Prefix to add to the copied value")
@click.option("--postfix", type=str, default=None, help="Postfix to add to the copied value")
@click.option("--separator", type=str, default="", help="Separator between prefix/postfix and value")
@click.option("--add_if_missing", is_flag=True, default=False, help="Add the target key if it doesn't exist")
def copy_value_command(
    source_key: str,
    target_key: str,
    assets_path: str,
    canonical_name: Optional[str] = None,
    prefix: Optional[str] = None,
    postfix: Optional[str] = None,
    separator: str = "",
    add_if_missing: bool = False,
) -> None:
    """
    Copy a value from one key to another in metadata files. Optionally add prefix/postfix to string values with a separator.
    """
    MetadataManager(assets_path=assets_path).copy_value(
        source_key,
        target_key,
        canonical_name,
        prefix,
        postfix,
        separator,
        add_if_missing,
    )


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
    MetadataManager(assets_path=assets_path).rename_key(source_key, target_key, canonical_name)


@cli.command(name="add-recommended-deployments")
@assets_path_option
@click.option("--canonical_name", type=str, default=None, help="Filter by model canonical name (format: 'org/model')")
def add_recommended_deployments_command(assets_path: str, canonical_name: Optional[str] = None) -> None:
    """
    Add recommended deployment configurations based on available profiles.
    Automatically detects latency and throughput profiles for each GPU model.

    Args:
        metadata_path: Root directory containing metadata.yaml files
        canonical_name: If provided, only update files matching this canonical name
    """
    modified_count = _add_recommended_deployments(assets_path, canonical_name)
    sys.exit(1 if modified_count > 0 else 0)


@cli.command(name="add-all-recommended-deployments")
@assets_root_option
@click.option("--canonical_name", type=str, default=None, help="Filter by model canonical name (format: 'org/model')")
def add_all_recommended_deployments_command(assets_root: str = "assets", canonical_name: Optional[str] = None) -> None:
    """Add recommended deployments across all accelerator asset directories."""
    total_modified = 0
    for assets_path in discover_assets_paths(assets_root):
        total_modified += _add_recommended_deployments(assets_path, canonical_name)
    sys.exit(1 if total_modified > 0 else 0)


def _add_recommended_deployments(assets_path: str, canonical_name: Optional[str] = None) -> int:
    """Core logic: returns the number of modified files."""
    metadata_files = MetadataManager(assets_path=assets_path).get_yamls(CanonicalName.from_string(canonical_name))

    modified_count = 0
    for metadata_file in metadata_files:
        modified = _add_recommended_deployments_for_model(metadata_file)
        if modified:
            modified_count += 1

    if modified_count > 0:
        logger.info(f"✅ Recommended deployments added to {modified_count}/{len(metadata_files)} metadata files")

    return modified_count


def _add_recommended_deployments_for_model(metadata_file: Path) -> bool:
    """
    Add recommended deployments for a single model based on its profiles for each GPU and metric.

    Selection criteria:
    1. Prioritizes profiles with manual_selection_only=false
    2. Lowest precision int4 > int8 > fp4 > fp8 > fp16 > bf16 > fp32 (lower is better)
    3. Lowest GPU count (minimal TP):

    Args:
        metadata_file: Path to the metadata.yaml file
    """
    metadata = read_yaml(metadata_file)

    # Get canonical name from metadata
    canonical_name = get_value(metadata, "com.amd.aim.model.canonicalName")
    if not canonical_name:
        logger.warning(f"No canonical name found in {metadata_file}, skipping...")
        return False

    logger.debug(f"Processing {canonical_name}")

    # Find corresponding profiles directory
    profiles_path = metadata_file.parent / "profiles"

    if not profiles_path.exists():
        logger.warning(f"No profiles directory found at {profiles_path}, skipping...")
        return False

    # Get all profile files
    profile_files = list(profiles_path.glob("*.yaml"))
    if not profile_files:
        logger.warning(f"No profile files found in {profiles_path}, skipping...")
        return False

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
        gpu_model = profile_metadata.get("gpu")
        gpu_count = profile_metadata.get("gpu_count")
        metric = profile_metadata.get("metric")
        precision = profile_metadata.get("precision")

        if not all([gpu_model, gpu_count, metric, precision]):
            logger.debug(f"Skipping {profile_file.name} - missing required metadata")
            continue

        # Initialize nested structure
        if gpu_model not in profiles_by_gpu:
            profiles_by_gpu[gpu_model] = {"latency": [], "throughput": []}

        # Store the profile info with precision priority matching ProfileSelector
        profile_info = {
            "gpuModel": gpu_model,
            "gpuCount": gpu_count,
            "metric": metric,
            "precision": precision,
            "precision_priority": precision_priority.get(precision.lower(), UNKNOWN_PRECISION_PRIORITY),
            "manual_selection_only": manual,
            "profileId": profile_id,
        }

        profiles_by_gpu[gpu_model][metric].append(profile_info)

    # Select best profiles for each GPU model and metric
    # Strategy: For each GPU model and metric, select minimal precision and minimal TP
    recommended_deployments = []

    for gpu_model in sorted(profiles_by_gpu.keys()):
        for metric in ["latency", "throughput"]:
            profiles = profiles_by_gpu[gpu_model][metric]
            if profiles:
                # Sort by: 1) manual_selection_only (False preferred), 2) precision priority (lower is better), 3) GPU count (lower is better)
                # This heavily prioritizes manual_selection_only=False profiles
                best_profile = min(
                    profiles, key=lambda p: (p["manual_selection_only"], p["precision_priority"], p["gpuCount"])
                )

                deployment = {
                    "gpuModel": best_profile["gpuModel"],
                    "gpuCount": best_profile["gpuCount"],
                    "precision": best_profile["precision"],
                    "metric": metric,
                    "description": f"Optimized for {metric} on {best_profile['gpuModel']} using {best_profile['precision']} precision",
                }

                logger.debug(
                    f"  Selected {metric}: {best_profile['gpuModel']} tp{best_profile['gpuCount']} {best_profile['precision']}"
                )

                if best_profile["manual_selection_only"]:
                    deployment["profileId"] = best_profile["profileId"]
                    del deployment["precision"]

                recommended_deployments.append(deployment)

    if not recommended_deployments:
        logger.warning(f"No valid deployment configurations found for {canonical_name}")
        return False

    # Update metadata
    metadata = set_value(
        metadata,
        "com.amd.aim.model.recommendedDeployments",
        recommended_deployments,
        add_if_missing=True,
    )

    modified = save_yaml(metadata, path=metadata_file, enforce_double_quotes=True)
    if modified:
        logger.debug(f"Added {len(recommended_deployments)} recommended deployments to {metadata_file}")
    return modified


@cli.command(name="validate")
@assets_path_option
@click.option("--canonical_name", type=str, help="Filter by model canonical name (format: 'org/model')")
def validate_metadata_command(assets_path: str, canonical_name: Optional[str] = None) -> None:
    """
    Validate all metadata.yaml files against Pydantic models.

    Args:
        assets_path: Root directory containing assets
        canonical_name: If provided, only validate files matching this canonical name
    """
    results = validate_metadata(assets_path, canonical_name)
    _report_validation_results(results)
    sys.exit(0 if results["invalid_count"] == 0 else 1)


@cli.command(name="validate-all")
@assets_root_option
@click.option("--canonical_name", type=str, help="Filter by model canonical name (format: 'org/model')")
def validate_all_metadata_command(assets_root: str = "assets", canonical_name: Optional[str] = None) -> None:
    """Validate metadata across all accelerator asset directories."""
    totals = {"total_count": 0, "valid_count": 0, "invalid_count": 0}
    for assets_path in discover_assets_paths(assets_root):
        results = validate_metadata(assets_path, canonical_name)
        for key in totals:
            totals[key] += results[key]
    _report_validation_results(totals)
    sys.exit(0 if totals["invalid_count"] == 0 else 1)


def _report_validation_results(results: Dict[str, int]) -> None:
    if results["invalid_count"] == 0:
        logger.info(f"✅ All {results['total_count']} metadata files are valid!")
    else:
        logger.error(f"❌ {results['invalid_count']} out of {results['total_count']} metadata files failed validation")


def validate_metadata(assets_path: str, canonical_name: Optional[str] = None) -> Dict[str, int]:
    """
    Validate metadata.yaml files against Pydantic models.
    Uses BaseMetadataModel for base/metadata.yaml and ModelMetadataModel for all others.

    Args:
        assets_path: Path to the root assets directory
        canonical_name: If provided, only validate files matching this canonical name

    Returns:
        Dictionary with validation results: {"total_count": int, "valid_count": int, "invalid_count": int}
    """
    # Get metadata files to validate
    metadata_files = MetadataManager(assets_path=assets_path).get_yamls(CanonicalName.from_string(canonical_name))

    if not metadata_files:
        logger.warning("No metadata files found")
        return {"total_count": 0, "valid_count": 0, "invalid_count": 0}

    logger.info(f"Validating {len(metadata_files)} metadata files")

    valid_count = 0
    invalid_count = 0

    for metadata_file in metadata_files:
        try:
            # Determine which model to use based on file path
            is_base_metadata = "base" in metadata_file.parts
            model_class = BaseMetadataModel if is_base_metadata else ModelMetadataModel
            schema_label = "BaseMetadataModel" if is_base_metadata else "ModelMetadataModel"

            # Load and validate the metadata file
            metadata = read_yaml(metadata_file)
            model_class.model_validate(metadata)

            valid_count += 1
            logger.debug(f"✅ {metadata_file}: Valid (using {schema_label})")

        except ValidationError as e:
            invalid_count += 1
            logger.error(f"❌ {metadata_file}: Validation failed - {e}")

        except Exception as e:
            invalid_count += 1
            logger.error(f"❌ {metadata_file}: Failed to process - {e}")

    return {"total_count": len(metadata_files), "valid_count": valid_count, "invalid_count": invalid_count}


def extract_all_keys(data: Dict[str, Any], prefix: str = "") -> List[str]:
    """
    Recursively extract all full keys from a nested dictionary.

    Args:
        data: The data structure to extract keys from
        prefix: Current key prefix (for recursion)

    Returns:
        List of full keys in dot notation
    """
    keys = []

    if isinstance(data, dict):
        for key, value in data.items():
            current_path = f"{prefix}.{key}" if prefix else key

            if isinstance(value, dict):
                # Recurse into nested dictionaries
                keys.extend(extract_all_keys(value, current_path))
            else:
                # This is a leaf node, add the full path
                keys.append(current_path)

    return keys


@cli.command(name="list-keys")
@assets_path_option
@click.option("--canonical_name", type=str, help="Filter by model canonical name (format: 'org/model')")
def list_keys_command(assets_path: str, canonical_name: Optional[str] = None) -> None:
    """
    List all keys from metadata.yaml files.

    Args:
        assets_path: Root directory containing assets
        canonical_name: If provided, only process files matching this canonical name
    """

    metadata_files = MetadataManager(assets_path=assets_path).get_yamls(CanonicalName.from_string(canonical_name))
    logger.debug(f"Found {len(metadata_files)} metadata files to process")

    all_keys = set()

    for metadata_file in metadata_files:
        metadata = read_yaml(metadata_file)
        keys = extract_all_keys(metadata)
        all_keys.update(keys)

    keys = sorted(list(all_keys))
    if keys:
        logger.info("Found keys:")
        for key in keys:
            logger.info(f"{key}")
        logger.info(f"Total: {len(keys)} unique keys")
    else:
        logger.info("No keys found")


@cli.command(name="validate-recommended-deployments-profile-ids")
@assets_path_option
@click.option("--canonical_name", type=str, help="Filter by model canonical name (format: 'org/model')")
def validate_recommended_deployments_profile_ids(assets_path: str, canonical_name: Optional[str] = None) -> None:
    """Validate that profile ids in recommended deployments exist and match."""
    failed = _validate_recommended_deployments_profile_ids(assets_path, canonical_name)
    if failed:
        logger.error("❌ Recommended deployments validation failed")
        sys.exit(1)
    logger.info("✅ Recommended deployments validation passed.")


@cli.command(name="validate-all-recommended-deployments-profile-ids")
@assets_root_option
@click.option("--canonical_name", type=str, help="Filter by model canonical name (format: 'org/model')")
def validate_all_recommended_deployments_profile_ids(
    assets_root: str = "assets", canonical_name: Optional[str] = None
) -> None:
    """Validate recommended deployment profile ids across all accelerator asset directories."""
    any_failed = False
    for assets_path in discover_assets_paths(assets_root):
        if _validate_recommended_deployments_profile_ids(assets_path, canonical_name):
            any_failed = True
    if any_failed:
        logger.error("❌ Recommended deployments validation failed")
        sys.exit(1)
    logger.info("✅ All recommended deployments validation passed.")
    sys.exit(0)


def _validate_recommended_deployments_profile_ids(assets_path: str, canonical_name: Optional[str] = None) -> bool:
    """Core logic: returns True if validation failed."""

    def rd_to_profile_metadata(rd: Dict[str, Any], profile_metadata: ProfileMetadata) -> ProfileMetadata:
        key_mapping = {
            "gpuModel": "gpu",
            "gpuCount": "gpu_count",
        }

        rd = dict(rd)
        rd["engine"] = profile_metadata.engine
        rd["type"] = profile_metadata.type
        rd["manual_selection_only"] = profile_metadata.manual_selection_only
        if "precision" not in rd:
            rd["precision"] = profile_metadata.precision
        # Profile-intrinsic capabilities are not deployment-selection attributes and
        # cannot be expressed in a recommendedDeployment entry, so inherit them from
        # the referenced profile to keep the equality check focused on selection keys
        # (gpu/metric/precision/...). Without this, any recommendedDeployment pointing
        # at a profile that declares e.g. ``features: [adapters]`` (ADR-0004) fails.
        rd["features"] = profile_metadata.features
        rd["capabilities"] = profile_metadata.capabilities

        rd = rename_keys(rd, key_mapping)
        return ProfileMetadata.from_dict(rd)

    metadata_files = MetadataManager(assets_path=assets_path).get_yamls(CanonicalName.from_string(canonical_name))
    summary = {}
    for metadata_file in metadata_files:
        metadata = read_yaml(metadata_file)

        recommended_deployments = get_value(metadata, "com.amd.aim.model.recommendedDeployments", [])
        canonical_name_from_metadata = get_value(metadata, "com.amd.aim.model.canonicalName")

        if not canonical_name_from_metadata and "base" not in metadata_file.parts:
            logger.error(
                f"❌ Recommended deployments validation failed. {metadata_file} is missing canonical name ('com.amd.aim.model.canonicalName')."
            )
            return True

        count_non_existent_profiles = 0
        count_metadata_mismatches = 0
        count_rd_with_profiles = 0

        for rd in recommended_deployments:
            profile_id = rd.get("profileId")
            if profile_id:
                count_rd_with_profiles += 1
                profile_path = metadata_file.parent / "profiles" / f"{profile_id}.yaml"
                if not profile_path.exists():
                    logger.error(f"Recommended deployment references non-existent profile: {profile_path}")
                    count_non_existent_profiles += 1
                    continue

                profile = read_yaml(profile_path)
                profile_metadata = profile.get("metadata", {})
                profile_metadata.pop("accelerator_type", None)
                profile_metadata.pop("primary", None)
                profile_metadata.pop("variant", None)
                from_profile = ProfileMetadata.from_dict(profile_metadata)
                from_metadata = rd_to_profile_metadata(rd, from_profile)

                if from_profile != from_metadata:
                    logger.error(
                        f"Metadata mismatch for recommended deployment in {metadata_file} referencing profile {profile_path}"
                    )
                    count_metadata_mismatches += 1

        summary[metadata_file] = {
            "total_recommended_deployments": len(recommended_deployments),
            "count_non_existent_profiles": count_non_existent_profiles,
            "count_metadata_mismatches": count_metadata_mismatches,
            "count_rd_with_profiles": count_rd_with_profiles,
        }

    failed = False
    for k, v in summary.items():
        count_non_existent_profiles = v["count_non_existent_profiles"]
        count_metadata_mismatches = v["count_metadata_mismatches"]
        if count_non_existent_profiles > 0 or count_metadata_mismatches > 0:
            failed = True
            logger.error(
                f"Validation failed for {k}: {count_non_existent_profiles} non-existent profiles, {count_metadata_mismatches} metadata mismatches"
            )

    if failed:
        return True

    logger.info(f"✅ All {len(metadata_files)} metadata files passed recommended deployment check.")
    return False


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
    cli()
