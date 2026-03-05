# Copyright © Advanced Micro Devices, Inc., or its affiliates.
#
# SPDX-License-Identifier: MIT
import copy
import json
import logging
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

import click
import jsonschema

from aim_common.object_model import CanonicalName

from .asset_utils import Initializer
from .dict_utils import delete_key, get_value, set_value
from .file_utils import extract_model_info, get_leaf_dirs
from .yaml_utils import get_yamls, read_yaml, resolve_paths, save_yaml

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


class MetadataInitializer(Initializer):

    def __init__(
        self,
        assets_path: str = "assets",
        reference_path: Optional[str] = None,
        file_name: Optional[str] = None,
        recreate: bool = False,
    ) -> None:
        if file_name is None:
            file_name = "metadata.yaml"
        super().__init__(
            assets_path=assets_path,
            reference_path=reference_path,
            file_name=file_name,
            recreate=recreate,
        )

    def initialize(self, model_dir: Path) -> None:
        org, model, _ = extract_model_info(model_dir)
        if not org or not model:
            logger.warning(f"Could not extract organization and model from path: {model_dir}")
            return

        output_path = Path(self.assets_path) / org / model / self.file_name  # type: ignore

        # Skip if file already exists
        if output_path.exists():
            logger.info(f"Metadata already exists for {output_path}, skipping...")
            return
        else:
            output_path.parent.mkdir(parents=True, exist_ok=True)

        metadata = copy.deepcopy(DEFAULT_METADATA)

        # Update metadata with info from profiles
        variants = get_model_variants(model_dir)
        set_value(metadata, "com.amd.aim.model.variants", variants)
        canonical_name = CanonicalName(org, model)
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

        logger.info(f"Generated metadata for {model_dir}")


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
@click.option(
    "--metadata_path",
    type=click.Path(exists=True, dir_okay=True, file_okay=False),
    default="metadata",
    help="Path to the root metadata directory",
)
@click.option(
    "--profiles_path",
    type=click.Path(exists=True, dir_okay=True, file_okay=False),
    default="profiles",
    help="Path to the root profiles directory",
)
def init_command(metadata_path: str = "metadata", profiles_path: str = "profiles") -> None:
    """Initialize metadata.yaml files for all models based on their profiles."""
    MetadataInitializer(assets_path=metadata_path, reference_path=profiles_path).initialize_all()


@cli.command(name="delete")
@click.option(
    "--metadata_path",
    type=click.Path(exists=True, dir_okay=True, file_okay=False),
    default="metadata",
    help="Path to the root metadata directory",
)
def delete_metadata_command(metadata_path: str = "metadata") -> None:
    """
    Delete all metadata.yaml files from the given directory and its subdirectories.
    :param metadata_path: path to the root metadata directory
    """
    metadata_dirs = get_leaf_dirs(Path(metadata_path))
    logger.debug(f"Found {len(metadata_dirs)} model-specific metadata directories")
    for metadata_dir in metadata_dirs:
        delete_metadata(metadata_dir)


def delete_metadata(metadata_dir: Path) -> None:
    """
    Delete metadata.yaml file from the given directory if it exists, then removes the directory if empty.
    :param metadata_dir: model-specific metadata directory
    :return:
    """
    file_path = metadata_dir / "metadata.yaml"

    # Skip if file doesn't exist
    if not file_path.exists():
        logger.info(f"No metadata file found at {metadata_dir}, skipping...")
        return

    # Delete the metadata file and directory if empty
    file_path.unlink()
    metadata_dir.rmdir()
    logger.info(f"Deleted metadata file from {metadata_dir}")


@cli.command("delete-key")
@click.argument("key", type=str)
@click.option(
    "--metadata_path",
    type=click.Path(exists=True, dir_okay=True, file_okay=False),
    default="metadata",
    help="Path to the root metadata directory",
)
@click.option("--canonical_name", type=str, default=None, help="Filter by model canonical name (format: 'org/model')")
def delete_key_command(key: str, metadata_path: str = "metadata", canonical_name: Optional[str] = None) -> None:
    """Remove a specific key from metadata files."""
    delete_key_from_files(key, metadata_path, canonical_name)


def delete_key_from_files(key: str, metadata_path: str = "metadata", canonical_name: Optional[str] = None) -> None:
    """
    Remove a specific key from all metadata.yaml files
    :param metadata_path: path to the root metadata directory
    :param key: key to remove in dot notation (e.g., "org.opencontainers.image.vendor")
    :param canonical_name: directory name to filter by (e.g., "TinyLlama/TinyLlama-1.1B-Chat-v1.0")
    """
    metadata_dir_path = Path(metadata_path)
    if not metadata_dir_path.exists():
        logger.error(f"Directory does not exist: {metadata_dir_path}")
        return

    metadata_files = get_yamls(metadata_dir_path, canonical_name)
    logger.debug(f"Found {len(metadata_files)} metadata files to update")

    for file_path in metadata_files:
        metadata = read_yaml(file_path)

        logger.info(f"Deleting key from metadata file: '{file_path}'")
        try:
            metadata = delete_key(metadata, key)
            save_yaml(metadata, path=file_path, enforce_double_quotes=True)
        except Exception as e:
            logger.error(f"Error removing key from '{file_path}': {str(e)}")
            raise e


@cli.command(name="update-value")
@click.argument("key", type=str)
@click.argument("new_value", type=str, default=None)
@click.option(
    "--metadata_path",
    type=click.Path(exists=True, dir_okay=True, file_okay=False),
    default="metadata",
    help="Path to the root metadata directory",
)
@click.option("--canonical_name", type=str, default=None, help="Filter by model canonical name (format: 'org/model')")
@click.option("--add_if_missing", type=bool, default=False, help="Add the key if it doesn't exist")
def update_value_command(
    key: str,
    new_value: Optional[Any] = None,
    metadata_path: str = "metadata",
    canonical_name: Optional[str] = None,
    add_if_missing: bool = False,
) -> None:
    """
    Update a specific field in all metadata.yaml files

    Args:
        metadata_path: Root directory containing metadata.yaml files
        key: Dot notation path to the key (e.g., "org.opencontainers.image.vendor")
        new_value: New value to set for the key
        add_if_missing: If True, add the key if it doesn't exist
        canonical_name: If provided, only update files matching this canonical name
    """

    logger.warning(
        "Currently, this command supports only string values. Non-string values will be set as strings. Use with caution."
    )

    # Mapping of keys to functions that calculate their values dynamically from metadata. Not all keys need this.
    update_mapping = {
        "com.amd.aim.model.publisher": lambda data: CanonicalName.from_string(
            get_value(data, "com.amd.aim.model.canonicalName")
        ).publisher,
        "com.amd.aim.title": lambda data: CanonicalName.from_string(
            get_value(data, "com.amd.aim.model.canonicalName")
        ).title,
    }

    # Find all metadata.yaml files
    metadata_files = get_yamls(Path(metadata_path), canonical_name)
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
                        save_yaml(metadata, path=file_path, enforce_double_quotes=True)
                    else:
                        logger.warning(f"Calculated value for key '{key}' is None, skipping update for {file_path}")
                else:
                    logger.warning(
                        f"No update mapping found for key '{key}' and no explicit value provided. Skipping {file_path}"
                    )
            else:
                metadata = set_value(metadata, key, new_value, add_if_missing=add_if_missing)
                save_yaml(metadata, path=file_path, enforce_double_quotes=True)
        except Exception as e:
            logger.error(f"Error updating '{file_path}': {str(e)}")
            raise e


@cli.command(name="copy-value")
@click.argument("source_key", type=str)
@click.argument("target_key", type=str)
@click.option(
    "--metadata_path",
    type=click.Path(exists=True, dir_okay=True, file_okay=False),
    default="metadata",
    help="Path to the root metadata directory",
)
@click.option("--canonical_name", type=str, default=None, help="Filter by model canonical name (format: 'org/model')")
@click.option("--prefix", type=str, default=None, help="Prefix to add to the copied value")
@click.option("--postfix", type=str, default=None, help="Postfix to add to the copied value")
@click.option("--separator", type=str, default="", help="Separator between prefix/postfix and value")
@click.option("--add_if_missing", type=bool, default=False, help="Add the target key if it doesn't exist")
def copy_value_command(
    source_key: str,
    target_key: str,
    metadata_path: str = "metadata",
    canonical_name: Optional[str] = None,
    prefix: Optional[str] = None,
    postfix: Optional[str] = None,
    separator: str = "",
    add_if_missing: bool = False,
) -> None:
    """
    Copy a value from one key to another in metadata files. Optionally add prefix/postfix to string values with a separator.
    """
    copy_value(
        source_key,
        target_key,
        metadata_path,
        canonical_name,
        prefix,
        postfix,
        separator,
        add_if_missing,
    )


def copy_value(
    source_key,
    target_key,
    metadata_path: str = "metadata",
    canonical_name: Optional[str] = None,
    prefix: Optional[str] = None,
    postfix: Optional[str] = None,
    separator: str = "",
    add_if_missing: bool = False,
) -> None:
    metadata_files = get_yamls(Path(metadata_path), canonical_name)
    logger.debug(f"Found {len(metadata_files)} metadata files to update")

    for file_path in metadata_files:
        metadata = read_yaml(file_path)
        try:
            metadata = _copy_value(
                metadata,
                source_key,
                target_key,
                prefix,
                postfix,
                separator,
                add_if_missing,
            )
            save_yaml(metadata, path=file_path, enforce_double_quotes=True)
        except Exception as e:
            logger.error(f"Error copying value in '{file_path}'")
            raise e


def _copy_value(
    metadata: Metadata,
    source_key: str,
    target_key: str,
    prefix: Optional[str] = None,
    postfix: Optional[str] = None,
    separator: str = "",
    add_if_missing: bool = False,
) -> Metadata:
    source_value = get_value(metadata, source_key)
    if isinstance(source_value, str):
        if prefix:
            source_value = f"{prefix}{separator}{source_value}"
        if postfix:
            source_value = f"{source_value}{separator}{postfix}"
    return set_value(metadata, target_key, source_value, add_if_missing)


@cli.command(name="rename-key")
@click.argument("source_key", type=str)
@click.argument("target_key", type=str)
@click.option(
    "--metadata_path",
    type=click.Path(exists=True, dir_okay=True, file_okay=False),
    default="metadata",
    help="Path to the root metadata directory",
)
@click.option("--canonical_name", type=str, default=None, help="Filter by model canonical name (format: 'org/model')")
def rename_key_command(
    source_key: str,
    target_key: str,
    metadata_path: str = "metadata",
    canonical_name: Optional[str] = None,
) -> None:
    """Rename a key by copying its value to a new key and deleting the original."""
    copy_value(source_key, target_key, metadata_path, canonical_name, add_if_missing=True)
    delete_key_from_files(source_key, metadata_path, canonical_name)


@cli.command(name="add-recommended-deployments")
@click.option(
    "--metadata_path",
    type=click.Path(exists=True, dir_okay=True, file_okay=False),
    default="metadata",
    help="Path to the root metadata directory",
)
@click.option("--canonical_name", type=str, default=None, help="Filter by model canonical name (format: 'org/model')")
def add_recommended_deployments_command(metadata_path: str = "metadata", canonical_name: Optional[str] = None) -> None:
    """
    Add recommended deployment configurations based on available profiles.
    Automatically detects latency and throughput profiles for each GPU model.

    Args:
        metadata_path: Root directory containing metadata.yaml files
        canonical_name: If provided, only update files matching this canonical name
    """
    metadata_files = resolve_paths(directory=metadata_path, canonical_name=canonical_name)

    modified_count = 0
    for metadata_file in metadata_files:
        modified = _add_recommended_deployments_for_model(metadata_file)
        if modified:
            modified_count += 1

    if modified_count > 0:
        logger.info(f"✅ Recommended deployments added to {modified_count}/{len(metadata_files)} metadata files")
        sys.exit(1)

    sys.exit(0)


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
    profiles_base = Path("profiles")
    profiles_path = profiles_base / canonical_name

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
@click.option(
    "--metadata_path",
    type=click.Path(exists=True, dir_okay=True, file_okay=False),
    default="metadata",
    help="Path to the root metadata directory",
)
@click.option("--canonical_name", type=str, help="Filter by model canonical name (format: 'org/model')")
def validate_metadata_command(metadata_path: str = "metadata", canonical_name: Optional[str] = None) -> None:
    """
    Validate all metadata.yaml files against the JSON schema.

    Args:
        metadata_path: Root directory containing metadata.yaml files
        canonical_name: If provided, only validate files matching this canonical name
    """
    results = validate_metadata(metadata_path, canonical_name)

    if results["valid_count"] == results["total_count"]:
        logger.info(f"✅ All {results['total_count']} metadata files are valid!")
        sys.exit(0)
    else:
        error_message = (
            f"❌ {results['invalid_count']} out of {results['total_count']} metadata files failed validation"
        )
        logger.error(error_message)
        sys.exit(1)


def validate_metadata(metadata_path: str, canonical_name: Optional[str] = None) -> Dict[str, int]:
    """
    Validate metadata.yaml files against JSON schemas.
    Uses base_metadata_schema.json for base/metadata.yaml and metadata_schema.json for all others.

    Args:
        metadata_path: Path to the root metadata directory
        canonical_name: If provided, only validate files matching this canonical name

    Returns:
        Dictionary with validation results: {"total_count": int, "valid_count": int, "invalid_count": int}
    """
    # Get metadata files to validate
    metadata_files = resolve_paths(directory=metadata_path, canonical_name=canonical_name)

    if not metadata_files:
        logger.warning("No metadata files found")
        return {"total_count": 0, "valid_count": 0, "invalid_count": 0}

    # Define schema paths
    base_schema_path = Path("schemas/base_metadata_schema.json")
    regular_schema_path = Path("schemas/metadata_schema.json")

    # Load schemas
    schemas = {}
    for schema_name, schema_path in [("base", base_schema_path), ("regular", regular_schema_path)]:
        if not schema_path.exists():
            logger.error(f"Schema file does not exist: {schema_path}")
            return {"total_count": 0, "valid_count": 0, "invalid_count": 0}

        try:
            with open(schema_path, "r") as f:
                schemas[schema_name] = json.load(f)
            logger.debug(f"Loaded {schema_name} schema from {schema_path}")
        except (json.JSONDecodeError, IOError) as e:
            logger.error(f"Failed to load schema from {schema_path}: {e}")
            return {"total_count": 0, "valid_count": 0, "invalid_count": 0}

    logger.info(f"Validating {len(metadata_files)} metadata files against schemas")

    valid_count = 0
    invalid_count = 0

    for metadata_file in metadata_files:
        try:
            # Determine which schema to use based on file path
            is_base_metadata = "base" in metadata_file.parts
            schema = schemas["base"] if is_base_metadata else schemas["regular"]
            schema_type = "base_metadata_schema.json" if is_base_metadata else "metadata_schema.json"

            # Load and validate the metadata file
            metadata = read_yaml(metadata_file)
            jsonschema.validate(metadata, schema)

            valid_count += 1
            logger.debug(f"✅ {metadata_file}: Valid (using {schema_type})")

        except jsonschema.ValidationError as e:
            invalid_count += 1
            logger.error(f"❌ {metadata_file}: Validation failed - {e.message}")

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
@click.option(
    "--metadata_path",
    type=click.Path(exists=True, dir_okay=True, file_okay=False),
    default="metadata",
    help="Path to the root metadata directory",
)
@click.option("--canonical_name", type=str, help="Filter by model canonical name (format: 'org/model')")
def list_keys_command(metadata_path: str = "metadata", canonical_name: Optional[str] = None) -> None:
    """
    List all keys from metadata.yaml files.

    Args:
        metadata_path: Root directory containing metadata.yaml files
        canonical_name: If provided, only process files matching this canonical name
    """

    metadata_files = get_yamls(Path(metadata_path), canonical_name)
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


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
    cli()
