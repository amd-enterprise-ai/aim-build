# Copyright © Advanced Micro Devices, Inc., or its affiliates.
#
# SPDX-License-Identifier: MIT

import argparse
import logging
from pathlib import Path
from typing import Any, Dict

import yaml


class DoubleQuoted(str):
    """String subclass for double-quoted YAML output."""


def quoted_presenter(dumper, data: Any):
    """YAML representer for double-quoted strings."""
    return dumper.represent_scalar("tag:yaml.org,2002:str", data, style='"')


def save_yaml(path: Path, yaml_data: Dict[str, Any]) -> None:
    """
    Save YAML data to a file with double-quoted strings.

    Args:
        path: Path where to save the YAML file
        yaml_data: The data to save
    """

    # Convert all string values to DoubleQuoted
    def process_values(data: Any) -> Any:
        if isinstance(data, dict):
            return {k: process_values(v) for k, v in data.items()}
        elif isinstance(data, list):
            return [process_values(item) for item in data]
        elif isinstance(data, str):
            return DoubleQuoted(data)
        else:
            return data

    # Process the data
    processed_data = process_values(yaml_data)

    # Ensure directory exists
    path.parent.mkdir(parents=True, exist_ok=True)

    # Add representer for DoubleQuoted
    yaml.add_representer(DoubleQuoted, quoted_presenter)

    # Write to file
    with open(path, "w") as f:
        yaml.dump(processed_data, f, sort_keys=False, default_flow_style=False)

    logging.info(f"Saved YAML to {path}")


def clone_profiles(directory: str, old_gpu_name: str, new_gpu_name: str):
    """
    Recursively scans a directory for YAML files with old_gpu_name in their name,
    updates the GPU profile from old_gpu_name to new_gpu_name, and saves a new file.
    """
    root_path = Path(directory)
    if not root_path.is_dir():
        logging.error(f"Directory '{directory}' not found.")
        return

    logging.info(f"Scanning directory: '{directory}'...")

    for filepath in root_path.rglob("*.yaml"):
        # Process only YAML files containing old_gpu_name
        if old_gpu_name.lower() in filepath.name.lower():
            try:
                with open(filepath, "r") as f:
                    data = yaml.safe_load(f)

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
                    save_yaml(new_filepath, data)

                    logging.info(f"  -> Saved new profile to '{new_filepath}'")
                else:
                    logging.warning(f"Skipping '{filepath.name}': '{old_gpu_name}' not found in metadata.gpu field.")

            except yaml.YAMLError as e:
                logging.error(f"Error parsing YAML file '{filepath.name}': {e}")
            except Exception as e:
                logging.error(f"An error occurred while processing '{filepath.name}': {e}")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    parser = argparse.ArgumentParser(description="Clone GPU profiles by updating the GPU name.")
    parser.add_argument("--directory", help="The directory to scan for profiles.", required=False, default="profiles")
    parser.add_argument("--old-gpu-name", required=True, help="The old GPU name to replace (e.g., MI300X).")
    parser.add_argument(
        "--new-gpu-name", required=True, help="The new GPU name to use for the cloned profile (e.g., MI325X)."
    )

    args = parser.parse_args()

    clone_profiles(args.directory, args.old_gpu_name, args.new_gpu_name)
