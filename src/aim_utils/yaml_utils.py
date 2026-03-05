# Copyright © Advanced Micro Devices, Inc., or its affiliates.
#
# SPDX-License-Identifier: MIT
import logging
from dataclasses import dataclass
from enum import StrEnum
from io import StringIO
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml

# Configure logging
logger = logging.getLogger(__name__)


class DoubleQuoted(str):
    pass


def read_header(path: Path) -> Optional[str]:
    if not path or not path.exists():
        return None

    with open(path, "r", encoding="utf-8") as f:
        original_content = f.read()

    # Preserve the copyright header from original file
    lines = original_content.split("\n")
    header_lines = []
    for i, line in enumerate(lines):
        if line.startswith("#") or line.strip() == "":
            header_lines.append(line)
        else:
            break

    if header_lines:
        header = "\n".join(header_lines).rstrip() + "\n\n"
        return header

    return None


def is_modified(file_path: Path, new_content: str) -> bool:
    """
    Check if the content of a file is different from the provided new content.

    Args:
        file_path: Path to the file to check.
        new_content: New content to compare against the file's current content.

    Returns:
        True if the file content is different from the new content, False otherwise.
    """
    if not file_path.exists():
        return True

    with open(file_path, "r", encoding="utf-8") as f:
        original_content = f.read()

    return original_content != new_content


def save_yaml(
    yaml_data: Dict[str, Any],
    path: Path,
    enforce_double_quotes: bool = False,
    enforce_null_as_empty: bool = False,
) -> bool:
    """
    Save YAML data to a file

    Args:
        yaml_data: The data to save.
        path: Path where to save the YAML file.
        enforce_double_quotes: If True, all string values will be saved with double quotes.
        enforce_null_as_empty: If True, None values will be saved as empty strings. It is relevant for "flag" values the
            presence of which indicates True.

    Returns:
        True if the file was modified and rewritten, False otherwise.
    """

    if not path:
        raise ValueError("Path must be provided to save the YAML data.")

    def process_values(data: Any) -> Any:
        """
        This method should be extended if other types need special processing. It processes the data recursively.

        Args:
            data: The data to process. Can be of any type

        Returns:
            Processed data.
        """
        if isinstance(data, dict):
            return {k: process_values(v) for k, v in data.items()}
        elif isinstance(data, list):
            return [process_values(item) for item in data]
        elif isinstance(data, str):
            if enforce_double_quotes:
                return DoubleQuoted(data)
            return data
        else:
            return data

    def dump(data: Any, stream: Any) -> None:
        """
        A function to save data. Added for convenience and to avoid changing global state of yaml module.

        Args:
            data: Data to save.
            stream: Stream to write the data to.
        """

        class CustomDumper(yaml.SafeDumper):
            pass

        def string_double_quoted(dumper: yaml.SafeDumper, data: Any):
            return dumper.represent_scalar("tag:yaml.org,2002:str", data, style='"')

        def none_as_empty(dumper, _):
            return dumper.represent_scalar("tag:yaml.org,2002:null", "")

        if enforce_double_quotes:
            CustomDumper.add_representer(DoubleQuoted, string_double_quoted)

        if enforce_null_as_empty:
            CustomDumper.add_representer(type(None), none_as_empty)

        yaml.dump(
            data,
            stream,
            Dumper=CustomDumper,
            sort_keys=False,
            default_flow_style=False,
            width=float("inf"),
            allow_unicode=True,
        )

    # Process the data
    processed_data = process_values(yaml_data)

    # Ensure directory exists
    path.parent.mkdir(parents=True, exist_ok=True)

    header = read_header(path)

    stream = StringIO()
    if header:
        stream.write(header)
    dump(processed_data, stream)
    new_content = stream.getvalue()
    modified = is_modified(path, new_content)
    if modified:
        with open(path, "w") as f:
            f.write(new_content)
            logger.debug(f"Saved YAML to {path}")
    else:
        logger.debug(f"No modifications. Saving was skipped for {path}")

    return modified


def read_yaml(path: Path) -> Dict[str, Any]:
    """
    Read YAML data from a file

    Args:
        path: Path to the YAML file

    Returns:
        Dictionary containing the parsed YAML data
    """
    if not path or not path.exists():
        error_message = f"YAML file does not exist: {path}"
        logger.error(error_message)
        raise ValueError(error_message)

    with open(path, "r") as f:
        data = yaml.safe_load(f)
    logger.debug(f"Read YAML from {path}")
    return data


def get_yamls(yaml_dir_path: Path, subfolder: Optional[str] = None) -> List[Path]:
    """
    Get all YAML files in a directory (and optional subfolder)

    Args:
        yaml_dir_path: Path to the directory containing YAML files.
        subfolder: Optional subfolder within the directory to search for YAML files.
    Returns:
        List of paths to YAML files found in the specified directory (and subfolder).
    """
    if subfolder:
        yaml_dir_path = yaml_dir_path / subfolder

    if not yaml_dir_path.exists():
        logger.error(f"Directory does not exist: {yaml_dir_path}")
        return []

    yaml_files = list(yaml_dir_path.glob("**/*.yaml"))
    logger.debug(f"Found {len(yaml_files)} YAML files")

    return yaml_files


class FileType(StrEnum):
    METADATA = "metadata"
    PROFILE = "profile"


@dataclass
class FileTypeParams:
    sections: List[str]
    enforce_double_quotes: bool


def sort_yaml_file(file_path: Path, file_type: FileType) -> bool:
    """
    Sort non-top-level keys in a YAML file alphabetically.

    Args:
        file_path: Path to the YAML file
        file_type: A type of file to determine which sections to sort and formatting options

    Returns:
        True if file was modified, False otherwise
    """

    params_mapping: Dict[FileType, FileTypeParams] = {
        FileType.PROFILE: FileTypeParams(["metadata", "engine_args", "env_vars"], False),
    }

    params = params_mapping.get(file_type)
    if not params:
        logger.error(f"Unsupported file type: {file_type}")
        return False

    # Parse YAML
    data = read_yaml(file_path)

    if not data:
        return False

    for section in params.sections:
        if section in data and isinstance(data[section], dict):
            original_keys = list(data[section].keys())
            sorted_keys = sorted(original_keys)

            if original_keys != sorted_keys:
                # Create new ordered dict with sorted keys
                sorted_section = {k: data[section][k] for k in sorted_keys}
                data[section] = sorted_section

    return save_yaml(
        data,
        path=file_path,
        enforce_double_quotes=params.enforce_double_quotes,
    )


def resolve_paths(
    files: Optional[List[str]] = None, directory: Optional[str] = None, canonical_name: Optional[str] = None
) -> List[Path]:
    if files:
        paths = [Path(file) for file in files]
    else:
        if not directory:
            logger.error("Either 'files' or 'directory' params must be provided.")
            return []

        directory_path = Path(directory)
        if not directory_path.exists() or not directory_path.is_dir():
            logger.error(f"Directory does not exist: {directory_path}")
            return []

        paths = get_yamls(directory_path, canonical_name)
        logger.debug(f"Found {len(paths)} files to process in {directory_path}")

    return paths
