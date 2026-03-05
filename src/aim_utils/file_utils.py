# Copyright © Advanced Micro Devices, Inc., or its affiliates.
#
# SPDX-License-Identifier: MIT

from pathlib import Path
from typing import List, Optional, Tuple

from .dict_utils import get_value

try:
    import tomllib
except ModuleNotFoundError:
    import tomli as tomllib  # type: ignore[no-redef]


def is_leaf_directory(path: Path) -> bool:
    """Check if the directory is a leaf directory (contains no subdirectories)"""
    return path.is_dir() and not any(item.is_dir() for item in path.iterdir())


def get_leaf_dirs(path: Path) -> List[Path]:
    result: List[Path] = []
    for subpath in path.glob("**/*"):
        if is_leaf_directory(subpath):
            result.append(subpath)
    return result


def extract_model_info(profile_path: Path) -> Tuple[Optional[str], Optional[str], bool]:
    """Extract organization and model name from the path"""
    parts = profile_path.parts

    if not parts:
        return None, None, False

    if len(parts) not in (2, 3):
        return None, None, False

    if len(parts) == 3:
        _, org, model_name = parts
        return org, model_name, False

    is_general = "general" in parts or "base" in parts
    return None, None, is_general


class KeyValueFileReader:

    def __init__(self, file_path: Path, separator: str = "\n") -> None:
        self.file_path = file_path
        self.separator = separator

        if not file_path.exists():
            raise FileNotFoundError(file_path)

        with open(file_path, "r") as file:
            self.content = file.read()

    def read_value(self, key: str) -> str:
        """Extract the value for a given key from a key=value formatted string."""
        lines = self.content.split(self.separator)
        for line in lines:
            # First, quickly filter lines that do not contain the key at all
            if key not in line:
                continue
            # Safely split around the first '=' and ensure the key matches exactly (ignoring surrounding whitespace)
            before, sep, after = line.partition("=")
            if sep and before.strip() == key:
                return after.strip()
        return ""


class TomlFileReader(KeyValueFileReader):

    def read_value(self, key: str) -> str:
        """Extract the value for a given key from a TOML formatted string."""
        toml_content = tomllib.loads(self.content)
        return get_value(toml_content, key) or ""
