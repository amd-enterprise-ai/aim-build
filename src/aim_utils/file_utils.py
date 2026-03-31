# Copyright © Advanced Micro Devices, Inc., or its affiliates.
#
# SPDX-License-Identifier: MIT

from pathlib import Path

from .dict_utils import get_value

try:
    import tomllib
except ModuleNotFoundError:
    import tomli as tomllib  # type: ignore[no-redef]


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
