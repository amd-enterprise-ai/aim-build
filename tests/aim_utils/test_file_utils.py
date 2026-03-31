# Copyright © Advanced Micro Devices, Inc., or its affiliates.
#
# SPDX-License-Identifier: MIT

import pytest

from aim_utils.file_utils import (
    KeyValueFileReader,
    TomlFileReader,
)


@pytest.fixture
def temp_dir_structure(tmp_path):
    """Create a directory structure for testing."""
    # Create leaf directories
    leaf1 = tmp_path / "org1" / "model1"
    leaf1.mkdir(parents=True)

    leaf2 = tmp_path / "org2" / "model2"
    leaf2.mkdir(parents=True)

    # Create a non-leaf directory (has subdirectories)
    non_leaf = tmp_path / "org3"
    non_leaf.mkdir(parents=True)
    (non_leaf / "subdir").mkdir()

    # Create base and general directories
    base_dir = tmp_path / "base"
    base_dir.mkdir(parents=True)

    general_dir = tmp_path / "general"
    general_dir.mkdir(parents=True)

    return tmp_path


@pytest.fixture
def key_value_file(tmp_path):
    """Create a key=value formatted file for testing."""
    file_path = tmp_path / "config.txt"
    content = """BASE_REGISTRY_NAMESPACE=test-namespace
BASE_REPOSITORY=test-repo
BASE_TAG=1.0.0
EMPTY_VALUE=
SPACED_KEY = spaced-value
"""
    file_path.write_text(content)
    return file_path


@pytest.fixture
def toml_file(tmp_path):
    """Create a TOML formatted file for testing."""
    file_path = tmp_path / "config.toml"
    content = """[project]
name = "test-project"
version = "2.0.0"

[project.nested]
value = "nested-value"
"""
    file_path.write_text(content)
    return file_path


class TestKeyValueFileReader:
    def test_read_value_existing_key(self, key_value_file):
        reader = KeyValueFileReader(key_value_file)
        assert reader.read_value("BASE_REGISTRY_NAMESPACE") == "test-namespace"
        assert reader.read_value("BASE_REPOSITORY") == "test-repo"
        assert reader.read_value("BASE_TAG") == "1.0.0"

    def test_read_value_empty_value(self, key_value_file):
        reader = KeyValueFileReader(key_value_file)
        assert reader.read_value("EMPTY_VALUE") == ""

    def test_read_value_spaced_key(self, key_value_file):
        reader = KeyValueFileReader(key_value_file)
        assert reader.read_value("SPACED_KEY") == "spaced-value"

    def test_read_value_nonexistent_key(self, key_value_file):
        reader = KeyValueFileReader(key_value_file)
        assert reader.read_value("NONEXISTENT") == ""

    def test_read_value_partial_match(self, key_value_file):
        reader = KeyValueFileReader(key_value_file)
        # Should not match partial keys
        assert reader.read_value("BASE") == ""
        assert reader.read_value("BASE_TAG_EXTRA") == ""

    def test_file_not_found(self, tmp_path):
        nonexistent_file = tmp_path / "nonexistent.txt"
        with pytest.raises(FileNotFoundError):
            KeyValueFileReader(nonexistent_file)

    def test_custom_separator(self, tmp_path):
        file_path = tmp_path / "custom.txt"
        file_path.write_text("KEY1=value1;KEY2=value2")
        reader = KeyValueFileReader(file_path, separator=";")
        assert reader.read_value("KEY1") == "value1"
        assert reader.read_value("KEY2") == "value2"


class TestTomlFileReader:
    def test_read_value_top_level(self, toml_file):
        reader = TomlFileReader(toml_file)
        assert reader.read_value("project.name") == "test-project"
        assert reader.read_value("project.version") == "2.0.0"

    def test_read_value_nested(self, toml_file):
        reader = TomlFileReader(toml_file)
        assert reader.read_value("project.nested.value") == "nested-value"

    def test_read_value_nonexistent_key(self, toml_file):
        reader = TomlFileReader(toml_file)
        assert reader.read_value("nonexistent.key") == ""

    def test_read_value_partial_path(self, toml_file):
        reader = TomlFileReader(toml_file)
        # Returns dict, not string, so get_value returns the dict which is truthy but not a string
        result = reader.read_value("project")
        assert result != ""  # Returns the dict as-is

    def test_file_not_found(self, tmp_path):
        nonexistent_file = tmp_path / "nonexistent.toml"
        with pytest.raises(FileNotFoundError):
            TomlFileReader(nonexistent_file)
