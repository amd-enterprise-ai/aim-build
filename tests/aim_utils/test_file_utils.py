# Copyright © Advanced Micro Devices, Inc., or its affiliates.
#
# SPDX-License-Identifier: MIT

from pathlib import Path

import pytest

from aim_utils.file_utils import (
    KeyValueFileReader,
    TomlFileReader,
    extract_model_info,
    get_leaf_dirs,
    is_leaf_directory,
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


class TestIsLeafDirectory:
    def test_is_leaf_directory_true(self, temp_dir_structure):
        leaf_dir = temp_dir_structure / "org1" / "model1"
        assert is_leaf_directory(leaf_dir) is True

    def test_is_leaf_directory_false(self, temp_dir_structure):
        non_leaf_dir = temp_dir_structure / "org3"
        assert is_leaf_directory(non_leaf_dir) is False

    def test_is_leaf_directory_with_file(self, tmp_path):
        file_path = tmp_path / "file.txt"
        file_path.write_text("content")
        assert is_leaf_directory(file_path) is False

    def test_is_leaf_directory_empty(self, tmp_path):
        empty_dir = tmp_path / "empty"
        empty_dir.mkdir()
        assert is_leaf_directory(empty_dir) is True

    def test_is_leaf_directory_with_files_only(self, tmp_path):
        dir_with_files = tmp_path / "dir_with_files"
        dir_with_files.mkdir()
        (dir_with_files / "file1.txt").write_text("content1")
        (dir_with_files / "file2.txt").write_text("content2")
        assert is_leaf_directory(dir_with_files) is True


class TestGetLeafDirs:
    def test_get_leaf_dirs(self, temp_dir_structure):
        leaf_dirs = get_leaf_dirs(temp_dir_structure)
        leaf_names = {d.name for d in leaf_dirs}
        assert "model1" in leaf_names
        assert "model2" in leaf_names
        assert "subdir" in leaf_names
        assert "base" in leaf_names
        assert "general" in leaf_names

    def test_get_leaf_dirs_empty(self, tmp_path):
        leaf_dirs = get_leaf_dirs(tmp_path)
        assert leaf_dirs == []

    def test_get_leaf_dirs_single_leaf(self, tmp_path):
        single_leaf = tmp_path / "single"
        single_leaf.mkdir()
        leaf_dirs = get_leaf_dirs(tmp_path)
        assert len(leaf_dirs) == 1
        assert leaf_dirs[0].name == "single"


class TestExtractModelInfo:
    def test_extract_model_info_valid_path(self):
        path = Path("profiles") / "meta-llama" / "Llama-2-7b"
        org, model, is_general = extract_model_info(path)
        assert org == "meta-llama"
        assert model == "Llama-2-7b"
        assert is_general is False

    def test_extract_model_info_base_path(self):
        path = Path("profiles") / "base"
        org, model, is_general = extract_model_info(path)
        assert org is None
        assert model is None
        assert is_general is True

    def test_extract_model_info_general_path(self):
        path = Path("profiles") / "general"
        org, model, is_general = extract_model_info(path)
        assert org is None
        assert model is None
        assert is_general is True

    def test_extract_model_info_empty_path(self):
        path = Path("")
        org, model, is_general = extract_model_info(path)
        assert org is None
        assert model is None
        assert is_general is False

    def test_extract_model_info_single_part(self):
        path = Path("single")
        org, model, is_general = extract_model_info(path)
        assert org is None
        assert model is None
        assert is_general is False

    def test_extract_model_info_too_many_parts(self):
        path = Path("a") / "b" / "c" / "d"
        org, model, is_general = extract_model_info(path)
        assert org is None
        assert model is None
        assert is_general is False


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
