# Copyright © Advanced Micro Devices, Inc., or its affiliates.
#
# SPDX-License-Identifier: MIT


import pytest
import yaml

from aim_utils.yaml_utils import get_yamls, read_yaml, save_yaml


@pytest.fixture
def temp_yaml_dir(tmp_path):
    yaml_dir = tmp_path / "yaml_files"
    yaml_dir.mkdir()
    return yaml_dir


@pytest.fixture
def sample_yaml_data():
    return {"key1": "value1", "key2": {"nested_key": "nested_value"}, "key3": None, "key4": ["item1", "item2"]}


class TestSaveYaml:
    def test_save_yaml_to_file(self, temp_yaml_dir, sample_yaml_data):
        yaml_path = temp_yaml_dir / "test.yaml"
        save_yaml(sample_yaml_data, path=yaml_path)

        assert yaml_path.exists()
        with open(yaml_path, "r") as f:
            loaded_data = yaml.safe_load(f)
        assert loaded_data == sample_yaml_data

    def test_save_yaml_with_double_quotes(self, temp_yaml_dir, sample_yaml_data):
        yaml_path = temp_yaml_dir / "test_quoted.yaml"
        save_yaml(sample_yaml_data, path=yaml_path, enforce_double_quotes=True)

        assert yaml_path.exists()
        content = yaml_path.read_text()
        assert '"value1"' in content
        assert '"nested_value"' in content

    def test_save_yaml_with_null_as_empty(self, temp_yaml_dir, sample_yaml_data):
        yaml_path = temp_yaml_dir / "test_null.yaml"
        save_yaml(sample_yaml_data, path=yaml_path, enforce_null_as_empty=True)

        assert yaml_path.exists()
        content = yaml_path.read_text()
        assert "key3:" in content
        assert "key3: null" not in content

    def test_save_yaml_creates_parent_dirs(self, temp_yaml_dir, sample_yaml_data):
        nested_path = temp_yaml_dir / "subdir1" / "subdir2" / "test.yaml"
        save_yaml(sample_yaml_data, path=nested_path)

        assert nested_path.exists()
        assert nested_path.parent.exists()


class TestReadYaml:
    def test_read_yaml(self, temp_yaml_dir, sample_yaml_data):
        yaml_path = temp_yaml_dir / "test.yaml"
        with open(yaml_path, "w") as f:
            yaml.dump(sample_yaml_data, f)

        result = read_yaml(yaml_path)
        assert result == sample_yaml_data

    def test_read_yaml_empty_file(self, temp_yaml_dir):
        yaml_path = temp_yaml_dir / "empty.yaml"
        yaml_path.write_text("")

        result = read_yaml(yaml_path)
        assert result is None


class TestGetYamls:
    def test_get_yamls_in_directory(self, temp_yaml_dir):
        (temp_yaml_dir / "file1.yaml").write_text("key: value1")
        (temp_yaml_dir / "file2.yaml").write_text("key: value2")
        (temp_yaml_dir / "file.txt").write_text("not yaml")

        result = get_yamls(temp_yaml_dir)

        assert len(result) == 2
        assert all(p.suffix == ".yaml" for p in result)

    def test_get_yamls_with_subfolder(self, temp_yaml_dir):
        subfolder = temp_yaml_dir / "sub"
        subfolder.mkdir()
        (subfolder / "file1.yaml").write_text("key: value1")
        (temp_yaml_dir / "file2.yaml").write_text("key: value2")

        result = get_yamls(temp_yaml_dir, subfolder="sub")

        assert len(result) == 1
        assert result[0].parent == subfolder

    def test_get_yamls_recursive(self, temp_yaml_dir):
        nested_dir = temp_yaml_dir / "level1" / "level2"
        nested_dir.mkdir(parents=True)
        (temp_yaml_dir / "file1.yaml").write_text("key: value1")
        (nested_dir / "file2.yaml").write_text("key: value2")

        result = get_yamls(temp_yaml_dir)

        assert len(result) == 2

    def test_get_yamls_nonexistent_directory(self, temp_yaml_dir):
        nonexistent = temp_yaml_dir / "does_not_exist"

        result = get_yamls(nonexistent)

        assert result == []

    def test_get_yamls_empty_directory(self, temp_yaml_dir):
        result = get_yamls(temp_yaml_dir)

        assert result == []
