# Copyright © Advanced Micro Devices, Inc., or its affiliates.
#
# SPDX-License-Identifier: MIT

import pytest

from aim_utils.dict_utils import delete_key, get_value, rename_key, rename_keys, set_value, sort_dict_keys_in_list


@pytest.fixture
def sample_dict():
    return {
        "level1": {
            "level2": {
                "level3": "deep_value",
                "another_key": 42,
            },
            "simple_key": "simple_value",
        },
        "top_level": "top_value",
        "boolean_key": False,
        "none_key": None,
    }


class TestGetValue:
    def test_get_value_existing_key(self, sample_dict):
        assert get_value(sample_dict, "top_level") == "top_value"
        assert get_value(sample_dict, "level1.simple_key") == "simple_value"
        assert get_value(sample_dict, "level1.level2.level3") == "deep_value"
        assert get_value(sample_dict, "level1.level2.another_key") == 42

    def test_get_value_missing_key(self, sample_dict):
        assert get_value(sample_dict, "nonexistent") is None
        assert get_value(sample_dict, "level1.nonexistent") is None
        assert get_value(sample_dict, "level1.level2.nonexistent") is None

    def test_get_value_partial_path(self, sample_dict):
        result = get_value(sample_dict, "level1.level2")
        assert isinstance(result, dict)
        assert "level3" in result
        assert "another_key" in result

    def test_get_value_boolean_false(self, sample_dict):
        assert get_value(sample_dict, "boolean_key") is False

    def test_get_value_none_value(self, sample_dict):
        assert get_value(sample_dict, "none_key") is None

    def test_get_value_empty_dict(self):
        assert get_value({}, "any.key") is None

    def test_get_value_single_key(self, sample_dict):
        assert get_value(sample_dict, "top_level") == "top_value"


class TestSetValue:
    def test_set_value_existing_key(self, sample_dict):
        updated = set_value(sample_dict, "top_level", "new_value")
        assert get_value(updated, "top_level") == "new_value"

    def test_set_value_nested_existing_key(self, sample_dict):
        updated = set_value(sample_dict, "level1.level2.level3", "updated_deep")
        assert get_value(updated, "level1.level2.level3") == "updated_deep"

    def test_set_value_missing_key_no_add(self, sample_dict):
        updated = set_value(sample_dict, "new_key", "new_value", add_if_missing=False)
        assert get_value(updated, "new_key") is None

    def test_set_value_missing_key_with_add(self, sample_dict):
        updated = set_value(sample_dict, "new_key", "new_value", add_if_missing=True)
        assert get_value(updated, "new_key") == "new_value"

    def test_set_value_missing_nested_key_with_add(self, sample_dict):
        updated = set_value(sample_dict, "level1.new_nested.key", "nested_value", add_if_missing=True)
        assert get_value(updated, "level1.new_nested.key") == "nested_value"

    def test_set_value_missing_nested_key_no_add(self, sample_dict):
        updated = set_value(sample_dict, "level1.nonexistent.key", "value", add_if_missing=False)
        assert get_value(updated, "level1.nonexistent.key") is None

    def test_set_value_returns_dict(self, sample_dict):
        result = set_value(sample_dict, "top_level", "new")
        assert isinstance(result, dict)
        assert result is sample_dict

    def test_set_value_deep_missing_path_with_add(self, sample_dict):
        updated = set_value(sample_dict, "a.b.c.d", "deep", add_if_missing=True)
        assert get_value(updated, "a.b.c.d") == "deep"

    def test_set_value_overwrite_with_different_type(self, sample_dict):
        updated = set_value(sample_dict, "level1.level2.another_key", "string_now")
        assert get_value(updated, "level1.level2.another_key") == "string_now"


class TestDeleteKey:
    def test_delete_key_top_level(self, sample_dict):
        updated = delete_key(sample_dict, "top_level")
        assert "top_level" not in updated
        assert get_value(updated, "top_level") is None

    def test_delete_key_nested(self, sample_dict):
        updated = delete_key(sample_dict, "level1.simple_key")
        assert get_value(updated, "level1.simple_key") is None
        assert get_value(updated, "level1.level2") is not None

    def test_delete_key_deep_nested(self, sample_dict):
        updated = delete_key(sample_dict, "level1.level2.level3")
        assert get_value(updated, "level1.level2.level3") is None
        assert get_value(updated, "level1.level2.another_key") == 42

    def test_delete_key_nonexistent(self, sample_dict):
        original_keys = set(sample_dict.keys())
        updated = delete_key(sample_dict, "nonexistent")
        assert set(updated.keys()) == original_keys

    def test_delete_key_nonexistent_nested(self, sample_dict):
        updated = delete_key(sample_dict, "level1.nonexistent.key")
        assert get_value(updated, "level1.level2") is not None

    def test_delete_key_returns_dict(self, sample_dict):
        result = delete_key(sample_dict, "top_level")
        assert isinstance(result, dict)
        assert result is sample_dict

    def test_delete_key_empty_dict(self):
        result = delete_key({}, "any.key")
        assert result == {}

    def test_delete_key_boolean_value(self, sample_dict):
        updated = delete_key(sample_dict, "boolean_key")
        assert "boolean_key" not in updated

    def test_delete_key_none_value(self, sample_dict):
        updated = delete_key(sample_dict, "none_key")
        assert "none_key" not in updated


class TestSortDictKeysInList:
    def test_sort_dict_keys_in_list_single_dict(self):
        input_list = [{"z": 1, "a": 2, "m": 3}]
        result = sort_dict_keys_in_list(input_list)
        assert list(result[0].keys()) == ["a", "m", "z"]
        assert result[0] == {"a": 2, "m": 3, "z": 1}

    def test_sort_dict_keys_in_list_multiple_dicts(self):
        input_list = [
            {"z": 1, "a": 2, "m": 3},
            {"x": "first", "b": "second", "d": "third"},
            {"y": True, "c": False, "e": None},
        ]
        result = sort_dict_keys_in_list(input_list)
        assert list(result[0].keys()) == ["a", "m", "z"]
        assert list(result[1].keys()) == ["b", "d", "x"]
        assert list(result[2].keys()) == ["c", "e", "y"]

    def test_sort_dict_keys_in_list_already_sorted(self):
        input_list = [{"a": 1, "b": 2, "c": 3}]
        result = sort_dict_keys_in_list(input_list)
        assert list(result[0].keys()) == ["a", "b", "c"]
        assert result[0] == {"a": 1, "b": 2, "c": 3}

    def test_sort_dict_keys_in_list_empty_list(self):
        result = sort_dict_keys_in_list([])
        assert result == []

    def test_sort_dict_keys_in_list_empty_dict(self):
        input_list = [{}]
        result = sort_dict_keys_in_list(input_list)
        assert result == [{}]

    def test_sort_dict_keys_in_list_nested_values(self):
        input_list = [{"z": {"nested": "value"}, "a": [1, 2, 3], "m": {"another": "dict"}}]
        result = sort_dict_keys_in_list(input_list)
        assert list(result[0].keys()) == ["a", "m", "z"]
        assert result[0]["z"] == {"nested": "value"}
        assert result[0]["a"] == [1, 2, 3]
        assert result[0]["m"] == {"another": "dict"}

    def test_sort_dict_keys_in_list_preserves_values(self):
        input_list = [
            {"name": "Alice", "age": 30, "city": "NYC"},
            {"name": "Bob", "age": 25, "city": "LA"},
        ]
        result = sort_dict_keys_in_list(input_list)
        assert result[0] == {"age": 30, "city": "NYC", "name": "Alice"}
        assert result[1] == {"age": 25, "city": "LA", "name": "Bob"}

    def test_sort_dict_keys_in_list_mixed_types(self):
        input_list = [{"z": 1, "a": "string", "m": True, "b": None, "y": [1, 2, 3]}]
        result = sort_dict_keys_in_list(input_list)
        assert list(result[0].keys()) == ["a", "b", "m", "y", "z"]
        assert result[0]["z"] == 1
        assert result[0]["a"] == "string"
        assert result[0]["m"] is True
        assert result[0]["b"] is None
        assert result[0]["y"] == [1, 2, 3]

    def test_sort_dict_keys_in_list_case_sensitive_sorting(self):
        input_list = [{"Z": 1, "a": 2, "B": 3, "c": 4}]
        result = sort_dict_keys_in_list(input_list)
        # Python sorts uppercase before lowercase
        assert list(result[0].keys()) == ["B", "Z", "a", "c"]

    def test_sort_dict_keys_in_list_numeric_string_keys(self):
        input_list = [{"3": "three", "1": "one", "2": "two"}]
        result = sort_dict_keys_in_list(input_list)
        assert list(result[0].keys()) == ["1", "2", "3"]
        assert result[0] == {"1": "one", "2": "two", "3": "three"}

    def test_sort_dict_keys_in_list_does_not_modify_original(self):
        input_list = [{"z": 1, "a": 2, "m": 3}]
        original_keys = list(input_list[0].keys())
        result = sort_dict_keys_in_list(input_list)
        # Original list should not be modified
        assert list(input_list[0].keys()) == original_keys
        # Result should have sorted keys
        assert list(result[0].keys()) == ["a", "m", "z"]


class TestRenameKey:
    def test_rename_existing_top_level_key(self):
        d = {"gpuModel": "MI300X", "count": 2}
        result = rename_key(d, "gpuModel", "gpu")
        assert result == {"gpu": "MI300X", "count": 2}
        assert "gpuModel" not in result

    def test_rename_missing_key_is_noop(self):
        d = {"count": 2}
        result = rename_key(d, "gpuModel", "gpu")
        assert result == {"count": 2}

    def test_rename_modifies_in_place(self):
        d = {"gpuModel": "MI300X"}
        updated = rename_key(d, "gpuModel", "gpu")
        assert "gpu" in d
        assert updated is d


class TestRenameKeys:
    def test_rename_multiple_keys(self):
        d = {"gpuModel": "MI300X", "gpuCount": 4, "engine": "vllm"}
        result = rename_keys(d, {"gpuModel": "gpu", "gpuCount": "gpu_count"})
        assert result == {"gpu": "MI300X", "gpu_count": 4, "engine": "vllm"}

    def test_rename_keys_empty_mapping(self):
        d = {"gpuModel": "MI300X"}
        result = rename_keys(d, {})
        assert result == {"gpuModel": "MI300X"}

    def test_rename_keys_skips_missing(self):
        d = {"engine": "vllm"}
        result = rename_keys(d, {"gpuModel": "gpu", "gpuCount": "gpu_count"})
        assert result == {"engine": "vllm"}
