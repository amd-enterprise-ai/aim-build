# Copyright © Advanced Micro Devices, Inc., or its affiliates.
#
# SPDX-License-Identifier: MIT

import pytest

from aim_utils.dict_utils import delete_key, get_value, set_value


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
