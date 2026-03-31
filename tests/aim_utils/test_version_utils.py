# Copyright © Advanced Micro Devices, Inc., or its affiliates.
#
# SPDX-License-Identifier: MIT
import pytest

from aim_utils.version_utils import AIMVersion, AIMVersionSuffixType, validate_version_tag


def test_sort_order():
    versions = ["0.7.0", "0.6.0", "0.7.1", "0.6.10", "0.7.1-rc2", "0.7.0-preview", "0.7.0-rc1", "0.7.1-rc1"]

    aim_versions = []
    for version in versions:
        aim_versions.append(AIMVersion(version))

    actual = sorted(aim_versions)
    actual = list(map(str, actual))
    expected = ["0.6.0", "0.6.10", "0.7.0-rc1", "0.7.0-preview", "0.7.0", "0.7.1-rc1", "0.7.1-rc2", "0.7.1"]

    assert actual == expected


def test_sort_order_base():
    versions = ["0.7", "0.6", "0.7-rc2", "0.7-preview", "0.7-rc1"]

    aim_versions = []
    for version in versions:
        aim_versions.append(AIMVersion(version, is_base=True))

    actual = sorted(aim_versions)
    actual = list(map(str, actual))
    expected = ["0.6", "0.7-rc1", "0.7-rc2", "0.7-preview", "0.7"]

    assert actual == expected


def test_eq_comparison():
    """Test __eq__ method for version equality"""
    # Test equal versions
    v1 = AIMVersion("1.0.0")
    v2 = AIMVersion("1.0.0")
    assert v1 == v2

    # Test equal preview versions
    v3 = AIMVersion("1.0.0-preview")
    v4 = AIMVersion("1.0.0-preview")
    assert v3 == v4

    # Test equal rc versions
    v5 = AIMVersion("1.0.0-rc1")
    v6 = AIMVersion("1.0.0-rc1")
    assert v5 == v6

    # Test unequal versions
    v7 = AIMVersion("1.0.0")
    v8 = AIMVersion("1.0.1")
    assert not (v7 == v8)


def test_lt_comparison():
    """Test __lt__ method for less than comparison"""
    # Test basic version comparison
    v1 = AIMVersion("1.0.0")
    v2 = AIMVersion("1.0.1")
    assert v1 < v2

    # Test rc vs release
    v3 = AIMVersion("1.0.0-rc1")
    v4 = AIMVersion("1.0.0")
    assert v3 < v4

    # Test preview vs rc (preview should come after rc)
    v5 = AIMVersion("1.0.0-rc1")
    v6 = AIMVersion("1.0.0-preview")
    assert v5 < v6

    # Test preview vs release
    v7 = AIMVersion("1.0.0-preview")
    v8 = AIMVersion("1.0.0")
    assert v7 < v8

    # Test rc versions
    v9 = AIMVersion("1.0.0-rc1")
    v10 = AIMVersion("1.0.0-rc2")
    assert v9 < v10

    # Test rc versions
    v11 = AIMVersion("1.0.0-rc9")
    v12 = AIMVersion("1.0.0-rc46")
    assert v11 < v12


def test_gt_comparison():
    """Test __gt__ method for greater than comparison"""
    # Test basic version comparison
    v1 = AIMVersion("1.0.1")
    v2 = AIMVersion("1.0.0")
    assert v1 > v2

    # Test release vs rc
    v3 = AIMVersion("1.0.0")
    v4 = AIMVersion("1.0.0-rc1")
    assert v3 > v4

    # Test preview vs rc (preview should come after rc)
    v5 = AIMVersion("1.0.0-preview")
    v6 = AIMVersion("1.0.0-rc1")
    assert v5 > v6

    # Test release vs preview
    v7 = AIMVersion("1.0.0")
    v8 = AIMVersion("1.0.0-preview")
    assert v7 > v8

    # Test rc versions
    v9 = AIMVersion("1.0.0-rc2")
    v10 = AIMVersion("1.0.0-rc1")
    assert v9 > v10


def test_le_comparison():
    """Test __le__ method for less than or equal comparison"""
    # Test equal versions
    v1 = AIMVersion("1.0.0")
    v2 = AIMVersion("1.0.0")
    assert v1 <= v2
    assert v2 <= v1

    # Test less than
    v3 = AIMVersion("1.0.0")
    v4 = AIMVersion("1.0.1")
    assert v3 <= v4

    # Test rc vs release
    v5 = AIMVersion("1.0.0-rc1")
    v6 = AIMVersion("1.0.0")
    assert v5 <= v6

    # Test preview vs release
    v7 = AIMVersion("1.0.0-preview")
    v8 = AIMVersion("1.0.0")
    assert v7 <= v8

    # Test equal preview versions
    v9 = AIMVersion("1.0.0-preview")
    v10 = AIMVersion("1.0.0-preview")
    assert v9 <= v10


def test_ge_comparison():
    """Test __ge__ method for greater than or equal comparison"""
    # Test equal versions
    v1 = AIMVersion("1.0.0")
    v2 = AIMVersion("1.0.0")
    assert v1 >= v2
    assert v2 >= v1

    # Test greater than
    v3 = AIMVersion("1.0.1")
    v4 = AIMVersion("1.0.0")
    assert v3 >= v4
    assert not (v4 >= v3)

    # Test release vs rc
    v5 = AIMVersion("1.0.0")
    v6 = AIMVersion("1.0.0-rc1")
    assert v5 >= v6

    # Test release vs preview
    v7 = AIMVersion("1.0.0")
    v8 = AIMVersion("1.0.0-preview")
    assert v7 >= v8

    # Test equal rc versions
    v9 = AIMVersion("1.0.0-rc1")
    v10 = AIMVersion("1.0.0-rc1")
    assert v9 >= v10


def test_invalid_version_format():
    """Test that invalid version formats raise ValueError"""
    with pytest.raises(ValueError):
        AIMVersion("1.0")


def test_core_no_suffix():
    actual = AIMVersion("1.2.3").core
    expected = "1.2.3"
    assert actual == expected


def test_core_with_suffix():
    actual = AIMVersion("1.2.3-rc1").core
    expected = "1.2.3"
    assert actual == expected


def test_core_base_no_suffix():
    actual = AIMVersion("1.2", is_base=True).core
    expected = "1.2"
    assert actual == expected


def test_core_base_with_suffix():
    actual = AIMVersion("1.2-rc1", is_base=True).core
    expected = "1.2"
    assert actual == expected


def test_suffix_no_suffix():
    suffix = AIMVersion("1.2.3").suffix
    assert suffix.suffix == ""
    assert suffix.suffix_type == AIMVersionSuffixType.STABLE


def test_suffix_with_suffix():
    suffix = AIMVersion("1.2.3-rc1").suffix
    assert suffix.suffix == "rc1"
    assert suffix.suffix_type == AIMVersionSuffixType.RC
    assert suffix.rc_number == 1


def test_suffix_base_no_suffix():
    suffix = AIMVersion("1.2", is_base=True).suffix
    assert suffix.suffix == ""
    assert suffix.suffix_type == AIMVersionSuffixType.STABLE


def test_suffix_base_with_suffix():
    suffix = AIMVersion("1.2-rc2", is_base=True).suffix
    assert suffix.suffix == "rc2"
    assert suffix.suffix_type == AIMVersionSuffixType.RC
    assert suffix.rc_number == 2


def test_major_no_suffix():
    actual = AIMVersion("1.2.3").major
    expected = 1
    assert actual == expected


def test_major_with_suffix():
    actual = AIMVersion("2.5.7-rc1").major
    expected = 2
    assert actual == expected


def test_major_base_no_suffix():
    actual = AIMVersion("3.4", is_base=True).major
    expected = 3
    assert actual == expected


def test_major_base_with_suffix():
    actual = AIMVersion("5.6-preview", is_base=True).major
    expected = 5
    assert actual == expected


def test_major_zero():
    actual = AIMVersion("0.7.2").major
    expected = 0
    assert actual == expected


def test_minor_no_suffix():
    actual = AIMVersion("1.2.3").minor
    expected = 2
    assert actual == expected


def test_minor_with_suffix():
    actual = AIMVersion("2.5.7-rc1").minor
    expected = 5
    assert actual == expected


def test_minor_base_no_suffix():
    actual = AIMVersion("3.4", is_base=True).minor
    expected = 4
    assert actual == expected


def test_minor_base_with_suffix():
    actual = AIMVersion("5.6-preview", is_base=True).minor
    expected = 6
    assert actual == expected


def test_minor_zero():
    actual = AIMVersion("7.0.2").minor
    expected = 0
    assert actual == expected


def test_minor_large_number():
    actual = AIMVersion("0.42.1").minor
    expected = 42
    assert actual == expected


# ---------------------------------------------------------------------------
# Tests for validate_version_tag()
# ---------------------------------------------------------------------------


class TestValidateVersionTag:
    """Tests for the standalone validate_version_tag function."""

    # --- Valid full (model-specific) versions ---

    @pytest.mark.parametrize(
        "version",
        [
            "0.4.0",
            "0.4.2",
            "1.0.0",
            "0.4.2-rc1",
            "0.4.2-rc99",
            "0.4.2-preview",
            "10.20.30",
        ],
    )
    def test_valid_full_versions(self, version):
        validate_version_tag(version, is_base=False)

    # --- Valid base versions ---

    @pytest.mark.parametrize(
        "version",
        [
            "0.4",
            "1.0",
            "0.11",
            "0.4-rc1",
            "0.4-rc10",
            "0.4-preview",
            "10.20",
        ],
    )
    def test_valid_base_versions(self, version):
        validate_version_tag(version, is_base=True)

    # --- Invalid full versions ---

    @pytest.mark.parametrize(
        "version",
        [
            "",
            "abc",
            "0.4",  # base format, not full
            "0.4.2.1",  # four components
            "0.4.2-rc",  # missing rc number
            "0.4.2-rc0",  # rc0 not allowed
            "0.4.2-beta",  # unsupported suffix
            "v0.4.2",  # leading 'v'
            "0.04.2",  # leading zero in minor
            "0.4.02",  # leading zero in patch
        ],
    )
    def test_invalid_full_versions(self, version):
        with pytest.raises(ValueError):
            validate_version_tag(version, is_base=False)

    # --- Invalid base versions ---

    @pytest.mark.parametrize(
        "version",
        [
            "",
            "abc",
            "0.4.0",  # full format, not base
            "0.4-rc",  # missing rc number
            "0.4-rc0",  # rc0 not allowed
            "0.4-beta",  # unsupported suffix
            "v0.4",  # leading 'v'
            "0.04",  # leading zero in minor
        ],
    )
    def test_invalid_base_versions(self, version):
        with pytest.raises(ValueError):
            validate_version_tag(version, is_base=True)
