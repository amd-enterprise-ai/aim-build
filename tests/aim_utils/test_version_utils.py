# Copyright © Advanced Micro Devices, Inc., or its affiliates.
#
# SPDX-License-Identifier: MIT
import pytest

from aim_utils.version_utils import AIMVersion, AIMVersionSuffix, AIMVersionSuffixType, validate_version_tag


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


@pytest.mark.parametrize(
    "version, is_base, expected",
    [
        ("1.2.3", False, "1.2"),
        ("1.2.3-rc1", False, "1.2"),
        ("1.2", True, "1.2"),
        ("1.2-rc1", True, "1.2"),
        # Three-part base tags emitted since PR #1188 collapse to their MAJOR.MINOR series.
        ("0.13.0", True, "0.13"),
        ("0.13.0-rc5", True, "0.13"),
        ("0.13.0-preview", True, "0.13"),
    ],
)
def test_major_minor(version, is_base, expected):
    assert AIMVersion(version, is_base=is_base).major_minor == expected


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
# Tests for AIMVersionSuffixType.read_enum()
# ---------------------------------------------------------------------------


class TestAIMVersionSuffixTypeReadEnum:
    def test_read_enum_rc(self):
        result = AIMVersionSuffixType.read_enum("rc")
        assert result == AIMVersionSuffixType.RC

    def test_read_enum_preview(self):
        result = AIMVersionSuffixType.read_enum("preview")
        assert result == AIMVersionSuffixType.PREVIEW

    def test_read_enum_stable(self):
        result = AIMVersionSuffixType.read_enum("")
        assert result == AIMVersionSuffixType.STABLE

    def test_read_enum_none_returns_none(self):
        result = AIMVersionSuffixType.read_enum(None)
        assert result is None

    def test_read_enum_invalid_raises(self):
        with pytest.raises(ValueError):
            AIMVersionSuffixType.read_enum("beta")


# ---------------------------------------------------------------------------
# Tests for AIMVersionSuffix
# ---------------------------------------------------------------------------


class TestAIMVersionSuffix:
    def test_empty_suffix_is_stable(self):
        s = AIMVersionSuffix("")
        assert s.suffix == ""
        assert s.suffix_type == AIMVersionSuffixType.STABLE

    def test_preview_suffix(self):
        s = AIMVersionSuffix("preview")
        assert s.suffix == "preview"
        assert s.suffix_type == AIMVersionSuffixType.PREVIEW

    def test_rc_suffix(self):
        s = AIMVersionSuffix("rc1")
        assert s.suffix == "rc1"
        assert s.suffix_type == AIMVersionSuffixType.RC

    def test_rc_suffix_large_number(self):
        s = AIMVersionSuffix("rc42")
        assert s.suffix == "rc42"
        assert s.suffix_type == AIMVersionSuffixType.RC

    def test_invalid_suffix_raises(self):
        with pytest.raises(ValueError):
            AIMVersionSuffix("beta")

    def test_invalid_suffix_rc0_raises(self):
        # rc0 does not match the pattern rc[1-9]\d* used in AIMVersion,
        # but AIMVersionSuffix uses a looser pattern; verify behaviour
        # The SUFFIX_PATTERN allows rc\d+ (any digits), so "rc0" actually
        # matches the suffix pattern directly. This test documents that.
        s = AIMVersionSuffix("rc0")
        assert s.suffix_type == AIMVersionSuffixType.RC
        assert s.rc_number == 0

    def test_rc_number_returns_int(self):
        s = AIMVersionSuffix("rc5")
        assert s.rc_number == 5

    def test_rc_number_multidigit(self):
        s = AIMVersionSuffix("rc99")
        assert s.rc_number == 99

    def test_rc_number_none_for_preview(self):
        s = AIMVersionSuffix("preview")
        assert s.rc_number is None

    def test_rc_number_none_for_stable(self):
        s = AIMVersionSuffix("")
        assert s.rc_number is None


# ---------------------------------------------------------------------------
# Tests for AIMVersion.is_stable
# ---------------------------------------------------------------------------


class TestAIMVersionIsStable:
    def test_stable_full_version(self):
        assert AIMVersion("1.0.0").is_stable is True

    def test_stable_full_version_zero(self):
        assert AIMVersion("0.11.0").is_stable is True

    def test_rc_full_version_not_stable(self):
        assert AIMVersion("1.0.0-rc1").is_stable is False

    def test_preview_full_version_not_stable(self):
        assert AIMVersion("1.0.0-preview").is_stable is False

    def test_stable_base_version(self):
        assert AIMVersion("1.0", is_base=True).is_stable is True

    def test_stable_base_version_zero(self):
        assert AIMVersion("0.11", is_base=True).is_stable is True

    def test_rc_base_version_not_stable(self):
        assert AIMVersion("1.0-rc1", is_base=True).is_stable is False

    def test_preview_base_version_not_stable(self):
        assert AIMVersion("1.0-preview", is_base=True).is_stable is False


# ---------------------------------------------------------------------------
# Tests for AIMVersion.__str__
# ---------------------------------------------------------------------------


class TestAIMVersionStr:
    def test_str_full_stable(self):
        assert str(AIMVersion("1.2.3")) == "1.2.3"

    def test_str_full_rc(self):
        assert str(AIMVersion("1.2.3-rc1")) == "1.2.3-rc1"

    def test_str_full_preview(self):
        assert str(AIMVersion("1.2.3-preview")) == "1.2.3-preview"

    def test_str_base_stable(self):
        assert str(AIMVersion("0.11", is_base=True)) == "0.11"

    def test_str_base_rc(self):
        assert str(AIMVersion("0.11-rc2", is_base=True)) == "0.11-rc2"


# ---------------------------------------------------------------------------
# Tests for AIMVersion.from_string()
# ---------------------------------------------------------------------------


class TestAIMVersionFromString:
    def test_from_string_full_stable(self):
        v = AIMVersion.from_string("1.2.3")
        assert str(v) == "1.2.3"
        assert v.is_base is False

    def test_from_string_full_rc(self):
        v = AIMVersion.from_string("1.2.3-rc1")
        assert str(v) == "1.2.3-rc1"
        assert v.is_base is False

    def test_from_string_full_preview(self):
        v = AIMVersion.from_string("1.2.3-preview")
        assert str(v) == "1.2.3-preview"
        assert v.is_base is False

    def test_from_string_base_stable(self):
        v = AIMVersion.from_string("0.11")
        assert str(v) == "0.11"
        assert v.is_base is True

    def test_from_string_base_rc(self):
        v = AIMVersion.from_string("0.11-rc1")
        assert str(v) == "0.11-rc1"
        assert v.is_base is True

    def test_from_string_base_preview(self):
        v = AIMVersion.from_string("1.0-preview")
        assert str(v) == "1.0-preview"
        assert v.is_base is True

    def test_from_string_major_only_is_base(self):
        # Only one dot-separated component → treated as base
        # "1" splits into ["1"] which is length 1 < 3, so is_base=True,
        # but "1" is not a valid base version (needs MAJOR.MINOR).
        with pytest.raises(ValueError):
            AIMVersion.from_string("1")


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
            # Three-part base tags emitted since PR #1188 ("Enable patch version for base images")
            "0.13.0",
            "0.13.0-rc5",
            "0.13.0-preview",
            "0.4.2-rc1",
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
            "0.4.2.1",  # four components
            "0.4-rc",  # missing rc number
            "0.4-rc0",  # rc0 not allowed
            "0.4.0-rc0",  # rc0 not allowed (three-part)
            "0.4-beta",  # unsupported suffix
            "v0.4",  # leading 'v'
            "0.04",  # leading zero in minor
            "0.4.02",  # leading zero in patch
        ],
    )
    def test_invalid_base_versions(self, version):
        with pytest.raises(ValueError):
            validate_version_tag(version, is_base=True)
