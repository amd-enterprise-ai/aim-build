# Copyright © Advanced Micro Devices, Inc., or its affiliates.
#
# SPDX-License-Identifier: MIT

import functools
import logging
import re
from typing import Optional

from aim_common.compat import StrEnum

logger = logging.getLogger(__name__)


class AIMVersionSuffixType(StrEnum):
    RC = "rc"
    PREVIEW = "preview"
    STABLE = ""

    @classmethod
    def read_enum(cls, value: str) -> Optional["AIMVersionSuffixType"]:
        if value is not None:
            try:
                return AIMVersionSuffixType(value)
            except ValueError as e:
                logger.error(
                    f"Invalid filter_suffix: {value}. Must be one of {[suffix.value for suffix in AIMVersionSuffixType]}"
                )
                raise e
        return None


class AIMVersionSuffix:

    SUFFIX_PATTERN = re.compile(r"^(rc\d+|preview|)$")

    def __init__(self, suffix: str) -> None:
        self.suffix_type = None

        match = self.SUFFIX_PATTERN.match(suffix)
        if match:
            if suffix.startswith("rc"):
                self.suffix_type = AIMVersionSuffixType.RC
            else:
                self.suffix_type = AIMVersionSuffixType.read_enum(suffix)
        else:
            raise ValueError(
                f"Invalid suffix: {suffix}. Must be one of {[suffix.value for suffix in AIMVersionSuffixType]}"
            )

        self.suffix = suffix

    @property
    def rc_number(self) -> Optional[int]:
        """
        Extract the RC number from the suffix if it's an RC version.

        Returns:
            Optional[int]: The RC number if suffix is rcN, None otherwise
        """
        if self.suffix_type == AIMVersionSuffixType.RC:
            rc_match = re.match(r"rc(\d+)", self.suffix)
            if rc_match:
                return int(rc_match.group(1))
        return None


@functools.total_ordering
class AIMVersion:

    # Compiled patterns aligned with .github/versioning-strategy.md.
    # Full version: MAJOR.MINOR.PATCH with optional -rcN or -preview suffix (model-specific images)
    pattern = re.compile(r"^(0|[1-9]\d*)\.(0|[1-9]\d*)\.(0|[1-9]\d*)(?:-(rc[1-9]\d*|preview))?$")
    # Base version: MAJOR.MINOR with an optional .PATCH and optional -rcN or -preview suffix (base images).
    # PATCH is optional so both legacy two-part tags (0.13-rc5) and the three-part tags emitted since
    # PR #1188 (0.13.0-rc5) validate.
    base_pattern = re.compile(r"^(0|[1-9]\d*)\.(0|[1-9]\d*)(?:\.(0|[1-9]\d*))?(?:-(rc[1-9]\d*|preview))?$")
    # Stable full version: MAJOR.MINOR.PATCH without any suffix
    pattern_stable = re.compile(r"^(0|[1-9]\d*)\.(0|[1-9]\d*)\.(0|[1-9]\d*)$")
    # Stable base version: MAJOR.MINOR with an optional .PATCH and without any suffix
    base_pattern_stable = re.compile(r"^(0|[1-9]\d*)\.(0|[1-9]\d*)(?:\.(0|[1-9]\d*))?$")

    def __init__(self, version: str, is_base: bool = False) -> None:
        pattern = self.base_pattern if is_base else self.pattern
        if not pattern.fullmatch(version):
            expected_fmt = "MAJOR.MINOR[.PATCH][-rcN|-preview]" if is_base else "MAJOR.MINOR.PATCH[-rcN|-preview]"
            raise ValueError(f"Invalid version format: '{version}'. Expected {expected_fmt}")

        self.version = version
        self.is_base = is_base

    @classmethod
    def __patch_preview(cls, version: str) -> str:
        return version.replace("preview", "rcpreview")

    @classmethod
    def __patch_rc(cls, version: str) -> str:
        return version.replace("rc", "rc.")

    def __patch_base(self, version: str) -> str:
        def convert_base_to_semver(version: str) -> str:
            match = self.base_pattern.fullmatch(version)

            if match:
                major, minor, patch, suffix = match.groups()
                patch = patch if patch is not None else "0"
                if suffix:
                    return f"{major}.{minor}.{patch}-{suffix}"
                else:
                    return f"{major}.{minor}.{patch}"

            return version

        return convert_base_to_semver(version)

    def __lt__(self, other) -> bool:
        """
        This ensures that versions are sorted correctly:
        - Base versions (X.Y-suffix) are converted to X.Y.0-suffix
        - Preview versions are converted to rcpreview for proper ordering
        :param other: other version to compare
        :return: bool indicating if self is less than other
        """
        self_version = self.version
        other_version = other.version

        if self.is_base:
            self_version = self.__patch_base(self_version)

        if other.is_base:
            other_version = self.__patch_base(other.version)

        self_version = self.__patch_preview(self_version)
        other_version = self.__patch_preview(other_version)

        self_version = self.__patch_rc(self_version)
        other_version = self.__patch_rc(other_version)

        from semver import VersionInfo

        return VersionInfo.parse(self_version) < VersionInfo.parse(other_version)

    def __eq__(self, other) -> bool:
        return self.version == other.version

    def __str__(self) -> str:
        return self.version

    @property
    def is_stable(self) -> bool:
        """
        Checks if the version is stable (i.e., has no suffix). Useful for generating documentation specifically for
        stable (i.e. released) versions

        Returns:
            bool: True if the version is stable, False otherwise
        """
        if self.is_base:
            return self.base_pattern_stable.fullmatch(self.version) is not None
        return self.pattern_stable.fullmatch(self.version) is not None

    @property
    def core(self) -> Optional[str]:
        if self.is_base:
            match = self.base_pattern.match(self.version)
            if match:
                major, minor, patch, _ = match.groups()
                if patch is not None:
                    return f"{major}.{minor}.{patch}"
                return f"{major}.{minor}"
        else:
            match = self.pattern.match(self.version)

            if match:
                major, minor, patch, _ = match.groups()
                return f"{major}.{minor}.{patch}"

        return None

    @property
    def suffix(self) -> Optional[AIMVersionSuffix]:
        if self.is_stable:
            return AIMVersionSuffix("")

        if self.is_base:
            match = self.base_pattern.match(self.version)
            if match:
                _, _, _, suffix = match.groups()
                return AIMVersionSuffix(suffix)
        else:
            match = self.pattern.match(self.version)
            if match:
                _, _, _, suffix = match.groups()
                return AIMVersionSuffix(suffix)

        return None

    @property
    def major(self) -> Optional[int]:
        core = self.core
        if not core:
            return None

        parts = core.split(".")
        if len(parts) < 1:
            return None

        return int(parts[0])

    @property
    def minor(self) -> Optional[int]:
        core = self.core
        if not core:
            return None

        parts = core.split(".")
        if len(parts) < 2:
            return None

        return int(parts[1])

    @property
    def major_minor(self) -> Optional[str]:
        """Return the ``MAJOR.MINOR`` series string, dropping any patch/suffix.

        This is the canonical "version series" a model image inherits from its
        base image, e.g. both base tags ``0.13`` and ``0.13.0`` yield ``0.13``.
        """
        major, minor = self.major, self.minor
        if major is None or minor is None:
            return None
        return f"{major}.{minor}"

    @classmethod
    def from_string(cls, version: str) -> "AIMVersion":
        version_parts = version.split(".")
        is_base = len(version_parts) < 3
        return cls(version, is_base=is_base)


def validate_version_tag(version: str, is_base: bool = False) -> None:
    """Validate a version string against the AIM versioning strategy.

    Base images use MAJOR.MINOR[-rcN|-preview] format.
    Model-specific images use MAJOR.MINOR.PATCH[-rcN|-preview] format.
    RC numbers start at 1 (rc0 is not valid).

    Args:
        version: The version string to validate.
        is_base: If True, validate as base version (no patch component).

    Raises:
        ValueError: If the version string does not match the expected format.
    """
    AIMVersion(version, is_base=is_base)
