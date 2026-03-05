# Copyright © Advanced Micro Devices, Inc., or its affiliates.
#
# SPDX-License-Identifier: MIT

import functools
import logging
import re
import sys
from enum import Enum
from typing import Optional

from semver import VersionInfo

# TODO: Remove this compatibility workaround once the ROCm base image is updated to Python 3.12
# Python 3.10 compatibility: define StrEnum if not available
if sys.version_info >= (3, 11):
    from enum import StrEnum
else:

    class StrEnum(str, Enum):
        """Minimal StrEnum for Python <3.11."""


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

    def __init__(self, version: str, is_base: bool = False) -> None:
        # Matches full version format: major.minor.patch with optional -rc# or -preview suffix
        self.pattern = re.compile(r"^(0|[1-9]\d*)\.(0|[1-9]\d*)\.(0|[1-9]\d*)(?:-(rc\d+|preview))?$")
        # Matches base version format: major.minor with optional -rc# or -preview suffix (no patch)
        self.base_pattern = re.compile(r"^(0|[1-9]\d*)\.(0|[1-9]\d*)(?:-(rc\d+|preview))?$")
        # Matches stable full version format: major.minor.patch without any suffix
        self.pattern_stable = re.compile(r"^(0|[1-9]\d*)\.(0|[1-9]\d*)\.(0|[1-9]\d*)$")
        # Matches stable base version format: major.minor without any suffix (no patch)
        self.base_pattern_stable = re.compile(r"^(0|[1-9]\d*)\.(0|[1-9]\d*)$")

        pattern = self.base_pattern if is_base else self.pattern
        if not pattern.fullmatch(version):
            raise ValueError(f"Invalid version format: {version}")

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
                major, minor, suffix = match.groups()
                if suffix:
                    return f"{major}.{minor}.0-{suffix}"
                else:
                    return f"{major}.{minor}.0"

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
                major, minor, _ = match.groups()
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
                _, _, suffix = match.groups()
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
