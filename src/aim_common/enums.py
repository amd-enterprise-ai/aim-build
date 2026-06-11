#!/usr/bin/env python3

# Copyright © Advanced Micro Devices, Inc., or its affiliates.
#
# SPDX-License-Identifier: MIT
"""Shared enums for AIM utilities."""

from typing import Optional, TypeVar

from aim_common.compat import StrEnum

T = TypeVar("T", bound="ParseableStrEnum")


class ParseableStrEnum(StrEnum):
    """StrEnum with try_parse support for safe case-insensitive parsing."""

    @classmethod
    def try_parse(cls: type[T], value: str) -> Optional[T]:
        """Try to parse a string into an enum value. Returns None if not valid."""
        if value is None:
            return None
        folded = value.casefold()
        for member in cls:
            if member.value.casefold() == folded:
                return member  # type: ignore[return-value]
        return None
