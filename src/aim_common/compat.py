# Copyright © Advanced Micro Devices, Inc., or its affiliates.
#
# SPDX-License-Identifier: MIT

"""Compatibility shims for older Python versions."""

import sys
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    # Mypy always sees the real stdlib StrEnum (available in Python >= 3.11).
    from enum import StrEnum
elif sys.version_info >= (3, 11):
    from enum import StrEnum
else:
    from enum import Enum

    class StrEnum(str, Enum):  # type: ignore[no-redef]
        """Backport of enum.StrEnum for Python < 3.11."""

        @staticmethod
        def _generate_next_value_(name: str, start: int, count: int, last_values: list) -> str:
            return name.lower()

        def __str__(self) -> str:
            return self.value


__all__ = ["StrEnum"]
