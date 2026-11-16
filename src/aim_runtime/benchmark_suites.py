# Copyright © Advanced Micro Devices, Inc., or its affiliates.
#
# SPDX-License-Identifier: MIT
"""Benchmark suite selection from declarative accelerator rules."""

from collections.abc import Mapping
from typing import Any


def select_benchmark_suite(
    rules: object,
    *,
    accelerator_type: str,
    accelerator_count: int | None,
) -> str | None:
    """Return the suite from the first rule matching the accelerator details."""
    if rules is None:
        return None
    if not isinstance(rules, list):
        raise ValueError("suite_selection must be a list")

    values = {
        "accelerator_type": accelerator_type,
        "accelerator_count": accelerator_count,
    }
    for index, rule in enumerate(rules, start=1):
        if not isinstance(rule, Mapping):
            raise ValueError(f"suite_selection rule {index} must be a mapping")
        suite = rule.get("suite")
        match = rule.get("match")
        if not isinstance(suite, str) or not suite:
            raise ValueError(f"suite_selection rule {index} must provide a non-empty suite")
        if not isinstance(match, Mapping) or not match:
            raise ValueError(f"suite_selection rule {index} must provide a non-empty match mapping")
        unknown_keys = set(match) - set(values)
        if unknown_keys:
            raise ValueError(f"suite_selection rule {index} contains unsupported match fields: {sorted(unknown_keys)}")
        if all(_matches(values[field], expected) for field, expected in match.items()):
            return suite
    return None


def _matches(value: Any, expected: Any) -> bool:
    return value in expected if isinstance(expected, list) else value == expected
