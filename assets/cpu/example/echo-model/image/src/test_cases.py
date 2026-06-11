#!/usr/bin/env python3

# Copyright © Advanced Micro Devices, Inc., or its affiliates.
#
# SPDX-License-Identifier: MIT

"""
Shared test fixtures for the echo-model.

This module is imported by harness.py via a sibling import
(``from test_cases import KNOWN_INPUTS, expected_output``).
It lives in the same image/src/ directory and is copied to
/workspace/model/src/ alongside the harness at build time.
"""

from __future__ import annotations

KNOWN_INPUTS: list[dict[str, object]] = [
    {"text": "hello", "expected_reversed": "olleh", "expected_length": 5},
    {"text": "AIM harness", "expected_reversed": "ssenrah MIA", "expected_length": 11},
    {"text": "", "expected_reversed": "", "expected_length": 0},
    {"text": "racecar", "expected_reversed": "racecar", "expected_length": 7},
]


def expected_output(text: str) -> dict[str, object]:
    """Return the expected echo-service response for a given input string."""
    return {"reversed": text[::-1], "length": len(text)}
