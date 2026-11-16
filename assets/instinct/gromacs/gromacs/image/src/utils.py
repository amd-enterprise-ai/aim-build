#!/usr/bin/env python3

# Copyright © Advanced Micro Devices, Inc., or its affiliates.
#
# SPDX-License-Identifier: MIT

"""
Shared utilities for the GROMACS job AIM (entrypoint.py and runner.py).
"""


def parse_performance(log_text: str) -> "tuple[float, float]":
    """Return (ns_per_day, hours_per_ns) from gmx mdrun log, or (0.0, 0.0).

    Searches for the line: 'Performance:  XX.XXX  YY.YYY'
    where the first column is ns/day and the second is hours/ns.
    Returns (0.0, 0.0) if the line is not found or cannot be parsed.
    """
    for line in log_text.splitlines():
        stripped = line.strip()
        if stripped.startswith("Performance:"):
            parts = stripped.split()
            # Performance:  <ns/day>  <hours/ns>
            if len(parts) >= 3:
                try:
                    return float(parts[1]), float(parts[2])
                except ValueError:
                    pass
    return 0.0, 0.0
