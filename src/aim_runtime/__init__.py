# Copyright © Advanced Micro Devices, Inc., or its affiliates.
#
# SPDX-License-Identifier: MIT

"""
AIM Runtime Package

This package provides classes and utilities for AIM runtime operations including
profile selection, command generation, GPU detection, and configuration management.
"""

from .aim_runtime import AIMRuntime
from .command_generator import CommandGenerator
from .config import DEFAULT_CACHE_PATH, DEFAULT_PROFILE_BASE_PATH, DEFAULT_SCHEMA_SEARCH_PATH, AIMConfig
from .gpu_detector import GPUDetector
from .logging_config import configure_logging
from .profile_registry import Profile, ProfileMetadata, ProfileRegistry
from .profile_selector import ProfileCompatibilityResult, ProfileCompatibilityState, ProfileSelector

__version__ = "0.1.0"
__all__ = [
    "AIMConfig",
    "DEFAULT_CACHE_PATH",
    "DEFAULT_PROFILE_BASE_PATH",
    "DEFAULT_SCHEMA_SEARCH_PATH",
    "ProfileSelector",
    "ProfileRegistry",
    "Profile",
    "ProfileMetadata",
    "ProfileCompatibilityState",
    "ProfileCompatibilityResult",
    "CommandGenerator",
    "AIMRuntime",
    "GPUDetector",
    "configure_logging",
]
