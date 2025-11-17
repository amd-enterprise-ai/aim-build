# Copyright © Advanced Micro Devices, Inc., or its affiliates.
#
# SPDX-License-Identifier: MIT

"""
AIM Runtime Logging Configuration

This module handles logging configuration for the AIM runtime.
"""

import logging
import sys


def configure_logging(root_log_level: str = "WARNING", aim_log_level: str = "INFO") -> None:
    """
    Configure logging for the AIM runtime with separate controls for root and AIM loggers.

    Args:
        root_log_level: Log level for the root logger (DEBUG, INFO, WARNING, ERROR, CRITICAL).
                       Controls third-party and external package logging. Default: WARNING.
        aim_log_level: Log level for aim_runtime package loggers (DEBUG, INFO, WARNING, ERROR, CRITICAL).
                      Controls AIM-specific logging. Default: INFO.

    Note:
        - Root logger controls all loggers by default, but aim_runtime loggers override this
        - For maximum verbosity, set both to DEBUG
        - For production, use WARNING for root and INFO for aim_runtime (defaults)
    """
    # Parse log level strings to logging constants
    root_level = getattr(logging, root_log_level.upper(), logging.WARNING)
    aim_level = getattr(logging, aim_log_level.upper(), logging.INFO)

    # Configure the root logger
    logging.basicConfig(
        level=root_level,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=[logging.StreamHandler(sys.stderr)],  # Use stderr for logs
        force=True,  # Reconfigure even if already configured
    )

    # Configure aim_runtime logger separately
    aim_logger = logging.getLogger("aim_runtime")
    aim_logger.setLevel(aim_level)

    # Ensure aim_runtime logs propagate to root handler
    aim_logger.propagate = True
