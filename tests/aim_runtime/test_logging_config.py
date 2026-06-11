# Copyright © Advanced Micro Devices, Inc., or its affiliates.
#
# SPDX-License-Identifier: MIT

"""
Test module for AIM runtime logging configuration.

This module tests the logging configuration functionality including
proper log level setting for root and aim_runtime loggers.
"""

import logging

from aim_runtime.logging_config import configure_logging


class TestLoggingConfiguration:
    """Test cases for logging configuration functionality."""

    def test_configure_logging_defaults(self):
        """Test that configure_logging uses default levels (WARNING for root, INFO for aim)."""
        configure_logging()

        # Check that root logger is set to WARNING level (default)
        root_logger = logging.getLogger()
        assert root_logger.level == logging.WARNING

        # Check that aim_runtime logger is set to INFO level (default)
        aim_logger = logging.getLogger("aim_runtime")
        assert aim_logger.level == logging.INFO

    def test_configure_logging_debug_both(self):
        """Test that both loggers can be set to DEBUG."""
        configure_logging(root_log_level="DEBUG", aim_log_level="DEBUG")

        # Check that root logger is set to DEBUG level
        root_logger = logging.getLogger()
        assert root_logger.level == logging.DEBUG

        # Check that aim_runtime logger is also DEBUG
        aim_logger = logging.getLogger("aim_runtime")
        assert aim_logger.level == logging.DEBUG

        # Test that debug messages would be logged
        logger = logging.getLogger(__name__)
        assert logger.isEnabledFor(logging.DEBUG)

    def test_configure_logging_separate_levels(self):
        """Test that root and aim_runtime loggers can have separate levels."""
        configure_logging(root_log_level="ERROR", aim_log_level="DEBUG")

        root_logger = logging.getLogger()
        assert root_logger.level == logging.ERROR

        aim_logger = logging.getLogger("aim_runtime")
        assert aim_logger.level == logging.DEBUG

        # Test that aim_runtime logger can log DEBUG
        assert aim_logger.isEnabledFor(logging.DEBUG)

        # Test that non-aim_runtime loggers follow root level
        other_logger = logging.getLogger("some_other_package")
        assert not other_logger.isEnabledFor(logging.WARNING)
        assert other_logger.isEnabledFor(logging.ERROR)

    def test_configure_logging_custom_levels(self):
        """Test that custom log levels are properly applied."""
        configure_logging(root_log_level="CRITICAL", aim_log_level="WARNING")

        root_logger = logging.getLogger()
        assert root_logger.level == logging.CRITICAL

        aim_logger = logging.getLogger("aim_runtime")
        assert aim_logger.level == logging.WARNING

    def test_multiple_loggers_inherit_configuration(self):
        """Test that aim_runtime child loggers inherit aim_runtime configuration."""
        configure_logging(root_log_level="WARNING", aim_log_level="DEBUG")

        logger1 = logging.getLogger("aim_runtime.module1")
        logger2 = logging.getLogger("aim_runtime.module2")
        logger3 = logging.getLogger("aim_runtime.module3")

        # All aim_runtime children should be able to log DEBUG
        assert logger1.isEnabledFor(logging.DEBUG)
        assert logger2.isEnabledFor(logging.DEBUG)
        assert logger3.isEnabledFor(logging.DEBUG)

    def test_reconfigure_logging(self):
        """Test that logging can be reconfigured with different settings."""
        # First configuration
        configure_logging(root_log_level="WARNING", aim_log_level="INFO")
        assert logging.getLogger().level == logging.WARNING
        assert logging.getLogger("aim_runtime").level == logging.INFO

        # Reconfigure to debug both
        configure_logging(root_log_level="DEBUG", aim_log_level="DEBUG")
        assert logging.getLogger().level == logging.DEBUG
        assert logging.getLogger("aim_runtime").level == logging.DEBUG

        # Reconfigure with custom levels
        configure_logging(root_log_level="ERROR", aim_log_level="WARNING")
        assert logging.getLogger().level == logging.ERROR
        assert logging.getLogger("aim_runtime").level == logging.WARNING

    def test_logging_format(self):
        """Test that logging format is configured correctly."""
        configure_logging(root_log_level="DEBUG", aim_log_level="DEBUG")

        # Get the root logger and check its handlers
        root_logger = logging.getLogger()
        assert len(root_logger.handlers) > 0

        # Check that the handler has the expected format
        handler = root_logger.handlers[0]
        assert handler.formatter is not None
        format_string = handler.formatter._fmt

        # Verify the format contains expected components
        assert "%(asctime)s" in format_string
        assert "%(name)s" in format_string
        assert "%(levelname)s" in format_string
        assert "%(message)s" in format_string

    def test_aim_runtime_logger_propagates(self):
        """Test that aim_runtime logger propagates to root handler."""
        configure_logging(root_log_level="WARNING", aim_log_level="INFO")

        aim_logger = logging.getLogger("aim_runtime")
        assert aim_logger.propagate is True
