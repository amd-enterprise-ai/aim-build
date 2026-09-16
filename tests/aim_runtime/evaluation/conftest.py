# Copyright © Advanced Micro Devices, Inc., or its affiliates.
#
# SPDX-License-Identifier: MIT

"""
Shared fixtures for the evaluation tests.

Every environment variable the evaluation package reads is an override of a
default, and these tests assert those defaults. An AIM image exports
``AIM_EVAL_BACKEND_COMMAND`` and the accuracy job exports the output-path
variables, so a suite that inherited the ambient environment would assert the
defaults everywhere except the two places the code actually ships.
"""

import pytest

from aim_runtime.evaluation.config import (
    BACKEND_COMMAND_ENV,
    LM_EVAL_PREFERRED_VARIANT_ENV,
    RESULTS_CSV_FILE_ENV,
    RESULTS_JSON_FILE_ENV,
)

#: Overrides cleared before each test. A test that wants one sets it itself.
EVALUATION_ENV_OVERRIDES = (
    BACKEND_COMMAND_ENV,
    LM_EVAL_PREFERRED_VARIANT_ENV,
    RESULTS_JSON_FILE_ENV,
    RESULTS_CSV_FILE_ENV,
)


@pytest.fixture(autouse=True)
def unset_evaluation_overrides(monkeypatch):
    """Run every test against the shipped defaults, wherever pytest was started."""
    for name in EVALUATION_ENV_OVERRIDES:
        monkeypatch.delenv(name, raising=False)
