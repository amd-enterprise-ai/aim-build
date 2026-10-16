# Copyright © Advanced Micro Devices, Inc., or its affiliates.
#
# SPDX-License-Identifier: MIT

"""Integration tests for CPU detection against the container's real environment."""

import os

import pytest

from aim_runtime.accelerator_detector import AcceleratorDetector
from aim_runtime.object_model import AcceleratorFamily, AcceleratorType

pytestmark = pytest.mark.skipif(
    os.environ.get("AIM_ACCELERATOR_TYPE") != "cpu",
    reason="CPU detector integration checks apply only to CPU images",
)


@pytest.mark.integration
def test_cpu_accelerator_detector_with_real_hardware():
    """CPU base images must detect available CPU resources without GPU hardware."""
    accelerator_family = AcceleratorFamily(os.environ["AIM_ACCELERATOR_FAMILY"])

    result = AcceleratorDetector().detect(
        accelerator_type=AcceleratorType.CPU,
        accelerator_family=accelerator_family,
    )

    assert result.accelerator_type == AcceleratorType.CPU
    assert result.gpu_info == []
    assert result.cpu_info is not None
    assert result.cpu_cores >= 1
    assert result.accelerator_count >= result.cpu_cores
    assert result.cpu_info.available_cores == result.cpu_cores
