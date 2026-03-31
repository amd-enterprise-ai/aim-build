# Copyright © Advanced Micro Devices, Inc., or its affiliates.
#
# SPDX-License-Identifier: MIT
import pytest

from aim_common import GPUModel


def test_gpu_model_from_string_happy_path():
    """Test GPUModel.from_string method for various inputs."""
    assert GPUModel.from_string("MI300X") == GPUModel.MI300X
    assert GPUModel.from_string("mi300x") == GPUModel.MI300X
    assert GPUModel.from_string("Mi300X") == GPUModel.MI300X
    assert GPUModel.from_string("0x740c") == GPUModel.MI250X
    assert GPUModel.from_string("0x74a1") == GPUModel.MI300X
    assert GPUModel.from_string(None) is None
    assert GPUModel.from_string("0x740C") == GPUModel.MI250X


def test_gpu_model_from_string_raise_value_error():
    """Test GPUModel.from_string method for various inputs."""
    with pytest.raises(ValueError):
        GPUModel.from_string("UNKNOWN_MODEL")

    with pytest.raises(ValueError):
        GPUModel.from_string("0x1234")


def test_gpu_model_from_string_with_default_happy_path():
    """Test GPUModel.from_string_with_default method for various inputs."""
    assert GPUModel.from_string_with_default("UNKNOWN_MODEL", GPUModel.MI100) == GPUModel.MI100
    assert GPUModel.from_string_with_default("0x1234", GPUModel.MI100) == GPUModel.MI100
    assert GPUModel.from_string_with_default("0x1234") is None
    assert GPUModel.from_string_with_default("Mi300X", GPUModel.MI355X) == GPUModel.MI300X
    assert GPUModel.from_string_with_default("0x740c", GPUModel.MI355X) == GPUModel.MI250X
    assert GPUModel.from_string_with_default(None) is None
