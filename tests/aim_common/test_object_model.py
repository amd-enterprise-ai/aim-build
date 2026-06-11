# Copyright © Advanced Micro Devices, Inc., or its affiliates.
#
# SPDX-License-Identifier: MIT
import pytest

from aim_common import Engine, GPUModel, Metric, Precision, ProfileCapabilities, ProfileMetadata, ProfileType


def test_gpu_model_from_string_happy_path():
    """Test GPUModel.from_string method for various inputs."""
    assert GPUModel.from_string("MI300X") == GPUModel.MI300X
    assert GPUModel.from_string("mi300x") == GPUModel.MI300X
    assert GPUModel.from_string("Mi300X") == GPUModel.MI300X
    assert GPUModel.from_string("0x740c") == GPUModel.MI250X
    assert GPUModel.from_string("0x74a1") == GPUModel.MI300X
    assert GPUModel.from_string("0x7551") == GPUModel.R9700
    assert GPUModel.from_string("R9700") == GPUModel.R9700
    assert GPUModel.from_string("r9700") == GPUModel.R9700
    assert GPUModel.from_string("0x7448") == GPUModel.W7900
    assert GPUModel.from_string("W7900") == GPUModel.W7900
    assert GPUModel.from_string("w7900") == GPUModel.W7900
    assert GPUModel.from_string(None) is None
    assert GPUModel.from_string("0x740C") == GPUModel.MI250X
    # EPYC CPU models
    assert GPUModel.from_string("EPYC_9965") == GPUModel.EPYC_9965
    assert GPUModel.from_string("9965") == GPUModel.EPYC_9965
    assert GPUModel.from_string("EPYC_ZEN5") == GPUModel.EPYC_ZEN5
    assert GPUModel.from_string("9J14") == GPUModel.EPYC_ZEN4


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


def test_profile_metadata_capabilities_default_all_false():
    profile = ProfileMetadata(
        engine=Engine.VLLM,
        accelerator_model=GPUModel.MI300X,
        precision=Precision.FP16,
        accelerator_count=1,
        metric=Metric.LATENCY,
        manual_selection_only=False,
        type=ProfileType.GENERAL,
    )

    assert profile.capabilities.tool_calling is False
    assert profile.capabilities.structured_outputs is False
    assert profile.capabilities.reasoning is False


def test_profile_metadata_capabilities_round_trip():
    profile = ProfileMetadata(
        engine=Engine.VLLM,
        accelerator_model=GPUModel.MI300X,
        precision=Precision.FP16,
        accelerator_count=1,
        metric=Metric.LATENCY,
        manual_selection_only=False,
        type=ProfileType.GENERAL,
        capabilities=ProfileCapabilities(tool_calling=True, structured_outputs=True, reasoning=True),
    )

    restored = ProfileMetadata.from_dict(profile.to_dict())

    assert restored.capabilities.tool_calling is True
    assert restored.capabilities.structured_outputs is True
    assert restored.capabilities.reasoning is True


def test_profile_metadata_capabilities_partial_round_trip():
    restored = ProfileMetadata.from_dict(
        {
            "engine": "vllm",
            "accelerator_model": "MI300X",
            "precision": "fp16",
            "accelerator_count": 1,
            "metric": "latency",
            "manual_selection_only": False,
            "type": "general",
            "capabilities": {"reasoning": True},
        }
    )

    assert restored.capabilities.reasoning is True
    assert restored.capabilities.tool_calling is False
    assert restored.capabilities.structured_outputs is False
