# Copyright © Advanced Micro Devices, Inc., or its affiliates.
#
# SPDX-License-Identifier: MIT

"""Integration tests for GPUDetector with real amdsmi library."""

import pytest

from aim_runtime.gpu_detector import GPUDetector


@pytest.mark.integration
def test_amdsmi_library_interface_compatibility():
    """Test that amdsmi library interface remains compatible."""
    try:
        import amdsmi

        # Test library can be imported and initialized
        amdsmi.amdsmi_init()

        # Test expected functions exist with correct signatures
        required_functions = [
            "amdsmi_get_processor_handles",
            "amdsmi_get_gpu_asic_info",
            "amdsmi_get_gpu_vram_usage",
            "amdsmi_get_gpu_activity",
            "amdsmi_shut_down",
        ]

        for func_name in required_functions:
            assert hasattr(amdsmi, func_name), f"Missing required function: {func_name}"

        # Test graceful handling when no GPUs present
        handles = amdsmi.amdsmi_get_processor_handles()
        assert isinstance(handles, list), "Expected list return type"

        # If GPUs are present, test data structure format
        if handles:
            # Test first GPU info structure
            info = amdsmi.amdsmi_get_gpu_asic_info(handles[0])
            assert isinstance(info, dict), "Expected dict return type"
            assert "device_id" in info, "Missing device_id field"

            vram = amdsmi.amdsmi_get_gpu_vram_usage(handles[0])
            assert isinstance(vram, dict), "Expected dict return type"
            assert "vram_total" in vram, "Missing vram_total field"
            assert "vram_used" in vram, "Missing vram_used field"

            activity = amdsmi.amdsmi_get_gpu_activity(handles[0])
            assert isinstance(activity, dict), "Expected dict return type"
            assert "gfx_activity" in activity, "Missing gfx_activity field"
            assert "umc_activity" in activity, "Missing umc_activity field"

        amdsmi.amdsmi_shut_down()

    except ImportError:
        pytest.skip("amdsmi library not available")
    except Exception as e:
        pytest.fail(f"amdsmi library interface incompatible: {e}")


@pytest.mark.integration
def test_gpu_detector_with_real_environment():
    """Test GPUDetector with real amdsmi library."""
    detector = GPUDetector()

    # Should not crash regardless of GPU presence
    assert isinstance(detector.has_gpus, bool)
    assert isinstance(detector.gpu_count, int)
    assert detector.gpu_count >= 0

    if detector.has_gpus:
        assert detector.device_ids is not None
        assert len(detector.device_ids) == detector.gpu_count
        assert detector.total_free_vram is not None
        assert detector.total_free_vram >= 0
        assert detector.gpu_models is not None
        assert len(detector.gpu_models) == detector.gpu_count
    else:
        assert detector.device_ids is None
        assert detector.total_free_vram is None
        assert detector.gpu_models is None


if __name__ == "__main__":
    # Run integration tests manually for debugging
    test_amdsmi_library_interface_compatibility()
    test_gpu_detector_with_real_environment()
