# Copyright © Advanced Micro Devices, Inc., or its affiliates.
#
# SPDX-License-Identifier: MIT

"""Integration tests for GPUDetector with real hardware (sysfs + amdsmi)."""

from pathlib import Path

import pytest

from aim_runtime.gpu_detector import DRM_BASE, GPUDetector


@pytest.mark.integration
def test_sysfs_interface_compatibility():
    """Test that sysfs GPU attributes exist and are readable on real hardware."""
    if not DRM_BASE.exists():
        pytest.skip("sysfs DRM base not found — not running on Linux with amdgpu driver")

    card_dirs = sorted(DRM_BASE.glob("card[0-9]*/device"), key=lambda p: p.parent.name)
    gpu_cards = [d for d in card_dirs if (d / "mem_info_vram_total").exists()]

    if not gpu_cards:
        pytest.skip("No AMD GPU card directories found in sysfs")

    device_dir = gpu_cards[0]

    # Required attributes for GPU detection
    required_files = ["device", "mem_info_vram_total", "mem_info_vram_used"]
    for fname in required_files:
        fpath = device_dir / fname
        assert fpath.exists(), f"Missing required sysfs attribute: {fpath}"
        content = fpath.read_text().strip()
        assert len(content) > 0, f"Empty sysfs attribute: {fpath}"

    # Utilization files (may not always be present but should be on discrete GPUs)
    for fname in ["gpu_busy_percent", "mem_busy_percent"]:
        fpath = device_dir / fname
        if fpath.exists():
            content = fpath.read_text().strip()
            val = int(content)
            assert 0 <= val <= 100, f"Unexpected utilization value {val} from {fpath}"


@pytest.mark.integration
def test_gpu_detector_with_real_hardware():
    """Test GPUDetector produces valid results on real AMD GPU hardware."""
    detector = GPUDetector()

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
        assert detector.detection_method in ("sysfs", "amdsmi")

        for gpu in detector.gpus:
            assert gpu.device_id.startswith("0x")
            assert gpu.vram_total > 0
            assert gpu.vram_used >= 0
            assert gpu.vram_free >= 0
            assert 0 <= gpu.gfx_utilization <= 100
            assert 0 <= gpu.mem_utilization <= 100
    else:
        assert detector.device_ids is None
        assert detector.total_free_vram is None
        assert detector.gpu_models is None


@pytest.mark.integration
def test_amdsmi_fallback_with_real_hardware():
    """Test that the amdsmi fallback path produces valid results on real hardware."""
    try:
        import amdsmi  # noqa: F401
    except ImportError:
        pytest.skip("amdsmi library not available")

    detector = GPUDetector(drm_base=Path("/nonexistent"))

    if not detector.has_gpus:
        pytest.skip("amdsmi detected no GPUs (no hardware or driver issue)")

    assert detector.detection_method == "amdsmi"
    assert detector.gpu_count > 0

    for gpu in detector.gpus:
        assert gpu.device_id.startswith("0x")
        assert gpu.vram_total > 0
        assert gpu.vram_used >= 0
        assert gpu.vram_free >= 0
        assert 0 <= gpu.gfx_utilization <= 100
        assert 0 <= gpu.mem_utilization <= 100


if __name__ == "__main__":
    test_sysfs_interface_compatibility()
    test_gpu_detector_with_real_hardware()
    test_amdsmi_fallback_with_real_hardware()
