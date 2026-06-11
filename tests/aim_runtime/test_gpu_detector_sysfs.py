# Copyright © Advanced Micro Devices, Inc., or its affiliates.
#
# SPDX-License-Identifier: MIT

"""Unit tests for sysfs GPU detection backend."""

from pathlib import Path

from aim_runtime.gpu_detector import GPUDetector


def test_sysfs_single_gpu(tmp_path):
    """Test sysfs detection with a single GPU."""
    drm = tmp_path / "drm"
    card0_device = drm / "card0" / "device"
    card0_device.mkdir(parents=True)

    (card0_device / "device").write_text("0x74a1\n")
    (card0_device / "mem_info_vram_total").write_text(str(192 * 1024**3) + "\n")  # 192 GB
    (card0_device / "mem_info_vram_used").write_text(str(50 * 1024**3) + "\n")  # 50 GB
    (card0_device / "gpu_busy_percent").write_text("5\n")
    (card0_device / "mem_busy_percent").write_text("3\n")

    # Use nonexistent dev_dri to avoid interference from host /dev/dri
    detector = GPUDetector(drm_base=drm, dev_dri=tmp_path / "no_dri")

    assert detector.has_gpus is True
    assert detector.gpu_count == 1
    assert detector.device_ids == ["0x74a1"]
    assert detector.gpu_models == ["MI300X"]
    assert detector.detection_method == "sysfs"

    gpu = detector.gpus[0]
    assert gpu.device_id == "0x74a1"
    assert gpu.model == "MI300X"
    assert gpu.vram_total == 192 * 1024  # MB
    assert gpu.vram_used == 50 * 1024  # MB
    assert gpu.vram_free == 142 * 1024  # MB
    assert gpu.gfx_utilization == 5.0
    assert gpu.mem_utilization == 3.0


def test_sysfs_multiple_gpus_numeric_sorting(tmp_path):
    """Test sysfs detection with multiple GPUs and numeric card sorting."""
    drm = tmp_path / "drm"
    dev_dri = tmp_path / "dev_dri"
    dev_dri.mkdir()

    # Create card0, card1, card2, card10 to test numeric sorting
    # Use different device IDs to distinguish them
    device_ids = {0: "0x74a1", 1: "0x7410", 2: "0x7430", 10: "0x7420"}

    for card_num in [0, 1, 10, 2]:  # Create in non-sorted order
        card_device = drm / f"card{card_num}" / "device"
        card_device.mkdir(parents=True)
        (card_device / "device").write_text(f"{device_ids[card_num]}\n")
        (card_device / "mem_info_vram_total").write_text(str(100 * 1024**3) + "\n")
        (card_device / "mem_info_vram_used").write_text("0\n")
        (card_device / "gpu_busy_percent").write_text("0\n")
        (card_device / "mem_busy_percent").write_text("0\n")

        # Also create card in /dev/dri so it's not filtered
        (dev_dri / f"card{card_num}").touch()

    detector = GPUDetector(drm_base=drm, dev_dri=dev_dri)

    assert detector.gpu_count == 4
    # Should be sorted numerically: card0, card1, card2, card10 (not card0, card1, card10, card2)
    assert detector.device_ids == ["0x74a1", "0x7410", "0x7430", "0x7420"]


def test_sysfs_dev_dri_filtering(tmp_path):
    """Test /dev/dri filtering for container GPU assignment."""
    drm = tmp_path / "drm"
    dev_dri = tmp_path / "dev_dri"
    dev_dri.mkdir()

    # Create 4 GPUs in sysfs
    for i in range(4):
        card_device = drm / f"card{i}" / "device"
        card_device.mkdir(parents=True)
        (card_device / "device").write_text("0x74a1\n")
        (card_device / "mem_info_vram_total").write_text(str(100 * 1024**3) + "\n")
        (card_device / "mem_info_vram_used").write_text("0\n")
        (card_device / "gpu_busy_percent").write_text("0\n")
        (card_device / "mem_busy_percent").write_text("0\n")

    # But only card1 and card3 are visible in /dev/dri (container assignment)
    (dev_dri / "card1").touch()
    (dev_dri / "card3").touch()
    (dev_dri / "renderD128").touch()

    detector = GPUDetector(drm_base=drm, dev_dri=dev_dri)

    # Should only see card1 and card3
    assert detector.gpu_count == 2
    assert detector.detection_method == "sysfs"


def test_sysfs_dev_dri_only_renderd_nodes(tmp_path, monkeypatch):
    """Test /dev/dri with only renderD* nodes falls back to amdsmi."""
    drm = tmp_path / "drm"
    dev_dri = tmp_path / "dev_dri"
    dev_dri.mkdir()

    # Create GPU in sysfs
    card_device = drm / "card0" / "device"
    card_device.mkdir(parents=True)
    (card_device / "device").write_text("0x74a1\n")
    (card_device / "mem_info_vram_total").write_text(str(100 * 1024**3) + "\n")
    (card_device / "mem_info_vram_used").write_text("0\n")

    # Only renderD nodes, no card nodes — visible_cards returns empty set,
    # sysfs can't map renderD -> card so falls back to amdsmi
    (dev_dri / "renderD128").touch()
    (dev_dri / "renderD129").touch()

    monkeypatch.setattr(GPUDetector, "_get_gpu_info_amdsmi", lambda self: None)
    detector = GPUDetector(drm_base=drm, dev_dri=dev_dri)

    # sysfs finds no visible cards -> amdsmi disabled -> no GPUs
    assert detector.has_gpus is False


def test_sysfs_non_gpu_filtered_by_missing_vram(tmp_path):
    """Test that non-GPU devices without VRAM are filtered out."""
    drm = tmp_path / "drm"

    # Valid AMD GPU with VRAM
    card0_device = drm / "card0" / "device"
    card0_device.mkdir(parents=True)
    (card0_device / "device").write_text("0x74a1\n")
    (card0_device / "mem_info_vram_total").write_text(str(100 * 1024**3) + "\n")
    (card0_device / "mem_info_vram_used").write_text("0\n")

    # Non-GPU device (IOMMU) — has a valid PCI ID but no VRAM
    card1_device = drm / "card1" / "device"
    card1_device.mkdir(parents=True)
    (card1_device / "device").write_text("0x2000\n")
    # No mem_info_vram_total file -> skipped

    detector = GPUDetector(drm_base=drm, dev_dri=tmp_path / "no_dri")

    assert detector.gpu_count == 1
    assert detector.device_ids == ["0x74a1"]


def test_sysfs_vram_total_zero_skipped(tmp_path):
    """Test that GPUs with vram_total=0 are skipped."""
    drm = tmp_path / "drm"
    card0_device = drm / "card0" / "device"
    card0_device.mkdir(parents=True)

    (card0_device / "device").write_text("0x74a1\n")
    (card0_device / "mem_info_vram_total").write_text("0\n")  # Invalid
    (card0_device / "mem_info_vram_used").write_text("0\n")

    detector = GPUDetector(drm_base=drm, dev_dri=tmp_path / "no_dri")

    # GPU with zero VRAM should be skipped
    assert detector.has_gpus is False
    assert detector.gpu_count == 0


def test_sysfs_missing_vram_total_file(tmp_path):
    """Test that GPUs without mem_info_vram_total are skipped."""
    drm = tmp_path / "drm"
    card0_device = drm / "card0" / "device"
    card0_device.mkdir(parents=True)

    (card0_device / "device").write_text("0x74a1\n")
    # mem_info_vram_total file missing
    (card0_device / "mem_info_vram_used").write_text("0\n")

    detector = GPUDetector(drm_base=drm, dev_dri=tmp_path / "no_dri")

    assert detector.has_gpus is False
    assert detector.gpu_count == 0


def test_sysfs_missing_device_file(tmp_path):
    """Test that cards without device file are skipped."""
    drm = tmp_path / "drm"
    card0_device = drm / "card0" / "device"
    card0_device.mkdir(parents=True)

    # device file missing
    (card0_device / "mem_info_vram_total").write_text(str(100 * 1024**3) + "\n")
    (card0_device / "mem_info_vram_used").write_text("0\n")

    detector = GPUDetector(drm_base=drm, dev_dri=tmp_path / "no_dri")

    assert detector.has_gpus is False


def test_sysfs_optional_utilization_files(tmp_path):
    """Test that missing utilization files default to 0."""
    drm = tmp_path / "drm"
    card0_device = drm / "card0" / "device"
    card0_device.mkdir(parents=True)

    (card0_device / "device").write_text("0x74a1\n")
    (card0_device / "mem_info_vram_total").write_text(str(100 * 1024**3) + "\n")
    (card0_device / "mem_info_vram_used").write_text(str(10 * 1024**3) + "\n")
    # gpu_busy_percent and mem_busy_percent missing

    detector = GPUDetector(drm_base=drm, dev_dri=tmp_path / "no_dri")

    assert detector.has_gpus is True
    gpu = detector.gpus[0]
    assert gpu.gfx_utilization == 0.0
    assert gpu.mem_utilization == 0.0


def test_sysfs_nonexistent_drm_base(tmp_path, monkeypatch):
    """Test that nonexistent DRM base returns None when both backends fail."""
    monkeypatch.setattr(GPUDetector, "_get_gpu_info_amdsmi", lambda self: None)
    detector = GPUDetector(drm_base=tmp_path / "nonexistent", dev_dri=tmp_path / "no_dri")

    assert detector.has_gpus is False


def test_sysfs_dev_dri_permission_error(tmp_path, monkeypatch):
    """Test that PermissionError on /dev/dri disables filtering."""
    drm = tmp_path / "drm"
    dev_dri = tmp_path / "dev_dri"
    dev_dri.mkdir()

    # Create GPU in sysfs
    card_device = drm / "card0" / "device"
    card_device.mkdir(parents=True)
    (card_device / "device").write_text("0x74a1\n")
    (card_device / "mem_info_vram_total").write_text(str(100 * 1024**3) + "\n")
    (card_device / "mem_info_vram_used").write_text("0\n")

    # Monkeypatch iterdir() to raise PermissionError
    original_iterdir = Path.iterdir

    def fake_iterdir(self):
        if self == dev_dri:
            raise PermissionError("Permission denied")
        return original_iterdir(self)

    monkeypatch.setattr(Path, "iterdir", fake_iterdir)

    detector = GPUDetector(drm_base=drm, dev_dri=dev_dri)

    # Should detect GPU without filtering (permission error disables filter)
    assert detector.gpu_count == 1
    assert detector.detection_method == "sysfs"


def test_sysfs_device_id_normalization(tmp_path):
    """Test device ID normalization (handles with/without 0x prefix)."""
    drm = tmp_path / "drm"
    dev_dri = tmp_path / "dev_dri"
    dev_dri.mkdir()

    # card0: device ID without 0x prefix
    card0_device = drm / "card0" / "device"
    card0_device.mkdir(parents=True)
    (card0_device / "device").write_text("74a1\n")  # No 0x prefix
    (card0_device / "mem_info_vram_total").write_text(str(100 * 1024**3) + "\n")
    (card0_device / "mem_info_vram_used").write_text("0\n")
    (dev_dri / "card0").touch()

    # card1: device ID with 0x prefix (use 0x7xxx range for AMD GPUs)
    card1_device = drm / "card1" / "device"
    card1_device.mkdir(parents=True)
    (card1_device / "device").write_text("0x7499\n")  # With 0x prefix, valid AMD GPU ID
    (card1_device / "mem_info_vram_total").write_text(str(100 * 1024**3) + "\n")
    (card1_device / "mem_info_vram_used").write_text("0\n")
    (dev_dri / "card1").touch()

    detector = GPUDetector(drm_base=drm, dev_dri=dev_dri)

    assert detector.gpu_count == 2
    # Both should be normalized to 0x format
    assert detector.device_ids == ["0x74a1", "0x7499"]
