# Copyright © Advanced Micro Devices, Inc., or its affiliates.
#
# SPDX-License-Identifier: MIT

import json
import logging
import re
from dataclasses import dataclass
from functools import cached_property
from pathlib import Path
from typing import Mapping, Optional

from .object_model import AcceleratorModel

logger = logging.getLogger(__name__)

# GPU model to GFX architecture mapping for AMD GPUs
# Reference: https://rocm.docs.amd.com/en/latest/reference/gpu-arch-specs.html
# Also mirrored (lowercase, AITER-only subset) in docker/prebuild_aiter_kernels.py
GPU_TO_GFX_ARCH = {
    # CDNA1 (gfx908)
    "MI100": "gfx908",
    # CDNA2 (gfx90a)
    "MI210": "gfx90a",
    "MI250X": "gfx90a",
    # CDNA3 (gfx942)
    "MI300A": "gfx942",
    "MI300X": "gfx942",
    "MI308X": "gfx942",
    "MI325X": "gfx942",
    # CDNA4 (gfx950)
    "MI350X": "gfx950",
    "MI350P": "gfx950",
    "MI355X": "gfx950",
    # RDNA3 (gfx1100)
    "W7900": "gfx1100",
    "W7800": "gfx1100",
    # RDNA4 (gfx1201)
    "R9700": "gfx1201",
}


def get_gfx_arch(gpu_model: str) -> Optional[str]:
    """Get GFX architecture for a GPU model.

    Args:
        gpu_model: GPU model name (e.g., "MI300X", "MI325X")

    Returns:
        GFX architecture string (e.g., "gfx942", "gfx950") or None if unknown
    """
    return GPU_TO_GFX_ARCH.get(gpu_model)


DRM_BASE = Path("/sys/class/drm")
DEV_DRI = Path("/dev/dri")
# Matches any 16-bit PCI device ID; non-GPU devices (IOMMU etc.) are filtered
# by requiring mem_info_vram_total to exist and be > 0.
GPU_DEVICE_ID_PATTERN = re.compile(r"^0x[0-9a-fA-F]{4}$")


@dataclass
class GPUInfo:
    """Represents information about a single GPU."""

    device_id: str
    model: Optional[AcceleratorModel]
    vram_total: int  # in MB
    vram_used: int  # in MB
    gfx_utilization: float  # percentage
    mem_utilization: float  # percentage

    @property
    def vram_free(self) -> int:
        """Get free VRAM in MB."""
        return max(0, self.vram_total - self.vram_used)

    @property
    def is_idle(self) -> bool:
        """Check if GPU is idle (no graphics utilization and memory is free)."""
        # A GPU is idle if it has no graphics activity AND memory is considered free
        return self.gfx_utilization == 0 and self.mem_utilization <= 5

    def to_dict(self) -> dict:
        """Convert to dictionary for logging/serialization."""
        return {
            "device_id": self.device_id,
            "model": self.model.value if self.model else None,
            "vram_total": self.vram_total,
            "vram_used": self.vram_used,
            "vram_free": self.vram_free,
            "gfx_utilization": self.gfx_utilization,
            "mem_utilization": self.mem_utilization,
            "is_idle": self.is_idle,
        }


class GPUDetector:
    """Detects AMD GPUs using a two-tier strategy: sysfs primary, amdsmi fallback.

    Primary (sysfs): Reads /sys/class/drm/card*/device/ attributes directly.
    Zero dependency, works on any Linux system with the amdgpu kernel driver.
    Filters to container-assigned GPUs by cross-referencing /dev/dri/ card nodes.

    Fallback (amdsmi): Uses the AMD SMI Python library when sysfs detection fails
    (e.g., missing sysfs attributes, non-standard kernel). Handles container GPU
    visibility natively via the ROCm runtime.
    """

    def __init__(self, drm_base: Optional[Path] = None, dev_dri: Optional[Path] = None):
        self._gpus: Optional[list[GPUInfo]] = None
        self._detected = False
        self._detection_method: Optional[str] = None
        self._drm_base = drm_base or DRM_BASE
        self._dev_dri = dev_dri or DEV_DRI

    def group_by_device(self) -> Mapping[str, list[GPUInfo]]:
        """Group detected GPUs by their normalized device id."""
        groups: dict[str, list[GPUInfo]] = {}
        if not self.gpus:
            return groups
        for gpu in self.gpus:
            key = self._normalize_device_id(gpu.device_id)
            groups.setdefault(key, []).append(gpu)
        return groups

    @cached_property
    def is_homogeneous(self) -> bool:
        """Return True if all detected GPUs share the same device id."""
        groups = self.group_by_device()
        if not groups:
            return True
        return len(groups) == 1

    @cached_property
    def gpus(self) -> Optional[list[GPUInfo]]:
        """Get list of detected GPUs with full information."""
        if not self._detected:
            self._detect_gpus()
        return self._gpus

    @cached_property
    def device_ids(self) -> Optional[list[str]]:
        """Get list of detected GPU device IDs."""
        if not self.gpus:
            return None
        return [gpu.device_id for gpu in self.gpus]

    @cached_property
    def total_free_vram(self) -> Optional[int]:
        """Get total free VRAM across all GPUs in MB."""
        if not self.gpus:
            return None
        return sum(gpu.vram_free for gpu in self.gpus)

    @cached_property
    def gpu_count(self) -> int:
        """Get number of detected GPUs."""
        return len(self.gpus) if self.gpus else 0

    @cached_property
    def has_gpus(self) -> bool:
        """Check if any GPUs were detected."""
        return self.gpu_count > 0

    @cached_property
    def all_gpus_idle(self) -> bool:
        """Check if all GPUs are idle."""
        if not self.gpus:
            return True
        return all(gpu.is_idle for gpu in self.gpus)

    @property
    def detection_method(self) -> Optional[str]:
        """Return which detection method was used: 'sysfs', 'amdsmi', or None."""
        if not self._detected:
            self._detect_gpus()
        return self._detection_method

    def get_gpu_model(self, device_id: str) -> Optional[AcceleratorModel]:
        """Get GPU model from device ID, or None if unrecognized."""
        norm = self._normalize_device_id(device_id)
        return AcceleratorModel.from_string_with_default(norm)

    @cached_property
    def gpu_models(self) -> Optional[list[Optional[AcceleratorModel]]]:
        """Get list of detected GPU model names."""
        if not self.gpus:
            return None
        return [gpu.model for gpu in self.gpus]

    def get_gpu_info(self) -> Optional[list[dict]]:
        """
        Get detailed GPU information including device ID and model name.

        Returns:
            List of dictionaries with GPU information, or None if no GPUs
        """
        if not self.gpus:
            return None
        return [gpu.to_dict() for gpu in self.gpus]

    def _normalize_device_id(self, device_id: str) -> str:
        """
        Normalize device ID to ensure consistent "0x" prefix format.

        Args:
            device_id: Raw device ID that may or may not have "0x" prefix

        Returns:
            Normalized device ID with "0x" prefix
        """
        device_id_str = str(device_id).strip()
        if device_id_str.startswith("0x"):
            return device_id_str
        else:
            return f"0x{device_id_str}"

    def _log_gpu_health(self, gpus: list[GPUInfo]) -> None:
        """Log GPU health status and warn about non-idle GPUs."""
        for gpu in gpus:
            logger.info("GPU %s: %s", gpu.device_id, json.dumps(gpu.to_dict(), indent=2))

            # Consider it okay if GPU is idle or effectively free by memory criteria
            if not gpu.is_idle:
                logger.warning(
                    "GPU %s is not idle - GFX utilization: %s%%, MEM utilization: %s%%",
                    gpu.device_id,
                    gpu.gfx_utilization,
                    gpu.mem_utilization,
                )

    def _detect_gpus(self) -> None:
        """Perform GPU detection: sysfs first, amdsmi fallback."""
        if self._detected:
            return

        gpus = self._get_gpu_info_sysfs()
        if gpus is not None:
            self._detection_method = "sysfs"
        else:
            logger.info("sysfs detection failed or found no GPUs, trying amdsmi fallback")
            gpus = self._get_gpu_info_amdsmi()
            if gpus is not None:
                self._detection_method = "amdsmi"

        if gpus is None:
            logger.warning("No AMD GPUs detected (tried sysfs and amdsmi)")
            self._gpus = None
        else:
            self._gpus = gpus
            logger.info("Detected %d AMD GPU(s) via %s", len(self._gpus), self._detection_method)
            self._log_gpu_health(self._gpus)

        self._detected = True

    # ── sysfs backend ──────────────────────────────────────────────────────

    @staticmethod
    def _card_index(path: Path) -> int:
        """Sort key: extract the numeric index from a card directory path."""
        m = re.search(r"card(\d+)", path.parent.name)
        return int(m.group(1)) if m else 0

    def _read_sysfs(self, path: Path) -> Optional[str]:
        """Read a sysfs attribute file, returning None on any error."""
        try:
            return path.read_text().strip()
        except (OSError, IOError):
            return None

    def _read_sysfs_int(self, path: Path, default: int = 0) -> int:
        """Read an integer value from a sysfs attribute file."""
        val = self._read_sysfs(path)
        if val is None:
            return default
        try:
            return int(val)
        except ValueError:
            return default

    def _get_visible_cards(self) -> Optional[set[str]]:
        """Return the set of card names visible in /dev/dri/.

        In Kubernetes, the AMD GPU device plugin mounts only the allocated
        card and renderD nodes into the container.  When /dev/dri/ exists we
        use it as the authoritative list of assigned GPUs so that sysfs
        (which always exposes every physical GPU on the node) is filtered
        down to only the GPUs this container owns.

        Returns:
            None  – /dev/dri/ does not exist (bare-metal / VM), no filtering.
            set[str] – visible card node names (may be empty when only
                       renderD* nodes are present, meaning no cards visible).
        """
        if not self._dev_dri.exists():
            return None

        try:
            cards: set[str] = set()
            for entry in self._dev_dri.iterdir():
                if entry.name.startswith("card") and not entry.name.startswith("card-"):
                    cards.add(entry.name)
            return cards
        except (OSError, PermissionError) as e:
            logger.debug("Unable to list %s (%s); disabling device filtering", self._dev_dri, e)
            return None

    def _get_gpu_info_sysfs(self) -> Optional[list[GPUInfo]]:
        """Detect GPUs by reading /sys/class/drm/card*/device/ attributes.

        Reads device_id, mem_info_vram_total, mem_info_vram_used,
        gpu_busy_percent, and mem_busy_percent from sysfs. Only GPUs whose
        card node is present in /dev/dri/ are included, so that containers
        with a subset of GPUs mounted report only their assigned devices.
        """
        try:
            if not self._drm_base.exists():
                logger.debug("sysfs DRM base not found: %s", self._drm_base)
                return None

            visible_cards = self._get_visible_cards()
            if visible_cards is not None:
                logger.debug("Visible card devices in %s: %s", self._dev_dri, visible_cards)

            gpus = []

            card_dirs = sorted(
                self._drm_base.glob("card[0-9]*/device"),
                key=GPUDetector._card_index,
            )

            for device_dir in card_dirs:
                card_name = device_dir.parent.name

                if visible_cards is not None and card_name not in visible_cards:
                    continue

                device_file = device_dir / "device"
                raw_id = self._read_sysfs(device_file)
                if raw_id is None:
                    continue

                normalized_id = self._normalize_device_id(raw_id)

                if not GPU_DEVICE_ID_PATTERN.match(normalized_id):
                    continue

                vram_total_path = device_dir / "mem_info_vram_total"
                if not vram_total_path.exists():
                    continue

                vram_total_bytes = self._read_sysfs_int(vram_total_path)
                if vram_total_bytes <= 0:
                    continue

                vram_used_bytes = self._read_sysfs_int(device_dir / "mem_info_vram_used")
                gpu_busy = self._read_sysfs_int(device_dir / "gpu_busy_percent")
                mem_busy = self._read_sysfs_int(device_dir / "mem_busy_percent")

                vram_total_mb = vram_total_bytes // (1024 * 1024)
                vram_used_mb = vram_used_bytes // (1024 * 1024)

                model = self.get_gpu_model(normalized_id)

                gpu_info = GPUInfo(
                    device_id=normalized_id,
                    model=model,
                    vram_total=vram_total_mb,
                    vram_used=vram_used_mb,
                    gfx_utilization=float(gpu_busy),
                    mem_utilization=float(mem_busy),
                )
                gpus.append(gpu_info)

            return gpus if gpus else None

        except Exception as e:
            logger.error("Error reading GPU info from sysfs: %s", e, exc_info=True)
            return None

    # ── amdsmi backend (fallback) ──────────────────────────────────────────

    def _get_gpu_info_amdsmi(self) -> Optional[list[GPUInfo]]:
        """Fallback: detect GPUs using the AMD SMI Python library.

        This is used when sysfs detection fails (e.g., missing attributes,
        non-standard kernel). amdsmi handles container GPU visibility
        natively via the ROCm runtime (/dev/kfd + /dev/dri/renderD*).
        """
        try:
            from amdsmi import (
                amdsmi_get_gpu_activity,
                amdsmi_get_gpu_asic_info,
                amdsmi_get_gpu_vram_usage,
                amdsmi_get_processor_handles,
                amdsmi_init,
                amdsmi_shut_down,
            )

            amdsmi_init()
            try:
                handles = amdsmi_get_processor_handles()
                if not handles:
                    return None

                gpus = []
                for handle in handles:
                    asic_info = amdsmi_get_gpu_asic_info(handle)
                    vram_info = amdsmi_get_gpu_vram_usage(handle)
                    util_info = amdsmi_get_gpu_activity(handle)

                    normalized_id = self._normalize_device_id(asic_info["device_id"])
                    model = self.get_gpu_model(normalized_id)

                    gpu_info = GPUInfo(
                        device_id=normalized_id,
                        model=model,
                        vram_total=vram_info["vram_total"],
                        vram_used=vram_info["vram_used"],
                        gfx_utilization=util_info["gfx_activity"],
                        mem_utilization=util_info["umc_activity"],
                    )
                    gpus.append(gpu_info)

                return gpus

            finally:
                amdsmi_shut_down()

        except ImportError:
            logger.debug("amdsmi Python library not available")
            return None
        except Exception as e:
            logger.error("Error using amdsmi library: %s", e, exc_info=True)
            return None
