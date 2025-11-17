# Copyright © Advanced Micro Devices, Inc., or its affiliates.
#
# SPDX-License-Identifier: MIT

import json
import logging
from dataclasses import dataclass
from functools import cached_property
from typing import List, Mapping, Optional

from .object_model import GPUModel

logger = logging.getLogger(__name__)


@dataclass
class GPUInfo:
    """Represents information about a single GPU."""

    device_id: str
    model: GPUModel
    vram_total: int  # in MB
    vram_used: int  # in MB
    gfx_utilization: float  # percentage
    mem_utilization: float  # percentage

    @property
    def vram_free(self) -> int:
        """Get free VRAM in MB."""
        return self.vram_total - self.vram_used

    @property
    def is_idle(self) -> bool:
        """Check if GPU is idle (no graphics utilization and memory is free)."""
        # A GPU is idle if it has no graphics activity AND memory is considered free
        return self.gfx_utilization == 0 and self.mem_utilization <= 5

    def to_dict(self) -> dict:
        """Convert to dictionary for logging/serialization."""
        return {
            "device_id": self.device_id,
            "model": self.model.value,
            "vram_total": self.vram_total,
            "vram_used": self.vram_used,
            "vram_free": self.vram_free,
            "gfx_utilization": self.gfx_utilization,
            "mem_utilization": self.mem_utilization,
            "is_idle": self.is_idle,
        }


class GPUDetector:
    """Detects AMD GPUs and provides information about them."""

    # GPU device ID to model name mapping
    # reference: https://github.com/ROCm/gpu-operator/blob/main/helm-charts-k8s/templates/gpu-nfd-default-rule.yaml
    GPU_MODEL_MAPPING = {
        # AMD Instinct
        "0x738c": GPUModel.MI100,
        "0x738e": GPUModel.MI100,
        "0x7408": GPUModel.MI250X,
        "0x740c": GPUModel.MI250X,  # MI250/MI250X
        "0x740f": GPUModel.MI210,
        "0x7410": GPUModel.MI210,  # MI210 VF
        "0x74a0": GPUModel.MI300A,
        "0x74a1": GPUModel.MI300X,
        "0x74a2": GPUModel.MI308X,
        "0x74a5": GPUModel.MI325X,
        "0x74a8": GPUModel.MI308X,  # MI308X HF
        "0x74a9": GPUModel.MI300X,  # MI300X HF
        "0x74b5": GPUModel.MI300X,  # MI300X VF
        "0x74b6": GPUModel.MI308X,
        "0x74b9": GPUModel.MI325X,  # MI325X VF
        "0x74bd": GPUModel.MI300X,  # MI300X HF
        "0x75a0": GPUModel.MI350X,
        "0x75a3": GPUModel.MI355X,
        "0x75b0": GPUModel.MI350X,  # MI350X VF
        "0x75b3": GPUModel.MI355X,  # MI355X VF
        # AMD Radeon Pro
        "0x7460": GPUModel.V710,
        "0x7461": GPUModel.V710,  # Radeon Pro V710 MxGPU
        "0x7448": GPUModel.W7900,
        "0x744a": GPUModel.W7900,  # W7900 Dual Slot
        "0x7449": GPUModel.W7800,  # W7800 48GB
        "0x745e": GPUModel.W7800,
        "0x73a2": GPUModel.W6900X,
        "0x73a3": GPUModel.W6800,  # W6800 GL-XL
        "0x73ab": GPUModel.W6800X,  # W6800X / W6800X Duo
        "0x73a1": GPUModel.V620,
        "0x73ae": GPUModel.V620,  # Radeon Pro V620 MxGPU
        # AMD Radeon
        "0x7550": GPUModel.RX9070,  # RX 9070 / 9070 XT
        "0x744c": GPUModel.RX7900,  # RX 7900 XT / 7900 XTX / 7900 GRE / 7900M
        "0x73af": GPUModel.RX6900,
        "0x73bf": GPUModel.RX6800,  # RX 6800 / 6800 XT / 6900 XT
    }

    def __init__(self):
        self._gpus: Optional[List[GPUInfo]] = None
        self._detected = False

    def group_by_device(self) -> Mapping[str, List[GPUInfo]]:
        """Group detected GPUs by their normalized device id."""
        groups: dict[str, List[GPUInfo]] = {}
        if not self.gpus or self.gpus is None:
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
    def gpus(self) -> Optional[List[GPUInfo]]:
        """Get list of detected GPUs with full information."""
        if not self._detected:
            self._detect_gpus()
        return self._gpus

    @cached_property
    def device_ids(self) -> Optional[List[str]]:
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

    def get_gpu_model(self, device_id: str) -> GPUModel:
        """Get GPU model name from device ID.

        Args:
            device_id: GPU device ID (e.g., "74a1")

        Returns:
            GPU model name (e.g., "MI300X") or "Unknown" if not found
        """
        norm = self._normalize_device_id(device_id)
        return self.GPU_MODEL_MAPPING.get(norm, GPUModel.UNKNOWN)

    @cached_property
    def gpu_models(self) -> Optional[List[GPUModel]]:
        """Get list of detected GPU model names."""
        if not self.gpus:
            return None
        return [gpu.model for gpu in self.gpus]

    def get_gpu_info(self) -> Optional[List[dict]]:
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

    def _log_gpu_health(self, gpus: List[GPUInfo]) -> None:
        """Log GPU health status and warn about non-idle GPUs."""
        for gpu in gpus:
            logger.info(f"GPU {gpu.device_id}: {json.dumps(gpu.to_dict(), indent=2)}")

            # Consider it okay if GPU is idle or effectively free by memory criteria
            if not gpu.is_idle:
                logger.error(
                    f"GPU {gpu.device_id} is not idle - "
                    f"GFX utilization: {gpu.gfx_utilization}%, "
                    f"MEM utilization: {gpu.mem_utilization}%"
                )

    def _detect_gpus(self) -> None:
        """Perform GPU detection using AMD SMI Python library."""
        if self._detected:
            return

        # Detect GPUs
        gpus = self._get_gpu_info()

        if gpus is None:
            logger.warning("No AMD GPUs detected")
            self._gpus = None
        else:
            self._gpus = gpus
            logger.info(f"Detected {len(self._gpus)} AMD GPU(s)")
            self._log_gpu_health(self._gpus)

        self._detected = True

    def _get_gpu_info(self) -> Optional[List[GPUInfo]]:
        """Get GPU info using AMD SMI Python library."""
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
            logger.debug("AMD SMI Python library not available")
            return None
        except Exception as e:
            logger.error(f"Error using AMD SMI library: {e}")
            return None
