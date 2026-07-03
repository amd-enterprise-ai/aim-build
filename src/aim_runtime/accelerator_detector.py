# Copyright © Advanced Micro Devices, Inc., or its affiliates.
#
# SPDX-License-Identifier: MIT

"""
Unified Accelerator Detection

Provides a single entry point for detecting hardware accelerators (GPU or CPU).
Delegates to GPUDetector or EpycDetector based on the configured accelerator type
and returns a unified AcceleratorDetectionResult.
"""

import logging
from dataclasses import dataclass, field
from typing import List, Optional, Union

from .cpu_detector import CpuDetector, CPUInfo, EpycDetector
from .gpu_detector import GPUDetector, GPUInfo
from .object_model import AcceleratorFamily, AcceleratorModel, AcceleratorType

logger = logging.getLogger(__name__)


@dataclass
class AcceleratorDetectionResult:
    """Unified result from hardware detection."""

    accelerator_type: AcceleratorType
    accelerator_model: Optional[AcceleratorModel] = None
    accelerator_count: int = 0
    gpu_info: List[GPUInfo] = field(default_factory=list)
    cpu_info: Optional[CPUInfo] = None
    cpu_cores: int = 0
    cpuset_bind: Optional[str] = None

    def to_label_dicts(self) -> List[dict]:
        """Return a list of dicts for node labelling.

        Each dict contains accelerator_type (uppercase), accelerator_model,
        and accelerator_count.  Returns an empty list when no model was detected.
        """
        if self.accelerator_model is None:
            return []
        return [
            {
                "accelerator_type": self.accelerator_type.value.upper(),
                "accelerator_model": self.accelerator_model.value,
                "accelerator_count": self.accelerator_count,
            }
        ]

    def to_detail_dict(self) -> dict:
        """Return full detection details including GPUInfo / CPUInfo."""
        result: dict = {
            "accelerator_type": self.accelerator_type.value,
            "accelerator_model": self.accelerator_model.value if self.accelerator_model else None,
            "accelerator_count": self.accelerator_count,
        }
        if self.gpu_info:
            result["gpu_info"] = [g.to_dict() for g in self.gpu_info]
        if self.cpu_info:
            result["cpu_info"] = self.cpu_info.to_dict()
            result["cpu_cores"] = self.cpu_cores
            result["node_cores"] = self.cpu_info.node_cores
            result["cpuset_bind"] = self.cpuset_bind
        return result


class AcceleratorDetector:
    """Unified hardware detector that delegates to GPUDetector or EpycDetector.

    Usage::

        detector = AcceleratorDetector()
        result = detector.detect(
            accelerator_type=AcceleratorType.GPU,
            accelerator_model_override=config.accelerator_model,
            accelerator_count_override=config.accelerator_count,
        )
        # result.accelerator_model, result.accelerator_count, etc.
    """

    def detect(
        self,
        accelerator_type: AcceleratorType = AcceleratorType.GPU,
        accelerator_family: AcceleratorFamily = AcceleratorFamily.INSTINCT,
        accelerator_model_override: Optional[AcceleratorModel] = None,
        accelerator_count_override: Union[int, str] = "auto",
    ) -> AcceleratorDetectionResult:
        """Detect hardware accelerators.

        Args:
            accelerator_type: Accelerator type — ``AcceleratorType.GPU`` or ``AcceleratorType.CPU``.
            accelerator_family: Accelerator family (e.g. ``AcceleratorFamily.INSTINCT``) for accelerator detection.
            accelerator_model_override: If set, skip auto-detection and use
                this model directly (from ``AIM_ACCELERATOR_MODEL``).
            accelerator_count_override: ``"auto"`` for auto-detection, or an
                ``int`` to override (from ``AIM_ACCELERATOR_COUNT``).

        Returns:
            An ``AcceleratorDetectionResult`` with the detected (or overridden)
            hardware information.
        """
        if accelerator_type == AcceleratorType.CPU:
            return self._detect_cpu(accelerator_family)
        return self._detect_gpu(accelerator_model_override, accelerator_count_override)

    # ------------------------------------------------------------------
    # GPU detection
    # ------------------------------------------------------------------

    def _detect_gpu(
        self,
        model_override: Optional[AcceleratorModel],
        count_override: Union[int, str],
    ) -> AcceleratorDetectionResult:
        """Detect GPU hardware, respecting config overrides."""
        if model_override is not None:
            logger.info(f"Using accelerator model from config: {model_override}")
            if count_override == "auto":
                logger.warning(
                    "AIM_ACCELERATOR_MODEL is set but AIM_ACCELERATOR_COUNT is 'auto'. "
                    "Defaulting to 1. Set AIM_ACCELERATOR_COUNT explicitly if needed."
                )
                count = 1
            else:
                count = int(count_override)
            return AcceleratorDetectionResult(
                accelerator_type=AcceleratorType.GPU,
                accelerator_model=model_override,
                accelerator_count=count,
            )

        gpu_detector = GPUDetector()
        if not gpu_detector.all_gpus_idle:
            logger.warning("Some GPUs are not idle! Check GPU usage.")

        if gpu_detector.has_gpus and gpu_detector.gpu_models:
            model = gpu_detector.gpu_models[0]
            if count_override == "auto":
                count = gpu_detector.gpu_count
            else:
                count = int(count_override)
            return AcceleratorDetectionResult(
                accelerator_type=AcceleratorType.GPU,
                accelerator_model=model,
                accelerator_count=count,
                gpu_info=gpu_detector.gpus or [],
            )

        return AcceleratorDetectionResult(
            accelerator_type=AcceleratorType.GPU,
            accelerator_model=None,
            accelerator_count=0,
        )

    # ------------------------------------------------------------------
    # CPU detection
    # ------------------------------------------------------------------

    def _detect_cpu(self, accelerator_family: AcceleratorFamily = AcceleratorFamily.EPYC) -> AcceleratorDetectionResult:
        """Detect EPYC CPU model and core counts.

        ``accelerator_count`` reports the node's full CPU capacity (``node_cores``),
        independent of cpuset/cgroup limits, so node labelling via ``detect-hardware``
        reflects the host rather than the pod's allocation.  ``cpu_cores`` keeps the
        cpuset-limited count used by ``serve``/``dry-run`` to size OMP threads and
        bind vLLM to the container's cpus only.

        Falls back to ``AcceleratorModel.CPU`` when no EPYC CPU is detected.
        """
        cpu_detectors_map = {
            AcceleratorFamily.EPYC: EpycDetector,
            AcceleratorFamily.CPU: CpuDetector,
        }
        cpu_detector = cpu_detectors_map.get(accelerator_family, CpuDetector)()

        cpu_cores = cpu_detector.available_cores
        node_cores = cpu_detector.node_cores
        cpu_model = cpu_detector.cpu_model
        logger.info(f"Auto-detected CPU cores available to container: {cpu_cores}")
        logger.info(f"Auto-detected node CPU capacity: {node_cores}")
        logger.info(f"Auto-detected CPU model: {cpu_model}")
        logger.debug(f"CPU cpuset bind: {cpu_detector.cpuset_bind}")

        if cpu_model is None:
            logger.warning("No known CPU detected, falling back to generic CPU model")
            cpu_model = AcceleratorModel.CPU
            cpu_cores = cpu_cores or 1
            node_cores = node_cores or 1

        return AcceleratorDetectionResult(
            accelerator_type=AcceleratorType.CPU,
            accelerator_model=cpu_model,
            accelerator_count=node_cores,
            cpu_info=cpu_detector.cpu_info,
            cpu_cores=cpu_cores,
            cpuset_bind=cpu_detector.cpuset_bind,
        )
