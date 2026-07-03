# Copyright © Advanced Micro Devices, Inc., or its affiliates.
#
# SPDX-License-Identifier: MIT

import logging
import os
import re
from dataclasses import dataclass
from functools import cached_property
from pathlib import Path
from typing import Any, Dict, Optional

from .object_model import AcceleratorModel

logger = logging.getLogger(__name__)

CGROUP_V2_CPU_MAX = "/sys/fs/cgroup/cpu.max"
CGROUP_V1_QUOTA = "/sys/fs/cgroup/cpu/cpu.cfs_quota_us"
CGROUP_V1_PERIOD = "/sys/fs/cgroup/cpu/cpu.cfs_period_us"
CPUSET_V2_EFFECTIVE = "/sys/fs/cgroup/cpuset.cpus.effective"
CPUSET_V2 = "/sys/fs/cgroup/cpuset.cpus"
CPUSET_V1 = "/sys/fs/cgroup/cpuset/cpuset.cpus"

# Node-level CPU inventory sources, unaffected by cpuset/cgroup limits.
# Used to report the host's full capacity for node labelling (detect-hardware).
CPU_PRESENT = "/sys/devices/system/cpu/present"
PROC_CPUINFO = "/proc/cpuinfo"

EPYC_MODEL_PATTERN = re.compile(r"EPYC\s+(9[\dA-Z]{2}[45][A-Z]?)")


@dataclass
class CPUInfo:
    """Represents information about detected CPU hardware."""

    # CPU vendor (e.g. 'AMD')
    vendor: str
    # CPU model (e.g. AcceleratorModel.EPYC_9965), None if non-AMD or unrecognized
    model: Optional[AcceleratorModel]
    # EPYC model number (e.g. '9965')
    model_number: Optional[str]
    # Total logical CPU count from os.cpu_count() (includes hyperthreads; may reflect cgroup limits in containers)
    physical_cores: int
    # Number of cores available to the container (e.g. this can be less than physical_cores if the container is limited by the cgroup)
    available_cores: int
    # Raw cpuset string if detected from cpuset file, else None
    cpuset_bind: Optional[str]
    # Node's full logical CPU capacity, independent of cpuset/cgroup limits
    # (from /sys/devices/system/cpu/present). Used for node labelling.
    node_cores: int = 0

    def to_dict(self) -> dict:
        return {
            "vendor": self.vendor,
            "model": self.model.value if self.model else None,
            "model_number": self.model_number,
            "physical_cores": self.physical_cores,
            "available_cores": self.available_cores,
            "cpuset_bind": self.cpuset_bind,
            "node_cores": self.node_cores,
        }


class CpuDetector:
    """Detects CPU information and available cores in containerized environments."""

    def __init__(self):
        self._cpu_info: Optional[CPUInfo] = None
        self._detected = False

    @cached_property
    def cpu_info(self) -> Optional[CPUInfo]:
        if not self._detected:
            self._detect()
        return self._cpu_info

    @cached_property
    def cpu_model(self) -> Optional[AcceleratorModel]:
        info = self.cpu_info
        return info.model if info else None

    @cached_property
    def available_cores(self) -> int:
        info = self.cpu_info
        return info.available_cores if info else (os.cpu_count() or 1)

    @cached_property
    def node_cores(self) -> int:
        """Node's full logical CPU capacity, independent of cpuset/cgroup limits.

        Reflects the host's total cores (e.g. 384 on a dual-socket EPYC 9965) even
        when the container is restricted via --cpuset-cpus or a Kubernetes static
        CPU policy. Used by detect-hardware for node labelling, where presence and
        full capacity matter rather than the pod's allocation.
        """
        info = self.cpu_info
        return info.node_cores if info else (os.cpu_count() or 1)

    @cached_property
    def cpuset_bind(self) -> Optional[str]:
        """Raw cpuset string when cores were detected from a cpuset file (e.g. '0-127', '16-47,64-95').

        Returns None when cores were detected from cgroup quota or sched_getaffinity,
        because in those cases the specific CPU IDs assigned to the container are unknown.
        Use 'auto' for VLLM_CPU_OMP_THREADS_BIND when this is None.
        """
        info = self.cpu_info
        return info.cpuset_bind if info else None

    def _detect(self) -> None:
        if self._detected:
            return

        vendor, model_name = self._parse_proc_cpuinfo()
        model_number = self._extract_model_number(model_name)
        cpu_model = self._resolve_cpu_model(model_number, vendor)
        physical_cores = os.cpu_count() or 1
        available_cores, cpuset_bind = self._detect_available_cores(physical_cores)
        node_cores = self._detect_node_cores(physical_cores)

        self._cpu_info = CPUInfo(
            vendor=vendor,
            model=cpu_model,
            model_number=model_number,
            physical_cores=physical_cores,
            available_cores=available_cores,
            cpuset_bind=cpuset_bind,
            node_cores=node_cores,
        )
        self._detected = True

        logger.info(f"CPU detection: {self._cpu_info.to_dict()}")

    @staticmethod
    def _parse_proc_cpuinfo() -> tuple:
        """Parse /proc/cpuinfo to extract vendor and model name."""
        vendor = "unknown"
        model_name = "unknown"
        try:
            with open("/proc/cpuinfo", "r") as f:
                for line in f:
                    if line.startswith("vendor_id"):
                        vendor = line.split(":", 1)[1].strip()
                    elif line.startswith("model name"):
                        model_name = line.split(":", 1)[1].strip()
                        break
        except FileNotFoundError:
            logger.warning("/proc/cpuinfo not found")
        except Exception as e:
            logger.error(f"Error reading /proc/cpuinfo: {e}")
        return vendor, model_name

    @staticmethod
    def _extract_model_number(model_name: str) -> Optional[str]:
        return None

    @classmethod
    def _resolve_cpu_model(cls, model_number: Optional[str], vendor: str) -> Optional[AcceleratorModel]:
        """Resolve CPU model from EPYC model number and vendor.

        Returns:
            - Specific AcceleratorModel (e.g. EPYC_9965) if recognized EPYC model
            - None for unrecognized AMD CPUs or non-AMD CPUs
        """
        if model_number:
            result = AcceleratorModel.from_string_with_default(model_number)
            if result is None:
                logger.warning(
                    f"Unrecognized EPYC model number '{model_number}'. "
                    "CPU inference may not work; using generic EPYC defaults."
                )
            else:
                logger.debug(f"Recognized EPYC model: {result.value}")
            return result

        return cls._fallback_to_generic_cpu(vendor)

    @staticmethod
    def _fallback_to_generic_cpu(vendor: str) -> Optional[AcceleratorModel]:
        if "amd" in vendor.lower():
            logger.info("AMD CPU detected.")
            return AcceleratorModel.CPU

        logger.debug(f"Non-AMD CPU detected: {vendor}")
        return None

    @classmethod
    def _detect_node_cores(cls, physical_cores: int) -> int:
        """Detect the node's full logical CPU capacity, ignoring cpuset/cgroup limits.

        Unlike ``_detect_available_cores`` (which honours container restrictions),
        this reports the host's total core count for node labelling. ``--cpuset-cpus``
        and Kubernetes static CPU policies restrict scheduling affinity but do not
        mask the node inventory exposed by ``/sys/devices/system/cpu/present`` or
        ``/proc/cpuinfo``.

        Priority: /sys present -> /proc/cpuinfo processor count -> os.cpu_count().
        """
        cores = cls._read_cpu_present()
        if cores is not None:
            logger.info(f"Detected {cores} node cores from {CPU_PRESENT}")
            return cores

        cores = cls._read_proc_cpuinfo_count()
        if cores is not None:
            logger.info(f"Detected {cores} node cores from {PROC_CPUINFO} processor count")
            return cores

        logger.info(f"Falling back to os.cpu_count() for node cores: {physical_cores}")
        return physical_cores

    @staticmethod
    def _read_cpu_present() -> Optional[int]:
        """Count CPUs listed in /sys/devices/system/cpu/present (e.g. '0-383' -> 384)."""
        try:
            content = Path(CPU_PRESENT).read_text().strip()
            if not content:
                return None
            cores = len(CpuDetector._parse_cpuset_to_set(content))
            return cores if cores > 0 else None
        except (FileNotFoundError, OSError, ValueError):
            return None

    @staticmethod
    def _read_proc_cpuinfo_count() -> Optional[int]:
        """Count 'processor' entries in /proc/cpuinfo (host-wide logical CPUs)."""
        try:
            count = 0
            with open(PROC_CPUINFO, "r") as f:
                for line in f:
                    if line.startswith("processor"):
                        count += 1
            return count if count > 0 else None
        except (FileNotFoundError, OSError):
            return None

    @classmethod
    def _detect_available_cores(cls, physical_cores: int) -> tuple[int, Optional[str]]:
        """Detect available CPU cores respecting container limits.

        Priority: cgroup v2 quota -> cgroup v1 quota -> cpuset -> sched_getaffinity -> os fallback.

        Returns:
            (available_cores, cpuset_bind) where cpuset_bind is the raw cpuset string
            (e.g. '0-127', '16-47,64-95') when detected from a cpuset file, or None
            when detected from cgroup quota or affinity (CPU IDs are unknown in those cases).
        """
        cores = cls._read_cgroup_v2_quota()
        if cores is not None:
            logger.info(f"Detected {cores} cores from cgroup v2 cpu.max")
            return cores, None  # CPU IDs unknown from quota

        cores = cls._read_cgroup_v1_quota()
        if cores is not None:
            logger.info(f"Detected {cores} cores from cgroup v1 CFS quota")
            return cores, None  # CPU IDs unknown from quota

        result = cls._read_cpuset()
        if result is not None:
            cores, cpuset_str = result
            logger.info(f"Detected {cores} cores from cpuset ({cpuset_str})")
            return cores, cpuset_str  # CPU IDs known from cpuset

        try:
            affinity_cores = len(os.sched_getaffinity(0))  # type: ignore[attr-defined]
            logger.info(f"Detected {affinity_cores} cores from sched_getaffinity")
            return affinity_cores, None  # CPU IDs unknown from affinity
        except (AttributeError, OSError) as e:
            logger.debug("sched_getaffinity unavailable, counting all sockets: %s", e)

        logger.info(f"Falling back to os.cpu_count(): {physical_cores} cores")
        return physical_cores, None

    @staticmethod
    def _read_cgroup_v2_quota() -> Optional[int]:
        """Read CPU quota from cgroup v2 cpu.max (format: '$MAX $PERIOD' or 'max $PERIOD')."""
        try:
            content = Path(CGROUP_V2_CPU_MAX).read_text().strip()
            parts = content.split()
            if len(parts) != 2 or parts[0] == "max":
                return None
            quota, period = int(parts[0]), int(parts[1])
            if quota <= 0 or period <= 0:
                return None

            # Calculate exact cores (fractional)
            exact_cores = quota / period
            # Use floor division (conservative - avoids over-subscription)
            cores = max(1, quota // period)

            # Warn if fractional allocation detected
            if exact_cores != cores:
                logger.warning(
                    f"Fractional CPU allocation detected: {exact_cores:.2f} cores. "
                    f"Using {cores} cores (conservative). "
                    f"For optimal performance, ensure Kubernetes CPU requests/limits are whole numbers "
                    f"aligned with NUMA topology."
                )
            else:
                logger.debug(f"Detected {cores} cores from cgroup v2 quota")

            return cores
        except (FileNotFoundError, ValueError, OSError):
            return None

    @staticmethod
    def _read_cgroup_v1_quota() -> Optional[int]:
        """Read CPU quota from cgroup v1 CFS files."""
        try:
            quota = int(Path(CGROUP_V1_QUOTA).read_text().strip())
            period = int(Path(CGROUP_V1_PERIOD).read_text().strip())
            if quota <= 0 or period <= 0:
                return None

            # Calculate exact cores (fractional)
            exact_cores = quota / period
            # Use floor division (conservative - avoids over-subscription)
            cores = max(1, quota // period)

            # Warn if fractional allocation detected
            if exact_cores != cores:
                logger.warning(
                    f"Fractional CPU allocation detected: {exact_cores:.2f} cores. "
                    f"Using {cores} cores (conservative). "
                    f"For optimal performance, ensure Kubernetes CPU requests/limits are whole numbers "
                    f"aligned with NUMA topology."
                )
            else:
                logger.debug(f"Detected {cores} cores from cgroup v1 quota")

            return cores
        except (FileNotFoundError, ValueError, OSError):
            return None

    @classmethod
    def _read_cpuset(cls) -> Optional[tuple[int, str]]:
        """Read available CPUs from cpuset (cgroup v2 effective, v2, then v1).

        Returns:
            (core_count, raw_cpuset_string) when a cpuset file is found and parseable,
            e.g. (128, '0-127') or (192, '0-95,128-223').
            None if no cpuset file is found or parseable.
        """
        for path in [CPUSET_V2_EFFECTIVE, CPUSET_V2, CPUSET_V1]:
            try:
                content = Path(path).read_text().strip()
                if content:
                    cores = len(cls._parse_cpuset_to_set(content))
                    if cores > 0:
                        return cores, content
            except (FileNotFoundError, OSError):
                continue
        return None

    @staticmethod
    def _parse_cpuset_to_set(cpuset_str: str) -> set:
        """Parse a cpuset range string like '0-3,5,7-9' into a set of CPU IDs."""
        cpus: set[int] = set()
        for part in cpuset_str.split(","):
            part = part.strip()
            if "-" in part:
                lo, hi = part.split("-", 1)
                cpus.update(range(int(lo), int(hi) + 1))
            elif part.isdigit():
                cpus.add(int(part))
        return cpus

    @staticmethod
    def _override_omp_num_threads(env_vars: Dict[str, Any], detected_cores: int) -> None:
        """Override OMP_NUM_THREADS based on detected core count."""
        old_threads = env_vars.get("OMP_NUM_THREADS")

        if old_threads:
            try:
                required_threads = int(old_threads)
                new_threads = min(required_threads, detected_cores)
                if detected_cores < required_threads:
                    logger.warning(
                        f"Scaling down CPU configuration: profile requests {required_threads} cores "
                        f"(OMP_NUM_THREADS={old_threads}), but only {detected_cores} cores detected. "
                        f"OMP_NUM_THREADS will be set to {detected_cores}."
                    )

                if new_threads != required_threads:
                    logger.info(f"Overriding CPU env vars: OMP_NUM_THREADS {old_threads}->{new_threads}")

                env_vars["OMP_NUM_THREADS"] = str(new_threads)
                return

            except ValueError:
                logger.warning(f"Could not parse OMP_NUM_THREADS value from profile: {old_threads}")

        logger.info(f"No OMP_NUM_THREADS found, using auto detection for OMP_NUM_THREADS ({detected_cores} cores)")
        env_vars["OMP_NUM_THREADS"] = str(detected_cores)

    @staticmethod
    def _override_omp_threads_bind(env_vars: Dict[str, Any], detected_cores: int, cpuset_bind: Optional[str]) -> None:
        """Override VLLM_CPU_OMP_THREADS_BIND based on detected core count and cpuset bind."""
        old_bind = env_vars.get("VLLM_CPU_OMP_THREADS_BIND")
        if old_bind == "auto" or old_bind is None:
            bind_value = "auto"
        elif cpuset_bind is not None:
            bind_value = cpuset_bind
        else:
            bind_value = "auto"

        if bind_value != old_bind:
            logger.info(f"Overriding CPU env vars: VLLM_CPU_OMP_THREADS_BIND {old_bind}->{bind_value}")

        env_vars["VLLM_CPU_OMP_THREADS_BIND"] = bind_value

    @classmethod
    def override_cpu_env_vars(
        cls, env_vars: Dict[str, Any], detected_cores: int, cpuset_bind: Optional[str] = None
    ) -> None:
        """Override CPU-related env vars based on detected core count.

        When cpuset_bind is provided (cores detected from a cpuset file), the exact
        CPU IDs are known and used for explicit binding. Otherwise 'auto' is used so
        vLLM/OpenMP discovers the actual core assignments — required for Kubernetes
        where NUMA-aware scheduling may assign non-sequential IDs.

        If the profile requests more cores (via OMP_NUM_THREADS) than are available,
        the value is scaled down to the detected core count with a warning — the
        profile is never rejected.

        Args:
            env_vars: Environment variables from the SELECTED PROFILE (not os.environ).
                     OMP_NUM_THREADS here comes from the profile YAML, not the base
                     container environment.
            detected_cores: Number of cores detected by CpuDetector.
            cpuset_bind: Raw cpuset string (e.g. '0-127', '16-47,64-95') when CPU IDs
                        are known from a cpuset file. None when IDs are unknown.
        """
        cls._override_omp_num_threads(env_vars, detected_cores)

        cls._override_omp_threads_bind(env_vars, detected_cores, cpuset_bind)


class EpycDetector(CpuDetector):
    """Detects AMD EPYC CPUs and available core counts in containerized environments."""

    def __init__(self):
        super().__init__()

    @staticmethod
    def _extract_model_number(model_name: str) -> Optional[str]:
        """Extract EPYC model number (e.g. '9965') from a model name string."""
        match = EPYC_MODEL_PATTERN.search(model_name)
        if match:
            return match.group(1)
        return None

    @staticmethod
    def _fallback_to_generic_cpu(vendor: str) -> Optional[AcceleratorModel]:
        if "amd" in vendor.lower():
            logger.info("AMD CPU detected but not an EPYC processor. CPU inference is only supported on EPYC CPUs.")
            return None

        logger.debug(f"Non-AMD CPU detected: {vendor}")
        return None
